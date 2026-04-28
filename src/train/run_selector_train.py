#!/usr/bin/env python3
"""Train a bi-encoder style checklist selector on oracle labels.

MVP design:
- frozen text encoder (sample/question embeddings)
- trainable MLP head over [s, q, s*q]
- listwise ranking loss (ListMLE)
- optional auxiliary heads (answerable / dimension)

Usage:
    python src/train/run_selector_train.py \
        --oracle data/oracle/train_oracle_v3.parquet \
        --out results/checkpoints/selector_v1 --epochs 3
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import hashlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - dependency availability is environment-specific
    AutoModel = None
    AutoTokenizer = None

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


@dataclass
class OracleTensors:
    sample_ids: list[str]
    qids: list[int]
    sample_texts: list[str]
    question_texts: list[str]
    qid_to_index: dict[int, int]
    q_dim_ids: torch.Tensor
    rank_target: torch.Tensor
    u2_target: torch.Tensor
    ans_target: torch.Tensor
    active_mask: torch.Tensor
    ranking_sample_mask: torch.Tensor
    dim_vocab: list[str]
    bank_df: pd.DataFrame


class SelectorHead(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, n_dims: int, dropout: float = 0.1):
        super().__init__()
        in_dim = emb_dim * 3
        hid2 = max(hidden_dim // 2, 64)
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hid2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rank_head = nn.Linear(hid2, 1)
        self.ans_head = nn.Linear(hid2, 1)
        self.dim_head = nn.Linear(hid2, n_dims)

    def forward(self, sample_emb: torch.Tensor, q_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample_emb: [B, D], q_emb: [Q, D]
        bsz, emb_dim = sample_emb.shape
        n_q = q_emb.shape[0]
        s = sample_emb.unsqueeze(1).expand(bsz, n_q, emb_dim)
        q = q_emb.unsqueeze(0).expand(bsz, n_q, emb_dim)
        feat = torch.cat([s, q, s * q], dim=-1)
        hid = self.backbone(feat)
        rank_logits = self.rank_head(hid).squeeze(-1)      # [B, Q]
        ans_logits = self.ans_head(hid).squeeze(-1)        # [B, Q]
        dim_logits = self.dim_head(hid)                    # [B, Q, C]
        return rank_logits, ans_logits, dim_logits


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        sample_emb: torch.Tensor,
        rank_target: torch.Tensor,
        u2_target: torch.Tensor,
        ans_target: torch.Tensor,
        active_mask: torch.Tensor,
        ranking_sample_mask: torch.Tensor,
        indices: list[int],
    ):
        self.sample_emb = sample_emb
        self.rank_target = rank_target
        self.u2_target = u2_target
        self.ans_target = ans_target
        self.active_mask = active_mask
        self.ranking_sample_mask = ranking_sample_mask
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = self.indices[idx]
        return {
            "sample_emb": self.sample_emb[i],
            "rank_target": self.rank_target[i],
            "u2_target": self.u2_target[i],
            "ans_target": self.ans_target[i],
            "active_mask": self.active_mask[i],
            "ranking_sample_mask": self.ranking_sample_mask[i],
        }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_sample_text(context: str, response_a: str, response_b: str) -> str:
    return (
        "[Context]\n" + str(context) + "\n\n"
        "[Response A]\n" + str(response_a) + "\n\n"
        "[Response B]\n" + str(response_b)
    )


def _load_oracle_tables(oracle_path: Path, sample_oracle_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not oracle_path.exists():
        raise FileNotFoundError(oracle_path)
    q_df = pd.read_parquet(oracle_path)

    if sample_oracle_path is None:
        sample_oracle_path = oracle_path.with_name(f"{oracle_path.stem}_sample.parquet")
    if not sample_oracle_path.exists():
        raise FileNotFoundError(
            f"{sample_oracle_path} not found. Pass --oracle-sample explicitly."
        )
    s_df = pd.read_parquet(sample_oracle_path)

    required_q = {
        "sample_id",
        "qid",
        "u1_answerable",
        "u2_abs_contrib",
        "u3_decisive",
        "dim",
        "question_text",
    }
    missing_q = required_q - set(q_df.columns)
    if missing_q:
        raise ValueError(f"oracle parquet missing columns: {sorted(missing_q)}")

    required_s = {"sample_id", "context", "response_a", "response_b"}
    missing_s = required_s - set(s_df.columns)
    if missing_s:
        raise ValueError(f"oracle sample parquet missing columns: {sorted(missing_s)}")

    return q_df, s_df


def _prepare_oracle_tensors(
    q_df: pd.DataFrame,
    s_df: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    only_oracle_correct_for_ranking: bool,
    target_mode: str = "oracle_baseline",
    human_relevance_df: pd.DataFrame | None = None,
    oracle_fallback_eps: float = 0.1,
) -> OracleTensors:
    if target_mode not in {"oracle_baseline", "pure_human", "human_oracle_fallback"}:
        raise ValueError(f"unknown target_mode: {target_mode}")
    if target_mode != "oracle_baseline" and human_relevance_df is None:
        raise ValueError("human_relevance_df is required when target_mode != oracle_baseline")

    bank_df = (
        q_df[["qid", "dim", "question_text"]]
        .drop_duplicates(subset=["qid"], keep="first")
        .sort_values("qid", kind="stable")
        .reset_index(drop=True)
    )

    qids = bank_df["qid"].astype(int).tolist()
    qid_to_index = {q: i for i, q in enumerate(qids)}
    n_q = len(qids)

    dim_vocab = sorted(bank_df["dim"].astype(str).unique().tolist())
    dim_to_id = {d: i for i, d in enumerate(dim_vocab)}
    q_dim_ids = torch.tensor([dim_to_id[str(d)] for d in bank_df["dim"].tolist()], dtype=torch.long)

    sample_table = s_df.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
    sample_ids = sample_table["sample_id"].astype(str).tolist()
    n_s = len(sample_ids)
    sid_to_index = {sid: i for i, sid in enumerate(sample_ids)}

    sample_texts = [
        _build_sample_text(r["context"], r["response_a"], r["response_b"])
        for _, r in sample_table.iterrows()
    ]

    rank_target = torch.zeros((n_s, n_q), dtype=torch.float32)
    u2_target = torch.zeros((n_s, n_q), dtype=torch.float32)
    ans_target = torch.zeros((n_s, n_q), dtype=torch.float32)
    active_mask = torch.zeros((n_s, n_q), dtype=torch.bool)

    n_skipped_parse_fail = 0
    n_skipped_null_util = 0

    # First pass: preserve the oracle-derived supervision tensors exactly.
    for _, row in q_df.iterrows():
        sid = str(row["sample_id"])
        qid = int(row["qid"])
        if sid not in sid_to_index or qid not in qid_to_index:
            continue

        # Skip rows where the oracle judge failed to parse either side.
        # `parse_fail` was added by build_oracle_labels.py to disambiguate
        # "not decisive" (u3=0) from "unknown" (u3=NaN).
        if bool(row.get("parse_fail", False)):
            n_skipped_parse_fail += 1
            continue

        # Any required utility signal being null means we cannot trust the
        # rank target for this (sample, qid) row — skip it.
        u1_raw = row.get("u1_answerable")
        u2_raw = row.get("u2_abs_contrib")
        u3_raw = row.get("u3_decisive")
        if pd.isna(u1_raw) or pd.isna(u2_raw) or pd.isna(u3_raw):
            n_skipped_null_util += 1
            continue

        i = sid_to_index[sid]
        j = qid_to_index[qid]

        u1 = float(u1_raw)
        u2 = float(u2_raw)
        u3 = float(u3_raw)

        y = alpha * u2 + beta * u3 + gamma * u1

        rank_target[i, j] = y
        u2_target[i, j] = u2
        ans_target[i, j] = u1
        active_mask[i, j] = True

    if target_mode != "oracle_baseline":
        h_mat = torch.zeros((n_s, n_q), dtype=torch.float32)
        n_h_used = 0
        assert human_relevance_df is not None
        for _, row in human_relevance_df.iterrows():
            sid = str(row["sample_id"])
            qid = int(row["qid"])
            if sid not in sid_to_index or qid not in qid_to_index:
                continue

            i = sid_to_index[sid]
            j = qid_to_index[qid]
            if not bool(active_mask[i, j]):
                continue

            h_raw = row.get("h")
            if pd.isna(h_raw):
                continue
            h_mat[i, j] = float(h_raw)
            n_h_used += 1

        if target_mode == "pure_human":
            rank_target = torch.where(
                active_mask,
                h_mat,
                rank_target.new_zeros(rank_target.shape),
            )
        else:
            fallback = oracle_fallback_eps * u2_target
            rank_target = torch.where(
                active_mask,
                torch.where(h_mat > 0, h_mat, fallback),
                rank_target.new_zeros(rank_target.shape),
            )

        log.info(
            "target_mode=%s; %d (sample,qid) human_relevance entries applied",
            target_mode,
            n_h_used,
        )

    if n_skipped_parse_fail or n_skipped_null_util:
        log.info(
            "Skipped %d parse-fail rows and %d null-utility rows from oracle",
            n_skipped_parse_fail,
            n_skipped_null_util,
        )

    ranking_sample_mask = torch.ones(n_s, dtype=torch.bool)
    if only_oracle_correct_for_ranking and {"winner_pred_full", "winner_gt"}.issubset(sample_table.columns):
        ranking_sample_mask = torch.tensor(
            (sample_table["winner_pred_full"] == sample_table["winner_gt"]).fillna(False).tolist(),
            dtype=torch.bool,
        )

    return OracleTensors(
        sample_ids=sample_ids,
        qids=qids,
        sample_texts=sample_texts,
        question_texts=bank_df["question_text"].astype(str).tolist(),
        qid_to_index=qid_to_index,
        q_dim_ids=q_dim_ids,
        rank_target=rank_target,
        u2_target=u2_target,
        ans_target=ans_target,
        active_mask=active_mask,
        ranking_sample_mask=ranking_sample_mask,
        dim_vocab=dim_vocab,
        bank_df=bank_df,
    )


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: Iterable[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    normalize: bool,
) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []
    texts_list = list(texts)
    for start in tqdm(range(0, len(texts_list), batch_size), desc="Encode", leave=False):
        batch = texts_list[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
            emb = _mean_pool(h, enc["attention_mask"])  # [B, D]
            if normalize:
                emb = F.normalize(emb, p=2, dim=-1)
            outputs.append(emb.detach().cpu())
    return torch.cat(outputs, dim=0)


def _listmle_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    active_mask: torch.Tensor,
    ranking_sample_mask: torch.Tensor,
) -> torch.Tensor:
    """ListMLE over active questions only."""
    losses: list[torch.Tensor] = []

    bsz = scores.shape[0]
    for i in range(bsz):
        if not bool(ranking_sample_mask[i]):
            continue

        idx = torch.nonzero(active_mask[i], as_tuple=False).squeeze(-1)
        if idx.numel() <= 1:
            continue

        s = scores[i, idx]
        t = targets[i, idx]

        order = torch.argsort(t, descending=True)
        s_ord = s[order]

        # PL log-likelihood: sum(logsumexp(s_i..s_n) - s_i)
        log_den = torch.logcumsumexp(torch.flip(s_ord, dims=[0]), dim=0)
        log_den = torch.flip(log_den, dims=[0])
        losses.append((log_den - s_ord).sum())

    if not losses:
        return scores.new_tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


def _compute_ndcg_recall(
    scores: torch.Tensor,
    gains: torch.Tensor,
    active_mask: torch.Tensor,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    out: dict[str, float] = {}
    bsz, _ = scores.shape

    for k in ks:
        ndcgs: list[float] = []
        recalls: list[float] = []

        for i in range(bsz):
            idx = torch.nonzero(active_mask[i], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            s = scores[i, idx]
            g = gains[i, idx]

            k_eff = min(k, idx.numel())
            if k_eff == 0:
                continue

            pred_rank = torch.argsort(s, descending=True)
            true_rank = torch.argsort(g, descending=True)

            pred_idx = pred_rank[:k_eff]
            true_idx = true_rank[:k_eff]

            # NDCG@k
            pred_g = g[pred_idx]
            true_g = g[true_idx]
            discounts = 1.0 / torch.log2(torch.arange(2, 2 + k_eff, dtype=torch.float32))

            dcg = float(((2.0 ** pred_g - 1.0) * discounts).sum().item())
            idcg = float(((2.0 ** true_g - 1.0) * discounts).sum().item())
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            pred_set = set(pred_idx.tolist())
            true_set = set(true_idx.tolist())
            recalls.append(len(pred_set & true_set) / len(true_set) if true_set else 0.0)

        out[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        out[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0

    return out


def _make_sample_id(prompt_id: str, response_a: str, response_b: str, winner: str) -> str:
    payload = json.dumps(
        {
            "prompt_id": prompt_id,
            "response_a": response_a,
            "response_b": response_b,
            "winner": winner,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prompt_id}_{digest}"


def _sample_ids_for_leak_check(df: pd.DataFrame, *, winner_col: str) -> set[str] | None:
    """Return stable sample ids, materializing them for raw split parquets."""
    if "sample_id" in df.columns:
        return set(df["sample_id"].dropna().astype(str).tolist())

    required = {"prompt_id", "response_a", "response_b", winner_col}
    missing = required - set(df.columns)
    if missing:
        log.warning(
            "Cannot run sample_id leakage check; missing columns: %s",
            sorted(missing),
        )
        return None

    return set(
        df.apply(
            lambda r: _make_sample_id(
                prompt_id=str(r["prompt_id"]),
                response_a=str(r["response_a"]),
                response_b=str(r["response_b"]),
                winner=str(r[winner_col]),
            ),
            axis=1,
        ).tolist()
    )


def _assert_no_holdout_leak(s_df: pd.DataFrame, holdout_paths: list[Path]) -> None:
    """Ensure no exact holdout sample appears in the oracle sample table.

    Rationale: the selector must be evaluated on dev_600 / test splits it has
    never seen during training. If someone accidentally builds an oracle parquet
    from a union of train+dev or from the wrong tier, this guard catches it
    before training burns GPU hours.
    """
    oracle_ids = _sample_ids_for_leak_check(s_df, winner_col="winner_gt")
    if oracle_ids is None:
        return

    for path in holdout_paths:
        p = Path(path)
        if not p.exists():
            log.warning("Holdout split %s not found — skipping leakage check for this file", p)
            continue
        holdout_df = pd.read_parquet(p)
        holdout_ids = _sample_ids_for_leak_check(holdout_df, winner_col="winner")
        if holdout_ids is None:
            log.warning("Skipping leakage check for %s", p)
            continue
        overlap = oracle_ids & holdout_ids
        if overlap:
            sample = sorted(overlap)[:5]
            raise AssertionError(
                f"{len(overlap)} holdout sample_ids from {p} leak into oracle "
                f"(first 5: {sample}). Rebuild the oracle on train-only data, "
                f"or pass --allow-holdout-leak to override."
            )
        log.info("Leakage guard OK: no sample_id overlap between oracle and %s", p)


def _split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(math.ceil(n * val_ratio))) if n > 1 else 0
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if not train_idx and val_idx:
        train_idx, val_idx = val_idx, []
    return train_idx, val_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--oracle-sample", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)

    parser.add_argument("--encoder-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.2)

    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--lambda-ans", type=float, default=0.3)
    parser.add_argument("--lambda-dim", type=float, default=0.1)

    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--normalize-emb", action="store_true")
    parser.add_argument("--only-oracle-correct-for-ranking", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout-splits",
        type=Path,
        nargs="*",
        default=[Path("data/splits/dev_600.parquet")],
        help="Parquet splits whose sample_ids must NOT appear in the oracle (leakage guard).",
    )
    parser.add_argument(
        "--allow-holdout-leak",
        action="store_true",
        help="Disable the holdout leakage guard (DANGEROUS — only for diagnostic runs).",
    )
    parser.add_argument(
        "--target-mode",
        choices=["oracle_baseline", "pure_human", "human_oracle_fallback"],
        default="oracle_baseline",
        help="Source of the rank target.",
    )
    parser.add_argument(
        "--human-relevance",
        type=Path,
        default=None,
        help=(
            "Path to data/oracle/<split>_human_relevance_v3.parquet. "
            "Required when --target-mode != oracle_baseline."
        ),
    )
    parser.add_argument(
        "--oracle-fallback-eps",
        type=float,
        default=0.1,
        help="Used only when --target-mode == human_oracle_fallback.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device(args.device)

    if AutoModel is None or AutoTokenizer is None:
        raise SystemExit(
            "Missing dependency: transformers. Install it before running selector training."
        )

    q_df, s_df = _load_oracle_tables(args.oracle, args.oracle_sample)

    if not args.allow_holdout_leak:
        _assert_no_holdout_leak(s_df, args.holdout_splits)
    else:
        log.warning("Holdout leakage guard disabled via --allow-holdout-leak")

    human_df: pd.DataFrame | None = None
    if args.target_mode != "oracle_baseline":
        if args.human_relevance is None:
            raise SystemExit("--human-relevance is required when --target-mode != oracle_baseline")
        if not args.human_relevance.exists():
            raise SystemExit(f"--human-relevance not found: {args.human_relevance}")

        human_df = pd.read_parquet(args.human_relevance)
        required = {"sample_id", "qid", "h"}
        missing = required - set(human_df.columns)
        if missing:
            raise SystemExit(f"human_relevance parquet missing columns: {sorted(missing)}")

        if not args.allow_holdout_leak:
            human_sids = set(human_df["sample_id"].astype(str).tolist())
            oracle_sids = set(_sample_ids_for_leak_check(s_df, winner_col="winner_gt") or [])
            extra = human_sids - oracle_sids
            if extra:
                log.warning(
                    "human_relevance has %d sample_ids not in oracle and will be ignored: %s",
                    len(extra),
                    sorted(extra)[:5],
                )

    tensors = _prepare_oracle_tensors(
        q_df=q_df,
        s_df=s_df,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        only_oracle_correct_for_ranking=args.only_oracle_correct_for_ranking,
        target_mode=args.target_mode,
        human_relevance_df=human_df,
        oracle_fallback_eps=args.oracle_fallback_eps,
    )

    log.info("Oracle samples: %d  questions: %d", len(tensors.sample_ids), len(tensors.qids))

    t0_embed = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.encoder_model, trust_remote_code=True).to(device)

    sample_emb = _encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=tensors.sample_texts,
        max_length=args.max_length,
        batch_size=args.embed_batch_size,
        device=device,
        normalize=args.normalize_emb,
    )
    q_emb = _encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=tensors.question_texts,
        max_length=min(args.max_length, 256),
        batch_size=args.embed_batch_size,
        device=device,
        normalize=args.normalize_emb,
    )
    embed_elapsed = time.time() - t0_embed
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    emb_dim = sample_emb.shape[1]
    head = SelectorHead(
        emb_dim=emb_dim,
        hidden_dim=args.hidden_dim,
        n_dims=len(tensors.dim_vocab),
        dropout=args.dropout,
    ).to(device)

    q_emb_dev = q_emb.to(device)

    train_idx, val_idx = _split_indices(len(tensors.sample_ids), args.val_ratio, args.seed)
    train_ds = EmbeddingDataset(
        sample_emb=sample_emb,
        rank_target=tensors.rank_target,
        u2_target=tensors.u2_target,
        ans_target=tensors.ans_target,
        active_mask=tensors.active_mask,
        ranking_sample_mask=tensors.ranking_sample_mask,
        indices=train_idx,
    )
    val_ds = EmbeddingDataset(
        sample_emb=sample_emb,
        rank_target=tensors.rank_target,
        u2_target=tensors.u2_target,
        ans_target=tensors.ans_target,
        active_mask=tensors.active_mask,
        ranking_sample_mask=tensors.ranking_sample_mask,
        indices=val_idx,
    )

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.train_batch_size, shuffle=False) if val_idx else None

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        epoch_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            s = batch["sample_emb"].to(device)
            y_rank = batch["rank_target"].to(device)
            y_ans = batch["ans_target"].to(device)
            mask = batch["active_mask"].to(device)
            rank_sample_mask = batch["ranking_sample_mask"].to(device)

            rank_logits, ans_logits, dim_logits = head(s, q_emb_dev)

            loss_rank = _listmle_loss(rank_logits, y_rank, mask, rank_sample_mask)

            if mask.any():
                loss_ans = F.binary_cross_entropy_with_logits(ans_logits[mask], y_ans[mask])
                dim_targets = tensors.q_dim_ids.to(device).unsqueeze(0).expand_as(ans_logits)
                loss_dim = F.cross_entropy(dim_logits[mask], dim_targets[mask])
            else:
                loss_ans = rank_logits.new_tensor(0.0)
                loss_dim = rank_logits.new_tensor(0.0)

            loss = args.lambda_rank * loss_rank + args.lambda_ans * loss_ans + args.lambda_dim * loss_dim

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_steps += 1

        avg_train_loss = epoch_loss / max(1, n_steps)

        eval_metrics: dict[str, float] = {}
        if val_loader is not None:
            head.eval()
            with torch.no_grad():
                all_scores: list[torch.Tensor] = []
                all_rank: list[torch.Tensor] = []
                all_u2: list[torch.Tensor] = []
                all_mask: list[torch.Tensor] = []
                for batch in val_loader:
                    s = batch["sample_emb"].to(device)
                    rank_logits, _, _ = head(s, q_emb_dev)
                    all_scores.append(rank_logits.cpu())
                    all_rank.append(batch["rank_target"].cpu())
                    all_u2.append(batch["u2_target"].cpu())
                    all_mask.append(batch["active_mask"].cpu())

                score_mat = torch.cat(all_scores, dim=0)
                rank_mat = torch.cat(all_rank, dim=0)
                u2_mat = torch.cat(all_u2, dim=0)
                mask_mat = torch.cat(all_mask, dim=0)
                # eval against the same target the model was trained on
                eval_metrics = _compute_ndcg_recall(score_mat, rank_mat, mask_mat)
                # also report u2-based metrics for cross-mode comparison
                u2_metrics = _compute_ndcg_recall(score_mat, u2_mat, mask_mat)
                for k, v in u2_metrics.items():
                    eval_metrics[f"{k}_u2"] = v

        row = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **eval_metrics,
        }
        history.append(row)
        log.info(
            "Epoch %d/%d  loss=%.4f  %s",
            epoch,
            args.epochs,
            avg_train_loss,
            "  ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items()) if eval_metrics else "",
        )

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(head.state_dict(), out_dir / "selector_head.pt")
    torch.save(
        {
            "qids": tensors.qids,
            "question_embeddings": q_emb,
        },
        out_dir / "q_embeds.pt",
    )
    tensors.bank_df.to_parquet(out_dir / "bank_index.parquet", index=False)

    cfg_payload = {
        "encoder_model": args.encoder_model,
        "max_length": args.max_length,
        "normalize_emb": args.normalize_emb,
        "emb_dim": int(emb_dim),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "dim_vocab": tensors.dim_vocab,
        "qids": tensors.qids,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "lambda_rank": args.lambda_rank,
        "lambda_ans": args.lambda_ans,
        "lambda_dim": args.lambda_dim,
        "seed": args.seed,
        "embed_time_s": embed_elapsed,
        "n_samples": len(tensors.sample_ids),
        "n_questions": len(tensors.qids),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "only_oracle_correct_for_ranking": args.only_oracle_correct_for_ranking,
        "target_mode": args.target_mode,
        "human_relevance": str(args.human_relevance) if args.human_relevance else None,
        "oracle_fallback_eps": args.oracle_fallback_eps,
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_payload, f, indent=2, ensure_ascii=False)

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    summary = {
        "last_epoch": history[-1] if history else {},
        "best_ndcg@20": float(max((h.get("ndcg@20", 0.0) for h in history), default=0.0)),
        "best_recall@20": float(max((h.get("recall@20", 0.0) for h in history), default=0.0)),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("Saved selector checkpoint -> %s", out_dir)


if __name__ == "__main__":
    main()
