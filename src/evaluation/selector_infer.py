#!/usr/bin/env python3
"""Run selector inference and export top-k question picks per sample.

Usage:
    python src/evaluation/selector_infer.py \
        --selector results/checkpoints/selector_v1 \
        --split dev_600 --k 20 \
        --out results/dynamic_dev_600/selector_topk_dev600.parquet
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(
    0,
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data_process"),
)

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - dependency availability is environment-specific
    AutoModel = None
    AutoTokenizer = None

import config as cfg
from prepare_data_reasoning import make_sample_id
from utils import _select_dimensions

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


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

    def forward(self, sample_emb: torch.Tensor, q_emb: torch.Tensor) -> torch.Tensor:
        bsz, emb_dim = sample_emb.shape
        n_q = q_emb.shape[0]
        s = sample_emb.unsqueeze(1).expand(bsz, n_q, emb_dim)
        q = q_emb.unsqueeze(0).expand(bsz, n_q, emb_dim)
        feat = torch.cat([s, q, s * q], dim=-1)
        hid = self.backbone(feat)
        return self.rank_head(hid).squeeze(-1)


@dataclass
class SelectorBundle:
    selector_dir: Path
    config: dict
    bank_df: pd.DataFrame
    qids: list[int]
    qid_to_idx: dict[int, int]
    q_emb: torch.Tensor
    head: SelectorHead
    tokenizer: AutoTokenizer
    encoder: AutoModel
    device: torch.device


def build_sample_texts(df: pd.DataFrame) -> list[str]:
    return [
        "[Context]\n"
        + str(r["context"])
        + "\n\n[Response A]\n"
        + str(r["response_a"])
        + "\n\n[Response B]\n"
        + str(r["response_b"])
        for _, r in df.iterrows()
    ]


def load_eval_pairs(
    split: str,
    subset: str | None = None,
    input_path: Path | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    if input_path is not None:
        path = input_path
    elif subset and subset != "full":
        path = cfg.SPLITS_DIR / f"train_{subset}.parquet"
    else:
        path = cfg.SPLITS_DIR / f"{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    required = {"prompt_id", "domain", "context", "response_a", "response_b", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if max_samples is not None:
        df = df.head(max_samples).reset_index(drop=True)

    if "sample_id" not in df.columns:
        df["sample_id"] = df.apply(
            lambda r: make_sample_id(
                prompt_id=r["prompt_id"],
                response_a=r["response_a"],
                response_b=r["response_b"],
                winner=r["winner"],
            ),
            axis=1,
        )

    return df.reset_index(drop=True)


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _encode_texts(
    encoder: AutoModel,
    tokenizer: AutoTokenizer,
    texts: Iterable[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    normalize: bool,
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    text_list = list(texts)
    encoder.eval()

    for start in tqdm(range(0, len(text_list), batch_size), desc="Selector encode", leave=False):
        batch = text_list[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = encoder(**enc).last_hidden_state
            emb = _mean_pool(h, enc["attention_mask"])
            if normalize:
                emb = F.normalize(emb, p=2, dim=-1)
            out.append(emb.cpu())

    return torch.cat(out, dim=0) if out else torch.zeros((0, 1), dtype=torch.float32)


def load_selector_bundle(selector_dir: Path, device: torch.device) -> SelectorBundle:
    if AutoModel is None or AutoTokenizer is None:
        raise SystemExit(
            "Missing dependency: transformers. Install it before running selector inference."
        )

    selector_dir = selector_dir.resolve()
    cfg_path = selector_dir / "config.json"
    q_embed_path = selector_dir / "q_embeds.pt"
    head_path = selector_dir / "selector_head.pt"
    bank_path = selector_dir / "bank_index.parquet"

    for p in [cfg_path, q_embed_path, head_path, bank_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    with cfg_path.open("r", encoding="utf-8") as f:
        conf = json.load(f)

    bank_df = pd.read_parquet(bank_path).sort_values("qid", kind="stable").reset_index(drop=True)

    q_blob = torch.load(q_embed_path, map_location="cpu")
    qids = [int(x) for x in q_blob["qids"]]
    q_emb = q_blob["question_embeddings"].float().to(device)
    qid_to_idx = {q: i for i, q in enumerate(qids)}

    head = SelectorHead(
        emb_dim=int(conf["emb_dim"]),
        hidden_dim=int(conf["hidden_dim"]),
        n_dims=len(conf["dim_vocab"]),
        dropout=float(conf.get("dropout", 0.1)),
    ).to(device)
    head.load_state_dict(torch.load(head_path, map_location=device))
    head.eval()

    tokenizer = AutoTokenizer.from_pretrained(conf["encoder_model"], trust_remote_code=True)
    encoder = AutoModel.from_pretrained(conf["encoder_model"], trust_remote_code=True).to(device)
    encoder.eval()

    return SelectorBundle(
        selector_dir=selector_dir,
        config=conf,
        bank_df=bank_df,
        qids=qids,
        qid_to_idx=qid_to_idx,
        q_emb=q_emb,
        head=head,
        tokenizer=tokenizer,
        encoder=encoder,
        device=device,
    )


def score_samples_with_bundle(
    bundle: SelectorBundle,
    sample_texts: list[str],
    batch_size: int,
) -> torch.Tensor:
    emb = _encode_texts(
        encoder=bundle.encoder,
        tokenizer=bundle.tokenizer,
        texts=sample_texts,
        max_length=int(bundle.config.get("max_length", 1024)),
        batch_size=batch_size,
        device=bundle.device,
        normalize=bool(bundle.config.get("normalize_emb", False)),
    ).to(bundle.device)

    scores: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, emb.shape[0], batch_size):
            chunk = emb[start : start + batch_size]
            s = bundle.head(chunk, bundle.q_emb)
            scores.append(s.cpu())

    return torch.cat(scores, dim=0)


def active_qids_for_domain(bank_df: pd.DataFrame, domain: str) -> list[int]:
    allowed = _select_dimensions(str(domain))
    allowed_lower = {d.lower() for d in allowed}
    dim_col = "dimension" if "dimension" in bank_df.columns else "dim"

    active = bank_df[
        bank_df[dim_col].map(lambda d: d in allowed or str(d).lower() in allowed_lower)
    ]
    if active.empty:
        active = bank_df
    return active["qid"].astype(int).tolist()


def select_topk_with_quota(
    ranked_active_qids: list[int],
    bank_df: pd.DataFrame,
    domain: str,
    k: int,
    enforce_quota: bool = True,
) -> list[int]:
    k_eff = min(k, len(ranked_active_qids))
    if k_eff <= 0:
        return []
    if not enforce_quota:
        return ranked_active_qids[:k_eff]

    dim_col = "dimension" if "dimension" in bank_df.columns else "dim"
    q_to_dim = {int(r["qid"]): str(r[dim_col]) for _, r in bank_df[["qid", dim_col]].iterrows()}

    allowed_dims = []
    seen: set[str] = set()
    for qid in ranked_active_qids:
        d = q_to_dim[qid]
        if d not in seen:
            seen.add(d)
            allowed_dims.append(d)

    n_dims = len(allowed_dims)
    if n_dims == 0:
        return ranked_active_qids[:k_eff]

    min_per_dim = k_eff // n_dims

    selected: list[int] = []
    used: set[int] = set()

    if min_per_dim > 0:
        for dim in allowed_dims:
            dim_q = [q for q in ranked_active_qids if q_to_dim[q] == dim]
            for q in dim_q[:min_per_dim]:
                if q not in used:
                    selected.append(q)
                    used.add(q)

    for q in ranked_active_qids:
        if len(selected) >= k_eff:
            break
        if q in used:
            continue
        selected.append(q)
        used.add(q)

    return selected[:k_eff]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--split", type=str, default="dev_600")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-dim-quota", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    bundle = load_selector_bundle(args.selector, device=device)

    df = load_eval_pairs(
        split=args.split,
        subset=args.subset,
        input_path=args.input_path,
        max_samples=args.max_samples,
    )
    sample_texts = build_sample_texts(df)

    t0 = time.time()
    score_matrix = score_samples_with_bundle(bundle, sample_texts, batch_size=args.batch_size)
    elapsed = time.time() - t0

    rows: list[dict] = []
    for i, row in df.iterrows():
        domain = str(row["domain"])
        active_qids = active_qids_for_domain(bundle.bank_df, domain)

        qidx = torch.tensor([bundle.qid_to_idx[q] for q in active_qids], dtype=torch.long)
        s = score_matrix[i, qidx]
        order = torch.argsort(s, descending=True)
        ranked_active_qids = [active_qids[int(j)] for j in order.tolist()]

        selected_qids = select_topk_with_quota(
            ranked_active_qids=ranked_active_qids,
            bank_df=bundle.bank_df,
            domain=domain,
            k=args.k,
            enforce_quota=(not args.no_dim_quota),
        )

        rows.append(
            {
                "sample_id": row["sample_id"],
                "prompt_id": row["prompt_id"],
                "domain": domain,
                "k": int(args.k),
                "selected_qids": selected_qids,
                "ranked_qids": ranked_active_qids,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    meta = {
        "selector": str(args.selector.resolve()),
        "split": args.split,
        "subset": args.subset,
        "k": args.k,
        "n_samples": len(out_df),
        "inference_time_s": elapsed,
        "samples_per_second": (len(out_df) / elapsed) if elapsed > 0 else None,
        "dim_quota": not args.no_dim_quota,
    }
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("Saved selector inference -> %s", out_path)


if __name__ == "__main__":
    main()
