"""
No-gold scorer for self-checklist traces (STaR / Expert-Iteration usage).

The existing reward ORMs in ``judge_selfcheck_reward.py`` all require a gold
``winner`` column. This module provides a parallel set of helpers that derive
a *pseudo* winner from the model's own rollouts (K samples + swap-order
agreement) and score each trace against that pseudo-winner — no gold labels
ever touched.

Used by:
  - ``data_process/build_selfchk_ei_data.py`` to build expert-iteration SFT data.
  - Future GRPO reward (online self-supervised variant) that batches K rollouts
    per prompt and feeds the group into ``score_group``.

Reward components (per trace, given pseudo Ŵ ∈ {A,B} or None):
  parse_ok                : 1 if checklist + verdicts + final all parse cleanly.
  final_matches_pseudo    : 1 if τ.winner == Ŵ.
  item_margin_pseudo      : (n_Ŵ - n_other) / n_total in [-1, 1].
  diversity               : 1 - mean pairwise cosine over checklist Qs in [0, 1].
  discriminative          : fraction of verdicts that take A or B (not Tie).
  length_band             : 1 if n_q in [min_q, max_q] else linear decay.
  dominance_penalty       : negative when one side > threshold of all verdicts.

Pair-level signals (added equally to every accepted trace in a pair):
  swap_consistency        : fraction of orig vs permuted-swap rollouts that agree on Ŵ.
  pseudo_confidence       : max-vote / total-votes across orig+swap rollouts.

Defaults are tuned so a perfectly-formed trace that matches the pseudo winner
on a high-confidence pair lands near +1.0.
"""
from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent.parent
for _p in (_SRC_DIR, _SRC_DIR / "data_process"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_process.prepare_self_checklist_sft import parse_self_checklist_trace

log = logging.getLogger(__name__)


# ── Defaults ──

DEFAULTS = {
    "min_q": 6,
    "max_q": 25,
    "soft_max_q": 20,
    "dominance_threshold": 0.90,
    "dominance_penalty": 0.10,
    "weight_parse_ok": 0.10,
    "weight_final": 0.45,
    "weight_item_margin": 0.15,
    "weight_diversity": 0.10,
    "weight_discriminative": 0.10,
    "weight_length": 0.05,
    "weight_swap_consistency": 0.10,
    "weight_pseudo_confidence": 0.05,
    "min_pseudo_confidence": 0.6,
    "min_swap_consistency": 0.5,
}


# ── Pseudo-winner derivation ──

def permute_winner_for_swap(w: str | None) -> str | None:
    """Remap a verdict produced on (B,A) order back to original (A,B) frame."""
    if w == "A":
        return "B"
    if w == "B":
        return "A"
    return w  # Tie or None


def derive_pseudo_winner(
    orig_winners: Iterable[str | None],
    swap_winners: Iterable[str | None],
    *,
    min_confidence: float = DEFAULTS["min_pseudo_confidence"],
    min_swap_consistency: float = DEFAULTS["min_swap_consistency"],
) -> dict:
    """Aggregate K-sample rollouts under both orders into a pseudo winner.

    Returns:
        {
          "pseudo_winner": "A" | "B" | None,   # None ⇒ pair rejected
          "confidence":     float,             # in [0, 1]
          "swap_consistency": float,           # in [0, 1]
          "vote_counts":   {"A": int, "B": int, "Tie": int, "None": int},
          "n_orig_valid":  int,
          "n_swap_valid":  int,
        }
    """
    orig = list(orig_winners)
    swap = list(swap_winners)
    swap_remapped = [permute_winner_for_swap(w) for w in swap]

    valid_orig = [w for w in orig if w in ("A", "B")]
    valid_swap = [w for w in swap_remapped if w in ("A", "B")]

    all_votes = [w for w in (orig + swap_remapped) if w is not None]
    if not all_votes:
        return {
            "pseudo_winner": None,
            "confidence": 0.0,
            "swap_consistency": 0.0,
            "vote_counts": {"A": 0, "B": 0, "Tie": 0, "None": 0},
            "n_orig_valid": 0,
            "n_swap_valid": 0,
        }

    counts = Counter(all_votes)
    none_n = sum(1 for w in (orig + swap_remapped) if w is None)
    vote_counts = {
        "A": counts.get("A", 0),
        "B": counts.get("B", 0),
        "Tie": counts.get("Tie", 0),
        "None": none_n,
    }

    ab_total = vote_counts["A"] + vote_counts["B"]
    if ab_total == 0:
        # All Tie / None — skip; final pseudo winner must be A or B for SFT.
        return {
            "pseudo_winner": None,
            "confidence": 0.0,
            "swap_consistency": 0.0,
            "vote_counts": vote_counts,
            "n_orig_valid": len(valid_orig),
            "n_swap_valid": len(valid_swap),
        }

    pseudo = "A" if vote_counts["A"] >= vote_counts["B"] else "B"
    confidence = max(vote_counts["A"], vote_counts["B"]) / ab_total

    if valid_orig and valid_swap:
        orig_majority = Counter(valid_orig).most_common(1)[0][0]
        swap_majority = Counter(valid_swap).most_common(1)[0][0]
        agree_orig = sum(1 for w in valid_orig if w == orig_majority) / len(valid_orig)
        agree_swap = sum(1 for w in valid_swap if w == swap_majority) / len(valid_swap)
        cross = 1.0 if orig_majority == swap_majority else 0.0
        swap_consistency = 0.5 * (agree_orig + agree_swap) * cross
    elif valid_orig or valid_swap:
        # Only one side has valid rollouts: cannot verify positional invariance.
        swap_consistency = 0.0
    else:
        swap_consistency = 0.0

    accepted = (
        confidence >= min_confidence
        and swap_consistency >= min_swap_consistency
    )

    return {
        "pseudo_winner": pseudo if accepted else None,
        "confidence": float(confidence),
        "swap_consistency": float(swap_consistency),
        "vote_counts": vote_counts,
        "n_orig_valid": len(valid_orig),
        "n_swap_valid": len(valid_swap),
    }


# ── Per-trace scoring ──

def _length_band(n_q: int, min_q: int, max_q: int, soft_max_q: int) -> float:
    """1 inside [min_q, soft_max_q]; linear decay to 0 at max_q; 0 below min_q."""
    if n_q < min_q:
        return 0.0
    if n_q <= soft_max_q:
        return 1.0
    if n_q >= max_q:
        return 0.0
    return max(0.0, 1.0 - (n_q - soft_max_q) / max(1, max_q - soft_max_q))


def _strict_parse_ok(parsed: dict, min_q: int, max_q: int) -> bool:
    if not parsed.get("checklist"):
        return False
    if not parsed.get("verdicts"):
        return False
    if parsed.get("winner") not in ("A", "B"):
        return False
    if not parsed.get("checklist_matched"):
        return False
    n = int(parsed.get("n_questions") or 0)
    if n < min_q or n > max_q:
        return False
    expected = set(range(1, n + 1))
    return set(parsed["verdicts"].keys()) == expected


def _item_margin(parsed: dict, pseudo: str) -> float:
    verdicts = parsed.get("verdicts") or {}
    n = len(verdicts)
    if n == 0 or pseudo not in ("A", "B"):
        return 0.0
    n_w = sum(1 for v in verdicts.values() if v == pseudo)
    n_l = sum(1 for v in verdicts.values() if v in ("A", "B") and v != pseudo)
    return (n_w - n_l) / n


def _discriminative(parsed: dict) -> float:
    verdicts = parsed.get("verdicts") or {}
    n = len(verdicts)
    if n == 0:
        return 0.0
    return sum(1 for v in verdicts.values() if v in ("A", "B")) / n


def _dominance(parsed: dict, threshold: float, penalty: float) -> float:
    verdicts = parsed.get("verdicts") or {}
    n = len(verdicts)
    if n == 0:
        return 0.0
    n_a = sum(1 for v in verdicts.values() if v == "A")
    n_b = sum(1 for v in verdicts.values() if v == "B")
    if max(n_a, n_b) / n > threshold:
        return -abs(penalty)
    return 0.0


def _diversity(parsed: dict, encoder) -> float:
    qs = parsed.get("checklist") or []
    n = len(qs)
    if n <= 1 or encoder is None:
        return 0.0 if n <= 1 else 1.0
    emb = encoder.encode(qs, normalize_embeddings=True, batch_size=64)
    emb = np.asarray(emb, dtype=np.float32)
    sim = emb @ emb.T
    mask = ~np.eye(n, dtype=bool)
    return max(0.0, min(1.0, 1.0 - float(sim[mask].mean())))


def score_trace(
    raw: str,
    pseudo_winner: str | None,
    *,
    swap_consistency: float = 0.0,
    pseudo_confidence: float = 0.0,
    encoder=None,
    weights: dict | None = None,
) -> dict:
    """Score one rollout against the pair's pseudo winner.

    Returns a dict with ``reward`` and per-component diagnostics. Trace whose
    ``winner != pseudo_winner`` still gets a reward but with the final-match
    bonus zeroed — useful for ranking near-misses.
    """
    w = {**DEFAULTS, **(weights or {})}
    parsed = parse_self_checklist_trace(raw or "")

    parse_ok = _strict_parse_ok(parsed, w["min_q"], w["max_q"])
    parse_ok_f = 1.0 if parse_ok else 0.0
    n_q = int(parsed.get("n_questions") or 0)
    pred = parsed.get("winner")

    final_match = 1.0 if (pred == pseudo_winner and pseudo_winner in ("A", "B")) else 0.0

    item_margin = _item_margin(parsed, pseudo_winner) if pseudo_winner else 0.0
    disc = _discriminative(parsed)
    div = _diversity(parsed, encoder) if parse_ok else 0.0
    length = _length_band(n_q, w["min_q"], w["soft_max_q"], w["max_q"])
    dom = _dominance(parsed, w["dominance_threshold"], w["dominance_penalty"])

    reward = (
        w["weight_parse_ok"] * parse_ok_f
        + w["weight_final"] * final_match
        + w["weight_item_margin"] * item_margin * parse_ok_f
        + w["weight_diversity"] * div
        + w["weight_discriminative"] * disc * parse_ok_f
        + w["weight_length"] * length
        + w["weight_swap_consistency"] * swap_consistency
        + w["weight_pseudo_confidence"] * pseudo_confidence
        + dom
    )

    return {
        "reward": float(reward),
        "parse_ok": parse_ok,
        "n_questions": n_q,
        "pred_winner": pred,
        "final_match": final_match,
        "item_margin": float(item_margin),
        "diversity": float(div),
        "discriminative": float(disc),
        "length_band": float(length),
        "dominance_penalty": float(dom),
        "parsed": parsed,
    }


# ── Group-level driver ──

def score_group(
    raws_orig: list[str],
    raws_swap: list[str],
    *,
    encoder=None,
    weights: dict | None = None,
) -> dict:
    """Score a full K-sample group for one prompt.

    Args:
        raws_orig: K rollouts on (A, B) order.
        raws_swap: K rollouts on (B, A) order.

    Returns:
        {
          "pseudo": {...},                          # output of derive_pseudo_winner
          "orig_scores": list[dict],                # score_trace per orig rollout
          "swap_scores": list[dict],                # score_trace per swap rollout (raw winner not permuted)
          "best_orig_idx": int | None,              # index of highest-reward orig trace, or None
          "best_orig_reward": float,
        }
    """
    w = {**DEFAULTS, **(weights or {})}

    orig_parsed = [parse_self_checklist_trace(r or "") for r in raws_orig]
    swap_parsed = [parse_self_checklist_trace(r or "") for r in raws_swap]

    pseudo = derive_pseudo_winner(
        [p.get("winner") for p in orig_parsed],
        [p.get("winner") for p in swap_parsed],
        min_confidence=w["min_pseudo_confidence"],
        min_swap_consistency=w["min_swap_consistency"],
    )

    orig_scores = [
        score_trace(
            r,
            pseudo["pseudo_winner"],
            swap_consistency=pseudo["swap_consistency"],
            pseudo_confidence=pseudo["confidence"],
            encoder=encoder,
            weights=w,
        )
        for r in raws_orig
    ]

    # Swap scores are diagnostic only; we never SFT on swap-order outputs since
    # the prompt shows responses in (B, A) order — student must learn under the
    # canonical (A, B) order. Pseudo winner is permuted for fair item-margin.
    swap_pseudo_for_swap = permute_winner_for_swap(pseudo["pseudo_winner"])
    swap_scores = [
        score_trace(
            r,
            swap_pseudo_for_swap,
            swap_consistency=pseudo["swap_consistency"],
            pseudo_confidence=pseudo["confidence"],
            encoder=encoder,
            weights=w,
        )
        for r in raws_swap
    ]

    if pseudo["pseudo_winner"] is None:
        best_idx, best_r = None, float("-inf")
    else:
        candidates = [
            (i, s["reward"]) for i, s in enumerate(orig_scores)
            if s["parse_ok"] and s["pred_winner"] == pseudo["pseudo_winner"]
        ]
        if candidates:
            best_idx, best_r = max(candidates, key=lambda kv: kv[1])
        else:
            best_idx, best_r = None, float("-inf")

    return {
        "pseudo": pseudo,
        "orig_scores": orig_scores,
        "swap_scores": swap_scores,
        "best_orig_idx": best_idx,
        "best_orig_reward": float(best_r if best_idx is not None else 0.0),
    }


# ── Diversity encoder loader ──

def load_diversity_encoder(name: str | None = None, device: str = "cpu"):
    """Lazy load a sentence-transformers encoder for diversity scoring."""
    from sentence_transformers import SentenceTransformer
    name = name or os.environ.get(
        "JUDGE_GRPO_DIVERSITY_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    log.info("Loading diversity encoder: %s on %s", name, device)
    return SentenceTransformer(name, device=device)
