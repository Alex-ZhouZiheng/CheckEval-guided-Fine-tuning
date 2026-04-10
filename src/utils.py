"""
Shared utilities for baseline inference scripts.

Includes:
- vLLM model loading
- Prompt formatting helpers
- Batched chat generation
- Result persistence and metrics computation
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any
from tqdm import tqdm

import pandas as pd
from rich.console import Console
from sklearn.metrics import accuracy_score, classification_report, f1_score

import config as cfg

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)
console = Console()


# ────────────────────────── data loading ─────────────────────


def load_eval_data(eval_split: str = "test", subset: str | None = None) -> pd.DataFrame:
    """Load evaluation data from parquet splits.

    Parameters
    ----------
    eval_split : str
        Which standard split to load ("train", "dev", or "test").
    subset : str or None
        If provided and not "full", loads a training tier subset
        (e.g. "debug_5k", "tier_10k").
    """
    if subset and subset != "full":
        path = cfg.SPLITS_DIR / f"train_{subset}.parquet"
    else:
        path = cfg.SPLITS_DIR / f"{eval_split}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    df = pd.read_parquet(path)
    log.info("Loaded %s pairs from %s", f"{len(df):,}", path.name)
    return df


# ────────────────────────── model loading ────────────────────


def load_judge_model(
    model_id: str = cfg.JUDGE_MODEL_ID,
    cache_dir: str | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    dtype: str | None = None,
    language_model_only: bool | None = None,
    max_num_seqs: int | None = None,
    enable_prefix_caching: bool | None = True,
    max_num_batched_tokens: int | None = 16384,
    num_gpu_blocks_override: int | None = None,
    enable_lora: bool = False,
    max_lora_rank: int | None = None,
    max_loras: int | None = None,
) -> LLM:
    """Load the judge LLM through vLLM."""
    from vllm import LLM

    model_id = str(model_id)

    engine_kwargs = dict(cfg.VLLM_ENGINE_KWARGS)
    overrides = {
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "dtype": dtype,
        "language_model_only": language_model_only,
        "max_num_seqs": max_num_seqs,
        "enable_prefix_caching": enable_prefix_caching,
        "max_num_batched_tokens": max_num_batched_tokens,
        "num_gpu_blocks_override": num_gpu_blocks_override,
    }
    engine_kwargs.update({k: v for k, v in overrides.items() if v is not None})

    if enable_lora:
        engine_kwargs["enable_lora"] = True
        if max_lora_rank is not None:
            engine_kwargs["max_lora_rank"] = max_lora_rank
        if max_loras is not None:
            engine_kwargs["max_loras"] = max_loras

    if num_gpu_blocks_override is None:
        engine_kwargs.pop("num_gpu_blocks_override", None)

    if cache_dir is not None:
        engine_kwargs["download_dir"] = cache_dir

    log.info("Loading model with vLLM: %s", model_id)
    log.info(
        "  tp=%s  max_model_len=%s  gpu_mem=%.2f  max_num_seqs=%s  "
        "max_num_batched_tokens=%s  prefix_caching=%s",
        engine_kwargs.get("tensor_parallel_size"),
        engine_kwargs.get("max_model_len"),
        engine_kwargs.get("gpu_memory_utilization"),
        engine_kwargs.get("max_num_seqs"),
        engine_kwargs.get("max_num_batched_tokens"),
        engine_kwargs.get("enable_prefix_caching"),
    )

    t0 = time.time()
    model = LLM(
        model=model_id,
        tokenizer=model_id,
        **engine_kwargs,
    )
    elapsed = time.time() - t0
    log.info("  Model loaded in %.1fs", elapsed)
    return model


# ────────────────────────── generation ───────────────────────


def build_sampling_params(**gen_kwargs) -> SamplingParams:
    """Convert shared generation kwargs into vLLM SamplingParams."""
    from vllm import SamplingParams
    merged = {**cfg.GENERATION_KWARGS, **gen_kwargs}

    if "max_new_tokens" in merged and "max_tokens" not in merged:
        merged["max_tokens"] = merged.pop("max_new_tokens")

    merged.pop("do_sample", None)

    allowed_keys = {
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "stop",
        "stop_token_ids",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
    }
    sampling_kwargs = {
        k: v for k, v in merged.items() if k in allowed_keys and v is not None
    }
    return SamplingParams(**sampling_kwargs)


def _extract_output_text(output: Any) -> str:
    """Extract the first completion text from a vLLM RequestOutput."""
    if not getattr(output, "outputs", None):
        return ""
    return output.outputs[0].text.strip()


def generate_single(
    model: LLM,
    messages: list[dict[str, str]],
    **gen_kwargs,
) -> str:
    """Generate a response for a single chat-formatted input."""
    return generate_batch(model, [messages], batch_size=1, **gen_kwargs)[0]


def generate_batch(
    model: LLM,
    messages_list: list[list[dict[str, str]]],
    batch_size: int = 16,
    lora_request: Any = None,
    **gen_kwargs,
) -> list[str]:
    """Batched generation for a list of chat-formatted inputs via vLLM."""
    sampling_params = build_sampling_params(**gen_kwargs)
    chat_kwargs = dict(cfg.VLLM_CHAT_KWARGS)
    if lora_request is not None:
        chat_kwargs["lora_request"] = lora_request
    outputs = model.chat(
        messages_list,
        sampling_params=sampling_params,
        use_tqdm=True,
        add_generation_prompt=True,
        **chat_kwargs,
    )
    return [_extract_output_text(output) for output in outputs]


# ────────────────────────── prompts ──────────────────────────


VANILLA_JUDGE_PROMPT = """\
<Task Overview>
Your task is to evaluate two candidate responses to the same user request and decide which response is better overall.

<Evaluation Definition>
Judge overall response quality based on instruction following, relevance, correctness, completeness, clarity, and reasoning quality when applicable.

<Instructions>

1. Read the User Request and both responses carefully.
2. Compare Response A and Response B based only on their quality for this request.
3. Do not be influenced by response order, response length, or formatting alone.
4. Choose [[A]] if Response A is better overall.
5. Choose [[B]] if Response B is better overall.
6. Choose [[Tie]] only if the two responses are equally strong or equally weak.
7. Keep the explanation concise.
8. Output exactly in the required format.

<Answer Format>
Verdict: A or B

### Conversation Context
{context}

### Response A
{response_a}

### Response B
{response_b}

Which response is better? Answer with ONLY "A" or "B"."""


def build_vanilla_prompt(row: dict | pd.Series) -> str:
    """Format a single pairwise comparison prompt."""
    return VANILLA_JUDGE_PROMPT.format(
        context=row["context"],
        response_a=row["response_a"],
        response_b=row["response_b"],
    )

CHECKEVAL_POINTWISE_PROMPT = """\
<Task Overview>
You will be given a conversation between a user and an assistant, followed by \
a candidate response for the next turn. Your task is to read the conversation \
history and the response, then answer 'yes','no' or 'N/A' to specific quality \
checklist questions about that response.

<Dimension Definitions>
{dimension_block}

<Instructions>
1. Read these instructions thoroughly.
2. Carefully read the Conversation History and the Response.
3. Understand the given questions and the definitions of each dimension.
4. For each question, answer 'yes','no' or 'N/A' based on a clear rationale.
5. Follow the specified format for your answers.
6. Use exactly one label for each question: yes, no, or N/A.
7. Use N/A only when the question's condition does not apply to this case at all.
8. If the question applies but the requirement is not satisfied, answer no.
9. If the question applies and the requirement is satisfied, answer yes.
10. Do not provide explanations, rationale, notes, or extra text.
11. Output only the answer lines.

Additional rule for N/A:
- Use N/A only when the prerequisite of the question is absent.
- Do NOT use N/A merely because the response is poor, incomplete, uncertain, or hard to judge.
- When in doubt between no and N/A, choose no unless the question is clearly inapplicable.

<Answer Format>
Q1: yes
Q2: no
Q3: N/A
...

# Conversation History #

{context}

# Response #

{response}

# Questions #

{checklist_text}

# Your Answer #

Following the specified Answer Format.
Think silently, then output ONLY the final answer lines.
"""



def load_checklists(checklists_dir: Path = cfg.CHECKLISTS_DIR) -> dict[str, list[str]]:
    """Load filtered checklist questions from YAML files.

    Returns {dimension_name: [question, ...]}.
    """
    import yaml

    checklists: dict[str, list[str]] = {}
    definitions: dict[str, str] = {}
    for yaml_path in sorted(checklists_dir.glob("*_filtered.yaml")):
        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        dim_name = data.get("dimension", yaml_path.stem)
        definitions[dim_name] = data.get("definition", "")
        questions = []
        for sub_data in data.get("sub_aspects", {}).values():
            seen = set()
            for question in sub_data.get("filtered_questions", []):
                question_text = question.strip()
                if question_text not in seen:
                    seen.add(question_text)
                    questions.append(question_text)
        checklists[dim_name] = questions
    return checklists, definitions


_SHARED_DIMS = frozenset({
    "clarity_and_communication",
    "correctness_and_completeness",
    "helpfulness_and_usefulness",
})
_CODE_DIMS = frozenset({"coding_communication_conditional"})


def _select_dimensions(domain: str) -> frozenset[str]:
    """Return the set of dimension names applicable to *domain*."""
    if domain == "code":
        return _SHARED_DIMS | _CODE_DIMS
    return _SHARED_DIMS


def build_checkeval_prompt(
    row: dict | pd.Series,
    checklists: dict[str, list[str]],
    definitions: dict[str, str] | None = None,
    domain: str | None = None,
    side: str = "A",
):
    """Build a pointwise CheckEval prompt for one response.

    Parameters
    ----------
    side : str
        "A" or "B" – selects which response column to evaluate.
    """
    domain = domain or row.get("domain", "general")
    definitions = definitions or {}
    allowed = _select_dimensions(domain)

    dim_lines: list[str] = []
    all_questions: list[str] = []
    for dim_name, questions in checklists.items():
        if dim_name not in allowed and dim_name.lower() not in {d.lower() for d in allowed}:
            continue
        defn = definitions.get(dim_name, "")
        dim_lines.append(f"{dim_name} - {defn}" if defn else dim_name)
        all_questions.extend(questions)

    dimension_block = "\n".join(dim_lines)
    checklist_lines = [f"Q{i + 1}: {q}" for i, q in enumerate(all_questions)]
    checklist_text = "\n".join(checklist_lines)

    response_key = "response_a" if side.upper() == "A" else "response_b"

    return CHECKEVAL_POINTWISE_PROMPT.format(
        dimension_block=dimension_block,
        context=row["context"],
        response=row[response_key],
        checklist_text=checklist_text,
    )


# ────────────────────────── parsing ──────────────────────────

_LINE_RE = re.compile(r"^\s*Q(\d+):\s*(yes|no)\s*$", re.IGNORECASE)
_NA_RE = re.compile(
    r"^\s*Q(\d+):\s*(?:N\s*/\s*A|NA|not\s*applicable)\s*[.。]?\s*$",
    re.IGNORECASE,
)
# Bare yes/no/N/A without Q-prefix (e.g. gpt-5.4-mini sometimes omits "Q{n}: ")
_BARE_YN_RE = re.compile(r"^\s*(yes|no)\s*$", re.IGNORECASE)
_BARE_NA_RE = re.compile(
    r"^\s*(?:N\s*/\s*A|NA|not\s*applicable)\s*[.。]?\s*$",
    re.IGNORECASE,
)


def parse_checkeval_output(raw: str, expected_n: int | None = None) -> dict:
    """Parse checklist output, tolerating N/A lines instead of stopping."""

    answers: list[dict] = []
    na_answers: list[dict] = []
    seen: set[int] = set()
    started = False
    hit_na = False
    n_duplicates = 0
    n_out_of_range = 0
    stop_reason: str | None = None
    first_bad_line: str | None = None
    first_bad_lineno: int | None = None
    positional_q = 0          # counter for bare yes/no/N/A lines

    for lineno, line in enumerate(raw.splitlines(), 1):
        stripped = line.strip()

        if not stripped:
            continue

        # ── try yes/no match ──
        m = _LINE_RE.match(stripped)
        if m:
            started = True
            q = int(m.group(1))
            ans = m.group(2).lower()

            if expected_n is not None and not (1 <= q <= expected_n):
                n_out_of_range += 1
                continue

            if q in seen:
                n_duplicates += 1
                continue

            seen.add(q)
            answers.append({"q": q, "answer": ans})
            continue

        # ── try N/A match ──
        m_na = _NA_RE.match(stripped)
        if m_na:
            started = True
            hit_na = True
            q = int(m_na.group(1))

            if expected_n is not None and not (1 <= q <= expected_n):
                n_out_of_range += 1
                continue

            if q in seen:
                n_duplicates += 1
                continue

            seen.add(q)
            na_answers.append({"q": q})
            continue

        # ── try bare yes/no (no Q-prefix) → assign positional Q number ──
        m_bare = _BARE_YN_RE.match(stripped)
        if m_bare:
            started = True
            positional_q += 1
            q = positional_q
            ans = m_bare.group(1).lower()

            if expected_n is not None and not (1 <= q <= expected_n):
                n_out_of_range += 1
                continue

            if q in seen:
                n_duplicates += 1
                continue

            seen.add(q)
            answers.append({"q": q, "answer": ans})
            continue

        # ── try bare N/A (no Q-prefix) → assign positional Q number ──
        m_bare_na = _BARE_NA_RE.match(stripped)
        if m_bare_na:
            started = True
            hit_na = True
            positional_q += 1
            q = positional_q

            if expected_n is not None and not (1 <= q <= expected_n):
                n_out_of_range += 1
                continue

            if q in seen:
                n_duplicates += 1
                continue

            seen.add(q)
            na_answers.append({"q": q})
            continue

        # ── unrecognized line after block started → stop ──
        if started:
            stop_reason = "unrecognized_line_after_started"
            first_bad_line = stripped
            first_bad_lineno = lineno
            break

    # ── determine stop_reason if not already set ──
    if stop_reason is None:
        if not answers and not na_answers:
            stop_reason = "empty_output"
        elif not answers and na_answers:
            stop_reason = "out_of_range_only" if n_out_of_range > 0 else "empty_output"
        else:
            stop_reason = "completed"

    # ── fallback: nothing useful parsed ──
    if not answers:
        return {
            "_raw_fallback": True,
            "raw_text": raw,
            "n_questions_parsed": 0,
            "n_yes": 0,
            "n_na": len(na_answers),
            "na_qnums": sorted(a["q"] for a in na_answers),
            "score": 0.0,
            "stop_reason": stop_reason,
            "first_bad_line": first_bad_line,
            "first_bad_lineno": first_bad_lineno,
            "n_duplicates": n_duplicates,
            "n_out_of_range": n_out_of_range,
        }

    # ── sort by question id ──
    answers = sorted(answers, key=lambda x: x["q"])
    na_answers = sorted(na_answers, key=lambda x: x["q"])

    n_yes = sum(1 for a in answers if a["answer"] == "yes")
    n_no = sum(1 for a in answers if a["answer"] == "no")
    n_parsed = len(answers)
    n_na = len(na_answers)
    na_qnums = [a["q"] for a in na_answers]

    # ── strict complete: all yes/no, no N/A ──
    complete = False
    if expected_n is not None:
        complete = (
            n_parsed == expected_n
            and [a["q"] for a in answers] == list(range(1, expected_n + 1))
        )

    # ── complete_with_na: answered ∪ na covers {1..expected_n} exactly ──
    complete_with_na = False
    if expected_n is not None:
        answered_ids = {a["q"] for a in answers}
        na_ids = {a["q"] for a in na_answers}
        complete_with_na = (answered_ids | na_ids) == set(range(1, expected_n + 1))

    # ── refine stop_reason for N/A cases ──
    if stop_reason == "completed" and hit_na:
        stop_reason = "hit_na_but_continued"

    return {
        "answers": answers,
        "na_answers": na_answers,
        "n_questions_parsed": n_parsed,
        "n_yes": n_yes,
        "n_no": n_no,
        "n_na": n_na,
        "na_qnums": na_qnums,
        "score": n_yes / n_parsed,
        "complete": complete,
        "complete_with_na": complete_with_na,
        "stop_reason": stop_reason,
        "first_bad_line": first_bad_line,
        "first_bad_lineno": first_bad_lineno,
        "n_duplicates": n_duplicates,
        "n_out_of_range": n_out_of_range,
    }

def expected_question_count(domain: str, checklists: dict[str, list[str]]) -> int:
    allowed = _select_dimensions(domain)
    total = 0
    allowed_lower = {d.lower() for d in allowed}
    for dim_name, questions in checklists.items():
        if dim_name in allowed or dim_name.lower() in allowed_lower:
            total += len(questions)
    return total

def build_question_index(
    checklists: dict[str, list[str]],
    domain: str,
) -> dict[int, dict[str, str]]:
    """Map 1-based question ID to its dimension and text for *domain*.

    Uses the same iteration order as :func:`build_checkeval_prompt` so
    that Q-numbers match the prompt the model saw.
    """
    allowed = _select_dimensions(domain)
    allowed_lower = {d.lower() for d in allowed}

    qmap: dict[int, dict[str, str]] = {}
    qid = 1
    for dim_name, questions in checklists.items():
        if dim_name not in allowed and dim_name.lower() not in allowed_lower:
            continue
        for q in questions:
            qmap[qid] = {"dimension": dim_name, "question": q}
            qid += 1
    return qmap


def compute_question_diagnostics(
    parsed_a_list: list[dict],
    parsed_b_list: list[dict],
    domains: list[str],
    checklists: dict[str, list[str]],
) -> list[dict]:
    """Per-question yes/no/NA rates across all samples.

    Returns a list of dicts (one per question-domain combination) with:
    ``domain, dimension, qid, question_text, n_yes, n_no, n_na, n_total,
    yes_rate, no_rate, na_rate``.
    """
    # accumulator: (domain, qid) → {n_yes, n_no, n_na}
    acc: dict[tuple[str, int], dict[str, int]] = {}

    for parsed_a, parsed_b, domain in zip(parsed_a_list, parsed_b_list, domains):
        for parsed in (parsed_a, parsed_b):
            for a in parsed.get("answers", []):
                key = (domain, a["q"])
                if key not in acc:
                    acc[key] = {"n_yes": 0, "n_no": 0, "n_na": 0}
                if a["answer"] == "yes":
                    acc[key]["n_yes"] += 1
                else:
                    acc[key]["n_no"] += 1
            for a in parsed.get("na_answers", []):
                key = (domain, a["q"])
                if key not in acc:
                    acc[key] = {"n_yes": 0, "n_no": 0, "n_na": 0}
                acc[key]["n_na"] += 1

    rows = []
    for (domain, qid), counts in sorted(acc.items()):
        qindex = build_question_index(checklists, domain)
        info = qindex.get(qid, {"dimension": "?", "question": "?"})
        total = counts["n_yes"] + counts["n_no"] + counts["n_na"]
        rows.append({
            "domain": domain,
            "dimension": info["dimension"],
            "qid": qid,
            "question_text": info["question"],
            "n_yes": counts["n_yes"],
            "n_no": counts["n_no"],
            "n_na": counts["n_na"],
            "n_total": total,
            "yes_rate": counts["n_yes"] / total if total else 0.0,
            "no_rate": counts["n_no"] / total if total else 0.0,
            "na_rate": counts["n_na"] / total if total else 0.0,
        })
    return rows


def compute_dimension_accuracy(
    results_df: pd.DataFrame,
    checklists: dict[str, list[str]],
    parsed_a_list: list[dict],
    parsed_b_list: list[dict],
) -> dict[str, dict[str, Any]]:
    """Per-dimension accuracy: does each dimension independently predict the winner?

    For each sample where both sides have parsed answers, group questions by
    dimension, compute per-dimension score, determine dimension-level winner,
    and compare to ground truth.
    """
    # per-dimension accumulators
    dim_stats: dict[str, dict[str, Any]] = {}

    for idx, (_, row) in enumerate(results_df.iterrows()):
        pa = parsed_a_list[idx]
        pb = parsed_b_list[idx]

        if pa.get("_raw_fallback") or pb.get("_raw_fallback"):
            continue

        domain = row["domain"]
        ground_truth = row["winner"]
        qindex = build_question_index(checklists, domain)

        # group answers by dimension
        dim_yes_a: dict[str, int] = {}
        dim_n_a: dict[str, int] = {}
        dim_na_a: dict[str, int] = {}
        for a in pa.get("answers", []):
            dim = qindex.get(a["q"], {}).get("dimension", "?")
            dim_yes_a[dim] = dim_yes_a.get(dim, 0) + (1 if a["answer"] == "yes" else 0)
            dim_n_a[dim] = dim_n_a.get(dim, 0) + 1
        for a in pa.get("na_answers", []):
            dim = qindex.get(a["q"], {}).get("dimension", "?")
            dim_na_a[dim] = dim_na_a.get(dim, 0) + 1

        dim_yes_b: dict[str, int] = {}
        dim_n_b: dict[str, int] = {}
        dim_na_b: dict[str, int] = {}
        for a in pb.get("answers", []):
            dim = qindex.get(a["q"], {}).get("dimension", "?")
            dim_yes_b[dim] = dim_yes_b.get(dim, 0) + (1 if a["answer"] == "yes" else 0)
            dim_n_b[dim] = dim_n_b.get(dim, 0) + 1
        for a in pb.get("na_answers", []):
            dim = qindex.get(a["q"], {}).get("dimension", "?")
            dim_na_b[dim] = dim_na_b.get(dim, 0) + 1

        all_dims = set(dim_n_a) | set(dim_n_b)
        for dim in all_dims:
            na = dim_n_a.get(dim, 0)
            nb = dim_n_b.get(dim, 0)
            if na == 0 or nb == 0:
                continue

            score_a = dim_yes_a.get(dim, 0) / na
            score_b = dim_yes_b.get(dim, 0) / nb
            n_na_dim = dim_na_a.get(dim, 0) + dim_na_b.get(dim, 0)

            if score_a > score_b:
                dim_winner = "A"
            elif score_b > score_a:
                dim_winner = "B"
            else:
                dim_winner = "Tie"

            if dim not in dim_stats:
                dim_stats[dim] = {
                    "n_samples": 0,
                    "n_correct": 0,
                    "sum_score_a": 0.0,
                    "sum_score_b": 0.0,
                    "sum_na": 0,
                    "sum_questions": 0,
                }
            s = dim_stats[dim]
            s["n_samples"] += 1
            s["sum_score_a"] += score_a
            s["sum_score_b"] += score_b
            s["sum_na"] += n_na_dim
            s["sum_questions"] += na + nb + n_na_dim
            if ground_truth in ("A", "B") and dim_winner == ground_truth:
                s["n_correct"] += 1

    # summarize
    result = {}
    for dim, s in sorted(dim_stats.items()):
        n = s["n_samples"]
        result[dim] = {
            "n_samples": n,
            "avg_score_a": s["sum_score_a"] / n if n else 0,
            "avg_score_b": s["sum_score_b"] / n if n else 0,
            "dimension_accuracy": s["n_correct"] / n if n else 0,
            "avg_na_rate": s["sum_na"] / s["sum_questions"] if s["sum_questions"] else 0,
        }
    return result


def aggregate_checklist_score(
    parsed: dict,
    na_policy: str = "strict",
    coverage_threshold: float = 0.8,
    expected_n: int | None = None,
) -> dict[str, Any] | None:
    """Aggregate a pointwise checklist parse into n_yes / n_answered / score.

    Parameters
    ----------
    na_policy : str
        - ``"strict"`` (default): returns None if any N/A exists.
        - ``"as_no"``: treat N/A as "no".  score = n_yes / (n_answered + n_na).
        - ``"skip"``: score = n_yes / n_answered, N/A excluded from denominator.
        - ``"partial"``: like skip, but requires n_answered / expected_n >= *coverage_threshold*.
    coverage_threshold : float
        Only used when *na_policy* is ``"partial"``.
    expected_n : int | None
        Total expected questions.  Required for ``"partial"`` policy.
    """
    if parsed.get("_raw_fallback"):
        return None

    n_yes = parsed["n_yes"]
    n_answered = parsed["n_questions_parsed"]
    n_na = parsed.get("n_na", 0)
    coverage = n_answered / expected_n if expected_n and expected_n > 0 else None

    if na_policy == "strict":
        if n_na > 0:
            return None
        if "complete" in parsed and not parsed["complete"]:
            return None
        score = parsed["score"]

    elif na_policy == "as_no":
        if not parsed.get("complete") and not parsed.get("complete_with_na"):
            return None
        denom = n_answered + n_na
        score = n_yes / denom if denom > 0 else 0.0

    elif na_policy == "skip":
        if not parsed.get("complete") and not parsed.get("complete_with_na"):
            return None
        score = n_yes / n_answered if n_answered > 0 else 0.0

    elif na_policy == "partial":
        if not parsed.get("complete") and not parsed.get("complete_with_na"):
            # allow incomplete if coverage is high enough
            if coverage is None or coverage < coverage_threshold:
                return None
        if coverage is not None and coverage < coverage_threshold:
            return None
        score = n_yes / n_answered if n_answered > 0 else 0.0

    else:
        raise ValueError(f"Unknown na_policy: {na_policy!r}")

    return {
        "n_yes": n_yes,
        "n_no": parsed.get("n_no", n_answered - n_yes),
        "n_na": n_na,
        "n_answered": n_answered,
        "score": score,
        "na_policy": na_policy,
        "coverage": coverage,
    }


# ── Pairwise margin scoring ──────────────────────────────────────────

_PAIRWISE_TABLE: dict[tuple[str, str], float] = {
    ("yes", "no"):  +1.0,
    ("no",  "yes"): -1.0,
    ("yes", "na"):  +0.5,
    ("na",  "yes"): -0.5,
    ("no",  "na"):   0.0,
    ("na",  "no"):   0.0,
    ("yes", "yes"):  0.0,
    ("no",  "no"):   0.0,
    ("na",  "na"):   0.0,
}


def compare_checklists_pairwise(
    parsed_a: dict,
    parsed_b: dict,
    expected_n: int,
    tie_delta: float = 0.05,
) -> dict[str, Any] | None:
    """Compare two parsed checklists question-by-question.

    For each question, labels are aligned to {yes, no, na} and a per-question
    contribution is looked up from ``_PAIRWISE_TABLE``.  The final margin is
    the mean contribution over all ``expected_n`` questions.

    Returns ``None`` when either side failed to parse (``_raw_fallback``).
    """
    if parsed_a.get("_raw_fallback") or parsed_b.get("_raw_fallback"):
        return None

    # Build q -> label maps (default "na" for missing questions)
    def _label_map(parsed: dict) -> dict[int, str]:
        m: dict[int, str] = {}
        for a in parsed.get("answers", []):
            m[a["q"]] = a["answer"]          # "yes" or "no"
        for a in parsed.get("na_answers", []):
            m[a["q"]] = "na"
        return m

    map_a = _label_map(parsed_a)
    map_b = _label_map(parsed_b)

    total = 0.0
    n_aligned = 0
    for q in range(1, expected_n + 1):
        la = map_a.get(q, "na")
        lb = map_b.get(q, "na")
        total += _PAIRWISE_TABLE[(la, lb)]
        if la != "na" and lb != "na":
            n_aligned += 1

    margin = total / expected_n

    if margin > tie_delta:
        winner = "A"
    elif margin < -tie_delta:
        winner = "B"
    else:
        winner = "Tie"

    return {
        "margin": margin,
        "winner": winner,
        "n_aligned": n_aligned,
    }


def parse_winner(raw: str) -> str | None:
    """Extract the predicted winner (A or B) from raw model output.

    Handles various formats: plain "A"/"B", "Response A/B", JSON with
    "winner" key.
    """
    raw = raw.strip()

    # Try JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "winner" in data:
            winner = str(data["winner"]).strip().upper()
            if winner in ("A", "B"):
                return winner
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try embedded JSON
    json_match = re.search(r'\{[^{}]*"winner"\s*:\s*"([AB])"[^{}]*\}', raw, re.IGNORECASE)
    if json_match:
        return json_match.group(1).upper()

    raw_upper = raw.upper()

    if raw_upper in ("A", "B"):
        return raw_upper

    match = re.match(r"(?:RESPONSE\s+)?([AB])\b", raw_upper)
    if match:
        return match.group(1)

    matches = re.findall(r"\b([AB])\b", raw_upper)
    if matches:
        return matches[-1]

    return None


# ────────────────────────── metrics ──────────────────────────


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    domains: list[str] | None = None,
    scores_a: list[float] | None = None,
    scores_b: list[float] | None = None,
    n_answered_a: list[int] | None = None,
    n_answered_b: list[int] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro-F1, per-domain metrics, and pointwise diagnostics."""
    domains_list = domains or ["all"] * len(y_true)
    n_total = len(y_true)
    n_tie = sum(1 for p in y_pred if p == "Tie")

    # Only A/B predictions count for accuracy/F1
    valid = [
        (truth, pred, domain)
        for truth, pred, domain in zip(y_true, y_pred, domains_list)
        if pred in ("A", "B")
    ]
    n_unparseable = n_total - len(valid) - n_tie

    if not valid:
        return {"error": "no valid predictions", "n_total": n_total,
                "n_tie": n_tie, "n_unparseable": n_unparseable}

    true_v, pred_v, dom_v = zip(*valid)
    ab_labels = ["A", "B"]

    metrics: dict[str, Any] = {
        "n_total": n_total,
        "n_valid": len(valid),
        "n_tie": n_tie,
        "n_unparseable": n_unparseable,
        "parse_rate": len(valid) / n_total,
        "accuracy": accuracy_score(true_v, pred_v),
        "macro_f1": f1_score(true_v, pred_v, labels=ab_labels, average="macro", zero_division=0),
        "classification_report": classification_report(
            true_v, pred_v, labels=ab_labels, zero_division=0, output_dict=True
        ),
    }

    pred_counts = Counter(y_pred)
    metrics["pred_distribution"] = dict(pred_counts)
    metrics["position_bias_A"] = pred_counts.get("A", 0) / max(len(valid), 1)

    # ── pointwise diagnostics ──
    if scores_a is not None and scores_b is not None:
        valid_sa = [s for s in scores_a if s is not None]
        valid_sb = [s for s in scores_b if s is not None]
        gaps = [
            sa - sb for sa, sb in zip(scores_a, scores_b)
            if sa is not None and sb is not None
        ]
        tie_gaps = [
            abs(sa - sb) for sa, sb, p in zip(scores_a, scores_b, y_pred)
            if sa is not None and sb is not None and p == "Tie"
        ]
        metrics["avg_score_a"] = sum(valid_sa) / len(valid_sa) if valid_sa else None
        metrics["avg_score_b"] = sum(valid_sb) / len(valid_sb) if valid_sb else None
        metrics["avg_score_gap"] = sum(gaps) / len(gaps) if gaps else None
        metrics["avg_tie_score_gap"] = sum(tie_gaps) / len(tie_gaps) if tie_gaps else None

    if n_answered_a is not None and n_answered_b is not None:
        valid_na = [n for n in n_answered_a if n is not None]
        valid_nb = [n for n in n_answered_b if n is not None]
        metrics["avg_questions_answered_a"] = sum(valid_na) / len(valid_na) if valid_na else None
        metrics["avg_questions_answered_b"] = sum(valid_nb) / len(valid_nb) if valid_nb else None

    if domains is not None:
        domain_metrics = {}
        for domain_name in sorted(set(dom_v)):
            d_true = [t for t, d in zip(true_v, dom_v) if d == domain_name]
            d_pred = [p for p, d in zip(pred_v, dom_v) if d == domain_name]
            domain_metrics[domain_name] = {
                "n": len(d_true),
                "accuracy": accuracy_score(d_true, d_pred),
                "macro_f1": f1_score(d_true, d_pred, average="macro"),
            }
        metrics["per_domain"] = domain_metrics

    return metrics


def save_results(
    results_df: pd.DataFrame,
    metrics: dict,
    experiment_name: str,
    output_dir: Path = cfg.RESULTS_DIR,
) -> None:
    """Persist predictions and metrics for an experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / f"{experiment_name}_predictions.parquet"
    results_df.to_parquet(pred_path, index=False)
    log.info("Saved predictions -> %s", pred_path)

    metrics_path = output_dir / f"{experiment_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False, default=str)
    log.info("Saved metrics -> %s", metrics_path)

    console.print(f"\n[bold]{experiment_name}[/bold]")
    if "accuracy" not in metrics:
        console.print(metrics)
        return

    console.print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    console.print(f"  Macro-F1:     {metrics['macro_f1']:.4f}")
    console.print(f"  Parse rate:   {metrics['parse_rate']:.4f}")
    console.print(f"  Pos bias (A): {metrics['position_bias_A']:.4f}")
    if "per_domain" in metrics:
        for domain_name, dm in metrics["per_domain"].items():
            console.print(
                f"  [{domain_name}] acc={dm['accuracy']:.4f}  "
                f"f1={dm['macro_f1']:.4f}  n={dm['n']}"
            )
