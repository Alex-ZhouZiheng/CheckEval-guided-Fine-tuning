from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from data_process.prepare_self_checklist_sft import parse_self_checklist_trace
from utils import build_question_index, load_checklists, parse_checkeval_output

OUT = Path("results/coverage_dev200")
OUT.mkdir(parents=True, exist_ok=True)

RNG = 20260508
TARGETS = {"code": 67, "general": 66, "stem": 67}


def norm_pred(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() == "tie":
        return "Tie"
    if s.upper() in {"A", "B"}:
        return s.upper()
    return s


def is_correct(pred, gold):
    pred = norm_pred(pred)
    gold = norm_pred(gold)
    return pred in {"A", "B"} and pred == gold


def parse_reasoning_text(s):
    if not isinstance(s, str) or not s.strip():
        return []
    fixed = s.replace("}\r\n {", "},\n {").replace("}\n {", "},\n {")
    try:
        obj = ast.literal_eval(fixed)
    except Exception:
        return [s.strip()]
    out = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                txt = item.get("reasoning") or item.get("preference") or ""
                if txt:
                    out.append(str(txt).strip())
    return out


def add_pairwise_from_static_answers(a, b):
    if a == "yes" and b == "no":
        return "A"
    if a == "no" and b == "yes":
        return "B"
    if a is not None or b is not None:
        return "Tie"
    return None


def main():
    dev = pd.read_parquet("data/splits/dev_600.parquet").copy()
    dev["row_idx"] = range(len(dev))
    dev["gold_winner"] = dev["winner"].map(norm_pred)

    direct = pd.read_parquet("results/vanilla_judge_dev_600_predictions.parquet").copy()
    direct["row_idx"] = range(len(direct))
    direct = direct[["row_idx", "predicted_winner"]].rename(
        columns={"predicted_winner": "direct_pred"}
    )
    self4 = pd.read_parquet(
        "results/selfchecklist_checkpoint-558_dev_600_4B_sft_v1_predictions.parquet"
    ).copy()
    self4["row_idx"] = range(len(self4))
    self4 = self4[["row_idx", "predicted_winner"]].rename(
        columns={"predicted_winner": "selfcheck_sft_pred"}
    )
    selector = pd.read_parquet(
        "results/dynamic_dev_600/selector_v4bank_v4hr_noOracleFilter_best_k15_td005_pairwise/predictions.parquet"
    ).copy()
    selector["row_idx"] = range(len(selector))
    selector = selector[["row_idx", "sample_id", "predicted_winner"]].rename(
        columns={"predicted_winner": "static_v4_selector_pred"}
    )

    base = (
        dev.merge(direct, on="row_idx", how="left")
        .merge(self4, on="row_idx", how="left")
        .merge(selector, on="row_idx", how="left")
    )
    base["instance_id"] = base["sample_id"].fillna(
        base["prompt_id"].astype(str) + "_row" + base["row_idx"].astype(str)
    )
    for c in ["direct_pred", "selfcheck_sft_pred", "static_v4_selector_pred"]:
        base[c] = base[c].map(norm_pred)
    base["direct_correct"] = [
        is_correct(p, g) for p, g in zip(base["direct_pred"], base["gold_winner"])
    ]
    base["selfcheck_sft_correct"] = [
        is_correct(p, g)
        for p, g in zip(base["selfcheck_sft_pred"], base["gold_winner"])
    ]
    base["selector_tie"] = base["static_v4_selector_pred"].eq("Tie")

    reason = pd.read_parquet("data/with_reason/dev_600_reasoning.parquet")
    reason = reason[reason["swap_flag"] == False].copy()  # noqa: E712
    key_cols = ["domain", "context", "response_a", "response_b", "winner"]
    reason["_key"] = reason[key_cols].astype(str).agg("\u241f".join, axis=1)
    base["_key"] = base[key_cols].astype(str).agg("\u241f".join, axis=1)
    reason_map = dict(zip(reason["_key"], reason["reasoning_text"].map(parse_reasoning_text)))
    base["human_reasonings"] = base["_key"].map(reason_map).apply(
        lambda x: x if isinstance(x, list) else []
    )

    selected_parts = []
    for domain, target in TARGETS.items():
        ddf = base[base["domain"] == domain].copy()
        ddf = ddf.sample(frac=1.0, random_state=RNG).reset_index(drop=True)
        groups = defaultdict(list)
        for idx, row in ddf.iterrows():
            key = (
                row["gold_winner"],
                bool(row["direct_correct"]),
                bool(row["selfcheck_sft_correct"]),
                bool(row["selector_tie"]),
            )
            groups[key].append(idx)
        cursors = {k: 0 for k in groups}
        chosen = []
        keys = sorted(groups, key=lambda k: (len(groups[k]), str(k)))
        while len(chosen) < target:
            progressed = False
            for k in keys:
                cur = cursors[k]
                if cur < len(groups[k]) and len(chosen) < target:
                    chosen.append(groups[k][cur])
                    cursors[k] += 1
                    progressed = True
            if not progressed:
                break
        if len(chosen) != target:
            raise RuntimeError(f"domain {domain}: selected {len(chosen)} != target {target}")
        selected_parts.append(ddf.loc[chosen])

    sample = pd.concat(selected_parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=RNG + 1).reset_index(drop=True)

    instance_cols = [
        "instance_id",
        "domain",
        "context",
        "response_a",
        "response_b",
        "gold_winner",
        "human_reasonings",
        "direct_pred",
        "selfcheck_sft_pred",
        "static_v4_selector_pred",
    ]
    sample[instance_cols].to_parquet(OUT / "dev200_instances.parquet", index=False)
    sample[instance_cols].to_json(
        OUT / "dev200_instances.preview.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    dev20_parts = []
    for domain, target in {"code": 7, "general": 6, "stem": 7}.items():
        dev20_parts.append(sample[sample["domain"] == domain].head(target))
    dev20 = (
        pd.concat(dev20_parts, ignore_index=True)
        .sample(frac=1.0, random_state=RNG + 2)
        .reset_index(drop=True)
    )
    atomic_in_rows = []
    for _, r in dev20.iterrows():
        for i, txt in enumerate(r["human_reasonings"] or [], start=1):
            atomic_in_rows.append(
                {
                    "instance_id": r["instance_id"],
                    "domain": r["domain"],
                    "annotator_id": f"ann{i}",
                    "rationale_text": txt,
                    "gold_winner": r["gold_winner"],
                    "context": r["context"],
                    "response_a": r["response_a"],
                    "response_b": r["response_b"],
                }
            )
    atomic_in = pd.DataFrame(atomic_in_rows)
    atomic_in.to_parquet(OUT / "atomic_claims_dev20_input.parquet", index=False)
    atomic_in[
        ["instance_id", "domain", "annotator_id", "gold_winner", "rationale_text"]
    ].to_json(
        OUT / "atomic_claims_dev20_input.preview.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    selected_rows = set(int(x) for x in sample["row_idx"].tolist())
    row_to_instance = dict(zip(sample["row_idx"], sample["instance_id"]))
    rows = []

    def add_selfcheck(judge_name, path):
        df = pd.read_parquet(path).copy()
        df["row_idx"] = range(len(df))
        df = df[df["row_idx"].isin(selected_rows)]
        for _, r in df.iterrows():
            iid = str(row_to_instance[int(r["row_idx"])])
            parsed = parse_self_checklist_trace(str(r.get("raw_output", "")))
            checklist = parsed.get("checklist") or []
            verdicts = parsed.get("verdicts") or {}
            for local_i, q in enumerate(checklist, start=1):
                ans = verdicts.get(local_i) or verdicts.get(str(local_i))
                rows.append(
                    {
                        "instance_id": iid,
                        "judge_name": judge_name,
                        "question_id": local_i,
                        "question_text": q,
                        "answer_a": None,
                        "answer_b": None,
                        "pairwise_answer": norm_pred(ans) if ans is not None else None,
                        "dimension": None,
                        "source": "generated",
                        "final_pred": norm_pred(r.get("predicted_winner")),
                        "parse_ok": bool(r.get("parse_ok"))
                        if not pd.isna(r.get("parse_ok"))
                        else None,
                        "tie": norm_pred(r.get("predicted_winner")) == "Tie",
                    }
                )

    add_selfcheck(
        "selfcheck_4b_sft",
        "results/selfchecklist_checkpoint-558_dev_600_4B_sft_v1_predictions.parquet",
    )
    add_selfcheck(
        "selfcheck_27b_teacher",
        "results/selfchecklist_base_dev_600_27B_teacher_predictions.parquet",
    )

    checklists_v3, _defs = load_checklists(Path("checklists/v3_frozen"))
    bank_v3 = pd.read_parquet("checklists/v3_frozen/bank_index.parquet")
    text_to_qid_v3 = {
        str(r["question_text"]): int(r["qid"]) for _, r in bank_v3.iterrows()
    }
    static3 = pd.read_parquet("results/checkeval_pairwise_naaware_dev_600_v3_q9b_predictions.parquet")
    static3["row_idx"] = range(len(static3))
    static3 = static3[static3["row_idx"].isin(selected_rows)]
    for _, r in static3.iterrows():
        iid = str(row_to_instance[int(r["row_idx"])])
        domain = r["domain"]
        qindex = build_question_index(checklists_v3, domain)
        expected = int(r.get("expected_n_questions") or len(qindex))
        pa = parse_checkeval_output(str(r.get("raw_output_a", "")), expected_n=expected)
        pb = parse_checkeval_output(str(r.get("raw_output_b", "")), expected_n=expected)
        ans_a = {int(a["q"]): a["answer"] for a in pa.get("answers", [])}
        ans_b = {int(a["q"]): a["answer"] for a in pb.get("answers", [])}
        for qn in pa.get("na_qnums", []):
            ans_a[int(qn)] = "na"
        for qn in pb.get("na_qnums", []):
            ans_b[int(qn)] = "na"
        for qn, meta in qindex.items():
            a = ans_a.get(qn)
            b = ans_b.get(qn)
            qtext = meta["question"]
            rows.append(
                {
                    "instance_id": iid,
                    "judge_name": "static_v3",
                    "question_id": text_to_qid_v3.get(qtext, qn),
                    "question_text": qtext,
                    "answer_a": a,
                    "answer_b": b,
                    "pairwise_answer": add_pairwise_from_static_answers(a, b),
                    "dimension": meta["dimension"],
                    "source": "static",
                    "final_pred": norm_pred(r.get("predicted_winner")),
                    "parse_ok": bool(r.get("checklist_parsed"))
                    if not pd.isna(r.get("checklist_parsed"))
                    else None,
                    "tie": norm_pred(r.get("predicted_winner")) == "Tie",
                }
            )

    bank_v4 = pd.read_parquet("checklists/v4_frozen/bank_index.parquet")
    qmeta_v4 = {int(r["qid"]): r.to_dict() for _, r in bank_v4.iterrows()}

    def add_dynamic_qids(judge_name, path, source):
        df = pd.read_parquet(path).copy()
        df["row_idx"] = range(len(df))
        df = df[df["row_idx"].isin(selected_rows)]
        for _, r in df.iterrows():
            iid = str(row_to_instance[int(r["row_idx"])])
            qids = r.get("asked_qids") if "asked_qids" in r else r.get("selected_qids")
            if isinstance(qids, np.ndarray):
                qids = qids.tolist()
            if qids is None:
                qids = []
            for qid in qids:
                qid = int(qid)
                meta = qmeta_v4.get(qid, {})
                rows.append(
                    {
                        "instance_id": iid,
                        "judge_name": judge_name,
                        "question_id": qid,
                        "question_text": meta.get("question_text"),
                        "answer_a": None,
                        "answer_b": None,
                        "pairwise_answer": None,
                        "dimension": meta.get("dimension"),
                        "source": source,
                        "final_pred": norm_pred(r.get("predicted_winner")),
                        "parse_ok": bool(r.get("parse_ok"))
                        if not pd.isna(r.get("parse_ok"))
                        else None,
                        "tie": norm_pred(r.get("predicted_winner")) == "Tie",
                    }
                )

    add_dynamic_qids(
        "static_v4",
        "results/dynamic_dev_600/static_v4_td005_pairwise/predictions.parquet",
        "static",
    )
    add_dynamic_qids(
        "static_v4_selector_k15",
        "results/dynamic_dev_600/selector_v4bank_v4hr_noOracleFilter_best_k15_td005_pairwise/predictions.parquet",
        "selected",
    )

    jdf = pd.DataFrame(rows)
    cols = [
        "instance_id",
        "judge_name",
        "question_id",
        "question_text",
        "answer_a",
        "answer_b",
        "pairwise_answer",
        "dimension",
        "source",
        "final_pred",
        "parse_ok",
        "tie",
    ]
    jdf = jdf[cols].sort_values(["judge_name", "instance_id", "question_id"]).reset_index(drop=True)
    jdf.to_parquet(OUT / "judge_checklists.parquet", index=False)
    jdf.head(500).to_json(
        OUT / "judge_checklists.preview.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    summary = {
        "dev200_path": str(OUT / "dev200_instances.parquet"),
        "atomic_dev20_input_path": str(OUT / "atomic_claims_dev20_input.parquet"),
        "judge_checklists_path": str(OUT / "judge_checklists.parquet"),
        "domain_counts": sample["domain"].value_counts().sort_index().to_dict(),
        "gold_counts_by_domain": {
            d: sample[sample.domain == d]["gold_winner"].value_counts().to_dict()
            for d in TARGETS
        },
        "direct_correct_counts_by_domain": {
            d: sample[sample.domain == d]["direct_correct"].value_counts().to_dict()
            for d in TARGETS
        },
        "selfcheck_correct_counts_by_domain": {
            d: sample[sample.domain == d]["selfcheck_sft_correct"].value_counts().to_dict()
            for d in TARGETS
        },
        "selector_tie_counts_by_domain": {
            d: sample[sample.domain == d]["selector_tie"].value_counts().to_dict()
            for d in TARGETS
        },
        "human_reasonings_len_counts": sample["human_reasonings"].map(len).value_counts().sort_index().to_dict(),
        "atomic_dev20_instances": int(dev20["instance_id"].nunique()),
        "atomic_dev20_rationales": int(len(atomic_in)),
        "judge_checklists_rows_by_judge": jdf["judge_name"].value_counts().sort_index().to_dict(),
        "answer_missing_rows_by_judge": jdf[
            jdf["answer_a"].isna()
            & jdf["answer_b"].isna()
            & jdf["pairwise_answer"].isna()
        ]["judge_name"].value_counts().sort_index().to_dict(),
        "note": (
            "static_v4 and static_v4_selector_k15 qids/final_pred came from existing "
            "dynamic_dev_600 predictions, which did not store raw per-question answers."
        ),
    }
    (OUT / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
