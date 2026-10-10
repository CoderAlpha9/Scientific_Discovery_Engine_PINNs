import re

import numpy as np

from .common import read_jsonl, write_json
from .validation import assess_case


def score_answer(answer, groups):
    text = re.sub(r"[^a-z0-9_ ]", " ", answer.lower())
    matches = [any(synonym.lower() in text for synonym in group) for group in groups]
    return bool(all(matches)), float(np.mean(matches))


def run_benchmark(path, evidence, output):
    questions = read_jsonl(path)
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Benchmark identifiers must be unique.")
    requests = [evidence.answer_request(q["question"], graph)[2]
                for q in questions if q["category"] != "adversarial" for graph in (False, True)]
    evidence.cache.generate_many(requests)
    results = []
    for question in questions:
        record = {"id": question["id"], "category": question["category"], "question": question["question"]}
        if question["category"] == "adversarial":
            verdict = assess_case(question, evidence.by_id)
            record.update(verdict)
            record["correct"] = verdict["status"] == question["expected"]
        else:
            record["systems"] = {}
            for name, use_graph in (("rag", False), ("rag_kg", True)):
                answer = evidence.answer(question["question"], graph=use_graph, prepared=True)
                correct, fraction = score_answer(answer["answer"], question["answer_groups"])
                answer.update({"correct": correct and answer["valid_citations"], "concept_fraction": fraction,
                               "evidence_recall": len(set(answer["evidence"]) & set(question["evidence"])) / len(question["evidence"])})
                record["systems"][name] = answer
        results.append(record)
        write_json(output, {"items": results})
        print(f"benchmark {len(results)}/{len(questions)} {question['id']}", flush=True)
    summary = {}
    for category in ("scientific", "multi_hop"):
        subset = [r for r in results if r["category"] == category]
        summary[category] = {"count": len(subset)}
        for name in ("rag", "rag_kg"):
            summary[category][name] = {
                "answer_accuracy": float(np.mean([r["systems"][name]["correct"] for r in subset])),
                "concept_fraction": float(np.mean([r["systems"][name]["concept_fraction"] for r in subset])),
                "evidence_recall": float(np.mean([r["systems"][name]["evidence_recall"] for r in subset])),
                "citation_validity": float(np.mean([r["systems"][name]["valid_citations"] for r in subset])),
            }
    adversarial = [r for r in results if r["category"] == "adversarial"]
    summary["adversarial"] = {"count": len(adversarial), "accuracy": float(np.mean([r["correct"] for r in adversarial]))}
    summary["answer_metric"] = "Deterministic synonym-group coverage with citation validation; heuristic, not an expert semantic grader."
    write_json(output, {"summary": summary, "items": results})
    return summary
