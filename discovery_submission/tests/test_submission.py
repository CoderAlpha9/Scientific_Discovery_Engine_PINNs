from pathlib import Path

import numpy as np
import pytest
import torch

from discovery.common import read_json, read_jsonl
from discovery.evidence import ModelCache
from discovery.experiments import load_arrays
from discovery.validation import assess_case
from discovery.verify import verify_results


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_counts_uniqueness_and_adversarial_behaviour():
    items = read_jsonl(ROOT / "data" / "benchmark.jsonl")
    evidence = read_jsonl(ROOT / "data" / "evidence.jsonl")
    ids = {e["id"] for e in evidence}
    assert len(items) == len({q["id"] for q in items}) == len({q["question"] for q in items}) == 95
    for case in items:
        if case["category"] == "adversarial":
            assert assess_case(case, ids)["status"] == case["expected"]


def test_model_response_provenance_and_request_hashes():
    cache = ModelCache(ROOT / "data" / "model_cache.json")
    generations = [r for r in cache.records.values() if r["kind"] == "generation"]
    assert len(generations) >= 75
    for key, record in cache.records.items():
        if record["kind"] == "generation":
            content = {k: record[k] for k in ("messages", "maximum", "decoding")}
            if "precision" in record:
                content["precision"] = record["precision"]
            assert cache.key("generation", content) == key
            assert record["output_tokens"] > 0
            assert len(record["revision"]) == 40


@pytest.mark.parametrize("batch", [False, True])
def test_cached_request_survives_caller_mutation(tmp_path, batch):
    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **options):
            return torch.tensor([[1]]) if options["tokenize"] else "prompt"

        def __call__(self, texts, **options):
            values = torch.ones((len(texts), 1), dtype=torch.long)
            return {"input_ids": values, "attention_mask": values}

        def decode(self, tokens, **options):
            return "response"

    class Generator:
        def generate(self, inputs=None, **options):
            values = options["input_ids"] if inputs is None else inputs
            return torch.cat((values, 2 * torch.ones_like(values)), dim=1)

    path = tmp_path / "cache.json"
    cache = ModelCache(path)
    cache.generator = Tokenizer(), Generator()
    messages = [{"role": "user", "content": "first"}]
    if batch:
        cache.generate_many([messages], maximum=8)
    else:
        cache.generate(messages, maximum=8)
    messages[0]["content"] = "changed"
    messages.append({"role": "user", "content": "retry"})
    cache.persist()
    record = next(iter(ModelCache(path).records.values()))
    assert record["messages"] == [{"role": "user", "content": "first"}]


def test_hypotheses_have_resolved_multisource_evidence():
    evidence = {r["id"]: r for r in read_jsonl(ROOT / "data" / "evidence.jsonl")}
    hypotheses = read_json(ROOT / "results" / "hypotheses.json")["hypotheses"]
    assert len({tuple(h["terms"]) for h in hypotheses}) >= 3
    for h in hypotheses:
        assert len({evidence[i]["source"] for i in h["evidence"]}) >= 2


def test_saved_end_to_end_submission():
    result = verify_results(ROOT)
    assert result["status"] == "passed"
    metrics = read_json(ROOT / "results" / "metrics.json")
    assert metrics["inverse"]["ood_relative_error"] < 0.1
    assert metrics["ode"]["parameter_relative_error"] < 0.1
    intervals = load_arrays(ROOT / "results" / "uncertainty.npz")
    assert np.isfinite(intervals["mean"]).all()
    assert (intervals["lower"] <= intervals["upper"]).all()
