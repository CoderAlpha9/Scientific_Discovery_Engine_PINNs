import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np

from .common import read_jsonl, write_json
from .equations import canonical_support


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
LANGUAGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LANGUAGE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TERM_CONCEPT = {"u_xx": "diffusion", "u_x": "advection", "u": "reaction", "u2": "reaction"}
TERM_MEANING = {"u_xx": "diffusive spreading", "u_x": "advective displacement",
                "u": "concentration-proportional reaction", "u2": "quadratic concentration-dependent reaction",
                "u3": "cubic concentration-dependent reaction", "u*u_x": "concentration-dependent transport",
                "1": "uniform production or removal"}


class ModelCache:
    def __init__(self, path, cache_dir=None, live=False, offline=False):
        self.path = Path(path)
        self.records = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.cache_dir, self.live = cache_dir, live
        if live and offline:
            raise ValueError("Live regeneration and offline replay cannot be combined.")
        self.offline = offline
        self.encoder = self.generator = None

    def key(self, kind, content):
        revision = EMBEDDING_REVISION if kind == "embedding" else LANGUAGE_REVISION
        return hashlib.sha256(json.dumps([kind, revision, content], sort_keys=True).encode()).hexdigest()

    def persist(self):
        write_json(self.path, self.records)

    def embed(self, text):
        key = self.key("embedding", text)
        if not self.live and key in self.records:
            return np.asarray(self.records[key]["vector"])
        if self.offline:
            raise RuntimeError("This embedding is not cached. Disable offline mode to load the pinned model.")
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        if self.encoder is None:
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, cache_dir=self.cache_dir)
            model = AutoModel.from_pretrained(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, cache_dir=self.cache_dir).eval()
            self.encoder = (tokenizer, model)
        tokenizer, model = self.encoder
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            vector = (hidden * mask).sum(1) / mask.sum(1)
            vector = torch.nn.functional.normalize(vector, dim=1)[0].numpy()
        self.records[key] = {"kind": "embedding", "model": EMBEDDING_MODEL,
                             "revision": EMBEDDING_REVISION, "input": text, "vector": vector.tolist()}
        self.persist()
        return vector

    def generate(self, messages, maximum=256, prepared=False):
        content = {"messages": deepcopy(messages), "maximum": maximum, "decoding": "greedy", "precision": "bfloat16"}
        key = self.key("generation", content)
        if (not self.live or prepared) and key in self.records:
            return self.records[key]["output"], key
        if self.offline:
            raise RuntimeError("This language request is not cached. Disable offline mode to run the pinned model.")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(4)
        if self.generator is None:
            tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL, revision=LANGUAGE_REVISION, cache_dir=self.cache_dir)
            model = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL, revision=LANGUAGE_REVISION,
                                                        cache_dir=self.cache_dir, dtype=torch.bfloat16).eval()
            self.generator = (tokenizer, model)
        tokenizer, model = self.generator
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(inputs, max_new_tokens=maximum, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id,
                                    attention_mask=torch.ones_like(inputs))
        result = tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True)
        self.records[key] = {"kind": "generation", "model": LANGUAGE_MODEL, "revision": LANGUAGE_REVISION,
                             **content, "output": result, "input_tokens": inputs.shape[1],
                             "output_tokens": output.shape[1] - inputs.shape[1]}
        self.persist()
        return result, key

    def generate_many(self, requests, maximum=100, batch_size=8):
        missing, seen = [], set()
        for messages in requests:
            content = {"messages": deepcopy(messages), "maximum": maximum, "decoding": "greedy", "precision": "bfloat16"}
            key = self.key("generation", content)
            if key not in seen and (self.live or key not in self.records):
                missing.append((key, content))
                seen.add(key)
        if not missing:
            return
        if self.offline:
            raise RuntimeError("The benchmark has uncached requests. Disable offline mode to generate them.")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(4)
        if missing and self.generator is None:
            tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL, revision=LANGUAGE_REVISION, cache_dir=self.cache_dir)
            tokenizer.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL, revision=LANGUAGE_REVISION,
                                                        cache_dir=self.cache_dir, dtype=torch.bfloat16).eval()
            self.generator = tokenizer, model
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            tokenizer, model = self.generator
            tokenizer.padding_side = "left"
            texts = [tokenizer.apply_chat_template(c["messages"], add_generation_prompt=True, tokenize=False) for _, c in batch]
            inputs = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=maximum, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
            for i, (key, content) in enumerate(batch):
                sequence = outputs[i, inputs["input_ids"].shape[1]:]
                text = tokenizer.decode(sequence, skip_special_tokens=True)
                stops = (sequence == tokenizer.eos_token_id).nonzero().flatten()
                token_count = int(stops[0]) + 1 if len(stops) else len(sequence)
                self.records[key] = {"kind": "generation", "model": LANGUAGE_MODEL, "revision": LANGUAGE_REVISION,
                                     **content, "output": text, "input_tokens": int(inputs["attention_mask"][i].sum()),
                                     "output_tokens": token_count, "batch_size": len(batch)}
            self.persist()
            print(f"language generation {min(start + batch_size, len(missing))}/{len(missing)}", flush=True)


class Evidence:
    def __init__(self, path, cache):
        self.records = read_jsonl(path)
        self.by_id = {r["id"]: r for r in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError("Evidence IDs must be unique.")
        self.cache = cache
        self.vectors = np.vstack([cache.embed(r["text"] + " " + r["keywords"]) for r in self.records])
        self.graph = {"nodes": [], "edges": []}
        concepts = sorted({c for r in self.records for c in r["concepts"]})
        self.graph["nodes"] = [{"id": c, "type": "concept"} for c in concepts]
        for r in self.records:
            self.graph["nodes"].append({"id": r["id"], "type": "evidence", "source": r["source"]})
            self.graph["edges"].extend({"from": r["id"], "to": c, "relation": "supports",
                                         "source": r["source"], "location": r["location"]} for c in r["concepts"])

    def ground_terms(self, terms):
        ids, paths = set(), []
        for term in terms:
            concept = TERM_CONCEPT.get(term)
            anchors = [r for r in self.records if term in r["concepts"] and concept in r["concepts"]]
            if not anchors:
                raise ValueError(f"No direct evidence anchor for {term}.")
            anchor = max(anchors, key=lambda r: int(term in r["keywords"].split()))
            bridges = [r for r in self.records if concept in r["concepts"] and r["source"] != anchor["source"]]
            if not bridges:
                raise ValueError(f"No independent source supports the mechanism for {term}.")
            bridge = max(bridges, key=lambda r: int(r["concepts"][0] == concept))
            ids.update((anchor["id"], bridge["id"]))
            paths.append({"term": term, "path": [anchor["id"], concept, bridge["id"]]})
        return sorted(ids), paths

    def retrieve(self, question, graph=False, count=6):
        query = self.cache.embed(question)
        tokens = set(re.findall(r"[a-z_]+", question.lower()))
        lexical = np.array([len(tokens & set(re.findall(r"[a-z_]+", r["text"].lower() + " " + r["keywords"])))
                            / max(len(tokens), 1) for r in self.records])
        scores = self.vectors @ query + 0.25 * lexical
        order = np.argsort(-scores)
        selected, paths = list(order[:count]), []
        if graph:
            selected = list(order[:max(2, count // 2)])
            bridges = {c for i in selected for c in self.records[i]["concepts"]}
            ranked = sorted((i for i in range(len(self.records)) if i not in selected),
                            key=lambda i: scores[i] + 0.08 * len(bridges & set(self.records[i]["concepts"])), reverse=True)
            for i in ranked[:count - len(selected)]:
                for j in selected[:max(2, count // 2)]:
                    shared = sorted(set(self.records[i]["concepts"]) & set(self.records[j]["concepts"]))
                    if shared:
                        paths.append([self.records[j]["id"], shared[0], self.records[i]["id"]])
                        break
                selected.append(i)
        return [self.records[i] for i in selected], paths

    def answer_request(self, question, graph=False):
        records, paths = self.retrieve(question, graph)
        context = "\n".join(f"{r['id']} [{r['title']}]: {r['text']}" for r in records)
        messages = [{"role": "system", "content": "Answer the scientific question using only the evidence. Be concise. Cite evidence IDs. If unsupported, say insufficient evidence."},
                    {"role": "user", "content": context + "\nQuestion: " + question}]
        return records, paths, messages

    def answer(self, question, graph=False, prepared=False):
        records, paths, messages = self.answer_request(question, graph)
        answer, key = self.cache.generate(messages, maximum=100, prepared=prepared)
        cited = re.findall(r"\bE\d{2}\b", answer)
        return {"answer": answer, "evidence": [r["id"] for r in records], "paths": paths,
                "valid_citations": bool(cited) and all(c in {r["id"] for r in records} for c in cited),
                "cited_ids": cited, "cache_key": key}

    def hypotheses(self, question, graph=True):
        records, paths = self.retrieve(question, graph=graph, count=10)
        context = "\n".join(f"{r['id']} [{r['title']}]: {r['text']}" for r in records)
        prompt = (context + "\nExperiment: " + question +
                  '\nReturn JSON only: {"hypotheses":[{"terms":["u_xx","u_x"],"evidence":["E06","E08"],"reason":"...","prediction":"..."}, ...]}. '
                  'Generate at least three different plausible term combinations for u_t. Allowed terms: u_xx, u_x, u, u2, u3, u*u_x, 1. '
                  'Include simpler competing explanations as well as transport with nonlinear reaction. Use evidence IDs from the supplied context. Each hypothesis needs evidence from at least two sources. No coefficient values.')
        messages = [{"role": "system", "content": "You propose falsifiable scientific hypotheses. Evidence is a prior, not proof. Follow the JSON schema exactly."},
                    {"role": "user", "content": prompt}]
        attempts = []
        for attempt in range(3):
            raw, key = self.cache.generate(messages, maximum=850)
            attempts.append({"cache_key": key, "response": raw})
            try:
                value = re.search(r"\{.*\}", raw, re.S)
                if value is None:
                    raise ValueError("No JSON object.")
                parsed = json.loads(value.group())["hypotheses"]
                proposals = []
                for hypothesis in parsed:
                    terms = canonical_support(hypothesis["terms"])
                    if not terms or len(terms) > 4 or (graph and any(t not in TERM_CONCEPT for t in terms)):
                        continue
                    evidence_ids, support_paths = self.ground_terms(terms) if graph else (hypothesis.get("evidence", []), [])
                    if graph and len({self.by_id[e]["source"] for e in evidence_ids}) < 2:
                        continue
                    mechanisms = [TERM_MEANING[t] for t in terms]
                    proposals.append({"id": f"H{len(proposals) + 1}", "terms": list(terms), "evidence": evidence_ids,
                                      "support_paths": support_paths,
                                      "reason": ("Test the combined contributions of " + ", ".join(mechanisms) + ".") if graph else str(hypothesis.get("reason", "")),
                                      "prediction": "The fitted trajectory must exhibit the rate contributions implied by these terms; omitted mechanisms must not be needed to explain held-out observations.",
                                      "language_evidence": hypothesis.get("evidence", []),
                                      "language_reason": str(hypothesis.get("reason", "")),
                                      "explanation_origin": "term semantics and verified evidence graph; language prose is untrusted" if graph else "raw language proposal without graph grounding"})
                unique = {tuple(p["terms"]): p for p in proposals}
                if len(unique) < 3:
                    raise ValueError("Need three distinct hypotheses, each citing two different source titles from the context.")
                break
            except (KeyError, TypeError, ValueError) as error:
                if attempt == 2:
                    raise RuntimeError("Hypothesis generation failed after three validated attempts.") from error
                messages.extend([{"role": "assistant", "content": raw},
                                 {"role": "user", "content": "Correct the response. " + str(error) +
                                  ' Evidence IDs must be quoted JSON strings such as "E01". Each hypothesis must cite two different source titles. Return only a compact JSON object. Reasons and predictions must match the chosen terms.'}])
        return {"question": question, "hypotheses": list(unique.values()), "evidence": [r["id"] for r in records],
                "paths": paths, "raw_response": raw, "cache_key": key,
                "provenance_policy": "The model proposes structure. The evidence graph resolves support and term semantics independently; model citations and prose are not accepted as proof." if graph else "Uncorrected RAG-only structural proposals for the ablation.",
                "model": LANGUAGE_MODEL, "revision": LANGUAGE_REVISION, "attempts": attempts}
