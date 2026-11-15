import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from test_knowledge_base import KnowledgeBase
from tooling.sag_index import tokenize
from tooling.sag_retrieval import EvidenceJudge


@pytest.fixture
def kb(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHOMIND_SEED_DEMO_DOCS", "false")
    monkeypatch.setenv("ECHOMIND_KB_MIGRATE_CHROMA", "off")
    monkeypatch.setenv("ECHOMIND_SAG2_JUDGE", "lexical")
    knowledge = KnowledgeBase(sqlite_path=str(tmp_path / "knowledge.sqlite3"))
    yield knowledge
    knowledge.close()


def test_sag_chinese_fts_and_source_citations(kb):
    kb.add_documents([{"title": "退款规则", "content": "付款后七天可以申请退款，退回原支付账户。"},
                      {"title": "配送", "content": "同城配送当天到达。"}])
    results = kb.search("退款多久到账", top_k=2)
    assert results[0]["title"] == "退款规则"
    assert results[0]["document_id"] and results[0]["version"]
    assert results[0]["citation_id"] == results[0]["id"]
    assert results[0]["retrieval_method"] == "sag2"
    assert results[0]["scoring_method"] == "lexical"
    assert kb.search('" OR * NOT missingxyz') == []
    assert "退款" in tokenize("如何申请退款？")


def test_sag_replacement_is_atomic_on_partial_write_failure(kb, monkeypatch):
    kb.add_documents([{"document_id": "policy", "title": "规则", "content": "原来的规则。"}])
    before = kb._index.get()
    original = kb._index.upsert

    def fail_after_write(**kwargs):
        original(**kwargs)
        raise RuntimeError("interrupted write")

    monkeypatch.setattr(kb._index, "upsert", fail_after_write)
    with pytest.raises(RuntimeError):
        kb.add_documents([{"document_id": "policy", "title": "规则", "content": "新规则" * 300}])
    assert kb._index.get() == before
    assert kb.search("原来")[0]["content"] == "原来的规则。"


def test_sag_concurrent_instances_replace_without_stale_chunks(kb, monkeypatch):
    path = kb._index.db.execute("PRAGMA database_list").fetchone()[2]
    other = KnowledgeBase(sqlite_path=path)
    versions = ["new " * 250, "last revision"]
    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(store.add_documents, [{"document_id": "same", "title": "Policy", "content": content}])
                       for store, content in zip((kb, other), versions)]
            for future in futures:
                future.result()
        rows = kb._index.get()
        assert len({meta["version"] for meta in rows["metadatas"]}) == 1
        assert len(rows["ids"]) == rows["metadatas"][0]["total_chunks"]
    finally:
        other.close()


def test_sag_multihop_expansion_uses_bridge_and_rejudges_context(kb):
    kb.add_documents([{"title": "Project Orion", "content": "Project Orion was founded by Alice Mercer."},
                      {"title": "Alice Mercer", "content": "Alice Mercer's birthplace is Kyoto."},
                      {"title": "Distractor", "content": "Orion telescope observation."}])
    calls = []

    async def judge(query, candidates, evidence):
        calls.append((candidates, evidence))
        return [{**item, "score": .95 if "Alice Mercer" in item["content"] else .05} for item in candidates], "llm"

    kb._retriever.judge.score = judge
    results = kb.search("Where was the founder of Project Orion born?", top_k=3)
    assert any("Kyoto" in item["content"] and item["hop"] > 0 for item in results)
    assert len(calls) >= 2
    assert calls[0][1] == []
    assert any("founded" in item["content"] for item in calls[1][1])


def test_sag_empty_query_and_top_k_validation(kb):
    for query, top_k in [("", 5), (None, 5), ("policy", True), ("policy", 21)]:
        with pytest.raises(ValueError):
            kb.search(query, top_k)


def test_sag_chunks_preserve_quoted_source_text(kb):
    source = ("原文保留。\nNext paragraph: a.b? " * 70)
    kb.add_documents([{"title": "Source", "content": source}])
    rows = kb._index.get()
    ordered = sorted(zip(rows["documents"], rows["metadatas"]), key=lambda item: item[1]["chunk_index"])
    assert "".join(text for text, _ in ordered) == source
    assert all(len(text) <= 500 for text, _ in ordered)


def test_sag_legacy_migration_idempotent_preserves_ids_and_replacement(kb):
    legacy = {"ids": ["legacy-1", "legacy-2"], "documents": ["旧退款规则", "旧配送规则"],
              "metadatas": [{"title": "退款", "chunk_index": 0}, {"title": "配送", "chunk_index": 0}]}
    collection = SimpleNamespace(count=lambda: 2, get=lambda **kwargs: legacy)
    assert kb.migrate_from_chroma(collection) == 2
    assert kb.migrate_from_chroma(collection) == 0
    assert kb.doc_count == 2
    assert kb.search("退款")[0]["citation_id"] == "legacy-1"
    kb.add_documents([{"title": "退款", "content": "新退款规则"}])
    assert kb.doc_count == 2
    assert "legacy-1" not in kb._index.get()["ids"]


def test_sag_migration_does_not_resurrect_replaced_legacy_title(kb):
    kb.add_documents([{"title": "Policy", "content": "new revision"}])
    legacy = {"ids": ["old"], "documents": ["old revision"], "metadatas": [{"title": "Policy"}]}
    collection = SimpleNamespace(count=lambda: 1, get=lambda **kwargs: legacy)
    assert kb.migrate_from_chroma(collection) == 0
    assert kb._index.get()["documents"] == ["new revision"]


def test_sag_failed_migration_can_retry_without_partial_rows(kb, monkeypatch):
    legacy = {"ids": ["one", "two"], "documents": ["first", "second"],
              "metadatas": [{"title": "one"}, {"title": "two"}]}
    collection = SimpleNamespace(count=lambda: 2, get=lambda **kwargs: legacy)
    upsert = kb._index.upsert

    def interrupted(**kwargs):
        upsert(**kwargs)
        if kwargs["ids"] == ["two"]:
            raise RuntimeError("interrupted")

    monkeypatch.setattr(kb._index, "upsert", interrupted)
    with pytest.raises(RuntimeError):
        kb.migrate_from_chroma(collection)
    assert kb.doc_count == 0
    assert kb._index.setting("chroma_migration") is None
    monkeypatch.setattr(kb._index, "upsert", upsert)
    assert kb.migrate_from_chroma(collection) == 2


def test_sag_failed_judge_batch_cancels_pending_requests(monkeypatch):
    monkeypatch.setenv("ECHOMIND_SAG2_JUDGE", "llm")
    pending_started = asyncio.Event()
    cancelled = []

    async def create(**kwargs):
        items = json.loads(kwargs["messages"][0]["content"])["candidates"]
        if items[0]["id"] == "0":
            await pending_started.wait()
            return SimpleNamespace(content=[{"type": "text", "text": "invalid JSON"}])
        pending_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    judge = EvidenceJudge(SimpleNamespace(messages=SimpleNamespace(create=create)), "test-model")
    candidates = [{"id": str(i), "title": "test", "content": "source", "score": .1} for i in range(21)]
    _, method = asyncio.run(judge.score("question", candidates, []))
    assert method == "lexical-fallback"
    assert cancelled == [True]


def test_sag_llm_judge_validates_scores_and_has_visible_fallback(monkeypatch):
    monkeypatch.setenv("ECHOMIND_SAG2_JUDGE", "llm")
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[{"type": "text", "text": '{"scores":{"first":1.0}}'}])

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    judge = EvidenceJudge(client, "test-model")
    candidate = {"id": "first", "title": "test", "content": "source", "score": .2}
    scored, method = asyncio.run(judge.score("question", [candidate], []))
    assert scored[0]["score"] == 1 and method == "llm"
    assert json.loads(calls[0]["messages"][0]["content"])["original_question"] == "question"
    scored, method = asyncio.run(judge.score("question", [{**candidate, "id": "second"}], []))
    assert method == "lexical-fallback" and scored[0]["score"] == 1.0


def test_sag_jev_protocol_validates_noul(monkeypatch):
    monkeypatch.setenv("ECHOMIND_SAG2_JUDGE", "jev")
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    payloads = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, endpoint, *, json, headers):
            payloads.append(json)
            assert endpoint.endswith("/decisions")
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"answers": {"candidate_0": {"type": "noul", "noul": .9}}})

    monkeypatch.setattr("tooling.sag_retrieval.httpx.AsyncClient", Client)
    candidate = {"id": "source-id", "title": "test", "content": "source", "score": .2}
    scored, method = asyncio.run(EvidenceJudge().score("question", [candidate], []))
    assert method == "jev" and scored[0]["score"] == .9
    assert payloads[0]["questions"]["candidate_0"]["type"] == "noul"
