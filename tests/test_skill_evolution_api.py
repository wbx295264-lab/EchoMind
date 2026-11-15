"""Learning and SAG2 must remain connected to authenticated, retryable API flows."""
import json
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from test_chat_reliability import body, services, text_response
import api.main as api
from agents.agent_orchestrator import (
    AgentResponse, AgentType, GeneralAgent, Request, ResponseComposer,
)
from tooling.tool_manager import MCPToolManager, Tool
from core.skill_evolution import SkillEvolution
from core.skill_loader import SkillManager


def user_headers(monkeypatch, role="user"):
    token = "skill-test-token-with-at-least-24-characters"
    monkeypatch.setenv("ECHOMIND_API_TOKENS", json.dumps({
        token: {"user_id": "authenticated-user", "role": role},
    }))
    return {"Authorization": "Bearer " + token}


@pytest.mark.asyncio
async def test_learning_is_submitted_after_commit_and_replay_uses_same_identity(services, monkeypatch):
    memory, orch, model, http = services
    calls = []
    commit = memory.commit_turn

    async def record_commit(*args, **kwargs):
        result = await commit(*args, **kwargs)
        calls.append("commit")
        return result

    monkeypatch.setattr(memory, "commit_turn", record_commit)
    submit = Mock(side_effect=lambda *args, **kwargs: calls.append("submit"))
    monkeypatch.setattr(api, "_skill_evolution", SimpleNamespace(submit_turn=submit))
    headers = user_headers(monkeypatch)
    first = await http.post("/chat", json=body(), headers=headers)
    assert first.status_code == 200
    assert calls == ["commit", "submit"]
    replay = await http.post("/chat", json=body(), headers=headers)
    assert replay.json() == first.json()
    assert model.messages.create.await_count == 1
    assert submit.call_count == 2
    assert submit.call_args_list[0] == submit.call_args_list[1]
    submit.assert_called_with(
        user_id="authenticated-user", conv_id="c", request_id="one",
        message="退款规则", response=first.json()["response"],
        agent_type=first.json()["agent_type"],
    )


@pytest.mark.asyncio
async def test_real_chat_feedback_learns_once_and_reaches_only_owner_prompt(services, monkeypatch, tmp_path):
    memory, orch, model, http = services
    skills = SkillManager(str(tmp_path / "base-skills"))
    feedback = "以后退款咨询先给简短步骤"
    rule = "先用三个简短步骤说明退款咨询流程。"
    learner = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=[
        text_response(json.dumps({"skill": {
            "name": "退款步骤", "description": "用户偏好的退款说明格式", "instructions": rule,
            "evidence": feedback, "keywords": ["退款"], "agents": ["billing"],
        }}, ensure_ascii=False)),
        text_response('{"action":"add","reason":"明确可复用的表达偏好"}'),
    ])))
    evolution = SkillEvolution(learner, "test-model", skills, tmp_path / "evolution")
    monkeypatch.setattr(api, "_skill_evolution", evolution)
    orch.set_skill_manager(skills)
    first = await http.post("/chat", json=body())
    assert first.status_code == 200
    second_body = body(feedback, request_id="two")
    second = await http.post("/chat", json=second_body)
    assert second.status_code == 200
    assert (await http.post("/chat", json=second_body)).json() == second.json()
    await evolution.recover_pending()
    summary = evolution.summary("u")
    assert summary["jobs"] == {"done": 1}
    assert len(summary["skills"]) == 1 and summary["skills"][0]["version"] == 1
    await http.post("/chat", json=body("退款如何办理", request_id="three"))
    assert rule in model.messages.create.call_args.kwargs["system"]
    another_user = body("退款如何办理", request_id="four") | {"user_id": "another-owner"}
    await http.post("/chat", json=another_user)
    assert rule not in model.messages.create.call_args.kwargs["system"]
    assert learner.messages.create.await_count == 2
    await evolution.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("text,stop", [("不完整", "max_tokens"), ("", "end_turn")])
async def test_incomplete_answers_do_not_become_learning_windows(services, monkeypatch, text, stop):
    memory, orch, model, http = services
    model.messages.create.return_value = text_response(text, stop)
    submit = Mock()
    monkeypatch.setattr(api, "_skill_evolution", SimpleNamespace(submit_turn=submit))
    await http.post("/chat", json=body())
    submit.assert_not_called()


@pytest.mark.asyncio
async def test_learning_storage_failure_preserves_committed_chat(services, monkeypatch):
    memory, orch, model, http = services
    submit = Mock(side_effect=OSError("read-only evolution database"))
    monkeypatch.setattr(api, "_skill_evolution", SimpleNamespace(submit_turn=submit))
    first = await http.post("/chat", json=body())
    replay = await http.post("/chat", json=body())
    assert first.status_code == replay.status_code == 200
    assert first.json() == replay.json()
    assert model.messages.create.await_count == 1
    assert await memory.get_turn_receipt("u", "c", "one")


@pytest.mark.asyncio
async def test_evolution_status_uses_authenticated_owner(services, monkeypatch):
    _, _, _, http = services
    summary = Mock(return_value={"enabled": True, "skills": []})
    monkeypatch.setattr(api, "_skill_evolution", SimpleNamespace(summary=summary))
    response = await http.get("/skills/evolution?user_id=another-user", headers=user_headers(monkeypatch))
    assert response.status_code == 200
    summary.assert_called_once_with(user_id="authenticated-user")


@pytest.mark.asyncio
async def test_admin_can_inspect_explicit_owner_but_default_stays_scoped(services, monkeypatch):
    _, _, _, http = services
    summary = Mock(return_value={"enabled": True, "skills": []})
    monkeypatch.setattr(api, "_skill_evolution", SimpleNamespace(summary=summary))
    headers = user_headers(monkeypatch, role="admin")
    assert (await http.get("/skills/evolution?user_id=another-user", headers=headers)).status_code == 200
    summary.assert_called_with(user_id="another-user")
    assert (await http.get("/skills/evolution", headers=headers)).status_code == 200
    summary.assert_called_with(user_id="authenticated-user")


@pytest.mark.asyncio
async def test_manual_extraction_binds_owner_and_passes_idempotency_key(services, monkeypatch):
    _, _, _, http = services
    extract = AsyncMock(return_value={"action": "discard"})
    monkeypatch.setattr(api, "_skill_evolution", SimpleNamespace(extract_now=extract))
    response = await http.post("/skills/evolution/extract", headers=user_headers(monkeypatch), json={
        "user_id": "another-user", "conv_id": "c", "feedback": "以后先给简短步骤",
        "client_request_id": "feedback-one",
    })
    assert response.status_code == 200
    extract.assert_awaited_once_with(
        user_id="authenticated-user", conv_id="c", feedback="以后先给简短步骤",
        request_id="feedback-one",
    )


@pytest.mark.asyncio
async def test_uninitialized_evolution_reports_unavailable(services, monkeypatch):
    _, _, _, http = services
    monkeypatch.setattr(api, "_skill_evolution", None)
    assert (await http.get("/skills/evolution")).status_code == 503
    result = await http.post("/skills/evolution/extract", json={
        "conv_id": "c", "feedback": "以后先给简短步骤", "client_request_id": "one",
    })
    assert result.status_code == 503


@pytest.mark.asyncio
async def test_agent_and_composer_forward_skill_owner():
    skills = SimpleNamespace(prompt_for=Mock(return_value="该用户的规则"))
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=text_response())))
    req = Request("退款和登录", "owner-a", "conversation")
    agent = GeneralAgent(client, "test-model", skill_manager=skills)
    assert "该用户的规则" in agent._build_system_prompt(req)
    skills.prompt_for.assert_called_once_with(req.message, "general", user_id="owner-a")
    skills.prompt_for.reset_mock()
    composer = ResponseComposer(client, "test-model", skill_manager=skills)
    await composer.compose(req, [
        AgentResponse(AgentType.TECHNICAL, "登录步骤", True),
        AgentResponse(AgentType.BILLING, "退款规则", True),
    ])
    skills.prompt_for.assert_called_once_with(req.message, "general", user_id="owner-a")


@pytest.mark.asyncio
async def test_sag2_pipeline_preserves_query_evidence_order_and_empty_results():
    manager = MCPToolManager(api_key="test")
    docs = [{"title": "桥接证据", "content": "A -> B", "hop": 1},
            {"title": "答案证据", "content": "B -> C", "hop": 2}]
    handler = AsyncMock(return_value=docs)
    manager.register(Tool("knowledge_search", "SAG2", handler, {}, cache_ttl=30,
                          handles_retrieval=True, supports_rerank=False))
    manager.rewrite_query = AsyncMock(side_effect=AssertionError("SAG2 owns expansion"))
    manager._rerank_with_status = AsyncMock(side_effect=AssertionError("SAG2 owns ranking"))
    try:
        result = await manager.search_with_rewrite("knowledge_search", "原始多跳问题", 2)
        assert result.success and result.data == docs
        handler.assert_awaited_once_with({"query": "原始多跳问题", "top_k": 2}, None)
        cached = await manager.search_with_rewrite("knowledge_search", "原始多跳问题", 2)
        assert cached.cached and cached.data == docs
        manager.invalidate_cache("knowledge_search")
        handler.return_value = []
        empty = await manager.search_with_rewrite("knowledge_search", "无匹配内容", 2)
        assert empty.success and empty.data == []
        manager.rewrite_query.assert_not_awaited()
        manager._rerank_with_status.assert_not_awaited()
    finally:
        await manager._client.close()


@pytest.mark.asyncio
async def test_lifespan_connects_sag2_import_search_mcp_and_evolution(monkeypatch, tmp_path):
    from evaluation.jobs import EvaluationJobs
    from tooling import mcp_server
    import memory.conversation_memory as memory_module
    import monitor.performance_monitor as monitor_module

    # MCP session managers are single-use. Give this lifespan its own transport.
    monkeypatch.setattr(mcp_server.server, "_session_manager", None)
    isolated_mcp = mcp_server.server.streamable_http_app()
    mount = next(route for route in api.app.routes if getattr(route, "path", None) == "/mcp")
    monkeypatch.setattr(mount, "app", isolated_mcp)
    for name in ("_orchestrator", "_memory", "_tool_manager", "_monitor", "_evaluator",
                 "_eval_orchestrator", "_eval_tool_manager", "_skill_manager",
                 "_skill_evolution", "_profile_updater", "_mcp_client"):
        monkeypatch.setattr(api, name, None)
    monkeypatch.setattr(api, "_eval_jobs", EvaluationJobs())
    configuration = {
        "ANTHROPIC_API_KEY": "test-key", "ECHOMIND_ENV": "development",
        "ECHOMIND_KB_BACKEND": "sag2", "ECHOMIND_KB_MIGRATE_CHROMA": "off",
        "ECHOMIND_SEED_DEMO_DOCS": "false", "ECHOMIND_SAG2_JUDGE": "lexical",
        "ECHOMIND_SAG2_PATH": str(tmp_path / "knowledge.sqlite3"),
        "ECHOMIND_SKILL_EVOLUTION_DIR": str(tmp_path / "evolution"),
        "ECHOMIND_SKILLS_DIR": str(tmp_path / "skills"),
        "ECHOMIND_AUTO_SKILL_EVOLUTION": "true", "MCP_EXTERNAL_ENABLED": "false",
        "INTENT_EMBEDDING_ENABLED": "false", "EVAL_BASELINE_PATH": str(tmp_path / "baseline.json"),
    }
    for name, value in configuration.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ECHOMIND_API_TOKENS", raising=False)
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()), close=AsyncMock())
    for name in ("agents.agent_orchestrator", "core.intent_recognizer", "tooling.tool_manager", "evaluation.evaluator"):
        monkeypatch.setattr(importlib.import_module(name), "AsyncAnthropic", Mock(return_value=client))
    memory = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(memory_module, "MemoryManager", Mock(return_value=memory))
    monitor = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), _prom={})
    monkeypatch.setattr(monitor_module, "PerformanceMonitor", Mock(return_value=monitor))

    async with api.lifespan(api.app):
        kb = api._tool_manager._tools["knowledge_search"].handler.__self__
        assert kb.backend == "sag2"
        assert kb._retriever.judge.client is client
        assert api._skill_manager.evolution is api._skill_evolution
        assert api._skill_evolution.client is client
        assert api._skill_evolution._worker_task is not None
        assert api._tool_manager._tools["knowledge_search"].handles_retrieval
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://localhost") as http:
            added = await http.post("/knowledge/add", json={"documents": [
                {"title": "退款规则", "content": "退款申请需要提供订单编号。", "document_id": "refund"},
            ]})
            assert added.status_code == 200 and added.json()["added_chunks"] == 1
            initial = await http.post("/search", params={"query": "退款", "top_k": 2})
            assert initial.status_code == 200 and initial.json()["results"][0]["title"] == "退款规则"
            assert initial.json()["results"][0]["document_id"]
            cached = await http.post("/search", params={"query": "退款", "top_k": 2})
            assert cached.json()["cached"]
            uploaded = await http.post("/knowledge/upload", files={
                "file": ("refund.md", "退款审核需要三个工作日。".encode("utf-8"), "text/markdown"),
            })
            assert uploaded.status_code == 200 and uploaded.json()["total_chunks"] == 2
            refreshed = await http.post("/search", params={"query": "退款", "top_k": 2})
            assert not refreshed.json()["cached"]
            assert {row["title"] for row in refreshed.json()["results"]} == {"退款规则", "refund"}
            stats = (await http.get("/knowledge/stats")).json()
            assert stats["backend"] == "sag2" and stats["total_chunks"] == 2
            async with streamable_http_client("http://localhost/mcp/", http_client=http) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("search_local_knowledge", {"query": "退款", "top_k": 2})
                    assert not result.isError
                    assert len(result.structuredContent["results"]) == 2
                    resource = await session.read_resource("echomind://knowledge/stats")
                    assert json.loads(resource.contents[0].text)["total_chunks"] == 2
        evolution = api._skill_evolution
    assert evolution._worker_task is None
    assert mcp_server.knowledge_base is None
    memory.close.assert_awaited_once()
    monitor.stop.assert_awaited_once()
