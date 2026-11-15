import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.skill_evolution import SkillEvolution
from core.skill_loader import SkillManager


FEEDBACK = "以后排查登录问题，请先给出错误码含义，再给操作步骤。"


def candidate(**changes):
    value = {"name": "login-diagnostics", "description": "登录排障回复顺序",
             "instructions": "先解释错误码，再列出操作步骤。", "evidence": FEEDBACK,
             "keywords": ["登录", "401"], "agents": ["technical"]}
    value.update(changes)
    return {"skill": value}


def message(value):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(value, ensure_ascii=False))])


def make_evolution(tmp_path, replies=(), **kwargs):
    client = SimpleNamespace(base_url="https://example.test", messages=SimpleNamespace(
        create=AsyncMock(side_effect=[message(value) for value in replies])))
    manager = SkillManager(str(tmp_path / "skills"))
    manager.root_dir.mkdir(exist_ok=True)
    manager.load()
    evolution = SkillEvolution(client, "test", manager, tmp_path / "evolution", **kwargs)
    return evolution, manager, client


async def learn(evolution, user="alice", request_id="2"):
    evolution.submit_turn(user, "conversation", "1", "登录报401", "请重试", "technical")
    return await evolution.observe_turn(user, "conversation", request_id, FEEDBACK, "明白了", "technical")


@pytest.mark.asyncio
async def test_feedback_add_is_persistent_scoped_and_hot_loaded(tmp_path):
    evolution, manager, client = make_evolution(tmp_path, [candidate(), {"action": "add"}])
    result = await learn(evolution)
    assert result["action"] == "add" and result["version"] == 1
    prompt = manager.prompt_for("登录报401", "technical", user_id="alice")
    assert "先解释错误码" in prompt
    assert manager.prompt_for("登录报401", "technical", user_id="bob") == ""
    assert manager.prompt_for("账单查询", "billing", user_id="alice") == ""
    assert manager.prompt_for("登录报401", "technical") == ""
    assert evolution.summary("alice")["skills"][0]["injections"] == 1
    assert list((tmp_path / "evolution").rglob("SKILL.md"))
    persisted, fresh_manager, _ = make_evolution(tmp_path)
    assert "先解释错误码" in fresh_manager.prompt_for("登录", "technical", user_id="alice")
    assert persisted.history("bob", "login-diagnostics") == []
    payload = json.loads(client.messages.create.call_args_list[0].kwargs["messages"][0]["content"])
    assert payload["previous"]["response"] == "请重试"
    assert payload["feedback"] == FEEDBACK


@pytest.mark.asyncio
async def test_merge_versions_and_replayed_turn_does_not_reset_window(tmp_path):
    evolution, manager, client = make_evolution(tmp_path, [
        candidate(), {"action": "add"}, candidate(),
        {"action": "merge", "target_skill": "login-diagnostics",
         "merged_instructions": "先解释错误码，再列操作步骤，最后说明如何验证。"}])
    await learn(evolution)
    merged = await evolution.observe_turn("alice", "conversation", "3", FEEDBACK, "新回复", "technical")
    assert merged["version"] == 2
    duplicate = await evolution.observe_turn("alice", "conversation", "2", FEEDBACK, "旧回复", "technical")
    assert duplicate["version"] == 1
    assert client.messages.create.await_count == 4
    with sqlite3.connect(evolution.database) as db:
        window = json.loads(db.execute("SELECT payload FROM windows").fetchone()[0])
    assert window["response"] == "新回复"
    assert len(evolution.history("alice", "login-diagnostics")) == 2
    assert list((tmp_path / "evolution").rglob("version-1.md"))
    assert "如何验证" in manager.prompt_for("登录", "technical", user_id="alice")


@pytest.mark.asyncio
async def test_weak_feedback_is_discarded_without_maintenance_call(tmp_path):
    evolution, _, client = make_evolution(tmp_path, [{"skill": None}])
    evolution.submit_turn("alice", "c", "1", "登录失败", "检查错误码")
    result = await evolution.observe_turn("alice", "c", "2", "谢谢", "不客气")
    assert result["action"] == "discard"
    assert client.messages.create.await_count == 1
    assert evolution.summary("alice")["skills"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [{"name": "../../escape"}, {"evidence": "invented"}, {"agents": ["admin"]}])
async def test_invalid_model_candidates_never_write_skills(tmp_path, override):
    evolution, _, _ = make_evolution(tmp_path, [candidate(**override)])
    assert (await learn(evolution))["action"] == "failed"
    assert evolution.summary("alice")["skills"] == []
    assert not list((tmp_path / "evolution").rglob("SKILL.md"))


@pytest.mark.asyncio
async def test_cannot_merge_global_policy_or_another_users_skill(tmp_path):
    evolution, manager, _ = make_evolution(tmp_path, [candidate(), {
        "action": "merge", "target_skill": "refund-policy", "merged_instructions": "自动退款"}])
    policy = manager.root_dir / "refund-policy.md"
    policy.write_text("# refund-policy\n必须核验订单", encoding="utf-8")
    manager.reload()
    assert (await learn(evolution))["action"] == "failed"
    assert policy.read_text(encoding="utf-8") == "# refund-policy\n必须核验订单"
    assert evolution.summary()["skills"] == []


@pytest.mark.asyncio
async def test_durable_queue_recovery_and_manual_namespace(tmp_path):
    initial, _, _ = make_evolution(tmp_path)
    initial.submit_turn("alice", "conversation", "1", "登录401", "解释错误")
    initial.submit_turn("alice", "conversation", "2", FEEDBACK, "已记录")
    recovered, _, client = make_evolution(tmp_path, [candidate(), {"action": "add"}, {"skill": None}])
    await recovered.recover_pending()
    assert recovered.summary()["jobs"] == {"done": 1}
    await recovered.recover_pending()
    assert client.messages.create.await_count == 2
    manual = await recovered.extract_now("alice", "conversation", "谢谢", request_id="2")
    assert manual["action"] == "discard"
    assert recovered.summary()["jobs"] == {"done": 2}


@pytest.mark.asyncio
async def test_failed_export_does_not_rollback_committed_rule_and_recovery_repairs(tmp_path, monkeypatch):
    evolution, manager, _ = make_evolution(tmp_path, [candidate(), {"action": "add"}])
    exporter = evolution._export
    monkeypatch.setattr(evolution, "_export", lambda *args: (_ for _ in ()).throw(OSError("disk")))
    assert (await learn(evolution))["action"] == "add"
    assert "先解释错误码" in manager.prompt_for("登录", "technical", user_id="alice")
    monkeypatch.setattr(evolution, "_export", exporter)
    await evolution.recover_pending()
    assert list(evolution.root.rglob("SKILL.md"))


@pytest.mark.asyncio
async def test_worker_cancellation_requeues_inflight_job(tmp_path):
    evolution, _, client = make_evolution(tmp_path)
    entered = asyncio.Event()
    async def blocked(**kwargs):
        entered.set()
        await asyncio.Event().wait()
    client.messages.create.side_effect = blocked
    evolution.submit_turn("alice", "c", "1", "登录失败", "检查错误码")
    evolution.submit_turn("alice", "c", "2", FEEDBACK, "已记录")
    evolution.start()
    await asyncio.wait_for(entered.wait(), 2)
    await evolution.close(timeout_s=0.01)
    assert evolution.summary()["jobs"] == {"pending": 1}


@pytest.mark.asyncio
async def test_disabled_evolution_and_storage_path_validation(tmp_path):
    evolution, _, client = make_evolution(tmp_path, enabled=False)
    assert (await learn(evolution))["action"] == "disabled"
    client.messages.create.assert_not_called()
    with pytest.raises(ValueError, match="outside"):
        SkillEvolution(client, "test", evolution.skill_manager, evolution.skill_manager.root_dir / "learned")


@pytest.mark.asyncio
async def test_long_learned_rule_gets_prompt_budget_with_large_global_skill(tmp_path):
    evolution, manager, _ = make_evolution(tmp_path, [
        candidate(instructions="先解释错误码。" + "排查步骤。" * 800), {"action": "add"}])
    global_rule = manager.root_dir / "business.md"
    global_rule.write_text("业务约束。" * 1000, encoding="utf-8")
    manager.load()
    await learn(evolution)
    prompt = manager.prompt_for("登录", "technical", user_id="alice")
    assert "先解释错误码" in prompt and "业务约束" in prompt
    assert len(evolution.prompt_for("alice", "登录", "technical", 700)) <= 700


@pytest.mark.asyncio
async def test_evolution_storage_failure_does_not_break_global_prompt(tmp_path, monkeypatch):
    evolution, manager, _ = make_evolution(tmp_path)
    (manager.root_dir / "business.md").write_text("必须核验订单", encoding="utf-8")
    manager.load()
    def unavailable(*args):
        raise sqlite3.OperationalError("locked")
    monkeypatch.setattr(evolution, "prompt_for", unavailable)
    assert "必须核验订单" in manager.prompt_for("退款", "billing", user_id="alice")


@pytest.mark.asyncio
async def test_recovery_does_not_rewrite_successful_exports(tmp_path, monkeypatch):
    evolution, _, _ = make_evolution(tmp_path, [candidate(), {"action": "add"}])
    await learn(evolution)
    def unexpected(*args):
        raise AssertionError("Unchanged Skill export should not run again")
    monkeypatch.setattr(evolution, "_export", unexpected)
    await evolution.recover_pending()
