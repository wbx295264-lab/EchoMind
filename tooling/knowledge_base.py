"""SAG2 source knowledge base with a compatible Chroma migration path."""
import asyncio
import hashlib
import json
import logging
import os
import re
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
except ModuleNotFoundError:
    chromadb = None

from tooling.sag_index import SQLiteKnowledgeIndex
from tooling.sag_retrieval import EvidenceJudge, SAG2Retriever

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Source import, citation metadata and bounded multi-hop evidence retrieval."""

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        *,
        backend: Optional[str] = None,
        sqlite_path: Optional[str] = None,
        llm_client: Any = None,
        model: Optional[str] = None,
    ):
        self._write_lock = threading.RLock()
        self._use_server = False
        self.backend = (backend or os.getenv("ECHOMIND_KB_BACKEND", "sag2")).lower()
        if self.backend not in {"sag2", "chroma"}:
            raise ValueError("ECHOMIND_KB_BACKEND 必须为 sag2 或 chroma")
        self.supports_rerank = self.backend == "chroma"
        if self.backend == "sag2":
            self._index = SQLiteKnowledgeIndex(sqlite_path or os.getenv("ECHOMIND_SAG2_PATH", "./data/knowledge/sag2.sqlite3"))
            self._collection = self._index
            self._retriever = SAG2Retriever(self._index, EvidenceJudge(llm_client, model))
            self._initialize_migration(chroma_host, chroma_port, chroma_path)
            self._seed_if_empty()
            return
        # Explicit compatibility backend retains existing Chroma behavior.
        if chromadb is None:
            raise RuntimeError("Chroma 兼容后端需要安装 chromadb")
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception as ex:
            if os.getenv("ECHOMIND_ENV", "development").lower() == "production":
                raise RuntimeError("生产环境知识库 ChromaDB 不可用，禁止切换到本地库") from ex
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 不传 embedding_function 时使用 ChromaDB 客户端默认模型；
        # Docker 已预下载模型，本地运行首次使用可能触发下载。
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "EchoMind RAG 知识库"},
        )

        self._seed_if_empty()

    def _seed_if_empty(self):
        seed_demo = os.getenv("ECHOMIND_SEED_DEMO_DOCS", "true").lower() in {"1", "true", "yes"}
        if os.getenv("ECHOMIND_ENV", "development").lower() != "production" and seed_demo and self._collection.count() == 0:
            self._load_default_docs()

    def _initialize_migration(self, host, port, path):
        mode = os.getenv("ECHOMIND_KB_MIGRATE_CHROMA", "auto").lower()
        if mode not in {"auto", "local", "server", "off"}:
            raise ValueError("ECHOMIND_KB_MIGRATE_CHROMA 必须为 auto、local、server 或 off")
        previous = self._index.setting("chroma_migration")
        if previous or mode == "off":
            self.migration_status = previous or {"status": "disabled"}
            return
        local_exists = Path(path, "chroma.sqlite3").is_file()
        source = "local" if mode == "local" or (mode == "auto" and local_exists) else "server"
        try:
            if source == "local":
                if not local_exists:
                    raise FileNotFoundError("existing Chroma database not found")
                client = chromadb.PersistentClient(path=path, settings=chromadb.Settings(anonymized_telemetry=False))
            else:
                # Bound first-start discovery before constructing the legacy client.
                import httpx
                httpx.get(f"http://{host}:{port}/api/v1/heartbeat", timeout=1.5, trust_env=False).raise_for_status()
                client = chromadb.HttpClient(host=host, port=port, settings=chromadb.Settings(anonymized_telemetry=False))
            try:
                collection = client.get_collection(self.COLLECTION_NAME, embedding_function=None)
            except Exception as exc:
                if type(exc).__name__ in {"InvalidCollectionException", "NotFoundError"}:
                    self.migration_status = {"status": "no_legacy_collection", "source": source}
                    return
                raise
            self.migrate_from_chroma(collection, source=source)
        except Exception as exc:
            self.migration_status = {"status": "unavailable", "source": source, "error": type(exc).__name__}
            if mode != "auto" or local_exists or os.getenv("ECHOMIND_ENV", "development").lower() == "production":
                raise RuntimeError("旧知识库迁移失败，请恢复 Chroma 连接，或明确配置 ECHOMIND_KB_MIGRATE_CHROMA=off 使用独立新库") from exc
            logger.warning("旧知识库未发现或暂时不可用，迁移状态为 unavailable")

    def migrate_from_chroma(self, collection, *, source="provided"):
        """Copy original chunks and metadata exactly once, without embedding calls.

        Existing SQLite documents win over the migration snapshot. The migration
        marker and all imported rows commit together, so retries are idempotent.
        """
        if getattr(self, "backend", None) != "sag2":
            raise ValueError("migration requires the SAG2 backend")
        rows = []
        expected = collection.count()
        for offset in range(0, expected, 500):
            batch = collection.get(limit=500, offset=offset, include=["documents", "metadatas"])
            rows.extend(zip(batch["ids"], batch["documents"], batch["metadatas"]))
        if len(rows) != expected or collection.count() != expected:
            raise RuntimeError("legacy knowledge changed during migration; retry from a stable source")
        with self._write_lock, self._index.transaction():
            existing_marker = self._index.setting("chroma_migration")
            if existing_marker:
                self.migration_status = existing_marker
                return 0
            existing = self._index.get()
            existing_ids = set(existing["ids"])
            existing_documents = {meta.get("document_id") for meta in existing["metadatas"] if meta.get("document_id")}
            count = 0
            for passage_id, content, metadata in rows:
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("legacy passage contains no source text")
                metadata = dict(metadata or {})
                metadata.setdefault("title", "")
                metadata.setdefault("version", hashlib.sha256(content.encode()).hexdigest())
                title = metadata["title"]
                canonical_title_id = hashlib.sha256(json.dumps(["title", title], ensure_ascii=True).encode()).hexdigest()
                legacy_title_id = hashlib.sha256(f"title:{title}".encode()).hexdigest()
                replaced_title = canonical_title_id in existing_documents and metadata.get("document_id") in {None, "", legacy_title_id}
                if passage_id in existing_ids or metadata.get("document_id") in existing_documents or replaced_title:
                    continue
                self._index.upsert(ids=[passage_id], documents=[content], metadatas=[metadata])
                count += 1
            status = {"status": "complete", "source": source, "imported_chunks": count}
            self._index.set_setting("chroma_migration", status)
        self.migration_status = status
        return count

    def stats(self):
        return {"backend": self.backend, "total_chunks": self.doc_count,
                "judge": self._retriever.judge.provider if self.backend == "sag2" else None,
                "migration": getattr(self, "migration_status", None)}

    def close(self):
        if getattr(self, "backend", None) == "sag2":
            self._index.close()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        # 相同文档身份在同一批次只采用最后一个版本，避免重复 ID。
        prepared = {}
        for doc in documents:
            title, content = doc.get("title", ""), doc.get("content", "")
            if not isinstance(title, str) or not isinstance(content, str):
                raise ValueError("文档 title 和 content 必须是字符串")
            if not title.strip() or not content.strip():
                raise ValueError("文档标题和内容不能为空")
            explicit_id = doc.get("document_id")
            if explicit_id is not None and (not isinstance(explicit_id, str) or not explicit_id.strip()):
                raise ValueError("document_id 必须是非空字符串")
            identity = json.dumps(["explicit", explicit_id] if explicit_id else ["title", title], ensure_ascii=True)
            if not isinstance(identity, str):
                raise ValueError("document_id 必须是字符串")
            prepared[identity] = (title, content, bool(doc.get("document_id")))

        count = 0
        transaction = self._index.transaction() if getattr(self, "backend", None) == "sag2" else nullcontext()
        with self._write_lock, transaction:
            for identity, (title, content, explicit) in prepared.items():
                document_id = hashlib.sha256(identity.encode()).hexdigest()
                chunks = self._chunk_text(content, chunk_size=500)
                version = hashlib.sha256(content.encode()).hexdigest()
                ids = [f"{document_id}:{i}" for i in range(len(chunks))]
                existing = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
                old_ids = set(existing["ids"])
                # 仅迁移归属可验证的旧身份；旧显式 ID 与 title: 前缀冲突时保留旧记录供核验。
                kind, value = json.loads(identity)
                legacy_identity = value if explicit else f"title:{title}"
                if not explicit or not value.startswith("title:"):
                    legacy_id = hashlib.sha256(legacy_identity.encode()).hexdigest()
                    previous = self._collection.get(where={"document_id": legacy_id}, include=["metadatas"])
                    old_ids.update(i for i, meta in zip(previous["ids"], previous["metadatas"])
                                   if (meta or {}).get("title") == title)
                if not explicit:
                    # 兼容旧版只有 title/chunk_index 元数据的片段。
                    legacy = self._collection.get(where={"title": title}, include=["metadatas"])
                    old_ids.update(i for i, meta in zip(legacy["ids"], legacy["metadatas"])
                                   if not (meta or {}).get("document_id"))
                self._collection.upsert(ids=ids, documents=chunks, metadatas=[
                    {"title": title, "document_id": document_id, "version": version,
                     "chunk_index": i, "total_chunks": len(chunks)} for i in range(len(chunks))
                ])
                stale = old_ids - set(ids)
                if stale:
                    self._collection.delete(ids=list(stale))
                count += len(chunks)
        logger.info("知识库写入 %s 个文档片段", count)
        return count

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """Run transactional document imports outside the event loop."""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Synchronous entry point. Async callers should await search_async()."""
        if getattr(self, "backend", None) == "sag2":
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.search_async(query, top_k))
            raise RuntimeError("异步环境请使用 await search_async()")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k 必须是 1–20 的整数")
        count = self._collection.count()
        if not count:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                    "document_id": meta.get("document_id", ""),
                    "version": meta.get("version", ""),
                })

        return items

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve and judge evidence, then expand through source-grounded terms."""
        if getattr(self, "backend", None) == "sag2":
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query 必须是非空字符串")
            if type(top_k) is not int or not 1 <= top_k <= 20:
                raise ValueError("top_k 必须是 1–20 的整数")
            return await self._retriever.search(query, top_k)
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await self.search_async(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Bound chunks while preserving original punctuation and source text."""
        if chunk_size < 1:
            raise ValueError("chunk_size 必须大于 0")
        chunks, start = [], 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                boundaries = list(re.finditer(r"[。！？.!?\n]", text[start:end]))
                if boundaries and boundaries[-1].end() >= chunk_size // 2:
                    end = start + boundaries[-1].end()
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end
        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（客服场景常见问题）。"""
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
