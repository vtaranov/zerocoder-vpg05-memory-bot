"""Автономные проверки ветвлений, изоляции и обработчика; внешние API заменены."""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock
from config import Config
from pinecone_manager import PineconeManager, optional_profile
from service import MemoryService
from bot import build_bot


class FakeIndex:
    def __init__(self):
        self.rows = {}

    def upsert(self, namespace, vectors):
        self.rows.setdefault(namespace, {}).update({v["id"]: dict(v) for v in vectors})

    def fetch(self, namespace, ids):
        rows = self.rows.get(namespace, {})
        return NS(vectors={i: NS(metadata=rows[i]["metadata"]) for i in ids if i in rows})

    def query(self, namespace, vector, top_k, include_metadata, filter=None):
        rows = self.rows.get(namespace, {})
        matches = []
        for i, r in rows.items():
            if filter and r["metadata"].get("memory_id") != filter["memory_id"]["$eq"]:
                continue
            v = r["values"]
            score = sum(a*b for a,b in zip(v,vector)) / (math.sqrt(sum(a*a for a in v))*math.sqrt(sum(a*a for a in vector)))
            matches.append(NS(id=i, metadata=r["metadata"], score=score))
        return NS(matches=sorted(matches, key=lambda m: m.score, reverse=True)[:top_k])

    def list(self, namespace):
        yield list(self.rows.get(namespace, {}))

    def delete(self, namespace, delete_all):
        self.rows.pop(namespace, None)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.config = Config("dummy", "dummy", "dummy", dimension=2, allowed_users=(7,))
        self.ai = Mock()
        self.ai.embeddings.create.side_effect = lambda **kw: NS(data=[NS(embedding=[0.,1.] if kw["input"]=="pizza" else [1.,0.])])
        self.judge = Mock(return_value={"relation": "duplicate", "reason": "перефразирование"})
        self.index = FakeIndex()
        self.m = PineconeManager(self.config, ai=self.ai, index=self.index, judge=self.judge)

    def test_exact_repeat_skips_without_model(self):
        a = self.m.upsert_document("alice", "Mars")
        b = self.m.upsert_document("alice", "Mars")
        self.assertEqual((a["action"], b["action"]), ("inserted", "skipped"))
        self.assertEqual(a["id"], b["id"])
        self.judge.assert_not_called()

    def test_paraphrase_skips(self):
        self.m.upsert_document("alice", "Mars")
        self.assertEqual(self.m.upsert_document("alice", "To Mars")["action"], "skipped")
        self.assertEqual(len(self.m.list_memories("alice")), 1)

    def test_revision_replaces_text_same_id(self):
        a = self.m.upsert_document("alice", "Mars")
        self.judge.return_value = {"relation": "revision", "reason": "отрицание"}
        b = self.m.upsert_document("alice", "Not Mars")
        self.assertEqual(b["action"], "updated")
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(self.m.list_memories("alice")[0]["metadata"]["user_message"], "Not Mars")

    def test_similar_but_different_subject_is_inserted(self):
        self.m.upsert_document("alice", "Mars")
        self.judge.return_value = {"relation": "different", "reason": "другой субъект"}
        b = self.m.upsert_document("alice", "Brother Mars")
        self.assertEqual(b["action"], "inserted")
        self.assertEqual(len(self.m.list_memories("alice")), 2)

    def test_low_similarity_is_inserted_without_judge(self):
        self.m.upsert_document("alice", "Mars")
        b = self.m.upsert_document("alice", "pizza")
        self.assertEqual(b["action"], "inserted")
        self.judge.assert_not_called()

    def test_owner_isolation_and_scoped_reset(self):
        self.m.upsert_document("alice", "Mars")
        b = self.m.upsert_document("bob", "Mars")
        self.assertEqual(b["action"], "inserted")
        self.m.reset("alice")
        self.assertEqual(self.m.list_memories("alice"), [])
        self.assertEqual(len(self.m.list_memories("bob")), 1)

    def test_missing_profile_and_metadata_allowlist(self):
        self.assertEqual(optional_profile(NS(first_name=None)), {})
        self.m.upsert_document("alice", "  Mars  ", {"first_name": None, "last_name": "",
                              "username": "learner", "bot_response": "invented", "user_message": "wrong"})
        meta = self.m.list_memories("alice")[0]["metadata"]
        self.assertEqual(meta["user_message"], "  Mars  ")
        self.assertNotIn("bot_response", meta)
        self.assertNotIn("first_name", meta)
        self.assertEqual(meta["username"], "learner")

    def test_judge_error_does_not_overwrite(self):
        a = self.m.upsert_document("alice", "Mars")
        self.judge.side_effect = RuntimeError("API down")
        with self.assertRaises(RuntimeError):
            self.m.upsert_document("alice", "New Mars")
        self.assertEqual(self.m.list_memories("alice")[0]["metadata"]["user_message"], "Mars")

    def test_bad_embedding_rejected(self):
        self.ai.embeddings.create.side_effect = None
        self.ai.embeddings.create.return_value = NS(data=[NS(embedding=[float("nan"), 0])])
        with self.assertRaises(ValueError):
            self.m.upsert_document("alice", "Mars")
        self.assertEqual(self.index.rows, {})

    def test_base_url_passed_to_client(self):
        from unittest.mock import patch
        with patch("pinecone_manager.OpenAI") as client:
            PineconeManager(replace(self.config, base_url="https://example.test/v1"), index=self.index)
            self.assertEqual(client.call_args.kwargs["base_url"], "https://example.test/v1")

    def test_answer_not_saved_and_failure_is_explicit(self):
        self.m.response = Mock(side_effect=RuntimeError("API unavailable"))
        r = MemoryService(self.m).handle("alice", "Mars")
        self.assertFalse(r["answer_ok"])
        self.assertEqual(self.m.list_memories("alice")[0]["metadata"]["user_message"], "Mars")

    def test_telegram_handler_missing_names_and_access_control(self):
        # Реальный TeleBot dispatch handler, только send_message и сервис заменены.
        from telebot import TeleBot
        bot = TeleBot("123456:dummy", threaded=False)
        bot.send_message = Mock()
        svc = Mock()
        svc.handle.return_value = {"answer": "Ответ", "answer_ok": True}
        build_bot(self.config, svc, bot)
        handler = bot.message_handlers[0]["function"]
        msg = NS(from_user=NS(id=7), chat=NS(id=7, type="private"), text="Привет")
        handler(msg)
        svc.handle.assert_called_once_with("telegram:7:7", "Привет", msg.from_user)
        msg.from_user.id = 8
        handler(msg)
        self.assertEqual(svc.handle.call_count, 1)


if __name__ == "__main__":
    unittest.main()

