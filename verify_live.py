"""Реальная проверка API в уникальных namespace VPg05, без отправки в Telegram."""
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from config import ROOT, load_config
from pinecone_manager import PineconeManager
from service import MemoryService


def run():
    config = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    owner = "validation:" + run_id
    other = owner + ":other"
    output = ROOT / "evidence" / ("live-" + run_id + ".json")
    output.parent.mkdir(exist_ok=True)
    report = {"run_id": run_id, "model": config.model, "threshold": config.threshold,
              "steps": [], "checks": {}, "passed": False}
    manager = None
    try:
        manager = PineconeManager(config)
        report["probe"] = manager.probe()
        report["namespace"] = manager.namespace(owner)
        assert manager.list_memories(owner) == []
        phrases = [
            ("Я хочу полететь на Марс.", "inserted"),
            ("Я хотел бы полететь на Марс.", "skipped"),
            ("Я хочу полететь на Марс.", "skipped"),
            ("Я не хочу полететь на Марс.", "updated"),
            ("Мой брат хочет полететь на Марс.", "inserted"),
            ("Я люблю пиццу.", "inserted"),
        ]
        for text, expected in phrases:
            result = manager.upsert_document(owner, text, {"first_name": None, "bot_response": "NEVER_STORE"})
            report["steps"].append({"text": text, "expected": expected, **result})
            print(json.dumps({"text": text, **result}, ensure_ascii=False), flush=True)
            assert result["action"] == expected, (text, result)
        rows = manager.list_memories(owner)
        report["records_before_restart"] = rows
        report["checks"]["three_memories"] = len(rows) == 3
        assert len(rows) == 3
        texts = [r["metadata"]["user_message"] for r in rows]
        assert "Я не хочу полететь на Марс." in texts and "Я хочу полететь на Марс." not in texts
        assert all("bot_response" not in r["metadata"] and "first_name" not in r["metadata"] for r in rows)
        report["checks"]["raw_user_text_only"] = True

        assert manager.query_by_text(other, "Марс") == []
        other_saved = manager.upsert_document(other, "Я хочу полететь на Марс.")
        assert other_saved["action"] == "inserted"
        report["checks"]["owner_isolation"] = True
        manager.close()
        manager = PineconeManager(config)
        restored = manager.list_memories(owner)
        assert restored == rows
        report["checks"]["reconnect_persistence"] = True
        answer = MemoryService(manager).handle(owner, "Хочу ли я полететь на Марс?", SimpleNamespace())
        report["answer_after_reconnect"] = answer
        print(json.dumps({"answer_after_reconnect": answer}, ensure_ascii=False), flush=True)
        assert answer["answer_ok"]
        report["checks"]["answer_generated"] = True
        manager.reset(owner)
        assert manager.list_memories(owner) == []
        assert len(manager.list_memories(other)) == 1
        report["checks"]["scoped_reset_preserves_other"] = True
        manager.reset(other)
        report["checks"]["test_namespaces_cleaned"] = True
        report["passed"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        # Ошибки SDK могут содержать чувствительные данные; в отчёт только тип.
        print("LIVE CHECK FAILED:", type(exc).__name__, flush=True)
    finally:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if manager:
            manager.close()
    print("Evidence:", output.name, "passed=", report["passed"], flush=True)
    return report["passed"]


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

