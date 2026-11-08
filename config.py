"""Настройки только из .env этого проекта или переменных процесса."""
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    api_key: str = field(repr=False)
    pinecone_key: str = field(repr=False)
    index_name: str
    bot_token: str = field(default="", repr=False)
    base_url: str = ""
    model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"
    namespace_prefix: str = "vpg05-memory"
    threshold: float = 0.75
    dimension: int = 1536
    max_tokens: int = 800
    allowed_users: tuple[int, ...] = ()
    visibility_timeout: float = 30.0


def load_config(require_bot=False):
    load_dotenv(ROOT / ".env")
    required = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    if require_bot:
        required += ["TELEGRAM_BOT_TOKEN", "ALLOWED_USER_IDS"]
    if any(not os.getenv(k, "").strip() for k in required):
        raise ValueError("Заполните настройки: " + ", ".join(required))
    c = Config(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        pinecone_key=os.getenv("PINECONE_API_KEY", ""),
        index_name=os.getenv("PINECONE_INDEX_NAME", ""),
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        model=os.getenv("CHAT_MODEL", "gpt-6-luna"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        namespace_prefix=os.getenv("PINECONE_NAMESPACE_PREFIX", "vpg05-memory"),
        threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.75")),
        allowed_users=tuple(int(x.strip()) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()),
    )
    if not math.isfinite(c.threshold) or not 0 <= c.threshold <= 1:
        raise ValueError("SIMILARITY_THRESHOLD должен быть от 0 до 1.")
    if not c.namespace_prefix.startswith("vpg05-") or len(c.namespace_prefix) > 40:
        raise ValueError("Нужен namespace prefix vpg05- длиной до 40 символов.")
    return c

