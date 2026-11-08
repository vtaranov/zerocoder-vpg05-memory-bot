"""Telegram-интерфейс pyTelegramBotAPI; один процесс и последовательная обработка."""
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import telebot
from config import ROOT, load_config
from pinecone_manager import PineconeManager
from service import MemoryService

HELP = """Учебный бот VPg05 с памятью.
Напишите сообщение: я использую сохранённые сведения для ответа.
/memories - посмотреть сохранённые сообщения
/status - настройки памяти
/reset - инструкция очистки своей памяти
/reset confirm - очистить свою память
/help - эта справка
Память содержит только ваши сообщения. Данные передаются в OpenAI и Pinecone."""
logger = logging.getLogger(__name__)


def owner_for(message):
    return f"telegram:{message.chat.id}:{message.from_user.id}"


def build_bot(config, service, bot=None):
    bot = bot or telebot.TeleBot(config.bot_token, threaded=False, parse_mode=None)

    def send(chat_id, text):
        for start in range(0, len(text), 3500):
            bot.send_message(chat_id, text[start:start + 3500])

    @bot.message_handler(content_types=["text"])
    def handle(message):
        user = message.from_user
        if not user or user.id not in config.allowed_users or message.chat.type != "private":
            return
        text = message.text or ""
        owner = owner_for(message)
        try:
            if not text.strip():
                send(message.chat.id, "Отправьте непустое текстовое сообщение.")
            elif text.split()[0] in {"/start", "/help"}:
                send(message.chat.id, HELP)
            elif text == "/status":
                send(message.chat.id, f"VPg05: память включена. Модель: {config.model}. Порог: {config.threshold}.")
            elif text == "/memories":
                rows = service.manager.list_memories(owner)
                send(message.chat.id, "\n\n".join(r["metadata"]["user_message"] for r in rows) or "Память пуста.")
            elif text == "/reset":
                send(message.chat.id, "Для удаления вашей памяти VPg05 отправьте /reset confirm.")
            elif text == "/reset confirm":
                service.manager.reset(owner)
                send(message.chat.id, "Ваша память VPg05 очищена.")
            elif text.startswith("/"):
                send(message.chat.id, "Неизвестная команда. /help")
            else:
                result = service.handle(owner, text, user)
                send(message.chat.id, result["answer"])
        except Exception as exc:
            logger.error("message_error=%s", type(exc).__name__)
            send(message.chat.id, "Обработка не завершена. Запись могла успеть сохраниться; проверьте /memories и повторите позже.")

    return bot


def main():
    log_dir = ROOT / "evidence"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "bot.log", encoding="utf-8")])
    for name in ("httpx", "httpcore", "urllib3", "telebot"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    c = load_config(require_bot=True)
    # Не допускаем двух экземпляров данного проекта с одним токеном.
    lock_dir = ROOT / "data"
    lock_dir.mkdir(exist_ok=True)
    lock_file = (lock_dir / (hashlib.sha256(c.bot_token.encode()).hexdigest()[:16] + ".lock")).open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Этот бот VPg05 уже запущен.")
    manager = PineconeManager(c)
    bot = build_bot(c, MemoryService(manager))
    try:
        me = bot.get_me()
        if bot.get_webhook_info().url:
            raise RuntimeError("У токена активен webhook. Запуск остановлен.")
        logger.info("bot_ready username=%s model=%s threshold=%s", me.username, c.model, c.threshold)
        # Старую очередь прежнего учебного бота не обрабатываем как новую память.
        bot.infinity_polling(skip_pending=True, timeout=20, long_polling_timeout=20,
                             allowed_updates=["message"], logger_level=logging.CRITICAL)
    finally:
        bot.stop_polling()
        manager.close()
        lock_file.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("Бот остановлен:", type(exc).__name__)
        raise SystemExit(1)

