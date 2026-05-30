import asyncio
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.session.aiohttp import AiohttpSession
from .config import BOT_TOKEN
from .handlers import router

async def main():
    session = AiohttpSession(
        proxy="socks5://217.12.209.4:1080"  # replace with working proxy
    )
    bot = Bot(token=BOT_TOKEN, session=session)

    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(router)

    print("Bot started...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())