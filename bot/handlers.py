"""
Handlers for the questionnaire bot (aiogram version).
"""

import os
import tempfile
from aiogram import Router, F
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command

from .config import TOTAL_QUESTIONS
from .questions import QUESTIONS
from .states import Questionnaire
from .transcriber import transcribe_audio
from .utils import load_progress, save_progress, append_dialogue

router = Router()

@router.message(Command("start"))
async def start_command(message: Message, state: FSMContext):
    """Greet user and send the first (or next) question."""
    user_id = str(message.from_user.id)
    progress = load_progress()
    
    # Determine starting question index
    if user_id in progress:
        index = progress[user_id] + 1
        if index >= TOTAL_QUESTIONS:
            await message.answer("Вы уже ответили на все вопросы. Спасибо!")
            return
    else:
        index = 0

    # Store current index in FSM data
    await state.update_data(question_index=index)
    # Set state to waiting for voice answer
    await state.set_state(Questionnaire.waiting_for_answer)

    question = QUESTIONS[index]
    await message.answer(
        f"Привет! Я задам тебе 250 вопросов. Отвечай голосовым сообщением.\n\n"
        f"Вопрос {index+1}/{TOTAL_QUESTIONS}: {question}\n\n"
        f"Запиши голосовое сообщение и отправь мне."
    )

@router.message(Questionnaire.waiting_for_answer, F.voice)
async def voice_handler(message: Message, state: FSMContext):
    """Handle voice message: transcribe, save, move to next question."""
    user_id = str(message.from_user.id)
    data = await state.get_data()
    current_index = data.get("question_index", 0)

    # Confirm processing
    await message.answer_chat_action("typing")

    # Download voice file
    voice_file = await message.voice.get_file()
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
        await voice_file.download_to_file(tmp_path)

    try:
        transcribed_text = transcribe_audio(tmp_path)
    except Exception as e:
        await message.answer(f"Ошибка при распознавании голоса: {e}")
        os.unlink(tmp_path)
        return
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not transcribed_text:
        await message.answer("Не удалось распознать текст. Пожалуйста, повторите запись.")
        return

    # Save dialogue
    question = QUESTIONS[current_index]
    append_dialogue(transcribed_text, question)

    # Update progress in file and FSM
    progress = load_progress()
    progress[user_id] = current_index
    save_progress(progress)

    next_index = current_index + 1
    if next_index >= TOTAL_QUESTIONS:
        await message.answer(
            f"Ваш ответ записан: \"{transcribed_text}\".\n\n"
            f"Поздравляю! Вы ответили на все 250 вопросов. Спасибо!"
        )
        await state.clear()
    else:
        # Set next question and stay in same state
        await state.update_data(question_index=next_index)
        next_question = QUESTIONS[next_index]
        await message.answer(
            f"Ответ принят: \"{transcribed_text}\".\n\n"
            f"Следующий вопрос ({next_index+1}/{TOTAL_QUESTIONS}): {next_question}"
        )

@router.message(Questionnaire.waiting_for_answer, F.text)
async def text_handler(message: Message):
    """Prompt the user to send voice messages, not text."""
    await message.answer("Пожалуйста, отвечайте голосовым сообщением. "
                         "Отправьте аудиозапись с вашим ответом.")

@router.message(Command("progress"))
async def progress_command(message: Message, state: FSMContext):
    """Show current progress."""
    user_id = str(message.from_user.id)
    progress = load_progress()
    answered = progress.get(user_id, -1) + 1
    total = TOTAL_QUESTIONS
    await message.answer(f"Вы ответили на {answered} из {total} вопросов.")