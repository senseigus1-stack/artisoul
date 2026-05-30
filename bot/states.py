"""
FSM states for the questionnaire bot.
"""

from aiogram.fsm.state import State, StatesGroup

class Questionnaire(StatesGroup):
    waiting_for_answer = State()