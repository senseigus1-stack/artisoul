# main.py
# Entry point: starts the interactive chat loop.

from config import Config
from bot import EndocrineConversationalBot

def main():
    print("=" * 60)
    print(" Endocrine-modulated conversational AI (the 'bitch' version) ")
    print(" Type 'exit' to quit.")
    print("=" * 60)

    config = Config()
    bot = EndocrineConversationalBot(config)

    print("\n[Bot's initial state]")
    print(bot.endocrine.state_description())

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'выход']:
            break
        reply, state = bot.generate_reply(user_input)
        print(f"\n[{state}]")
        print(f"Bot: {reply}")

if __name__ == "__main__":
    main()