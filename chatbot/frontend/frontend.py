# frontend.py
# Frontend for chatbot (CLI)
from backend.backend import chatbot
from langchain_core.messages import HumanMessage

def run_chat():
    thread_id = '1'
    while True:
        user_message = input('Type here: ')
        print("User:", user_message)
        if user_message.strip().lower() in ['exit', 'quit', 'bye']:
            break
        config = {'configurable': {'thread_id': thread_id}}
        response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)
        print("AI:", response['messages'][-1].content)

if __name__ == "__main__":
    run_chat()
