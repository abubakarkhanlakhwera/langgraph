# backend.py
# Backend logic for chatbot
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from dotenv import load_dotenv

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Global LLM instance that will be configured per request
llm = ChatOpenAI()
graph = StateGraph(ChatState)


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


def create_chatbot_with_config(
    model='gpt-3.5-turbo',
    temperature=0.7,
    system_prompt='',
):
    """Create a configured chatbot instance"""
    global llm
    llm = ChatOpenAI(model=model, temperature=temperature)
    
    def configured_chat_node(state: ChatState):
        messages = state['messages']
        
        # Add system message if provided
        if system_prompt and system_prompt.strip():
            # Check if first message is already a system message
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_prompt)] + messages
            else:
                # Replace existing system message
                messages[0] = SystemMessage(content=system_prompt)
        
        response = llm.invoke(messages)
        return {'messages': [response]}
    
    # Create new graph with configured node
    local_graph = StateGraph(ChatState)
    local_checkpointer = SqliteSaver(conn=conn)
    local_graph.add_node('chat_node', configured_chat_node)
    local_graph.add_edge(START, 'chat_node')
    local_graph.add_edge('chat_node', END)
    return local_graph.compile(checkpointer=local_checkpointer)


# Default chatbot instance
checkpointer = SqliteSaver(conn=conn)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)
chatbot = graph.compile(checkpointer=checkpointer)

 
def retrieve_all_threads():
    """Return a list of all thread_ids stored in the checkpointer.

    Using a list ensures the frontend can safely use list operations like
    .append() and slicing for display (e.g., reversing order).
    """
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    # Return a stable list (sorted by string representation for consistency)
    return sorted(list(all_threads), key=lambda x: str(x))
