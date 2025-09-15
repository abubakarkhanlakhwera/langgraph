# backend.py
# Backend logic for chatbot
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

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


def create_chatbot_with_config(model='gpt-3.5-turbo', temperature=0.7, system_prompt=''):
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
    checkpointer = MemorySaver()
    local_graph.add_node('chat_node', configured_chat_node)
    local_graph.add_edge(START, 'chat_node')
    local_graph.add_edge('chat_node', END)
    return local_graph.compile(checkpointer=checkpointer)


# Default chatbot instance
checkpointer = MemorySaver()
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)
chatbot = graph.compile(checkpointer=checkpointer)
