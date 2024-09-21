from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from .models import State
import os


def initialize_chatbot(env_vars):
    os.environ["LANGCHAIN_API_KEY"] = env_vars['langsmith']
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "CourseLanggraph"

    llm = ChatGroq(groq_api_key=env_vars['groq_api_key'], model_name="Gemma2-9b-It")

    graph_builder = StateGraph(State)

    def chatbot(state: State):
        return {"messages": llm.invoke(state.messages)}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    return graph_builder.compile()
