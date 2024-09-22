import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
from pymongo import MongoClient
from typing_extensions import TypedDict

load_dotenv()

groq_api_key = os.getenv('groq_api_key')
langsmith = os.getenv('langsmith')
os.environ["LANGCHAIN_API_KEY"] = langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CourseLanggraph"

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

tools = [wiki_tool]

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
llm_with_tools = llm.bind_tools(tools=tools)

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()

# from IPython.display import Image, display
# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

client = MongoClient('mongodb://root:example@localhost:27017/')
db = client.chatbot_db
collection = db.queries

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "q"]:
        print("Good Bye")
        break

    human_response = None
    ai_response = None
    tool_response = None
    tool_messages = []
    for event in graph.stream({'messages': ("user", user_input)}, stream_mode="values"):
        message = event["messages"][-1]
        if isinstance(message, AIMessage):
            ai_response = message.content
        if isinstance(message, ToolMessage):
            tool_response = message.content
        if isinstance(message, HumanMessage):
            human_response = message.content

    print("Human Response:", human_response)
    print("AI Response:", ai_response)
    print("Tool Response:", tool_response)

    collection.insert_one({
        "humanresponse": human_response,
        "AIResponse": ai_response,
        "toolresponse": tool_response
    })
