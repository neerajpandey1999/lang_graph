import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('groq_api_key')
langsmith = os.getenv('langsmith')

# print(groq_api_key, langsmith)

import os
os.environ["LANGCHAIN_API_KEY"] = langsmith
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="CourseLanggraph"

from langchain_groq import ChatGroq

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages


class State(TypedDict):
  messages:Annotated[list,add_messages]

graph_builder=StateGraph(State)

def chatbot(state:State):
  return {"messages":llm.invoke(state['messages'])}

graph_builder.add_node("chatbot",chatbot)

graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

graph=graph_builder.compile()

# import os
#
# try:
#   image_path = 'graph_image.png'
#   image_data = graph.get_graph().draw_mermaid_png()
#
#   if image_data:
#     with open(image_path, 'wb') as f:
#       f.write(image_data)
#     print(f"Image saved to {os.path.abspath(image_path)}")
#     os.system(f'open {image_path}')
#   else:
#     print("No image data returned from the graph.")
# except Exception as e:
#   print("An error occurred:", e)

while True:
  user_input=input("User: ")
  if user_input.lower() in ["quit","q"]:
    print("Good Bye")
    break

  for event in graph.stream({'messages':("user",user_input)}):
    # print(event.values())  # Dictionary values
    for value in event.values():
      # print(value['messages'])
      print("Assistant:",value["messages"].content)