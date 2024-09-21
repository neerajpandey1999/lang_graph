from fastapi import FastAPI, HTTPException
from .models import UserInput
from .logic import initialize_chatbot
from .utils import load_env_variables

env_vars = load_env_variables()
graph = initialize_chatbot(env_vars)

app = FastAPI()

conversation_history = []

@app.post("/chat/")
async def chat(input_data: UserInput):
    global conversation_history
    state = {'messages': conversation_history + [{"role": "user", "content": input_data.user_input}]}
    try:
        response = []
        for event in graph.stream(state):
            for value in event.values():
                assistant_response = value["messages"].content
                response.append({"content": assistant_response})
                conversation_history = state['messages'] + [{"role": "assistant", "content": assistant_response}]
        return {"responses": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
