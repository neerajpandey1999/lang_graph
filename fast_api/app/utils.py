import os
from dotenv import load_dotenv

def load_env_variables():
    load_dotenv()
    return {
        "groq_api_key": os.getenv('groq_api_key'),
        "langsmith": os.getenv('langsmith')
    }
