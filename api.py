# import os   ## api_key=os.environ.get("OPENAI_API_KEY")
from openai import OpenAI

def get_chatgpt_response(question):
    # load and set our key
    client = OpenAI(api_key=open("key.txt", "r").read().strip("\n"))

    # Use the text from the voice to chat data as input to the ChatGPT 3.5 API
    chatgpt_response = client.chat.completions.create(
        messages=[{"role": "user", "content": question}],
        model="gpt-3.5-turbo",
    )
    return chatgpt_response
