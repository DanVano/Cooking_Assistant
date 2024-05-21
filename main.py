import whisper
import openai
import time

from api import get_chatgpt_response




def main():
    # Record the start time
    start_time = time.time()

    #Important
    voice_to_chat()
    chatgpt_call()

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")


def voice_to_chat():
    print('Voice to Chat function')
    # Load whisper - the voice to text import
    model = whisper.load_model("base")
    
    # Save the voice to chat data into a variable
    result = model.transcribe("audio.mp3", fp16=False)
    print(result["text"])
    return

# test question for chatgpt_call()
def get_question():
    # Prompt the user for a question
    question = input("Please enter your question: ")
    return question

def chatgpt_call():
    print('chat gpt call')
    # Get the question from the user
    question = get_question()

    # Get the response from the API call
    chatgpt_response = get_chatgpt_response(question)

    # Access the content of the response
    content = chatgpt_response.choices[0].message.content
    print(content)


if __name__ == "__main__":
    main()
