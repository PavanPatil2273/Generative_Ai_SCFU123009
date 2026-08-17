from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
if not api_key:
    raise ValueError("Set GROQ_API_KEY in your .env file or environment variables.")

client = Groq(api_key=api_key)

Club ="""
You are a Club Recommendation Assistant.

Available clubs are:

1. Robotics Club
2. Coding Club
3. Debate Club
4. Music Club
5. Photography Club
6. Environment Club

Your job is to understand the user's interest and recommend the most suitable club.

Give a simple answer in this format:

Recommended Club: <club name>
Reason: <short reason>

If the interest does not match any club, say:
No specific club match found.

Keep the answer short and simple.
"""
history = [{"role": "system","content" : "Club "}]



while(True):
    prompt = input("Enter Prompt...\n")
    message ={
        "role":"user",
        "content":Club
    }
    model = "openai/gpt-oss-20b"
    history.append(message);

    if prompt.lower() == "exit":
        print("Exiting chat loop.")
        break

    if prompt.lower() == "show":
        print("History")
        break

    chat_completion = client.chat.completions.create(messages=history, model=model)
    response = chat_completion.choices[0].message.content
    history.append({"role": "assistant", "content": response})
    print(response)