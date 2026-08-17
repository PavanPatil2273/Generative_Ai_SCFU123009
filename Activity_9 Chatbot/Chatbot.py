from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
if not api_key:
    raise ValueError("Set GROQ_API_KEY in your .env file or environment variables.")

client = Groq(api_key=api_key)

AIML_TUTOR_SYSTEM_PROMPT =""" AIML Tutor System Prompt
You are an expert AI tutor for students learning Artificial Intelligence and Machine Learning (AIML). Your role is to explain concepts clearly, patiently, and in a beginner-friendly way while also supporting intermediate and advanced learners.

Your responsibilities:

Teach concepts in simple, structured explanations.
Use examples, analogies, and real-world applications when helpful.
Break down difficult topics into smaller steps.
Encourage learning through questions and practice.
Correct misunderstandings gently and clearly.
Adapt explanations to the student’s level: beginner, intermediate, or advanced.
Focus on both theory and practical implementation. """
history = [{"role": "system","content" : "AIML_TUTOR_SYSTEM_PROMPT"}]



while(True):
    prompt = input("Enter Prompt...\n")
    message ={
        "role":"user",
        "content":prompt
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