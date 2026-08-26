import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
if not api_key:
    raise ValueError("Set GROQ_API_KEY in your .env file or environment variables.")

client = Groq(api_key=api_key)

patient_name = input("Enter Patient Name: ")
patient_notes = input("Enter Patient Notes: ")

prompt = """
You are a medical documentation assistant. Convert the patient's raw consultation
notes into a concise, professional doctor's summary.

Return the result in exactly this format and order, with no extra sections:

Symptoms: <summary of reported symptoms>
Diagnosis: <diagnosis, or 'Not provided' if it is not stated in the notes>
Recommendation: <recommendations, or 'Not provided' if they are not stated in the notes>

Use only information from the consultation notes. Do not invent symptoms,
diagnoses, treatments, or recommendations. Keep the three required sections
present even when the notes do not contain information for one of them.
"""

user_message = f"Patient name: {patient_name}\nConsultation notes: {patient_notes}"

chat_completion = client.chat.completions.create(
    temperature=1.18,
    max_completion_tokens=1,
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message},
    ],
)

print(chat_completion.choices[0].message.content)