import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")


if not NVIDIA_KEY:
    raise ValueError("NVIDIA_API_KEY is missing from your .env file.")

client = OpenAI(
    api_key=NVIDIA_KEY,
    base_url="https://integrate.api.nvidia.com/v1"
)

MODEL = "meta/llama-3.1-8b-instruct"

# Get user inputs
job_posting = input("Enter the job posting: ").strip()
candidate_profile = input("Enter the candidate profile: ").strip()

# Validation: Check if inputs are empty
if not job_posting or not candidate_profile:
    raise ValueError("Both the job posting and candidate profile are required.")

print("\nProcessing... please wait.")

try:
    req_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Extract key job requirements. Return ONLY valid JSON with fields: job_title, company, location, required_skills, preferred_skills, experience, education, responsibilities."
            },
            {"role": "user", "content": job_posting}
        ],
        response_format={"type": "json_object"}
    )
    
    # FIXED: Added [0] to correctly access the first choice element
    requirements = json.loads(req_response.choices[0].message.content)

    # Step 2: Create the Outreach Message
    outreach_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Write a short, friendly, personalized candidate outreach message. Include a subject, greeting, relevant role matches, and a call to action."
            },
            {
                "role": "user",
                "content": f"Job requirements:\n{json.dumps(requirements, indent=2)}\n\nCandidate profile:\n{candidate_profile}"
            }
        ]
    )

    # Print the final results
    print("\n=== Structured Requirements ===")
    print(json.dumps(requirements, indent=2))

    print("\n=== Personalized Outreach Message ===")
    print(outreach_response.choices[0].message.content)

except Exception as e:
    print(f"\nNVIDIA API request failed: {e}")
    print("Check that NVIDIA_API_KEY is a valid NVIDIA API key and that the endpoint is reachable.")
