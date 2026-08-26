import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_KEY:
    raise ValueError("NVIDIA_API_KEY is missing from your .env file.")

# Initialize the OpenAI client pointed to NVIDIA
client = OpenAI(
    api_key=NVIDIA_KEY,
    base_url="https://nvidia.com"
)

MODEL = "meta/llama-3.1-8b-instruct"

# Get the news article input
article = input("Enter the news article text: ").strip()

if not article:
    raise ValueError("The news article text cannot be empty.")

print("\nProcessing... please wait.")

try:
    # STEP 1: Extract core claims as a strict JSON list
    step1_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the core factual claims from the news article. "
                    "Return ONLY a valid JSON object containing a single key 'claims' "
                    "linked to a list of strings. Do not include markdown blocks or conversational text."
                )
            },
            {"role": "user", "content": article}
        ],
        response_format={"type": "json_object"}
    )
    
    # Parse claims safely into a Python dictionary
    extracted_data = json.loads(step1_response.choices[0].message.content)
    claims_list = extracted_data.get("claims", [])

    # STEP 2: Generate a Fact Card using ONLY the extracted claims
    step2_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict fact-checking assistant. Build a clear Fact Card using ONLY the claims provided. "
                    "Do not bring in outside knowledge or extrapolate. "
                    "Format your output exactly with these headers:\n"
                    "HEADLINE:\n"
                    "BULLET POINTS:\n"
                    "- [Point 1]\n"
                    "- [Point 2]\n"
                    "- [Point 3]\n"
                    "SOURCE CONFIDENCE NOTE:"
                )
            },
            {
                "role": "user",
                "content": f"Extracted Claims to use:\n{json.dumps(claims_list, indent=2)}"
            }
        ]
    )

    # Print the pipeline results
    print("\n=== [STEP 1] Extracted Core Claims ===")
    print(json.dumps(claims_list, indent=2))

    print("\n=== [STEP 2] Generated Fact Card ===")
    print(step2_response.choices[0].message.content.strip())

except Exception as e:
    print(f"\nPipeline failed: {e}")
