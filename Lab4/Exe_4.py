
"""
Exercise 4: Product Idea to Pitch Pipeline
Build a 2-step pipeline that:
1. Expands a one-line product idea into a structured pitch (problem, solution, target user)
2. Generates an investor-style pitch paragraph using ONLY the structured version from Step 1
"""

from groq import Groq
from dotenv import load_dotenv
import json
import os

load_dotenv()

# Define prompts for each step
STEP_1_PROMPT = """You are a product strategist. Given this one-line product idea, 
expand it into a structured pitch with the following JSON format (and ONLY this format):
{{
    "problem": "describe the core problem this product solves",
    "solution": "describe how the product solves this problem",
    "target_user": "describe the primary target user/customer"
}}

Product idea: {product_idea}

Respond with ONLY valid JSON, no additional text."""

STEP_2_PROMPT = """You are an expert pitch coach preparing an executive summary for investors.
Based on ONLY the following structured pitch information, write a compelling investor-style pitch paragraph (2-3 sentences):

Problem: {problem}
Solution: {solution}
Target User: {target_user}

Write a persuasive pitch for venture capitalists. Be specific and compelling."""

MODEL = "llama-3.3-70b-versatile"
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Set GROQ_API_KEY in your .env file or environment variables.")

# Initialize Groq client
client = Groq(api_key=api_key)

def step_1_structured_pitch(product_idea: str) -> dict:
    """Step 1: Expand product idea into structured pitch"""
    message = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": STEP_1_PROMPT.format(product_idea=product_idea)}
        ]
    )
    structured_pitch = json.loads(message.choices[0].message.content)
    return structured_pitch

def step_2_investor_pitch(structured_pitch: dict) -> str:
    """Step 2: Generate investor pitch from structured pitch ONLY"""
    message = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "user", "content": STEP_2_PROMPT.format(
                problem=structured_pitch['problem'],
                solution=structured_pitch['solution'],
                target_user=structured_pitch['target_user']
            )}
        ]
    )
    return message.choices[0].message.content

def product_idea_to_pitch_pipeline(product_idea: str) -> dict:
    """Complete 2-step pipeline"""
    print(f"\nProduct Idea: {product_idea}\n")
    
    # Step 1: Structured Pitch
    print("STEP 1: Structured Pitch")
    structured_pitch = step_1_structured_pitch(product_idea)
    print(json.dumps(structured_pitch, indent=2))
    
    # Step 2: Investor Pitch
    print("\nSTEP 2: Investor Pitch")
    investor_pitch = step_2_investor_pitch(structured_pitch)
    print(investor_pitch)
    
    return {
        "structured_pitch": structured_pitch,
        "investor_pitch": investor_pitch
    }

if __name__ == "__main__":
    test_idea = "An AI-powered app that helps busy professionals meal plan"
    output = product_idea_to_pitch_pipeline(test_idea)