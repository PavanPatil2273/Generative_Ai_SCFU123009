from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3
)


extract_answer_prompt = PromptTemplate.from_template("""
You are an interview transcript analyst.

Read the transcript below and extract the candidate's answer to each interview
question. Preserve the candidate's meaning and use concise paraphrases. Include
the question, the answer, and a short evidence quote when the transcript makes
one available.

Rules:
- Extract only information stated by the candidate in the transcript.
- Do not infer skills, intent, seniority, or results that are not supported.
- If a question was not answered, write "Not answered".
- Keep answers separate; do not combine responses to different questions.

Return a numbered list with this format:
1. Question: <question>
    Answer: <answer or Not answered>
    Evidence: <short quote or Not available>

Transcript:
{transcript_text}
""")

evaluate_answers_prompt = PromptTemplate.from_template("""
You are a structured interview evaluator.

Assess the extracted answers against the requested skills. For every skill,
assign exactly one rating: Strong, Adequate, Weak, or No evidence. Give a brief
reason based only on the extracted answers and cite the relevant question when
possible.

Rules:
- Evaluate demonstrated evidence, not assumptions or writing quality.
- Do not reward a skill unless the answers clearly support it.
- Distinguish "Weak" evidence from "No evidence".
- Flag missing, vague, or contradictory evidence as a concern.
- Do not introduce skills that are not in the requested list.

Return one section per skill:
Skill: <skill>
Rating: <Strong | Adequate | Weak | No evidence>
Evidence: <brief evidence or No evidence>
Concern: <missing or unclear detail, or None>

Extracted answers:
{answers}

Skills to assess:
{skills_to_assess}
""")

format_output_prompt = PromptTemplate.from_template("""
You are a hiring scorecard editor.

Convert the evaluation below into a concise, neutral hiring scorecard. Preserve
every skill rating and the evidence behind it. Do not add facts, scores, or
qualifications that are absent from the evaluation.

Use exactly this format:

HIRING SCORECARD
Overall recommendation: <Strong hire | Hire | Mixed evidence | Do not recommend>
Confidence: <High | Medium | Low>

Skill assessment:
- <skill>: <rating> — <one-sentence evidence summary>

Key strengths:
- <strength or None identified>

Key concerns:
- <concern or None identified>

Follow-up questions:
- <question needed to resolve an evidence gap or None>

Recommendation rules:
- Use Strong hire only when the evidence is consistently strong.
- Use Hire when the evidence is mostly adequate with no critical gaps.
- Use Mixed evidence when important skills are weak, unclear, or incomplete.
- Use Do not recommend only when the evaluation contains clear critical weaknesses.
- Set confidence lower when the transcript contains unanswered or vague questions.

Evaluation:
{evaluation}
""")

transcript_text = input("Enter the transcript text: ").strip()
skills_to_assess = input("Enter skills to assess (comma-separated): ").strip()


extract_chain = extract_answer_prompt | model | StrOutputParser()
evaluate_chain = evaluate_answers_prompt | model | StrOutputParser()
format_chain = format_output_prompt | model | StrOutputParser()

extracted_answers = extract_chain.invoke({
    "transcript_text": transcript_text
})

evaluation = evaluate_chain.invoke({
    "answers": extracted_answers,
    "skills_to_assess": skills_to_assess
})

scorecard = format_chain.invoke({
    "evaluation": evaluation
})

print("\nStep 1 - Extracted answers:")
print(extracted_answers)
print("\nStep 2 - Skills evaluation:")
print(evaluation)
print("\nStep 3 - Hiring scorecard:")
print(scorecard)

