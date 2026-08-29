import os
from groq import Groq
from dotenv import load_dotenv
from prompts import (
    SUMMARY_PROMPT, 
    CONTRIBUTIONS_PROMPT, 
    LIMITATIONS_PROMPT, 
    FUTURE_WORK_PROMPT,
    QA_PROMPT,
    COMPARISON_PROMPT,
    COMPARISON_SUMMARY_PROMPT
)

class LLMManager:
    def __init__(self, model_name="openai/gpt-oss-120b"):
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def ask_groq(self, prompt, temperature=0.2):
        """Helper to send a prompt to Groq API"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content

    def ask_groq_conversational(self, messages, temperature=0.2):
        """Helper to send a conversation history to Groq API"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    def generate_analysis(self, context, analysis_type):
        """
        Generates a specific type of analysis given the context.
        """
        if analysis_type == "summary":
            prompt = SUMMARY_PROMPT.format(context=context)
        elif analysis_type == "contributions":
            prompt = CONTRIBUTIONS_PROMPT.format(context=context)
        elif analysis_type == "limitations":
            prompt = LIMITATIONS_PROMPT.format(context=context)
        elif analysis_type == "future_work":
            prompt = FUTURE_WORK_PROMPT.format(context=context)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
            
        return self.ask_groq(prompt)

    def answer_question(self, context, question):
        """Answers a single question based on context"""
        prompt = QA_PROMPT.format(context=context, question=question)
        return self.ask_groq(prompt)

    def compare_papers(self, paper_chunks_dict, question=None):
        """
        Compares multiple papers given a dict of {paper_name: [chunks]}.
        Formats each paper's context with clear labels before sending to the LLM.
        If no question is provided, generates a comprehensive side-by-side summary.
        """
        combined_context = ""
        for paper_name, chunks in paper_chunks_dict.items():
            paper_context = "\n".join(c["text"] for c in chunks)
            combined_context += f"\n\n### {paper_name}\n{paper_context}"

        if question and question.strip():
            prompt = COMPARISON_PROMPT.format(
                context=combined_context.strip(),
                question=question
            )
        else:
            prompt = COMPARISON_SUMMARY_PROMPT.format(
                context=combined_context.strip()
            )
        return self.ask_groq(prompt)
