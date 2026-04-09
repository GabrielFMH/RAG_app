import os
from typing import List, Optional, Any

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import Generation
from huggingface_hub import InferenceClient
from pydantic import Field


class HuggingFaceChatLLM(LLM):
    """Custom LLM wrapper for Hugging Face chat completions."""
    
    client: Any = Field(default=None, exclude=True)
    model_name: str = Field(default="meta-llama/Llama-3.1-8B-Instruct")
    
    def __init__(self, hf_token: str, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", **kwargs):
        super().__init__(**kwargs)
        self.client = InferenceClient(
            provider="novita",
            api_key=hf_token,
        )
        self.model_name = model_name
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question concisely."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=500,
            temperature=0,
        )
        
        return completion.choices[0].message.content


class RAGEvaluator:
    """Evaluate RAG pipeline using RAGAS metrics with Hugging Face models."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found. Set it as env var or pass it to the constructor.")

        self.llm = HuggingFaceChatLLM(hf_token=token, model_name=model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
        )

        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def evaluate_rag(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> dict:
        """Evaluate RAG responses using RAGAS metrics."""
        
        if len(questions) != len(answers) or len(questions) != len(contexts):
            raise ValueError("questions, answers, and contexts must have the same length")

        if ground_truths is None:
            ground_truths = [""] * len(questions)

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        df = result.to_pandas()
        
        # Extract scores from the dataframe (we know this works based on user output)
        scores_dict = {}
        for metric in self.metrics:
            if metric.name in df.columns:
                # Take the mean score across all examples
                scores_dict[metric.name] = float(df[metric.name].mean())
        
        return {
            "scores": scores_dict,
            "df": df,
            "summary": result,
        }

    def format_results(self, eval_result: dict) -> str:
        """Format evaluation results for display."""
        scores = eval_result["scores"]
        df = eval_result["df"]
        
        output = "## RAGAS Evaluation Results\n\n"
        output += "| Metric | Score |\n"
        output += "|--------|-------|\n"
        for metric_name, score in scores.items():
            # Handle different possible score formats from RAGAS
            try:
                # If score is already a number
                if isinstance(score, (int, float)):
                    score_value = float(score)
                # If score is a dict with a 'score' key
                elif isinstance(score, dict) and 'score' in score:
                    score_value = float(score['score'])
                # If score is a dict with a 'value' key
                elif isinstance(score, dict) and 'value' in score:
                    score_value = float(score['value'])
                # If score is a list/tuple, take first element
                elif isinstance(score, (list, tuple)) and len(score) > 0:
                    score_value = float(score[0])
                # If score is a string that can be converted to float
                elif isinstance(score, str):
                    score_value = float(score)
                # Default fallback
                else:
                    score_value = 0.0
                output += f"| {metric_name} | {score_value:.4f} |\n"
            except (ValueError, TypeError):
                # If conversion fails, show 0.0000
                output += f"| {metric_name} | 0.0000 |\n"

        output += "\n### Detailed Results (per question)\n\n"
        # Format the dataframe nicely for display with proper column handling
        if df is not None and not df.empty:
            # Select only the most relevant columns for display to avoid overflow
            display_cols = ['question', 'answer', 'faithfulness', 'answer_relevancy', 
                           'context_precision', 'context_recall']
            # Filter to only columns that exist in the dataframe
            available_cols = [col for col in display_cols if col in df.columns]
            if available_cols:
                display_df = df[available_cols].copy()
                # Truncate long text fields for better readability
                for col in ['question', 'answer']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].astype(str).str[:100] + '...'
                output += display_df.to_string(index=False, max_colwidth=20)
            else:
                # Fallback to showing all columns with truncation
                display_df = df.copy()
                for col in display_df.columns:
                    if display_df[col].dtype == 'object':  # String columns
                        display_df[col] = display_df[col].astype(str).str[:50] + '...'
                output += display_df.to_string(index=False, max_colwidth=15)
        else:
            output += "*No detailed results available*"

        return output

    def get_default_questions(self) -> List[str]:
        """Return default evaluation questions."""
        return [
            "What is the main topic of this document?",
            "What are the key points discussed?",
            "What conclusions are drawn?",
        ]
