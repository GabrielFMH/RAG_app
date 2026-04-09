import os
import tempfile
import gradio as gr
from dotenv import load_dotenv

from pdf_processor import PDFProcessor
from rag_pipeline import RAGPipeline
from evaluation import RAGEvaluator

load_dotenv()


class RAGApp:
    """Gradio interface for PDF RAG with evaluation."""

    def __init__(self):
        load_dotenv()
        
        self.hf_token = os.environ.get("HF_TOKEN", "")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in environment. Please set it in .env file.")
        
        self.processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
        self.rag = RAGPipeline(
            hf_token=self.hf_token,
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            k_retriever=4,
        )
        self.evaluator = RAGEvaluator(hf_token=self.hf_token)
        
        self.full_text = ""
        self.chunks = []
        self.pdf_loaded = False

    def upload_pdf(self, pdf_file):
        """Process uploaded PDF file."""
        if pdf_file is None:
            raise gr.Error("Please upload a PDF file first.")

        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()

        self.full_text, self.chunks = self.processor.process_pdf_bytes(pdf_bytes)
        self.rag.load_documents([self.full_text])

        stats = self.rag.get_collection_stats()
        metadata = self.processor.get_pdf_metadata(pdf_file)

        info = f"PDF loaded successfully.\n"
        info += f"Pages: {metadata['page_count']}\n"
        info += f"Chunks created: {stats['documents']}\n"
        info += f"Chunk size: 1000, Overlap: 200"

        return info

    def ask_question(self, question: str, history):
        """Answer a question using the RAG pipeline."""
        if not question.strip():
            return history + [{"role": "user", "content": question}, {"role": "assistant", "content": "Please enter a question."}], ""

        if self.rag is None or not self.chunks:
            return history + [{"role": "user", "content": question}, {"role": "assistant", "content": "Please upload and process a PDF first."}], ""

        result = self.rag.query(question)
        answer = result["answer"]
        sources = result["source_documents"]

        source_info = ""
        if sources:
            source_info = "\n\n**Sources:**\n"
            for i, doc in enumerate(sources, 1):
                source_info += f"\n{i}. {doc.page_content[:200]}..."

        answer_with_sources = answer + source_info
        return history + [{"role": "user", "content": question}, {"role": "assistant", "content": answer_with_sources}], ""

    def run_evaluation(self, questions_text: str):
        """Run RAGAS evaluation."""
        if self.rag is None or not self.chunks:
            raise gr.Error("Please upload and process a PDF first.")

        questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
        if not questions:
            raise gr.Error("Please provide at least one question.")

        answers = []
        contexts = []

        for q in questions:
            result = self.rag.query(q)
            answers.append(result["answer"])
            contexts.append([doc.page_content for doc in result["source_documents"]])

        eval_result = self.evaluator.evaluate_rag(
            questions=questions,
            answers=answers,
            contexts=contexts,
        )

        formatted = self.evaluator.format_results(eval_result)
        return formatted


def create_demo():
    """Create the Gradio demo interface."""
    app = RAGApp()

    with gr.Blocks(title="PDF RAG with Evaluation") as demo:
        gr.Markdown("# PDF RAG System with RAGAS Evaluation")
        gr.Markdown("Upload PDFs, ask questions, and evaluate response quality.")

        with gr.Tab("Upload PDF"):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Process PDF")
            upload_output = gr.Textbox(label="PDF Info", interactive=False)

            upload_btn.click(
                fn=app.upload_pdf,
                inputs=[pdf_input],
                outputs=[upload_output],
            )

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Conversation")
            question_input = gr.Textbox(label="Your Question", placeholder="Ask about the PDF...")
            clear_btn = gr.Button("Clear Chat")

            question_input.submit(
                fn=app.ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input],
            )

            clear_btn.click(lambda: ([], ""), outputs=[chatbot, question_input])

        with gr.Tab("Evaluation"):
            questions_input = gr.Textbox(
                label="Evaluation Questions (one per line)",
                lines=5,
                placeholder="What is the main topic?\nWhat are the key points?\nWhat conclusions are drawn?",
                value="What is the main topic of this document?\nWhat are the key points discussed?\nWhat conclusions are drawn?",
            )
            eval_btn = gr.Button("Run Evaluation")
            eval_output = gr.Markdown(label="Results")

            eval_btn.click(
                fn=app.run_evaluation,
                inputs=[questions_input],
                outputs=[eval_output],
            )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
