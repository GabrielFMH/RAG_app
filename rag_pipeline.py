import os
from typing import List, Optional

from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGPipeline:
    """RAG pipeline with LangChain, ChromaDB, and Hugging Face models."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k_retriever: int = 4,
    ):
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found. Set it as env var or pass it to the constructor.")

        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
        )
        self.llm_client = InferenceClient(
            provider="novita",
            api_key=token,
        )
        self.vectorstore = None
        self.retriever = None
        self.k_retriever = k_retriever
        self.chunks = []
        self.loaded = False

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_documents(self, texts: List[str]):
        """Load text documents into the vector store."""
        self.chunks = self.text_splitter.split_text("\n\n".join(texts))
        documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(self.chunks)]

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k_retriever},
        )
        self.loaded = True

    def query(self, question: str) -> dict:
        """Query the RAG pipeline."""
        if not self.loaded:
            raise ValueError("No documents loaded. Call load_documents() first.")

        docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            }
        ]

        completion = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=500,
            temperature=0,
        )

        answer = completion.choices[0].message.content

        return {
            "answer": answer,
            "source_documents": docs,
        }

    def get_relevant_chunks(self, question: str) -> List[str]:
        """Get relevant chunks for a question without generating an answer."""
        if not self.retriever:
            raise ValueError("No documents loaded. Call load_documents() first.")

        docs = self.retriever.invoke(question)
        return [doc.page_content for doc in docs]

    def get_collection_stats(self) -> dict:
        """Get statistics about the loaded collection."""
        if not self.vectorstore:
            return {"documents": 0}

        return {
            "documents": len(self.chunks),
            "k_retriever": self.k_retriever,
        }
