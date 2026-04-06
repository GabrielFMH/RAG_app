import os
from typing import List, Optional

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document


class RAGPipeline:
    """RAG pipeline with LangChain, ChromaDB, and Hugging Face models."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k_retriever: int = 4,
    ):
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found. Set it as env var or pass it to the constructor.")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
        )
        self.llm = HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=0,
            huggingfacehub_api_token=token,
        )
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.k_retriever = k_retriever
        self.chunks = []

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Use three sentences maximum and keep the answer concise.\n\n"
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
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

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True,
        )

    def query(self, question: str) -> dict:
        """Query the RAG pipeline."""
        if not self.qa_chain:
            raise ValueError("No documents loaded. Call load_documents() first.")

        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result.get("source_documents", []),
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
