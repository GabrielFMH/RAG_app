---
title: PDF RAG with Evaluation (HF Models)
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
python_version: 3.10
---

# PDF RAG System with RAGAS Evaluation

A complete Retrieval-Augmented Generation (RAG) system for PDFs using open-source Hugging Face models with quality evaluation.

## Architecture

| Layer | Components |
|-------|------------|
| **Interface** | Gradio UI · PDF upload · Chat |
| **Orchestration** | LangChain chains · retrievers · prompts |
| **Processing** | PyMuPDF parsing · RecursiveTextSplitter chunking · HF embeddings |
| **Vector Store** | ChromaDB in-memory |
| **Generation** | Mistral-7B-Instruct (HF) |
| **Evaluation** | RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) |

## Local Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Set your Hugging Face token:

```bash
export HF_TOKEN="hf_..."
```

Or configure it in Hugging Face Spaces secrets.

## Usage

### Local

```bash
python app.py
```

### Hugging Face Spaces Deployment

1. Create a new Space at https://huggingface.co/new-space
2. Select **Gradio** as the SDK
3. Push files to the Space:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
cd YOUR_SPACE
cp ../RAG_app/* .
git add .
git commit -m "Initial commit"
git push
```

4. Add `HF_TOKEN` as a **Space Secret** in Settings

## Features

- **PDF Upload**: Parse and chunk PDFs with PyMuPDF
- **Semantic Search**: ChromaDB vector store with sentence-transformers embeddings
- **Q&A Chat**: Ask questions about your documents using Mistral-7B
- **RAGAS Evaluation**: Measure response quality with:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall

## File Structure

```
├── app.py              # Gradio interface
├── pdf_processor.py    # PDF parsing and chunking
├── rag_pipeline.py     # RAG orchestration
├── evaluation.py       # RAGAS evaluation
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Environment Variables

Set your Hugging Face token:

```bash
export HF_TOKEN="hf_..."
```

Or configure it in Hugging Face Spaces secrets.

## Usage

### Local

```bash
python app.py
```

### Hugging Face Spaces Deployment

1. Create a new Space at https://huggingface.co/new-space
2. Select **Gradio** as the SDK
3. Push files to the Space:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
cd YOUR_SPACE
cp ../RAG_app/* .
git add .
git commit -m "Initial commit"
git push
```

4. Add `HF_TOKEN` as a **Space Secret** in Settings

## Features

- **PDF Upload**: Parse and chunk PDFs with PyMuPDF
- **Semantic Search**: ChromaDB vector store with sentence-transformers embeddings
- **Q&A Chat**: Ask questions about your documents using Mistral-7B
- **RAGAS Evaluation**: Measure response quality with:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall

## File Structure

```
├── app.py              # Gradio interface
├── pdf_processor.py    # PDF parsing and chunking
├── rag_pipeline.py     # RAG orchestration
├── evaluation.py       # RAGAS evaluation
├── requirements.txt    # Dependencies
└── README.md           # This file
```
