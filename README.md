# RAG-based-QA-system

## Description
RAG-based Q&A system built with LangChain, Hugging Face, and Chroma. It retrieves relevant document chunks using embeddings and generates accurate answers with an LLM.

## Steps
1. Load documents (currently .txt, extendable to other doc types)
2. Split documents into chunks for efficient retrieval
3. Create embeddings and store in Vector Database (Chroma)
5. User asks a Question (query with natural language)
6. Retrieve top-k relevant Chunks (Currently, 3)
7. Generate final answer using Hugging Face LLM
8. Results include answer and source chunks

## Project Structure
```
├── rag.py/            # Main pipeline
├── sample.txt/         # Sample knowledge base text
├── requirements.txt   # Dependencies
└── README.md          # Project overview
```

## Setup
1. 1. Clone the Repository
```bash
git clone https://github.com/meenu1204/RAG-based-QA-system
cd RAG-based-QA-system
```
2. Create virtual environment
   ```bash
python -m venv rag_env
source rag_env/bin/activate
```
3. Install dependencies
```bash
 pip install -r requirements.txt
 ```
4. Run main script
```bash
 python rag.py
 ```
5. Sample Output
![Screenshot of RAG QA result](rag_output)
