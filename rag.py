# File: rag.py

# import necessary libraries
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline

# Load document
document = TextLoader("sample.txt").load()
print("Step 1: Loaded documents:", len(document))

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(document)
print("Step 2: Split document into", len(docs), "chunks")

# Create embeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Step 3: Loaded embeddings:", embeddings_model)

# Store in Vector DB
vector_store = Chroma.from_documents(docs, embeddings_model)
print("Step 4: Docs and embeddings are stored in", vector_store)

# Building retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("Step 5: Retrieving retriever results:", retriever)

# Loading Hugging Face LLM
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
print("Step 6: Loaded Hugging Face LLM pipeline", llm_pipeline)

# Wrapping LLM pipeline
llm = HuggingFacePipeline(pipeline=llm_pipeline)
print("Step 7: Wrapped LLM pipeline into LangChain HuggingFacePipeline", llm)

# Build retrieval-QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
print("Step 8: Connect retriever + LLM", qa)

# User question
query = "What is RAG?"
print("Step 9: User Question:", query)

# Retrieve relevant chunks
relevant_docs = retriever.invoke(query)
print("Step 10: Retrieved relevant documents:", len(relevant_docs))
for doc in relevant_docs:
    print(doc.page_content[:200])

# Generate answer
result =qa.invoke(query)
print("Step 11:", result["result"])