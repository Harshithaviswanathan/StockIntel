from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import os

# Use the correct import for Groq
try:
    from langchain_groq import ChatGroq
except ImportError:
    try:
        from langchain_community.chat_models import ChatGroq
    except ImportError:
        ChatGroq = None

# Import other necessary libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

# Import local components
from rag_components.vector_store import VectorStoreManager
from rag_components.data_ingestor import DataIngestor

# Initialize router
router = APIRouter()

# Initialize data ingester and vector store manager
data_ingester = DataIngestor()
vector_store_manager = VectorStoreManager()

@router.post("/rag/query")
async def rag_query(query: Dict[str, str]):
    """
    Perform RAG (Retrieval Augmented Generation) query
    
    :param query: Dictionary containing the query text
    :return: Generated response based on retrieved context
    """
    try:
        # Extract query text
        query_text = query.get('query', '')
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")
        
        # Retrieve relevant documents
        retrieved_docs = vector_store_manager.similarity_search(query_text, k=3)
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Prepare prompt template
        prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful AI assistant. Given the following context:\n{context}\n\n"
            "Answer the following question, using the context to inform your response if relevant:\n{question}"
        )
        
        # Initialize LLM (use a fallback if Groq is not available)
        if ChatGroq is not None:
            try:
                llm = ChatGroq(
                    temperature=0.2, 
                    model_name="mixtral-8x7b-32768"  # or another appropriate model
                )
            except Exception:
                llm = None
        else:
            llm = None
        
        # Fallback to a default language model if Groq is not available
        if llm is None:
            from langchain_community.llms import OpenAI
            llm = OpenAI(temperature=0.2)
        
        # Create the chain
        chain = (
            {"context": lambda x: context, "question": lambda x: query_text}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        # Generate response
        response = chain.invoke({})
        
        return {
            "response": response,
            "context": context,
            "retrieved_docs": [doc.metadata for doc in retrieved_docs]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))