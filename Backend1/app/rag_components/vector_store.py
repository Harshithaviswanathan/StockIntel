import os
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

class VectorStoreManager:
    def __init__(self, persist_directory='./chroma_db'):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "stock_documents"
    
    def add_documents(self, documents, metadata=None):
        """
        Add documents to the vector store
        
        :param documents: List of document texts
        :param metadata: Optional list of metadata for documents
        """
        # Create or get collection
        collection = self.client.get_or_create_collection(name=self.collection_name)
        
        # Create Langchain Chroma vector store
        vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        
        # Convert documents to Document objects if they're not already
        from langchain_core.documents import Document
        if documents and not isinstance(documents[0], Document):
            doc_objects = []
            for i, doc in enumerate(documents):
                meta = metadata[i] if metadata and i < len(metadata) else {}
                doc_objects.append(Document(page_content=doc, metadata=meta))
            vectorstore.add_documents(doc_objects)
        else:
            # Add documents
            vectorstore.add_documents(documents, metadatas=metadata)
        
        return True
    
    def similarity_search(self, query, k=3):
        """
        Perform similarity search
        
        :param query: Search query
        :param k: Number of top results to return
        :return: List of most similar documents
        """
        # Create Langchain Chroma vector store
        vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        
        # Perform similarity search
        return vectorstore.similarity_search(query, k=k)
    
    def clear_collection(self):
        """
        Clear all documents from the collection
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass
        self.client.create_collection(name=self.collection_name)