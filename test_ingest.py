# force_recreate.py
import os
import shutil
from llm_rag.rag_handler import load_and_split_documents, get_embeddings_model
from langchain_chroma import Chroma
from core.config import RAG_VECTORSTORE_PATH

def force_recreate_vectorstore():
    print("🔥 FORCE RECREATING VECTORSTORE")
    print("=" * 50)
    
    # 1. Xóa vectorstore cũ hoàn toàn
    if os.path.exists(RAG_VECTORSTORE_PATH):
        print(f"🗑️ Deleting old vectorstore: {RAG_VECTORSTORE_PATH}")
        shutil.rmtree(RAG_VECTORSTORE_PATH)
        print("✅ Old vectorstore deleted")
    
    # 2. Load documents
    print("📚 Loading documents...")
    docs = load_and_split_documents()
    if not docs:
        print("❌ No documents found!")
        return False
    
    print(f"✅ Loaded {len(docs)} document chunks")
    
    # 3. Initialize embeddings
    print("🔧 Initializing embeddings...")
    embeddings_model = get_embeddings_model()
    
    # 4. Create new vectorstore
    print("🏗️ Creating new vectorstore...")
    try:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings_model,
            persist_directory=RAG_VECTORSTORE_PATH
        )
        print("✅ New vectorstore created!")
        
        # 5. Test immediately
        print("🧪 Testing new vectorstore...")
        results = vectorstore.similarity_search("Bitcoin", k=3)
        print(f"📊 Test query 'Bitcoin': {len(results)} results")
        
        if results:
            print("🎉 SUCCESS! VectorStore has data!")
            print(f"📄 Sample result: {results[0].page_content[:100]}...")
            return True
        else:
            print("❌ VectorStore still empty after creation!")
            return False
            
    except Exception as e:
        print(f"❌ Error creating vectorstore: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = force_recreate_vectorstore()
    if success:
        print("\n🎉 VectorStore recreation successful!")
        print("Now you can test RAG Q&A!")
    else:
        print("\n❌ VectorStore recreation failed!")