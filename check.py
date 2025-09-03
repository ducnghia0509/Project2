# test_vectorstore.py
from llm_rag.rag_handler import get_embeddings_model, ask_question
from langchain_chroma import Chroma
from core.config import RAG_VECTORSTORE_PATH

def test_vectorstore_data():
    print("🧪 TESTING VECTORSTORE AFTER INGEST")
    print("=" * 50)
    
    try:
        # 1. Load vectorstore
        embeddings_model = get_embeddings_model()
        vectorstore = Chroma(persist_directory=RAG_VECTORSTORE_PATH, embedding_function=embeddings_model)
        
        # 2. Test basic search
        results = vectorstore.similarity_search("Bitcoin", k=5)
        print(f"📊 Query 'Bitcoin': {len(results)} results")
        
        if results:
            print("✅ VectorStore có data!")
            for i, doc in enumerate(results[:2]):
                print(f"\n📄 Result {i+1}:")
                print(f"Content: {doc.page_content[:150]}...")
                print(f"Source: {doc.metadata.get('source', 'N/A')}")
        else:
            print("❌ VectorStore trống!")
            return False
        
        # 3. Test với các keywords khác
        test_queries = ["Satoshi", "blockchain", "cryptocurrency", "wallet"]
        for query in test_queries:
            results = vectorstore.similarity_search(query, k=1)
            print(f"📊 Query '{query}': {len(results)} results")
        
        # 4. Test thông qua RAG system
        print("\n🤖 Testing RAG Q&A:")
        test_questions = [
            "Bitcoin do ai tạo ra?",
            "Blockchain là gì?"
        ]
        
        for question in test_questions:
            print(f"\n❓ {question}")
            result = ask_question(question)
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
            else:
                answer = result.get('answer', '')
                sources = len(result.get('sources', []))
                print(f"💬 Answer: {answer[:100]}...")
                print(f"📚 Sources: {sources}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vectorstore_data()
    if success:
        print("\n🎉 VectorStore hoạt động tốt!")
    else:
        print("\n❌ VectorStore có vấn đề!")