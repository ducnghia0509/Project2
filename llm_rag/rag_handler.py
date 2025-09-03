# llm_rag/rag_handler.py
import os
import shutil 
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Import các cấu hình RAG trực tiếp và sử dụng chúng
from core.config import (
    RAG_DOCUMENTS_PATH, RAG_VECTORSTORE_PATH, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP,
    RAG_EMBEDDING_PROVIDER, RAG_EMBEDDING_MODEL_NAME_HF, RAG_EMBEDDING_MODEL_NAME_GOOGLE,
    RAG_LLM_PROVIDER, RAG_LLM_MODEL_NAME_GOOGLE, GOOGLE_API_KEY as CFG_GOOGLE_API_KEY 
)

load_dotenv()

# Kiểm tra API Key từ config
if not CFG_GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or core.config. Please set it.")

# --- Khởi tạo các thành phần ---
def get_embeddings_model():
    if RAG_EMBEDDING_PROVIDER == "HuggingFace":
        print(f"Using HuggingFace Embeddings: {RAG_EMBEDDING_MODEL_NAME_HF}")
        return HuggingFaceEmbeddings(
            model_name=RAG_EMBEDDING_MODEL_NAME_HF,
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
    elif RAG_EMBEDDING_PROVIDER == "Google":
        print(f"Using Google Embeddings: {RAG_EMBEDDING_MODEL_NAME_GOOGLE}")
        return GoogleGenerativeAIEmbeddings(
            model=RAG_EMBEDDING_MODEL_NAME_GOOGLE,
            google_api_key=CFG_GOOGLE_API_KEY
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {RAG_EMBEDDING_PROVIDER}")

def get_llm():
    if RAG_LLM_PROVIDER == "Google":
        print(f"Using Google LLM: {RAG_LLM_MODEL_NAME_GOOGLE}")
        return ChatGoogleGenerativeAI(
            model=RAG_LLM_MODEL_NAME_GOOGLE,
            google_api_key=CFG_GOOGLE_API_KEY,
            temperature=0.2,
            # convert_system_message_to_human=True
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {RAG_LLM_PROVIDER}")

# --- Các hàm xử lý ---
def load_and_split_documents(docs_path=RAG_DOCUMENTS_PATH): 
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(
        docs_path,
        glob="**/*[.txt,.pdf]",
        loader_cls=lambda path: PDFPlumberLoader(path) if path.endswith(".pdf") else TextLoader(path, **text_loader_kwargs),
        recursive=True,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    if not documents:
        print(f"Không có tài liệu nào được tải từ: {docs_path}")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,       
        chunk_overlap=RAG_CHUNK_OVERLAP, 
        length_function=len
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Đã tải {len(documents)} tài liệu, chia thành {len(split_docs)} chunks.")
    return split_docs

def create_or_load_vectorstore(split_docs, embeddings, persist_dir=RAG_VECTORSTORE_PATH): 
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"Đang tải VectorStore từ: {persist_dir}")
        vectorstore = Chroma(persist_directory=RAG_VECTORSTORE_PATH, embedding_function=embeddings_model)
        sample_docs = vectorstore.similarity_search("bitcoin adoption", k=5)
        for doc in sample_docs:
            print(f"Sample doc: {doc.page_content[:200]}... Metadata: {doc.metadata}")
    else:
        if not split_docs:
            print(f"Không có documents để tạo VectorStore mới tại {persist_dir}. Vui lòng chạy ingest_data() trước.")
            return None
        print(f"Đang tạo VectorStore mới và lưu vào: {persist_dir}")
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
    return vectorstore


# --- Workflow chính ---
embeddings_model = get_embeddings_model() 
llm_model = get_llm() # Gemini

# --- Script để ingest dữ liệu lần đầu (hoặc khi có tài liệu mới) ---
def ingest_data_to_vectorstore():
    print("Bắt đầu quá trình Ingest dữ liệu vào VectorStore...")
    docs = load_and_split_documents()
    if not docs:
        print("Không có tài liệu để ingest.")
        return

    create_or_load_vectorstore(docs, embeddings_model) 
    print("Quá trình Ingest dữ liệu hoàn tất.")

# --- Hàm để thiết lập QA chain ---
_vectorstore_cache = None
_qa_chain_cache = None

def get_qa_chain():
    global _vectorstore_cache, _qa_chain_cache
    if _qa_chain_cache:
        return _qa_chain_cache

    if not _vectorstore_cache:
        if os.path.exists(RAG_VECTORSTORE_PATH) and os.listdir(RAG_VECTORSTORE_PATH):
            _vectorstore_cache = Chroma(persist_directory=RAG_VECTORSTORE_PATH, embedding_function=embeddings_model)
            print("VectorStore đã được tải từ cache hoặc disk.")
        else:
            print("CẢNH BÁO: VectorStore chưa tồn tại. Hãy chạy ingest_data_to_vectorstore() trước.")
            return None

    retriever = _vectorstore_cache.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    prompt_template = """Bạn là một chuyên gia về tiền điện tử, có nhiệm vụ cung cấp thông tin chi tiết và chính xác về Bitcoin. 
    Chỉ dựa trên dữ liệu được cung cấp, hãy trả lời câu hỏi một cách rõ ràng, ngắn gọn và bằng tiếng Việt. 

Dữ liệu:  
{context}  

Câu hỏi:  
{question}  

Trả lời (bằng tiếng Việt):  
"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    _qa_chain_cache = qa_chain
    print("QA Chain đã được thiết lập với Gemini.")
    return qa_chain

def ask_question(query: str):
    qa_chain = get_qa_chain()
    if not qa_chain:
        return {"error": "Hệ thống RAG chưa sẵn sàng. VectorStore có thể chưa được tạo."}

    print(f"\nĐang xử lý câu hỏi với Gemini: {query}")
    try:
        result = qa_chain.invoke({"query": query})
        answer = result.get("result")
        source_documents = result.get("source_documents")

        response = {
            "question": query,
            "answer": answer,
            "sources": []
        }
        if source_documents:
            for doc in source_documents:
                response["sources"].append({
                    "content_preview": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
        return response
    except Exception as e:
        print(f"Lỗi khi xử lý câu hỏi với Gemini: {e}")
        error_message = str(e)
        if "BlockedPromptException" in str(type(e)):
            error_message = "Câu hỏi hoặc ngữ cảnh có thể đã bị chặn do chính sách nội dung."
        elif "StopCandidateException" in str(type(e)):
             error_message = "Phản hồi từ Gemini đã bị dừng do chính sách nội dung."

        return {"error": error_message, "question": query}


if __name__ == "__main__":
    # 1: Chạy một lần để ingest dữ liệu
    # ingest_data_to_vectorstore()

    # 2: Đặt câu hỏi
    if not (os.path.exists(RAG_VECTORSTORE_PATH) and os.listdir(RAG_VECTORSTORE_PATH)):
         print(f"Chưa có VectorStore tại {RAG_VECTORSTORE_PATH}. Hãy chạy ingest_data_to_vectorstore() trước.")
    else:
        test_question_1 = "Bitcoin do ai tạo ra?"
        answer_1 = ask_question(test_question_1)
        print("\n--- Kết quả cho câu hỏi 1 (Gemini) ---")
        print(f"Câu hỏi: {answer_1.get('question')}")
        print(f"Trả lời: {answer_1.get('answer')}")
        if answer_1.get('sources'):
            print("Nguồn tham khảo:")
            for i, src in enumerate(answer_1.get('sources')):
                print(f"  Nguồn {i+1}: {src.get('metadata', {}).get('source', 'N/A')} - Preview: {src.get('content_preview')}")

        test_question_2 = "Làm sao để tạo ra một loại tiền điện tử kiểu như ETH, hay bitcoin?"
        answer_2 = ask_question(test_question_2)
        print("\n--- Kết quả cho câu hỏi 2 (Gemini) ---")
        print(f"Câu hỏi: {answer_2.get('question')}")
        print(f"Trả lời: {answer_2.get('answer')}")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        print("Chạy ingest từ command line...")
        ingest_data_to_vectorstore()