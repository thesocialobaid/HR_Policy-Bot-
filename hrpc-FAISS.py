from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

import os
import time
from dotenv import load_dotenv

load_dotenv()

DEBUG_LIMIT = 300  # Limit for documents in debug mode (set to None for full run)

def upload_htmls():
    print("🚀 Starting document ingestion...")

    loader = DirectoryLoader(
        path="hr-policies",
        glob="**/*.html",
        loader_cls=BSHTMLLoader,
        loader_kwargs={"bs_kwargs": {"features": "html.parser"}, "open_encoding": "utf-8"}
    )
    documents = loader.load()

    if DEBUG_LIMIT:
        documents = documents[:DEBUG_LIMIT]

    print(f"📄 {len(documents)} HTML pages loaded")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    split_documents = text_splitter.split_documents(documents=documents)
    print(f"✂️ Split into {len(split_documents)} chunks")

    print("🔎 Preview of first chunk:")
    print(split_documents[0].page_content[:300], "...")
    print(split_documents[0].metadata)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    batch_size = 500
    db = None

    for i in range(0, len(split_documents), batch_size):
        batch_docs = split_documents[i:i + batch_size]
        print(f"⚙️ Processing batch {i} to {i + len(batch_docs)}...")

        try:
            if db is None:
                db = FAISS.from_documents(batch_docs, embeddings)
            else:
                db.add_documents(batch_docs)

            db.save_local("faiss_index")
            print(f"💾 Saved FAISS index after batch {i}")

        except Exception as e:
            print(f"❌ Error in batch {i}: {e}")
        
        time.sleep(1)  # prevent API throttling

    if os.path.exists("faiss_index/index.faiss"):
        print("✅ index.faiss successfully saved!")
    else:
        print("❌ Failed to save index.faiss")


def faiss_query():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    try:
        new_db = FAISS.load_local("faiss_index", embeddings)
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        return

    query = "Explain the Candidate onBoarding Process"
    docs = new_db.similarity_search(query)

    print(f"🔍 Results for: {query}")
    for i, doc in enumerate(docs):
        print(f"\n📄 Document {i + 1}")
        print("Source:", doc.metadata.get('source', 'N/A'))
        print(doc.page_content[:500], "...")  # preview first 500 chars


if __name__ == "__main__":
    upload_htmls()
    faiss_query()
