from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "contract-index-v2"
EMBED_DIMENSION = 768   # ✅ gemini-embedding-001 dimension

def ensure_index():
    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print("⚠️ Index not found. Creating new index...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("✅ Index created successfully!")
    else:
        print("⚡ Index already exists.")

    return pc.Index(INDEX_NAME)


if __name__ == "__main__":
    ensure_index()
