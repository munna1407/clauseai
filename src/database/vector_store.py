import os
import uuid
from google import genai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# =========================
# Configuration
# =========================
INDEX_NAME = "contract-index-v2"
EMBED_MODEL = "gemini-embedding-001"

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# =========================
# Detect Embedding Dimension (only once)
# =========================
def get_embedding_dimension():
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=["dimension check"]
    )
    return len(response.embeddings[0].values)


EMBED_DIMENSION = get_embedding_dimension()
print(f"✅ Embedding dimension detected: {EMBED_DIMENSION}")


# =========================
# Ensure Pinecone Index Exists
# =========================
def ensure_index():
    existing_indexes = pc.list_indexes().names()

    if INDEX_NAME not in existing_indexes:
        print("⚠️ Index not found. Creating new Pinecone index...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("✅ Pinecone index created successfully!")

    return pc.Index(INDEX_NAME)


# =========================
# Store Chunks
# =========================
def store_chunks(chunks, metadata_list):
    if not chunks:
        return

    index = ensure_index()

    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=chunks
    )

    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": emb.values,
            "metadata": metadata_list[i]
        }
        for i, emb in enumerate(response.embeddings)
    ]

    index.upsert(vectors=vectors)

    print(f"✅ Successfully stored {len(vectors)} chunks in Pinecone")


# =========================
# Query Chunks
# =========================
def query_chunks(query, top_k=5):
    if not query:
        return {}

    index = ensure_index()

    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query]
    )

    query_embedding = response.embeddings[0].values

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results
