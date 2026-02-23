import os
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.database.vector_store import store_chunks, query_chunks

load_dotenv()

# =========================
# Configure Gemini Client
# =========================
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = "gemini-1.5-flash"  # Text generation model


# =========================
# Process Contract
# =========================
def process_contract(text: str) -> str:
    """
    Splits the contract text into chunks, embeds them, 
    and stores them in Pinecone.
    """
    if not text:
        return "No contract text provided."

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    # Prepare metadata for Pinecone
    metadata_list = [
        {"source": "contract", "chunk_id": i, "text": chunks[i]}
        for i in range(len(chunks))
    ]

    # Store chunks in Pinecone (embeddings)
    store_chunks(chunks, metadata_list)

    return f"✅ Contract processed successfully. {len(chunks)} chunks stored."


# =========================
# Analyze Contract
# =========================
def analyze(query: str) -> str:
    """
    Queries Pinecone for relevant contract chunks and generates
    an AI analysis using the Gemini model.
    """
    if not query:
        return "No query provided."

    # Query relevant chunks from Pinecone
    results = query_chunks(query)

    context_texts = []

    # Safe extraction of matches
    matches = results.matches if hasattr(results, "matches") else []

    for match in matches:
        metadata = match.metadata
        text = metadata.get("text", "") if metadata else ""    
        if text:
            context_texts.append(text)

    if not context_texts:
        return "⚠️ No relevant contract context found."

    # Combine context chunks
    context = "\n\n".join(context_texts)

    # Prepare prompt for Gemini
    prompt = f"""
You are a senior legal contract analyst.

Analyze the contract strictly based on the provided context.
If information is missing, clearly state that.

=====================
CONTRACT CONTEXT:
=====================
{context}

=====================
QUESTION:
=====================
{query}

=====================
RESPONSE FORMAT:
=====================
1. Summary of relevant clauses
2. Legal interpretation
3. Risks (if any)
4. Recommendations
"""

    # ✅ Generate text using modern client method
    try:
        response = client.generations.create(
            model=MODEL_NAME,
            prompt=prompt,
            temperature=0.2,
            max_output_tokens=1024
        )

        # Extract generated text safely
        text_output = ""
        if response and hasattr(response, "generations") and len(response.generations) > 0:
            text_output = response.generations[0].text

        return text_output or "⚠️ Failed to generate response."

    except Exception as e:
        return f"❌ Error generating analysis: {str(e)}"
