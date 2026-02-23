# ClauseAI

AI-powered contract analysis system using LLM + Vector Search.

---

## 🚀 Project Structure

```
clauseai/
│
├── app/                 # Application entry layer (API routes / main app)
│
├── src/                 # Core business logic
│   ├── services/        # Contract processing & AI services
│   ├── utils/           # Helper functions
│   └── config/          # Configuration settings
│
├── create_index.py      # Script to create Pinecone index
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignored files (.env etc.)
└── README.md            # Project documentation
```

---

## 🧠 System Architecture

```
                ┌────────────────────┐
                │     User Upload    │
                │   (Contract PDF)   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  Backend (FastAPI) │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Text Chunking      │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Gemini Embeddings  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Pinecone Vector DB │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Semantic Search    │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ AI Clause Analysis │
                └────────────────────┘
```

---

## ⚙️ Tech Stack

- Python
- FastAPI
- Pinecone (Vector Database)
- Google Gemini (Embeddings + LLM)
- dotenv

---

## 🔐 Environment Variables

Create a `.env` file locally:

```
GOOGLE_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## 👨‍💻 Author

Munna Ansari
