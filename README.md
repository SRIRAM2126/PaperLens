# 📚 PaperLens – Research Paper Recommendation System using BERT

PaperLens is an AI-powered research paper recommendation system that uses **BERT sentence embeddings** and **cosine similarity** to recommend semantically relevant research papers from the **arXiv dataset**.

Unlike traditional keyword search, PaperLens understands the meaning of a query and recommends papers based on semantic similarity.

---

# 🚀 Features

- 🔍 Semantic search using BERT
- 🤖 Sentence Transformers (all-MiniLM-L6-v2)
- 📄 Research paper recommendation
- ⚡ FastAPI backend
- 🎨 Responsive HTML frontend
- 💾 SQLite database
- 📦 Precomputed BERT embeddings for fast inference
- 📚 Dataset of approximately **297,357 arXiv research papers**

---

# 🛠️ Tech Stack

## Backend
- Python
- FastAPI

## Machine Learning
- Sentence Transformers
- BERT (all-MiniLM-L6-v2)
- Scikit-learn
- NumPy
- Pandas

## Database
- SQLite

## Frontend
- HTML
- CSS
- JavaScript

---

# 📂 Project Structure

```
PaperLens/
│
├── app.py
├── recommender.py
├── preprocessing.py
├── etl_warehouse.py
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── models/
│   ├── bert_embeddings.pkl
│   └── papers_df.pkl
│
├── data/
│   ├── papers_clean.csv
│   ├── raw_dataset.csv
│   └── papers_warehouse.db
│
└── README.md
```

---

# ⚙️ How It Works

1. User enters a research topic.
2. Query is converted into a BERT embedding.
3. Cosine similarity is computed against all paper embeddings.
4. Top matching papers are retrieved.
5. Recommended papers are displayed with titles, abstracts, and links.

---

# 🧠 Recommendation Pipeline

```
User Query
      │
      ▼
Sentence Transformer (BERT)
      │
      ▼
Query Embedding
      │
      ▼
Cosine Similarity
      │
      ▼
Top-K Similar Papers
      │
      ▼
FastAPI Backend
      │
      ▼
Web Interface
```

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/PaperLens.git
```

Move into the project

```bash
cd PaperLens
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python -m uvicorn app:app --reload
```

Open

```
http://127.0.0.1:8000
```

---

# 📈 Dataset

- Source: arXiv
- Total Papers: ~297,357
- Database: SQLite
- Embeddings: BERT Sentence Embeddings

---

# 📊 Machine Learning Model

Model Used:

```
sentence-transformers/all-MiniLM-L6-v2
```

Similarity Metric:

```
Cosine Similarity
```

---

# 🎯 Future Improvements

- PDF summarization
- Hybrid recommendation system
- Citation graph recommendations
- User authentication
- Personalized recommendations
- Vector database integration (FAISS)

---

# 👨‍💻 Author

**Vadthyavath Sriram**
**Kartavya Gupta**

B.Tech Student | AI & Machine Learning Enthusiast

---

# ⭐ If you like this project

Please give it a ⭐ on GitHub.