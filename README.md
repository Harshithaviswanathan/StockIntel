# 📊 Advanced Stock Analysis Retrieval System 📈

A Comprehensive RAG-Powered Investment Intelligence Platform

## 🔍 Overview

This project introduces an innovative Retrieval-Augmented Generation (RAG) powered stock analysis platform that revolutionizes financial intelligence gathering and portfolio management through advanced artificial intelligence and machine learning technologies. The system integrates multiple sophisticated components to provide comprehensive, context-aware stock insights and intelligent portfolio optimization.

## 🚀 Key Technological Components

1. **🔄 Data Ingestion and Vectorization**
   - Utilizes cutting-edge sentence transformer embeddings
   - Converts complex financial documents into semantic vector representations
   - Enables advanced semantic search and contextual analysis

2. **📊 Multi-Dimensional Stock Analysis**
   - Aggregates data from multiple sources including:
     - Historical price movements
     - Financial statements
     - Company fundamentals
     - Real-time market information

3. **🧠 Intelligent Query Processing**
   - Implements a RAG (Retrieval-Augmented Generation) architecture
   - Uses large language models to generate contextually rich financial insights
   - Provides nuanced, data-driven responses to complex financial queries

## ✨ Unique Features

- **📝 Comprehensive Stock Analysis Endpoint**: Generates holistic stock evaluations combining fundamental and technical analyses
- **💼 Advanced Portfolio Optimization**: Dynamically allocates portfolio weights based on risk preferences and stock characteristics
- **🔍 Semantic Vector Search**: Enables deep, context-aware information retrieval
- **⚖️ Flexible Risk-Based Allocation**: Supports low, medium, and high-risk investment strategies

## 🛠️ Technical Architecture

- **🔙 Backend Framework**: FastAPI
- **🧩 Embedding Model**: HuggingFace Sentence Transformers
- **🗃️ Vector Database**: Chroma
- **🤖 AI Model Integration**: Groq API for advanced language processing
- **📊 Data Source**: Yahoo Finance (yfinance)

## 🚀 Getting Started

### 🔧 Local development

**Backend**

```bash
cd Backend1
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend**

```bash
cd Frontend1
npm install
cp .env.example .env   # set VITE_API_URL=http://localhost:8000
npm run dev
```

Open http://localhost:5174

### 🐳 Docker deployment (recommended)

1. Copy `Backend1/.env.example` to `Backend1/.env` and set `GROQ_API_KEY`
2. From the project root:

```bash
docker compose up --build
```

| Service  | URL |
|----------|-----|
| Frontend | http://localhost:8080 |
| Backend  | http://localhost:8000 |
| Health   | http://localhost:8000/health |

Vector data persists in the Docker volume `chroma_data`.

### ☁️ Cloud deployment checklist

- Set `GROQ_API_KEY` in your host environment (never commit it)
- Set `ALLOWED_ORIGINS` to your frontend URL(s)
- Build frontend with `VITE_API_URL=https://your-api-domain.com`
- Mount a persistent volume at `VECTOR_DB_PATH` (default `/app/data/new_vector_db`)
- Set `DEBUG=False` in production


## 💡 Potential Applications

- 👨‍💼 Individual investor research
- 💼 Financial advisory support
- 🤖 Automated portfolio management
- 📈 Investment strategy development

## 📖 About

The system represents a convergence of machine learning, financial data analysis, and artificial intelligence, offering a powerful tool for extracting meaningful insights from complex financial landscapes.

---
✨ Developed by HARSHITHA V (22PT12) ✨
