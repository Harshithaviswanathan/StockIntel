from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import traceback
import json
import yfinance as yf
import numpy as np
import time
from functools import wraps
from typing import Callable, Any
from app.config import VECTOR_DB_PATH, DEBUG, PORT, DEFAULT_LLM_MODEL, require_groq_api_key, check_groq_api_key
from app.utils import extract_tickers_from_query

app = FastAPI(title="StockIntel RAG API", version="1.2.0")
APP_BUILD = "groq-first-v4"

load_dotenv()

# Rate limiting and retry logic for yfinance
def retry_yfinance_call(max_retries: int = 3, base_delay: float = 2.0, backoff_factor: float = 2.0):
    """
    Decorator to retry yfinance calls with exponential backoff on rate limit errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds before retry
        backoff_factor: Multiplier for delay on each retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    # Add a small delay between requests to avoid rate limiting
                    if attempt > 0:
                        delay = base_delay * (backoff_factor ** (attempt - 1))
                        print(f"Waiting {delay:.2f} seconds before retry attempt {attempt + 1}...")
                        time.sleep(delay)
                    else:
                        # Small delay even on first attempt to space out requests
                        time.sleep(0.5)
                    
                    return func(*args, **kwargs)
                except yf.exceptions.YFRateLimitError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (backoff_factor ** attempt)
                        print(f"Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        print(f"Max retries reached. Rate limit error persists.")
                        raise
                except Exception as e:
                    # For other exceptions, don't retry
                    raise
            if last_exception:
                raise last_exception
        return wrapper
    return decorator

# Helper function to safely get stock ticker with retry logic
@retry_yfinance_call(max_retries=3, base_delay=2.0, backoff_factor=2.0)
def get_stock_ticker(ticker: str):
    """Get yfinance Ticker object with retry logic"""
    return yf.Ticker(ticker)

# Helper function to safely get stock history with retry logic
@retry_yfinance_call(max_retries=3, base_delay=2.0, backoff_factor=2.0)
def get_stock_history(ticker_obj, period: str = "1y"):
    """Get stock history with retry logic"""
    return ticker_obj.history(period=period)

# Helper function to safely get stock info with retry logic
@retry_yfinance_call(max_retries=3, base_delay=2.0, backoff_factor=2.0)
def get_stock_info(ticker_obj):
    """Get stock info with retry logic"""
    info = ticker_obj.info
    return info if isinstance(info, dict) else {}

# Helper function to safely get financials with retry logic
@retry_yfinance_call(max_retries=3, base_delay=2.0, backoff_factor=2.0)
def get_stock_financials(ticker_obj):
    """Get stock financials with retry logic"""
    return {
        "income_statement": ticker_obj.financials,
        "balance_sheet": ticker_obj.balance_sheet,
        "cashflow": ticker_obj.cashflow
    }

# Add CORS middleware (configure ALLOWED_ORIGINS for production, comma-separated)
from app.config import ALLOWED_ORIGINS as _allowed_origins_env

DEFAULT_ALLOWED_ORIGINS = [
    "https://stock-intel-theta.vercel.app",
    "http://localhost:5174",
    "http://localhost:5173",
    "http://localhost:8080",
]

def _clean_origin(value: str) -> str:
    return value.strip().strip('"').strip("'")

def _origin_allowed(origin: str) -> bool:
    if not origin:
        return False
    if _allowed_origins_raw == "*":
        return True
    if origin in _cors_origins:
        return True
    import re
    return bool(
        re.fullmatch(r"https://[\w-]+\.vercel\.app", origin)
        or re.fullmatch(r"http://localhost(:\d+)?", origin)
    )

_allowed_origins_raw = _allowed_origins_env.strip()
_origin_regex = r"https://.*\.vercel\.app|http://localhost(:\d+)?"

if _allowed_origins_raw == "*":
    _cors_origins = ["*"]
else:
    _cors_origins = list(
        dict.fromkeys(
            DEFAULT_ALLOWED_ORIGINS
            + [_clean_origin(o) for o in _allowed_origins_raw.split(",") if o.strip()]
        )
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def ensure_cors_headers(request: Request, call_next):
    """Ensure CORS headers are present even on error responses."""
    origin = request.headers.get("origin", "")
    try:
        response = await call_next(request)
    except Exception as exc:
        print(f"Unhandled request error: {exc}")
        response = JSONResponse(status_code=500, content={"error": "Internal server error"})

    if _origin_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    return response

_embeddings_ready = False
_embeddings_error: str | None = None
_shared_embeddings = None
_shared_vectordb = None
_init_lock = __import__("threading").Lock()

# Simple vector store setup
def setup_embeddings():
    """ONNX-based embeddings — ~10x less RAM than PyTorch sentence-transformers."""
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def setup_vectordb(embeddings, persist_directory=None):
    import chromadb
    from langchain_chroma import Chroma

    path = persist_directory or VECTOR_DB_PATH
    os.makedirs(path, exist_ok=True)
    client = chromadb.PersistentClient(path=path)
    return Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name="stock_documents",
    )

def get_shared_embeddings():
    """Load embedding model once per process (expensive — do not call on /health)."""
    global _shared_embeddings, _embeddings_ready, _embeddings_error
    if _shared_embeddings is not None:
        return _shared_embeddings
    with _init_lock:
        if _shared_embeddings is not None:
            return _shared_embeddings
        try:
            print("Loading embedding model (one-time)...")
            _shared_embeddings = setup_embeddings()
            _embeddings_ready = True
            _embeddings_error = None
            print("✅ Embedding model loaded")
        except Exception as exc:
            _embeddings_ready = False
            _embeddings_error = str(exc)
            raise
    return _shared_embeddings

def get_shared_vectordb():
    """Return shared Chroma instance; initializes embeddings lazily on first use."""
    global _shared_vectordb
    if _shared_vectordb is not None:
        return _shared_vectordb
    with _init_lock:
        if _shared_vectordb is not None:
            return _shared_vectordb
        _shared_vectordb = setup_vectordb(get_shared_embeddings())
    return _shared_vectordb

def try_rag_search(query: str, k: int = 2) -> tuple[str, list]:
    """Optional RAG lookup — returns empty context if embeddings/vector store unavailable."""
    try:
        vectordb = get_shared_vectordb()
        docs = vectordb.similarity_search(query, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)
        sources = [doc.metadata for doc in docs]
        return context, sources
    except Exception as exc:
        print(f"RAG search skipped: {exc}")
        return "", []

def _format_ticker_market_data(tickers: list[str]) -> str:
    """Live Yahoo Finance snapshot for prompt context."""
    if not tickers:
        return ""
    lines = ["Current market data:"]
    for ticker in tickers:
        try:
            stock = get_stock_ticker(ticker)
            hist = get_stock_history(stock, period="1d")
            price = hist["Close"].iloc[-1]
            info = get_stock_info(stock) or {}
            lines.append(f"- {ticker}: ${price:.2f}")
            lines.append(f"  P/E Ratio: {info.get('trailingPE', 'N/A')}")
            lines.append(f"  Market Cap: ${info.get('marketCap', 0) / 1e9:.2f}B")
        except yf.exceptions.YFRateLimitError:
            lines.append(f"- {ticker}: Rate limited. Please try again later.")
        except Exception as exc:
            lines.append(f"- {ticker}: Error retrieving data: {exc}")
    return "\n".join(lines)

def _build_fundamental_context(ticker: str, company_info: dict, stock_data: dict | None) -> str:
    """Fundamental context from Yahoo Finance, with optional ingested RAG docs."""
    parts = [
        f"Ticker: {ticker}",
        f"Company: {company_info.get('longName', ticker)}",
        f"Sector: {company_info.get('sector', 'N/A')}",
        f"Industry: {company_info.get('industry', 'N/A')}",
        f"P/E Ratio: {company_info.get('trailingPE', 'N/A')}",
        f"EPS: {company_info.get('trailingEps', 'N/A')}",
        f"Market Cap: {company_info.get('marketCap', 'N/A')}",
    ]
    summary = company_info.get("longBusinessSummary")
    if summary:
        parts.append(f"Business summary: {summary[:1500]}")

    rag_context, _ = try_rag_search(f"{ticker} company financials", k=3)
    if rag_context.strip():
        parts.append(f"Ingested research:\n{rag_context}")
    elif not stock_data:
        parts.append(
            "Limited live data available. Use well-known public information about this company."
        )
    return "\n".join(parts)

# Add this class definition to your main.py file before the endpoints
class VectorStoreManager:
    def __init__(self):
        """Initialize the vector store manager"""
        self.embeddings = get_shared_embeddings()
        self.vectordb = get_shared_vectordb()
    
    def clear_collection(self, ticker):
        """Clear only documents for a specific ticker"""
        try:
            if hasattr(self.vectordb, "_collection"):
            # Only clear documents for this specific ticker
                self.vectordb._collection.delete(
                    where={"ticker": ticker}
                )
                return True
            return False
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def add_documents(self, documents, metadata):
        """Add documents to the vector store"""
        if not documents:
            return False
        
        texts = documents
        metadatas = metadata
        
        # Add to vector database
        self.vectordb.add_texts(texts=texts, metadatas=metadatas)
        #self.vectordb.persist()
        
        print(f"Added {len(texts)} documents to vector database")
        return True

def _company_info(stock_data: dict | None) -> dict:
    if not stock_data:
        return {}
    company_info = stock_data.get("company_info")
    return company_info if isinstance(company_info, dict) else {}

class DataIngestor:
    def __init__(self):
        """Initialize data ingestor; vector store loads only when ingesting."""
        self._vector_store_manager: VectorStoreManager | None = None

    @property
    def vector_store_manager(self) -> VectorStoreManager:
        if self._vector_store_manager is None:
            self._vector_store_manager = VectorStoreManager()
        return self._vector_store_manager

    def fetch_stock_data(self, ticker: str, period: str = "1y"):
        """Fetch stock data for a given ticker"""
        try:
            # Fetch stock information with retry logic
            stock = get_stock_ticker(ticker)

            # Get historical market data with retry logic
            hist = get_stock_history(stock, period=period)

            # Get company info with retry logic
            info = get_stock_info(stock)

            # Get financial statements with retry logic
            financials = get_stock_financials(stock)

            return {
                "ticker": ticker,
                "historical_data": hist.to_dict(),
                "company_info": info,
                "financials": {k: v.to_dict() for k, v in financials.items()}
            }
        except yf.exceptions.YFRateLimitError as e:
            print(f"Rate limit error fetching data for {ticker}: {e}")
            print("Please wait a few minutes before trying again.")
            return {}
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return {}

    def prepare_document_text(self, stock_data):
        """Prepare document text from stock data for vector store"""
        if not stock_data:
            return []

        documents = []
        company_info = _company_info(stock_data)

        # Company overview
        company_doc = f"""
        Stock Ticker: {stock_data.get('ticker', 'N/A')}
        
        Company Overview:
        - Name: {company_info.get('longName', 'N/A')}
        - Sector: {company_info.get('sector', 'N/A')}
        - Industry: {company_info.get('industry', 'N/A')}
        - Description: {company_info.get('longBusinessSummary', 'No description available')}
        """
        documents.append(company_doc)

        # Historical price summary
        hist_data = stock_data.get('historical_data', {})
        if hist_data and 'Close' in hist_data:
            # Convert dictionary values to a list and then calculate statistics
            close_values = list(hist_data['Close'].values())
            high_values = list(hist_data['High'].values()) if 'High' in hist_data else []
            low_values = list(hist_data['Low'].values()) if 'Low' in hist_data else []
            
            avg_close = sum(close_values) / len(close_values) if close_values else 'N/A'
            max_high = max(high_values) if high_values else 'N/A'
            min_low = min(low_values) if low_values else 'N/A'
            
            price_doc = f"""
            Historical Price Summary:
            - Average Close Price: {avg_close if avg_close != 'N/A' else 'N/A'}
            - Highest Price: {max_high if max_high != 'N/A' else 'N/A'}
            - Lowest Price: {min_low if min_low != 'N/A' else 'N/A'}
            """
            documents.append(price_doc)

        # Financial statement summary
        financials = stock_data.get('financials', {})
        for statement_type, statement_data in financials.items():
            if statement_data:
                fin_doc = f"""
                {statement_type.replace('_', ' ').title()} Summary:
                """
                # Get first 5 items, handling nested dictionary structure
                items = list(statement_data.items())[:5]
                for key, value in items:
                    fin_doc += f"- {key}: {value}\n"
                documents.append(fin_doc)

        return documents

    def fetch_and_ingest_all_data(self, ticker: str):
        """Fetch stock data and ingest into vector store"""
        # Clear existing collection for this ticker
        self.vector_store_manager.clear_collection(ticker)

        # Fetch stock data
        stock_data = self.fetch_stock_data(ticker)
        if not stock_data:
            raise ValueError(
                "Could not fetch stock data from Yahoo Finance. "
                "It may be rate-limiting — wait 2-3 minutes and retry."
            )

        # Prepare document texts
        documents = self.prepare_document_text(stock_data)
        if not documents:
            raise ValueError(
                "No documents could be built from stock data. Try again in a few minutes."
            )

        # Prepare metadata
        metadata = [{"ticker": ticker, "type": "stock_info"} for _ in documents]

        # Add documents to vector store
        if documents:
            return self.vector_store_manager.add_documents(documents, metadata)
        return False

# Initialize components at startup
@app.on_event("startup")
async def startup_event():
    print("Initializing components...")
    groq_status = check_groq_api_key()
    if groq_status["valid"]:
        print("✅ Groq API key validated")
    else:
        print(f"⚠️ Groq API key issue: {groq_status['message']}")
    print("✅ Server ready (embedding model loads on first RAG request)")

# Root endpoint to display all routes
@app.get("/")
async def root():
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": route.methods
        })
    return {"available_routes": routes}

# Test endpoint (legacy)
@app.get("/test")
async def test():
    return {"message": "Test endpoint working!"}


@app.get("/test/groq")
async def test_groq():
    """Verify Groq works without loading embeddings (safe on Render free tier)."""
    try:
        result = call_groq_api("Reply with exactly: Groq is working on Render.")
        return {"ok": True, "result": result, "build": APP_BUILD}
    except ValueError as e:
        return {"ok": False, "error": str(e), "build": APP_BUILD}


@app.get("/health")
async def health():
    """Fast health check for Render deploy probes — must respond in under 5 seconds."""
    return {"status": "ok", "service": "stockintel-api", "build": APP_BUILD}


@app.get("/health/status")
async def health_status():
    """Detailed status for the frontend dashboard."""
    import chromadb

    vector_path = VECTOR_DB_PATH
    vector_exists = os.path.isdir(vector_path)
    doc_count = 0
    if vector_exists:
        try:
            db = chromadb.PersistentClient(path=vector_path)
            col = db.get_or_create_collection("stock_documents")
            doc_count = col.count()
        except Exception:
            doc_count = 0

    groq_status = check_groq_api_key()

    return {
        "status": "ok" if groq_status["valid"] and _embeddings_ready else "degraded",
        "groq_configured": groq_status["configured"],
        "groq_valid": groq_status["valid"],
        "groq_message": groq_status["message"],
        "embeddings_ok": _embeddings_ready,
        "embeddings_error": _embeddings_error,
        "vector_db_path": vector_path,
        "vector_db_exists": vector_exists,
        "document_count": doc_count,
    }

# Define request models
class IngestDataRequest(BaseModel):
    ticker: str

class RAGQueryRequest(BaseModel):
    question: str
    ticker: str = None

class AgentQueryRequest(BaseModel):
    query: str

class PortfolioOptimizationRequest(BaseModel):
    tickers: list[str]
    allocations: dict[str, float]
    risk_preference: str

# Create the ingest_stock_data endpoint
@app.post("/rag/ingest_stock_data")
async def ingest_stock_data(request: IngestDataRequest):
    """Ingest data for a specific stock into the RAG system"""
    try:
        # Create a new DataIngestor instance
        data_ingester = DataIngestor()
        
        print(f"Ingesting data for ticker: {request.ticker}")
        success = data_ingester.fetch_and_ingest_all_data(request.ticker)
        print(f"Ingestion result: {success}")
        if not success:
            return {
                "success": False,
                "error": "Ingestion completed but no documents were stored.",
                "ticker": request.ticker,
            }
        return {"success": success, "ticker": request.ticker}
    except ValueError as e:
        print(f"Ingestion validation error: {e}")
        return {"success": False, "error": str(e), "ticker": request.ticker}
    except Exception as e:
        print(f"Error ingesting data: {e}")
        print(traceback.format_exc())
        return {"success": False, "error": str(e), "ticker": request.ticker}

# Regular query endpoint
@app.post("/rag/query")
async def rag_query(request: RAGQueryRequest):
    """Query the RAG system for stock information"""
    try:
        # Initialize components
        vectordb = get_shared_vectordb()
        
        # Format the query
        query = request.question
        if request.ticker:
            query = f"{request.ticker}: {request.question}"
        
        # Search the vector store
        docs = vectordb.similarity_search(query, k=2)
        
        # Extract content from docs
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata for doc in docs]
        
        # Create a prompt for Groq
        prompt = f"""
        You are a financial expert assistant. Use the following information to answer the question.
        If the information doesn't contain the answer, say that you don't have enough information.
        
        Information:
        {context}
        
        Question: {request.question} about {request.ticker if request.ticker else 'the company'}.
        
        Provide a well-structured, detailed analysis based on the information. Include specific 
        numbers and data points where available. Format the response using markdown for readability.
        """
        
        # Call Groq API
        answer = call_groq_api(prompt)
        
        return {
            "answer": answer,
            "sources": sources
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        print(f"Error querying RAG system: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}

def call_groq_api(prompt, model=None):
    """Call Groq API directly. Raises ValueError when the key or request is invalid."""
    import requests

    model = model or DEFAULT_LLM_MODEL
    api_key = require_groq_api_key()

    if DEBUG:
        print("Groq API request initiated")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a financial expert assistant providing detailed stock and financial analysis."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 800
    }
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]

    print(f"Error from Groq API: {response.status_code} - {response.text}")
    if response.status_code == 401:
        raise ValueError(
            "Invalid Groq API key. Create a new key at console.groq.com and update GROQ_API_KEY in Render."
        )
    try:
        detail = response.json().get("error", {}).get("message", response.text)
    except Exception:
        detail = response.text
    raise ValueError(f"Groq API error ({response.status_code}): {detail}")

# Agent query endpoint
@app.post("/rag/agent_query")
async def agent_query(request: AgentQueryRequest):
    """Run the stock agent with a user query"""
    try:
        require_groq_api_key()

        context, _ = try_rag_search(request.query, k=2)
        max_tokens = 2000
        if len(context) > max_tokens:
            context = context[:max_tokens] + "..."
        if not context.strip():
            context = "No ingested documents yet — use live market data and general financial knowledge."

        tickers = extract_tickers_from_query(request.query)
        ticker_info = _format_ticker_market_data(tickers)
        
        prompt = f"""
        You are a sophisticated stock analysis agent that helps users with financial questions.
        
        User query: {request.query}
        
        Relevant information from my knowledge base:
        {context}
        
        {ticker_info if ticker_info else ""}
        
        Provide a detailed, well-structured response to the user's query. Use specific data points and 
        numbers where available. If you're analyzing a stock, include both fundamental and technical 
        factors. Format your response using markdown for readability.
        """
        
        response = call_groq_api(prompt)
        return {"result": response}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        print(f"Error in agent query: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
# Comprehensive analysis endpoint
@app.get("/rag/comprehensive_analysis/{ticker}")
async def comprehensive_analysis(ticker: str):
    """Perform a comprehensive analysis of a stock"""
    try:
        require_groq_api_key()

        data_ingester = DataIngestor()
        stock_data = data_ingester.fetch_stock_data(ticker)
        company_info = _company_info(stock_data)
        fundamental_context = _build_fundamental_context(ticker, company_info, stock_data)
        _, rag_sources = try_rag_search(f"{ticker} company financials", k=3)
        
        # Get live data using yfinance for technical analysis
        import numpy as np
        
        try:
            stock = get_stock_ticker(ticker)
            hist = get_stock_history(stock, period="200d")  # Get 200 days of data for technical analysis
        except yf.exceptions.YFRateLimitError as e:
            print(f"Rate limit error in comprehensive analysis for {ticker}: {e}")
            return {
                "error": "Rate limit error. Yahoo Finance has rate limited our requests. Please wait a few minutes and try again.",
                "ticker": ticker
            }
        except Exception as e:
            print(f"Error fetching stock data for technical analysis: {e}")
            return {
                "error": f"Error fetching stock data: {str(e)}",
                "ticker": ticker
            }
        
        if hist.empty:
            return {
                "error": "No historical data available for this ticker.",
                "ticker": ticker
            }
        
        # Calculate technical indicators
        # Simple Moving Averages
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        
        # RSI - Relative Strength Index
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD - Moving Average Convergence Divergence
        hist['EMA_12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA_26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
        hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        hist['Middle_Band'] = hist['Close'].rolling(window=20).mean()
        hist['STD'] = hist['Close'].rolling(window=20).std()
        hist['Upper_Band'] = hist['Middle_Band'] + (hist['STD'] * 2)
        hist['Lower_Band'] = hist['Middle_Band'] - (hist['STD'] * 2)
        
        # Get current values for technical indicators
        current_price = hist['Close'].iloc[-1]
        current_sma_50 = hist['SMA_50'].iloc[-1]
        current_sma_200 = hist['SMA_200'].iloc[-1]
        current_rsi = hist['RSI'].iloc[-1]
        current_macd = hist['MACD'].iloc[-1]
        current_signal = hist['Signal_Line'].iloc[-1]
        current_upper_band = hist['Upper_Band'].iloc[-1]
        current_middle_band = hist['Middle_Band'].iloc[-1]
        current_lower_band = hist['Lower_Band'].iloc[-1]
        
        # Determine support and resistance levels (simplified)
        recent_lows = hist['Low'].tail(30).nsmallest(3).mean()
        recent_highs = hist['High'].tail(30).nlargest(3).mean()
        
        # Prepare technical data
        technical_data = {
            "Current Price": current_price,
            "SMA_50": current_sma_50,
            "SMA_200": current_sma_200,
            "RSI_14": current_rsi,
            "MACD": current_macd,
            "Signal_Line": current_signal,
            "Upper_Band": current_upper_band,
            "Middle_Band": current_middle_band,
            "Lower_Band": current_lower_band,
            "Support_Level": recent_lows,
            "Resistance_Level": recent_highs
        }
        
        # Create separate prompts for fundamental and technical analysis
        fundamental_prompt = f"""
        Provide a fundamental analysis of {ticker} based on the following information.
        
        Company information and financial data:
        {fundamental_context}
        
        Focus on business model, financial performance, growth prospects, and valuation metrics.
        Format your response using markdown for readability.
        Keep your analysis concise and focused on fundamental factors only.
        """
        
        technical_prompt = f"""
        Provide a technical analysis of {ticker} based on the following technical indicators:
        
        Current Price: ${technical_data['Current Price']:.2f}
        50-Day Moving Average: ${technical_data['SMA_50']:.2f}
        200-Day Moving Average: ${technical_data['SMA_200']:.2f}
        Relative Strength Index (RSI-14): {technical_data['RSI_14']:.2f}
        MACD: {technical_data['MACD']:.4f}
        MACD Signal Line: {technical_data['Signal_Line']:.4f}
        Bollinger Bands:
        - Upper Band: ${technical_data['Upper_Band']:.2f}
        - Middle Band: ${technical_data['Middle_Band']:.2f}
        - Lower Band: ${technical_data['Lower_Band']:.2f}
        Support Level: ${technical_data['Support_Level']:.2f}
        Resistance Level: ${technical_data['Resistance_Level']:.2f}
        
        Focus ONLY on technical analysis patterns, price movements, and trading signals.
        Identify the trend direction, support/resistance levels, and any actionable signals.
        Do NOT include fundamental analysis or company information.
        Format your response using markdown for readability.
        """
        
        # Get separate analyses from LLM
        fundamental_analysis = call_groq_api(fundamental_prompt)
        technical_analysis = call_groq_api(technical_prompt)
        
        # Create combined prompt for overall recommendation
        outlook_prompt = f"""
        Based on both fundamental and technical analysis, provide an overall outlook and recommendation for {ticker}.
        
        Fundamental Analysis Summary:
        {fundamental_analysis[:500]}...
        
        Technical Analysis Summary:
        {technical_analysis[:500]}...
        
        Provide a concise outlook and clear investment recommendation (Buy, Hold, or Sell).
        Include a target price range and key risks to watch.
        Format your response using markdown for readability.
        """
        
        # Get overall recommendation
        outlook = call_groq_api(outlook_prompt)
        
        # Extract basic information for structured response
        company_name = company_info.get("longName", ticker)
        sector = company_info.get("sector", "N/A")
        market_cap = company_info.get("marketCap", 0)
        pe_ratio = company_info.get("trailingPE", "N/A")
        eps = company_info.get("trailingEps", "N/A")
        
        # Create structured response
        response = {
            "stock_info": {
                "name": company_name,
                "symbol": ticker,
                "sector": sector,
                "price": current_price,
                "market_cap": market_cap
            },
            "fundamental_analysis": {
                "metrics": {
                    "PE Ratio": pe_ratio,
                    "EPS": eps,
                },
                "analysis": fundamental_analysis
            },
            "technical_analysis": {
                "indicators": technical_data,
                "analysis": technical_analysis
            },
            "rag_insights": {
                "answer": outlook,
                "sources": rag_sources
            }
        }
        
        return response
    except ValueError as e:
        return {"error": str(e), "ticker": ticker}
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
    
@app.get("/rag/list_documents/{ticker}")
async def list_documents(ticker: str):
    """List documents in the vector database for a specific ticker"""
    try:
        # Initialize components
        vectordb = get_shared_vectordb()
        
        # Query for documents with the ticker
        results = vectordb.similarity_search(ticker, k=20)
        
        # Format the results
        documents = []
        for i, doc in enumerate(results):
            documents.append({
                "id": i,
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        print(f"Error listing documents: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
# Portfolio optimization endpoint
@app.post("/rag/optimize_portfolio")
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """Optimize a portfolio using the RAG agent"""
    try:
        require_groq_api_key()

        # Portfolio uses live Yahoo Finance + Groq (no vector store required)
        tickers_data = {}
        for ticker in request.tickers:
            ticker_context = ""
            try:
                stock = get_stock_ticker(ticker)
                hist_1d = get_stock_history(stock, period="1d")
                info = get_stock_info(stock) or {}
                
                current_data = {
                    "price": hist_1d["Close"].iloc[-1],
                    "pe_ratio": info.get("trailingPE", "N/A"),
                    "beta": info.get("beta", 1.0),
                    "market_cap": info.get("marketCap", 0),
                    "dividend_yield": info.get("dividendYield", 0)
                }
                
                hist = get_stock_history(stock, period="1y")
                if not hist.empty:
                    returns = hist["Close"].pct_change().dropna()
                    current_data["avg_return"] = returns.mean() * 252 * 100
                    current_data["volatility"] = returns.std() * np.sqrt(252) * 100

                rag_context, _ = try_rag_search(f"{ticker} analysis", k=1)
                ticker_context = rag_context
                
                tickers_data[ticker] = {
                    "context": ticker_context or f"Live market data for {ticker}.",
                    "current_data": current_data
                }
            except yf.exceptions.YFRateLimitError as e:
                print(f"Rate limit error fetching data for {ticker}: {e}")
                tickers_data[ticker] = {
                    "context": ticker_context,
                    "current_data": {"error": "Rate limited. Please try again later."}
                }
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                tickers_data[ticker] = {
                    "context": ticker_context,
                    "current_data": {"error": str(e)}
                }
        
        # Create a prompt for portfolio optimization
        portfolio_context = "\n\n".join([
            f"--- {ticker} ---\n{data['context']}\nCurrent Data: {json.dumps(data['current_data'])}"
            for ticker, data in tickers_data.items()
        ])
        
        # Truncate if too long
        max_tokens = 3000
        if len(portfolio_context) > max_tokens:
            portfolio_context = portfolio_context[:max_tokens] + "..."
        
        prompt = f"""
        As a portfolio optimization expert, optimize the following portfolio based on a {request.risk_preference} risk profile:
        
        Tickers: {', '.join(request.tickers)}
        Current allocations: {json.dumps(request.allocations)}
        Risk preference: {request.risk_preference} (low, medium, or high)
        
        Stock information:
        {portfolio_context}
        
        Please optimize this portfolio considering:
        1. The risk profile ({request.risk_preference})
        2. Diversification principles
        3. Current market conditions
        4. Each stock's historical performance and metrics
        
        First, analyze each stock's strengths, weaknesses, and fit with the risk profile.
        Then, provide an optimized allocation with clear justification for each weight.
        
        IMPORTANT: Your final allocation recommendations MUST be presented in a clear table format with the exact format:
        | Ticker | Optimized Allocation |
        | --- | --- |
        | TICKER1 | XX% |
        | TICKER2 | YY% |
        
        Where the percentages must sum to exactly 100%.
        
        Return your analysis in markdown format, followed by your allocation table and justification.
        """
        
        # Call LLM for portfolio optimization
        analysis = call_groq_api(prompt)
        
        # Try to extract the LLM's allocation recommendations first
        try:
            import re
            # Look for a table or list format in the analysis
            allocation_pattern = r'([A-Z]{1,5})\s*\|\s*(\d+\.?\d*)%'
            matches = re.findall(allocation_pattern, analysis)
            
            llm_portfolio = {}
            if matches:
                for ticker, allocation in matches:
                    if ticker in request.tickers:  # Only include tickers from the request
                        llm_portfolio[ticker] = float(allocation)
                
                # Check if all requested tickers are included and allocations sum close to 100%
                if (all(ticker in llm_portfolio for ticker in request.tickers) and 
                    abs(sum(llm_portfolio.values()) - 100) < 10):
                    
                    # Normalize to exactly 100%
                    total = sum(llm_portfolio.values())
                    optimized_portfolio = {
                        ticker: round((alloc / total) * 100, 2) 
                        for ticker, alloc in llm_portfolio.items()
                    }
                else:
                    # If allocations don't sum to ~100% or not all tickers found, 
                    # use calculated allocations
                    optimized_portfolio = calculate_portfolio_allocation(
                        tickers_data, request.tickers, request.risk_preference
                    )
            else:
                # If we didn't find any matches, try a different pattern
                allocation_pattern = r'([A-Z]{1,5})(?:[:\s-]+)(\d+\.?\d*)%'
                matches = re.findall(allocation_pattern, analysis)
                
                if matches:
                    for ticker, allocation in matches:
                        if ticker in request.tickers:
                            llm_portfolio[ticker] = float(allocation)
                    
                    if (all(ticker in llm_portfolio for ticker in request.tickers) and 
                        abs(sum(llm_portfolio.values()) - 100) < 10):
                        
                        total = sum(llm_portfolio.values())
                        optimized_portfolio = {
                            ticker: round((alloc / total) * 100, 2) 
                            for ticker, alloc in llm_portfolio.items()
                        }
                    else:
                        optimized_portfolio = calculate_portfolio_allocation(
                            tickers_data, request.tickers, request.risk_preference
                        )
                else:
                    # If no pattern matches, use calculated allocations
                    optimized_portfolio = calculate_portfolio_allocation(
                        tickers_data, request.tickers, request.risk_preference
                    )
        except Exception as e:
            print(f"Error extracting allocations from LLM response: {e}")
            print(traceback.format_exc())
            # Use calculated portfolio as fallback
            optimized_portfolio = calculate_portfolio_allocation(
                tickers_data, request.tickers, request.risk_preference
            )
        
        return {
            "optimized_portfolio": optimized_portfolio,
            "analysis": analysis
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        print(f"Error optimizing portfolio: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}

def calculate_portfolio_allocation(tickers_data, tickers, risk_preference):
    """Calculate portfolio allocation based on risk preference and stock metrics"""
    import numpy as np
    
    optimized_portfolio = {}
    total_weight = 0
    
    for ticker in tickers:
        if ticker in tickers_data and "current_data" in tickers_data[ticker]:
            current_data = tickers_data[ticker]["current_data"]
            
            # Get metrics for optimization (with defaults if missing)
            volatility = current_data.get("volatility", 20)
            avg_return = current_data.get("avg_return", 10)
            beta = current_data.get("beta", 1.0)
            
            # Adjust weight based on risk preference
            if risk_preference == "low":
                # For low risk, prioritize low volatility and beta
                if volatility > 0 and beta > 0:
                    weight = 100 / (volatility * beta)
                else:
                    weight = 10
            elif risk_preference == "high":
                # For high risk, prioritize return
                weight = max(avg_return, 5)  # Minimum weight of 5
            else:  # medium
                # For medium risk, balance return and risk
                if volatility > 0:
                    weight = (avg_return + 10) / (volatility + 10)  # Adding constants to avoid division by zero
                else:
                    weight = avg_return + 1
            
            optimized_portfolio[ticker] = weight
            total_weight += weight
        else:
            # Equal weight if data is missing
            optimized_portfolio[ticker] = 100 / len(tickers)
            total_weight += 100 / len(tickers)
    
    # Normalize to 100%
    for ticker in optimized_portfolio:
        optimized_portfolio[ticker] = round((optimized_portfolio[ticker] / total_weight) * 100, 2)
    
    return optimized_portfolio

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)