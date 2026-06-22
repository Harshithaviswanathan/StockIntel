import yfinance as yf
import time
from typing import List, Dict, Any
from .vector_store import VectorStoreManager

# Helper functions for yfinance with retry logic
def get_stock_ticker_safe(ticker: str, max_retries: int = 3, base_delay: float = 2.0):
    """Get yfinance Ticker object with retry logic"""
    last_exception = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2.0 ** (attempt - 1))
                print(f"Waiting {delay:.2f} seconds before retry attempt {attempt + 1}...")
                time.sleep(delay)
            else:
                time.sleep(0.5)  # Small delay to space out requests
            return yf.Ticker(ticker)
        except yf.exceptions.YFRateLimitError as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2.0 ** attempt)
                print(f"Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise
    if last_exception:
        raise last_exception

def get_stock_history_safe(ticker_obj, period: str = "1y", max_retries: int = 3, base_delay: float = 2.0):
    """Get stock history with retry logic"""
    last_exception = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2.0 ** (attempt - 1))
                time.sleep(delay)
            else:
                time.sleep(0.5)
            return ticker_obj.history(period=period)
        except yf.exceptions.YFRateLimitError as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2.0 ** attempt)
                time.sleep(delay)
            else:
                raise
    if last_exception:
        raise last_exception

def get_stock_info_safe(ticker_obj, max_retries: int = 3, base_delay: float = 2.0):
    """Get stock info with retry logic"""
    last_exception = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2.0 ** (attempt - 1))
                time.sleep(delay)
            else:
                time.sleep(0.5)
            return ticker_obj.info
        except yf.exceptions.YFRateLimitError as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2.0 ** attempt)
                time.sleep(delay)
            else:
                raise
    if last_exception:
        raise last_exception

def get_stock_financials_safe(ticker_obj, max_retries: int = 3, base_delay: float = 2.0):
    """Get stock financials with retry logic"""
    last_exception = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2.0 ** (attempt - 1))
                time.sleep(delay)
            else:
                time.sleep(0.5)
            return {
                "income_statement": ticker_obj.financials,
                "balance_sheet": ticker_obj.balance_sheet,
                "cashflow": ticker_obj.cashflow
            }
        except yf.exceptions.YFRateLimitError as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2.0 ** attempt)
                time.sleep(delay)
            else:
                raise
    if last_exception:
        raise last_exception

class DataIngestor:
    def __init__(self):
        """
        Initialize data ingestor with vector store manager
        """
        self.vector_store_manager = VectorStoreManager()
    
    def fetch_stock_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Fetch stock data for a given ticker
        
        :param ticker: Stock ticker symbol
        :param period: Data period (default: 1 year)
        :return: Dictionary of stock data
        """
        try:
            # Fetch stock information with retry logic
            stock = get_stock_ticker_safe(ticker)
            
            # Get historical market data with retry logic
            hist = get_stock_history_safe(stock, period=period)
            
            # Get company info with retry logic
            info = get_stock_info_safe(stock)
            
            # Get financial statements with retry logic
            financials = get_stock_financials_safe(stock)
            
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
    
    def prepare_document_text(self, stock_data: Dict[str, Any]) -> List[str]:
        """
        Prepare document text from stock data for vector store
        
        :param stock_data: Stock data dictionary
        :return: List of document texts
        """
        if not stock_data:
            return []
        
        documents = []
        
        # Company overview
        company_doc = f"""
        Stock Ticker: {stock_data.get('ticker', 'N/A')}
        
        Company Overview:
        - Name: {stock_data.get('company_info', {}).get('longName', 'N/A')}
        - Sector: {stock_data.get('company_info', {}).get('sector', 'N/A')}
        - Industry: {stock_data.get('company_info', {}).get('industry', 'N/A')}
        - Description: {stock_data.get('company_info', {}).get('longBusinessSummary', 'No description available')}
        """
        documents.append(company_doc)
        
        # Historical price summary
        hist_data = stock_data.get('historical_data', {})
        if hist_data:
            price_doc = f"""
            Historical Price Summary:
            - Average Close Price: {hist_data.get('Close', []).mean() if 'Close' in hist_data else 'N/A'}
            - Highest Price: {hist_data.get('High', []).max() if 'High' in hist_data else 'N/A'}
            - Lowest Price: {hist_data.get('Low', []).min() if 'Low' in hist_data else 'N/A'}
            """
            documents.append(price_doc)
        
        # Financial statement summary
        financials = stock_data.get('financials', {})
        for statement_type, statement_data in financials.items():
            if statement_data:
                fin_doc = f"""
                {statement_type.replace('_', ' ').title()} Summary:
                """
                for key, value in list(statement_data.items())[:5]:  # Limit to first 5 items
                    fin_doc += f"- {key}: {value}\n"
                documents.append(fin_doc)
        
        return documents
    
    def fetch_and_ingest_all_data(self, ticker: str):
        """
        Fetch stock data and ingest into vector store
        
        :param ticker: Stock ticker symbol
        """
        # Clear existing collection for this ticker
        self.vector_store_manager.clear_collection()
        
        # Fetch stock data
        stock_data = self.fetch_stock_data(ticker)
        
        # Prepare document texts
        documents = self.prepare_document_text(stock_data)
        
        # Prepare metadata
        metadata = [{"ticker": ticker, "type": "stock_info"} for _ in documents]
        
        # Add documents to vector store
        if documents:
            self.vector_store_manager.add_documents(documents, metadata)