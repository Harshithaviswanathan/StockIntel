# rag_components/stock_agent.py

from langchain.chains import RetrievalQA
from langchain_community.llms import Groq
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
import yfinance as yf
import json
import os
import numpy as np
import pandas as pd
import time
from functools import wraps
from typing import Dict, List, Any, Optional, Callable

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

class StockAgent:
    """Agentic interface for stock analysis using LLM and RAG"""
    def __init__(self, rag_engine, api_key):
        self.rag_engine = rag_engine
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self):
        """Create tools for the agent"""
        tools = [
            Tool(
                name="StockInfo",
                func=self._get_stock_info,
                description="Get basic information about a stock by ticker symbol"
            ),
            Tool(
                name="FundamentalAnalysis",
                func=self._get_fundamental_analysis,
                description="Get fundamental analysis of a stock including PE ratio, EPS, etc."
            ),
            Tool(
                name="TechnicalAnalysis",
                func=self._get_technical_analysis,
                description="Get technical indicators like SMA, RSI, MACD for a stock"
            ),
            Tool(
                name="RAGQuery",
                func=self._rag_query,
                description="Search for specific financial information about a company using the RAG system"
            ),
            Tool(
                name="PortfolioOptimization",
                func=self._optimize_portfolio,
                description="Optimize a portfolio based on tickers, allocations, and risk preference"
            )
        ]
        return tools
    
    def _create_agent(self):
        """Create the agent with tools"""
        llm = Groq(api_key=self.api_key, model_name="llama-3.1-8b-instant")
        
        # Create prompt for the financial agent
        prompt = PromptTemplate.from_template(
            """
            You are a sophisticated financial analysis agent designed to help with stock analysis and portfolio optimization.
            
            Your key capabilities:
            1. Looking up current stock information
            2. Performing fundamental analysis
            3. Performing technical analysis
            4. Retrieving detailed financial information from your knowledge base
            5. Optimizing portfolios based on risk preferences
            
            To solve a problem, carefully decompose it into steps and use the appropriate tools.
            When analyzing stocks, consider both fundamental and technical factors.
            When optimizing portfolios, consider risk-reward tradeoffs.
            
            {tools}
            
            {agent_scratchpad}
            
            User's request: {input}
            """
        )
        
        agent = create_react_agent(llm, self.tools, prompt)
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10
        )
    
    def _get_stock_info(self, ticker: str):
        """Get basic stock information"""
        try:
            stock = get_stock_ticker_safe(ticker)
            hist = get_stock_history_safe(stock, period="1d")
            info_data = get_stock_info_safe(stock)
            info = {
                "name": info_data.get("longName"),
                "symbol": info_data.get("symbol"),
                "sector": info_data.get("sector"),
                "industry": info_data.get("industry"),
                "market_cap": info_data.get("marketCap"),
                "price": hist["Close"].iloc[-1] if not hist.empty else None,
                "currency": info_data.get("currency")
            }
            return json.dumps(info)
        except yf.exceptions.YFRateLimitError as e:
            return f"Rate limit error retrieving stock info for {ticker}. Please try again later."
        except Exception as e:
            return f"Error retrieving stock info for {ticker}: {str(e)}"
    
    def _get_fundamental_analysis(self, ticker: str):
        """Get fundamental analysis"""
        try:
            stock = get_stock_ticker_safe(ticker)
            info_data = get_stock_info_safe(stock)
            fundamentals = {
                "PE Ratio": info_data.get("trailingPE"),
                "Forward PE": info_data.get("forwardPE"),
                "PB Ratio": info_data.get("priceToBook"),
                "EPS": info_data.get("trailingEps"),
                "Forward EPS": info_data.get("forwardEps"),
                "Dividend Yield": info_data.get("dividendYield"),
                "ROE": info_data.get("returnOnEquity"),
                "Profit Margin": info_data.get("profitMargins"),
                "Revenue Growth": info_data.get("revenueGrowth"),
                "Debt to Equity": info_data.get("debtToEquity")
            }
            
            # Add analysis text
            analysis = "Fundamental Analysis:\n"
            
            # PE Ratio analysis
            pe = info_data.get("trailingPE")
            if pe:
                if pe < 15:
                    analysis += f"- PE Ratio of {pe:.2f} is relatively low, potentially indicating undervaluation.\n"
                elif pe > 30:
                    analysis += f"- PE Ratio of {pe:.2f} is relatively high, potentially indicating overvaluation.\n"
                else:
                    analysis += f"- PE Ratio of {pe:.2f} is moderate.\n"
            
            # Dividend analysis
            div_yield = info_data.get("dividendYield")
            if div_yield:
                if div_yield > 0.04:  # 4%
                    analysis += f"- Dividend yield of {div_yield*100:.2f}% is relatively high.\n"
                elif div_yield > 0:
                    analysis += f"- Dividend yield of {div_yield*100:.2f}%.\n"
                else:
                    analysis += "- No dividend is paid by this company.\n"
            
            # Return combined data
            return json.dumps({
                "metrics": fundamentals,
                "analysis": analysis
            })
        except yf.exceptions.YFRateLimitError as e:
            return f"Rate limit error retrieving fundamental analysis for {ticker}. Please try again later."
        except Exception as e:
            return f"Error retrieving fundamental analysis for {ticker}: {str(e)}"
    
    def _get_technical_analysis(self, ticker: str):
        """Get technical indicators"""
        try:
            stock = get_stock_ticker_safe(ticker)
            history = get_stock_history_safe(stock, period="1y")
            
            if history.empty:
                return "No historical data available for this ticker."
            
            # Calculate indicators
            history["SMA_50"] = history["Close"].rolling(window=50).mean()
            history["SMA_200"] = history["Close"].rolling(window=200).mean()
            
            # RSI
            delta = history["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            history["RSI"] = 100 - (100 / (1 + rs))
            
            # MACD
            short_ema = history["Close"].ewm(span=12, adjust=False).mean()
            long_ema = history["Close"].ewm(span=26, adjust=False).mean()
            history["MACD"] = short_ema - long_ema
            history["Signal_Line"] = history["MACD"].ewm(span=9, adjust=False).mean()
            
            # Fill NaN values
            history.fillna(0, inplace=True)
            
            # Get the latest values
            latest = history.iloc[-1].to_dict()
            
            # Technical signals
            signals = {
                "Current Price": latest["Close"],
                "SMA_50": latest["SMA_50"],
                "SMA_200": latest["SMA_200"],
                "RSI": latest["RSI"],
                "MACD": latest["MACD"],
                "Signal_Line": latest["Signal_Line"],
            }
            
            # Generate analysis
            analysis = "Technical Analysis:\n"
            
            # Trend analysis based on MAs
            if latest["SMA_50"] > latest["SMA_200"]:
                analysis += "- BULLISH TREND: The 50-day moving average is above the 200-day moving average.\n"
            else:
                analysis += "- BEARISH TREND: The 50-day moving average is below the 200-day moving average.\n"
            
            # Price vs MA
            if latest["Close"] > latest["SMA_50"]:
                analysis += "- BULLISH: Price is above the 50-day moving average.\n"
            else:
                analysis += "- BEARISH: Price is below the 50-day moving average.\n"
            
            # RSI analysis
            if latest["RSI"] > 70:
                analysis += "- OVERBOUGHT: RSI is above 70, indicating the stock may be overbought.\n"
            elif latest["RSI"] < 30:
                analysis += "- OVERSOLD: RSI is below 30, indicating the stock may be oversold.\n"
            else:
                analysis += f"- NEUTRAL: RSI is at {latest['RSI']:.2f}, indicating neutral momentum.\n"
            
            # MACD analysis
            if latest["MACD"] > latest["Signal_Line"]:
                analysis += "- BULLISH SIGNAL: MACD is above the signal line, indicating bullish momentum.\n"
            else:
                analysis += "- BEARISH SIGNAL: MACD is below the signal line, indicating bearish momentum.\n"
            
            return json.dumps({
                "indicators": signals,
                "analysis": analysis
            })
        except yf.exceptions.YFRateLimitError as e:
            return f"Rate limit error retrieving technical analysis for {ticker}. Please try again later."
        except Exception as e:
            return f"Error retrieving technical analysis for {ticker}: {str(e)}"
    
    def _rag_query(self, query_data: str):
        """Query the RAG system"""
        try:
            # Parse the query data
            try:
                data = json.loads(query_data)
                question = data.get("question", "")
                ticker = data.get("ticker", None)
            except:
                # If not valid JSON, treat the input as the question
                question = query_data
                ticker = None
                
                # Try to extract ticker from the question
                import re
                ticker_match = re.search(r'\b[A-Z]{1,5}\b', question)
                if ticker_match:
                    ticker = ticker_match.group(0)
            
            # Query the RAG system
            result = self.rag_engine.query(question, ticker)
            return json.dumps(result)
        except Exception as e:
            return f"Error querying RAG system: {str(e)}"
    
    def _optimize_portfolio(self, portfolio_request: str):
        """Optimize a portfolio"""
        try:
            # Parse the portfolio request
            data = json.loads(portfolio_request)
            tickers = data.get("tickers", [])
            risk_preference = data.get("risk_preference", "medium")
            
            if not tickers:
                return "No tickers provided for portfolio optimization."
            
            # Collect historical data
            stock_data = {}
            for ticker in tickers:
                try:
                    stock = get_stock_ticker_safe(ticker)
                    history = get_stock_history_safe(stock, period="1y")
                    info_data = get_stock_info_safe(stock)
                    if not history.empty:
                        # Calculate returns and risk metrics
                        returns = history["Close"].pct_change().dropna()
                        stock_data[ticker] = {
                            "avg_return": returns.mean() * 252 * 100,  # Annualized return %
                            "volatility": returns.std() * np.sqrt(252) * 100,  # Annualized volatility %
                            "beta": info_data.get("beta", 1.0),
                            "current_price": history["Close"].iloc[-1]
                        }
                except yf.exceptions.YFRateLimitError as e:
                    print(f"Rate limit error for {ticker} in portfolio optimization")
                    # Skip this ticker or use default values
                    stock_data[ticker] = {
                        "avg_return": 10.0,
                        "volatility": 20.0,
                        "beta": 1.0,
                        "current_price": 0.0
                    }
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {e}")
                    stock_data[ticker] = {
                        "avg_return": 10.0,
                        "volatility": 20.0,
                        "beta": 1.0,
                        "current_price": 0.0
                    }
            
            # Simple portfolio optimization based on risk preference
            optimized_allocations = {}
            
            if risk_preference == "low":
                # For low risk: favor low volatility and beta stocks
                weights = {}
                for ticker, data in stock_data.items():
                    # Inverse of volatility * inverse of beta
                    weight = 1 / (data["volatility"] * data["beta"] if data["beta"] > 0 else 1)
                    weights[ticker] = weight
            
            elif risk_preference == "high":
                # For high risk: favor high return stocks
                weights = {}
                for ticker, data in stock_data.items():
                    # Return to volatility ratio (Sharpe-like without risk-free rate)
                    weight = max(data["avg_return"], 1) / data["volatility"] if data["volatility"] > 0 else 1
                    weights[ticker] = weight
            
            else:  # medium risk
                # For medium risk: balanced approach
                weights = {}
                for ticker, data in stock_data.items():
                    # Balance of return and risk
                    weight = (data["avg_return"] + 5) / (data["volatility"] + 5)  # Adding constants to avoid division by zero
                    weights[ticker] = weight
            
            # Normalize weights to percentages
            total_weight = sum(weights.values())
            for ticker, weight in weights.items():
                optimized_allocations[ticker] = round((weight / total_weight) * 100, 2)
            
            # Sort by allocation (highest first)
            optimized_allocations = dict(sorted(optimized_allocations.items(), key=lambda x: x[1], reverse=True))
            
            # Prepare analysis
            analysis = f"Portfolio Optimization Analysis ({risk_preference.upper()} risk profile):\n\n"
            
            # Portfolio characteristics
            total_return = sum(stock_data[ticker]["avg_return"] * (optimized_allocations[ticker]/100) for ticker in optimized_allocations)
            
            # Calculate weighted average volatility (simplified without covariance)
            portfolio_volatility = sum(stock_data[ticker]["volatility"] * (optimized_allocations[ticker]/100) for ticker in optimized_allocations)
            
            analysis += f"Expected Annual Return: {total_return:.2f}%\n"
            analysis += f"Portfolio Volatility: {portfolio_volatility:.2f}%\n\n"
            
            # Add rationale for each allocation
            analysis += "Allocation Rationale:\n"
            for ticker, allocation in optimized_allocations.items():
                if risk_preference == "low":
                    analysis += f"- {ticker}: {allocation:.2f}% - Selected for its lower volatility ({stock_data[ticker]['volatility']:.2f}%) and beta ({stock_data[ticker]['beta']:.2f}).\n"
                elif risk_preference == "high":
                    analysis += f"- {ticker}: {allocation:.2f}% - Selected for its higher potential return ({stock_data[ticker]['avg_return']:.2f}%).\n"
                else:
                    analysis += f"- {ticker}: {allocation:.2f}% - Balanced allocation based on return ({stock_data[ticker]['avg_return']:.2f}%) and risk ({stock_data[ticker]['volatility']:.2f}%).\n"
            
            return json.dumps({
                "optimized_portfolio": optimized_allocations,
                "analysis": analysis,
                "metrics": {ticker: data for ticker, data in stock_data.items()}
            })
            
        except Exception as e:
            return f"Error optimizing portfolio: {str(e)}"
    
    def run(self, query: str):
        """Run the agent with a user query"""
        return self.agent.run(query)