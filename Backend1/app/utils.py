"""Shared helpers for the StockIntel backend."""

import re

# Common English / finance acronyms that are not stock tickers
_TICKER_BLOCKLIST = frozenset({
    "A", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "IT", "ME", "MY",
    "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "ALL", "AND", "ARE", "BUT", "CAN", "FOR", "GET", "HAD", "HAS", "HER", "HIM", "HIS",
    "HOW", "ITS", "MAY", "NEW", "NOT", "NOW", "OLD", "ONE", "OUR", "OUT", "OWN", "SAY",
    "SHE", "THE", "TOO", "TWO", "USE", "WAY", "WHO", "WHY", "YES", "YOU", "ANY", "ASK",
    "BEST", "BUY", "CEO", "ETF", "IPO", "LLC", "NYSE", "SEC", "YOY", "QOQ", "EPS", "ROE",
    "ROI", "API", "RAG", "AI", "PE", "PB", "PS", "EV", "GDP", "CPI", "FED", "USD", "EUR",
    "ATH", "ATL", "YTD", "MOM", "TTM", "FCF", "DEBT", "CASH", "NEWS", "WHAT", "WHEN",
    "WHERE", "WHICH", "THAN", "THAT", "THIS", "THEY", "THEM", "THEN", "ALSO", "JUST",
    "LIKE", "MAKE", "MADE", "MORE", "MOST", "MUCH", "MANY", "OVER", "INTO", "ONLY",
    "SOME", "SUCH", "TAKE", "TELL", "WELL", "WERE", "WORK", "YEAR", "YOUR", "ABOUT",
    "AFTER", "BEEN", "BEING", "COULD", "EVERY", "FIRST", "GOOD", "GREAT", "HERE",
    "HIGH", "HOLD", "KNOW", "LAST", "LONG", "LOOK", "NEED", "NEXT", "RISK", "RISKS",
    "SHOW", "STOCK", "STOCKS", "THINK", "THOSE", "UNDER", "VERY", "WANT", "WITH",
    "WOULD", "PRICE", "SHARE", "MARKET", "VALUE", "GROWTH", "MIGHT", "SHOULD",
    "INVEST", "MONEY", "FUNDS", "RATE", "RATES", "GAIN", "LOSS", "BULL", "BEAR",
})


def extract_tickers_from_query(query: str) -> list[str]:
    """Extract likely ticker symbols, ignoring common English words."""
    upper = query.upper()
    explicit = re.findall(r"\$([A-Z]{1,5})\b", upper)
    candidates = re.findall(r"\b[A-Z]{2,5}\b", upper)

    tickers: list[str] = []
    seen: set[str] = set()
    for symbol in explicit + candidates:
        if symbol not in _TICKER_BLOCKLIST and symbol not in seen:
            seen.add(symbol)
            tickers.append(symbol)
    return tickers
