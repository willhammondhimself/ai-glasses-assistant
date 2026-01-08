"""Stocks tool using yfinance (free, no API key required)."""
import logging
import re
from typing import Optional, Tuple
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)

# Common stock name to ticker mappings
STOCK_ALIASES = {
    "apple": "AAPL",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "disney": "DIS",
    "coca cola": "KO",
    "coke": "KO",
    "pepsi": "PEP",
    "walmart": "WMT",
    "target": "TGT",
    "nike": "NKE",
    "intel": "INTC",
    "amd": "AMD",
    "spotify": "SPOT",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "coinbase": "COIN",
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "spy": "SPY",
    "s&p": "SPY",
    "s and p": "SPY",
    "dow": "DIA",
    "nasdaq": "QQQ",
}


class StocksTool(VoiceTool):
    """Get stock quotes and market data using yfinance."""

    name = "stocks"
    description = "Get stock prices and market information"

    keywords = [
        r"\bstock\s+price\b",
        r"\bstock\s+quote\b",
        r"\bhow\s+is\s+\w+\s+(doing|trading)\b",
        r"\bwhat('s| is)\s+\w+\s+(at|trading|worth)\b",
        r"\bprice\s+of\s+\w+\b",
        r"\b(AAPL|GOOGL|MSFT|TSLA|AMZN|NVDA|META)\b",  # Common tickers
        r"\bmarket\b",
        r"\bcrypto\b",
        r"\bbitcoin\b",
        r"\bethereum\b",
    ]

    priority = 10

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Get stock information.

        Args:
            query: The user's voice query

        Returns:
            VoiceToolResult with stock data
        """
        ticker, company_name = self._extract_ticker(query)

        if not ticker:
            return VoiceToolResult(
                success=False,
                message="I couldn't identify which stock you're asking about. Try saying the company name or ticker symbol."
            )

        try:
            result = await self._get_stock_quote(ticker, company_name)
            return VoiceToolResult(
                success=True,
                message=result,
                data={"ticker": ticker, "company": company_name}
            )
        except Exception as e:
            logger.error(f"Stock lookup failed for {ticker}: {e}")
            return VoiceToolResult(
                success=False,
                message=f"Sorry, I couldn't get the stock data for {company_name or ticker}."
            )

    def _extract_ticker(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract stock ticker from query.

        Returns:
            Tuple of (ticker, company_name) or (None, None)
        """
        query_lower = query.lower()

        # Check for explicit ticker symbols (uppercase 1-5 letters)
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', query)
        if ticker_match:
            ticker = ticker_match.group(1)
            if ticker in ["I", "A", "THE", "IS", "AT", "OF", "FOR"]:
                pass  # Skip common words
            else:
                return ticker, None

        # Check for company names
        for name, ticker in STOCK_ALIASES.items():
            if name in query_lower:
                return ticker, name.title()

        # Try to extract from patterns like "stock price of X" or "how is X doing"
        patterns = [
            r"stock\s+(?:price|quote)\s+(?:of|for)\s+([a-zA-Z\s]+)",
            r"how\s+is\s+([a-zA-Z]+)\s+(?:doing|trading)",
            r"what('s| is)\s+([a-zA-Z]+)\s+(?:at|trading|worth)",
            r"price\s+of\s+([a-zA-Z]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip().lower()
                if potential_name in STOCK_ALIASES:
                    return STOCK_ALIASES[potential_name], potential_name.title()
                # Assume it's a ticker
                return potential_name.upper(), None

        return None, None

    async def _get_stock_quote(self, ticker: str, company_name: Optional[str]) -> str:
        """Get stock quote using yfinance.

        Args:
            ticker: Stock ticker symbol
            company_name: Optional company name for response

        Returns:
            Voice-friendly stock summary
        """
        try:
            import yfinance as yf
        except ImportError:
            return "Stock data is not available. The yfinance package is not installed."

        stock = yf.Ticker(ticker)

        # Get current data
        info = stock.info

        # Handle different data availability
        current_price = info.get("regularMarketPrice") or info.get("currentPrice")
        previous_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
        company = company_name or info.get("shortName") or ticker

        if not current_price:
            # Try getting from history
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist["Close"].iloc[-1]

        if not current_price:
            return f"I couldn't find current price data for {company}."

        # Calculate change
        if previous_close:
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100
            direction = "up" if change >= 0 else "down"
            change_str = f", {direction} {abs(change_pct):.1f}% today"
        else:
            change_str = ""

        # Format price
        if current_price >= 1:
            price_str = f"${current_price:,.2f}"
        else:
            price_str = f"${current_price:.4f}"  # For penny stocks/crypto

        return f"{company} is trading at {price_str}{change_str}."
