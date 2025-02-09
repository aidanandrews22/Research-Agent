from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchResult:
    """Represents a single search result from any search provider."""
    title: str
    url: str
    snippet: str
    source: str  # e.g. "DuckDuckGo" or "Tavily"
    relevance_score: Optional[float] = None
    ranking_rationale: Optional[str] = None

@dataclass
class FetchedContent:
    """Represents the fetched content from a search result URL."""
    url: str
    content: str
    title: str
    success: bool = True
    error: Optional[str] = None 