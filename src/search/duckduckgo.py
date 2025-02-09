import logging
import asyncio
from typing import List
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from .models import SearchResult

logger = logging.getLogger(__name__)

class DuckDuckGoSearch:
    """DuckDuckGo search implementation using LangChain's DuckDuckGoSearchAPIWrapper."""
    
    def __init__(self, max_results: int = 50):
        """Initialize DuckDuckGo search with max results limit."""
        self.max_results = max_results
        self.wrapper = DuckDuckGoSearchAPIWrapper(
            max_results=max_results,
            time="y",  # All time results
            region="wt-wt",  # Worldwide
            backend="api"  # Use the API backend which is more stable
        )
    
    async def search(self, query: str, region: str = "wt-wt", safesearch: str = "moderate") -> List[SearchResult]:
        """
        Perform an async search using DuckDuckGo.
        
        Args:
            query: The search query string
            region: Region for search results (default: worldwide)
            safesearch: SafeSearch setting ("on", "moderate", or "off")
            
        Returns:
            List of SearchResult objects
        """
        max_retries = 3
        retry_delay = 0  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Performing DuckDuckGo search for query: {query} (attempt {attempt + 1}/{max_retries})")
                
                # Use LangChain's wrapper to perform the search
                results = self.wrapper.results(query, max_results=self.max_results)
                
                # Convert results to our standard format
                processed_results = []
                for result in results:
                    processed_results.append(SearchResult(
                        title=result.get("title", ""),
                        url=result.get("link", ""),
                        snippet=result.get("snippet", ""),
                        source="DuckDuckGo"
                    ))
                
                logger.info(f"DuckDuckGo search returned {len(processed_results)} results")
                return processed_results
                
            except Exception as e:
                logger.error(f"Error performing DuckDuckGo search (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if "202 Ratelimit" in str(e) and attempt < max_retries - 1:
                    logger.info(f"Rate limit hit, waiting {retry_delay} seconds before retry...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                if attempt == max_retries - 1:
                    return []
        
        return []
    
    async def close(self):
        """Close the DuckDuckGo search session."""
        self.wrapper = None 