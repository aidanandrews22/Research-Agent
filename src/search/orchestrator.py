import logging
import asyncio
from typing import List, Optional
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel

from .models import SearchResult, FetchedContent
from .duckduckgo import DuckDuckGoSearch
from .deduplicator import SearchDeduplicator
from .ranker import ResultRanker
from .content_fetcher import ContentFetcher

logger = logging.getLogger(__name__)

@dataclass
class SearchConfig:
    """Configuration for the search orchestrator."""
    max_results_per_source: int = 20  # Reduced from 50 to get faster initial results
    min_relevance_score: float = 50.0  # Lowered threshold to reduce processing
    max_concurrent_fetches: int = 10  # Increased from 5 for faster content fetching
    fetch_timeout: int = 15  # Reduced timeout for faster failure detection
    fetch_retries: int = 2  # Reduced retries to speed up process
    ranking_batch_size: int = 10  # Increased from 5 for faster ranking
    max_ranked_results: int = 5  # Reduced from 10 to minimize content fetching

class SearchOrchestrator:
    """Coordinates the entire search process across multiple providers."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        tavily_client=None,  # Optional Tavily client
        config: Optional[SearchConfig] = None
    ):
        """
        Initialize the search orchestrator.
        
        Args:
            llm: LangChain chat model for ranking
            tavily_client: Optional Tavily search client
            config: Search configuration
        """
        self.config = config or SearchConfig()
        self.tavily = None  # Disabled Tavily by default for speed
        
        # Initialize components
        self.ddg = DuckDuckGoSearch(max_results=self.config.max_results_per_source)
        self.deduper = SearchDeduplicator()
        self.ranker = ResultRanker(
            llm,
            batch_size=self.config.ranking_batch_size
        )
        self.fetcher = ContentFetcher(
            max_retries=self.config.fetch_retries,
            timeout=self.config.fetch_timeout
        )
    
    async def _get_tavily_results(self, query: str) -> List[SearchResult]:
        """Get search results from Tavily if available."""
        if not self.tavily:
            return []
            
        try:
            results = await self.tavily.search(query)
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    source="Tavily"
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error getting Tavily results: {str(e)}")
            return []
    
    async def search(
        self,
        query: str,
        num_results: int
    ) -> List[SearchResult]:
        """
        Perform a search across all providers and rank results.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of ranked SearchResult objects
        """
        try:
            # Get results from all sources
            ddg_results = await self.ddg.search(query)
            tavily_results = await self._get_tavily_results(query)
            
            # Combine and deduplicate results
            all_results = self.deduper.deduplicate(ddg_results + tavily_results)
            
            # Rank results with batch processing and max results limit
            ranked_results = await self.ranker.rank_results(
                results=all_results,
                query=query,
                min_score=self.config.min_relevance_score,
                max_results=min(num_results, self.config.max_ranked_results)
            )
            
            # Log detailed ranking information
            logger.info("Top ranked results:")
            for i, result in enumerate(ranked_results, 1):
                logger.info(f"{i}. [{result.source}] {result.title} - Score: {result.relevance_score:.2f}")
                logger.info(f"   URL: {result.url}")
                logger.info(f"   Rationale: {result.ranking_rationale}")
            
            logger.info(
                f"Search complete - DDG: {len(ddg_results)}, "
                f"Tavily: {len(tavily_results)}, "
                f"Unique: {len(all_results)}, "
                f"Final Ranked: {len(ranked_results)}"
            )
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Error in search process: {str(e)}")
            return []
    
    async def search_and_fetch(
        self,
        query: str,
        num_results: int
    ) -> List[FetchedContent]:
        """
        Perform a search and fetch content for top results.
        
        Args:
            query: Search query string
            num_results: Number of results to fetch content for
            
        Returns:
            List of FetchedContent objects
        """
        # Get ranked results
        results = await self.search(query, num_results)
        
        if not results:
            return []
        
        # Fetch content for top results
        return await self.fetcher.fetch_multiple(
            results=results,
            max_concurrent=self.config.max_concurrent_fetches
        )
    
    async def close(self):
        """Clean up resources."""
        await self.ddg.close()
        await self.fetcher.close() 