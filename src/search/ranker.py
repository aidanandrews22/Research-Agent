import logging
import json
from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .models import SearchResult

logger = logging.getLogger(__name__)

RANKING_PROMPT = """Rank these search results by relevance to the query. Focus on direct relevance and information quality.

Query: {query}

Results:
{results}

Return JSON array:
{{
    "ranked_results": [
        {{
            "url": "URL",
            "relevance_score": 0-100,
            "rationale": "brief reason"
        }}
    ]
}}"""

class ResultRanker:
    """Ranks search results using an LLM."""
    
    def __init__(self, llm: BaseChatModel, batch_size: int = 10):
        """
        Initialize the result ranker.
        
        Args:
            llm: LangChain chat model to use for ranking
            batch_size: Number of results to rank in each batch
        """
        self.llm = llm
        self.batch_size = batch_size
        self.prompt = ChatPromptTemplate.from_template(RANKING_PROMPT)
        self.parser = JsonOutputParser()
    
    def _format_results_for_prompt(self, results: List[SearchResult]) -> str:
        """Format search results for the ranking prompt."""
        return "\n".join(
            f"{i}. {r.title} - {r.url} - {r.snippet[:150]}..."
            for i, r in enumerate(results, 1)
        )

    async def _rank_batch(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rank a single batch of results."""
        if not results:
            return []
            
        try:
            formatted_results = self._format_results_for_prompt(results)
            chain = self.prompt | self.llm | self.parser
            
            response = await chain.ainvoke({
                "query": query,
                "results": formatted_results
            })
            
            result_map = {r.url: r for r in results}
            ranked_results = []
            
            for ranking in response.get("ranked_results", []):
                url = ranking.get("url")
                if url in result_map:
                    result = result_map[url]
                    score = float(ranking.get("relevance_score", 0))
                    result.relevance_score = score
                    result.ranking_rationale = ranking.get("rationale")
                    ranked_results.append(result)
            
            ranked_results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            return ranked_results
            
        except Exception as e:
            logger.error(f"Error ranking batch: {str(e)}")
            return sorted(results, key=lambda x: len(x.snippet), reverse=True)  # Fallback to snippet length
    
    async def rank_results(
        self,
        results: List[SearchResult],
        query: str,
        min_score: float = 0.0,
        max_results: int = 5  # Default reduced to 5 for speed
    ) -> List[SearchResult]:
        """
        Rank search results using the LLM.
        
        Args:
            results: List of SearchResult objects to rank
            query: Original search query
            min_score: Minimum relevance score to include in results
            max_results: Maximum number of results to return after ranking
            
        Returns:
            List of ranked SearchResult objects
        """
        if not results:
            return []
            
        try:
            # Only process enough results to get max_results
            results_needed = min(len(results), max_results * 2)  # Process 2x max_results to ensure quality
            results_to_process = results[:results_needed]
            
            # Process results in batches
            all_ranked_results = []
            for i in range(0, len(results_to_process), self.batch_size):
                batch = results_to_process[i:i + self.batch_size]
                ranked_batch = await self._rank_batch(batch, query)
                all_ranked_results.extend(ranked_batch)
                
                # Early stopping if we have enough high-scoring results
                if len([r for r in all_ranked_results if r.relevance_score >= min_score]) >= max_results:
                    break
            
            # Sort and filter results
            all_ranked_results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            filtered_results = [r for r in all_ranked_results if r.relevance_score >= min_score]
            
            # Return top results
            final_results = filtered_results[:max_results]
            
            logger.info(
                f"Ranked {len(all_ranked_results)} results in batches of {self.batch_size}. "
                f"Found {len(filtered_results)} results above score {min_score}. "
                f"Returning top {len(final_results)} results."
            )
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error ranking results: {str(e)}")
            # On error, return top results based on snippet length
            return sorted(results[:max_results], key=lambda x: len(x.snippet), reverse=True) 