import logging
from typing import List, Set
from urllib.parse import urlparse
from .models import SearchResult

logger = logging.getLogger(__name__)

class SearchDeduplicator:
    """Handles deduplication of search results from multiple sources."""
    
    def __init__(self):
        """Initialize the deduplicator with empty sets for tracking seen items."""
        self.seen_urls: Set[str] = set()
        self.seen_normalized_urls: Set[str] = set()
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize a URL by removing common variations that point to the same content.
        
        Args:
            url: The URL to normalize
            
        Returns:
            Normalized URL string
        """
        try:
            # Parse the URL
            parsed = urlparse(url)
            
            # Remove www. prefix if present
            hostname = parsed.netloc
            if hostname.startswith('www.'):
                hostname = hostname[4:]
            
            # Reconstruct the normalized URL
            normalized = f"{parsed.scheme}://{hostname}{parsed.path}"
            
            # Remove trailing slash if present
            if normalized.endswith('/'):
                normalized = normalized[:-1]
                
            return normalized.lower()
            
        except Exception as e:
            logger.warning(f"Error normalizing URL {url}: {str(e)}")
            return url.lower()
    
    def is_duplicate(self, result: SearchResult) -> bool:
        """
        Check if a search result is a duplicate based on URL.
        
        Args:
            result: SearchResult to check
            
        Returns:
            True if duplicate, False otherwise
        """
        if not result.url:
            return True
            
        # Check exact URL match
        if result.url in self.seen_urls:
            return True
            
        # Check normalized URL match
        normalized_url = self._normalize_url(result.url)
        if normalized_url in self.seen_normalized_urls:
            return True
            
        return False
    
    def deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results from a list of search results.
        
        Args:
            results: List of SearchResult objects to deduplicate
            
        Returns:
            Deduplicated list of SearchResult objects
        """
        unique_results = []
        duplicates = 0
        
        for result in results:
            if not self.is_duplicate(result):
                unique_results.append(result)
                self.seen_urls.add(result.url)
                self.seen_normalized_urls.add(self._normalize_url(result.url))
            else:
                duplicates += 1
        
        logger.info(f"Deduplication removed {duplicates} duplicate results")
        return unique_results
    
    def reset(self):
        """Reset the deduplicator's state."""
        self.seen_urls.clear()
        self.seen_normalized_urls.clear() 