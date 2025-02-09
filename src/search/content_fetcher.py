import logging
import aiohttp
import asyncio
from typing import List, Optional
from bs4 import BeautifulSoup
from .models import SearchResult, FetchedContent

logger = logging.getLogger(__name__)

class ContentFetcher:
    """Handles fetching and processing content from search result URLs."""
    
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        """
        Initialize the content fetcher.
        
        Args:
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def _extract_text_content(self, html: str) -> tuple[str, str]:
        """
        Extract readable text content from HTML.
        
        Args:
            html: Raw HTML content
            
        Returns:
            Tuple of (title, main content text)
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get title
            title = ""
            if soup.title:
                title = soup.title.string
            
            # Get main content
            text = soup.get_text()
            
            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # Drop blank lines
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            return title, content
            
        except Exception as e:
            logger.error(f"Error extracting text content: {str(e)}")
            return "", ""
    
    async def fetch_content(self, result: SearchResult) -> FetchedContent:
        """
        Fetch and process content from a search result URL.
        
        Args:
            result: SearchResult to fetch content for
            
        Returns:
            FetchedContent object containing the processed content
        """
        await self._ensure_session()
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(
                    result.url,
                    timeout=self.timeout,
                    headers={'User-Agent': 'Mozilla/5.0'}
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        title, content = await self._extract_text_content(html)
                        
                        return FetchedContent(
                            url=result.url,
                            content=content,
                            title=title or result.title
                        )
                    else:
                        logger.warning(
                            f"Failed to fetch content from {result.url} "
                            f"(status: {response.status})"
                        )
                        
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout fetching content from {result.url} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
            except Exception as e:
                logger.error(
                    f"Error fetching content from {result.url}: {str(e)} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return FetchedContent(
            url=result.url,
            content="",
            title=result.title,
            success=False,
            error=f"Failed to fetch content after {self.max_retries} attempts"
        )
    
    async def fetch_multiple(
        self,
        results: List[SearchResult],
        max_concurrent: int = 5
    ) -> List[FetchedContent]:
        """
        Fetch content for multiple search results concurrently.
        
        Args:
            results: List of SearchResult objects to fetch content for
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of FetchedContent objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(result: SearchResult) -> FetchedContent:
            async with semaphore:
                return await self.fetch_content(result)
        
        tasks = [fetch_with_semaphore(result) for result in results]
        return await asyncio.gather(*tasks)
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None 