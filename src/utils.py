import asyncio

from tavily import TavilyClient, AsyncTavilyClient
from state import Section, SearchQuery
from langsmith import traceable

tavily_client = TavilyClient()
tavily_async_client = AsyncTavilyClient()

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.
    
    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
            
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str

@traceable
def tavily_search(query: SearchQuery | str):
    """ Search the web using the Tavily API.
    
    Args:
        query (Union[SearchQuery, str]): The search query to execute
        
    Returns:
        dict: Tavily search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""
     
    if isinstance(query, SearchQuery):
        query = query.search_query
        
    return tavily_client.search(query, 
                         max_results=5, 
                         include_raw_content=True)

@traceable
async def tavily_search_async(search_queries: list[SearchQuery | str], tavily_topic: str, tavily_days: int | None):
    """
    Performs concurrent web searches using the Tavily API.

    Args:
        search_queries (List[Union[SearchQuery, str]]): List of search queries to process
        tavily_topic (str): Type of search to perform ('news' or 'general')
        tavily_days (Optional[int]): Number of days to look back for news articles (only used when tavily_topic='news')

    Returns:
        List[dict]: List of search results from Tavily API, one per query
    """
    if not search_queries:
        return []
        
    async with AsyncTavilyClient() as client:
        search_tasks = []
        for query in search_queries:
            if isinstance(query, SearchQuery):
                query = query.search_query
                
            if tavily_topic == "news":
                search_tasks.append(
                    client.search(
                        query,
                        max_results=5,
                        include_raw_content=True,
                        search_depth="advanced",
                        topic="news",
                        days=tavily_days
                    )
                )
            else:
                search_tasks.append(
                    client.search(
                        query,
                        max_results=5,
                        include_raw_content=True,
                        search_depth="advanced",
                        topic="general"
                    )
                )

        # Execute all searches concurrently
        try:
            search_docs = await asyncio.gather(*search_tasks, return_exceptions=True)
            # Filter out any failed searches
            search_docs = [doc for doc in search_docs if not isinstance(doc, Exception)]
            return search_docs
        except Exception as e:
            print(f"Error during Tavily search: {str(e)}")
            return []
