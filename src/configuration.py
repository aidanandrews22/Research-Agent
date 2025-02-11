import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Literal

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
from dataclasses import dataclass

DEFAULT_REPORT_STRUCTURE = """The report structure should focus on breaking-down the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2-5. At Least 3 Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   - Include any key concepts and definitions
   - Provide real-world examples or case studies where applicable
   
6. Conclusion
   - Aim for 1 structural element (either a list or table) that distills the main body sections 
   - Provide a concise summary of the report"""

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    # Report settings
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    number_of_queries: int = 8  # Increased for better initial research coverage
    section_queries: int = 5  # Increased per-section queries for deeper research
    tavily_topic: str = "general"
    tavily_days: str = None
    
    # Model settings
    planner_model_type: Literal[1, 2] = 1  # 1 for local LLM, 2 for gpt-4o
    planner_model: str = "gpt-4o"  # This will be set based on planner_model_type
    writer_model: str = "claude-3-5-sonnet-latest"
    
    # Search settings
    max_results_per_source: int = 15  # Focused on quality over quantity
    min_relevance_score: float = 80.0  # Higher relevance threshold
    max_concurrent_fetches: int = 5  # Balanced for reliability
    fetch_timeout: int = 30  # Standard timeout
    fetch_retries: int = 3  # Standard retries
    
    # Context settings
    max_context_length: int = 4000  # Standard context length
    min_context_overlap: int = 200  # Reduced overlap - wasn't providing significant benefit

    def __post_init__(self):
        # Set the planner model based on the model type
        if self.planner_model_type == 1:
            self.planner_model = "local"
        else:
            self.planner_model = "gpt-4o"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})