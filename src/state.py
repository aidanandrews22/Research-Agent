from typing import Annotated, List, TypedDict
from pydantic import BaseModel, Field
from enum import Enum
import operator

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )   

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(
        description="Query for web search.",
    )
    explanation: str = Field(
        default="",
        description="Optional explanation of what this query aims to find."
    )

    def __str__(self) -> str:
        return self.search_query

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {"search_query": self.search_query}

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

    @classmethod
    def from_plain_queries(cls, queries: List[str]) -> "Queries":
        """Create a Queries instance from a list of plain string queries."""
        return cls(queries=[SearchQuery(search_query=q) for q in queries])

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {"queries": [q.to_dict() for q in self.queries]}

class ReportStateInput(TypedDict):
    topic: str # Report topic
    feedback_on_report_plan: str # Feedback on the report structure from review
    accept_report_plan: bool  # Whether to accept or reject the report plan
    
class ReportStateOutput(TypedDict):
    final_report: str # Final report

class ReportState(TypedDict):
    topic: str # Report topic    
    feedback_on_report_plan: str # Feedback on the report structure from review
    accept_report_plan: bool  # Whether to accept or reject the report plan
    sections: list[Section] # List of report sections 
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report

class SectionState(TypedDict):
    section: Section # Report section   
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
