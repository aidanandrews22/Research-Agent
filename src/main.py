import asyncio
from pathlib import Path
from typing import Optional, List
import sys
import os
import re
import json
import logging
from datetime import datetime
import uuid
import dotenv

dotenv.load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from configuration import Configuration
from local_llm import LocalDeepSeekLLM
from state import (
    ReportState,
    Section,
    SearchQuery,
    Sections,
    Queries
)
from search import (
    SearchOrchestrator,
    SearchConfig,
    FetchedContent
)
from utils import (
    tavily_search_async,
    deduplicate_and_format_sources,
    format_sections
)
from prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions,
    section_writer_instructions,
    final_section_writer_instructions
)

class ReportGenerator:
    def __init__(self, config: Optional[Configuration] = None):
        self.config = config or Configuration()
        
        # Initialize models
        if self.config.planner_model_type == 1:
            self.planner = LocalDeepSeekLLM()
        else:
            self.planner = ChatOpenAI(model=self.config.planner_model, temperature=0)
        self.writer = ChatAnthropic(model=self.config.writer_model, temperature=0)
        
        # Initialize search
        self.search = SearchOrchestrator(
            llm=self.writer,  # Use the writer model for ranking
            tavily_client=tavily_search_async,  # Pass existing Tavily client
            config=SearchConfig(
                max_results_per_source=self.config.max_results_per_source,
                min_relevance_score=self.config.min_relevance_score,
                max_concurrent_fetches=self.config.max_concurrent_fetches,
                fetch_timeout=self.config.fetch_timeout,
                fetch_retries=self.config.fetch_retries
            )
        )
        
        # Initialize logging
        self.run_id = str(uuid.uuid4())
        self.log_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "raw_responses": [],
            "search_results": []  # Add search results tracking
        }
        
    def _log_llm_response(self, stage: str, prompt: dict, response: str):
        """Log a raw LLM response with metadata."""
        self.log_data["raw_responses"].append({
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "raw_response": response
        })
        
        # Write to file after each response
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"../logs/report_gen_{datetime.now().strftime('%Y%m%d')}_{self.run_id}.json"
        with open(log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        
    def _extract_queries_from_text(self, text: str) -> list[str]:
        """Extract queries from the planner's text output."""
        # First try to parse as JSON
        try:
            # Try to find a JSON object in the text
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                data = json.loads(json_str)
                if isinstance(data, dict) and "queries" in data:
                    queries = []
                    for q in data["queries"]:
                        if isinstance(q, dict) and "search_query" in q:
                            query = q["search_query"].strip()
                            if query and not query.startswith(("{", "[")):
                                queries.append(query)
                        elif isinstance(q, str):
                            query = q.strip()
                            if query and not query.startswith(("{", "[")):
                                queries.append(query)
                    if queries:
                        return queries[:self.config.number_of_queries]
        except:
            pass

        # Try to find quoted queries
        quoted_queries = []
        for line in text.split('\n'):
            # Look for text in quotes
            matches = re.findall(r'"([^"]*)"', line)
            for match in matches:
                if match and not match.startswith(("{", "[")):
                    quoted_queries.append(match)
            if not matches:
                # Also look for text after numbers or bullet points
                match = re.search(r'(?:\d+\.|\-)\s*(.+)', line)
                if match:
                    query = match.group(1).strip(' "')
                    if query and not query.startswith(("{", "[")):
                        quoted_queries.append(query)
        
        return quoted_queries[:self.config.number_of_queries]
    
    def _get_response_content(self, response) -> str:
        """Extract content from an LLM response."""
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    
    def _log_search_results(self, query: str, results: List[FetchedContent], stage: str):
        """Log search results with metadata."""
        self.log_data["search_results"].append({
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.content[:1000] if r.content else None,  # Truncate long content
                    "success": r.success,
                    "error": str(r.error) if r.error else None
                }
                for r in results
            ]
        })
        
        # Write to file after each batch of results
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"../logs/report_gen_{datetime.now().strftime('%Y%m%d')}_{self.run_id}.json"
        with open(log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    async def generate_report(self, topic: str, output_path: Optional[str] = None) -> str:
        """
        Generate a report on the given topic.
        
        Args:
            topic: The topic to generate a report about
            output_path: Optional path to save the report
            
        Returns:
            The generated report text
        """
        # Initialize report state
        state = {
            "topic": topic,
            "feedback_on_report_plan": "",
            "accept_report_plan": True,
            "sections": [],
            "completed_sections": [],
            "report_sections_from_research": "",
            "final_report": ""
        }
        
        try:
            print("\n📝 Generating initial research queries...")
            # Step 1: Generate initial search queries
            queries = await self._generate_planner_queries(state)
            state["queries"] = queries
            print("Generated queries:")
            for i, query in enumerate(queries, 1):
                print(f"  {i}. {query.search_query}")
            
            print("\n🌐 Gathering initial research...")
            # Step 2: Perform searches and fetch content
            all_content = []
            for query in queries:
                results = await self.search.search_and_fetch(
                    query=query.search_query,
                    num_results=self.config.number_of_queries
                )
                self._log_search_results(query.search_query, results, "initial_research")
                all_content.extend(results)
            
            # Format search results for context
            context = self._format_search_results(all_content)
            
            print("\n📋 Generating report outline...")
            # Step 3: Generate report plan
            sections = await self._generate_report_plan(state, context)
            state["sections"] = sections
            
            print("\nProposed sections:")
            for i, section in enumerate(sections.sections, 1):
                print(f"\n  {i}. {section.name}")
            
            print("\n📚 Generating section content...")
            # Step 4: Generate section-specific queries and content
            for i, section in enumerate(sections.sections, 1):
                print(f"\nWorking on section {i}/{len(sections.sections)}: {section.name}")
                
                # Generate queries for this section
                print("  Generating search queries...")
                section.queries = await self._generate_section_queries(section)
                print("  Generated queries:")
                for j, query in enumerate(section.queries, 1):
                    print(f"    {j}. {query.search_query}")
                
                # Search and fetch content for section queries
                print("  Gathering research...")
                section_content = []
                for query in section.queries:
                    results = await self.search.search_and_fetch(
                        query=query.search_query,
                        num_results=self.config.number_of_queries
                    )
                    self._log_search_results(query.search_query, results, f"section_research_{section.name}")
                    section_content.extend(results)
                
                # Format section context
                section_context = self._format_search_results(section_content)
                
                # Generate section content
                print("  Writing content...")
                is_final = section == sections.sections[-1]
                section.content = await self._write_section_content(
                    section,
                    section_context,
                    is_final
                )
                print("  ✓ Section completed")
            
            print("\n📄 Combining sections into final report...")
            # Combine all sections into final report
            report = self._format_final_report(sections)
            
            # Save report if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(report)
                print(f"\n✅ Report saved to: {output_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
        
        finally:
            # Clean up resources
            await self.search.close()
    
    def _format_search_results(self, results: List[FetchedContent]) -> str:
        """Format fetched content into a context string."""
        formatted = []
        for i, result in enumerate(results, 1):
            if result.success and result.content:
                formatted.append(
                    f"Source {i}:\n"
                    f"Title: {result.title}\n"
                    f"URL: {result.url}\n"
                    f"Content:\n{result.content[:1000]}...\n"  # Truncate long content
                )
        return "\n\n".join(formatted)
    
    def _format_final_report(self, sections: Sections) -> str:
        """Format the final report from all sections."""
        logger.info(f"Formatting final report with {len(sections.sections)} sections")
        
        try:
            formatted_sections = []
            for section in sections.sections:
                logger.info(f"Processing section '{section.name}' - Content length: {len(section.content) if section.content else 0}")
                if not section.content:
                    logger.warning(f"Section '{section.name}' has no content")
                formatted_sections.append(f"## {section.name}\n\n{section.content}\n")
            
            return "\n".join(formatted_sections)
            
        except Exception as e:
            logger.error(f"Error formatting final report: {str(e)}")
            return "Error generating report"
    
    async def _generate_planner_queries(self, state: dict) -> list[SearchQuery]:
        """Generate initial search queries for the topic."""
        prompt = ChatPromptTemplate.from_template(report_planner_query_writer_instructions)
        
        response = await self.planner.ainvoke(
            prompt.format(
                topic=state["topic"],
                report_organization=self.config.report_structure,
                number_of_queries=self.config.number_of_queries
            )
        )
        
        content = self._get_response_content(response)
        self._log_llm_response("planner_queries", prompt.dict(), content)
        
        # Extract queries from response
        query_texts = self._extract_queries_from_text(content)
        return [SearchQuery(search_query=q) for q in query_texts]
    
    async def _generate_report_plan(self, state: dict, context: str) -> Sections:
        """Generate the report plan based on search results."""
        prompt = ChatPromptTemplate.from_template(report_planner_instructions)
        
        response = await self.planner.ainvoke(
            prompt.format(
                topic=state["topic"],
                report_organization=self.config.report_structure,
                context=context,
                feedback=state["feedback_on_report_plan"]
            )
        )
        
        content = self._get_response_content(response)
        self._log_llm_response("report_plan", prompt.dict(), content)
        
        try:
            # Try to parse JSON response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
                if isinstance(data, dict) and "sections" in data:
                    sections = []
                    for section_data in data["sections"]:
                        sections.append(Section(
                            name=section_data.get("name", "Untitled Section"),
                            description=section_data.get("description", ""),
                            research=section_data.get("research", True),
                            content=""
                        ))
                    return Sections(sections=sections)
            
            # Fallback to parsing text format if JSON fails
            sections = []
            current_section = None
            
            for line in content.split('\n'):
                if line.startswith('#'):
                    if current_section:
                        sections.append(current_section)
                    title = line.lstrip('#').strip()
                    current_section = Section(
                        name=title,
                        description="",
                        research=True,
                        content=""
                    )
            
            if current_section:
                sections.append(current_section)
            
            return Sections(sections=sections)
            
        except Exception as e:
            logger.error(f"Error parsing report plan: {str(e)}")
            # Return a default structure if parsing fails
            return Sections(sections=[
                Section(name="Introduction", description="Introduction to the topic", research=False, content=""),
                Section(name="Main Content", description="Main discussion", research=True, content=""),
                Section(name="Conclusion", description="Summary and conclusions", research=False, content="")
            ])
    
    async def _generate_section_queries(self, section: Section) -> list[SearchQuery]:
        """Generate search queries for a specific section."""
        prompt = ChatPromptTemplate.from_template(query_writer_instructions)
        
        response = await self.planner.ainvoke(
            prompt.format(
                topic=section.topic,
                section_title=section.title
            )
        )
        
        content = self._get_response_content(response)
        self._log_llm_response(
            f"section_queries_{section.title}",
            prompt.dict(),
            content
        )
        
        # Extract queries from response
        query_texts = self._extract_queries_from_text(content)
        return [SearchQuery(search_query=q) for q in query_texts]
    
    async def _write_section_content(
        self,
        section: Section,
        context: str,
        is_final_section: bool
    ) -> str:
        """Write content for a section using the provided context."""
        logger.info(f"Writing content for section: {section.name}")
        logger.info(f"Context length: {len(context) if context else 0}")
        logger.info(f"Is final section: {is_final_section}")
        
        try:
            template = (
                final_section_writer_instructions
                if is_final_section
                else section_writer_instructions
            )
            
            prompt = ChatPromptTemplate.from_template(template)
            
            response = await self.writer.ainvoke(
                prompt.format(
                    topic=section.topic,
                    section_title=section.title,
                    context=context
                )
            )
            
            content = self._get_response_content(response)
            self._log_llm_response(
                f"section_content_{section.title}",
                prompt.dict(),
                content
            )
            
            # Log the response before returning
            logger.info(f"Generated content length for {section.name}: {len(content) if content else 0}")
            return content
            
        except Exception as e:
            logger.error(f"Error writing section content: {str(e)}")
            return ""

async def interactive_report_generation():
    """Interactive terminal interface for report generation."""
    print("\n=== Report Generator ===")
    print("This tool will help you generate a comprehensive report on any topic.")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not found.")
        print("Please set your OpenAI API key first:")
        print("export OPENAI_API_KEY='your-key-here'")
        return
        
    if not os.getenv("TAVILY_API_KEY"):
        print("\n❌ Error: TAVILY_API_KEY environment variable not found.")
        print("Please set your Tavily API key first:")
        print("export TAVILY_API_KEY='your-key-here'")
        return
    
    # Get topic
    print("\nWhat topic would you like a report on?")
    topic = input().strip()
    
    if not topic:
        print("❌ Error: Topic cannot be empty")
        return
        
    # Get output path
    print("\nWhere would you like to save the report? (default: report.md)")
    output_path = input().strip() or "report.md"
    
    # Optional: Configure advanced settings
    print("\nWould you like to configure advanced settings? (y/n)")
    if input().lower().strip() == 'y':
        print("\nSelect planner model type:")
        print("1. Local LLM (default)")
        print("2. GPT-4")
        try:
            model_type = int(input().strip() or "1")
            if model_type not in [1, 2]:
                print("Invalid input, using default value: 1")
                model_type = 1
        except ValueError:
            print("Invalid input, using default value: 1")
            model_type = 1
            
        print("\nHow many search queries per section? (default: 2)")
        try:
            num_queries = int(input().strip() or "2")
        except ValueError:
            print("Invalid input, using default value: 2")
            num_queries = 2
            
        print("\nSearch type (news/general)? (default: general)")
        search_type = input().strip() or "general"
        
        if search_type == "news":
            print("How many days back to search? (default: 7)")
            try:
                days = int(input().strip() or "7")
            except ValueError:
                print("Invalid input, using default value: 7")
                days = 7
        else:
            days = None
            
        config = Configuration(
            planner_model_type=model_type,
            number_of_queries=num_queries,
            tavily_topic=search_type,
            tavily_days=days
        )
    else:
        config = None
    
    print(f"\n🔍 Starting report generation on topic: {topic}")
    
    try:
        generator = ReportGenerator(config)
        report = await generator.generate_report(topic, output_path)
        
        print("\n🎉 Report generation complete!")
        print(f"Your report has been saved to: {output_path}")
        
        print("\nWould you like to view the report now? (y/n)")
        if input().lower().strip() == 'y':
            print("\n" + "="*60)
            print(report)
            print("="*60)
            
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please try again or contact support if the problem persists.")

if __name__ == "__main__":
    try:
        asyncio.run(interactive_report_generation())
    except KeyboardInterrupt:
        print("\n\n⚠️  Report generation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        sys.exit(1) 