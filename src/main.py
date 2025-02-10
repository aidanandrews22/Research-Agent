import asyncio
from pathlib import Path
from typing import Optional, List, Dict
import sys
import os
import re
import json
import logging
from datetime import datetime
import uuid
import dotenv
import torch
import itertools

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

# Global semaphore for LLM access
llm_semaphore = asyncio.Semaphore(1)

MAX_CHARS_FOR_CONTEXT = 4000  # Reduced from 8000 to be more conservative

def _clear_memory():
    """Aggressively clear memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # More aggressive GPU memory clearing
        torch.cuda.synchronize()

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
            llm=self.planner,  # Use the planner model for ranking
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
        
    async def _llm_call(self, llm, prompt, *args, **kwargs):
        """Wrapper for LLM calls to ensure sequential execution"""
        async with llm_semaphore:
            # Clear CUDA cache before each LLM call
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return await llm.ainvoke(prompt, *args, **kwargs)

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
                # Clean up any escaped quotes and normalize to proper JSON
                json_str = json_str.replace('\\"', '"').replace('\\n', ' ').strip()
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
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
        except Exception as e:
            logger.warning(f"Error extracting queries: {e}")

        # Fallback: Try to find quoted queries
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
        
        return quoted_queries[:self.config.number_of_queries] if quoted_queries else ["No valid queries found"]
    
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
                section_queries = await self._generate_section_queries(section)
                print("  Generated queries:")
                for j, query in enumerate(section_queries, 1):
                    print(f"    {j}. {query.search_query}")
                
                # Search and fetch content for section queries
                print("  Gathering research...")
                section_content = []
                for query in section_queries:
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
                try:
                    # Ensure output_path is not empty and has a valid directory
                    if not output_path.strip():
                        output_path = "../reports/report.md"  # Default filename
                    
                    # Get the directory path
                    dir_path = os.path.dirname(output_path)
                    
                    # Create directory if it doesn't exist and if there's a directory path
                    if dir_path:
                        os.makedirs(dir_path, exist_ok=True)
                    
                    # Write the report
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                    print(f"\n✅ Report saved to: {output_path}")
                except Exception as e:
                    logger.error(f"Error saving report to file: {str(e)}")
                    print(f"\n⚠️  Could not save report to file: {str(e)}")
                    print("Report content will still be displayed below.\n")
            
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
        
        response = await self._llm_call(
            self.planner,
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
    
    async def _process_context_chunks(
        self,
        chunks: List[str],
        prompt_template: ChatPromptTemplate,
        stage: str = "processing",  # Add stage parameter
        **prompt_kwargs
    ) -> str:
        """Process multiple context chunks and combine their results."""
        all_responses = []
        total_chunks = len(chunks)
        
        # Log the start of chunk processing
        logger.info(f"Starting to process {total_chunks} chunks for {stage}")
        self._log_llm_response(
            f"{stage}_chunks_start",
            {"total_chunks": total_chunks},
            f"Beginning processing of {total_chunks} chunks"
        )
        
        try:
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i}/{total_chunks} for {stage}")
                
                # Force garbage collection before processing each chunk
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Add chunk metadata to help the model understand context
                chunk_context = f"[Processing chunk {i}/{total_chunks}]\n\n{chunk}"
                
                # Include previous summary if this isn't the first chunk
                if all_responses:
                    chunk_context = f"Previous summary:\n{all_responses[-1]}\n\nContinuing with new content:\n{chunk_context}"
                
                # Log the chunk being processed
                self._log_llm_response(
                    f"{stage}_chunk_{i}",
                    {"chunk_size": len(chunk_context)},
                    f"Processing chunk {i}/{total_chunks}"
                )
                
                try:
                    response = await self._llm_call(
                        self.planner,
                        prompt_template.format(
                            context=chunk_context,
                            **prompt_kwargs
                        )
                    )
                    
                    content = self._get_response_content(response)
                    all_responses.append(content)
                    
                    # Log successful chunk completion
                    self._log_llm_response(
                        f"{stage}_chunk_{i}_complete",
                        {"chunk_size": len(chunk_context)},
                        content
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}/{total_chunks}: {str(e)}")
                    self._log_llm_response(
                        f"{stage}_chunk_{i}_error",
                        {"error": str(e)},
                        f"Failed to process chunk {i}"
                    )
                    # Continue with next chunk instead of failing completely
                    continue
                
                # Force cleanup after each chunk
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Combine all responses
            if len(all_responses) == 1:
                return all_responses[0]
            
            # For multiple chunks, generate a final summary
            logger.info(f"Generating final summary for {stage}")
            combined_response = "\n\n".join(all_responses)
            summary_prompt = ChatPromptTemplate.from_template(
                "Synthesize these separate analyses into a single coherent response:\n\n{text}"
            )
            
            # Log start of summary generation
            self._log_llm_response(
                f"{stage}_summary_start",
                {"response_count": len(all_responses)},
                "Starting final summary generation"
            )
            
            try:
                final_response = await self._llm_call(
                    self.planner,
                    summary_prompt.format(text=combined_response)
                )
                
                result = self._get_response_content(final_response)
                
                # Log successful summary completion
                self._log_llm_response(
                    f"{stage}_summary_complete",
                    {"summary_length": len(result)},
                    result
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error generating final summary: {str(e)}")
                self._log_llm_response(
                    f"{stage}_summary_error",
                    {"error": str(e)},
                    "Failed to generate final summary"
                )
                # Return the concatenated responses if summary fails
                return combined_response
                
        except Exception as e:
            logger.error(f"Error in chunk processing: {str(e)}")
            self._log_llm_response(
                f"{stage}_error",
                {"error": str(e)},
                "Failed during chunk processing"
            )
            raise

    async def _generate_report_plan(self, state: dict, context: str) -> Sections:
        """Generate the report plan based on search results."""
        prompt = ChatPromptTemplate.from_template(report_planner_instructions)
        
        # Split context into manageable chunks
        chunks = self._chunk_text(context)
        
        # Process all chunks
        content = await self._process_context_chunks(
            chunks,
            prompt,
            stage="report_plan",
            topic=state["topic"],
            report_organization=self.config.report_structure,
            feedback=state["feedback_on_report_plan"]
        )
        
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
        
        response = await self._llm_call(
            self.planner,
            prompt.format(
                section_name=section.name,
                section_description=section.description
            )
        )
        
        content = self._get_response_content(response)
        self._log_llm_response(
            f"section_queries_{section.name}",
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
            
            # Split context into chunks
            chunks = self._chunk_text(context)
            
            # Process all chunks
            content = await self._process_context_chunks(
                chunks,
                prompt,
                stage=f"section_{section.name}",
                section_topic=section.name
            )
            
            self._log_llm_response(
                f"section_content_{section.name}",
                prompt.dict(),
                content
            )
            
            logger.info(f"Generated content length for {section.name}: {len(content) if content else 0}")
            return content
            
        except Exception as e:
            logger.error(f"Error writing section content: {str(e)}")
            return ""

    def _chunk_text(self, text: str, max_chars: int = MAX_CHARS_FOR_CONTEXT) -> List[str]:
        """Split text into chunks of size <= max_chars."""
        if len(text) <= max_chars:
            return [text]
            
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Split by double newlines to preserve paragraph structure
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # If a single paragraph is too long, split it further
            if len(paragraph) > max_chars:
                # Split into sentences (crude but effective)
                sentences = paragraph.replace('. ', '.\n').split('\n')
                for sentence in sentences:
                    if current_size + len(sentence) + 2 <= max_chars:
                        current_chunk.append(sentence)
                        current_size += len(sentence) + 2
                    else:
                        if current_chunk:
                            chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [sentence]
                        current_size = len(sentence)
            else:
                if current_size + len(paragraph) + 2 <= max_chars:
                    current_chunk.append(paragraph)
                    current_size += len(paragraph) + 2
                else:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [paragraph]
                    current_size = len(paragraph)
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

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
    output_path = input().strip()
    
    # Use default path if none provided
    if not output_path:
        output_path = "../reports/report.md"
    
    # Ensure the path has .md extension
    if not output_path.lower().endswith('.md'):
        output_path += '.md'
    
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
        
        # Always show the report content
        print("\nReport content:")
        print("="*60)
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