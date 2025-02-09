# Prompt to generate a search query to help with planning the report
report_planner_query_writer_instructions="""You are an expert technical writer, helping to plan a report. 

The report will be focused on the following topic:

{topic}

The report structure will follow these guidelines:

{report_organization}

Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information for planning the report sections.

The queries should:
1. Be directly related to the topic
2. Help gather information useful for planning the report structure
3. Be specific and focused
4. Avoid JSON or structural elements in the query text itself

IMPORTANT: Return your response in EXACTLY this format, with no additional text:
{{
    "queries": [
        {{"search_query": "First specific query here"}},
        {{"search_query": "Second specific query here"}},
        {{"search_query": "Third specific query here"}}
    ]
}}

Example of GOOD queries for a CSV processing topic:
{{
    "queries": [
        {{"search_query": "best practices for CSV data extraction and processing in Python"}},
        {{"search_query": "efficient CSV parsing and indexing techniques for large datasets"}},
        {{"search_query": "integrating CSV data with language models and NLP systems"}}
    ]
}}

Example of BAD queries to avoid:
- Queries containing JSON characters like {{ or }}
- Vague queries like "CSV processing"
- Queries that are just keywords without context
- Structural elements like "queries:" or "search_query:"
"""

# Prompt generating the report outline
report_planner_instructions="""I want a plan for a report. \n\nThe topic of the report is:\n\n{topic}\n\nThe report should follow this organization: \n\n{report_organization}\n\nHere is context to use to plan the sections of the report: \n\n{context}\n\nHere is feedback on the report structure from review (if any):\n\n{feedback}\n\nIMPORTANT: Return your response as a valid JSON object in the following format (do not include any additional text):\n{{"sections": [{{"name": "Introduction", "description": "Brief overview of the topic.", "research": false, "content": ""}}, {{"name": "Main Section", "description": "Detailed discussion of key aspects.", "research": true, "content": ""}}, {{"name": "Conclusion", "description": "Summary and final thoughts.", "research": false, "content": ""}}]}}\n\nMake sure to:\n1. Include an introduction (research: false) and conclusion (research: false)\n2. Leave all content fields empty (just "")\n3. Set research to true for sections needing web research\n4. Use proper JSON formatting with double quotes"""

# Query writer instructions
query_writer_instructions="""Your goal is to generate targeted web search queries that will gather comprehensive information for writing a technical report section.

Topic for this section:
{section_topic}

When generating {number_of_queries} search queries, ensure they:
1. Cover different aspects of the topic
2. Include specific technical terms
3. Target recent information where relevant
4. Look for comparisons or differentiators
5. Search for both documentation and practical examples

IMPORTANT: Return your response in the following JSON format:
{{
    "queries": [
        {{"search_query": "your first query here"}},
        {{"search_query": "your second query here"}}
    ]
}}
"""

# Section writer instructions
section_writer_instructions = """You are an expert technical writer crafting one section of a technical report.

Topic for this section:
{section_topic}

Guidelines for writing:

1. Technical Accuracy:
- Include specific version numbers
- Reference concrete metrics/benchmarks
- Cite official documentation
- Use technical terminology precisely

2. Length and Style:
- Strict 150-200 word limit
- No marketing language
- Technical focus
- Write in simple, clear language
- Start with your most important insight in **bold**
- Use short paragraphs (2-3 sentences max)

3. Structure:
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
  * Either a focused table comparing 2-3 key items (using Markdown table syntax)
  * Or a short list (3-5 items) using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- End with ### Sources that references the below source material formatted as:
  * List each source with title, date, and URL
  * Format: `- Title : URL`

3. Writing Approach:
- Include at least one specific example or case study
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the section content
- Focus on your single most important point

4. Use this source material to help write the section:
{context}

5. Quality Checks:
- Exactly 150-200 words (excluding title and sources)
- Careful use of only ONE structural element (table or list) and only if it helps clarify your point
- One specific example / case study
- Starts with bold insight
- No preamble prior to creating the section content
- Sources cited at end"""

final_section_writer_instructions="""You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

Section to write: 
{section_topic}

Available report content:
{context}

1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill insights from the report
    * Keep table entries clear and concise
- For non-comparative reports: 
    * Only use ONE structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps or implications
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point

4. Quality Checks:
- For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
- For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Do not include word count or any preamble in your response"""