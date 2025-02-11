# Prompt to generate a search query to help with planning the report
report_planner_query_writer_instructions="""You are an expert technical writer, helping to plan a report. 

The report will be focused on the following topic:

{topic}

The report structure will follow these guidelines:

{report_organization}

Your goal is to generate {number_of_queries} high-quality search queries that will help gather comprehensive information for planning the report sections.

The queries should:
1. Be directly related to the topic and its key aspects
2. Help gather information useful for planning the report structure
3. Be specific, focused, and contextually relevant
4. Build upon each other to cover different aspects of the topic
5. Avoid redundancy and overlapping coverage
6. Avoid JSON or structural elements in the query text itself

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
        {{"search_query": "latest advancements in CSV data extraction and processing Python 2024"}},
        {{"search_query": "performance comparison of CSV parsing libraries for large datasets"}},
        {{"search_query": "real-world case studies CSV data integration with language models"}}
    ]
}}

Example of BAD queries to avoid:
- Queries containing JSON characters like {{ or }}
- Vague queries like "CSV processing"
- Queries that are just keywords without context
- Structural elements like "queries:" or "search_query:"
- Redundant queries that cover the same ground
"""

# Prompt generating the report outline
report_planner_instructions="""I want a plan for a report. \n\nThe topic of the report is:\n\n{topic}\n\nThe report should follow this organization: \n\n{report_organization}\n\nHere is context to use to plan the sections of the report: \n\n{context}\n\nHere is feedback on the report structure from review (if any):\n\n{feedback}\n\nIMPORTANT: Return your response as a valid JSON object in the following format (do not include any additional text):\n{{"sections": [{{"name": "Introduction", "description": "Brief overview of the topic.", "research": false, "content": ""}}, {{"name": "Main Section", "description": "Detailed discussion of key aspects.", "research": true, "content": ""}}, {{"name": "Conclusion", "description": "Summary and final thoughts.", "research": false, "content": ""}}]}}\n\nMake sure to:\n1. Include an introduction (research: false) and conclusion (research: false)\n2. Leave all content fields empty (just "")\n3. Set research to true for sections needing web research\n4. Use proper JSON formatting with double quotes"""

# Query writer instructions
query_writer_instructions = """You are an expert at generating focused search queries to gather information for a section of a technical report.

Section Name: {section_name}
Section Description: {section_description}

Your task is to generate 3-5 specific search queries that will help gather comprehensive information for this section.

Guidelines for queries:
1. Make queries specific and targeted
2. Focus on technical details and implementation
3. Include version numbers or dates where relevant
4. Prioritize authoritative sources
5. Avoid marketing content

IMPORTANT: Return your response in EXACTLY this format, with no additional text:
{{"queries": [{{"search_query": "first specific query"}}, {{"search_query": "second specific query"}}, {{"search_query": "third specific query"}}]}}

Example of GOOD queries:
{{"queries": [
    {{"search_query": "python pandas read_csv performance optimization techniques 2024"}},
    {{"search_query": "efficient CSV data indexing methods for large datasets"}},
    {{"search_query": "best practices CSV data extraction python libraries"}}
]}}

Example of BAD queries to avoid:
- Queries with line breaks or extra spaces
- Queries containing JSON characters like {{ or }}
- Vague queries like "CSV processing"
- Queries that are just keywords without context"""

# Section writer instructions
section_writer_instructions = """You are an expert technical writer crafting one section of a technical report.

Section Topic:
{section_topic}

Guidelines for writing:

1. Technical Accuracy:
- Include specific version numbers and dates
- Reference concrete metrics/benchmarks
- Cite official documentation and primary sources
- Use technical terminology precisely
- Cross-reference information from multiple sources

2. Length and Style:
- Strict 150-200 word limit
- No marketing language
- Technical focus with practical insights
- Write in simple, clear language
- Start with your most important insight in **bold**
- Use short paragraphs (2-3 sentences max)
- Ensure logical flow between paragraphs

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
  * Format: `- Title (Date): URL`

4. Writing Approach:
- Include at least one specific example or case study
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the section content
- Focus on your single most important point
- Ensure content builds on previous sections when relevant

5. Use this source material to help write the section:
{context}

6. Quality Checks:
- Exactly 150-200 words (excluding title and sources)
- Careful use of only ONE structural element (table or list) and only if it helps clarify your point
- One specific example / case study
- Starts with bold insight
- No preamble prior to creating the section content
- Sources cited at end
- Information is cross-referenced and verified
- Content flows logically from previous sections"""

final_section_writer_instructions="""You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

Section Topic: 
{section_topic}

Available report content:
{context}

1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Focus on the core motivation and context for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed
- Set clear expectations for what the report will cover

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill key insights from the report
    * Keep table entries clear and concise
    * Ensure table adds value beyond simply restating content
- For non-comparative reports: 
    * Only use ONE structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps, implications, or future directions
- No sources section needed
- Ensure synthesis of key themes across sections

2. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point
- Draw connections between sections
- Highlight key patterns or trends
- Address any gaps or areas for future research

3. Quality Checks:
- For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
- For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Clear synthesis of report content
- Actionable next steps or implications
- Do not include word count or any preamble in your response"""