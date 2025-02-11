# AI-Powered Research Report Generator

A sophisticated report generation system that combines local and cloud-based Language Models (LLMs) with advanced web search capabilities to create comprehensive research reports on any topic.

## Features

- **Hybrid LLM Architecture**
  - Local DeepSeek model for planning and content ranking
  - Claude 3.5 Sonnet for high-quality writing
  - GPT-4 option for planning (configurable)

- **Advanced Search Capabilities**
  - Multi-source search (DuckDuckGo + optional Tavily)
  - LLM-powered result ranking
  - Smart deduplication system
  - Concurrent content fetching

- **Optimized Performance**
  - 4-bit quantization for local models
  - Flash Attention 2 for faster inference
  - Dynamic GPU memory management
  - Concurrent operations with rate limiting

- **Intelligent Content Processing**
  - Smart HTML content extraction
  - Adaptive chunk-based processing
  - Early stopping optimization
  - Comprehensive error handling

## Requirements

### System Requirements
- Linux (tested on Arch Linux)
- CUDA-capable GPU (recommended)
- Python 3.8+

### Dependencies
```
# Core dependencies
langchain-core
langchain-openai
langchain-anthropic
aiohttp
beautifulsoup4
duckduckgo-search

# LLM providers
openai
anthropic

# Utilities
python-dotenv
typing-extensions
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aidanandrews22/Research-Agent.git
cd Research-Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
TAVILY_API_KEY=your_tavily_key  # Optional
```

## Usage

Run the interactive report generator:

```bash
python src/main.py
```

The tool will guide you through:
1. Topic selection
2. Output file configuration
3. Optional advanced settings:
   - Model selection (Local LLM or GPT-4)
   - Number of search queries
   - Search type (news/general)
   - Time range for news searches

## Project Structure

```
.
├── src/
│   ├── main.py              # Main entry point
│   ├── configuration.py     # Configuration settings
│   ├── local_llm.py         # Local LLM implementation
│   ├── prompts.py          # System prompts
│   ├── state.py            # State management
│   ├── utils.py            # Utility functions
│   └── search/             # Search implementation
│       ├── orchestrator.py  # Search coordination
│       ├── ranker.py       # Result ranking
│       ├── deduplicator.py # Result deduplication
│       └── content_fetcher.py # Content retrieval
├── reports/                # Generated reports
├── logs/                   # System logs
└── requirements.txt        # Project dependencies
```

## Advanced Configuration

The system can be configured through the `Configuration` class in `src/configuration.py`:

- `planner_model_type`: Choose between local LLM (1) or GPT-4 (2)
- `number_of_queries`: Number of search queries per section
- `max_results_per_source`: Maximum results from each search provider
- `min_relevance_score`: Minimum score for ranked results
- `max_concurrent_fetches`: Control concurrent operations

## Performance Optimization

The system includes several optimizations for efficient operation:

1. **Memory Management**
   - Proactive CUDA cache clearing
   - Expandable memory segments
   - Semaphore-based LLM access control

2. **Search Optimization**
   - Batch processing of results
   - Early stopping when quality threshold is met
   - Smart retry mechanisms with exponential backoff

3. **Model Optimization**
   - 4-bit quantization for local models
   - Flash Attention 2 implementation
   - Conservative context window management

## Logging

The system maintains detailed logs in the `logs/` directory, including:
- Search results and metadata
- LLM responses
- Error tracking
- Performance metrics

## Error Handling

Comprehensive error handling includes:
- Retry mechanisms with exponential backoff
- Fallback options for failed operations
- Graceful degradation
- Detailed error logging

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

- DeepSeek AI for the local LLM model
- Anthropic for Claude
- OpenAI for GPT-4
- DuckDuckGo for search capabilities 
