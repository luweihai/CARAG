# Earnings Call Agentic RAG

A research implementation of context-enriched agentic Retrieval Augmented Generation (RAG) for analyzing earnings call transcripts and predicting post-earnings price movements.

## Overview

This repository implements a multi-agent system that processes earnings call transcripts to extract financial insights and predict stock price movements following earnings announcements. The system combines:

- **Agentic RAG Architecture**: Multiple specialized agents for different aspects of financial analysis
- **Context-Enriched Data**: Integration of earnings call transcripts with quarterly financial statements
- **Parallel Processing**: Efficient handling of large-scale financial datasets
- **Neo4j Graph Database**: Advanced relationship analysis and fact storage

## Dataset

| File | Description | 
|------|-------------|
| `merged_data_nasdaq.csv` | NASDAQ earnings call data |
| `merged_data_nyse.csv` | NYSE earnings call data |
| `financial_statements/` | Quarterly financial data organized by ticker |
| `gics_sector_map_nasdaq.csv` | GICS sector mappings for NASDAQ |
| `gics_sector_map_nyse.csv` | GICS sector mappings for NYSE |
| `gics_sector_map_maec.csv` | GICS sector mappings for MAEC dataset |
| `results/` | Analysis results and example outputs |
| `exampleResults.csv` | Example analysis outputs |
| `nyse_final_results.csv` | Final NYSE analysis results |

### Data Processing

The system includes utilities to filter large datasets to the most significant cases:

- **Top Returns Selection**: Selects calls with significant absolute `actual_return_log` values
- **Financial Statement Filtering**: Keeps only financial data for relevant tickers in the datasets
- **Sector Mapping**: Maps tickers to GICS sectors for comparative analysis

## Architecture

### Core Components

- **`orchestrator_parallel_facts.py`**: Main orchestration script for parallel processing
- **`agents/`**: Specialized analysis agents
  - `mainAgent.py`: Primary coordination agent
  - `comparativeAgent.py`: Comparative analysis between periods and peers
  - `historicalEarningsAgent.py`: Historical earnings pattern analysis
  - `historicalPerformanceAgent.py`: Stock performance analysis
- **`utils/`**: Utility functions and indexing
  - `indexFacts.py`: Neo4j integration and fact indexing
- **`baseline/`**: Baseline sentiment analysis implementations

### Key Features

- **Multi-Agent Processing**: Specialized agents for different financial analysis tasks
- **Parallel Execution**: Configurable worker pools (default: 10 workers)
- **Context Integration**: Combines transcript text with structured financial data
- **Token Usage Tracking**: Monitors LLM API costs and usage patterns
- **Neo4j Integration**: Graph database for relationship analysis and fact storage
- **Quarter Parsing**: Advanced parsing of financial report types and quarters
- **Memory System**: Tracks historical analysis for each ticker

## Setup

### Prerequisites

```bash
pip install openai pandas numpy langchain-openai scikit-learn tqdm neo4j transformers torch accelerate
```

### Configuration

1. **Configure credentials**: Create `credentials.json` with your API keys and database credentials, place it under your main directory:
   ```json
   {
     "openai_api_key": "your-openai-api-key",
     "neo4j_uri": "your-neo4j-database-uri",
     "neo4j_username": "your-neo4j-username", 
     "neo4j_password": "your-neo4j-password"
   }
   ```

2. **Data Preparation**: Ensure you have the required CSV files and financial statements directory

## Usage

### Running the Main Orchestrator

The main agentic RAG system processes earnings calls using multiple specialized agents:

```bash
# Process NYSE data (default)
python orchestrator_parallel_facts.py

# Process NASDAQ data
python orchestrator_parallel_facts.py --data merged_data_nasdaq.csv --sector-map gics_sector_map_nasdaq.csv
```

### Configuration Options

The orchestrator supports several configuration constants:

```python
# In orchestrator_parallel_facts.py
DATA_FILE = "merged_data_nyse.csv"  # Input data file
LOG_PATH = "nyse_final_results.csv"  # Output results file
TIMEOUT_SEC = 1000  # Timeout per call in seconds
MAX_WORKERS = 10    # Number of parallel workers
CHUNK_SIZE = 300    # Process tickers in chunks
```

### Processing Workflow

1. **Data Loading**: Loads earnings call transcripts and financial statements
2. **Sector Grouping**: Groups data by GICS sectors for parallel processing
3. **Agent Processing**: Each sector is processed by specialized agents:
   - Financial statement fact extraction
   - Transcript analysis and fact extraction
   - Comparative analysis
   - Historical pattern analysis
4. **Neo4j Indexing**: Facts are stored in Neo4j graph database
5. **Results Output**: Analysis results saved to CSV with predictions

### Output Format

The system generates comprehensive analysis results including:

- **Ticker and Quarter**: Company and reporting period
- **Transcript**: Original earnings call transcript
- **Parsed Facts**: Structured financial facts extracted from statements
- **Research Notes**: Agent-generated analysis and insights
- **Actual Returns**: Observed post-earnings price movements
- **Error Logging**: Any processing errors or issues

## Monitoring and Logging

### Token Usage Tracking

The system tracks LLM API usage across all agents:

- **Token Logs**: Stored in `token_logs/` directory
- **Cost Tracking**: Estimates API costs based on usage
- **Agent Breakdown**: Separate logs for each agent type

### Timing Analysis

Performance monitoring for optimization:

- **Timing Logs**: Stored in `timing_logs/` directory
- **Agent Performance**: Individual agent execution times
- **Overall Processing**: Total processing time and throughput

### Neo4j Database Logs

Database operation monitoring:

- **Deletion Logs**: Track database clearing operations
- **Performance Metrics**: Node and relationship counts

## Research Context

This implementation accompanies research on context-enriched agentic RAG systems for financial analysis. The system demonstrates:

- **Multi-Agent Cooperation**: How specialized agents can work together
- **Context Integration**: Combining unstructured text with structured data
- **Scalable Processing**: Efficient handling of large financial datasets
- **Predictive Analytics**: Using LLMs for financial prediction tasks

## File Structure

```
code/
├── orchestrator_parallel_facts.py    # Main orchestration script
├── agents/                           # Specialized analysis agents
│   ├── comparativeAgent.py         # Comparative analysis between periods and peers
│   ├── historicalEarningsAgent.py  # Historical earnings pattern analysis
│   ├── historicalPerformanceAgent.py # Stock performance analysis
│   ├── mainAgent.py                # Primary coordination agent
│   └── prompts/
│       └── prompts.py              # Agent prompts and templates
├── utils/                           # Utility functions and indexing
│   ├── __init__.py
│   └── indexFacts.py               # Neo4j integration and fact indexing
├── baseline/                        # Baseline implementations
│   ├── README.md                   # Baseline methods documentation
│   ├── FinBERT/
│   │   └── finbert_classifier.py  # FinBERT sentiment classifier
│   ├── HierFinBERT/
│   │   ├── dataset.py              # Hierarchical FinBERT dataset
│   │   ├── model.py                # Hierarchical FinBERT model
│   │   ├── segmentation.py         # Text segmentation utilities
│   │   └── train_baseline.py       # Training script for baseline
│   ├── LM/
│   │   └── sentiment_analysis.py   # Language model sentiment analysis
│   ├── graphOfThought/
│   │   └── baseline_graphOfThought.py # Graph of Thought baseline
│   ├── langGraph/
│   │   └── baseline_langgraph.py   # LangGraph baseline implementation
│   └── zeroShotGPT/
│       ├── prompts.py              # Zero-shot GPT prompts
│       └── zeroShot.py             # Zero-shot GPT implementation
├── financial_statements/            # Quarterly financial data
│   ├── [TICKER]_balance_sheet.csv  # Balance sheet data by ticker
│   ├── [TICKER]_cash_flow_statement.csv # Cash flow data by ticker
│   └── [TICKER]_income_statement.csv # Income statement data by ticker
├── results/                         # Analysis results and examples
│   ├── MAEC_Results_Examples.csv.zip # MAEC dataset results
│   ├── NASDAQ_Results_Examples.csv.zip # NASDAQ results
│   └── NYSE_Results_Examples.csv.zip # NYSE results
├── token_logs/                      # LLM API usage tracking
├── timing_logs/                     # Performance monitoring
│   └── transcript_level_timing.csv # Transcript processing times
├── neo4j_logs/                      # Database operation logs
│   └── neo4j_deletion_log.csv      # Database deletion operations
├── gics_sector_map_*.csv           # Sector mapping files
│   ├── gics_sector_map_maec.csv    # MAEC sector mappings
│   ├── gics_sector_map_nasdaq.csv  # NASDAQ sector mappings
│   └── gics_sector_map_nyse.csv    # NYSE sector mappings
├── merged_data_nasdaq.csv          # NASDAQ earnings call data
├── merged_data_nyse.csv            # NYSE earnings call data
├── nyse_final_results.csv          # Final NYSE analysis results
├── exampleResults.csv              # Example analysis outputs
├── credentials.json                # API and database credentials
└── LICENSE                         # Project license
```

## Performance

The system is optimized for processing large financial datasets:

- **Parallel Processing**: Up to 10 concurrent workers
- **Memory Efficient**: Processes data in configurable chunks
- **Database Integration**: Neo4j for scalable fact storage
- **Error Handling**: Robust error recovery and logging

## Contributing

This is a research implementation. For questions or contributions, please refer to the accompanying research paper and documentation.