# Baseline Methods

This directory contains various baseline implementations for earnings call analysis and sentiment prediction. Each baseline provides a different approach to analyzing financial transcripts and predicting post-earnings price movements.

## Overview

The baseline methods range from traditional sentiment analysis to advanced neural network approaches:

- **Loughran-McDonald Sentiment Analysis**: Traditional financial lexicon-based approach
- **FinBERT**: Fine-tuned BERT model for financial sentiment
- **Hierarchical FinBERT**: Advanced hierarchical BERT with document segmentation
- **Zero-Shot GPT**: Direct LLM analysis without fine-tuning
- **LangGraph**: Multi-agent reasoning with critic feedback
- **Graph of Thoughts**: Advanced reasoning with multiple candidate thoughts

## Prerequisites

Install required packages:

```bash
pip install pandas numpy scikit-learn torch transformers accelerate tqdm langchain langgraph openai
```

## Configuration

All baselines require a `credentials.json` file in the root directory:

```json
{
  "openai_api_key": "your-openai-api-key"
}
```

## Baseline Methods

### 1. Loughran-McDonald Sentiment Analysis

**Location**: `baseline/LM/sentiment_analysis.py`

Traditional financial sentiment analysis using the Loughran-McDonald dictionary.

#### Setup
1. Download the Loughran-McDonald dictionary and place it at `baseline/LM/LM_Master.csv`
2. The dictionary should have columns: `Word`, `Positive`, `Negative`

#### Configuration
```python
# In sentiment_analysis.py
LM_DICT_PATH = "baseline/LM/LM_Master.csv"
DATA_PATH = "merged_data_nyse.csv"  # Change to your data file
OUTPUT_PATH = "baseline/nyse_sentiment_results.csv"
```

#### Usage
```bash
cd baseline/LM
python sentiment_analysis.py
```

#### Output
- CSV file with sentiment scores and classifications
- Metrics: Balanced accuracy, precision, recall, F1-score
- Predictions based on positive/negative word counts

---

### 2. FinBERT Classifier

**Location**: `baseline/FinBERT/finbert_classifier.py`

Fine-tuned BERT model for financial sentiment classification using k-fold cross-validation.

#### Configuration
```python
# In finbert_classifier.py
DATA_PATH = "merged_data_nasdaq.csv"  # Change to your data file
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = "baseline/finbert_returns_classifier"
N_FOLDS = 5  # Number of cross-validation folds
```

#### Usage
```bash
cd baseline/FinBERT
python finbert_classifier.py
```

#### Output
- Trained models saved to `baseline/finbert_returns_classifier/`
- Cross-validation results with accuracy, F1, precision, recall
- Model checkpoints for each fold

---

### 3. Hierarchical FinBERT

**Location**: `baseline/HierFinBERT/`

Advanced hierarchical BERT model with document segmentation for long financial documents.

#### Files
- `train_baseline.py`: Main training script
- `model.py`: Hierarchical BERT model definition
- `dataset.py`: Dataset loading and preprocessing
- `segmentation.py`: Document segmentation utilities

#### Configuration
```python
# In train_baseline.py
DATA_FILE = "merged_data_nasdaq.csv"  # Change to your data file
MODEL_NAME = "ProsusAI/finbert"
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
N_SPLITS = 5  # Cross-validation folds
```

#### Usage
```bash
cd baseline/HierFinBERT
python train_baseline.py
```

#### Output
- Trained hierarchical models
- Cross-validation results with macro F1, balanced accuracy
- Model checkpoints and evaluation metrics

---

### 4. Zero-Shot GPT

**Location**: `baseline/zeroShotGPT/`

Direct analysis using GPT models without fine-tuning.

#### Files
- `zeroShot.py`: Main analysis script
- `prompts.py`: Prompt templates

#### Configuration
```python
# In zeroShot.py
DATA_PATH = "transcripts_nasdaq.csv"  # Change to your data file
LOG_PATH = "baseline_nasdaq3.csv"
MODEL = "gpt-3.5-turbo"
RETURN_THRESHOLD = 0.05  # 5% threshold for reprocessing
```

#### Usage
```bash
cd baseline/zeroShotGPT
python zeroShot.py
```

#### Output
- CSV file with GPT analysis results
- Incremental processing with threshold-based updates
- Error logging and recovery

---

### 5. LangGraph Baseline

**Location**: `baseline/langGraph/baseline_langgraph.py`

Multi-agent reasoning system with analyst and critic agents using LangGraph.

#### Configuration
```python
# In baseline_langgraph.py
DATA_PATH = "transcripts_nasdaq.csv"  # Change to your data file
LOG_PATH = "baseline_nasdaq_lang2.csv"
MODEL_NAME = "gpt-4o-mini"
RETURN_THRESHOLD = 0.05
MAX_TRANSCRIPT_CHARS = 15_000
```

#### Usage
```bash
cd baseline/langGraph
python baseline_langgraph.py
```

#### Output
- CSV file with multi-agent analysis results
- Robust error handling and state management
- Analyst and critic agent interactions

---

### 6. Graph of Thoughts

**Location**: `baseline/graphOfThought/baseline_graphOfThought.py`

Advanced reasoning with multiple candidate thoughts and value-based selection.

#### Configuration
```python
# In baseline_graphOfThought.py
DATA_PATH = "transcripts_nyse.csv"  # Change to your data file
LOG_PATH = "baseline_nyse_got.csv"
MODEL_NAME = "gpt-4o-mini"
NUM_THOUGHT_SAMPLES = 3  # Number of candidate thoughts
THOUGHT_TEMP = 0.7  # Temperature for thought generation
VALUE_TEMP = 0.0  # Temperature for value estimation
```

#### Usage
```bash
cd baseline/graphOfThought
python baseline_graphOfThought.py
```

#### Output
- CSV file with Graph of Thoughts analysis
- Multiple candidate thoughts with value scores
- Best thought selection based on value function

## Data Format Requirements

All baselines expect CSV files with the following columns:

### Required Columns
- `ticker`: Stock ticker symbol
- `q`: Quarter (e.g., "2023-Q4")
- `transcript`: Earnings call transcript text
- `future_3bday_cum_return`: Actual 3-day cumulative return (for evaluation)

### Optional Columns
- `clean_transcript`: Alternative transcript column name
- `call_text`: Alternative transcript column name
- `text`: Alternative transcript column name

## Output Format

All baselines generate CSV files with:

### Common Output Columns
- `ticker`: Stock ticker
- `q`: Quarter
- `actual_return`: Observed return (from input data)
- `error`: Any processing errors

### Method-Specific Columns

#### Loughran-McDonald
- `positive_count`: Count of positive words
- `negative_count`: Count of negative words
- `polarity`: Sentiment polarity score
- `sentiment`: Sentiment classification (Positive/Negative/Neutral)
- `pred`: Binary prediction (1 for Positive, 0 for Negative)

#### FinBERT & Hierarchical FinBERT
- `prediction`: Model prediction
- `confidence`: Prediction confidence score
- `fold`: Cross-validation fold number

#### GPT-based Methods
- `analysis`: LLM-generated analysis
- `draft`: Initial draft (for multi-agent methods)
- `final_note`: Final analysis after refinement

## Performance Comparison

To compare baseline performance:

1. **Run all baselines** on the same dataset
2. **Use consistent evaluation metrics**:
   - Balanced Accuracy
   - Macro F1-Score
   - Precision and Recall
3. **Compare against actual returns** using the `future_3bday_cum_return` column

## Troubleshooting

### Common Issues

1. **Missing credentials.json**: Ensure the file exists in the root directory
2. **Missing data files**: Update DATA_PATH in each baseline script
3. **CUDA out of memory**: Reduce batch size in FinBERT baselines
4. **API rate limits**: Add delays between API calls for GPT-based methods

### Error Handling

- All baselines include error logging
- Check the output CSV files for error messages
- Most baselines support incremental processing and can resume from failures

## Customization

### Adding New Baselines

1. Create a new directory in `baseline/`
2. Follow the existing pattern for configuration and output
3. Include error handling and logging
4. Use consistent data format and evaluation metrics

### Modifying Existing Baselines

- Update configuration variables at the top of each script
- Modify prompts for GPT-based methods
- Adjust model parameters for neural network approaches
- Change evaluation metrics as needed

## Research Context

These baselines provide comparison points for the main agentic RAG system. They represent different approaches to financial text analysis:

- **Traditional**: Loughran-McDonald lexicon
- **Neural**: FinBERT and hierarchical variants
- **LLM-based**: Zero-shot and multi-agent approaches
- **Advanced Reasoning**: Graph of Thoughts and LangGraph

Each baseline demonstrates different trade-offs between accuracy, interpretability, and computational requirements. 