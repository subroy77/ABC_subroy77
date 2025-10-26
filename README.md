# Prompt Classifier Agent

An AI-powered agent for classifying prompts using DSPy and ollama.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys
4. Run the tests:
   ```bash
   pytest
   ```

## Environment Variables

- `DSPY_MODEL`: The DSPy model to use (default: ollama)
- `DSPY_MODEL_NAME`: The specific model name
- `SERPAPI_API_KEY`: API key for SerpAPI integration
