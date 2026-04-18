# Intelligent Credit Risk Scoring & Agentic Lending Decision Support

An AI-powered system that evaluates borrower credit risk and supports lending decisions through a policy-aware, agentic decision pipeline.

## Project Overview

Credit risk assessment is one of the most consequential decisions a financial institution makes — yet traditional evaluation methods are slow, inconsistent, and susceptible to human bias. This project tackles that problem by combining machine learning-based risk scoring with a Retrieval-Augmented Generation (RAG) pipeline and an agentic reasoning layer, so lenders get decisions that are fast, explainable, and grounded in actual policy.

## Key Features

- Automated preprocessing pipeline — imputation, one-hot encoding, and feature scaling included
- Train and compare multiple ML models side by side
- Real-time credit risk prediction with structured risk payloads
- RAG over lending-policy PDFs using a FAISS vector index
- Agentic lending verdicts that combine ML scores with policy context
- Follow-up Q&A with conversation memory across the session
- PDF report export covering the borrower profile, verdict, and policy passages
- Evaluation visualizations: ROC Curve, Confusion Matrix, Feature Importance
- Clean Streamlit interface with no setup friction for end users


## Machine Learning Models

- **Logistic Regression** — probabilistic classification that estimates the likelihood of default.
- **Decision Tree Classifier** — rule-based model that surfaces the key features driving each risk decision.

Both models are evaluated on Accuracy, ROC-AUC, and Confusion Matrix results, with visual outputs for each.

## Installation & Setup

**1. Clone the Repository**
```bash
git clone https://github.com/CWAbhi/Gen-AI_Capstone.git
cd Gen-AI_Capstone
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Add Policy PDFs**
Place any lending-policy or regulatory PDF files in:
`data/policies/`
The app will build the FAISS index automatically on first run.

**4. Launch the App**
```bash
streamlit run app.py
```

## Policy Retrieval (RAG)

The RAG pipeline in `src/rag_pipeline.py` handles all policy-related lookups:

- `ingest_policy_documents()` — parses PDFs and builds the FAISS index
- `get_policy_context(query)` — retrieves relevant policy passages for a given borrower scenario
- `build_policy_query(profile, risk_score)` — converts borrower risk factors into a retrieval query

```python
from src.rag_pipeline import get_policy_context

context = get_policy_context("High DTI borrower with low savings needs compensating-factor policy")
print(context)
```

## Agentic Lending Layer

The agent logic in `src/lending_agent.py` wraps the ML model and RAG pipeline as LangChain tools, then orchestrates a final lending verdict via `run_agentic_lending_decision(profile)`.

Configure your preferred LLM provider before running:

**Groq**
```bash
export GROQ_API_KEY=...
export LENDING_AGENT_PROVIDER=groq
export LENDING_AGENT_MODEL=llama-3.3-70b-versatile
```

**OpenAI**
```bash
export OPENAI_API_KEY=...
export LENDING_AGENT_PROVIDER=openai
export LENDING_AGENT_MODEL=gpt-4o
```

**Anthropic**
```bash
export ANTHROPIC_API_KEY=...
export LENDING_AGENT_PROVIDER=anthropic
export LENDING_AGENT_MODEL=claude-3-5-sonnet-latest
```

If no API key is set, the system falls back to a deterministic decision engine that still uses the ML score and retrieved policy context.

## Phase 4 Enhancements

The latest iteration added three meaningful upgrades: uploaded borrower CSVs now run through a full sklearn preprocessing layer automatically; the Agentic Analysis tab maintains conversation memory so follow-up questions don't lose context; and an Export Report action produces a clean PDF summary of each lending decision.

## Team

| Member | Contribution |
| :--- | :--- |
| Anshika Seth (2401010080) | Model Development, Streamlit UI, Deployment  |
| Abhijeet Dey (2401010014) | Model Development, Report|
| Aditya Ranjan(2401010035) | Video Explanation , Analysis | 
## Conclusion

This project shows what's possible when classical ML and modern generative AI are used together thoughtfully. The result is a credit risk system that doesn't just predict — it reasons, references policy, and explains itself. That combination is what makes it genuinely useful in a real lending context.

