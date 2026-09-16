# SpotifyCares Support Agent Evaluation

## Project Summary
This project builds and evaluates an automated customer support agent for the SpotifyCares brand on Twitter. Using a carefully curated 181-row hand-labeled dataset and a 42k+ pair historical retrieval corpus, the project compares a naive baseline, a deterministic TF-IDF/keyword baseline, and an LLM-powered RAG Agent. The Agent leverages few-shot prompting, schema validation, and deterministic overrides to classify intents, draft evidence-grounded replies, and determine escalation risk, significantly outperforming the deterministic baselines while revealing crucial insights about LLM-as-a-judge reliability and rate-limit infrastructure bottlenecks.

## Live Demo
Try the agent live: https://spotifycares-support-agent-jade.vercel.app/
Paste any customer support message and see the agent classify intent, draft a grounded reply, and decide whether to auto-handle or escalate -- in real time.

This is a demo UI for interactively testing the agent pipeline in a browser. It wraps the same `agent.py`/`retrieval.py` logic already verified in the evaluation results. **It does not affect or change any reported numbers**, and is provided purely as a bonus/demo addition, not part of the graded evaluation deliverables.

* **Frontend (try it here):** https://spotifycares-support-agent-jade.vercel.app/
* **Backend API:** https://spotifycares-demo-backend.onrender.com

**(Verified working as of 2026-09-16.)**


This demo defaults to `qwen/qwen3.8-27b`. `openai/gpt-oss-20b` (used for the graded evaluation results in reports/REPORT.md) was found to intermittently fail strict JSON validation in this live-request context during testing -- this does not affect or change any of the evaluated/reported numbers, which were independently verified against real batch runs.

*(Note: The backend is deployed on a free tier. The first request after idle time may take 10-30s due to cold start. This is expected behavior, not a bug.)*


## Additional Exploration: BERT-Based Classification

We also tested whether a BERT-embedding-based classifier (no LLM calls needed at 
inference time) could match or beat our TF-IDF baseline for intent classification. 
It underperformed on our small dataset (42.4% vs. 51.0% accuracy) -- likely due to 
having only ~30 dev-fold examples to train on, not a flaw in the approach itself. 
See `reports/DECISION_LOG.md` for the full analysis and reasoning, and 
`reports/REPORT.md`'s Next Steps section for why this is a promising direction 
worth revisiting with more labeled data (faster, cheaper inference at scale once 
enough training data exists).

## Setup Instructions
1. Clone this repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the `.env.example` file to `.env` and fill in your Groq API key:
   ```bash
   cp .env.example .env
   # Edit .env with your key
   ```

## Data Note
The raw Kaggle dataset (thoughtvector/customer-support-on-twitter) is **NOT** included in this repo due to its massive file size. Before running anything that touches the full corpus, you must:
1. Download the `twcs.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
2. Place the extracted file at exactly: `data/raw/twcs/twcs.csv`.
*(Note: The pre-processed evaluation sets, splits, and retrieval corpus are already provided in the `data/` folder).*

After placing `twcs.csv`, you MUST build the retrieval corpus and conversation groups in this exact order BEFORE running `agent.py` or `judge.py`:
```bash
python src/build_conversation_groups.py
python src/build_retrieval_corpus.py
```
*Expected output self-check:* The final step should output exactly **42,658 pairs** after leakage exclusion. *(Note: This was verified accurate via script logic, but since the raw `twcs.csv` is not present in this lightweight repo, please confirm the final count matches when running on your local machine).*

## Reproduction Steps

This project evaluates the systems in two distinct tiers. **Please read carefully before attempting to reproduce**, as the full pipeline took multiple hours across several days in practice.

### TIER 1 (Fast, ~2-5 min, No API Calls Needed)
You can instantly recompute the core baseline metrics from the already-saved prediction files without needing an API key or running long evaluation loops.

Run the following exact command from the root directory to verify the Baseline B metrics (51.0% Accuracy, 0.5037 Macro F1) from the cache:
```bash
python -c "import pandas as pd; from sklearn.metrics import accuracy_score, f1_score; df=pd.read_csv('results/baseline_predictions_full.csv'); print(f'Accuracy: {accuracy_score(df[\"true_intent\"], df[\"baseline_b_pred\"]):.4f}'); print(f'Macro F1: {f1_score(df[\"true_intent\"], df[\"baseline_b_pred\"], average=\"macro\")}')"
```

### TIER 2 (Slow, API-Dependent)
Re-running the Agent and LLM-as-a-Judge pipelines from scratch against the Groq API requires significant time and patience.
* **Actual Observed Time:** Due to strict free-tier rate limits, the LLM calls required a ~9s/call pacing and consistently exhausted the daily API quota after ~100-150 calls. Completing the agent and judge runs took **multiple hours spanned across several days** (forcing mid-run model switches to bypass quotas).
* **Important:** The 37-row Agent evaluation result and the 111-row Judge result are already saved in the `results/` folder. A fresh run is **NOT required** to see the reported numbers--only to verify them independently. If you attempt a fresh run without a paid tier, expect heavy rate limits and hard halts.
* Commands to initiate a fresh run:
  ```bash
  python src/agent.py
  python src/judge.py
  ```

## Repo Structure Explanation
* `data/`: Pre-processed datasets (golden eval set, train/test folds, few-shot pool).
* `reports/`: Detailed findings, failure analyses, and decisions (`REPORT.md`, `DECISION_LOG.md`).
* `results/`: Final output predictions, judge scores, and human audit files.
* `src/`: Python scripts for baselines, RAG agent, LLM judge, and evaluation.
* `.env.example`: Template for environment variables.
* `.gitignore`: Files excluded from source control (including raw data and logs).
* `README.md`: Project overview and instructions.
* `requirements.txt`: Python package dependencies.

## Documentation
* **[Final Report](reports/REPORT.md):** Details problem framing, system architectures, full results, and 5 verified failure modes.
* **[Decision Log](reports/DECISION_LOG.md):** 10 detailed entries recording critical methodology choices, taxonomy updates, and bugs caught via independent verification.

## Known Limitations
* **Exclusions & Narrow Evaluation:** We explicitly excluded 19 complex/contextless rows from the golden set.
* **Baseline Overfitting:** Baseline B suffers from a massive 70% dev vs 51% test accuracy gap.
* **Small, Mixed-Backend Agent Sample:** The agent was only evaluated on a **37-row subset** of the 151-row test fold, and mixes two different LLM backends (`groq/compound` and `openai/gpt-oss-20b`) due to rate limits.
* **Judge Unreliability:** The LLM judge exhibited **weak agreement with human ratings** on 3 out of 4 evaluation dimensions, meaning the LLM-judged "agent wins" conclusion carries real structural uncertainty.

For a full breakdown of these limitations and "what is misleading about the headline numbers", see Section 7 of the [Final Report](reports/REPORT.md).






