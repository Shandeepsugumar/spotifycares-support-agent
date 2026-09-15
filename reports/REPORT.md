# Hiver SDE Intern Assignment: SpotifyCares Support Agent

## 1. Problem Framing
The objective of this project is to build and evaluate an automated customer support agent capable of classifying customer intents and drafting grounded, helpful, and safe replies. "Good" performance in this context means accurately triaging requests and synthesizing historical resolutions without hallucinating brand policies or escalating unnecessarily. 

To maintain focus on core agentic logic and evaluation rigor, several architectural simplifications were made:
* **No Fine-Tuning:** The agent relies entirely on few-shot prompting and in-context learning.
* **No Vector Database:** Retrieval utilizes standard `scikit-learn` TF-IDF and cosine similarity.
* **No Custom UI:** The pipeline operates purely via CLI and CSV artifacts.

## 2. Data, Sampling, Annotation, and Leakage Prevention
* **Brand Selection:** **SpotifyCares** was chosen after comparing volume and content-sample quality against Delta, AmericanAir, and ChipotleTweets. SpotifyCares provided both high volume (43k+ matched pairs) and actual in-thread resolutions, whereas airlines frequently punted to a generic "DM us" without public resolution.
* **Golden Eval Set:** Constructed a 181-row hand-labeled dataset from an original 200-row stratified random sample. We explicitly excluded 19 rows consisting of context-less mid-thread fragments (e.g., bare "Yes", "Nope"), 1 empty-text row, and 1 non-English row. Keyword heuristics were used solely to suggest starting guesses for manual annotation.
* **Data Splitting:** The golden set was split into `dev_fold.csv` (30 rows) and `test_fold.csv` (151 rows) using a stratified split (seed=42) with zero `tweet_id` overlap. 
* **Leakage Prevention:** A conversation-group split was enforced via reply-chain union-find (rather than author-grouping). It was mathematically verified that the golden eval set, few-shot pool, and the 42,658-pair historical retrieval corpus share **zero conversation threads**.

## 3. The Three Systems
1. **Baseline A (Trivial):** A naive baseline that always predicts the dev-fold majority class (`app_technical_issue`), issues a fixed generic reply, and always escalates.
2. **Baseline B (Simple):** A deterministic pipeline using keyword/regex rules (tuned exclusively on the 30-row dev fold) combined with a TF-IDF nearest-neighbor retriever for drafting replies. No LLM is used.
3. **Agent (RAG + LLM):** An LLM pipeline featuring few-shot intent classification and a TF-IDF-retrieved evidence-grounded reply drafter that cites specific `evidence_ids`. It includes stated escalation reasons, schema validation with a safe-escalate fallback, deterministic policy overrides on hacking/fraud keywords, and prompt-injection guards.

## 4. Results vs. Baselines
The systems were evaluated on the full 151-row frozen test fold (baselines) and a 37 valid-schema example subset (agent).

| Metric | Baseline A | Baseline B | Agent (LLM) |
| :--- | :--- | :--- | :--- |
| **Intent Accuracy** | 22.5% | 51.0% | **67.6%** |
| **Intent Macro F1** | 0.0525 | 0.5037 | **0.6544** |
| **Escalation Rate** | 100.0% | 11.3% | 39.1% |
| **Valid-Schema Rate** | N/A | N/A | **80.4%** |

*Note: Baseline B's performance highlighted two known weak spots: a 5% recall on `content_catalog_request` and a 14% precision on `general_inquiry_other`.*

**LLM-as-a-Judge Mean Scores (1-5 Scale)**
Scored on the same 37-example subset using `qwen/qwen3.8-27b` and a shared evidence bundle per example. Safety violations were 0 across all systems.
* **Relevance:** Agent (**4.86**) vs Baseline B (3.35) vs Baseline A (2.43)
* **Grounding:** Agent (**4.76**) vs Baseline B (4.16) vs Baseline A (3.00)
* **Helpfulness:** Agent (**3.78**) vs Baseline B (3.00) vs Baseline A (1.97)
* **Tone:** Agent (**4.57**) vs Baseline B (4.03) vs Baseline A (3.92)

*Known Limitation:* The judge scored against general rubric anchors. The ideal best-practice design requires human-authored per-example `acceptable_reply_points` and `forbidden_claims`, which the golden set unfortunately did not capture.

## 5. Human-Judge Agreement Results
To validate the LLM judge, we computed Weighted Cohen's Kappa on 45 blinded items (15 examples × 3 systems) comparing human ratings to judge ratings:
* **Relevance:** kappa=0.588 (Moderate agreement)
* **Helpfulness:** kappa=0.389 (Fair agreement)
* **Tone:** kappa=0.290 (Weak agreement)
* **Grounding:** kappa=0.206 (Weak agreement)
* **Safety violation:** Undefined. Both the human and the judge flagged 0 violations resulting in 100% exact agreement, but zero variance renders the kappa mathematically undefined.

*Finding:* This weak-to-moderate agreement on 3 of 4 dimensions is a crucial outcome. It implies that the LLM judge is significantly more trustworthy for objective dimensions (Relevance) than for subjective dimensions (Grounding, Tone).

## 6. Verified Failure Modes
1. **Keyword Overfitting (Baseline B):** 
    * *Example:* `tweet_id 2496166` ("put chance the rappers entire discography...")
    * *Failure:* True intent is `feature_request_feedback`, but predicted as `general_inquiry_other`.
    * *Mechanical Reason:* Keyword rules were heavily overfit to the dev set's narrow vocabulary ("feature", "add", "wish"), entirely missing phrasings like "put". 
    * *Hypothesis:* Manual regex rules will always fail to generalize to the massive lexical variety of real user requests.
2. **Semantic Blindspot (Baseline B):** 
    * *Example:* `tweet_id 572077` ("WHY ISN'T BJORK'S ENTIRE DISCOGRAPHY ON SPOTIFY")
    * *Failure:* True intent is `content_catalog_request`, but predicted as `general_inquiry_other`.
    * *Mechanical Reason:* The word "discography" is absent from the keyword list (which strictly expects "song", "album", or "artist"). This perfectly explains Baseline B's abysmal 5% recall on catalog requests.
3. **Judge Harsher than Human on "Unable to Help" Replies:** 
    * *Example:* `ITEM_032`, `tweet_id 1403655` (Baseline B Reply: "We're afraid we don't have any additional info to share...")
    * *Failure:* Human scored Relevance=5, but Judge scored Relevance=1.
    * *Hypothesis:* An LLM judge rigidly interprets rubrics and penalizes an honest "can't help" response as ignoring the issue. The judge may also be distracted by the mismatched historical Twitter handle in the verbatim retrieved reply.
4. **Judge Blind to a Hallucinated Feature:** 
    * *Example:* `ITEM_008`, `tweet_id 2264883` (Agent Reply: "...check the release date or look for it in the app’s upcoming releases section.")
    * *Failure:* Human scored Grounding=1 (catching the hallucination), Judge scored Grounding=4.
    * *Hypothesis:* Without explicit domain/product knowledge, an LLM judge cannot reliably distinguish confident, plausible-sounding fabrications from real information (explaining the 0.206 Grounding kappa).
5. **Failure Mode Shifted from Schema to Infrastructure:** 
    * *Example:* The Agent run resulted in 8 real hard-failures. 
    * *Failure:* 100% were API rate-limit failures (`api_call_failed`); 0% were JSON schema failures. 
    * *Hypothesis:* Modern LLMs can easily follow JSON schemas, shifting the primary engineering bottleneck to infrastructure and rate limits. *Note:* Adaptive pacing was implemented to address this, but it was **not yet validated** under live conditions because testing immediately hit an already-exhausted daily quota. 

## 7. What is misleading about my headline number?
The conclusion that "the Agent significantly outperforms the baselines" carries real, substantive uncertainty due to the following structural limitations:
1. **Exclusions:** 19 complex/contextless rows were excluded from the golden set entirely, artificially simplifying the evaluation environment.
2. **Dev/Test Gap:** Baseline B's performance suffered from a massive dev/test gap (70% dev vs 51% test accuracy), meaning the baseline it beat was highly overfit.
3. **Small, Mixed-Backend Sample (CRITICAL):** The agent was only evaluated on a 37-row subset of the 151-row test fold due to free-tier quota limits. More importantly, these 37 rows **mix two different LLM backends** (`groq/compound` n=30, `openai/gpt-oss-20b` n=7). 
4. **Judge Unreliability:** The LLM-as-a-judge showed weak agreement with human ratings on 3 out of 4 dimensions (including a 0.206 kappa on Grounding). The LLM's high scores for the Agent's replies may partly reflect the judge's inability to penalize plausible hallucinations.

## 8. Next Steps (With One More Week)
* **Infrastructure:** Secure a paid API tier or implement a multi-key pool to successfully evaluate the agent on the full 151-row test fold, and validate the new adaptive pacing logic live.
* **Consistency:** Re-run the entire evaluation utilizing a single, consistent model backend rather than mixing models mid-run.
* **Judge Anchoring:** Hand-author `acceptable_reply_points` and `forbidden_claims` for each example in the golden set to explicitly anchor the judge and correct its blindness to hallucinations.
