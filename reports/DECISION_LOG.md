* **Brand Selection:** Chose SpotifyCares over Delta and AmericanAir because it provided high volume (43k+ pairs) and contained actual in-thread resolutions rather than generic "DM us" punts.
* **Taxonomy Rename:** Renamed `playback_technical_issue` to `app_technical_issue` after analysis showed non-playback bugs (e.g., search errors, Sonos sync issues) shared the exact same resolution paths.
* **Exclusion Criteria:** Decided to exclude 19 context-less mid-thread fragments (e.g., "Yes", "Nope") from the golden set to ensure a fair evaluation of the agent's few-shot classification and RAG-grounded reply generation.
* **Leakage-Prevention Method:** Selected conversation-group reconstruction (via reply-chain union-find) over author-grouping. Limitation acknowledged: reply-chain reconstruction depends on the dataset's recorded response_tweet_id/in_response_to_tweet_id links being complete -- if Twitter's API missed a link when this dataset was originally compiled, one real conversation could be incorrectly split into two separate groups, understating leakage risk slightly. Author-grouping was rejected as the alternative because it would over-merge a single customer's unrelated conversations from different days into one group.
* **Catch-all Threshold:** Set `general_inquiry_other` as a ~15% catch-all bucket to prevent forcing obscure or highly unique customer intents into unrelated taxonomic categories.
* **Model-Switching:** Switched LLM backends mid-evaluation (`groq/compound` and `openai/gpt-oss-20b`) due to strict rate limit exhaustion, accepting the consequence of a mixed-backend agent sample.
* **Subsample Fallback:** Opted to formally evaluate and compare the Agent on a 37 valid-schema subset rather than blocking the entire project on the inability to process the full 151-row test fold on a free tier.
* **Judge Model Choice:** Specifically selected `qwen/qwen3.8-27b` to serve as the LLM-judge because it belongs to a different model family than the agent backends, mitigating self-preference bias.
* **Undefined Kappa Reporting:** Decided to report the Safety kappa honestly as "undefined" rather than artificially claiming a 1.0 or 0.0, preserving statistical integrity in the face of zero-variance data (no violations flagged by either side).
* **Process Rigor via Independent Verification:** Caught and fixed three major bugs via independent verification scripts, not by chance: a Pandas `NaN` merge deduplication bug that dropped valid rows, a judge duplicate-write bug, and a corrupted header-row-as-data artifact in `agent_predictions_merged.csv` (which corrected the row count from 46 to 45 while keeping the valid-schema count unchanged at 37).
* **BERT Embedding Classifier Experiment (Negative Result, Documented):** Tested 
  a frozen BERT-embedding (all-MiniLM-L6-v2) + logistic regression classifier as 
  an alternative to keyword+TF-IDF intent classification. Result: 42.4% accuracy 
  / 0.284 macro F1, underperforming Baseline B's TF-IDF approach (51.0% / 0.5037) 
  on the same 151-row test fold. Root cause identified: the dev fold has only 
  ~30 examples across 7 classes (~4 per class), which is insufficient for a 
  statistical classifier to learn reliable decision boundaries from embeddings 
  alone -- unlike TF-IDF's hard keyword matching or the LLM agent's few-shot 
  reasoning, a from-scratch classifier has no fallback when trained on too little 
  data. Hypothesis (not verified in this project): with a substantially larger 
  labeled dataset (likely low hundreds of examples per class or more), BERT 
  embeddings would plausibly outperform both TF-IDF and possibly approach LLM-based 
  classification, while being far cheaper and faster at inference time (no API 
  call, no rate limits, runs locally in milliseconds). Not pursued further due to 
  labeled-data constraints in this project's scope; documented as a candidate 
  next step given more data.
