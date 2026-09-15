"""
Build the blinded human-agreement audit sheet.

Samples 15 of the 37 fully-judged examples, pulls all 3 systems' candidate
replies for each (45 outputs total), shuffles them into random order with an
anonymous item_id, and writes:
  - human_rating_sheet.csv   -- what the human (you) actually rates. No
    system names, no judge scores, no ordering pattern that reveals identity.
  - ANSWER_KEY_do_not_open_yet.csv -- item_id -> (tweet_id, system) mapping,
    plus the judge's own scores for that item. Kept separate so rating isn't
    accidentally influenced by seeing the judge's scores first.

Each item includes the same retrieved evidence bundle the judge saw, so the
human is rating with the same information, not less.
"""
import pandas as pd
from retrieval import Retriever

SEED = 7  # different seed than the eval/fewshot splits, no reason to correlate
N_EXAMPLES = 15

judge = pd.read_csv("judge_scores_clean.csv")
retriever = Retriever()

unique_ids = judge["tweet_id"].unique()
sampled_ids = pd.Series(unique_ids).sample(n=N_EXAMPLES, random_state=SEED).tolist()

rows = []
for tid in sampled_ids:
    subset = judge[judge["tweet_id"] == tid]
    customer_text = subset.iloc[0]["customer_text"]
    evidence = retriever.retrieve(customer_text, k=3)
    evidence_text = " | ".join(
        f'"{e["customer_text"][:100]}" -> "{e["brand_reply_text"][:100]}"' for e in evidence
    )
    for _, r in subset.iterrows():
        rows.append({
            "tweet_id": tid,
            "system": r["system"],
            "customer_text": customer_text,
            "evidence_context": evidence_text,
            "candidate_reply": r["candidate_reply"],
            "judge_relevance": r["relevance"],
            "judge_grounding": r["grounding"],
            "judge_helpfulness": r["helpfulness"],
            "judge_tone": r["tone"],
            "judge_safety_violation": r["safety_violation"],
        })

full_df = pd.DataFrame(rows)

shuffled = full_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
shuffled["item_id"] = [f"ITEM_{i+1:03d}" for i in range(len(shuffled))]

human_sheet = shuffled[["item_id", "customer_text", "evidence_context", "candidate_reply"]].copy()
human_sheet["relevance"] = ""
human_sheet["grounding"] = ""
human_sheet["helpfulness"] = ""
human_sheet["tone"] = ""
human_sheet["safety_violation"] = ""
human_sheet["safety_reason"] = ""
human_sheet.to_csv("human_rating_sheet.csv", index=False)

answer_key = shuffled[["item_id", "tweet_id", "system",
                        "judge_relevance", "judge_grounding",
                        "judge_helpfulness", "judge_tone", "judge_safety_violation"]]
answer_key.to_csv("ANSWER_KEY_do_not_open_yet.csv", index=False)

print(f"Sampled {N_EXAMPLES} unique examples x 3 systems = {len(shuffled)} items.")
print("Wrote human_rating_sheet.csv (rate this one) and "
      "ANSWER_KEY_do_not_open_yet.csv (don't open until done).")
print()
print("System balance check (should be ~15 each):")
print(shuffled["system"].value_counts())
