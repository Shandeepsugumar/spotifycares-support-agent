"""
Task 5, part A: RAG retrieval component.

Given a new customer message, retrieves the top-K most similar historical
(customer_text, brand_reply_text) pairs from the leakage-safe corpus. These
become the "evidence" the LLM agent grounds its reply on.

This is the SAME corpus and SAME leakage guarantee used for Baseline B's
TF-IDF retrieval -- reused deliberately so the comparison is apples-to-apples.

Difference from Baseline B: Baseline B returns ONE historical reply verbatim.
The real agent retrieves multiple (default 4) candidates and gives them to
the LLM as grounding context to synthesize a NEW reply from, rather than
copy-pasting an old one.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, corpus_path="historical_retrieval_corpus_LEAKAGE_SAFE.csv", max_features=20000):
        self.corpus = pd.read_csv(corpus_path).dropna(subset=["customer_text", "brand_reply_text"])
        self.corpus = self.corpus.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.corpus["customer_text"].astype(str))
        print(f"Retriever ready: {len(self.corpus)} historical pairs indexed, "
              f"vocabulary size {len(self.vectorizer.vocabulary_)}")

    def retrieve(self, query_text, k=4):
        """Return top-k similar (evidence_id, customer_text, brand_reply_text, similarity) records."""
        q_vec = self.vectorizer.transform([str(query_text)])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for i in top_idx:
            row = self.corpus.iloc[i]
            results.append({
                "evidence_id": str(row["tweet_id"]),
                "customer_text": row["customer_text"],
                "brand_reply_text": row["brand_reply_text"],
                "similarity": round(float(sims[i]), 4),
            })
        return results


if __name__ == "__main__":
    # Quick smoke test -- no API calls, just verifies retrieval works
    r = Retriever()
    test_queries = [
        "my account got hacked and I can't log in anymore",
        "I was charged twice for premium this month",
        "the app keeps crashing when I try to search for a song",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        for ev in r.retrieve(q, k=3):
            print(f"  [sim={ev['similarity']}] {ev['customer_text'][:80]}")
            print(f"    -> reply: {ev['brand_reply_text'][:80]}")
