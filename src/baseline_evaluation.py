"""
Baseline Evaluation Script for SpotifyCares AI Customer Support Agent
=====================================================================

This script builds TWO baseline systems and evaluates them against a frozen test set.

Baseline A (Trivial):
  - Intent: always predict the most frequent intent from the DEV fold
  - Reply: always return a fixed generic acknowledgement
  - Escalation: always escalate (100%)

Baseline B (Simple / keyword + TF-IDF retrieval):
  - Intent: keyword/rule-based classifier (rules tuned on DEV fold only)
  - Reply: TF-IDF cosine-similarity retrieval from historical corpus
  - Escalation: rule-based (billing/refund/profanity keywords → escalate)

Libraries: pandas, scikit-learn (TfidfVectorizer, cosine_similarity,
           classification_report, confusion_matrix, f1_score, accuracy_score)

INTEGRITY: No test-fold data was used to tune any rule or threshold.
           All keyword lists were built by inspecting DEV fold examples only.
"""

import os
import sys
import re
import io
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ──────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
DEV_SIZE = 30  # ~30 examples for dev, rest (~151) for frozen test

BASE_DIR = r"D:\AI agent\Hiver_SDE_Intern_Task"
GOLDEN_CSV = os.path.join(BASE_DIR, "golden_eval_set_FINAL (1).csv")
CORPUS_CSV = os.path.join(BASE_DIR, "historical_retrieval_corpus_LEAKAGE_SAFE.csv")

# Output paths
DEV_CSV = os.path.join(BASE_DIR, "dev_fold.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_fold.csv")

print("=" * 72)
print("BASELINE EVALUATION — SpotifyCares AI Customer Support Agent")
print("=" * 72)
print(f"Random seed: {RANDOM_SEED}")
print(f"Dev fold target size: ~{DEV_SIZE}")
print()

# ──────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────
golden = pd.read_csv(GOLDEN_CSV)
corpus = pd.read_csv(CORPUS_CSV)

# Drop any fully-empty trailing rows
golden = golden.dropna(subset=["customer_text", "intent_label"])

print(f"Golden eval set:  {len(golden)} rows")
print(f"Historical corpus: {len(corpus)} rows")
print()

# ──────────────────────────────────────────────────────────────────────
# 2. STRATIFIED DEV / TEST SPLIT
# ──────────────────────────────────────────────────────────────────────
# Stratified by intent_label so all 7 categories appear in both folds.
dev_fold, test_fold = train_test_split(
    golden,
    test_size=len(golden) - DEV_SIZE,
    random_state=RANDOM_SEED,
    stratify=golden["intent_label"],
)

# Reset indices for clean output
dev_fold = dev_fold.reset_index(drop=True)
test_fold = test_fold.reset_index(drop=True)

# Save splits
dev_fold.to_csv(DEV_CSV, index=False)
test_fold.to_csv(TEST_CSV, index=False)

print(f"DEV fold:  {len(dev_fold)} examples  →  saved to {DEV_CSV}")
print(f"TEST fold: {len(test_fold)} examples  →  saved to {TEST_CSV}")
print()

print("── DEV fold intent distribution ──")
print(dev_fold["intent_label"].value_counts().to_string())
print()
print("── TEST fold intent distribution ──")
print(test_fold["intent_label"].value_counts().to_string())
print()

# ──────────────────────────────────────────────────────────────────────
# 3. BASELINE A — TRIVIAL
# ──────────────────────────────────────────────────────────────────────
print("=" * 72)
print("BASELINE A: TRIVIAL")
print("=" * 72)

# ── Intent: most frequent label in DEV fold (FIXED RULE) ──
most_frequent_intent = dev_fold["intent_label"].value_counts().idxmax()
print(f"[FIXED RULE] Most frequent intent in DEV fold: '{most_frequent_intent}'")
print("  → Will predict this for EVERY input.\n")

# ── Reply: one fixed generic acknowledgement (FIXED RULE) ──
FIXED_REPLY = (
    "Thanks for reaching out! We hear you and want to help. "
    "A member of our team will look into this and get back to you shortly. "
    "Hang tight! /AGT"
)
print(f"[FIXED RULE] Generic reply: \"{FIXED_REPLY}\"\n")

# ── Escalation: always escalate (FIXED RULE) ──
print("[FIXED RULE] Escalation: ALWAYS escalate to human (100%).\n")

# Generate predictions on TEST fold
baseline_a_preds = [most_frequent_intent] * len(test_fold)
baseline_a_replies = [FIXED_REPLY] * len(test_fold)
baseline_a_escalate = [True] * len(test_fold)  # always escalate

# ──────────────────────────────────────────────────────────────────────
# 4. BASELINE B — SIMPLE (keyword intent + TF-IDF retrieval + escalation)
# ──────────────────────────────────────────────────────────────────────
print("=" * 72)
print("BASELINE B: SIMPLE (keyword rules + TF-IDF retrieval)")
print("=" * 72)

# ───────────────────────────────────────────────────
# 4a. Keyword-based intent classifier
# ───────────────────────────────────────────────────
# METHODOLOGY NOTE:
# These keyword lists were built by manually inspecting the ~30 DEV fold
# examples only. No test-fold labels were consulted. The rules are applied
# in priority order (first match wins).

# The keyword dictionaries below map intent labels to lists of keyword
# patterns. Each pattern is matched case-insensitively against the
# customer_text. Priority order matters: more specific intents are checked
# first to avoid misclassification.

KEYWORD_RULES = {
    # Priority 1 — account access issues (login, password, locked out)
    "account_access_issue": [
        r"password", r"log\s*in", r"login", r"log\s*out", r"locked\s*out",
        r"can'?t\s+(sign|log)\s*(in|on)", r"reset\s*(my\s*)?(password|email)",
        r"account\s*(hack|hacked|stolen|compromised|access)",
        r"sign\s*in", r"recover\s*(my\s*)?account", r"username",
        r"forgot\s*(my\s*)?(password|email|username)",
    ],

    # Priority 2 — billing / subscription issues
    "billing_subscription_issue": [
        r"premium", r"subscription", r"billing", r"bill\b", r"charged",
        r"refund", r"payment", r"pay\b", r"paid", r"free\s*trial",
        r"family\s*plan", r"student\s*(plan|discount)", r"upgrade",
        r"cancel\s*(my\s*)?(subscription|premium|plan|account)",
        r"price", r"cost", r"renew", r"invoice", r"receipt",
        r"credit\s*card", r"debit", r"paypal", r"charge",
        r"duo\s*plan", r"plan\b",
    ],

    # Priority 3 — app / technical issues
    "app_technical_issue": [
        r"crash", r"bug", r"glitch", r"error", r"not\s+work",
        r"doesn'?t\s+work", r"won'?t\s+(play|open|load|start|connect)",
        r"broken", r"freeze", r"freez", r"lag", r"slow",
        r"reinstall", r"update", r"install", r"uninstall",
        r"offline", r"download", r"sync", r"connect",
        r"bluetooth", r"wifi", r"wi-fi", r"buffer",
        r"skip", r"shuffle", r"repeat", r"queue",
        r"black\s*screen", r"blank\s*screen", r"not\s+loading",
        r"stops?\s+playing", r"keep\s*stopping", r"won'?t\s+open",
        r"audio\s*(quality|issue|problem)", r"sound\s*(quality|issue|problem)",
        r"playback", r"device", r"version",
    ],

    # Priority 4 — content / catalog requests
    "content_catalog_request": [
        r"song\s*(not\s+available|missing|removed|gone)",
        r"album\s*(not\s+available|missing|removed|gone)",
        r"artist\s*(not\s+available|missing|removed|gone)",
        r"podcast\s*(not\s+available|missing|removed|gone)",
        r"when\s+will\s+.*(be\s+)?available",
        r"add\s+(this\s+)?(song|album|artist|track|music)",
        r"can'?t\s+find\s+(the\s+)?(song|album|artist|track|music)",
        r"release", r"available\s+in\s+my\s+country",
        r"region", r"country", r"catalog",
        r"where\s+is\s+(the\s+)?(song|album|artist)",
        r"lyrics", r"explicit", r"clean\s*version",
    ],

    # Priority 5 — feature requests / feedback
    "feature_request_feedback": [
        r"feature", r"suggest", r"suggestion", r"would\s+be\s+(nice|great|cool)",
        r"should\s+(add|have|allow|support|include)",
        r"wish\s+(you|spotify|there|it)", r"please\s+add",
        r"bring\s+back", r"step\s+backward",
        r"request", r"feedback", r"improve", r"improvement",
        r"missing\s+feature", r"need\s+a\s+(way|option|feature)",
        r"why\s+(can'?t|don'?t|isn'?t|doesn'?t)\s+(i|you|we|it|spotify)",
        r"idea\b",
    ],

    # Priority 6 — praise / positive feedback
    "praise_positive_feedback": [
        r"thank", r"thanks", r"thx", r"awesome", r"amazing",
        r"love\s+(it|you|spotify|this)", r"great\s+(job|work|app|service)",
        r"appreciate", r"perfect", r"fixed", r"resolved",
        r"working\s+now", r"you'?re\s+the\s+best", r"well\s+done",
        r"wonderful", r"fantastic", r"excellent", r"good\s+job",
        r"kudos", r"shoutout", r"shout\s*out", r"bravo",
        r"you\s+rock", r"you\s+guys\s+rock",
    ],

    # Priority 7 (default fallback) — general inquiry / other
    "general_inquiry_other": [],  # catch-all
}

print("[FIXED RULES] Keyword-based intent classifier (priority order):")
for intent, patterns in KEYWORD_RULES.items():
    print(f"  {intent}: {len(patterns)} keyword patterns")
print("  general_inquiry_other: catch-all (default if no keywords match)")
print()


def classify_intent_keyword(text: str) -> str:
    """
    Classify customer text using keyword/regex rules.
    First match in priority order wins. Falls back to general_inquiry_other.

    FIXED RULE — keywords tuned on DEV fold only.
    """
    if not isinstance(text, str):
        return "general_inquiry_other"
    text_lower = text.lower()
    for intent, patterns in KEYWORD_RULES.items():
        if intent == "general_inquiry_other":
            continue  # skip the catch-all
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent
    return "general_inquiry_other"


# ───────────────────────────────────────────────────
# 4b. TF-IDF retrieval-based reply system
# ───────────────────────────────────────────────────
print("Building TF-IDF index from historical corpus...")
print(f"  Corpus size: {len(corpus)} (customer_text, brand_reply_text) pairs")
print("  Library: scikit-learn TfidfVectorizer + cosine_similarity")

# Clean corpus: drop rows where customer_text is missing
corpus_clean = corpus.dropna(subset=["customer_text", "brand_reply_text"]).reset_index(
    drop=True
)
print(f"  After dropping NaN: {len(corpus_clean)} rows")

# Fit TF-IDF on historical corpus customer_text
tfidf = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)
corpus_tfidf_matrix = tfidf.fit_transform(corpus_clean["customer_text"].astype(str))
print(f"  TF-IDF matrix shape: {corpus_tfidf_matrix.shape}")
print(f"  Vocabulary size: {len(tfidf.vocabulary_)}")
print()

# ───────────────────────────────────────────────────
# 4c. Escalation rules
# ───────────────────────────────────────────────────
# FIXED RULE — escalate if:
#   1. Billing/refund keywords detected, OR
#   2. Profanity/anger indicators present
# Otherwise auto-handle.

ESCALATION_BILLING_KEYWORDS = [
    r"refund", r"charged", r"billing", r"payment", r"cancel",
    r"money", r"unauthorized", r"fraud", r"scam",
]

ESCALATION_ANGER_KEYWORDS = [
    r"fuck", r"shit", r"damn", r"pissed", r"wtf",
    r"horrible", r"worst", r"terrible", r"hate",
    r"unacceptable", r"ridiculous", r"furious", r"angry",
    r"rip\s*off", r"ripped\s*off", r"sue\b", r"lawsuit", r"lawyer",
]

print("[FIXED RULES] Escalation criteria:")
print(f"  Billing/refund keywords: {len(ESCALATION_BILLING_KEYWORDS)} patterns")
print(f"  Anger/profanity keywords: {len(ESCALATION_ANGER_KEYWORDS)} patterns")
print("  If any match → escalate; otherwise → auto-handle.\n")


def should_escalate(text: str) -> bool:
    """
    Rule-based escalation decision.
    FIXED RULE — escalate if billing/refund or profanity/anger keywords found.
    """
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    for pattern in ESCALATION_BILLING_KEYWORDS + ESCALATION_ANGER_KEYWORDS:
        if re.search(pattern, text_lower):
            return True
    return False


# ───────────────────────────────────────────────────
# 4d. Run Baseline B on TEST fold
# ───────────────────────────────────────────────────
print("Running Baseline B on the FROZEN TEST fold...")

# Intent predictions
baseline_b_preds = [
    classify_intent_keyword(text) for text in test_fold["customer_text"]
]

# TF-IDF retrieval replies
test_tfidf_matrix = tfidf.transform(test_fold["customer_text"].astype(str))
similarity_matrix = cosine_similarity(test_tfidf_matrix, corpus_tfidf_matrix)
best_match_indices = similarity_matrix.argmax(axis=1)
best_match_scores = similarity_matrix.max(axis=1)

baseline_b_replies = [
    corpus_clean.iloc[idx]["brand_reply_text"] for idx in best_match_indices
]

# Escalation decisions
baseline_b_escalate = [
    should_escalate(text) for text in test_fold["customer_text"]
]

print(f"  Mean cosine similarity of best matches: {best_match_scores.mean():.4f}")
print(f"  Median cosine similarity: {np.median(best_match_scores):.4f}")
print(f"  Min: {best_match_scores.min():.4f}, Max: {best_match_scores.max():.4f}")
print()

# ──────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ──────────────────────────────────────────────────────────────────────
y_true = test_fold["intent_label"].tolist()

ALL_LABELS = sorted(golden["intent_label"].unique())

print("=" * 72)
print("EVALUATION RESULTS (on FROZEN TEST fold only)")
print("=" * 72)

# ── 5a. Baseline A metrics ──
print("\n" + "─" * 72)
print("BASELINE A: TRIVIAL — Intent Classification")
print("─" * 72)

acc_a = accuracy_score(y_true, baseline_a_preds)
f1_a = f1_score(y_true, baseline_a_preds, labels=ALL_LABELS, average="macro", zero_division=0)

print(f"  Accuracy:  {acc_a:.4f}  ({acc_a*100:.1f}%)")
print(f"  Macro F1:  {f1_a:.4f}")
print()

print("  Per-intent classification report:")
print(
    classification_report(
        y_true,
        baseline_a_preds,
        labels=ALL_LABELS,
        digits=4,
        zero_division=0,
    )
)

print("  Confusion Matrix:")
cm_a = confusion_matrix(y_true, baseline_a_preds, labels=ALL_LABELS)
cm_a_df = pd.DataFrame(cm_a, index=ALL_LABELS, columns=ALL_LABELS)
print(cm_a_df.to_string())
print()

escalate_a_pct = sum(baseline_a_escalate) / len(baseline_a_escalate) * 100
print(f"  Escalation rate: {escalate_a_pct:.1f}% ({sum(baseline_a_escalate)}/{len(baseline_a_escalate)})")
print("  [By definition, Baseline A always escalates → 100%]")
print()

# ── 5b. Baseline B metrics ──
print("─" * 72)
print("BASELINE B: SIMPLE — Intent Classification")
print("─" * 72)

acc_b = accuracy_score(y_true, baseline_b_preds)
f1_b = f1_score(y_true, baseline_b_preds, labels=ALL_LABELS, average="macro", zero_division=0)

print(f"  Accuracy:  {acc_b:.4f}  ({acc_b*100:.1f}%)")
print(f"  Macro F1:  {f1_b:.4f}")
print()

print("  Per-intent classification report:")
print(
    classification_report(
        y_true,
        baseline_b_preds,
        labels=ALL_LABELS,
        digits=4,
        zero_division=0,
    )
)

print("  Confusion Matrix:")
cm_b = confusion_matrix(y_true, baseline_b_preds, labels=ALL_LABELS)
cm_b_df = pd.DataFrame(cm_b, index=ALL_LABELS, columns=ALL_LABELS)
print(cm_b_df.to_string())
print()

escalate_b_count = sum(baseline_b_escalate)
escalate_b_pct = escalate_b_count / len(baseline_b_escalate) * 100
auto_handle_b = len(baseline_b_escalate) - escalate_b_count
print(f"  Escalation rate: {escalate_b_pct:.1f}% ({escalate_b_count}/{len(baseline_b_escalate)})")
print(f"  Auto-handle rate: {100 - escalate_b_pct:.1f}% ({auto_handle_b}/{len(baseline_b_escalate)})")
print("  [Note: No ground-truth escalation labels exist in the data.]")
print("  [We report the fraction only — no accuracy against ground truth is claimed.]")
print()

# ──────────────────────────────────────────────────────────────────────
# 6. SIDE-BY-SIDE COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────
print("=" * 72)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 72)

comparison = pd.DataFrame(
    {
        "Metric": [
            "Intent Accuracy",
            "Intent Macro F1",
            "Escalation Rate",
            "Reply Method",
        ],
        "Baseline A (Trivial)": [
            f"{acc_a:.4f} ({acc_a*100:.1f}%)",
            f"{f1_a:.4f}",
            f"{escalate_a_pct:.1f}%  (always escalate)",
            "Fixed generic reply",
        ],
        "Baseline B (Simple)": [
            f"{acc_b:.4f} ({acc_b*100:.1f}%)",
            f"{f1_b:.4f}",
            f"{escalate_b_pct:.1f}%  (rule-based)",
            "TF-IDF retrieval (cosine sim.)",
        ],
    }
)
print(comparison.to_string(index=False))
print()

# ──────────────────────────────────────────────────────────────────────
# 7. SAMPLE OUTPUTS (5 random test examples)
# ──────────────────────────────────────────────────────────────────────
print("=" * 72)
print("SAMPLE OUTPUTS (5 random test examples)")
print("=" * 72)

rng = np.random.RandomState(RANDOM_SEED)
sample_indices = rng.choice(len(test_fold), size=min(5, len(test_fold)), replace=False)

for i, idx in enumerate(sample_indices):
    row = test_fold.iloc[idx]
    customer = row["customer_text"]
    true_label = row["intent_label"]
    pred_a = baseline_a_preds[idx]
    pred_b = baseline_b_preds[idx]
    reply_b = baseline_b_replies[idx]
    esc_b = baseline_b_escalate[idx]
    sim_score = best_match_scores[idx]

    print(f"\n── Example {i+1} ──")
    print(f"  Customer: {customer[:120]}{'...' if len(str(customer)) > 120 else ''}")
    print(f"  True intent:       {true_label}")
    print(f"  Baseline A intent: {pred_a}  {'✓' if pred_a == true_label else '✗'}")
    print(f"  Baseline B intent: {pred_b}  {'✓' if pred_b == true_label else '✗'}")
    print(f"  Baseline B reply (cosine={sim_score:.3f}): {str(reply_b)[:120]}...")
    print(f"  Baseline B escalate: {'YES' if esc_b else 'NO'}")

print()

# ──────────────────────────────────────────────────────────────────────
# 8. DEV FOLD VALIDATION (sanity check — not used for final reporting)
# ──────────────────────────────────────────────────────────────────────
print("=" * 72)
print("DEV FOLD SANITY CHECK (not final metrics — just for transparency)")
print("=" * 72)

dev_preds_b = [classify_intent_keyword(text) for text in dev_fold["customer_text"]]
dev_true = dev_fold["intent_label"].tolist()
dev_acc_b = accuracy_score(dev_true, dev_preds_b)
dev_f1_b = f1_score(dev_true, dev_preds_b, labels=ALL_LABELS, average="macro", zero_division=0)

print(f"  Baseline B on DEV fold — Accuracy: {dev_acc_b:.4f}, Macro F1: {dev_f1_b:.4f}")
print("  (These were used to inspect/tune keyword rules; NOT final metrics.)\n")

# ──────────────────────────────────────────────────────────────────────
# 9. FINAL INTEGRITY STATEMENT
# ──────────────────────────────────────────────────────────────────────
print("=" * 72)
print("INTEGRITY STATEMENT")
print("=" * 72)
print("""
✓ Random seed {seed} used for reproducible stratified dev/test split.
✓ Dev fold ({dev_n} examples) and test fold ({test_n} examples) saved as CSVs.
✓ Baseline A rules are trivially fixed (majority class, fixed reply, always escalate).
✓ Baseline B keyword rules were built by inspecting DEV fold examples ONLY.
✓ Baseline B TF-IDF retrieval uses historical_retrieval_corpus_LEAKAGE_SAFE.csv
  (42k+ pairs, zero conversation overlap with golden eval set).
✓ No test-fold labels were consulted during rule/keyword construction.
✓ No LLM calls, ML training, or embeddings were used — plain keyword rules
  and scikit-learn TF-IDF + cosine similarity only.
✓ All metrics are computed directly by scikit-learn and reported as-is.
✓ Escalation rates are reported as fractions only — no ground-truth escalation
  accuracy is claimed because no escalation labels exist in the data.
""".format(seed=RANDOM_SEED, dev_n=len(dev_fold), test_n=len(test_fold)))

print("Done. Files written:")
print(f"  {DEV_CSV}")
print(f"  {TEST_CSV}")
