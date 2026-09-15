"""
Task 1, part A: Reconstruct conversation groups.

Groups tweets into conversation threads by following reply-chain links
(in_response_to_tweet_id / response_tweet_id), NOT by author_id. Author-level
grouping would be overly conservative (it would merge a customer's totally
unrelated conversations from different days into one group) -- so we use the
actual thread structure instead, and document this choice.

Limitation to state in the decision log: response_tweet_id / in_response_to_tweet_id
only capture direct reply chains as recorded in this dataset export; if Twitter's
API missed a link (rare but possible), two tweets that were "really" the same
conversation could end up in different groups. We accept this as a reasonable
best-effort reconstruction, per Astra's plan's guidance to state such limitations
honestly rather than claim perfect thread reconstruction.
"""
import pandas as pd

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


print("Loading full dataset...")
df = pd.read_csv(
    "twcs/twcs.csv",
    usecols=["tweet_id", "in_response_to_tweet_id", "response_tweet_id"],
    dtype={"tweet_id": "int64", "in_response_to_tweet_id": "string", "response_tweet_id": "string"},
)
print(f"Loaded {len(df)} rows")

uf = UnionFind()

print("Building union-find over reply chains...")
for row in df.itertuples():
    tid = row.tweet_id
    uf.find(tid)  # ensure it's registered even if it has no links

    if pd.notna(row.in_response_to_tweet_id):
        parent_id = int(float(row.in_response_to_tweet_id))
        uf.union(tid, parent_id)

    if pd.notna(row.response_tweet_id):
        # response_tweet_id can contain multiple comma-separated ids
        for rid in str(row.response_tweet_id).split(","):
            rid = rid.strip()
            if rid:
                uf.union(tid, int(float(rid)))

print("Assigning group ids...")
df["conversation_group_id"] = df["tweet_id"].apply(uf.find)

group_map = df[["tweet_id", "conversation_group_id"]]
group_map.to_csv("tweet_to_conversation_group.csv", index=False)

n_groups = group_map["conversation_group_id"].nunique()
print(f"Done. {len(group_map)} tweets grouped into {n_groups} conversation threads.")
print(f"Average thread size: {len(group_map) / n_groups:.2f} tweets")
print("Saved tweet_to_conversation_group.csv")
