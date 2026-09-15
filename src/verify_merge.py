import sys
import pandas as pd

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 verify_merge.py <run1.csv> <run2.csv> <merged.csv>")
        sys.exit(1)

    run1 = pd.read_csv(sys.argv[1])
    run2 = pd.read_csv(sys.argv[2])
    merged = pd.read_csv(sys.argv[3])

    run1_valid = set(run1.loc[run1["valid_schema"] == True, "tweet_id"])
    run2_valid = set(run2.loc[run2["valid_schema"] == True, "tweet_id"])
    merged_valid = set(merged.loc[merged["valid_schema"] == True, "tweet_id"])

    expected_min_valid = run1_valid | run2_valid

    print(f"Run 1 valid tweet_ids: {len(run1_valid)}")
    print(f"Run 2 valid tweet_ids: {len(run2_valid)}")
    print(f"Union of both (expected MINIMUM valid in merge): {len(expected_min_valid)}")
    print(f"Merged file's actual valid tweet_ids: {len(merged_valid)}")

    lost = expected_min_valid - merged_valid
    if lost:
        print(f"\n*** BUG CONFIRMED: {len(lost)} previously-valid rows are MISSING "
              f"from the merged file ***")
        print("Lost tweet_ids:", sorted(lost))
        print("\nThese rows were valid in at least one run but are not valid in the "
              "merged result. The merge logic is discarding good data. Do not use "
              "the merged file's metrics until this is fixed.")
    else:
        print("\nNo valid rows were lost in the merge. Merged count is consistent.")

    if len(merged) != len(set(merged["tweet_id"])):
        dupes = merged["tweet_id"].duplicated().sum()
        print(f"\n*** WARNING: merged file has {dupes} duplicate tweet_id rows -- "
              f"should have been deduplicated to one row per tweet_id ***")

if __name__ == "__main__":
    main()
