"""
The Bradley-Terry (BT) model converts pairwise comparison data into a single ratio-scale score per item.

Basic idea:
  1. Build pairwise comparison data
  2. Fit the BT model with the classic iterative MM algorithm
  3. Convert to readable scores
  4. Plot the result
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
def build_pairwise_df(items, wins_a, wins_b):
    pairs = list(combinations(items, 2))
    if len(pairs) != len(wins_a) or len(pairs) != len(wins_b):
        raise ValueError(
            f"Expected {len(pairs)} win counts for {len(items)} items, "
            f"got {len(wins_a)} (wins_a) and {len(wins_b)} (wins_b)."
        )

    df = pd.DataFrame(pairs, columns=["Item A", "Item B"])
    df["Wins A"] = wins_a
    df["Wins B"] = wins_b
    return df
def bradley_terry_fit(df, max_iters=1000, error_tol=1e-6):
   
    wins_a = df.groupby("Item A")["Wins A"].sum()
    wins_b = df.groupby("Item B")["Wins B"].sum()
    total_wins = wins_a.add(wins_b, fill_value=0)
    num_games = Counter()
    for _, row in df.iterrows():
        key = tuple(sorted([row["Item A"], row["Item B"]]))
        num_games[key] += row["Wins A"] + row["Wins B"]

    items = sorted(set(df["Item A"]) | set(df["Item B"]))
    n = len(items)
    ranks = pd.Series(np.ones(n) / n, index=items)

    for iteration in range(max_iters):
        old_ranks = ranks.copy()

        for item in items:
            denom = 0.0
            for other in items:
                if other == item:
                    continue
                games = num_games[tuple(sorted([item, other]))]
                if games:
                    denom += games / (ranks[item] + ranks[other])

            ranks[item] = total_wins.get(item, 0) / denom if denom > 0 else 0.0
        ranks /= ranks.sum()

        if np.abs(ranks - old_ranks).sum() < error_tol:
            print(f" * Converged after {iteration + 1} iterations.")
            break
    else:
        print(f" * Max iterations reached ({max_iters}) without full convergence.")

    return ranks.sort_values(ascending=False)
def scale_scores(ranks, scale=100):
    scores = (ranks * scale).round(2)
    return scores.reset_index().rename(
        columns={"index": "Item", 0: "Score"}
    ).set_axis(["Item", "Score"], axis=1)
if __name__ == "__main__":
    items = ["Excerpt 1", "Excerpt 2", "Excerpt 3", "Excerpt 4", "Excerpt 5"]
    wins_a = [3, 11, 11, 5, 19, 20, 15, 16, 9, 7]
    wins_b = [22, 14, 14, 20, 6, 4, 10, 9, 16, 18]

    df = build_pairwise_df(items, wins_a, wins_b)
    print("Pairwise comparison data:")
    print(df, "\n")

    fitted_ranks = bradley_terry_fit(df)
    scores = scale_scores(fitted_ranks)

    print("\nFinal Bradley-Terry scores:")
    print(scores.to_string(index=False))

    if HAS_PLOTLY:
        fig = px.bar(
            scores.sort_values("Score"),
            x="Score", y="Item",
            title="Bradley-Terry readability scores",
            text="Score",
            width=600,
        )
        fig.write_html("bt_scores.html")
        print("\nSaved chart to bt_scores.html")
