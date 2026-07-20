"""
Bradley-Terry Model — using the `choix` library
`choix`implements Luce-family choice models, including Bradley-Terry, with fast, well-tested solvers.
"""

import numpy as np
import pandas as pd
from itertools import combinations
import choix

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


def df_to_choix_data(df):
    items = sorted(set(df["Item A"]) | set(df["Item B"]))
    item_to_idx = {item: i for i, item in enumerate(items)}

    data = []
    for _, row in df.iterrows():
        a_idx = item_to_idx[row["Item A"]]
        b_idx = item_to_idx[row["Item B"]]
        data.extend([(a_idx, b_idx)] * int(row["Wins A"]))  # A beat B
        data.extend([(b_idx, a_idx)] * int(row["Wins B"]))  # B beat A

    return data, items
def bradley_terry_fit_choix(df, method="ilsr", alpha=0.01):
    data, items = df_to_choix_data(df)
    n_items = len(items)

    if method == "ilsr":
        params = choix.ilsr_pairwise(n_items, data, alpha=alpha)
    elif method == "mm":
        params = choix.opt_pairwise(n_items, data, alpha=alpha)
    else:
        raise ValueError("method must be 'ilsr' or 'mm'")
    strengths = np.exp(params)
    strengths = strengths / strengths.sum()

    scores = pd.Series(strengths, index=items).sort_values(ascending=False)
    return scores


def scale_scores(scores, scale=100):
    return (scores * scale).round(2).reset_index().set_axis(["Item", "Score"], axis=1)
if __name__ == "__main__":
    items = ["Excerpt 1", "Excerpt 2", "Excerpt 3", "Excerpt 4", "Excerpt 5"]
    wins_a = [3, 11, 11, 5, 19, 20, 15, 16, 9, 7]
    wins_b = [22, 14, 14, 20, 6, 4, 10, 9, 16, 18]

    df = build_pairwise_df(items, wins_a, wins_b)
    print("Pairwise comparison data:")
    print(df, "\n")

    scores_ilsr = bradley_terry_fit_choix(df, method="ilsr")
    print("ILSR (spectral) scores:")
    print(scale_scores(scores_ilsr).to_string(index=False))

    scores_mm = bradley_terry_fit_choix(df, method="mm")
    print("\nMM (classic max-likelihood) scores:")
    print(scale_scores(scores_mm).to_string(index=False))

    if HAS_PLOTLY:
        plot_df = scale_scores(scores_ilsr).sort_values("Score")
        fig = px.bar(
            plot_df, x="Score", y="Item",
            title="Bradley-Terry readability scores (choix / ILSR)",
            text="Score", width=600,
        )
        fig.write_html("bt_scores_choix.html")
        print("\nSaved chart to bt_scores_choix.html")
