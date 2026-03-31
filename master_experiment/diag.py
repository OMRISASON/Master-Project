"""Quick diagnostic: check what columns exist and if q0_vector/r_vector are populated."""
import ast
import pandas as pd
from pathlib import Path

df = pd.read_excel("experiments/exp/results_all.xlsx")

with open("diag_out.txt", "w") as f:
    f.write(f"Columns: {list(df.columns)}\n")
    f.write(f"Rows: {len(df)}\n")
    f.write(f"\nq0_vector present: {'q0_vector' in df.columns}\n")
    f.write(f"r_vector  present: {'r_vector' in df.columns}\n")
    f.write(f"skill_vector present: {'skill_vector' in df.columns}\n")

    for col in ["q0_vector", "r_vector", "skill_vector"]:
        if col in df.columns:
            sample = df[col].iloc[0]
            f.write(f"\n{col} type: {type(sample)}, value[:60]: {str(sample)[:120]}\n")

    # Check after parsing
    def _parse(val):
        if isinstance(val, list): return val
        if isinstance(val, str):
            try: return ast.literal_eval(val)
            except: return []
        return []

    if "q0_vector" in df.columns:
        parsed_counts = df["q0_vector"].apply(_parse).apply(len)
        f.write(f"\nq0_vector lengths: {parsed_counts.value_counts().to_dict()}\n")

    if "r_vector" in df.columns:
        parsed_counts2 = df["r_vector"].apply(_parse).apply(len)
        f.write(f"r_vector  lengths: {parsed_counts2.value_counts().to_dict()}\n")

    f.write("\nDone.\n")

print("Wrote diag_out.txt")
