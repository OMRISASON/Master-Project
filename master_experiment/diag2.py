"""Minimal test."""
from pathlib import Path
out = Path("diag2_out.txt")
out.write_text("Python ran successfully\n")
print("Done")
