"""One-time fixture builder. Run manually if the npy needs to be regenerated.

Usage: python tests/fixtures/folio/_build_npy.py
"""
import numpy as np
from pathlib import Path

vectors = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],   # automatic_stay
        [0.0, 1.0, 0.0, 0.0],   # adequate_protection
        [0.0, 0.0, 1.0, 0.0],   # proof_of_claim
    ],
    dtype=np.float32,
)
out = Path(__file__).parent / "concepts.npy"
np.save(out, vectors)
print(f"Wrote {out} shape={vectors.shape}")
