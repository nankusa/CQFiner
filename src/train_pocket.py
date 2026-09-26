"""Historical CLI name; use src/train.py for new experiments."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.runtime import PocketRuntime, build_pocket_runtime, main

if __name__ == "__main__":
    main()
