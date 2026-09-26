"""Historical CLI name; use src/evaluate.py for new experiments."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.evaluation import main

if __name__ == "__main__":
    main()
