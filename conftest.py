import sys
from pathlib import Path

# Put the repo root on the path so `import ontime` resolves when pytest runs
# from inside the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
