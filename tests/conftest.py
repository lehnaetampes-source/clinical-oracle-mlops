import sys
from pathlib import Path

# Permet à pytest de trouver rag_core.py à la racine du projet
sys.path.insert(0, str(Path(__file__).parent.parent))
