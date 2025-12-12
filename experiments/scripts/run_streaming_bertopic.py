import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from src.models.streaming_bertopic import load_config, main, parse_args

if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        args = load_config()
    main(args)
