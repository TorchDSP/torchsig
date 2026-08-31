import json
import sys
from pathlib import Path


def main() -> int:
    for filename in sys.argv[1:]:
        path = Path(filename)

        with path.open(encoding="utf-8") as f:
            notebook = json.load(f)

        if "metadata" not in notebook:
            print(f"Adding missing metadata to {path}")
            notebook["metadata"] = {}

            with path.open("w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=1)
                f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
