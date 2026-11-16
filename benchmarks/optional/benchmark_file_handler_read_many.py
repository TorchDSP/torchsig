"""Compare portable repeated reads with ``read_many`` for a selected reader."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from torchsig.utils.file_handlers.registry import file_handler_registry


def main() -> int:
    """Measure scalar and portable multi-record access for one dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=sorted(file_handler_registry.specs))
    parser.add_argument("root", type=Path)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    reader = file_handler_registry.resolve(args.backend).reader(args.root)
    indices = list(range(min(args.count, len(reader))))
    start = time.perf_counter()
    scalar = [reader.read(index) for index in indices]
    scalar_seconds = time.perf_counter() - start
    start = time.perf_counter()
    batch = reader.read_many(indices)
    batch_seconds = time.perf_counter() - start
    if len(scalar) != len(batch):
        raise RuntimeError("read_many returned a different record count")
    print({"backend": args.backend, "records": len(indices), "scalar_seconds": scalar_seconds, "read_many_seconds": batch_seconds})
    reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
