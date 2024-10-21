#!/bin/bash
python3 ../../scripts/generate_wideband.py --root=. --impaired --num-iq-samples=1048576 --num-workers=16 --batch-size=16
