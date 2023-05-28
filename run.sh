#!/bin/bash

# no distil (just stack AB)
python3 main_informer.py
# no distil + passthrough
python3 main_informer.py --passthrough
# distil + stack(replica)
python3 main_informer.py --distil --model informerstack
# distil + passthrough
python3 main_informer.py --distil --passthrough

