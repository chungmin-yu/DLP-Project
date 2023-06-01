#!/bin/bash

# no distil (just stack AB)
python3 main_informer.py
# no distil + passthrough
python3 main_informer.py --passthrough
# distil
python3 main_informer.py --distil
# distil + stack(replica)
python3 main_informer.py --distil --model informerstack
# distil + passthrough
python3 main_informer.py --distil --passthrough

