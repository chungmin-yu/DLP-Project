#!/bin/bash

nvidia-smi --query-gpu=memory.used --format=csv -i 0 -l 1 > memory_usage

