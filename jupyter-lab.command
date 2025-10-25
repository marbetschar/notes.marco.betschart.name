#!/bin/bash

# Mac OS: change workdir to script's dir
# https://stackoverflow.com/questions/5125907/how-to-run-a-shell-script-in-os-x-by-double-clicking
cd -- "$(dirname "$0")"

source ./.venv/bin/activate
jupyter lab
open "http://localhost:8888"