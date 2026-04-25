#!/bin/zsh

cd /Users/kimscarabello/Desktop/Repos/equity-momentum-rotation || exit 1

source .venv/bin/activate

python3 -m live.run_alpaca_live_trader --live --verbose

