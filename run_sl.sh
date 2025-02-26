# uv venv --python=3.8
uv pip install .
uv pip install torch==1.9
python -m distar.bin.sl_train --data ./replay
