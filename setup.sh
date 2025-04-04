# initialize venv
uv venv --seed
uv pip install pip
source .venv/bin/activate

# setup packages
uv sync --refresh
uv run python3 -V

# install spacy models
uv run python3 -m spacy download en_core_web_sm
uv run python3 -m spacy download en_core_web_lg


