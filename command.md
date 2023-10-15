### pyenv setup
pyenv install 3.11.5

export PYENV_ROOT="$HOME/.pyenv" export PATH="$PYENV_ROOT/bin:$PATH" eval "$(pyenv init -)"

### manage multiple pythons
pyenv shell 3.11.5

pyenv versions

### poetry setup
poetry config virtualenvs.in-project true
poetry config virtualenvs.prefer-active-python true

export PATH="$HOME/.local/bin:$PATH"

### activate poetry
source .venv/bin/activate 
deactivate

### poetry commands
poetry show
poetry add torch=2.0.0
poetry lock --no-update
poetry install --sync
poetry run pip install flash-attn
