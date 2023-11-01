### pyenv setup

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pyenv install 3.11.6

### manage multiple pythons

pyenv shell 3.11.6

pyenv versions

### poetry setup
export PATH="$HOME/.local/bin:$PATH"

poetry config virtualenvs.in-project true
poetry config virtualenvs.prefer-active-python true

### activate poetry

source .venv/bin/activate
deactivate

### install flash-attn
poetry run pip install flash-attn==2.2.1

git clone [flash-attn]
cd [flash-attn]
PIP_NO_BUILD_ISOLATION=1 pip install -e .

### poetry commands

poetry init
poetry show
poetry add torch=2.0.0
poetry lock --no-update
poetry install --sync
poetry env use 3.11.6


### CUDA
export CUDA_HOME=/usr/local/cuda-11.7
export CUDA_HOME=/opt/conda/

watch -n 0.5 nvidia-smi
gpustat -cup --watch 0.5
https://github.com/wookayin/gpustat

### huggingface cache
/root/.cache/huggingface/hub/

### Linux
cat /etc/os-release

apt update
apt install tmux
tmux

exec bash - switch to bash
exec zsh - Switch to zsh

### training
torchrun --nproc_per_node=auto sagemaker_7b_supervised_fine_tune.py
torchrun --nproc_per_node=auto sagemaker_7b_fine_tune.py
torchrun --nproc_per_node=auto sagemaker_7b_supervised_fine_tune_qlora.py

### inference
python merge_lora_weights_and_save_hf_model.py \
--base_model meta-llama/Llama-2-7b-chat-hf \
--peft_model ./data/download \
--context_size 32768 \
--save_path ./output/Llama-2-7b-supervised-qlora-32k-merged

python inference.py  \
--base_model ./output/Llama-2-7b-supervised-qlora-32k-merged \
--question "What are the main contributions and novelties of this work?" \
--context_size 32768 \
--max_gen_len 512 \
--flash_attn True \
--material "materials/2309.12307.txt"
