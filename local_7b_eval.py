import os

from eval import main, parse_config
import huggingface_hub


def run_eval():
    args = []
    args.extend([
        "--seq_len", "8192",
        "--context_size", "8192",
        "--batch_size", "1",
        "--base_model", "/media/cwei/WD_BLACK/model_weights/llama-2-7b-hf/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9/",
        "--peft_model", "Yukang/Llama-2-7b-longlora-8k",
        "--data_path", "/media/cwei/WD_BLACK/datasets/pg19/test.bin",
        # "--output_dir", "/media/cwei/WD_BLACK/model_weights/longlora-llama-2-7b-hf",
        "--cache_dir", "./data/.cache",
        # "--flash_attn", "False",
    ])
    huggingface_hub.login("xx")

    main(parse_config(args))


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    run_eval()
