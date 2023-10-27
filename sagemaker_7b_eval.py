import os

from eval import main, parse_config
import huggingface_hub


def run_eval():
    args = []
    args.extend([
        "--seq_len", "8192",
        "--context_size", "8192",
        "--batch_size", "1",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--peft_model", "Yukang/Llama-2-7b-longlora-8k",
        "--data_path", "./datasets/pg19/test.bin",
        "--cache_dir", "./data/.cache",
        "--flash_attn", "True",
    ])
    huggingface_hub.login("xx")

    main(parse_config(args))


if __name__ == "__main__":
    run_eval()