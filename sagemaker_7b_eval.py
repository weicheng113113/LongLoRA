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
        "--peft_model", "/home/ubuntu/models/Llama-2-7b-longlora-8k",
        # 8
        # {'val_acc': 0.4987945556640625, 'val_loss': 2.3734896183013916, 'val_perplexity': 10.734770170795082, 'val_perplexity_per_chunk': tensor([10.7348])}
        # 30
        # {'val_acc': 0.507311999797821, 'val_loss': 2.3371968269348145, 'val_perplexity': 10.352160633794169, 'val_perplexity_per_chunk': tensor([10.3522])}
        # 100; used 4:38
        # {'val_acc': 0.5085864067077637, 'val_loss': 2.3561558723449707, 'val_perplexity': 10.550299914896874, 'val_perplexity_per_chunk': tensor([10.5503])}
        "--data_path", "./datasets/pg19/test.bin",
        "--cache_dir", "./data/.cache",
        "--flash_attn", "True",
    ])
    huggingface_hub.login("xx")

    main(parse_config(args))


if __name__ == "__main__":
    run_eval()
