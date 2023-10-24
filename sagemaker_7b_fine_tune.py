import os

import huggingface_hub

from fine_tune import train


def main():
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    args = []
    args.extend([
        # "--model_name_or_path", "/media/cwei/WD_BLACK/model_weights/llama2/llama-2-7b/",
        # "--cache_dir", "path_to_cache/",
        "--model_name_or_path", "meta-llama/Llama-2-7b-hf",
        "--output_dir", "./data/download/",
        "--cache_dir", "./data/.cache/",
        "--model_max_length", "8192",
        "--use_flash_attn", "True",
        "--low_rank_training", "True",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "2",
        "--gradient_accumulation_steps", "8",
        "--evaluation_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "1000",
        "--save_total_limit", "2",
        "--learning_rate", "2e-5",
        "--weight_decay", "0.0",
        "--warmup_steps", "20",
        "--lr_scheduler_type", "constant_with_warmup",
        "--logging_steps", "1",
        "--deepspeed", "ds_configs/stage2.json",
        "--max_steps", "1000",
        "--tf32", "True",
        "--bf16", "True",
        "--dataset_num_workers", "16",
    ])
    huggingface_hub.login("xx")
    train(args)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    main()
