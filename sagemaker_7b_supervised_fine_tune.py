import huggingface_hub

from supervised_fine_tune import train


def main():
    args = []
    args.extend([
        "--model_name_or_path", "meta-llama/Llama-2-7b-chat-hf",
        "--output_dir", "./output/7b_supervised_32k",
        # "--cache_dir", "./data/.cache/",
        # "--model_max_length", "32768",
        "--model_max_length", "8192",
        "--use_flash_attn", "True",
        "--low_rank_training", "True",
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "2",
        "--gradient_accumulation_steps", "1",
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
        "--model_type", "llama2",
        #
        # dataset v2
        "--model_max_length", "8192",
        "--gradient_accumulation_steps", "8",
        "--tf32", "True",
        "--bf16", "True",
        "--data_path", "./data/",
        # "--dataloader_num_workers", f"{int(os.cpu_count()//2)}",
        "--dataloader_num_workers", f"{int(os.cpu_count()//3)}",
        # "--dataloader_num_workers", "0",
        #
        # dataset v1
        # "--model_max_length", "8192",
        # "--gradient_accumulation_steps", "8",
        # "--data_path", "LongAlpaca-12k.json",
        # "--dataloader_num_workers", "0"
    ])
    huggingface_hub.login("xx")
    train(args)


if __name__ == "__main__":
    main()
