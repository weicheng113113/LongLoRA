import os

import huggingface_hub

from supervised_fine_tune import train


def main():
    args = []
    args.extend([
        "--model_name_or_path", "meta-llama/Llama-2-7b-chat-hf",
        "--output_dir", "/media/cwei/WD_BLACK/model_weights/longlora-llama-2-7b-chat-hf",
        "--cache_dir", "/media/cwei/WD_BLACK/model_weights/llama-2-7b-chat-hf",
        # "--model_max_length", "32768",
        "--model_max_length", "8192",
        # "--use_flash_attn", "True",
        "--data_path", "LongAlpaca-12k.json",
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
        # "--tf32", "True",
        # "--bf16", "True",
        # "--tf32", "False",
        # "--fp16", "True",
        "--use_flash_attn", "False",
        "--model_type", "llama2",
    ])
    huggingface_hub.login("xx")
    train(args)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    main()
