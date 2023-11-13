from eval_distributed_qlora import main
import huggingface_hub


def run_eval():
    args = []
    args.extend([
        "--eval_accumulation_steps", "1",
        "--output_dir", "./output/tmp",
        "--seq_len", "8192",
        "--context_size", "8192",
        "--per_device_eval_batch_size", "1",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        # "--peft_model", "./output/7b_qlora_8k/checkpoint-1000/",
        # {'eval_loss': 11.007083892822266, 'eval_accuracy': 0.5008697509765625, 'eval_perplexity': 60299.7890625, 'eval_runtime': 29.6625, 'eval_samples_per_second': 0.27, 'eval_steps_per_second': 0.27}

        "--peft_model", "/home/ubuntu/models/Llama-2-7b-longlora-8k",
        # 8
        # {'eval_loss': 2.3734896183013916, 'eval_accuracy': 0.49877914784519595, 'eval_perplexity': 10.734786987304688, 'eval_runtime': 49.87, 'eval_samples_per_second': 0.16, 'eval_steps_per_second': 0.16}
        # 30
        # {'eval_loss': 2.3371965885162354, 'eval_accuracy': 0.5073006958857282, 'eval_perplexity': 10.352173805236816, 'eval_runtime': 164.1781, 'eval_samples_per_second': 0.183, 'eval_steps_per_second': 0.183}
        # 100; 4gpus; used 13:01
        # {'eval_loss': 2.356156349182129, 'eval_accuracy': 0.5085825906482725, 'eval_perplexity': 10.550321578979492, 'eval_runtime': 818.0428, 'eval_samples_per_second': 0.122, 'eval_steps_per_second': 0.031}
        "--data_path", "./datasets/pg19/test.bin",
        "--cache_dir", "./data/.cache",
        "--flash_attn", "True",
    ])
    huggingface_hub.login("xx")

    main(args)


if __name__ == "__main__":
    run_eval()
