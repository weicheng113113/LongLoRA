import copy
import os
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
import transformers
from functools import partial
from torchdata.datapipes.iter import FileOpener, IterableWrapper, IterDataPipe
from torchtext._download_hooks import HttpReader

from supervised_dataset_v1 import SupervisedDatasetV1

URL = {
    "train": "https://huggingface.co/datasets/Yukang/LongAlpaca-12k/resolve/main/LongAlpaca-12k.json"
}

MD5 = {
    "train": "ea66002d453098877d9aa63ca690f37a",
}

IGNORE_INDEX = -100


class SupervisedDataPipe(IterDataPipe):
    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe
        self.length = sum([len(stream) for _, stream in self.source_datapipe])

    def __iter__(self):
        for _, stream in self.source_datapipe:
            for row in stream:
                yield row

    def __len__(self):
        return self.length


class SupervisedDatasetV2(IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_dir: str, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer

        self.source_datapipe = self._data_pipe_and_length(split="train")

    def _data_pipe_and_length(self, split: str) -> SupervisedDataPipe:
        url_dp = IterableWrapper([URL[split]])
        cache_dp = url_dp.on_disk_cache(
            filepath_fn=partial(self._filepath_fn, split),
            hash_dict={self._filepath_fn(split): MD5[split]},
            hash_type="md5",
        )
        cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)
        cache_dp = FileOpener(cache_dp, encoding="utf-8")
        # return SupervisedDataPipe(cache_dp.parse_json_files()).map(self._preprocess).shuffle().set_shuffle(False).sharding_filter()
        return SupervisedDataPipe(cache_dp.parse_json_files()).map(self._preprocess).sharding_filter()

    def _filepath_fn(self, split: str, _=None):
        return os.path.join(self.data_dir, os.path.basename(URL[split]))

    def _preprocess(self, row):
        source = row["instruction"]
        target = f"{row['output']}{self.tokenizer.eos_token}"
        example = source + target

        example_tokenized = self.tokenizer(
            example,
            return_tensors="pt",
            # padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_tokenized = self.tokenizer(
            source,
            return_tensors="pt",
            # padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        input_id = example_tokenized.input_ids[0]
        label = copy.deepcopy(input_id)
        source_len = source_tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()

        label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_id, labels=label, instruction=source, output=target)

    def __len__(self):
        return len(self.source_datapipe)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for example in self.source_datapipe:
            yield example


def main():
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    cache_dir = "./data/.cache/"
    model_max_length = 8192

    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    train_data_v2 = SupervisedDatasetV2(data_dir=cache_dir, tokenizer=tokenizer)
    print(f"length: {len(train_data_v2)}")

    first_number_examples_to_check = 500
    train_data_v1 = SupervisedDatasetV1(tokenizer=tokenizer, data_path="./LongAlpaca-12k.json", num_examples=first_number_examples_to_check)
    it_v2 = iter(train_data_v2)
    for i in range(len(train_data_v1)):
        example_v1 = train_data_v1[i]
        example_v2 = next(it_v2)
        torch.testing.assert_close(example_v1["input_ids"], example_v2["input_ids"])
        torch.testing.assert_close(example_v1["labels"], example_v2["labels"])


    # print((example["labels"] == -100).sum())
    # example = next(myit)
    # print((example["labels"] == -100).sum())


if __name__ == "__main__":
    main()
