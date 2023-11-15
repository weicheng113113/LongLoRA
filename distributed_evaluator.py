import inspect
from abc import ABC, abstractmethod
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


class EvalMetric(ABC):
    @abstractmethod
    def add(self, logits: torch.FloatTensor, labels: torch.LongTensor, model_output: object) -> dict[str, object]:
        pass

    @abstractmethod
    def compute(self) -> dict[str, object]:
        pass


class DistributedEvaluator:
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module],
                 batch_size: int,
                 refresh_rate: int,
                 gpu_id: int):
        self.gpu_id = gpu_id
        print(f"self.gpu_id: {self.gpu_id}")
        self.batch_size = batch_size
        self.refresh_rate = refresh_rate

        self.model = DDP(model, device_ids=[self.gpu_id])

    def evaluate(self, dataset: Dataset, metric: EvalMetric) -> dict[str, object]:
        data_loader = self._prepare_dataloader(dataset)
        self.model.eval()
        with torch.no_grad():
            if self.is_first_device():
                data_loader = tqdm(data_loader)
            for i, example_dict in enumerate(data_loader):
                sig = inspect.signature(self.model.forward)
                used = set(list(sig.parameters.keys()) + ["input_ids", "labels"])
                inputs = {key: example_dict[key].to(self.gpu_id) for key in used if key in example_dict}
                outputs = self.model(**inputs)
                metric_result = metric.add(logits=outputs["logits"], labels=inputs["labels"], model_output=outputs)

                if self.is_first_device() and (i % self.refresh_rate == 0):
                    data_loader.set_postfix(metric_result)
            return metric.compute()

    def is_first_device(self):
        return self.gpu_id == 0

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset)
        )
