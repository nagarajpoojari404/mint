import copy
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from mint.data.datasets.base import SFTEvalDataset
from mint.eval.base import EvalConfig, Evaluator
from mint.nn.base import LogitsWrapper
from mint.tokenizer import Tokenizer
from mint.utils.device import Device


class ChatCoreEvaluator(Evaluator):
    def __init__(
        self,
        model: nn.Module,
        config: EvalConfig,
        tokenizer: Tokenizer,
        device: Device,
        datasets: list[SFTEvalDataset],
    ) -> None:
        super().__init__(LogitsWrapper(model), config, device)
        self.tokenizer = tokenizer
        self.datasets = datasets

    def evaluate(self, num_examples: int | None = None) -> dict[str, Any]:
        self.model.eval()
        eval_start_time = time.perf_counter()
        results = {}

        with torch.no_grad():
            for dataset in self.datasets:
                results[dataset.__class__.__name__] = self._evaluate_dataset(dataset, num_examples)

        self.device.synchronize()
        eval_time = time.perf_counter() - eval_start_time

        if not results:
            return {"tasks": {}, "accuracy": 0.0, "eval_time_sec": eval_time}

        avg_accuracy = sum(r["accuracy"] for r in results.values()) / len(results)
        return {
            "tasks": results,
            "accuracy": avg_accuracy,
            "eval_time_sec": eval_time,
        }

    def _evaluate_dataset(self, dataset: SFTEvalDataset, num_examples: int | None = None):  # noqa: ANN202
        total = len(dataset) if num_examples is None else min(num_examples, len(dataset))

        rank = self.process_info["rank"]
        world_size = self.process_info["world_size"]

        correct = 0
        count = 0

        for idx in range(rank, total, world_size):
            conversation = dataset[idx]
            completion = self._generate_completion(conversation)
            print("*" * 50)
            print(completion)
            if dataset.evaluate(conversation, completion):
                correct += 1
            count += 1

        if self.process_info["distributed"]:
            correct_tensor = torch.tensor(correct, dtype=torch.float32, device=self.device.device)
            count_tensor = torch.tensor(count, dtype=torch.float32, device=self.device.device)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            correct = correct_tensor.item()
            count = count_tensor.item()

        accuracy = correct / count if count > 0 else 0.0
        return {"accuracy": accuracy, "num_examples": int(count)}

    def _generate_completion(self, conversation: dict) -> str:
        conversation = copy.deepcopy(conversation)
        conversation["messages"].pop()  # remove ground-truth assistant turn

        ids, _ = self.tokenizer.render_conversation(conversation, self.config.seq_length - 1)
        assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        ids.append(assistant_start)

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device.device)

        output_ids = self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=self.tokenizer.pad_token,
            max_new_tokens=self.config.max_new_tokens,
            eos_token_id=assistant_end,
            do_sample=False,
            use_cache=True,
        )

        generated = output_ids[0, input_ids.shape[1] :].tolist()
        return self.tokenizer.decode(torch.tensor(generated))[0]
