import random
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

from mint.data.dataloader import DataloaderConfig, DistributedDataloader
from mint.tokenizer import Tokenizer
from mint.utils.device import Device


class ContentPart(BaseModel):
    type: Literal["text", "python", "python_output"]
    text: str


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class Row(BaseModel):
    prompt: list[Message]
    chosen: list[Message]
    rejected: list[Message]


class Batch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    chosen_input_ids: torch.Tensor
    chosen_attn_mask: torch.Tensor
    chosen_labels: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attn_mask: torch.Tensor
    rejected_labels: torch.Tensor


class DistributedDPODataloader(DistributedDataloader):
    def __init__(
        self,
        device: Device,
        config: DataloaderConfig,
        tokenizer: Tokenizer,
        filename: str,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(
            device,
            config.data_dir,
            config.batch_size,
            config.seq_length,
            tokenizer,
            *args,
            **kwargs,
        )
        self.device = device
        self.filepath = Path(config.data_dir) / Path(filename)

        proc_info = device.process_info()
        self.rank = proc_info["rank"]
        self.world_size = proc_info["world_size"]

        assert Path(self.filepath).exists()

    def _iter_rows(self) -> Iterator[Row]:
        with Path(self.filepath).open("r", encoding="utf-8") as file:
            for idx, line in enumerate(file):
                if idx % self.world_size != self.rank:
                    continue

                clean_line = line.strip()
                if not clean_line:
                    continue

                yield Row.model_validate_json(clean_line)

    def _collate_batch(self, rows: list[Row]) -> Batch:
        chosen_ids_list = []
        chosen_attn_mask_list = []
        chosen_labels_list = []

        rejected_ids_list = []
        rejected_attn_mask_list = []
        rejected_labels_list = []

        for row in rows:
            chosen_conv = {"messages": [m.model_dump() for m in (row.prompt + row.chosen)]}
            rejected_conv = {"messages": [m.model_dump() for m in (row.prompt + row.rejected)]}

            c_ids, c_mask = self.tokenizer.render_conversation(chosen_conv, max_tokens=self.T)
            r_ids, r_mask = self.tokenizer.render_conversation(rejected_conv, max_tokens=self.T)

            c_labels = [tid if m == 1 else -100 for tid, m in zip(c_ids, c_mask, strict=False)]
            r_labels = [tid if m == 1 else -100 for tid, m in zip(r_ids, r_mask, strict=False)]

            chosen_ids_list.append(torch.tensor(c_ids, dtype=torch.long))
            chosen_attn_mask_list.append(torch.tensor([1] * len(c_ids), dtype=torch.bool))
            chosen_labels_list.append(torch.tensor(c_labels, dtype=torch.long))

            rejected_ids_list.append(torch.tensor(r_ids, dtype=torch.long))
            rejected_attn_mask_list.append(torch.tensor([1] * len(r_ids), dtype=torch.bool))
            rejected_labels_list.append(torch.tensor(r_labels, dtype=torch.long))

        # Combine all sequences to find the maximum length across both chosen and rejected
        all_sequences = chosen_ids_list + rejected_ids_list
        max_len = max(seq.size(0) for seq in all_sequences)

        def pad_and_move(sequences, pad_value) -> torch.Tensor:
            # Pad all sequences to the same max_len
            padded = []
            for seq in sequences:
                if seq.size(0) < max_len:
                    padding = torch.full((max_len - seq.size(0),), pad_value, dtype=seq.dtype)
                    seq = torch.cat([seq, padding])
                padded.append(seq)
            return torch.stack(padded).to(self.device.device)

        return Batch(
            chosen_input_ids=pad_and_move(chosen_ids_list, self.tokenizer.pad_token),
            chosen_attn_mask=pad_and_move(chosen_attn_mask_list, 0),
            chosen_labels=pad_and_move(chosen_labels_list, -100),
            rejected_input_ids=pad_and_move(rejected_ids_list, self.tokenizer.pad_token),
            rejected_attn_mask=pad_and_move(rejected_attn_mask_list, 0),
            rejected_labels=pad_and_move(rejected_labels_list, -100),
        )

    def batch_loader(self, split="train", resume_state=None) -> Iterator[Batch]:  # noqa: ANN001, ARG002
        current_batch_rows = []

        for row in self._iter_rows():
            current_batch_rows.append(row)

            if len(current_batch_rows) == self.B:
                yield self._collate_batch(current_batch_rows)
                current_batch_rows = []

    def get_state(self):  # noqa: ANN201
        return NotImplementedError()

    def set_state(self, state):  # noqa: ANN001, ANN201, ARG002
        return NotImplementedError()

    def sample(self, num_samples: int = 1) -> list[dict]:
        samples = []

        all_rows = list(self._iter_rows())

        if not all_rows:
            return samples

        for _ in range(num_samples):
            row = random.choice(all_rows)  # noqa: S311

            chosen_conv = {"messages": [m.model_dump() for m in (row.prompt + row.chosen)]}
            c_ids, c_mask = self.tokenizer.render_conversation(chosen_conv, max_tokens=self.T)

            rejected_conv = {"messages": [m.model_dump() for m in (row.prompt + row.rejected)]}
            r_ids, r_mask = self.tokenizer.render_conversation(rejected_conv, max_tokens=self.T)

            # Truncate if needed
            if len(c_ids) > self.T:
                c_ids = c_ids[: self.T]
                c_mask = c_mask[: self.T]

            if len(r_ids) > self.T:
                r_ids = r_ids[: self.T]
                r_mask = r_mask[: self.T]

            chosen_tokens = torch.tensor(c_ids, dtype=torch.long)
            rejected_tokens = torch.tensor(r_ids, dtype=torch.long)

            chosen_str = self.tokenizer.decode(
                chosen_tokens.unsqueeze(0), skip_special_tokens=True
            )[0]
            rejected_str = self.tokenizer.decode(
                rejected_tokens.unsqueeze(0), skip_special_tokens=True
            )[0]

            samples.append(
                {
                    "chosen_tokens": chosen_tokens,
                    "rejected_tokens": rejected_tokens,
                    "chosen_str": chosen_str,
                    "rejected_str": rejected_str,
                    "chosen_mask": c_mask,
                    "rejected_mask": r_mask,
                }
            )

        return samples
