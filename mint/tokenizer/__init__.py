import copy
from abc import ABC, abstractmethod
from typing import Any
from typing_extensions import deprecated

import tiktoken
import torch


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @property
    @abstractmethod
    def bos_token(self) -> int:
        """Return the beginning-of-sequence token ID."""
        pass

    @property
    @abstractmethod
    def eos_token(self) -> int:
        """Return the end-of-sequence token ID."""
        pass

    @property
    @abstractmethod
    def pad_token(self) -> int:
        """Return the padding token ID."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        pass

    @abstractmethod
    def encode(
        self,
        batch: list[str],
        *,
        add_bos: bool = True,
        add_eos: bool = False,
        padding: bool = True,
        max_length: int | None = None,
    ) -> torch.Tensor:
        """Encode a batch of strings to token IDs."""
        pass

    @abstractmethod
    def encode_to_list(
        self,
        batch: list[str],
        *,
        add_bos: bool = True,
        add_eos: bool = False,
        padding: bool = True,
        max_length: int | None = None,
    ) -> list[Any]:
        """Encode a batch of strings to list of token IDs."""
        pass

    @abstractmethod
    def decode(self, batch: torch.Tensor, *, skip_special_tokens: bool = True) -> list[str]:
        """Decode token IDs back to strings."""
        pass

    @abstractmethod
    def encode_special(self, token: str) -> int:
        """Encode a special token to its ID."""
        pass

    @abstractmethod
    def render_conversation(
        self, conversation: dict, max_tokens: int = 2048
    ) -> tuple[list[int], list[int]]:
        """Render a conversation to token IDs and attention mask."""
        pass


class TikTokenizer(Tokenizer):
    """TikToken-based tokenizer implementation."""

    def __init__(self, model_name: str = "gpt-2") -> None:
        self.encoding = tiktoken.encoding_for_model(model_name=model_name)
        self._bos_token = self.encoding.max_token_value + 1
        self._eos_token = self.encoding.eot_token
        self._pad_token = self.encoding.max_token_value + 2

        self.special_tokens = {
            "<|user_start|>": self.encoding.max_token_value + 3,
            "<|user_end|>": self.encoding.max_token_value + 4,
            "<|assistant_start|>": self.encoding.max_token_value + 5,
            "<|assistant_end|>": self.encoding.max_token_value + 6,
            "<|python_start|>": self.encoding.max_token_value + 7,
            "<|python_end|>": self.encoding.max_token_value + 8,
            "<|output_start|>": self.encoding.max_token_value + 9,
            "<|output_end|>": self.encoding.max_token_value + 10,
        }

    @property
    def bos_token(self) -> int:
        return self._bos_token

    @property
    def eos_token(self) -> int:
        return self._eos_token

    @property
    def pad_token(self) -> int:
        return self._pad_token

    @property
    def vocab_size(self) -> int:
        return self.encoding.max_token_value + 11

    def encode(
        self,
        batch: list[str],
        *,
        add_bos: bool = True,
        add_eos: bool = False,
        padding: bool = True,
        max_length: int | None = None,
    ) -> torch.Tensor:
        return torch.tensor(
            self.encode_to_list(
                batch=batch,
                add_bos=add_bos,
                add_eos=add_eos,
                padding=padding,
                max_length=max_length,
            ),
            dtype=torch.long,
        )

    def encode_to_list(
        self,
        batch: list[str],
        *,
        add_bos: bool = True,
        add_eos: bool = False,
        padding: bool = True,
        max_length: int | None = None,
    ) -> list[Any]:
        encoded = self.encoding.encode_batch(text=batch, disallowed_special=())

        processed = []
        for enc in encoded:
            tokens = enc.copy()

            if max_length is not None:
                available = max_length - int(add_bos) - int(add_eos)
                tokens = tokens[:available]

            if add_bos:
                tokens = [self.bos_token, *tokens]
            if add_eos:
                tokens = [*tokens, self.eos_token]

            processed.append(tokens)

        if padding:
            max_len = max(len(seq) for seq in processed)
            return [seq + [self.pad_token] * (max_len - len(seq)) for seq in processed]
        return list(processed)

    def decode(self, batch: torch.Tensor, *, skip_special_tokens: bool = True) -> list[str]:
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)

        batch_list = batch.tolist()

        if skip_special_tokens:
            special = {
                self.bos_token,
                self.eos_token,
                self.pad_token,
                *self.special_tokens.values(),
            }
            batch_list = [[t for t in seq if t not in special] for seq in batch_list]

        max_valid_token = self.encoding.max_token_value
        batch_list = [[t for t in seq if t <= max_valid_token] for seq in batch_list]

        return self.encoding.decode_batch(batch_list)

    def encode_special(self, token: str) -> int:
        return self.special_tokens[token]

    def render_conversation(
        self, conversation: dict, max_tokens: int = 2048
    ) -> tuple[list[int], list[int]]:
        ids, mask = [], []

        def add_tokens(token_ids: int | list[int], mask_val: int) -> None:
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        messages = conversation["messages"]
        if messages[0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]

        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")

        add_tokens(self.bos_token, 0)

        for _i, message in enumerate(messages):
            content = message["content"]

            if message["role"] == "user":
                value_ids = self.encoding.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    value_ids = self.encoding.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encoding.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                add_tokens(assistant_end, 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

@DeprecationWarning
class HFTokenizer(Tokenizer):
    """HuggingFace tokenizer wrapper."""

    def __init__(self, hf_tokenizer: Any) -> None:  # noqa: ANN401
        """Initialize with a HuggingFace tokenizer.

        Args:
            hf_tokenizer: An instance of transformers.PreTrainedTokenizer or PreTrainedTokenizerFast
        """
        self.hf_tokenizer = hf_tokenizer

        # Ensure pad token exists
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

        # Define special tokens mapping
        self.special_tokens = {
            "<|user_start|>": self._get_or_add_special_token("<|user_start|>"),
            "<|user_end|>": self._get_or_add_special_token("<|user_end|>"),
            "<|assistant_start|>": self._get_or_add_special_token("<|assistant_start|>"),
            "<|assistant_end|>": self._get_or_add_special_token("<|assistant_end|>"),
            "<|python_start|>": self._get_or_add_special_token("<|python_start|>"),
            "<|python_end|>": self._get_or_add_special_token("<|python_end|>"),
            "<|output_start|>": self._get_or_add_special_token("<|output_start|>"),
            "<|output_end|>": self._get_or_add_special_token("<|output_end|>"),
        }

    def _get_or_add_special_token(self, token: str) -> int:
        """Get token ID or add it as a special token if not present."""
        token_id = self.hf_tokenizer.convert_tokens_to_ids(token)
        if token_id == self.hf_tokenizer.unk_token_id:
            # Token doesn't exist, add it
            self.hf_tokenizer.add_special_tokens({"additional_special_tokens": [token]})
            token_id = self.hf_tokenizer.convert_tokens_to_ids(token)
        return token_id

    @property
    def bos_token(self) -> int:
        return self.hf_tokenizer.bos_token_id or self.hf_tokenizer.eos_token_id

    @property
    def eos_token(self) -> int:
        return self.hf_tokenizer.eos_token_id

    @property
    def pad_token(self) -> int:
        return self.hf_tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.hf_tokenizer)

    def encode(
        self,
        batch: list[str],
        *,
        add_bos: bool = True,
        add_eos: bool = False,
        padding: bool = True,
        max_length: int | None = None,
    ) -> torch.Tensor:
        return torch.tensor(
            self.encode_to_list(
                batch=batch,
                add_bos=add_bos,
                add_eos=add_eos,
                padding=padding,
                max_length=max_length,
            ),
            dtype=torch.long,
        )

    def encode_to_list(
        self,
        batch: list[str],
        *,
        add_bos: bool = True,
        add_eos: bool = False,
        padding: bool = True,
        max_length: int | None = None,
    ) -> list[Any]:
        # Use HF tokenizer's batch encoding
        encoded = self.hf_tokenizer(
            batch,
            add_special_tokens=False,  # We'll add them manually
            padding=False,  # We'll handle padding ourselves
            truncation=False,
            return_attention_mask=False,
        )

        processed = []
        for token_ids in encoded["input_ids"]:
            tokens = token_ids.copy() if isinstance(token_ids, list) else token_ids.tolist()

            if max_length is not None:
                available = max_length - int(add_bos) - int(add_eos)
                tokens = tokens[:available]

            if add_bos:
                tokens = [self.bos_token, *tokens]
            if add_eos:
                tokens = [*tokens, self.eos_token]

            processed.append(tokens)

        if padding:
            max_len = max(len(seq) for seq in processed)
            return [seq + [self.pad_token] * (max_len - len(seq)) for seq in processed]
        return list(processed)

    def decode(self, batch: torch.Tensor, *, skip_special_tokens: bool = True) -> list[str]:
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)

        batch_list = batch.tolist()
        return self.hf_tokenizer.batch_decode(batch_list, skip_special_tokens=skip_special_tokens)

    def encode_special(self, token: str) -> int:
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.hf_tokenizer.convert_tokens_to_ids(token)

    def render_conversation(
        self, conversation: dict, max_tokens: int = 2048
    ) -> tuple[list[int], list[int]]:
        ids, mask = [], []

        def add_tokens(token_ids: int | list[int], mask_val: int) -> None:
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        messages = conversation["messages"]
        if messages[0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]

        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")

        add_tokens(self.bos_token, 0)

        for _i, message in enumerate(messages):
            content = message["content"]

            if message["role"] == "user":
                value_ids = self.hf_tokenizer.encode(content, add_special_tokens=False)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    value_ids = self.hf_tokenizer.encode(content, add_special_tokens=False)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.hf_tokenizer.encode(part["text"], add_special_tokens=False)
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                add_tokens(assistant_end, 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask
