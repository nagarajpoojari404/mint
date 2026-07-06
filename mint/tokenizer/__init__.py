import copy
from abc import ABC, abstractmethod
from typing import Any

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


class HFTokenizer(Tokenizer):
    """HuggingFace tokenizer wrapper.

    Uses the model's own native special tokens (derived from its chat_template)
    for conversation structure — no new tokens are ever added, so the embedding
    table is never resized and the model can leverage its pre-trained understanding
    of turn boundaries from day one.

    Supported chat-template families (auto-detected):
      - ChatML  : Qwen, SmolLM, Phi-3, Yi  (<|im_start|> / <|im_end|>)
      - Llama-3 : Meta-Llama-3              (<|start_header_id|> / <|eot_id|>)
      - Gemma   : google/gemma              (<start_of_turn> / <end_of_turn>)
      - Mistral : Mistralai                 ([INST] / [/INST])
      - Fallback: renders a minimal 2-message conversation and scrapes the
                  boundary strings directly from the chat_template output.
    """

    _ROLE_USER = "user"
    _ROLE_ASSISTANT = "assistant"

    def __init__(self, hf_tokenizer: Any) -> None:  # noqa: ANN401
        self.hf_tokenizer = hf_tokenizer

        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

        # _role_start_ids / _role_end_ids map role → list[int]
        # (list because some formats emit multiple tokens for a boundary,
        #  e.g. Llama-3: "<|start_header_id|>" + "assistant" + "<|end_header_id|>")
        self._role_start_ids: dict[str, list[int]] = {}
        self._role_end_ids: dict[str, list[int]] = {}
        self._detect_chat_format()

    # ── Chat-format detection ──────────────────────────────────────────────────

    def _tok_ids(self, text: str) -> list[int]:
        """Encode text without BOS/EOS added."""
        return self.hf_tokenizer.encode(text, add_special_tokens=False)

    def _detect_chat_format(self) -> None:
        """Populate _role_start_ids / _role_end_ids from the model's own vocabulary."""
        tmpl = getattr(self.hf_tokenizer, "chat_template", None) or ""

        # ── ChatML  (Qwen, SmolLM, Phi-3, Yi, …) ─────────────────────────────
        if "<|im_start|>" in tmpl:
            im_start = self._tok_ids("<|im_start|>")
            im_end = self._tok_ids("<|im_end|>")
            user_role = self._tok_ids("user")
            assistant_role = self._tok_ids("assistant")
            nl = self._tok_ids("\n")
            self._role_start_ids[self._ROLE_USER] = im_start + user_role + nl
            self._role_start_ids[self._ROLE_ASSISTANT] = im_start + assistant_role + nl
            self._role_end_ids[self._ROLE_USER] = im_end + nl
            self._role_end_ids[self._ROLE_ASSISTANT] = im_end + nl

        # ── Llama-3  (<|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>) ──
        elif "<|start_header_id|>" in tmpl:
            soh = self._tok_ids("<|start_header_id|>")
            eoh = self._tok_ids("<|end_header_id|>")
            eot = self._tok_ids("<|eot_id|>")
            user_role = self._tok_ids("user")
            assistant_role = self._tok_ids("assistant")
            nl2 = self._tok_ids("\n\n")
            self._role_start_ids[self._ROLE_USER] = soh + user_role + eoh + nl2
            self._role_start_ids[self._ROLE_ASSISTANT] = soh + assistant_role + eoh + nl2
            self._role_end_ids[self._ROLE_USER] = eot
            self._role_end_ids[self._ROLE_ASSISTANT] = eot

        # ── Gemma  (<start_of_turn>role\n…<end_of_turn>\n) ───────────────────
        elif "<start_of_turn>" in tmpl:
            sot = self._tok_ids("<start_of_turn>")
            eot = self._tok_ids("<end_of_turn>")
            user_role = self._tok_ids("user")
            assistant_role = self._tok_ids("model")  # Gemma uses "model" not "assistant"
            nl = self._tok_ids("\n")
            self._role_start_ids[self._ROLE_USER] = sot + user_role + nl
            self._role_start_ids[self._ROLE_ASSISTANT] = sot + assistant_role + nl
            self._role_end_ids[self._ROLE_USER] = eot + nl
            self._role_end_ids[self._ROLE_ASSISTANT] = eot + nl

        # ── Mistral  ([INST]…[/INST]) ─────────────────────────────────────────
        elif "[INST]" in tmpl:
            inst_open = self._tok_ids("[INST]")
            inst_close = self._tok_ids("[/INST]")
            self._role_start_ids[self._ROLE_USER] = inst_open
            self._role_start_ids[self._ROLE_ASSISTANT] = inst_close
            self._role_end_ids[self._ROLE_USER] = []
            self._role_end_ids[self._ROLE_ASSISTANT] = [self.eos_token]

        # ── Fallback: scrape from a rendered 2-turn example ───────────────────
        else:
            self._detect_chat_format_fallback()

    def _detect_chat_format_fallback(self) -> None:
        """Last-resort: render a known conversation and find delimiters by sentinel."""
        sentinel_user = "XUSERTEXTX"
        sentinel_asst = "XASSTTEXTX"
        try:
            rendered = self.hf_tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": sentinel_user},
                    {"role": "assistant", "content": sentinel_asst},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            nl = self._tok_ids("\n")
            self._role_start_ids[self._ROLE_USER] = nl
            self._role_start_ids[self._ROLE_ASSISTANT] = nl
            self._role_end_ids[self._ROLE_USER] = nl
            self._role_end_ids[self._ROLE_ASSISTANT] = [self.eos_token]
            return

        before_user, after_user = rendered.split(sentinel_user, 1)
        between, after_asst = after_user.split(sentinel_asst, 1)

        self._role_start_ids[self._ROLE_USER] = self._tok_ids(before_user) if before_user else []
        self._role_end_ids[self._ROLE_USER] = self._tok_ids(between) if between else []
        self._role_start_ids[self._ROLE_ASSISTANT] = self._tok_ids(between) if between else []
        self._role_end_ids[self._ROLE_ASSISTANT] = (
            self._tok_ids(after_asst) if after_asst else [self.eos_token]
        )

    # ── Tokenizer ABC properties ───────────────────────────────────────────────

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
        """Map a logical role-boundary name to its first native token ID.

        Accepts the legacy mint marker names (``<|assistant_start|>`` etc.) used
        by ``_log_sample_predictions`` and maps them to the model's real native
        token. No new tokens are ever created.
        """
        ids = self.encode_special_ids(token)
        if ids:
            return ids[0]
        return self.hf_tokenizer.convert_tokens_to_ids(token)

    def encode_special_ids(self, token: str) -> list[int]:
        """Return the full multi-token boundary sequence for a logical role marker.

        Unlike ``encode_special``, which returns only the first token ID, this
        returns the complete list of token IDs that make up the boundary.  Use
        this whenever you need to *locate* a boundary in a token stream, so that
        boundaries whose first token is shared with other roles (e.g. ChatML
        ``<|im_start|>`` used by both user and assistant turns) are matched
        correctly via their full sequence rather than a single ambiguous token.
        """
        _LEGACY: dict[str, tuple[str, str]] = {
            "<|user_start|>":      (self._ROLE_USER,      "start"),
            "<|user_end|>":        (self._ROLE_USER,       "end"),
            "<|assistant_start|>": (self._ROLE_ASSISTANT, "start"),
            "<|assistant_end|>":   (self._ROLE_ASSISTANT,  "end"),
        }
        if token in _LEGACY:
            role, boundary = _LEGACY[token]
            mapping = self._role_start_ids if boundary == "start" else self._role_end_ids
            return mapping.get(role, [])
        return [self.hf_tokenizer.convert_tokens_to_ids(token)]

    def render_conversation(
        self, conversation: dict, max_tokens: int = 2048
    ) -> tuple[list[int], list[int]]:
        """Render a conversation dict to (token_ids, loss_mask).

        Uses the model's own native boundary tokens — no custom tokens added.
        Loss mask is 1 only on assistant content + turn-end marker, 0 elsewhere.
        """
        ids: list[int] = []
        mask: list[int] = []

        def add(token_ids: int | list[int], mask_val: int) -> None:
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        messages = conversation["messages"]

        # Fold a leading system message into the first user turn
        if messages and messages[0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]

        add(self.bos_token, 0)

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == self._ROLE_USER:
                add(self._role_start_ids[role], 0)
                add(self.hf_tokenizer.encode(content, add_special_tokens=False), 0)
                add(self._role_end_ids[role], 0)

            elif role == self._ROLE_ASSISTANT:
                add(self._role_start_ids[role], 0)  # boundary header — not predicted
                if isinstance(content, str):
                    add(self.hf_tokenizer.encode(content, add_special_tokens=False), 1)
                elif isinstance(content, list):
                    for part in content:
                        part_ids = self.hf_tokenizer.encode(part["text"], add_special_tokens=False)
                        # python_output is model-received, not generated
                        mask_val = 0 if part.get("type") == "python_output" else 1
                        add(part_ids, mask_val)
                add(self._role_end_ids[role], 1)  # turn-end token IS predicted

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask
