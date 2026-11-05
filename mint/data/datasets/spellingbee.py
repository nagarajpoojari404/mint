import random
import re
import urllib.request
from pathlib import Path
from typing import Any, Literal

from mint.data.datasets.base import SFTEvalDataset, SFTTrainDataset


LETTERS = "abcdefghijklmnopqrstuvwxyz"
WORD_LIST_URL = (
    "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
)
TEST_SEED_OFFSET = 10_000_000

USER_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
]


class SpellingBee(SFTTrainDataset, SFTEvalDataset):
    ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    def __init__(self, size: int = 1000, split: Literal["train", "test"] = "train") -> None:
        super().__init__()
        self.size = size
        self.split = split
        self.words = self._load_words()

    @property
    def eval_type(self) -> Literal["categorical", "generative"]:
        return "generative"

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        seed = index if self.split == "train" else TEST_SEED_OFFSET + index
        rng = random.Random(seed)  # noqa: S311

        word = rng.choice(self.words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)
        count = word.count(letter)

        user_msg = self._create_user_message(rng, word, letter)
        assistant_parts = self._create_assistant_parts(word, letter, count)

        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_parts},
            ]
        }

    def evaluate(self, conversation: dict[str, Any], completion: str) -> bool:
        ground_truth_parts = conversation["messages"][-1]["content"]
        ground_truth_text = ground_truth_parts[-1]["text"]

        ref_num = self._extract_answer(ground_truth_text)
        pred_num = self._extract_answer(completion)
        return pred_num == ref_num

    def _load_words(self) -> list:

        cache_path = "/tmp/words_alpha.txt"  # noqa: S108
        if not Path(cache_path).exists():
            urllib.request.urlretrieve(WORD_LIST_URL, cache_path)

        with Path(cache_path).open() as f:
            return [line.strip() for line in f]

    def _create_user_message(self, rng: random.Random, word: str, letter: str) -> str:
        template = rng.choice(USER_TEMPLATES)
        if rng.random() < 0.3:
            template = template.lower()

        quote = rng.choice(["", "'", '"'])
        msg = template.format(letter=f"{quote}{letter}{quote}", word=f"{quote}{word}{quote}")
        return msg + "?" if rng.random() < 0.5 else msg

    def _create_assistant_parts(self, word: str, letter: str, count: int) -> list:
        parts = []
        word_letters = ",".join(list(word))

        manual_text = f"Let me spell '{word}': {word_letters}\n\nCounting '{letter}':\n"
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"
        manual_text += f"\nManual count: {running_count}"

        parts.append({"type": "text", "text": manual_text})
        parts.append({"type": "text", "text": "\n\nVerifying with Python:\n\n"})
        parts.append({"type": "python", "text": f"'{word}'.count('{letter}')"})
        parts.append({"type": "python_output", "text": str(count)})
        parts.append({"type": "text", "text": f"\n\nFinal answer:\n\n#### {count}"})

        return parts

    def _extract_answer(self, text: str) -> str | None:
        match = self.ANSWER_RE.search(text)
        return match.group(1).strip().replace(",", "") if match else None


class SimpleSpelling(SFTTrainDataset):
    def __init__(self, size: int = 1000, split: Literal["train", "test"] = "train") -> None:
        super().__init__()
        self.size = size
        self.split = split
        self.words = self._load_words()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        seed = index if self.split == "train" else TEST_SEED_OFFSET + index
        rng = random.Random(seed)  # noqa: S311
        word = rng.choice(self.words)

        return {
            "messages": [
                {"role": "user", "content": f"Spell the word: {word}"},
                {"role": "assistant", "content": f"{word}:{','.join(list(word))}"},
            ]
        }

    def _load_words(self) -> list:

        cache_path = "/tmp/words_alpha.txt"  # noqa: S108
        if not Path(cache_path).exists():
            urllib.request.urlretrieve(WORD_LIST_URL, cache_path)

        with Path(cache_path).open() as f:
            words = [line.strip() for line in f]

        rng = random.Random(42)  # noqa: S311
        rng.shuffle(words)
        return words
