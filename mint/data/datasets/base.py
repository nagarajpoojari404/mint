from abc import abstractmethod
from typing import Any, Literal


class SFTDataset:
    def __init__(self) -> None:
        super().__init__()
        self.index = 0

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]: ...


class SFTTrainDataset(SFTDataset):
    def next(self) -> dict[str, Any]:
        example = self[self.index]
        self.index = (self.index + 1) % len(self)
        return example

    def reset(self) -> None:
        self.index = 0


class SFTEvalDataset(SFTDataset):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def eval_type(self) -> Literal["categorical", "generative"]: ...

    @abstractmethod
    def evaluate(self, conversation: dict[str, Any], completion: str) -> bool: ...


def build_mc_prompt(question: str, letters: tuple[str, ...], choices: list[str]) -> str:
    query = f"Multiple Choice question: {question}\n"
    query += "".join(
        [f"- {choice}={letter}\n" for letter, choice in zip(letters, choices, strict=True)]
    )
    query += "\nRespond only with the letter of the correct answer."
    return query
