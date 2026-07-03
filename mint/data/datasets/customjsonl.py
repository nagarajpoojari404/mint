import json
from pathlib import Path
from typing import Any

from mint.data.datasets.base import SFTTrainDataset


class CustomJSON(SFTTrainDataset):
    def __init__(self, filepath: str) -> None:
        super().__init__()
        self.filepath = filepath
        self.conversations = []

        if Path(filepath).exists():
            with Path(filepath).open(encoding="utf-8") as f:
                for line in f:
                    line_ = line.strip()
                    if line_:
                        self.conversations.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # return {"messages": self.conversations[index]}  # noqa: ERA001
        return self.conversations[index]
