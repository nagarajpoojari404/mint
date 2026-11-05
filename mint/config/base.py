import tomllib
import types
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Self, Union, get_args, get_origin


def _resolve_config_type(field_type: Any) -> Any:  # noqa: ANN401
    if get_origin(field_type) in (Union, types.UnionType):
        return next(
            (
                arg
                for arg in get_args(field_type)
                if isinstance(arg, type) and issubclass(arg, Config)
            ),
            field_type,
        )
    return field_type


@dataclass
class Config:
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        field_types = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for key, raw_val in data.items():
            field_type = _resolve_config_type(field_types.get(key))
            if (
                isinstance(raw_val, dict)
                and isinstance(field_type, type)
                and issubclass(field_type, Config)
            ):
                val = field_type.from_dict(raw_val)
            else:
                val = raw_val
            kwargs[key] = val
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, toml_path: str) -> Self:
        path = Path(toml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {toml_path}")

        with path.open("rb") as f:
            return cls.from_dict(tomllib.load(f))
