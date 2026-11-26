from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path
import json
import math
import sys


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise ArgumentTypeError("expected true or false")


def _parse_nullable_number(value):
    if value.lower() in ("none", "null"):
        return None
    try:
        result = json.loads(value)
    except ValueError as error:
        raise ArgumentTypeError("expected a number or null") from error
    if isinstance(result, bool) or not isinstance(result, (int, float)) or not math.isfinite(result):
        raise ArgumentTypeError("expected a finite number or null")
    return result


class Args:
    def __init__(self):
        self._data = {}

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __getattr__(self, key):
        if key in self._data:
            return self._data[key]
        raise AttributeError(f"'Args' object has no attribute '{key}'")

    def as_dict(self):
        return self._data

    def get_parser(self, include_config=False):
        parser = ArgumentParser(description="Argument parser for the script")
        if include_config or "config" in self._data:
            parser.add_argument("--config", type=Path, default=None,
                                help="JSON overrides defaults; explicit CLI values take precedence")
        for key, value in self._data.items():
            if key == "config":
                continue
            if isinstance(value, bool):
                # Bare flags keep the original toggle polarity, while =true/false
                # permits an explicit override of a JSON boolean.
                parser.add_argument(f"--{key}", nargs="?", type=_parse_bool,
                                    const=not value, default=value)
            else:
                arg_type = _parse_nullable_number if value is None else type(value)
                parser.add_argument(f"--{key}", type=arg_type, default=value)
        return parser

    def parse_args(self, argv=None):
        argv = list(sys.argv[1:] if argv is None else argv)
        probe = ArgumentParser(add_help=False)
        probe.add_argument("--config", type=Path)
        config_path, _ = probe.parse_known_args(argv)
        parser = self.get_parser(include_config=True)
        if config_path.config is not None:
            try:
                values = json.loads(config_path.config.read_text())
                if not isinstance(values, dict):
                    raise ValueError("config must be a JSON object")
                unknown = set(values) - set(self._data) - {"config"}
                if unknown:
                    raise ValueError("unknown config keys: " + ", ".join(sorted(unknown)))
                overrides = {}
                for key, value in values.items():
                    if key == "config":
                        continue
                    original = self._data[key]
                    if isinstance(original, bool):
                        if not isinstance(value, bool):
                            raise ValueError(f"{key} requires a JSON boolean")
                    elif isinstance(original, Path):
                        if not isinstance(value, str):
                            raise ValueError(f"{key} requires a path string")
                        value = Path(value)
                    elif original is None:
                        if value is not None and (isinstance(value, bool) or
                                not isinstance(value, (int, float)) or not math.isfinite(value)):
                            raise ValueError(f"{key} requires a number or null")
                    elif isinstance(original, int):
                        if isinstance(value, bool) or not isinstance(value, int):
                            raise ValueError(f"{key} requires an integer")
                    elif isinstance(original, float):
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                            raise ValueError(f"{key} requires a finite number")
                        value = float(value)
                    elif not isinstance(value, type(original)):
                        raise ValueError(f"{key} requires {type(original).__name__}")
                    overrides[key] = value
                parser.set_defaults(**overrides)
            except (OSError, ValueError) as error:
                parser.error(str(error))
        return self.from_namespace(parser.parse_args(argv))

    @classmethod
    def from_namespace(cls, namespace: Namespace):
        obj = cls()
        obj._data = dict(vars(namespace))
        return obj

    @classmethod
    def load_from_json(cls, json_path: str):
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config must be a JSON object")
        obj = cls()
        obj._data = data
        return obj

    def __add__(self, other):
        if not isinstance(other, Args):
            raise TypeError("Can only add Args objects together.")
        new_args = Args()
        new_args._data = {**self._data, **other._data}
        return new_args
