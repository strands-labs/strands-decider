"""Synthetic request workload used by the benchmark runner."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from strands_decider.prompting import render_question, render_state
from strands_decider.schema import ChoiceQuestion

INPUT_LENGTHS = (256, 1024, 2048, 4096)
THROUGHPUT_INPUT_TOKENS = 256
QUESTIONS_PER_REQUEST = 4
OPTIONS = {
    "alpha": "first",
    "bravo": "second",
    "charlie": "third",
    "delta": "fourth",
}
WORDS = (
    "A customer requested help with an account workflow. Review the supplied context and choose "
    "the most appropriate route. The record contains operational notes, constraints, and prior "
    "events that may affect the decision. "
)


@dataclass(frozen=True)
class Shape:
    input_tokens: int

    @property
    def questions(self) -> int:
        return QUESTIONS_PER_REQUEST

    @property
    def key(self) -> str:
        return f"{self.input_tokens}t"


@dataclass(frozen=True)
class Request:
    body: bytes
    shape: Shape
    repeated_state: bool


class Workload:
    """Deterministic equal-weight synthetic workload.

    Half of requests reuse one state per input-length bucket and half use a unique
    fixed-width state identifier. Question payloads rotate while retaining the same
    rendered token length.
    """

    def __init__(self, tokenizer: Any, seed: int = 0):
        self.tokenizer = tokenizer
        self.seed = seed
        self.shapes = [Shape(tokens) for tokens in INPUT_LENGTHS]
        self._cycle = [(shape, repeated) for shape in self.shapes for repeated in (True, False)]
        random.Random(seed).shuffle(self._cycle)
        self._state_chars: dict[int, int] = {}
        self._token_ranges: dict[int, tuple[int, int]] = {}
        for tokens in INPUT_LENGTHS:
            self._state_chars[tokens] = self._calibrate_state_chars(tokens)
            measured = [
                self._rendered_tokens(self._state(self._state_chars[tokens], identifier))
                for identifier in range(32)
            ]
            self._token_ranges[tokens] = (min(measured), max(measured))
            if max(abs(value - tokens) for value in measured) > 4:
                raise ValueError(
                    f"synthetic {tokens}-token states vary from {min(measured)} to "
                    f"{max(measured)} tokens across identifiers"
                )

    def _question(self, index: int) -> dict[str, Any]:
        keys = list(OPTIONS)
        shift = index % len(keys)
        rotated = keys[shift:] + keys[:shift]
        criteria = {key: OPTIONS[key] for key in rotated}
        return {
            "type": "choice",
            "instructions": "Choose the appropriate route.",
            "criteria": criteria,
        }

    def _rendered_tokens(self, state: str) -> int:
        question = ChoiceQuestion(**self._question(0))
        state_text = render_state(state)
        question_text = render_question(question).text
        state_ids = self.tokenizer(state_text, add_special_tokens=True)["input_ids"]
        question_ids = self.tokenizer(question_text, add_special_tokens=False)["input_ids"]
        return len(state_ids) + len(question_ids)

    def _state(self, chars: int, state_id: int) -> str:
        prefix = f"Record {state_id:016x}. "
        text = (WORDS * (chars // len(WORDS) + 2))[:chars]
        return prefix + text

    def _calibrate_state_chars(self, target: int) -> int:
        empty_tokens = self._rendered_tokens(self._state(0, 0))
        if empty_tokens > target:
            raise ValueError(
                f"the minimal synthetic request is {empty_tokens} tokens, above target {target}"
            )
        lo, hi = 0, max(32, 4 * target)
        while self._rendered_tokens(self._state(hi, 0)) < target:
            hi *= 2
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            got = self._rendered_tokens(self._state(mid, 0))
            if got <= target:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        got = self._rendered_tokens(self._state(best, 0))
        if target - got > 2:
            raise ValueError(f"could only calibrate {target}-token shape to {got} tokens")
        return best

    def _request_with_chars(
        self,
        shape: Shape,
        *,
        repeated: bool,
        sequence: int,
        state_chars: int,
    ) -> Request:
        state_id = shape.input_tokens if repeated else sequence + 1
        state = self._state(state_chars, state_id)
        questions = {f"q{i:02d}": self._question(sequence + i) for i in range(shape.questions)}
        body = json.dumps(
            {"state": state, "questions": questions},
            separators=(",", ":"),
        ).encode()
        return Request(body=body, shape=shape, repeated_state=repeated)

    def calibrate(self, measure: Callable[[Request], int | None]) -> None:
        """Adjust state lengths to the input-token counts reported by the server."""
        base_shape = self.shapes[0]
        base = measure(
            self._request_with_chars(
                base_shape,
                repeated=True,
                sequence=0,
                state_chars=0,
            )
        )
        if base is None:
            raise ValueError("server refused the minimal synthetic request during calibration")

        ratio: float | None = None
        for target in INPUT_LENGTHS:
            if target <= base:
                self._state_chars[target] = 0
                self._token_ranges[target] = (base, base)
                continue
            shape = Shape(target)
            chars = (
                self._state_chars[target] if ratio is None else max(0, int((target - base) / ratio))
            )
            best: tuple[int, int] | None = None
            tolerance = max(2, int(0.005 * target))
            for _ in range(8):
                got = measure(
                    self._request_with_chars(
                        shape,
                        repeated=True,
                        sequence=0,
                        state_chars=chars,
                    )
                )
                if got is None:
                    chars = max(0, int(chars * 0.9))
                    continue
                if best is None or abs(got - target) < abs(best[1] - target):
                    best = (chars, got)
                if abs(got - target) <= tolerance:
                    break
                if chars == 0:
                    break
                measured_ratio = (got - base) / chars
                if measured_ratio <= 0:
                    break
                ratio = measured_ratio
                next_chars = max(0, int((target - base) / ratio))
                if next_chars == chars:
                    next_chars += 1 if got < target else -1
                chars = max(0, next_chars)

            if best is None or abs(best[1] - target) > tolerance:
                got = best[1] if best else None
                raise ValueError(f"could only calibrate {target}-token requests to {got} tokens")
            self._state_chars[target] = best[0]
            self._token_ranges[target] = (best[1], best[1])
            if best[0] > 0:
                ratio = (best[1] - base) / best[0]

    def request(self, shape: Shape, *, repeated: bool, sequence: int) -> Request:
        return self._request_with_chars(
            shape,
            repeated=repeated,
            sequence=sequence,
            state_chars=self._state_chars[shape.input_tokens],
        )

    def requests(self, count: int, sequence: int = 0) -> list[Request]:
        requests = []
        for index in range(count):
            current = sequence + index
            shape, repeated = self._cycle[current % len(self._cycle)]
            requests.append(self.request(shape, repeated=repeated, sequence=current))
        return requests

    def throughput_requests(self, count: int, sequence: int = 0) -> list[Request]:
        shape = Shape(THROUGHPUT_INPUT_TOKENS)
        return [
            self.request(
                shape,
                repeated=(sequence + index) % 2 == 0,
                sequence=sequence + index,
            )
            for index in range(count)
        ]

    def latency_warmups(self, sequence: int = 0) -> list[Request]:
        return [
            self.request(shape, repeated=True, sequence=sequence + index)
            for index, shape in enumerate(self.shapes)
        ]

    def manifest(self) -> dict[str, Any]:
        return {
            "input_length_targets": list(INPUT_LENGTHS),
            "throughput_input_tokens": THROUGHPUT_INPUT_TOKENS,
            "questions_per_request": QUESTIONS_PER_REQUEST,
            "question_type": "choice",
            "options_per_question": 4,
            "state_reuse": {"repeated": 0.5, "unique": 0.5},
            "latency_shape_weight": 1 / len(self.shapes),
            "seed": self.seed,
            "state_characters": self._state_chars,
            "reported_token_ranges": {
                key: {"minimum": value[0], "maximum": value[1]}
                for key, value in self._token_ranges.items()
            },
        }
