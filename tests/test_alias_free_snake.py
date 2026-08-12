#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[1]
TAPS = 12


def clamp(value: int, minimum: int, maximum: int) -> int:
    return minimum if value < minimum else maximum if value > maximum else value


def upsample_at(
    samples: list[list[list[float]]],
    up_filter: list[float],
    batch: int,
    up_time: int,
    channel: int,
) -> float:
    length = len(samples[batch])
    raw_time = up_time + 15
    result = 0.0
    for up_k in range(TAPS):
        numerator = raw_time - up_k
        if numerator < 0 or numerator & 1:
            continue
        padded_time = numerator // 2
        source_time = clamp(padded_time - 5, 0, length - 1)
        result += samples[batch][source_time][channel] * 2.0 * up_filter[up_k]
    return result


def staged_reference(
    samples: list[list[list[float]]],
    alpha_log: list[float],
    beta_log: list[float],
    up_filter: list[float],
    down_filter: list[float],
) -> list[list[list[float]]]:
    batches = len(samples)
    length = len(samples[0])
    channels = len(samples[0][0])
    activated = [
        [[0.0 for _ in range(channels)] for _ in range(length * 2)]
        for _ in range(batches)
    ]
    for batch in range(batches):
        for up_time in range(length * 2):
            for channel in range(channels):
                value = upsample_at(samples, up_filter, batch, up_time, channel)
                alpha = math.exp(alpha_log[channel])
                beta = math.exp(beta_log[channel])
                sine = math.sin(alpha * value)
                activated[batch][up_time][channel] = (
                    value + sine * sine / (beta + 1e-9)
                )

    output = [
        [[0.0 for _ in range(channels)] for _ in range(length)]
        for _ in range(batches)
    ]
    for batch in range(batches):
        for time in range(length):
            for channel in range(channels):
                total = 0.0
                for down_k in range(TAPS):
                    up_time = clamp(time * 2 + down_k - 5, 0, length * 2 - 1)
                    total += activated[batch][up_time][channel] * down_filter[down_k]
                output[batch][time][channel] = total
    return output


def fused_reference(
    samples: list[list[list[float]]],
    alpha_log: list[float],
    beta_log: list[float],
    up_filter: list[float],
    down_filter: list[float],
) -> list[list[list[float]]]:
    batches = len(samples)
    length = len(samples[0])
    channels = len(samples[0][0])
    output = [
        [[0.0 for _ in range(channels)] for _ in range(length)]
        for _ in range(batches)
    ]
    for batch in range(batches):
        for time in range(length):
            for channel in range(channels):
                alpha = math.exp(alpha_log[channel])
                beta = math.exp(beta_log[channel])
                result = 0.0
                for down_k in range(TAPS):
                    up_time = clamp(time * 2 + down_k - 5, 0, length * 2 - 1)
                    upsampled = upsample_at(
                        samples, up_filter, batch, up_time, channel
                    )
                    sine = math.sin(alpha * upsampled)
                    activated = upsampled + sine * sine / (beta + 1e-9)
                    result += activated * down_filter[down_k]
                output[batch][time][channel] = result
    return output


class AliasFreeSnakeTest(unittest.TestCase):
    def assert_nested_close(
        self,
        left: list[list[list[float]]],
        right: list[list[list[float]]],
    ) -> None:
        for left_batch, right_batch in zip(left, right, strict=True):
            for left_row, right_row in zip(left_batch, right_batch, strict=True):
                for left_value, right_value in zip(left_row, right_row, strict=True):
                    self.assertAlmostEqual(left_value, right_value, places=12)

    def test_fused_polyphase_matches_materialized_pipeline(self) -> None:
        generator = random.Random(0x48334353)
        for length in (1, 2, 7, 19):
            batches = 2
            channels = 3
            samples = [
                [
                    [generator.uniform(-1.5, 1.5) for _ in range(channels)]
                    for _ in range(length)
                ]
                for _ in range(batches)
            ]
            alpha_log = [generator.uniform(-1.0, 1.0) for _ in range(channels)]
            beta_log = [generator.uniform(-1.0, 1.0) for _ in range(channels)]
            up_filter = [generator.uniform(-0.2, 0.4) for _ in range(TAPS)]
            down_filter = [generator.uniform(-0.2, 0.4) for _ in range(TAPS)]
            self.assert_nested_close(
                fused_reference(
                    samples, alpha_log, beta_log, up_filter, down_filter
                ),
                staged_reference(
                    samples, alpha_log, beta_log, up_filter, down_filter
                ),
            )

    def test_cuda_source_contains_released_twelve_tap_fir_path(self) -> None:
        source = (ROOT / "src/h3_gpu_cuda_vae.cu").read_text(encoding="utf-8")
        self.assertIn("alias_free_snake_kernel", source)
        self.assertIn("for (int down_k = 0; down_k < 12; down_k++)", source)
        self.assertIn("for (int up_k = 0; up_k < 12; up_k++)", source)
        self.assertIn("2.0f * upsample_filter[up_k]", source)
        self.assertIn("downsample_filter[down_k]", source)
        self.assertNotIn("alias-free Snake fallback", source)


if __name__ == "__main__":
    unittest.main()
