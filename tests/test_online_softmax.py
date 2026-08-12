#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import unittest


def reference_attention(scores: list[float], values: list[list[float]]) -> list[float]:
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denominator = sum(weights)
    return [
        sum(weight * row[column] for weight, row in zip(weights, values)) /
        denominator
        for column in range(len(values[0]))
    ]


def online_attention(scores: list[float], values: list[list[float]]) -> list[float]:
    running_max = -math.inf
    denominator = 0.0
    accumulator = [0.0] * len(values[0])
    for score, row in zip(scores, values):
        new_max = max(running_max, score)
        alpha = math.exp(running_max - new_max) if denominator > 0.0 else 0.0
        beta = math.exp(score - new_max)
        denominator = denominator * alpha + beta
        accumulator = [old * alpha + beta * value
                       for old, value in zip(accumulator, row)]
        running_max = new_max
    return [value / denominator for value in accumulator]


class OnlineSoftmaxTest(unittest.TestCase):
    def test_random_inputs_match_stable_reference(self) -> None:
        random.seed(42)
        for length in (1, 2, 3, 17, 64):
            for width in (1, 7, 32):
                scores = [random.uniform(-40.0, 40.0) for _ in range(length)]
                values = [
                    [random.uniform(-3.0, 3.0) for _ in range(width)]
                    for _ in range(length)
                ]
                expected = reference_attention(scores, values)
                actual = online_attention(scores, values)
                for left, right in zip(expected, actual):
                    self.assertAlmostEqual(left, right, places=12)

    def test_extreme_scores_stay_finite(self) -> None:
        scores = [-10_000.0, 10_000.0, 9_999.0, -9_999.0]
        values = [[1.0, -1.0], [2.0, 3.0], [4.0, 5.0], [9.0, 8.0]]
        actual = online_attention(scores, values)
        expected = reference_attention(scores, values)
        self.assertTrue(all(math.isfinite(value) for value in actual))
        for left, right in zip(expected, actual):
            self.assertAlmostEqual(left, right, places=12)


if __name__ == "__main__":
    unittest.main()
