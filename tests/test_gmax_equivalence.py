# SPDX-License-Identifier: Apache-2.0
# Adapted for this vLLM V1 plugin from the JITServe artifact. See NOTICE.
"""V1 CutoffTuner port vs V0 original: identical trajectory + convergence."""
import math
import random

from jitserve_vllm31.gmax import CutoffTuner as V1CutoffTuner


class V0CutoffTunerRef:
    """Reference copy of the V0 algorithm (slo_scheduler.py CutoffTuner)."""
    GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)

    def __init__(self, initial, adaptive=True, epsilon=0.1, alpha=0.2, seed=0):
        self.adaptive = adaptive
        self.grid = sorted(set(self.GRID) | {initial})
        self.value = {p: None for p in self.grid}
        self.current = initial
        self.epsilon = epsilon
        self.alpha = alpha
        self._rng = random.Random(seed)

    def update(self, goodput_rate):
        if not self.adaptive:
            return self.current
        v = self.value[self.current]
        self.value[self.current] = goodput_rate if v is None else (
            (1 - self.alpha) * v + self.alpha * goodput_rate)
        tried = [p for p in self.grid if self.value[p] is not None]
        best = max(tried, key=lambda p: self.value[p])
        if self._rng.random() < self.epsilon:
            i = self.grid.index(best)
            neighbours = self.grid[max(0, i - 1):i] + self.grid[i + 1:i + 2]
            self.current = self._rng.choice(neighbours) if neighbours else best
        else:
            self.current = best
        return self.current


def _env_rate(p, rng):
    pdf = math.exp(-0.5 * ((p - 0.8) / 0.12) ** 2)
    return pdf * 10.0 + rng.gauss(0.0, 0.3)


def test_v1_port_matches_v0_across_200_frames():
    v0 = V0CutoffTunerRef(0.7, epsilon=0.1, alpha=0.2, seed=7)
    v1 = V1CutoffTuner(0.7, epsilon=0.1, alpha=0.2, seed=7)
    rng0 = random.Random(999)
    rng1 = random.Random(999)
    seq0 = []
    seq1 = []
    for _ in range(200):
        assert v0.current == v1.current
        r0 = v0.update(_env_rate(v0.current, rng0))
        r1 = v1.update(_env_rate(v1.current, rng1))
        seq0.append(r0)
        seq1.append(r1)
    assert seq0 == seq1
    assert v1.current == 0.8
    for p in V0CutoffTunerRef.GRID:
        c0, c1 = v0.value[p], v1.value[p]
        assert c0 is None and c1 is None or abs(c0 - c1) < 1e-9


def test_nonadaptive_keeps_initial():
    v1 = V1CutoffTuner(0.95, adaptive=False)
    assert v1.update(5.0) == 0.95
    assert v1.update(0.0) == 0.95


def test_grid_contains_initial_points():
    v1 = V1CutoffTuner(0.77)
    assert sorted(set(v1.GRID) | {0.77}) == v1.grid
    assert 0.77 in v1.grid
