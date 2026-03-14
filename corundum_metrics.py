#!/usr/bin/env python3
# corundum_metrics.py
# CORUNDUM metrics engine — rule-based, no LLM

import time
from typing import Dict
import logging

log = logging.getLogger("corundum")


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


GEAR_INTERVALS = {
    "OVERDRIVE": 1.0,
    "FOCUS":     3.0,
    "THINK":     8.0,
    "NORMAL":    20.0,
    "SAVE":      60.0,
    "LOW":       180.0,
    "SLEEP":     600.0,
    "DREAM":     1800.0,
}
GEAR_ORDER = ["OVERDRIVE", "FOCUS", "THINK", "NORMAL", "SAVE", "LOW", "SLEEP", "DREAM"]


class MetricsEngine:
    """
    Corundum metrics controller. Fully rule-based.

    Tracks:
      energy  [0-1] : consumed per interaction, recovers over time
      fatigue [0-1] : accumulates slowly, cleared during sleep
      gear    [str] : processing gear determined by energy/fatigue/idle
    """

    W_INTERACT = 0.018
    W_THINK    = 0.010
    W_IDLE     = 0.0002

    RECOVER_IDLE  = 0.0004
    RECOVER_SLEEP = 0.40

    FATIGUE_PER_INTERACT = 0.008
    FATIGUE_IDLE_DECAY   = 0.0001
    FATIGUE_SLEEP_DECAY  = 0.35

    SLEEP_ENERGY_THR  = 0.12
    SLEEP_FATIGUE_THR = 0.88

    def __init__(self):
        self.energy:  float = 1.0
        self.fatigue: float = 0.0
        self.gear:    str   = "NORMAL"

        self._last_tick_t:     float = time.time()
        self._last_interact_t: float = time.time()
        self._is_sleeping:     bool  = False
        self._sleep_start_t:   float = 0.0
        self._interaction_count: int = 0

        self._downshift_cnt:    int   = 0
        self._last_downshift_t: float = 0.0
        self._DS_CONFIRM  = 3
        self._DS_COOLDOWN = 15.0

    def tick(self, idle_sec: float = 0.0, interaction_count: int = 0) -> Dict:
        now = time.time()
        dt  = now - self._last_tick_t
        self._last_tick_t = now

        if self._is_sleeping:
            self.energy  = clamp(self.energy  + self.RECOVER_IDLE * dt * 3.0)
            self.fatigue = clamp(self.fatigue - self.FATIGUE_IDLE_DECAY * dt * 2.0)
        else:
            self.energy  = clamp(self.energy  - self.W_IDLE * dt + self.RECOVER_IDLE * dt)
            self.fatigue = clamp(self.fatigue - self.FATIGUE_IDLE_DECAY * dt)

        self._check_sleep()
        self._update_gear(idle_sec)
        return self.snapshot()

    def on_response(self, response: str):
        self._interaction_count += 1
        self.energy  = clamp(self.energy  - self.W_INTERACT)
        self.fatigue = clamp(self.fatigue + self.FATIGUE_PER_INTERACT)

        if len(response) > 800:
            self.energy  = clamp(self.energy  - self.W_THINK * 2)
            self.fatigue = clamp(self.fatigue + self.FATIGUE_PER_INTERACT * 0.5)

        self._last_interact_t = time.time()
        log.debug("metrics after response: energy=%.2f fatigue=%.2f", self.energy, self.fatigue)

    def _check_sleep(self):
        if not self._is_sleeping:
            if self.energy <= self.SLEEP_ENERGY_THR or self.fatigue >= self.SLEEP_FATIGUE_THR:
                self._is_sleeping   = True
                self._sleep_start_t = time.time()
                log.info("metrics: entering sleep")
        else:
            sleep_sec = time.time() - self._sleep_start_t
            if self.energy > 0.70 and sleep_sec > 600:
                self._is_sleeping = False
                self.energy  = clamp(self.energy  + self.RECOVER_SLEEP)
                self.fatigue = clamp(self.fatigue - self.FATIGUE_SLEEP_DECAY)
                log.info("metrics: woke up after %.1f min", sleep_sec / 60)

    def _update_gear(self, idle_sec: float):
        target  = self._compute_gear(idle_sec)
        cur_idx = GEAR_ORDER.index(self.gear)
        tgt_idx = GEAR_ORDER.index(target)

        # 업시프트(더 빠른 기어)는 즉시 반영.
        # 다운시프트는 _DS_CONFIRM번 연속 요청 + cooldown 이후에만 실행 (진동 방지).
        if tgt_idx < cur_idx:
            self._downshift_cnt = 0
            self.gear = target
            log.debug("gear up: %s", target)
        elif tgt_idx > cur_idx:
            self._downshift_cnt += 1
            cooldown_ok = (time.time() - self._last_downshift_t) >= self._DS_COOLDOWN
            if self._downshift_cnt >= self._DS_CONFIRM and cooldown_ok:
                self._downshift_cnt    = 0
                self._last_downshift_t = time.time()
                self.gear = GEAR_ORDER[cur_idx + 1]
                log.debug("gear down: %s", self.gear)
        else:
            self._downshift_cnt = max(0, self._downshift_cnt - 1)

    def _compute_gear(self, idle_sec: float) -> str:
        # MetricsEngine은 energy/fatigue 기반 rule-only.
        # OVERDRIVE/FOCUS는 emotion skepticism 기반이라 CorundumEmotion이 담당.
        # ContextAssembler에서 emotion gear가 metrics gear를 오버라이드함 — 의도적 설계.
        if self.energy <= 0.10:
            return "DREAM"
        if self._is_sleeping:
            return "SLEEP"

        # score: energy 가중치 0.65, 피로 반전 0.35
        score = self.energy * 0.65 + (1.0 - self.fatigue) * 0.35

        if score >= 0.82:   base = "THINK"
        elif score >= 0.65: base = "NORMAL"
        elif score >= 0.48: base = "SAVE"
        elif score >= 0.30: base = "LOW"
        elif score >= 0.15: base = "SLEEP"
        else:               base = "DREAM"

        if idle_sec > 1800:
            idx  = min(GEAR_ORDER.index(base) + 1, len(GEAR_ORDER) - 1)
            base = GEAR_ORDER[idx]

        return base

    def snapshot(self) -> Dict:
        return {
            "energy":       self.energy,
            "fatigue":      self.fatigue,
            "gear":         self.gear,
            "is_sleeping":  self._is_sleeping,
            "interactions": self._interaction_count,
        }
