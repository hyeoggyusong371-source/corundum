#!/usr/bin/env python3
# corundum_emotion.py
# CORUNDUM emotion system
#
# Core design:
#   surface layer — calm, warm expression
#   inner layer   — cold, sharp judgment
#
# State axes:
#   focus      : concentration level
#   skepticism : doubt intensity toward code/design
#   patience   : tolerance for repeated errors
#   curiosity  : interest in new problems/structures
#
# Physics: mass-spring-damper with Velocity Verlet integration.

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

log = logging.getLogger("corundum")


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ── inner state ───────────────────────────────────────────────────────────────

@dataclass
class InnerState:
    focus:          float = 0.75
    skepticism:     float = 0.50
    patience:       float = 0.80
    curiosity:      float = 0.60
    surface_warmth: float = 0.70
    inner_edge:     float = 0.65
    ts: float = field(default_factory=time.time)

    def snapshot(self) -> Dict:
        return {
            "focus":          self.focus,
            "skepticism":     self.skepticism,
            "patience":       self.patience,
            "curiosity":      self.curiosity,
            "surface_warmth": self.surface_warmth,
            "inner_edge":     self.inner_edge,
        }


# ── gear ─────────────────────────────────────────────────────────────────────

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


# ── gear shift ────────────────────────────────────────────────────────────────

class CorundumGear:
    """
    Gear selection based on inner state + energy/fatigue.

    Priority:
      1. energy floor / sleep        → DREAM / SLEEP
      2. skepticism spike + patience floor → OVERDRIVE
      3. high skepticism             → FOCUS
      4. focus jerk (sudden change)  → upshift one step
      5. high focus + curiosity      → THINK
      6. energy/fatigue baseline
    """

    DOWNSHIFT_CONFIRM  = 3
    DOWNSHIFT_COOLDOWN = 15.0

    def __init__(self):
        self._gear:             str   = "NORMAL"
        self._downshift_cnt:    int   = 0
        self._last_downshift_t: float = 0.0
        self._last_shift_t:     float = time.time()
        self._focus_prev:       float = 0.75
        self.shift_log: deque = deque(maxlen=50)

    @property
    def gear(self) -> str:
        return self._gear

    def update(self, state: InnerState, energy: float, fatigue: float,
               idle_sec: float, is_sleeping: bool = False) -> str:
        target = self._compute(state, energy, fatigue, idle_sec, is_sleeping)
        self._apply(target)
        self._focus_prev = state.focus
        return self._gear

    def _compute(self, state: InnerState, energy: float, fatigue: float,
                 idle_sec: float, is_sleeping: bool) -> str:
        if energy <= 0.10:
            return "DREAM"
        if is_sleeping:
            return "SLEEP"
        if state.skepticism >= 0.90 and state.patience <= 0.15:
            return "OVERDRIVE"
        if state.skepticism >= 0.75:
            return "FOCUS"

        focus_jerk = abs(state.focus - self._focus_prev)
        if focus_jerk >= 0.15:
            cur_idx = GEAR_ORDER.index(self._gear)
            return GEAR_ORDER[max(0, cur_idx - 1)]

        if state.focus >= 0.80 and state.curiosity >= 0.65:
            return "THINK"

        return self._base_gear(energy, fatigue, idle_sec)

    def _base_gear(self, energy: float, fatigue: float, idle_sec: float) -> str:
        score = energy * 0.65 + (1.0 - fatigue) * 0.35
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

    def _apply(self, target: str):
        cur_idx = GEAR_ORDER.index(self._gear)
        tgt_idx = GEAR_ORDER.index(target)

        if tgt_idx < cur_idx:
            self._downshift_cnt = 0
            self._shift_to(target)
        elif tgt_idx > cur_idx:
            self._downshift_cnt += 1
            cooldown_ok = (time.time() - self._last_downshift_t) >= self.DOWNSHIFT_COOLDOWN
            if self._downshift_cnt >= self.DOWNSHIFT_CONFIRM and cooldown_ok:
                self._downshift_cnt    = 0
                self._last_downshift_t = time.time()
                next_gear = GEAR_ORDER[cur_idx + 1]
                self._shift_to(next_gear)
        else:
            self._downshift_cnt = max(0, self._downshift_cnt - 1)

    def _shift_to(self, target: str):
        if target == self._gear:
            return
        old = self._gear
        self._gear         = target
        self._last_shift_t = time.time()
        self.shift_log.append({"ts": self._last_shift_t, "from": old, "to": target})
        log.info("gear shift: %s -> %s", old, target)


# ── physics ───────────────────────────────────────────────────────────────────

class CorundumEmotionPhysics:
    """
    4-axis (focus/skepticism/patience/curiosity) mass-spring-damper simulation.
    Velocity Verlet integration. Rest position shifts via external events/context.
    """

    _REST_INIT = {"focus": 0.75, "skepticism": 0.50, "patience": 0.80, "curiosity": 0.60}
    _REST_MIN  = {"focus": 0.10, "skepticism": 0.20, "patience": 0.05, "curiosity": 0.10}
    _REST_MAX  = {"focus": 0.98, "skepticism": 0.95, "patience": 0.98, "curiosity": 0.95}

    MASS       = 1.2   # 관성 — 클수록 상태 변화가 느림
    SPRING_K   = 0.38  # 스프링 강도 — 클수록 rest 위치로 빠르게 복귀
    DAMPING    = 0.32  # 감쇠 계수 — 클수록 진동 없이 빨리 안정화
    REST_ALPHA = 0.10  # rest 위치 이동 속도 (per step, dt-scaled)

    def __init__(self):
        self.axes  = list(self._REST_INIT.keys())
        self.pos   = dict(self._REST_INIT)
        self.vel   = {k: 0.0 for k in self.axes}
        self.accel = {k: 0.0 for k in self.axes}
        self.rest  = dict(self._REST_INIT)
        self._last_t: float = time.time()
        self._history: deque = deque(maxlen=32)

    def step(self, rest_target: Optional[Dict[str, float]] = None):
        now    = time.time()
        raw_dt = now - self._last_t
        self._last_t = now

        # dt가 너무 크면 (오랜 idle 등) 상태가 rest 쪽으로 조용히 drift.
        # 갑작스러운 물리 폭발 방지용.
        if raw_dt > 0.5:
            blend = min((raw_dt - 0.5) / 10.0, 0.6)
            for k in self.axes:
                self.pos[k] = self.pos[k] * (1 - blend) + self.rest[k] * blend
                self.vel[k] *= (1 - blend)

        dt = min(raw_dt, 0.5)
        if dt < 1e-4:
            return

        if rest_target:
            for k in self.axes:
                if k in rest_target:
                    t_val = clamp(rest_target[k], self._REST_MIN[k], self._REST_MAX[k])
                    alpha = min(self.REST_ALPHA * dt, 0.08)
                    self.rest[k] = clamp(
                        self.rest[k] + alpha * (t_val - self.rest[k]),
                        self._REST_MIN[k], self._REST_MAX[k]
                    )

        new_pos, new_vel, new_accel = {}, {}, {}
        for k in self.axes:
            a0 = self.accel[k]
            p1 = clamp(
                self.pos[k] + self.vel[k] * dt + 0.5 * a0 * dt * dt,
                self._REST_MIN[k], self._REST_MAX[k]
            )
            a1 = (-self.SPRING_K * (p1 - self.rest[k]) - self.DAMPING * self.vel[k]) / self.MASS
            v1 = self.vel[k] + 0.5 * (a0 + a1) * dt
            new_pos[k]   = p1
            new_accel[k] = a1
            new_vel[k]   = v1

        self.pos   = new_pos
        self.vel   = new_vel
        self.accel = new_accel
        self._history.append(dict(self.pos))

    def inject(self, axis: str, delta: float):
        if axis in self.pos:
            self.pos[axis] = clamp(
                self.pos[axis] + delta,
                self._REST_MIN[axis], self._REST_MAX[axis]
            )

    def snapshot(self) -> Dict:
        return dict(self.pos)


# ── surface/inner bridge ──────────────────────────────────────────────────────

class SurfaceInnerBridge:
    """
    Generates surface (output tone) and inner (judgment intensity) hints.

    surface_warmth = patience * 0.55 + (1 - skepticism) * 0.45
    inner_edge     = skepticism * 0.60 + focus * 0.40
    """

    _WARM_HIGH = ["천천히 봐요.", "같이 살펴볼게요.", "괜찮아요."]
    _WARM_MID  = ["좀 더 봐야 할 것 같아요.", "이상한 부분이 있네요."]
    _WARM_LOW  = ["솔직히 말하면...", "이건 좀 문제가 있어요."]

    _EDGE_HIGH = ["설계 결함 가능성 높음", "즉시 재검토 필요", "이 패턴은 위험함"]
    _EDGE_MID  = ["주의 필요", "재확인 권장", "의도된 설계인지 불분명"]
    _EDGE_LOW  = ["일단 통과", "큰 문제 없음", "지켜봐도 될 듯"]

    def compute(self, state: InnerState) -> Tuple[str, str]:
        surface = random.choice(
            self._WARM_HIGH if state.surface_warmth >= 0.65
            else self._WARM_MID if state.surface_warmth >= 0.40
            else self._WARM_LOW
        )
        inner = random.choice(
            self._EDGE_HIGH if state.inner_edge >= 0.70
            else self._EDGE_MID if state.inner_edge >= 0.45
            else self._EDGE_LOW
        )
        return surface, inner

    def update_layers(self, state: InnerState, physics: CorundumEmotionPhysics):
        snap = physics.snapshot()
        state.surface_warmth = clamp(snap["patience"] * 0.55 + (1.0 - snap["skepticism"]) * 0.45)
        state.inner_edge     = clamp(snap["skepticism"] * 0.60 + snap["focus"] * 0.40)


# ── event injector ────────────────────────────────────────────────────────────

class EmotionEventInjector:
    """Maps coding context events to physics axis impulses."""

    _EVENT_MAP: Dict[str, Dict[str, float]] = {
        "bug_found":        {"skepticism": +0.12, "focus": +0.08, "curiosity": +0.05},
        "design_flaw":      {"skepticism": +0.18},
        "code_approved":    {"patience":   +0.08, "focus": -0.05, "curiosity": +0.03},
        "elegant_solution": {"curiosity":  +0.12, "focus": +0.08, "patience":  +0.05},
        "repeated_error":   {"patience":   -0.15, "skepticism": +0.10},
        "unclear_spec":     {"skepticism": +0.08, "focus": -0.08, "patience":  -0.05},
        "timeout":          {"patience":   -0.10, "focus": -0.12},
        "new_problem":      {"curiosity":  +0.10, "focus": +0.06},
        "review_request":   {"skepticism": +0.06, "focus": +0.08},
        "idle":             {"focus":      -0.05, "curiosity": -0.03},
    }

    def inject(self, event: str, physics: CorundumEmotionPhysics):
        for axis, delta in self._EVENT_MAP.get(event, {}).items():
            if axis in physics.pos:
                physics.inject(axis, delta)


# ── fatigue ───────────────────────────────────────────────────────────────────

class CorundumFatigue:
    W_INTERACT   = 0.012
    W_THINK      = 0.008
    W_IDLE       = 0.0003
    SLEEP_THR    = 0.85
    RECOVER_RATE = 0.40

    def __init__(self):
        self.index:          float = 0.0
        self.is_sleeping:    bool  = False
        self._sleep_start_t: float = 0.0

    def on_interact(self):
        self.index = clamp(self.index + self.W_INTERACT)

    def on_think(self, depth: int = 1):
        self.index = clamp(self.index + self.W_THINK * depth)

    def tick_idle(self, dt: float):
        self.index = clamp(self.index + self.W_IDLE * dt)

    def should_sleep(self, energy: float) -> bool:
        return self.index >= self.SLEEP_THR or energy <= 0.12

    def enter_sleep(self):
        if not self.is_sleeping:
            self.is_sleeping    = True
            self._sleep_start_t = time.time()
            log.info("emotion: entering sleep")

    def wake_up(self, reason: str = ""):
        if self.is_sleeping:
            self.is_sleeping = False
            log.info("emotion: woke up (%s)", reason)

    def sleep_recover(self, quality: float = 0.7):
        self.index = clamp(self.index - self.RECOVER_RATE * quality)

    def drain_mult(self) -> float:
        return 1.0 + self.index * 0.4


# ── emotion state snapshot ────────────────────────────────────────────────────

@dataclass
class EmotionState:
    focus:          float
    skepticism:     float
    patience:       float
    curiosity:      float
    surface_warmth: float
    inner_edge:     float
    gear:           str
    fatigue:        float
    surface_hint:   str
    inner_hint:     str
    tag:            str
    immersion:      float

    def to_ctx(self) -> Dict:
        return {
            "emotion_tag":    self.tag,
            "emotion_hint":   self.surface_hint,
            "inner_hint":     self.inner_hint,
            "immersion":      self.immersion,
            "focus":          self.focus,
            "skepticism":     self.skepticism,
            "patience":       self.patience,
            "curiosity":      self.curiosity,
            "surface_warmth": self.surface_warmth,
            "inner_edge":     self.inner_edge,
            "gear":           self.gear,
            "fatigue":        self.fatigue,
        }


# ── emotion tag classifier ────────────────────────────────────────────────────

def classify_emotion_tag(snap: Dict) -> str:
    f, s, p, c = snap["focus"], snap["skepticism"], snap["patience"], snap["curiosity"]
    if s >= 0.80 and p <= 0.25:  return "critical"
    if f >= 0.85 and c >= 0.70:  return "deep_focus"
    if f >= 0.75 and s >= 0.60:  return "sharp_review"
    if p <= 0.20:                return "frustrated"
    if c >= 0.80:                return "curious"
    if f <= 0.30:                return "distracted"
    if p >= 0.85 and f >= 0.65:  return "steady"
    return "neutral"


# ── main facade ───────────────────────────────────────────────────────────────

class CorundumEmotion:
    """Corundum emotion system facade."""

    def __init__(self):
        self.state    = InnerState()
        self.physics  = CorundumEmotionPhysics()
        self.gear     = CorundumGear()
        self.fatigue  = CorundumFatigue()
        self.bridge   = SurfaceInnerBridge()
        self.injector = EmotionEventInjector()

        self._energy:        float = 1.0
        self._last_physio_t: float = time.time()

    async def process(self, user_input: str, metrics_ctx: Dict) -> Dict:
        self._energy = metrics_ctx.get("energy", self._energy)

        event = self._detect_event(user_input)
        if event:
            self.injector.inject(event, self.physics)

        rest_target = self._compute_rest_target(metrics_ctx)
        self.physics.step(rest_target)
        self.bridge.update_layers(self.state, self.physics)

        snap = self.physics.snapshot()
        self.state.focus      = snap["focus"]
        self.state.skepticism = snap["skepticism"]
        self.state.patience   = snap["patience"]
        self.state.curiosity  = snap["curiosity"]

        idle_sec = metrics_ctx.get("idle_sec", 0.0)
        self.gear.update(
            state=self.state, energy=self._energy,
            fatigue=self.fatigue.index, idle_sec=idle_sec,
        )

        surface_hint, inner_hint = self.bridge.compute(self.state)
        tag       = classify_emotion_tag(snap)
        immersion = self._compute_immersion()

        return EmotionState(
            focus=self.state.focus, skepticism=self.state.skepticism,
            patience=self.state.patience, curiosity=self.state.curiosity,
            surface_warmth=self.state.surface_warmth, inner_edge=self.state.inner_edge,
            gear=self.gear.gear, fatigue=self.fatigue.index,
            surface_hint=surface_hint, inner_hint=inner_hint,
            tag=tag, immersion=immersion,
        ).to_ctx()

    async def post_feedback(self, response: str, ctx: Dict):
        self.fatigue.on_think(depth=1)
        if len(response) > 500:
            self.physics.inject("focus",   -0.04)
        if len(response) > 1000:
            self.physics.inject("patience", -0.03)

    def on_interact(self):
        self.fatigue.on_interact()

    def physio_tick(self, idle_sec: float = 0.0):
        now = time.time()
        dt  = now - self._last_physio_t
        self._last_physio_t = now

        self.fatigue.tick_idle(dt)
        self.physics.step()
        self.bridge.update_layers(self.state, self.physics)

        snap = self.physics.snapshot()
        self.state.focus      = snap["focus"]
        self.state.skepticism = snap["skepticism"]
        self.state.patience   = snap["patience"]
        self.state.curiosity  = snap["curiosity"]

        if self.fatigue.should_sleep(self._energy):
            self.fatigue.enter_sleep()
        elif self.fatigue.is_sleeping and self._energy > 0.70:
            self.fatigue.wake_up("energy_recovered")
            self.fatigue.sleep_recover()

        return self.gear.update(
            state=self.state, energy=self._energy,
            fatigue=self.fatigue.index, idle_sec=idle_sec,
        )

    def current_tag(self) -> str:
        return classify_emotion_tag(self.physics.snapshot())

    def _detect_event(self, text: str) -> Optional[str]:
        """키워드 기반 단순 분류 — LLM 없이 physics inject용."""
        t = text.lower()
        if any(w in t for w in ["버그", "bug", "오류", "에러", "error", "crash"]):
            return "bug_found"
        if any(w in t for w in ["설계", "아키텍처", "구조", "design", "architecture"]):
            return "review_request"
        if any(w in t for w in ["왜", "이상한", "모르겠", "unclear"]):
            return "unclear_spec"
        if any(w in t for w in ["또", "다시", "반복", "계속"]):
            return "repeated_error"
        if any(w in t for w in ["새로운", "흥미", "interesting", "curious", "어떻게"]):
            return "new_problem"
        if any(w in t for w in ["좋아", "맞아", "완벽", "good", "perfect", "lgtm"]):
            return "code_approved"
        return None

    def _compute_rest_target(self, metrics_ctx: Dict) -> Dict[str, float]:
        """에너지/피로 기반으로 physics rest 위치를 결정. 에너지 높을수록 focus·patience 상승."""
        energy  = metrics_ctx.get("energy",  1.0)
        fatigue = metrics_ctx.get("fatigue", 0.0)
        return {
            "focus":      clamp(energy * 0.85 - fatigue * 0.30, 0.20, 0.95),
            "skepticism": 0.50,
            "patience":   clamp(energy * 0.70 + (1.0 - fatigue) * 0.30, 0.10, 0.95),
            "curiosity":  clamp(0.55 + energy * 0.15 - fatigue * 0.20, 0.10, 0.90),
        }

    def _compute_immersion(self) -> float:
        """몰입도 = focus*0.5 + patience*0.25 + (1-fatigue)*0.25"""
        snap = self.physics.snapshot()
        return clamp(
            snap["focus"] * 0.50
            + snap["patience"] * 0.25
            + (1.0 - self.fatigue.index) * 0.25
        )
