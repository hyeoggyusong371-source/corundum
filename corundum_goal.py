#!/usr/bin/env python3
# corundum_goal.py
# CORUNDUM goal formation + self-critique engine

import asyncio, json, logging, re, time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

log = logging.getLogger("corundum")

def _safe_parse_json(raw):
    from corundum_utils import safe_parse_json
    return safe_parse_json(raw)

try:
    from ollama import AsyncClient as OllamaClient; OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

try:
    from corundum_config import Cfg, CORUNDUM_IDENTITY
except ImportError:
    class Cfg:
        GOAL_MODEL = "deepseek-r1:14b"; TEMP_GOAL = 0.55; TIMEOUT_GOAL = 45.0; AUTO_INTERVAL = 60.0
    CORUNDUM_IDENTITY = "너는 코런덤이야."

def clamp(x, lo=0.0, hi=1.0): return max(lo, min(hi, x))


# ── goal status ───────────────────────────────────────────────────────────────

class GoalStatus(Enum):
    PENDING   = "pending"
    ACTIVE    = "active"
    BLOCKED   = "blocked"
    DONE      = "done"
    ABANDONED = "abandoned"


@dataclass
class CorundumGoalItem:
    name:        str
    description: str
    origin:      str
    confidence:  float      = 0.5
    progress:    float      = 0.0
    active:      bool       = True
    status:      GoalStatus = field(default=GoalStatus.PENDING)
    depends_on:  List[str]  = field(default_factory=list)
    formed_at:   float      = field(default_factory=time.time)
    last_updated: float     = field(default_factory=time.time)

    def to_str(self):
        status_mark = {
            "pending": "o", "active": ">", "blocked": "x",
            "done": "done", "abandoned": "-"
        }.get(self.status.value, "o")
        return f"[{status_mark}] {self.name} ({self.confidence:.0%})"


@dataclass
class CritiqueRecord:
    input_summary:    str
    response_summary: str
    error_type:       str
    correction:       str
    ts: float = field(default_factory=time.time)


# ── goal formation ────────────────────────────────────────────────────────────

class GoalFormationEngine:
    MAX_GOALS   = 6     # 동시에 유지할 최대 목표 수
    PATTERN_MIN = 3     # 목표 형성을 트리거하는 최소 패턴 반복 횟수
    CONF_GROW   = 0.08  # 패턴 반복 시 confidence 증가량
    CONF_DECAY  = 0.03  # (현재 미사용, 향후 자동 decay용)

    def __init__(self):
        self.goals: List[CorundumGoalItem] = []
        self._pattern_counts: Dict[str, int]        = {}
        self._outcome_map:    Dict[str, List[str]]  = {}

    def observe(self, event_tag, action, outcome):
        key = f"{event_tag}:{action[:15]}"
        self._pattern_counts[key] = self._pattern_counts.get(key, 0) + 1
        self._outcome_map.setdefault(key, []).append(outcome)
        if self._pattern_counts[key] >= self.PATTERN_MIN:
            self._try_form(key)

    def _try_form(self, key):
        for g in self.goals:
            if g.origin == key:
                g.confidence  = min(1.0, g.confidence + self.CONF_GROW)
                g.last_updated = time.time()
                return

        if len(self.goals) >= self.MAX_GOALS:
            weakest = min(self.goals, key=lambda g: g.confidence)
            if weakest.confidence < 0.3:
                self.goals.remove(weakest)
            else:
                return

        outcomes = self._outcome_map.get(key, [])
        positive = sum(1 for o in outcomes if o in ("pass", "collab"))
        negative = sum(1 for o in outcomes if o in ("flag", "veto"))
        total    = max(len(outcomes), 1)
        etag, _, action_hint = key.partition(":")

        if positive / total >= 0.6:
            name = f"{action_hint} 강화"
            desc = f"{etag} 상황에서 '{action_hint}'이 {positive}/{total}번 통과됐어."
        elif negative / total >= 0.6:
            name = f"{action_hint} 개선"
            desc = f"{etag} 상황에서 '{action_hint}'이 {negative}/{total}번 걸렸어."
        else:
            name = f"{action_hint} 탐색"
            desc = f"'{action_hint}' 패턴이 {total}번 쌓였어."

        self.goals.append(CorundumGoalItem(
            name=name, description=desc, origin=key,
            confidence=0.4, status=GoalStatus.PENDING
        ))

    def top_goal(self):
        done_names = {g.name for g in self.goals if g.status == GoalStatus.DONE}
        candidates = [
            g for g in self.goals
            if g.active
            and g.status not in (GoalStatus.DONE, GoalStatus.ABANDONED)
            and all(dep in done_names for dep in g.depends_on)
        ]
        return max(candidates, key=lambda g: g.confidence) if candidates else None

    def top_goal_name(self):
        g = self.top_goal()
        return g.name if g else "없음"

    def summary(self):
        active = [
            g for g in self.goals
            if g.active and g.status not in (GoalStatus.DONE, GoalStatus.ABANDONED)
        ]
        if not active:
            return "활성 목표 없음"
        active.sort(key=lambda g: g.confidence, reverse=True)
        return "\n".join(g.to_str() for g in active[:4])

    def add_user_goal(self, name, desc="", depends_on=None):
        g = CorundumGoalItem(
            name=name, description=desc or name, origin="user",
            confidence=0.8, status=GoalStatus.ACTIVE,
            depends_on=list(depends_on) if depends_on else [],
        )
        self.goals.append(g)
        return g


# ── self-critique ─────────────────────────────────────────────────────────────

class SelfWhipEngine:
    WINDOW          = 30    # 최근 N개 기록만 유지 (rolling window)
    ERROR_THRESHOLD = 0.40  # 이 비율 이상 오류 시 correction_hint 활성화

    def __init__(self):
        self._records:     deque             = deque(maxlen=self.WINDOW)
        self._error_counts: Dict[str, int]   = {}
        self._awareness:   float             = 0.0

    def record(self, error_type, input_summary, correction=""):
        self._records.append(CritiqueRecord(
            input_summary=input_summary[:40], response_summary="",
            error_type=error_type, correction=correction[:80]
        ))
        if error_type != "correct":
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        self._awareness = min(1.0, len(self._records) / self.WINDOW)

    def get_correction_hint(self):
        if len(self._records) < 5:
            return ""
        recent     = list(self._records)[-15:]
        error_rate = sum(1 for r in recent if r.error_type != "correct") / len(recent)
        if error_rate < self.ERROR_THRESHOLD:
            return ""
        type_counts = {}
        for r in recent:
            if r.error_type != "correct":
                type_counts[r.error_type] = type_counts.get(r.error_type, 0) + 1
        if not type_counts:
            return ""
        dominant = max(type_counts, key=lambda k: type_counts[k])
        return {
            "missed_flaw": "최근 설계/코드 결함을 자주 놓치고 있어.",
            "too_soft":    "최근 경고를 너무 부드럽게 흐리고 있어.",
            "no_basis":    "최근 근거 없는 승인이 반복되고 있어.",
        }.get(dominant, "")

    def error_rate(self):
        if not self._records:
            return 0.0
        recent = list(self._records)[-15:]
        return sum(1 for r in recent if r.error_type != "correct") / len(recent)


# ── goal LLM ──────────────────────────────────────────────────────────────────

class GoalLLM:
    SYS_CONTEXT = (
        CORUNDUM_IDENTITY
        + "\n목적+자기채찍질 역할.\n"
        + "JSON만 응답:\n"
        + '{"goal_hint":"","critique":"","top_goal":"","urgency":0.0}'
    )
    SYS_AUTO = (
        CORUNDUM_IDENTITY
        + "\n목적 재검토 역할.\n"
        + "JSON만 응답:\n"
        + '{"reflection":"","new_goal":"","new_goal_desc":"","drop_goal":""}'
    )

    _FORMAT_CTX = {
        "type": "object",
        "properties": {
            "goal_hint": {"type": "string"},
            "critique":  {"type": "string"},
            "top_goal":  {"type": "string"},
            "urgency":   {"type": "number"},
        },
        "required": ["goal_hint", "critique", "top_goal", "urgency"],
    }
    _FORMAT_AUTO = {
        "type": "object",
        "properties": {
            "reflection":     {"type": "string"},
            "new_goal":       {"type": "string"},
            "new_goal_desc":  {"type": "string"},
            "drop_goal":      {"type": "string"},
        },
        "required": ["reflection", "new_goal", "new_goal_desc", "drop_goal"],
    }

    def __init__(self):
        self.call_count = 0

    async def get_context(self, user_input, goals_summary, whip_hint, metrics_ctx):
        self.call_count += 1
        parts = []
        if goals_summary: parts.append(f"[현재 목표]\n{goals_summary}")
        if whip_hint:     parts.append(f"[채찍질] {whip_hint}")
        parts.append(f"사용자: {user_input}")
        raw = await self._call("\n".join(parts), self.SYS_CONTEXT, 256, self._FORMAT_CTX)
        return self._parse_context(raw)

    async def autonomous_reflect(self, goals_summary, metrics_ctx):
        self.call_count += 1
        raw = await self._call(goals_summary or "활성 목표 없음", self.SYS_AUTO, 512, self._FORMAT_AUTO)
        return self._parse_auto(raw)

    async def _call(self, prompt, sys_p, max_tokens=512, fmt=None):
        if not OLLAMA_OK:
            return "{}"
        try:
            client = OllamaClient()
            kwargs = {
                "model":    Cfg.GOAL_MODEL,
                "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": prompt}],
                "options":  {"temperature": Cfg.TEMP_GOAL, "num_predict": max_tokens},
            }
            if fmt:
                kwargs["format"] = fmt
            resp = await asyncio.wait_for(client.chat(**kwargs), timeout=Cfg.TIMEOUT_GOAL)
            return re.sub(r"<think>.*?</think>", "", resp["message"]["content"].strip(), flags=re.DOTALL).strip()
        except:
            return "{}"

    def _parse_context(self, raw):
        try:
            return json.loads(raw)
        except:
            return _safe_parse_json(raw) or {"goal_hint": "", "critique": "", "top_goal": "", "urgency": 0.3}

    def _parse_auto(self, raw):
        try:
            return json.loads(raw)
        except:
            return _safe_parse_json(raw) or {"reflection": "", "new_goal": "", "new_goal_desc": "", "drop_goal": ""}


# ── facade ────────────────────────────────────────────────────────────────────

class CorundumGoal:
    def __init__(self):
        self.formation = GoalFormationEngine()
        self.whip      = SelfWhipEngine()
        self.llm       = GoalLLM()
        self._last_auto_t = 0.0

    async def process(self, user_input, metrics_ctx):
        result = await self.llm.get_context(
            user_input, self.formation.summary(),
            self.whip.get_correction_hint(), metrics_ctx
        )
        return {
            "current_goal": result.get("top_goal",  self.formation.top_goal_name()),
            "goal_hint":    result.get("goal_hint", ""),
            "critique":     result.get("critique",  self.whip.get_correction_hint()),
            "urgency":      result.get("urgency",   0.3),
        }

    async def autonomous_tick(self, metrics_ctx):
        now = time.time()
        if now - self._last_auto_t < getattr(Cfg, "AUTO_INTERVAL", 60.0):
            return ""
        self._last_auto_t = now
        result = await self.llm.autonomous_reflect(self.formation.summary(), metrics_ctx)
        if result.get("new_goal"):
            self.formation.add_user_goal(result["new_goal"], result.get("new_goal_desc", ""))
        if result.get("drop_goal"):
            for g in self.formation.goals:
                if result["drop_goal"] in g.name:
                    g.active = False
                    break
        return result.get("reflection", "")

    def observe_outcome(self, event_tag, action_summary, outcome, error_type="correct", correction=""):
        self.formation.observe(event_tag, action_summary, outcome)
        self.whip.record(error_type, action_summary, correction)

    def add_goal(self, name, desc=""):
        g = self.formation.add_user_goal(name, desc)
        return f"목표 추가됨: {g.name}"

    def top_goal_name(self):
        return self.formation.top_goal_name()

    def summary(self):
        return "\n".join([
            "CORUNDUM 목표 현황",
            self.formation.summary(),
            "",
            f"자기채찍질 오류율: {self.whip.error_rate():.0%}",
        ])

    def stats(self):
        return {
            "active_goals": len([g for g in self.formation.goals if g.active]),
            "error_rate":   self.whip.error_rate(),
            "awareness":    self.whip._awareness,
            "llm_calls":    self.llm.call_count,
        }
