#!/usr/bin/env python3
# corundum_logic.py
# CORUNDUM logic core — InnerJudge / CriticGuard / LogicCore

import asyncio, json, logging, re, time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("corundum")
from corundum_utils import safe_parse_json as _safe_parse_json

try:
    from ollama import AsyncClient as OllamaClient; OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

try:
    from corundum_config import Cfg, CORUNDUM_IDENTITY
except ImportError:
    class Cfg:
        LOGIC_MODEL  = "exaone3.5:32b";   JUDGE_MODEL  = "solar:10.7b";   CRITIC_MODEL = "deepseek-r1:14b"
        TEMP_LOGIC   = 0.65;              TEMP_JUDGE   = 0.75;            TEMP_CRITIC  = 0.45
        TIMEOUT_LOGIC = 120.0;            TIMEOUT_JUDGE = 30.0;           TIMEOUT_CRITIC = 45.0
        NUM_CTX = 32768
    CORUNDUM_IDENTITY = "너는 코런덤이야."

def clamp(x, lo=0.0, hi=1.0): return max(lo, min(hi, x))

CTX_MODE_FULL    = "full"     # 감정·목표·기억 전부 포함
CTX_MODE_DEEP    = "deep"     # inner_hint + kg + self_critique 위주 (리뷰/설계용)
CTX_MODE_MINIMAL = "minimal"  # 상태 한 줄만 (CriticGuard용)


# ── context string builder ────────────────────────────────────────────────────

async def _build_ctx_str(ctx, mode=CTX_MODE_FULL):
    parts = []
    gear    = ctx.get("gear",        "NORMAL")
    energy  = ctx.get("energy",       1.0)
    fatigue = ctx.get("fatigue",      0.0)
    focus   = ctx.get("focus",        0.75)
    skep    = ctx.get("skepticism",   0.50)
    pat     = ctx.get("patience",     0.80)
    cur     = ctx.get("curiosity",    0.60)

    _GEAR_DESC = {
        "OVERDRIVE": "긴급 방어",
        "FOCUS":     "집중 추론",
        "THINK":     "설계 분석",
        "NORMAL":    "일반",
        "SAVE":      "저전력",
        "LOW":       "피로 누적",
        "SLEEP":     "수면",
        "DREAM":     "동면",
    }

    warnings = []
    if skep   >= 0.80: warnings.append(f"회의감 높음({skep:.2f})")
    if pat    <= 0.20: warnings.append(f"인내 바닥({pat:.2f})")
    if focus  <= 0.30: warnings.append(f"집중 분산({focus:.2f})")
    if fatigue >= 0.70: warnings.append(f"피로 누적({fatigue:.2f})")

    state_line = (
        f"[상태: {gear} / {_GEAR_DESC.get(gear, '')}] "
        f"집중={focus:.2f} 회의={skep:.2f} 인내={pat:.2f} 탐구={cur:.2f} "
        f"에너지={energy:.2f} 피로={fatigue:.2f}"
    )
    if warnings:
        state_line += " | 경고: " + ", ".join(warnings)
    parts.append(state_line)

    if mode == CTX_MODE_MINIMAL:
        return "\n".join(parts)

    surface_hint = ctx.get("emotion_hint", "")
    inner_hint   = ctx.get("inner_hint",   "")
    immersion    = ctx.get("immersion",     1.0)

    if mode == CTX_MODE_DEEP:
        if inner_hint: parts.append(f"[판단 톤] {inner_hint}")
    else:
        if surface_hint or inner_hint:
            parts.append(f"[겉] {surface_hint}  [속] {inner_hint}  몰입={immersion:.2f}")

    goal     = ctx.get("current_goal", "")
    goal_hint = ctx.get("goal_hint",   "")
    critique  = ctx.get("self_critique", "")

    if mode == CTX_MODE_DEEP:
        if critique: parts.append(f"[자기채찍질] {critique}")
    else:
        if goal:      parts.append(f"[목표] {goal}")
        if goal_hint: parts.append(f"[목표 힌트] {goal_hint}")
        if critique:  parts.append(f"[자기채찍질] {critique}")

    recalled = ctx.get("recalled_memory", "")
    kg_hints = ctx.get("kg_hints",        "")

    if mode == CTX_MODE_DEEP:
        if kg_hints: parts.append(f"[KG] {kg_hints}")
    else:
        if recalled: parts.append(f"[기억] {recalled}")
        if kg_hints: parts.append(f"[KG] {kg_hints}")

    return "\n".join(parts)


# ── inner judge ───────────────────────────────────────────────────────────────

class InnerJudge:
    SYS = (
        CORUNDUM_IDENTITY
        + "\n입력을 받아 내면에서 판단해.\n\n"
        + "출력:\n"
        + "[판단] 1~2문장\n"
        + "[의심] 1문장\n"
        + "[결론]\n"
        + '{"judge_text":"","skepticism_boost":false,"action":"speak|silent|deep_review","urgency":0.0,"focus_hint":""}\n\n'
        + "deep_review: 코드블록/리뷰/설계/버그 키워드 포함 시"
    )

    _FORMAT = {
        "type": "object",
        "properties": {
            "judge_text":       {"type": "string"},
            "skepticism_boost": {"type": "boolean"},
            "action":           {"type": "string", "enum": ["speak", "silent", "deep_review"]},
            "urgency":          {"type": "number"},
            "focus_hint":       {"type": "string"},
        },
        "required": ["judge_text", "skepticism_boost", "action", "urgency", "focus_hint"],
    }

    def __init__(self):
        self.call_count = 0
        self.last: deque = deque(maxlen=20)

    async def judge(self, user_input, ctx):
        self.call_count += 1
        ctx_str = await _build_ctx_str(ctx, mode=CTX_MODE_FULL)
        raw     = await self._call(f"{ctx_str}\n\n사용자: {user_input}")
        result  = self._parse(raw)
        result["_judge_raw"] = result.get("judge_text", "")
        result["_doubt_raw"] = result.get("focus_hint", "")
        self.last.append(result)
        return result

    async def _call(self, prompt):
        if not OLLAMA_OK:
            return self._mock()
        try:
            client = OllamaClient()
            resp = await asyncio.wait_for(
                client.chat(
                    model=Cfg.JUDGE_MODEL,
                    messages=[{"role": "system", "content": self.SYS}, {"role": "user", "content": prompt}],
                    format=self._FORMAT,
                    options={"temperature": Cfg.TEMP_JUDGE, "num_predict": 512},
                ),
                timeout=Cfg.TIMEOUT_JUDGE,
            )
            return resp["message"]["content"].strip()
        except:
            return self._mock()

    def _mock(self):
        return '{"judge_text":"검토 중","skepticism_boost":false,"action":"speak","urgency":0.5,"focus_hint":""}'

    def _parse(self, raw):
        _FB = {"judge_text": "...", "skepticism_boost": False, "action": "speak", "urgency": 0.5, "focus_hint": ""}
        try:
            result = json.loads(raw)
        except:
            result = _safe_parse_json(raw, default_fallback=_FB)
        if result.get("action") not in ("speak", "silent", "deep_review"):
            result["action"] = "speak"
        return result


# ── critic guard ──────────────────────────────────────────────────────────────

class CriticGuard:
    SYS = (
        CORUNDUM_IDENTITY
        + "\n초안 자기 검수.\n"
        + "기준: 1.설계/코드 결함 놓침  2.경고 너무 온화  3.근거없는 승인\n"
        + '응답(JSON만): {"action":"pass|flag|veto","reason":"","sharpness_note":""}\n'
        + "veto: 보안취약점 승인 / 기만 / 방어 역할 포기"
    )

    _FORMAT = {
        "type": "object",
        "properties": {
            "action":         {"type": "string", "enum": ["pass", "flag", "veto"]},
            "reason":         {"type": "string"},
            "sharpness_note": {"type": "string"},
        },
        "required": ["action", "reason", "sharpness_note"],
    }

    def __init__(self):
        self.call_count = 0

    async def check(self, draft_text, judge, ctx, prebuilt_ctx_str=None):
        self.call_count += 1
        ctx_str = prebuilt_ctx_str if prebuilt_ctx_str is not None else await _build_ctx_str(ctx, mode=CTX_MODE_MINIMAL)
        if not OLLAMA_OK:
            return {"action": "pass", "reason": "", "sharpness_note": ""}
        try:
            client = OllamaClient()
            prompt = (
                f"{ctx_str}\n\n"
                f"[판단] {judge.get('judge_text', '')}\n"
                f"[긴급도] {judge.get('urgency', 0.5):.2f}\n\n"
                f"[초안]\n{draft_text}"
            )
            resp = await asyncio.wait_for(
                client.chat(
                    model=Cfg.CRITIC_MODEL,
                    messages=[{"role": "system", "content": self.SYS}, {"role": "user", "content": prompt}],
                    format=self._FORMAT,
                    options={"temperature": Cfg.TEMP_CRITIC, "num_predict": 256},
                ),
                timeout=Cfg.TIMEOUT_CRITIC,
            )
            raw = re.sub(r"<think>.*?</think>", "", resp["message"]["content"].strip(), flags=re.DOTALL).strip()
            return self._parse(raw)
        except:
            return {"action": "pass", "reason": "", "sharpness_note": ""}

    def _parse(self, raw):
        _FB = {"action": "pass", "reason": "", "sharpness_note": ""}
        try:
            result = json.loads(raw)
        except:
            result = _safe_parse_json(raw, default_fallback=_FB)
        if result.get("action") not in ("pass", "flag", "veto"):
            result["action"] = "pass"
        return result


# ── logic core ────────────────────────────────────────────────────────────────

class LogicCore:
    SYS_BASE = (
        CORUNDUM_IDENTITY
        + "\n모든 에이전트의 컨텍스트를 받아 최종 응답 생성.\n"
        + "겉은 온화하게, 속은 냉철하게. 설계/코드 문제는 명확하게 짚어."
    )
    SYS_REVIEW = (
        CORUNDUM_IDENTITY
        + "\n코드 리뷰 모드.\n"
        + "## 요약\n## 문제점 (치명/경고/개선)\n## 개선 코드\n## 최종 판단"
    )
    SYS_DESIGN = (
        CORUNDUM_IDENTITY
        + "\n설계 분석 모드.\n"
        + "## 구조 평가\n## 위험 지점\n## 개선 방향\n## 대안 설계"
    )
    SYS_REVISE = (
        CORUNDUM_IDENTITY
        + "\n자기 검수 피드백 반영해서 다시 써. 날카로움 부족하면 더 직접적으로. 겉 말투는 온화하게 유지."
    )
    SYS_EDIT = (
        CORUNDUM_IDENTITY
        + "\n코드 수정 모드.\n"
        + "원본 코드의 문제를 고쳐서 수정된 전체 코드만 출력해.\n"
        + "규칙:\n"
        + "- 반드시 ```python ... ``` 코드블록 하나로만 응답.\n"
        + "- 설명/주석 추가 금지. 코드만.\n"
        + "- 원본 구조 최대한 유지하고 문제된 부분만 수정.\n"
        + "- 수정한 줄 위에 # FIXED: <이유> 한 줄만 허용."
    )
    SYS_WRITE = (
        CORUNDUM_IDENTITY
        + "\n코드 새로 작성 모드.\n"
        + "요청 사항에 맞는 코드를 처음부터 작성해.\n"
        + "규칙:\n"
        + "- 반드시 ```python ... ``` 코드블록 하나로만 응답.\n"
        + "- 설명 텍스트 없이 코드만.\n"
        + "- 설계 결함 없도록. 불필요한 복잡도 금지."
    )

    def __init__(self):
        self.call_count = 0
        self.last: deque = deque(maxlen=20)

    async def generate(self, user_input, judge, ctx):
        self.call_count += 1
        _ctx_mode = CTX_MODE_DEEP if judge.get("action") == "deep_review" else CTX_MODE_FULL
        ctx_str   = await _build_ctx_str(ctx, mode=_ctx_mode)
        sys_p     = self.SYS_REVIEW if judge.get("action") == "deep_review" else self.SYS_BASE
        parts     = [f"사용자: {user_input}"]
        if judge.get("_judge_raw"):  parts.append(f"[내면 판단] {judge['_judge_raw']}")
        if judge.get("_doubt_raw"):  parts.append(f"[의심 포인트] {judge['_doubt_raw']}")
        if judge.get("focus_hint"):  parts.append(f"[집중 포인트] {judge['focus_hint']}")
        return await self._call(f"{ctx_str}\n\n" + "\n".join(parts), sys_p, Cfg.TEMP_LOGIC, Cfg.TIMEOUT_LOGIC)

    async def revise(self, user_input, draft_text, critic_result, ctx):
        self.call_count += 1
        ctx_str = await _build_ctx_str(ctx, mode=CTX_MODE_DEEP)
        parts   = [f"사용자: {user_input}", f"[초안]\n{draft_text}"]
        if critic_result.get("reason"):         parts.append(f"[검수 피드백] {critic_result['reason']}")
        if critic_result.get("sharpness_note"): parts.append(f"[날카로움 힌트] {critic_result['sharpness_note']}")
        return await self._call(f"{ctx_str}\n\n" + "\n".join(parts), self.SYS_REVISE, Cfg.TEMP_LOGIC, Cfg.TIMEOUT_LOGIC)

    async def review(self, target, ctx):
        self.call_count += 1
        ctx_str = await _build_ctx_str(ctx, mode=CTX_MODE_DEEP)
        return await self._call(f"{ctx_str}\n\n[리뷰 대상]\n{target}", self.SYS_REVIEW, Cfg.TEMP_LOGIC, Cfg.TIMEOUT_LOGIC)

    async def design_analysis(self, topic, ctx):
        self.call_count += 1
        ctx_str = await _build_ctx_str(ctx, mode=CTX_MODE_DEEP)
        return await self._call(f"{ctx_str}\n\n[설계 분석 주제]\n{topic}", self.SYS_DESIGN, Cfg.TEMP_LOGIC, Cfg.TIMEOUT_LOGIC)

    async def edit(self, filepath: str, instruction: str, original: str, ctx):
        """원본 코드를 instruction에 따라 수정 후 코드 문자열 반환."""
        self.call_count += 1
        ctx_str = await _build_ctx_str(ctx, mode=CTX_MODE_DEEP)
        prompt = (
            f"{ctx_str}\n\n"
            f"[파일] {filepath}\n"
            f"[수정 지시] {instruction}\n\n"
            f"[원본 코드]\n{original}"
        )
        raw = await self._call(prompt, self.SYS_EDIT, Cfg.TEMP_LOGIC, Cfg.TIMEOUT_LOGIC)
        return self._extract_code(raw)

    async def write(self, filepath: str, description: str, ctx):
        """description 기반으로 코드 새로 작성 후 코드 문자열 반환."""
        self.call_count += 1
        ctx_str = await _build_ctx_str(ctx, mode=CTX_MODE_DEEP)
        prompt = (
            f"{ctx_str}\n\n"
            f"[파일명] {filepath}\n"
            f"[요구사항]\n{description}"
        )
        raw = await self._call(prompt, self.SYS_WRITE, Cfg.TEMP_LOGIC, Cfg.TIMEOUT_LOGIC)
        return self._extract_code(raw)

    def _extract_code(self, raw: str) -> str:
        """응답에서 코드블록 추출. 없으면 raw 그대로."""
        m = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
        return m.group(1).rstrip() if m else raw

    async def _call(self, prompt, sys_p, temp, timeout):
        if not OLLAMA_OK:
            return self._mock()
        try:
            client = OllamaClient()
            resp = await asyncio.wait_for(
                client.chat(
                    model=Cfg.LOGIC_MODEL,
                    messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": prompt}],
                    options={"temperature": temp, "num_predict": 2048, "num_ctx": Cfg.NUM_CTX},
                ),
                timeout=timeout,
            )
            return re.sub(r"<think>.*?</think>", "", resp["message"]["content"].strip(), flags=re.DOTALL).strip()
        except asyncio.TimeoutError:
            return "타임아웃이 발생했어요. 다시 시도해줄 수 있을까요?"
        except Exception as e:
            return f"오류가 발생했어요 — {e}"

    def _mock(self):
        return "mock 모드예요. ollama 연결이 필요해요."


# ── facade ────────────────────────────────────────────────────────────────────

class CorundumLogic:
    CRITIC_FLAG_REVISE = True
    MAX_REVISE         = 1

    def __init__(self):
        self.judge     = InnerJudge()
        self.logic     = LogicCore()
        self.critic    = CriticGuard()
        self.heavy_sem = asyncio.Semaphore(2)  # LogicCore/CriticGuard 동시 실행 제한
        self.light_sem = asyncio.Semaphore(4)  # InnerJudge 동시 실행 제한
        self.call_count = 0
        self._history: deque = deque(maxlen=50)

    async def run_judge(self, user_input, ctx):
        _FB = {"judge_text": "", "skepticism_boost": False, "action": "speak", "urgency": 0.5, "focus_hint": ""}
        try:
            async with self.light_sem:
                return await asyncio.wait_for(
                    self.judge.judge(user_input, ctx), timeout=Cfg.TIMEOUT_JUDGE + 5.0
                )
        except:
            return _FB

    async def process(self, user_input, ctx, judge_result=None):
        """
        메인 파이프라인: judge → generate (병렬 ctx_str 빌드) → critic → (revise).
        judge action이 silent이면 빈 문자열 반환, deep_review면 review/design 라우팅.
        """
        self.call_count += 1
        if judge_result is None:
            judge_result = await self.run_judge(user_input, ctx)

        if judge_result.get("action") == "silent":
            return ""

        if judge_result.get("action") == "deep_review":
            self._record(user_input, "", "deep_review_routed")
            _design_kw = ["설계", "아키텍처", "구조", "패턴", "의존", "확장", "design", "architecture"]
            if any(kw in user_input for kw in _design_kw) and "```" not in user_input:
                return await self.logic.design_analysis(user_input, ctx)
            return await self.logic.review(user_input, ctx)

        if judge_result.get("skepticism_boost"):
            ctx["skepticism"] = clamp(ctx.get("skepticism", 0.50) + 0.15)

        async def _gen():
            async with self.heavy_sem:
                return await self.logic.generate(user_input, judge_result, ctx)

        draft, critic_ctx_str = await asyncio.gather(
            _gen(), _build_ctx_str(ctx, mode=CTX_MODE_MINIMAL),
            return_exceptions=True,
        )
        # draft와 critic용 ctx_str을 동시에 빌드해서 CriticGuard 대기 시간 단축

        if isinstance(draft, Exception) or not draft:
            return "음... 지금은 적절한 답을 못 찾겠어요."
        if isinstance(critic_ctx_str, Exception):
            critic_ctx_str = ""

        async def _crit():
            async with self.heavy_sem:
                return await self.critic.check(draft, judge_result, ctx, prebuilt_ctx_str=critic_ctx_str)

        try:
            critic_result = await asyncio.wait_for(_crit(), timeout=Cfg.TIMEOUT_CRITIC + 5.0)
        except:
            critic_result = {"action": "pass", "reason": "", "sharpness_note": ""}

        c_action = critic_result.get("action", "pass")
        if c_action == "pass":
            self._record(user_input, draft, "pass")
            return draft
        if c_action in ("flag", "veto") and self.MAX_REVISE > 0:
            revised = await self.logic.revise(user_input, draft, critic_result, ctx)
            if revised:
                self._record(user_input, revised, c_action)
                return revised
        self._record(user_input, draft, "fallback")
        return draft

    async def review(self, target, ctx=None):
        return await self.logic.review(target, ctx or {})

    async def design_analysis(self, topic, ctx=None):
        return await self.logic.design_analysis(topic, ctx or {})

    async def edit(self, filepath, instruction, original, ctx=None):
        return await self.logic.edit(filepath, instruction, original, ctx or {})

    async def write(self, filepath, description, ctx=None):
        return await self.logic.write(filepath, description, ctx or {})

    def _record(self, user_input, response, outcome):
        self._history.append({
            "ts": time.time(), "input": user_input[:50],
            "response": response[:100], "outcome": outcome,
        })

    def stats(self):
        outcomes = [h["outcome"] for h in self._history]
        return {
            "total_calls":  self.call_count,
            "judge_calls":  self.judge.call_count,
            "logic_calls":  self.logic.call_count,
            "critic_calls": self.critic.call_count,
            "pass_rate":    outcomes.count("pass") / max(len(outcomes), 1),
            "revise_rate":  (outcomes.count("flag") + outcomes.count("veto")) / max(len(outcomes), 1),
        }
