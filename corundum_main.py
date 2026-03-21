#!/usr/bin/env python3
# corundum_main.py
# CORUNDUM — code defense + design analysis AI

import asyncio
import copy
import logging
import os
import time
from typing import Dict, Optional
from importlib import import_module as _im

log = logging.getLogger("corundum")

# ── module imports ────────────────────────────────────────────────────────────

_cfg_mod          = _im("corundum_config")
Cfg               = _cfg_mod.Cfg
CORUNDUM_IDENTITY = _cfg_mod.CORUNDUM_IDENTITY
BOOT_MSG          = _cfg_mod.BOOT_MSG

CorundumMemory  = _im("corundum_memory").CorundumMemory
CorundumEmotion = _im("corundum_emotion").CorundumEmotion
CorundumGoal    = _im("corundum_goal").CorundumGoal
CorundumLogic   = _im("corundum_logic").CorundumLogic
MetricsEngine   = _im("corundum_metrics").MetricsEngine

from corundum_agency import CorundumAgency
from corundum_voice  import make_listener

try:
    from ollama import AsyncClient as OllamaClient
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False
    log.warning("ollama not found — running in mock mode")


# ── context assembler ─────────────────────────────────────────────────────────

class ContextAssembler:
    """
    Assembles all agent outputs into a single context dict for LogicCore.

    Key mapping:
      memory  -> recalled, kg_hints, working
      emotion -> emotion_tag, emotion_hint, inner_hint,
                 immersion, focus, skepticism, patience, curiosity,
                 surface_warmth, inner_edge, gear, fatigue
      goal    -> current_goal, goal_hint, self_critique, urgency
      metrics -> energy, fatigue, gear  (emotion gear takes priority)

    Note: gear는 emotion 쪽이 우선됨.
    MetricsEngine은 energy/fatigue 기반 rule-only이고 OVERDRIVE/FOCUS를 내지 않음.
    OVERDRIVE/FOCUS는 emotion skepticism 기반이라 CorundumEmotion이 담당.
    """

    @staticmethod
    def assemble(memory_ctx: Dict, emotion_ctx: Dict, goal_ctx: Dict, metrics_ctx: Dict) -> Dict:
        gear    = emotion_ctx.get("gear")    or metrics_ctx.get("gear",    "NORMAL")
        fatigue = emotion_ctx.get("fatigue") or metrics_ctx.get("fatigue", 0.0)

        return {
            "recalled_memory": memory_ctx.get("recalled", ""),
            "kg_hints":        memory_ctx.get("kg_hints", ""),
            "working_memory":  memory_ctx.get("working",  ""),

            "emotion_tag":    emotion_ctx.get("emotion_tag",    "neutral"),
            "emotion_hint":   emotion_ctx.get("emotion_hint",   ""),
            "inner_hint":     emotion_ctx.get("inner_hint",     ""),
            "immersion":      emotion_ctx.get("immersion",      1.0),
            "focus":          emotion_ctx.get("focus",          0.75),
            "skepticism":     emotion_ctx.get("skepticism",     0.50),
            "patience":       emotion_ctx.get("patience",       0.80),
            "curiosity":      emotion_ctx.get("curiosity",      0.60),
            "surface_warmth": emotion_ctx.get("surface_warmth", 0.70),
            "inner_edge":     emotion_ctx.get("inner_edge",     0.65),

            "current_goal":  goal_ctx.get("current_goal", ""),
            "goal_hint":     goal_ctx.get("goal_hint",    ""),
            "self_critique": goal_ctx.get("critique",     ""),
            "urgency":       float(goal_ctx.get("urgency", 0.3)),  # LLM이 str로 반환할 수 있음

            "energy":  metrics_ctx.get("energy", 1.0),
            "fatigue": fatigue,
            "gear":    gear,
        }


# ── main engine ───────────────────────────────────────────────────────────────

class Corundum:
    """
    CORUNDUM — code defense + design analysis AI.

    Processing flow:
      input
       |- [parallel] CorundumMemory      (KG/vector recall)
       |- [parallel] CorundumEmotion     (emotion physics)
       |- [parallel] CorundumGoal        (goal + self-critique LLM)
       |- [parallel] InnerJudge          (judgment impulse, partial ctx)
       +- MetricsEngine                  (energy/fatigue/gear, rule-based)
              |
       ContextAssembler
              |
       CorundumLogic.process()
         |- [parallel] LogicCore.generate  (draft, 120s)
         +- [parallel] _build_ctx_str      (pre-build for CriticGuard)
                       |
         CriticGuard.check()               (reuses prebuilt ctx)
              |
       post: metrics.on_response / memory.record /
             emotion.post_feedback / goal.observe_outcome
    """

    def __init__(self):
        log.info("corundum: initializing")

        self.memory  = CorundumMemory()
        self.metrics = MetricsEngine()
        self.emotion = CorundumEmotion()
        self.goal    = CorundumGoal()
        self.logic   = CorundumLogic()

        # agency (린 구조)
        self.agency = CorundumAgency(safe_mode=True)
        self.agency.attach(self)

        # 음성
        self.voice = make_listener(
            wake_name  = Cfg.WAKE_NAME,
            on_wake    = self._on_voice_wake,
            on_command = self._on_voice_command,
            model_size = getattr(Cfg, "WHISPER_MODEL", "small"),
        )

        # DORMANT: True면 LLM 호출 없음. 호출어/키보드로 깨어남.
        self._dormant: bool = True
        self.on_autonomous: Optional[Callable] = None

        self.running:           bool  = False
        self.last_input_t:      float = time.time()
        self.interaction_count: int   = 0
        self._bg_tasks:         set   = set()

        self._nx_task:      Optional[asyncio.Task] = None
        self._nx_interrupt: asyncio.Event          = asyncio.Event()

        log.info("corundum: ready")
        self._print_status()

    def _print_status(self):
        ok = lambda b: "ok" if b else "missing"
        print(
            f"CORUNDUM | code defense + design | "
            f"ollama:{ok(OLLAMA_OK)} | "
            f"logic:{Cfg.LOGIC_MODEL} | "
            f"judge:{Cfg.JUDGE_MODEL}"
        )

    # ── main process ──────────────────────────────────────────────────────────

    async def process(self, user_input: str) -> str:
        if self._dormant:
            return ""
        self.last_input_t = time.time()

        # agency 임무 감지
        agency_result = await self.agency.on_impulse(user_input, {})
        if agency_result and agency_result.get("task_started"):
            return (
                f"임무 시작: {agency_result['goal'][:60]}\n"
                "잠수합니다. 완료되면 알려드릴게요. (/abort 로 중단)"
            )

        if self._nx_task and not self._nx_task.done():
            self._nx_interrupt.set()
            self._nx_task.cancel()
            try:
                await asyncio.wait_for(self._nx_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            log.debug("process: nx autonomous reasoning interrupted")

        idle = 0.0

        metrics_ctx = self.metrics.tick(idle_sec=idle, interaction_count=self.interaction_count)

        # memory / emotion / goal / judge 4개 동시 실행 후 gather
        memory_task  = asyncio.create_task(self.memory.recall(user_input, metrics_ctx))
        emotion_task = asyncio.create_task(self.emotion.process(user_input, metrics_ctx))
        goal_task    = asyncio.create_task(self.goal.process(user_input, metrics_ctx))
        # episodes는 memory 로드 전에 None일 수 있으므로 getattr로 안전하게 접근
        _episodes    = getattr(self.memory, "episodes", None)
        _working_ctx = _episodes.working_context(n=4) if _episodes else ""
        judge_ctx    = {**metrics_ctx, "working_memory": _working_ctx}
        judge_task   = asyncio.create_task(self.logic.run_judge(user_input, judge_ctx))

        memory_ctx, emotion_ctx, goal_ctx, judge_result = await asyncio.gather(
            memory_task, emotion_task, goal_task, judge_task,
            return_exceptions=True,
        )

        if isinstance(memory_ctx,   Exception): log.warning("memory agent failed: %s",  memory_ctx);  memory_ctx  = {}
        if isinstance(emotion_ctx,  Exception): log.warning("emotion agent failed: %s", emotion_ctx); emotion_ctx = {}
        if isinstance(goal_ctx,     Exception): log.warning("goal agent failed: %s",    goal_ctx);    goal_ctx    = {}
        if isinstance(judge_result, Exception): log.warning("judge agent failed: %s",   judge_result); judge_result = None

        ctx = ContextAssembler.assemble(memory_ctx, emotion_ctx, goal_ctx, metrics_ctx)

        if judge_result and judge_result.get("skepticism_boost"):
            ctx["skepticism"] = min(1.0, ctx.get("skepticism", 0.50) + 0.15)

        frozen_ctx = copy.deepcopy(ctx)
        response   = await self.logic.process(user_input, frozen_ctx, judge_result=judge_result)
        if not response:
            response = "..."

        self.interaction_count += 1
        self.metrics.on_response(response)
        self.memory.record(user_input, response, emotion_ctx, goal_ctx)

        self._bg(self.emotion.post_feedback(response, ctx))
        self._bg(self._observe_goal(user_input, response,
                                    "pass" if self.logic.stats().get("pass_rate", 1.0) > 0.5 else "flag",
                                    emotion_ctx.get("emotion_tag", "neutral")))
        self.emotion.on_interact()

        return response

    async def _observe_goal(self, user_input, response, outcome, emotion_tag):
        self.goal.observe_outcome(event_tag=emotion_tag, action_summary=response[:40], outcome=outcome)

    def _bg(self, coro):
        task = asyncio.ensure_future(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    # ── autonomous loop ───────────────────────────────────────────────────────

    async def _autonomous_loop(self):
        while self.running:
            interval = (
                getattr(Cfg, "DORMANT_AUTO_INTERVAL", 120.0)
                if self._dormant else Cfg.AUTO_INTERVAL
            )
            await asyncio.sleep(interval)
            if self._dormant or self.agency.is_running:
                continue
            idle = time.time() - self.last_input_t
            if idle < Cfg.AUTO_IDLE_THRESHOLD:
                continue
            metrics_ctx = self.metrics.tick(idle_sec=idle)
            gear        = metrics_ctx.get("gear", "NORMAL")
            result = await self.goal.autonomous_tick(metrics_ctx)
            if result:
                print(f"\n[autonomous] {result}")
            if gear in ("SAVE", "LOW", "SLEEP", "DREAM"):
                continue
            asyncio.create_task(self._cogn_tick(metrics_ctx))

    async def _cogn_tick(self, metrics_ctx: dict):
        """
        Autonomous cognition tick: InnerJudge (AG) -> LogicCore (NX).

        impulse < 0.35  -> zoning (idle, no output)
        0.35 <= imp < 0.55 -> wandering (code association monologue)
        imp >= 0.55     -> NX: LogicCore autonomous reasoning task
        """
        try:
            recent     = self.memory.episodes.working_context(n=6)
            judge_ctx  = {**metrics_ctx, "working_memory": recent}
            judge_result = await self.logic.run_judge("(alone)", judge_ctx)
            action = judge_result.get("action", "silent")
            imp    = float(judge_result.get("urgency", 0.0))

            if action == "silent" and imp < 0.35:
                await self._do_zoning(imp)
                return

            if imp < 0.55:
                await self._do_wander(recent, judge_result, metrics_ctx)
                return

            if self._nx_task and not self._nx_task.done():
                log.debug("cogn_tick: nx already running, skip")
                return

            self._nx_interrupt.clear()

            async def _run_nx():
                try:
                    result = await self._nx_think(recent, judge_result, metrics_ctx)
                    if result:
                        print(f"\n코런덤: {result}\n")
                except asyncio.CancelledError:
                    log.debug("nx: autonomous reasoning cancelled")
                except Exception as e:
                    log.warning("nx: autonomous reasoning failed: %s", e)

            self._nx_task = asyncio.create_task(_run_nx())
            log.info("cogn_tick: nx started (impulse=%.2f)", imp)

        except Exception as e:
            log.error("cogn_tick: crash guard: %s", e)

    async def _do_zoning(self, imp: float):
        zoning_sec = 2.0 + imp * 6.0
        log.debug("zoning: %.1fs", zoning_sec)
        await asyncio.sleep(zoning_sec)

    async def _do_wander(self, recent: str, judge: dict, metrics_ctx: dict):
        """Lightweight code association monologue using the judge model."""
        _SYS = (
            CORUNDUM_IDENTITY
            + "\n\n혼자 멍하니 이전 대화를 떠올리는 중이야.\n"
            + "코드나 설계 관련 연상이 머릿속에서 흘러가고 있어.\n\n"
            + "규칙:\n"
            + "- 연상 흐름을 1~2문장으로. 독백처럼.\n"
            + "- 결론 내리려 하지 마. 그냥 흘러가는 생각.\n"
            + "- 없으면 빈 문자열.\n\n"
            + "응답: 독백 1~2문장 또는 빈 문자열."
        )
        focus_hint = judge.get("focus_hint", "")
        _prompt    = "\n".join(filter(None, [
            f"[이전 대화]\n{recent}" if recent else None,
            f"[걸리는 부분] {focus_hint}" if focus_hint else None,
        ])) or "(이전 대화 없음)"

        if not OLLAMA_OK:
            return
        try:
            from ollama import AsyncClient as OllamaClient
            client = OllamaClient()
            resp   = await asyncio.wait_for(
                client.chat(
                    model=Cfg.JUDGE_MODEL,
                    messages=[{"role": "system", "content": _SYS}, {"role": "user", "content": _prompt}],
                    options={"temperature": 0.90, "num_predict": 60},
                ),
                timeout=15.0,
            )
            result = resp["message"]["content"].strip()
            if result:
                print(f"\n코런덤: {result}\n")
        except Exception as e:
            log.debug("wander: failed: %s", e)

    async def _nx_think(self, recent: str, judge: dict, metrics_ctx: dict) -> str:
        """NX autonomous reasoning — deep thought via LogicCore."""
        _SYS = (
            CORUNDUM_IDENTITY
            + "\n\n지금 아무도 없는데 혼자 생각하고 있어.\n"
            + "이전 대화에서 걸리는 게 있거나, 더 파고들고 싶은 게 생겼어.\n\n"
            + "규칙:\n"
            + "- 2~3문장. 독백 또는 혼잣말.\n"
            + "- 설계/코드에 대한 통찰이나 의문.\n"
            + "- 사용자한테 말하는 게 아니라 혼자 중얼거리는 거야.\n"
            + "- 없으면 빈 문자열.\n\n"
            + "응답: 혼잣말 2~3문장 또는 빈 문자열."
        )
        focus_hint  = judge.get("focus_hint", "")
        judge_inner = judge.get("_judge_raw", "")
        _prompt     = "\n".join(filter(None, [
            f"[이전 대화]\n{recent}" if recent else None,
            f"[걸리는 부분] {focus_hint}" if focus_hint else None,
            f"[내면 판단] {judge_inner}" if judge_inner else None,
        ])) or "(이전 대화 없음)"

        if self._nx_interrupt.is_set():
            return ""
        if not OLLAMA_OK:
            return ""
        try:
            from ollama import AsyncClient as OllamaClient
            client = OllamaClient()
            resp   = await asyncio.wait_for(
                client.chat(
                    model=Cfg.LOGIC_MODEL,
                    messages=[{"role": "system", "content": _SYS}, {"role": "user", "content": _prompt}],
                    options={"temperature": 0.75, "num_predict": 100},
                ),
                timeout=30.0,
            )
            if self._nx_interrupt.is_set():
                return ""
            return resp["message"]["content"].strip()
        except asyncio.TimeoutError:
            log.debug("nx_think: timeout")
            return ""
        except Exception as e:
            log.debug("nx_think: failed: %s", e)
            return ""

    # ── physio loop ───────────────────────────────────────────────────────────

    async def _physio_loop(self):
        while self.running:
            idle  = time.time() - self.last_input_t
            state = self.metrics.tick(idle_sec=idle)
            gear  = state.get("gear", "NORMAL")
            self.emotion.physio_tick(idle_sec=idle)
            if gear in ("SLEEP", "DREAM"):
                await self.memory.consolidate()
            if self._dormant:
                interval = getattr(Cfg, "DORMANT_PHYSIO_INTERVAL", 30.0)
            else:
                interval = Cfg.GEAR_INTERVALS.get(gear, 20.0)
            await asyncio.sleep(min(interval, 30.0))
            await self.agency.tick(state)

    # ── chat loop ─────────────────────────────────────────────────────────────

    async def chat_loop(self):
        self.running = True
        loop         = asyncio.get_event_loop()

        print(f"DORMANT — \"{Cfg.WAKE_NAME}\" 라고 부르거나 아무 키나 누르세요.\n")

        physio_task = asyncio.create_task(self._physio_loop())
        auto_task   = asyncio.create_task(self._autonomous_loop())

        self.on_autonomous = lambda msg: print(f"\n코런덤: {msg}")
        await self.voice.start()

        while self.running:
            try:
                user_input = await loop.run_in_executor(None, input, "")
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "종료", "끝"):
                break

            # 키보드 입력 → DORMANT 해제
            if self._dormant:
                self._dormant = False
                self.voice.force_awake()
                print("\n코런덤: 네.")

            # 임무 중 잠수
            if self.agency.is_running:
                c = user_input.strip().lower()
                if c in ("/abort", "abort", "중단", "멈춰", "그만"):
                    print(f"\n코런덤: {self.agency.abort()}")
                else:
                    print(f"\n코런덤: {self.agency.dive_response()}")
                continue

            if user_input.startswith("/"):
                response = await self._handle_command(user_input)
            else:
                response = await self.process(user_input)

            if response:
                print(f"\n코런덤: {response}")

        physio_task.cancel()
        auto_task.cancel()
        self.voice.stop()
        await asyncio.gather(physio_task, auto_task, return_exceptions=True)
        await self.memory.save()
        print("\n[ CORUNDUM session ended ]")

    # ── 음성 콜백 ─────────────────────────────────────────────────────────────

    async def _on_voice_wake(self, remainder: str = ""):
        if self._dormant:
            self._dormant = False
            print("\n코런덤: 네.")
        if remainder:
            await self._on_voice_command(remainder)

    async def _on_voice_command(self, text: str):
        if not text:
            return
        print(f"\n[음성] {text}")
        if self.agency.is_running:
            if text.strip().lower() in ("abort", "중단", "멈춰", "그만"):
                print(f"코런덤: {self.agency.abort()}")
            else:
                print(f"코런덤: {self.agency.dive_response()}")
            return
        if text.startswith("/"):
            response = await self._handle_command(text)
        else:
            response = await self.process(text)
        if response:
            print(f"코런덤: {response}")

    # ── command handler ───────────────────────────────────────────────────────

    async def _handle_command(self, cmd: str) -> str:
        parts = cmd.strip().split()
        c     = parts[0].lower()

        if c == "/status":
            m  = self.metrics.snapshot()
            et = self.emotion.current_tag()
            ls = self.logic.stats()
            gs = self.goal.stats()
            return (
                f"CORUNDUM status\n"
                f"  gear:        {m['gear']}\n"
                f"  energy:      {m['energy']:.2f}\n"
                f"  fatigue:     {m['fatigue']:.2f}\n"
                f"  emotion:     {et}\n"
                f"  interactions:{self.interaction_count}\n"
                f"  top goal:    {self.goal.top_goal_name()}\n"
                f"  pass rate:   {ls.get('pass_rate', 1.0):.0%}\n"
                f"  error rate:  {gs.get('error_rate', 0.0):.0%}"
            )

        elif c == "/goals":
            return self.goal.summary()

        elif c == "/goal":
            if len(parts) < 2:
                return "usage: /goal <name>"
            return self.goal.add_goal(" ".join(parts[1:]))

        elif c == "/memory":
            return self.memory.recent_summary()

        elif c == "/kg":
            q = " ".join(parts[1:]) if len(parts) > 1 else ""
            return self.memory.kg.query(q, top_k=10) or "KG empty"

        elif c == "/review":
            if len(parts) < 2:
                return "usage: /review <code or filepath>"
            target = " ".join(parts[1:])
            from pathlib import Path
            p = Path(target)
            if p.exists() and p.is_file():
                try:
                    target = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    try:
                        target = p.read_text(encoding="cp949")
                    except Exception as e:
                        return f"파일 읽기 실패 (인코딩 오류): {e}"
                except PermissionError as e:
                    return f"파일 읽기 실패 (권한 없음): {e}"
                except Exception as e:
                    return f"파일 읽기 실패: {e}"
            elif p.exists() and p.is_dir():
                return f"디렉토리는 리뷰할 수 없어요: {target}"
            elif "/" in target or "\\" in target:
                # 경로처럼 보이는데 파일이 없는 경우
                return f"파일을 찾을 수 없어요: {target}"
            # 경로가 아니면 코드 문자열로 그대로 리뷰
            return await self.logic.review(target, ctx=self.metrics.snapshot())

        elif c == "/edit":
            # usage: /edit <filepath> <instruction>
            if len(parts) < 3:
                return "usage: /edit <filepath> <instruction>"
            from pathlib import Path
            filepath = parts[1]
            instruction = " ".join(parts[2:])
            p = Path(filepath)
            if not p.exists() or not p.is_file():
                return f"파일을 찾을 수 없어요: {filepath}"
            try:
                original = p.read_text(encoding="utf-8")
            except Exception as e:
                return f"파일 읽기 실패: {e}"
            code = await self.logic.edit(filepath, instruction, original, ctx=self.metrics.snapshot())
            if not code or code == original:
                return "수정 사항이 없거나 생성 실패예요."
            # 백업 후 덮어쓰기
            backup = p.with_suffix(p.suffix + ".bak")
            backup.write_text(original, encoding="utf-8")
            p.write_text(code, encoding="utf-8")
            return f"수정 완료 → {filepath}\n백업 → {backup}\n\n```python\n{code[:800]}{'...' if len(code) > 800 else ''}\n```"

        elif c == "/write":
            # usage: /write <filepath> <description>
            if len(parts) < 3:
                return "usage: /write <filepath> <description>"
            from pathlib import Path
            filepath = parts[1]
            description = " ".join(parts[2:])
            p = Path(filepath)
            code = await self.logic.write(filepath, description, ctx=self.metrics.snapshot())
            if not code:
                return "코드 생성 실패예요."
            if p.exists():
                backup = p.with_suffix(p.suffix + ".bak")
                backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            p.write_text(code, encoding="utf-8")
            return f"생성 완료 → {filepath}\n\n```python\n{code[:800]}{'...' if len(code) > 800 else ''}\n```"

        elif c == "/design":
            if len(parts) < 2:
                return "usage: /design <topic>"
            return await self.logic.design_analysis(" ".join(parts[1:]), ctx=self.metrics.snapshot())

        elif c == "/stats":
            ls = self.logic.stats()
            gs = self.goal.stats()
            return (
                f"stats\n"
                f"  logic calls:  {ls.get('total_calls', 0)}\n"
                f"  pass rate:    {ls.get('pass_rate', 1.0):.0%}\n"
                f"  revise rate:  {ls.get('revise_rate', 0.0):.0%}\n"
                f"  active goals: {gs.get('active_goals', 0)}\n"
                f"  error rate:   {gs.get('error_rate', 0.0):.0%}"
            )

        elif c in ("/help", "/?"):
            return "\n".join([
                "/status              — 현재 상태",
                "/goals               — 목표 목록",
                "/goal <n>            — 목표 추가",
                "/memory              — 최근 기억",
                "/kg [query]          — 지식 그래프",
                "/review <code>       — 코드 리뷰 (파일 경로 가능)",
                "/edit <file> <inst>  — 파일 수정 (자동 백업)",
                "/write <file> <desc> — 새 파일 작성",
                "/design <topic>      — 설계 분석",
                "/task <description>  — 임무 부여 (잠수 모드)",
                "/abort               — 임무 중단",
                "/computer <goal>     — 컴퓨터 직접 제어",
                "/agency              — agency 상태",
                "/dormant             — 대기 모드 전환",
                "/stats               — 통계",
                "/help                — 이 메시지",
                "quit                 — 종료",
            ])

        elif c == "/task":
            if len(parts) < 2:
                return "usage: /task <임무 설명>"
            description = " ".join(parts[1:])
            def _cb(msg): print(f"  {msg}")
            await self.agency.do_task(description, on_status=_cb)
            return (
                f"임무 시작: {description[:60]}\n"
                "잠수합니다. 완료되면 알려드릴게요. (/abort 로 중단)"
            )

        elif c == "/abort":
            return self.agency.abort()

        elif c == "/computer":
            if len(parts) < 2:
                return "usage: /computer <goal>"
            goal   = " ".join(parts[1:])
            result = await self.agency.do_computer(goal)
            status = "완료" if result.get("success") else "실패"
            output = result.get("output", "")[:300]
            return f"[computer] {status}: {result.get('reason','')}\n{output}" if output else f"[computer] {status}"

        elif c == "/agency":
            return self.agency.status_str()

        elif c == "/dormant":
            self._dormant = True
            self.voice.force_dormant()
            return f"대기 모드. \"{Cfg.WAKE_NAME}\" 라고 불러주세요."

        return f"알 수 없는 명령어예요: {c}  (/help 참고)"

    # ── wakeup words ──────────────────────────────────────────────────────────

    async def _wakeup_words(self) -> str:
        await self.memory._ensure_loaded()

        metrics_ctx = self.metrics.tick()
        recent      = self.memory.episodes.working_context(n=6)
        top_goal    = self.goal.top_goal_name()
        m           = metrics_ctx

        _SYS = (
            CORUNDUM_IDENTITY
            + "\n\n막 켜졌어. 처음으로 하는 말.\n\n"
            + "조건:\n"
            + "- 2문장 이내.\n"
            + "- 이전 대화 기록이 있으면 언급해도 돼.\n"
            + "- 현재 상태(에너지/피로/목표)가 말투에 자연스럽게 배어나오면 좋아.\n"
            + "- 인사말로 시작하지 마.\n"
        )

        _prompt_parts = []
        if recent:
            _prompt_parts.append(f"[이전 세션 기억]\n{recent}")
        if top_goal and top_goal != "없음":
            _prompt_parts.append(f"[현재 목표] {top_goal}")
        _prompt_parts.append(
            f"[현재 상태] 에너지={m['energy']:.2f} 피로={m['fatigue']:.2f} 기어={m['gear']}"
        )

        if not OLLAMA_OK:
            e, f = m["energy"], m["fatigue"]
            if f >= 0.60:   return "좀 쌓인 게 있네요. 가볍게 시작해볼게요."
            if recent:      return "저번에 보던 거 있었죠. 이어서 볼게요."
            if e >= 0.85:   return "코드 가져와요."
            return "왔어요."

        try:
            from ollama import AsyncClient as OllamaClient
            client = OllamaClient()
            resp   = await asyncio.wait_for(
                client.chat(
                    model=Cfg.LOGIC_MODEL,
                    messages=[
                        {"role": "system", "content": _SYS},
                        {"role": "user",   "content": "\n".join(_prompt_parts)},
                    ],
                    options={"temperature": 0.85, "num_predict": 80},
                ),
                timeout=15.0,
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            log.debug("wakeup: failed: %s", e)
            return "왔어요."

    # ── run ───────────────────────────────────────────────────────────────────

    async def run(self):
        print(BOOT_MSG)
        wakeup = await self._wakeup_words()
        print(f"코런덤: {wakeup}\n" + "-" * 50)
        await self.chat_loop()

    def __del__(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.memory.save())
        except Exception:
            pass


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CORUNDUM — code defense + design AI")
    parser.add_argument("--debug",    action="store_true", help="enable debug logging")
    parser.add_argument("--no-color", action="store_true", help="disable color output")
    args = parser.parse_args()

    logging.basicConfig(
        level  = logging.DEBUG if args.debug else logging.INFO,
        format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    corundum = Corundum()
    try:
        asyncio.run(corundum.run())
    except KeyboardInterrupt:
        asyncio.run(corundum.memory.save())
        print("\n[ interrupted ]")
