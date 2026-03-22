#!/usr/bin/env python3
# corundum_agency.py
# CorundumAgency — 에이전시 메인 모듈
#
# 구성:
#   CorundumVision      — 화면 캡처 + LLM 비전 분석
#   CorundumActor       — GUI + 터미널 + 파일 실행
#   CorundumSemanticAnchor — AT-SPI → UIAuto → OCR 폴백
#   CorundumBrain       — 화면 보고 행동 결정
#   CorundumComputer    — Brain 래퍼
#   CorundumWeb         — 검색 + 요약
#   CorundumTask        — 임무 상태 머신 (IDLE→RUNNING→DONE, 잠수 모드)
#   CorundumAgency      — 메인 통합
#
# 핵심 API:
#   agency = CorundumAgency()
#   agency.attach(corundum_instance)
#   await agency.on_impulse(text, ctx)   — 임무 감지 시
#   await agency.do_task(description)    — /task 커맨드
#   agency.abort()                       — /abort
#   await agency.tick(ctx)               — physio_loop 주기 호출

import asyncio
import base64
import io
import json
import logging
import re
import subprocess
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("corundum.agency")

# ── 선택적 의존성 ──────────────────────────────────────────────────────────────

try:
    import pyautogui
    PYAUTOGUI_OK = True
except ImportError:
    PYAUTOGUI_OK = False

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import gi
    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi
    ATSPI_OK = True
except Exception:
    ATSPI_OK = False

try:
    import uiautomation as _uiauto
    UIAUTO_OK = True
except ImportError:
    UIAUTO_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

try:
    from json_repair import repair_json
    JSON_REPAIR_OK = True
except ImportError:
    JSON_REPAIR_OK = False

DEPS = {
    "pyautogui": PYAUTOGUI_OK,
    "PIL":       PIL_OK,
}


def _safe_parse_json(raw: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    if JSON_REPAIR_OK:
        try:
            result = repair_json(text, return_objects=True)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}
    start = text.find("{")
    if start == -1:
        return {}
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


def _get_cfg():
    from importlib import import_module as _im
    return _im("corundum_config").Cfg


# ══════════════════════════════════════════════════════════════════════════════
#  CorundumVision — 화면 캡처 + LLM 비전 분석
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScreenState:
    timestamp:   float
    width:       int  = 1920
    height:      int  = 1080
    b64:         str  = ""
    description: str  = ""
    focused_app: str  = ""
    ocr_text:    str  = ""


class CorundumVision:
    _VIS_SYS = (
        "/no_think\n"
        "화면 스크린샷을 보고 JSON으로 반환해:\n"
        '{"description":"화면에 무엇이 보이는지 (터미널/에디터/브라우저 등 구체적으로)",'
        '"focused_app":"포커스된 앱",'
        '"text_content":"주요 텍스트 300자 (코드/오류 메시지 우선)",'
        '"has_error":"true|false",'
        '"error_summary":"오류 메시지 요약 (있을 때만)"}\n'
        "JSON만."
    )

    def capture(self) -> ScreenState:
        if not DEPS["pyautogui"] or not DEPS["PIL"]:
            return ScreenState(timestamp=time.time(), description="[mock] 캡처 불가")
        try:
            shot = pyautogui.screenshot()
            w, h = shot.size
            buf  = io.BytesIO()
            shot.save(buf, format="PNG")
            b64  = base64.b64encode(buf.getvalue()).decode()
            return ScreenState(timestamp=time.time(), width=w, height=h, b64=b64)
        except Exception as e:
            log.error("[Vision] 캡처 실패: %s", e)
            return ScreenState(timestamp=time.time(), description="[error] 캡처 실패")

    async def understand(self, state: ScreenState) -> ScreenState:
        if not state.b64:
            return state
        try:
            from ollama import AsyncClient
            Cfg = _get_cfg()
            resp = await asyncio.wait_for(
                AsyncClient().chat(
                    model=Cfg.VISION_MODEL,   # VL 전용 모델
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text",  "text": "이 화면을 분석해줘."},
                            {"type": "image", "data": state.b64},
                        ],
                    }],
                    system=self._VIS_SYS,
                    options={"temperature": 0.1, "num_predict": 300},
                ),
                timeout=Cfg.TIMEOUT_VISION,
            )
            data = _safe_parse_json(resp["message"]["content"])
            if data:
                state.description = data.get("description", "")
                state.focused_app = data.get("focused_app", "")
                state.ocr_text    = data.get("text_content", "")
                if data.get("has_error") == "true" and data.get("error_summary"):
                    state.description += f" [오류: {data['error_summary']}]"
        except asyncio.TimeoutError:
            log.debug("[Vision] 타임아웃 — 화면 설명 없이 진행")
        except Exception as e:
            log.debug("[Vision] 이해 실패: %s", e)
        return state


# ══════════════════════════════════════════════════════════════════════════════
#  CorundumActor — 실제 입력 실행
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComputerAction:
    action_type: str
    params:      Dict
    confidence:  float = 1.0
    reason:      str   = ""
    safe:        bool  = True

@dataclass
class ActionResult:
    success: bool
    output:  Any  = None
    error:   str  = ""


class CorundumActor:
    _DESTRUCTIVE = {"delete", "format", "rm", "sudo", "shutdown", "reboot"}

    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode

    async def execute(self, action: ComputerAction, confirm_fn=None) -> ActionResult:
        if self.safe_mode and not action.safe:
            if confirm_fn:
                approved = await confirm_fn(action)
                if not approved:
                    return ActionResult(success=False, error="사용자가 거부함")
            else:
                return ActionResult(success=False, error="SAFE_MODE — 파괴적 행동 차단")

        atype = action.action_type
        p     = action.params

        # ── 터미널 명령 (코런덤 주력) ─────────────────────────────────────────
        if atype == "run_command":
            cmd = p.get("cmd", "")
            if self.safe_mode:
                for kw in self._DESTRUCTIVE:
                    if re.search(rf"\b{kw}\b", cmd):
                        return ActionResult(success=False, error=f"SAFE_MODE: 차단 ({kw})")
            try:
                res = subprocess.run(
                    cmd, shell=True,
                    capture_output=True, text=True,
                    timeout=p.get("timeout", 30),
                    cwd=p.get("cwd"),
                )
                return ActionResult(
                    success=(res.returncode == 0),
                    output=res.stdout.strip()[:2000],
                    error=res.stderr.strip()[:500] if res.returncode != 0 else "",
                )
            except subprocess.TimeoutExpired:
                return ActionResult(success=False, error="명령 타임아웃")
            except Exception as e:
                return ActionResult(success=False, error=str(e))

        # ── 파일 읽기/쓰기 ────────────────────────────────────────────────────
        if atype == "read_file":
            try:
                content = Path(p["path"]).read_text(encoding="utf-8")
                return ActionResult(success=True, output=content[:5000])
            except Exception as e:
                return ActionResult(success=False, error=str(e))

        if atype == "write_file":
            try:
                path = Path(p["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(p.get("content", ""), encoding="utf-8")
                return ActionResult(success=True, output=f"저장: {path}")
            except Exception as e:
                return ActionResult(success=False, error=str(e))

        # ── GUI 조작 ──────────────────────────────────────────────────────────
        if not DEPS["pyautogui"]:
            return ActionResult(success=True, output=f"[mock] {atype} {p}")

        try:
            pyautogui.FAILSAFE = True

            if atype == "click":
                pyautogui.click(p.get("x", 0), p.get("y", 0), button=p.get("button", "left"))
                return ActionResult(success=True, output=f"클릭 ({p.get('x')},{p.get('y')})")
            elif atype == "double_click":
                pyautogui.doubleClick(p.get("x", 0), p.get("y", 0))
                return ActionResult(success=True, output=f"더블클릭 ({p.get('x')},{p.get('y')})")
            elif atype == "type":
                pyautogui.typewrite(p.get("text", ""), interval=p.get("interval", 0.03))
                return ActionResult(success=True, output=f"타이핑: {p.get('text','')[:30]}")
            elif atype == "type_raw":
                subprocess.run(["xdotool", "type", "--clearmodifiers", p.get("text", "")],
                               check=True, timeout=5)
                return ActionResult(success=True, output=f"xdotool: {p.get('text','')[:30]}")
            elif atype == "key":
                keys = p.get("keys", [])
                if isinstance(keys, list):
                    pyautogui.hotkey(*keys)
                else:
                    pyautogui.press(keys)
                return ActionResult(success=True, output=f"키: {keys}")
            elif atype == "scroll":
                pyautogui.scroll(p.get("clicks", 3), x=p.get("x", 960), y=p.get("y", 540))
                return ActionResult(success=True, output=f"스크롤 {p.get('clicks', 3)}")
            elif atype == "move":
                pyautogui.moveTo(p.get("x", 0), p.get("y", 0), duration=p.get("duration", 0.3))
                return ActionResult(success=True, output=f"이동 ({p.get('x')},{p.get('y')})")
            elif atype == "screenshot":
                state = CorundumVision().capture()
                return ActionResult(success=True, output=state.b64[:100] + "...")
            else:
                return ActionResult(success=False, error=f"미지원: {atype}")
        except Exception as e:
            return ActionResult(success=False, error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  CorundumSemanticAnchor — UI 요소 의미 기반 탐색
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UIElement:
    role:      str  = ""
    name:      str  = ""
    x:         int  = -1
    y:         int  = -1
    width:     int  = 0
    height:    int  = 0
    enabled:   bool = True
    visible:   bool = True
    source:    str  = "unknown"

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def matches(self, role: str = "", text: str = "", partial: bool = True) -> bool:
        role_ok = (not role) or (role.lower() in self.role.lower())
        if not text:
            return role_ok
        t = text.lower()
        name_ok = (t in self.name.lower()) if partial else (t == self.name.lower())
        return role_ok and name_ok


class CorundumSemanticAnchor:
    _CACHE_TTL = 1.5

    def __init__(self):
        self._cache:    List[UIElement] = []
        self._cache_ts: float = 0.0

    async def get_tree(self, force: bool = False) -> List[UIElement]:
        now = time.time()
        if not force and (now - self._cache_ts) < self._CACHE_TTL:
            return self._cache
        elements: List[UIElement] = []
        if ATSPI_OK:
            elements = await asyncio.get_event_loop().run_in_executor(None, self._atspi_tree)
        elif UIAUTO_OK:
            elements = await asyncio.get_event_loop().run_in_executor(None, self._uiauto_tree)
        self._cache    = elements
        self._cache_ts = now
        return elements

    def _atspi_tree(self) -> List[UIElement]:
        elems: List[UIElement] = []
        try:
            def _walk(node, depth=0):
                if depth > 10:
                    return
                try:
                    role = node.get_role_name() or ""
                    name = node.get_name() or ""
                    comp = node.get_component()
                    ext  = comp.get_extents(Atspi.CoordType.SCREEN) if comp else None
                    e = UIElement(
                        role=role, name=name,
                        x=ext.x if ext else -1, y=ext.y if ext else -1,
                        width=ext.width if ext else 0, height=ext.height if ext else 0,
                        enabled=node.get_state_set().contains(Atspi.StateType.ENABLED),
                        visible=node.get_state_set().contains(Atspi.StateType.VISIBLE),
                        source="atspi",
                    )
                    if e.visible and (e.name or e.role):
                        elems.append(e)
                    for i in range(node.get_child_count()):
                        ch = node.get_child_at_index(i)
                        if ch:
                            _walk(ch, depth + 1)
                except Exception:
                    pass
            _walk(Atspi.get_desktop(0))
        except Exception as ex:
            log.debug("[Anchor/ATSPI] 실패: %s", ex)
        return elems

    def _uiauto_tree(self) -> List[UIElement]:
        elems: List[UIElement] = []
        try:
            def _walk(ctrl, depth=0):
                if depth > 8 or ctrl is None:
                    return
                try:
                    rect = ctrl.BoundingRectangle
                    e = UIElement(
                        role=ctrl.ControlTypeName or "",
                        name=ctrl.Name or "",
                        x=rect.left, y=rect.top,
                        width=rect.width(), height=rect.height(),
                        enabled=ctrl.IsEnabled,
                        visible=not ctrl.IsOffscreen,
                        source="uiauto",
                    )
                    if e.visible:
                        elems.append(e)
                    for ch in ctrl.GetChildren():
                        _walk(ch, depth + 1)
                except Exception:
                    pass
            _walk(_uiauto.GetRootControl())
        except Exception as ex:
            log.debug("[Anchor/UIAuto] 실패: %s", ex)
        return elems

    async def find_element(self, role: str = "", text: str = "",
                           screenshot_b64: str = "") -> Optional[UIElement]:
        """AT-SPI → UIAuto → OCR → VLM 순으로 UI 요소 탐색."""
        # 1. 접근성 트리
        tree = await self.get_tree()
        for elem in tree:
            if elem.matches(role=role, text=text):
                log.debug("[Anchor] 발견(tree/%s): %s", elem.source, elem.name[:30])
                return elem

        # 2. OCR 폴백
        if text:
            for elem in await self._ocr_elements(screenshot_b64):
                if elem.matches(role=role, text=text):
                    log.debug("[Anchor] 발견(OCR): %s", elem.name[:30])
                    return elem

        # 3. VLM 최후 수단
        if screenshot_b64 and (role or text):
            elem = await self._vlm_find(f"role={role} text={text}", screenshot_b64)
            if elem:
                log.debug("[Anchor] 발견(VLM): %s", elem.name[:30])
                return elem

        log.debug("[Anchor] 요소 없음: role=%r text=%r", role, text)
        return None

    async def _ocr_elements(self, screenshot_b64: str = "") -> List[UIElement]:
        if not (PYAUTOGUI_OK and PIL_OK):
            return []
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_ocr, screenshot_b64
        )

    def _sync_ocr(self, screenshot_b64: str) -> List[UIElement]:
        elems: List[UIElement] = []
        try:
            import pytesseract
            from PIL import Image
            import io as _io
            if screenshot_b64:
                img = Image.open(_io.BytesIO(base64.b64decode(screenshot_b64)))
            else:
                img = pyautogui.screenshot()
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                if not txt or data["conf"][i] < 40:
                    continue
                elems.append(UIElement(
                    role="text", name=txt,
                    x=data["left"][i], y=data["top"][i],
                    width=data["width"][i], height=data["height"][i],
                    source="ocr",
                ))
        except Exception as ex:
            log.debug("[Anchor/OCR] 실패: %s", ex)
        return elems

    async def _vlm_find(self, description: str, screenshot_b64: str) -> Optional[UIElement]:
        try:
            from ollama import AsyncClient
            Cfg = _get_cfg()
            prompt = (
                f"화면에서 다음 UI 요소를 찾아 좌표를 반환해:\n찾는 요소: {description}\n\n"
                '응답 형식 (JSON만):\n{"found":true,"x":정수,"y":정수,"w":정수,"h":정수,"role":"역할","name":"텍스트"}'
            )
            resp = await asyncio.wait_for(
                AsyncClient().chat(
                    model=Cfg.VISION_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "data": screenshot_b64},
                        {"type": "text",  "text": prompt},
                    ]}],
                    options={"temperature": 0.1, "num_predict": 128},
                ),
                timeout=Cfg.TIMEOUT_VISION,
            )
            data = _safe_parse_json(resp["message"]["content"])
            if data.get("found"):
                return UIElement(
                    role=data.get("role", "unknown"),
                    name=data.get("name", ""),
                    x=int(data["x"]), y=int(data["y"]),
                    width=int(data.get("w", 50)), height=int(data.get("h", 30)),
                    source="vlm",
                )
        except Exception as ex:
            log.debug("[Anchor/VLM] 실패: %s", ex)
        return None

    def invalidate(self):
        self._cache_ts = 0.0
        self._cache    = []


# ══════════════════════════════════════════════════════════════════════════════
#  CorundumPlanner — 목표 분해 + 재플래닝
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SubTask:
    idx:         int
    description: str
    done:        bool  = False
    failed:      bool  = False
    output:      str   = ""

    def to_str(self) -> str:
        mark = "✓" if self.done else ("✗" if self.failed else "○")
        return f"[{mark}] {self.idx+1}. {self.description[:60]}"


class CorundumPlanner:
    """
    목표를 서브태스크로 분해하고 실행 순서를 관리.
    실패 시 재플래닝 가능.
    """
    _PLAN_SYS = (
        "/no_think\n"
        "너는 코런덤의 플래너야.\n"
        "주어진 목표를 구체적이고 순차적인 서브태스크로 분해해.\n\n"
        "규칙:\n"
        "- 최대 8개 서브태스크\n"
        "- 각 태스크는 독립적으로 실행 가능한 단위\n"
        "- 터미널 우선, GUI는 필요할 때만\n"
        '응답(JSON만): {"subtasks": ["태스크1", "태스크2", ...]}'
    )
    _REPLAN_SYS = (
        "/no_think\n"
        "너는 코런덤의 플래너야.\n"
        "서브태스크가 실패했어. 나머지 계획을 수정해.\n\n"
        "규칙:\n"
        "- 실패한 태스크를 다른 방법으로 대체하거나 건너뛰어\n"
        "- 최대 5개\n"
        '응답(JSON만): {"subtasks": ["수정된 태스크1", ...]}'
    )
    _FORMAT = {
        "type": "object",
        "properties": {"subtasks": {"type": "array", "items": {"type": "string"}}},
        "required": ["subtasks"],
    }

    def __init__(self):
        self.plan:    List[SubTask] = []
        self._replan_count: int    = 0
        self._MAX_REPLAN: int      = 2

    async def make_plan(self, goal: str) -> List[SubTask]:
        """목표 → 서브태스크 리스트."""
        raw = await self._call(goal, self._PLAN_SYS)
        subtasks = self._parse(raw)
        if not subtasks:
            subtasks = [goal]  # 분해 실패 시 목표 자체를 단일 태스크로
        self.plan = [SubTask(idx=i, description=t) for i, t in enumerate(subtasks)]
        log.info("[Planner] 플랜 생성: %d 태스크", len(self.plan))
        return self.plan

    async def replan(self, failed_task: SubTask, remaining: List[SubTask]) -> List[SubTask]:
        """실패한 태스크 이후 계획 수정."""
        if self._replan_count >= self._MAX_REPLAN:
            log.info("[Planner] 재플래닝 한도 초과")
            return remaining
        self._replan_count += 1
        prompt = (
            f"실패한 태스크: {failed_task.description}\n"
            f"남은 태스크:\n" +
            "\n".join(f"- {t.description}" for t in remaining)
        )
        raw      = await self._call(prompt, self._REPLAN_SYS)
        new_descs = self._parse(raw)
        if not new_descs:
            return remaining
        base_idx = failed_task.idx + 1
        new_tasks = [SubTask(idx=base_idx + i, description=d) for i, d in enumerate(new_descs)]
        log.info("[Planner] 재플래닝: %d → %d 태스크", len(remaining), len(new_tasks))
        return new_tasks

    def plan_summary(self) -> str:
        if not self.plan:
            return "플랜 없음"
        return "\n".join(t.to_str() for t in self.plan)

    def _parse(self, raw: str) -> List[str]:
        try:
            import json
            data = json.loads(raw)
        except Exception:
            data = _safe_parse_json(raw)
        items = data.get("subtasks", [])
        return [str(s).strip() for s in items if str(s).strip()][:8]

    async def _call(self, prompt: str, sys_p: str) -> str:
        try:
            from ollama import AsyncClient
            Cfg = _get_cfg()
            resp = await asyncio.wait_for(
                AsyncClient().chat(
                    model=Cfg.JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user",   "content": prompt},
                    ],
                    format=self._FORMAT,
                    options={"temperature": 0.3, "num_predict": 512},
                ),
                timeout=Cfg.TIMEOUT_JUDGE,
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            log.debug("[Planner] LLM 실패: %s", e)
            return "{}"


# ══════════════════════════════════════════════════════════════════════════════
#  CorundumBrain — 화면 보고 행동 결정 루프
# ══════════════════════════════════════════════════════════════════════════════

class CorundumBrain:
    _ACT_SYS = (
        "/no_think\n"
        "너는 코런덤의 컴퓨터 조작 모듈이야.\n"
        "화면과 목표를 보고 다음 행동을 JSON으로 결정해.\n\n"
        "우선순위:\n"
        "1. run_command — 터미널이 가능하면 무조건 우선\n"
        "2. read_file / write_file — 파일 직접 접근\n"
        "3. click / type / key / scroll — GUI (터미널 불가 시만)\n"
        "4. done — 목표 달성\n"
        "5. wait — 확신 부족 또는 대기 필요\n\n"
        'OUTPUT: {"action_type":"run_command|read_file|write_file|click|double_click|type|type_raw|key|scroll|move|screenshot|wait|done",'
        '"params":{},"confidence":0.0~1.0,"reason":"이유","safe":true,'
        '"target_text":"click 대상 텍스트","target_role":"버튼/링크 등"}\n\n'
        "- run_command params: {\"cmd\":\"명령어\",\"timeout\":30,\"cwd\":\"/path\"}\n"
        "- read_file params: {\"path\":\"/경로\"}\n"
        "- write_file params: {\"path\":\"/경로\",\"content\":\"내용\"}\n"
        "- confidence < 0.65면 wait 권장\n"
        "- 화면에 오류가 보이면 즉시 수정 시도\n"
        "JSON만."
    )
    MAX_STEPS      = 25
    CONF_THRESHOLD = 0.65

    def __init__(self, safe_mode: bool = True):
        self.vision = CorundumVision()
        self.actor  = CorundumActor(safe_mode=safe_mode)
        self.anchor = CorundumSemanticAnchor()
        self._kg    = None   # CorundumComputer에서 주입

    def set_kg(self, kg):
        """LightKG 인스턴스 주입 — 메모리 연동용."""
        self._kg = kg

    async def execute_goal(
        self,
        goal: str,
        on_action: Optional[Callable] = None,
        confirm_fn=None,
        cogn_interval: float = 1.5,
    ) -> Dict:
        Cfg         = _get_cfg()
        steps_taken = []
        outputs     = []

        # KG에서 앱/목표 관련 힌트 조회
        kg_hint = ""
        if self._kg:
            keywords = [w for w in goal.split() if len(w) >= 3][:3]
            hints = []
            for kw in keywords:
                h = self._kg.query(kw, top_k=2)
                if h:
                    hints.append(h)
            if hints:
                kg_hint = "[과거 경험]\n" + "\n".join(hints[:4])
                log.debug("[Brain] KG 힌트 주입: %d개", len(hints))

        try:
            from ollama import AsyncClient
            OLLAMA_OK = True
        except ImportError:
            OLLAMA_OK = False

        for step_n in range(self.MAX_STEPS):
            state = self.vision.capture()
            if state.b64:
                try:
                    state = await self.vision.understand(state)
                except Exception:
                    pass

            screen_summary = (
                f"화면: {state.description} | "
                f"앱: {state.focused_app} | "
                f"텍스트: {state.ocr_text[:150]}"
            )

            recent = "\n".join(
                f"[{s['action']}] {'✓' if s['success'] else '✗'} {s.get('output','')[:80]}"
                for s in steps_taken[-3:]
            ) if steps_taken else "없음"

            prompt = (
                f"목표: {goal}\n"
                f"현재 상태: {screen_summary}\n"
                f"단계: {step_n + 1}/{self.MAX_STEPS}\n"
                f"이전 행동:\n{recent}"
                + (f"\n{kg_hint}" if kg_hint and step_n == 0 else "")
            )

            if not OLLAMA_OK:
                return {"success": True, "steps": [], "output": "[mock] 완료", "reason": "mock"}

            try:
                from ollama import AsyncClient
                resp = await asyncio.wait_for(
                    AsyncClient().chat(
                        model=Cfg.JUDGE_MODEL,
                        messages=[
                            {"role": "system", "content": self._ACT_SYS},
                            {"role": "user",   "content": prompt},
                        ],
                        options={"temperature": 0.2, "num_predict": 256},
                    ),
                    timeout=Cfg.TIMEOUT_JUDGE,
                )
                action_data = _safe_parse_json(resp["message"]["content"])
                if not action_data:
                    await asyncio.sleep(1.0)
                    continue
            except Exception as e:
                log.warning("[Brain] 행동 결정 실패: %s", e)
                break

            action = ComputerAction(
                action_type=action_data.get("action_type", "wait"),
                params=action_data.get("params", {}),
                confidence=float(action_data.get("confidence", 0.5)),
                reason=action_data.get("reason", ""),
                safe=action_data.get("safe", True),
            )

            if (action.action_type in ("click", "double_click")
                    and action_data.get("target_text")):
                elem = await self.anchor.find_element(
                    role=action_data.get("target_role", ""),
                    text=action_data["target_text"],
                    screenshot_b64=state.b64,
                )
                if elem and elem.x >= 0:
                    action.params["x"] = elem.center[0]
                    action.params["y"] = elem.center[1]
                    log.info("[Brain] Anchor 보정: '%s' → (%d,%d)",
                             elem.name[:20], elem.center[0], elem.center[1])

            if action.action_type == "done":
                log.info("[Brain] 목표 달성: %s", goal[:50])
                return {
                    "success": True,
                    "steps":   steps_taken,
                    "output":  "\n".join(outputs[-5:]),
                    "reason":  action.reason,
                }

            if action.action_type == "wait" or action.confidence < self.CONF_THRESHOLD:
                wait_sec = min(max(cogn_interval, 1.0), 5.0)
                await asyncio.sleep(wait_sec)
                continue

            result = await self.actor.execute(action, confirm_fn=confirm_fn)
            step_rec = {
                "step":    step_n,
                "action":  action.action_type,
                "reason":  action.reason[:60],
                "success": result.success,
                "output":  str(result.output or "")[:200],
            }
            steps_taken.append(step_rec)
            if result.output:
                outputs.append(str(result.output)[:300])

            if on_action:
                on_action(action, result)

            log.info("[Brain] 스텝%d: %s (%s) → %s",
                     step_n + 1, action.action_type, action.reason[:30],
                     "✓" if result.success else f"✗ {result.error[:40]}")

            if not result.success:
                if sum(1 for s in steps_taken[-3:] if not s["success"]) >= 3:
                    # 전략 전환 시도
                    strategy = self._recover_strategy(steps_taken, goal)
                    if strategy:
                        log.info("[Brain] 실패 복구 전략 전환: %s", strategy)
                        if on_action:
                            on_action(None, ActionResult(success=False,
                                      error=f"전략 전환: {strategy}"))
                        # 전환 힌트를 goal에 주입해서 다음 스텝에서 반영
                        goal = f"{goal}\n[복구 힌트] {strategy}"
                        # 연속 실패 카운터 리셋을 위해 더미 성공 레코드 삽입
                        steps_taken.append({
                            "step": step_n, "action": "recover",
                            "reason": strategy, "success": True,
                            "output": "전략 전환",
                        })
                    else:
                        return {"success": False, "steps": steps_taken,
                                "output": "\n".join(outputs[-3:]), "reason": "연속 실패 — 복구 전략 없음"}

            await asyncio.sleep(0.4)

        return {"success": False, "steps": steps_taken,
                "output": "\n".join(outputs[-3:]), "reason": "최대 단계 도달"}

    def _recover_strategy(self, steps: List[Dict], goal: str) -> Optional[str]:
        """
        연속 실패 시 전략 전환 힌트 반환.
        최근 3개 실패 액션 패턴 분석 → 다른 접근 제안.
        두 번 이상 같은 전략 전환을 시도하면 None 반환 (무한루프 방지).
        """
        recent = steps[-6:]
        failed_actions = [s["action"] for s in recent if not s.get("success", True)]
        recover_count  = sum(1 for s in recent if s.get("action") == "recover")

        if recover_count >= 2:
            return None  # 이미 두 번 전환했으면 포기

        # GUI 액션 연속 실패 → 터미널로 전환
        gui_actions = {"click", "double_click", "type", "type_raw", "key", "scroll", "move"}
        if all(a in gui_actions for a in failed_actions[-3:]):
            return "GUI 조작 실패 — run_command로 터미널에서 직접 처리 시도"

        # 터미널 명령 연속 실패 → GUI로 전환
        if all(a == "run_command" for a in failed_actions[-3:]):
            return "터미널 명령 실패 — GUI 직접 조작으로 전환 시도"

        # 파일 작업 실패 → 경로/권한 확인
        if all(a in ("read_file", "write_file") for a in failed_actions[-3:]):
            return "파일 접근 실패 — 경로 확인 및 권한 체크 필요 (ls -la, pwd 등)"

        # 혼합 실패 → 스크린샷 찍고 현재 상태 재파악
        return "현재 상태 불명확 — screenshot으로 화면 재파악 후 재시도"


# ══════════════════════════════════════════════════════════════════════════════
#  CorundumComputer — Brain 래퍼
# ══════════════════════════════════════════════════════════════════════════════

class CorundumComputer:
    def __init__(self, safe_mode: bool = True):
        self._brain   = CorundumBrain(safe_mode=safe_mode)
        self._planner = CorundumPlanner()
        self._run_log: deque = deque(maxlen=30)
        self._kg      = None  # CorundumAgency에서 주입

    def set_anchor(self, anchor: CorundumSemanticAnchor):
        self._brain.anchor = anchor

    def set_kg(self, kg):
        """LightKG 주입 — 성공/실패 패턴 저장 + 힌트 조회."""
        self._kg = kg
        self._brain.set_kg(kg)

    def _save_pattern(self, goal: str, steps: List[Dict], success: bool):
        """성공/실패 패턴을 KG에 저장."""
        if not self._kg or not steps:
            return
        try:
            # 가장 많이 쓴 액션 타입 추출
            action_counts: Dict[str, int] = {}
            for s in steps:
                a = s.get("action", "")
                action_counts[a] = action_counts.get(a, 0) + 1
            if not action_counts:
                return
            dominant = max(action_counts, key=lambda k: action_counts[k])
            goal_kw  = goal[:20].strip()
            rel      = "성공패턴" if success else "실패패턴"
            conf     = 0.7 if success else 0.3
            self._kg.upsert(goal_kw, rel, dominant, confidence=conf)
            log.debug("[Computer] KG 저장: %s --%s--> %s", goal_kw, rel, dominant)
        except Exception as e:
            log.debug("[Computer] KG 저장 실패: %s", e)

    async def execute(
        self,
        goal: str,
        on_status: Optional[Callable] = None,
        confirm_fn=None,
        cogn_interval: float = 1.5,
        use_planner: bool = True,
    ) -> Dict:
        log.info("[Computer] 목표: %s", goal[:60])

        def _on_action(action, result):
            if action is None:
                return
            msg = f"[{action.action_type}] {'✓' if result.success else '✗'} {action.reason[:40]}"
            if result.output:
                msg += f"\n  → {str(result.output)[:100]}"
            if on_status:
                on_status(msg)

        # 단순 목표는 플래너 없이 직접 실행
        if not use_planner or len(goal) < 30:
            result = await self._brain.execute_goal(
                goal, on_action=_on_action,
                confirm_fn=confirm_fn, cogn_interval=cogn_interval,
            )
            self._save_pattern(goal, result.get("steps", []), result.get("success", False))
            self._run_log.append({"ts": time.time(), "goal": goal[:60], "success": result.get("success", False)})
            return result

        # 복잡한 목표 → 플래너로 분해 후 서브태스크 순차 실행
        subtasks = await self._planner.make_plan(goal)
        if on_status:
            on_status(f"[Planner] {len(subtasks)}개 태스크로 분해\n{self._planner.plan_summary()}")

        all_steps:   List[Dict] = []
        all_outputs: List[str]  = []
        remaining = list(subtasks)

        while remaining:
            task = remaining.pop(0)
            if on_status:
                on_status(f"[Planner] 태스크 {task.idx+1}: {task.description[:50]}")

            result = await self._brain.execute_goal(
                task.description,
                on_action=_on_action,
                confirm_fn=confirm_fn,
                cogn_interval=cogn_interval,
            )
            all_steps.extend(result.get("steps", []))
            if result.get("output"):
                all_outputs.append(result["output"])

            if result.get("success"):
                task.done   = True
                task.output = result.get("output", "")
            else:
                task.failed = True
                log.info("[Planner] 태스크 실패: %s — 재플래닝 시도", task.description[:40])
                remaining = await self._planner.replan(task, remaining)
                if not remaining:
                    break

        success = all(t.done for t in subtasks if not t.failed)
        final   = {
            "success": success,
            "steps":   all_steps,
            "output":  "\n".join(all_outputs[-5:]),
            "reason":  "플래너 완료" if success else "일부 태스크 실패",
            "plan":    self._planner.plan_summary(),
        }
        self._save_pattern(goal, all_steps, success)
        self._run_log.append({"ts": time.time(), "goal": goal[:60], "success": success})
        return final

    def recent_log(self, n: int = 5) -> List[Dict]:
        return list(self._run_log)[-n:]


# ══════════════════════════════════════════════════════════════════════════════
#  CorundumWeb — 검색 + 요약
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WebDoc:
    url:      str
    title:    str  = ""
    raw_text: str  = ""
    summary:  str  = ""
    facts:    list = field(default_factory=list)
    error:    str  = ""


class CorundumWeb:
    HEADERS = {
        "User-Agent": "Corundum/1.0 (code-defense agent)",
        "Accept-Language": "ko,en;q=0.9",
    }
    TIMEOUT   = 12.0
    MAX_BYTES = 400_000
    _SKIP     = {"script", "style", "nav", "footer", "header", "aside", "iframe"}
    RATE_LIMIT = 12
    TECH_DOMAINS = {
        "stackoverflow.com", "github.com", "docs.python.org",
        "developer.mozilla.org", "pypi.org", "npmjs.com",
    }

    _SUM_SYS = (
        "/no_think\n"
        "기술 문서/코드 관련 페이지를 분석해 JSON으로.\n"
        '{"summary":"핵심 2~3문장","facts":["기술 사실1","사실2"]}\n'
        "JSON만."
    )

    def __init__(self):
        self._req_times: list = []
        self._lock = asyncio.Lock()

    async def _acquire(self) -> bool:
        async with self._lock:
            now = time.time()
            self._req_times = [t for t in self._req_times if now - t < 60]
            if len(self._req_times) >= self.RATE_LIMIT:
                return False
            self._req_times.append(now)
            return True

    def _do_fetch(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers=self.HEADERS)
        with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
            return resp.read()

    def _extract(self, html: str):
        if BS4_OK:
            soup  = BeautifulSoup(html, "html.parser")
            title = soup.find("title")
            title = title.get_text(strip=True) if title else ""
            for tag in soup(self._SKIP):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            m     = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
            title = m.group(1).strip() if m else ""
            text  = re.sub(r"<[^>]+>", " ", html)
        lines  = [l.strip() for l in text.splitlines() if len(l.strip()) > 15]
        seen, unique = set(), []
        for l in lines:
            if l not in seen:
                seen.add(l)
                unique.append(l)
        return title, "\n".join(unique[:400])

    async def fetch(self, url: str) -> WebDoc:
        doc = WebDoc(url=url)
        for _ in range(3):
            if await self._acquire():
                break
            await asyncio.sleep(5)
        else:
            doc.error = "레이트 리밋"
            return doc
        try:
            loop = asyncio.get_event_loop()
            raw  = await asyncio.wait_for(
                loop.run_in_executor(None, self._do_fetch, url),
                timeout=self.TIMEOUT,
            )
            html = raw[:self.MAX_BYTES].decode("utf-8", errors="replace")
            doc.title, doc.raw_text = self._extract(html)
        except asyncio.TimeoutError:
            doc.error = "타임아웃"
        except Exception as e:
            doc.error = str(e)[:80]

        if doc.raw_text and not doc.error:
            try:
                from ollama import AsyncClient
                Cfg  = _get_cfg()
                resp = await asyncio.wait_for(
                    AsyncClient().chat(
                        model=Cfg.JUDGE_MODEL,
                        messages=[
                            {"role": "system", "content": self._SUM_SYS},
                            {"role": "user", "content": f"URL: {url}\n{doc.raw_text[:2000]}"},
                        ],
                        options={"temperature": 0.15, "num_predict": 300},
                    ),
                    timeout=20.0,
                )
                data = _safe_parse_json(resp["message"]["content"])
                if data:
                    doc.summary = data.get("summary", "")
                    doc.facts   = data.get("facts", [])
            except Exception:
                pass
        return doc

    async def search(self, query: str, max_results: int = 3) -> List[WebDoc]:
        q   = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={q}"
        try:
            loop = asyncio.get_event_loop()
            raw  = await asyncio.wait_for(
                loop.run_in_executor(None, self._do_fetch, url),
                timeout=12,
            )
            html = raw.decode("utf-8", errors="replace")
            urls = re.findall(r'href="(https?://[^"&]+)"', html)
            tech = [u for u in urls if any(d in u for d in self.TECH_DOMAINS)]
            rest = [u for u in urls if u not in tech and "duckduckgo" not in u]
            seen, final = set(), []
            for u in tech + rest:
                domain = urllib.parse.urlparse(u).netloc
                if domain not in seen:
                    seen.add(domain)
                    final.append(u)
                if len(final) >= max_results:
                    break
            docs = []
            for u in final:
                docs.append(await self.fetch(u))
                await asyncio.sleep(0.5)
            return docs
        except Exception as e:
            log.error("[Web] 검색 실패: %s", e)
            return []

    def format_results(self, docs: List[WebDoc]) -> str:
        if not docs:
            return ""
        lines = ["[웹 검색 결과]"]
        for i, doc in enumerate(docs, 1):
            if doc.error:
                continue
            lines.append(f"  {i}. {doc.title or doc.url[:50]}")
            if doc.summary:
                lines.append(f"     {doc.summary[:150]}")
            for f in doc.facts[:2]:
                lines.append(f"     • {f[:80]}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CorundumTask — 임무 상태 머신
# ══════════════════════════════════════════════════════════════════════════════

class TaskState(Enum):
    IDLE    = "idle"
    RUNNING = "running"
    DONE    = "done"
    ABORTED = "aborted"


@dataclass
class Task:
    description:  str
    state:        TaskState = TaskState.IDLE
    steps:        list      = field(default_factory=list)
    started_at:   float     = field(default_factory=time.time)
    finished_at:  float     = 0.0
    final_output: str       = ""
    error:        str       = ""

    def duration_str(self) -> str:
        sec = int((self.finished_at or time.time()) - self.started_at)
        return f"{sec // 60}분 {sec % 60}초" if sec >= 60 else f"{sec}초"

    def summary(self) -> str:
        lines = [
            f"[임무] {self.description[:60]}",
            f"상태: {self.state.value} | 소요: {self.duration_str()} | 스텝: {len(self.steps)}",
        ]
        if self.final_output:
            lines.append(f"결과:\n{self.final_output[:400]}")
        if self.error:
            lines.append(f"오류: {self.error[:100]}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CorundumAgency — 메인 통합
# ══════════════════════════════════════════════════════════════════════════════

class CorundumAgency:
    DIVE_MSGS = [
        "임무 중이에요. 끝나면 말할게요. (/abort 로 중단)",
        "잠수 중. /abort 로 중단 가능해요.",
        "집중 중. 방해하지 마요.",
    ]
    _dive_idx = 0

    def __init__(self, safe_mode: bool = True):
        self.computer = CorundumComputer(safe_mode=safe_mode)
        self.web      = CorundumWeb()

        self._corundum    = None
        self._task:       Optional[Task]         = None
        self._bg_task:    Optional[asyncio.Task] = None
        self._abort_evt:  asyncio.Event          = asyncio.Event()
        self._status_log: deque                  = deque(maxlen=50)

        log.info("[Agency] 초기화 완료")

    def attach(self, corundum_instance):
        self._corundum = corundum_instance
        log.info("[Agency] Corundum 연결됨")

    # ── 상태 ──────────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._task is not None and self._task.state == TaskState.RUNNING

    def dive_response(self) -> str:
        msg = self.DIVE_MSGS[self._dive_idx % len(self.DIVE_MSGS)]
        self._dive_idx += 1
        return msg

    # ── 훅 ───────────────────────────────────────────────────────────────────

    async def on_impulse(self, impulse_text: str, ctx: Dict) -> Optional[Dict]:
        """process()에서 호출. 임무 키워드 감지 시 do_task 시작."""
        if self.is_running:
            return None
        task_goal = self._detect_task(impulse_text)
        if task_goal:
            await self.do_task(task_goal)
            return {"task_started": True, "goal": task_goal}
        return None

    async def tick(self, ctx: Dict) -> None:
        """physio_loop에서 주기 호출. 상태 로깅."""
        if self._task and self._task.state == TaskState.RUNNING:
            log.debug("[Agency.tick] 임무 진행 중: %s", self._task.description[:40])

    # ── 임무 실행 ─────────────────────────────────────────────────────────────

    async def do_task(
        self,
        description: str,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Task:
        if self.is_running:
            return self._task

        self._abort_evt.clear()
        task = Task(description=description, state=TaskState.RUNNING)
        self._task = task

        _on_status = on_status or self._make_status_fn()

        self._bg_task = asyncio.create_task(
            self._run_task(task, _on_status)
        )
        log.info("[Agency] 임무 시작: %s", description[:60])
        return task

    def abort(self) -> str:
        if not self.is_running:
            return "실행 중인 임무 없음"
        self._abort_evt.set()
        if self._bg_task:
            self._bg_task.cancel()
        if self._task:
            self._task.state       = TaskState.ABORTED
            self._task.finished_at = time.time()
      
        if self._corundum:
            self._corundum._dormant = False
        return "임무 중단됐어요."

    async def do_computer(self, goal: str, cogn_interval: float = 1.5) -> Dict:
        """직접 컴퓨터 제어 (/computer 커맨드)."""
        return await self.computer.execute(
            goal,
            on_status=self._make_status_fn(),
            cogn_interval=cogn_interval,
        )

    # ── 임무 실행 루프 ────────────────────────────────────────────────────────

    async def _run_task(self, task: Task, on_status: Callable):
        try:
            on_status(f"[임무 시작] {task.description[:60]}")

            needs_web = any(
                kw in task.description
                for kw in ["검색", "찾아", "조사", "문서", "어떻게", "방법", "search"]
            )

            web_context = ""
            if needs_web:
                on_status("[웹 검색 중...]")
                docs = await self.web.search(task.description, max_results=2)
                web_context = self.web.format_results(docs)
                if web_context:
                    on_status(f"[검색 완료] {len(docs)}개")

            if self._abort_evt.is_set():
                task.state       = TaskState.ABORTED
                task.finished_at = time.time()
                self._notify_done(task)
                return

            goal_with_context = task.description
            if web_context:
                goal_with_context = f"{task.description}\n\n참고:\n{web_context[:500]}"

            result = await self.computer.execute(
                goal=goal_with_context,
                on_status=on_status,
                cogn_interval=getattr(_get_cfg(), "AUTO_INTERVAL", 60.0) / 10,
            )

            task.steps        = result.get("steps", [])
            task.final_output = result.get("output", "")
            task.state        = TaskState.DONE if result.get("success") else TaskState.ABORTED
            task.error        = result.get("reason", "") if not result.get("success") else ""
            task.finished_at  = time.time()

            on_status(f"[임무 {'완료' if task.state == TaskState.DONE else '실패'}] {task.duration_str()} 소요")
            self._notify_done(task)

        except asyncio.CancelledError:
            task.state       = TaskState.ABORTED
            task.finished_at = time.time()
        except Exception as e:
            task.state       = TaskState.ABORTED
            task.error       = str(e)
            task.finished_at = time.time()
            log.error("[Agency] 임무 크래시: %s", e)
            self._notify_done(task)

    def _notify_done(self, task: Task):
        if not self._corundum:
            return
        on_auto = getattr(self._corundum, "on_autonomous", None)
        if on_auto:
            on_auto(task.summary())
        self._corundum._dormant = False

    # ── 유틸 ──────────────────────────────────────────────────────────────────

    def _make_status_fn(self) -> Callable:
        def _cb(msg: str):
            self._status_log.append({"ts": time.time(), "msg": msg})
            log.info("[Agency] %s", msg)
            if self._corundum:
                on_auto = getattr(self._corundum, "on_autonomous", None)
                if on_auto:
                    on_auto(msg)
        return _cb

    def _detect_task(self, text: str) -> Optional[str]:
        """임무 키워드 감지 (오발 방지: action + target 둘 다 있어야)."""
        t = text.lower()
        action_kws = ["해줘", "해봐", "실행해", "만들어", "수정해", "고쳐", "찾아줘",
                      "설치해", "테스트해", "분석해", "빌드해", "배포해"]
        target_kws = ["파일", "코드", "프로젝트", "버그", "오류", "스크립트",
                      "터미널", "폴더", "모듈", "패키지", "서버"]
        if any(k in t for k in action_kws) and any(k in t for k in target_kws):
            return text[:200]
        return None

    def status_str(self) -> str:
        lines = ["[Agency status]"]
        if self._task:
            lines.append(f"  임무: {self._task.description[:50]} ({self._task.state.value})")
        else:
            lines.append("  임무: 없음")
        comp_log = self.computer.recent_log(3)
        if comp_log:
            last = comp_log[-1]
            ts   = time.strftime("%H:%M", time.localtime(last["ts"]))
            lines.append(f"  last computer: [{ts}] {last['goal']} ({'ok' if last['success'] else 'fail'})")
        return "\n".join(lines)
