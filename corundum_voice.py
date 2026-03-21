#!/usr/bin/env python3
# corundum_voice.py
# CORUNDUM voice listener — whisper 기반 호출어 감지 + 음성 입력
#
# 대기 중: 마이크 청취만. LLM 호출 없음. CPU 최소화.
# 호출어 감지 시: on_wake 콜백 → Corundum 깨움.
# 임무 중: 음성 입력을 텍스트로 변환해서 process()로 전달.
#
# 핵심 API:
#   listener = VoiceListener(wake_name="코런덤", on_wake=cb, on_command=cb)
#   await listener.start()   # 백그라운드 루프 시작
#   listener.stop()

import asyncio
import logging
import re
import time
import unicodedata
from typing import Callable, Optional

log = logging.getLogger("corundum.voice")

# ── 선택적 의존성 ──────────────────────────────────────────────────────────────

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

try:
    import sounddevice as sd
    SD_OK = True
except ImportError:
    SD_OK = False
    log.warning("[Voice] sounddevice 없음 — 음성 비활성화")

try:
    import whisper as _whisper_mod
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False
    log.warning("[Voice] openai-whisper 없음 — 음성 비활성화")


def _normalize(text: str) -> str:
    """유니코드 정규화 + 소문자 + 공백 정리."""
    return unicodedata.normalize("NFC", text).lower().strip()


def _name_variants(name: str):
    """
    '코런덤' → ['코런덤', '코런덤아', '코런덤야', '야 코런덤', 'corundum', ...]
    영문 이름이면 한글 변형 스킵.
    """
    n = _normalize(name)
    variants = {n, n + "아", n + "야", n + "!", "야 " + n, "야, " + n}
    # 영문 변형
    en = name.lower()
    if en != n:
        variants.update({en, en + "!", "hey " + en})
    return variants


# ── Whisper 모델 관리 (지연 로드) ─────────────────────────────────────────────

_WHISPER_MODEL = None
_WHISPER_SIZE  = "small"  # "tiny" 가장 빠름, "small" 균형

def _get_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None and WHISPER_OK:
        log.info("[Voice] whisper 모델 로드: %s", _WHISPER_SIZE)
        _WHISPER_MODEL = _whisper_mod.load_model(_WHISPER_SIZE)
    return _WHISPER_MODEL


# ── 오디오 청취 ───────────────────────────────────────────────────────────────

class AudioChunk:
    SAMPLE_RATE   = 16000
    CHUNK_SEC     = 3.0      # 한 번에 처리할 오디오 길이 (초)
    SILENCE_THRESH = 0.01    # 이 이하면 무음으로 판단
    SILENCE_RATIO  = 0.85    # 청크의 이 비율 이상이 무음이면 스킵

    @classmethod
    def record(cls) -> Optional["np.ndarray"]:
        if not (SD_OK and NUMPY_OK):
            return None
        try:
            frames = sd.rec(
                int(cls.SAMPLE_RATE * cls.CHUNK_SEC),
                samplerate=cls.SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = frames.flatten()
            # 무음 스킵
            silence = np.sum(np.abs(audio) < cls.SILENCE_THRESH) / len(audio)
            if silence > cls.SILENCE_RATIO:
                return None
            return audio
        except Exception as e:
            log.debug("[Voice] 오디오 녹음 실패: %s", e)
            return None


# ── Whisper 전사 ──────────────────────────────────────────────────────────────

def _transcribe(audio: "np.ndarray") -> str:
    model = _get_whisper()
    if model is None:
        return ""
    try:
        result = model.transcribe(audio, language="ko", fp16=False)
        return result.get("text", "").strip()
    except Exception as e:
        log.debug("[Voice] 전사 실패: %s", e)
        return ""


# ── VoiceListener ─────────────────────────────────────────────────────────────

class VoiceListener:
    """
    상시 대기 음성 감지기.

    states:
      DORMANT  — 호출어만 감지. Whisper 호출 최소화 (무음 스킵).
      AWAKE    — 모든 음성을 텍스트로 변환해 on_command 콜백.

    전환:
      DORMANT → AWAKE : 호출어 감지 → on_wake() 호출
      AWAKE → DORMANT : idle_timeout 초 동안 입력 없음
    """

    IDLE_TIMEOUT = 60.0   # 깨어난 후 이 시간 동안 입력 없으면 다시 대기

    def __init__(
        self,
        wake_name:  str,
        on_wake:    Optional[Callable] = None,
        on_command: Optional[Callable[[str], None]] = None,
        model_size: str = "small",
    ):
        global _WHISPER_SIZE
        _WHISPER_SIZE = model_size

        self.wake_name   = wake_name
        self._variants   = _name_variants(wake_name)
        self.on_wake     = on_wake
        self.on_command  = on_command

        self._state      = "DORMANT"   # "DORMANT" | "AWAKE"
        self._last_cmd_t = 0.0
        self._running    = False
        self._task: Optional[asyncio.Task] = None

        log.info("[Voice] 초기화: 호출어=%r 변형=%s", wake_name, self._variants)

    # ── 공개 API ──────────────────────────────────────────────────────────────

    async def start(self):
        if not (SD_OK and WHISPER_OK and NUMPY_OK):
            log.warning("[Voice] 의존성 없음 — 음성 비활성화 (sounddevice/whisper/numpy 필요)")
            return
        self._running = True
        self._task    = asyncio.create_task(self._loop())
        log.info("[Voice] 대기 시작 (state=DORMANT)")

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        log.info("[Voice] 정지")

    def force_awake(self):
        """키보드 입력 등 외부 트리거로 강제 AWAKE."""
        self._state      = "AWAKE"
        self._last_cmd_t = time.time()

    def force_dormant(self):
        """임무 완수 후 다시 DORMANT."""
        self._state = "DORMANT"
        log.info("[Voice] → DORMANT")

    @property
    def is_awake(self) -> bool:
        return self._state == "AWAKE"

    # ── 내부 루프 ─────────────────────────────────────────────────────────────

    async def _loop(self):
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                # AWAKE idle timeout 체크
                if self._state == "AWAKE":
                    if time.time() - self._last_cmd_t > self.IDLE_TIMEOUT:
                        log.info("[Voice] idle timeout → DORMANT")
                        self._state = "DORMANT"

                # 오디오 녹음 (blocking → executor)
                audio = await loop.run_in_executor(None, AudioChunk.record)
                if audio is None:
                    # 무음 → 짧게 쉬고 다시
                    await asyncio.sleep(0.1)
                    continue

                # 전사 (blocking → executor)
                text = await loop.run_in_executor(None, _transcribe, audio)
                if not text:
                    await asyncio.sleep(0.05)
                    continue

                log.debug("[Voice] 전사: %r (state=%s)", text[:60], self._state)

                if self._state == "DORMANT":
                    if self._detect_wake(text):
                        self._state      = "AWAKE"
                        self._last_cmd_t = time.time()
                        log.info("[Voice] 호출어 감지: %r → AWAKE", text[:40])
                        # 호출어 이후 텍스트가 있으면 명령으로 처리
                        remainder = self._strip_wake(text)
                        if self.on_wake:
                            if asyncio.iscoroutinefunction(self.on_wake):
                                await self.on_wake(remainder)
                            else:
                                self.on_wake(remainder)
                        elif remainder and self.on_command:
                            await self._dispatch(remainder)

                elif self._state == "AWAKE":
                    self._last_cmd_t = time.time()
                    await self._dispatch(text)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("[Voice] 루프 오류: %s", e)
                await asyncio.sleep(1.0)

    async def _dispatch(self, text: str):
        if not text or not self.on_command:
            return
        if asyncio.iscoroutinefunction(self.on_command):
            await self.on_command(text)
        else:
            self.on_command(text)

    def _detect_wake(self, text: str) -> bool:
        t = _normalize(text)
        for v in self._variants:
            if v in t:
                return True
        # 편집거리 기반 퍼지 매칭 (짧은 이름 대비)
        name = _normalize(self.wake_name)
        if len(name) >= 3:
            for word in t.split():
                if _edit_distance(word, name) <= 1:
                    return True
        return False

    def _strip_wake(self, text: str) -> str:
        """호출어 제거 후 나머지 반환."""
        t = _normalize(text)
        for v in sorted(self._variants, key=len, reverse=True):
            t = t.replace(v, "").strip(" ,!.")
        return t.strip()


def _edit_distance(a: str, b: str) -> int:
    """레벤슈타인 거리 (짧은 문자열 전용, O(n*m))."""
    if abs(len(a) - len(b)) > 2:
        return 99
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        ndp = [i + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (ca != cb), dp[j + 1] + 1, ndp[-1] + 1))
        dp = ndp
    return dp[-1]


# ── mock (의존성 없을 때) ─────────────────────────────────────────────────────

class MockVoiceListener:
    """sounddevice/whisper 없을 때 사용. 키보드 입력만."""

    def __init__(self, wake_name="코런덤", on_wake=None, on_command=None, **_):
        self.wake_name  = wake_name
        self.on_wake    = on_wake
        self.on_command = on_command
        self.is_awake   = False
        log.info("[Voice] MockVoiceListener — 음성 비활성화, 키보드만")

    async def start(self): pass
    def stop(self): pass
    def force_awake(self): self.is_awake = True
    def force_dormant(self): self.is_awake = False


def make_listener(**kwargs) -> "VoiceListener | MockVoiceListener":
    if SD_OK and WHISPER_OK and NUMPY_OK:
        return VoiceListener(**kwargs)
    return MockVoiceListener(**kwargs)
