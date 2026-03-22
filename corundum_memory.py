#!/usr/bin/env python3
# corundum_memory.py
# CORUNDUM memory subsystem — DB/KG based, no LLM
#
# Components:
#   EpisodeMemory  : short-term + long-term episode buffer
#   SnippetMemory  : frequently referenced code patterns
#   LightKG        : entity-relation knowledge graph (Triple-based)
#   VectorIndex    : similarity search (sentence-transformers; keyword fallback)
#   MemoryDB       : async SQLite persistence
#   CorundumMemory : facade

import asyncio
import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import aiosqlite
    AIOSQLITE_OK = True
except ImportError:
    import sqlite3
    AIOSQLITE_OK = False

log = logging.getLogger("corundum")

try:
    from corundum_config import Cfg
except ImportError:
    class Cfg:
        MEMORY_DB_PATH   = "corundum_memory.db"
        MEMORY_TOP_K     = 4
        MEMORY_EMBED_DIM = 384

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    EMBED_OK = True
    log.info("memory: sentence-transformers loaded")
except ImportError:
    EMBED_OK = False
    log.warning("memory: sentence-transformers not found, using keyword fallback")


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ── MMR 유틸 ──────────────────────────────────────────────────────────────────

_MMR_LAMBDA   = 0.65   # 1.0 = 순수 유사도, 0.0 = 순수 다양성
_TIME_DECAY_K = 1e-6   # exp(-k * age_sec) — 약 11일에 절반


def _time_decay(ts: float) -> float:
    age = max(0.0, time.time() - ts)
    return math.exp(-_TIME_DECAY_K * age)


def _cosine_np(a, b) -> float:
    import numpy as np
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mmr_select(
    indices: List[int],
    scores: List[float],
    vectors: List,
    ts_list: List[float],
    k: int,
    lam: float = _MMR_LAMBDA,
) -> List[int]:
    """MMR 재순위 — 관련성과 다양성 균형. 시간 감쇠를 초기 점수에 적용."""
    if not indices:
        return []
    try:
        import numpy as np
        decayed = [s * _time_decay(ts_list[i]) for s, i in zip(scores, indices)]
        selected: List[int] = []
        remaining = list(range(len(indices)))
        for _ in range(min(k, len(indices))):
            if not remaining:
                break
            best_pos, best_score = -1, -float("inf")
            for pos in remaining:
                rel = decayed[pos]
                if not selected:
                    mmr_s = rel
                else:
                    max_sim = max(
                        _cosine_np(vectors[indices[pos]], vectors[indices[sel]])
                        for sel in selected
                    )
                    mmr_s = lam * rel - (1.0 - lam) * max_sim
                if mmr_s > best_score:
                    best_score, best_pos = mmr_s, pos
            selected.append(best_pos)
            remaining.remove(best_pos)
        return [indices[i] for i in selected]
    except Exception as e:
        log.debug("mmr_select fallback: %s", e)
        return [i for _, i in sorted(zip(scores, indices), reverse=True)[:k]]


# ── gear → layer 매핑 ─────────────────────────────────────────────────────────

def _gear_to_layer(gear: str) -> int:
    """SURFACE=0 / VORTEX=1 / DEEP=2"""
    return {
        "OVERDRIVE": 2, "FOCUS": 2,
        "THINK": 1, "NORMAL": 1,
        "SAVE": 0, "LOW": 0, "SLEEP": 0, "DREAM": 0,
    }.get(gear, 1)


# ── episode memory ────────────────────────────────────────────────────────────

@dataclass
class Episode:
    text:       str
    source:     str
    importance: float = 0.5
    ts:         float = field(default_factory=time.time)
    tags:       List[str] = field(default_factory=list)


class EpisodeMemory:
    SHORT_MAXLEN   = 20
    LONG_MAXLEN    = 200
    IMPORTANCE_THR = 0.6

    def __init__(self):
        self.short_term: deque = deque(maxlen=self.SHORT_MAXLEN)
        self.long_term:  deque = deque(maxlen=self.LONG_MAXLEN)

    def record(self, text: str, source: str, importance: float = 0.5, tags: List[str] = None):
        ep = Episode(text=text, source=source, importance=importance, tags=tags or [])
        self.short_term.append(ep)
        if importance >= self.IMPORTANCE_THR:
            self.long_term.append(ep)

    def working_context(self, n: int = 6) -> str:
        recent = list(self.short_term)[-n:]
        if not recent:
            return ""
        lines = []
        for ep in recent:
            role = "나" if ep.source == "user" else "코런덤"
            lines.append(f"[{role}] {ep.text[:80]}")
        return "\n".join(lines)

    def keyword_recall(self, query: str, top_k: int = 4) -> List[Episode]:
        words  = set(query.lower().split())
        scored = []
        for ep in self.long_term:
            overlap = sum(1 for w in words if w in ep.text.lower())
            if overlap > 0:
                scored.append((overlap, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def sleep_prune(self, min_nodes: int = 30) -> int:
        """
        SLEEP/DREAM gear 때 호출 — 중요도 하위 항목 정리.
        min_nodes 미만이면 손대지 않음.
        Returns: 삭제된 수
        """
        nodes = list(self.long_term)
        if len(nodes) < min_nodes:
            return 0

        _IMP_KW = {"버그", "오류", "에러", "설계", "보안", "취약", "중요", "핵심", "수정", "리뷰"}

        def _importance(ep: Episode) -> float:
            score = ep.importance
            score += min((time.time() - ep.ts) / 86400.0 * -0.01, 0.0)  # 오래될수록 소폭 감소
            if any(kw in ep.text for kw in _IMP_KW):
                score += 0.2
            return clamp(score)

        scored = sorted(nodes, key=_importance)
        # 하위 10% 삭제
        n_delete = max(0, int(len(scored) * 0.10))
        to_delete = set(id(ep) for ep in scored[:n_delete])

        before = len(self.long_term)
        self.long_term = deque(
            (ep for ep in self.long_term if id(ep) not in to_delete),
            maxlen=self.LONG_MAXLEN,
        )
        deleted = before - len(self.long_term)
        log.info("sleep_prune: removed %d episodes", deleted)
        return deleted


# ── snippet memory ────────────────────────────────────────────────────────────

@dataclass
class CodeSnippet:
    code:        str
    description: str
    language:    str  = "python"
    tags:        List[str] = field(default_factory=list)
    used_count:  int  = 0
    ts:          float = field(default_factory=time.time)


class SnippetMemory:
    MAXLEN = 50

    def __init__(self):
        self._snippets: List[CodeSnippet] = []

    def add(self, code: str, description: str, language: str = "python", tags: List[str] = None):
        s = CodeSnippet(code=code, description=description, language=language, tags=tags or [])
        self._snippets.append(s)
        if len(self._snippets) > self.MAXLEN:
            self._snippets.sort(key=lambda s: s.used_count)
            self._snippets = self._snippets[1:]

    def search(self, query: str, top_k: int = 3) -> List[CodeSnippet]:
        words  = set(query.lower().split())
        scored = []
        for s in self._snippets:
            score = (
                sum(1 for w in words if w in s.description.lower()) * 2
                + sum(1 for w in words if any(w in t for t in s.tags))
            )
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [s for _, s in scored[:top_k]]
        for s in results:
            s.used_count += 1
        return results


# ── knowledge graph ───────────────────────────────────────────────────────────

@dataclass
class Triple:
    """
    Knowledge triple unit.
    subject  : entity (function, class, file, ...)
    relation : calls / contains / fixes / related_to / causes / depends_on
    obj      : entity
    confidence increases on repeated observation, decreases on conflict.
    """
    subject:    str
    relation:   str
    obj:        str
    confidence: float = 0.5
    ts:         float = field(default_factory=time.time)
    count:      int   = 1

    @property
    def key(self) -> str:
        return f"{self.subject}|{self.relation}|{self.obj}"


class LightKG:
    """
    Coding context knowledge graph — Triple-based.
    Nodes: functions / classes / files / bugs / patterns
    Relations: calls, contains, fixes, related_to, causes, depends_on

    Conflict detection: opposing relations on same (subject, obj) pair
    cause confidence decay on the existing triple and discount on the new one.
    Overflow: lowest-confidence triples removed first.
    """
    MAXTRIPLES = 600

    _OPPOSITES = [
        {"fixes",      "causes"},
        {"calls",      "never_calls"},
        {"depends_on", "independent_of"},
        {"contains",   "excludes"},
    ]

    def __init__(self):
        self._triples: Dict[str, Triple] = {}
        self._nodes:   Dict[str, Dict]   = {}

    def upsert(self, subject: str, relation: str, obj: str,
               confidence: float = 0.5) -> str:
        key = f"{subject}|{relation}|{obj}"

        conflict_key = self._find_conflict(subject, relation, obj)
        if conflict_key and conflict_key in self._triples:
            self._triples[conflict_key].confidence = max(
                0.05, self._triples[conflict_key].confidence * 0.6
            )
            log.info("kg: conflict detected [%s] vs [%s]", key, conflict_key)
            self._triples[key] = Triple(subject=subject, relation=relation,
                                        obj=obj, confidence=confidence * 0.8)
            self._trim()
            return "conflict"

        if key in self._triples:
            t = self._triples[key]
            t.count      += 1
            t.confidence  = min(1.0, t.confidence + confidence * 0.15)
            t.ts          = time.time()
            return "reinforced"

        self._triples[key] = Triple(subject=subject, relation=relation,
                                    obj=obj, confidence=confidence)
        self._nodes.setdefault(subject, {"type": "entity", "desc": "", "ts": time.time()})
        self._nodes.setdefault(obj,     {"type": "entity", "desc": "", "ts": time.time()})
        self._trim()
        return "new"

    def _find_conflict(self, subject: str, relation: str, obj: str) -> Optional[str]:
        for pair in self._OPPOSITES:
            if relation in pair:
                opposite = (pair - {relation}).pop()
                ckey     = f"{subject}|{opposite}|{obj}"
                if ckey in self._triples:
                    return ckey
        return None

    def _trim(self):
        if len(self._triples) > self.MAXTRIPLES:
            sorted_keys = sorted(self._triples, key=lambda k: self._triples[k].confidence)
            for k in sorted_keys[:len(self._triples) - self.MAXTRIPLES]:
                t = self._triples.pop(k)
                if not any(tr.subject == t.subject for tr in self._triples.values()):
                    self._nodes.pop(t.subject, None)
                if not any(tr.obj == t.obj for tr in self._triples.values()):
                    self._nodes.pop(t.obj, None)

    def add_node(self, name: str, node_type: str, desc: str = ""):
        self._nodes[name] = {"type": node_type, "desc": desc, "ts": time.time()}

    def add_edge(self, src: str, rel: str, dst: str):
        self.upsert(src, rel, dst, confidence=0.5)

    def query(self, keyword: str, top_k: int = 5) -> str:
        kw   = keyword.lower()
        hits = [
            t for t in self._triples.values()
            if kw in t.subject.lower() or kw in t.obj.lower() or kw in t.relation.lower()
        ]
        hits.sort(key=lambda t: t.confidence, reverse=True)
        lines = [
            f"{t.subject} --{t.relation}({t.confidence:.2f})--> {t.obj}"
            for t in hits[:top_k]
        ]
        return "\n".join(lines)

    def stats(self) -> str:
        avg_conf = (
            sum(t.confidence for t in self._triples.values()) / max(len(self._triples), 1)
        )
        return f"triples={len(self._triples)} nodes={len(self._nodes)} avg_conf={avg_conf:.2f}"


# ── vector index ──────────────────────────────────────────────────────────────

class VectorIndex:
    """
    Sentence-transformers vector search.
    Falls back to keyword search when unavailable.

    Composite score = cosine_similarity * ebbinghaus_decay * (1 + importance)
    Access count extends half-life (frequently used memories persist longer).
    """
    HALF_LIFE_BASE = 86400.0 * 3
    HALF_LIFE_MAX  = 86400.0 * 30

    def __init__(self):
        self._texts:      List[str]   = []
        self._vectors:    List        = []
        self._ts:         List[float] = []
        self._importance: List[float] = []
        self._access:     List[int]   = []

    def add(self, text: str, importance: float = 0.5):
        if not EMBED_OK:
            return
        try:
            vec = _EMBED_MODEL.encode(text, convert_to_numpy=True)
            self._texts.append(text)
            self._vectors.append(vec)
            self._ts.append(time.time())
            self._importance.append(clamp(importance))
            self._access.append(0)
            if len(self._texts) > 500:
                self._texts      = self._texts[-400:]
                self._vectors    = self._vectors[-400:]
                self._ts         = self._ts[-400:]
                self._importance = self._importance[-400:]
                self._access     = self._access[-400:]
        except Exception as e:
            log.debug("vector_index: add failed: %s", e)

    def _decay(self, idx: int) -> float:
        age = time.time() - self._ts[idx]
        ac  = max(self._access[idx], 1)
        hl  = min(self.HALF_LIFE_BASE * (1.0 + math.log(1 + ac)), self.HALF_LIFE_MAX)
        return math.exp(-math.log(2) * age / hl)

    def search(self, query: str, top_k: int = 4) -> List[str]:
        if not EMBED_OK or not self._texts:
            return []
        try:
            import numpy as np
            q_vec  = _EMBED_MODEL.encode(query, convert_to_numpy=True)
            matrix = np.array(self._vectors)
            cos    = matrix @ q_vec / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec) + 1e-9)

            # 후보를 넉넉하게 뽑은 뒤 MMR로 줄임
            fetch_k = min(len(self._texts), max(top_k * 3, 20))
            top_raw = sorted(range(len(cos)), key=lambda i: cos[i], reverse=True)[:fetch_k]
            raw_scores = [float(cos[i]) for i in top_raw]

            selected = _mmr_select(
                indices  = top_raw,
                scores   = raw_scores,
                vectors  = self._vectors,
                ts_list  = self._ts,
                k        = top_k,
            )

            results = []
            for i in selected:
                if cos[i] > 0.3:
                    self._access[i] += 1
                    results.append(self._texts[i])
            return results
        except Exception as e:
            log.debug("vector_index: search failed: %s", e)
            return []


# ── persistence (SQLite) ──────────────────────────────────────────────────────

class MemoryDB:
    """
    Async SQLite persistence.
    Uses aiosqlite if available; falls back to asyncio.to_thread.
    WAL mode for concurrent read/write without blocking.
    """

    _INIT_SQL = """
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT, source TEXT,
            importance REAL, ts REAL,
            tags TEXT
        );
        CREATE TABLE IF NOT EXISTS snippets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT, description TEXT,
            language TEXT, tags TEXT,
            used_count INTEGER, ts REAL
        );
        CREATE TABLE IF NOT EXISTS kg_nodes (
            name TEXT PRIMARY KEY,
            type TEXT, desc TEXT, ts REAL
        );
    """

    def __init__(self, path: str):
        self._path        = path
        self._initialized = False
        if not AIOSQLITE_OK:
            import sqlite3 as _sqlite3
            self._sync_conn = _sqlite3.connect(path, check_same_thread=False)
            self._sync_conn.executescript(self._INIT_SQL)
            self._sync_conn.commit()
            self._initialized = True

    async def _ensure_init(self):
        if self._initialized:
            return
        async with aiosqlite.connect(self._path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA synchronous=NORMAL;")
            await db.executescript(self._INIT_SQL)
            await db.commit()
        self._initialized = True

    async def save_episode(self, ep: Episode):
        try:
            if AIOSQLITE_OK:
                await self._ensure_init()
                async with aiosqlite.connect(self._path) as db:
                    await db.execute(
                        "INSERT INTO episodes (text, source, importance, ts, tags) VALUES (?,?,?,?,?)",
                        (ep.text, ep.source, ep.importance, ep.ts, json.dumps(ep.tags))
                    )
                    await db.commit()
            else:
                def _sync():
                    self._sync_conn.execute(
                        "INSERT INTO episodes (text, source, importance, ts, tags) VALUES (?,?,?,?,?)",
                        (ep.text, ep.source, ep.importance, ep.ts, json.dumps(ep.tags))
                    )
                    self._sync_conn.commit()
                await asyncio.to_thread(_sync)
        except Exception as e:
            log.debug("memory_db: save_episode failed: %s", e)

    async def load_recent_episodes(self, n: int = 50) -> List[Episode]:
        try:
            if AIOSQLITE_OK:
                await self._ensure_init()
                async with aiosqlite.connect(self._path) as db:
                    async with db.execute(
                        "SELECT text, source, importance, ts, tags FROM episodes ORDER BY ts DESC LIMIT ?", (n,)
                    ) as cursor:
                        rows = await cursor.fetchall()
            else:
                def _sync():
                    return self._sync_conn.execute(
                        "SELECT text, source, importance, ts, tags FROM episodes ORDER BY ts DESC LIMIT ?", (n,)
                    ).fetchall()
                rows = await asyncio.to_thread(_sync)

            return [
                Episode(text=r[0], source=r[1], importance=r[2], ts=r[3],
                        tags=json.loads(r[4] or "[]"))
                for r in reversed(rows)
            ]
        except Exception as e:
            log.debug("memory_db: load failed: %s", e)
            return []

    async def close(self):
        if not AIOSQLITE_OK and hasattr(self, "_sync_conn"):
            try:
                await asyncio.to_thread(self._sync_conn.close)
            except Exception:
                pass


# ── facade ────────────────────────────────────────────────────────────────────

class CorundumMemory:
    """
    Corundum memory facade.
    No LLM. Fully rule/DB based.

    recall() pipeline:
      _fetch_raw() — sync, fast keyword/vector search
      _refine()    — rule-based formatting (swap for async LLM if needed)
    """

    def __init__(self):
        self.episodes = EpisodeMemory()
        self.snippets = SnippetMemory()
        self.kg       = LightKG()
        self.vectors  = VectorIndex()

        db_path  = getattr(Cfg, "MEMORY_DB_PATH", "corundum_memory.db")
        self._db = MemoryDB(db_path)
        self._loaded = False

    async def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        eps = await self._db.load_recent_episodes(50)
        for ep in eps:
            self.episodes.long_term.append(ep)
            self.vectors.add(ep.text)
        log.info("memory: restored %d episodes", len(eps))

    async def recall(self, user_input: str, metrics_ctx: Dict) -> Dict:
        await self._ensure_loaded()
        raw = self._fetch_raw(user_input, metrics_ctx)
        return self._refine(raw, user_input)

    def _fetch_raw(self, user_input: str, metrics_ctx: Dict) -> Dict:
        energy = metrics_ctx.get("energy", 1.0)
        gear   = metrics_ctx.get("gear", "NORMAL")
        layer  = _gear_to_layer(gear)
        top_k  = max(2, int(getattr(Cfg, "MEMORY_TOP_K", 4) * energy))

        # layer별 검색 깊이
        # SURFACE(0): 벡터만
        # VORTEX (1): 벡터 + KG top-2
        # DEEP   (2): 벡터 + KG top-4 + 최근 에피소드 추가
        if layer >= 2:
            top_k = min(top_k + 2, 8)

        if EMBED_OK:
            similar_texts = self.vectors.search(user_input, top_k=top_k)
        else:
            similar_texts = [ep.text for ep in self.episodes.keyword_recall(user_input, top_k=top_k)]

        kg_lines = []
        if layer >= 1:
            keywords = [w for w in user_input.split() if len(w) >= 3][:3]
            kg_top_k = 4 if layer >= 2 else 2
            for kw in keywords:
                hint = self.kg.query(kw, top_k=kg_top_k)
                if hint:
                    kg_lines.append(hint)

        working_n = 8 if layer >= 2 else 6
        return {
            "_raw_texts": similar_texts,
            "_kg_lines":  kg_lines,
            "_working":   self.episodes.working_context(n=working_n),
            "_layer":     layer,
        }

    def _refine(self, raw: Dict, user_input: str) -> Dict:
        texts    = raw.get("_raw_texts", [])
        recalled = "\n".join(f"- {t[:80]}" for t in texts) if texts else ""
        kg_hints = "\n".join(raw.get("_kg_lines", [])).strip()
        working  = raw.get("_working", "")
        return {"recalled": recalled, "kg_hints": kg_hints, "working": working}

    def record(self, user_input: str, response: str, emotion_ctx: Dict, goal_ctx: Dict):
        urgency    = goal_ctx.get("urgency", 0.3) if goal_ctx else 0.3
        importance = clamp(0.5 + urgency * 0.3)

        user_ep = Episode(text=user_input, source="user",     importance=importance)
        resp_ep = Episode(text=response,   source="corundum", importance=importance * 0.8)

        self.episodes.record(user_input, "user",     importance)
        self.episodes.record(response,   "corundum", importance * 0.8)
        self.vectors.add(user_input, importance=importance)
        self.vectors.add(response,   importance=importance * 0.8)

        if importance >= 0.6:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._db.save_episode(user_ep))
                    loop.create_task(self._db.save_episode(resp_ep))
            except RuntimeError:
                pass

    def recent_summary(self) -> str:
        return self.episodes.working_context(n=10) or "no memory"

    async def consolidate(self):
        if len(self.episodes.long_term) >= EpisodeMemory.LONG_MAXLEN * 0.9:
            pruned = self.episodes.sleep_prune(min_nodes=30)
            if pruned == 0:
                # prune로 못 줄였으면 기존 단순 정렬로 폴백
                sorted_eps = sorted(self.episodes.long_term, key=lambda e: e.importance)
                to_remove  = sorted_eps[:10]
                for ep in to_remove:
                    try:
                        self.episodes.long_term.remove(ep)
                    except ValueError:
                        pass
                log.info("memory: fallback consolidate, removed %d episodes", len(to_remove))

    async def save(self):
        try:
            if not AIOSQLITE_OK and hasattr(self._db, "_sync_conn"):
                await asyncio.to_thread(self._db._sync_conn.commit)
            log.info("memory: saved")
        except Exception as e:
            log.warning("memory: save failed: %s", e)

    async def close(self):
        await self._db.close()

    def __del__(self):
        try:
            asyncio.get_event_loop().create_task(self._db.close())
        except Exception:
            pass
