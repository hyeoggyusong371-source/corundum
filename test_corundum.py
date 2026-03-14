#!/usr/bin/env python3
"""
CORUNDUM 통합 테스트 — 실제 LLM 호출 없이 mock으로 파이프라인 전체 검증
"""
import asyncio
import sys
import traceback
sys.path.insert(0, '/home/claude')
sys.path.insert(0, '/mnt/user-data/outputs')

PASS = "✅"
FAIL = "❌"
results = []

def check(name, expr, expected=None):
    try:
        val = expr() if callable(expr) else expr
        if expected is not None:
            ok = val == expected
        else:
            ok = bool(val) if val is not None else True
        results.append((name, ok, val))
        print(f"  {PASS if ok else FAIL} {name}: {repr(val)[:80]}")
        return val
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL} {name}: {e}")
        traceback.print_exc()
        return None

# ── 1. config 임포트 ──────────────────────────────────────────────────────────
print("\n[1] Config")
try:
    from corundum_config import Cfg, CORUNDUM_IDENTITY, BOOT_MSG
    check("Cfg.LOGIC_MODEL",  lambda: isinstance(Cfg.LOGIC_MODEL, str))
    check("Cfg.JUDGE_MODEL",  lambda: isinstance(Cfg.JUDGE_MODEL, str))
    check("Cfg.GOAL_MODEL",   lambda: isinstance(Cfg.GOAL_MODEL, str))
    check("Cfg.CRITIC_MODEL", lambda: isinstance(Cfg.CRITIC_MODEL, str))
    check("CORUNDUM_IDENTITY", lambda: len(CORUNDUM_IDENTITY) > 10)
    check("GEAR_INTERVALS",   lambda: "NORMAL" in Cfg.GEAR_INTERVALS)
except Exception as e:
    print(f"  {FAIL} config 임포트 실패: {e}")

# ── 2. metrics ────────────────────────────────────────────────────────────────
print("\n[2] MetricsEngine")
try:
    from corundum_metrics import MetricsEngine
    m = MetricsEngine()
    snap = m.tick(idle_sec=0.0)
    check("tick 반환 키",     lambda: all(k in snap for k in ["energy","fatigue","gear"]))
    check("energy 범위",      lambda: 0.0 <= snap["energy"] <= 1.0)
    check("fatigue 범위",     lambda: 0.0 <= snap["fatigue"] <= 1.0)
    check("gear 유효값",      lambda: snap["gear"] in ["OVERDRIVE","FOCUS","THINK","NORMAL","SAVE","LOW","SLEEP","DREAM"])
    m.on_response("테스트 응답")
    snap2 = m.tick()
    check("on_response 후 fatigue 증가", lambda: snap2["fatigue"] >= snap["fatigue"])
    snap3 = m.snapshot()
    check("snapshot 키 일치", lambda: snap3.keys() == snap.keys())
except Exception as e:
    print(f"  {FAIL} metrics 실패: {e}")
    traceback.print_exc()

# ── 3. emotion ────────────────────────────────────────────────────────────────
print("\n[3] CorundumEmotion")
try:
    from corundum_emotion import CorundumEmotion
    emo = CorundumEmotion()
    metrics_ctx = {"energy": 0.9, "fatigue": 0.1, "gear": "NORMAL", "idle_sec": 0.0}
    ctx = asyncio.run(emo.process("버그 찾았어요", metrics_ctx))
    EMOTION_KEYS = ["emotion_tag","emotion_hint","inner_hint","immersion",
                    "focus","skepticism","patience","curiosity",
                    "surface_warmth","inner_edge","gear","fatigue"]
    check("process 반환 타입", lambda: isinstance(ctx, dict))
    check("필수 키 전부 있음",  lambda: all(k in ctx for k in EMOTION_KEYS))
    check("immersion 범위",    lambda: 0.0 <= ctx["immersion"] <= 1.0)
    check("emotion_tag 문자열",lambda: isinstance(ctx["emotion_tag"], str))
    emo.on_interact()
    emo.physio_tick(idle_sec=5.0)
    check("current_tag 반환", lambda: isinstance(emo.current_tag(), str))
    asyncio.run(emo.post_feedback("테스트 응답", ctx))
    check("post_feedback OK", True)
except Exception as e:
    print(f"  {FAIL} emotion 실패: {e}")
    traceback.print_exc()

# ── 4. memory ─────────────────────────────────────────────────────────────────
print("\n[4] CorundumMemory")
try:
    import os; os.environ["CORUNDUM_MEMORY_DB"] = "/tmp/test_corundum.db"
    from corundum_memory import CorundumMemory
    mem = CorundumMemory()
    metrics_ctx = {"energy": 0.9, "fatigue": 0.1}
    result = asyncio.run(mem.recall("파이썬 버그", metrics_ctx))
    check("recall 반환 타입",  lambda: isinstance(result, dict))
    check("recall 키 존재",    lambda: all(k in result for k in ["recalled","kg_hints","working"]))
    mem.record("테스트 입력", "테스트 응답", {}, {})
    check("record OK", True)
    summary = mem.recent_summary()
    check("recent_summary 문자열", lambda: isinstance(summary, str))
    asyncio.run(mem.consolidate())
    check("consolidate OK", True)
    asyncio.run(mem.save())
    check("save OK", True)
except Exception as e:
    print(f"  {FAIL} memory 실패: {e}")
    traceback.print_exc()

# ── 5. goal ───────────────────────────────────────────────────────────────────
print("\n[5] CorundumGoal")
try:
    from corundum_goal import CorundumGoal
    goal = CorundumGoal()
    metrics_ctx = {"energy": 0.9, "fatigue": 0.1}
    # LLM 없이도 폴백 동작하는지
    result = asyncio.run(goal.process("코드 리뷰 해줘", metrics_ctx))
    check("process 반환 타입", lambda: isinstance(result, dict))
    check("process 키 존재",   lambda: all(k in result for k in ["current_goal","goal_hint","critique","urgency"]))
    ret = goal.add_goal("테스트 목표", "설명")
    check("add_goal 반환 문자열", lambda: isinstance(ret, str))
    check("top_goal_name",     lambda: isinstance(goal.top_goal_name(), str))
    summary = goal.summary()
    check("summary 문자열",    lambda: isinstance(summary, str))
    goal.observe_outcome("bug_found", "리뷰 완료", "pass")
    check("observe_outcome OK", True)
    stats = goal.stats()
    check("stats 키 존재",     lambda: all(k in stats for k in ["active_goals","error_rate","llm_calls"]))
    auto = asyncio.run(goal.autonomous_tick(metrics_ctx))
    check("autonomous_tick 문자열", lambda: isinstance(auto, str))
except Exception as e:
    print(f"  {FAIL} goal 실패: {e}")
    traceback.print_exc()

# ── 6. logic ──────────────────────────────────────────────────────────────────
print("\n[6] CorundumLogic")
try:
    from corundum_logic import CorundumLogic
    logic = CorundumLogic()
    ctx = {
        "energy": 0.9, "fatigue": 0.1, "gear": "NORMAL",
        "focus": 0.7, "skepticism": 0.5, "patience": 0.8, "curiosity": 0.6,
        "surface_warmth": 0.7, "inner_edge": 0.65,
        "emotion_hint": "", "inner_hint": "",
        "immersion": 1.0, "current_goal": "", "goal_hint": "", "self_critique": "",
        "recalled_memory": "", "kg_hints": "", "working_memory": "",
    }
    # mock 모드 동작 확인 (ollama 없으면)
    result = asyncio.run(logic.process("파이썬 리스트 컴프리헨션이 뭐예요?", ctx))
    check("process 반환 문자열", lambda: isinstance(result, str) and len(result) > 0)
    stats = logic.stats()
    check("stats 반환", lambda: isinstance(stats, dict))
    check("stats 키 존재", lambda: all(k in stats for k in ["total_calls","pass_rate"]))
    # review mock
    rev = asyncio.run(logic.review("def foo(): pass", ctx={}))
    check("review 반환 문자열", lambda: isinstance(rev, str))
    # design mock
    des = asyncio.run(logic.design_analysis("단일 책임 원칙", ctx={}))
    check("design_analysis 반환 문자열", lambda: isinstance(des, str))
    # edit mock
    ed = asyncio.run(logic.edit("test.py", "변수명 개선", "x = 1\ny = 2", ctx={}))
    check("edit 반환 문자열", lambda: isinstance(ed, str))
    # write mock
    wr = asyncio.run(logic.write("new_module.py", "두 수를 더하는 함수", ctx={}))
    check("write 반환 문자열", lambda: isinstance(wr, str))
except Exception as e:
    print(f"  {FAIL} logic 실패: {e}")
    traceback.print_exc()

# ── 7. ContextAssembler ───────────────────────────────────────────────────────
print("\n[7] ContextAssembler")
try:
    from corundum_main import ContextAssembler
    memory_ctx  = {"recalled": "기억1", "kg_hints": "힌트", "working": "대화"}
    emotion_ctx = {
        "emotion_tag": "focused", "emotion_hint": "집중", "inner_hint": "날카롭게",
        "immersion": 0.9, "focus": 0.8, "skepticism": 0.6, "patience": 0.7,
        "curiosity": 0.5, "surface_warmth": 0.7, "inner_edge": 0.8,
        "gear": "FOCUS", "fatigue": 0.2,
    }
    goal_ctx    = {"current_goal": "목표1", "goal_hint": "힌트", "critique": "채찍질", "urgency": 0.4}
    metrics_ctx = {"energy": 0.85, "fatigue": 0.15, "gear": "NORMAL"}
    ctx = ContextAssembler.assemble(memory_ctx, emotion_ctx, goal_ctx, metrics_ctx)
    EXPECTED_KEYS = [
        "recalled_memory","kg_hints","working_memory",
        "emotion_tag","emotion_hint","inner_hint","immersion",
        "focus","skepticism","patience","curiosity","surface_warmth","inner_edge",
        "current_goal","goal_hint","self_critique","urgency",
        "energy","fatigue","gear",
    ]
    check("조립 키 전부 있음", lambda: all(k in ctx for k in EXPECTED_KEYS))
    check("emotion gear 우선", lambda: ctx["gear"] == "FOCUS")  # emotion이 metrics보다 우선
    check("recalled_memory",  lambda: ctx["recalled_memory"] == "기억1")
except Exception as e:
    print(f"  {FAIL} ContextAssembler 실패: {e}")
    traceback.print_exc()

# ── 8. 전체 파이프라인 (mock) ─────────────────────────────────────────────────
print("\n[8] 전체 파이프라인 (mock)")
try:
    import os; os.environ["CORUNDUM_MEMORY_DB"] = "/tmp/test_corundum2.db"
    # 모듈 캐시 비우고 재임포트
    for mod in list(sys.modules.keys()):
        if "corundum" in mod:
            del sys.modules[mod]
    from corundum_main import Corundum

    async def run_pipeline():
        c = Corundum()
        r1 = await c.process("파이썬에서 리스트와 튜플의 차이가 뭐예요?")
        r2 = await c.process("이 코드 리뷰해줘: def add(a,b): return a+b")
        cmd_status = await c._handle_command("/status")
        cmd_goals  = await c._handle_command("/goals")
        cmd_help   = await c._handle_command("/help")
        cmd_stats  = await c._handle_command("/stats")
        cmd_memory = await c._handle_command("/memory")
        cmd_unk    = await c._handle_command("/unknown")

        # /edit — 파일 없을 때 에러 메시지
        cmd_edit_nofile = await c._handle_command("/edit /nonexistent.py 변수명개선")
        # /write — 실제 파일 생성
        cmd_write = await c._handle_command("/write /tmp/corundum_test_write.py 두 수를 더하는 함수")
        # /edit — 실제 파일 수정
        import pathlib
        pathlib.Path("/tmp/corundum_test_edit.py").write_text("x=1\ny=2\n", encoding="utf-8")
        cmd_edit = await c._handle_command("/edit /tmp/corundum_test_edit.py 변수명을 a, b로 바꿔줘")

        await c.memory.save()
        return r1, r2, cmd_status, cmd_goals, cmd_help, cmd_stats, cmd_memory, cmd_unk, \
               cmd_edit_nofile, cmd_write, cmd_edit

    r1, r2, cs, cg, ch, cst, cm, cu, cen, cw, ce = asyncio.run(run_pipeline())
    check("process 응답1 문자열",  lambda: isinstance(r1, str) and len(r1) > 0)
    check("process 응답2 문자열",  lambda: isinstance(r2, str) and len(r2) > 0)
    check("/status 응답",          lambda: "CORUNDUM" in cs)
    check("/goals 응답",           lambda: isinstance(cg, str))
    check("/help 응답",            lambda: "/review" in ch)
    check("/help /edit 포함",      lambda: "/edit" in ch)
    check("/help /write 포함",     lambda: "/write" in ch)
    check("/stats 응답",           lambda: "stats" in cst)
    check("/memory 응답",          lambda: isinstance(cm, str))
    check("/unknown 응답",         lambda: "알 수 없는" in cu)
    check("/edit 파일없음 에러",   lambda: "찾을 수 없" in cen)
    check("/write 응답 문자열",    lambda: isinstance(cw, str) and len(cw) > 0)
    check("/edit 응답 문자열",     lambda: isinstance(ce, str) and len(ce) > 0)
except Exception as e:
    print(f"  {FAIL} 파이프라인 실패: {e}")
    traceback.print_exc()

# ── 결과 요약 ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed
print(f"결과: {passed}/{total} 통과 | {failed}개 실패")
if failed:
    print("\n실패 목록:")
    for name, ok, val in results:
        if not ok:
            print(f"  ❌ {name}: {val}")
