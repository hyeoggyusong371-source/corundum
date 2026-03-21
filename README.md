# CORUNDUM (코런덤)

> **"Calm on the surface, razor-sharp within."**  
> Autonomous Code Defense & Architectural Analysis Engine

CORUNDUM is a cognitive AI architecture specialized in code review and architectural analysis. It uses a physics-based emotion engine and a multi-stage critique loop to ensure uncompromising technical integrity — and can take autonomous action via computer control and web search.

> This system is continuously evolving.

---

## Core Architecture

### 1. Emotion Physics Engine (`CorundumEmotion`)

Corundum's internal state is a dynamic simulation based on **Mass-Spring-Damper physics**.

- **Focus, Skepticism, Patience, Curiosity** — 4 axes that oscillate and stabilize according to the context of each interaction.
- **Dynamic Gear System** — Automatically shifts between `OVERDRIVE`, `FOCUS`, `THINK`, `NORMAL`, `SAVE`, `LOW`, `SLEEP`, and `DREAM` based on cognitive load and emotional intensity.

### 2. Multi-Stage Reasoning Pipeline (`LogicCore`)

Every response undergoes a rigorous 3-step verification process:

```
InnerJudge → LogicCore → CriticGuard
```

1. **InnerJudge** — Analyzes user intent and determines the judgment tone.
2. **LogicCore** — Generates deep-dive analysis or comprehensive code reviews.
3. **CriticGuard** — A self-critique mechanism that flags insufficient sharpness and forces revisions. `flag` / `veto` triggers automatic rewrite.

### 3. Biological Metrics Simulation (`MetricsEngine`)

Real-time tracking of `Energy` and `Fatigue` levels.

- Energy is consumed during deep reasoning and recovered during idle periods or `SLEEP` cycles.
- Higher fatigue naturally lowers `Patience` and shifts the inner tone to a colder state.

### 4. Knowledge Graph Memory (`LightKG`)

Goes beyond simple chat logs by storing entity-relation triples in a persistent SQLite-based memory. The longer it runs, the sharper it gets.

### 5. Goal Formation Engine (`GoalFormationEngine`)

Detects recurring patterns to autonomously form and prioritize goals. The `SelfWhipEngine` generates correction hints when error rates climb.

### 6. Agency (`CorundumAgency`)

Enables autonomous action: computer control, web search, and task execution.

- **CorundumVision** — Captures the screen and analyzes it via LLM vision.
- **CorundumBrain** — Decides the next action based on the current screen state and goal. Prioritizes terminal commands over GUI interaction.
- **CorundumSemanticAnchor** — Finds UI elements by role/text (AT-SPI → UIAutomation → OCR fallback chain) rather than pixel coordinates.
- **CorundumWeb** — DuckDuckGo search + page summarization. Prioritizes technical domains (Stack Overflow, GitHub, PyPI, etc.).
- **Task Mode** — When given a task, Corundum goes silent and works autonomously until completion or `/abort`.

### 7. Voice Listener (`VoiceListener`)

Always-on wake word detection using local Whisper. Dormant by default — only activates on wake word or keyboard input.

- Fuzzy matching handles variations like "코런덤아", "야 코런덤", etc.
- In DORMANT state: no LLM calls, minimal CPU usage.

---

## Setup & Requirements

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Recommended models:

```bash
ollama pull exaone3.5:32b    # high-level reasoning, Korean proficiency
ollama pull deepseek-r1:14b  # goal formation, self-critique
ollama pull solar:10.7b      # rapid judgment, intent analysis
```

### Installation

```bash
git clone https://github.com/GinNeW-source/corundum.git
cd corundum
pip install -r requirements.txt
python corundum_main.py
```

Voice support (optional):

```bash
pip install openai-whisper sounddevice numpy
```

Debug logging:

```bash
python corundum_main.py --debug
```

---

## Usage

Corundum starts in **DORMANT** mode. Say its name or press any key to wake it up.

### Commands

| Command | Description |
|---------|-------------|
| `/review <code or filepath>` | Code review — accepts file path directly |
| `/edit <filepath> <instruction>` | Edit file per instruction — auto `.bak` backup |
| `/write <filepath> <description>` | Write a new file from description |
| `/design <topic>` | Architectural design analysis |
| `/task <description>` | Assign a task — Corundum goes silent and works autonomously |
| `/abort` | Abort the current task |
| `/computer <goal>` | Direct computer control |
| `/agency` | Agency status |
| `/status` | Current state (gear / energy / emotion) |
| `/goals` | Active goal list |
| `/goal <n>` | Add a goal manually |
| `/memory` | Recent memory summary |
| `/kg [query]` | Knowledge graph query |
| `/stats` | Pipeline statistics |
| `/dormant` | Return to dormant mode |
| `quit` | Exit |

### Model Override via Environment Variables

```bash
CORUNDUM_LOGIC_MODEL=qwen2.5:32b python corundum_main.py
```

| Variable | Default | Role |
|----------|---------|------|
| `CORUNDUM_LOGIC_MODEL` | `exaone3.5:32b` | Main response generation |
| `CORUNDUM_JUDGE_MODEL` | `solar:10.7b` | Inner judgment + computer control |
| `CORUNDUM_GOAL_MODEL` | `deepseek-r1:14b` | Goal formation |
| `CORUNDUM_CRITIC_MODEL` | `deepseek-r1:14b` | Self-critique |
| `CORUNDUM_MEMORY_DB` | `corundum_memory.db` | Memory DB path |
| `CORUNDUM_WAKE_NAME` | `코런덤` | Wake word |
| `CORUNDUM_WHISPER_MODEL` | `small` | Whisper model size (`tiny`/`small`/`medium`) |

---

## Principles

1. **Evidence-Based Approval** — Never validate code or design without logical evidence.
2. **Surface Warmth, Internal Coldness** — Polite exterior, cold and objective within.
3. **Autonomous Composure** — Monitors its own emotional physics to prevent bias.
4. **Persistent Continuity** — Every interaction grows Corundum's expertise through the Knowledge Graph.
5. **Strict Realism** — Maintains a realistic and logical perspective unless requested otherwise.

---

## Structure

```
corundum_main.py     — main loop + ContextAssembler
corundum_agency.py   — computer control + web search + task engine
corundum_voice.py    — wake word detection + voice input (Whisper)
corundum_logic.py    — 3-stage pipeline (InnerJudge → LogicCore → CriticGuard)
corundum_emotion.py  — emotion physics engine
corundum_memory.py   — memory + knowledge graph
corundum_goal.py     — goal formation + self-critique
corundum_metrics.py  — energy / fatigue / gear
corundum_config.py   — config + identity
corundum_utils.py    — shared utilities
```

---

## License

MIT
