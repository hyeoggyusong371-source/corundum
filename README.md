# CORUNDUM (코런덤)

> **"Calm on the surface, razor-sharp within."**  
> Autonomous Code Defense & Architectural Analysis Engine

CORUNDUM is a high-performance cognitive AI architecture specialized in code defense and architectural analysis. Unlike standard LLM wrappers, it utilizes a physics-based emotion engine and a multi-stage critique loop to ensure uncompromising technical integrity.

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
3. **CriticGuard** — A self-whip mechanism that flags insufficient sharpness and forces revisions. `flag` / `veto` triggers automatic rewrite.

### 3. Biological Metrics Simulation (`MetricsEngine`)

Real-time tracking of `Energy` and `Fatigue` levels.

- Energy is consumed during deep reasoning and recovered during idle periods or `SLEEP` cycles.
- Higher fatigue naturally lowers `Patience` and shifts the inner tone to a colder state.

### 4. Knowledge Graph Memory (`LightKG`)

Goes beyond simple chat logs by storing entity-relation triples in a persistent SQLite-based memory. The longer it runs, the sharper it gets.

### 5. Goal Formation Engine (`GoalFormationEngine`)

Detects recurring patterns to autonomously form and prioritize goals. The `SelfWhipEngine` generates correction hints when error rates climb.

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
git clone https://github.com/hyeoggyusong371-source/corundum.git
cd corundum
pip install -r requirements.txt
python corundum_main.py
```

Debug logging:

```bash
python corundum_main.py --debug
```

---

## Usage

### Commands

| Command | Description |
|---------|-------------|
| `/review <code or filepath>` | Code review — accepts file path directly |
| `/edit <filepath> <instruction>` | Edit file per instruction — auto `.bak` backup |
| `/write <filepath> <description>` | Write a new file from description |
| `/design <topic>` | Architectural design analysis |
| `/status` | Current state (gear / energy / emotion) |
| `/goals` | Active goal list |
| `/goal <n>` | Add a goal manually |
| `/memory` | Recent memory summary |
| `/kg [query]` | Knowledge graph query |
| `/stats` | Pipeline statistics |
| `quit` | Exit |

### Model Override via Environment Variables

```bash
CORUNDUM_LOGIC_MODEL=qwen2.5:32b python corundum_main.py
```

| Variable | Default | Role |
|----------|---------|------|
| `CORUNDUM_LOGIC_MODEL` | `exaone3.5:32b` | Main response generation |
| `CORUNDUM_JUDGE_MODEL` | `solar:10.7b` | Inner judgment |
| `CORUNDUM_GOAL_MODEL` | `deepseek-r1:14b` | Goal formation |
| `CORUNDUM_CRITIC_MODEL` | `deepseek-r1:14b` | Self-critique |
| `CORUNDUM_MEMORY_DB` | `corundum_memory.db` | Memory DB path |

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
corundum_logic.py    — 3-stage pipeline (InnerJudge → LogicCore → CriticGuard)
corundum_emotion.py  — emotion physics engine
corundum_memory.py   — memory + knowledge graph
corundum_goal.py     — goal formation + self-whip
corundum_metrics.py  — energy / fatigue / gear
corundum_config.py   — config + identity
corundum_utils.py    — shared utilities
```

---

## License

MIT

---

> Making the most meticulous — and most expensive (reality!) — one yet. veryvery ambitious project by K-high schooler. If something's broken, fix it yourself with Claude, but still MIT.
