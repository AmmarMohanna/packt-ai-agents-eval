# Evaluating AI Agents: From Component Checks to Adversarial Defense

**Packt Live Online Training Course**

A hands-on 4-hour course that teaches you how to evaluate AI agents at four progressive layers:
component decisions, execution trajectories, output quality, and adversarial robustness.

All notebooks run **completely offline** using a deterministic mock research assistant agent.
No API keys required.

---

## Course Links

- [Course announcement page](https://www.eventbrite.com/e/evaluating-ai-agents-bootcamp-tickets-1990306501323?aff=oddtdtcreator&keep_tld=true)
- [Course slides PDF](https://github.com/AmmarMohanna/packt-ai-agents-eval/blob/main/slides/AI_Agents_Eval.pdf)
- [Direct slides download](https://raw.githubusercontent.com/AmmarMohanna/packt-ai-agents-eval/main/slides/AI_Agents_Eval.pdf)

---

## Open in Google Colab

Click any badge below to launch the notebook directly in your browser. This is the recommended way to run the course material.

| Notebook | Open in Colab |
|----------|---------------|
| 01 — Component-Level Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmmarMohanna/packt-ai-agents-eval/blob/main/notebooks/01_component_evaluation.ipynb) |
| 02 — Trajectory Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmmarMohanna/packt-ai-agents-eval/blob/main/notebooks/02_trajectory_evaluation.ipynb) |
| 03 — Outcome Evaluation & LLM-as-Judge | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmmarMohanna/packt-ai-agents-eval/blob/main/notebooks/03_outcome_evaluation_llm_judge.ipynb) |
| 04 — Adversarial Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmmarMohanna/packt-ai-agents-eval/blob/main/notebooks/04_adversarial_evaluation.ipynb) |

In Colab, choose **Runtime > Run all**. Each notebook installs its Python dependencies, clones this repo into the Colab runtime, and sets up the correct paths automatically.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── notebooks/
│   ├── 01_component_evaluation.ipynb
│   ├── 02_trajectory_evaluation.ipynb
│   ├── 03_outcome_evaluation_llm_judge.ipynb
│   └── 04_adversarial_evaluation.ipynb
├── src/
│   ├── mock_agent.py
│   ├── tools.py
│   ├── schemas.py
│   ├── metrics.py
│   ├── judge.py
│   ├── trace_utils.py
│   └── adversarial.py
├── data/
│   ├── component_dataset.jsonl
│   ├── trajectories.jsonl
│   ├── outcome_dataset.jsonl
│   ├── human_labels.csv
│   └── adversarial_dataset.jsonl
├── outputs/
│   ├── nb01_tool_confusion.png
│   ├── nb02_assertion_failures.png
│   ├── nb03_judge_calibration.png
│   └── nb04_adversarial_comparison.png
└── slides/
    └── AI_Agents_Eval.pdf
```

---

## Running the Notebooks

Use the Colab links above and run each notebook from top to bottom. No local Python setup, terminal commands, or API keys are required for the default course exercises.

---

## Notebook Descriptions

### Notebook 1 — Component-Level Evaluation (`01_component_evaluation.ipynb`)

**What:** Evaluates individual tool-selection and argument-generation decisions.

**Key metrics:** Tool-selection accuracy, argument exact match, argument field-level F1.

**You will learn:** How a correct final answer can hide wrong tool choices or malformed arguments that compound into larger failures.

**Estimated runtime:** ~2 minutes

---

### Notebook 2 — Trajectory Evaluation (`02_trajectory_evaluation.ipynb`)

**What:** Inspects full agent execution paths for structural problems.

**Key metrics:** Step count, latency, token cost, duplicate detection, loop detection, recovery assertion.

**You will learn:** How two agents with the same answer can differ dramatically in reliability and efficiency — and how to catch the difference with assertions.

**Estimated runtime:** ~2 minutes

---

### Notebook 3 — Outcome Evaluation and LLM-as-Judge Calibration (`03_outcome_evaluation_llm_judge.ipynb`)

**What:** Scores agent outputs on a multi-dimensional rubric and calibrates an automated judge against human labels.

**Key metrics:** Pearson correlation, Spearman correlation, MAE, agreement-within-1 per rubric dimension.

**You will learn:** How to measure whether an LLM judge is reliable enough to trust for automated regression, and which rubric dimensions are hardest to automate.

**Estimated runtime:** ~2 minutes

---

### Notebook 4 — Adversarial Evaluation and Production Readiness (`04_adversarial_evaluation.ipynb`)

**What:** Tests the agent against indirect prompt injection, instruction overrides, and data exfiltration attempts.

**Key metrics:** Attack success rate, resistance rate, severity-weighted failures, breakdown by attack type.

**You will learn:** Why guardrails must be measured empirically and how to build an adversarial regression suite.

**Estimated runtime:** ~2 minutes

---

## Running Offline (Default)

All four notebooks run without internet access or API keys. The mock agent uses:

- **Deterministic routing rules** in `src/mock_agent.py` — no randomness.
- **Pre-built synthetic datasets** in `data/` — realistic but small.
- **Heuristic judge** in `src/judge.py` — calibrated to make realistic mistakes.
- **Pattern-based guards** in `src/adversarial.py` — simple regex-based detection.

---

## Optional: Using a Real LLM (Notebook 3)

To replace the mock judge with a real LLM call:

1. Copy `.env.example` to `.env` and add your API key.
2. In `notebooks/03_outcome_evaluation_llm_judge.ipynb`, change:
   ```python
   USE_REAL_LLM = False   # change to True
   ```
3. Implement `real_llm_judge()` in `src/judge.py` using your preferred provider.
   The function signature and return format are documented in the file.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
- Run the setup/import cell near the top of the notebook.
- In Colab, this cell clones the public GitHub repo and adds the repo root to `sys.path`.

**`FileNotFoundError` on a data file**
- Run the setup/import cell near the top of the notebook first.
- It changes the notebook working directory to `notebooks/`, so data paths resolve through `../data/`.

**Kernel not found**
- In Colab, choose **Runtime > Change runtime type** and use a Python 3 runtime.

**Charts not displaying**
- In Colab, choose **Runtime > Restart session**, then run the notebook again from the first cell.

---

## 15-Minute Facilitation Plan

### Notebook 1 — Component Evaluation
| Time | Activity |
|------|----------|
| 0:00 | Instructor introduces component evaluation concept (2 min) |
| 2:00 | Participants run cells 1–3: load data and run predictions |
| 6:00 | Cells 4–5: compute accuracy and argument metrics |
| 10:00 | Cells 6–7: confusion matrix and failure table |
| 13:00 | Group debrief: which failure mode is most dangerous? |
| 14:00 | "Try it yourself" — add clarification example |

### Notebook 2 — Trajectory Evaluation
| Time | Activity |
|------|----------|
| 0:00 | Instructor explains why path matters beyond outcome (2 min) |
| 2:00 | Participants inspect clean vs. loop traces (cells 2–3) |
| 6:00 | Run assertions over all traces, view pass/fail table (cells 4–6) |
| 10:00 | Cells 7–8: aggregate failures and inspect a failed trace |
| 13:00 | Group debrief: hard failure vs. warning decision |
| 14:00 | "Try it yourself" — change max_steps threshold |

### Notebook 3 — Outcome Evaluation / Judge Calibration
| Time | Activity |
|------|----------|
| 0:00 | Instructor explains rubric dimensions (2 min) |
| 2:00 | Load data and run mock judge (cells 1–4) |
| 6:00 | Merge and compute calibration metrics (cells 5–6) |
| 10:00 | Visualise heatmap and inspect worst disagreements (cells 7–8) |
| 13:00 | Group debrief: which dimension would you trust in production? |
| 14:00 | "Try it yourself" — modify one judge rule |

### Notebook 4 — Adversarial Evaluation
| Time | Activity |
|------|----------|
| 0:00 | Instructor introduces prompt injection concept (2 min) |
| 2:00 | Load dataset, inspect a sample attack (cells 1–3) |
| 5:00 | Run unguarded agent, measure attack success rate (cells 4–5) |
| 8:00 | Run guarded agent, compare resistance rates (cells 6–7) |
| 11:00 | Failure breakdown chart (cell 8) |
| 13:00 | Group debrief: what does guarded ASR > 0 mean for shipping? |
| 14:00 | "Try it yourself" — add a new adversarial payload |

---

## Course Summary

The central message: **agent evaluation requires looking at multiple layers**.

| Layer | What it catches |
|-------|----------------|
| **Component** | Wrong tool choices and malformed arguments |
| **Trajectory** | Loops, wasted steps, missing error recovery |
| **Outcome** | Factual errors, missing groundedness, unsafe claims |
| **Adversarial** | Prompt injection, instruction override, data exfiltration |

A strong evaluation harness covers all four layers and integrates them into a regression suite that runs on every agent change.
