# Evaluating AI Agents: From Component Checks to Adversarial Defense

**Packt Live Online Training Course**

A hands-on 4-hour course that teaches you how to evaluate AI agents at four progressive layers:
component decisions, execution trajectories, output quality, and adversarial robustness.

By default, the notebooks run in **Google Colab** using a deterministic mock research assistant agent.
No API keys are required for the core course exercises.

---

## Course Links

- [Course announcement page](https://www.eventbrite.com/e/evaluating-ai-agents-bootcamp-tickets-1990306501323?aff=oddtdtcreator&keep_tld=true)
- [Download Course Slides](https://raw.githubusercontent.com/AmmarMohanna/packt-ai-agents-eval/main/slides/AI_Agents_Eval.pdf)

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
│   ├── openai_agents.py
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

## Default Mock Mode

All four notebooks run without API keys by default. The mock agent uses:

- **Deterministic routing rules** in `src/mock_agent.py` — no randomness.
- **Pre-built synthetic datasets** in `data/` — realistic but small.
- **Heuristic judge** in `src/judge.py` — calibrated to make realistic mistakes.
- **Pattern-based guards** in `src/adversarial.py` — simple regex-based detection.

---

## Optional: Using OpenAI in the Notebooks

Each notebook includes a Colab checkbox named `USE_OPENAI` in the first setup cell. Leave it unchecked for the free deterministic course path, or check it to make real OpenAI API calls for that notebook's agent behavior.

1. In Colab, open the **Secrets** panel and add a secret named `OPENAI_API_KEY`.
2. In the first setup cell, check:
   ```python
   USE_OPENAI = True
   ```
3. Optionally change `OPENAI_MODEL`, then run the notebook from the first cell.

The notebooks read the key directly from Colab Secrets. Do not paste API keys into notebook cells.

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
