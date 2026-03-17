# Beginner Project 48: Multi-Armed Bandit Simulator

**Time:** 4-6 hours  
**Difficulty:** Intermediate Beginner  
**Focus:** Exploration-vs-exploitation strategy simulation, policy benchmarking, persistent run history, and runtime/regret visualizations

---

## Why This Project?

Multi-armed bandits are one of the most practical entry points to reinforcement learning and online optimization. Instead of training a heavy model, bandits let you learn from immediate feedback and balance exploration with exploitation over time. This project turns that concept into a reproducible simulator workflow where you can compare policies, inspect action traces, and track regret trends across repeated sessions.

This project teaches end-to-end bandit simulation workflow concepts where you can:

- load or auto-seed a reusable arm library for deterministic demos,
- run a fixed-horizon online decision process over multiple reward arms,
- compare epsilon-greedy, UCB, and Thompson sampling side by side,
- track cumulative reward and cumulative regret across step sequences,
- persist session and step-level history for reproducible audit trails,
- benchmark policy runtime across increasing horizon lengths,
- visualize runtime behavior as a multi-series chart,
- visualize regret behavior as a trend chart across decisions,
- export trace bundles and benchmark bundles to JSON for inspection,
- persist historical run summaries for repeatable profiling,
- and print a readable terminal report with step previews and artifact paths.

---

## More Projects

You can access this project and more in this separate repository:

[student-interview-prep](https://github.com/ShamShamsw/student-interview-prep.git)

---

## What You Will Build

You will build a multi-armed bandit simulation workflow that:

1. Loads arm definitions from `data/arm_sets.json` (or seeds a starter set of 6 arms).
2. Selects a deterministic demo sequence for policy walkthroughs.
3. Simulates reward pulls under epsilon-greedy, UCB, and Thompson sampling.
4. Compares policy quality using cumulative reward, average reward, and regret.
5. Stores session and sampled-step events in `data/bandit_history.json`.
6. Benchmarks policy runtime across configured horizon lengths and repeated trials.
7. Aggregates runtime points into strategy-level average performance summaries.
8. Saves a strategy runtime chart under `data/outputs/`.
9. Saves a cumulative regret trend chart under `data/outputs/`.
10. Persists trace, benchmark, and run-summary artifacts for future sessions.
11. Maintains a run catalog in `data/runs.json` for startup profiling.

---

## Requirements

- Python 3.11+
- `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Example Session

```text
======================================================================
   MULTI-ARMED BANDIT SIMULATOR
======================================================================

Configuration:
   Project type:           multi_armed_bandit_simulator
   Strategy selected:      epsilon_greedy
   Strategy set:           epsilon_greedy, ucb, thompson_sampling
   Demo sequence:          classroom_campaign
   Horizon steps:          1200
   Epsilon:                0.10
   UCB c-value:            2.00
   Benchmark horizons:     200, 500, 1000, 2000, 4000
   Trials per horizon:     4
   Runtime chart:          True
   Regret chart:           True
   Max preview rows:       8
   Random seed:            42

Startup:
   Data directory:         data/
   Outputs directory:      data/outputs/
   Arm library:            data/arm_sets.json (loaded 0 arms)
   Run catalog:            data/runs.json (loaded 0 runs)
   Bandit history:         data/bandit_history.json (loaded 0 entries)
   Recent runs:            None yet

---

Session complete:
   Session ID:             20260317_235921
   Arms available:         6
   Demo sequence:          classroom_campaign
   Strategy selected:      epsilon_greedy
   Strategy runs:          3
   Steps completed:        1200
   Runtime points:         60
   Elapsed time:           295.64 ms

Bandit metrics: best_strategy=epsilon_greedy | best_avg_runtime_ms=2.203111 | selected_total_reward=389.194000 | selected_avg_reward=0.324328 | selected_cumulative_regret=66.540000 | history_size=21

Selected strategy totals: total_reward=389.194 | avg_reward=0.3243 | cumulative_regret=66.540

Step previews:
   # | Strategy          | Arm                      | Reward   | CumReward | CumRegret
   --+-------------------+--------------------------+----------+-----------+----------
   1  | epsilon_greedy   | Referral Bonus           | 0.3277   | 0.328     | 0.000
   2  | epsilon_greedy   | Referral Bonus           | 0.4132   | 0.741     | 0.000
   3  | epsilon_greedy   | Landing Page Refresh     | 0.2901   | 1.031     | 0.040

Artifacts saved:
   Session record:         data/outputs/run_20260317_235921.json
   Trace bundle:           data/outputs/trace_20260317_235921.json
   Benchmark file:         data/outputs/benchmark_20260317_235921.json
   Runtime chart:          data/outputs/strategy_runtime_20260317_235921.png
   Regret chart:           data/outputs/regret_trend_20260317_235921.png
```

---

## Run

```bash
python main.py
```
