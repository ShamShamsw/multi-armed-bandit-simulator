"""Business logic for Project 48: Multi-Armed Bandit Simulator."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from models import (
    create_arm_case,
    create_bandit_config,
    create_history_entry,
    create_runtime_point,
    create_session_summary,
    create_step_record,
    create_strategy_result,
)
from storage import (
    OUTPUTS_DIR,
    ensure_data_dirs,
    load_arm_library,
    load_history,
    load_run_catalog,
    save_arm_library,
    save_benchmark_file,
    save_history,
    save_run_record,
    save_trace_file,
)


PLOT_COLORS = {
    'epsilon_greedy': '#264653',
    'ucb': '#2a9d8f',
    'thompson_sampling': '#e9c46a',
    'selected_plan': '#e76f51',
}


def _session_id() -> str:
    """Build a compact session ID from UTC timestamp."""
    return datetime.utcnow().strftime('%Y%m%d_%H%M%S')


def _default_arm_library() -> List[Dict[str, Any]]:
    """Return deterministic starter arms used on first run."""
    return [
        create_arm_case(
            arm_id='arm_001',
            label='classroom_campaign',
            name='Email Reminder',
            true_mean=0.18,
            reward_std=0.08,
            description='Low-conversion but cheap baseline arm.',
            tags=['baseline', 'email'],
        ),
        create_arm_case(
            arm_id='arm_002',
            label='classroom_campaign',
            name='Push Notification',
            true_mean=0.24,
            reward_std=0.10,
            description='Fast user reach with moderate variance.',
            tags=['mobile', 'push'],
        ),
        create_arm_case(
            arm_id='arm_003',
            label='classroom_campaign',
            name='Short Video Teaser',
            true_mean=0.29,
            reward_std=0.12,
            description='Higher upside with content quality swings.',
            tags=['video', 'creative'],
        ),
        create_arm_case(
            arm_id='arm_004',
            label='classroom_campaign',
            name='Referral Bonus',
            true_mean=0.35,
            reward_std=0.09,
            description='Best expected conversion with stable behavior.',
            tags=['referral', 'growth'],
        ),
        create_arm_case(
            arm_id='arm_005',
            label='classroom_campaign',
            name='Coupon Offer',
            true_mean=0.27,
            reward_std=0.11,
            description='Good conversion with occasional noisy spikes.',
            tags=['promo', 'discount'],
        ),
        create_arm_case(
            arm_id='arm_006',
            label='classroom_campaign',
            name='Landing Page Refresh',
            true_mean=0.31,
            reward_std=0.10,
            description='Strong performer after optimization updates.',
            tags=['web', 'ux'],
        ),
    ]


def _sample_reward(arm: Dict[str, Any], rng: random.Random) -> float:
    """Draw one clipped stochastic reward for an arm."""
    reward = rng.gauss(float(arm['true_mean']), float(arm['reward_std']))
    return min(1.0, max(0.0, reward))


def _select_epsilon_greedy(estimates: List[float], counts: List[int], epsilon: float, rng: random.Random) -> int:
    """Select an arm index with epsilon-greedy policy."""
    if rng.random() < epsilon or all(count == 0 for count in counts):
        return rng.randrange(len(estimates))
    best_estimate = max(estimates)
    best_indexes = [index for index, value in enumerate(estimates) if value == best_estimate]
    return rng.choice(best_indexes)


def _select_ucb(estimates: List[float], counts: List[int], step_index: int, c_value: float) -> int:
    """Select an arm index with upper-confidence-bound policy."""
    for index, count in enumerate(counts):
        if count == 0:
            return index

    bonus_values = []
    step = max(1, step_index)
    for index, estimate in enumerate(estimates):
        bonus = c_value * math.sqrt(math.log(step + 1) / max(1, counts[index]))
        bonus_values.append(estimate + bonus)
    best_score = max(bonus_values)
    return bonus_values.index(best_score)


def _beta_sample(alpha: int, beta: int, rng: random.Random) -> float:
    """Sample from Beta(alpha, beta) using gamma variates."""
    left = rng.gammavariate(max(1, alpha), 1.0)
    right = rng.gammavariate(max(1, beta), 1.0)
    total = left + right
    return 0.5 if total <= 0.0 else left / total


def _select_thompson(alpha_params: List[int], beta_params: List[int], rng: random.Random) -> int:
    """Select an arm index with Thompson sampling policy."""
    samples = [
        _beta_sample(alpha_params[index], beta_params[index], rng)
        for index in range(len(alpha_params))
    ]
    best_sample = max(samples)
    return samples.index(best_sample)


def _run_strategy(
    strategy: str,
    arms: List[Dict[str, Any]],
    horizon_steps: int,
    config: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    """Run one strategy over a fixed horizon and return full step trace."""
    arm_count = len(arms)
    estimates = [0.0] * arm_count
    counts = [0] * arm_count
    alpha_params = [1] * arm_count
    beta_params = [1] * arm_count
    arm_names = [str(arm['name']) for arm in arms]

    best_mean = max(float(arm['true_mean']) for arm in arms) if arms else 0.0
    best_index = next(
        (index for index, arm in enumerate(arms) if float(arm['true_mean']) == best_mean),
        0,
    )

    steps: List[Dict[str, Any]] = []
    cumulative_reward = 0.0
    cumulative_regret = 0.0

    for step in range(1, max(1, horizon_steps) + 1):
        if strategy == 'epsilon_greedy':
            selected_index = _select_epsilon_greedy(estimates, counts, config['epsilon'], rng)
        elif strategy == 'ucb':
            selected_index = _select_ucb(estimates, counts, step, config['ucb_c'])
        else:
            selected_index = _select_thompson(alpha_params, beta_params, rng)

        selected_arm = arms[selected_index]
        reward = _sample_reward(selected_arm, rng)
        counts[selected_index] += 1

        previous = estimates[selected_index]
        estimates[selected_index] = previous + ((reward - previous) / counts[selected_index])

        if strategy == 'thompson_sampling':
            outcome = 1 if reward >= 0.5 else 0
            alpha_params[selected_index] += outcome
            beta_params[selected_index] += 1 - outcome

        cumulative_reward += reward
        instant_regret = best_mean - float(selected_arm['true_mean'])
        cumulative_regret += instant_regret

        steps.append(
            create_step_record(
                step_index=step,
                strategy=strategy,
                selected_arm=arm_names[selected_index],
                reward=reward,
                cumulative_reward=cumulative_reward,
                instant_regret=instant_regret,
                cumulative_regret=cumulative_regret,
            )
        )

    total_reward = cumulative_reward
    average_reward = total_reward / max(1, horizon_steps)
    best_arm_share_pct = (counts[best_index] / max(1, horizon_steps)) * 100.0

    return create_strategy_result(
        strategy=strategy,
        total_reward=total_reward,
        average_reward=average_reward,
        cumulative_regret=cumulative_regret,
        best_arm_share_pct=best_arm_share_pct,
        steps=steps,
        feasible=bool(steps),
    )


def _compare_strategies(arms: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run configured bandit strategies for side-by-side comparison."""
    results: List[Dict[str, Any]] = []
    base_seed = int(config['random_seed'])
    for offset, strategy in enumerate(config['enabled_strategies'], start=1):
        strategy_rng = random.Random(base_seed + (offset * 97))
        results.append(
            _run_strategy(
                strategy=strategy,
                arms=arms,
                horizon_steps=config['horizon_steps'],
                config=config,
                rng=strategy_rng,
            )
        )
    return results


def _benchmark_runtime(config: Dict[str, Any], arms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Benchmark strategy runtime across increasing horizon lengths."""
    points: List[Dict[str, Any]] = []
    base_seed = int(config['random_seed'])

    for horizon in config['benchmark_horizons']:
        for strategy_index, strategy in enumerate(config['enabled_strategies'], start=1):
            for trial in range(1, config['benchmark_trials'] + 1):
                trial_seed = base_seed + (horizon * 13) + (strategy_index * 101) + trial
                trial_rng = random.Random(trial_seed)
                started = time.perf_counter()
                _run_strategy(strategy, arms, horizon, config, trial_rng)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                points.append(create_runtime_point(strategy, float(horizon), trial, elapsed_ms))
    return points


def _aggregate_runtime(points: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, float]]]:
    """Average runtime benchmark points by strategy and horizon size."""
    grouped: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for point in points:
        grouped[point['series']][point['size']].append(point['elapsed_ms'])

    summary: Dict[str, List[Dict[str, float]]] = {}
    for strategy, buckets in grouped.items():
        rows: List[Dict[str, float]] = []
        for size in sorted(buckets):
            rows.append({'size': size, 'elapsed_ms': round(mean(buckets[size]), 6)})
        summary[strategy] = rows
    return summary


def _save_runtime_chart(runtime_summary: Dict[str, List[Dict[str, float]]], session_id: str) -> str:
    """Persist benchmark runtime chart."""
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    for series, rows in runtime_summary.items():
        axis.plot(
            [row['size'] for row in rows],
            [row['elapsed_ms'] for row in rows],
            marker='o',
            linewidth=2,
            label=series.replace('_', ' ').title(),
            color=PLOT_COLORS.get(series, '#6c757d'),
        )
    axis.set_title('Average Runtime By Horizon Length And Strategy')
    axis.set_xlabel('Horizon steps')
    axis.set_ylabel('Elapsed time (ms)')
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    file_path = OUTPUTS_DIR / f'strategy_runtime_{session_id}.png'
    figure.savefig(file_path, dpi=150)
    plt.close(figure)
    return str(file_path)


def _save_regret_chart(selected_steps: List[Dict[str, Any]], session_id: str) -> str:
    """Persist cumulative regret trend chart for selected strategy."""
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    if selected_steps:
        axis.plot(
            [int(row['step_index']) for row in selected_steps],
            [float(row['cumulative_regret']) for row in selected_steps],
            marker='o',
            markersize=2.2,
            linewidth=1.8,
            color=PLOT_COLORS['selected_plan'],
        )
    axis.set_title('Cumulative Regret Trend Across Demo Horizon')
    axis.set_xlabel('Step index')
    axis.set_ylabel('Cumulative regret')
    axis.grid(alpha=0.25)
    figure.tight_layout()
    file_path = OUTPUTS_DIR / f'regret_trend_{session_id}.png'
    figure.savefig(file_path, dpi=150)
    plt.close(figure)
    return str(file_path)


def load_bandit_profile() -> Dict[str, Any]:
    """Return startup profile built from previously saved catalogs."""
    ensure_data_dirs()
    run_catalog = load_run_catalog()
    library = load_arm_library()
    history = load_history()
    recent_runs = [
        f"{item.get('session_id', '')}:{item.get('best_strategy', '')}:{item.get('demo_sequence_label', '')}"
        for item in run_catalog[-5:]
    ]
    return {
        'catalog_file': 'data/runs.json',
        'library_file': 'data/arm_sets.json',
        'history_file': 'data/bandit_history.json',
        'runs_stored': len(run_catalog),
        'arms_available': len(library),
        'history_entries': len(history),
        'recent_runs': recent_runs,
    }


def run_core_flow() -> Dict[str, Any]:
    """Run one complete bandit simulation, benchmark, and plotting session."""
    ensure_data_dirs()
    config = create_bandit_config()
    session_id = _session_id()
    started = time.perf_counter()

    arm_library = load_arm_library()
    if not arm_library:
        arm_library = _default_arm_library()
        save_arm_library(arm_library)

    demo_arms = [item for item in arm_library if item['label'] == config['demo_sequence_label']]
    if not demo_arms:
        demo_arms = arm_library[:]

    comparison = _compare_strategies(demo_arms, config)
    selected_result = next((item for item in comparison if item['strategy'] == config['strategy']), None)
    if selected_result is None and comparison:
        selected_result = max(comparison, key=lambda item: float(item['total_reward']))
    selected_result = selected_result or create_strategy_result('none', 0.0, 0.0, 0.0, 0.0, [], False)

    history = load_history()
    history.append(
        create_history_entry(
            'bandit_session',
            {
                'session_id': session_id,
                'strategy': selected_result['strategy'],
                'steps_completed': len(selected_result['steps']),
                'total_reward': selected_result['total_reward'],
                'cumulative_regret': selected_result['cumulative_regret'],
            },
        )
    )
    for step in selected_result['steps'][:: max(1, config['horizon_steps'] // 20)]:
        history.append(create_history_entry('bandit_step', step))
    history = history[-500:]
    history_file = save_history(history)

    runtime_points = _benchmark_runtime(config, demo_arms)
    runtime_summary = _aggregate_runtime(runtime_points)

    trace_payload = {
        'session_id': session_id,
        'config': config,
        'arms': demo_arms,
        'strategy_comparison': comparison,
        'selected_result': selected_result,
        'history_file': history_file,
    }
    benchmark_payload = {
        'session_id': session_id,
        'runtime_points': runtime_points,
        'runtime_summary': runtime_summary,
    }

    runtime_chart_file = ''
    regret_chart_file = ''
    if config['include_runtime_plot']:
        runtime_chart_file = _save_runtime_chart(runtime_summary, session_id)
    if config['include_regret_plot']:
        regret_chart_file = _save_regret_chart(selected_result['steps'], session_id)

    trace_file = save_trace_file(trace_payload, session_id)
    benchmark_file = save_benchmark_file(benchmark_payload, session_id)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    average_by_series = {
        series: mean(row['elapsed_ms'] for row in rows)
        for series, rows in runtime_summary.items()
        if rows
    }
    best_strategy = min(average_by_series.items(), key=lambda item: item[1])[0] if average_by_series else ''

    metrics = {
        'best_strategy': best_strategy,
        'best_avg_runtime_ms': round(average_by_series.get(best_strategy, 0.0), 6),
        'selected_total_reward': round(float(selected_result['total_reward']), 6),
        'selected_avg_reward': round(float(selected_result['average_reward']), 6),
        'selected_cumulative_regret': round(float(selected_result['cumulative_regret']), 6),
        'history_size': len(history),
    }

    artifacts = {
        'trace_file': trace_file,
        'benchmark_file': benchmark_file,
        'runtime_chart_file': runtime_chart_file,
        'regret_chart_file': regret_chart_file,
        'history_file': history_file,
    }

    step_previews = selected_result['steps'][: config['max_preview_rows']]
    summary = create_session_summary(
        session_id=session_id,
        arms_available=len(arm_library),
        demo_sequence_label=config['demo_sequence_label'],
        strategy_selected=selected_result['strategy'],
        strategy_runs=len(comparison),
        steps_completed=len(selected_result['steps']),
        runtime_points=len(runtime_points),
        elapsed_ms=elapsed_ms,
        artifacts=artifacts,
        step_previews=step_previews,
        metrics=metrics,
    )
    summary['history_previews'] = history[-config['max_preview_rows'] :]
    summary['max_preview_rows'] = config['max_preview_rows']
    summary['selected_result'] = selected_result

    run_record = dict(summary)
    session_file = save_run_record(run_record)
    summary['artifacts']['session_file'] = session_file
    return summary
