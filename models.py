"""Data models for Project 48: Multi-Armed Bandit Simulator."""

from datetime import datetime
from typing import Any, Dict, List


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.utcnow().isoformat(timespec='seconds') + 'Z'


def create_bandit_config(
    strategy: str = 'epsilon_greedy',
    enabled_strategies: List[str] | None = None,
    demo_sequence_label: str = 'classroom_campaign',
    horizon_steps: int = 1200,
    benchmark_horizons: List[int] | None = None,
    benchmark_trials: int = 4,
    include_runtime_plot: bool = True,
    include_regret_plot: bool = True,
    max_preview_rows: int = 8,
    random_seed: int = 42,
    epsilon: float = 0.1,
    ucb_c: float = 2.0,
) -> Dict[str, Any]:
    """Create a validated configuration record for one simulator session."""
    strategies = (
        enabled_strategies
        if enabled_strategies
        else ['epsilon_greedy', 'ucb', 'thompson_sampling']
    )
    benchmark_sizes = benchmark_horizons if benchmark_horizons else [200, 500, 1000, 2000, 4000]
    normalized_strategy = str(strategy).strip().lower()
    selected_strategy = normalized_strategy if normalized_strategy in strategies else 'epsilon_greedy'
    return {
        'project_type': 'multi_armed_bandit_simulator',
        'strategy': selected_strategy,
        'enabled_strategies': [str(value) for value in strategies],
        'demo_sequence_label': str(demo_sequence_label),
        'horizon_steps': max(50, int(horizon_steps)),
        'benchmark_horizons': [max(50, int(value)) for value in benchmark_sizes],
        'benchmark_trials': max(1, int(benchmark_trials)),
        'include_runtime_plot': bool(include_runtime_plot),
        'include_regret_plot': bool(include_regret_plot),
        'max_preview_rows': max(1, int(max_preview_rows)),
        'random_seed': int(random_seed),
        'epsilon': min(0.95, max(0.01, float(epsilon))),
        'ucb_c': max(0.1, float(ucb_c)),
        'created_at': _utc_timestamp(),
    }


def create_arm_case(
    arm_id: str,
    label: str,
    name: str,
    true_mean: float,
    reward_std: float,
    description: str,
    tags: List[str],
) -> Dict[str, Any]:
    """Create one reusable bandit arm record."""
    return {
        'arm_id': str(arm_id),
        'label': str(label),
        'name': str(name),
        'true_mean': round(float(true_mean), 6),
        'reward_std': max(0.001, round(float(reward_std), 6)),
        'description': str(description),
        'tags': [str(tag) for tag in tags],
    }


def create_step_record(
    step_index: int,
    strategy: str,
    selected_arm: str,
    reward: float,
    cumulative_reward: float,
    instant_regret: float,
    cumulative_regret: float,
) -> Dict[str, Any]:
    """Create one per-step simulation output record."""
    return {
        'step_index': int(step_index),
        'strategy': str(strategy),
        'selected_arm': str(selected_arm),
        'reward': round(float(reward), 6),
        'cumulative_reward': round(float(cumulative_reward), 6),
        'instant_regret': round(float(instant_regret), 6),
        'cumulative_regret': round(float(cumulative_regret), 6),
    }


def create_strategy_result(
    strategy: str,
    total_reward: float,
    average_reward: float,
    cumulative_regret: float,
    best_arm_share_pct: float,
    steps: List[Dict[str, Any]],
    feasible: bool,
) -> Dict[str, Any]:
    """Create one strategy result record."""
    return {
        'strategy': str(strategy),
        'total_reward': round(float(total_reward), 6),
        'average_reward': round(float(average_reward), 6),
        'cumulative_regret': round(float(cumulative_regret), 6),
        'best_arm_share_pct': round(float(best_arm_share_pct), 6),
        'steps': list(steps),
        'feasible': bool(feasible),
    }


def create_runtime_point(series: str, size: float, trial: int, elapsed_ms: float) -> Dict[str, Any]:
    """Create one benchmark runtime point."""
    return {
        'series': str(series),
        'size': float(size),
        'trial': int(trial),
        'elapsed_ms': round(float(elapsed_ms), 6),
    }


def create_history_entry(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create one persistent history entry."""
    return {
        'event_type': str(event_type),
        'payload': dict(payload),
        'created_at': _utc_timestamp(),
    }


def create_session_summary(
    session_id: str,
    arms_available: int,
    demo_sequence_label: str,
    strategy_selected: str,
    strategy_runs: int,
    steps_completed: int,
    runtime_points: int,
    elapsed_ms: float,
    artifacts: Dict[str, Any],
    step_previews: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Create final session summary for reporting and persistence."""
    return {
        'session_id': str(session_id),
        'arms_available': int(arms_available),
        'demo_sequence_label': str(demo_sequence_label),
        'strategy_selected': str(strategy_selected),
        'strategy_runs': int(strategy_runs),
        'steps_completed': int(steps_completed),
        'runtime_points': int(runtime_points),
        'elapsed_ms': round(float(elapsed_ms), 5),
        'artifacts': dict(artifacts),
        'step_previews': list(step_previews),
        'metrics': dict(metrics),
        'finished_at': _utc_timestamp(),
    }


def create_record(**kwargs):
    """Backwards-compatible generic record factory."""
    return dict(kwargs)
