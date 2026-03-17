"""Presentation helpers for Project 48: Multi-Armed Bandit Simulator."""

from typing import Any, Dict, List


def format_header() -> str:
    """Format session header banner."""
    return '=' * 70 + '\n' + '   MULTI-ARMED BANDIT SIMULATOR\n' + '=' * 70


def format_startup_guide(config: Dict[str, Any], profile: Dict[str, Any]) -> str:
    """Format startup configuration and historical profile."""
    recent = ', '.join(profile.get('recent_runs', [])) or 'None yet'
    lines = [
        '',
        'Configuration:',
        f"   Project type:           {config['project_type']}",
        f"   Strategy selected:      {config['strategy']}",
        f"   Strategy set:           {', '.join(config['enabled_strategies'])}",
        f"   Demo sequence:          {config['demo_sequence_label']}",
        f"   Horizon steps:          {config['horizon_steps']}",
        f"   Epsilon:                {config['epsilon']:.2f}",
        f"   UCB c-value:            {config['ucb_c']:.2f}",
        f"   Benchmark horizons:     {', '.join(str(value) for value in config['benchmark_horizons'])}",
        f"   Trials per horizon:     {config['benchmark_trials']}",
        f"   Runtime chart:          {config['include_runtime_plot']}",
        f"   Regret chart:           {config['include_regret_plot']}",
        f"   Max preview rows:       {config['max_preview_rows']}",
        f"   Random seed:            {config['random_seed']}",
        '',
        'Startup:',
        '   Data directory:         data/',
        '   Outputs directory:      data/outputs/',
        (
            f"   Arm library:            {profile['library_file']} "
            f"(loaded {profile['arms_available']} arms)"
        ),
        (
            f"   Run catalog:            {profile['catalog_file']} "
            f"(loaded {profile['runs_stored']} runs)"
        ),
        (
            f"   Bandit history:         {profile['history_file']} "
            f"(loaded {profile['history_entries']} entries)"
        ),
        f"   Recent runs:            {recent}",
        '',
        '---',
    ]
    return '\n'.join(lines)


def _clip_preview(value: str, width: int = 36) -> str:
    """Return a compact preview string for table output."""
    compact = value.replace('\n', ' ').strip()
    if len(compact) <= width:
        return compact
    return compact[: width - 3] + '...'


def format_step_table(steps: List[Dict[str, Any]]) -> str:
    """Format strategy step preview table."""
    if not steps:
        return 'No step previews generated.'
    lines = [
        'Step previews:',
        '   # | Strategy          | Arm                      | Reward   | CumReward | CumRegret',
        '   --+-------------------+--------------------------+----------+-----------+----------',
    ]
    for row in steps:
        lines.append(
            '   '
            f"{int(row.get('step_index', 0)):<2} | "
            f"{_clip_preview(str(row.get('strategy', '')), 17):<17} | "
            f"{_clip_preview(str(row.get('selected_arm', '')), 24):<24} | "
            f"{float(row.get('reward', 0.0)):<8.4f} | "
            f"{float(row.get('cumulative_reward', 0.0)):<9.3f} | "
            f"{float(row.get('cumulative_regret', 0.0)):<8.3f}"
        )
    return '\n'.join(lines)


def format_history_table(history: List[Dict[str, Any]], max_rows: int) -> str:
    """Format history preview table."""
    if not history:
        return 'No history entries available.'
    lines = [
        'Recent history:',
        '   Event type      | Preview                              | Created at',
        '   ----------------+--------------------------------------+----------------------',
    ]
    for row in history[-max_rows:]:
        payload = row.get('payload', {})
        if row.get('event_type') == 'bandit_session':
            preview = (
                f"{payload.get('strategy', '')} | "
                f"steps={payload.get('steps_completed', 0)} | "
                f"reward={payload.get('total_reward', 0)}"
            )
        elif row.get('event_type') == 'bandit_step':
            preview = f"{payload.get('step_index', 0)}. {payload.get('selected_arm', '')}"
        else:
            preview = str(payload)
        lines.append(
            '   '
            f"{_clip_preview(str(row.get('event_type', '')), 14):<14} | "
            f"{_clip_preview(preview, 36):<36} | "
            f"{_clip_preview(str(row.get('created_at', '')), 20):<20}"
        )
    return '\n'.join(lines)


def format_run_report(summary: Dict[str, Any]) -> str:
    """Format final session report."""
    artifacts = summary.get('artifacts', {})
    metrics = summary.get('metrics', {})
    selected_result = summary.get('selected_result', {})
    lines = [
        '',
        'Session complete:',
        f"   Session ID:             {summary['session_id']}",
        f"   Arms available:         {summary['arms_available']}",
        f"   Demo sequence:          {summary['demo_sequence_label']}",
        f"   Strategy selected:      {summary['strategy_selected']}",
        f"   Strategy runs:          {summary['strategy_runs']}",
        f"   Steps completed:        {summary['steps_completed']}",
        f"   Runtime points:         {summary['runtime_points']}",
        f"   Elapsed time:           {summary['elapsed_ms']:.2f} ms",
        '',
        (
            'Bandit metrics: '
            f"best_strategy={metrics.get('best_strategy', 'N/A')} | "
            f"best_avg_runtime_ms={metrics.get('best_avg_runtime_ms', 0.0):.6f} | "
            f"selected_total_reward={metrics.get('selected_total_reward', 0.0):.5f} | "
            f"selected_avg_reward={metrics.get('selected_avg_reward', 0.0):.5f} | "
            f"selected_cumulative_regret={metrics.get('selected_cumulative_regret', 0.0):.5f} | "
            f"history_size={metrics.get('history_size', 0)}"
        ),
        '',
        (
            'Selected strategy totals: '
            f"total_reward={float(selected_result.get('total_reward', 0.0)):.3f} | "
            f"avg_reward={float(selected_result.get('average_reward', 0.0)):.4f} | "
            f"cumulative_regret={float(selected_result.get('cumulative_regret', 0.0)):.3f}"
        ),
        '',
        format_step_table(summary.get('step_previews', [])),
        '',
        format_history_table(summary.get('history_previews', []), max_rows=summary.get('max_preview_rows', 8)),
        '',
        'Artifacts saved:',
        f"   Session record:         {artifacts.get('session_file', 'N/A')}",
        f"   Trace bundle:           {artifacts.get('trace_file', 'N/A')}",
        f"   Benchmark file:         {artifacts.get('benchmark_file', 'N/A')}",
        f"   Runtime chart:          {artifacts.get('runtime_chart_file', 'N/A')}",
        f"   Regret chart:           {artifacts.get('regret_chart_file', 'N/A')}",
    ]
    return '\n'.join(lines)


def format_message(message: str) -> str:
    """Format a user-facing message string."""
    return f'[Project 48] {message}'
