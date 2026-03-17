import json
from pathlib import Path
from typing import Any, Dict, List

"""Persistence layer for Project 48: Multi-Armed Bandit Simulator."""

DATA_DIR = Path(__file__).resolve().parent / 'data'
OUTPUTS_DIR = DATA_DIR / 'outputs'
ARM_LIBRARY_FILE = DATA_DIR / 'arm_sets.json'
RUN_CATALOG_FILE = DATA_DIR / 'runs.json'
HISTORY_FILE = DATA_DIR / 'bandit_history.json'


def ensure_data_dirs() -> None:
    """Create local data and output directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_list_file(path: Path) -> List[Dict[str, Any]]:
    """Safely load a list-based JSON file."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _save_list_file(path: Path, data: List[Dict[str, Any]]) -> None:
    """Persist list-based JSON with indentation."""
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def load_arm_library() -> List[Dict[str, Any]]:
    """Load reusable bandit arms from disk."""
    ensure_data_dirs()
    return _load_list_file(ARM_LIBRARY_FILE)


def save_arm_library(library: List[Dict[str, Any]]) -> str:
    """Persist arm library to disk."""
    ensure_data_dirs()
    _save_list_file(ARM_LIBRARY_FILE, library)
    return str(ARM_LIBRARY_FILE)


def load_run_catalog() -> List[Dict[str, Any]]:
    """Load run catalog entries from disk."""
    ensure_data_dirs()
    return _load_list_file(RUN_CATALOG_FILE)


def save_run_catalog(catalog: List[Dict[str, Any]]) -> None:
    """Persist run catalog entries to disk."""
    ensure_data_dirs()
    _save_list_file(RUN_CATALOG_FILE, catalog)


def load_history() -> List[Dict[str, Any]]:
    """Load simulator history entries from disk."""
    ensure_data_dirs()
    return _load_list_file(HISTORY_FILE)


def save_history(history: List[Dict[str, Any]]) -> str:
    """Persist simulator history entries to disk."""
    ensure_data_dirs()
    _save_list_file(HISTORY_FILE, history)
    return str(HISTORY_FILE)


def save_trace_file(trace_payload: Dict[str, Any], session_id: str) -> str:
    """Persist simulation trace payload for a session."""
    ensure_data_dirs()
    file_path = OUTPUTS_DIR / f'trace_{session_id}.json'
    file_path.write_text(json.dumps(trace_payload, indent=2), encoding='utf-8')
    return str(file_path)


def save_benchmark_file(benchmark_payload: Dict[str, Any], session_id: str) -> str:
    """Persist benchmark payload for a session."""
    ensure_data_dirs()
    file_path = OUTPUTS_DIR / f'benchmark_{session_id}.json'
    file_path.write_text(json.dumps(benchmark_payload, indent=2), encoding='utf-8')
    return str(file_path)


def save_run_record(run_record: Dict[str, Any]) -> str:
    """Persist one run record and update the run catalog."""
    ensure_data_dirs()
    session_id = run_record['session_id']
    file_path = OUTPUTS_DIR / f'run_{session_id}.json'
    file_path.write_text(json.dumps(run_record, indent=2), encoding='utf-8')

    catalog = load_run_catalog()
    catalog.append(
        {
            'session_id': session_id,
            'demo_sequence_label': run_record.get('demo_sequence_label', ''),
            'strategy_selected': run_record.get('strategy_selected', ''),
            'strategy_runs': run_record.get('strategy_runs', 0),
            'steps_completed': run_record.get('steps_completed', 0),
            'runtime_points': run_record.get('runtime_points', 0),
            'elapsed_ms': run_record.get('elapsed_ms', 0.0),
            'best_strategy': run_record.get('metrics', {}).get('best_strategy', ''),
            'finished_at': run_record.get('finished_at', ''),
            'run_file': str(file_path),
        }
    )
    save_run_catalog(catalog)
    return str(file_path)


def load_json(filename: str):
    """Load JSON data from the local data directory."""
    ensure_data_dirs()
    path = DATA_DIR / filename
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return []


def save_json(filename: str, data) -> None:
    """Save JSON data to the local data directory."""
    ensure_data_dirs()
    path = DATA_DIR / filename
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
