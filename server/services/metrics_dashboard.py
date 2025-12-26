from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class TurnMetrics:
    turn_id: int
    mode: str
    end_ts: float  # perf_counter timestamp
    stt_done_ms: Optional[float] = None
    llm_first_ms: Optional[float] = None
    client_audio_start_ms: Optional[float] = None
    total_turn_ms: Optional[float] = None
    audio_chunks: int = 0
    chunk_sizes: List[int] = field(default_factory=list)


class MetricsStore:
    def __init__(self, max_turns: int = 25):
        self.max_turns = max_turns
        self._lock = threading.Lock()
        self._turns: Deque[TurnMetrics] = deque(maxlen=max_turns)
        self._inflight: Dict[int, TurnMetrics] = {}

    def start_turn(self, turn_id: int, mode: str, end_ts: float) -> None:
        with self._lock:
            tm = TurnMetrics(turn_id=turn_id, mode=mode, end_ts=end_ts)
            self._inflight[turn_id] = tm

    def set_stt_done(self, turn_id: int, ms: float) -> None:
        with self._lock:
            tm = self._inflight.get(turn_id)
            if tm:
                tm.stt_done_ms = ms

    def set_llm_first(self, turn_id: int, ms: float) -> None:
        with self._lock:
            tm = self._inflight.get(turn_id)
            if tm and tm.llm_first_ms is None:
                tm.llm_first_ms = ms

    def set_client_audio_start(self, turn_id: int, ms: float) -> None:
        with self._lock:
            tm = self._inflight.get(turn_id)
            if tm:
                # Record only the first time the client starts playback for this turn.
                if tm.client_audio_start_ms is None:
                    tm.client_audio_start_ms = max(0.0, float(ms))

    def add_audio_chunk(self, turn_id: int, size_bytes: int) -> None:
        with self._lock:
            tm = self._inflight.get(turn_id)
            if tm:
                tm.audio_chunks += 1
                tm.chunk_sizes.append(int(size_bytes))

    def finish_turn(self, turn_id: int, total_turn_ms: float) -> Optional[TurnMetrics]:
        with self._lock:
            tm = self._inflight.pop(turn_id, None)
            if not tm:
                return None
            tm.total_turn_ms = total_turn_ms
            self._turns.append(tm)
            return tm

    def snapshot(self) -> List[TurnMetrics]:
        with self._lock:
            turns = list(self._turns)
            inflight = list(self._inflight.values())
        # show inflight first (not finished) then recent finished
        return inflight + turns


_STORE: Optional[MetricsStore] = None


def get_store() -> MetricsStore:
    global _STORE
    if _STORE is None:
        _STORE = MetricsStore(max_turns=int(os.getenv("METRICS_MAX_TURNS", "25")))
    return _STORE


def _render_rich_table(turns: List[TurnMetrics]):
    from rich.table import Table

    t = Table(title="Voice Assistant Metrics (live)")
    t.add_column("turn", justify="right")
    t.add_column("mode")
    t.add_column("end→stt", justify="right")
    t.add_column("end→llm1", justify="right")
    t.add_column("end→client_audio", justify="right")
    t.add_column("turn_total", justify="right")
    t.add_column("chunks", justify="right")
    t.add_column("chunk_bytes", overflow="fold")

    def fmt(v: Optional[float]) -> str:
        return "-" if v is None else f"{v:.0f}ms"

    for tm in turns[:25]:
        sizes = ",".join(str(x) for x in tm.chunk_sizes[-12:])
        if len(tm.chunk_sizes) > 12:
            sizes = "…," + sizes
        t.add_row(
            str(tm.turn_id),
            tm.mode,
            fmt(tm.stt_done_ms),
            fmt(tm.llm_first_ms),
            fmt(tm.client_audio_start_ms),
            fmt(tm.total_turn_ms),
            str(tm.audio_chunks),
            sizes or "-",
        )
    return t


def start_dashboard() -> None:
    """
    Starts a live-updating terminal UI using Rich.
    Enabled by setting METRICS_DASHBOARD=1.
    """
    if os.getenv("METRICS_DASHBOARD", "0") != "1":
        return

    # Avoid starting multiple dashboards in the same process.
    if getattr(start_dashboard, "_started", False):
        return
    setattr(start_dashboard, "_started", True)

    def run():
        try:
            from rich.console import Console
            from rich.live import Live
        except Exception as e:
            print(f"Metrics dashboard disabled (install 'rich'): {e}")
            return

        console = Console()
        store = get_store()
        with Live(_render_rich_table(store.snapshot()), console=console, refresh_per_second=4, transient=False) as live:
            while True:
                time.sleep(0.25)
                live.update(_render_rich_table(store.snapshot()))

    th = threading.Thread(target=run, daemon=True, name="metrics_dashboard")
    th.start()


