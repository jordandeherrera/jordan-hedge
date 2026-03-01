"""
Ralph Worker Monitor — guardrails for running RALPH subprocesses.

Responsibilities:
1. Write a PID/state file when a Ralph worker starts
2. Heartbeat: poll the process and log liveness at interval
3. Detect stall: if no stdout progress for N seconds, mark as stalled
4. Kill: SIGTERM + SIGKILL escalation with timeout
5. Cleanup: remove PID file, .ralph/ working state, temp files
6. Status: read all active worker state files and report

State file location: ~/.jordan-hedge/workers/<thread_slug>.json
"""

import os
import json
import time
import signal
import shutil
import re
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

STATE_DIR = Path.home() / '.jordan-hedge' / 'workers'
HEARTBEAT_INTERVAL_S = 15       # check liveness every 15s
STALL_TIMEOUT_S = 120           # no output for 2 min → stalled
KILL_GRACE_S = 10               # SIGTERM → wait → SIGKILL


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def _state_path(thread_name: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"{_slug(thread_name)}.json"


@dataclass
class WorkerState:
    thread_name: str
    pid: int
    working_dir: str
    story_preview: str
    started_at: float
    last_heartbeat: float
    last_output_at: float
    status: str                 # running | stalled | complete | failed | killed
    exit_code: Optional[int] = None
    error: str = ''

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> Optional['WorkerState']:
        try:
            d = json.loads(path.read_text())
            return cls(**d)
        except Exception:
            return None

    def is_stalled(self) -> bool:
        return (
            self.status == 'running'
            and time.time() - self.last_output_at > STALL_TIMEOUT_S
        )

    def elapsed(self) -> float:
        return time.time() - self.started_at


class RalphMonitor:
    """
    Attach to a running ralph-loop subprocess and monitor it.
    Call .start() in a background thread; call .kill() from any thread.
    """

    def __init__(
        self,
        proc,                   # subprocess.Popen
        thread_name: str,
        working_dir: str,
        story: str,
        stall_timeout_s: int = STALL_TIMEOUT_S,
        on_stall=None,          # callback(state: WorkerState) — called once on stall
    ):
        self.proc = proc
        self.thread_name = thread_name
        self.working_dir = working_dir
        self.stall_timeout_s = stall_timeout_s
        self.on_stall = on_stall
        self._stop = threading.Event()
        self._stall_notified = False

        now = time.time()
        self.state = WorkerState(
            thread_name=thread_name,
            pid=proc.pid,
            working_dir=working_dir,
            story_preview=story[:200],
            started_at=now,
            last_heartbeat=now,
            last_output_at=now,
            status='running',
        )
        self._state_path = _state_path(thread_name)
        self.state.save(self._state_path)

    def touch_output(self):
        """Call whenever output is detected from the subprocess."""
        self.state.last_output_at = time.time()

    def start(self):
        """Run the heartbeat loop in the current thread (call from a daemon thread)."""
        while not self._stop.is_set():
            time.sleep(HEARTBEAT_INTERVAL_S)

            if self.proc.poll() is not None:
                # Process finished
                self.state.exit_code = self.proc.returncode
                self.state.status = 'complete' if self.proc.returncode == 0 else 'failed'
                self.state.last_heartbeat = time.time()
                self.state.save(self._state_path)
                break

            self.state.last_heartbeat = time.time()

            if self.state.is_stalled() and not self._stall_notified:
                self._stall_notified = True
                self.state.status = 'stalled'
                self.state.save(self._state_path)
                print(f"\n[MONITOR] ⚠️  Ralph worker STALLED on '{self.thread_name}' "
                      f"(no output for {self.stall_timeout_s}s, PID {self.proc.pid})")
                print(f"[MONITOR] Run: python3 -m src.cli.orchestrate --kill '{self.thread_name}'")
                if self.on_stall:
                    try:
                        self.on_stall(self.state)
                    except Exception as e:
                        print(f"[MONITOR] on_stall callback error: {e}")
            else:
                self.state.save(self._state_path)

    def stop(self):
        self._stop.set()

    def kill(self, reason: str = 'manual') -> bool:
        """
        Escalating kill: SIGTERM → wait KILL_GRACE_S → SIGKILL.
        Returns True if the process is dead.
        """
        if self.proc.poll() is not None:
            print(f"[MONITOR] Process {self.proc.pid} already exited.")
            self._finalize('killed', -1)
            return True

        print(f"[MONITOR] Sending SIGTERM to PID {self.proc.pid} (reason: {reason})")
        try:
            self.proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass

        deadline = time.time() + KILL_GRACE_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                print(f"[MONITOR] PID {self.proc.pid} exited after SIGTERM.")
                self._finalize('killed', self.proc.returncode)
                return True
            time.sleep(0.5)

        # Escalate to SIGKILL
        print(f"[MONITOR] SIGTERM ignored — sending SIGKILL to PID {self.proc.pid}")
        try:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait(timeout=5)
        except (ProcessLookupError, Exception):
            pass

        self._finalize('killed', -9)
        return True

    def cleanup(self, remove_ralph_dir: bool = True):
        """
        Remove worker state file and optionally .ralph/ directory in working_dir.
        """
        # Remove state file
        if self._state_path.exists():
            self._state_path.unlink()
            print(f"[MONITOR] Removed state file: {self._state_path}")

        # Remove .ralph/ working state
        if remove_ralph_dir:
            ralph_dir = Path(self.working_dir) / '.ralph'
            if ralph_dir.exists():
                shutil.rmtree(ralph_dir)
                print(f"[MONITOR] Removed Ralph sandbox: {ralph_dir}")

    def _finalize(self, status: str, exit_code: int):
        self.state.status = status
        self.state.exit_code = exit_code
        self.state.last_heartbeat = time.time()
        self.state.save(self._state_path)
        self.stop()


# ---------------------------------------------------------------------------
# Standalone functions for CLI use (no monitor instance required)
# ---------------------------------------------------------------------------

def list_workers() -> list[WorkerState]:
    """Return all worker state files."""
    if not STATE_DIR.exists():
        return []
    workers = []
    for f in STATE_DIR.glob('*.json'):
        state = WorkerState.load(f)
        if state:
            # Refresh: check if PID is still alive
            if state.status == 'running':
                try:
                    os.kill(state.pid, 0)  # signal 0 = existence check
                except (ProcessLookupError, PermissionError):
                    state.status = 'dead'
                    state.save(f)
            workers.append(state)
    return workers


def kill_worker(thread_name: str, remove_ralph_dir: bool = True) -> bool:
    """Kill a worker by thread name. Returns True if killed or already dead."""
    path = _state_path(thread_name)
    if not path.exists():
        # Try fuzzy match
        slug = _slug(thread_name)
        matches = list(STATE_DIR.glob(f'*{slug}*.json')) if STATE_DIR.exists() else []
        if not matches:
            print(f"[MONITOR] No worker state found for '{thread_name}'")
            return False
        path = matches[0]

    state = WorkerState.load(path)
    if not state:
        print(f"[MONITOR] Could not read state from {path}")
        return False

    # Try to kill by PID
    killed = False
    try:
        os.kill(state.pid, 0)  # check alive
        print(f"[MONITOR] Killing PID {state.pid} ({state.thread_name})")
        os.kill(state.pid, signal.SIGTERM)
        time.sleep(KILL_GRACE_S)
        try:
            os.kill(state.pid, 0)
            os.kill(state.pid, signal.SIGKILL)
            print(f"[MONITOR] SIGKILL sent to PID {state.pid}")
        except ProcessLookupError:
            pass
        killed = True
    except ProcessLookupError:
        print(f"[MONITOR] PID {state.pid} already dead.")
        killed = True
    except PermissionError:
        print(f"[MONITOR] Permission denied killing PID {state.pid}")

    # Cleanup
    path.unlink(missing_ok=True)
    print(f"[MONITOR] Removed state file: {path}")

    if remove_ralph_dir:
        ralph_dir = Path(state.working_dir) / '.ralph'
        if ralph_dir.exists():
            shutil.rmtree(ralph_dir)
            print(f"[MONITOR] Removed Ralph sandbox: {ralph_dir}")

    return killed


def kill_all_workers(remove_ralph_dirs: bool = True) -> int:
    """Kill all running workers. Returns count killed."""
    workers = list_workers()
    count = 0
    for w in workers:
        if w.status in ('running', 'stalled', 'dead'):
            if kill_worker(w.thread_name, remove_ralph_dirs):
                count += 1
    return count


def print_worker_status():
    """Print a human-readable status table of all workers."""
    workers = list_workers()
    if not workers:
        print("\n  No active Ralph workers.\n")
        return

    print(f"\n{'='*70}")
    print(f"  RALPH WORKERS")
    print(f"{'='*70}")
    for w in workers:
        elapsed = int(w.elapsed())
        stall_age = int(time.time() - w.last_output_at)
        icon = {
            'running': '🟢',
            'stalled': '🟡',
            'complete': '✅',
            'failed': '❌',
            'killed': '🔴',
            'dead': '💀',
        }.get(w.status, '❓')

        print(f"\n  {icon} [{w.status.upper()}] {w.thread_name}")
        print(f"     PID: {w.pid}  |  Elapsed: {elapsed}s  |  No-output: {stall_age}s")
        print(f"     Dir: {w.working_dir}")
        print(f"     Story: {w.story_preview[:80]}...")
    print(f"\n{'='*70}\n")
