"""
HEDGE → RALPH Convergence Orchestrator

Watches HEDGE thread state. When a thread's top next_action is:
- Implementable (code/file/config change, not "call X" or "wait for Y")
- High priority (score above threshold)
- Low enough entropy (action is clear)

...it generates a RALPH story spec and dispatches to ralph-loop for execution.
RALPH outcome feeds back as a signal into the HEDGE thread.
"""
