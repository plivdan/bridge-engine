"""Decision tracing for debugging AI bid and play choices.

When enabled via BridgeParams.trace_enabled, the bidding and card play
agents emit DecisionTrace entries explaining each decision. The TraceLog
accumulates entries for a single board and can produce human-readable
summaries or JSON-serializable dicts.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict


@dataclass
class DecisionTrace:
    """A single AI decision record."""
    action_type: str          # 'bid' or 'play'
    seat: int                 # 0=N, 1=E, 2=S, 3=W
    phase: str                # bidding phase or play phase name
    chosen: str               # string repr of chosen action
    reason: str               # why this action was chosen
    candidates: List[str] = field(default_factory=list)  # alternatives considered
    details: Dict[str, Any] = field(default_factory=dict)  # HCP, combined, plan, etc.

    def __str__(self):
        seat_names = ['N', 'E', 'S', 'W']
        s = f"[{seat_names[self.seat]}] {self.action_type.upper()} "
        s += f"phase={self.phase} -> {self.chosen}"
        if self.reason:
            s += f"  ({self.reason})"
        if self.details:
            detail_str = ', '.join(f"{k}={v}" for k, v in self.details.items())
            s += f"  [{detail_str}]"
        return s


class TraceLog:
    """Accumulates decision traces for a single board."""

    def __init__(self):
        self.entries: List[DecisionTrace] = []
        self.board_num: int = 0

    def add(self, entry: DecisionTrace):
        self.entries.append(entry)

    def clear(self):
        self.entries.clear()
        self.board_num = 0

    def summary(self) -> str:
        """Human-readable summary of all decisions."""
        if not self.entries:
            return "(no trace entries)"
        lines = [f"=== Board {self.board_num} Trace ({len(self.entries)} decisions) ==="]
        for e in self.entries:
            lines.append(f"  {e}")
        return '\n'.join(lines)

    def to_list(self) -> List[Dict]:
        """JSON-serializable list of all entries."""
        result = []
        for e in self.entries:
            result.append({
                'action_type': e.action_type,
                'seat': e.seat,
                'phase': e.phase,
                'chosen': e.chosen,
                'reason': e.reason,
                'candidates': e.candidates,
                'details': e.details,
            })
        return result

    @property
    def bid_entries(self) -> List[DecisionTrace]:
        return [e for e in self.entries if e.action_type == 'bid']

    @property
    def play_entries(self) -> List[DecisionTrace]:
        return [e for e in self.entries if e.action_type == 'play']
