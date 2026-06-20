import json
import os
import uuid
from datetime import datetime, timezone

# --- CONFIGURATION ---
LOG_DIR = "./logs"


class ConversationLogger:
    """Persists each conversation turn to a JSON Lines file for post-mortem analysis.

    One file is created per session (named with a UTC timestamp + short id).
    Each line is a self-contained JSON record describing a single turn:
    the question, how it was routed, what was retrieved, the answer, latency,
    and any error. JSONL is append-only and trivially loadable with pandas
    (pd.read_json(path, lines=True)) or jq for later analysis.
    """

    def __init__(self, log_dir: str = LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = uuid.uuid4().hex[:8]
        started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = os.path.join(log_dir, f"session_{started}_{self.session_id}.jsonl")
        self.turn = 0

    def log_turn(self, question, result=None, latency_s=None, error=None):
        """Append a single turn to the session log.

        Args:
            question:  the raw user input.
            result:    the final state dict returned by the LangGraph app
                       (expects optional 'classification', 'documents', 'generation').
            latency_s: wall-clock seconds the turn took, if measured.
            error:     string description of any exception raised during the turn.
        """
        self.turn += 1
        result = result or {}

        record = {
            "session_id": self.session_id,
            "turn": self.turn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "classification": result.get("classification"),
            "retrieved_documents": result.get("documents", []),
            "num_retrieved": len(result.get("documents", []) or []),
            "generation": result.get("generation"),
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "error": error,
        }

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record
