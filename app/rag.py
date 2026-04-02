import json
import asyncio
import time
from collections import deque
from typing import Deque, Dict, List

# Simple in‑memory store that keeps the last N records per category
MAX_RECORDS = 5

class RAGStore:
    def __init__(self):
        # category -> deque of (timestamp, data)
        self.store: Dict[str, Deque[tuple[float, dict]]] = {}
        self.lock = asyncio.Lock()

    async def add(self, category: str, data: dict):
        async with self.lock:
            if category not in self.store:
                self.store[category] = deque(maxlen=MAX_RECORDS)
            self.store[category].append((time.time(), data))

    async def get_recent(self, category: str) -> List[dict]:
        async with self.lock:
            if category not in self.store:
                return []
            return [item[1] for item in list(self.store[category])]

    async def build_context(self, prompt: str) -> str:
        """Create a simple concatenated context string.
        We pull recent records from all categories and dump them as JSON.
        """
        async with self.lock:
            parts: List[str] = []
            for cat, dq in self.store.items():
                recent = [json.dumps(rec) for _, rec in dq]
                if recent:
                    parts.append(f"Category {cat}: " + ", ".join(recent))
            context = "\n".join(parts)
            return context

# Global store instance used by the routers
rag_store = RAGStore()
