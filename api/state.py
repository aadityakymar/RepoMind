"""
api/state.py — Shared mutable singletons: graph + checkpointer.

Using a module-level store avoids circular imports between server.py and routes.py.
server.py calls set_graph/set_checkpointer during lifespan startup;
routes.py calls get_graph to invoke the agent.
"""

from typing import Any, Optional

_graph: Optional[Any] = None
_checkpointer: Optional[Any] = None


def get_graph() -> Optional[Any]:
    return _graph


def set_graph(g: Any) -> None:
    global _graph
    _graph = g


def get_checkpointer() -> Optional[Any]:
    return _checkpointer


def set_checkpointer(c: Any) -> None:
    global _checkpointer
    _checkpointer = c
