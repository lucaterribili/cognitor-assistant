import threading

from agent.agent import Agent

_agent = None
_agent_lock = threading.Lock()


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                agent = Agent()
                agent.load_models()
                agent.load_knowledge()
                _agent = agent
    return _agent
