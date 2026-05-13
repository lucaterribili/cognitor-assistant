from agent.agent import Agent

_agent = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
        _agent.load_models()
        _agent.load_knowledge()
    return _agent
