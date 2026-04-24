from app.orchestration.state import GraphState


def test_defaults():
    state = GraphState(hypothesis="test hypothesis")
    assert state.hypothesis == "test hypothesis"
    assert state.messages == []
    assert state.plan == []
    assert state.retry_count == 0
    assert state.max_retries == 2
    assert state.pause_requested is False
    assert state.human_approval is False
    assert state.awaiting_approval is False
    assert state.confidence == 0.0
    assert state.failure_reason is None
    assert state.active_node == ""


def test_log_appends():
    state = GraphState(hypothesis="test")
    state.log("step one")
    state.log("step two")
    assert state.logs == ["step one", "step two"]


def test_state_from_dict():
    data = {"hypothesis": "from dict", "retry_count": 1, "max_retries": 3}
    state = GraphState(**data)
    assert state.hypothesis == "from dict"
    assert state.retry_count == 1
    assert state.max_retries == 3
