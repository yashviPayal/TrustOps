from app.api.rollout import RolloutManager

def test_champion_mode_never_selects_challenger():
    manager = RolloutManager(mode="champion", canary_percent=100)
    assert all(manager.choose_model() == "champion" for _ in range(20))

def test_canary_rolls_back_after_repeated_slow_candidate_predictions():
    manager = RolloutManager(
        mode="canary", canary_percent=100, max_latency_ms=10, latency_failures=3
    )
    manager.record_candidate_latency(20)
    manager.record_candidate_latency(20)
    assert not manager.rolled_back
    manager.record_candidate_latency(20)
    assert manager.rolled_back
    assert manager.choose_model() == "champion"

def test_shadow_mode_is_valid():
    manager = RolloutManager(mode="shadow")
    assert manager.status()["mode"] == "shadow"
