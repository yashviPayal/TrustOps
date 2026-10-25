from app.api.monitoring import DelayedPerformanceMonitor

def test_delayed_monitor_collecting_state():
    monitor = DelayedPerformanceMonitor(window_size=50, f1_alert_threshold=0.60)
    monitor.record_prediction("tx-1", 0.8, "review", 0.5, "champion")
    report = monitor.report()
    assert report["status"] == "collecting_labels"
    assert report["resolved_samples"] == 0
    assert report["pending_labels"] == 1

def test_delayed_monitor_calculates_performance_and_confusion_matrix():
    monitor = DelayedPerformanceMonitor(window_size=50, f1_alert_threshold=0.60)
    # Feed 20 resolved transactions
    # 10 true negatives (pred allow, true 0)
    for i in range(10):
        tx_id = f"tx-tn-{i}"
        monitor.record_prediction(tx_id, 0.1, "allow", 0.5, "champion")
        monitor.record_label(tx_id, 0)

    # 5 true positives (pred review, true 1)
    for i in range(5):
        tx_id = f"tx-tp-{i}"
        monitor.record_prediction(tx_id, 0.9, "review", 0.5, "champion")
        monitor.record_label(tx_id, 1)

    # 2 false positives (pred review, true 0)
    for i in range(2):
        tx_id = f"tx-fp-{i}"
        monitor.record_prediction(tx_id, 0.7, "review", 0.5, "champion")
        monitor.record_label(tx_id, 0)

    # 1 false negative (pred allow, true 1)
    monitor.record_prediction("tx-fn-1", 0.2, "allow", 0.5, "champion")
    monitor.record_label("tx-fn-1", 1)

    report = monitor.report()
    assert report["status"] == "healthy"
    assert report["resolved_samples"] == 18
    metrics = report["metrics"]
    assert metrics["tn"] == 10
    assert metrics["tp"] == 5
    assert metrics["fp"] == 2
    assert metrics["fn"] == 1
    assert metrics["precision"] > 0.6
    assert metrics["recall"] > 0.7
    assert metrics["f1"] > 0.7

def test_delayed_monitor_triggers_concept_drift_alert():
    monitor = DelayedPerformanceMonitor(window_size=50, f1_alert_threshold=0.60)
    # High false negative rate (concept drift: fraud disguised, model predicts allow)
    for i in range(20):
        tx_id = f"tx-missed-{i}"
        monitor.record_prediction(tx_id, 0.05, "allow", 0.5, "champion")
        monitor.record_label(tx_id, 1)  # All are true fraud!

    report = monitor.report()
    assert report["status"] == "performance_deteriorated"
    assert report["metrics"]["recall"] == 0.0
    assert report["metrics"]["f1"] == 0.0
    assert len(report["alerts"]) > 0
    assert "deterioration" in report["alerts"][0]
