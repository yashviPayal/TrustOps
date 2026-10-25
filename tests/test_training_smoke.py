import numpy as np
import pandas as pd
from app.ml.train import make_model, make_reference
from app.ml.metrics import choose_threshold, evaluate
from app.ml.data import feature_columns, split_data

def test_deterministic_training_smoke_pipeline(tmp_path):
    # 1. Create a lightweight synthetic dataset
    rng = np.random.default_rng(42)
    n_samples = 200
    features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    data = {feat: rng.normal(0, 1, n_samples) for feat in features}
    # 20 frauds, 180 normal
    labels = np.zeros(n_samples, dtype=int)
    labels[:20] = 1
    rng.shuffle(labels)
    data["Class"] = labels

    df = pd.DataFrame(data)
    cols = feature_columns(df)
    assert len(cols) == 29

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, cols)
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    # 2. Fit pipeline
    model = make_model(C=1.0)
    model.fit(X_train, y_train)

    # 3. Validation threshold search
    val_probs = model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_probs)
    assert 0.0 < threshold < 1.0

    # 4. Evaluate on test set
    test_probs = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, test_probs, threshold)
    assert "average_precision" in metrics
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == len(y_test)

    # 5. Generate reference distribution
    ref = make_reference(X_train, cols)
    assert "V1" in ref
    assert "edges" in ref["V1"]
    assert "proportions" in ref["V1"]
    assert "sample" in ref["V1"]
    assert len(ref["V1"]["sample"]) > 0
