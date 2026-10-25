export interface TransactionPrediction {
  fraud_probability: number;
  risk: 'low' | 'elevated' | 'high';
  threshold: number;
  model_version: string;
  rollout_mode: string;
  decision: 'allow' | 'review';
  shadow_challenger_probability?: number | null;
  canary_rolled_back: boolean;
}

export interface TransactionRecord {
  transaction_id: string;
  timestamp: string;
  data_type: string;
  scenario: string;
  features: Record<string, number>;
  prediction: TransactionPrediction;
  latency_ms: number;
  ground_truth?: number | null;
  label_status: 'pending' | 'resolved';
}

export interface ReplayStatus {
  is_running: boolean;
  is_paused: boolean;
  scenario: string;
  speed_tps: number;
  label_delay_steps: number;
  total_replayed: number;
  fraud_count: number;
  alert_count: number;
  uptime_seconds: number;
  dataset_available: number;
}

export interface FeatureDriftInfo {
  ks_statistic: number;
  p_value: number;
  psi: number;
  mean_shift: number;
  window_mean: number;
  reference_mean: number;
  drift_detected: boolean;
}

export interface DriftReport {
  status: 'healthy' | 'collecting' | 'drift_detected' | 'reference_missing' | 'not_ready';
  timestamp?: string;
  samples: number;
  window_size?: number;
  required?: number;
  reference_size?: number;
  alpha_nominal?: number;
  alpha_bonferroni?: number;
  max_psi?: number;
  max_ks_statistic?: number;
  drifted_feature_count?: number;
  drifted_features?: string[];
  alerts?: string[];
  features: Record<string, FeatureDriftInfo>;
}

export interface PerformanceReport {
  status: 'healthy' | 'collecting_labels' | 'performance_deteriorated' | 'not_ready';
  timestamp?: string;
  resolved_samples: number;
  window_size?: number;
  pending_labels: number;
  f1_alert_threshold?: number;
  metrics: {
    precision?: number;
    recall?: number;
    f1?: number;
    pr_auc?: number | null;
    tp?: number;
    fp?: number;
    tn?: number;
    fn?: number;
    true_fraud_count?: number;
    true_legit_count?: number;
  };
  alerts: string[];
}

export interface RolloutStatus {
  mode: 'champion' | 'shadow' | 'canary';
  canary_percent: number;
  canary_rolled_back: boolean;
  rollback_reason?: string | null;
  candidate_latency_samples: number;
  avg_candidate_latency_ms: number;
  avg_champion_latency_ms: number;
  max_candidate_latency_ms: number;
  consecutive_slow_calls: number;
  consecutive_errors: number;
  gate_rules: {
    min_pr_auc: number;
    min_precision: number;
    min_recall: number;
    max_latency_ms: number;
  };
  recent_audit_events: Array<{
    timestamp: string;
    event: string;
    details: string;
    mode: string;
  }>;
}

export interface ModelInfo {
  project: string;
  dataset: string;
  trained_at_utc: string;
  feature_names: string[];
  models: Record<string, {
    file: string;
    threshold: number;
    validation_metrics: Record<string, any>;
    test_metrics: Record<string, any>;
  }>;
  rollout: RolloutStatus;
  release_gate_status: {
    passed: boolean;
    reason: string;
    evaluated_at: string;
    gates: Record<string, any>;
  };
}

export interface ScamEvidence {
  category: string;
  description: string;
  matched_terms: string[];
  severity_weight: number;
}

export interface ScamAnalysisResult {
  source_type: string;
  text_length: number;
  risk_score: number;
  risk_level: 'low' | 'suspicious' | 'critical_scam';
  flagged_categories: string[];
  evidence: ScamEvidence[];
  explanation: string;
  analyzed_at: string;
}
