import {
  TransactionRecord,
  ReplayStatus,
  DriftReport,
  PerformanceReport,
  ModelInfo,
  RolloutStatus,
  ScamAnalysisResult,
} from './types';

// In development or production, use relative paths so Vite proxy or FastAPI serves seamlessly
const BASE_URL = '';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${url}`, options);
  if (!res.ok) {
    let errMessage = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const errObj = await res.json();
      if (errObj.detail) {
        errMessage = typeof errObj.detail === 'string' ? errObj.detail : JSON.stringify(errObj.detail);
      }
    } catch {}
    throw new Error(errMessage);
  }
  return res.json();
}

export const api = {
  getHealth: () => fetchJson<{ status: string; models_loaded: string[]; uptime_seconds: number; rollout_mode: string; canary_rolled_back: boolean }>('/health'),
  getModelInfo: () => fetchJson<ModelInfo>('/model-info'),
  
  // Replay
  getReplayStatus: () => fetchJson<ReplayStatus>('/replay/status'),
  startReplay: (scenario: string, speed_tps: number, label_delay_steps: number) =>
    fetchJson<ReplayStatus>('/replay/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario, speed_tps, label_delay_steps }),
    }),
  pauseReplay: () => fetchJson<ReplayStatus>('/replay/pause', { method: 'POST' }),
  resumeReplay: () => fetchJson<ReplayStatus>('/replay/resume', { method: 'POST' }),
  stopReplay: () => fetchJson<ReplayStatus>('/replay/stop', { method: 'POST' }),

  // Transactions
  getRecentTransactions: (limit: number = 50, risk?: string, decision?: string, model?: string) => {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (risk && risk !== 'all') params.append('risk', risk);
    if (decision && decision !== 'all') params.append('decision', decision);
    if (model && model !== 'all') params.append('model', model);
    return fetchJson<{ count: number; transactions: TransactionRecord[] }>(`/transactions/recent?${params.toString()}`);
  },
  getTransactionStats: () => fetchJson<any>('/transactions/stats/summary'),

  // Drift Monitoring
  getDriftReport: () => fetchJson<DriftReport>('/monitoring/drift'),
  resetDrift: () => fetchJson<{ status: string; message: string }>('/monitoring/drift/reset', { method: 'POST' }),

  // Performance & Concept Drift
  getPerformanceReport: () => fetchJson<PerformanceReport>('/monitoring/performance'),
  resetPerformance: () => fetchJson<{ status: string; message: string }>('/monitoring/performance/reset', { method: 'POST' }),

  // Rollout & Model Lifecycle
  setRolloutMode: (mode: string, canary_percent?: number) =>
    fetchJson<RolloutStatus>('/models/rollout-mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode, canary_percent }),
    }),
  evaluateCandidate: () => fetchJson<any>('/models/evaluate-candidate', { method: 'POST' }),
  promoteCandidate: (candidate_name: string = 'challenger', force: boolean = false, reason?: string) =>
    fetchJson<any>('/models/promote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate_name, force, reason }),
    }),
  rollbackModel: (reason?: string) =>
    fetchJson<RolloutStatus>('/models/rollback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    }),
  resetRollback: () => fetchJson<RolloutStatus>('/models/rollback/reset', { method: 'POST' }),

  // Scam Analyzer
  analyzeScamText: (text: string, source_type: string = 'email') =>
    fetchJson<ScamAnalysisResult>('/scam/analyze-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, source_type }),
    }),
  getScamSamples: () => fetchJson<Record<string, string>>('/scam/samples'),
};
