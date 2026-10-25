import React, { useEffect, useState } from 'react';
import {
  ShieldAlert,
  Activity,
  GitBranch,
  Radio,
  BarChart3,
  MailWarning,
  CheckCircle2,
  AlertTriangle,
  Play,
  Pause,
  RotateCcw,
  ExternalLink,
  ChevronRight,
  Database,
  Cpu,
  Zap,
  Info,
  Clock,
  Sparkles,
} from 'lucide-react';
import { api } from './api';
import {
  TransactionRecord,
  ReplayStatus,
  DriftReport,
  PerformanceReport,
  ModelInfo,
  ScamAnalysisResult,
} from './types';

export function App() {
  const [activeTab, setActiveTab] = useState<
    'transactions' | 'streaming' | 'rollout' | 'drift' | 'performance' | 'scam'
  >('transactions');

  // Backend state
  const [health, setHealth] = useState<any>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [replayStatus, setReplayStatus] = useState<ReplayStatus | null>(null);
  const [transactions, setTransactions] = useState<TransactionRecord[]>([]);
  const [driftReport, setDriftReport] = useState<DriftReport | null>(null);
  const [performanceReport, setPerformanceReport] = useState<PerformanceReport | null>(null);
  const [stats, setStats] = useState<any>(null);

  // Filters & selection
  const [selectedTx, setSelectedTx] = useState<TransactionRecord | null>(null);
  const [filterRisk, setFilterRisk] = useState<string>('all');
  const [filterDecision, setFilterDecision] = useState<string>('all');
  const [filterModel, setFilterModel] = useState<string>('all');

  // Streaming controls
  const [selectedScenario, setSelectedScenario] = useState<string>('baseline');
  const [replaySpeed, setReplaySpeed] = useState<number>(5.0);
  const [labelDelay, setLabelDelay] = useState<number>(10);

  // Rollout controls
  const [selectedMode, setSelectedMode] = useState<string>('champion');
  const [canaryPercent, setCanaryPercent] = useState<number>(15);
  const [forcePromotion, setForcePromotion] = useState<boolean>(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  // Scam analyzer
  const [scamText, setScamText] = useState<string>('');
  const [scamSourceType, setScamSourceType] = useState<string>('email');
  const [scamResult, setScamResult] = useState<ScamAnalysisResult | null>(null);
  const [scamSamples, setScamSamples] = useState<Record<string, string>>({});
  const [isScamLoading, setIsScamLoading] = useState<boolean>(false);

  // Periodic polling
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [h, mi, rs, txRes, dr, pr, st] = await Promise.all([
          api.getHealth().catch(() => null),
          api.getModelInfo().catch(() => null),
          api.getReplayStatus().catch(() => null),
          api.getRecentTransactions(40, filterRisk, filterDecision, filterModel).catch(() => ({ transactions: [] })),
          api.getDriftReport().catch(() => null),
          api.getPerformanceReport().catch(() => null),
          api.getTransactionStats().catch(() => null),
        ]);

        if (h) setHealth(h);
        if (mi) {
          setModelInfo(mi);
          setSelectedMode(mi.rollout.mode);
          setCanaryPercent(mi.rollout.canary_percent);
        }
        if (rs) setReplayStatus(rs);
        if (txRes && txRes.transactions) setTransactions(txRes.transactions);
        if (dr) setDriftReport(dr);
        if (pr) setPerformanceReport(pr);
        if (st) setStats(st);
      } catch (err) {
        console.error('Polling error:', err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [filterRisk, filterDecision, filterModel]);

  // Load scam samples on mount
  useEffect(() => {
    api.getScamSamples().then(setScamSamples).catch(() => {});
  }, []);

  const handleStartReplay = async () => {
    try {
      const res = await api.startReplay(selectedScenario, replaySpeed, labelDelay);
      setReplayStatus(res);
      setActionMessage(`Replay started: ${selectedScenario.toUpperCase()} at ${replaySpeed} tx/s.`);
      setTimeout(() => setActionMessage(null), 4000);
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handlePauseResumeReplay = async () => {
    if (!replayStatus) return;
    try {
      if (replayStatus.is_paused) {
        const res = await api.resumeReplay();
        setReplayStatus(res);
      } else {
        const res = await api.pauseReplay();
        setReplayStatus(res);
      }
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleStopReplay = async () => {
    try {
      const res = await api.stopReplay();
      setReplayStatus(res);
      setActionMessage('Replay stopped.');
      setTimeout(() => setActionMessage(null), 3000);
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleModeChange = async (mode: string, percent?: number) => {
    try {
      const res = await api.setRolloutMode(mode, percent ?? canaryPercent);
      setSelectedMode(res.mode);
      setCanaryPercent(res.canary_percent);
      setActionMessage(`Rollout mode updated to ${mode.toUpperCase()} (${res.canary_percent}% canary).`);
      setTimeout(() => setActionMessage(null), 4000);
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handlePromote = async () => {
    if (!confirm("Are you sure you want to promote Candidate Challenger to Champion?")) return;
    try {
      const res = await api.promoteCandidate('challenger', forcePromotion);
      setActionMessage(res.message);
      setTimeout(() => setActionMessage(null), 5000);
      const mi = await api.getModelInfo();
      setModelInfo(mi);
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleRollback = async () => {
    if (!confirm("Confirm immediate rollback of Candidate to Champion?")) return;
    try {
      await api.rollbackModel("Manual operator rollback from dashboard");
      setActionMessage("Candidate rolled back to Champion.");
      setTimeout(() => setActionMessage(null), 4000);
      const mi = await api.getModelInfo();
      setModelInfo(mi);
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleResetDrift = async () => {
    try {
      await api.resetDrift();
      const dr = await api.getDriftReport();
      setDriftReport(dr);
      setActionMessage("Drift monitoring window reset.");
      setTimeout(() => setActionMessage(null), 3000);
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleAnalyzeScam = async () => {
    if (!scamText.trim()) return;
    setIsScamLoading(true);
    try {
      const res = await api.analyzeScamText(scamText, scamSourceType);
      setScamResult(res);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setIsScamLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 pb-16">
      {/* Top Banner Navigation */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-tr from-indigo-600 to-cyan-500 rounded-xl shadow-lg shadow-indigo-500/20">
              <ShieldAlert className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">
                  TrustOps
                </h1>
                <span className="text-xs px-2 py-0.5 rounded-full font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  MLOps 2.0
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  LIVE DEMO
                </span>
              </div>
              <p className="text-xs text-slate-400">Automated Fraud Detection & MLOps Platform</p>
            </div>
          </div>

          {/* Quick Metrics & Links */}
          <div className="flex items-center gap-4 text-xs">
            <div className="hidden sm:flex items-center gap-3 px-3 py-1.5 bg-slate-800/60 rounded-lg border border-slate-700/60">
              <span className="text-slate-400">Active Champion:</span>
              <span className="font-mono font-semibold text-cyan-400">
                {modelInfo?.models?.champion?.file || 'champion.joblib'}
              </span>
              <span className="text-slate-600">|</span>
              <span className="text-slate-400">Rollout Mode:</span>
              <span className="font-semibold text-indigo-400 uppercase">
                {modelInfo?.rollout?.mode || 'champion'}
                {modelInfo?.rollout?.mode === 'canary' && ` (${modelInfo.rollout.canary_percent}%)`}
              </span>
              {modelInfo?.rollout?.canary_rolled_back && (
                <span className="px-1.5 py-0.5 bg-red-500/20 text-red-400 border border-red-500/40 rounded font-bold">
                  ROLLED BACK
                </span>
              )}
            </div>

            {/* External Links */}
            <div className="flex items-center gap-2">
              <a
                href="/docs"
                target="_blank"
                rel="noreferrer"
                className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 flex items-center gap-1 transition"
                title="FastAPI Swagger Documentation"
              >
                Docs <ExternalLink className="w-3 h-3 text-slate-400" />
              </a>
              <a
                href="http://localhost:3000"
                target="_blank"
                rel="noreferrer"
                className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 flex items-center gap-1 transition"
                title="Grafana Dashboard (Port 3000)"
              >
                Grafana <ExternalLink className="w-3 h-3 text-slate-400" />
              </a>
              <a
                href="http://localhost:5000"
                target="_blank"
                rel="noreferrer"
                className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 flex items-center gap-1 transition"
                title="MLflow Tracking (Port 5000)"
              >
                MLflow <ExternalLink className="w-3 h-3 text-slate-400" />
              </a>
              <a
                href="http://localhost:9090"
                target="_blank"
                rel="noreferrer"
                className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 flex items-center gap-1 transition"
                title="Prometheus Metrics (Port 9090)"
              >
                Prometheus <ExternalLink className="w-3 h-3 text-slate-400" />
              </a>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="max-w-7xl mx-auto flex items-center gap-2 mt-4 pt-3 border-t border-slate-800/80 overflow-x-auto text-sm font-medium">
          <button
            onClick={() => setActiveTab('transactions')}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition ${
              activeTab === 'transactions'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Activity className="w-4 h-4" /> Live Transactions
          </button>
          <button
            onClick={() => setActiveTab('streaming')}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition ${
              activeTab === 'streaming'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Radio className="w-4 h-4" /> Replay Controls
            {replayStatus?.is_running && !replayStatus?.is_paused && (
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
            )}
          </button>
          <button
            onClick={() => setActiveTab('rollout')}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition ${
              activeTab === 'rollout'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <GitBranch className="w-4 h-4" /> Rollout & Release Gates
          </button>
          <button
            onClick={() => setActiveTab('drift')}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition ${
              activeTab === 'drift'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Database className="w-4 h-4" /> Data Drift Monitor
            {driftReport?.status === 'drift_detected' && (
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
            )}
          </button>
          <button
            onClick={() => setActiveTab('performance')}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition ${
              activeTab === 'performance'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <BarChart3 className="w-4 h-4" /> Concept Drift & Delayed Labels
            {performanceReport?.status === 'performance_deteriorated' && (
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
            )}
          </button>
          <button
            onClick={() => setActiveTab('scam')}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition ${
              activeTab === 'scam'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <MailWarning className="w-4 h-4" /> Scam Analyzer
          </button>
        </div>
      </header>

      {/* Action Notification Toast */}
      {actionMessage && (
        <div className="max-w-7xl mx-auto px-6 mt-4">
          <div className="bg-indigo-950/80 border border-indigo-500/50 text-indigo-200 px-4 py-2.5 rounded-lg flex items-center justify-between text-sm shadow-lg shadow-indigo-950/50">
            <span className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-indigo-400" />
              {actionMessage}
            </span>
            <button onClick={() => setActionMessage(null)} className="text-slate-400 hover:text-white">
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-6 mt-6">
        {/* ========================================================================= */}
        {/* TAB 1: LIVE TRANSACTIONS */}
        {/* ========================================================================= */}
        {activeTab === 'transactions' && (
          <div className="space-y-6">
            {/* Top KPI Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <div className="card">
                <span className="text-xs text-slate-400 font-medium">Total Processed</span>
                <p className="text-2xl font-bold font-mono text-white mt-1">
                  {stats?.total_recorded || transactions.length}
                </p>
                <span className="text-[10px] text-slate-500">Live sliding window</span>
              </div>
              <div className="card">
                <span className="text-xs text-slate-400 font-medium">High Risk Alerts</span>
                <p className="text-2xl font-bold font-mono text-red-400 mt-1">
                  {stats?.risk_counts?.high || 0}
                </p>
                <span className="text-[10px] text-red-400/80">Immediate review</span>
              </div>
              <div className="card">
                <span className="text-xs text-slate-400 font-medium">Elevated Risk</span>
                <p className="text-2xl font-bold font-mono text-amber-400 mt-1">
                  {stats?.risk_counts?.elevated || 0}
                </p>
                <span className="text-[10px] text-amber-400/80">Flagged threshold</span>
              </div>
              <div className="card">
                <span className="text-xs text-slate-400 font-medium">Allowed</span>
                <p className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                  {stats?.decision_counts?.allow || 0}
                </p>
                <span className="text-[10px] text-emerald-400/80">Normal pass-through</span>
              </div>
              <div className="card">
                <span className="text-xs text-slate-400 font-medium">Resolved Labels</span>
                <p className="text-2xl font-bold font-mono text-cyan-400 mt-1">
                  {stats?.resolved_labels_count || 0}
                </p>
                <span className="text-[10px] text-slate-400">Delayed ground truth</span>
              </div>
              <div className="card">
                <span className="text-xs text-slate-400 font-medium">Confirmed Frauds</span>
                <p className="text-2xl font-bold font-mono text-pink-400 mt-1">
                  {stats?.confirmed_fraud_count || 0}
                </p>
                <span className="text-[10px] text-pink-400/80">Verified true class 1</span>
              </div>
            </div>

            {/* Filter Bar */}
            <div className="card flex flex-wrap items-center justify-between gap-4 py-3">
              <div className="flex items-center gap-3">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Filter Feed:
                </span>
                <select
                  value={filterRisk}
                  onChange={(e) => setFilterRisk(e.target.value)}
                  className="bg-slate-900 border border-slate-700 text-xs text-slate-200 px-3 py-1.5 rounded-lg focus:outline-none focus:border-indigo-500"
                >
                  <option value="all">All Risks</option>
                  <option value="high">High Risk</option>
                  <option value="elevated">Elevated Risk</option>
                  <option value="low">Low Risk</option>
                </select>

                <select
                  value={filterDecision}
                  onChange={(e) => setFilterDecision(e.target.value)}
                  className="bg-slate-900 border border-slate-700 text-xs text-slate-200 px-3 py-1.5 rounded-lg focus:outline-none focus:border-indigo-500"
                >
                  <option value="all">All Decisions</option>
                  <option value="review">Review</option>
                  <option value="allow">Allow</option>
                </select>

                <select
                  value={filterModel}
                  onChange={(e) => setFilterModel(e.target.value)}
                  className="bg-slate-900 border border-slate-700 text-xs text-slate-200 px-3 py-1.5 rounded-lg focus:outline-none focus:border-indigo-500"
                >
                  <option value="all">All Models</option>
                  <option value="champion">Champion</option>
                  <option value="challenger">Challenger</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400">
                  Showing {transactions.length} recent transactions
                </span>
                {replayStatus?.is_running && (
                  <span className="flex items-center gap-1.5 text-xs text-emerald-400 font-mono bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    Replay Active ({replayStatus.speed_tps} tx/s)
                  </span>
                )}
              </div>
            </div>

            {/* Transactions Table */}
            <div className="card overflow-hidden p-0">
              <div className="overflow-x-auto max-h-[600px]">
                <table>
                  <thead>
                    <tr>
                      <th>Transaction ID</th>
                      <th>Time (UTC)</th>
                      <th>Amount</th>
                      <th>Fraud Probability</th>
                      <th>Risk Level</th>
                      <th>Decision</th>
                      <th>Model</th>
                      <th>Latency</th>
                      <th>Delayed Label</th>
                      <th>Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {transactions.length === 0 ? (
                      <tr>
                        <td colSpan={10} className="text-center py-12 text-slate-500">
                          No transactions recorded in buffer. Start the replay engine to stream transactions.
                        </td>
                      </tr>
                    ) : (
                      transactions.map((tx) => {
                        const prob = tx.prediction?.fraud_probability ?? 0;
                        const risk = tx.prediction?.risk ?? 'low';
                        const decision = tx.prediction?.decision ?? 'allow';
                        const model = tx.prediction?.model_version ?? 'unknown';

                        return (
                          <tr key={tx.transaction_id}>
                            <td className="font-mono text-xs font-semibold text-slate-200">
                              {tx.transaction_id}
                            </td>
                            <td className="text-xs text-slate-400 font-mono">
                              {tx.timestamp ? tx.timestamp.slice(11, 19) : '-'}
                            </td>
                            <td className="font-mono text-xs font-semibold text-slate-300">
                              ${(tx.features?.Amount ?? 0).toFixed(2)}
                            </td>
                            <td className="font-mono text-xs font-bold">
                              <span
                                className={
                                  prob >= 0.75
                                    ? 'text-red-400'
                                    : prob >= 0.5
                                    ? 'text-amber-400'
                                    : 'text-slate-300'
                                }
                              >
                                {(prob * 100).toFixed(2)}%
                              </span>
                            </td>
                            <td>
                              <span className={`badge badge-${risk}`}>{risk}</span>
                            </td>
                            <td>
                              <span className={`badge badge-${decision}`}>{decision}</span>
                            </td>
                            <td className="font-mono text-xs text-indigo-400">{model}</td>
                            <td className="font-mono text-xs text-slate-400">
                              {tx.latency_ms?.toFixed(1) || '1.0'}ms
                            </td>
                            <td>
                              {tx.label_status === 'resolved' ? (
                                tx.ground_truth === 1 ? (
                                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-pink-500/20 text-pink-400 border border-pink-500/30">
                                    TRUE FRAUD (1)
                                  </span>
                                ) : (
                                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-300 border border-slate-700">
                                    NORMAL (0)
                                  </span>
                                )
                              ) : (
                                <span className="text-[10px] text-slate-500 italic">Pending</span>
                              )}
                            </td>
                            <td>
                              <button
                                onClick={() => setSelectedTx(tx)}
                                className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                              >
                                View <ChevronRight className="w-3 h-3" />
                              </button>
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: STREAMING / REPLAY CONTROLS */}
        {/* ========================================================================= */}
        {activeTab === 'streaming' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="card lg:col-span-2 space-y-6">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <Radio className="w-5 h-5 text-indigo-400" />
                  Transaction Replay Engine
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Stream held-out historical ULB transactions or simulated traffic scenarios to observe live model scoring, drift detection, and automated rollback.
                </p>
              </div>

              {/* Notice */}
              <div className="p-3.5 bg-slate-900/80 rounded-lg border border-slate-800 text-xs text-slate-300 space-y-1">
                <span className="font-semibold text-cyan-400 uppercase tracking-wide flex items-center gap-1.5">
                  <Info className="w-4 h-4" /> Realistic Benchmark Replay
                </span>
                <p>
                  Transactions are drawn from the held-out test split of the European Credit Card Fraud dataset (2,574 authentic transactions). Labels are held back and delivered with simulated delay.
                </p>
              </div>

              {/* Scenario Selection */}
              <div className="space-y-3">
                <label className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                  Select Traffic Scenario:
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div
                    onClick={() => setSelectedScenario('baseline')}
                    className={`p-4 rounded-xl border cursor-pointer transition ${
                      selectedScenario === 'baseline'
                        ? 'bg-indigo-950/60 border-indigo-500 shadow-md shadow-indigo-950/50'
                        : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-sm text-white">1. Baseline Traffic</span>
                      <span className="badge badge-low">Normal</span>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">
                      Held-out benchmark distribution (~0.2% fraud). Standard legitimate credit card transactions.
                    </p>
                  </div>

                  <div
                    onClick={() => setSelectedScenario('elevated_fraud')}
                    className={`p-4 rounded-xl border cursor-pointer transition ${
                      selectedScenario === 'elevated_fraud'
                        ? 'bg-indigo-950/60 border-indigo-500 shadow-md shadow-indigo-950/50'
                        : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-sm text-white">2. Elevated Fraud Attack</span>
                      <span className="badge badge-high">High Fraud</span>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">
                      Simulates an organized fraud ring attack (~20% fraud rate). Tests alert queues and precision/recall under surge.
                    </p>
                  </div>

                  <div
                    onClick={() => setSelectedScenario('feature_drift')}
                    className={`p-4 rounded-xl border cursor-pointer transition ${
                      selectedScenario === 'feature_drift'
                        ? 'bg-indigo-950/60 border-indigo-500 shadow-md shadow-indigo-950/50'
                        : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-sm text-white">3. Feature Data Drift</span>
                      <span className="badge badge-elevated">P(X) Shift</span>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">
                      Deliberately shifts numeric PCA components (V1..V4) and expands Amounts. Triggers KS-test and PSI drift detection.
                    </p>
                  </div>

                  <div
                    onClick={() => setSelectedScenario('concept_drift')}
                    className={`p-4 rounded-xl border cursor-pointer transition ${
                      selectedScenario === 'concept_drift'
                        ? 'bg-indigo-950/60 border-indigo-500 shadow-md shadow-indigo-950/50'
                        : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-sm text-white">4. Concept Drift</span>
                      <span className="badge badge-high">P(Y|X) Shift</span>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">
                      Fraudsters adapt and disguise frauds as benign low-value transactions. Inverts typical signals, degrading recall and triggering alerts.
                    </p>
                  </div>
                </div>
              </div>

              {/* Sliders */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-2">
                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-300">Replay Speed:</span>
                    <span className="text-indigo-400 font-mono">{replaySpeed} tx/sec</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    step="1"
                    value={replaySpeed}
                    onChange={(e) => setReplaySpeed(parseFloat(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>1 tx/s (Slow)</span>
                    <span>25 tx/s</span>
                    <span>50 tx/s (Fast)</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-300">Delayed Label Delay:</span>
                    <span className="text-indigo-400 font-mono">{labelDelay} transactions</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="30"
                    step="1"
                    value={labelDelay}
                    onChange={(e) => setLabelDelay(parseInt(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>2 tx (Near instant)</span>
                    <span>15 tx</span>
                    <span>30 tx (Delayed confirmation)</span>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center gap-3 pt-4 border-t border-slate-800">
                {!replayStatus?.is_running ? (
                  <button onClick={handleStartReplay} className="btn btn-primary px-6 py-2.5">
                    <Play className="w-4 h-4 fill-current" /> Start Replay Stream
                  </button>
                ) : (
                  <>
                    <button onClick={handlePauseResumeReplay} className="btn btn-warning px-5 py-2.5">
                      {replayStatus.is_paused ? (
                        <>
                          <Play className="w-4 h-4 fill-current" /> Resume Stream
                        </>
                      ) : (
                        <>
                          <Pause className="w-4 h-4 fill-current" /> Pause Stream
                        </>
                      )}
                    </button>
                    <button onClick={handleStopReplay} className="btn btn-danger px-5 py-2.5">
                      Stop Stream
                    </button>
                  </>
                )}

                <span className="text-xs text-slate-500 italic ml-auto">
                  {replayStatus?.is_running
                    ? `Running: ${replayStatus.scenario.toUpperCase()}`
                    : 'Stream is idle'}
                </span>
              </div>
            </div>

            {/* Replay Status Card */}
            <div className="card space-y-5">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Clock className="w-4 h-4 text-cyan-400" />
                Live Stream Metrics
              </h3>

              <div className="space-y-3">
                <div className="p-3 bg-slate-900 rounded-lg flex items-center justify-between">
                  <span className="text-xs text-slate-400">Stream Status:</span>
                  {replayStatus?.is_running ? (
                    replayStatus.is_paused ? (
                      <span className="px-2 py-0.5 rounded text-xs font-bold bg-amber-500/20 text-amber-400">
                        PAUSED
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded text-xs font-bold bg-emerald-500/20 text-emerald-400 flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        ACTIVE
                      </span>
                    )
                  ) : (
                    <span className="px-2 py-0.5 rounded text-xs font-bold bg-slate-800 text-slate-400">
                      STOPPED
                    </span>
                  )}
                </div>

                <div className="p-3 bg-slate-900 rounded-lg flex items-center justify-between">
                  <span className="text-xs text-slate-400">Active Scenario:</span>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase">
                    {replayStatus?.scenario || 'baseline'}
                  </span>
                </div>

                <div className="p-3 bg-slate-900 rounded-lg flex items-center justify-between">
                  <span className="text-xs text-slate-400">Total Streamed:</span>
                  <span className="text-sm font-mono font-bold text-white">
                    {replayStatus?.total_replayed || 0}
                  </span>
                </div>

                <div className="p-3 bg-slate-900 rounded-lg flex items-center justify-between">
                  <span className="text-xs text-slate-400">True Frauds Replayed:</span>
                  <span className="text-sm font-mono font-bold text-pink-400">
                    {replayStatus?.fraud_count || 0}
                  </span>
                </div>

                <div className="p-3 bg-slate-900 rounded-lg flex items-center justify-between">
                  <span className="text-xs text-slate-400">Alerts Triggered:</span>
                  <span className="text-sm font-mono font-bold text-red-400">
                    {replayStatus?.alert_count || 0}
                  </span>
                </div>

                <div className="p-3 bg-slate-900 rounded-lg flex items-center justify-between">
                  <span className="text-xs text-slate-400">Replay Uptime:</span>
                  <span className="text-xs font-mono text-slate-300">
                    {replayStatus?.uptime_seconds?.toFixed(1) || 0}s
                  </span>
                </div>
              </div>

              {/* CLI Command Helper */}
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide">
                  Run from PowerShell:
                </span>
                <pre className="font-mono text-[11px] text-cyan-300 mt-1 whitespace-pre-wrap">
                  python scripts/stream_transactions.py --scenario baseline --speed 5
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: ROLLOUT & RELEASE GATES */}
        {/* ========================================================================= */}
        {activeTab === 'rollout' && (
          <div className="space-y-6">
            {/* Rollout Mode Switcher Card */}
            <div className="card space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <GitBranch className="w-5 h-5 text-indigo-400" />
                    Model Rollout & Deployment Mode
                  </h2>
                  <p className="text-xs text-slate-400">
                    Control how production traffic is routed between Champion and Challenger models.
                  </p>
                </div>

                {modelInfo?.rollout?.canary_rolled_back && (
                  <div className="px-3 py-1.5 bg-red-950 border border-red-500/50 text-red-300 rounded-lg text-xs flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-red-400" />
                    <span>Rollback Active: {modelInfo.rollout.rollback_reason}</span>
                    <button
                      onClick={async () => {
                        await api.resetRollback();
                        const mi = await api.getModelInfo();
                        setModelInfo(mi);
                      }}
                      className="ml-2 underline text-white font-bold"
                    >
                      Clear
                    </button>
                  </div>
                )}
              </div>

              {/* Mode Buttons */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
                <div
                  onClick={() => handleModeChange('champion')}
                  className={`p-4 rounded-xl border cursor-pointer transition ${
                    selectedMode === 'champion'
                      ? 'bg-indigo-950/60 border-indigo-500 shadow-md shadow-indigo-950/50'
                      : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm text-white">1. Champion Only</span>
                    <span className="badge badge-low">Safe</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-2">
                    100% of live inference requests are served exclusively by the verified Champion model.
                  </p>
                </div>

                <div
                  onClick={() => handleModeChange('shadow')}
                  className={`p-4 rounded-xl border cursor-pointer transition ${
                    selectedMode === 'shadow'
                      ? 'bg-indigo-950/60 border-indigo-500 shadow-md shadow-indigo-950/50'
                      : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm text-white">2. Shadow Mode</span>
                    <span className="badge badge-elevated">Dual Score</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-2">
                    Scores events with Champion and Challenger simultaneously. Serves Champion decision; logs Challenger for side-by-side comparison.
                  </p>
                </div>

                <div
                  onClick={() => handleModeChange('canary', canaryPercent)}
                  className={`p-4 rounded-xl border cursor-pointer transition ${
                    selectedMode === 'canary'
                      ? 'bg-indigo-950/60 border-indigo-500 shadow-md shadow-indigo-950/50'
                      : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm text-white">3. Canary Rollout</span>
                    <span className="badge badge-high">Live Test</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-2">
                    Routes a controlled percentage of requests directly to Challenger. Rolls back automatically if latency or errors breach safeguards.
                  </p>
                </div>
              </div>

              {/* Canary Slider */}
              {selectedMode === 'canary' && (
                <div className="p-4 bg-slate-900/80 rounded-xl border border-slate-800 space-y-2 mt-4">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-300">Canary Traffic Allocation:</span>
                    <span className="text-indigo-400 font-mono font-bold">{canaryPercent}% to Challenger</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="100"
                    step="5"
                    value={canaryPercent}
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      setCanaryPercent(val);
                      handleModeChange('canary', val);
                    }}
                    className="w-full accent-indigo-500"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>5% (Cautious)</span>
                    <span>50% (A/B Test)</span>
                    <span>100% (Full candidate traffic)</span>
                  </div>
                </div>
              )}
            </div>

            {/* Model Comparison & Release Gates */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Champion Card */}
              <div className="card space-y-3 border-indigo-500/30">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-5 h-5 text-indigo-400" />
                    <h3 className="font-bold text-white">Active Champion Model</h3>
                  </div>
                  <span className="badge badge-low">Production</span>
                </div>
                <div className="p-3 bg-slate-900 rounded-lg text-xs font-mono space-y-1.5 text-slate-300">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Artifact:</span>
                    <span>{modelInfo?.models?.champion?.file}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Decision Threshold:</span>
                    <span className="text-cyan-400 font-bold">{modelInfo?.models?.champion?.threshold}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Test PR-AUC:</span>
                    <span>{(modelInfo?.models?.champion?.test_metrics?.average_precision * 100)?.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Test Recall:</span>
                    <span>{(modelInfo?.models?.champion?.test_metrics?.recall * 100)?.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Test Precision:</span>
                    <span>{(modelInfo?.models?.champion?.test_metrics?.precision * 100)?.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Test F1:</span>
                    <span className="text-emerald-400 font-bold">{(modelInfo?.models?.champion?.test_metrics?.f1 * 100)?.toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* Challenger Card */}
              <div className="card space-y-3 border-cyan-500/30">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Zap className="w-5 h-5 text-cyan-400" />
                    <h3 className="font-bold text-white">Candidate / Challenger Model</h3>
                  </div>
                  <span className="badge badge-elevated">Candidate</span>
                </div>
                <div className="p-3 bg-slate-900 rounded-lg text-xs font-mono space-y-1.5 text-slate-300">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Artifact:</span>
                    <span>{modelInfo?.models?.challenger?.file}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Decision Threshold:</span>
                    <span className="text-cyan-400 font-bold">{modelInfo?.models?.challenger?.threshold}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Test PR-AUC:</span>
                    <span>{(modelInfo?.models?.challenger?.test_metrics?.average_precision * 100)?.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Test Recall:</span>
                    <span>{(modelInfo?.models?.challenger?.test_metrics?.recall * 100)?.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Test Precision:</span>
                    <span>{(modelInfo?.models?.challenger?.test_metrics?.precision * 100)?.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Test F1:</span>
                    <span className="text-emerald-400 font-bold">{(modelInfo?.models?.challenger?.test_metrics?.f1 * 100)?.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Release Gate Checklist & Actions */}
            <div className="card space-y-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Pre-Promotion Release Gates Checklist
              </h3>
              <p className="text-xs text-slate-400">
                Before a candidate can be promoted to Champion, all automated governance criteria must pass.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
                {modelInfo?.release_gate_status?.gates &&
                  Object.entries(modelInfo.release_gate_status.gates).map(([gateName, gateData]: any) => (
                    <div
                      key={gateName}
                      className={`p-3 rounded-lg border text-xs ${
                        gateData.passed
                          ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-200'
                          : 'bg-red-950/20 border-red-500/30 text-red-200'
                      }`}
                    >
                      <div className="flex items-center justify-between font-bold">
                        <span className="capitalize">{gateName.replace(/_/g, ' ')}</span>
                        {gateData.passed ? '✓ PASS' : '✗ FAIL'}
                      </div>
                      <div className="mt-1 font-mono text-[11px] text-slate-400">
                        Target: {gateData.target ?? gateData.budget_ms ?? 'Match'} | Actual:{' '}
                        {gateData.actual ?? gateData.actual_ms ?? 'OK'}
                      </div>
                    </div>
                  ))}
              </div>

              {/* Promotion / Rollback Action Bar */}
              <div className="flex flex-wrap items-center justify-between gap-4 pt-4 border-t border-slate-800">
                <div className="flex items-center gap-3">
                  <button onClick={handlePromote} className="btn btn-primary px-5 py-2">
                    Promote Challenger to Champion
                  </button>
                  <button onClick={handleRollback} className="btn btn-danger px-4 py-2">
                    Emergency Rollback
                  </button>
                  <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={forcePromotion}
                      onChange={(e) => setForcePromotion(e.target.checked)}
                      className="accent-indigo-500"
                    />
                    Bypass gates (Force Admin override)
                  </label>
                </div>

                <span className="text-xs text-slate-500">
                  Audit log records all promotions and rollbacks with timestamp and reason.
                </span>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 4: DATA DRIFT MONITORING */}
        {/* ========================================================================= */}
        {activeTab === 'drift' && (
          <div className="space-y-6">
            {/* Status Banner */}
            <div
              className={`card p-4 flex flex-wrap items-center justify-between gap-4 border ${
                driftReport?.status === 'drift_detected'
                  ? 'bg-red-950/40 border-red-500/50'
                  : driftReport?.status === 'collecting'
                  ? 'bg-amber-950/40 border-amber-500/50'
                  : 'bg-emerald-950/40 border-emerald-500/50'
              }`}
            >
              <div className="flex items-center gap-3">
                <div
                  className={`p-2 rounded-lg ${
                    driftReport?.status === 'drift_detected'
                      ? 'bg-red-500 text-white'
                      : driftReport?.status === 'collecting'
                      ? 'bg-amber-500 text-white'
                      : 'bg-emerald-500 text-white'
                  }`}
                >
                  <Database className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-white text-base flex items-center gap-2">
                    Drift Status: <span className="uppercase">{driftReport?.status || 'COLLECTING'}</span>
                  </h3>
                  <p className="text-xs text-slate-300">
                    Two-sample Kolmogorov-Smirnov test + PSI with Bonferroni correction against training reference.
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-6 text-xs font-mono">
                <div>
                  <span className="text-slate-400 block">Window Samples:</span>
                  <span className="font-bold text-white text-sm">
                    {driftReport?.samples || 0} / {driftReport?.window_size || 500}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">Max KS Statistic:</span>
                  <span className="font-bold text-cyan-400 text-sm">
                    {driftReport?.max_ks_statistic ?? 0}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">Max PSI:</span>
                  <span className="font-bold text-indigo-400 text-sm">
                    {driftReport?.max_psi ?? 0}
                  </span>
                </div>
                <button onClick={handleResetDrift} className="btn btn-secondary text-xs py-1.5 px-3">
                  <RotateCcw className="w-3.5 h-3.5" /> Reset Window
                </button>
              </div>
            </div>

            {/* Feature Table */}
            <div className="card space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                  Feature-Level Statistical Drift Results (29 Features)
                </h3>
                <span className="text-xs text-slate-400 font-mono">
                  Bonferroni Alpha: {driftReport?.alpha_bonferroni || 0.0017} | Nominal: 0.05
                </span>
              </div>

              <div className="overflow-x-auto max-h-[500px]">
                <table>
                  <thead>
                    <tr>
                      <th>Feature Name</th>
                      <th>KS Statistic (Effect)</th>
                      <th>p-Value</th>
                      <th>PSI</th>
                      <th>Mean Shift (std)</th>
                      <th>Window Mean</th>
                      <th>Reference Mean</th>
                      <th>Drift Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {!driftReport?.features || Object.keys(driftReport.features).length === 0 ? (
                      <tr>
                        <td colSpan={8} className="text-center py-8 text-slate-500">
                          {driftReport?.status === 'collecting'
                            ? `Collecting minimum sample window (${driftReport.samples}/${driftReport.required || 50} collected)...`
                            : 'No drift metrics available.'}
                        </td>
                      </tr>
                    ) : (
                      Object.entries(driftReport.features).map(([feat, info]) => (
                        <tr key={feat}>
                          <td className="font-mono text-xs font-bold text-slate-200">{feat}</td>
                          <td className="font-mono text-xs text-slate-300">
                            {info.ks_statistic?.toFixed(4)}
                          </td>
                          <td className="font-mono text-xs text-slate-400">
                            {info.p_value < 0.0001 ? '< 0.0001' : info.p_value?.toFixed(4)}
                          </td>
                          <td className="font-mono text-xs text-indigo-300">
                            {info.psi?.toFixed(4)}
                          </td>
                          <td className="font-mono text-xs text-slate-300">
                            {info.mean_shift?.toFixed(4)}
                          </td>
                          <td className="font-mono text-xs text-slate-400">
                            {info.window_mean?.toFixed(2)}
                          </td>
                          <td className="font-mono text-xs text-slate-500">
                            {info.reference_mean?.toFixed(2)}
                          </td>
                          <td>
                            {info.drift_detected ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-400 border border-red-500/30">
                                DRIFTED
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400">
                                STABLE
                              </span>
                            )}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 5: CONCEPT DRIFT & DELAYED LABELS */}
        {/* ========================================================================= */}
        {activeTab === 'performance' && (
          <div className="space-y-6">
            {/* Deterioration Alert Banner */}
            {performanceReport?.status === 'performance_deteriorated' && (
              <div className="p-4 bg-red-950/60 border border-red-500 rounded-xl text-red-200 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="font-bold text-red-300">Concept Drift / Model Deterioration Detected!</h3>
                  <p className="text-xs text-red-200/90 mt-1">
                    Delayed ground truth labels reveal that online model F1 score has dropped below the safeguard threshold ({performanceReport.f1_alert_threshold}). Fraudsters have changed tactics or disguised fraud features.
                  </p>
                  {performanceReport.alerts?.map((alert, i) => (
                    <span key={i} className="block text-xs font-mono text-red-400 mt-1 font-semibold">
                      • {alert}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Performance KPIs */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="card">
                <span className="text-xs text-slate-400 font-medium">Online F1 Score</span>
                <p className="text-2xl font-bold font-mono text-cyan-400 mt-1">
                  {performanceReport?.metrics?.f1 !== undefined
                    ? `${(performanceReport.metrics.f1 * 100).toFixed(1)}%`
                    : 'N/A'}
                </p>
                <span className="text-[10px] text-slate-500">Alert threshold: 60%</span>
              </div>
              <div className="card">
                <span className="text-xs text-slate-400 font-medium">Online Recall</span>
                <p className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                  {performanceReport?.metrics?.recall !== undefined
                    ? `${(performanceReport.metrics.recall * 100).toFixed(1)}%`
                    : 'N/A'}
                </p>
                <span className="text-[10px] text-slate-500">True fraud catch rate</span>
              </div>
              <div className="card">
                <span className="text-xs text-slate-400 font-medium">Online Precision</span>
                <p className="text-2xl font-bold font-mono text-indigo-400 mt-1">
                  {performanceReport?.metrics?.precision !== undefined
                    ? `${(performanceReport.metrics.precision * 100).toFixed(1)}%`
                    : 'N/A'}
                </p>
                <span className="text-[10px] text-slate-500">Review accuracy</span>
              </div>
              <div className="card">
                <span className="text-xs text-slate-400 font-medium">Resolved Ground Truth</span>
                <p className="text-2xl font-bold font-mono text-white mt-1">
                  {performanceReport?.resolved_samples || 0}
                </p>
                <span className="text-[10px] text-slate-500">
                  {performanceReport?.pending_labels || 0} pending in queue
                </span>
              </div>
            </div>

            {/* Confusion Matrix Visual */}
            <div className="card space-y-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                Delayed Ground Truth Confusion Matrix
              </h3>
              <p className="text-xs text-slate-400">
                Calculated on real ground truth labels after the delayed confirmation interval.
              </p>

              <div className="grid grid-cols-2 gap-4 max-w-md mx-auto pt-2">
                <div className="p-4 bg-emerald-950/20 border border-emerald-500/30 rounded-xl text-center">
                  <span className="text-xs text-emerald-400 block font-semibold">True Negatives (TN)</span>
                  <span className="text-2xl font-mono font-bold text-white mt-1 block">
                    {performanceReport?.metrics?.tn ?? 0}
                  </span>
                  <span className="text-[10px] text-slate-400">Normal allowed correctly</span>
                </div>
                <div className="p-4 bg-amber-950/20 border border-amber-500/30 rounded-xl text-center">
                  <span className="text-xs text-amber-400 block font-semibold">False Positives (FP)</span>
                  <span className="text-2xl font-mono font-bold text-white mt-1 block">
                    {performanceReport?.metrics?.fp ?? 0}
                  </span>
                  <span className="text-[10px] text-slate-400">False alarms</span>
                </div>
                <div className="p-4 bg-red-950/20 border border-red-500/30 rounded-xl text-center">
                  <span className="text-xs text-red-400 block font-semibold">False Negatives (FN)</span>
                  <span className="text-2xl font-mono font-bold text-white mt-1 block">
                    {performanceReport?.metrics?.fn ?? 0}
                  </span>
                  <span className="text-[10px] text-slate-400">Missed fraud (Dangerous)</span>
                </div>
                <div className="p-4 bg-emerald-950/20 border border-emerald-500/30 rounded-xl text-center">
                  <span className="text-xs text-emerald-400 block font-semibold">True Positives (TP)</span>
                  <span className="text-2xl font-mono font-bold text-white mt-1 block">
                    {performanceReport?.metrics?.tp ?? 0}
                  </span>
                  <span className="text-[10px] text-slate-400">Frauds intercepted</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 6: SCAM ANALYZER (VOICE / EMAIL EXTENSION) */}
        {/* ========================================================================= */}
        {activeTab === 'scam' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card space-y-4">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <MailWarning className="w-5 h-5 text-indigo-400" />
                  Social Engineering & Scam Analyzer
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  NLP heuristic and pattern analysis for suspicious payment instructions, urgency, executive impersonation, and credential theft.
                </p>
              </div>

              {/* Sample Scenarios */}
              <div className="space-y-2">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                  Quick Load Test Samples:
                </span>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => {
                      setScamText(scamSamples.ceo_fraud_email || '');
                      setScamSourceType('email');
                    }}
                    className="btn btn-secondary text-xs py-1 px-2.5"
                  >
                    CEO Wire Fraud
                  </button>
                  <button
                    onClick={() => {
                      setScamText(scamSamples.fake_tech_support || '');
                      setScamSourceType('voice_transcript');
                    }}
                    className="btn btn-secondary text-xs py-1 px-2.5"
                  >
                    Fake IT Support (Audio Call)
                  </button>
                  <button
                    onClick={() => {
                      setScamText(scamSamples.irs_gift_card_scam || '');
                      setScamSourceType('voice_transcript');
                    }}
                    className="btn btn-secondary text-xs py-1 px-2.5"
                  >
                    IRS Gift Card Scam
                  </button>
                  <button
                    onClick={() => {
                      setScamText(scamSamples.legitimate_billing_email || '');
                      setScamSourceType('email');
                    }}
                    className="btn btn-secondary text-xs py-1 px-2.5"
                  >
                    Legitimate Invoice
                  </button>
                </div>
              </div>

              {/* Text Input */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-300">Email Text or Call Transcript:</label>
                  <select
                    value={scamSourceType}
                    onChange={(e) => setScamSourceType(e.target.value)}
                    className="bg-slate-900 border border-slate-700 text-slate-200 px-2 py-1 rounded"
                  >
                    <option value="email">Email</option>
                    <option value="voice_transcript">Voice Call Transcript</option>
                    <option value="sms">SMS Smishing</option>
                  </select>
                </div>
                <textarea
                  rows={8}
                  value={scamText}
                  onChange={(e) => setScamText(e.target.value)}
                  placeholder="Paste email text or voice recording transcript here..."
                  className="w-full bg-slate-900 border border-slate-700 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
                ></textarea>
              </div>

              <button
                onClick={handleAnalyzeScam}
                disabled={isScamLoading || !scamText.trim()}
                className="btn btn-primary w-full py-2.5"
              >
                {isScamLoading ? 'Evaluating NLP Patterns...' : 'Analyze Communication for Threats'}
              </button>
            </div>

            {/* Analysis Output Card */}
            <div className="card space-y-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                Threat Intelligence & Evidence
              </h3>

              {!scamResult ? (
                <div className="text-center py-16 text-slate-500 text-xs">
                  Load a sample scenario and click "Analyze" to see extracted signals.
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Risk Badge & Score */}
                  <div className="p-4 bg-slate-900 rounded-xl flex items-center justify-between">
                    <div>
                      <span className="text-xs text-slate-400 block">Threat Classification:</span>
                      <span
                        className={`badge mt-1 ${
                          scamResult.risk_level === 'critical_scam'
                            ? 'badge-high'
                            : scamResult.risk_level === 'suspicious'
                            ? 'badge-elevated'
                            : 'badge-low'
                        }`}
                      >
                        {scamResult.risk_level.replace('_', ' ')}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-slate-400 block">Risk Score:</span>
                      <span className="text-2xl font-mono font-bold text-white">
                        {(scamResult.risk_score * 100).toFixed(0)} / 100
                      </span>
                    </div>
                  </div>

                  <p className="text-xs text-slate-300 italic">{scamResult.explanation}</p>

                  {/* Flagged Evidence */}
                  <div className="space-y-2">
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                      Flagged Categories ({scamResult.evidence?.length || 0}):
                    </span>
                    {scamResult.evidence?.map((item, idx) => (
                      <div key={idx} className="p-3 bg-slate-900/60 rounded-lg border border-slate-800 text-xs">
                        <div className="flex justify-between font-bold text-slate-200">
                          <span className="capitalize">{item.category.replace(/_/g, ' ')}</span>
                          <span className="text-red-400 font-mono">Weight: {item.severity_weight}</span>
                        </div>
                        <p className="text-slate-400 text-[11px] mt-0.5">{item.description}</p>
                        <div className="flex flex-wrap gap-1 mt-2">
                          {item.matched_terms.map((term, tIdx) => (
                            <span
                              key={tIdx}
                              className="px-2 py-0.5 bg-red-500/10 border border-red-500/30 text-red-300 rounded text-[10px] font-mono"
                            >
                              "{term}"
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* ========================================================================= */}
      {/* TRANSACTION DETAIL MODAL */}
      {/* ========================================================================= */}
      {selectedTx && (
        <div className="modal-overlay" onClick={() => setSelectedTx(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between items-center pb-3 border-b border-slate-800">
              <h3 className="font-bold text-white flex items-center gap-2">
                <Info className="w-5 h-5 text-indigo-400" />
                Transaction Detail: {selectedTx.transaction_id}
              </h3>
              <button
                onClick={() => setSelectedTx(null)}
                className="text-slate-400 hover:text-white text-lg font-bold"
              >
                ✕
              </button>
            </div>

            <div className="mt-4 space-y-4 text-xs">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 bg-slate-900 rounded-lg">
                  <span className="text-slate-500 block">Fraud Probability:</span>
                  <span className="font-mono font-bold text-white text-sm">
                    {((selectedTx.prediction?.fraud_probability || 0) * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="p-3 bg-slate-900 rounded-lg">
                  <span className="text-slate-500 block">Model Decision:</span>
                  <span className="font-bold uppercase text-white">
                    {selectedTx.prediction?.decision}
                  </span>
                </div>
                <div className="p-3 bg-slate-900 rounded-lg">
                  <span className="text-slate-500 block">Model Version:</span>
                  <span className="font-mono text-indigo-400 font-bold">
                    {selectedTx.prediction?.model_version}
                  </span>
                </div>
                <div className="p-3 bg-slate-900 rounded-lg">
                  <span className="text-slate-500 block">Inference Latency:</span>
                  <span className="font-mono text-cyan-400 font-bold">
                    {selectedTx.latency_ms?.toFixed(2)} ms
                  </span>
                </div>
              </div>

              {/* Input Features Grid */}
              <div className="space-y-2">
                <span className="font-bold text-slate-300 uppercase tracking-wider block">
                  Input Features ({Object.keys(selectedTx.features || {}).length} PCA Features + Amount):
                </span>
                <div className="grid grid-cols-3 sm:grid-cols-5 gap-2 max-h-56 overflow-y-auto p-2 bg-slate-900/60 rounded-xl border border-slate-800">
                  {Object.entries(selectedTx.features || {}).map(([key, val]) => (
                    <div key={key} className="p-2 bg-slate-900 rounded border border-slate-800/80">
                      <span className="text-[10px] text-slate-500 block font-mono">{key}</span>
                      <span className="font-mono text-xs font-semibold text-slate-200">
                        {typeof val === 'number' ? val.toFixed(4) : val}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Raw JSON */}
              <div className="space-y-1">
                <span className="font-bold text-slate-400 uppercase tracking-wide text-[10px]">
                  Prediction Output Schema:
                </span>
                <pre className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto">
                  {JSON.stringify(selectedTx.prediction, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
