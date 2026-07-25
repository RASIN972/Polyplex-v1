import { useEffect, useMemo, useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type AppState, type RunRow, type TrainConfig } from "./api";

const OUTCOME_COLOR: Record<string, string> = {
  finished: "#34d399",
  crashed: "#f87171",
  timeout: "#fbbf24",
  off_track: "#ff8a3d",
};

const emptyState = (): AppState => ({
  live_on: false,
  live_label: "IDLE",
  training: false,
  train_status: "Connecting…",
  watch_status: "",
  watching: false,
  watch_port: 8099,
  config: {
    num_envs: 4,
    timesteps: 1_000_000,
    headless: true,
    watch_live: false,
    dummy_vec: false,
  },
  metrics: {
    best_fitness: 0,
    best_reward: 0,
    mean_fitness: 0,
    timesteps: 0,
    total_timesteps: 1,
    progress: 0,
    fps: 0,
    uptime_s: 0,
    uptime: "00:00:00",
    episodes: 0,
    finishes: 0,
    crashes: 0,
    off_tracks: 0,
  },
  history: { mean_fitness: [], mean_reward: [], timesteps: [] },
  last5: [],
});

type Tab = "overview" | "runs";

export default function App() {
  const [state, setState] = useState<AppState>(emptyState);
  const [cfg, setCfg] = useState<TrainConfig>(emptyState().config);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [graphOpen, setGraphOpen] = useState(true);
  const [tab, setTab] = useState<Tab>("overview");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const [s, r] = await Promise.all([api.state(), api.runs()]);
        if (!alive) return;
        setState(s);
        setRuns(r);
        setError(null);
      } catch (e) {
        if (!alive) return;
        setError(e instanceof Error ? e.message : "Failed to reach API");
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 1500);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  const chartData = useMemo(() => {
    const { mean_fitness, mean_reward, timesteps } = state.history;
    const n = Math.max(mean_fitness.length, mean_reward.length, timesteps.length);
    const rows = [];
    for (let i = 0; i < n; i++) {
      rows.push({
        step: timesteps[i] ?? i,
        fitness: mean_fitness[i] ?? null,
        reward: mean_reward[i] ?? null,
      });
    }
    return rows;
  }, [state.history]);

  const sparkFitness = useMemo(
    () => chartData.slice(-12).map((d, i) => ({ i, v: d.fitness ?? 0 })),
    [chartData],
  );

  const selected = runs.find((r) => r.id === selectedId) ?? null;
  const m = state.metrics;
  const progressPct = Math.min(100, Math.max(0, m.progress * 100));
  const circumference = 2 * Math.PI * 88;
  const dashOffset = circumference * (1 - progressPct / 100);

  const startTraining = async () => {
    setBusy(true);
    try {
      const res = await api.startTraining(cfg);
      setState((s) => ({ ...s, train_status: res.message }));
    } finally {
      setBusy(false);
    }
  };

  const stopTraining = async () => {
    setBusy(true);
    try {
      const res = await api.stopTraining();
      setState((s) => ({ ...s, train_status: res.message }));
    } finally {
      setBusy(false);
    }
  };

  const watchSelected = async (id?: number) => {
    const runId = id ?? selectedId;
    if (runId == null) {
      setState((s) => ({ ...s, watch_status: "Select a run first." }));
      return;
    }
    setBusy(true);
    try {
      const res = await api.startWatch(runId);
      setState((s) => ({ ...s, watch_status: res.message }));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="bg-scene" aria-hidden />
      <div className="app fade-in">
        <header className="nav glass">
          <div className="brand">
            <div className="brand-mark">P</div>
            <div>
              <h1>Polyplex</h1>
              <p>Training control</p>
            </div>
          </div>

          <div className="tabs">
            <button
              type="button"
              className={`tab ${tab === "overview" ? "active" : ""}`}
              onClick={() => setTab("overview")}
            >
              Overview
            </button>
            <button
              type="button"
              className={`tab ${tab === "runs" ? "active" : ""}`}
              onClick={() => setTab("runs")}
            >
              Best runs
            </button>
          </div>

          <div className="nav-right">
            <div className={`pill ${state.live_on ? "on" : ""}`}>
              <span className="dot" />
              {state.live_label}
            </div>
          </div>
        </header>

        {error && (
          <div className="error-banner">
            API offline ({error}). Is <code>start_gui.py</code> running?
          </div>
        )}

        <section className="hero-grid">
          <div className="hero glass">
            <div className="hero-kicker">PPO training</div>
            <h2>Train your racing agent</h2>
            <p className="sub">
              Watch live rollouts, track reward and distance, and replay elite
              runs as the policy improves.
            </p>

            <div className="hero-stage">
              <div className="gauge">
                <svg viewBox="0 0 200 200">
                  <defs>
                    <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#8b5cf6" />
                      <stop offset="55%" stopColor="#c4b5fd" />
                      <stop offset="100%" stopColor="#ff8a3d" />
                    </linearGradient>
                  </defs>
                  <circle className="track" cx="100" cy="100" r="88" />
                  <circle
                    className="fill"
                    cx="100"
                    cy="100"
                    r="88"
                    strokeDasharray={circumference}
                    strokeDashoffset={dashOffset}
                  />
                </svg>
                <div className="gauge-center">
                  <div>
                    <strong>{progressPct.toFixed(1)}%</strong>
                    <span>training progress</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="hero-meta">
              <span className="chip">
                Steps <b>{m.timesteps.toLocaleString()}</b> /{" "}
                {m.total_timesteps.toLocaleString()}
              </span>
              <span className="chip">
                Uptime <b>{m.uptime}</b>
              </span>
              <span className="chip">
                Episodes <b>{m.episodes}</b>
              </span>
              <span className="chip">
                FPS <b>{m.fps.toFixed(0)}</b>
              </span>
            </div>
          </div>

          <div className="side-stack">
            <MetricCard
              label="Best distance"
              value={m.best_fitness.toFixed(1)}
              hint="metres · all-time"
              color="#34d399"
            />
            <MetricCard
              label="Best reward"
              value={
                m.best_reward >= 0
                  ? `+${m.best_reward.toFixed(2)}`
                  : m.best_reward.toFixed(2)
              }
              hint="episode reward"
              color="#ff8a3d"
            />
            <MetricCard
              label="Mean distance"
              value={m.mean_fitness.toFixed(1)}
              hint="last 10 episodes"
              color="#c4b5fd"
              spark={sparkFitness}
            />
            <MetricCard
              label="Rollout FPS"
              value={m.fps.toFixed(0)}
              hint="env steps / s"
              color="#fbbf24"
            />
            <div className="metric-card glass wide">
              <div className="label">Outcome mix</div>
              <div className="outcomes" style={{ marginTop: 12 }}>
                <Outcome label="Finishes" value={m.finishes} color="#34d399" />
                <Outcome label="Crashes" value={m.crashes} color="#f87171" />
                <Outcome label="Off-track" value={m.off_tracks} color="#ff8a3d" />
              </div>
            </div>
          </div>
        </section>

        <section className="panel glass">
          <h3 className="panel-title">Training controls</h3>
          <div className="controls-row">
            <div className="field">
              <label>ENVS</label>
              <input
                type="number"
                min={1}
                max={8}
                value={cfg.num_envs}
                onChange={(e) =>
                  setCfg((c) => ({ ...c, num_envs: Number(e.target.value) }))
                }
              />
            </div>
            <div className="field">
              <label>TIMESTEPS</label>
              <input
                type="number"
                min={10000}
                step={50000}
                value={cfg.timesteps}
                onChange={(e) =>
                  setCfg((c) => ({ ...c, timesteps: Number(e.target.value) }))
                }
              />
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={cfg.headless}
                onChange={(e) =>
                  setCfg((c) => ({ ...c, headless: e.target.checked }))
                }
              />
              Headless
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={cfg.watch_live}
                onChange={(e) =>
                  setCfg((c) => ({ ...c, watch_live: e.target.checked }))
                }
              />
              Watch env 0
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={cfg.dummy_vec}
                onChange={(e) =>
                  setCfg((c) => ({ ...c, dummy_vec: e.target.checked }))
                }
              />
              Dummy vec
            </label>
          </div>
          <div className="actions">
            <button
              className="btn btn-primary"
              disabled={busy || state.training}
              onClick={() => void startTraining()}
            >
              Start training
            </button>
            <button
              className="btn btn-danger"
              disabled={busy}
              onClick={() => void stopTraining()}
            >
              Stop training
            </button>
            <span className="status">{state.train_status}</span>
          </div>
        </section>

        {tab === "overview" && (
          <>
            <section className="panel glass">
              <div className="section-head">
                <h3>Progress graph</h3>
                <button
                  type="button"
                  className="linkish"
                  onClick={() => setGraphOpen((v) => !v)}
                >
                  {graphOpen ? "Collapse" : "Expand"} · lavender distance · orange reward
                </button>
              </div>
              {graphOpen && (
                <div className="chart-wrap">
                  {chartData.length < 2 ? (
                    <div className="chart-empty">
                      Start training to see live curves…
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={chartData}>
                        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                        <XAxis
                          dataKey="step"
                          tickFormatter={(v: number) => v.toLocaleString()}
                          stroke="#65657a"
                          fontSize={11}
                          tickLine={false}
                          axisLine={false}
                        />
                        <YAxis
                          yAxisId="fit"
                          stroke="#8b7ec8"
                          fontSize={11}
                          tickLine={false}
                          axisLine={false}
                          width={44}
                        />
                        <YAxis
                          yAxisId="rew"
                          orientation="right"
                          stroke="#ff8a3d"
                          fontSize={11}
                          tickLine={false}
                          axisLine={false}
                          width={40}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "rgba(16,16,22,0.92)",
                            border: "1px solid rgba(255,255,255,0.1)",
                            borderRadius: 14,
                            fontSize: 12,
                            backdropFilter: "blur(12px)",
                          }}
                        />
                        <Area
                          yAxisId="fit"
                          type="monotone"
                          dataKey="fitness"
                          stroke="#c4b5fd"
                          fill="url(#areaFill)"
                          strokeWidth={2.5}
                          name="Mean distance"
                          dot={false}
                          isAnimationActive
                        />
                        <Line
                          yAxisId="rew"
                          type="monotone"
                          dataKey="reward"
                          stroke="#ff8a3d"
                          strokeWidth={2.5}
                          name="Mean reward"
                          dot={false}
                          isAnimationActive
                        />
                        <defs>
                          <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#c4b5fd" stopOpacity={0.35} />
                            <stop offset="100%" stopColor="#c4b5fd" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                      </ComposedChart>
                    </ResponsiveContainer>
                  )}
                </div>
              )}
            </section>

            <section className="split">
              <div className="panel glass">
                <div className="section-head">
                  <h3>Last 5 episodes</h3>
                </div>
                <div className="episodes">
                  {Array.from({ length: 5 }).map((_, i) => {
                    const e = state.last5[i];
                    if (!e) {
                      return (
                        <div key={i} className="ep">
                          —
                        </div>
                      );
                    }
                    return (
                      <div
                        key={i}
                        className="ep"
                        style={{ color: OUTCOME_COLOR[e.outcome] ?? "#9a9aaf" }}
                      >
                        {`ep ${String(e.ep).padStart(3, " ")}  dist ${e.fitness.toFixed(1).padStart(5)}m  r ${e.reward.toFixed(1)}  cp ${e.checkpoints}  ${e.outcome}`}
                      </div>
                    );
                  })}
                </div>
              </div>

              <RunsPanel
                runs={runs}
                selectedId={selectedId}
                selected={selected}
                watchPort={state.watch_port}
                watchStatus={state.watch_status}
                busy={busy}
                onSelect={setSelectedId}
                onWatch={(id) => void watchSelected(id)}
                onStopWatch={() =>
                  void api.stopWatch().then((res) =>
                    setState((s) => ({ ...s, watch_status: res.message })),
                  )
                }
              />
            </section>
          </>
        )}

        {tab === "runs" && (
          <RunsPanel
            runs={runs}
            selectedId={selectedId}
            selected={selected}
            watchPort={state.watch_port}
            watchStatus={state.watch_status}
            busy={busy}
            onSelect={setSelectedId}
            onWatch={(id) => void watchSelected(id)}
            onStopWatch={() =>
              void api.stopWatch().then((res) =>
                setState((s) => ({ ...s, watch_status: res.message })),
              )
            }
          />
        )}
      </div>
    </>
  );
}

function MetricCard({
  label,
  value,
  hint,
  color,
  spark,
}: {
  label: string;
  value: string;
  hint: string;
  color: string;
  spark?: { i: number; v: number }[];
}) {
  return (
    <div className="metric-card glass">
      <div className="label">{label}</div>
      <div className="value" style={{ color }}>
        {value}
      </div>
      {spark && spark.length > 1 && (
        <div className="spark">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={spark}>
              <Bar dataKey="v" fill={color} opacity={0.75} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="hint">{hint}</div>
    </div>
  );
}

function Outcome({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="outcome">
      <div className="label">{label}</div>
      <div className="val" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

function RunsPanel({
  runs,
  selectedId,
  selected,
  watchPort,
  watchStatus,
  busy,
  onSelect,
  onWatch,
  onStopWatch,
}: {
  runs: RunRow[];
  selectedId: number | null;
  selected: RunRow | null;
  watchPort: number;
  watchStatus: string;
  busy: boolean;
  onSelect: (id: number) => void;
  onWatch: (id?: number) => void;
  onStopWatch: () => void;
}) {
  return (
    <div className="panel glass">
      <div className="section-head">
        <h3>Best runs</h3>
        <span>Double-click to watch · :{watchPort}</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Tag</th>
              <th>Dist</th>
              <th>Reward</th>
              <th>CPs</th>
              <th>Outcome</th>
              <th>Times</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr
                key={r.id}
                className={selectedId === r.id ? "selected" : ""}
                onClick={() => onSelect(r.id)}
                onDoubleClick={() => onWatch(r.id)}
              >
                <td>{r.label}</td>
                <td>{r.tag}</td>
                <td>{r.dist.toFixed(0)}</td>
                <td>
                  {r.reward >= 0 ? `+${r.reward.toFixed(2)}` : r.reward.toFixed(2)}
                </td>
                <td>{r.checkpoints}</td>
                <td style={{ color: OUTCOME_COLOR[r.outcome] ?? undefined }}>
                  {r.outcome}
                </td>
                <td>{r.times}</td>
              </tr>
            ))}
            {runs.length === 0 && (
              <tr>
                <td colSpan={7} style={{ color: "#65657a" }}>
                  No elite runs yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="detail">
        {selected
          ? `#${selected.id}  gen=${selected.generation}  kind=${selected.kind}  outcome=${selected.outcome}\ndistance=${selected.dist.toFixed(1)} m  reward=${selected.reward.toFixed(2)}  steps=${selected.steps}\ncheckpoint times: ${selected.times}\nmodel: ${selected.model_path}`
          : "Select a run to inspect or Watch."}
      </p>
      <div className="actions">
        <button
          className="btn btn-primary"
          disabled={busy}
          onClick={() => onWatch()}
        >
          Watch selected run
        </button>
        <button className="btn btn-ghost" disabled={busy} onClick={onStopWatch}>
          Stop watch
        </button>
        <span className="status">{watchStatus}</span>
      </div>
    </div>
  );
}
