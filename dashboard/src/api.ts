export type TrainConfig = {
  num_envs: number;
  timesteps: number;
  headless: boolean;
  watch_live: boolean;
  dummy_vec: boolean;
};

export type Episode = {
  ep: number;
  fitness: number;
  reward: number;
  checkpoints: number;
  outcome: string;
};

export type RunRow = {
  id: number;
  label: string;
  tag: string;
  dist: number;
  reward: number;
  checkpoints: number;
  outcome: string;
  times: string;
  generation?: number;
  kind?: string;
  steps?: number;
  model_path?: string;
};

export type AppState = {
  live_on: boolean;
  live_label: string;
  training: boolean;
  train_status: string;
  watch_status: string;
  watching: boolean;
  config: TrainConfig;
  watch_port: number;
  metrics: {
    best_fitness: number;
    best_reward: number;
    mean_fitness: number;
    timesteps: number;
    total_timesteps: number;
    progress: number;
    fps: number;
    uptime_s: number;
    uptime: string;
    episodes: number;
    finishes: number;
    crashes: number;
    off_tracks: number;
  };
  history: {
    mean_fitness: number[];
    mean_reward: number[];
    timesteps: number[];
  };
  last5: Episode[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  state: () => request<AppState>("/api/state"),
  runs: () => request<RunRow[]>("/api/runs"),
  startTraining: (cfg: TrainConfig) =>
    request<{ ok: boolean; message: string }>("/api/training/start", {
      method: "POST",
      body: JSON.stringify(cfg),
    }),
  stopTraining: () =>
    request<{ ok: boolean; message: string }>("/api/training/stop", {
      method: "POST",
    }),
  startWatch: (run_id: number) =>
    request<{ ok: boolean; message: string }>("/api/watch/start", {
      method: "POST",
      body: JSON.stringify({ run_id }),
    }),
  stopWatch: () =>
    request<{ ok: boolean; message: string }>("/api/watch/stop", {
      method: "POST",
    }),
};
