export type CapabilityStatus = {
  controller: { ready: boolean; version: string };
  paths: { project: string; trajectory: string; checkpoints: string };
  sdk: { ready: boolean };
  torch: { ready: boolean };
  engine: { source_available: boolean; revision: string | null; pinned_revision: string | null; synced: boolean; dirty: boolean; built: boolean; url: string; healthy: boolean };
  pixi: { source_available: boolean; built: boolean; build_present: boolean; built_revision: string | null; revision: string | null; pinned_revision: string | null; synced: boolean; dirty: boolean };
  hosted: { api_key_configured: boolean; trajectory_training: boolean; reason: string };
  workflows: Record<string, boolean>;
};

export type AccountStatus = {
  configured: boolean;
  valid: boolean | null;
  reason: string;
};

export type Job = {
  id: string;
  kind: string;
  label: string;
  argv: string[];
  status: 'queued' | 'running' | 'completed' | 'failed' | 'stopped';
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  artifact_path: string | null;
  logs: string[];
  model_id?: string | null;
  worker_slots?: number;
  details?: {
    engineUrl?: string;
    sessionId?: string;
    selectionOrder?: string;
    playerDeck?: { id: string; name: string };
    opponentDeck?: { id: string; name: string };
  } | null;
};

export type DeckPresentation = {
  versionId: string;
  name: string;
  cards: Array<Record<string, unknown>>;
};

export type DeckSummary = {
  id: string;
  deckId?: string;
  name: string;
  creator?: string;
  version: number;
  format?: string;
  colors: string[];
  playableCardCount: number;
};

export type CompetitionSummary = {
  versionId: string;
  name: string;
  status: string;
  format: string;
  timeControl: string;
};

export type LocalDeck = { deckSessionId: string; deckName: string };

export type LocalModel = {
  id: string;
  name: string;
  architecture: string;
  format: "legacy" | "commander";
  description: string;
  createdAt: string;
  runPath: string;
  checkpointPath: string;
  status: string;
  ready: boolean;
  reservePlaytest: boolean;
  source?: "user-trained" | "local-frozen-checkpoint";
  selfPlayAllSeats?: boolean;
  decks: DeckSummary[];
  diskBytes: number;
  weightsBytes: number;
  trainingState: {
    phase?: string | null;
    desiredState?: string | null;
    completedGames: number;
    trainingStep: number;
    parallelGames: number;
    activeGames: number;
    updatedAtUnixMs?: number | null;
  };
};

export type ActiveGame = {
  id: string;
  sessionId: string | null;
  source: 'training' | 'local' | 'league';
  jobId: string | null;
  modelId: string | null;
  modelName: string;
  worker: number;
  status: string;
  mode: string | null;
  decks: Array<string | null>;
  players: number;
  playersState: Array<{
    id?: string;
    name?: string;
    life?: number;
    hasLost?: boolean;
    handCount?: number;
    battlefieldCount?: number;
  }>;
  turnNumber: number | null;
  roundNumber: number | null;
  decisions: number;
  startedAtUnixMs: number | null;
  updatedAtUnixMs: number | null;
  canCancel: boolean;
  watchUrl?: string | null;
};

export type TrainingStatistic = {
  modelId: string;
  modelName: string;
  architecture: 'v11' | 'v12' | 'v13';
  format: string;
  completedGames: number;
  totalRLEpisodes?: number;
  engineCompletedGames?: number;
  trainingStep: number;
  parallelGames: number;
  activeGames: number;
  phase: string;
  desiredState: string;
  trainingElapsedSeconds: number;
  simulationSeconds: number;
  modelTrainingSeconds: number;
  averageGameSeconds: number | null;
  metricWindow: '50' | '200' | '1000' | '5000' | 'all';
  metricRecordCount: number;
  metricPointsReturned: number;
  windowSummary: {
    sampleType: 'games' | 'rl-episodes' | 'synthetic-batches';
    gameMetricsAvailable: boolean;
    engineGameMetrics: boolean;
    attemptedSamples?: number;
    generatedSamples: number;
    usedSamples: number;
    failedSamples: number;
    utilizationRate: number | null;
    coveredTrainingSteps: number;
    generatedDecisions: number;
    usedDecisions: number;
    averageGameSeconds: number | null;
    p50GameSeconds: number | null;
    p95GameSeconds: number | null;
    simulationSeconds: number | null;
    trainingSeconds: number | null;
    collectionWallSeconds: number | null;
    gamesPerHour: number | null;
  };
  latestMetrics: Array<{
    episode: number;
    trainingStep: number;
    loss: number | null;
    policyLoss: number | null;
    valueLoss: number | null;
    entropy: number | null;
    episodeReward: number | null;
    approxKl: number | null;
    clipFraction: number | null;
    dataSource: string | null;
    losses?: Record<string, number>;
    gameDurationSeconds: number | null;
  }>;
  updatedAtUnixMs: number | null;
};

export type ResourcePlan = {
  trainingMatches: number;
  leagueMatches: number;
  localMatches: number;
  gpuMemoryMb: number;
};

export type ResourceWorker = {
  jobId: string;
  modelId: string | null;
  label: string;
  kind: string;
  pids: number[];
  workerSlots: number;
  ramBytes: number;
  gpuBytes: number | null;
  ramPerWorkerEstimate: number;
  gpuPerWorkerEstimate: number | null;
};

export type ResourceSnapshot = {
  system: {
    ramTotalBytes: number;
    ramUsedBytes: number;
    ramAvailableBytes: number;
    gpuTotalBytes: number | null;
    gpuUsedBytes: number | null;
    gpuProcessTelemetry: boolean;
  };
  workers: ResourceWorker[];
  engine: {
    ramBytes: number;
    activeLocalGames: number;
    ramPerGameEstimate: number;
    attribution: string;
  };
};

export type DeckStatistic = {
  modelId: string;
  modelName: string;
  architecture: "v11" | "v12" | "v13";
  deckVersionId: string;
  deckName: string;
  format: string;
  ratingSystem: "plackett-luce" | "engine-self-play";
  mu: number;
  sigma: number;
  ordinal: number;
  rank: number | null;
  matches: number;
  gameWins: number;
  gameLosses: number;
  draws?: number;
  winRate: number | null;
};

export type TrainingSettings = {
  schemaVersion: 'training-platform/v1';
  modelId: string;
  architecture: string;
  savedGameLimit: number;
  checkpointEvery: number;
  evaluationEvery: number;
  evaluationGamesPerScenario: number;
  stages: {
    worldModel: boolean;
    reinforcementLearning: boolean;
    engineEvaluation: boolean;
  };
  targets: {
    worldModelSteps: number;
    reinforcementLearningEpisodes: number;
  };
  curriculum: {
    enabled: boolean;
    adaptive: boolean;
    scenarioIds: string[];
  };
  engineEvaluationSupported: boolean;
  requiresRestart: boolean;
};

export type CurriculumScenario = {
  id: string;
  family: string;
  revision: number;
  label: string;
  description: string;
  difficulty: number;
  objective: string;
  target_round: number | null;
  opening_hand_pool_size: number | null;
  opening_hand_candidate_cards: string[];
  opening_hand_roles: string[][];
  free_casts: boolean;
  miracle_pressure: number;
  legacy_match: boolean;
  deck_hint: string | null;
  opponent_deck_hint: string | null;
  fixed_hand: string[];
  success_cards: string[];
  success_zones: string[];
  success_action_sequence: string[];
  sideboard_target_cards: string[];
  sideboard_cut_cards: string[];
  tags: string[];
};

export type TrainingCurriculum = {
  schemaVersion: 'deepdeck-training-curriculum/v1';
  scenarios: CurriculumScenario[];
};

export type TrainingEvidence = {
  modelId: string;
  status: 'not-connected' | 'not-evaluated' | 'insufficient' | 'measured' | 'verified';
  verdict: string;
  engineEvaluationSupported: boolean;
  magicEvidence: boolean;
  fixedSeeds: boolean;
  evaluationPeriods: number;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  winRate: number | null;
  lower95: number | null;
  upper95: number | null;
  promotions: number;
  trend: Array<{
    period: number;
    trainingStep: number;
    opponent: string | null;
    games: number;
    winRate: number | null;
    lower95: number | null;
    upper95: number | null;
    promoted: boolean;
  }>;
  byDeck: Array<{
    deck: string;
    games: number;
    wins: number;
    losses: number;
    draws: number;
    winRate: number | null;
    lower95: number | null;
    upper95: number | null;
  }>;
  curriculum: {
    games: number;
    evaluationPeriods: number;
    evaluationByScenario: Array<{
      scenarioId: string;
      games: number;
      successes: number;
      successRate: number;
      meanMastery: number;
    }>;
    latestEvaluation: {
      expectedScenarios: number;
      completedScenarios: number;
      successes: number;
      successRate: number;
      meanMastery: number;
      failedScenarios: number;
      recoveredScenarios?: number;
      engineErrors?: number;
    } | null;
    byScenario: Array<{
      scenarioId: string;
      family: string;
      scenarioVersion: number;
      objective: string;
      difficulty: number | null;
      games: number;
      successes: number;
      successRate: number | null;
      averageRound: number | null;
      averageDecisions: number | null;
      averageMilestoneProgress: number | null;
      averageDamageProgress: number | null;
      averageSideboardCards: number | null;
      sideboardTargetCoverage: number | null;
      sideboardCutCoverage: number | null;
      recent: boolean[];
      adaptiveWeight: number | null;
    }>;
  };
};

export type SavedReplaySummary = {
  id: string;
  createdAtUnixMs: number | null;
  episode: number | null;
  seed: number | null;
  matchupId: string | null;
  decks: string[];
  gameMode: string | null;
  gameStatus: string | null;
  roundNumber: number | null;
  winner: string | null;
  durationSeconds: number | null;
  frameCount: number;
  bytes: number;
  saved: boolean;
  viewing: boolean;
};

export type SavedReplay = {
  schemaVersion: 'deepdeck-replay/v1';
  id: string;
  createdAtUnixMs: number;
  metadata: Record<string, unknown>;
  frames: Array<Record<string, unknown>>;
};

export type AgentTrainingParameter = {
  key: string;
  label: string;
  kind: 'integer' | 'number' | 'boolean' | 'string' | 'choice';
  default: string | number | boolean | null;
  description: string;
  min: number | null;
  max: number | null;
  step: number | null;
  choices: Array<string | number | boolean | null>;
  apply: 'live' | 'next-phase' | 'next-run';
};

export type AgentTrainingContract = {
  schemaVersion: 'deepdeck-agent-training/v1';
  agentId: string;
  displayName: string;
  ownsCurriculum: boolean;
  ownsPromotionCriteria: boolean;
  proposesExperiments: boolean;
  replaySupport: 'none' | 'summary' | 'full';
  phases: Array<{
    id: string;
    label: string;
    description: string;
    controls: Array<'start' | 'pause' | 'resume' | 'stop' | 'evaluate' | 'checkpoint'>;
    metrics: Array<{
      key: string;
      label: string;
      kind: 'loss' | 'reward' | 'rate' | 'count' | 'duration' | 'scalar';
      direction: 'minimize' | 'maximize' | 'target' | 'neutral';
      unit: string;
      description: string;
      group: string;
    }>;
    parameters: AgentTrainingParameter[];
  }>;
};

export type AgentTrainingPublication = {
  contract: AgentTrainingContract;
  state: Record<string, unknown> | null;
  control: Record<string, unknown> | null;
  parameterValues: Record<string, string | number | boolean | null>;
  metrics: Array<{
    schemaVersion: string;
    agentId: string;
    phase: string;
    step: number;
    metrics: Record<string, number>;
    recordedAtUnixMs: number | null;
    context?: Record<string, unknown>;
  }>;
};

type Page<T> = {
  items: T[];
  pagination?: {
    page?: number;
    totalPages?: number;
    total_pages?: number;
    hasNextPage?: boolean;
    has_next_page?: boolean;
  };
};

let sessionToken = '';

async function json<T>(response: Response): Promise<T> {
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  return body as T;
}

async function token(forceRefresh = false): Promise<string> {
  if (forceRefresh) {
    sessionToken = '';
  }
  if (!sessionToken) {
    sessionToken = (await json<{ token: string }>(await fetch('/api/v1/session'))).token;
  }
  return sessionToken;
}

async function authorizedFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set('X-DeepDeck-Token', await token());
  let response = await fetch(input, { ...init, headers });
  if (response.status !== 403) return response;

  const body = await response.clone().json().catch(() => null) as { detail?: string } | null;
  if (body?.detail !== 'Invalid local session token.') return response;

  const retryHeaders = new Headers(init.headers);
  retryHeaders.set('X-DeepDeck-Token', await token(true));
  response = await fetch(input, { ...init, headers: retryHeaders });
  return response;
}

export async function loadStatus(): Promise<CapabilityStatus> {
  return json<CapabilityStatus>(await fetch('/api/v1/status'));
}

export async function loadJobs(): Promise<Job[]> {
  return json<Job[]>(await fetch('/api/v1/jobs'));
}

export async function loadModels(): Promise<LocalModel[]> {
  return (await json<Page<LocalModel>>(await fetch('/api/v1/models'))).items;
}

export async function loadAccountStatus(): Promise<AccountStatus> {
  return json<AccountStatus>(await fetch('/api/v1/account/status'));
}

export async function createModel(payload: Record<string, unknown>): Promise<LocalModel> {
  return json<LocalModel>(
    await authorizedFetch('/api/v1/models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  );
}

export async function deleteModel(modelId: string): Promise<{ id: string; name: string; deleted: boolean; reclaimedBytes: number }> {
  return json(
    await authorizedFetch(`/api/v1/models/${encodeURIComponent(modelId)}`, {
      method: 'DELETE',
    }),
  );
}

export async function loadResources(): Promise<ResourceSnapshot> {
  return json<ResourceSnapshot>(await fetch('/api/v1/resources'));
}

export async function loadModelResources(modelId: string): Promise<ResourcePlan> {
  return json<ResourcePlan>(
    await fetch(`/api/v1/models/${encodeURIComponent(modelId)}/resources`),
  );
}

export async function saveModelResources(
  modelId: string,
  plan: ResourcePlan,
): Promise<ResourcePlan> {
  return json<ResourcePlan>(
    await authorizedFetch(`/api/v1/models/${encodeURIComponent(modelId)}/resources`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(plan),
    }),
  );
}

export async function loadDeckStatistics(): Promise<DeckStatistic[]> {
  return (await json<Page<DeckStatistic>>(await fetch('/api/v1/statistics/decks'))).items;
}

export async function loadTrainingStatistics(window: TrainingStatistic['metricWindow'] = '200'): Promise<TrainingStatistic[]> {
  const query = new URLSearchParams({ window });
  return (await json<Page<TrainingStatistic>>(await fetch(`/api/v1/statistics/training?${query}`))).items;
}

export async function loadTrainingSettings(modelId: string): Promise<TrainingSettings> {
  return json<TrainingSettings>(
    await fetch(`/api/v1/models/${encodeURIComponent(modelId)}/training-settings`),
  );
}

export async function loadTrainingCurriculum(): Promise<TrainingCurriculum> {
  return json<TrainingCurriculum>(await fetch('/api/v1/training/curriculum'));
}

export async function saveTrainingSettings(
  modelId: string,
  settings: TrainingSettings,
): Promise<TrainingSettings> {
  return json<TrainingSettings>(
    await authorizedFetch(`/api/v1/models/${encodeURIComponent(modelId)}/training-settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    }),
  );
}

export async function loadTrainingEvidence(modelId: string): Promise<TrainingEvidence> {
  return json<TrainingEvidence>(
    await fetch(`/api/v1/models/${encodeURIComponent(modelId)}/evidence`),
  );
}

export async function loadAgentTrainingContract(modelId: string): Promise<AgentTrainingPublication | null> {
  const response = await fetch(`/api/v1/models/${encodeURIComponent(modelId)}/training-contract`);
  if (response.status === 404) return null;
  if (!response.headers.get('Content-Type')?.includes('application/json')) return null;
  const publication = await json<AgentTrainingPublication | unknown>(response);
  if (!publication || typeof publication !== 'object' || !('contract' in publication)) return null;
  return publication as AgentTrainingPublication;
}

export async function saveAgentTrainingControl(
  modelId: string,
  payload: { phase: string; action: string; parameters: Record<string, unknown> },
): Promise<Record<string, unknown>> {
  return json(
    await authorizedFetch(`/api/v1/models/${encodeURIComponent(modelId)}/training-control`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  );
}

export async function loadSavedReplays(modelId: string): Promise<SavedReplaySummary[]> {
  return (
    await json<Page<SavedReplaySummary>>(
      await fetch(`/api/v1/models/${encodeURIComponent(modelId)}/replays`),
    )
  ).items;
}

export async function loadSavedReplay(
  modelId: string,
  replayId: string,
): Promise<SavedReplay> {
  return json<SavedReplay>(
    await fetch(`/api/v1/models/${encodeURIComponent(modelId)}/replays/${encodeURIComponent(replayId)}`),
  );
}

export async function acquireSavedReplay(
  modelId: string,
  replayId: string,
): Promise<{ replay: SavedReplay; leaseId: string; expiresAtUnixMs: number; saved: boolean }> {
  const response = await authorizedFetch(`/api/v1/models/${encodeURIComponent(modelId)}/replays/${encodeURIComponent(replayId)}/leases`, {
    method: 'POST',
  });
  if (response.status === 404 || !response.headers.get('Content-Type')?.includes('application/json')) {
    return {
      replay: await loadSavedReplay(modelId, replayId),
      leaseId: '',
      expiresAtUnixMs: 0,
      saved: false,
    };
  }
  return json(response);
}

export async function renewSavedReplayLease(
  modelId: string,
  replayId: string,
  leaseId: string,
): Promise<{ expiresAtUnixMs: number }> {
  return json(
    await authorizedFetch(`/api/v1/models/${encodeURIComponent(modelId)}/replays/${encodeURIComponent(replayId)}/leases/${encodeURIComponent(leaseId)}`, {
      method: 'PUT',
    }),
  );
}

export async function releaseSavedReplayLease(
  modelId: string,
  replayId: string,
  leaseId: string,
): Promise<{ released: boolean }> {
  return json(
    await authorizedFetch(`/api/v1/models/${encodeURIComponent(modelId)}/replays/${encodeURIComponent(replayId)}/leases/${encodeURIComponent(leaseId)}`, {
      method: 'DELETE',
    }),
  );
}

export async function saveReplayForever(
  modelId: string,
  replayId: string,
): Promise<{ saved: boolean }> {
  return json(
    await authorizedFetch(`/api/v1/models/${encodeURIComponent(modelId)}/replays/${encodeURIComponent(replayId)}/save`, {
      method: 'POST',
    }),
  );
}

export async function loadActiveGames(): Promise<ActiveGame[]> {
  return (await json<Page<ActiveGame>>(await fetch('/api/v1/games'))).items;
}

export async function stopGame(gameId: string): Promise<ActiveGame> {
  return json<ActiveGame>(
    await authorizedFetch(`/api/v1/games/${encodeURIComponent(gameId)}/stop`, {
      method: 'POST',
    }),
  );
}

export async function saveApiKey(apiKey: string): Promise<{ configured: boolean }> {
  return json<{ configured: boolean }>(
    await authorizedFetch('/api/v1/settings/api-key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey }),
    }),
  );
}

export async function searchDeckPage(search: string, format: string, page = 1): Promise<Page<DeckSummary>> {
  const query = new URLSearchParams({ search, format, page: String(page) });
  return json<Page<DeckSummary>>(await fetch(`/api/v1/catalog/decks?${query}`));
}

export async function updateModel(
  modelId: string,
  payload: Pick<LocalModel, 'name' | 'decks' | 'reservePlaytest' | 'selfPlayAllSeats'>,
): Promise<LocalModel> {
  return json<LocalModel>(
    await authorizedFetch(`/api/v1/models/${encodeURIComponent(modelId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  );
}

export async function searchDecks(search: string, format: string): Promise<DeckSummary[]> {
  return (await searchDeckPage(search, format)).items;
}

export async function loadAllFormatDecks(format: string): Promise<DeckSummary[]> {
  const decks = new Map<string, DeckSummary>();
  for (let page = 1; page <= 100; page += 1) {
    const result = await searchDeckPage("", format, page);
    result.items.forEach((deck) => decks.set(deck.id, deck));
    const pagination = result.pagination;
    const totalPages = pagination?.totalPages ?? pagination?.total_pages;
    const hasNextPage = pagination?.hasNextPage ?? pagination?.has_next_page;
    if (
      result.items.length === 0 ||
      (typeof totalPages === "number" && page >= totalPages) ||
      hasNextPage === false ||
      (!pagination && result.items.length < 12)
    ) {
      break;
    }
  }
  return [...decks.values()];
}

export async function downloadDeck(versionId: string): Promise<{ versionId: string; name: string; format: string; cardCount: number; rawCardCount: number; path: string }> {
  return json(await authorizedFetch(`/api/v1/catalog/decks/${encodeURIComponent(versionId)}/download`, {
    method: 'POST',
  }));
}

export async function loadTrainingDeckPool(): Promise<{ decks: DeckSummary[] }> {
  return json(await fetch('/api/v1/training/deck-pool'));
}

export async function saveTrainingDeckPool(decks: DeckSummary[]): Promise<{ decks: DeckSummary[] }> {
  return json(await authorizedFetch('/api/v1/training/deck-pool', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decks }),
  }));
}

export async function loadCompetitions(): Promise<CompetitionSummary[]> {
  return (await json<Page<CompetitionSummary>>(await fetch('/api/v1/catalog/competitions'))).items;
}

export async function loadLocalDecks(format: string, engineUrl: string): Promise<LocalDeck[]> {
  const query = new URLSearchParams({ format, engine_url: engineUrl });
  return json<LocalDeck[]>(await fetch(`/api/v1/catalog/local-decks?${query}`));
}

export async function startJob(payload: Record<string, unknown>): Promise<Job> {
  return json<Job>(
    await authorizedFetch('/api/v1/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  );
}

export async function stopJob(jobId: string): Promise<Job> {
  return json<Job>(
    await authorizedFetch(`/api/v1/jobs/${jobId}/stop`, {
      method: 'POST',
    }),
  );
}
