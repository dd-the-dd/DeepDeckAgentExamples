import {
  Component,
  FormEvent,
  Suspense,
  lazy,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import {
  createModel,
  deleteModel,
  updateModel,
  loadAccountStatus,
  loadCompetitions,
  loadActiveGames,
  loadDeckStatistics,
  loadJobs,
  loadAllFormatDecks,
  loadModelResources,
  loadModels,
  loadResources,
  loadStatus,
  loadTrainingStatistics,
  loadTrainingEvidence,
  loadTrainingSettings,
  loadTrainingCurriculum,
  loadAgentTrainingContract,
  loadSavedReplays,
  acquireSavedReplay,
  releaseSavedReplayLease,
  renewSavedReplayLease,
  saveReplayForever,
  downloadDeck,
  loadTrainingDeckPool,
  saveApiKey,
  saveModelResources,
  saveTrainingDeckPool,
  saveTrainingSettings,
  saveAgentTrainingControl,
  searchDecks,
  startJob,
  stopGame,
  stopJob,
  type CapabilityStatus,
  type AccountStatus,
  type ActiveGame,
  type CompetitionSummary,
  type DeckSummary,
  type DeckStatistic,
  type Job,
  type LocalModel,
  type ResourcePlan,
  type ResourceSnapshot,
  type TrainingStatistic,
  type TrainingEvidence,
  type TrainingSettings,
  type TrainingCurriculum,
  type SavedReplay,
  type SavedReplaySummary,
  type AgentTrainingPublication,
} from "./api";
import { workflowBlockers, type Workflow } from "./readiness";
import AsyncActionButton from "./AsyncActionButton";

const LocalPixiTable = lazy(() => import("./LocalPixiTable"));
const LocalPixiReplay = lazy(() => import("./LocalPixiReplay"));

class PixiErrorBoundary extends Component<
  { children: ReactNode; onClose: () => void },
  { error: string }
> {
  state = { error: "" };

  static getDerivedStateFromError(reason: unknown) {
    return {
      error: reason instanceof Error ? reason.message : "Pixi could not open this table.",
    };
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <section className="pixi-recovery" role="alert">
        <span className="eyebrow">Local table recovery</span>
        <h2>Pixi could not display this game</h2>
        <p>{this.state.error}</p>
        <button className="primary" type="button" onClick={this.props.onClose}>
          Return to the workbench
        </button>
      </section>
    );
  }
}

type Page =
  | "overview"
  | "train"
  | "jobs"
  | "playtest"
  | "compete"
  | "statistics"
  | "representation"
  | "models";

const pages: Array<{ id: Page; label: string; glyph: string }> = [
  { id: "train", label: "Agent configuration", glyph: "01" },
  { id: "jobs", label: "Jobs", glyph: "02" },
  { id: "playtest", label: "Play against your AI", glyph: "03" },
  { id: "statistics", label: "Training statistics", glyph: "04" },
];

const pageHeadings: Record<Page, string> = {
  overview: "What do you want to do?",
  train: "Agent configuration",
  jobs: "Jobs and live games",
  playtest: "Play against your AI",
  compete: "Play in the League",
  statistics: "Training statistics",
  representation: "Understand the tensor",
  models: "Choose a model family",
};

const leagueUrl = "https://staging.deepdeckleague.com";
const leagueLogoUrl = `${leagueUrl}/deep-deck-league-logo.png`;
const patreonUrl = "https://www.patreon.com/DeepDeckLeague";

const workflowCopy: Record<
  Workflow,
  { title: string; kicker: string; description: string }
> = {
  "local-training": {
    title: "Train an agent",
    kicker: "Recommended first step",
    description:
      "Choose a V11 or V12 model, format, and training deck pool before launching a run.",
  },
  "online-training": {
    title: "Train online",
    kicker: "Hosted opponents",
    description:
      "Connect your account when the versioned hosted trajectory contract is available.",
  },
  "local-playtest": {
    title: "Test an agent locally",
    kicker: "I want to test behavior",
    description:
      "Prepare Engine and Pixi, choose two decks, then launch a local behavior test.",
  },
  matchmaking: {
    title: "Send an agent to the League",
    kicker: "I am ready to compete",
    description:
      "Connect your account key, find a legal deck by name, and join matchmaking.",
  },
};

function StatusDot({ ready, label }: { ready: boolean; label: string }) {
  return (
    <span className={`status-chip ${ready ? "ready" : "missing"}`}>
      <i />
      {label}
    </span>
  );
}

function PatreonMark() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M14.82 2.4a7.18 7.18 0 1 0 0 14.36 7.18 7.18 0 0 0 0-14.36ZM2.4 21.6h3.5V2.4H2.4v19.2Z" />
    </svg>
  );
}

function WorkflowCard({
  workflow,
  status,
  onSelect,
}: {
  workflow: Workflow;
  status: CapabilityStatus | null;
  onSelect: (workflow: Workflow) => void;
}) {
  const blockers = workflowBlockers(status, workflow);
  const copy = workflowCopy[workflow];
  return (
    <button
      className={`workflow-card${workflow === "local-training" ? " recommended" : ""}`}
      type="button"
      onClick={() => onSelect(workflow)}
    >
      <span className="card-kicker">{copy.kicker}</span>
      <span className="card-title">
        {copy.title}
        <b aria-hidden="true">→</b>
      </span>
      <span className="card-copy">{copy.description}</span>
      <span className={`card-state ${blockers.length ? "blocked" : "ready"}`}>
        {blockers.length ? "Guided setup included" : "Ready to continue"}
      </span>
    </button>
  );
}

function WorkflowJourney({
  active,
  steps,
}: {
  active: number;
  steps: Array<{ label: string; detail: string }>;
}) {
  return (
    <ol className="workflow-journey" aria-label="Workflow progress">
      {steps.map((step, index) => {
        const number = index + 1;
        const state =
          number < active
            ? "complete"
            : number === active
              ? "active"
              : "upcoming";
        return (
          <li className={state} key={step.label}>
            <span>{number < active ? "✓" : number}</span>
            <div>
              <strong>{step.label}</strong>
              <small>{step.detail}</small>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function WorkspaceSummary({ status, account }: { status: CapabilityStatus | null; account: AccountStatus | null }) {
  return (
    <section className="workspace-summary" aria-label="Workspace readiness">
      <div>
        <span className="eyebrow">Your workspace</span>
        <strong>
          {status?.torch.ready
            ? "You can train an agent now."
            : "One training dependency is missing."}
        </strong>
        <small>
          Engine and Pixi are only needed when you run a local behavior test.
        </small>
      </div>
      <div className="workspace-status">
        <StatusDot ready={Boolean(status?.torch.ready)} label="Training" />
        <StatusDot
          ready={Boolean(status?.engine.healthy && status?.pixi.built)}
          label="Local play"
        />
        <StatusDot
          ready={account?.valid === true}
          label="League account"
        />
      </div>
    </section>
  );
}

function AccountSetup({ status, account, refresh }: { status: CapabilityStatus | null; account: AccountStatus | null; refresh: () => void }) {
  const [apiKey, setApiKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [replacing, setReplacing] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    try {
      await saveApiKey(apiKey);
      setApiKey("");
      setReplacing(false);
      setMessage("API key saved on this computer.");
      refresh();
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "Unable to save the API key.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel account-setup">
      <div>
        <span className="eyebrow">Application setup</span>
        <h2>Connect your League account</h2>
        <p>The key stays on this computer and is never displayed again after saving.</p>
      </div>
      {!status || !account ? (
        <div className="notice" role="status"><strong>Checking this computer…</strong><span>Looking for a previously saved League account key.</span></div>
      ) : account.valid === true && !replacing ? (
        <div className="notice success account-connected" role="status">
          <div><strong>Account connected</strong><span>Saved in this workbench's private local data folder and restored after restart.</span></div>
          <button type="button" onClick={() => setReplacing(true)}>Replace key</button>
        </div>
      ) : (
        <form onSubmit={submit}>
          {account.configured && <div className="notice warning"><strong>{account.valid === false ? "Saved key rejected" : "League unavailable"}</strong><span>{account.reason}</span></div>}
          <label>
            Deep Deck League API key
            <input type="password" autoComplete="off" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="ddl_agent_…" />
          </label>
          <AsyncActionButton className="primary" type="submit" loading={busy} loadingLabel="Saving API key…" disabled={apiKey.trim().length < 24}>Save API key</AsyncActionButton>
          {account.valid === true && <button type="button" onClick={() => { setReplacing(false); setApiKey(""); }}>Cancel</button>}
        </form>
      )}
      {message && <p className={message.startsWith("API key saved") ? "form-success" : "form-error"} role="status">{message}</p>}
    </section>
  );
}

function LockedNextStep() {
  return (
    <section className="panel locked-step" aria-disabled="true">
      <span>Next</span>
      <div>
        <h2>Choose the matchup</h2>
        <p>
          This step opens automatically as soon as Engine and Pixi are ready.
        </p>
      </div>
    </section>
  );
}

function shortRevision(revision: string | null) {
  return revision ? revision.slice(0, 8) : "not installed";
}

const dependencyKinds = new Set([
  "dependency.stack.prepare",
  "dependency.engine.start",
  "dependency.pixi.prepare",
  "dependency.sync",
]);

function dependencyActivity(
  kind: string,
  status: CapabilityStatus | null,
  job?: Job,
) {
  const engine = status?.engine;
  const pixi = status?.pixi;
  if (kind === "dependency.stack.prepare") {
    if (!engine?.synced || !pixi?.synced) {
      return {
        title: "Syncing compatible Engine and Pixi sources",
        detail: "Retrieving the reviewed revisions selected for this Learner release.",
        step: 1,
      };
    }
    if (!pixi.built) {
      return {
        title: "Building the Pixi visual client",
        detail: "Installing locked packages and preparing the local game table.",
        step: 2,
      };
    }
    if (!engine?.built) {
      return {
        title: "Compiling DeepDeckEngine",
        detail: "Rust is building the local rules server. The first build can take a few minutes.",
        step: 3,
      };
    }
    return {
      title: engine.healthy ? "Verifying the local game stack" : "Starting DeepDeckEngine",
      detail: engine.healthy
        ? "Checking that Engine and Pixi are ready for local games."
        : "Starting the freshly built rules server on this computer.",
      step: 3,
    };
  }
  if (kind === "dependency.engine.start") {
    return {
      title: engine?.built ? "Starting DeepDeckEngine" : "Building DeepDeckEngine",
      detail: engine?.built
        ? "Starting the local rules server and checking its health."
        : "Compiling the Rust rules server before starting it locally.",
      step: 3,
    };
  }
  if (kind === "dependency.pixi.prepare") {
    return {
      title: "Building the Pixi visual client",
      detail: "Preparing the full-screen local table and its browser assets.",
      step: 2,
    };
  }
  const syncingPixi = kind === "dependency.sync.pixi" || job?.label.toLowerCase().includes("pixi");
  return {
    title: syncingPixi ? "Syncing DeepDeckPixi" : "Syncing DeepDeckEngine",
    detail: "Switching to the compatible reviewed revision without overwriting local work.",
    step: 1,
  };
}

function DependencyLoader({
  kind,
  status,
  job,
}: {
  kind: string;
  status: CapabilityStatus | null;
  job?: Job;
}) {
  const activity = dependencyActivity(kind, status, job);
  const latestLog = job?.logs.filter((line) => line.trim()).at(-1);
  return (
    <aside
      className="operation-loader"
      role="status"
      aria-label={activity.title}
      aria-live="polite"
    >
      <div className="operation-loader-visual" aria-hidden="true">
        <i />
        <i />
        <span>DD</span>
      </div>
      <div className="operation-loader-copy">
        <span className="eyebrow">Working locally · safe to leave this tab open</span>
        <strong>{activity.title}</strong>
        <p>{activity.detail}</p>
        <div className="operation-loader-track" aria-hidden="true"><i /></div>
        <small>{latestLog || (job?.status === "queued" ? "Waiting for the local worker…" : "Preparing the next step…")}</small>
      </div>
      <ol className="operation-loader-stages" aria-label="Setup progress">
        {["Sources", "Pixi", "Engine"].map((label, index) => {
          const number = index + 1;
          return (
            <li
              className={number < activity.step ? "complete" : number === activity.step ? "active" : ""}
              key={label}
            >
              <span>{number < activity.step ? "✓" : number}</span>
              {label}
            </li>
          );
        })}
      </ol>
    </aside>
  );
}

function DependencyPanel({
  status,
  jobs,
  refresh,
}: {
  status: CapabilityStatus | null;
  jobs: Job[];
  refresh: () => void | Promise<void>;
}) {
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const engine = status?.engine;
  const pixi = status?.pixi;
  const stackReady = Boolean(engine?.healthy && pixi?.built);
  const stackJob = jobs.find(
    (job) =>
      job.kind === "dependency.stack.prepare" &&
      ["queued", "running"].includes(job.status),
  );
  const stackFailure = jobs.find(
    (job) => job.kind === "dependency.stack.prepare" && job.status === "failed",
  );
  const activeDependencyJob = jobs.find(
    (job) => dependencyKinds.has(job.kind) && ["queued", "running"].includes(job.status),
  );
  const busyKind = {
    stack: "dependency.stack.prepare",
    engine: "dependency.engine.start",
    pixi: "dependency.pixi.prepare",
    "engine-sync": "dependency.sync.engine",
    "pixi-sync": "dependency.sync.pixi",
  }[busy];
  const activeKind = activeDependencyJob?.kind ?? busyKind;
  const dirty = Boolean(engine?.dirty || pixi?.dirty);

  async function run(action: string, payload: Record<string, unknown>) {
    setBusy(action);
    setError("");
    try {
      await startJob(payload);
      await refresh();
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Unable to start dependency task.",
      );
    } finally {
      setBusy("");
    }
  }

  async function startStack() {
    setBusy("stack");
    setError("");
    try {
      await startJob({ kind: "dependency.stack.prepare" });
      await refresh();
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Unable to start the local stack.",
      );
    } finally {
      setBusy("");
    }
  }

  return (
    <section className={`panel dependency-panel${stackReady ? " ready" : ""}`}>
      <div className="section-heading dependency-heading">
        <div>
          <span className="eyebrow">Step 1 · automatic setup</span>
          <h2>Prepare the local game table</h2>
          <p>
            One action retrieves the pinned Engine and Pixi sources, builds what
            is needed, and starts Engine.
          </p>
        </div>
        <AsyncActionButton
          className="primary"
          type="button"
          loading={Boolean(busy) || Boolean(activeDependencyJob)}
          loadingLabel={stackReady ? "Verifying local stack…" : "Setting up local stack…"}
          disabled={!status || dirty}
          onClick={() => void startStack()}
        >
          {stackReady
            ? stackJob || busy === "stack"
              ? "Verifying…"
              : "Verify & repair Engine + Pixi"
            : stackJob || busy === "stack"
              ? "Setting up…"
              : "Set up Engine + Pixi"}
        </AsyncActionButton>
      </div>
      {activeKind && (
        <DependencyLoader
          kind={activeKind}
          status={status}
          job={activeDependencyJob}
        />
      )}
      <ol className="setup-progress">
        <li
          className={
            engine?.synced && pixi?.synced
              ? "complete"
              : stackJob
                ? "active"
                : ""
          }
        >
          <span>{engine?.synced && pixi?.synced ? "✓" : "1"}</span>
          <div>
            <strong>Get compatible sources</strong>
            <small>Uses the reviewed commits pinned by DeepDeckLearner.</small>
          </div>
        </li>
        <li
          className={
            pixi?.built
              ? "complete"
              : stackJob && engine?.synced && pixi?.synced
                ? "active"
                : ""
          }
        >
          <span>{pixi?.built ? "✓" : "2"}</span>
          <div>
            <strong>Build the visual client</strong>
            <small>Installs locked packages and prepares DeepDeckPixi.</small>
          </div>
        </li>
        <li
          className={
            engine?.healthy
              ? "complete"
              : stackJob && pixi?.built
                ? "active"
                : ""
          }
        >
          <span>{engine?.healthy ? "✓" : "3"}</span>
          <div>
            <strong>Start the rules engine</strong>
            <small>
              Builds once when necessary, then listens only on this computer.
            </small>
          </div>
        </li>
      </ol>
      {dirty && (
        <p className="form-error" role="alert">
          Engine or Pixi contains local changes. Commit or move those changes
          before automatic setup; nothing will be overwritten.
        </p>
      )}
      {(error || stackFailure?.logs.at(-1)) && (
        <p className="form-error" role="alert">
          {error || stackFailure?.logs.at(-1)}
        </p>
      )}
      <details className="runtime-details">
        <summary>Technical details and individual controls</summary>
        <div className="dependency-list">
          <article>
            <div className="dependency-icon engine">E</div>
            <div className="dependency-copy">
              <span>DeepDeckEngine</span>
              <strong>
                {engine?.healthy
                  ? "Running"
                  : engine?.synced
                    ? engine.built
                      ? "Ready to start"
                      : "Build required"
                    : engine?.source_available
                      ? "Update available"
                      : "Not installed"}
              </strong>
              <small>
                Current {shortRevision(engine?.revision ?? null)} · compatible{" "}
                {shortRevision(engine?.pinned_revision ?? null)}
              </small>
              {engine?.dirty && (
                <em>Local changes prevent automatic synchronization.</em>
              )}
            </div>
            <div className="dependency-actions">
              <button
                type="button"
                disabled={
                  !engine?.source_available ||
                  !engine.synced ||
                  engine.healthy ||
                  Boolean(busy) ||
                  Boolean(activeDependencyJob)
                }
                onClick={() =>
                  void run("engine", { kind: "dependency.engine.start" })
                }
              >
                {engine?.healthy
                  ? "Running"
                  : engine?.built
                    ? "Start"
                    : "Build & start"}
              </button>
              <button
                type="button"
                disabled={
                  engine?.synced ||
                  engine?.dirty ||
                  Boolean(busy) ||
                  Boolean(activeDependencyJob)
                }
                onClick={() =>
                  void run("engine-sync", {
                    kind: "dependency.sync",
                    dependency: "engine",
                  })
                }
              >
                {engine?.synced ? "Up to date" : "Sync version"}
              </button>
            </div>
          </article>
          <article>
            <div className="dependency-icon pixi">P</div>
            <div className="dependency-copy">
              <span>DeepDeckPixi</span>
              <strong>
                {pixi?.built
                  ? "Ready"
                  : pixi?.synced
                    ? "Build required"
                    : pixi?.source_available
                      ? "Update available"
                      : "Not installed"}
              </strong>
              <small>
                Current {shortRevision(pixi?.revision ?? null)} · compatible{" "}
                {shortRevision(pixi?.pinned_revision ?? null)}
              </small>
              {pixi?.dirty && (
                <em>Local changes prevent automatic synchronization.</em>
              )}
            </div>
            <div className="dependency-actions">
              <button
                type="button"
                disabled={
                  !pixi?.source_available ||
                  !pixi.synced ||
                  pixi.built ||
                  Boolean(busy) ||
                  Boolean(activeDependencyJob)
                }
                onClick={() =>
                  void run("pixi", { kind: "dependency.pixi.prepare" })
                }
              >
                {pixi?.built ? "Ready" : "Prepare"}
              </button>
              <button
                type="button"
                disabled={
                  pixi?.synced ||
                  pixi?.dirty ||
                  Boolean(busy) ||
                  Boolean(activeDependencyJob)
                }
                onClick={() =>
                  void run("pixi-sync", {
                    kind: "dependency.sync",
                    dependency: "pixi",
                  })
                }
              >
                {pixi?.synced ? "Up to date" : "Sync version"}
              </button>
            </div>
          </article>
        </div>
        <p className="dependency-note">
          Individual controls are recovery tools. Normal setup should only
          require the main button above.
        </p>
      </details>
    </section>
  );
}

export function TrainingForm({
  status,
  refresh,
}: {
  status: CapabilityStatus | null;
  refresh: () => void;
}) {
  const [source, setSource] = useState<"smoke" | "dataset">("smoke");
  const [model, setModel] = useState("v12");
  const [dataset, setDataset] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [epochs, setEpochs] = useState(3);
  const [learningRate, setLearningRate] = useState(0.0003);
  const [device, setDevice] = useState("cuda");
  const [error, setError] = useState("");
  const [formats, setFormats] = useState<string[]>(["legacy"]);
  const [decks, setDecks] = useState<DeckSummary[]>([]);
  const [selectedDecks, setSelectedDecks] = useState<string[]>([]);
  const [deckError, setDeckError] = useState("");
  const [loadingDecks, setLoadingDecks] = useState(false);
  const blockers = workflowBlockers(status, "local-training");
  const deckTrainingAvailable = Boolean(status?.workflows.training_decks);

  useEffect(() => {
    if (!status?.hosted.api_key_configured) {
      setDecks([]);
      return;
    }
    let active = true;
    setLoadingDecks(true);
    setDeckError("");
    void Promise.all(formats.map((format) => searchDecks("", format)))
      .then((groups) => {
        if (!active) return;
        const items = groups.flat();
        setDecks(items);
        setSelectedDecks((current) =>
          current.filter((id) => items.some((deck) => deck.id === id)),
        );
      })
      .catch((reason) => {
        if (!active) return;
        setDecks([]);
        setDeckError(
          reason instanceof Error
            ? reason.message
            : "Unable to load training decks from Deep Deck League.",
        );
      })
      .finally(() => {
        if (active) setLoadingDecks(false);
      });
    return () => {
      active = false;
    };
  }, [formats, status?.hosted.api_key_configured]);

  function toggleFormat(format: string) {
    setFormats((current) =>
      current.includes(format)
        ? current.length === 1
          ? current
          : current.filter((item) => item !== format)
        : [...current, format],
    );
    setSelectedDecks([]);
  }

  function toggleDeck(deckVersionId: string) {
    setSelectedDecks((current) =>
      current.includes(deckVersionId)
        ? current.filter((id) => id !== deckVersionId)
        : [...current, deckVersionId],
    );
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      await startJob({
        kind: source === "smoke" ? "training.smoke" : "training.dataset",
        model,
        dataset,
        epochs,
        learning_rate: learningRate,
        device,
      });
      refresh();
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Unable to start training.",
      );
    }
  }

  return (
    <form className="panel configure" onSubmit={submit}>
      <div className="section-heading">
        <div>
          <span className="eyebrow">Steps 1–2 · configure the training run</span>
          <h2>Choose a model and its training decks</h2>
          <p className="section-lead">
            Select the actual pool the agent should learn from before starting.
            V12 is the recommended two-player model.
          </p>
        </div>
        <span className="step">01</span>
      </div>
      <div className="field-grid beginner-fields">
        <label>
          Model
          <select
            value={model}
            onChange={(event) => setModel(event.target.value)}
          >
            <option value="v12">V12 · two-player</option>
            <option value="v11">V11 · multiplayer</option>
          </select>
        </label>
        <fieldset className="format-picker">
          <legend>Formats</legend>
          <div>
            {["legacy", "commander"].map((format) => (
              <button
                key={format}
                type="button"
                className={formats.includes(format) ? "selected" : ""}
                aria-pressed={formats.includes(format)}
                onClick={() => toggleFormat(format)}
              >
                {format[0].toUpperCase() + format.slice(1)}
              </button>
            ))}
          </div>
          <small>Choose one or more formats.</small>
        </fieldset>
        <div className="automatic-setting">
          <span>Compute</span>
          <strong>GPU preferred</strong>
          <small>Automatically uses CPU when CUDA is unavailable.</small>
        </div>
      </div>
      <fieldset className="training-pool">
        <legend>Training pool</legend>
        <div className="training-pool-heading">
          <div>
            <strong>Select one or more decks</strong>
            <small>
              {selectedDecks.length
                ? `${selectedDecks.length} deck${selectedDecks.length === 1 ? "" : "s"} selected`
                : "No deck selected — training cannot start"}
            </small>
          </div>
        </div>
        {!status?.hosted.api_key_configured ? (
          <div className="notice warning" role="status">
            <strong>Connect your Deep Deck League account</strong>
            <span>
              Add your account API key in Application setup, then return here
              to choose decks from your authenticated training pool.
            </span>
          </div>
        ) : loadingDecks ? (
          <p className="deck-pool-empty" role="status">Loading legal decks…</p>
        ) : deckError ? (
          <p className="form-error" role="alert">{deckError}</p>
        ) : decks.length === 0 ? (
          <p className="deck-pool-empty">
            Your account has no deck for the selected formats.
          </p>
        ) : (
          <div className="training-deck-grid" aria-label="Training decks">
            {decks.map((deck) => {
              const selected = selectedDecks.includes(deck.id);
              return (
                <button
                  key={deck.id}
                  type="button"
                  aria-pressed={selected}
                  className={selected ? "selected" : ""}
                  onClick={() => toggleDeck(deck.id)}
                >
                  <span aria-hidden="true">{selected ? "✓" : "+"}</span>
                  <strong>{deck.name}</strong>
                  <small>v{deck.version} · {deck.playableCardCount} cards</small>
                </button>
              );
            })}
          </div>
        )}
      </fieldset>
      <div className="notice warning" role="status">
        <strong>Deck training is not available yet</strong>
        <span>
          Your pool can be configured here, but Engine does not publish the
          trajectory-v1 collector yet. DeepDeckLearner will not replace your
          chosen decks with sample data.
        </span>
      </div>
      <div className="form-actions">
        <button
          className="primary"
          type="button"
          disabled={
            blockers.length > 0 ||
            selectedDecks.length === 0 ||
            !deckTrainingAvailable
          }
        >
          Train {model.toUpperCase()} on selected decks <span>→</span>
        </button>
        <small>
          Select decks first. This action unlocks only when the real trajectory
          collector is available.
        </small>
      </div>
      <button
        className="advanced-toggle"
        type="button"
        aria-expanded={advanced}
        onClick={() => setAdvanced(!advanced)}
      >
        <span>{advanced ? "−" : "+"}</span> Advanced trainer validation
      </button>
      {advanced && (
        <div className="advanced-fields trainer-validation">
          <p className="advanced-explanation">
            This separately validates the encoder, optimizer and checkpoint
            pipeline. It uses sample or existing JSONL data—not the decks
            selected above.
          </p>
          <label>
            Training input
            <select
              value={source}
              onChange={(event) => {
                const next = event.target.value as "smoke" | "dataset";
                setSource(next);
                if (next === "dataset" && !dataset)
                  setDataset(
                    status?.paths.trajectory ??
                      ".deepdeck/trajectories/decisions.jsonl",
                  );
              }}
            >
              <option value="smoke">Built-in smoke trajectory</option>
              <option value="dataset">Trajectory JSONL file</option>
            </select>
          </label>
          {source === "dataset" && (
            <label className="wide-field">
              Trajectory file
              <input
                value={dataset}
                onChange={(event) => setDataset(event.target.value)}
                placeholder={
                  status?.paths.trajectory ??
                  ".deepdeck/trajectories/decisions.jsonl"
                }
                required
              />
              <small>
                Created automatically inside this project when the workbench
                starts.
              </small>
            </label>
          )}
          <label>
            Epochs
            <input
              type="number"
              min="1"
              max="1000"
              value={epochs}
              onChange={(event) => setEpochs(Number(event.target.value))}
            />
          </label>
          <label>
            Learning rate
            <input
              type="number"
              min="0.00000001"
              max="1"
              step="0.0001"
              value={learningRate}
              onChange={(event) => setLearningRate(Number(event.target.value))}
            />
          </label>
          <label>
            Device
            <select
              value={device}
              onChange={(event) => setDevice(event.target.value)}
            >
              <option value="cuda">CUDA · fallback to CPU</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
        </div>
      )}
      {blockers.length > 0 && (
        <div className="notice warning">
          <strong>Before you start</strong>
          {blockers.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      )}
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
      {advanced && (
        <div className="form-actions validation-actions">
          <button className="secondary" disabled={blockers.length > 0}>
            Validate {model.toUpperCase()} trainer
          </button>
          <small>This does not train on the selected deck pool.</small>
        </div>
      )}
    </form>
  );
}

function LocalTrainingForm({ status, account, refresh }: { status: CapabilityStatus | null; account: AccountStatus | null; refresh: () => void }) {
  const [query, setQuery] = useState("");
  const [decks, setDecks] = useState<DeckSummary[]>([]);
  const [selected, setSelected] = useState<DeckSummary[]>([]);
  const [busy, setBusy] = useState("");
  const [model, setModel] = useState("v12");
  const [modelName, setModelName] = useState("");
  const [reservePlaytest, setReservePlaytest] = useState(true);
  const [selfPlayAllSeats, setSelfPlayAllSeats] = useState(true);
  const [starting, setStarting] = useState(false);
  const isLatentPretraining = model === "v13" || model === "v15";
  const [created, setCreated] = useState(false);
  const [error, setError] = useState("");
  const requiredFormat = model === "v11" ? "commander" : "legacy";
  const compatibleDeckCount = selected.filter(
    (deck) => deck.format?.toLowerCase() === requiredFormat,
  ).length;

  useEffect(() => {
    if (account?.valid !== true) return;
    let active = true;
    const timer = window.setTimeout(() => {
      void Promise.all([searchDecks(query, "legacy"), searchDecks(query, "commander")])
        .then((groups) => { if (active) setDecks(groups.flat()); })
        .catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : "Unable to search decks."); });
    }, 250);
    return () => { active = false; window.clearTimeout(timer); };
  }, [account?.valid, query]);

  useEffect(() => {
    void loadTrainingDeckPool().then((pool) => setSelected(pool.decks)).catch(() => undefined);
  }, []);

  async function toggleDeck(deck: DeckSummary) {
    if (selected.some((item) => item.id === deck.id)) {
      const next = selected.filter((item) => item.id !== deck.id);
      setSelected(next);
      void saveTrainingDeckPool(next);
      return;
    }
    setBusy(deck.id);
    setError("");
    try {
      const downloaded = await downloadDeck(deck.id);
      const next = [
        ...selected.filter((item) => !isSameDeckLineage(item, deck)),
        { ...deck, playableCardCount: downloaded.cardCount },
      ];
      setSelected(next);
      await saveTrainingDeckPool(next);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to download this deck.");
    } finally {
      setBusy("");
    }
  }

  async function configureAgent() {
    setStarting(true);
    setCreated(false);
    setError("");
    try {
      await createModel({
        model,
        model_name: modelName.trim(),
        parallel_matches: 1,
        gpu_memory_mb: 0,
        reserve_playtest: reservePlaytest,
        self_play_all_seats: selfPlayAllSeats,
      });
      setCreated(true);
      refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to configure this agent.");
    } finally {
      setStarting(false);
    }
  }

  return <section className="panel configure local-training-form">
    <div className="section-heading"><div><span className="eyebrow">New agent</span><h2>Architecture and training decks</h2><p className="section-lead">Create the agent identity here. CPU, GPU and simultaneous games are assigned later from Jobs.</p></div><span className="step">CONFIG</span></div>
    {account?.valid !== true && <div className="notice warning"><strong>{account?.valid === false && account.configured ? "Replace the rejected API key" : account?.configured ? "League platform unavailable" : "API key required"}</strong><span>{isLatentPretraining ? `${model.toUpperCase()} world-model pretraining is local and does not require a League API key.` : selected.length > 0 ? "Your downloaded deck snapshots remain available locally. A valid key is only required to add more decks or join the League." : account?.reason ?? "Configure your Deep Deck League API key to access training decks."}</span></div>}
    <label>Architecture<select value={model} onChange={(event) => setModel(event.target.value)}><option value="v15">V15 · typed latent planner</option><option value="v13">V13 · graph-belief experimental</option><option value="v12" disabled={!selected.some((deck) => deck.format?.toLowerCase() === "legacy")}>V12 · Legacy</option><option value="v11" disabled={!selected.some((deck) => deck.format?.toLowerCase() === "commander")}>V11 · Commander</option></select><small>{isLatentPretraining ? "Local world-model pretraining; no API key, deck, or Engine session required." : `${compatibleDeckCount} ${requiredFormat} deck${compatibleDeckCount === 1 ? "" : "s"} available for this model.`}</small></label>
    <>
      <div className="training-pool-heading"><div><strong>Training deck pool</strong><small>{selected.length === 0 ? "No deck selected" : `${selected.length} deck${selected.length === 1 ? "" : "s"} ready locally`}</small></div></div>
      {selected.length > 0 && <div className="selected-training-pool">{selected.map((deck) => <button type="button" key={deck.id} onClick={() => void toggleDeck(deck)} title="Remove from training pool"><span><strong>{deck.name}</strong><small>{deck.format ?? "Deck"} · v{deck.version}</small></span><b aria-hidden="true">×</b></button>)}</div>}
      {account?.valid === true && <><label className="deck-search">Add decks<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search any deck by name…" autoFocus /></label>
      <div className="training-deck-grid" aria-label="Training deck results">
        {decks.map((deck) => { const isSelected = selected.some((item) => item.id === deck.id); return <button key={deck.id} type="button" className={isSelected ? "selected" : ""} onClick={() => void toggleDeck(deck)} disabled={Boolean(busy)}><span aria-hidden="true">{busy === deck.id ? "…" : isSelected ? "✓" : "+"}</span><strong>{deck.name}</strong><small>{deck.format ?? "Deck"} · v{deck.version} · {deck.playableCardCount} cards</small></button>; })}
      </div></>}
      {busy && <p className="deck-pool-empty" role="status">Adding the deck to the local pool…</p>}
      {selected.length > 0 && <div className="notice success" role="status"><strong>Training pool ready locally</strong><span>{selected.length} immutable deck snapshot{selected.length === 1 ? "" : "s"} stored in .deepdeck/decks.</span></div>}
      {error && <p className="form-error" role="alert">{error}</p>}
      {(selected.length > 0 || isLatentPretraining) && <div className="training-launch">
        <label>Model name<input value={modelName} maxLength={64} onChange={(event) => setModelName(event.target.value)} placeholder="Example: Montréal Control" /><small>This is your AI's name; its architecture remains fixed after creation.</small></label>
        <div className="model-implementation"><strong>{model === "v15" ? "V15 · typed latent conditional planner" : model === "v13" ? "V13 · graph-belief world model" : model === "v12" ? "V12 · structured two-player policy" : "V11 · structured multiplayer policy"}</strong><p>{model === "v15" ? "Multi-level latent states, composable cost/announcement/response/resolution deltas, sampled reconstruction, conditional plans and adaptive compute budgets." : model === "v13" ? "Experimental local pretraining with typed graph observations, recurrent latent dynamics, explicit belief and opponent heads, and separate loss telemetry." : model === "v12" ? "Designed for two-player Legacy: structured observations, legal-action encoding, two relative value slots, self-play collection and PPO updates." : "Designed for Commander: structured observations, legal-action encoding, four multiplayer value slots, shared-policy self-play and PPO updates."}</p><small>Deep Deck provides the architecture and training implementation. The generated weights, name and local model belong to this workspace.</small></div>
        {!isLatentPretraining && <div className="agent-mode-grid">
          <label className="check-setting"><input type="checkbox" checked={selfPlayAllSeats} onChange={(event) => setSelfPlayAllSeats(event.target.checked)} /><span><strong>Shared-model self-play</strong><small>{selfPlayAllSeats ? "The same model controls every player in each training game." : "Half of training games use the built-in anchor opponent."}</small></span></label>
          <label className="check-setting"><input type="checkbox" checked={reservePlaytest} onChange={(event) => setReservePlaytest(event.target.checked)} /><span><strong>Publish playable weights</strong><small>Keeps a stable checkpoint available while newer weights train.</small></span></label>
        </div>}
        {created && <div className="notice success" role="status"><strong>Agent configured</strong><span>Open Jobs to assign simultaneous games and start training.</span></div>}
        <div className="form-actions"><AsyncActionButton className="primary" type="button" onClick={() => void configureAgent()} loading={starting} loadingLabel="Preparing agent…" disabled={created || modelName.trim().length < 2 || (!isLatentPretraining && compatibleDeckCount === 0) || !status?.torch.ready || (!isLatentPretraining && !status?.engine.healthy)}>{created ? "Agent configured" : `Create ${modelName.trim() || model.toUpperCase()}`}</AsyncActionButton><small>{modelName.trim().length < 2 ? "Give your model a name first." : !isLatentPretraining && compatibleDeckCount === 0 ? `Add a ${requiredFormat} deck for this agent.` : "No training starts until you explicitly start it in Jobs."}</small></div>
      </div>}
    </>
  </section>;
}

function formatBytes(bytes: number | null | undefined) {
  if (bytes === null || bytes === undefined) return "Unavailable";
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MiB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GiB`;
}

function isSameDeckLineage(left: DeckSummary, right: DeckSummary) {
  return left.id === right.id || Boolean(left.deckId && right.deckId && left.deckId === right.deckId);
}

function AgentEditor({ model, activeWorkers, onClose, refresh }: {
  model: LocalModel;
  activeWorkers: ResourceSnapshot["workers"];
  onClose: () => void;
  refresh: () => void;
}) {
  const [name, setName] = useState(model.name);
  const [decks, setDecks] = useState<DeckSummary[]>(model.decks);
  const [results, setResults] = useState<DeckSummary[]>([]);
  const [query, setQuery] = useState("");
  const [reservePlaytest, setReservePlaytest] = useState(model.reservePlaytest);
  const [selfPlayAllSeats, setSelfPlayAllSeats] = useState(model.selfPlayAllSeats !== false);
  const [busy, setBusy] = useState("");
  const [saving, setSaving] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    const timer = window.setTimeout(() => {
      void searchDecks(query, model.format)
        .then((items) => { if (active) setResults(items); })
        .catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : "Unable to search decks."); });
    }, 250);
    return () => { active = false; window.clearTimeout(timer); };
  }, [model.format, query]);

  async function toggleDeck(deck: DeckSummary) {
    if (decks.some((item) => item.id === deck.id)) {
      setDecks((current) => current.filter((item) => item.id !== deck.id));
      return;
    }
    setBusy(deck.id);
    setError("");
    try {
      const downloaded = await downloadDeck(deck.id);
      setDecks((current) => [
        ...current.filter((item) => !isSameDeckLineage(item, deck)),
        { ...deck, playableCardCount: downloaded.cardCount },
      ]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to download this deck.");
    } finally {
      setBusy("");
    }
  }

  async function save() {
    setSaving(true);
    setError("");
    try {
      await updateModel(model.id, {
        name: name.trim(),
        decks,
        reservePlaytest,
        selfPlayAllSeats,
      });
      refresh();
      onClose();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to update this agent.");
    } finally {
      setSaving(false);
    }
  }

  async function stopActiveWorkers() {
    setStopping(true);
    setError("");
    try {
      await Promise.all(activeWorkers.map((worker) => stopJob(worker.jobId)));
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to stop this agent's active services.");
    } finally {
      setStopping(false);
    }
  }

  return <div className="agent-editor">
    <div className="training-launch">
      <label>Agent name<input value={name} maxLength={64} onChange={(event) => setName(event.target.value)} /></label>
      <div className="automatic-setting"><span>Architecture</span><strong>{model.architecture.toUpperCase()} · {model.format}</strong><small>The architecture stays fixed so existing weights remain compatible.</small></div>
    </div>
    <div className="training-pool-heading"><div><strong>Decks this agent can play and train with</strong><small>{decks.length} selected</small></div></div>
    {decks.length > 0 && <div className="selected-training-pool">{decks.map((deck) => <button type="button" key={deck.id} onClick={() => void toggleDeck(deck)} title="Remove from this agent"><span><strong>{deck.name}</strong><small>{deck.format} · v{deck.version}</small></span><b aria-hidden="true">×</b></button>)}</div>}
    <label className="deck-search">Add {model.format} decks<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search decks by name…" /></label>
    <div className="training-deck-grid" aria-label={`Deck results for ${model.name}`}>
      {results.map((deck) => { const selected = decks.some((item) => item.id === deck.id); return <button key={deck.id} type="button" className={selected ? "selected" : ""} disabled={Boolean(busy)} onClick={() => void toggleDeck(deck)}><span aria-hidden="true">{busy === deck.id ? "…" : selected ? "✓" : "+"}</span><strong>{deck.name}</strong><small>v{deck.version} · {deck.playableCardCount} cards</small></button>; })}
    </div>
    <div className="agent-mode-grid">
      <label className="check-setting"><input type="checkbox" checked={selfPlayAllSeats} onChange={(event) => setSelfPlayAllSeats(event.target.checked)} /><span><strong>Shared-model self-play</strong><small>The same model controls every training seat.</small></span></label>
      <label className="check-setting"><input type="checkbox" checked={reservePlaytest} onChange={(event) => setReservePlaytest(event.target.checked)} /><span><strong>Publish playable weights</strong><small>Keep a stable checkpoint available for playtests.</small></span></label>
    </div>
    {activeWorkers.length > 0 && <div className="notice warning" role="status"><strong>{activeWorkers.length} active service{activeWorkers.length === 1 ? "" : "s"} must stop before saving</strong><span>Your edits stay in this form. Stop the agent's services, then save the changes.</span><AsyncActionButton className="danger subtle" type="button" loading={stopping} loadingLabel="Stopping services…" onClick={() => void stopActiveWorkers()}>{`Stop ${activeWorkers.length} active service${activeWorkers.length === 1 ? "" : "s"}`}</AsyncActionButton></div>}
    {error && <p className="form-error" role="alert">{error}</p>}
    <div className="form-actions"><button type="button" onClick={onClose}>Cancel</button><AsyncActionButton className="primary" type="button" loading={saving} loadingLabel="Saving agent…" disabled={activeWorkers.length > 0 || name.trim().length < 2 || decks.length === 0} onClick={() => void save()}>Save agent</AsyncActionButton><small>{activeWorkers.length > 0 ? "Stop the active services above to enable saving." : "The existing weights are preserved when these settings change."}</small></div>
  </div>;
}

function AgentCatalog({
  models,
  resources,
  refresh,
}: {
  models: LocalModel[];
  resources: ResourceSnapshot | null;
  refresh: () => void;
}) {
  const [confirming, setConfirming] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [deleting, setDeleting] = useState("");
  const [editing, setEditing] = useState("");
  const [message, setMessage] = useState("");

  async function remove(model: LocalModel) {
    setDeleting(model.id);
    setMessage("");
    try {
      const result = await deleteModel(model.id);
      setConfirming("");
      setConfirmation("");
      setMessage(`${result.name} deleted · ${formatBytes(result.reclaimedBytes)} reclaimed.`);
      refresh();
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "Unable to delete this agent.");
    } finally {
      setDeleting("");
    }
  }

  if (models.length === 0) {
    return <div className="empty-agent"><span>AI</span><div><strong>No configured agent yet</strong><p>Choose an architecture, give it a deck pool, then create its local identity.</p></div></div>;
  }
  return <div className="agent-catalog">
    {models.map((model) => {
      const activeWorkers = resources?.workers.filter((worker) => worker.modelId === model.id) ?? [];
      const isConfirming = confirming === model.id;
      return <article className="agent-card" key={model.id}>
        <div className="agent-card-main">
          <span className={`model-ready ${model.ready ? "ready" : ""}`}>{model.ready ? "Playable weights" : model.status === "running" ? "Preparing weights" : "Configured"}</span>
          <h3>{model.name}</h3>
          <p>{model.description}</p>
          <div className="agent-card-tags"><span>{model.architecture.toUpperCase()}</span><span>{model.format}</span><span>{model.selfPlayAllSeats === false ? "Mixed opponents" : "Shared-model self-play"}</span></div>
        </div>
        <dl className="agent-card-facts">
          <div><dt>Training decks</dt><dd>{model.decks.length}</dd></div>
          <div><dt>Disk space</dt><dd>{formatBytes(model.diskBytes)}</dd><small>{formatBytes(model.weightsBytes)} weights</small></div>
          <div><dt>Live workers</dt><dd>{activeWorkers.reduce((total, worker) => total + worker.workerSlots, 0)}</dd></div>
          <div><dt>Games learned</dt><dd>{model.trainingState?.completedGames ?? 0}</dd></div>
        </dl>
        <div className="agent-deck-list">{model.decks.map((deck) => <span key={deck.id}>{deck.name}</span>)}</div>
        {!isConfirming && <div className="agent-card-actions"><button type="button" onClick={() => { setEditing(editing === model.id ? "" : model.id); setMessage(""); }}>{editing === model.id ? "Close editor" : "Edit agent and decks"}</button><button className="danger subtle" type="button" onClick={() => { setEditing(""); setConfirming(model.id); setConfirmation(""); setMessage(""); }}>Delete agent and weights</button></div>}
        {editing === model.id && <AgentEditor model={model} activeWorkers={activeWorkers} refresh={refresh} onClose={() => setEditing("")} />}
        {isConfirming && <div className="delete-agent-confirm" role="alertdialog" aria-label={`Delete ${model.name}`}>
          <strong>Delete {formatBytes(model.diskBytes)} permanently?</strong>
          <p>The agent, checkpoints, training history and local statistics in this run will be removed.</p>
          {activeWorkers.length > 0 ? <small>Stop its active jobs first.</small> : <label>Type <b>{model.name}</b><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /></label>}
          <div><button type="button" onClick={() => { setConfirming(""); setConfirmation(""); }}>Keep agent</button><AsyncActionButton className="danger" type="button" loading={deleting === model.id} loadingLabel="Deleting files…" disabled={activeWorkers.length > 0 || confirmation !== model.name} onClick={() => void remove(model)}>Delete files</AsyncActionButton></div>
        </div>}
      </article>;
    })}
    {message && <p className={message.includes("deleted") ? "form-success" : "form-error"} role="status">{message}</p>}
  </div>;
}

function ResourceSystemSummary({ resources }: { resources: ResourceSnapshot | null }) {
  if (!resources) return null;
  const ramPercent = resources.system.ramTotalBytes
    ? Math.round((resources.system.ramUsedBytes / resources.system.ramTotalBytes) * 100)
    : 0;
  return (
    <section className="resource-summary" aria-label="Local resource usage">
      <article><span>System RAM</span><strong>{formatBytes(resources.system.ramUsedBytes)}</strong><small>{ramPercent}% of {formatBytes(resources.system.ramTotalBytes)}</small></article>
      <article><span>GPU memory</span><strong>{formatBytes(resources.system.gpuUsedBytes)}</strong><small>{resources.system.gpuTotalBytes === null ? "NVIDIA telemetry unavailable" : `of ${formatBytes(resources.system.gpuTotalBytes)}`}</small></article>
      <article><span>Local Engine</span><strong>{formatBytes(resources.engine.ramBytes)}</strong><small>{resources.engine.activeLocalGames ? `≈ ${formatBytes(resources.engine.ramPerGameEstimate)} per active game` : "No active local game"}</small></article>
    </section>
  );
}

function AgentAllocationRow({
  model,
  resources,
  jobs,
  refresh,
}: {
  model: LocalModel;
  resources: ResourceSnapshot | null;
  jobs: Job[];
  refresh: () => void;
}) {
  const [plan, setPlan] = useState<ResourcePlan | null>(null);
  const [savedPlan, setSavedPlan] = useState<ResourcePlan | null>(null);
  const [saving, setSaving] = useState(false);
  const [working, setWorking] = useState(false);
  const [message, setMessage] = useState("");
  const [messageTone, setMessageTone] = useState<"progress" | "success" | "error">("success");
  const workers = resources?.workers.filter((worker) => worker.modelId === model.id) ?? [];
  const trainingWorker = workers.find((worker) => worker.kind === "training.pool");
  const trainingJob = jobs.find((job) => job.model_id === model.id && job.kind === "training.pool" && ["queued", "running"].includes(job.status));
  const latestTrainingJob = jobs.find((job) => job.model_id === model.id && job.kind === "training.pool");
  const latestTrainingLog = latestTrainingJob?.logs.filter((line) => line.trim()).at(-1);
  const trainingActive = Boolean(trainingWorker || trainingJob);

  useEffect(() => {
    let active = true;
    void loadModelResources(model.id)
      .then((value) => { if (active) { setPlan(value); setSavedPlan(value); } })
      .catch((reason) => { if (active) { setMessageTone("error"); setMessage(reason instanceof Error ? reason.message : "Unable to load resources."); } });
    return () => { active = false; };
  }, [model.id]);

  useEffect(() => {
    if (!latestTrainingJob) return;
    if (latestTrainingJob.status === "failed") {
      setMessageTone("error");
      setMessage(`Self-play failed: ${latestTrainingLog || `trainer exited with code ${latestTrainingJob.exit_code ?? "unknown"}`}`);
    } else if (latestTrainingJob.status === "queued") {
      setMessageTone("progress");
      setMessage("Self-play is queued. Preparing the trainer configuration and worker processâ€¦");
    } else if (latestTrainingJob.status === "running") {
      setMessageTone(trainingWorker ? "success" : "progress");
      setMessage(trainingWorker
        ? `Self-play is running with ${trainingWorker.workerSlots} worker${trainingWorker.workerSlots === 1 ? "" : "s"}.${latestTrainingLog ? ` ${latestTrainingLog}` : ""}`
        : `Trainer process started. Waiting for the first Engine gameâ€¦${latestTrainingLog ? ` ${latestTrainingLog}` : ""}`);
    } else if (latestTrainingJob.status === "completed") {
      setMessageTone("success");
      setMessage("Self-play completed successfully. The latest weights and metrics were saved.");
    } else if (latestTrainingJob.status === "stopped") {
      setMessageTone("success");
      setMessage("Self-play is stopped. The agent and its saved weights are preserved.");
    }
  }, [latestTrainingJob, latestTrainingLog, trainingWorker]);

  function update(key: keyof ResourcePlan, value: number) {
    setPlan((current) => current ? { ...current, [key]: value } : current);
  }

  async function save() {
    if (!plan) return;
    setSaving(true);
    setMessageTone("progress");
    setMessage("Saving the resource allocationâ€¦");
    try {
      const saved = await saveModelResources(model.id, plan);
      setPlan(saved);
      setSavedPlan(saved);
      setMessageTone("success");
      setMessage(trainingActive ? "Allocation saved · trainer updates after the current batch." : "Allocation saved · ready to start.");
      refresh();
    } catch (reason) {
      setMessageTone("error");
      setMessage(reason instanceof Error ? reason.message : "Unable to save resources.");
    } finally {
      setSaving(false);
    }
  }

  async function toggleTraining() {
    if (!plan) return;
    setWorking(true);
    setMessageTone("progress");
    setMessage(trainingActive ? "Requesting a safe stop after the current training boundaryâ€¦" : "Starting self-playâ€¦ Preparing configuration, checkpoint and Engine workers.");
    try {
      if (trainingActive) {
        const jobId = trainingJob?.id ?? trainingWorker?.jobId;
        if (jobId) await stopJob(jobId);
        setMessageTone("success");
        setMessage("Training stopped. The agent and its weights are preserved.");
      } else {
        await startJob({ kind: "training.pool", model_id: model.id });
        setMessageTone("progress");
        setMessage("Self-play request accepted. Waiting for the trainer process and first Engine gameâ€¦");
      }
      refresh();
    } catch (reason) {
      setMessageTone("error");
      setMessage(reason instanceof Error ? reason.message : "Unable to update training.");
    } finally {
      setWorking(false);
    }
  }

  const ramBytes = workers.reduce((total, worker) => total + worker.ramBytes, 0);
  const gpuBytes = workers.reduce(
    (total, worker) => total + (worker.gpuBytes ?? 0),
    0,
  );

  return (
    <tr>
      <th scope="row">
        <span className={`model-ready ${model.ready ? "ready" : ""}`}>
          {trainingActive ? "Training live" : model.ready ? "Playable" : "Configured"}
        </span>
        <strong>{model.name}</strong>
        <small>{model.architecture.toUpperCase()} · {model.format}</small>
      </th>
      {!plan ? (
        <td colSpan={5}>Loading allocation…</td>
      ) : (
        <>
          <td className="agent-allocation-resources">
            <label>
              GPU cap
              <span>
                <input
                  aria-label={`GPU memory limit for ${model.name}`}
                  type="number"
                  min="0"
                  max="24576"
                  step="256"
                  value={plan.gpuMemoryMb}
                  onChange={(event) => update("gpuMemoryMb", Number(event.target.value))}
                />
                MiB
              </span>
            </label>
            <small>{formatBytes(ramBytes)} RAM · {formatBytes(gpuBytes)} GPU active</small>
          </td>
          {([
            ["trainingMatches", "Self-play", 32],
            ["localMatches", "Playtest", 8],
            ["leagueMatches", "League", 32],
          ] as const).map(([key, label, maximum]) => (
            <td className="agent-allocation-slot" key={key}>
              <input
                aria-label={`${label} slots for ${model.name}`}
                type="number"
                min="0"
                max={maximum}
                value={plan[key]}
                onChange={(event) => update(key, Number(event.target.value))}
              />
              <small>{plan[key] === 0 ? "Paused" : `${plan[key]} simultaneous`}</small>
            </td>
          ))}
        </>
      )}
      <td className="agent-allocation-save">
        <AsyncActionButton className="secondary" type="button" loading={saving} loadingLabel="Saving allocation…" disabled={!plan || JSON.stringify(plan) === JSON.stringify(savedPlan)} onClick={() => void save()}>
          {saving ? "Saving…" : JSON.stringify(plan) === JSON.stringify(savedPlan) ? "Allocation saved" : "Save allocation"}
        </AsyncActionButton>
        <AsyncActionButton className={trainingActive ? "danger" : "primary"} type="button" loading={working} loadingLabel={trainingActive ? "Stopping safely…" : "Starting training…"} disabled={!plan || (!trainingActive && plan.trainingMatches < 1)} onClick={() => void toggleTraining()}>{trainingActive ? "Stop training" : "Start training"}</AsyncActionButton>
        {message && <div className={`training-operation-status ${messageTone}`} role={messageTone === "error" ? "alert" : "status"} aria-live="polite">{messageTone === "progress" && <span className="inline-spinner" aria-hidden="true" />}<span>{message}</span></div>}
      </td>
    </tr>
  );
}

function AgentAllocationTable({
  models,
  resources,
  jobs,
  refresh,
}: {
  models: LocalModel[];
  resources: ResourceSnapshot | null;
  jobs: Job[];
  refresh: () => void;
}) {
  return (
    <div className="agent-allocation-scroll">
      <table className="agent-allocation-table" aria-label="Agent resource allocation">
        <thead>
          <tr>
            <th scope="col">Agent</th>
            <th scope="col">Resources</th>
            <th scope="col">Self-play games</th>
            <th scope="col">Playtest slots</th>
            <th scope="col">League connections</th>
            <th scope="col">Actions</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <AgentAllocationRow
              key={model.id}
              model={model}
              resources={resources}
              jobs={jobs}
              refresh={refresh}
            />
          ))}
        </tbody>
      </table>
      <p className="resource-caveat">
        Values are per-agent limits. Zero pauses that activity without deleting the agent.
        RAM and GPU usage come from the active process tree.
      </p>
    </div>
  );
}

function ActiveWorkersPanel({
  resources,
  jobs,
  refresh,
}: {
  resources: ResourceSnapshot | null;
  jobs: Job[];
  refresh: () => void;
}) {
  const [stopping, setStopping] = useState("");
  const [error, setError] = useState("");
  const workers = resources?.workers ?? [];
  const queued = jobs.filter(
    (job) => ["queued", "running"].includes(job.status) && !workers.some((worker) => worker.jobId === job.id),
  );

  async function stop(workerId: string) {
    setStopping(workerId);
    setError("");
    try {
      await stopJob(workerId);
      refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to stop this worker.");
    } finally {
      setStopping("");
    }
  }

  return <section className="panel active-workers">
    <div className="section-heading"><div><span className="eyebrow">Processes owned by this workbench</span><h2>Active services</h2><p className="section-lead">This is the source of truth for occupied slots. Recovered processes can be stopped here too.</p></div><button className="text-button" type="button" onClick={refresh}>Refresh</button></div>
    {workers.length === 0 && queued.length === 0 ? <div className="empty"><span>◇</span><p>No worker is holding a training, playtest or League slot.</p></div> : <div className="worker-table" role="table">
      <div className="worker-row worker-head" role="row"><span>Service</span><span>Slots</span><span>RAM</span><span>GPU</span><span>Process</span><span /></div>
      {workers.map((worker) => <div className="worker-row" role="row" key={worker.jobId}>
        <span><strong>{worker.label}</strong><small>{worker.kind === "training.pool" ? "Local self-play trainer" : worker.kind === "matchmaking.agent" ? "League connection" : "Local playtest"}</small></span>
        <span><strong>{worker.workerSlots}</strong><small>{worker.workerSlots === 1 ? "worker" : "workers"}</small></span>
        <span><strong>{formatBytes(worker.ramBytes)}</strong><small>{formatBytes(worker.ramPerWorkerEstimate)} / worker</small></span>
        <span><strong>{formatBytes(worker.gpuBytes)}</strong><small>{worker.gpuBytes === null ? "No process telemetry" : `${formatBytes(worker.gpuPerWorkerEstimate)} / worker`}</small></span>
        <span><code>{worker.pids.join(", ")}</code><small>{worker.jobId.startsWith("recovered-") ? "Recovered after restart" : "Controller attached"}</small></span>
        <span><button className="danger subtle" type="button" disabled={stopping === worker.jobId} onClick={() => void stop(worker.jobId)}>{stopping === worker.jobId ? "Stopping…" : "Stop"}</button></span>
      </div>)}
      {queued.map((job) => <div className="worker-row pending" role="row" key={job.id}><span><strong>{job.label}</strong><small>{job.kind}</small></span><span>—</span><span>Starting…</span><span>—</span><span><code>{job.id.slice(0, 8)}</code></span><span><button className="danger subtle" type="button" disabled={stopping === job.id} onClick={() => void stop(job.id)}>Cancel</button></span></div>)}
    </div>}
    {error && <p className="form-error" role="alert">{error}</p>}
  </section>;
}

function LeagueConnectionsPanel({
  account,
  models,
  resources,
  jobs,
  refresh,
}: {
  account: AccountStatus | null;
  models: LocalModel[];
  resources: ResourceSnapshot | null;
  jobs: Job[];
  refresh: () => void;
}) {
  const [competitions, setCompetitions] = useState<CompetitionSummary[]>([]);
  const [catalogs, setCatalogs] = useState<Record<string, DeckSummary[]>>({});
  const [deckModes, setDeckModes] = useState<Record<string, "all" | "pool" | "single">>({});
  const [singleDecks, setSingleDecks] = useState<Record<string, string>>({});
  const [deckPools, setDeckPools] = useState<Record<string, string[]>>({});
  const [loadingFormats, setLoadingFormats] = useState<string[]>([]);
  const [working, setWorking] = useState("");
  const [messages, setMessages] = useState<Record<string, string>>({});
  const leagueFormats = [...new Set(models.filter((model) => model.ready).map((model) => model.format))].sort().join(",");

  useEffect(() => {
    if (account?.valid !== true) {
      setCompetitions([]);
      setCatalogs({});
      return;
    }
    const formats = leagueFormats.split(",").filter(Boolean);
    setLoadingFormats(formats);
    void Promise.all([
      loadCompetitions(),
      ...formats.map(async (format) => [format, await loadAllFormatDecks(format)] as const),
    ])
      .then(([loadedCompetitions, ...loadedCatalogs]) => {
        setCompetitions(loadedCompetitions as CompetitionSummary[]);
        setCatalogs(Object.fromEntries(loadedCatalogs as ReadonlyArray<readonly [string, DeckSummary[]]>));
        setMessages((current) => ({ ...current, global: "" }));
      })
      .catch((reason) => setMessages((current) => ({ ...current, global: reason instanceof Error ? reason.message : "Unable to load League decks and competitions." })))
      .finally(() => setLoadingFormats([]));
  }, [account?.valid, leagueFormats]);

  function decksFor(model: LocalModel) {
    return catalogs[model.format] ?? model.decks.filter((deck) => deck.format?.toLowerCase() === model.format);
  }

  function togglePoolDeck(modelId: string, deckId: string) {
    setDeckPools((current) => {
      const selected = current[modelId] ?? [];
      return {
        ...current,
        [modelId]: selected.includes(deckId)
          ? selected.filter((id) => id !== deckId)
          : [...selected, deckId],
      };
    });
  }

  async function connect(model: LocalModel) {
    const availableDecks = decksFor(model);
    const mode = deckModes[model.id] ?? "single";
    const selectedIds = mode === "all"
      ? availableDecks.map((deck) => deck.id)
      : mode === "pool"
        ? deckPools[model.id] ?? []
        : [singleDecks[model.id] ?? availableDecks[0]?.id].filter((id): id is string => Boolean(id));
    const selected = selectedIds
      .map((id) => availableDecks.find((deck) => deck.id === id))
      .filter((deck): deck is DeckSummary => Boolean(deck));
    const competition = competitions.find((item) => item.format.toLowerCase() === model.format);
    if (!competition) return;
    if (selected.length === 0) {
      setMessages((current) => ({ ...current, [model.id]: mode === "pool" ? "Choose at least one deck for this League pool." : `No valid ${model.format} deck is available.` }));
      return;
    }
    setWorking(model.id);
    setMessages((current) => ({ ...current, [model.id]: "" }));
    try {
      const plan = await loadModelResources(model.id);
      if (plan.leagueMatches === 0) {
        setMessages((current) => ({ ...current, [model.id]: "Set League connections above to at least 1, then save the allocation." }));
        return;
      }
      const existingWorkers = (resources?.workers ?? []).filter(
        (worker) => worker.modelId === model.id && worker.kind === "matchmaking.agent",
      );
      const queuedJobs = jobs.filter(
        (job) => job.model_id === model.id && job.kind === "matchmaking.agent" && job.status === "queued",
      );
      for (const jobId of new Set([
        ...existingWorkers.map((worker) => worker.jobId),
        ...queuedJobs.map((job) => job.id),
      ])) {
        await stopJob(jobId);
      }
      await startJob({
        kind: "matchmaking.agent",
        model_id: model.id,
        agent: model.architecture,
        checkpoint: model.checkpointPath,
        speed: "1s",
        continuous: true,
        connections: plan.leagueMatches,
        competition_version_id: competition.versionId,
        deck_version_ids: selected.map((deck) => deck.id),
      });
      const selfPlay = plan.leagueMatches >= 2
        ? " The League prioritizes distinct agents, then pairs these seats together only when too few distinct agents are queued to fill the table."
        : " Allocate at least 2 League connections to allow self-play when the queue is empty.";
      setMessages((current) => ({ ...current, [model.id]: `${plan.leagueMatches} League seat${plan.leagueMatches === 1 ? "" : "s"} now advertises the same pool of ${selected.length} deck${selected.length === 1 ? "" : "s"}. The League selects a Plackett–Luce-balanced deck for every match.${selfPlay}` }));
      refresh();
    } catch (reason) {
      setMessages((current) => ({ ...current, [model.id]: reason instanceof Error ? reason.message : "Unable to connect this agent to the League." }));
    } finally {
      setWorking("");
    }
  }

  return <section className="panel league-connections">
    <div className="section-heading"><div><span className="eyebrow">Optional remote activity</span><h2>League connections</h2><p className="section-lead">Start the number of persistent connections saved in the allocation table. The League fills tables with distinct agents first, then uses your additional seats as self-play fallback.</p></div></div>
    {account?.valid !== true ? <div className="notice warning"><strong>{account?.configured ? "Saved League key rejected" : "League account not connected"}</strong><span>{account?.reason ?? "Add the API key in Agent configuration. Nothing else is required on a separate page."}</span></div> : <div className="league-agent-list">
      {models.filter((model) => model.ready).map((model) => {
        const competition = competitions.find((item) => item.format.toLowerCase() === model.format);
        const live = (resources?.workers ?? []).filter((worker) => worker.modelId === model.id && worker.kind === "matchmaking.agent").length;
        const availableDecks = decksFor(model);
        const mode = deckModes[model.id] ?? "single";
        const pool = deckPools[model.id] ?? [];
        const loading = loadingFormats.includes(model.format);
        return <article key={model.id}>
          <div><span className="model-ready ready">{live} connected</span><strong>{model.name}</strong><small>{competition ? competition.name : `No active ${model.format} competition`}</small></div>
          <div className="league-deck-configuration">
            <span className="field-label">League decks</span>
            <div className="league-deck-modes" role="group" aria-label={`Deck selection for ${model.name}`}>
              {(["all", "pool", "single"] as const).map((candidate) => <button key={candidate} type="button" className={mode === candidate ? "selected" : ""} aria-pressed={mode === candidate} onClick={() => setDeckModes((current) => ({ ...current, [model.id]: candidate }))}>{candidate === "all" ? "All legal" : candidate === "pool" ? "Deck pool" : "One deck"}</button>)}
            </div>
            {loading ? <small>Loading valid {model.format} decks…</small> : mode === "single" ? <select aria-label={`Deck for ${model.name}`} value={singleDecks[model.id] ?? availableDecks[0]?.id ?? ""} onChange={(event) => setSingleDecks((current) => ({ ...current, [model.id]: event.target.value }))}>{availableDecks.map((deck) => <option value={deck.id} key={deck.id}>{deck.name}</option>)}</select> : mode === "pool" ? <div className="league-deck-pool">{availableDecks.map((deck) => <label key={deck.id}><input type="checkbox" checked={pool.includes(deck.id)} onChange={() => togglePoolDeck(model.id, deck.id)} /><span>{deck.name}</span></label>)}</div> : <small>{availableDecks.length} valid {model.format} deck{availableDecks.length === 1 ? "" : "s"} will be offered to the League for server-side selection at every match.</small>}
          </div>
          <AsyncActionButton className="secondary" type="button" loading={working === model.id} loadingLabel="Connecting allocated slots…" disabled={loading || !competition || availableDecks.length === 0 || (mode === "pool" && pool.length === 0)} onClick={() => void connect(model)}>Connect allocated slots</AsyncActionButton>
          {messages[model.id] && <small role="status">{messages[model.id]}</small>}
        </article>;
      })}
      {models.every((model) => !model.ready) && <div className="empty"><span>◇</span><p>Playable weights are required before connecting to the League.</p></div>}
    </div>}
    {messages.global && <p className="form-error" role="alert">{messages.global}</p>}
  </section>;
}

function elapsedLabel(startedAtUnixMs: number | null) {
  if (!startedAtUnixMs) return "Starting";
  const seconds = Math.max(0, Math.round((Date.now() - startedAtUnixMs) / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function ActiveGamesPanel({ games, refresh, onOpen }: { games: ActiveGame[]; refresh: () => void; onOpen: (game: ActiveGame) => void }) {
  const [stopping, setStopping] = useState("");
  const [error, setError] = useState("");

  async function cancel(game: ActiveGame) {
    setStopping(game.id);
    setError("");
    try {
      await stopGame(game.id);
      refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to cancel this game.");
    } finally {
      setStopping("");
    }
  }

  return <section className="panel live-games">
    <div className="section-heading"><div><span className="eyebrow"><i className={games.length ? "live-pulse" : ""} /> Live game telemetry</span><h2>Games in progress</h2><p className="section-lead">Training, local playtest, and League matches appear here as soon as a game starts.</p></div><span className="live-game-count">{games.length} live</span></div>
    {games.length === 0 ? <div className="empty live-empty"><span>0</span><div><strong>No game is currently running</strong><p>If a slot still looks occupied, check Active services above: the process can be stopped even before an Engine session appears.</p></div></div> : <div className="live-game-grid">
      {games.map((game) => <article className="live-game-card" key={game.id}>
        <header><span className={`game-source ${game.source}`}>{game.source === "training" ? `Worker ${game.worker}` : game.source === "league" ? "League" : "Local"}</span><strong>{elapsedLabel(game.startedAtUnixMs)}</strong></header>
        <h3>{game.modelName}</h3>
        <p>{game.decks.filter(Boolean).join(" vs ") || "Preparing matchup"}</p>
        <div className="game-progress"><span>Round <b>{game.roundNumber ?? "—"}</b></span><span>Turn <b>{game.turnNumber ?? "—"}</b></span><span>Decisions <b>{game.decisions}</b></span><span>Players <b>{game.players}</b></span></div>
        {game.playersState.length > 0 && <div className="game-player-strip">{game.playersState.map((player, index) => <span className={player.hasLost ? "lost" : ""} key={player.id ?? index}><b>{player.life ?? "—"}</b><small>{player.name ?? `P${index + 1}`} · {player.handCount ?? 0} cards</small></span>)}</div>}
        <footer><small>{game.mode ?? game.status}{game.sessionId ? ` · ${game.sessionId}` : " · session opening"}</small><div className="live-game-actions">{game.source === "local" && game.jobId && <button type="button" onClick={() => onOpen(game)}>Open game</button>}{game.source === "league" && game.watchUrl && <a className="button-link" href={game.watchUrl} target="_blank" rel="noreferrer">Watch League</a>}{game.source !== "league" && <AsyncActionButton className="danger" type="button" loading={stopping === game.id} loadingLabel="Cancelling game…" disabled={!game.canCancel} onClick={() => void cancel(game)}>{game.canCancel ? "Cancel game" : "Opening…"}</AsyncActionButton>}</div></footer>
      </article>)}
    </div>}
    {error && <p className="form-error" role="alert">{error}</p>}
  </section>;
}

function JobsDashboard({
  account,
  models,
  resources,
  jobs,
  games,
  refresh,
  onOpenGame,
}: {
  account: AccountStatus | null;
  models: LocalModel[];
  resources: ResourceSnapshot | null;
  jobs: Job[];
  games: ActiveGame[];
  refresh: () => void;
  onOpenGame: (game: ActiveGame) => void;
}) {
  return <div className="jobs-dashboard">
    <section className="jobs-intro"><div><span className="eyebrow">Capacity → activity → game</span><h2>One operational view</h2><p>Save the capacity you want, start the trainer, then follow every worker and Engine game below. League connections also appear here; they no longer need a separate page.</p></div><div className="jobs-legend"><span><i className="training" /> Training</span><span><i className="local" /> Playtest</span><span><i className="league" /> League</span></div></section>
    <ResourceSystemSummary resources={resources} />
    <section className="panel allocation-panel"><div className="section-heading"><div><span className="eyebrow">Desired capacity</span><h2>Agent allocations</h2><p className="section-lead">Self-play games use the same model for every seat when that mode is enabled in Agent configuration.</p></div></div>
      {models.length ? <AgentAllocationTable models={models} resources={resources} jobs={jobs} refresh={refresh} /> : <div className="empty"><span>AI</span><p>Create an agent configuration before allocating jobs.</p></div>}
    </section>
    <LeagueConnectionsPanel account={account} models={models} resources={resources} jobs={jobs} refresh={refresh} />
    <ActiveWorkersPanel resources={resources} jobs={jobs} refresh={refresh} />
    <ActiveGamesPanel games={games} refresh={refresh} onOpen={onOpenGame} />
  </div>;
}

function trainingMetrics(job: Job) {
  for (const line of [...job.logs].reverse()) {
    try {
      const value = JSON.parse(line) as Record<string, unknown>;
      if (typeof value.loss === "number") return value;
      const training = value.training;
      if (training && typeof training === "object") {
        const record = training as Record<string, unknown>;
        const ppo = record.ppo;
        if (ppo && typeof ppo === "object") {
          return {
            ...(ppo as Record<string, unknown>),
            updates: record.trainingStep ?? record.episode,
          };
        }
      }
    } catch { /* Non-JSON log line. */ }
  }
  return null;
}

const lossInformation: Record<string, string> = {
  loss: "Weighted total objective optimized by the trainer. Lower is generally better, but compare it only within the same training phase and configuration.",
  policy_loss: "Clipped PPO policy objective. Its sign may change and a value near zero is normal; interpret it together with reward, entropy and KL.",
  value_loss: "Error between the predicted value and the observed return or outcome. Lower means the critic estimates results more accurately.",
  reconstruction_loss: "Error when reconstructing visible graph features. Lower means the encoder retains more information from the observation.",
  dynamics_loss: "Error between the predicted next latent state and the encoded next state. Lower means action-conditioned transitions are modeled more accurately.",
  kl_loss: "Divergence between posterior and prior latent distributions. Large values indicate mismatch; values pinned near zero can indicate posterior collapse.",
  belief_loss: "Binary cross-entropy for hidden-state beliefs. Lower is better, but calibration and the class-frequency baseline also matter.",
  opponent_loss: "Cross-entropy for predicting the opponent's action. Compare it with the random baseline log(number of action classes).",
  search_loss: "Cross-entropy against the search-policy target. Lower means the policy follows search more closely; a plateau may reflect noisy or impossible targets.",
};

function lossHelp(name: string): string {
  return lossInformation[name]
    ?? "Optimization error for this objective. Lower is usually better; interpret its scale against the same objective and training phase.";
}

type DeckSortKey = "deck" | "rank" | "ordinal" | "mu" | "matches" | "record" | "winRate";
type DeckSortDirection = "asc" | "desc";

const initialDeckSort: { key: DeckSortKey; direction: DeckSortDirection } = {
  key: "ordinal",
  direction: "desc",
};

function deckSortValue(deck: DeckStatistic, key: DeckSortKey): string | number | null {
  switch (key) {
    case "deck": return `${deck.modelName}\u0000${deck.deckName}`;
    case "rank": return deck.rank;
    case "ordinal": return deck.ordinal;
    case "mu": return deck.mu;
    case "matches": return deck.matches;
    case "record": return deck.gameWins - deck.gameLosses;
    case "winRate": return deck.winRate;
  }
}

function compareDeckStatistics(
  left: DeckStatistic,
  right: DeckStatistic,
  key: DeckSortKey,
  direction: DeckSortDirection,
): number {
  const leftValue = deckSortValue(left, key);
  const rightValue = deckSortValue(right, key);
  if (leftValue === null && rightValue !== null) return 1;
  if (leftValue !== null && rightValue === null) return -1;
  let comparison = 0;
  if (typeof leftValue === "string" && typeof rightValue === "string") {
    comparison = leftValue.localeCompare(rightValue, undefined, { sensitivity: "base" });
  } else if (typeof leftValue === "number" && typeof rightValue === "number") {
    comparison = leftValue - rightValue;
  }
  if (comparison !== 0) return direction === "asc" ? comparison : -comparison;
  if (key === "mu" && left.sigma !== right.sigma) return left.sigma - right.sigma;
  return left.deckName.localeCompare(right.deckName, undefined, { sensitivity: "base" });
}

function MetricLineChart({ points }: { points: TrainingStatistic["latestMetrics"] }) {
  const values = points.map((point) => point.loss).filter((value): value is number => typeof value === "number");
  if (values.length === 0) return null;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const span = Math.max(0.000001, maximum - minimum);
  const coordinates = values.map((value, index) => {
    const x = values.length === 1 ? 50 : 4 + (index / (values.length - 1)) * 92;
    const y = 8 + ((maximum - value) / span) * 72;
    return `${x},${y}`;
  }).join(" ");
  return <div className="metric-line-chart" aria-label="Training loss line chart">
    <svg viewBox="0 0 100 92" preserveAspectRatio="none" role="img">
      <line x1="4" y1="8" x2="96" y2="8" />
      <line x1="4" y1="44" x2="96" y2="44" />
      <line x1="4" y1="80" x2="96" y2="80" />
      <polyline points={coordinates} />
    </svg>
    <span className="chart-maximum">{maximum.toFixed(4)}</span>
    <span className="chart-minimum">{minimum.toFixed(4)}</span>
    <small>step {points[0]?.trainingStep ?? 0}</small>
    <small>step {points.at(-1)?.trainingStep ?? 0}</small>
  </div>;
}

function StatisticsPage({ jobs, decks, training, metricWindow, onMetricWindowChange }: { jobs: Job[]; decks: DeckStatistic[]; training: TrainingStatistic[]; metricWindow: TrainingStatistic["metricWindow"]; onMetricWindowChange: (window: TrainingStatistic["metricWindow"]) => void }) {
  const [selectedId, setSelectedId] = useState("");
  const [settings, setSettings] = useState<TrainingSettings | null>(null);
  const [curriculum, setCurriculum] = useState<TrainingCurriculum | null>(null);
  const [evidence, setEvidence] = useState<TrainingEvidence | null>(null);
  const [replays, setReplays] = useState<SavedReplaySummary[]>([]);
  const [replay, setReplay] = useState<SavedReplay | null>(null);
  const [replayFrame, setReplayFrame] = useState(0);
  const [replayLease, setReplayLease] = useState<{ modelId: string; replayId: string; leaseId: string } | null>(null);
  const [platformMessage, setPlatformMessage] = useState("");
  const [savingSettings, setSavingSettings] = useState(false);
  const [agentTraining, setAgentTraining] = useState<AgentTrainingPublication | null>(null);
  const [agentParameters, setAgentParameters] = useState<Record<string, string | number | boolean>>({});
  const [selectedPhaseId, setSelectedPhaseId] = useState("");
  const [pendingPhaseId, setPendingPhaseId] = useState("");
  const [phaseBusy, setPhaseBusy] = useState(false);
  const [phaseFeedback, setPhaseFeedback] = useState<{ tone: "progress" | "success" | "error"; message: string } | null>(null);
  const [deckSort, setDeckSort] = useState(initialDeckSort);
  const runs = jobs.filter((job) => job.kind.startsWith("training."));
  const selected = training.find((item) => item.modelId === selectedId) ?? training[0];
  const selectedDecks = selected ? decks.filter((deck) => deck.modelId === selected.modelId) : decks;
  const sortedSelectedDecks = [...selectedDecks].sort((left, right) => (
    compareDeckStatistics(left, right, deckSort.key, deckSort.direction)
  ));
  const decidedGames = selectedDecks.reduce((total, deck) => total + deck.gameWins + deck.gameLosses, 0);
  const wins = selectedDecks.reduce((total, deck) => total + deck.gameWins, 0);
  const lossPoints = selected?.latestMetrics.filter((point) => typeof point.loss === "number") ?? [];
  const lossNames = [...new Set(lossPoints.flatMap((point) => Object.keys(point.losses ?? {})))];
  const publishedMetricDescriptions = new Map(agentTraining?.contract.phases.flatMap((phase) => phase.metrics.map((metric) => [metric.key, metric.description])) ?? []);
  const summary = selected?.windowSummary;
  const rewardValues = selected?.latestMetrics.map((point) => point.episodeReward).filter((value): value is number => typeof value === "number") ?? [];
  const meanReward = rewardValues.length ? rewardValues.reduce((total, value) => total + value, 0) / rewardValues.length : null;
  const isInProcessRl = summary?.sampleType === "rl-episodes";
  const activeTrainingJob = selected
    ? jobs.find((job) => job.model_id === selected.modelId && job.kind === "training.pool" && ["queued", "running"].includes(job.status))
    : undefined;
  const latestSelectedTrainingJob = selected
    ? jobs.find((job) => job.model_id === selected.modelId && job.kind === "training.pool")
    : undefined;
  const latestSelectedTrainingLog = latestSelectedTrainingJob?.logs.filter((line) => line.trim()).at(-1);
  const activePhaseId = typeof agentTraining?.state?.phase === "string" ? agentTraining.state.phase : "";
  const activePhaseStatus = String(agentTraining?.state?.status ?? "");
  const selectedPhase = agentTraining?.contract.phases.find((phase) => phase.id === selectedPhaseId)
    ?? agentTraining?.contract.phases.find((phase) => phase.id === activePhaseId)
    ?? agentTraining?.contract.phases[0];
  useEffect(() => {
    if (!selected?.modelId) return;
    let live = true;
    setPlatformMessage("");
    setAgentTraining(null);
    setReplay(null);
    setReplayLease(null);
    setReplayFrame(0);
    Promise.all([
      loadTrainingSettings(selected.modelId),
      loadTrainingCurriculum(),
      loadTrainingEvidence(selected.modelId),
      loadSavedReplays(selected.modelId),
      loadAgentTrainingContract(selected.modelId),
    ]).then(([nextSettings, nextCurriculum, nextEvidence, nextReplays, nextAgentTraining]) => {
      if (!live) return;
      setSettings(nextSettings);
      setCurriculum(nextCurriculum);
      setEvidence(nextEvidence);
      setReplays(nextReplays);
      setAgentTraining(nextAgentTraining);
      setAgentParameters(Object.fromEntries(nextAgentTraining?.contract.phases.flatMap((phase) => phase.parameters.map((parameter) => [parameter.key, nextAgentTraining.parameterValues?.[parameter.key] ?? parameter.default ?? ""])) ?? []));
      setSelectedPhaseId(String(nextAgentTraining?.state?.phase ?? nextAgentTraining?.contract.phases[0]?.id ?? ""));
    }).catch((reason) => {
      if (live) setPlatformMessage(reason instanceof Error ? reason.message : "Unable to load training controls.");
    });
    return () => { live = false; };
  }, [selected?.modelId]);

  useEffect(() => {
    if (!latestSelectedTrainingJob) return;
    if (latestSelectedTrainingJob.status === "failed") {
      setPhaseFeedback({ tone: "error", message: `Training failed: ${latestSelectedTrainingLog || `trainer exited with code ${latestSelectedTrainingJob.exit_code ?? "unknown"}`}` });
    } else if (latestSelectedTrainingJob.status === "queued") {
      setPhaseFeedback({ tone: "progress", message: "Training is queued. The controller is preparing the selected phaseâ€¦" });
    } else if (latestSelectedTrainingJob.status === "running") {
      setPhaseFeedback({ tone: "progress", message: latestSelectedTrainingLog || "Trainer started. Waiting for the first completed self-play stepâ€¦" });
    } else if (latestSelectedTrainingJob.status === "completed") {
      setPhaseFeedback({ tone: "success", message: "Training completed and the latest checkpoint was saved." });
    }
  }, [latestSelectedTrainingJob, latestSelectedTrainingLog]);

  useEffect(() => {
    if (!selected?.modelId) return;
    const interval = window.setInterval(() => {
      void loadAgentTrainingContract(selected.modelId).then((publication) => {
        if (publication) setAgentTraining(publication);
      });
    }, 5_000);
    return () => window.clearInterval(interval);
  }, [selected?.modelId]);

  useEffect(() => {
    if (!pendingPhaseId || activeTrainingJob || !selected || !agentTraining) return;
    const phase = agentTraining.contract.phases.find((item) => item.id === pendingPhaseId);
    if (!phase) return;
    setPendingPhaseId("");
    setPhaseBusy(true);
    setPhaseFeedback({ tone: "progress", message: `Starting ${phase.label}â€¦` });
    const parameters = Object.fromEntries(
      phase.parameters.map((parameter) => [parameter.key, agentParameters[parameter.key]]),
    );
    const action = phase.controls.includes("start") ? "start" : "evaluate";
    void saveAgentTrainingControl(selected.modelId, { phase: phase.id, action, parameters })
      .then((control) => startJob({ kind: "training.pool", model_id: selected.modelId, phase_id: phase.id }).then(() => control))
      .then((control) => {
        setAgentTraining((current) => current ? { ...current, control } : current);
        setPhaseFeedback({ tone: "progress", message: `${phase.label} accepted. Waiting for the trainer process and first resultâ€¦` });
        setPlatformMessage(`${phase.label} started with the displayed parameters.`);
      })
      .catch((reason) => {
        const message = reason instanceof Error ? reason.message : "Unable to start the next training phase.";
        setPhaseFeedback({ tone: "error", message });
        setPlatformMessage(message);
      })
      .finally(() => setPhaseBusy(false));
  }, [activeTrainingJob, agentParameters, agentTraining, pendingPhaseId, selected]);

  useEffect(() => {
    if (!replayLease) return;
    const interval = window.setInterval(() => {
      void renewSavedReplayLease(replayLease.modelId, replayLease.replayId, replayLease.leaseId);
    }, 30_000);
    return () => {
      window.clearInterval(interval);
      void releaseSavedReplayLease(replayLease.modelId, replayLease.replayId, replayLease.leaseId);
    };
  }, [replayLease]);

  async function persistSettings() {
    if (!selected || !settings) return;
    setSavingSettings(true);
    setPlatformMessage("");
    try {
      setSettings(await saveTrainingSettings(selected.modelId, settings));
      setPlatformMessage("Training plan saved. It will apply on the next trainer start.");
    } catch (reason) {
      setPlatformMessage(reason instanceof Error ? reason.message : "Unable to save training controls.");
    } finally {
      setSavingSettings(false);
    }
  }

  async function trainOnlyScenario(scenarioId: string) {
    if (!selected || !settings || !agentTraining || activeTrainingJob) return;
    setPhaseBusy(true);
    setPlatformMessage("");
    try {
      const focused = {
        ...settings,
        stages: { ...settings.stages, reinforcementLearning: true },
        curriculum: { enabled: true, adaptive: false, scenarioIds: [scenarioId] },
      };
      const saved = await saveTrainingSettings(selected.modelId, focused);
      setSettings(saved);
      const phase = "reinforcement-learning";
      const control = await saveAgentTrainingControl(selected.modelId, {
        phase,
        action: "start",
        parameters: phaseParameters(phase),
      });
      await startJob({ kind: "training.pool", model_id: selected.modelId, phase_id: phase });
      setAgentTraining({ ...agentTraining, control });
      setPhaseFeedback({ tone: "progress", message: `Focused scenario ${scenarioId} is starting.` });
      setPlatformMessage(`Training started with only ${scenarioId}.`);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "Unable to start this focused scenario.";
      setPhaseFeedback({ tone: "error", message });
      setPlatformMessage(message);
    } finally {
      setPhaseBusy(false);
    }
  }

  async function openReplay(item: SavedReplaySummary) {
    if (!selected) return;
    setPlatformMessage("");
    try {
      const acquired = await acquireSavedReplay(selected.modelId, item.id);
      setReplay(acquired.replay);
      setReplayLease(acquired.leaseId ? { modelId: selected.modelId, replayId: item.id, leaseId: acquired.leaseId } : null);
      setReplays((current) => current.map((candidate) => candidate.id === item.id ? { ...candidate, viewing: true } : candidate));
      setReplayFrame(0);
    } catch (reason) {
      setPlatformMessage(reason instanceof Error ? reason.message : "Unable to open the saved game.");
    }
  }

  async function keepReplayForever() {
    if (!selected || !replay) return;
    setPlatformMessage("");
    try {
      await saveReplayForever(selected.modelId, replay.id);
      setReplays((current) => current.map((candidate) => candidate.id === replay.id ? { ...candidate, saved: true, viewing: true } : candidate));
      setReplayLease(null);
      setPlatformMessage("Replay saved permanently. Retention cleanup will never remove it.");
    } catch (reason) {
      setPlatformMessage(reason instanceof Error ? reason.message : "Unable to save this replay.");
    }
  }

  function phaseParameters(phase: string) {
    const phaseContract = agentTraining?.contract.phases.find((item) => item.id === phase);
    return Object.fromEntries(phaseContract?.parameters.map((parameter) => [parameter.key, agentParameters[parameter.key]]) ?? []);
  }

  async function launchPhase(phase: string) {
    if (!selected || !agentTraining) return;
    const phaseContract = agentTraining.contract.phases.find((item) => item.id === phase);
    if (!phaseContract) return;
    if (activeTrainingJob) {
      if (activePhaseId === phase) {
        setPhaseFeedback({ tone: "success", message: `${phaseContract.label} is already running.` });
        setPlatformMessage(`${phaseContract.label} is already running.`);
        return;
      }
      setPhaseBusy(true);
      setPhaseFeedback({ tone: "progress", message: `Stopping the current phase safely before ${phaseContract.label} startsâ€¦` });
      try {
        if (activePhaseId) {
          const active = agentTraining.contract.phases.find((item) => item.id === activePhaseId);
          if (active?.controls.includes("stop")) {
            await saveAgentTrainingControl(selected.modelId, { phase: activePhaseId, action: "stop", parameters: {} });
          } else {
            await stopJob(activeTrainingJob.id);
          }
        } else {
          await stopJob(activeTrainingJob.id);
        }
        setPendingPhaseId(phase);
        setPlatformMessage(`Stopping the current phase safely; ${phaseContract.label} will start automatically.`);
      } catch (reason) {
        const message = reason instanceof Error ? reason.message : "Unable to switch training phases.";
        setPhaseFeedback({ tone: "error", message });
        setPlatformMessage(message);
      } finally {
        setPhaseBusy(false);
      }
      return;
    }
    setPhaseBusy(true);
    setPhaseFeedback({ tone: "progress", message: `Starting ${phaseContract.label}â€¦ Validating parameters and preparing the trainer.` });
    setPlatformMessage("");
    try {
      const action = phaseContract.controls.includes("start") ? "start" : "evaluate";
      const control = await saveAgentTrainingControl(selected.modelId, { phase, action, parameters: phaseParameters(phase) });
      await startJob({ kind: "training.pool", model_id: selected.modelId, phase_id: phase });
      setAgentTraining({ ...agentTraining, control });
      setPhaseFeedback({ tone: "progress", message: `${phaseContract.label} accepted. Waiting for the trainer process and first resultâ€¦` });
      setPlatformMessage(`${phaseContract.label} started with the displayed parameters.`);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "Unable to start this training phase.";
      setPhaseFeedback({ tone: "error", message });
      setPlatformMessage(message);
    } finally {
      setPhaseBusy(false);
    }
  }

  async function requestAgentControl(phase: string, action: string) {
    if (!selected || !agentTraining) return;
    if (action === "start" || action === "evaluate") {
      await launchPhase(phase);
      return;
    }
    setPhaseBusy(true);
    setPlatformMessage("");
    try {
      const control = await saveAgentTrainingControl(selected.modelId, { phase, action, parameters: phaseParameters(phase) });
      setAgentTraining({ ...agentTraining, control });
      setPlatformMessage(`${action} requested for ${agentTraining.contract.phases.find((item) => item.id === phase)?.label ?? phase}.`);
    } catch (reason) {
      setPlatformMessage(reason instanceof Error ? reason.message : "Unable to control this agent.");
    } finally {
      setPhaseBusy(false);
    }
  }

  const selectedPhaseMetrics = (agentTraining?.metrics ?? []).filter((record) => record.phase === selectedPhase?.id);
  const latestPhaseMetrics = selectedPhaseMetrics.at(-1);
  const previousPhaseMetrics = selectedPhaseMetrics.at(-2);
  const formatPhaseMetric = (value: number | undefined, kind: string) => {
    if (value === undefined) return "—";
    if (kind === "rate") return `${(value * 100).toFixed(1)}%`;
    return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(4);
  };

  const activeReplaySummary = replay ? replays.find((item) => item.id === replay.id) : undefined;
  const replayDeckLabels = Array.isArray(replay?.metadata.decks)
    ? replay.metadata.decks.map(String)
    : [];
  const replayDeckVersionIds = selectedDecks
    .filter((deck) => replayDeckLabels.length === 0 || replayDeckLabels.some((label) => label.includes(deck.deckName)))
    .map((deck) => deck.deckVersionId);
  const evidenceTrend = evidence?.trend.filter((point) => point.winRate !== null) ?? [];
  const evidenceTrendPoints = evidenceTrend.map((point, index) => `${evidenceTrend.length === 1 ? 50 : 4 + (index / (evidenceTrend.length - 1)) * 92},${82 - (point.winRate ?? 0) * 72}`).join(" ");
  const seconds = (value: number | null | undefined) => value == null ? "—" : value < 60 ? `${value.toFixed(1)}s` : `${(value / 60).toFixed(1)}m`;
  const sortDecksBy = (key: DeckSortKey) => {
    setDeckSort((current) => current.key === key
      ? { key, direction: current.direction === "asc" ? "desc" : "asc" }
      : { key, direction: key === "deck" || key === "rank" ? "asc" : "desc" });
  };
  const deckSortHeader = (key: DeckSortKey, label: string) => {
    const active = deckSort.key === key;
    return <span role="columnheader" aria-sort={active ? (deckSort.direction === "asc" ? "ascending" : "descending") : "none"}>
      <button type="button" className={active ? "stats-sort-button active" : "stats-sort-button"} onClick={() => sortDecksBy(key)}>
        {label}<i aria-hidden="true">{active ? deckSort.direction === "asc" ? "↑" : "↓" : "↕"}</i>
      </button>
    </span>;
  };
  return <section className="statistics-page">
    <section className="statistics-toolbar"><div><span className="eyebrow">Local training history</span><h2>{selected?.modelName ?? "No trained agent yet"}</h2></div><div className="statistics-filters">{training.length > 0 && <label>Agent<select value={selected?.modelId ?? ""} onChange={(event) => setSelectedId(event.target.value)}>{training.map((item) => <option value={item.modelId} key={item.modelId}>{item.modelName} · {item.architecture.toUpperCase()}</option>)}</select></label>}<label>Window<select aria-label="Metric window" value={metricWindow} onChange={(event) => onMetricWindowChange(event.target.value as TrainingStatistic["metricWindow"])}><option value="50">Last 50</option><option value="200">Last 200</option><option value="1000">Last 1,000</option><option value="5000">Last 5,000</option><option value="all">All time</option></select></label></div></section>
    <div className="stats-summary training-kpis"><article><span>{isInProcessRl ? "Completed RL episodes" : "Completed games"}</span><strong>{selected?.completedGames ?? 0}</strong><small>{selected?.activeGames ?? 0} live now</small></article><article><span>Training updates</span><strong>{selected?.trainingStep ?? 0}</strong><small>{selected?.parallelGames ?? 0} simultaneous slots</small></article><article><span>{isInProcessRl ? "Mean window reward" : "Deck-pool win rate"}</span><strong>{isInProcessRl ? meanReward?.toFixed(3) ?? "—" : decidedGames ? `${Math.round((wins / decidedGames) * 100)}%` : "—"}</strong><small>{isInProcessRl ? `${rewardValues.length} PPO rollouts` : `${decidedGames} decided games`}</small></article><article><span>{isInProcessRl ? "Average episode" : "Average game"}</span><strong>{selected?.averageGameSeconds ? `${selected.averageGameSeconds.toFixed(1)}s` : "—"}</strong><small>{selected?.phase ?? "not started"}</small></article></div>
    {evidence && <section className={`panel evidence-panel evidence-${evidence.status}`}><div className="section-heading"><div><span className="eyebrow">Magic skill evidence</span><h2>{evidence.status === "verified" ? "Progression verified" : evidence.status === "not-connected" ? "Not proven on Magic" : evidence.status === "not-evaluated" ? "Evaluation required" : evidence.status === "insufficient" ? "Early evidence only" : "Performance measured"}</h2><p className="section-lead">{evidence.verdict}</p></div><span className="evidence-badge">{evidence.magicEvidence ? `${evidence.games} Engine games` : "No Engine evidence"}</span></div>
      <div className="evidence-layout"><div className="evidence-kpis"><article><span>Fixed-seed win rate</span><strong>{evidence.winRate == null ? "—" : `${(evidence.winRate * 100).toFixed(1)}%`}</strong><small>{evidence.lower95 == null || evidence.upper95 == null ? "No interval yet" : `95% CI ${(evidence.lower95 * 100).toFixed(1)}–${(evidence.upper95 * 100).toFixed(1)}%`}</small></article><article><span>Record</span><strong>{evidence.wins}–{evidence.losses}–{evidence.draws}</strong><small>win–loss–draw vs {selected?.architecture === "v13" ? "frozen pre-Engine baseline" : "saved champion"}</small></article><article><span>Promotions</span><strong>{evidence.promotions}</strong><small>{evidence.fixedSeeds ? "reproducible fixed seeds" : "fixed seeds not confirmed"}</small></article></div>
      <div className="evidence-trend"><span>Evaluation trend</span>{evidenceTrend.length ? <svg viewBox="0 0 100 90" preserveAspectRatio="none"><line x1="4" y1="46" x2="96" y2="46" /><polyline points={evidenceTrendPoints} /></svg> : <p>Run an Engine evaluation period to create this curve.</p>}</div></div>
      {evidence.byDeck.length > 0 && <div className="evidence-decks"><strong>Controlled results by deck</strong>{evidence.byDeck.map((deck) => <span key={deck.deck}><b>{deck.deck}</b><em>{deck.wins}–{deck.losses}–{deck.draws}</em><small>{deck.winRate == null ? "—" : `${(deck.winRate * 100).toFixed(1)}%`} · 95% CI {deck.lower95 == null || deck.upper95 == null ? "—" : `${(deck.lower95 * 100).toFixed(0)}–${(deck.upper95 * 100).toFixed(0)}%`}</small></span>)}</div>}
      {(evidence.curriculum?.byScenario.length ?? 0) > 0 && <div className="curriculum-evidence"><strong>Focused skills · weakest first</strong><div>{evidence.curriculum.byScenario.map((scenario) => <article key={scenario.scenarioId}><span>{scenario.scenarioId.replaceAll("-", " ")}</span><b>{scenario.successRate == null ? "—" : `${(scenario.successRate * 100).toFixed(0)}%`}</b><small>{scenario.successes}/{scenario.games} successes · {scenario.averageRound == null ? "—" : `${scenario.averageRound.toFixed(1)} turns`} · {scenario.averageMilestoneProgress == null ? "" : `${(scenario.averageMilestoneProgress * 100).toFixed(0)}% line · `}{scenario.averageDamageProgress == null ? "" : `${(scenario.averageDamageProgress * 100).toFixed(0)}% damage · `}{scenario.averageSideboardCards == null ? "" : `${scenario.averageSideboardCards.toFixed(1)} sideboard cards · `}{scenario.sideboardTargetCoverage == null ? "" : `${(scenario.sideboardTargetCoverage * 100).toFixed(0)}% cards in · `}{scenario.sideboardCutCoverage == null ? "" : `${(scenario.sideboardCutCoverage * 100).toFixed(0)}% cards out · `}weight {scenario.adaptiveWeight?.toFixed(2) ?? "—"}</small><i>{scenario.recent.map((success, index) => <em className={success ? "success" : "failure"} key={`${scenario.scenarioId}-${index}`} />)}</i></article>)}</div></div>}
      {evidence.curriculum?.latestEvaluation && <div className="curriculum-holdout"><strong>Fixed-seed curriculum evaluation</strong><span><b>{(evidence.curriculum.latestEvaluation.successRate * 100).toFixed(0)}%</b> success</span><span><b>{(evidence.curriculum.latestEvaluation.meanMastery * 100).toFixed(0)}%</b> mastery</span><small>{evidence.curriculum.latestEvaluation.completedScenarios}/{evidence.curriculum.latestEvaluation.expectedScenarios} scenarios · {evidence.curriculum.latestEvaluation.recoveredScenarios ?? 0} recovered · {evidence.curriculum.latestEvaluation.engineErrors ?? evidence.curriculum.latestEvaluation.failedScenarios} Engine errors · {evidence.curriculum.evaluationPeriods} periods</small></div>}
    </section>}
    {agentTraining && selectedPhase && <section className="panel training-controls phase-cockpit"><div className="section-heading"><div><span className="eyebrow">Agent SDK · phase cockpit</span><h2>Run, inspect, adjust, repeat</h2><p className="section-lead">Choose one phase, see its own results and parameters, then start or switch without leaving this page.</p></div><span className={`evidence-badge ${activeTrainingJob ? "live" : ""}`}>{activeTrainingJob ? `● ${String(agentTraining.state?.status ?? "running")}` : "idle"}</span></div>
      <nav className="phase-rail" aria-label="Training phases">{agentTraining.contract.phases.map((phase, index) => {
        const records = (agentTraining.metrics ?? []).filter((record) => record.phase === phase.id);
        const isActive = activePhaseId === phase.id && Boolean(activeTrainingJob);
        return <button type="button" className={`${selectedPhase.id === phase.id ? "selected" : ""} ${isActive ? "active" : ""}`} key={phase.id} onClick={() => setSelectedPhaseId(phase.id)}><span>{index + 1}</span><strong>{phase.label}</strong><small>{isActive ? "running now" : records.length ? `${records.length} results` : "ready"}</small></button>;
      })}</nav>
      <div className="phase-workbench"><article className="phase-results"><header><div><span className="eyebrow">Results · step {latestPhaseMetrics?.step ?? "—"}</span><h3>{selectedPhase.label}</h3><p>{selectedPhase.description}</p></div><AsyncActionButton className="primary" type="button" loading={phaseBusy || pendingPhaseId === selectedPhase.id} loadingLabel={pendingPhaseId === selectedPhase.id ? "Waiting to switch…" : "Preparing phase…"} disabled={Boolean(activeTrainingJob && activePhaseId === selectedPhase.id)} onClick={() => void launchPhase(selectedPhase.id)}>{activeTrainingJob && activePhaseId !== selectedPhase.id ? "Switch to this phase" : activeTrainingJob ? activePhaseStatus === "paused" ? "Paused" : "Running" : selectedPhase.id === "engine-evaluation" ? "Run evaluation" : "Start phase"}</AsyncActionButton></header>
        <div className="phase-metric-grid">{selectedPhase.metrics.map((metric) => {
          const value = latestPhaseMetrics?.metrics[metric.key];
          const previous = previousPhaseMetrics?.metrics[metric.key];
          const delta = value !== undefined && previous !== undefined ? value - previous : undefined;
          const improving = delta !== undefined && ((metric.direction === "minimize" && delta < 0) || (metric.direction === "maximize" && delta > 0));
          return <span key={metric.key}><small>{metric.label}<button type="button" className="metric-info" aria-label={`About ${metric.label}`} data-help={metric.description} title={metric.description}>i</button></small><strong>{formatPhaseMetric(value, metric.kind)}</strong><em className={delta === undefined ? "" : improving ? "improving" : "watch"}>{delta === undefined ? metric.direction : `${delta > 0 ? "+" : ""}${delta.toFixed(4)} vs prior`}</em></span>;
        })}</div>
        {selectedPhaseMetrics.length === 0 && <p className="phase-empty">No result yet for this phase. Starting it will populate these cards automatically.</p>}
      </article>
      <aside className="phase-setup"><header><span className="eyebrow">Parameters</span><strong>{selectedPhase.label}</strong></header><div className="training-parameter-grid">{selectedPhase.parameters.map((parameter) => <label key={parameter.key}>{parameter.label}<span className="apply-boundary">{parameter.apply.replace("-", " ")}</span>{parameter.kind === "boolean" ? <input type="checkbox" checked={Boolean(agentParameters[parameter.key])} onChange={(event) => setAgentParameters({ ...agentParameters, [parameter.key]: event.target.checked })} /> : parameter.kind === "choice" ? <select value={String(agentParameters[parameter.key] ?? "")} onChange={(event) => setAgentParameters({ ...agentParameters, [parameter.key]: event.target.value })}>{parameter.choices.map((choice) => <option key={String(choice)} value={String(choice)}>{String(choice)}</option>)}</select> : <input type={parameter.kind === "integer" || parameter.kind === "number" ? "number" : "text"} min={parameter.min ?? undefined} max={parameter.max ?? undefined} step={parameter.step ?? (parameter.kind === "integer" ? 1 : undefined)} value={String(agentParameters[parameter.key] ?? "")} onChange={(event) => setAgentParameters({ ...agentParameters, [parameter.key]: parameter.kind === "integer" || parameter.kind === "number" ? Number(event.target.value) : event.target.value })} />}<small>{parameter.description || `Applied ${parameter.apply.replace("-", " ")}.`}</small></label>)}</div>
        {activePhaseId === selectedPhase.id && activeTrainingJob && <div className="agent-contract-actions">{selectedPhase.controls.filter((action) => action !== "start" && action !== "evaluate" && (action !== "pause" || activePhaseStatus !== "paused") && (action !== "resume" || activePhaseStatus === "paused")).map((action) => <button type="button" disabled={phaseBusy} key={action} onClick={() => void requestAgentControl(selectedPhase.id, action)}>{action}</button>)}</div>}
      </aside></div>
      {phaseFeedback && <div className={`training-operation-status phase-feedback ${phaseFeedback.tone}`} role={phaseFeedback.tone === "error" ? "alert" : "status"} aria-live="polite">{phaseFeedback.tone === "progress" && <span className="inline-spinner" aria-hidden="true" />}<span>{phaseFeedback.message}</span></div>}
      {platformMessage && <p className="training-platform-message" role="status">{platformMessage}</p>}
    </section>}
    {settings && <section className="panel training-controls"><div className="section-heading"><div><span className="eyebrow">Training cockpit</span><h2>Stages, retention and evaluation cadence</h2><p className="section-lead">Changes are validated and applied the next time this trainer starts.</p></div><AsyncActionButton className="primary" type="button" loading={savingSettings} loadingLabel="Saving training plan…" onClick={() => void persistSettings()}>Save training plan</AsyncActionButton></div>
      <div className="training-stage-grid"><label className={!(["v13", "v15"].includes(settings.architecture)) ? "unavailable" : ""}><input type="checkbox" disabled={!(["v13", "v15"].includes(settings.architecture))} checked={settings.stages.worldModel} onChange={(event) => setSettings({ ...settings, stages: { ...settings.stages, worldModel: event.target.checked } })} /><span><strong>1 · World model</strong><small>Graph reconstruction, typed latent dynamics and belief pretraining.</small></span></label><label className={settings.architecture === "v15" ? "unavailable" : ""}><input type="checkbox" disabled={settings.architecture === "v15"} checked={settings.stages.reinforcementLearning} onChange={(event) => setSettings({ ...settings, stages: { ...settings.stages, reinforcementLearning: event.target.checked } })} /><span><strong>2 · Reinforcement learning</strong><small>{settings.architecture === "v15" ? "Plan-level RL follows real Engine transition training." : "Policy/value optimization from self-play trajectories."}</small></span></label><label className={!settings.engineEvaluationSupported ? "unavailable" : ""}><input type="checkbox" disabled={!settings.engineEvaluationSupported} checked={settings.stages.engineEvaluation} onChange={(event) => setSettings({ ...settings, stages: { ...settings.stages, engineEvaluation: event.target.checked } })} /><span><strong>3 · Engine evaluation</strong><small>{settings.engineEvaluationSupported ? "Fixed-seed games against the current champion." : "Authoritative Engine evaluation is not configured for this architecture."}</small></span></label></div>
      {settings.architecture === "v13" && curriculum && <div className="curriculum-workbench"><div className="training-pool-heading"><div><strong>Focused game curriculum</strong><small>Short controlled games train one skill at a time; failed scenarios are sampled more often.</small></div><label className="curriculum-master"><input type="checkbox" checked={settings.curriculum.enabled} onChange={(event) => setSettings({ ...settings, curriculum: { ...settings.curriculum, enabled: event.target.checked } })} /> Enable</label></div>
        <label className="check-setting"><input type="checkbox" disabled={!settings.curriculum.enabled} checked={settings.curriculum.adaptive} onChange={(event) => setSettings({ ...settings, curriculum: { ...settings.curriculum, adaptive: event.target.checked } })} /><span><strong>Adaptive difficulty</strong><small>Increase sampling weight for scenarios with a low recent success rate.</small></span></label>
        <div className="curriculum-grid">{curriculum.scenarios.map((scenario) => {
          const selectedScenario = settings.curriculum.scenarioIds.length === 0 || settings.curriculum.scenarioIds.includes(scenario.id);
          return <label className={`${selectedScenario ? "selected" : ""} ${!settings.curriculum.enabled ? "unavailable" : ""}`} key={scenario.id}><input type="checkbox" disabled={!settings.curriculum.enabled} checked={selectedScenario} onChange={() => {
            const current = settings.curriculum.scenarioIds.length === 0 ? curriculum.scenarios.map((item) => item.id) : settings.curriculum.scenarioIds;
            const scenarioIds = selectedScenario ? current.filter((id) => id !== scenario.id) : [...current, scenario.id];
            setSettings({ ...settings, curriculum: { ...settings.curriculum, scenarioIds } });
          }} /><span><strong>{scenario.label}</strong><small>{scenario.description}</small><em>Difficulty {scenario.difficulty}/5 · {scenario.tags.join(" · ")}</em>{scenario.fixed_hand.length > 0 && <small>Fixed hand: {scenario.fixed_hand.join(", ")}</small>}{scenario.opening_hand_roles.length > 0 && <small>Hand roles: {scenario.opening_hand_roles.map((role) => role.join(" / ")).join(" · ")}</small>}{scenario.success_action_sequence.length > 0 && <small>Required line: {scenario.success_action_sequence.join(" → ")}</small>}{scenario.sideboard_target_cards.length > 0 && <small>Bring in: {scenario.sideboard_target_cards.join(", ")}</small>}{scenario.sideboard_cut_cards.length > 0 && <small>Take out: {scenario.sideboard_cut_cards.join(", ")}</small>}<button type="button" disabled={Boolean(activeTrainingJob) || phaseBusy} onClick={(event) => { event.preventDefault(); event.stopPropagation(); void trainOnlyScenario(scenario.id); }}>Train only this</button></span></label>;
        })}</div>
      </div>}
      <div className="training-parameter-grid"><label>Saved games<input type="number" min="0" max="500" value={settings.savedGameLimit} onChange={(event) => setSettings({ ...settings, savedGameLimit: Number(event.target.value) })} /><small>Keep the most recent complete Engine replays. 0 disables capture.</small></label><label>Checkpoint every<input type="number" min="1" value={settings.checkpointEvery} onChange={(event) => setSettings({ ...settings, checkpointEvery: Number(event.target.value) })} /></label><label>Evaluate every<input type="number" min="1" value={settings.evaluationEvery} onChange={(event) => setSettings({ ...settings, evaluationEvery: Number(event.target.value) })} /><small>{settings.architecture === "v13" ? "PPO updates between paired fixed-seed evaluations." : "Completed training games between evaluations."}</small></label><label>Games / scenario<input type="number" min="1" max="100" value={settings.evaluationGamesPerScenario} onChange={(event) => setSettings({ ...settings, evaluationGamesPerScenario: Number(event.target.value) })} /></label>{["v13", "v15"].includes(settings.architecture) && <><label>World-model steps<input type="number" min="0" value={settings.targets.worldModelSteps} onChange={(event) => setSettings({ ...settings, targets: { ...settings.targets, worldModelSteps: Number(event.target.value) } })} /></label>{settings.architecture === "v13" && <label>RL episodes<input type="number" min="0" value={settings.targets.reinforcementLearningEpisodes} onChange={(event) => setSettings({ ...settings, targets: { ...settings.targets, reinforcementLearningEpisodes: Number(event.target.value) } })} /></label>}</>}</div>
      {platformMessage && <p className="training-platform-message" role="status">{platformMessage}</p>}
    </section>}
    <section className="panel training-curve"><div className="section-heading"><div><span className="eyebrow">{selected?.metricRecordCount ?? 0} records · {lossPoints.length} plotted points</span><h2>Learning curve</h2><p className="section-lead">Total loss across the selected window. Large windows are evenly downsampled for display while summaries use every record.</p></div>{selected && <span className={`training-phase ${selected.activeGames ? "live" : ""}`}>{selected.activeGames ? "● collecting games" : selected.desiredState}</span>}</div>
      {lossPoints.length === 0 ? <div className="empty"><span>↗</span><p>The curve appears after the first completed training game.</p></div> : <MetricLineChart points={lossPoints} />}
    </section>
    {lossNames.length > 0 && <section className="panel objective-dashboard"><div className="section-heading"><div><span className="eyebrow">V13 objective telemetry</span><h2>Every loss, one dashboard</h2><p className="section-lead">Each objective uses its own recent range so small auxiliary losses remain readable.</p></div></div><div className="objective-grid">{lossNames.map((name) => {
      const values = lossPoints.map((point) => point.losses?.[name]).filter((value): value is number => typeof value === "number");
      const peak = Math.max(0.000001, ...values.map(Math.abs));
      const points = values.map((value, index) => `${values.length === 1 ? 100 : (index / (values.length - 1)) * 100},${36 - (Math.abs(value) / peak) * 32}`).join(" ");
      const latest = values.at(-1);
      const help = publishedMetricDescriptions.get(name) || lossHelp(name);
      return <article className="objective-card" key={name}><header><div className="objective-title"><span>{name.replaceAll("_", " ")}</span><button type="button" className="metric-info" aria-label={`About ${name.replaceAll("_", " ")}`} data-help={help} title={help}>i</button></div><strong>{latest?.toFixed(4) ?? "—"}</strong></header><svg viewBox="0 0 100 40" preserveAspectRatio="none" role="img" aria-label={`${name} curve`}><polyline points={points} /></svg></article>;
    })}</div></section>}
    {summary && <section className="panel production-dashboard"><div className="section-heading"><div><span className="eyebrow">Selected window</span><h2>{isInProcessRl ? "RL episode production and utilization" : summary.gameMetricsAvailable ? "Game production and utilization" : "Synthetic pretraining activity"}</h2><p className="section-lead">{isInProcessRl ? "On-policy PPO episodes generated in the local benchmark environment. These are real RL trajectories, not Engine Magic games." : summary.gameMetricsAvailable ? "Generated games, samples consumed by updates, latency distribution and throughput." : "V13 currently trains on synthetic graph batches; no Engine games are claimed for this run."}</p></div></div><div className="production-grid">
      <article><span>{isInProcessRl ? "Episodes generated" : summary.gameMetricsAvailable ? "Games generated" : "Synthetic batches"}</span><strong>{summary.generatedSamples.toLocaleString()}</strong><small>{summary.gameMetricsAvailable ? `${summary.failedSamples} failed of ${summary.attemptedSamples ?? summary.generatedSamples + summary.failedSamples} attempts` : `${selected?.metricRecordCount ?? 0} metric records in this window`}</small></article>
      <article><span>{isInProcessRl ? "Episodes used" : summary.gameMetricsAvailable ? "Games used" : "Batches used"}</span><strong>{summary.usedSamples.toLocaleString()}</strong><small>{summary.utilizationRate == null ? "—" : `${(summary.utilizationRate * 100).toFixed(1)}% utilization`}</small></article>
      <article><span>Decisions</span><strong>{summary.gameMetricsAvailable ? summary.generatedDecisions.toLocaleString() : "—"}</strong><small>{summary.gameMetricsAvailable ? `${summary.usedDecisions.toLocaleString()} used for updates` : "Available after Engine trajectory collection"}</small></article>
      <article><span>{isInProcessRl ? "Episode duration" : "Game duration"}</span><strong>{seconds(summary.averageGameSeconds)}</strong><small>p50 {seconds(summary.p50GameSeconds)} · p95 {seconds(summary.p95GameSeconds)}</small></article>
      <article><span>Simulation time</span><strong>{seconds(summary.simulationSeconds)}</strong><small>{summary.gamesPerHour == null ? "No Engine games" : `${summary.gamesPerHour.toFixed(1)} ${isInProcessRl ? "episodes" : "games"}/hour`}</small></article>
      <article><span>Model training</span><strong>{seconds(summary.trainingSeconds)}</strong><small>Collection wall {seconds(summary.collectionWallSeconds)}</small></article>
    </div></section>}
    <section className="panel replay-library"><div className="section-heading"><div><span className="eyebrow">Retained Engine games</span><h2>Replay library</h2><p className="section-lead">Recent games rotate automatically. An open replay is leased against deletion; a permanently saved replay never expires.</p></div><span className="evidence-badge">{replays.length} retained · {replays.filter((item) => item.saved).length} permanent</span></div>
      {replays.length === 0 ? <div className="empty"><span>▶</span><p>No complete Engine replay has been captured for this agent yet.</p></div> : <div className="replay-workbench"><div className="replay-list">{replays.map((item) => <button type="button" className={replay?.id === item.id ? "selected" : ""} key={item.id} onClick={() => void openReplay(item)}><strong>Game {item.episode ?? item.id}</strong><span>{item.decks.join(" vs ") || item.matchupId || "Saved matchup"}</span><small>{item.saved ? "★ permanent" : item.viewing ? "● protected while open" : "temporary"} · {item.frameCount} frames · {item.roundNumber ?? "—"} rounds · {seconds(item.durationSeconds)}</small></button>)}</div>
      <div className="replay-viewer">{!replay ? <div className="empty"><span>↗</span><p>Choose a saved game to watch it on the Pixi table.</p></div> : <><header><div><strong>{replay.id}</strong><small>Interactive Pixi replay · {replay.frames.length} frames</small></div><div>{activeReplaySummary?.saved === false && <button type="button" onClick={() => void keepReplayForever()}>Save forever</button>}</div></header><Suspense fallback={<div className="pixi-replay-loading" role="status">Loading Pixi replay…</div>}><LocalPixiReplay replay={replay} frameIndex={replayFrame} onFrameIndexChange={setReplayFrame} deckVersionIds={replayDeckVersionIds} /></Suspense></>}</div></div>}
    </section>
    <section className="panel"><div className="section-heading"><div><span className="eyebrow">Balanced self-play matchmaking</span><h2>Deck ratings</h2><p className="section-lead">{selected?.architecture === "v13" ? "Per-deck Engine results count every seat independently, so mirror matches remain measurable while the graph policy learns Legacy." : "The trainer uses a Plackett–Luce rating, similar in purpose to Elo but designed for ranked multiplayer outcomes. Conservative rating (μ − 3σ) is used to sample closer matchups."}</p></div></div>
      {selectedDecks.length === 0 ? <div className="empty"><span>◇</span><p>Deck ratings will appear when this agent completes games.</p></div> : <div className="stats-table deck-stats-table" role="table">
        <div className="stats-row stats-head" role="row">{deckSortHeader("deck", "Agent · deck")}{deckSortHeader("rank", "PL rank")}{deckSortHeader("ordinal", "Elo-like rating")}{deckSortHeader("mu", "μ ± σ")}{deckSortHeader("matches", "Matches")}{deckSortHeader("record", "Game W–L")}{deckSortHeader("winRate", "Win rate")}</div>
        {sortedSelectedDecks.map((deck) => <div className="stats-row" role="row" key={`${deck.modelId}-${deck.deckVersionId}`}><span><strong>{deck.deckName}</strong><small>{deck.modelName} · {deck.format}</small></span><span>{deck.ratingSystem === "plackett-luce" ? deck.rank ?? "—" : "self-play"}</span><span>{deck.ratingSystem === "plackett-luce" ? deck.ordinal.toFixed(2) : "—"}</span><span>{deck.ratingSystem === "plackett-luce" ? `${deck.mu.toFixed(2)} ± ${deck.sigma.toFixed(2)}` : "—"}</span><span>{deck.matches}</span><span>{deck.gameWins}–{deck.gameLosses}{deck.draws ? `–${deck.draws}` : ""}</span><span>{deck.winRate === null ? "—" : `${Math.round(deck.winRate * 100)}%`}</span></div>)}
      </div>}
    </section>
    <section className="panel"><div className="section-heading"><div><span className="eyebrow">Stored locally in .deepdeck/learner.db</span><h2>Controller history</h2></div></div>{runs.length === 0 ? <div className="empty"><span>◇</span><p>No local training job has been launched yet.</p></div> : <div className="stats-table" role="table"><div className="stats-row stats-head" role="row"><span>Run</span><span>Status</span><span>Loss</span><span>Policy</span><span>Value</span><span>Updates</span></div>{runs.map((job) => { const metrics = trainingMetrics(job); return <div className="stats-row" role="row" key={job.id}><span><strong>{job.label}</strong><small>{new Date(job.created_at).toLocaleString()}</small></span><span>{job.status}</span><span>{metrics ? Number(metrics.loss).toFixed(4) : "—"}</span><span>{metrics ? Number(metrics.policy_loss).toFixed(4) : "—"}</span><span>{metrics ? Number(metrics.value_loss).toFixed(4) : "—"}</span><span>{metrics ? String(metrics.updates) : "—"}</span></div>; })}</div>}</section>
  </section>;
}

function PlaytestForm({
  status,
  models,
  refresh,
  onStarted,
}: {
  status: CapabilityStatus | null;
  models: LocalModel[];
  refresh: () => void;
  onStarted: (jobId: string) => void;
}) {
  const playableModels = models.filter((model) => model.ready);
  const [modelId, setModelId] = useState("");
  const selectedModel = playableModels.find((model) => model.id === modelId);
  const format = selectedModel?.format ?? "legacy";
  const [ownDeck, setOwnDeck] = useState("");
  const [opponentDeck, setOpponentDeck] = useState("");
  const [deckSearch, setDeckSearch] = useState("");
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState("");
  const blockers = workflowBlockers(status, "local-playtest");
  useEffect(() => {
    if (!playableModels.some((model) => model.id === modelId)) {
      const first = playableModels[0];
      setModelId(first?.id ?? "");
      setOwnDeck(first ? "random" : "");
      setOpponentDeck(first ? "random" : "");
    }
  }, [modelId, playableModels]);
  const poolDecks = selectedModel?.decks ?? [];
  const visibleDecks = poolDecks.filter((deck) =>
    deck.name.toLowerCase().includes(deckSearch.toLowerCase()),
  );
  async function submit(event: FormEvent) {
    event.preventDefault();
    setLaunching(true);
    setError("");
    try {
      const job = await startJob({
        kind: "playtest.agent",
        model_id: selectedModel?.id,
        agent: selectedModel?.architecture,
        checkpoint: selectedModel?.checkpointPath,
        format,
        deck_version_id: ownDeck,
        opponent_deck_version_id: opponentDeck,
        engine_url: status?.engine.url,
      });
      onStarted(job.id);
      refresh();
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Unable to launch playtest.",
      );
    } finally {
      setLaunching(false);
    }
  }
  return (
    <form className="panel configure" onSubmit={submit}>
      <div className="section-heading">
        <div>
          <span className="eyebrow">Step 2 · choose the matchup</span>
          <h2>Play against your agent</h2>
        </div>
        <span className="step">02</span>
      </div>
      <div className="field-grid">
        <label>
          Agent
          <select
            required
            value={modelId}
            onChange={(event) => {
              const next = playableModels.find((model) => model.id === event.target.value);
              setModelId(event.target.value);
              setOwnDeck(next ? "random" : "");
              setOpponentDeck(next ? "random" : "");
            }}
          >
            <option value="">Choose one of your models</option>
            {playableModels.map((model) => <option key={model.id} value={model.id}>{model.name} · {model.architecture.toUpperCase()}</option>)}
          </select>
          <small>Only models with local playable weights are listed.</small>
        </label>
        <div className="automatic-setting"><span>Format</span><strong>{format[0].toUpperCase() + format.slice(1)}</strong><small>Defined by your model's architecture.</small></div>
      </div>
      {playableModels.length === 0 && <div className="notice warning"><strong>No playable local model yet</strong><span>Start training and wait for the model's first local checkpoint.</span></div>}
      <label>
        Search this model's training pool
        <input
          type="search"
          value={deckSearch}
          onChange={(event) => setDeckSearch(event.target.value)}
          placeholder="Reanimator, Alexios, Atraxa…"
        />
      </label>
      <div className="field-grid">
        <label>
          Your deck
          <select
            required
            value={ownDeck}
            onChange={(event) => setOwnDeck(event.target.value)}
          >
            <option value="random">Random from this pool</option>
            {visibleDecks.map((deck) => (
              <option key={deck.id} value={deck.id}>
                {deck.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Opponent deck
          <select
            required
            value={opponentDeck}
            onChange={(event) => setOpponentDeck(event.target.value)}
          >
            <option value="random">Random near your deck's Plackett–Luce strength</option>
            {visibleDecks.map((deck) => (
              <option key={deck.id} value={deck.id}>
                {deck.name}
              </option>
            ))}
          </select>
        </label>
      </div>
      {selectedModel && poolDecks.length === 0 && <div className="notice warning"><strong>No deck snapshot is attached to this model</strong><span>Start a new named training run from a deck pool.</span></div>}
      {blockers.length > 0 && (
        <div className="notice warning">
          <strong>Before you start</strong>
          {blockers.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      )}
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
      <div className="notice">
        <strong>Selection order</strong>
        <span>
          Your random deck is resolved first. The AI deck is then sampled by
          Plackett–Luce proximity, while preserving some matchup diversity.
        </span>
      </div>
      <AsyncActionButton
        className="primary"
        loading={launching}
        loadingLabel="Preparing game table…"
        disabled={blockers.length > 0 || !selectedModel || !ownDeck || !opponentDeck}
      >
        Launch behavior test <span>▶</span>
      </AsyncActionButton>
    </form>
  );
}

function MatchmakingForm({
  status,
  models,
  refresh,
}: {
  status: CapabilityStatus | null;
  models: LocalModel[];
  refresh: () => void;
}) {
  const playableModels = models.filter((model) => model.ready);
  const [modelId, setModelId] = useState("");
  const selectedModel = playableModels.find((model) => model.id === modelId);
  const [format, setFormat] = useState("legacy");
  const [query, setQuery] = useState("");
  const [decks, setDecks] = useState<DeckSummary[]>([]);
  const [selectedDeck, setSelectedDeck] = useState<DeckSummary | null>(null);
  const [competitions, setCompetitions] = useState<CompetitionSummary[]>([]);
  const [speed, setSpeed] = useState("1s");
  const [continuous, setContinuous] = useState(false);
  const [searching, setSearching] = useState(false);
  const [joining, setJoining] = useState(false);
  const [error, setError] = useState("");
  const blockers = workflowBlockers(status, "matchmaking");
  const competition = competitions.find(
    (item) => item.format.toLowerCase() === format,
  );

  useEffect(() => {
    if (!playableModels.some((model) => model.id === modelId)) {
      setModelId(playableModels[0]?.id ?? "");
      return;
    }
    if (selectedModel && selectedModel.format !== format) {
      setFormat(selectedModel.format);
      setDecks([]);
      setSelectedDeck(null);
    }
  }, [format, modelId, playableModels, selectedModel]);

  useEffect(() => {
    if (!status?.hosted.api_key_configured) {
      setCompetitions([]);
      return;
    }
    void loadCompetitions()
      .then(setCompetitions)
      .catch((reason) => {
        setError(
          reason instanceof Error
            ? reason.message
            : "Unable to load active competitions.",
        );
      });
  }, [status?.hosted.api_key_configured]);

  async function findDecks(event: FormEvent) {
    event.preventDefault();
    if (!status?.hosted.api_key_configured) {
      setError(
        "Add your account API key and restart DeepDeckLearner before searching decks.",
      );
      return;
    }
    setSearching(true);
    setError("");
    setSelectedDeck(null);
    try {
      setDecks(await searchDecks(query, format));
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Unable to search decks.",
      );
    } finally {
      setSearching(false);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    if (!selectedDeck || !competition) return;
    setJoining(true);
    try {
      await startJob({
        kind: "matchmaking.agent",
        model_id: selectedModel?.id,
        agent: selectedModel?.architecture,
        speed,
        continuous,
        checkpoint: selectedModel?.checkpointPath,
        competition_version_id: competition.versionId,
        deck_version_id: selectedDeck.id,
      });
      refresh();
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Unable to join matchmaking.",
      );
    } finally {
      setJoining(false);
    }
  }

  return (
    <section className="panel configure matchmaking-setup">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Guided League setup</span>
          <h2>Connect, choose a deck, and queue</h2>
        </div>
        <span className="step">01—03</span>
      </div>
      <ol className="setup-steps">
        <li className={status?.hosted.api_key_configured ? "complete" : ""}>
          <span>1</span>
          <div>
            <strong>Create your account key</strong>
            <p>
              Sign in, open Account → Autonomous agents, and press Generate API
              key.
            </p>
            <a
              href="https://staging.deepdeckleague.com/account#autonomous-agents"
              target="_blank"
              rel="noreferrer"
            >
              Open the exact account section ↗
            </a>
          </div>
        </li>
        <li className={status?.hosted.api_key_configured ? "complete" : ""}>
          <span>2</span>
          <div>
            <strong>
              Add the copied line to <code>.env</code>
            </strong>
            <p>
              Save <code>DEEPDECK_API_KEY=ddl_agent_…</code> in the
              DeepDeckLearner project root, then restart the workbench. The
              browser never receives the secret.
            </p>
          </div>
        </li>
        <li>
          <span>3</span>
          <div>
            <strong>Find your deck by name</strong>
            <p>
              DeepDeckLearner keeps the version identifier underneath this
              search.
            </p>
          </div>
        </li>
      </ol>
      <form className="deck-finder" onSubmit={findDecks}>
        <label>
          Format
          <input value={format[0].toUpperCase() + format.slice(1)} readOnly />
          <small>Set by the selected local model.</small>
        </label>
        <label>
          Deck name or creator
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Reanimator, Alexios, Andrea…"
          />
        </label>
        <AsyncActionButton
          className="primary"
          type="submit"
          loading={searching}
          loadingLabel="Searching decks…"
          disabled={!status?.hosted.api_key_configured}
        >
          Search decks
        </AsyncActionButton>
      </form>
      {decks.length > 0 && (
        <div
          className="deck-search-results"
          role="listbox"
          aria-label="Deck search results"
        >
          {decks.map((deck) => (
            <button
              type="button"
              role="option"
              aria-selected={selectedDeck?.id === deck.id}
              className={selectedDeck?.id === deck.id ? "selected" : ""}
              key={deck.id}
              onClick={() => setSelectedDeck(deck)}
            >
              <strong>{deck.name}</strong>
              <span>
                {deck.creator ? `by ${deck.creator}` : "Community deck"} ·{" "}
                {deck.format ?? format} · v{deck.version}
              </span>
              <small>{deck.playableCardCount} playable cards</small>
            </button>
          ))}
        </div>
      )}
      <form onSubmit={submit}>
        <div className="field-grid">
          <label>
            Agent
            <select
              required
              value={modelId}
              onChange={(event) => setModelId(event.target.value)}
            >
              <option value="">Choose one of your models</option>
              {playableModels.map((model) => <option key={model.id} value={model.id}>{model.name} · {model.architecture.toUpperCase()}</option>)}
            </select>
          </label>
          <label>
            Decision pace
            <select
              value={speed}
              onChange={(event) => setSpeed(event.target.value)}
            >
              <option value="100ms">100 ms</option>
              <option value="1s">1 second</option>
              <option value="10s">10 seconds</option>
            </select>
          </label>
        </div>
        {playableModels.length === 0 && <div className="notice warning"><strong>No local AI can join yet</strong><span>Start training and wait for your model's first playable checkpoint.</span></div>}
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={continuous}
            onChange={(event) => setContinuous(event.target.checked)}
          />
          Keep rejoining after each match
        </label>
        {selectedDeck && (
          <div className="selected-deck">
            <span>Selected deck</span>
            <strong>{selectedDeck.name}</strong>
            <small>
              {competition
                ? `${competition.name} · ${competition.timeControl}`
                : `No active ${format} competition`}
            </small>
          </div>
        )}
        {blockers.length > 0 && (
          <div className="notice warning">
            <strong>Before you queue</strong>
            {blockers.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        )}
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <AsyncActionButton
          className="primary"
          loading={joining}
          loadingLabel="Joining matchmaking…"
          disabled={
            blockers.length > 0 ||
            !selectedDeck ||
            !competition ||
            !selectedModel
          }
        >
          Join matchmaking <span>→</span>
        </AsyncActionButton>
      </form>
    </section>
  );
}

function JobsPanel({ jobs, refresh }: { jobs: Job[]; refresh: () => void | Promise<void> }) {
  const [stopping, setStopping] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  async function refreshJobs() {
    setRefreshing(true);
    setError("");
    try {
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to refresh jobs.");
    } finally {
      setRefreshing(false);
    }
  }

  async function stop(jobId: string) {
    setStopping(jobId);
    setError("");
    try {
      await stopJob(jobId);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to stop this job.");
    } finally {
      setStopping("");
    }
  }

  return (
    <section className="panel jobs">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Controller-owned</span>
          <h2>Recent jobs</h2>
        </div>
        <AsyncActionButton className="text-button" type="button" loading={refreshing} loadingLabel="Refreshingâ€¦" onClick={() => void refreshJobs()}>Refresh</AsyncActionButton>
      </div>
      {jobs.length === 0 ? (
        <div className="empty">
          <span>◇</span>
          <p>No jobs yet. A V12 smoke run is a safe first step.</p>
        </div>
      ) : (
        <div className="job-list">
          {jobs.map((job) => (
            <article className="job" key={job.id}>
              <div>
                <StatusDot
                  ready={job.status === "completed" || job.status === "running"}
                  label={job.status}
                />
                <h3>{job.label}</h3>
                <p>{job.logs.at(-1) ?? job.argv.join(" ")}</p>
                {job.artifact_path && <code>{job.artifact_path}</code>}
              </div>
              {job.status === "running" && (
                <AsyncActionButton
                  className="danger"
                  type="button"
                  loading={stopping === job.id}
                  loadingLabel="Stoppingâ€¦"
                  disabled={Boolean(stopping)}
                  onClick={() => void stop(job.id)}
                >
                  Stop
                </AsyncActionButton>
              )}
            </article>
          ))}
        </div>
      )}
      {error && <p className="form-error" role="alert">{error}</p>}
    </section>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("train");
  const [status, setStatus] = useState<CapabilityStatus | null>(null);
  const [account, setAccount] = useState<AccountStatus | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [resources, setResources] = useState<ResourceSnapshot | null>(null);
  const [games, setGames] = useState<ActiveGame[]>([]);
  const [deckStatistics, setDeckStatistics] = useState<DeckStatistic[]>([]);
  const [trainingStatistics, setTrainingStatistics] = useState<TrainingStatistic[]>([]);
  const [metricWindow, setMetricWindow] = useState<TrainingStatistic["metricWindow"]>("200");
  const [loadError, setLoadError] = useState("");
  const [setupOpen, setSetupOpen] = useState(true);
  const [trainingOpen, setTrainingOpen] = useState(true);
  const [creatingTraining, setCreatingTraining] = useState(false);
  const [openedPlaytestJobId, setOpenedPlaytestJobId] = useState("");
  const pageHeading = useRef<HTMLHeadingElement>(null);
  const setupWasResolved = useRef(false);
  const refreshInFlight = useRef<Promise<void> | null>(null);
  const metricWindowRef = useRef<TrainingStatistic["metricWindow"]>("200");

  function changeMetricWindow(window: TrainingStatistic["metricWindow"]) {
    metricWindowRef.current = window;
    setMetricWindow(window);
    void loadTrainingStatistics(window).then(setTrainingStatistics);
  }

  function refresh(): Promise<void> {
    if (refreshInFlight.current) return refreshInFlight.current;
    const pending = (async () => {
      try {
        const results = await Promise.allSettled([
          loadStatus().then(setStatus),
          loadAccountStatus().then(setAccount),
          loadJobs().then(setJobs),
          loadModels().then(setModels),
          loadResources().then(setResources),
          loadActiveGames().then(setGames),
          loadDeckStatistics().then(setDeckStatistics),
          loadTrainingStatistics(metricWindowRef.current).then(setTrainingStatistics),
        ]);
        const failure = results.find(
          (result): result is PromiseRejectedResult => result.status === "rejected",
        );
        setLoadError(
          failure
            ? failure.reason instanceof Error
              ? failure.reason.message
              : "Controller unavailable."
            : "",
        );
      } finally {
        refreshInFlight.current = null;
      }
    })();
    refreshInFlight.current = pending;
    return pending;
  }
  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    pageHeading.current?.focus();
  }, [page]);
  const activityJobs = useMemo(
    () => jobs.filter((job) => !job.kind.startsWith("dependency.")),
    [jobs],
  );
  const userModels = useMemo(
    () => models.filter((model) => model.source !== "local-frozen-checkpoint"),
    [models],
  );
  const running = resources?.workers.length ?? activityJobs.filter((job) => job.status === "running").length;
  const stackReady = Boolean(status?.engine.healthy && status?.pixi.built);
  const activePlaytest = openedPlaytestJobId ? jobs.find(
    (job) =>
      job.id === openedPlaytestJobId &&
      job.kind === "playtest.agent" &&
      job.status === "running" &&
      job.details?.sessionId,
  ) : undefined;
  const accountReady = account?.valid === true;
  const setupReady = Boolean(
    accountReady && status?.engine.synced && status?.pixi.synced,
  );
  useEffect(() => {
    if (setupReady && !setupWasResolved.current) setSetupOpen(false);
    if (status) setupWasResolved.current = setupReady;
  }, [setupReady, status]);
  const playtestStep = jobs.some((job) => job.kind === "playtest.agent")
    ? 3
    : stackReady
      ? 2
      : 1;
  const leagueStep = jobs.some((job) => job.kind === "matchmaking.agent")
    ? 3
    : accountReady
      ? 2
      : 1;
  function selectWorkflow(next: Workflow) {
    setPage(
      next === "local-playtest"
        ? "playtest"
        : next === "matchmaking"
          ? "compete"
          : "train",
    );
  }
  return (
    <div className="app-shell">
      {activePlaytest?.details?.sessionId && (
        <PixiErrorBoundary
          key={activePlaytest.details.sessionId}
          onClose={() => {
            setOpenedPlaytestJobId("");
            void stopJob(activePlaytest.id).then(refresh);
          }}
        >
          <Suspense fallback={<section className="pixi-recovery" role="status"><h2>Opening Pixi…</h2></section>}>
            <LocalPixiTable
              deckVersionIds={[
                activePlaytest.details.playerDeck?.id ?? "",
                activePlaytest.details.opponentDeck?.id ?? "",
              ].filter(Boolean)}
              engineUrl={activePlaytest.details.engineUrl ?? status?.engine.url ?? "http://127.0.0.1:8787"}
              sessionId={activePlaytest.details.sessionId}
              matchup={`${activePlaytest.details.playerDeck?.name ?? "Your deck"} vs ${activePlaytest.details.opponentDeck?.name ?? "AI deck"}`}
              onClose={() => {
                setOpenedPlaytestJobId("");
                void stopJob(activePlaytest.id).then(refresh);
              }}
            />
          </Suspense>
        </PixiErrorBoundary>
      )}
      <aside>
        <a
          className="brand"
          href={leagueUrl}
          target="_blank"
          rel="noreferrer"
          aria-label="Open Deep Deck League"
        >
          <img src={leagueLogoUrl} alt="Deep Deck League" />
          <span>Learner</span>
        </a>
        <span className="local-badge">● Local workbench</span>
        <nav aria-label="Main navigation">
          {pages.map((item) => (
            <button
              className={page === item.id ? "active" : ""}
              type="button"
              key={item.id}
              onClick={() => setPage(item.id)}
            >
              <span>{item.glyph}</span>
              {item.label}
            </button>
          ))}
        </nav>
        <div className="aside-foot">
          <small>Public AI laboratory</small>
          <a
            className="patreon-link"
            href={patreonUrl}
            target="_blank"
            rel="noreferrer"
          >
            <PatreonMark />
            <span>Support on Patreon</span>
            <b aria-hidden="true">↗</b>
          </a>
          <a
            className="source-link"
            href="https://github.com/dd-the-dd/DeepDeckLearner"
            target="_blank"
            rel="noreferrer"
          >
            View source <span aria-hidden="true">↗</span>
          </a>
        </div>
      </aside>
      <main>
        <header>
          <div>
            <span className="eyebrow">Deep Deck AI laboratory</span>
            <h1 ref={pageHeading} tabIndex={-1}>
              {pageHeadings[page]}
            </h1>
          </div>
          <div className="health">
            {page === "playtest" ? (
              <>
                <StatusDot
                  ready={Boolean(status?.engine.healthy)}
                  label="Engine"
                />
                <StatusDot ready={Boolean(status?.pixi.built)} label="Pixi" />
              </>
            ) : page === "compete" ? (
              <StatusDot ready={accountReady} label="League account" />
            ) : (
              <StatusDot
                ready={Boolean(status?.controller.ready)}
                label="Workbench"
              />
            )}
            {running > 0 && (
              <span className="running-count">{running} running</span>
            )}
          </div>
        </header>
        {loadError && (
          <p className="controller-error" role="alert">
            Local controller: {loadError}
          </p>
        )}
        {page === "overview" && (
          <>
            <section className="intro">
              <p>
                Choose your outcome. DeepDeckLearner will show only the setup
                and decisions needed to reach it.
              </p>
              <div className="first-run-callout">
                <span>New here?</span>
                <strong>Start with Train an agent.</strong>
                <p>
                  Choose the model and decks it should learn before starting a
                  real training run.
                </p>
              </div>
              <div className="workflow-grid">
                {(
                  [
                    "local-training",
                    "local-playtest",
                    "matchmaking",
                  ] as Workflow[]
                ).map((item) => (
                  <WorkflowCard
                    key={item}
                    workflow={item}
                    status={status}
                    onSelect={selectWorkflow}
                  />
                ))}
              </div>
            </section>
            <WorkspaceSummary status={status} account={account} />
            <AccountSetup status={status} account={account} refresh={refresh} />
            {activityJobs.length > 0 && (
              <JobsPanel jobs={activityJobs} refresh={refresh} />
            )}
          </>
        )}
        {page === "train" && (
          <>
            <section className={`flow-section ${setupReady ? "complete" : "attention"}`}>
              <button className="flow-section-toggle" type="button" aria-expanded={setupOpen} onClick={() => setSetupOpen(!setupOpen)}>
                <span><b>1</b><span><strong>Application setup</strong><small>API key, Engine and Pixi</small></span></span>
                <span className="flow-section-state">{setupReady ? "Ready" : "Action required"}<i>{setupOpen ? "−" : "+"}</i></span>
              </button>
              {setupOpen && <div className="flow-section-body"><AccountSetup status={status} account={account} refresh={refresh} /><DependencyPanel status={status} jobs={jobs} refresh={refresh} /></div>}
            </section>

            <section className="flow-section">
              <button className="flow-section-toggle" type="button" aria-expanded={trainingOpen} onClick={() => setTrainingOpen(!trainingOpen)}>
                <span><b>2</b><span><strong>Your agents</strong><small>Review trained agents and past runs</small></span></span>
                <span className="flow-section-state">{userModels.length} agent{userModels.length === 1 ? "" : "s"}<i>{trainingOpen ? "−" : "+"}</i></span>
              </button>
              {trainingOpen && <div className="flow-section-body">
                <div className="agent-toolbar">
                  <div><span className="eyebrow">Your AI</span><h2>Configured agents</h2><p>Identity, architecture, owned weights and the immutable deck pool each agent consumes.</p></div>
                  <button className="primary" type="button" onClick={() => setCreatingTraining(!creatingTraining)}>{creatingTraining ? "Cancel" : "+ New agent"}</button>
                </div>
                <AgentCatalog models={userModels} resources={resources} refresh={refresh} />
                {creatingTraining && <div className="new-training"><LocalTrainingForm status={status} account={account} refresh={refresh} /></div>}
              </div>}
            </section>
          </>
        )}
        {page === "jobs" && <JobsDashboard account={account} models={userModels} resources={resources} jobs={activityJobs} games={games} refresh={refresh} onOpenGame={(game) => { if (game.jobId) { setOpenedPlaytestJobId(game.jobId); setPage("playtest"); } }} />}
        {page === "playtest" && (
          <>
            <WorkflowJourney
              active={playtestStep}
              steps={[
                { label: "Prepare", detail: "Engine + Pixi" },
                { label: "Choose", detail: "Agent and decks" },
                { label: "Run", detail: "Inspect agent activity" },
              ]}
            />
            <DependencyPanel status={status} jobs={jobs} refresh={refresh} />
            {stackReady ? (
              <PlaytestForm status={status} models={userModels} refresh={refresh} onStarted={setOpenedPlaytestJobId} />
            ) : (
              <LockedNextStep />
            )}
            {activityJobs.length > 0 && (
              <JobsPanel jobs={activityJobs} refresh={refresh} />
            )}
          </>
        )}
        {page === "compete" && (
          <>
            <WorkflowJourney
              active={leagueStep}
              steps={[
                { label: "Connect", detail: "Add your account key" },
                { label: "Choose", detail: "Agent and legal deck" },
                { label: "Queue", detail: "Join matchmaking" },
              ]}
            />
            <MatchmakingForm
              status={status}
              models={userModels}
              refresh={refresh}
            />
            {activityJobs.length > 0 && (
              <JobsPanel jobs={activityJobs} refresh={refresh} />
            )}
          </>
        )}
        {page === "statistics" && <StatisticsPage jobs={activityJobs} decks={deckStatistics} training={trainingStatistics} metricWindow={metricWindow} onMetricWindowChange={changeMetricWindow} />}
        {page === "representation" && (
          <section className="panel prose">
            <span className="eyebrow">Magic → tensor</span>
            <h2>A decision, not a screenshot</h2>
            <p>
              The encoder combines the observable game state, each legal action,
              known deck context, and a previous-state delta. V11 keeps four
              multiplayer value slots; V12 specializes the value head for
              two-player Legacy.
            </p>
            <div className="tensor-flow">
              <span>Game observation</span>
              <b>+</b>
              <span>Legal action</span>
              <b>+</b>
              <span>Known deck</span>
              <b>→</b>
              <span>Feature tensor</span>
            </div>
            <p>
              Feature indices and masks are versioned in{" "}
              <code>deepdeck_examples.deep_learning.encoding</code>. See the ML
              guide before changing them: a checkpoint is only compatible with
              the schema it learned.
            </p>
          </section>
        )}
        {page === "models" && (
          <section className="model-grid">
            <article className="panel model">
              <span>V12</span>
              <h2>Structured two-player policy</h2>
              <p>
                Legacy architecture with structured observations, encoded legal
                actions, two relative value slots, self-play collection and PPO
                updates. Deep Deck publishes the implementation; training creates
                the user's own local weights.
              </p>
            </article>
            <article className="panel model">
              <span>V11</span>
              <h2>Structured multiplayer policy</h2>
              <p>
                Commander architecture with four multiplayer value slots,
                shared-policy self-play and PPO updates. Its weights are created
                and retained in the user's local workspace.
              </p>
            </article>
          </section>
        )}
      </main>
    </div>
  );
}
