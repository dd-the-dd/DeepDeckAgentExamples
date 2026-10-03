import {
  Component,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useState,
  type ErrorInfo,
  type ReactNode,
} from "react";

import type { DeckPresentation } from "./api";

type EngineAction = {
  id: string;
  kind?: string;
  label?: string;
  playerId?: string;
};

export type EngineView = {
  schemaVersion?: string;
  sessionId: string;
  revision: number;
  state?: {
    status?: string;
    turnNumber?: number;
    step?: string;
    activePlayer?: number;
    players?: Array<{
      id: string;
      name?: string;
      life?: number;
      hand?: unknown[];
      library?: unknown[];
      graveyard?: unknown[];
      battlefield?: unknown[];
    }>;
    outcome?: { winner?: string | null; reason?: string } | null;
  };
  decision?: {
    id: string;
    kind?: string;
    playerId?: string;
    prompt?: string;
    options?: EngineAction[];
  } | null;
};

const LocalPixiRenderer = lazy(() => import("./LocalPixiRenderer"));

class OptionalPixiBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.warn("The optional Pixi table could not be rendered.", error, info);
  }

  render() {
    if (this.state.failed) {
      return (
        <div className="pixi-renderer-fallback" role="status">
          Visual table unavailable. Reload the page to reconnect to this game.
        </div>
      );
    }
    return this.props.children;
  }
}

function engineEndpoint(engineUrl: string, sessionId: string) {
  // Engine session IDs contain a colon and the current Engine router matches it
  // literally instead of decoding an escaped path segment.
  return `${engineUrl.replace(/\/$/, "")}/game/sessions/${sessionId}`;
}

async function responseError(response: Response) {
  const message = await response.text();
  return message || `Engine request failed (${response.status}).`;
}

async function readSession(engineUrl: string, sessionId: string) {
  const response = await fetch(engineEndpoint(engineUrl, sessionId));
  if (!response.ok) throw new Error(await responseError(response));
  return response.json() as Promise<EngineView>;
}

async function submitSessionAction(
  engineUrl: string,
  view: EngineView,
  actionId: string,
  extra: Record<string, unknown> = {},
) {
  if (!view.decision) throw new Error("This decision is no longer active.");
  const response = await fetch(`${engineEndpoint(engineUrl, view.sessionId)}/actions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      actionId,
      decisionId: view.decision.id,
      revision: view.revision,
      ...extra,
    }),
  });
  if (!response.ok) throw new Error(await responseError(response));
  return response.json() as Promise<EngineView>;
}

// Polling and action requests can overlap. Never let a slower GET response
// replace the newer state returned by an action that just completed.
// eslint-disable-next-line react-refresh/only-export-components -- exported for race-condition coverage.
export function latestEngineView(current: EngineView | null, incoming: EngineView) {
  if (!current || current.sessionId !== incoming.sessionId) return incoming;
  return incoming.revision >= current.revision ? incoming : current;
}

function LoadingTable({ label }: { label: string }) {
  return (
    <div className="game-loading-visual" role="status" aria-live="polite">
      <div className="game-loading-cards" aria-hidden="true">
        <i />
        <i />
        <i />
      </div>
      <strong>{label}</strong>
      <span>Connecting the player, the agent, and their decks</span>
    </div>
  );
}

export default function LocalPixiTable({ deckVersionIds = [], engineUrl, sessionId, matchup, onClose }: {
  deckVersionIds?: string[];
  engineUrl: string;
  sessionId: string;
  matchup: string;
  onClose: () => void;
}) {
  const [view, setView] = useState<EngineView | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [deckSelections, setDeckSelections] = useState<DeckPresentation[]>([]);
  const deckVersionIdsKey = deckVersionIds.join("|");

  const acceptView = useCallback((incoming: EngineView) => {
    setView((current) => latestEngineView(current, incoming));
  }, []);

  const refresh = useCallback(async () => {
    const next = await readSession(engineUrl, sessionId);
    acceptView(next);
    setError("");
  }, [acceptView, engineUrl, sessionId]);

  const leave = useCallback(async () => {
    try {
      await fetch(engineEndpoint(engineUrl, sessionId), { method: "DELETE" });
    } catch {
      // The controller still stops the agent if Engine already closed the session.
    }
    onClose();
  }, [engineUrl, onClose, sessionId]);

  const submit = useCallback(async (
    actionId: string,
    extra: Record<string, unknown> = {},
  ) => {
    if (!view || busy) return;
    setBusy(true);
    setError("");
    try {
      acceptView(await submitSessionAction(engineUrl, view, actionId, extra));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The action was rejected.");
      try {
        await refresh();
      } catch {
        // Preserve the original action error.
      }
    } finally {
      setBusy(false);
    }
  }, [acceptView, busy, engineUrl, refresh, view]);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const next = await readSession(engineUrl, sessionId);
        if (active) {
          acceptView(next);
          setError("");
        }
      } catch (reason) {
        if (active) {
          setError(reason instanceof Error ? reason.message : "Unable to read the game session.");
        }
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 1_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [acceptView, engineUrl, sessionId]);

  useEffect(() => {
    let active = true;
    const uniqueIds = [...new Set(deckVersionIdsKey.split("|").filter(Boolean))];
    if (!uniqueIds.length) return;
    void Promise.all(uniqueIds.map(async (versionId) => {
      const response = await fetch(
        `/api/v1/catalog/decks/${encodeURIComponent(versionId)}/presentation`,
      );
      if (!response.ok) throw new Error(await responseError(response));
      return response.json() as Promise<DeckPresentation>;
    })).then((selections) => {
      if (active) setDeckSelections(selections);
    }).catch((reason) => {
      console.warn("Card artwork metadata could not be loaded.", reason);
    });
    return () => { active = false; };
  }, [deckVersionIdsKey]);

  return (
    <section className="local-pixi-table" aria-label="Local playable table">
      <main className="local-game-stage">
        {error && <p className="form-error" role="alert">{error}</p>}
        {!view ? (
          <LoadingTable label="The Rust engine is preparing your game…" />
        ) : (
          <>
            <OptionalPixiBoundary>
              <Suspense fallback={<LoadingTable label="Loading the visual table…" />}>
                <LocalPixiRenderer
                  deckSelections={deckSelections}
                  matchup={matchup}
                  view={view}
                  onAction={(actionId, extra) => void submit(actionId, extra)}
                  onExit={() => void leave()}
                />
              </Suspense>
            </OptionalPixiBoundary>
          </>
        )}
      </main>
    </section>
  );
}
