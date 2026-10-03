import {
  Component,
  Suspense,
  lazy,
  useEffect,
  useMemo,
  useState,
  type ErrorInfo,
  type ReactNode,
} from "react";

import type { EngineView } from "./LocalPixiTable";
import type { DeckPresentation, SavedReplay } from "./api";

const LocalPixiRenderer = lazy(() => import("./LocalPixiRenderer"));

class ReplayPixiBoundary extends Component<
  { children: ReactNode },
  { error: string }
> {
  state = { error: "" };

  static getDerivedStateFromError(reason: unknown) {
    return {
      error: reason instanceof Error ? reason.message : "Pixi could not render this replay.",
    };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.warn("The Pixi replay could not be rendered.", error, info);
  }

  render() {
    if (this.state.error) {
      return <div className="pixi-replay-error" role="alert">Pixi replay error: {this.state.error}</div>;
    }
    return this.props.children;
  }
}

// Saved self-play frames are Engine session views without a sessionId because
// they are immutable and no longer attached to a live Engine process.
// eslint-disable-next-line react-refresh/only-export-components -- exercised directly by regression tests.
export function replayFrameView(replay: SavedReplay, frameIndex: number): EngineView | null {
  const frame = replay.frames[frameIndex];
  if (!frame) return null;
  const rawState = frame.state as Record<string, unknown> | undefined;
  const players = Array.isArray(rawState?.players) ? rawState.players : [];
  const restoreCard = (candidate: unknown) => {
    if (!candidate || typeof candidate !== "object") return candidate;
    const card = candidate as Record<string, unknown>;
    if (card.definition && typeof card.definition === "object") return card;
    return {
      ...card,
      controller: card.controller ?? card.controllerId,
      owner: card.owner ?? card.ownerId,
      definition: {
        id: card.cardId ?? card.id ?? card.instanceId,
        name: card.name ?? "Unknown card",
        power: card.power ?? null,
        toughness: card.toughness ?? null,
      },
    };
  };
  const restoredPlayers = players.map((candidate) => {
    if (!candidate || typeof candidate !== "object") return candidate;
    const player = candidate as Record<string, unknown>;
    return Object.fromEntries(Object.entries(player).map(([key, value]) => [
      key,
      ["battlefield", "commandZone", "exile", "graveyard", "hand", "library", "sideboard"].includes(key) && Array.isArray(value)
        ? value.map(restoreCard)
        : value,
    ]));
  });
  const playerIndex = (playerId: unknown) => players.findIndex((candidate) => (
    candidate && typeof candidate === "object" && (candidate as Record<string, unknown>).id === playerId
  ));
  const rawStack = Array.isArray(rawState?.stack) ? rawState.stack : [];
  const restoredStack = rawStack.flatMap((candidate, index) => {
    if (!candidate || typeof candidate !== "object") return [];
    const item = candidate as Record<string, unknown>;
    if (item.card && typeof item.card === "object") return [{ ...item, card: restoreCard(item.card) }];
    if (!item.instanceId && !item.cardId && !item.name) return [];
    return [{
      id: item.id ?? `replay-stack:${index}`,
      controller: item.controller ?? item.controllerId,
      card: restoreCard(item),
    }];
  });
  const restoredState = rawState ? {
    ...rawState,
    activePlayer: typeof rawState.activePlayer === "number"
      ? rawState.activePlayer
      : Math.max(0, playerIndex(rawState.activePlayerId)),
    priorityPlayer: typeof rawState.priorityPlayer === "number"
      ? rawState.priorityPlayer
      : Math.max(0, playerIndex(rawState.priorityPlayerId)),
    players: restoredPlayers,
    stack: restoredStack,
  } : rawState;
  return {
    ...(frame as unknown as EngineView),
    schemaVersion: "mtg-game-session/v1",
    sessionId: `replay:${replay.id}`,
    revision: Number(frame.revision ?? frameIndex + 1),
    state: restoredState as EngineView["state"],
  };
}

async function loadDeckPresentation(versionId: string) {
  const response = await fetch(
    `/api/v1/catalog/decks/${encodeURIComponent(versionId)}/presentation`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Card artwork request failed (${response.status}).`);
  }
  return response.json() as Promise<DeckPresentation>;
}

export default function LocalPixiReplay({
  deckVersionIds,
  frameIndex,
  onFrameIndexChange,
  replay,
}: {
  deckVersionIds: string[];
  frameIndex: number;
  onFrameIndexChange: (frame: number) => void;
  replay: SavedReplay;
}) {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [deckSelections, setDeckSelections] = useState<DeckPresentation[]>([]);
  const [artworkWarning, setArtworkWarning] = useState("");
  const deckVersionIdsKey = [...new Set(deckVersionIds.filter(Boolean))].join("|");
  const lastFrame = Math.max(0, replay.frames.length - 1);
  const view = useMemo(
    () => replayFrameView(replay, Math.min(frameIndex, lastFrame)),
    [frameIndex, lastFrame, replay],
  );
  const state = view?.state;
  const selectedAction = replay.frames[frameIndex]?.selectedAction as
    | Record<string, unknown>
    | undefined;

  useEffect(() => {
    setPlaying(false);
  }, [replay.id]);

  useEffect(() => {
    if (!playing) return;
    if (frameIndex >= lastFrame) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(
      () => onFrameIndexChange(Math.min(lastFrame, frameIndex + 1)),
      700 / speed,
    );
    return () => window.clearTimeout(timer);
  }, [frameIndex, lastFrame, onFrameIndexChange, playing, speed]);

  useEffect(() => {
    let active = true;
    const ids = deckVersionIdsKey.split("|").filter(Boolean);
    setDeckSelections([]);
    setArtworkWarning("");
    if (!ids.length) return;
    void Promise.all(ids.map(loadDeckPresentation))
      .then((presentations) => {
        if (active) setDeckSelections(presentations);
      })
      .catch((reason: unknown) => {
        if (active) {
          setArtworkWarning(
            reason instanceof Error ? reason.message : "Card artwork metadata is unavailable.",
          );
        }
      });
    return () => { active = false; };
  }, [deckVersionIdsKey]);

  if (!view) {
    return <div className="pixi-replay-error" role="alert">This replay has no playable frames.</div>;
  }

  const togglePlayback = () => {
    if (!playing && frameIndex >= lastFrame) onFrameIndexChange(0);
    setPlaying((current) => !current);
  };

  return (
    <section className="pixi-replay" aria-label="Pixi replay player">
      <div className="pixi-replay-stage">
        <ReplayPixiBoundary>
          <Suspense fallback={<div className="pixi-replay-loading" role="status">Loading Pixi replay…</div>}>
            <LocalPixiRenderer
              deckSelections={deckSelections}
              matchup={String(replay.metadata.matchupId ?? replay.id)}
              mode="replay"
              view={view}
              onAction={() => undefined}
              onExit={() => undefined}
            />
          </Suspense>
        </ReplayPixiBoundary>
        {artworkWarning && <div className="pixi-replay-warning" role="status">Artwork unavailable; replay continues with card names.</div>}
      </div>
      <div className="pixi-replay-controls">
        <button type="button" onClick={() => onFrameIndexChange(0)} disabled={frameIndex === 0} aria-label="First frame">↤</button>
        <button type="button" onClick={() => onFrameIndexChange(Math.max(0, frameIndex - 1))} disabled={frameIndex === 0}>Previous</button>
        <button type="button" className="primary" onClick={togglePlayback}>{playing ? "Pause" : "Play"}</button>
        <button type="button" onClick={() => onFrameIndexChange(Math.min(lastFrame, frameIndex + 1))} disabled={frameIndex >= lastFrame}>Next</button>
        <label>Speed<select aria-label="Replay speed" value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
        <span>Frame <b>{frameIndex + 1}</b> / {replay.frames.length}</span>
      </div>
      <input
        aria-label="Replay frame"
        type="range"
        min="0"
        max={lastFrame}
        value={Math.min(frameIndex, lastFrame)}
        onChange={(event) => {
          setPlaying(false);
          onFrameIndexChange(Number(event.target.value));
        }}
      />
      <div className="pixi-replay-caption">
        <span>Turn <b>{String(state?.turnNumber ?? "—")}</b></span>
        <span>Step <b>{String(state?.step ?? "—")}</b></span>
        <span>Decision <b>{String(view.decision?.kind ?? "terminal")}</b></span>
        <span>Action <b>{String(selectedAction?.label ?? selectedAction?.id ?? "—")}</b></span>
      </div>
    </section>
  );
}
