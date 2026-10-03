import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import LocalPixiTable, { latestEngineView } from "./LocalPixiTable";

vi.mock("./LocalPixiRenderer", () => ({
  default: ({ matchup, onAction, view: currentView }: {
    matchup: string;
    onAction: (actionId: string) => void;
    view: typeof view;
  }) => <div data-testid="pixi-renderer">
    <span>{matchup}</span>
    {(currentView.decision?.options ?? []).map((action) => (
      <button key={action.id} type="button" onClick={() => onAction(action.id)}>
        {action.label}
      </button>
    ))}
  </div>,
}));

const view = {
  sessionId: "game-session:8",
  revision: 1,
  state: {
    status: "inProgress",
    turnNumber: 1,
    activePlayer: 0,
    step: "untap",
    players: [
      { id: "local-human", name: "Player deck", life: 20, hand: Array(7), library: Array(53) },
      { id: "local-agent", name: "Agent deck", life: 20, hand: Array(7), library: Array(53) },
    ],
    outcome: null,
  },
  decision: {
    id: "mulligan:local-human:0",
    kind: "mulligan",
    playerId: "local-human",
    options: [
      { id: "keep:local-human:0", kind: "keepHand", label: "Keep opening hand" },
      { id: "mulligan:local-human:0", kind: "takeMulligan", label: "Take a mulligan" },
    ],
  },
};

function response(body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  }));
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("local behavior test", () => {
  test("never rolls the table back when an older poll finishes late", () => {
    const latest = { ...view, revision: 12 };
    const stalePoll = { ...view, revision: 11 };

    expect(latestEngineView(latest, stalePoll)).toBe(latest);
    expect(latestEngineView(stalePoll, latest)).toBe(latest);
  });

  test("keeps Engine decisions usable independently from the Pixi renderer", async () => {
    const fetchMock = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
      if (String(_input).includes("/presentation")) {
        return response({ versionId: "deck-one", name: "Player deck", cards: [] });
      }
      if (init?.method === "POST") return response({ ...view, revision: 2, decision: null });
      return response(view);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <LocalPixiTable
        deckVersionIds={["deck-one", "deck-two"]}
        engineUrl="http://127.0.0.1:8787"
        sessionId="game-session:8"
        matchup="Player deck vs Agent deck"
        onClose={vi.fn()}
      />,
    );

    expect(await screen.findByText("Player deck vs Agent deck")).toBeInTheDocument();
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/v1/catalog/decks/deck-one/presentation",
      );
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/v1/catalog/decks/deck-two/presentation",
      );
    });
    fireEvent.click(screen.getByRole("button", { name: "Keep opening hand" }));

    await waitFor(() => {
      const request = fetchMock.mock.calls.find(([, init]) => init?.method === "POST");
      expect(request).toBeDefined();
      expect(String(request?.[0])).toBe(
        "http://127.0.0.1:8787/game/sessions/game-session:8/actions",
      );
      expect(JSON.parse(String(request?.[1]?.body))).toEqual({
        actionId: "keep:local-human:0",
        decisionId: "mulligan:local-human:0",
        revision: 1,
      });
    });
  });
});
