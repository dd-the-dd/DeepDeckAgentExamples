import { describe, expect, it } from "vitest";

import { replayFrameView } from "./LocalPixiReplay";
import type { SavedReplay } from "./api";

describe("Pixi replay frame adapter", () => {
  it("turns an immutable saved frame into an Engine view for Pixi", () => {
    const replay: SavedReplay = {
      schemaVersion: "deepdeck-replay/v1",
      id: "episode-42",
      createdAtUnixMs: 42,
      metadata: {},
      frames: [{
        revision: 7,
        state: {
          activePlayerId: "player-2",
          priorityPlayerId: "player-1",
          status: "inProgress",
          turnNumber: 3,
          stack: [{}],
          players: [
            { id: "player-1", hand: [{ cardId: "card-1", instanceId: "instance-1", name: "Ponder" }] },
            { id: "player-2", hand: [] },
          ],
        },
        decision: { id: "decision-7", kind: "priority", playerId: "player-1" },
        selectedAction: { id: "pass", label: "Pass priority" },
      }],
    };

    expect(replayFrameView(replay, 0)).toMatchObject({
      schemaVersion: "mtg-game-session/v1",
      sessionId: "replay:episode-42",
      revision: 7,
      state: {
        activePlayer: 1,
        priorityPlayer: 0,
        turnNumber: 3,
        stack: [],
        players: [{
          hand: [{
            definition: { id: "card-1", name: "Ponder" },
            instanceId: "instance-1",
          }],
        }, { id: "player-2" }],
      },
      decision: { id: "decision-7" },
    });
    expect(replayFrameView(replay, 1)).toBeNull();
  });
});
