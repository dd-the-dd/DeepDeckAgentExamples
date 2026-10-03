import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import App from "./App";
import type { CapabilityStatus, Job, LocalModel } from "./api";

const status: CapabilityStatus = {
  controller: { ready: true, version: "test" },
  paths: {
    project: "project",
    trajectory: "trajectory.jsonl",
    checkpoints: "checkpoints",
  },
  sdk: { ready: true },
  torch: { ready: true },
  engine: {
    source_available: false,
    revision: null,
    pinned_revision: "engine-pinned",
    synced: false,
    dirty: false,
    built: false,
    url: "http://127.0.0.1:8787",
    healthy: false,
  },
  pixi: {
    source_available: false,
    built: false,
    build_present: false,
    built_revision: null,
    revision: null,
    pinned_revision: "pixi-pinned",
    synced: false,
    dirty: false,
  },
  hosted: {
    api_key_configured: false,
    trajectory_training: false,
    reason: "Not available yet.",
  },
  workflows: {},
};

const stackJob: Job = {
  id: "stack-job",
  kind: "dependency.stack.prepare",
  label: "Local Engine + Pixi setup",
  argv: [],
  status: "queued",
  created_at: "2026-08-29T00:00:00Z",
  started_at: null,
  finished_at: null,
  exit_code: null,
  artifact_path: null,
  logs: [],
};

function response(body: unknown) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

const resources = {
  system: {
    ramTotalBytes: 16_000_000_000,
    ramUsedBytes: 4_000_000_000,
    ramAvailableBytes: 12_000_000_000,
    gpuTotalBytes: null,
    gpuUsedBytes: null,
  },
  workers: [],
  engine: {
    ramBytes: 0,
    activeLocalGames: 0,
    ramPerGameEstimate: 0,
    attribution: "Shared Engine RSS divided by active local games.",
  },
};

const selfPlayModel: LocalModel = {
  id: "self-play-agent",
  name: "Self-play Agent",
  architecture: "v13",
  format: "legacy",
  description: "Graph agent",
  createdAt: "2026-09-01T00:00:00Z",
  runPath: "self-play-run",
  checkpointPath: "self-play-checkpoint",
  status: "stopped",
  ready: true,
  reservePlaytest: true,
  selfPlayAllSeats: true,
  decks: [],
  diskBytes: 1024,
  weightsBytes: 512,
  trainingState: { completedGames: 0, trainingStep: 0, parallelGames: 0, activeGames: 0 },
};

function readResponse(url: string, currentStatus: CapabilityStatus = status) {
  if (url.endsWith("/api/v1/status")) return response(currentStatus);
  if (url.endsWith("/api/v1/account/status")) return response({
    configured: currentStatus.hosted.api_key_configured,
    valid: currentStatus.hosted.api_key_configured,
    reason: currentStatus.hosted.api_key_configured ? "League account connected." : "Add an agent API key.",
  });
  if (url.endsWith("/api/v1/models")) return response({ items: [] });
  if (url.endsWith("/api/v1/resources")) return response(resources);
  if (url.endsWith("/api/v1/games")) return response({ items: [] });
  if (url.endsWith("/api/v1/statistics/decks")) return response({ items: [] });
  if (url.endsWith("/api/v1/statistics/training")) return response({ items: [] });
  if (url.endsWith("/api/v1/training/deck-pool")) return response({ decks: [] });
  return response([]);
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("guided onboarding", () => {
  test("opens directly in Agent configuration without a League tab", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        return readResponse(url);
      }),
    );

    render(<App />);

    expect(
      await screen.findByRole("heading", { name: "Agent configuration" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Application setup/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "What do you want to do?" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "League" })).not.toBeInTheDocument();
  });

  test("does not reopen a recovered local game when the site loads", async () => {
    const recoveredPlaytest: Job = {
      id: "recovered-playtest",
      kind: "playtest.agent",
      label: "Recovered local game",
      argv: [],
      status: "running",
      created_at: "2026-08-31T00:00:00Z",
      started_at: "2026-08-31T00:00:01Z",
      finished_at: null,
      exit_code: null,
      artifact_path: null,
      logs: [],
      model_id: "local-model",
      worker_slots: 1,
      details: {
        sessionId: "game-session:6",
        engineUrl: "http://127.0.0.1:8787",
      },
    };
    const occupiedResources = {
      ...resources,
      workers: [{
        jobId: recoveredPlaytest.id,
        modelId: "local-model",
        label: recoveredPlaytest.label,
        kind: recoveredPlaytest.kind,
        pids: [123],
        workerSlots: 1,
        ramBytes: 100,
        gpuBytes: null,
        ramPerWorkerEstimate: 100,
        gpuPerWorkerEstimate: null,
      }],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/v1/jobs")) return response([recoveredPlaytest]);
        if (url.endsWith("/api/v1/resources")) return response(occupiedResources);
        return readResponse(url);
      }),
    );

    render(<App />);

    expect(await screen.findByText("1 running")).toBeInTheDocument();
    expect(screen.queryByText("Opening Pixi…")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Local playable table")).not.toBeInTheDocument();
  });

  test("stops and refreshes a playtest job with visible async feedback", async () => {
    const playtest: Job = {
      id: "playtest-job",
      kind: "playtest.agent",
      label: "Local AI playtest",
      argv: [],
      status: "running",
      created_at: "2026-09-01T00:00:00Z",
      started_at: "2026-09-01T00:00:01Z",
      finished_at: null,
      exit_code: null,
      artifact_path: null,
      logs: ["Game ready."],
      model_id: "local-model",
      worker_slots: 1,
    };
    let running = true;
    let finishStop: (value: Response) => void = () => undefined;
    const pendingStop = new Promise<Response>((resolve) => { finishStop = resolve; });
    let jobReads = 0;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/jobs/playtest-job/stop") && init?.method === "POST") {
        return pendingStop;
      }
      if (url.endsWith("/api/v1/jobs")) {
        jobReads += 1;
        return response(running ? [playtest] : [{ ...playtest, status: "stopped" }]);
      }
      if (url.endsWith("/api/v1/session")) return response({ token: "local-token" });
      return readResponse(url);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: /Play against your AI/ }));
    const stop = await screen.findByRole("button", { name: "Stop" });
    fireEvent.click(stop);
    expect(await screen.findByRole("button", { name: "Stoppingâ€¦" })).toHaveAttribute("aria-busy", "true");

    running = false;
    finishStop(new Response(JSON.stringify({ ...playtest, status: "stopped" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument());

    const readsBeforeRefresh = jobReads;
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(jobReads).toBeGreaterThan(readsBeforeRefresh));
  });

  test("one local setup button requests the composite controller job", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/status")) return response(status);
      if (url.endsWith("/api/v1/session"))
        return response({ token: "local-token" });
      if (url.endsWith("/api/v1/jobs") && init?.method === "POST")
        return response(stackJob);
      return readResponse(url);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.click(
      await screen.findByRole("button", { name: "Set up Engine + Pixi" }),
    );

    await waitFor(() => {
      const request = fetchMock.mock.calls.find(
        ([, init]) => init?.method === "POST",
      );
      expect(request).toBeDefined();
      expect(JSON.parse(String(request?.[1]?.body))).toEqual({
        kind: "dependency.stack.prepare",
      });
    });
  });

  test("shows live progress and logs while compatible sources are syncing", async () => {
    const runningStackJob: Job = {
      ...stackJob,
      status: "running",
      started_at: "2026-08-29T00:00:01Z",
      logs: ["Fetching the reviewed Engine revision…"],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/v1/jobs")) return response([runningStackJob]);
        return readResponse(url);
      }),
    );

    render(<App />);

    const loader = await screen.findByRole("status", {
      name: "Syncing compatible Engine and Pixi sources",
    });
    expect(within(loader).getByText("Fetching the reviewed Engine revision…")).toBeInTheDocument();
    expect(within(loader).getByText("Sources").closest("li")).toHaveClass("active");
  });

  test("identifies a standalone Pixi build as the active slow operation", async () => {
    const syncedStatus: CapabilityStatus = {
      ...status,
      engine: { ...status.engine, source_available: true, revision: "engine-pinned", synced: true },
      pixi: { ...status.pixi, source_available: true, revision: "pixi-pinned", synced: true },
    };
    const pixiJob: Job = {
      ...stackJob,
      id: "pixi-job",
      kind: "dependency.pixi.prepare",
      label: "Prepare DeepDeckPixi",
      status: "running",
      started_at: "2026-08-29T00:00:01Z",
      logs: ["Building the production Pixi bundle…"],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/v1/status")) return response(syncedStatus);
        if (url.endsWith("/api/v1/jobs")) return response([pixiJob]);
        return readResponse(url, syncedStatus);
      }),
    );

    render(<App />);

    const loader = await screen.findByRole("status", {
      name: "Building the Pixi visual client",
    });
    expect(within(loader).getByText("Building the production Pixi bundle…")).toBeInTheDocument();
    expect(within(loader).getByText("Pixi").closest("li")).toHaveClass("active");
  });

  test("a ready local stack can still be verified and repaired", async () => {
    const readyStatus: CapabilityStatus = {
      ...status,
      engine: {
        ...status.engine,
        source_available: true,
        revision: "engine-pinned",
        synced: true,
        built: true,
        healthy: true,
      },
      pixi: {
        ...status.pixi,
        source_available: true,
        revision: "pixi-pinned",
        built_revision: "pixi-pinned",
        synced: true,
        built: true,
        build_present: true,
      },
      hosted: { ...status.hosted, api_key_configured: true },
    };
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/status")) return response(readyStatus);
      if (url.endsWith("/api/v1/session")) return response({ token: "local-token" });
      if (url.endsWith("/api/v1/jobs") && init?.method === "POST") return response(stackJob);
      return readResponse(url, readyStatus);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    const setupToggle = await screen.findByRole("button", { name: /Application setup/i });
    await waitFor(() => expect(setupToggle).toHaveAttribute("aria-expanded", "false"));
    fireEvent.click(setupToggle);
    await waitFor(() => expect(setupToggle).toHaveAttribute("aria-expanded", "true"));
    expect(await screen.findByText("Account connected")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Verify & repair Engine + Pixi" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([, init]) => init?.method === "POST")).toBe(true));
  });

  test("shows every agent in one allocation table and saves a row", async () => {
    const models = [
      {
        id: "agent-one",
        name: "Agent One",
        architecture: "v12",
        format: "legacy",
        description: "Legacy agent",
        createdAt: "2026-08-30T00:00:00Z",
        runPath: "run-one",
        checkpointPath: "checkpoint-one",
        status: "running",
        ready: true,
        reservePlaytest: true,
        decks: [],
      },
      {
        id: "agent-two",
        name: "Agent Two",
        architecture: "v11",
        format: "commander",
        description: "Commander agent",
        createdAt: "2026-08-29T00:00:00Z",
        runPath: "run-two",
        checkpointPath: "checkpoint-two",
        status: "stopped",
        ready: false,
        reservePlaytest: true,
        decks: [],
      },
    ];
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url.endsWith("/api/v1/models")) return response({ items: models });
      if (url.includes("/api/v1/models/") && url.endsWith("/resources")) {
        return response({ trainingMatches: 2, leagueMatches: 1, localMatches: 1, gpuMemoryMb: 4096 });
      }
      if (url.endsWith("/api/v1/session")) return response({ token: "local-token" });
      return readResponse(url);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: /Jobs/ }));
    const table = await screen.findByRole("table", { name: "Agent resource allocation" });
    expect(within(table).getByRole("row", { name: /Agent One/i })).toBeInTheDocument();
    expect(within(table).getByRole("row", { name: /Agent Two/i })).toBeInTheDocument();
    const trainingSlots = await screen.findByLabelText("Self-play slots for Agent One");
    const firstRow = trainingSlots.closest("tr");
    expect(firstRow).not.toBeNull();
    fireEvent.change(trainingSlots, {
      target: { value: "3" },
    });
    fireEvent.click(within(firstRow as HTMLTableRowElement).getByRole("button", { name: "Save allocation" }));

    await waitFor(() => {
      const save = fetchMock.mock.calls.find(([input, init]) =>
        String(input).endsWith("/api/v1/models/agent-one/resources") && init?.method === "PUT",
      );
      expect(save).toBeDefined();
      expect(JSON.parse(String(save?.[1]?.body))).toMatchObject({ trainingMatches: 3 });
    });
  });

  test("advertises all valid format decks to every League connection", async () => {
    const readyStatus: CapabilityStatus = {
      ...status,
      hosted: { ...status.hosted, api_key_configured: true },
    };
    const model = {
      id: "league-agent",
      name: "League Agent",
      architecture: "v12",
      format: "legacy",
      description: "Legacy agent",
      createdAt: "2026-08-30T00:00:00Z",
      runPath: "run-league",
      checkpointPath: "checkpoint-league",
      status: "running",
      ready: true,
      reservePlaytest: true,
      decks: [],
    };
    const leagueDecks = [
      { id: "deck-one", name: "Deck One", version: 1, format: "legacy", colors: [], playableCardCount: 60 },
      { id: "deck-two", name: "Deck Two", version: 1, format: "legacy", colors: [], playableCardCount: 60 },
      { id: "deck-three", name: "Deck Three", version: 1, format: "legacy", colors: [], playableCardCount: 60 },
    ];
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/status")) return response(readyStatus);
      if (url.endsWith("/api/v1/account/status")) return response({ configured: true, valid: true, reason: "Connected" });
      if (url.endsWith("/api/v1/models")) return response({ items: [model] });
      if (url.includes("/api/v1/catalog/decks?")) return response({ items: leagueDecks, pagination: { page: 1, totalPages: 1 } });
      if (url.endsWith("/api/v1/catalog/competitions")) return response({ items: [{ versionId: "legacy-season", name: "Legacy season", status: "active", format: "legacy", timeControl: "standard" }] });
      if (url.endsWith("/api/v1/models/league-agent/resources")) return response({ trainingMatches: 0, leagueMatches: 2, localMatches: 0, gpuMemoryMb: 0 });
      if (url.endsWith("/api/v1/session")) return response({ token: "local-token" });
      if (url.endsWith("/api/v1/jobs") && init?.method === "POST") return response(stackJob);
      return readResponse(url, readyStatus);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: /Jobs/ }));
    const deckModes = await screen.findByRole("group", { name: "Deck selection for League Agent" });
    fireEvent.click(within(deckModes).getByRole("button", { name: "All legal" }));
    fireEvent.click(screen.getByRole("button", { name: "Connect allocated slots" }));

    await waitFor(() => {
      const payloads = fetchMock.mock.calls
        .filter(([input, init]) => String(input).endsWith("/api/v1/jobs") && init?.method === "POST")
        .map(([, init]) => JSON.parse(String(init?.body)));
      expect(payloads).toHaveLength(1);
      expect(payloads[0]).toMatchObject({
        connections: 2,
        deck_version_ids: ["deck-one", "deck-two", "deck-three"],
        speed: "1s",
      });
    });
  });

  test("downloads selected training decks without substituting smoke data", async () => {
    const trainingStatus: CapabilityStatus = {
      ...status,
      engine: {
        ...status.engine,
        source_available: true,
        synced: true,
        built: true,
        healthy: true,
      },
      hosted: {
        api_key_configured: true,
        trajectory_training: false,
        reason: "Not available yet.",
      },
      workflows: { training_decks: false },
    };
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/status")) return response(trainingStatus);
      if (url.endsWith("/api/v1/session")) return response({ token: "local-token" });
      if (url.endsWith("/api/v1/jobs") && init?.method === "POST") return response(stackJob);
      if (url.includes("/api/v1/catalog/decks/") && init?.method === "POST") return response({ versionId: "deck-1", name: "Legacy Reanimator", format: "legacy", cardCount: 60, path: "deck.json" });
      if (url.endsWith("/api/v1/training/deck-pool") && init?.method === "PUT") return response({ decks: [] });
      if (url.includes("/api/v1/catalog/decks") && url.includes("format=commander")) return response({ items: [] });
      if (url.includes("/api/v1/catalog/decks"))
        return response({
          items: [
            {
              id: "deck-1",
              name: "Legacy Reanimator",
              version: 1,
              format: "legacy",
              colors: ["B"],
              playableCardCount: 60,
            },
            {
              id: "deck-2",
              name: "Death and Taxes",
              version: 1,
              format: "legacy",
              colors: ["W"],
              playableCardCount: 60,
            },
          ],
        });
      return readResponse(url, trainingStatus);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: /New agent/i }));

    fireEvent.click(
      await screen.findByRole("button", { name: /Legacy Reanimator/i }),
    );
    await waitFor(() => {
      const request = fetchMock.mock.calls.find((call) => {
        const [input, init] = call as unknown as [
          RequestInfo | URL,
          RequestInit?,
        ];
        return String(input).includes("/api/v1/catalog/decks/deck-1/download") && init?.method === "POST";
      });
      expect(request).toBeDefined();
      expect(fetchMock.mock.calls.some((call) => {
        const [input, init] = call as unknown as [RequestInfo | URL, RequestInit?];
        return String(input).endsWith("/api/v1/jobs")
          && init?.method === "POST"
          && JSON.parse(String(init.body)).kind === "training.smoke";
      })).toBe(false);
    });
  });

  test("edits an agent, adds a deck, and replaces an older deck version", async () => {
    const readyStatus: CapabilityStatus = {
      ...status,
      hosted: { ...status.hosted, api_key_configured: true },
    };
    const firstDeck = { id: "deck-one-v1", deckId: "deck-one", name: "Deck One", version: 1, format: "legacy", colors: [], playableCardCount: 60 };
    const updatedFirstDeck = { ...firstDeck, id: "deck-one-v2", version: 2, playableCardCount: 61 };
    const secondDeck = { id: "deck-two", deckId: "deck-two", name: "Deck Two", version: 2, format: "legacy", colors: [], playableCardCount: 60 };
    const model = {
      id: "editable-agent",
      name: "Editable Agent",
      architecture: "v12",
      format: "legacy",
      description: "Legacy agent",
      createdAt: "2026-09-01T00:00:00Z",
      runPath: "existing-run",
      checkpointPath: "existing-checkpoint",
      status: "stopped",
      ready: true,
      reservePlaytest: true,
      selfPlayAllSeats: true,
      decks: [firstDeck],
      diskBytes: 1024,
      weightsBytes: 512,
      trainingState: { completedGames: 10, trainingStep: 2, parallelGames: 1, activeGames: 0 },
    };
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/status")) return response(readyStatus);
      if (url.endsWith("/api/v1/account/status")) return response({ configured: true, valid: true, reason: "Connected" });
      if (url.endsWith("/api/v1/models") && !init?.method) return response({ items: [model] });
      if (url.includes("/api/v1/catalog/decks?")) return response({ items: [updatedFirstDeck, secondDeck] });
      if (url.endsWith("/api/v1/catalog/decks/deck-one-v2/download") && init?.method === "POST") return response({ versionId: "deck-one-v2", name: "Deck One", format: "legacy", cardCount: 61, rawCardCount: 61, path: "deck-one-v2.json" });
      if (url.endsWith("/api/v1/catalog/decks/deck-two/download") && init?.method === "POST") return response({ versionId: "deck-two", name: "Deck Two", format: "legacy", cardCount: 60, rawCardCount: 60, path: "deck-two.json" });
      if (url.endsWith("/api/v1/models/editable-agent") && init?.method === "PUT") return response({ ...model, name: "Renamed Agent", decks: [updatedFirstDeck, secondDeck] });
      if (url.endsWith("/api/v1/session")) return response({ token: "local-token" });
      return readResponse(url, readyStatus);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Edit agent and decks" }));
    fireEvent.change(screen.getByLabelText("Agent name"), { target: { value: "Renamed Agent" } });
    fireEvent.click(await within(screen.getByLabelText("Deck results for Editable Agent")).findByRole("button", { name: /Deck One/i }));
    fireEvent.click(await screen.findByRole("button", { name: /Deck Two/i }));
    fireEvent.click(await screen.findByRole("button", { name: "Save agent" }));

    await waitFor(() => {
      const request = fetchMock.mock.calls.find(([input, init]) =>
        String(input).endsWith("/api/v1/models/editable-agent") && init?.method === "PUT",
      );
      expect(request).toBeDefined();
      expect(JSON.parse(String(request?.[1]?.body))).toMatchObject({
        name: "Renamed Agent",
        decks: [{ id: "deck-one-v2", deckId: "deck-one" }, { id: "deck-two" }],
      });
    });
  });

  test("opens the agent editor when a service is active and can stop it before saving", async () => {
    const model = {
      id: "busy-agent",
      name: "Busy Agent",
      architecture: "v12" as const,
      format: "legacy" as const,
      description: "Legacy agent",
      createdAt: "2026-09-01T00:00:00Z",
      runPath: "busy-run",
      checkpointPath: "busy-checkpoint",
      status: "running",
      ready: true,
      reservePlaytest: true,
      selfPlayAllSeats: true,
      decks: [{ id: "deck-one", name: "Deck One", version: 1, format: "legacy", colors: [], playableCardCount: 60 }],
      diskBytes: 1024,
      weightsBytes: 512,
      trainingState: { completedGames: 10, trainingStep: 2, parallelGames: 1, activeGames: 1 },
    };
    const worker = {
      jobId: "league-worker",
      modelId: model.id,
      label: "League worker",
      kind: "matchmaking.agent",
      pids: [123],
      workerSlots: 1,
      ramBytes: 100,
      gpuBytes: null,
      ramPerWorkerEstimate: 100,
      gpuPerWorkerEstimate: null,
    };
    let workerActive = true;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/models") && !init?.method) return response({ items: [model] });
      if (url.endsWith("/api/v1/resources")) return response({ ...resources, workers: workerActive ? [worker] : [] });
      if (url.includes("/api/v1/catalog/decks?")) return response({ items: [] });
      if (url.endsWith("/api/v1/session")) return response({ token: "local-token" });
      if (url.endsWith("/api/v1/jobs/league-worker/stop") && init?.method === "POST") {
        workerActive = false;
        return response({ id: worker.jobId, status: "stopped" });
      }
      return readResponse(url);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Edit agent and decks" }));

    expect(screen.getByLabelText("Agent name")).toHaveValue("Busy Agent");
    expect(screen.getByRole("button", { name: "Save agent" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Stop 1 active service" }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Save agent" })).toBeEnabled());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/jobs/league-worker/stop",
      expect.objectContaining({ method: "POST" }),
    );
  });

  test("shows self-play startup progress immediately", async () => {
    let finishStart: (value: Response) => void = () => undefined;
    const pendingStart = new Promise<Response>((resolve) => { finishStart = resolve; });
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/models") && !init?.method) return response({ items: [selfPlayModel] });
      if (url.endsWith("/api/v1/models/self-play-agent/resources")) return response({ trainingMatches: 1, localMatches: 1, leagueMatches: 0, gpuMemoryMb: 0 });
      if (url.endsWith("/api/v1/jobs") && init?.method === "POST") return pendingStart;
      if (url.endsWith("/api/v1/jobs")) return response([]);
      if (url.endsWith("/api/v1/session")) return response({ token: "local-token" });
      return readResponse(url);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: /Jobs/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Start training" }));

    expect(await screen.findByText(/Starting self-play/i)).toBeInTheDocument();
    finishStart(new Response(JSON.stringify({ id: "training-job", status: "queued" }), { status: 202, headers: { "Content-Type": "application/json" } }));
    expect(await screen.findByText(/Self-play request accepted/i)).toBeInTheDocument();
  });

  test("surfaces the latest self-play worker error", async () => {
    const failedJob: Job = {
      id: "failed-training",
      kind: "training.pool",
      label: "Self-play Agent Â· V13",
      argv: [],
      status: "failed",
      created_at: "2026-09-01T00:00:00Z",
      started_at: "2026-09-01T00:00:01Z",
      finished_at: "2026-09-01T00:00:02Z",
      exit_code: 1,
      artifact_path: "self-play-run",
      logs: ["CUDA ran out of memory while starting the worker."],
      model_id: selfPlayModel.id,
    };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/models") && !init?.method) return response({ items: [selfPlayModel] });
      if (url.endsWith("/api/v1/models/self-play-agent/resources")) return response({ trainingMatches: 1, localMatches: 1, leagueMatches: 0, gpuMemoryMb: 0 });
      if (url.endsWith("/api/v1/jobs")) return response([failedJob]);
      return readResponse(url);
    }));
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: /Jobs/ }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/Self-play failed.*CUDA ran out of memory/i);
  });
});

describe("deck rating table", () => {
  test("sorts by Elo-like rating by default and lets every column change the order", async () => {
    const deckRatings = [
      { modelId: "agent", modelName: "Agent", architecture: "v12", deckVersionId: "alpha", deckName: "Alpha", format: "legacy", ratingSystem: "plackett-luce", mu: 30, sigma: 3, ordinal: 20, rank: 2, matches: 12, gameWins: 7, gameLosses: 5, winRate: 7 / 12 },
      { modelId: "agent", modelName: "Agent", architecture: "v12", deckVersionId: "beta", deckName: "Beta", format: "legacy", ratingSystem: "plackett-luce", mu: 22, sigma: 4, ordinal: 10, rank: 3, matches: 8, gameWins: 3, gameLosses: 5, winRate: 3 / 8 },
      { modelId: "agent", modelName: "Agent", architecture: "v12", deckVersionId: "gamma", deckName: "Gamma", format: "legacy", ratingSystem: "plackett-luce", mu: 36, sigma: 1, ordinal: 33, rank: 1, matches: 3, gameWins: 3, gameLosses: 0, winRate: 1 },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/v1/statistics/decks")) return response({ items: deckRatings });
        if (url.includes("/api/v1/statistics/training?")) return response({ items: [] });
        return readResponse(url);
      }),
    );

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: /Training statistics/ }));

    const table = await screen.findByRole("table");
    const rowNames = () => within(table).getAllByRole("row").slice(1).map((row) => within(row).getAllByRole("strong")[0].textContent);
    expect(screen.getByRole("columnheader", { name: /Elo-like rating/i })).toHaveAttribute("aria-sort", "descending");
    expect(rowNames()).toEqual(["Gamma", "Alpha", "Beta"]);

    fireEvent.click(screen.getByRole("button", { name: "Matches" }));
    expect(rowNames()).toEqual(["Alpha", "Beta", "Gamma"]);
    expect(screen.getByRole("columnheader", { name: /Matches/i })).toHaveAttribute("aria-sort", "descending");

    fireEvent.click(screen.getByRole("button", { name: "Matches" }));
    expect(rowNames()).toEqual(["Gamma", "Beta", "Alpha"]);
    expect(screen.getByRole("columnheader", { name: /Matches/i })).toHaveAttribute("aria-sort", "ascending");
  });
});
