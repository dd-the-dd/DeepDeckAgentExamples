/* eslint-disable @typescript-eslint/no-explicit-any -- @deepdeck/pixi exposes versioned Engine JSON. */
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import { createApp, h, shallowRef, type App as VueApp, type ShallowRef } from "vue";
import * as DeepDeckPixi from "@deepdeck/pixi";

import type { EngineView } from "./LocalPixiTable";
import type { DeckPresentation } from "./api";
import {
  actionTargetKeys,
  cardOrderLegend,
  candidateTargetIds,
  castingPaymentPresentations,
  castingXNumberChoice,
  selectPlanTarget,
  selectOnlyAttackDefender,
  tableChoiceCardIds,
  targetPlanFromAction,
  type TargetPlan,
} from "./pixiInteraction";

type Projection = Record<string, any>;
type PixiRuntime = {
  PixiGame: any;
  gameSessionActionPresentations: (actions: any[]) => any[];
  gameSessionCardCatalogFromDeckSelections: (
    selections: DeckPresentation[],
  ) => Record<string, unknown>;
  pixiCombatLinks: (combat: Record<string, unknown>) => any[];
  projectGameSessionView: (view: EngineView, options: Record<string, unknown>) => Projection;
};

type Interaction = {
  orderedIds?: string[];
  selectedIds?: Set<string>;
  targetableIds?: Set<string>;
  targeting?: boolean;
  prompt?: string;
};

export type GameNoteSort = "observed" | "name" | "type" | "mana" | "quantity";

function normalizedCardName(value: unknown) {
  return String(value ?? "").normalize("NFKC").trim().toLocaleLowerCase().replace(/\s+/gu, " ");
}

// eslint-disable-next-line react-refresh/only-export-components -- exported for presentation regression coverage.
export function gameNoteCardPresentation(card: any, cardCatalog: Record<string, unknown>) {
  const cardName = normalizedCardName(card?.name);
  const presentations = Object.values(cardCatalog) as any[];
  const metadata = presentations.find((candidate) => {
    return [candidate?.name, candidate?.flavorName].some(name => normalizedCardName(name) === cardName);
  }) ?? presentations.find((candidate) => {
    return (candidate?.faces ?? []).some((face: any) => normalizedCardName(face?.name) === cardName);
  }) ?? {};
  const hasManaValue = metadata.manaValue !== null && metadata.manaValue !== undefined;
  const manaValue = Number(metadata.manaValue);
  return {
    ...card,
    imageUrl: metadata.imageUrl ?? metadata.urlFront ?? "",
    manaValue: hasManaValue && Number.isFinite(manaValue) ? manaValue : null,
    typeLine: metadata.typeLine ?? "",
  };
}

// eslint-disable-next-line react-refresh/only-export-components -- exported for presentation regression coverage.
export function sortGameNoteCards(cards: any[], sort: GameNoteSort, direction: "asc" | "desc") {
  if (sort === "observed") return [...cards];
  const multiplier = direction === "asc" ? 1 : -1;
  const text = (value: unknown) => String(value ?? "").toLocaleLowerCase();
  return [...cards].sort((left, right) => {
    if (sort === "mana") {
      const leftValue = left.manaValue !== null && left.manaValue !== undefined && Number.isFinite(Number(left.manaValue))
        ? Number(left.manaValue)
        : null;
      const rightValue = right.manaValue !== null && right.manaValue !== undefined && Number.isFinite(Number(right.manaValue))
        ? Number(right.manaValue)
        : null;
      if (leftValue === null || rightValue === null) {
        if (leftValue === rightValue) return text(left.name).localeCompare(text(right.name));
        return leftValue === null ? 1 : -1;
      }
      if (leftValue !== rightValue) return (leftValue - rightValue) * multiplier;
    } else if (sort === "quantity") {
      const comparison = (Number(left.count ?? 0) - Number(right.count ?? 0)) * multiplier;
      if (comparison !== 0) return comparison;
    } else {
      const field = sort === "type" ? "typeLine" : "name";
      const comparison = text(left[field]).localeCompare(text(right[field])) * multiplier;
      if (comparison !== 0) return comparison;
    }
    return text(left.name).localeCompare(text(right.name));
  });
}

export type StackPlaybackEntry = {
  cardInstanceId: string;
  count: number;
  durationMs: number;
  firstSequence: number;
  kind: "spell" | "stackCopy" | "activatedAbility" | "triggeredAbility";
  lastSequence: number;
  stackId: string;
};

const pixi = DeepDeckPixi as unknown as PixiRuntime;

function sceneCard(card: any, kind: string, zone: string, interaction: Interaction) {
  const id = String(card?.id ?? "");
  const selected = Boolean(interaction.selectedIds?.has(id));
  const targetable = !selected && Boolean(interaction.targetableIds?.has(id));
  const selectedIndex = interaction.orderedIds?.indexOf(id) ?? -1;
  return {
    actionable: selected || (!interaction.targeting && Boolean(card?.actionState?.actionable)),
    attachedTo: card?.state?.attachedTo ?? null,
    commander: Boolean(card?.isCommander || zone === "commandZone"),
    counters: Object.values(card?.state?.counters ?? {}).reduce<number>(
      (sum, value) => sum + Number(value ?? 0),
      0,
    ),
    faceDown: Boolean(card?.faceDown),
    id: card?.id,
    imageUrl: card?.imageUrl ?? "",
    kind,
    knownToViewer: Boolean(card?.knownToViewer),
    linkedTo: card?.state?.linkedExileSourceId ?? null,
    name: card?.name ?? "Unknown card",
    raw: card,
    selectionOrder: selectedIndex >= 0 ? selectedIndex + 1 : null,
    sourceZone: card?.sourceZone ?? zone,
    tapped: Boolean(card?.state?.tapped),
    targetable,
    typeLine: card?.typeLine ?? "",
  };
}

// eslint-disable-next-line react-refresh/only-export-components -- exported for interaction regression coverage.
export function pixiScene(projection: Projection, interaction: Interaction = {}) {
  const players = projection.players ?? [];
  const local = players.find((player: any) => player.role === "human") ?? players[0];
  const scenePlayers = players.map((player: any) => {
    const isLocal = player.key === local?.key;
    const battlefield = player.zones?.battlefield ?? {};
    const zone = (id: string) => ({
      cards: (player.zones?.[id]?.cards ?? []).map(
        (card: any) => sceneCard(card, "zone", id, interaction),
      ),
      count: Number(player.zones?.[id]?.count ?? 0),
      id,
    });
    return {
      active: player.key === projection.step?.playerKey,
      battlefield: [
        ...(battlefield.lands ?? []).map(
          (card: any) => sceneCard(card, "land", "battlefield", interaction),
        ),
        ...(battlefield.creatures ?? []).map(
          (card: any) => sceneCard(card, "creature", "battlefield", interaction),
        ),
        ...(battlefield.nonCreaturePermanents ?? []).map(
          (card: any) => sceneCard(card, "other", "battlefield", interaction),
        ),
      ],
      commanderDamage: player.commanderDamage ?? [],
      hand: (player.zones?.hand ?? []).map((card: any) => {
        const presented = sceneCard(card, "hand", "hand", interaction);
        return isLocal || card?.knownToViewer
          ? presented
          : {
              ...presented,
              actionable: false,
              faceDown: true,
              imageUrl: "",
              name: "Hidden card",
              raw: { id: presented.id, typeLine: "Card" },
              targetable: false,
              typeLine: "Card",
            };
      }),
      handCount: Number(player.zones?.handCount ?? 0),
      id: player.key,
      life: Number(player.life ?? 0),
      local: isLocal,
      mana: Object.entries(player.zones?.manaPool ?? {}).map(([symbol, amount]) => ({ symbol, amount })),
      name: player.name,
      priority: player.key === projection.priorityPlayer?.key,
      raw: player,
      targetable: Boolean(interaction.targetableIds?.has(String(player.key))),
      zones: [zone("commandZone"), zone("graveyard"), {
        cards: Number(player.zones?.libraryCount ?? 0) ? [
          player.zones?.libraryTop
            ? sceneCard(player.zones.libraryTop, "zone", "library", interaction)
            : { faceDown: true, id: `${player.key}:library`, kind: "zone", name: "Library" },
        ] : [],
        count: Number(player.zones?.libraryCount ?? 0),
        id: "library",
      }, zone("exile")],
    };
  });
  const outcome = projection.state?.outcome;
  const winner = outcome?.winner
    ? players.find((player: any) => player.key === outcome.winner)
    : null;
  return {
    combat: pixi.pixiCombatLinks(projection.combat ?? {}),
    controls: {
      advanceLabel: interaction.targeting
        ? "Choose a highlighted target"
        : "Use the decision panel",
      autoPassActive: false,
      canAdvance: false,
      canPassPriority: !interaction.targeting && Boolean(projection.passAction),
      canPassStack: false,
      passStackActive: false,
      showPassPriority: !interaction.targeting && Boolean(projection.passAction),
      showPassStack: false,
    },
    outcome: outcome ? {
      draw: !outcome.winner,
      reason: outcome.reason,
      winnerId: outcome.winner ?? null,
      winnerName: winner?.name ?? outcome.winner ?? null,
    } : null,
    phase: projection.step?.phase ?? "",
    players: scenePlayers,
    replay: false,
    stack: (projection.stack ?? []).map((item: any) => ({
      card: item.card,
      detail: item.type === "spell" ? "Spell" : "Ability on the stack",
      id: item.id,
      imageUrl: item.card?.imageUrl ?? "",
      name: item.label,
      raw: item,
      targetable: Boolean(
        interaction.targetableIds?.has(String(item.id)) ||
        interaction.targetableIds?.has(String(item.card?.id)),
      ),
      type: item.type,
    })),
    status: interaction.prompt || (outcome
      ? winner ? `GG, ${winner.name}!` : "Game over: draw"
      : `Turn ${projection.step?.turn ?? 1} · ${projection.step?.playerName ?? "Waiting"}`),
  };
}

type PixiEventRefs = {
  card: RefObject<(payload: any) => void>;
  exit: RefObject<() => void>;
  hover: RefObject<(payload: any) => void>;
  leave: RefObject<() => void>;
  pass: RefObject<() => void>;
  player: RefObject<(player: any) => void>;
  stack: RefObject<(item: any) => void>;
};

function mountPixi(
  host: HTMLDivElement,
  initialScene: Record<string, unknown>,
  events: PixiEventRefs,
  mode: "play" | "replay",
) {
  const currentScene = shallowRef(initialScene);
  const vue = createApp({
    render: () => h(pixi.PixiGame, {
      brandName: "Deep Deck Learner",
      cardBackUrl: "/api/scryfall-images/back.png",
      mode,
      scene: currentScene.value,
      onCardClick: (payload: any) => events.card.current(payload),
      onCardHover: (payload: any) => events.hover.current(payload),
      onCardLeave: () => events.leave.current(),
      onExit: () => events.exit.current(),
      onPassPriority: () => events.pass.current(),
      onPlayerClick: (player: any) => events.player.current(player),
      onStackClick: (item: any) => events.stack.current(item),
    }),
  });
  vue.config.errorHandler = (error) => {
    console.warn("Pixi table runtime error", error);
  };
  vue.mount(host);
  return { currentScene, vue };
}

function actionLabel(action: any) {
  return String(action?.label ?? action?.kind ?? action?.id ?? "Choose action")
    .replace(/^Choose\s+/u, "");
}

// eslint-disable-next-line react-refresh/only-export-components -- exported for visibility regression coverage.
export function visibleCardNameChoices(projection: Projection) {
  const players = projection.players ?? [];
  const local = players.find((player: any) => player.role === "human") ?? players[0];
  const cards = players.flatMap((player: any) => [
    ...(player.zones?.graveyard?.cards ?? []),
    ...(player.zones?.exile?.cards ?? []),
    ...(player.zones?.hand ?? []).filter(
      (card: any) => player.key === local?.key || card?.knownToViewer,
    ),
  ]);
  return [...new Map(cards
    .filter((card: any) => card?.id && card?.name)
    .map((card: any) => [String(card.id), card])).values()];
}

// The Engine transports its complete event log with every session view. Convert only newly
// observed stack insertions into short visual frames; rules execution itself never waits.
// eslint-disable-next-line react-refresh/only-export-components -- exported for playback regression coverage.
export function stackEventPlaybackEntries(events: any[], afterSequence: number) {
  const entries: StackPlaybackEntry[] = [];
  for (const event of events ?? []) {
    const sequence = Number(event?.sequence ?? 0);
    const kind = ({
      activatedAbilityPutOnStack: "activatedAbility",
      cardCycled: "activatedAbility",
      cardTypecycled: "activatedAbility",
      equipAbilityPutOnStack: "activatedAbility",
      reflexiveTriggeredAbilityPutOnStack: "triggeredAbility",
      spellCast: "spell",
      stackObjectCopied: "stackCopy",
      triggeredAbilityCopied: "triggeredAbility",
      triggeredAbilityPutOnStack: "triggeredAbility",
    } as Record<string, StackPlaybackEntry["kind"] | undefined>)[String(event?.kind)];
    if (!kind || sequence <= afterSequence) continue;
    const cardInstanceId = String(event?.cardInstanceId ?? "");
    const stackId = String(event?.detail?.stackId ?? `stack-event:${sequence}`);
    const previous = entries.at(-1);
    if (
      kind === "triggeredAbility" &&
      previous?.kind === kind &&
      previous.cardInstanceId === cardInstanceId &&
      sequence === previous.lastSequence + 1
    ) {
      previous.count += 1;
      previous.durationMs = 3;
      previous.lastSequence = sequence;
      continue;
    }
    entries.push({
      cardInstanceId,
      count: 1,
      durationMs: 300,
      firstSequence: sequence,
      kind,
      lastSequence: sequence,
      stackId,
    });
  }
  return entries;
}

function projectedCardById(projection: Projection, cardInstanceId: string) {
  const originalCopyId = cardInstanceId.match(/^copy:spell:(.*):\d+$/u)?.[1];
  const candidateIds = new Set([cardInstanceId, originalCopyId].filter(Boolean));
  const stackCard = (projection.stack ?? [])
    .find((item: any) => candidateIds.has(String(item?.card?.id ?? "")))?.card;
  if (stackCard) return stackCard;
  for (const player of projection.players ?? []) {
    const battlefield = player.zones?.battlefield ?? {};
    const cards = [
      ...(player.zones?.hand ?? []),
      ...(player.zones?.graveyard?.cards ?? []),
      ...(player.zones?.exile?.cards ?? []),
      ...(player.zones?.commandZone?.cards ?? []),
      ...(player.zones?.sideboard?.cards ?? []),
      ...(battlefield.lands ?? []),
      ...(battlefield.creatures ?? []),
      ...(battlefield.nonCreaturePermanents ?? []),
    ];
    const found = cards.find((card: any) => candidateIds.has(String(card?.id ?? "")));
    if (found) return found;
  }
  return null;
}

function sceneWithStackPlayback(
  scene: Record<string, any>,
  projection: Projection,
  playback: StackPlaybackEntry | undefined,
) {
  if (!playback) return scene;
  const event = (projection.state?.events ?? []).find(
    (candidate: any) => Number(candidate?.sequence) === playback.firstSequence,
  );
  const card = projectedCardById(projection, playback.cardInstanceId);
  const name = String(card?.name ?? event?.detail?.cardName ?? "Unknown card");
  const activity = playback.kind === "spell"
    ? "Spell cast"
    : playback.kind === "stackCopy"
      ? "Stack object copied"
      : playback.kind === "triggeredAbility" ? "Triggered ability" : "Activated ability";
  const count = playback.count > 1 ? ` ×${playback.count}` : "";
  const replayItem = {
    card: card ?? { id: playback.cardInstanceId, imageUrl: "", name },
    detail: `${activity}${count}`,
    id: playback.stackId,
    imageUrl: card?.imageUrl ?? "",
    name: `${name} — ${activity}${count}`,
    raw: { event, playback },
    targetable: false,
    type: playback.kind,
  };
  const stack = [...(scene.stack ?? [])];
  const existing = stack.findIndex((item: any) => item.id === playback.stackId);
  if (existing >= 0) stack[existing] = { ...stack[existing], ...replayItem };
  else stack.push(replayItem);
  return {
    ...scene,
    replay: true,
    stack,
    status: `${activity}${count}: ${name}`,
  };
}

function abstractResolutionLabel(action: any, projection: Projection) {
  const label = actionLabel(action);
  if (projection.decisionSourceCard?.name !== "Ponder") return label;
  const decisionId = projection.decision?.choice?.decisionId;
  const selected = decisionId ? action?.decisions?.[decisionId]?.[0] : null;
  if (selected === "Decline") return "Keep the current order";
  if (selected === "Perform the optional effects") return "Shuffle the library";
  return label;
}

export default function LocalPixiRenderer({ deckSelections, matchup, mode = "play", view, onAction, onExit }: {
  deckSelections: DeckPresentation[];
  matchup: string;
  mode?: "play" | "replay";
  view: EngineView;
  onAction: (actionId: string, extra?: Record<string, unknown>) => void;
  onExit: () => void;
}) {
  const isReplay = mode === "replay";
  const host = useRef<HTMLDivElement>(null);
  const vue = useRef<VueApp<Element> | null>(null);
  const sceneRef = useRef<ShallowRef<Record<string, unknown>> | null>(null);
  const [targetPlan, setTargetPlan] = useState<TargetPlan | null>(null);
  const [sourceChoices, setSourceChoices] = useState<any[]>([]);
  const [selectedCardIds, setSelectedCardIds] = useState<string[]>([]);
  const [numberValue, setNumberValue] = useState(0);
  const [cardNameQuery, setCardNameQuery] = useState("");
  const [actionQuery, setActionQuery] = useState("");
  const [searchedCardNames, setSearchedCardNames] = useState<string[]>([]);
  const [hoveredCard, setHoveredCard] = useState<any | null>(null);
  const [notesOpen, setNotesOpen] = useState(false);
  const [notePreview, setNotePreview] = useState<any | null>(null);
  const [noteSort, setNoteSort] = useState<GameNoteSort>("observed");
  const [noteSortDirection, setNoteSortDirection] = useState<"asc" | "desc">("asc");
  const [stackPlaybackQueue, setStackPlaybackQueue] = useState<StackPlaybackEntry[]>([]);
  const [visualError, setVisualError] = useState("");
  const playbackCursor = useRef({ sequence: 0, sessionId: "" });

  const cardCatalog = useMemo(
    () => pixi.gameSessionCardCatalogFromDeckSelections(deckSelections),
    [deckSelections],
  );
  const projection = useMemo(
    () => pixi.projectGameSessionView(view, {
      cardCatalog,
      playerRoleById: { "local-human": "human", "local-agent": "ai" },
    }),
    [cardCatalog, view],
  );
  const directPresentations = useMemo(
    () => pixi.gameSessionActionPresentations(projection.directActions ?? []),
    [projection.directActions],
  );
  const automaticTargetPlan = useMemo(() => {
    const resolution = projection.resolutionTargetChoice;
    if (resolution) {
      return targetPlanFromAction({
        engineTargetActions: resolution.actions,
        engineTargetKeys: [resolution.targetKey],
        label: resolution.prompt,
      }, resolution.prompt);
    }
    const visualDirect = directPresentations.filter(
      (action: any) => actionTargetKeys(action).length > 0,
    );
    return visualDirect.length === 1 ? targetPlanFromAction(visualDirect[0]) : null;
  }, [directPresentations, projection.resolutionTargetChoice]);
  const activeTargetPlan = targetPlan ?? automaticTargetPlan;
  const cardChoice = projection.resolutionCardChoice;
  const cardNameChoice = projection.cardNameChoice;
  const dungeonChoice = projection.dungeonChoice;
  const visibleNameCards = useMemo(() => visibleCardNameChoices(projection), [projection]);
  const visibleNameCardIds = useMemo(
    () => new Set<string>(visibleNameCards.map((card: any) => String(card.id))),
    [visibleNameCards],
  );
  const tableCardIds = useMemo(() => tableChoiceCardIds(projection), [projection]);
  const tableChoiceCards = useMemo(
    () => (cardChoice?.cards ?? []).filter((card: any) => tableCardIds.has(String(card.id))),
    [cardChoice, tableCardIds],
  );
  const zoneChoiceCards = useMemo(
    () => (cardChoice?.cards ?? []).filter((card: any) => !tableCardIds.has(String(card.id))),
    [cardChoice, tableCardIds],
  );
  const tableChoiceIds = useMemo(
    () => new Set<string>(tableChoiceCards.map((card: any) => String(card.id))),
    [tableChoiceCards],
  );
  const selectedIds = useMemo(() => new Set(selectedCardIds), [selectedCardIds]);
  const targetableIds = useMemo(() => {
    if (activeTargetPlan) return candidateTargetIds(activeTargetPlan);
    if (cardNameChoice) return visibleNameCardIds;
    const maximumReached = selectedCardIds.length >= Number(cardChoice?.maximum ?? 0);
    return new Set<string>([...tableChoiceIds].filter(
      (id) => !maximumReached || selectedIds.has(id),
    ));
  }, [activeTargetPlan, cardChoice?.maximum, cardNameChoice, selectedCardIds.length, selectedIds, tableChoiceIds, visibleNameCardIds]);
  const interaction = useMemo<Interaction>(() => isReplay ? { targeting: true } : ({
    orderedIds: selectedCardIds,
    prompt: activeTargetPlan
      ? `${activeTargetPlan.prompt} (${activeTargetPlan.index + 1}/${activeTargetPlan.keys.length})`
      : cardChoice && tableChoiceCards.length
        ? cardChoice.prompt
      : cardNameChoice
        ? `${cardNameChoice.prompt} Click a visible hand, graveyard, or exile card.`
      : undefined,
    selectedIds,
    targetableIds,
    targeting: Boolean(activeTargetPlan || cardChoice || cardNameChoice),
  }), [activeTargetPlan, cardChoice, cardNameChoice, isReplay, selectedCardIds, selectedIds, tableChoiceCards.length, targetableIds]);
  const activeStackPlayback = stackPlaybackQueue[0];
  const scene = useMemo(
    () => sceneWithStackPlayback(
      { ...pixiScene(projection, interaction), replay: isReplay },
      projection,
      activeStackPlayback,
    ),
    [activeStackPlayback, interaction, isReplay, projection],
  );
  const initialScene = useRef(scene);
  const initialSelectionKey = JSON.stringify(
    projection.resolutionCardChoice?.initialSelectedCardIds ?? [],
  );
  const minimumNumber = Number(projection.numberChoice?.minimum ?? 0);

  const submit = useCallback((actionId: string, extra?: Record<string, unknown>) => {
    setTargetPlan(null);
    setSourceChoices([]);
    onAction(actionId, extra);
  }, [onAction]);

  const chooseCard = useCallback((cardId: string) => {
    if (!cardChoice || !cardChoice.cards.some((card: any) => String(card.id) === cardId)) return;
    const minimum = Number(cardChoice.minimum ?? 0);
    const maximum = Number(cardChoice.maximum ?? 0);
    if (
      cardChoice.kind !== "cardOrder" &&
      minimum === 1 &&
      maximum === 1 &&
      cardChoice.options?.[0]?.actionId
    ) {
      submit(cardChoice.options[0].actionId, { cardInstanceIds: [cardId] });
      return;
    }
    setSelectedCardIds((current) => {
      if (current.includes(cardId)) return current.filter((id) => id !== cardId);
      if (current.length >= maximum) return current;
      return [...current, cardId];
    });
  }, [cardChoice, submit]);

  const beginAction = useCallback((action: any) => {
    const plan = targetPlanFromAction(action);
    setSourceChoices([]);
    if (!plan) {
      if (action?.id) submit(action.id);
      return;
    }
    const automaticDefender = selectOnlyAttackDefender(action, plan);
    if (automaticDefender?.actionId) {
      submit(automaticDefender.actionId);
      return;
    }
    if (automaticDefender?.plan) {
      setTargetPlan(automaticDefender.plan);
      return;
    }
    setTargetPlan(plan);
  }, [submit]);

  const chooseEntity = useCallback((ids: string[]) => {
    if (!activeTargetPlan) return;
    const result = selectPlanTarget(activeTargetPlan, ids);
    if (!result) return;
    if (result.actionId) submit(result.actionId);
    else if (result.plan) setTargetPlan(result.plan);
  }, [activeTargetPlan, submit]);

  const handleCard = useCallback((payload: any) => {
    const card = payload?.card ?? payload;
    if (activeTargetPlan) {
      chooseEntity([String(card?.id ?? card?.instanceId ?? "")].filter(Boolean));
      return;
    }
    const cardId = String(card?.id ?? card?.instanceId ?? "");
    if (cardNameChoice && visibleNameCardIds.has(cardId) && card?.name) {
      submit(cardNameChoice.actionId, { cardName: String(card.name) });
      return;
    }
    if (cardChoice && tableChoiceIds.has(cardId)) {
      chooseCard(cardId);
      return;
    }
    const choices = castingPaymentPresentations(card?.actionState?.options ?? []);
    if (choices.length === 1) beginAction(choices[0]);
    else if (choices.length > 1) {
      const xChoice = castingXNumberChoice(choices);
      if (xChoice) setNumberValue(xChoice.minimum);
      setSourceChoices(choices);
    }
  }, [activeTargetPlan, beginAction, cardChoice, cardNameChoice, chooseCard, chooseEntity, submit, tableChoiceIds, visibleNameCardIds]);

  const cardRef = useRef(handleCard);
  const hoverRef = useRef<(payload: any) => void>(() => undefined);
  const leaveRef = useRef<() => void>(() => undefined);
  const playerRef = useRef<(player: any) => void>(() => undefined);
  const stackRef = useRef<(item: any) => void>(() => undefined);
  const passRef = useRef<() => void>(() => undefined);
  const exitRef = useRef(onExit);
  useEffect(() => { cardRef.current = handleCard; }, [handleCard]);
  useEffect(() => {
    hoverRef.current = (payload: any) => setHoveredCard(payload?.card ?? payload ?? null);
    leaveRef.current = () => setHoveredCard(null);
  }, []);
  useEffect(() => {
    playerRef.current = (player: any) => {
      const id = player?.key ?? player?.id;
      if (id) chooseEntity([String(id)]);
    };
    stackRef.current = (item: any) => {
      const cardId = [item?.card?.id, item?.card?.instanceId]
        .filter(Boolean)
        .map(String)
        .find((id) => tableChoiceIds.has(id));
      if (cardChoice && cardId) {
        chooseCard(cardId);
        return;
      }
      chooseEntity([
        item?.id,
        item?.card?.id,
        item?.card?.instanceId,
      ].filter(Boolean).map(String));
    };
    passRef.current = () => {
      if (projection.passAction?.id) submit(projection.passAction.id);
    };
  }, [cardChoice, chooseCard, chooseEntity, projection.passAction, submit, tableChoiceIds]);
  useEffect(() => { exitRef.current = onExit; }, [onExit]);
  useEffect(() => {
    setTargetPlan(null);
    setSourceChoices([]);
    setHoveredCard(null);
    setSelectedCardIds(JSON.parse(initialSelectionKey) as string[]);
    setNumberValue(minimumNumber);
    setCardNameQuery("");
    setActionQuery("");
    setSearchedCardNames([]);
  }, [initialSelectionKey, minimumNumber, view.decision?.id]);
  useEffect(() => {
    if (!cardNameChoice || cardNameQuery.trim().length < 2) {
      setSearchedCardNames([]);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void fetch(`/api/v1/catalog/cards/autocomplete?query=${encodeURIComponent(cardNameQuery.trim())}`, {
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok) throw new Error(`Card search failed (${response.status})`);
          return response.json() as Promise<{ data?: string[] }>;
        })
        .then((body) => setSearchedCardNames(Array.isArray(body.data) ? body.data : []))
        .catch((error: unknown) => {
          if (!(error instanceof DOMException && error.name === "AbortError")) {
            setSearchedCardNames([]);
          }
        });
    }, 300);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [cardNameChoice, cardNameQuery]);
  useEffect(() => {
    const events = projection.state?.events ?? [];
    const latestSequence = events.reduce(
      (latest: number, event: any) => Math.max(latest, Number(event?.sequence ?? 0)),
      0,
    );
    const sessionId = String(view.sessionId ?? "");
    if (playbackCursor.current.sessionId !== sessionId) {
      playbackCursor.current = { sequence: latestSequence, sessionId };
      setStackPlaybackQueue([]);
      return;
    }
    if (latestSequence < playbackCursor.current.sequence) {
      playbackCursor.current.sequence = latestSequence;
      setStackPlaybackQueue([]);
      return;
    }
    const additions = stackEventPlaybackEntries(events, playbackCursor.current.sequence);
    playbackCursor.current.sequence = latestSequence;
    if (additions.length) setStackPlaybackQueue((current) => [...current, ...additions]);
  }, [projection.state?.events, view.sessionId]);
  useEffect(() => {
    if (!activeStackPlayback) return;
    const timer = window.setTimeout(() => {
      setStackPlaybackQueue((current) => current.slice(1));
    }, activeStackPlayback.durationMs);
    return () => window.clearTimeout(timer);
  }, [activeStackPlayback]);
  useEffect(() => {
    if (!host.current) return;
    try {
      const mounted = mountPixi(host.current, initialScene.current, {
        card: cardRef,
        exit: exitRef,
        hover: hoverRef,
        leave: leaveRef,
        pass: passRef,
        player: playerRef,
        stack: stackRef,
      }, mode);
      vue.current = mounted.vue;
      sceneRef.current = mounted.currentScene;
    } catch (reason) {
      setVisualError(reason instanceof Error ? reason.message : "Pixi could not start.");
    }
    return () => {
      vue.current?.unmount();
      vue.current = null;
      sceneRef.current = null;
    };
  }, [mode]);
  useEffect(() => {
    if (sceneRef.current) sceneRef.current.value = scene;
  }, [scene]);

  const visualDirectChoices = directPresentations.filter(
    (action: any) => actionTargetKeys(action).length > 0,
  );
  const abstractActions = directPresentations.filter((action: any) =>
    actionTargetKeys(action).length === 0 &&
    action.kind !== "passPriority" &&
    !projection.resolutionCardChoice &&
    !projection.dungeonChoice &&
    !projection.cardNameChoice &&
    !projection.numberChoice,
  );
  const cardSelectionValid = Boolean(
    cardChoice &&
    selectedCardIds.length >= Number(cardChoice.minimum) &&
    selectedCardIds.length <= Number(cardChoice.maximum),
  );
  const orderLegend = cardOrderLegend(cardChoice);
  const zoneChoicePanelWidth = Math.min(
    54,
    Math.max(16, Math.min(zoneChoiceCards.length, 6) * 8.25 + 1.6),
  );
  const searchableAbstractActions = abstractActions.length > 12;
  const filteredAbstractActions = searchableAbstractActions
    ? abstractActions.filter((action: any) => abstractResolutionLabel(action, projection)
      .toLocaleLowerCase()
      .includes(actionQuery.trim().toLocaleLowerCase()))
    : abstractActions;
  const cardNameSuggestions = [...new Set<string>([
    ...(cardNameChoice?.suggestions ?? []),
    ...visibleNameCards.map((card: any) => String(card.name)),
    ...searchedCardNames,
  ])].slice(0, 25);
  const sourceXChoice = castingXNumberChoice(sourceChoices);

  return <>
    <div className={`local-pixi-vue-host${isReplay ? " replay-pixi-vue-host" : ""}`} ref={host} />
    {!isReplay && <div className="pixi-matchup-badge">{matchup}</div>}
    {(projection.gameNotes?.length ?? 0) > 0 && <aside className={`pixi-game-notes${notesOpen ? " open" : ""}`}>
      <button type="button" aria-expanded={notesOpen} onClick={() => setNotesOpen((open) => {
        if (open) setNotePreview(null);
        return !open;
      })}>
        <span aria-hidden="true">◉</span> Notes <b>{projection.gameNotes.length}</b>
      </button>
      {notesOpen && <div className="pixi-game-notes-panel">
        <header><strong>Cards seen</strong><button type="button" aria-label="Close notes" onClick={() => { setNotesOpen(false); setNotePreview(null); }}>×</button></header>
        <div className="pixi-game-notes-toolbar">
          <label><span>Sort cards</span><select value={noteSort} onChange={(event) => setNoteSort(event.target.value as GameNoteSort)}>
            <option value="observed">Observed order</option>
            <option value="name">Name</option>
            <option value="type">Type</option>
            <option value="mana">Mana value</option>
            <option value="quantity">Quantity</option>
          </select></label>
          <button
            type="button"
            disabled={noteSort === "observed"}
            aria-label={`Sort ${noteSortDirection === "asc" ? "descending" : "ascending"}`}
            onClick={() => setNoteSortDirection((direction) => direction === "asc" ? "desc" : "asc")}
          >{noteSortDirection === "asc" ? "↑ Asc" : "↓ Desc"}</button>
        </div>
        {projection.gameNotes.map((note: any, noteIndex: number) => <section key={`${note.sourceName}:${note.turnNumber}:${noteIndex}`}>
          <div><strong>{note.playerName}</strong><small>{note.sourceName} · turn {note.turnNumber}</small></div>
          {note.zones.map((zone: any) => <div className="pixi-game-note-zone" key={zone.zone}>
            <h4>{zone.zone === "library" ? "Library" : zone.zone === "hand" ? "Hand" : "Graveyard"}</h4>
            {zone.cards.length > 0 ? <ul>{sortGameNoteCards(
              zone.cards.map((card: any) => gameNoteCardPresentation(card, cardCatalog)),
              noteSort,
              noteSortDirection,
            ).map((card: any) => <li key={card.name}>
              <b>{card.count}×</b>
              <button
                type="button"
                onMouseEnter={() => setNotePreview(card)}
                onMouseLeave={() => setNotePreview(null)}
                onFocus={() => setNotePreview(card)}
                onBlur={() => setNotePreview(null)}
              >
                <span>{card.name}</span>
                <small>{card.typeLine || "Unknown type"} · MV {card.manaValue ?? "—"}</small>
              </button>
            </li>)}</ul> : <p>Empty</p>}
          </div>)}
          {!note.orderKnown && note.zones.some((zone: any) => zone.zone === "library") && <p className="pixi-game-note-warning">Known contents; order unknown after shuffle.</p>}
        </section>)}
      </div>}
      {notesOpen && notePreview?.imageUrl && <div className="pixi-note-card-preview" aria-hidden="true">
        <img src={notePreview.imageUrl} alt="" />
      </div>}
    </aside>}
    {hoveredCard?.imageUrl && <div className="pixi-hover-preview" aria-hidden="true">
      <img src={hoveredCard.imageUrl} alt="" />
    </div>}
    {visualError && <div className="pixi-visual-error" role="alert">Pixi: {visualError}</div>}
    {!isReplay && <>{dungeonChoice && <div className="pixi-dungeon-choice" role="dialog" aria-label={dungeonChoice.prompt}>
      <header>
        <strong>{dungeonChoice.prompt}</strong>
        {dungeonChoice.currentRoom && <span>Current room: {dungeonChoice.currentRoom}</span>}
      </header>
      {dungeonChoice.kind === "dungeon" ? <div className="pixi-dungeon-cards">
        {dungeonChoice.options.map((option: any) => {
          const card = dungeonChoice.cards.find((candidate: any) => String(candidate.id) === `dungeon:${option.dungeonId}`);
          return <button key={option.actionId} type="button" onClick={() => submit(option.actionId)}>
            {card?.imageUrl ? <img src={card.imageUrl} alt={card.name} /> : <span className="dungeon-card-fallback">Dungeon</span>}
            <strong>{card?.name ?? option.label}</strong>
          </button>;
        })}
      </div> : <div className="pixi-dungeon-room-choice">
        {dungeonChoice.cards[0]?.imageUrl && <img src={dungeonChoice.cards[0].imageUrl} alt={dungeonChoice.cards[0].name} />}
        <div><small>{dungeonChoice.cards[0]?.name}</small>{dungeonChoice.options.map((option: any) => <button key={option.actionId} type="button" onClick={() => submit(option.actionId)}>{option.label}</button>)}</div>
      </div>}
    </div>}
    {cardNameChoice && <form
      className="pixi-card-name-choice"
      role="dialog"
      aria-label={cardNameChoice.prompt}
      onSubmit={(event) => {
        event.preventDefault();
        if (cardNameQuery.trim()) submit(cardNameChoice.actionId, { cardName: cardNameQuery.trim() });
      }}
    >
      <strong>{cardNameChoice.prompt}</strong>
      <span>Click a highlighted card in a visible hand, graveyard, or exile, or search any card name.</span>
      <div>
        <input
          autoFocus
          aria-label="Card name"
          list="pixi-card-name-suggestions"
          placeholder="e.g. Force of Will"
          value={cardNameQuery}
          onChange={(event) => setCardNameQuery(event.target.value)}
        />
        <datalist id="pixi-card-name-suggestions">
          {cardNameSuggestions.map((name) => <option key={name} value={name} />)}
        </datalist>
        <button type="submit" disabled={!cardNameQuery.trim()}>Choose</button>
      </div>
      {cardNameSuggestions.length > 0 && <div className="pixi-card-name-suggestions">
        {cardNameSuggestions.slice(0, 10).map((name) => <button
          key={name}
          type="button"
          onClick={() => submit(cardNameChoice.actionId, { cardName: name })}
        >{name}</button>)}
      </div>}
    </form>}
    {activeTargetPlan && <div className="pixi-target-prompt" role="status">
      <strong>{activeTargetPlan.prompt}</strong>
      {activeTargetPlan.keys.length > 1 && <span>Step {activeTargetPlan.index + 1}/{activeTargetPlan.keys.length}</span>}
      {targetPlan && <button type="button" onClick={() => setTargetPlan(null)}>Cancel</button>}
    </div>}
    {!activeTargetPlan && cardChoice && tableChoiceCards.length > 0 && <div className="pixi-target-prompt" role="status">
      <strong>{cardChoice.prompt}</strong>
      <span>Choose the highlighted cards directly on the table · selected {selectedCardIds.length}/{cardChoice.minimum}–{cardChoice.maximum}</span>
      {orderLegend && <span className="pixi-order-legend">{orderLegend}</span>}
      <button
        type="button"
        disabled={!cardSelectionValid}
        onClick={() => submit(cardChoice.options[0].actionId, { cardInstanceIds: selectedCardIds })}
      >Confirm</button>
    </div>}
    {!activeTargetPlan && sourceXChoice && <div className="pixi-action-palette">
      <strong>Choose X</strong>
      <span>Select a whole number from {sourceXChoice.minimum} to {sourceXChoice.maximum}.</span>
      <div><input
        aria-label="X value"
        type="number"
        min={sourceXChoice.minimum}
        max={sourceXChoice.maximum}
        step="1"
        value={numberValue}
        onChange={(event) => setNumberValue(Number(event.target.value))}
      /><button type="button" disabled={!sourceXChoice.options.has(numberValue)} onClick={() => beginAction(sourceXChoice.options.get(numberValue))}>Cast with X = {numberValue}</button></div>
      <button type="button" className="subtle" onClick={() => setSourceChoices([])}>Cancel</button>
    </div>}
    {!activeTargetPlan && sourceChoices.length > 0 && !sourceXChoice && <div className="pixi-action-palette">
      <strong>Choose how to use this card</strong>
      <div>{sourceChoices.map((action) => <button key={action.id} type="button" onClick={() => beginAction(action)}>{actionLabel(action)}</button>)}</div>
      <button type="button" className="subtle" onClick={() => setSourceChoices([])}>Cancel</button>
    </div>}
    {!activeTargetPlan && sourceChoices.length === 0 && visualDirectChoices.length > 1 && <div className="pixi-action-palette">
      <strong>Choose an action, then its highlighted targets</strong>
      <div>{visualDirectChoices.map((action: any) => <button key={action.id} type="button" onClick={() => beginAction(action)}>{actionLabel(action)}</button>)}</div>
    </div>}
    {!activeTargetPlan && sourceChoices.length === 0 && abstractActions.length > 0 && <div className={`pixi-action-palette${searchableAbstractActions ? " searchable" : ""}`}>
      <strong>{projection.decision?.kind === "mulligan" ? "Choose your opening hand" : projection.decision?.choice?.prompt ?? "Choose an action"}</strong>
      {searchableAbstractActions && <label className="pixi-action-search">
        <span>Search choices</span>
        <input
          autoFocus
          type="search"
          placeholder="Type a name…"
          value={actionQuery}
          onChange={(event) => setActionQuery(event.target.value)}
        />
      </label>}
      <div>{filteredAbstractActions.map((action: any) => <button key={action.id} type="button" onClick={() => submit(action.id)}>{abstractResolutionLabel(action, projection)}</button>)}</div>
      {searchableAbstractActions && filteredAbstractActions.length === 0 && <span>No matching choice.</span>}
    </div>}
    {cardChoice && zoneChoiceCards.length > 0 && <div
      className={`pixi-card-choice${tableChoiceCards.length ? " mixed-zone-choice" : ""}`}
      role="dialog"
      aria-label={cardChoice.prompt}
      style={tableChoiceCards.length ? undefined : {
        width: `min(calc(100vw - 2rem), ${zoneChoicePanelWidth}rem)`,
      }}
    >
      <header><div>
        <strong>{tableChoiceCards.length ? "Cards outside the table" : cardChoice.prompt}</strong>
        <span>Select {cardChoice.minimum}–{cardChoice.maximum}</span>
        {orderLegend && <small className="pixi-order-legend">{orderLegend}</small>}
      </div></header>
      <div className="pixi-card-choice-grid">{zoneChoiceCards.map((card: any) => {
        const order = selectedCardIds.indexOf(card.id);
        return <button
          key={card.id}
          type="button"
          className={order >= 0 ? "selected" : ""}
          onClick={() => chooseCard(String(card.id))}
          onMouseEnter={() => setHoveredCard(card)}
          onMouseLeave={() => setHoveredCard(null)}
          onFocus={() => setHoveredCard(card)}
          onBlur={() => setHoveredCard(null)}
        >
          <img src={card.imageUrl} alt="" />
          <span>{card.name}</span>
          {order >= 0 && <b>{order + 1}</b>}
        </button>;
      })}</div>
      {tableChoiceCards.length === 0 && <button type="button" disabled={!cardSelectionValid} onClick={() => submit(cardChoice.options[0].actionId, { cardInstanceIds: selectedCardIds })}>Confirm selection</button>}
    </div>}
    {projection.numberChoice && <div className="pixi-action-palette">
      <strong>{projection.numberChoice.prompt}</strong>
      <input type="number" min={projection.numberChoice.minimum} max={projection.numberChoice.maximum} value={numberValue} onChange={(event) => setNumberValue(Number(event.target.value))} />
      <button type="button" onClick={() => submit(projection.actions[0].id, { numberValue })}>Confirm</button>
    </div>}</>}
  </>;
}
