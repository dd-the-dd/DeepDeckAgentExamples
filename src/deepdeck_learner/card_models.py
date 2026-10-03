from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from .dependencies import current_revision, engine_binary


class CardModelError(RuntimeError):
    pass


def card_identifier(card: dict[str, Any]) -> str:
    return str(
        card.get("cardId")
        or card.get("id")
        or card.get("printingId")
        or card.get("scryfallId")
        or ""
    )


def engine_signature(root: Path) -> str | None:
    revision = current_revision(root, "engine")
    binary = engine_binary(root)
    if not revision or not binary.is_file():
        return None
    stat = binary.stat()
    return f"{revision}:{stat.st_mtime_ns}:{stat.st_size}"


def oracle_request(card: dict[str, Any]) -> dict[str, Any]:
    faces = card.get("faces")
    return {
        "cardName": str(card.get("name", "")),
        "typeLine": str(card.get("typeLine", "")),
        "manaCost": card.get("manaCost"),
        "oracleText": card.get("oracleText"),
        "layout": card.get("layout"),
        "faces": faces if isinstance(faces, list) else [],
    }


def requires_power_toughness(card: dict[str, Any]) -> bool:
    type_line = str(card.get("typeLine", "")).casefold()
    if (
        bool(card.get("isToken"))
        or bool(card.get("isGamePiece"))
        or type_line.startswith(("token ", "emblem", "dungeon"))
    ):
        return False
    return "creature" in type_line or "vehicle" in type_line


def requires_face_characteristics(card: dict[str, Any]) -> bool:
    faces = card.get("faces")
    faces_are_complete = (
        isinstance(faces, list)
        and len(faces) >= 2
        and bool(card.get("layout"))
        and all(
            not isinstance(face, dict)
            or "planeswalker" not in str(face.get("typeLine", "")).casefold()
            or face.get("loyalty") is not None
            for face in faces
        )
    )
    return (
        bool(card.get("imageBackUri"))
        or "//" in str(card.get("name", ""))
        or "//" in str(card.get("typeLine", ""))
    ) and not faces_are_complete


def enrich_card_characteristics(root: Path, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill absent printed P/T from Scryfall without trusting stale deck snapshots."""
    cache_path = root / ".deepdeck" / "scryfall-card-characteristics.json"
    try:
        raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw_cache = {}
    cached_cards = (
        dict(raw_cache.get("cards", {}))
        if isinstance(raw_cache, dict)
        and raw_cache.get("schemaVersion") == "scryfall-card-characteristics/v1"
        and isinstance(raw_cache.get("cards"), dict)
        else {}
    )
    missing: dict[str, dict[str, Any]] = {}
    for card in cards:
        scryfall_id = str(card.get("scryfallId") or "").strip()
        cached = cached_cards.get(scryfall_id, {})
        cached_has_power_toughness = isinstance(cached, dict) and (
            cached.get("power") is not None and cached.get("toughness") is not None
        )
        cached_has_faces = isinstance(cached, dict) and (
            bool(cached.get("layout"))
            and isinstance(cached.get("faces"), list)
            and len(cached["faces"]) >= 2
            and all(
                not isinstance(face, dict)
                or "planeswalker" not in str(face.get("typeLine", "")).casefold()
                or face.get("loyalty") is not None
                for face in cached["faces"]
            )
        )
        if (
            scryfall_id
            and (
                (
                    requires_power_toughness(card)
                    and (card.get("power") is None or card.get("toughness") is None)
                    and not cached_has_power_toughness
                )
                or (requires_face_characteristics(card) and not cached_has_faces)
            )
        ):
            missing[scryfall_id] = card

    for start in range(0, len(missing), 75):
        identifiers = list(missing)[start : start + 75]
        try:
            response = httpx.post(
                "https://api.scryfall.com/cards/collection",
                json={"identifiers": [{"id": card_id} for card_id in identifiers]},
                headers={
                    "Accept": "application/json;q=0.9,*/*;q=0.8",
                    "User-Agent": "DeepDeckLearner/0.2",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise CardModelError(
                f"Scryfall could not provide missing card characteristics: {error}"
            ) from error
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise CardModelError("Scryfall did not return the requested card characteristics.")
        for result in data:
            if not isinstance(result, dict):
                continue
            scryfall_id = str(result.get("id") or "")
            power = result.get("power")
            toughness = result.get("toughness")
            faces = result.get("card_faces")
            if (power is None or toughness is None) and isinstance(faces, list):
                creature_face = next(
                    (
                        face
                        for face in faces
                        if isinstance(face, dict)
                        and "creature" in str(face.get("type_line", "")).casefold()
                    ),
                    None,
                )
                if isinstance(creature_face, dict):
                    power = creature_face.get("power")
                    toughness = creature_face.get("toughness")
            if scryfall_id:
                normalized_faces = [
                    {
                        "id": f"{scryfall_id}:{index}",
                        "name": str(face.get("name", "")),
                        "typeLine": str(face.get("type_line", "")),
                        "manaCost": face.get("mana_cost"),
                        "oracleText": face.get("oracle_text"),
                        "loyalty": face.get("loyalty"),
                        "power": face.get("power"),
                        "toughness": face.get("toughness"),
                    }
                    for index, face in enumerate(faces if isinstance(faces, list) else [])
                    if isinstance(face, dict)
                ]
                facts: dict[str, Any] = {
                    "power": power,
                    "toughness": toughness,
                }
                if result.get("layout"):
                    facts["layout"] = result["layout"]
                if len(normalized_faces) >= 2:
                    facts["faces"] = normalized_faces
                cached_cards[scryfall_id] = facts

    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pending = cache_path.with_suffix(".pending")
        pending.write_text(
            json.dumps(
                {
                    "schemaVersion": "scryfall-card-characteristics/v1",
                    "cards": cached_cards,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        pending.replace(cache_path)

    enriched: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for card in cards:
        current = dict(card)
        facts = cached_cards.get(str(current.get("scryfallId") or ""), {})
        if requires_face_characteristics(current) and isinstance(facts, dict):
            if facts.get("layout"):
                current["layout"] = facts["layout"]
            if isinstance(facts.get("faces"), list) and len(facts["faces"]) >= 2:
                current["faces"] = facts["faces"]
        if requires_power_toughness(current):
            if isinstance(facts, dict):
                if current.get("power") is None and facts.get("power") is not None:
                    current["power"] = facts["power"]
                if current.get("toughness") is None and facts.get("toughness") is not None:
                    current["toughness"] = facts["toughness"]
            if current.get("power") is None or current.get("toughness") is None:
                unresolved.append(str(current.get("name") or card_identifier(current)))
        enriched.append(current)
    if unresolved:
        names = ", ".join(sorted(set(unresolved))[:5])
        raise CardModelError(f"Missing printed power/toughness for: {names}.")
    return enriched


def compile_oracle_rules(
    root: Path, engine_url: str, cards: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    """Compile card text with the running Oracle, caching only for this Engine build."""
    signature = engine_signature(root)
    cache_path = root / ".deepdeck" / "oracle-card-rules.json"
    try:
        raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw_cache = {}
    cache_matches = bool(
        signature
        and isinstance(raw_cache, dict)
        and raw_cache.get("schemaVersion") == "oracle-card-rules/v2"
        and raw_cache.get("engineBuild") == signature
        and isinstance(raw_cache.get("cards"), dict)
    )
    cached_cards: dict[str, Any] = dict(raw_cache.get("cards", {})) if cache_matches else {}
    compiled: dict[str, list[Any]] = {}
    missing: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    for card in cards:
        card_id = card_identifier(card)
        if not card_id:
            continue
        request = oracle_request(card)
        has_oracle_text = bool(request["oracleText"]) or any(
            isinstance(face, dict) and bool(face.get("oracleText")) for face in request["faces"]
        )
        if not has_oracle_text:
            existing = card.get("rules")
            if isinstance(existing, list):
                compiled[card_id] = existing
            continue
        cached = cached_cards.get(card_id)
        if (
            isinstance(cached, dict)
            and cached.get("request") == request
            and isinstance(cached.get("rules"), list)
        ):
            compiled[card_id] = list(cached["rules"])
        else:
            missing[card_id] = (card, request)

    def parse_rules(
        item: tuple[str, tuple[dict[str, Any], dict[str, Any]]],
    ) -> tuple[str, dict[str, Any], list[Any]]:
        card_id, (card, request) = item
        try:
            response = httpx.post(
                f"{engine_url.rstrip('/')}/oracle/rules",
                json=request,
                timeout=30.0,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise CardModelError(
                f"Engine could not compile {card.get('name', card_id)}: {error}"
            ) from error
        rules = body.get("rules") if isinstance(body, dict) else None
        if not isinstance(rules, list):
            raise CardModelError(f"Engine did not return rules for {card.get('name', card_id)}.")
        return card_id, request, rules

    if missing:
        worker_count = min(8, len(missing))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for card_id, request, rules in executor.map(parse_rules, missing.items()):
                compiled[card_id] = rules
                cached_cards[card_id] = {"request": request, "rules": rules}

    if signature and missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pending = cache_path.with_suffix(".pending")
        pending.write_text(
            json.dumps(
                {
                    "schemaVersion": "oracle-card-rules/v2",
                    "engineBuild": signature,
                    "cards": cached_cards,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        pending.replace(cache_path)

    cards_by_id = {
        card_identifier(card): card
        for card in cards
        if card_identifier(card)
    }
    for card_id, rules in compiled.items():
        faces = cards_by_id.get(card_id, {}).get("faces")
        if not isinstance(faces, list):
            continue
        for rule in rules:
            compiled_faces = rule.get("transformFaces") if isinstance(rule, dict) else None
            if not isinstance(compiled_faces, list):
                continue
            for compiled_face, source_face in zip(compiled_faces, faces, strict=False):
                if (
                    isinstance(compiled_face, dict)
                    and isinstance(source_face, dict)
                    and source_face.get("loyalty") is not None
                ):
                    compiled_face["loyalty"] = str(source_face["loyalty"])
    return compiled


def refresh_playtest_deck(
    root: Path,
    engine_url: str,
    version_id: str,
    cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    snapshot_path = root / ".deepdeck" / "decks" / f"{version_id}.json"
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        snapshot = {}
    snapshot_cards = snapshot.get("cards", []) if isinstance(snapshot, dict) else []
    sources = {
        card_identifier(card): card
        for card in snapshot_cards
        if isinstance(card, dict) and card_identifier(card)
    }
    for card in cards:
        card_id = card_identifier(card)
        if card_id and card_id not in sources:
            sources[card_id] = card
    sources = {
        card_identifier(card): card
        for card in enrich_card_characteristics(root, list(sources.values()))
    }
    compiled = compile_oracle_rules(root, engine_url, list(sources.values()))
    refreshed: list[dict[str, Any]] = []
    for card in cards:
        current = dict(card)
        card_id = card_identifier(card)
        source = sources.get(card_id, {})
        if current.get("power") is None and source.get("power") is not None:
            current["power"] = source["power"]
        if current.get("toughness") is None and source.get("toughness") is not None:
            current["toughness"] = source["toughness"]
        if source.get("layout"):
            current["layout"] = source["layout"]
        if isinstance(source.get("faces"), list) and len(source["faces"]) >= 2:
            current["faces"] = source["faces"]
        rules = compiled.get(card_id)
        if isinstance(rules, list):
            current["rules"] = rules
        refreshed.append(current)
    return refreshed
