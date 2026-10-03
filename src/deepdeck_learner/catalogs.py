from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from .jobs import is_loopback_url


class CatalogError(RuntimeError):
    pass


class CatalogAuthenticationError(CatalogError):
    pass


class CatalogNotFoundError(CatalogError):
    pass


def account_api_key() -> str:
    api_key = os.getenv("DEEPDECK_API_KEY", "").strip()
    if not api_key:
        raise CatalogAuthenticationError(
            "Add DEEPDECK_API_KEY to the project .env and restart DeepDeckLearner."
        )
    return api_key


def _platform_url() -> str:
    return os.getenv("DEEPDECK_PLATFORM_URL", "https://staging.deepdeckleague.com/api/v1").rstrip(
        "/"
    )


def _data(response: httpx.Response) -> Any:
    if response.status_code in {401, 403}:
        raise CatalogAuthenticationError(
            "The saved Deep Deck League API key was rejected. "
            "Replace it with a new active agent key."
        )
    try:
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise CatalogError(f"The deck catalog request failed: {error}") from error
    if not isinstance(body, dict) or "data" not in body:
        raise CatalogError("The deck catalog returned an unsupported response.")
    return body["data"]


def platform_decks(search: str, game_format: str, page: int = 1) -> dict[str, Any]:
    if game_format not in {"legacy", "commander"}:
        raise CatalogError("Format must be legacy or commander.")
    with httpx.Client(timeout=10.0) as client:
        response = client.get(
            f"{_platform_url()}/agents/catalog/decks",
            headers={"Authorization": f"Bearer {account_api_key()}"},
            params={
                "page": page,
                "page_size": 12,
                "search": search.strip() or None,
                "format": game_format,
            },
        )
    data = _data(response)
    if not isinstance(data, dict):
        raise CatalogError("The deck catalog did not return a page.")
    return data


def cached_decks(root: Path, search: str, game_format: str, page: int = 1) -> dict[str, Any]:
    """List downloaded deck snapshots when the hosted catalog is unavailable."""
    if game_format not in {"legacy", "commander"}:
        raise CatalogError("Format must be legacy or commander.")
    normalized = search.strip().casefold()
    items: list[dict[str, Any]] = []
    for path in (root / ".deepdeck" / "decks").glob("*.json"):
        try:
            deck = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(deck, dict) or str(deck.get("format", "")).casefold() != game_format:
            continue
        name = str(deck.get("name", "Local deck"))
        if normalized and normalized not in name.casefold():
            continue
        cards = deck.get("cards", [])
        items.append(
            {
                "id": str(deck.get("id") or path.stem),
                "deckId": deck.get("deckId"),
                "name": name,
                "creator": deck.get("creator"),
                "version": int(deck.get("version", 1) or 1),
                "format": game_format,
                "colors": deck.get("colors", []),
                "playableCardCount": _playable_card_count(cards) if isinstance(cards, list) else 0,
                "cached": True,
            }
        )
    items.sort(key=lambda item: (str(item["name"]).casefold(), -int(item["version"])))
    page_size = 12
    current_page = max(1, int(page))
    start = (current_page - 1) * page_size
    total_pages = max(1, (len(items) + page_size - 1) // page_size)
    return {
        "items": items[start : start + page_size],
        "pagination": {
            "page": current_page,
            "totalPages": total_pages,
            "hasNextPage": current_page < total_pages,
        },
        "source": "local-cache",
    }


def cached_deck_download(root: Path, version_id: str) -> dict[str, Any] | None:
    target = root / ".deepdeck" / "decks" / f"{version_id.strip()}.json"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cards = data.get("cards", []) if isinstance(data, dict) else []
    if not isinstance(cards, list):
        return None
    return {
        "versionId": version_id,
        "name": str(data.get("name", "Training deck")),
        "format": data.get("format"),
        "cardCount": _playable_card_count(cards),
        "rawCardCount": data.get("cardCount", len(cards)),
        "path": str(target),
        "cached": True,
    }


def _playable_card_count(cards: list[Any]) -> int:
    total = 0
    for card in cards:
        if not isinstance(card, dict):
            continue
        type_line = str(card.get("typeLine", "")).strip().casefold()
        section = str(card.get("section", "main")).strip().casefold()
        if (
            bool(card.get("isToken"))
            or bool(card.get("isGamePiece"))
            or type_line.startswith(("token ", "emblem", "dungeon"))
            or section in {"considering", "considered", "token", "tokens", "emblem", "dungeon"}
        ):
            continue
        try:
            total += max(0, int(card.get("quantity", 1)))
        except (TypeError, ValueError):
            continue
    return total


def download_platform_deck(root: Path, version_id: str) -> dict[str, Any]:
    deck_id = version_id.strip()
    if not deck_id:
        raise CatalogError("A deck version is required.")
    with httpx.Client(timeout=20.0) as client:
        response = client.get(
            f"{_platform_url()}/agents/catalog/decks/{deck_id}",
            headers={"Authorization": f"Bearer {account_api_key()}"},
        )
    data = _data(response)
    if not isinstance(data, dict) or not isinstance(data.get("cards"), list):
        raise CatalogError("The deck snapshot did not contain a card list.")
    target = root / ".deepdeck" / "decks" / f"{deck_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(".pending")
    pending.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(target)
    return {
        "versionId": deck_id,
        "name": str(data.get("name", "Training deck")),
        "format": data.get("format"),
        "cardCount": _playable_card_count(data["cards"]),
        "rawCardCount": data.get("cardCount", 0),
        "path": str(target),
    }


def local_deck_presentation(root: Path, version_id: str) -> dict[str, Any]:
    """Return the downloaded printing metadata used only to render a local game."""
    deck_id = version_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", deck_id):
        raise CatalogNotFoundError("The local deck snapshot was not found.")
    target = root / ".deepdeck" / "decks" / f"{deck_id}.json"
    try:
        snapshot = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CatalogNotFoundError("Download this deck again before playtesting.") from error
    except (OSError, ValueError) as error:
        raise CatalogError("The local deck snapshot could not be read.") from error
    raw_cards = snapshot.get("cards", []) if isinstance(snapshot, dict) else []
    if not isinstance(raw_cards, list):
        raise CatalogError("The local deck snapshot has no card list.")

    cards: list[dict[str, Any]] = []
    for raw in raw_cards:
        if not isinstance(raw, dict):
            continue
        card_id = str(
            raw.get("cardId") or raw.get("printingId") or raw.get("scryfallId") or ""
        ).strip()
        name = str(raw.get("name", "")).strip()
        if not card_id or not name:
            continue
        type_line = str(raw.get("typeLine", ""))
        image_front = str(raw.get("imageUri") or raw.get("imageUrl") or "")
        image_back = str(raw.get("imageBackUri") or raw.get("urlBack") or "")
        is_token = bool(raw.get("isToken")) or type_line.casefold().startswith("token ")
        card: dict[str, Any] = {
            "id": card_id,
            "imageUrl": image_front,
            "isGamePiece": bool(raw.get("isGamePiece")) or is_token,
            "isToken": is_token,
            "manaCost": raw.get("manaCost") or "",
            "name": name,
            "oracleId": raw.get("oracleId"),
            "oracleText": raw.get("oracleText") or "",
            "power": raw.get("power"),
            "printingId": raw.get("printingId"),
            "quantity": raw.get("quantity", 1),
            "scryfallId": raw.get("scryfallId"),
            "toughness": raw.get("toughness"),
            "typeLine": type_line,
            "urlBack": image_back,
            "urlFront": image_front,
        }
        if raw.get("flavorName"):
            card["flavorName"] = raw["flavorName"]
        if isinstance(raw.get("relatedTokens"), list):
            card["relatedTokens"] = raw["relatedTokens"]

        face_names = [part.strip() for part in name.split("//") if part.strip()]
        if len(face_names) > 1:
            card["faces"] = [
                {
                    "id": f"{card_id}:face:{index}",
                    "imageUrl": image_back if index == 1 and image_back else image_front,
                    "name": face_name,
                    "urlFront": image_back if index == 1 and image_back else image_front,
                }
                for index, face_name in enumerate(face_names)
            ]
        cards.append(card)
    return {
        "versionId": deck_id,
        "name": str(snapshot.get("name", "Local deck")),
        "cards": cards,
    }


def scryfall_image(image_path: str) -> tuple[bytes, str]:
    """Proxy allow-listed Scryfall card art so Pixi can safely create WebGL textures."""
    if (
        not image_path
        or ".." in image_path
        or not re.fullmatch(r"[A-Za-z0-9/_-]+\.[A-Za-z0-9]+", image_path)
    ):
        raise CatalogError("Invalid Scryfall image path.")
    try:
        response = httpx.get(
            f"https://cards.scryfall.io/{image_path}",
            headers={"Accept": "image/*", "User-Agent": "DeepDeckLearner/0.2"},
            follow_redirects=True,
            timeout=20.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise CatalogError(f"The Scryfall image request failed: {error}") from error
    content_type = response.headers.get("content-type", "application/octet-stream")
    if not content_type.casefold().startswith("image/") or len(response.content) > 20 * 1024 * 1024:
        raise CatalogError("Scryfall returned an unsupported image response.")
    return response.content, content_type


@lru_cache(maxsize=256)
def scryfall_card_names(query: str) -> list[str]:
    """Return Scryfall autocomplete names without exposing any game-zone contents."""
    normalized = query.strip()
    if len(normalized) < 2:
        return []
    if len(normalized) > 100 or any(character in "\r\n\0" for character in normalized):
        raise CatalogError("Invalid card-name search.")
    try:
        response = httpx.get(
            "https://api.scryfall.com/cards/autocomplete",
            params={"q": normalized, "include_extras": "false"},
            headers={
                "Accept": "application/json",
                "User-Agent": "DeepDeckLearner/0.2",
            },
            timeout=10.0,
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise CatalogError(f"The Scryfall card-name search failed: {error}") from error
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise CatalogError("Scryfall returned an unsupported autocomplete response.")
    return [name for name in data[:25] if isinstance(name, str) and name.strip()]


def active_competitions() -> dict[str, Any]:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(
            f"{_platform_url()}/agents/catalog/competitions",
            headers={"Authorization": f"Bearer {account_api_key()}"},
            params={"status": "active", "page": 1, "page_size": 100},
        )
    data = _data(response)
    if not isinstance(data, dict):
        raise CatalogError("The competition catalog did not return a page.")
    return data


def local_legal_decks(engine_url: str, game_format: str) -> list[dict[str, str]]:
    if not is_loopback_url(engine_url):
        raise CatalogError("Local deck discovery requires a loopback Engine URL.")
    if game_format not in {"legacy", "commander"}:
        raise CatalogError("Format must be legacy or commander.")
    try:
        response = httpx.post(
            f"{engine_url.rstrip('/')}/game/decks/legal",
            json={"gameMode": game_format, "openingHandSize": 7},
            timeout=20.0,
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise CatalogError(f"The local Engine deck catalog failed: {error}") from error
    if body.get("schemaVersion") != "mtg-legal-deck-catalog/v1":
        raise CatalogError("The local Engine returned an unsupported deck catalog.")
    decks = body.get("decks")
    if not isinstance(decks, list):
        raise CatalogError("The local Engine deck catalog has no deck list.")
    return [deck for deck in decks if isinstance(deck, dict)]
