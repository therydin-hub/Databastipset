# -*- coding: utf-8 -*-
"""
Svenska Spels externa API – isolerad hjälpkod för Databastipset.

Prioritet per match:
1. API-fältet randomResultProbability (startodds omräknat till 1-X-2-värden).
2. Egen omräkning av startOdds.
3. Aktuella odds används endast efter ett uttryckligt godkännande i testappen:
   favouriteOdds, annars egen omräkning av odds.

Fältet distribution (Svenska folkets streck) används aldrig som oddsvärde.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE_URL = "https://api.www.svenskaspel.se/external/1/draw"

PRODUCT_MAP: dict[str, str] = {
    "Stryktips": "stryktipset",
    "Europatips": "europatipset",
    "Topptips ST": "topptipsetstryk",
    "Topptips EU": "topptipseteuropa",
    "Topptips Övrigt": "topptipset",
    "Powerplay": "powerplay",
}

EXPECTED_MATCHES: dict[str, int] = {
    "Stryktips": 13,
    "Europatips": 13,
    "Topptips ST": 8,
    "Topptips EU": 8,
    "Topptips Övrigt": 8,
    "Powerplay": 8,
}


class SvenskaSpelApiError(RuntimeError):
    """Grundfel för Svenska Spels API."""


class SvenskaSpelAccessKeyError(SvenskaSpelApiError):
    """API-nyckel saknas eller nekas."""


class SvenskaSpelResponseError(SvenskaSpelApiError):
    """API-svaret saknar förväntad struktur."""


@dataclass(frozen=True)
class MatchOddsRow:
    event_number: int
    description: str
    start_odds: Optional[tuple[float, float, float]]
    start_values: Optional[tuple[int, int, int]]
    start_values_origin: str
    current_odds: Optional[tuple[float, float, float]]
    current_values: Optional[tuple[int, int, int]]
    current_values_origin: str
    comparison: str
    event_start: str
    event_status: str

    @property
    def has_start_values(self) -> bool:
        return self.start_values is not None

    @property
    def has_current_values(self) -> bool:
        return self.current_values is not None


def _read_text(response: Any) -> str:
    raw = response.read()
    charset = "utf-8"
    try:
        charset = response.headers.get_content_charset() or "utf-8"
    except Exception:
        pass
    return raw.decode(charset, errors="replace")


def fetch_current_draw(
    product_code: str,
    access_key: str,
    *,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    """
    Hämtar aktuell öppen omgång för en produkt.

    API-nyckeln skickas endast som query-parametern accesskey och skrivs inte ut.
    """
    key = str(access_key or "").strip()
    if not key:
        raise SvenskaSpelAccessKeyError(
            "Ingen API-nyckel angavs. Lägg in SVENSKA_SPEL_ACCESSKEY "
            "eller skriv nyckeln i testappens lösenordsfält."
        )

    product = str(product_code or "").strip().lower()
    if product not in set(PRODUCT_MAP.values()):
        raise ValueError(f"Okänd produktkod: {product_code!r}")

    query = urlencode({"accesskey": key})
    url = f"{API_BASE_URL}/{product}/draws?{query}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Databastipset-SvenskaSpel-API-test/1.0",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", 200))
            body = _read_text(response)
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass

        if exc.code in (401, 403):
            raise SvenskaSpelAccessKeyError(
                f"Svenska Spel nekade anropet (HTTP {exc.code}). "
                "Kontrollera att API-nyckeln är korrekt och aktiv."
            ) from exc

        compact = " ".join(body.split())[:300]
        suffix = f" Svar: {compact}" if compact else ""
        raise SvenskaSpelApiError(
            f"Svenska Spels API svarade med HTTP {exc.code}.{suffix}"
        ) from exc
    except URLError as exc:
        raise SvenskaSpelApiError(
            f"Kunde inte ansluta till Svenska Spels API: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise SvenskaSpelApiError(
            f"Anropet till Svenska Spels API tog längre än {timeout_seconds} sekunder."
        ) from exc

    if status < 200 or status >= 300:
        raise SvenskaSpelApiError(f"Oväntad HTTP-status från API:t: {status}")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SvenskaSpelResponseError(
            "API:t svarade inte med giltig JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise SvenskaSpelResponseError(
            f"Förväntade ett JSON-objekt men fick {type(payload).__name__}."
        )

    return payload


def _parse_decimal(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if not text or text.lower() in {"null", "none", "-", "–", "—"}:
        return None

    # Svenska decimaler brukar komma med kommatecken.
    if "," in text and "." in text:
        # Tolka sista separatorn som decimaltecken.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    else:
        text = text.replace(",", ".")

    # Ta bort ett eventuellt procenttecken.
    text = text.rstrip("%")

    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _get_triplet(
    source: Any,
    *,
    aliases: Sequence[Sequence[str]],
) -> Optional[tuple[float, float, float]]:
    if not isinstance(source, Mapping):
        return None

    values: list[float] = []
    for alias_group in aliases:
        found: Optional[float] = None
        for key in alias_group:
            if key in source:
                found = _parse_decimal(source.get(key))
                if found is not None:
                    break
        if found is None:
            return None
        values.append(found)

    return values[0], values[1], values[2]


ODDS_ALIASES: tuple[tuple[str, ...], ...] = (
    ("home", "one", "1"),
    ("draw", "x", "X"),
    ("away", "two", "2"),
)


def parse_odds_triplet(source: Any) -> Optional[tuple[float, float, float]]:
    values = _get_triplet(source, aliases=ODDS_ALIASES)
    if values is None:
        return None
    if any(value <= 1.0 for value in values):
        return None
    return values


def _largest_remainder(values: Iterable[float], target: int = 100) -> tuple[int, int, int]:
    raw = list(values)
    if len(raw) != 3:
        raise ValueError("Exakt tre värden krävs.")
    if any((not math.isfinite(value)) or value < 0 for value in raw):
        raise ValueError("Värdena måste vara ändliga och icke-negativa.")

    total = sum(raw)
    if total <= 0:
        raise ValueError("Summan måste vara större än noll.")

    scaled = [value / total * target for value in raw]
    floors = [math.floor(value) for value in scaled]
    remainder = target - sum(floors)

    order = sorted(
        range(3),
        key=lambda index: (scaled[index] - floors[index], -index),
        reverse=True,
    )
    result = floors[:]
    for index in order[:remainder]:
        result[index] += 1

    return int(result[0]), int(result[1]), int(result[2])


def parse_probability_triplet(source: Any) -> Optional[tuple[int, int, int]]:
    values = _get_triplet(source, aliases=ODDS_ALIASES)
    if values is None:
        return None
    if any(value < 0 or value > 100 for value in values):
        return None
    try:
        return _largest_remainder(values, 100)
    except ValueError:
        return None


def odds_to_values(
    odds: Sequence[float],
) -> tuple[int, int, int]:
    if len(odds) != 3:
        raise ValueError("Exakt tre decimalodds krävs.")
    checked = [float(value) for value in odds]
    if any((not math.isfinite(value)) or value <= 1.0 for value in checked):
        raise ValueError("Alla decimalodds måste vara större än 1,00.")
    inverse = [1.0 / value for value in checked]
    return _largest_remainder(inverse, 100)


def _event_number(event: Mapping[str, Any], fallback: int) -> int:
    for key in ("eventNumber", "number", "eventNo", "matchNumber"):
        value = _parse_decimal(event.get(key))
        if value is not None and value >= 1:
            return int(value)
    return fallback


def _event_description(event: Mapping[str, Any], number: int) -> str:
    for key in ("description", "name", "eventDescription", "matchDescription"):
        value = event.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    participants = event.get("participants")
    if isinstance(participants, list):
        names: list[str] = []
        for participant in participants:
            if isinstance(participant, Mapping):
                name = participant.get("name") or participant.get("description")
                if name:
                    names.append(str(name).strip())
            elif participant:
                names.append(str(participant).strip())
        if len(names) >= 2:
            return f"{names[0]} – {names[1]}"

    return f"Match {number}"


def _iso_datetime_score(value: Any) -> float:
    if not value:
        return float("-inf")
    text = str(value).strip()
    if not text:
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return float("-inf")


def _collect_draw_candidates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add(value: Any) -> None:
        if isinstance(value, Mapping) and isinstance(value.get("events"), list):
            candidates.append(dict(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping) and isinstance(item.get("events"), list):
                    candidates.append(dict(item))

    add(payload)
    for key in ("draws", "items", "results", "data"):
        add(payload.get(key))

    return candidates


def select_draw(
    payload: Mapping[str, Any],
    *,
    expected_matches: Optional[int] = None,
) -> dict[str, Any]:
    """
    Väljer den mest sannolika aktuella kupongen om svaret innehåller flera draws.
    """
    candidates = _collect_draw_candidates(payload)
    if not candidates:
        raise SvenskaSpelResponseError(
            "Hittade ingen omgång med en events-lista i API-svaret."
        )

    def score(draw: Mapping[str, Any]) -> tuple[int, int, float, float]:
        events = draw.get("events")
        event_count = len(events) if isinstance(events, list) else 0
        count_match = int(expected_matches is not None and event_count == expected_matches)

        status_text = " ".join(
            str(draw.get(key, ""))
            for key in ("status", "drawStatus", "state", "open")
        ).lower()
        open_score = int(
            any(token in status_text for token in ("open", "öppen", "active", "current"))
        )

        close_score = max(
            _iso_datetime_score(draw.get("closeTime")),
            _iso_datetime_score(draw.get("drawCloseTime")),
            _iso_datetime_score(draw.get("salesStop")),
            _iso_datetime_score(draw.get("stopTime")),
        )

        draw_number = _parse_decimal(
            draw.get("drawNumber")
            or draw.get("number")
            or draw.get("drawNo")
        )
        return count_match, open_score, close_score, draw_number or float("-inf")

    return max(candidates, key=score)


def extract_match_rows(draw: Mapping[str, Any]) -> list[MatchOddsRow]:
    events = draw.get("events")
    if not isinstance(events, list) or not events:
        raise SvenskaSpelResponseError("Den valda omgången saknar matcher i events.")

    rows: list[MatchOddsRow] = []
    for fallback_number, raw_event in enumerate(events, start=1):
        if not isinstance(raw_event, Mapping):
            continue

        number = _event_number(raw_event, fallback_number)
        description = _event_description(raw_event, number)

        start_odds = parse_odds_triplet(raw_event.get("startOdds"))
        api_start_values = parse_probability_triplet(
            raw_event.get("randomResultProbability")
        )
        calculated_start_values = (
            odds_to_values(start_odds) if start_odds is not None else None
        )

        if api_start_values is not None:
            start_values = api_start_values
            start_origin = "randomResultProbability"
        elif calculated_start_values is not None:
            start_values = calculated_start_values
            start_origin = "beräknat från startOdds"
        else:
            start_values = None
            start_origin = "saknas"

        current_odds = parse_odds_triplet(raw_event.get("odds"))
        api_current_values = parse_probability_triplet(raw_event.get("favouriteOdds"))
        calculated_current_values = (
            odds_to_values(current_odds) if current_odds is not None else None
        )

        if api_current_values is not None:
            current_values = api_current_values
            current_origin = "favouriteOdds"
        elif calculated_current_values is not None:
            current_values = calculated_current_values
            current_origin = "beräknat från odds"
        else:
            current_values = None
            current_origin = "saknas"

        if api_start_values is not None and calculated_start_values is not None:
            if api_start_values == calculated_start_values:
                comparison = "OK"
            else:
                api_text = "/".join(map(str, api_start_values))
                calc_text = "/".join(map(str, calculated_start_values))
                comparison = f"API {api_text} · beräknat {calc_text}"
        elif api_start_values is not None:
            comparison = "API-värde finns; startOdds saknas"
        elif calculated_start_values is not None:
            comparison = "Beräknat; API-värde saknas"
        else:
            comparison = "Startdata saknas"

        event_start = str(
            raw_event.get("sportEventStart")
            or raw_event.get("eventStart")
            or raw_event.get("startTime")
            or ""
        ).strip()
        event_status = str(
            raw_event.get("sportEventStatus")
            or raw_event.get("status")
            or ""
        ).strip()

        rows.append(
            MatchOddsRow(
                event_number=number,
                description=description,
                start_odds=start_odds,
                start_values=start_values,
                start_values_origin=start_origin,
                current_odds=current_odds,
                current_values=current_values,
                current_values_origin=current_origin,
                comparison=comparison,
                event_start=event_start,
                event_status=event_status,
            )
        )

    if not rows:
        raise SvenskaSpelResponseError("Inga giltiga matcher kunde läsas ur events.")

    rows.sort(key=lambda row: row.event_number)
    return rows


def build_coupon_vector(
    rows: Sequence[MatchOddsRow],
    *,
    allow_current_fallback: bool,
) -> tuple[list[int], list[str], list[int]]:
    """
    Returnerar:
    - hela kupongvektorn i ordningen 1, X, 2 per match,
    - datakälla per match,
    - matchnummer som fortfarande saknar användbara värden.
    """
    vector: list[int] = []
    sources: list[str] = []
    unresolved: list[int] = []

    for row in rows:
        if row.start_values is not None:
            values = row.start_values
            source = "Startodds"
        elif allow_current_fallback and row.current_values is not None:
            values = row.current_values
            source = "Aktuella odds (fallback)"
        else:
            unresolved.append(row.event_number)
            continue

        if sum(values) != 100:
            raise SvenskaSpelResponseError(
                f"Match {row.event_number} summerar till {sum(values)}, inte 100."
            )

        vector.extend(values)
        sources.append(source)

    return vector, sources, unresolved


def draw_metadata(draw: Mapping[str, Any]) -> dict[str, str]:
    def first(*keys: str) -> str:
        for key in keys:
            value = draw.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    return {
        "draw_number": first("drawNumber", "number", "drawNo"),
        "close_time": first("closeTime", "drawCloseTime", "salesStop", "stopTime"),
        "product_name": first("productName", "product", "name"),
        "status": first("status", "drawStatus", "state"),
    }


def raw_diagnostics(
    payload: Mapping[str, Any],
    draw: Mapping[str, Any],
) -> dict[str, Any]:
    events = draw.get("events")
    first_event = events[0] if isinstance(events, list) and events else {}
    return {
        "top_level_keys": sorted(str(key) for key in payload.keys()),
        "draw_keys": sorted(str(key) for key in draw.keys()),
        "first_event_keys": (
            sorted(str(key) for key in first_event.keys())
            if isinstance(first_event, Mapping)
            else []
        ),
    }
