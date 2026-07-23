# -*- coding: utf-8 -*-
"""Startfil för Databastipset med Svenska Spels kuponglayout.

Den befintliga stora ``app.py`` och ``motorn.py`` lämnas orörda. Den här filen
ersätter bara den gamla procent-/odds-textrutan när ``app.py`` körs och matar
sedan den färdiga startoddsvärdesvektorn till samma analys som tidigare.

Använd denna fil som Streamlit Cloud "Main file path":
    streamlit_app.py
"""

from __future__ import annotations

import html
import os
import runpy
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

import streamlit as st

from svenska_spel_api import (
    DATABASE_FILE_MAP,
    EXPECTED_MATCHES,
    PRODUCT_MAP,
    SvenskaSpelAccessKeyError,
    SvenskaSpelApiError,
    build_coupon_vector,
    draw_metadata,
    extract_match_rows,
    fetch_current_draw,
    select_draw,
)


_ORIGINAL_TEXT_AREA = st.text_area
_TARGET_LABEL_WORDS = ("klistra in", "procent/odds-värden")
_STOCKHOLM = ZoneInfo("Europe/Stockholm")
_MONTHS_SV = (
    "jan", "feb", "mar", "apr", "maj", "jun",
    "jul", "aug", "sep", "okt", "nov", "dec",
)


def _is_coupon_input_text_area(label: Any) -> bool:
    normalized = str(label or "").strip().lower()
    return all(word in normalized for word in _TARGET_LABEL_WORDS)


def _get_saved_access_key() -> str:
    key = ""
    try:
        key = str(st.secrets.get("SVENSKA_SPEL_ACCESSKEY", "")).strip()
    except Exception:
        key = ""
    if not key:
        key = os.getenv("SVENSKA_SPEL_ACCESSKEY", "").strip()
    return key


def _format_odds(values: Optional[Sequence[float]]) -> tuple[str, str, str]:
    if values is None or len(values) != 3:
        return "–", "–", "–"
    return tuple(f"{float(value):.2f}" for value in values)  # type: ignore[return-value]


def _format_percent(values: Optional[Sequence[int]]) -> tuple[str, str, str]:
    if values is None or len(values) != 3:
        return "–", "–", "–"
    return tuple(f"{int(value)}%" for value in values)  # type: ignore[return-value]


def _format_match_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Tid saknas"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_STOCKHOLM)
        parsed = parsed.astimezone(_STOCKHOLM)
        return f"{_MONTHS_SV[parsed.month - 1]} {parsed.day} - {parsed:%H:%M}"
    except (ValueError, TypeError, OverflowError):
        return text


def _format_stop_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_STOCKHOLM)
        parsed = parsed.astimezone(_STOCKHOLM)
        return f"{_MONTHS_SV[parsed.month - 1]} {parsed.day} {parsed:%H:%M}"
    except (ValueError, TypeError, OverflowError):
        return text


def _clear_api_coupon_state(*, keep_error: bool = False) -> None:
    keys = (
        "ss_main_payload",
        "ss_main_draw",
        "ss_main_rows",
        "ss_main_product",
        "ss_main_allow_current_fallback",
        "ss_main_vector_text",
        "ss_main_sources",
    )
    for key in keys:
        st.session_state.pop(key, None)
    if not keep_error:
        st.session_state.pop("ss_main_error", None)


def _render_coupon_css() -> None:
    st.markdown(
        """
        <style>
        .ss-coupon-shell {
            border: 1px solid rgba(132, 150, 166, .25);
            border-radius: 14px;
            padding: 10px;
            background: rgba(52, 72, 84, .18);
            margin: .2rem 0 .65rem 0;
        }
        .ss-coupon-topline {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 7px;
            margin: 0 0 9px 0;
        }
        .ss-pill {
            display: inline-flex;
            align-items: center;
            min-height: 27px;
            padding: 3px 9px;
            border-radius: 999px;
            border: 1px solid rgba(145, 164, 179, .28);
            background: rgba(72, 94, 107, .30);
            font-size: .78rem;
            font-weight: 700;
        }
        .ss-pill-good {
            border-color: rgba(77, 190, 150, .38);
            background: rgba(42, 132, 103, .18);
        }
        .ss-pill-warn {
            border-color: rgba(230, 176, 68, .45);
            background: rgba(176, 124, 25, .18);
        }
        .ss-match-list {
            display: grid;
            gap: 5px;
        }
        .ss-match-card {
            border: 1px solid rgba(145, 164, 179, .18);
            border-radius: 10px;
            overflow: hidden;
            background: rgba(62, 84, 97, .58);
        }
        .ss-match-card.ss-fallback {
            border-color: rgba(230, 176, 68, .48);
        }
        .ss-match-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            padding: 6px 9px 5px 9px;
            background: rgba(39, 58, 69, .72);
        }
        .ss-match-title {
            min-width: 0;
            font-weight: 800;
            font-size: .90rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .ss-match-time {
            flex: 0 0 auto;
            opacity: .70;
            font-size: .76rem;
            white-space: nowrap;
        }
        .ss-source {
            display: inline-block;
            margin-left: 7px;
            padding: 1px 6px;
            border-radius: 999px;
            font-size: .66rem;
            font-weight: 800;
            vertical-align: 1px;
            background: rgba(42, 132, 103, .24);
            border: 1px solid rgba(77, 190, 150, .35);
        }
        .ss-source-fallback {
            background: rgba(176, 124, 25, .24);
            border-color: rgba(230, 176, 68, .45);
        }
        .ss-odds-grid {
            display: grid;
            grid-template-columns: minmax(76px, .85fr) repeat(3, minmax(60px, 1fr));
            align-items: center;
            padding: 2px 9px 5px 9px;
        }
        .ss-row-label {
            opacity: .68;
            font-size: .76rem;
            line-height: 1.55;
        }
        .ss-cell {
            display: grid;
            grid-template-columns: 18px 1fr;
            align-items: baseline;
            gap: 4px;
            border-left: 1px solid rgba(145, 164, 179, .18);
            padding-left: 8px;
            min-width: 0;
            line-height: 1.55;
        }
        .ss-sign {
            opacity: .58;
            font-size: .72rem;
        }
        .ss-value {
            font-size: .82rem;
            font-weight: 800;
            text-align: right;
            padding-right: 6px;
            white-space: nowrap;
        }
        @media (max-width: 650px) {
            .ss-match-head {align-items: flex-start;}
            .ss-match-title {white-space: normal;}
            .ss-odds-grid {
                grid-template-columns: 66px repeat(3, minmax(48px, 1fr));
                padding-left: 7px;
                padding-right: 7px;
            }
            .ss-cell {padding-left: 5px; grid-template-columns: 14px 1fr; gap: 2px;}
            .ss-value {padding-right: 2px; font-size: .78rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _cell(sign: str, value: str) -> str:
    return (
        '<div class="ss-cell">'
        f'<span class="ss-sign">{html.escape(sign)}</span>'
        f'<span class="ss-value">{html.escape(value)}</span>'
        "</div>"
    )


def _render_match_cards(rows: Sequence[Any], allow_fallback: bool) -> None:
    cards: list[str] = []
    for row in rows:
        fallback_used = bool(
            allow_fallback
            and row.start_values is None
            and row.current_values is not None
        )
        source_badge = (
            '<span class="ss-source ss-source-fallback">Odds fallback</span>'
            if fallback_used
            else ""
        )
        card_class = "ss-match-card ss-fallback" if fallback_used else "ss-match-card"

        streck = _format_percent(row.distribution)
        odds = _format_odds(row.current_odds)
        start_odds = _format_odds(row.start_odds)

        rows_html = []
        for label, values in (
            ("Streck", streck),
            ("Odds", odds),
            ("Startodds", start_odds),
        ):
            rows_html.append(f'<div class="ss-row-label">{label}</div>')
            rows_html.append(_cell("1", values[0]))
            rows_html.append(_cell("X", values[1]))
            rows_html.append(_cell("2", values[2]))

        cards.append(
            f'<article class="{card_class}">'
            '<div class="ss-match-head">'
            f'<div class="ss-match-title">{int(row.event_number)}&nbsp; '
            f'{html.escape(str(row.description))}'
            f'{source_badge}</div>'
            f'<div class="ss-match-time">{html.escape(_format_match_time(row.event_start))}</div>'
            "</div>"
            f'<div class="ss-odds-grid">{"".join(rows_html)}</div>'
            "</article>"
        )

    st.markdown(
        '<div class="ss-coupon-shell"><div class="ss-match-list">'
        + "".join(cards)
        + "</div></div>",
        unsafe_allow_html=True,
    )


def _render_api_coupon_input(
    label: str,
    *args: Any,
    **kwargs: Any,
) -> str:
    del label, args
    current_spelform = str(st.session_state.get("v12_spelform", "Stryktips"))
    if current_spelform not in PRODUCT_MAP:
        current_spelform = "Stryktips"

    previous_spelform = st.session_state.get("ss_main_last_spelform")
    if previous_spelform and previous_spelform != current_spelform:
        if st.session_state.get("v12_coupon_input_mode") == "api":
            st.session_state["v12_input_text"] = ""
            st.session_state["v12_manual_input_text"] = ""
        _clear_api_coupon_state()
    st.session_state["ss_main_last_spelform"] = current_spelform

    expected_matches = EXPECTED_MATCHES[current_spelform]
    product_code = PRODUCT_MAP[current_spelform]
    saved_key = _get_saved_access_key()

    _render_coupon_css()

    with st.container(border=True):
        title_col, action_col = st.columns([2.5, 1.0])
        with title_col:
            st.markdown("**Kupong från Svenska Spel**")
            st.caption(
                "Analysen använder startoddsvärdet i bakgrunden. "
                "Streck, odds och startodds visas här."
            )
        with action_col:
            fetch_clicked = st.button(
                "Hämta aktuell kupong",
                key=f"ss_main_fetch_{current_spelform}",
                type="primary",
                use_container_width=True,
            )

        entered_key = ""
        if not saved_key:
            entered_key = st.text_input(
                "Svenska Spel accesskey",
                type="password",
                key="ss_main_access_key",
                help="Nyckeln kan senare läggas permanent i Streamlit Secrets.",
            )

    if fetch_clicked:
        _clear_api_coupon_state()
        access_key = saved_key or str(entered_key or "").strip()
        try:
            with st.spinner("Hämtar kupong, streck, odds och startodds..."):
                payload = fetch_current_draw(product_code, access_key)
                draw = select_draw(payload, expected_matches=expected_matches)
                rows = extract_match_rows(draw)
            st.session_state["ss_main_payload"] = payload
            st.session_state["ss_main_draw"] = draw
            st.session_state["ss_main_rows"] = rows
            st.session_state["ss_main_product"] = current_spelform
            st.session_state["ss_main_allow_current_fallback"] = False
            st.session_state["v12_coupon_input_mode"] = "api"
        except (SvenskaSpelAccessKeyError, SvenskaSpelApiError) as exc:
            st.session_state["ss_main_error"] = str(exc)
        except Exception as exc:
            st.session_state["ss_main_error"] = (
                f"Oväntat fel av typen {type(exc).__name__}: {exc}"
            )

    api_error = st.session_state.get("ss_main_error")
    if api_error:
        st.error(str(api_error))

    rows = st.session_state.get("ss_main_rows")
    draw = st.session_state.get("ss_main_draw")
    stored_product = st.session_state.get("ss_main_product")
    api_complete = False
    api_vector_text = ""

    if rows and isinstance(draw, dict) and stored_product == current_spelform:
        allow_fallback = bool(
            st.session_state.get("ss_main_allow_current_fallback", False)
        )
        metadata = draw_metadata(draw)
        missing_start = [row for row in rows if row.start_values is None]
        fallback_ready = [
            row for row in missing_start if row.current_values is not None
        ]
        impossible = [
            row for row in missing_start if row.current_values is None
        ]

        vector, sources, unresolved = build_coupon_vector(
            rows,
            allow_current_fallback=allow_fallback,
        )
        api_complete = (
            len(rows) == expected_matches
            and len(vector) == expected_matches * 3
            and not unresolved
        )
        if api_complete:
            api_vector_text = " ".join(str(value) for value in vector)
            st.session_state["ss_main_vector_text"] = api_vector_text
            st.session_state["ss_main_sources"] = sources
            st.session_state["v12_input_text"] = api_vector_text
            st.session_state["v12_coupon_input_mode"] = "api"

        stop_text = _format_stop_time(metadata.get("close_time"))
        start_count = sum(row.start_values is not None for row in rows)
        fallback_count = sum(
            1 for source in sources if source == "Aktuella odds (fallback)"
        )

        pills = [
            f'<span class="ss-pill">{html.escape(current_spelform)}</span>',
            f'<span class="ss-pill">Omgång {html.escape(metadata.get("draw_number") or "–")}</span>',
            f'<span class="ss-pill">{len(rows)}/{expected_matches} matcher</span>',
            f'<span class="ss-pill ss-pill-good">Startodds {start_count}/{len(rows)}</span>',
        ]
        if stop_text:
            pills.append(f'<span class="ss-pill">Spelstopp {html.escape(stop_text)}</span>')
        if fallback_count:
            pills.append(
                f'<span class="ss-pill ss-pill-warn">Fallback {fallback_count} matcher</span>'
            )

        st.markdown(
            '<div class="ss-coupon-topline">' + "".join(pills) + "</div>",
            unsafe_allow_html=True,
        )
        _render_match_cards(rows, allow_fallback)

        if len(rows) != expected_matches:
            st.error(
                f"API:t gav {len(rows)} matcher men {current_spelform} ska ha "
                f"{expected_matches}. Kupongen skickas inte till analysen."
            )

        if missing_start and not allow_fallback:
            missing_names = ", ".join(
                f"M{row.event_number} {row.description}" for row in missing_start
            )
            st.warning(
                f"Startodds saknas för {len(missing_start)} matcher: {missing_names}. "
                "Inga aktuella odds används automatiskt."
            )
            if fallback_ready:
                yes_col, no_col = st.columns(2)
                with yes_col:
                    if st.button(
                        "Ja, använd odds för saknade matcher",
                        key="ss_main_accept_fallback",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state["ss_main_allow_current_fallback"] = True
                        st.rerun()
                with no_col:
                    st.button(
                        "Nej, behåll kupongen ofullständig",
                        key="ss_main_reject_fallback",
                        use_container_width=True,
                    )

        if allow_fallback and missing_start:
            used = ", ".join(
                f"M{row.event_number}"
                for row in missing_start
                if row.current_values is not None
            )
            st.warning(
                f"Aktuella odds används i bakgrunden för {used}. "
                "Alla övriga matcher använder startodds."
            )
            if st.button("Ångra odds-fallback", key="ss_main_undo_fallback"):
                st.session_state["ss_main_allow_current_fallback"] = False
                st.rerun()

        if impossible:
            names = ", ".join(f"M{row.event_number}" for row in impossible)
            st.error(
                f"Varken startodds eller aktuella odds finns för {names}. "
                "Dessa matcher måste matas in manuellt."
            )

        if api_complete:
            database_name = DATABASE_FILE_MAP.get(current_spelform, "rätt statistikfil")
            if fallback_count:
                st.success(
                    f"Kupongen är klar: {len(vector)} värden skapade i bakgrunden. "
                    f"{fallback_count} matcher använder godkänd odds-fallback."
                )
            else:
                st.success(
                    f"Kupongen är klar: samtliga {len(rows)} matcher använder startodds."
                )
            st.caption(f"Analysen kopplas till: {database_name}")
        elif unresolved:
            st.info(
                "Kupongen är inte redo för analys. Olösta matcher: "
                + ", ".join(f"M{number}" for number in unresolved)
            )

    current_saved_text = str(st.session_state.get("v12_input_text", "") or "")
    if "v12_manual_input_text" not in st.session_state:
        st.session_state["v12_manual_input_text"] = (
            "" if api_complete else current_saved_text
        )

    with st.expander("Manuell inmatning (reservläge)", expanded=False):
        st.caption(
            "Använd bara detta om API:t inte kan ge en komplett kupong. "
            "Den manuella raden ska innehålla 1-X-2-värden i matchordning."
        )
        if "v12_use_manual_input" not in st.session_state:
            st.session_state["v12_use_manual_input"] = False
        use_manual = st.checkbox(
            "Använd manuell inmatning istället för API-kupongen",
            key="v12_use_manual_input",
        )
        manual_text = _ORIGINAL_TEXT_AREA(
            "Manuella startoddsvärden",
            height=int(kwargs.get("height", 90) or 90),
            key="v12_manual_input_text",
            placeholder=str(
                kwargs.get("placeholder", "Exempel: 62 23 15 48 29 23 ...")
            ),
        )

    if use_manual or not api_complete:
        active_text = str(manual_text or "")
        st.session_state["v12_coupon_input_mode"] = "manual"
    else:
        active_text = api_vector_text
        st.session_state["v12_coupon_input_mode"] = "api"

    st.session_state["v12_input_text"] = active_text
    return active_text


def _patched_text_area(label: str, *args: Any, **kwargs: Any) -> Any:
    if _is_coupon_input_text_area(label):
        return _render_api_coupon_input(label, *args, **kwargs)
    return _ORIGINAL_TEXT_AREA(label, *args, **kwargs)


st.text_area = _patched_text_area

_APP_PATH = Path(__file__).resolve().with_name("app.py")
if not _APP_PATH.exists():
    raise FileNotFoundError(
        "Hittar inte app.py. Lägg streamlit_app.py i samma mapp som app.py."
    )

runpy.run_path(str(_APP_PATH), run_name="__main__")
