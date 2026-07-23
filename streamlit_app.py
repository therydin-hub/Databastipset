# -*- coding: utf-8 -*-
"""Startfil för Databastipset med Svenska Spels kupong och integrerad grundram.

Den befintliga app.py och motorn.py lämnas orörda. Den här startfilen:

- hämtar aktuell kupong från Svenska Spels API,
- visar Streck, Odds och Startodds kompakt,
- använder startoddsvärdet i bakgrunden,
- låter användaren välja grundramens 1/X/2 direkt på varje matchrad,
- sparar grundramen direkt utan separat Spara-knapp,
- döljer gamla sektionen "Manuell grundram" när API-kupongen är hämtad.

Använd som Streamlit Cloud "Main file path":
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


# ---------------------------------------------------------------------------
# Spara originalfunktioner innan app.py körs.
# ---------------------------------------------------------------------------

_ORIGINAL_TEXT_AREA = st.text_area
_ORIGINAL_MARKDOWN = st.markdown
_ORIGINAL_CAPTION = st.caption
_ORIGINAL_WARNING = st.warning
_ORIGINAL_INFO = st.info
_ORIGINAL_SUCCESS = st.success
_ORIGINAL_COLUMNS = st.columns
_ORIGINAL_BUTTON = st.button
_ORIGINAL_EXPANDER = st.expander

_TARGET_LABEL_WORDS = ("klistra in", "procent/odds-värden")
_STOCKHOLM = ZoneInfo("Europe/Stockholm")
_MONTHS_SV = (
    "jan", "feb", "mar", "apr", "maj", "jun",
    "jul", "aug", "sep", "okt", "nov", "dec",
)

# Sätts till True endast medan app.py försöker rendera sin gamla manuella
# grundramssektion. Då låter vi beräkningarna köras men visar inget av UI:t.
_HIDE_OLD_FRAME_SECTION = False


# ---------------------------------------------------------------------------
# Grundfunktioner
# ---------------------------------------------------------------------------

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


def _normalize_signs(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw = list(values.upper())
    else:
        try:
            raw = [str(value).upper() for value in values]
        except TypeError:
            raw = []
    return [sign for sign in ("1", "X", "2") if sign in raw]


def _normalize_frame(frame: Any, match_count: int) -> list[list[str]]:
    normalized: list[list[str]] = []
    raw = frame if isinstance(frame, (list, tuple)) else []
    for index in range(match_count):
        values = raw[index] if index < len(raw) else ["1", "X", "2"]
        normalized.append(_normalize_signs(values))
    return normalized


def _frame_row_count(frame: Sequence[Sequence[str]]) -> int:
    total = 1
    for signs in frame:
        total *= len(signs)
    return total if frame else 0


def _clear_recommended_package_state() -> None:
    for key in (
        "v12_recommended_packages",
        "v12_recommended_candidate_audit",
        "v12_recommended_meta",
        "v12_applied_package_meta",
        "v12_applied_package_snapshot",
    ):
        st.session_state.pop(key, None)


def _save_direct_frame(frame: list[list[str]], spelform: str) -> None:
    """Sparar grundramen direkt när ett tecken klickas i kupongraden."""
    old_frame = _normalize_frame(
        st.session_state.get("v12_saved_frame"),
        len(frame),
    )
    changed = old_frame != frame

    st.session_state["v12_saved_frame"] = [list(signs) for signs in frame]
    st.session_state["v12_frame_defaults"] = [list(signs) for signs in frame]
    st.session_state["v12_frame_spelform"] = spelform
    st.session_state["v12_frame_saved"] = all(bool(signs) for signs in frame)

    if changed:
        st.session_state["v12_filter_saved"] = False
        st.session_state["v12_last_result_stale"] = True
        _clear_recommended_package_state()


def _initialize_direct_frame(match_count: int, spelform: str) -> None:
    frame = [["1", "X", "2"] for _ in range(match_count)]
    st.session_state["ss_frame_widget_token"] = (
        int(st.session_state.get("ss_frame_widget_token", 0) or 0) + 1
    )
    st.session_state["v12_saved_frame"] = frame
    st.session_state["v12_frame_defaults"] = [list(signs) for signs in frame]
    st.session_state["v12_frame_spelform"] = spelform
    st.session_state["v12_frame_saved"] = True
    st.session_state["v12_filter_saved"] = False
    st.session_state["v12_last_result_stale"] = True
    _clear_recommended_package_state()


def _toggle_direct_frame_sign(
    match_index: int,
    sign: str,
    match_count: int,
    spelform: str,
) -> None:
    """Växlar ett tecken direkt i den sparade grundramen."""
    frame = _normalize_frame(
        st.session_state.get("v12_saved_frame"),
        match_count,
    )
    if not 0 <= match_index < len(frame):
        return

    current = list(frame[match_index])
    if sign in current:
        # En match får aldrig lämnas utan tecken.
        if len(current) <= 1:
            return
        current.remove(sign)
    else:
        current.append(sign)

    frame[match_index] = _normalize_signs(current)
    _save_direct_frame(frame, spelform)


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


# ---------------------------------------------------------------------------
# Kompakt kupongdesign
# ---------------------------------------------------------------------------

def _render_coupon_css() -> None:
    _ORIGINAL_MARKDOWN(
        """
        <style>
        /* Lite mer arbetsyta, utan att påverka sidomenyn. */
        section.main > div.block-container {
            max-width: 1580px;
        }

        .ss-api-title {
            font-weight: 800;
            font-size: .94rem;
            margin-bottom: .12rem;
        }
        .ss-api-status {
            display: inline-flex;
            align-items: center;
            min-height: 25px;
            margin-top: 7px;
            padding: 3px 9px;
            border-radius: 999px;
            border: 1px solid rgba(67, 201, 170, .42);
            background: rgba(67, 201, 170, .12);
            color: #43c9aa;
            font-size: .70rem;
            font-weight: 800;
        }
        .ss-coupon-meta {
            margin: .28rem 0 .38rem .15rem;
            color: rgba(220, 229, 236, .62);
            font-size: .68rem;
        }
        .ss-coupon-meta strong {
            color: rgba(238, 244, 248, .88);
            font-weight: 760;
        }

        /* Gemensam tabellrubrik */
        .ss-table-head-wrap {
            margin: 0 0 .16rem 0;
        }
        .ss-head-main {
            padding-left: .50rem;
            color: rgba(238, 244, 248, .90);
            font-size: .72rem;
            font-weight: 760;
            white-space: nowrap;
        }
        .ss-head-group {
            color: rgba(238, 244, 248, .90);
            font-size: .70rem;
            font-weight: 760;
            text-align: center;
        }
        .ss-head-sub {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 5px;
            margin-top: 2px;
            color: rgba(220, 229, 236, .52);
            font-size: .61rem;
            text-align: center;
        }

        /*
        Varje match är ett vanligt st.columns-block, inte en Streamlit-container.
        Det gör raden betydligt lägre och närmare den godkända förhandsbilden.
        */
        div[data-testid="stHorizontalBlock"]:has(.ss-table-row-marker) {
            min-height: 47px !important;
            margin: 0 0 .18rem 0 !important;
            padding: .20rem .48rem !important;
            gap: .58rem !important;
            align-items: center !important;
            border: 1px solid rgba(145, 164, 179, .20);
            border-radius: 7px;
            background: rgba(42, 54, 65, .72);
        }
        div[data-testid="stHorizontalBlock"]:has(.ss-table-row-marker)
        div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.ss-table-row-marker)
        [data-testid="stMarkdownContainer"] p {
            margin: 0 !important;
        }
        .ss-table-row-marker {
            display: none;
        }

        .ss-match-cell {
            display: grid;
            grid-template-columns: 27px minmax(0, 1fr);
            column-gap: 9px;
            align-items: center;
            min-width: 0;
        }
        .ss-match-number {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 27px;
            height: 27px;
            border-radius: 7px;
            background: rgba(74, 91, 104, .58);
            color: rgba(246, 249, 251, .96);
            font-size: .72rem;
            font-weight: 850;
        }
        .ss-match-copy {
            min-width: 0;
        }
        .ss-match-title {
            overflow: hidden;
            color: rgba(246, 249, 251, .96);
            font-size: .78rem;
            font-weight: 820;
            line-height: 1.05;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .ss-match-time {
            margin-top: 3px;
            color: rgba(220, 229, 236, .52);
            font-size: .62rem;
            line-height: 1;
            white-space: nowrap;
        }
        .ss-source-fallback {
            display: inline-block;
            margin-left: 5px;
            padding: 0 5px;
            border: 1px solid rgba(230, 176, 68, .45);
            border-radius: 999px;
            background: rgba(176, 124, 25, .22);
            font-size: .53rem;
            font-weight: 800;
            vertical-align: 1px;
        }

        .ss-triple {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 5px;
            align-items: center;
            color: rgba(241, 246, 249, .94);
            font-size: .71rem;
            font-weight: 650;
            line-height: 1;
            text-align: center;
        }
        .ss-triple span {
            white-space: nowrap;
        }

        /*
        Egna st.button-knappar. Primär = valt tecken, sekundär = ej valt.
        Vi skriver över appens röda accent bara inne i kupongraderna.
        */
        div[data-testid="stHorizontalBlock"]:has(.ss-table-row-marker)
        div[data-testid="stButton"] {
            margin: 0 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.ss-table-row-marker)
        div[data-testid="stButton"] button,
        div[data-testid="stHorizontalBlock"]:has(.ss-table-row-marker)
        button[kind="secondary"],
        div[data-testid="stHorizontalBlock"]:has(.ss-table-row-marker)
        [data-testid="stBaseButton-secondary"] {
            min-height: 32px !important;
            height: 32px !important;
            padding: 0 .20rem !important;
            border: 1px solid rgba(145, 164, 179, .23) !important;
            border-radius: 5px !important;
            background: rgba(62, 75, 87, .62) !important;
            color: rgba(225, 233, 239, .62) !important;
            font-size: .76rem !important;
            font-weight: 850 !important;
            box-shadow: none !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.ss-table-row-marker)
        button[kind="primary"],
        div[data-testid="stHorizontalBlock"]:has(.ss-table-row-marker)
        [data-testid="stBaseButton-primary"] {
            border-color: #43c9aa !important;
            background: #43c9aa !important;
            color: #071b17 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.ss-table-row-marker)
        div[data-testid="stButton"] button:hover {
            border-color: #43c9aa !important;
        }

        .ss-summary-card {
            min-height: 60px;
            padding: 9px 11px;
            border: 1px solid rgba(145, 164, 179, .20);
            border-radius: 7px;
            background: rgba(31, 43, 53, .56);
        }
        .ss-summary-label {
            margin-bottom: 4px;
            color: rgba(220, 229, 236, .58);
            font-size: .64rem;
        }
        .ss-summary-value {
            color: rgba(246, 249, 251, .96);
            font-size: .82rem;
            font-weight: 850;
            white-space: nowrap;
        }
        .ss-summary-good {
            color: #43c9aa;
        }

        @media (max-width: 900px) {
            .ss-match-title {
                white-space: normal;
            }
            div[data-testid="stHorizontalBlock"]:has(.ss-table-row-marker) {
                min-width: 970px;
            }
            .ss-table-scroll-note {
                display: block;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_table_header() -> None:
    cols = _ORIGINAL_COLUMNS([2.70, 1.35, 1.35, 1.35, 1.55])
    with cols[0]:
        _ORIGINAL_MARKDOWN(
            '<div class="ss-table-head-wrap">'
            '<div class="ss-head-main">Match</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    for col, label in zip(
        cols[1:],
        ("Streck", "Odds", "Startodds", "Grundram"),
    ):
        with col:
            _ORIGINAL_MARKDOWN(
                '<div class="ss-table-head-wrap">'
                f'<div class="ss-head-group">{html.escape(label)}</div>'
                '<div class="ss-head-sub">'
                '<span>1</span><span>X</span><span>2</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )


def _render_match_cell(row: Any, fallback_used: bool) -> None:
    fallback_badge = (
        '<span class="ss-source-fallback">Odds fallback</span>'
        if fallback_used
        else ""
    )
    _ORIGINAL_MARKDOWN(
        '<div class="ss-table-row-marker"></div>'
        '<div class="ss-match-cell">'
        f'<div class="ss-match-number">{int(row.event_number)}</div>'
        '<div class="ss-match-copy">'
        f'<div class="ss-match-title">{html.escape(str(row.description))}'
        f'{fallback_badge}</div>'
        f'<div class="ss-match-time">'
        f'{html.escape(_format_match_time(row.event_start))}</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def _render_triple(values: Sequence[str]) -> None:
    safe = [html.escape(str(value)) for value in values]
    _ORIGINAL_MARKDOWN(
        '<div class="ss-triple">'
        f'<span>{safe[0]}</span>'
        f'<span>{safe[1]}</span>'
        f'<span>{safe[2]}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_sign_buttons(
    row_index: int,
    row: Any,
    selected_signs: Sequence[str],
    token: int,
    match_count: int,
    spelform: str,
) -> None:
    button_cols = _ORIGINAL_COLUMNS(3)
    for col, sign in zip(button_cols, ("1", "X", "2")):
        with col:
            is_selected = sign in selected_signs
            _ORIGINAL_BUTTON(
                sign,
                key=(
                    f"ss_frame_button_{token}_{spelform}_"
                    f"{int(row.event_number)}_{sign}"
                ),
                type="primary" if is_selected else "secondary",
                use_container_width=True,
                on_click=_toggle_direct_frame_sign,
                args=(row_index, sign, match_count, spelform),
                help=(
                    f"Ta bort {sign} från match {int(row.event_number)}"
                    if is_selected and len(selected_signs) > 1
                    else f"Lägg till {sign} i match {int(row.event_number)}"
                    if not is_selected
                    else "Minst ett tecken måste vara valt"
                ),
            )


def _render_summary_cards(
    frame: Sequence[Sequence[str]],
    match_count: int,
) -> None:
    rows_count = _frame_row_count(frame)
    spikar = sum(len(signs) == 1 for signs in frame)
    halv = sum(len(signs) == 2 for signs in frame)
    hel = sum(len(signs) == 3 for signs in frame)
    sign_counts = {
        sign: sum(sign in signs for signs in frame)
        for sign in ("1", "X", "2")
    }

    cards = _ORIGINAL_COLUMNS(4)
    values = (
        ("Förhandsvisning rader", f"{rows_count:,}".replace(",", " "), ""),
        ("Ramtyp", f"{spikar} spik · {hel} hel · {halv} halv", ""),
        (
            "Markerade tecken",
            f"1:{sign_counts['1']} · X:{sign_counts['X']} · 2:{sign_counts['2']}",
            "",
        ),
        (
            "Kupong",
            f"✓ {match_count}/{match_count} matcher",
            " ss-summary-good",
        ),
    )
    for col, (label, value, extra_class) in zip(cards, values):
        with col:
            _ORIGINAL_MARKDOWN(
                '<div class="ss-summary-card">'
                f'<div class="ss-summary-label">{html.escape(label)}</div>'
                f'<div class="ss-summary-value{extra_class}">'
                f'{html.escape(value)}</div>'
                '</div>',
                unsafe_allow_html=True,
            )


def _render_match_editor(
    rows: Sequence[Any],
    allow_fallback: bool,
    spelform: str,
) -> list[list[str]]:
    match_count = len(rows)
    frame = _normalize_frame(
        st.session_state.get("v12_saved_frame"),
        match_count,
    )
    token = int(st.session_state.get("ss_frame_widget_token", 0) or 0)

    _render_table_header()

    for index, row in enumerate(rows):
        fallback_used = bool(
            allow_fallback
            and row.start_values is None
            and row.current_values is not None
        )

        streck = _format_percent(row.distribution)
        odds = _format_odds(row.current_odds)
        start_odds = _format_odds(row.start_odds)

        cols = _ORIGINAL_COLUMNS([2.70, 1.35, 1.35, 1.35, 1.55])
        with cols[0]:
            _render_match_cell(row, fallback_used)
        with cols[1]:
            _render_triple(streck)
        with cols[2]:
            _render_triple(odds)
        with cols[3]:
            _render_triple(start_odds)
        with cols[4]:
            _render_sign_buttons(
                index,
                row,
                frame[index],
                token,
                match_count,
                spelform,
            )

    # Callbacken sparar vid klick. Detta säkerställer även rätt metadata
    # när kupongen precis har hämtats.
    _save_direct_frame(frame, spelform)

    missing = [
        f"M{index + 1}"
        for index, signs in enumerate(frame)
        if not signs
    ]
    if missing:
        _ORIGINAL_WARNING(
            "Grundramen är inte komplett. Välj minst ett tecken i "
            + ", ".join(missing)
            + "."
        )
        return frame

    _render_summary_cards(frame, match_count)
    return frame


# ---------------------------------------------------------------------------
# Ersätt app.py:s gamla kupong-textarea med API-kupongen.
# ---------------------------------------------------------------------------

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
        st.session_state["v12_frame_saved"] = False
    st.session_state["ss_main_last_spelform"] = current_spelform

    expected_matches = EXPECTED_MATCHES[current_spelform]
    product_code = PRODUCT_MAP[current_spelform]
    saved_key = _get_saved_access_key()

    _render_coupon_css()

    with st.container(border=True):
        title_col, action_col = _ORIGINAL_COLUMNS([2.5, 1.0])
        with title_col:
            _ORIGINAL_MARKDOWN(
                '<div class="ss-api-title">Kupong från Svenska Spel</div>',
                unsafe_allow_html=True,
            )
            _ORIGINAL_CAPTION(
                "Startoddsvärdet används i bakgrunden. "
                "Streck, odds och startodds visas här."
            )
            existing_rows = st.session_state.get("ss_main_rows")
            existing_product = st.session_state.get("ss_main_product")
            if existing_rows and existing_product == current_spelform:
                _ORIGINAL_MARKDOWN(
                    '<div class="ss-api-status">✓ Kupong hämtad '
                    f'{len(existing_rows)}/{expected_matches}</div>',
                    unsafe_allow_html=True,
                )
        with action_col:
            fetch_clicked = _ORIGINAL_BUTTON(
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
                help="Nyckeln kan läggas permanent i Streamlit Secrets.",
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
            _initialize_direct_frame(len(rows), current_spelform)
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
        else:
            st.session_state.pop("ss_main_vector_text", None)

        stop_text = _format_stop_time(metadata.get("close_time"))
        start_count = sum(row.start_values is not None for row in rows)
        fallback_count = sum(
            source == "Aktuella odds (fallback)"
            for source in sources
        )

        meta_parts = [
            f'<strong>{html.escape(current_spelform)}</strong>',
            f'Omgång {html.escape(metadata.get("draw_number") or "–")}',
            f'Startodds {start_count}/{len(rows)}',
        ]
        if stop_text:
            meta_parts.append(f'Spelstopp {html.escape(stop_text)}')
        if fallback_count:
            meta_parts.append(f'Fallback {fallback_count}')

        _ORIGINAL_MARKDOWN(
            '<div class="ss-coupon-meta">'
            + ' &nbsp;·&nbsp; '.join(meta_parts)
            + '</div>',
            unsafe_allow_html=True,
        )

        _render_match_editor(
            rows,
            allow_fallback,
            current_spelform,
        )

        if len(rows) != expected_matches:
            st.error(
                f"API:t gav {len(rows)} matcher men {current_spelform} ska ha "
                f"{expected_matches}. Kupongen skickas inte till analysen."
            )

        if missing_start and not allow_fallback:
            missing_names = ", ".join(
                f"M{row.event_number} {row.description}"
                for row in missing_start
            )
            st.warning(
                f"Startodds saknas för {len(missing_start)} matcher: "
                f"{missing_names}. Inga aktuella odds används automatiskt."
            )
            if fallback_ready:
                yes_col, no_col = _ORIGINAL_COLUMNS(2)
                with yes_col:
                    if _ORIGINAL_BUTTON(
                        "Ja, använd odds för saknade matcher",
                        key="ss_main_accept_fallback",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state["ss_main_allow_current_fallback"] = True
                        st.rerun()
                with no_col:
                    _ORIGINAL_BUTTON(
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
                "Övriga matcher använder startodds."
            )
            if _ORIGINAL_BUTTON(
                "Ångra odds-fallback",
                key="ss_main_undo_fallback",
            ):
                st.session_state["ss_main_allow_current_fallback"] = False
                st.rerun()

        if impossible:
            names = ", ".join(
                f"M{row.event_number}"
                for row in impossible
            )
            st.error(
                f"Varken startodds eller aktuella odds finns för {names}."
            )

        if api_complete:
            database_name = DATABASE_FILE_MAP.get(
                current_spelform,
                "rätt statistikfil",
            )
            st.caption(f"Analysen kopplas till: {database_name}")
        elif unresolved:
            st.info(
                "Kupongen är inte redo för analys. Olösta matcher: "
                + ", ".join(f"M{number}" for number in unresolved)
            )

    # Reservläge för kupongdata. Detta är inte den gamla manuella grundramen.
    current_saved_text = str(
        st.session_state.get("v12_input_text", "") or ""
    )
    if "v12_manual_input_text" not in st.session_state:
        st.session_state["v12_manual_input_text"] = (
            "" if api_complete else current_saved_text
        )

    with _ORIGINAL_EXPANDER(
        "Manuell kupongdata (reservläge)",
        expanded=False,
    ):
        st.caption(
            "Använd endast om API:t inte kan ge en komplett kupong. "
            "Grundramen väljs fortfarande på matchraderna när kupongen finns."
        )
        if "v12_use_manual_input" not in st.session_state:
            st.session_state["v12_use_manual_input"] = False
        use_manual = st.checkbox(
            "Använd manuell kupongdata istället för API-värdet",
            key="v12_use_manual_input",
        )
        manual_text = _ORIGINAL_TEXT_AREA(
            "Manuella startoddsvärden",
            height=int(kwargs.get("height", 90) or 90),
            key="v12_manual_input_text",
            placeholder=str(
                kwargs.get(
                    "placeholder",
                    "Exempel: 62 23 15 48 29 23 ...",
                )
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


# ---------------------------------------------------------------------------
# Dölj gamla "Steg 2 – Manuell grundram" när API-kupongen visas.
#
# Beräkningarna i app.py får fortfarande köra. Dummy-kolumnerna lämnar tillbaka
# de värden som redan ligger i v12_saved_frame. Därför fungerar resten av
# appens filtermotor utan att den gamla grundramen behöver visas.
# ---------------------------------------------------------------------------

class _SilentColumn:
    def __enter__(self) -> "_SilentColumn":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        return None

    def write(self, *args: Any, **kwargs: Any) -> None:
        return None

    def caption(self, *args: Any, **kwargs: Any) -> None:
        return None

    def checkbox(
        self,
        label: Any,
        value: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        del label, args, kwargs
        return bool(value)


def _api_coupon_is_visible() -> bool:
    rows = st.session_state.get("ss_main_rows")
    product = st.session_state.get("ss_main_product")
    current = st.session_state.get("v12_spelform")
    return bool(rows) and product == current


def _patched_markdown(body: Any, *args: Any, **kwargs: Any) -> Any:
    global _HIDE_OLD_FRAME_SECTION

    text = str(body or "")
    lowered = text.lower()

    if (
        not _HIDE_OLD_FRAME_SECTION
        and _api_coupon_is_visible()
        and "steg 2" in lowered
        and "manuell grundram" in lowered
    ):
        _HIDE_OLD_FRAME_SECTION = True
        return None

    if _HIDE_OLD_FRAME_SECTION:
        if "steg 3" in lowered and "filtercentral" in lowered:
            _HIDE_OLD_FRAME_SECTION = False
            return _ORIGINAL_MARKDOWN(body, *args, **kwargs)
        return None

    return _ORIGINAL_MARKDOWN(body, *args, **kwargs)


def _patched_columns(spec: Any, *args: Any, **kwargs: Any) -> Any:
    if _HIDE_OLD_FRAME_SECTION:
        count = int(spec) if isinstance(spec, int) else len(spec)
        return [_SilentColumn() for _ in range(count)]
    return _ORIGINAL_COLUMNS(spec, *args, **kwargs)


def _patched_caption(body: Any, *args: Any, **kwargs: Any) -> Any:
    if _HIDE_OLD_FRAME_SECTION:
        return None
    return _ORIGINAL_CAPTION(body, *args, **kwargs)


def _patched_warning(body: Any, *args: Any, **kwargs: Any) -> Any:
    if _HIDE_OLD_FRAME_SECTION:
        return None
    return _ORIGINAL_WARNING(body, *args, **kwargs)


def _patched_info(body: Any, *args: Any, **kwargs: Any) -> Any:
    if _HIDE_OLD_FRAME_SECTION:
        return None
    return _ORIGINAL_INFO(body, *args, **kwargs)


def _patched_success(body: Any, *args: Any, **kwargs: Any) -> Any:
    if _HIDE_OLD_FRAME_SECTION:
        return None
    return _ORIGINAL_SUCCESS(body, *args, **kwargs)


def _patched_button(label: Any, *args: Any, **kwargs: Any) -> Any:
    if _HIDE_OLD_FRAME_SECTION:
        return False

    normalized = str(label or "").strip().lower()
    if "läs in kupong" in normalized:
        has_coupon_data = bool(
            str(st.session_state.get("v12_input_text", "") or "").strip()
        )
        frame_complete = bool(
            st.session_state.get("v12_frame_saved", False)
        )
        kwargs["disabled"] = not (has_coupon_data and frame_complete)
        kwargs.setdefault(
            "help",
            (
                "Hämta en komplett kupong och välj minst ett tecken "
                "per match."
                if kwargs["disabled"]
                else "Läser in kupongen med grundramen som valts ovan."
            ),
        )

    return _ORIGINAL_BUTTON(label, *args, **kwargs)


def _patched_expander(
    label: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    global _HIDE_OLD_FRAME_SECTION

    # När app.py går vidare från den gamla grundramen är första komponenten
    # normalt panelen för manuella teckengrupper i Steg 3.
    if _HIDE_OLD_FRAME_SECTION:
        _HIDE_OLD_FRAME_SECTION = False
    return _ORIGINAL_EXPANDER(label, *args, **kwargs)


# Installera patcharna innan app.py körs.
st.text_area = _patched_text_area
st.markdown = _patched_markdown
st.columns = _patched_columns
st.caption = _patched_caption
st.warning = _patched_warning
st.info = _patched_info
st.success = _patched_success
st.button = _patched_button
st.expander = _patched_expander


_APP_PATH = Path(__file__).resolve().with_name("app.py")
if not _APP_PATH.exists():
    raise FileNotFoundError(
        "Hittar inte app.py. Lägg streamlit_app.py i samma mapp som app.py."
    )

runpy.run_path(str(_APP_PATH), run_name="__main__")
