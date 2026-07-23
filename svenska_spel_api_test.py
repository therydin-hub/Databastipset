# -*- coding: utf-8 -*-
"""
Isolerad Streamlit-test för Svenska Spels externa API.

Kör lokalt:
    streamlit run svenska_spel_api_test.py

Lägg denna fil och svenska_spel_api.py i samma mapp.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import pandas as pd
import streamlit as st

from svenska_spel_api import (
    EXPECTED_MATCHES,
    PRODUCT_MAP,
    SvenskaSpelAccessKeyError,
    SvenskaSpelApiError,
    build_coupon_vector,
    draw_metadata,
    extract_match_rows,
    fetch_current_draw,
    raw_diagnostics,
    select_draw,
)


st.set_page_config(
    page_title="Svenska Spel API-test",
    page_icon="🧪",
    layout="wide",
)


def get_saved_access_key() -> str:
    """Läser nyckeln utan att krascha om Streamlit Secrets saknas."""
    key = ""
    try:
        key = str(st.secrets.get("SVENSKA_SPEL_ACCESSKEY", "")).strip()
    except Exception:
        key = ""

    if not key:
        key = os.getenv("SVENSKA_SPEL_ACCESSKEY", "").strip()
    return key


def fmt_odds(values: Optional[tuple[float, float, float]]) -> tuple[str, str, str]:
    if values is None:
        return "–", "–", "–"
    return tuple(f"{value:.2f}".replace(".", ",") for value in values)  # type: ignore[return-value]


def fmt_values(values: Optional[tuple[int, int, int]]) -> tuple[str, str, str]:
    if values is None:
        return "–", "–", "–"
    return str(values[0]), str(values[1]), str(values[2])


def reset_result_state() -> None:
    for key in (
        "ss_payload",
        "ss_draw",
        "ss_rows",
        "ss_error",
        "ss_allow_current_fallback",
        "ss_result_product",
    ):
        st.session_state.pop(key, None)


def source_for_row(row: Any, allow_current_fallback: bool) -> str:
    if row.start_values is not None:
        return "Startodds"
    if allow_current_fallback and row.current_values is not None:
        return "Aktuella odds (fallback)"
    if row.current_values is not None:
        return "Väntar på godkännande"
    return "Saknar odds"


st.title("🧪 Svenska Spel API-test")
st.caption(
    "Testar kupong, startodds och omräknade 1-X-2-värden utan att röra "
    "Databastipsets analysmotor."
)

with st.container(border=True):
    col_product, col_key, col_button = st.columns([1.0, 2.3, 1.0])

    with col_product:
        selected_product = st.selectbox(
            "Spelform",
            list(PRODUCT_MAP.keys()),
            index=0,
        )

    saved_key = get_saved_access_key()
    with col_key:
        if saved_key:
            st.success("API-nyckel hittad i Secrets eller miljövariabel.", icon="🔐")
            entered_key = ""
        else:
            entered_key = st.text_input(
                "Svenska Spel accesskey",
                type="password",
                help=(
                    "Nyckeln används endast för detta API-anrop och visas inte i "
                    "diagnostiken."
                ),
            )

    with col_button:
        st.write("")
        st.write("")
        fetch_clicked = st.button(
            "Hämta aktuell kupong",
            type="primary",
            use_container_width=True,
        )

if fetch_clicked:
    reset_result_state()
    access_key = saved_key or entered_key.strip()
    product_code = PRODUCT_MAP[selected_product]
    expected_matches = EXPECTED_MATCHES[selected_product]

    try:
        with st.spinner("Hämtar kupongen från Svenska Spel..."):
            payload = fetch_current_draw(product_code, access_key)
            draw = select_draw(payload, expected_matches=expected_matches)
            rows = extract_match_rows(draw)

        st.session_state["ss_payload"] = payload
        st.session_state["ss_draw"] = draw
        st.session_state["ss_rows"] = rows
        st.session_state["ss_result_product"] = selected_product
        st.session_state["ss_allow_current_fallback"] = False
    except SvenskaSpelAccessKeyError as exc:
        st.session_state["ss_error"] = str(exc)
    except SvenskaSpelApiError as exc:
        st.session_state["ss_error"] = str(exc)
    except Exception as exc:
        st.session_state["ss_error"] = (
            f"Oväntat fel av typen {type(exc).__name__}: {exc}"
        )

if st.session_state.get("ss_error"):
    st.error(st.session_state["ss_error"])

payload = st.session_state.get("ss_payload")
draw = st.session_state.get("ss_draw")
rows = st.session_state.get("ss_rows")
result_product = st.session_state.get("ss_result_product")

if isinstance(payload, dict) and isinstance(draw, dict) and rows:
    allow_fallback = bool(
        st.session_state.get("ss_allow_current_fallback", False)
    )
    expected_matches = EXPECTED_MATCHES.get(result_product, len(rows))
    metadata = draw_metadata(draw)

    start_count = sum(row.start_values is not None for row in rows)
    missing_start_rows = [row for row in rows if row.start_values is None]
    fallback_ready_rows = [
        row
        for row in missing_start_rows
        if row.current_values is not None
    ]
    no_odds_rows = [
        row
        for row in missing_start_rows
        if row.current_values is None
    ]

    st.subheader("Hämtad kupong")

    info_cols = st.columns(5)
    info_cols[0].metric("Spelform", result_product or "–")
    info_cols[1].metric("Omgång", metadata["draw_number"] or "–")
    info_cols[2].metric("Matcher", f"{len(rows)} / {expected_matches}")
    info_cols[3].metric("Startodds hittade", f"{start_count} / {len(rows)}")
    info_cols[4].metric(
        "Aktuell fallback möjlig",
        f"{len(fallback_ready_rows)} / {len(missing_start_rows)}",
    )

    if metadata["close_time"]:
        st.caption(f"Spelstopp enligt API: {metadata['close_time']}")

    table_rows: list[dict[str, Any]] = []
    for row in rows:
        so1, sox, so2 = fmt_odds(row.start_odds)
        sv1, svx, sv2 = fmt_values(row.start_values)
        co1, cox, co2 = fmt_odds(row.current_odds)
        cv1, cvx, cv2 = fmt_values(row.current_values)

        table_rows.append(
            {
                "#": row.event_number,
                "Match": row.description,
                "Startodds 1": so1,
                "Startodds X": sox,
                "Startodds 2": so2,
                "Startvärde 1": sv1,
                "Startvärde X": svx,
                "Startvärde 2": sv2,
                "Aktuellt odds 1": co1,
                "Aktuellt odds X": cox,
                "Aktuellt odds 2": co2,
                "Aktuellt värde 1": cv1,
                "Aktuellt värde X": cvx,
                "Aktuellt värde 2": cv2,
                "Vald källa": source_for_row(row, allow_fallback),
                "Startkontroll": row.comparison,
            }
        )

    st.dataframe(
        pd.DataFrame(table_rows),
        hide_index=True,
        use_container_width=True,
        height=min(620, 92 + len(table_rows) * 35),
    )

    if len(rows) != expected_matches:
        st.error(
            f"API:t gav {len(rows)} matcher men {result_product} förväntas ha "
            f"{expected_matches}. Kupongen ska inte skickas vidare till analysen."
        )

    if missing_start_rows and not allow_fallback:
        missing_numbers = ", ".join(
            f"M{row.event_number}" for row in missing_start_rows
        )
        st.warning(
            f"Startodds saknas för {len(missing_start_rows)} matcher: "
            f"{missing_numbers}. Inga aktuella odds används automatiskt."
        )

        if fallback_ready_rows:
            ready_numbers = ", ".join(
                f"M{row.event_number}" for row in fallback_ready_rows
            )
            st.write(
                f"Aktuella odds finns för: **{ready_numbers}**. "
                "Vill du använda dem endast för dessa matcher?"
            )
            yes_col, no_col = st.columns(2)
            with yes_col:
                if st.button(
                    "Ja, använd aktuella odds för saknade matcher",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state["ss_allow_current_fallback"] = True
                    st.rerun()
            with no_col:
                if st.button(
                    "Nej, behåll kupongen ofullständig",
                    use_container_width=True,
                ):
                    st.info(
                        "Kupongen lämnas ofullständig. Du kan hämta igen senare "
                        "eller mata in värden manuellt i huvudappen."
                    )

    if allow_fallback and missing_start_rows:
        used_fallback = [
            row for row in missing_start_rows if row.current_values is not None
        ]
        used_numbers = ", ".join(
            f"M{row.event_number}" for row in used_fallback
        )
        st.warning(
            f"Fallback är godkänd. Aktuella odds används för: {used_numbers}. "
            "Övriga matcher använder fortfarande startodds."
        )
        if st.button("Ångra fallback", use_container_width=False):
            st.session_state["ss_allow_current_fallback"] = False
            st.rerun()

    if no_odds_rows:
        no_odds_numbers = ", ".join(
            f"M{row.event_number}" for row in no_odds_rows
        )
        st.error(
            f"Varken startodds eller aktuella odds finns för: {no_odds_numbers}. "
            "Kupongen kan inte bli komplett via API:t."
        )

    vector, sources, unresolved = build_coupon_vector(
        rows,
        allow_current_fallback=allow_fallback,
    )

    expected_values = expected_matches * 3
    complete = (
        len(rows) == expected_matches
        and len(vector) == expected_values
        and not unresolved
    )

    st.subheader("Resultat för Databastipset")

    if complete:
        start_source_count = sum(source == "Startodds" for source in sources)
        current_source_count = sum(
            source == "Aktuella odds (fallback)" for source in sources
        )
        st.success(
            f"{len(rows)} matcher hittade · {len(vector)} värden skapade · "
            f"alla matcher summerar till 100."
        )
        st.write(
            f"**Datakällor:** {start_source_count} matcher från startodds, "
            f"{current_source_count} från aktuella odds, 0 manuella."
        )

        vector_text = " ".join(str(value) for value in vector)
        st.text_area(
            f"Färdiga {len(vector)} startoddsvärden i ordningen 1-X-2",
            value=vector_text,
            height=110,
        )
        st.download_button(
            "Ladda ned värderaden som TXT",
            data=vector_text + "\n",
            file_name=(
                f"{str(result_product).lower().replace(' ', '_')}_"
                f"{metadata['draw_number'] or 'aktuell'}_startvarden.txt"
            ),
            mime="text/plain",
        )
        st.info(
            "Detta test startar ingen historikanalys och ändrar inte app.py "
            "eller motorn.py."
        )
    else:
        unresolved_text = (
            ", ".join(f"M{number}" for number in unresolved)
            if unresolved
            else "inga"
        )
        st.warning(
            f"Kupongen är ännu inte redo: {len(vector)} av {expected_values} "
            f"värden skapade. Olösta matcher: {unresolved_text}."
        )

    with st.expander("Teknisk diagnostik – inga API-nycklar visas"):
        diagnostics = raw_diagnostics(payload, draw)
        st.json(diagnostics)

        comparison_rows = [
            {
                "Match": f"M{row.event_number}",
                "API-fält för startvärde": row.start_values_origin,
                "Kontroll": row.comparison,
                "API-fält för aktuellt värde": row.current_values_origin,
            }
            for row in rows
        ]
        st.dataframe(
            pd.DataFrame(comparison_rows),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Ladda ned rått API-svar som JSON",
            data=json.dumps(payload, ensure_ascii=False, indent=2),
            file_name="svenska_spel_api_diagnostik.json",
            mime="application/json",
        )

with st.expander("Så lägger du in API-nyckeln permanent"):
    st.code(
        'SVENSKA_SPEL_ACCESSKEY = "DIN_NYCKEL_HÄR"',
        language="toml",
    )
    st.write(
        "Lägg raden i Streamlit Secrets. I ett publikt GitHub-repo ska nyckeln "
        "aldrig skrivas direkt i Python-filen eller läggas i en vanlig fil i repot."
    )
