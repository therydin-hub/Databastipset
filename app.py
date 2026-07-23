import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import html
import itertools
import bisect
import matplotlib.pyplot as plt
import time
import json
import hashlib
import math
import sys
import importlib.util
import argparse
import traceback
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Tipset AI-Analys", layout="wide", page_icon="🎯")
APP_VERSION = "v12.0dl – PM2K effektivt grundramsspann"


st.markdown("""
<style>
    .main .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {background: rgba(255,255,255,0.04); border: 1px solid rgba(128,128,128,0.22); padding: 0.75rem; border-radius: 0.9rem;}
    .tm-hero {border: 1px solid rgba(128,128,128,0.25); border-radius: 1rem; padding: 1rem 1.1rem; background: linear-gradient(135deg, rgba(60,120,255,0.10), rgba(20,20,20,0.02)); margin-bottom: 1rem;}
    .tm-step {font-size: 0.82rem; letter-spacing: .06em; text-transform: uppercase; opacity: .72; font-weight: 700; margin-bottom: .15rem;}
    .tm-title {font-size: 1.25rem; font-weight: 800; margin-bottom: .2rem;}
    .tm-muted {opacity: .72; font-size: .92rem;}
    .tm-pill {display:inline-block; padding: .16rem .50rem; border:1px solid rgba(128,128,128,.35); border-radius:999px; font-size:.82rem; margin-right:.25rem; margin-top:.2rem;}

    .v12-preview-grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:.55rem 0 .85rem 0;}
    .v12-preview-card {border:1px solid rgba(120,120,120,.24); border-radius:12px; padding:8px 11px; background:rgba(128,128,128,.04); min-height:58px;}
    .v12-preview-label {font-size:.76rem; opacity:.82; font-weight:800; margin-bottom:.18rem;}
    .v12-preview-value {font-size:1.08rem; font-weight:850; line-height:1.15; white-space:normal; overflow-wrap:anywhere; word-break:break-word;}
</style>
""", unsafe_allow_html=True)


def tm_download_button(label, data, file_name, mime, **kwargs):
    """Download-knapp som i nya Streamlit inte triggar full rerun vid klick.

    Om appen körs på en äldre Streamlit-version faller den tillbaka till vanlig
    download_button i stället för att krascha.
    """
    try:
        return st.download_button(
            label,
            data,
            file_name=file_name,
            mime=mime,
            on_click="ignore",
            **kwargs,
        )
    except Exception:
        return st.download_button(
            label,
            data,
            file_name=file_name,
            mime=mime,
            **kwargs,
        )


def _v12_debug_code(prefix, payload):
    """Skapar en kort, stabil felkod av samma felsökningsdata.

    Felkoden är inte ett hemligt id; den är bara en kompakt hash så att samma
    typ av mismatch får en lätt igenkännbar kod när användaren skickar rapport.
    """
    try:
        raw = json.dumps(_json_safe_value(payload), sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        raw = repr(payload)
    digest = hashlib.sha1(raw.encode('utf-8', errors='replace')).hexdigest()[:10].upper()
    return f"{prefix}-{digest}"


def _signature_debug_diff(expected_sig, active_sig, max_items=25):
    """Returnerar skillnad mellan paketets signatur och aktiv filtercentral.

    Signaturen innehåller aktiva filter, intervall, läge och gruppkrav. Detta är
    den viktigaste felsökningsinformationen vid paketkontroll-mismatch.
    """
    try:
        expected = set(tuple(x) for x in (expected_sig or []))
        active = set(tuple(x) for x in (active_sig or []))
        missing = sorted(expected - active, key=lambda x: str(x))[:max_items]
        extra = sorted(active - expected, key=lambda x: str(x))[:max_items]
        return {
            'saknas_i_aktiv_filtercentral': [list(x) for x in missing],
            'extra_i_aktiv_filtercentral': [list(x) for x in extra],
            'antal_saknas': int(len(expected - active)),
            'antal_extra': int(len(active - expected)),
        }
    except Exception as e:
        return {'diff_error': str(e)}


def _debug_active_filters(specs, settings, limit=80):
    rows = []
    by_key = {s.get('key'): s for s in specs or []}
    for k, v in sorted((settings or {}).items()):
        mode = v.get('mode', 'Av')
        if mode == 'Av':
            continue
        spec = by_key.get(k, {})
        interval = v.get('interval')
        rows.append({
            'key': k,
            'namn': spec.get('name', k),
            'kategori': spec.get('category', ''),
            'läge': mode,
            'intervall': _json_safe_value(interval),
        })
        if len(rows) >= limit:
            break
    return rows


def _debug_package_filters_from_snapshot(snapshot, limit=80):
    if not isinstance(snapshot, dict):
        return []
    rows = []
    for c in snapshot.get('filters', []) or []:
        rows.append({
            'key': c.get('key'),
            'namn': c.get('name', c.get('key')),
            'kategori': c.get('category', ''),
            'läge': c.get('package_mode', 'Tvingat'),
            'intervall': _json_safe_value(c.get('interval')),
            'hist_efter_steg': c.get('package_hist_after'),
        })
        if len(rows) >= limit:
            break
    return rows


def _render_copyable_error_report(title, prefix, payload, expanded=True):
    """Visar kopierbar felrapport i appen.

    Användaren kan kopiera hela rutan och skicka den direkt, vilket gör att vi
    kan felsöka utan skärmbild eller gissningar.
    """
    code = _v12_debug_code(prefix, payload)
    report_payload = _json_safe_value(payload)
    report = (
        f"FELKOD: {code}\n"
        f"APP_VERSION: {APP_VERSION}\n"
        f"TID: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"TYP: {title}\n\n"
        f"KOPIERA ALLT UNDER DENNA RAD:\n"
        f"{json.dumps(report_payload, ensure_ascii=False, indent=2, default=str)}"
    )
    st.error(f"{title} · Felkod: {code}")
    with st.expander("📋 Visa felrapport att skicka", expanded=expanded):
        st.caption("Kopiera hela texten nedan och skicka den direkt. Den innehåller bara appversion, paket-/filterdata och diagnostik för felet.")
        st.code(report, language="text")
        tm_download_button(
            "⬇️ Ladda ner felrapport",
            report,
            file_name=f"{code}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    return code


# --- v12.0bh: enkel intern profiler för att hitta flaskhalsar i Streamlit-reruns ---
def _perf_enabled():
    return bool(st.session_state.get('v12_show_perf_profile', False))

def _perf_mark(label, start_time, bucket='v12_perf_marks'):
    if not _perf_enabled():
        return time.perf_counter()
    now = time.perf_counter()
    st.session_state.setdefault(bucket, []).append({'Steg': label, 'Sekunder': round(now - start_time, 3)})
    return now

def _short_hash_items(items, limit=1200):
    """Billig signatur för listor som inte ska läggas helt i cache-nycklar."""
    try:
        items = list(items or [])
        if len(items) <= limit:
            sample = items
        else:
            step = max(1, len(items) // limit)
            sample = items[::step][:limit]
        return hash(tuple(sample))
    except Exception:
        return 0


# ==========================================
# 1. FUNKTIONER (FÖR 8 & 13 MATCHER)
# ==========================================

def parse_payout_str(val_str):
    return int(str(val_str).replace(' ', '').replace('kr', '').replace('+', ''))

def calculate_ai_matrix_from_values(values):
    matrix = [values[i:i+3] for i in range(0, len(values), 3)]
    all_scores = sorted([sum(combo) for combo in itertools.product(*matrix)], reverse=True)
    return matrix, all_scores[::-1], len(all_scores)

def get_exact_rank(row_str, matrix, scores_ascending, total_count):
    score = sum(matrix[i][0] if c == '1' else matrix[i][1] if c == 'X' else matrix[i][2] for i, c in enumerate(row_str))
    rank = total_count - bisect.bisect_left(scores_ascending, score)
    return max(rank, 1), score

def get_100_minus_sum(row_str, prob_vector):
    return sum((100 - (prob_vector[i*3] if c == '1' else prob_vector[i*3+1] if c == 'X' else prob_vector[i*3+2])) for i, c in enumerate(row_str))

def get_log_surprise_sum(row_str, prob_vector):
    """
    Skrälltryck Log Summa.
    Poäng per valt tecken = -10 * ln(procent / 100), avrundat till heltal.
    Låga procenttal får mycket hårdare poäng än i vanligt 100-minus.
    """
    total = 0
    matcher = min(len(row_str), len(prob_vector) // 3)
    for i in range(matcher):
        idx = i * 3
        if row_str[i] == '1': p = prob_vector[idx]
        elif row_str[i] == 'X': p = prob_vector[idx+1]
        elif row_str[i] == '2': p = prob_vector[idx+2]
        else: p = 0
        p = max(float(p), 0.1)
        total += round(-10 * np.log(p / 100.0))
    return int(total)

def get_rank_points(row_str, prob_vector):
    matcher = len(row_str)
    pm = [[0, 0, 0] for _ in range(matcher)]
    nf = []
    for m in range(matcher):
        mp = sorted([(prob_vector[m*3], 0), (prob_vector[m*3+1], 1), (prob_vector[m*3+2], 2)], key=lambda x: x[0], reverse=True)
        nf.extend([{'pct': mp[1][0], 'm': m, 'c': mp[1][1]}, {'pct': mp[2][0], 'm': m, 'c': mp[2][1]}])
    for rank, item in enumerate(sorted(nf, key=lambda x: x['pct'], reverse=True), 1):
        pm[item['m']][item['c']] = rank
    return sum(pm[i][0] if c == '1' else pm[i][1] if c == 'X' else pm[i][2] for i, c in enumerate(row_str))

def get_fat(row_str, prob_vector):
    f, a, t, fs = 0, 0, 0, 0
    for i, char in enumerate(row_str):
        idx = i * 3
        ranked = sorted([('1', prob_vector[idx]), ('X', prob_vector[idx+1]), ('2', prob_vector[idx+2])], key=lambda x: x[1], reverse=True)
        if char == ranked[0][0]: f += 1; fs += 1
        elif char == ranked[1][0]: a += 1; fs += 2
        else: t += 1; fs += 3
    return f, a, t, fs

def get_fat_zone_indices(prob_vector, antal_matcher=None):
    """Delar kupongen i tre svårighetszoner efter favoritens streck/procent.

    Zon A = största favoriterna, Zon B = mittenmatcherna, Zon C = mest öppna/svåra.
    För 13 matcher används 4/5/4, vilket matchar Stryktipset/Europatipset-logiken.
    För 8 matcher används 3/3/2. Övriga storlekar delas så jämnt som möjligt.
    """
    try:
        n = int(antal_matcher or (len(prob_vector) // 3))
    except Exception:
        n = len(prob_vector) // 3 if prob_vector is not None else 0
    n = max(0, min(n, len(prob_vector or []) // 3))
    if n <= 0:
        return {'A': [], 'B': [], 'C': []}

    fav_order = []
    for m in range(n):
        vals = [float(x) for x in (prob_vector[m*3:m*3+3] or [0, 0, 0])]
        fav_order.append((max(vals), m))
    fav_order.sort(key=lambda x: x[0], reverse=True)
    ordered_idx = [m for _, m in fav_order]

    if n >= 12:
        a, b = 4, 5
    elif n >= 8:
        a, b = 3, 3
    else:
        a = max(1, int(np.ceil(n / 3.0)))
        b = max(1, int(np.ceil((n - a) / 2.0))) if n - a > 0 else 0
    a = min(a, n)
    b = min(b, max(0, n - a))
    return {
        'A': ordered_idx[:a],
        'B': ordered_idx[a:a+b],
        'C': ordered_idx[a+b:],
    }


def get_fat_rank_for_match(row_str, prob_vector, match_idx):
    """FAT-rank för valt tecken i en match: favorit=1, andratecken=2, tredjetecken=3."""
    try:
        c = str(row_str)[int(match_idx)]
        idx = int(match_idx) * 3
        ranked = sorted([('1', prob_vector[idx]), ('X', prob_vector[idx+1]), ('2', prob_vector[idx+2])], key=lambda x: x[1], reverse=True)
        for rank, (sign, _) in enumerate(ranked, start=1):
            if c == sign:
                return int(rank)
    except Exception:
        pass
    return 3


def get_fat_zone_points(row_str, prob_vector, zone='A', antal_matcher=None):
    """FAT-poäng per zon. Låg poäng = favorittyngd, hög poäng = mer andra-/tredjetecken."""
    zones = get_fat_zone_indices(prob_vector, antal_matcher=antal_matcher)
    idxs = zones.get(str(zone).upper(), [])
    return int(sum(get_fat_rank_for_match(row_str, prob_vector, m) for m in idxs))


def get_fat_step_counts(row_str, prob_vector, antal_matcher=None):
    """Stigande/fallande FAT-poängsteg över matchordningen.

    Poängtabell: favorit=1, andratecken=2, tredjetecken=3 per match.
    Räknar övergångar mellan intilliggande matcher:
    1→2 eller 2→3 = stigande 1 steg, 1→3 = stigande 2 steg,
    3→2 eller 2→1 = fallande 1 steg, 3→1 = fallande 2 steg.
    Lika värde räknas inte.
    """
    row_str = normalize_single_row_text(row_str) if 'normalize_single_row_text' in globals() else str(row_str or '').strip().upper()
    n = int(antal_matcher or min(len(row_str), len(prob_vector or []) // 3))
    points = [get_fat_rank_for_match(row_str, prob_vector, m) for m in range(min(n, len(row_str)))]
    rise1 = rise2 = fall1 = fall2 = 0
    for a, b in zip(points, points[1:]):
        diff = int(b) - int(a)
        if diff == 1:
            rise1 += 1
        elif diff == 2:
            rise2 += 1
        elif diff == -1:
            fall1 += 1
        elif diff == -2:
            fall2 += 1
    return int(rise1), int(rise2), int(fall1), int(fall2)


def get_fat_point_sequence(row_str, prob_vector, antal_matcher=None):
    """Returnerar radens FAT-poängföljd, t.ex. 1-1-2-3-1."""
    row_str = normalize_single_row_text(row_str) if 'normalize_single_row_text' in globals() else str(row_str or '').strip().upper()
    n = int(antal_matcher or min(len(row_str), len(prob_vector or []) // 3))
    return [get_fat_rank_for_match(row_str, prob_vector, m) for m in range(min(n, len(row_str)))]


def describe_fat_zone(prob_vector, zone='A', antal_matcher=None):
    """Visar vilka matcher som ingår i zonen i aktuell kupong."""
    zones = get_fat_zone_indices(prob_vector, antal_matcher=antal_matcher)
    idxs = zones.get(str(zone).upper(), [])
    if not idxs:
        return "-"
    return ", ".join([f"M{i+1}" for i in idxs])


def get_abc_class_map(prob_vector, antal_matcher=None):
    """Global ABC-klass för alla tecken på kupongen.

    Alla 3*matcher tecken rankas efter streck/procent. För 13 matcher blir
    global rank 1-13 = klass 1, 14-26 = klass 2, 27-39 = klass 3.
    Detta skiljer sig från FAT, som rankar tecknen inom varje match.
    """
    prob_vector = list(prob_vector or [])
    n = int(antal_matcher or (len(prob_vector) // 3))
    signs = ['1', 'X', '2']
    flat = []
    for m in range(n):
        idx = m * 3
        for si, sign in enumerate(signs):
            pct = float(prob_vector[idx + si]) if idx + si < len(prob_vector) else 0.0
            # Stabil sortering vid lika streck: högre procent först, därefter matchordning, därefter 1-X-2.
            flat.append({'pct': pct, 'match': m, 'sign': sign, 'sign_idx': si})
    flat.sort(key=lambda x: (-x['pct'], x['match'], x['sign_idx']))
    class_map = {}
    for rank, item in enumerate(flat, start=1):
        if rank <= n:
            cls = 1
        elif rank <= 2 * n:
            cls = 2
        else:
            cls = 3
        class_map[(int(item['match']), item['sign'])] = cls
    return class_map


def get_abc_counts(row_str, prob_vector, class_map=None, antal_matcher=None):
    """ABC-räkning för en enkelrad: antal valda tecken i global klass 1/2/3 och ABC-summa."""
    row_str = normalize_single_row_text(row_str) if 'normalize_single_row_text' in globals() else str(row_str or '').strip().upper()
    prob_vector = list(prob_vector or [])
    n = int(antal_matcher or min(len(row_str), len(prob_vector) // 3))
    if class_map is None:
        class_map = get_abc_class_map(prob_vector, n)
    counts = {1: 0, 2: 0, 3: 0}
    for m, sign in enumerate(row_str[:n]):
        cls = int(class_map.get((m, sign), 3))
        counts[cls] = counts.get(cls, 0) + 1
    abc_sum = counts.get(1, 0) + counts.get(2, 0) * 2 + counts.get(3, 0) * 3
    return counts.get(1, 0), counts.get(2, 0), counts.get(3, 0), int(abc_sum)


def describe_abc_match_patterns(prob_vector, antal_matcher=None):
    """Visar klassningen per match i ordningen 1-X-2, t.ex. M4:1-3-3."""
    prob_vector = list(prob_vector or [])
    n = int(antal_matcher or (len(prob_vector) // 3))
    cmap = get_abc_class_map(prob_vector, n)
    parts = []
    for m in range(n):
        parts.append(f"M{m+1}:{cmap.get((m, '1'), 3)}-{cmap.get((m, 'X'), 3)}-{cmap.get((m, '2'), 3)}")
    return " | ".join(parts)

def get_sft_sum(row_str, prob_vector):
    return sum(prob_vector[i*3] if c == '1' else prob_vector[i*3+1] if c == 'X' else prob_vector[i*3+2] for i, c in enumerate(row_str))

def get_occurrences(row_str):
    if not row_str: return 0,0,0,0,0
    u1, ux, u2 = int(row_str[0] == '1'), int(row_str[0] == 'X'), int(row_str[0] == '2')
    for i in range(1, len(row_str)):
        if row_str[i] != row_str[i-1]:
            if row_str[i] == '1': u1 += 1
            elif row_str[i] == 'X': ux += 1
            else: u2 += 1
    return u1, ux, u2, u1+ux+u2, max(u1, ux, u2)

def get_triplets(row_str):
    t1, tx, t2, i = 0, 0, 0, 0
    while i < len(row_str):
        char, count = row_str[i], 1
        while i + 1 < len(row_str) and row_str[i+1] == char: count += 1; i += 1
        if count == 3:
            if char == '1': t1 += 1
            elif char == 'X': tx += 1
            else: t2 += 1
        i += 1
    return t1, tx, t2, t1+tx+t2, max(t1, tx, t2)

def get_doublets(row_str):
    d1, dx, d2, i = 0, 0, 0, 0
    while i < len(row_str):
        char, count = row_str[i], 1
        while i + 1 < len(row_str) and row_str[i+1] == char: count += 1; i += 1
        if count == 2:
            if char == '1': d1 += 1
            elif char == 'X': dx += 1
            else: d2 += 1
        i += 1
    return d1, dx, d2, d1+dx+d2, max(d1, dx, d2)

def get_singles(row_str):
    s1, sx, s2 = 0, 0, 0
    for i in range(len(row_str)):
        char = row_str[i]
        if (i == 0 or row_str[i-1] != char) and (i == len(row_str)-1 or row_str[i+1] != char):
            if char == '1': s1 += 1
            elif char == 'X': sx += 1
            else: s2 += 1
    return s1, sx, s2, s1+sx+s2, max(s1, sx, s2)

def get_gaps(row_str):
    g1 = len(max(row_str.split('1'), key=len)) if '1' in row_str else len(row_str)
    gx = len(max(row_str.split('X'), key=len)) if 'X' in row_str else len(row_str)
    g2 = len(max(row_str.split('2'), key=len)) if '2' in row_str else len(row_str)
    return g1, gx, g2, max(g1, gx, g2)

def get_streaks(row_str):
    m1, mx, m2, c, curr = 0, 0, 0, 0, ''
    for char in row_str:
        if char == curr: c += 1
        else: c, curr = 1, char
        if curr == '1': m1 = max(m1, c)
        elif curr == 'X': mx = max(mx, c)
        else: m2 = max(m2, c)
    return m1, mx, m2, max(m1, mx, m2)

def get_best_interval(values, target_coverage_percent):
    n = len(values)
    if n == 0: return (0, 0)
    values = sorted(values)
    min_items = int(np.ceil(n * (target_coverage_percent / 100)))
    if min_items == 0: return (min(values), max(values))
    best_diff = float('inf')
    best_int = (values[0], values[-1])
    for i in range(n - min_items + 1):
        j = i + min_items - 1
        if values[j] - values[i] < best_diff:
            best_diff = values[j] - values[i]
            best_int = (values[i], values[j])
    return best_int

def interval_width(interval):
    if interval is None:
        return 0
    try:
        return float(interval[1]) - float(interval[0])
    except Exception:
        return 0

def pct_values_in_interval(values, interval):
    clean = [v for v in values if not pd.isna(v)]
    if not clean:
        return 0.0, 0, 0
    hits = sum(1 for v in clean if in_range(v, interval))
    misses = len(clean) - hits
    return (hits / len(clean)) * 100, hits, misses

def candidate_keep_pct(candidate_rows, cand_getter, interval):
    if not candidate_rows or cand_getter is None:
        return None
    hits = 0
    total = 0
    for tr in candidate_rows:
        try:
            val = cand_getter(tr)
            if pd.isna(val):
                continue
            total += 1
            if in_range(val, interval):
                hits += 1
        except Exception:
            continue
    if total == 0:
        return None
    return (hits / total) * 100

def choose_rational_interval(hist_values, candidate_rows, cand_getter, target_coverage_percent=95, min_hist_pct=85, coverage_steps=None):
    """
    AutoTrim: testar flera intervallnivåer och väljer intervallet som ger bäst rationell effekt.
    Säkerhetsintervallet är fortfarande intervallet för användarens valda kärnprocent.
    """
    clean = [v for v in hist_values if not pd.isna(v)]
    if not clean:
        return (0, 0), {
            'AutoTrim': False, 'Säkerhetsintervall': (0, 0), 'Historisk träff %': 0.0,
            'Kvar rad %': None, 'Rationell faktor': None, 'Kapade outliers': 0, 'Testade intervall': 0
        }

    safety_interval = get_best_interval(clean, target_coverage_percent)
    safety_hist_pct, _, safety_misses = pct_values_in_interval(clean, safety_interval)
    safety_keep = candidate_keep_pct(candidate_rows, cand_getter, safety_interval)
    safety_lift = None if safety_keep is None else safety_hist_pct - safety_keep

    if coverage_steps is None:
        low = int(max(50, np.floor(float(min_hist_pct) / 5.0) * 5))
        high = int(max(low, np.ceil(float(target_coverage_percent) / 5.0) * 5))
        coverage_steps = list(range(low, min(100, high) + 1, 5))
        coverage_steps.append(float(target_coverage_percent))
        coverage_steps = sorted(set([float(x) for x in coverage_steps if 0 <= float(x) <= 100]))

    candidates = []
    for cov in coverage_steps:
        interval = get_best_interval(clean, cov)
        hist_pct, hist_hits, hist_misses = pct_values_in_interval(clean, interval)
        if hist_pct + 1e-9 < float(min_hist_pct):
            continue
        keep = candidate_keep_pct(candidate_rows, cand_getter, interval)
        if keep is None:
            keep = 100.0
        lift = hist_pct - keep
        reduction = 100.0 - keep
        width = interval_width(interval)
        candidates.append({
            'coverage': cov,
            'interval': interval,
            'hist_pct': hist_pct,
            'hist_hits': hist_hits,
            'hist_misses': hist_misses,
            'keep_pct': keep,
            'lift': lift,
            'reduction': reduction,
            'width': width,
        })

    if not candidates:
        chosen = {
            'coverage': float(target_coverage_percent),
            'interval': safety_interval,
            'hist_pct': safety_hist_pct,
            'hist_hits': len(clean) - safety_misses,
            'hist_misses': safety_misses,
            'keep_pct': 100.0 if safety_keep is None else safety_keep,
            'lift': None if safety_lift is None else safety_lift,
            'reduction': None if safety_keep is None else 100.0 - safety_keep,
            'width': interval_width(safety_interval),
        }
    else:
        # Rationell effekt först. Vid jämnt läge: välj mer reducering, därefter högre historisk träff.
        chosen = max(candidates, key=lambda r: (r['lift'], r['reduction'], r['hist_pct'], -r['width']))

    meta = {
        'AutoTrim': True,
        'Vald tighthet %': round(float(chosen['coverage']), 1),
        'Säkerhetsintervall': safety_interval,
        'Historisk träff %': round(float(chosen['hist_pct']), 1),
        'Kvar rad %': round(float(chosen['keep_pct']), 1) if chosen.get('keep_pct') is not None else None,
        'Reducerar %': round(float(chosen['reduction']), 1) if chosen.get('reduction') is not None else None,
        'Rationell faktor': round(float(chosen['lift']), 1) if chosen.get('lift') is not None else None,
        'Kapade outliers': int(chosen['hist_misses']),
        'Säkerhet historisk träff %': round(float(safety_hist_pct), 1),
        'Säkerhet kvar rad %': round(float(safety_keep), 1) if safety_keep is not None else None,
        'Säkerhet RF': round(float(safety_lift), 1) if safety_lift is not None else None,
        'Testade intervall': len(candidates),
    }
    return chosen['interval'], meta

def autotrim_caption(meta, decimals=0):
    if not meta or not meta.get('AutoTrim'):
        return ""
    safe = meta.get('Säkerhetsintervall')
    safe_txt = fmt_interval(safe, decimals) if safe is not None else '-'
    keep = meta.get('Kvar rad %')
    rf = meta.get('Rationell faktor')
    outliers = meta.get('Kapade outliers', 0)
    tight = meta.get('Vald tighthet %', '-')
    keep_txt = '-' if keep is None else f"{keep:.1f}%"
    rf_txt = '-' if rf is None else f"{rf:+.1f}"
    return f"AutoTrim {tight}% | säkerhet {safe_txt} | hist {meta.get('Historisk träff %', 0):.1f}% | kvar {keep_txt} | RF {rf_txt} | kapar {outliers}"

def parse_input_string(text, max_vals):
    # Tillåt svensk decimal-komma och klistra in från tabeller med %, kr, semikolon osv.
    cleaned = str(text).replace(',', '.')
    tokens = re.findall(r'-?\d+(?:\.\d+)?', cleaned)
    return [float(p) for p in tokens[:max_vals]]

def get_structural_vector(vec):
    matches = [sorted(vec[i:i+3], reverse=True) for i in range(0, len(vec), 3)]
    matches_sorted = sorted(matches, key=lambda m: m[0], reverse=True)
    return [val for m in matches_sorted for val in m]

def get_rank_sum(row_str, prob_vector):
    sorted_probs = sorted(prob_vector, reverse=True)
    rank_dict = {}
    for i, p in enumerate(sorted_probs):
        if p not in rank_dict:
            count = sorted_probs.count(p)
            avg_rank = sum(range(i + 1, i + count + 1)) / count
            rank_dict[p] = avg_rank
    total_rank = 0
    for i, char in enumerate(row_str):
        idx = i * 3
        if char == '1': p = prob_vector[idx]
        elif char == 'X': p = prob_vector[idx+1]
        elif char == '2': p = prob_vector[idx+2]
        else: p = 0
        total_rank += rank_dict.get(p, len(prob_vector))
    return total_rank

def weighted_euclidean(u, v, w):
    return np.sqrt(np.sum(w * (np.array(u) - np.array(v))**2))

def get_top_n_favs_wins(row_str, prob_vector, top_n):
    match_favs = []
    matcher = len(row_str)
    for m in range(matcher):
        idx = m * 3
        probs = [(prob_vector[idx], '1'), (prob_vector[idx+1], 'X'), (prob_vector[idx+2], '2')]
        probs.sort(key=lambda x: x[0], reverse=True)
        match_favs.append({'pct': probs[0][0], 'sign': probs[0][1], 'match_idx': m})
    match_favs.sort(key=lambda x: x['pct'], reverse=True)
    top_n_list = match_favs[:top_n]
    return sum(1 for fav in top_n_list if row_str[fav['match_idx']] == fav['sign'])

def calculate_total_diff(match_odds, correct_results):
    col_map = {'1': 0, 'X': 1, '2': 2}
    matcher = len(correct_results)
    
    flat_t1 = []
    for match_idx, odds in enumerate(match_odds):
        max_val = max(odds)
        max_idx = odds.index(max_val)
        for col_idx, val in enumerate(odds):
            is_max = (col_idx == max_idx)
            flat_t1.append({"match_idx": match_idx, "col_idx": col_idx, "val": 0 if is_max else val, "is_max": is_max})
            
    non_zeros_t1 = sorted([x for x in flat_t1 if not x["is_max"]], key=lambda x: x["val"], reverse=True)
    for rank, item in enumerate(non_zeros_t1, start=1): item["rank"] = rank
    for item in flat_t1: 
        if item["is_max"]: item["rank"] = 0
            
    t1_table = [[0]*3 for _ in range(matcher)]
    for item in flat_t1: t1_table[item["match_idx"]][item["col_idx"]] = item["rank"]

    flat_t2 = []
    for match_idx, odds in enumerate(match_odds):
        min_val = min(odds)
        min_idx = odds.index(min_val)
        for col_idx, val in enumerate(odds):
            is_min = (col_idx == min_idx)
            flat_t2.append({"match_idx": match_idx, "col_idx": col_idx, "val": 0 if is_min else val, "is_min": is_min})
            
    non_zeros_t2 = sorted([x for x in flat_t2 if not x["is_min"]], key=lambda x: x["val"], reverse=True)
    for rank, item in enumerate(non_zeros_t2, start=1): item["rank"] = rank
    for item in flat_t2: 
        if item["is_min"]: item["rank"] = 0
            
    t2_table = [[0]*3 for _ in range(matcher)]
    for item in flat_t2: t2_table[item["match_idx"]][item["col_idx"]] = item["rank"]

    total_diff = 0
    for i in range(matcher):
        correct_sign = str(correct_results[i]).strip().upper()
        if correct_sign not in col_map: continue
        col = col_map[correct_sign]
        total_diff += (t1_table[i][col] - t2_table[i][col])
        
    return total_diff

def calculate_delta(row_str, prob_vector):
    if not row_str or len(prob_vector) != len(row_str) * 3: return 0
    delta_sum = 0
    for i, char in enumerate(row_str):
        idx = i * 3
        probs = {'1': prob_vector[idx], 'X': prob_vector[idx+1], '2': prob_vector[idx+2]}
        fav_pct = max(probs.values())
        win_pct = probs.get(char, 0)
        delta_sum += (fav_pct - win_pct)
    return round(delta_sum, 1)



def get_favorite_threshold_counts(prob_vector, thresholds=(70, 60, 50)):
    """Antal matcher där favoritens procent ligger över respektive gräns."""
    counts = {int(t): 0 for t in thresholds}
    matcher = len(prob_vector) // 3
    for m in range(matcher):
        idx = m * 3
        fav_pct = max(prob_vector[idx:idx+3])
        for t in thresholds:
            if fav_pct >= t:
                counts[int(t)] += 1
    return counts


def get_favorite_pressure(row_str, prob_vector, thresholds=(70, 60, 50)):
    """
    Favorittryck: hur många favoriter över 70/60/50 % som fanns,
    och hur många av dessa som vann på raden.
    """
    counts = get_favorite_threshold_counts(prob_vector, thresholds)
    wins = {int(t): 0 for t in thresholds}
    matcher = min(len(row_str), len(prob_vector) // 3)
    for m in range(matcher):
        idx = m * 3
        probs = [('1', prob_vector[idx]), ('X', prob_vector[idx+1]), ('2', prob_vector[idx+2])]
        fav_sign, fav_pct = max(probs, key=lambda x: x[1])
        for t in thresholds:
            if fav_pct >= t and row_str[m] == fav_sign:
                wins[int(t)] += 1
    return {
        'F70_Count': counts.get(70, 0), 'F70_Wins': wins.get(70, 0),
        'F60_Count': counts.get(60, 0), 'F60_Wins': wins.get(60, 0),
        'F50_Count': counts.get(50, 0), 'F50_Wins': wins.get(50, 0),
    }


def get_shock_capacity(prob_vector, thresholds=(10, 15, 20)):
    """Max antal möjliga skrällträffar per gräns på dagens kupong, räknat max 1 per match."""
    counts = {int(t): 0 for t in thresholds}
    matcher = len(prob_vector) // 3
    for m in range(matcher):
        vals = prob_vector[m*3:m*3+3]
        for t in thresholds:
            if any(v < t for v in vals):
                counts[int(t)] += 1
    return counts


def get_favorite_threshold_details(prob_vector, threshold):
    """Favorittecken på kupongen där favoritens streck är minst threshold."""
    out = []
    signs = ['1', 'X', '2']
    matcher = len(prob_vector) // 3
    for m in range(matcher):
        vals = [float(x) for x in prob_vector[m*3:m*3+3]]
        if not vals:
            continue
        fav_idx = int(np.argmax(vals))
        fav_pct = vals[fav_idx]
        if fav_pct >= float(threshold):
            out.append({'match': m + 1, 'sign': signs[fav_idx], 'pct': fav_pct})
    return out


def get_shock_sign_details(prob_vector, threshold):
    """Alla tecken på kupongen under given streckgräns."""
    out = []
    signs = ['1', 'X', '2']
    matcher = len(prob_vector) // 3
    for m in range(matcher):
        vals = [float(x) for x in prob_vector[m*3:m*3+3]]
        for idx, pct in enumerate(vals):
            if pct < float(threshold):
                out.append({'match': m + 1, 'sign': signs[idx], 'pct': pct})
    return out


def _details_covered_by_frame(details, frame):
    """Filtrerar detail-lista till de tecken som faktiskt finns i manuell grundram."""
    if not frame:
        return list(details or [])
    covered = []
    for item in details or []:
        try:
            mi = int(item.get('match', 0)) - 1
            sign = str(item.get('sign', '')).upper()
            if 0 <= mi < len(frame) and sign in set(normalize_signs(frame[mi])):
                covered.append(item)
        except Exception:
            continue
    return covered


def _format_match_sign_details(details, max_items=8):
    parts = []
    for item in (details or [])[:int(max_items)]:
        try:
            parts.append(f"M{int(item.get('match'))}:{item.get('sign')} {float(item.get('pct')):.0f}%")
        except Exception:
            pass
    if len(details or []) > int(max_items):
        parts.append(f"+{len(details)-int(max_items)} till")
    return ", ".join(parts)


def get_shock_strength(row_str, prob_vector):
    """
    Skrällstyrka: hur många vinnande tecken som låg under 10/15/20 %,
    plus lägsta vinnande procent och summa på vinnande tecken under 20 %.
    """
    win_pcts = []
    matcher = min(len(row_str), len(prob_vector) // 3)
    for i in range(matcher):
        idx = i * 3
        if row_str[i] == '1': p = prob_vector[idx]
        elif row_str[i] == 'X': p = prob_vector[idx+1]
        elif row_str[i] == '2': p = prob_vector[idx+2]
        else: p = 0
        win_pcts.append(float(p))
    if not win_pcts:
        return {'U10_Wins': 0, 'U15_Wins': 0, 'U20_Wins': 0, 'Lowest_Win_Pct': 0.0, 'Shock_Sum_U20': 0.0}
    return {
        'U10_Wins': sum(1 for p in win_pcts if p < 10),
        'U15_Wins': sum(1 for p in win_pcts if p < 15),
        'U20_Wins': sum(1 for p in win_pcts if p < 20),
        'Lowest_Win_Pct': round(min(win_pcts), 1),
        'Shock_Sum_U20': round(sum(p for p in win_pcts if p < 20), 1),
    }


def get_favorite_delta(row_str, prob_vector):
    """
    Favorit-delta = faktiska favoritvinster - förväntade favoritvinster.
    Tar hänsyn till om kupongen hade stora eller små favoriter från början.
    """
    expected = 0.0
    actual = 0
    matcher = min(len(row_str), len(prob_vector) // 3)
    for m in range(matcher):
        idx = m * 3
        probs = [('1', prob_vector[idx]), ('X', prob_vector[idx+1]), ('2', prob_vector[idx+2])]
        fav_sign, fav_pct = max(probs, key=lambda x: x[1])
        expected += fav_pct / 100.0
        if row_str[m] == fav_sign:
            actual += 1
    return round(actual - expected, 2)


def clamp_interval(interval, low, high):
    """Klipper ett historiskt intervall så det är möjligt på dagens kupong."""
    a, b = interval
    a = max(low, min(a, high))
    b = max(low, min(b, high))
    if a > b:
        a = b
    return (a, b)

def get_stat_strings(hits, max_items):
    if not hits or max_items == 0: return "Kan inte beräkna."
    tot = len(hits)
    lines = [f"📊 **Historiskt utfall:** Min {min(hits)} | Max {max(hits)}"]
    for i in range(max_items + 1):
        pct = (hits.count(i) / tot) * 100
        lines.append(f"**Exakt {i} sitter:** {pct:.1f} %")
    return "\n".join(lines)

def get_compact_stat_strings(title, hits):
    if not hits: return f"**{title}:** Kan inte beräkna."
    tot = len(hits)
    lines = [f"📊 **{title}:** Min {min(hits)} | Max {max(hits)}"]
    stats = []
    for i in range(min(hits), max(hits) + 1):
        pct = (hits.count(i) / tot) * 100
        if pct > 0:
            stats.append(f"**{i} st:** {pct:.1f}%")
    lines.append(" | ".join(stats))
    return "\n\n".join(lines)

# --- NY FUNKTION: HITTAR LÄNGSTA BLOCK ---
def get_longest_subset_streak(row_str, allowed_chars):
    max_streak = 0
    current_streak = 0
    for char in row_str:
        if char in allowed_chars:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak

def get_fat_string(row_str, prob_vector):
    fat_str = ""
    for i, char in enumerate(row_str):
        idx = i * 3
        ranked = sorted([('1', prob_vector[idx]), ('X', prob_vector[idx+1]), ('2', prob_vector[idx+2])], key=lambda x: x[1], reverse=True)
        if char == ranked[0][0]: fat_str += '1'
        elif char == ranked[1][0]: fat_str += '2'
        else: fat_str += '3'
    return fat_str

def count_fat_sequence_hits(fat_str, sequences):
    """Antal valda FAT-sekvenser som förekommer minst en gång i FAT-strängen."""
    if not fat_str or not sequences:
        return 0
    return sum(1 for seq in sequences if seq in fat_str)


def count_fat_pair_hits(fat_str, pairs):
    """Antal sekvenspar där minst en av de två sekvenserna förekommer."""
    if not fat_str or not pairs:
        return 0
    return sum(1 for s1, s2 in pairs if s1 in fat_str or s2 in fat_str)


def choose_fat_sequences(fat_strings, base_fat_strings, length=2, top_n=5, min_lift=0.0):
    """
    Väljer FAT-sekvenser med rationell inriktning: hög träff i liknande omgångar,
    positiv lift mot hela databasen och fallback till ren träff om för få finns.
    """
    fat_strings = [str(x) for x in fat_strings if isinstance(x, str) and x]
    base_fat_strings = [str(x) for x in base_fat_strings if isinstance(x, str) and x]
    if not fat_strings:
        return []
    seqs = [''.join(p) for p in itertools.product('123', repeat=length)]
    rows = []
    total = len(fat_strings)
    base_total = len(base_fat_strings)
    for seq in seqs:
        count = sum(1 for r in fat_strings if seq in r)
        hist_pct = (count / total) * 100 if total else 0.0
        base_pct = (sum(1 for r in base_fat_strings if seq in r) / base_total) * 100 if base_total else 0.0
        lift = hist_pct - base_pct
        # Score: träff krävs, men positiv lift premieras kraftigt.
        score = hist_pct + max(lift, 0) * 1.75
        rows.append({
            'Sekvens': seq,
            'Antal': count,
            'Träff %': round(hist_pct, 1),
            'Bas %': round(base_pct, 1),
            'Lift': round(lift, 1),
            'Score': round(score, 3),
        })
    positive = [r for r in rows if r['Lift'] >= min_lift and r['Antal'] > 0]
    selected = sorted(positive, key=lambda r: (r['Score'], r['Träff %'], r['Lift']), reverse=True)[:top_n]
    if len(selected) < top_n:
        used = {r['Sekvens'] for r in selected}
        fallback = [r for r in sorted(rows, key=lambda r: (r['Träff %'], r['Lift']), reverse=True) if r['Sekvens'] not in used]
        selected += fallback[:top_n - len(selected)]
    return selected


def choose_fat_sequence_pairs(fat_strings, base_fat_strings, top_n=5, min_lift=0.0):
    """Väljer par av FAT-3-sekvenser där minst en av två ska finnas."""
    fat_strings = [str(x) for x in fat_strings if isinstance(x, str) and x]
    base_fat_strings = [str(x) for x in base_fat_strings if isinstance(x, str) and x]
    if not fat_strings:
        return []
    seqs3 = [''.join(p) for p in itertools.product('123', repeat=3)]
    total = len(fat_strings)
    base_total = len(base_fat_strings)
    rows = []
    for s1, s2 in itertools.combinations(seqs3, 2):
        count = sum(1 for r in fat_strings if s1 in r or s2 in r)
        hist_pct = (count / total) * 100 if total else 0.0
        base_pct = (sum(1 for r in base_fat_strings if s1 in r or s2 in r) / base_total) * 100 if base_total else 0.0
        lift = hist_pct - base_pct
        score = hist_pct + max(lift, 0) * 1.75
        rows.append({
            'Par': (s1, s2),
            'Visning': f"{s1}/{s2}",
            'Antal': count,
            'Träff %': round(hist_pct, 1),
            'Bas %': round(base_pct, 1),
            'Lift': round(lift, 1),
            'Score': round(score, 3),
        })
    positive = [r for r in rows if r['Lift'] >= min_lift and r['Antal'] > 0]
    selected = sorted(positive, key=lambda r: (r['Score'], r['Träff %'], r['Lift']), reverse=True)[:top_n]
    if len(selected) < top_n:
        used = {r['Par'] for r in selected}
        fallback = [r for r in sorted(rows, key=lambda r: (r['Träff %'], r['Lift']), reverse=True) if r['Par'] not in used]
        selected += fallback[:top_n - len(selected)]
    return selected


def fmt_interval(interval, decimals=0):
    if interval is None: return "-"
    a, b = interval
    if decimals == 0:
        return f"{float(a):.0f}-{float(b):.0f}"
    return f"{float(a):.{decimals}f}-{float(b):.{decimals}f}"

def in_range(value, interval):
    return interval[0] <= value <= interval[1]

def make_candidate_rows(antal_matcher, sample_size=25000):
    if antal_matcher == 8:
        return [''.join(tup) for tup in itertools.product(['1','X','2'], repeat=8)], True
    rng = np.random.default_rng(42)
    arr = rng.choice(['1', 'X', '2'], size=(sample_size, antal_matcher))
    return [''.join(row) for row in arr], False


# --- TIPSETMATRIX 12: EXAKT GRUNDRAM, 12-GARANTI OCH GARANTITABELL ---
def normalize_signs(signs):
    """Normaliserar valda tecken till Helgardering-ordning: 1, X, 2.

    Tomt val returnerar tom lista. Det är viktigt i manuell grundram:
    en match utan tecken ska ge felmeddelande, inte tyst bli 1X2.
    """
    if signs is None:
        return []
    if isinstance(signs, str):
        raw = list(signs.upper().replace(' ', '').replace(',', '').replace('|', '').replace('/', '').replace('-', ''))
    else:
        raw = [str(s).strip().upper() for s in signs]
    order = ['1', 'X', '2']
    return [s for s in order if s in raw]


def parse_frame_text(text, antal_matcher):
    """
    Läser en manuell grundram från text.
    Exempel: 1X2 / 1X / 12 / 1 / X2 ... eller en rad med 13 grupper.
    """
    if text is None or str(text).strip() == "":
        return None, "Ingen textgrundram angiven."
    tokens = re.findall(r'[1Xx2]{1,3}', str(text))
    if len(tokens) != antal_matcher:
        return None, f"Textgrundramen måste innehålla exakt {antal_matcher} teckengrupper. Hittade {len(tokens)}."
    frame = [normalize_signs(tok) for tok in tokens]
    return frame, ""


def generate_rows_from_frame(frame, max_rows=None):
    """Genererar alla enkelrader från en egen grundram.

    Ingen hård radspärr används här. Stora grundramar kan ta längre tid och
    mer minne, men appen ska inte stoppa användaren när ramen är medvetet bred.
    """
    if not frame:
        return [], 0, False, "Ingen grundram finns."
    frame = [normalize_signs(s) for s in frame]
    empty_matches = [i + 1 for i, signs in enumerate(frame) if len(signs) == 0]
    if empty_matches:
        return [], 0, False, "Alla matcher måste ha minst ett tecken. Saknar tecken i match: " + ", ".join(map(str, empty_matches))
    n_rows = frame_row_count(frame)
    rows = [''.join(tup) for tup in itertools.product(*frame)]
    return rows, n_rows, True, ""


def row_log_probability(row_str, prob_vector):
    """Log-sannolikhet enligt dagens procent. Robust mot felaktig/lös rad eller kort procentvektor."""
    if isinstance(row_str, (list, tuple)):
        row_str = ''.join(map(str, row_str))
    row_str = str(row_str).strip().upper()
    prob_vector = list(prob_vector or [])
    total = 0.0
    for i, c in enumerate(row_str):
        idx = i * 3
        if idx + 2 >= len(prob_vector):
            # Säker fallback i stället för krasch om kupongvektorn/raden är osynkad.
            p = 33.333
        elif c == '1':
            p = prob_vector[idx]
        elif c == 'X':
            p = prob_vector[idx + 1]
        else:
            p = prob_vector[idx + 2]
        total += np.log(max(float(p), 0.05) / 100.0)
    return float(total)


def weighted_13_share(all_rows, selected_rows, prob_vector):
    """Viktad 13-chans inom filtrerad radmassa, baserad på dagens procentvektor."""
    if not all_rows or not selected_rows:
        return 0.0
    all_rows = [normalize_single_row_text(r) for r in (all_rows or []) if normalize_single_row_text(r)]
    selected_rows = [normalize_single_row_text(r) for r in (selected_rows or []) if normalize_single_row_text(r)]
    selected_set = set(selected_rows)
    logs = np.array([row_log_probability(r, prob_vector) for r in all_rows], dtype=float)
    mx = float(np.max(logs)) if len(logs) else 0.0
    weights = np.exp(logs - mx)
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    sel_weight = float(sum(w for r, w in zip(all_rows, weights) if r in selected_set))
    return round((sel_weight / total) * 100, 2)


def build_12_cover_sets(rows):
    """
    Täckningsyta för 12-rättsgaranti inom given radmassa.
    En vald rad täcker sig själv + rader som skiljer sig i max en match.
    """
    row_to_idx = {r: i for i, r in enumerate(rows)}
    signs = ['1', 'X', '2']
    cover_sets = []
    for i, r in enumerate(rows):
        cov = {i}
        chars = list(r)
        for pos, old in enumerate(chars):
            for s in signs:
                if s == old:
                    continue
                chars[pos] = s
                j = row_to_idx.get(''.join(chars))
                if j is not None:
                    cov.add(j)
            chars[pos] = old
        cover_sets.append(cov)
    return cover_sets


def prune_tipsetmatrix_selection(selected, cover_sets, row_scores, n_targets):
    """Tar bort överflödiga rader utan att bryta 12-garantitäckningen."""
    if not selected:
        return []
    selected = list(selected)
    cover_count = np.zeros(n_targets, dtype=np.int32)
    for idx in selected:
        for j in cover_sets[idx]:
            cover_count[j] += 1
    # Försök ta bort lägst viktade / minst täckande rader först.
    removal_order = sorted(selected, key=lambda idx: (float(row_scores[idx]), len(cover_sets[idx])))
    selected_set = set(selected)
    for idx in removal_order:
        if idx not in selected_set:
            continue
        cov = cover_sets[idx]
        if all(cover_count[j] > 1 for j in cov):
            selected_set.remove(idx)
            for j in cov:
                cover_count[j] -= 1
    return [idx for idx in selected if idx in selected_set]


def tipsetmatrix12_reduce(rows, row_scores=None, mode="Balans", max_output_rows=None, seed=42):
    """
    IntelliMatrix-liknande greedy set-cover för 12-rättsgaranti inom filtrerad radmassa.
    Garantin gäller under förutsättning att rätt rad finns kvar i rows.
    """
    rows = list(dict.fromkeys([str(r).strip().upper() for r in rows if str(r).strip()]))
    n = len(rows)
    if n == 0:
        return [], {"covered_pct": 0.0, "covered_rows": 0, "target_rows": 0, "complete": False, "selected_count": 0}
    if row_scores is None:
        row_scores = np.zeros(n, dtype=float)
    else:
        row_scores = np.array(row_scores, dtype=float)
        if len(row_scores) != n:
            row_scores = np.zeros(n, dtype=float)

    mode_key = str(mode).lower()
    restarts = 1 if "snabb" in mode_key else 4 if "balans" in mode_key else 9
    jitter_scale = 0.001 if "snabb" in mode_key else 0.02 if "balans" in mode_key else 0.06
    rng = np.random.default_rng(int(seed))
    cover_sets = build_12_cover_sets(rows)

    best_selected = None
    best_meta = None
    candidate_indices = np.arange(n)

    for restart in range(restarts):
        jitter = rng.normal(0, jitter_scale, size=n)
        selected = []
        selected_set = set()
        uncovered = set(range(n))

        while uncovered:
            if max_output_rows is not None and len(selected) >= int(max_output_rows):
                break
            best_idx = None
            best_key = None
            # Greedy: max ny täckning, sedan radkvalitet som tie-breaker.
            for idx in candidate_indices:
                if int(idx) in selected_set:
                    continue
                gain = len(cover_sets[int(idx)] & uncovered)
                if gain <= 0:
                    continue
                key = (gain, float(row_scores[int(idx)]) + float(jitter[int(idx)]), len(cover_sets[int(idx)]))
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = int(idx)
            if best_idx is None:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered -= cover_sets[best_idx]

        covered_rows = n - len(uncovered)
        complete = (len(uncovered) == 0)
        if complete and mode_key != "snabb":
            selected = prune_tipsetmatrix_selection(selected, cover_sets, row_scores, n)
            # Säkerställ metadata efter pruning.
            covered = set()
            for idx in selected:
                covered |= cover_sets[idx]
            covered_rows = len(covered)
            complete = (covered_rows == n)

        meta = {
            "covered_pct": round((covered_rows / n) * 100, 2),
            "covered_rows": int(covered_rows),
            "target_rows": int(n),
            "complete": bool(complete),
            "selected_count": int(len(selected)),
            "mode": mode,
            "restarts": restarts,
        }
        score_sum = float(sum(row_scores[idx] for idx in selected))
        rank_key = (1 if complete else 0, covered_rows, -len(selected), score_sum)
        if best_selected is None:
            best_selected, best_meta, best_rank = selected, meta, rank_key
        elif rank_key > best_rank:
            best_selected, best_meta, best_rank = selected, meta, rank_key

    reduced_rows = [rows[idx] for idx in best_selected]
    best_meta["selected_count"] = len(reduced_rows)
    return reduced_rows, best_meta


def add_rows_for_13_chance(filtered_rows, reduced_rows, row_scores=None, target_13_pct=0.0):
    """Lägger till extra rader ovanpå minsta 12-garantival för att höja 13-chansen.

    12-garantin byggs först med så få rader som motorn hittar. Därefter kan
    användaren höja önskad 13-chans. Då adderas rader från den redan filtrerade
    massan, prioriterat efter rad_score/procentvikt. Det kan aldrig försämra
    12-garantin; det ökar bara slutradantalet och andelen exakta rader.
    """
    filtered_rows = list(dict.fromkeys([normalize_single_row_text(r) for r in (filtered_rows or []) if normalize_single_row_text(r)]))
    reduced_rows = list(dict.fromkeys([normalize_single_row_text(r) for r in (reduced_rows or []) if normalize_single_row_text(r)]))
    n = len(filtered_rows)
    if n == 0:
        return reduced_rows, {'target_13_pct': 0.0, 'target_rows': 0, 'base_12_rows': len(reduced_rows), 'extra_13_rows': 0, 'final_13_pct': 0.0}
    try:
        target_13_pct = float(target_13_pct or 0.0)
    except Exception:
        target_13_pct = 0.0
    target_13_pct = max(0.0, min(100.0, target_13_pct))
    target_rows = int(np.ceil(n * target_13_pct / 100.0))
    base_count = len(reduced_rows)
    if target_rows <= base_count:
        return reduced_rows, {'target_13_pct': target_13_pct, 'target_rows': target_rows, 'base_12_rows': base_count, 'extra_13_rows': 0, 'final_13_pct': round(100.0 * base_count / max(1, n), 2)}

    selected = set(reduced_rows)
    score_by_row = {}
    if row_scores is not None and len(row_scores) == len(filtered_rows):
        try:
            score_by_row = {r: float(sc) for r, sc in zip(filtered_rows, row_scores)}
        except Exception:
            score_by_row = {}
    remaining = [r for r in filtered_rows if r not in selected]
    # Högre log-probability först. Om scores saknas blir ordningen stabil.
    remaining.sort(key=lambda r: score_by_row.get(r, 0.0), reverse=True)
    need = max(0, target_rows - base_count)
    add = remaining[:need]
    final_rows = reduced_rows + add
    return final_rows, {
        'target_13_pct': target_13_pct,
        'target_rows': target_rows,
        'base_12_rows': base_count,
        'extra_13_rows': len(add),
        'final_13_pct': round(100.0 * len(final_rows) / max(1, n), 2),
    }


def build_tipsetmatrix_guarantee_table(filtered_rows, reduced_rows, antal_matcher, prob_vector=None):
    """Bygger garantitabell: varje filtrerad rad testas som möjlig facitrad.
    Robust mot att rader råkar skickas in som list/tuple/metadata i stället för rena radsträngar.
    """
    antal_matcher = int(antal_matcher)
    filtered_rows = list(dict.fromkeys([
        normalize_single_row_text(r) for r in (filtered_rows or [])
        if len(normalize_single_row_text(r)) == antal_matcher
    ]))
    reduced_rows = list(dict.fromkeys([
        normalize_single_row_text(r) for r in (reduced_rows or [])
        if len(normalize_single_row_text(r)) == antal_matcher
    ]))
    n = len(filtered_rows)
    if n == 0 or not reduced_rows:
        table = pd.DataFrame([
            {"Rättnivå": "13 rätt", "Antal möjliga facitrader": 0, "Andel %": 0.0},
            {"Rättnivå": "12 rätt", "Antal möjliga facitrader": 0, "Andel %": 0.0},
            {"Rättnivå": "11 rätt", "Antal möjliga facitrader": 0, "Andel %": 0.0},
            {"Rättnivå": "10 rätt", "Antal möjliga facitrader": 0, "Andel %": 0.0},
            {"Rättnivå": "Under 10", "Antal möjliga facitrader": 0, "Andel %": 0.0},
        ])
        return table, {"13_oviktad": 0.0, "13_viktad": 0.0, "12plus": 0.0, "min_garanti": 0, "reduceringsgrad": 0.0}

    counts = {k: 0 for k in range(antal_matcher + 1)}
    red = reduced_rows
    for target in filtered_rows:
        best = 0
        for rr in red:
            hit = sum(1 for a, b in zip(target, rr) if a == b)
            if hit > best:
                best = hit
                if best == antal_matcher:
                    break
        counts[best] += 1

    rows_out = []
    for level in [antal_matcher, antal_matcher - 1, antal_matcher - 2, antal_matcher - 3]:
        if level < 0:
            continue
        rows_out.append({
            "Rättnivå": f"{level} rätt",
            "Antal möjliga facitrader": int(counts.get(level, 0)),
            "Andel %": round((counts.get(level, 0) / n) * 100, 2)
        })
    under10 = sum(v for k, v in counts.items() if k <= antal_matcher - 4)
    rows_out.append({"Rättnivå": f"Under {antal_matcher - 3}", "Antal möjliga facitrader": int(under10), "Andel %": round((under10 / n) * 100, 2)})
    table = pd.DataFrame(rows_out)

    selected_set = set(reduced_rows)
    exact_13 = sum(1 for r in filtered_rows if r in selected_set)
    twelve_plus = sum(v for k, v in counts.items() if k >= antal_matcher - 1)
    weighted = weighted_13_share(filtered_rows, reduced_rows, prob_vector) if prob_vector is not None else 0.0
    summary = {
        "13_oviktad": round((exact_13 / n) * 100, 2),
        "13_viktad": weighted,
        "12plus": round((twelve_plus / n) * 100, 2),
        "min_garanti": int(min(k for k, v in counts.items() if v > 0)),
        "reduceringsgrad": round((1 - (len(reduced_rows) / n)) * 100, 2) if n else 0.0,
        "filtered_rows": int(n),
        "reduced_rows": int(len(reduced_rows)),
    }
    return table, summary



def parse_result_row(text, antal_matcher):
    """Läser facit/vinstrad. Tillåter 1X2 med mellanslag, bindestreck eller klistrad rad."""
    raw = str(text or "").strip().upper()
    if not raw:
        return None, ""
    chars = re.findall(r'[1X2]', raw)
    if len(chars) != antal_matcher:
        return None, f"Facitraden måste innehålla exakt {antal_matcher} tecken (1/X/2). Hittade {len(chars)}."
    return ''.join(chars), ""


def result_in_frame(result_row, frame):
    """Kontrollerar om facitraden ryms i den manuella/valda grundramen."""
    if not result_row or not frame or len(result_row) != len(frame):
        return False
    return all(result_row[i] in set(frame[i]) for i in range(len(result_row)))


def best_hit_against_rows(result_row, rows):
    """Returnerar bästa antal rätt samt närmaste reducerade rader."""
    if not result_row or not rows:
        return 0, []
    best = -1
    nearest = []
    for rr in rows:
        hit = sum(1 for a, b in zip(result_row, rr) if a == b)
        if hit > best:
            best = hit
            nearest = [rr]
        elif hit == best and len(nearest) < 10:
            nearest.append(rr)
    return int(max(best, 0)), nearest


def build_facit_check(result_row, frame, base_rows, filtered_rows, reduced_rows, antal_matcher):
    """Felsökningsrapport för en känd historisk vinstrad/facitrad."""
    base_set = set(base_rows or [])
    filtered_set = set(filtered_rows or [])
    reduced_set = set(reduced_rows or [])
    best_hit, nearest = best_hit_against_rows(result_row, list(reduced_rows or []))
    return {
        "Facitrad": result_row,
        "I grundram": result_in_frame(result_row, frame),
        "I genererade grundrader": result_row in base_set,
        "Efter filter": result_row in filtered_set,
        "Efter TipsetMatrix": result_row in reduced_set,
        "Bästa rätt efter TipsetMatrix": best_hit,
        "Närmaste reducerade rader": nearest,
        "Reduceringsgaranti giltig": result_row in filtered_set,
        "12+ uppnått": best_hit >= antal_matcher - 1,
    }


def yes_no(v):
    return "Ja" if bool(v) else "Nej"

def frame_export_text(frame):
    if not frame:
        return ""
    return " / ".join(_sort_signs_display(set(s)) for s in frame)

def submission_game_header_and_code(spelform):
    """Returnerar rubrik och radprefix för inlämningsformatet egna rader.

    Svenska Spels externa spelfil ska använda radprefix E på varje rad,
    oavsett speltyp. Rubriken skiljer fortfarande spelformerna åt.

    Exempel:
    STRYKTIPSET
    E,2,2,1,X,2,...
    """
    sf = str(spelform or "").strip().lower()
    if "europa" in sf:
        return "EUROPATIPSET", "E"
    if "stryk" in sf:
        return "STRYKTIPSET", "E"
    if "topp" in sf:
        return "TOPPTIPSET", "E"
    if "power" in sf:
        return "POWERPLAY", "E"
    return str(spelform or "TIPSET").upper(), "E"


def normalize_single_row_text(row):
    """Gör en rad robust: tar bort kommatecken/mellanslag och behåller bara 1/X/2."""
    raw = str(row or "").upper()
    return "".join(ch for ch in raw if ch in {"1", "X", "2"})


def rows_to_submission_text(rows, spelform, antal_matcher):
    """Bygger inlämningsfil för Egna rader.

    Varje enkelrad får prefix enligt spelform och kommatecken mellan alla tecken:
    E,2,2,1,X,2,X,X,2,1,2,2,1,1
    """
    header, code = submission_game_header_and_code(spelform)
    out = [header]
    for row in rows or []:
        clean = normalize_single_row_text(row)
        if len(clean) == int(antal_matcher):
            out.append(",".join([code] + list(clean)))
    return "\n".join(out)

def selected_signs_missing(rows, frame, antal_matcher):
    """Returnerar manuellt valda tecken som saknas helt i en radmassa.

    Exempel: om M5 är helgarderad i grundramen men inga kvarvarande rader har 1 på M5,
    returneras (5, '1'). Detta skyddar mot att filter/reducering omedvetet blankar ett tecken.
    """
    if not frame:
        return []
    rows = [normalize_single_row_text(r) for r in (rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    missing = []
    for mi in range(int(antal_matcher)):
        selected = normalize_signs(frame[mi]) if mi < len(frame) else []
        if not selected:
            continue
        present = {r[mi] for r in rows} if rows else set()
        for sign in selected:
            if sign not in present:
                missing.append((mi + 1, sign))
    return missing


def format_missing_signs(missing):
    if not missing:
        return ""
    return ", ".join([f"M{m}:{s}" for m, s in missing])


def build_sign_distribution_df(rows, frame, antal_matcher):
    """Bygger en enkel teckenfördelning per match för filtrerad/reducerad radmassa."""
    rows = [normalize_single_row_text(r) for r in (rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    total = len(rows)
    out = []
    for mi in range(int(antal_matcher)):
        selected = normalize_signs(frame[mi]) if frame and mi < len(frame) else ['1', 'X', '2']
        counts = {s: sum(1 for r in rows if r[mi] == s) for s in ['1', 'X', '2']}
        zero_selected = [s for s in selected if counts.get(s, 0) == 0]
        out.append({
            "Match": mi + 1,
            "Grundram": _sort_signs_display(selected),
            "1": counts['1'],
            "X": counts['X'],
            "2": counts['2'],
            "1 %": round((counts['1'] / total * 100), 1) if total else 0.0,
            "X %": round((counts['X'] / total * 100), 1) if total else 0.0,
            "2 %": round((counts['2'] / total * 100), 1) if total else 0.0,
            "Teckenskydd": "⚠️ " + "".join(zero_selected) if zero_selected else "OK",
        })
    return pd.DataFrame(out)


def _fmt_counts_with_pct(counts, total):
    parts = []
    for s in ['1', 'X', '2']:
        n = int(counts.get(s, 0))
        pct = (n / total * 100.0) if total else 0.0
        parts.append(f"{s}: {n:,} ({pct:.1f}%)".replace(',', ' '))
    return " | ".join(parts)


def build_combined_sign_distribution_df(filter_rows, reduced_rows, frame, antal_matcher):
    """Kompakt matchvis teckenfördelning: grundram, efter filter och efter inlämningsrader."""
    filter_rows = [normalize_single_row_text(r) for r in (filter_rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    reduced_rows = [normalize_single_row_text(r) for r in (reduced_rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    filter_total = len(filter_rows)
    reduced_total = len(reduced_rows)
    out = []
    for mi in range(int(antal_matcher)):
        selected = normalize_signs(frame[mi]) if frame and mi < len(frame) else ['1', 'X', '2']
        f_counts = {s: sum(1 for r in filter_rows if r[mi] == s) for s in ['1', 'X', '2']}
        r_counts = {s: sum(1 for r in reduced_rows if r[mi] == s) for s in ['1', 'X', '2']}
        zero_filter = [s for s in selected if f_counts.get(s, 0) == 0]
        zero_reduced = [s for s in selected if r_counts.get(s, 0) == 0]
        status = []
        if zero_filter:
            status.append("Filter saknar " + "".join(zero_filter))
        if zero_reduced:
            status.append("Slutrader saknar " + "".join(zero_reduced))
        out.append({
            "Match/lag": f"M{mi + 1}",
            "Grundram": _sort_signs_display(selected),
            "Efter filter": _fmt_counts_with_pct(f_counts, filter_total),
            "Efter inlämning": _fmt_counts_with_pct(r_counts, reduced_total),
            "Status": "⚠️ " + "; ".join(status) if status else "OK",
        })
    return pd.DataFrame(out)


def add_rows_for_sign_coverage(source_rows, selected_rows, frame, row_scores=None, antal_matcher=None):
    """Lägger till bästa tillgängliga rader så att alla manuellt valda tecken finns i slutraderna.

    Detta ändrar inte filtermassan. Det lägger bara till extra reducerade rader från redan filtrerad radmassa
    om TipsetMatrix-valet råkar ge 0 rader på ett tecken som spelaren aktivt markerat i grundramen.
    """
    if antal_matcher is None:
        antal_matcher = len(frame) if frame else 0
    source_rows = [normalize_single_row_text(r) for r in (source_rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    selected_rows = list(dict.fromkeys([normalize_single_row_text(r) for r in (selected_rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]))
    if not frame or not source_rows or not selected_rows:
        return selected_rows, [], selected_signs_missing(selected_rows, frame, antal_matcher)

    if row_scores is None or len(row_scores) != len(source_rows):
        scores = [0.0] * len(source_rows)
    else:
        scores = [float(x) for x in row_scores]

    selected_set = set(selected_rows)
    added = []
    missing = selected_signs_missing(selected_rows, frame, antal_matcher)
    safety = 0
    while missing and safety < int(antal_matcher) * 3:
        best_idx = None
        best_key = None
        for idx, rr in enumerate(source_rows):
            if rr in selected_set:
                continue
            covers = sum(1 for m, s in missing if rr[m - 1] == s)
            if covers <= 0:
                continue
            key = (covers, scores[idx])
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is None:
            break
        chosen = source_rows[best_idx]
        selected_rows.append(chosen)
        selected_set.add(chosen)
        added.append(chosen)
        missing = selected_signs_missing(selected_rows, frame, antal_matcher)
        safety += 1
    return selected_rows, added, missing



def add_rows_for_min_13_chance(source_rows, selected_rows, target_pct=0.0, row_scores=None, antal_matcher=None):
    """Lägger till högst rankade rader från filtermassan tills oviktad 13-chans når målet.

    13-chansen här är andelen inlämnade rader av den filtrerade radmassan.
    Funktionen ändrar inte filtermassan och kan därför inte sänka 12-garantin; den ökar bara antal slutrader.
    """
    if antal_matcher is None:
        antal_matcher = len(source_rows[0]) if source_rows else 0
    source_rows = [normalize_single_row_text(r) for r in (source_rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    selected_rows = list(dict.fromkeys([normalize_single_row_text(r) for r in (selected_rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]))
    try:
        target_pct = float(target_pct)
    except Exception:
        target_pct = 0.0
    target_pct = max(0.0, min(100.0, target_pct))
    if not source_rows or target_pct <= 0:
        return selected_rows, []

    needed = int(np.ceil(len(source_rows) * (target_pct / 100.0)))
    needed = max(0, min(len(source_rows), needed))
    if len(selected_rows) >= needed:
        return selected_rows, []

    if row_scores is None or len(row_scores) != len(source_rows):
        scores = [0.0] * len(source_rows)
    else:
        scores = [float(x) for x in row_scores]

    selected_set = set(selected_rows)
    candidates = []
    for idx, rr in enumerate(source_rows):
        if rr not in selected_set:
            candidates.append((float(scores[idx]), idx, rr))
    candidates.sort(reverse=True)

    added = []
    for _, _, rr in candidates:
        if len(selected_rows) >= needed:
            break
        selected_rows.append(rr)
        selected_set.add(rr)
        added.append(rr)
    return selected_rows, added


def submission_file_stem(spelform):
    header, _ = submission_game_header_and_code(spelform)
    return header.lower().replace("å", "a").replace("ä", "a").replace("ö", "o")


# --- AI-RAM / U-FILTER ---
def _sort_signs_display(signs):
    order = {'1': 0, 'X': 1, '2': 2}
    return ''.join(sorted(list(signs), key=lambda s: order.get(s, 9)))

def build_match_ai_stats(v_m, filter_vec, antal_matcher):
    """Bygger matchvis historisk träffbild från de liknande omgångarna."""
    rows = []
    total = len(v_m)
    for m in range(antal_matcher):
        outcomes = [str(r).strip().upper()[m] for r in v_m['Correct_Row'].astype(str) if len(str(r).strip()) == antal_matcher]
        tot = len(outcomes) if outcomes else total
        counts = {s: outcomes.count(s) for s in ['1', 'X', '2']}
        hist_pct = {s: (counts[s] / tot) * 100 if tot else 0.0 for s in ['1', 'X', '2']}
        input_pct = {
            '1': float(filter_vec[m*3]),
            'X': float(filter_vec[m*3+1]),
            '2': float(filter_vec[m*3+2]),
        }
        # Sortering: historisk träff först, dagens procent som tie-breaker.
        order = sorted(['1', 'X', '2'], key=lambda s: (hist_pct[s], input_pct[s]), reverse=True)
        top1 = order[0]
        top2 = _sort_signs_display(order[:2])
        top3 = '1X2'
        rows.append({
            'Match': m + 1,
            'Ordning': ''.join(order),
            'Top1': top1,
            'Top2': top2,
            'Top3': top3,
            '1 hist %': round(hist_pct['1'], 1),
            'X hist %': round(hist_pct['X'], 1),
            '2 hist %': round(hist_pct['2'], 1),
            '1 idag %': round(input_pct['1'], 1),
            'X idag %': round(input_pct['X'], 1),
            '2 idag %': round(input_pct['2'], 1),
        })
    return rows

def frame_row_count(frame):
    if not frame:
        return 0
    n = 1
    for signs in frame:
        signs = normalize_signs(signs)
        if len(signs) == 0:
            return 0
        n *= len(signs)
    return int(n)

def evaluate_frame(frame, hist_rows, antal_matcher):
    """Returnerar täckning: hur många rätt som ryms i ramen mot historiska vinnarrader."""
    if not hist_rows:
        return {'13': 0.0, '12+': 0.0, '11+': 0.0, '10+': 0.0, 'Radantal': frame_row_count(frame)}
    hit_counts = []
    for r in hist_rows:
        inside = sum(1 for i, c in enumerate(r[:antal_matcher]) if c in frame[i])
        hit_counts.append(inside)
    total = len(hit_counts)
    return {
        '13': round((sum(1 for h in hit_counts if h >= antal_matcher) / total) * 100, 1),
        '12+': round((sum(1 for h in hit_counts if h >= antal_matcher - 1) / total) * 100, 1),
        '11+': round((sum(1 for h in hit_counts if h >= antal_matcher - 2) / total) * 100, 1),
        '10+': round((sum(1 for h in hit_counts if h >= antal_matcher - 3) / total) * 100, 1),
        'Min rätt i ram': min(hit_counts) if hit_counts else 0,
        'Max rätt i ram': max(hit_counts) if hit_counts else 0,
        'Radantal': frame_row_count(frame),
        'Spikar': sum(1 for s in frame if len(s) == 1),
        'Halvor': sum(1 for s in frame if len(s) == 2),
        'Hela': sum(1 for s in frame if len(s) == 3),
    }

def optimize_frame_greedy(hist_rows, match_stats, antal_matcher, target_pct=95.0, metric='12+'):
    """
    Greedy-ram: börjar med AI-top1 i alla matcher och lägger till nästa historiskt bästa tecken
    där det ger mest täckning per radkostnad. Målet är hög täckning med låg ramkostnad.
    """
    orders = [list(ms['Ordning']) for ms in match_stats]
    frame = [set([orders[i][0]]) for i in range(antal_matcher)]
    current_eval = evaluate_frame(frame, hist_rows, antal_matcher)

    guard = 0
    while current_eval.get(metric, 0.0) < target_pct and guard < antal_matcher * 2:
        guard += 1
        current_rows = max(1, frame_row_count(frame))
        best = None
        for m in range(antal_matcher):
            if len(frame[m]) >= 3:
                continue
            # Lägg bara till nästa tecken i historisk ordning, så ramen förblir logisk.
            next_sign = orders[m][len(frame[m])]
            test_frame = [set(x) for x in frame]
            test_frame[m].add(next_sign)
            ev = evaluate_frame(test_frame, hist_rows, antal_matcher)
            new_rows = max(1, ev['Radantal'])
            gain_main = ev.get(metric, 0.0) - current_eval.get(metric, 0.0)
            gain_13 = ev.get('13', 0.0) - current_eval.get('13', 0.0)
            gain_11 = ev.get('11+', 0.0) - current_eval.get('11+', 0.0)
            cost_mult = new_rows / current_rows
            # Balanserar förbättring mot radkostnad. En liten bonus gör att algoritmen inte fastnar
            # om en förbättring kräver flera tillägg innan den syns på huvudmåttet.
            score = (gain_main * 4.0 + gain_13 * 1.25 + gain_11 * 0.25 + ev.get(metric, 0.0) * 0.015) / np.log2(cost_mult + 1.0)
            candidate = (score, gain_main, gain_13, -new_rows, m, test_frame, ev)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            break
        frame = best[5]
        current_eval = best[6]

    return frame, current_eval


def optimize_frame_budgeted(hist_rows, match_stats, antal_matcher, max_rows=25000, max_full=4, max_halves=None):
    """
    Radbudget-ram: börjar med AI-top1 i alla matcher och lägger till tecken så länge
    de ger förbättrad historisk täckning utan att spränga radbudget eller max antal helgarderingar.

    Detta undviker att rammotorn jagar 12+-målet genom att fylla ramen med för många helgarderingar.
    Målet är i stället: bästa täckning inom en spelbar ram.
    """
    orders = [list(ms['Ordning']) for ms in match_stats]
    frame = [set([orders[i][0]]) for i in range(antal_matcher)]
    current_eval = evaluate_frame(frame, hist_rows, antal_matcher)

    if max_halves is None:
        max_halves = antal_matcher

    def avg_inside(fr):
        if not hist_rows:
            return 0.0
        vals = [sum(1 for i, c in enumerate(r[:antal_matcher]) if c in fr[i]) for r in hist_rows]
        return sum(vals) / len(vals)

    current_avg = avg_inside(frame)
    guard = 0
    while guard < antal_matcher * 2:
        guard += 1
        current_rows = max(1, frame_row_count(frame))
        best = None

        for m in range(antal_matcher):
            if len(frame[m]) >= 3:
                continue

            next_sign = orders[m][len(frame[m])]
            test_frame = [set(x) for x in frame]
            test_frame[m].add(next_sign)
            new_rows = frame_row_count(test_frame)
            if new_rows > max_rows:
                continue
            if sum(1 for s in test_frame if len(s) == 3) > max_full:
                continue
            if sum(1 for s in test_frame if len(s) == 2) > max_halves:
                continue

            ev = evaluate_frame(test_frame, hist_rows, antal_matcher)
            new_avg = avg_inside(test_frame)
            cost_mult = max(1.0001, new_rows / current_rows)

            gain_13 = ev.get('13', 0.0) - current_eval.get('13', 0.0)
            gain_12 = ev.get('12+', 0.0) - current_eval.get('12+', 0.0)
            gain_11 = ev.get('11+', 0.0) - current_eval.get('11+', 0.0)
            gain_10 = ev.get('10+', 0.0) - current_eval.get('10+', 0.0)
            gain_avg = new_avg - current_avg

            # Täckningsförbättring per radkostnad. Snitt-rätt ger algoritmen signal även
            # när 12+-nivån inte ökar direkt av ett enskilt teckentillägg.
            raw_gain = gain_13 * 3.0 + gain_12 * 2.0 + gain_11 * 0.8 + gain_10 * 0.15 + gain_avg * 18.0
            if raw_gain <= 0:
                continue
            score = raw_gain / np.log2(cost_mult + 1.0)
            candidate = (score, gain_12, gain_13, -new_rows, m, test_frame, ev, new_avg)
            if best is None or candidate > best:
                best = candidate

        if best is None:
            break

        frame = best[5]
        current_eval = best[6]
        current_avg = best[7]

    return frame, current_eval

def frame_to_string(frame):
    return ' '.join(_sort_signs_display(s) for s in frame)

def frame_compact_string(frame):
    return '-'.join(_sort_signs_display(s) for s in frame)

def build_ai_u_row(match_stats):
    return ''.join(ms['Top1'] for ms in match_stats)

def evaluate_u_row(u_row, hist_rows, antal_matcher, target_pct=90.0):
    hits = []
    for r in hist_rows:
        hits.append(sum(1 for i, c in enumerate(r[:antal_matcher]) if c == u_row[i]))
    total = len(hits) if hits else 0
    if not total:
        return {'hits': [], 'recommended_min': 0, 'coverage': 0.0, 'dist': pd.DataFrame()}
    recommended = 0
    coverage = 100.0
    for k in range(antal_matcher, -1, -1):
        pct = (sum(1 for h in hits if h >= k) / total) * 100
        if pct >= target_pct:
            recommended = k
            coverage = pct
            break
    dist = pd.DataFrame([
        {'U-träffar': k, 'Antal': sum(1 for h in hits if h == k), 'Andel %': round((sum(1 for h in hits if h == k) / total) * 100, 1)}
        for k in range(antal_matcher, -1, -1)
        if sum(1 for h in hits if h == k) > 0
    ])
    return {'hits': hits, 'recommended_min': recommended, 'coverage': round(coverage, 1), 'dist': dist}




def _u_signs_from_text(value):
    """Normaliserar ett utgångssystemfält. Tomt fält är tillåtet och betyder ingen markering i matchen."""
    return normalize_signs(value)


def _u_signs_display(signs):
    signs = _u_signs_from_text(signs)
    return _sort_signs_display(signs) if signs else ""


def u_system_to_text(system):
    """Kompakt visning av ett utgångssystem, t.ex. 1X / 1 / X2 / -."""
    parts = []
    for signs in (system or []):
        s = _u_signs_display(signs)
        parts.append(s if s else "-")
    return " / ".join(parts)


def u_system_marked_count(system):
    """Antal matcher i utgångssystemet där minst ett tecken är markerat."""
    return int(sum(1 for signs in (system or []) if len(_u_signs_from_text(signs)) > 0))


def u_system_from_single_row(row, antal_matcher):
    """Gör en enkel U-rad till ett utgångssystem med ett markerat tecken per match."""
    row = normalize_single_row_text(row)
    if len(row) != int(antal_matcher):
        return [[] for _ in range(int(antal_matcher))]
    return [[c] for c in row]


def parse_u_row_text(text, antal_matcher):
    """Bakåtkompatibel läsning av gammal U-rad. Returnerar ren rad eller tom sträng."""
    row = normalize_single_row_text(text)
    if len(row) != int(antal_matcher):
        return ''
    return row


def parse_u_system_text(text, antal_matcher):
    """Läser textad utgångsram: 1X / 12 / - / 2 ...  Tomma/- blir omarkerade matcher."""
    raw = str(text or '').strip().upper()
    if not raw:
        return None, "Ingen utgångsram angiven."
    tokens = re.findall(r'[1X2x]{1,3}|[-–—.]', raw)
    if len(tokens) != int(antal_matcher):
        return None, f"Utgångsramen måste innehålla exakt {antal_matcher} fält. Hittade {len(tokens)}."
    system = []
    for tok in tokens:
        if tok in ['-', '–', '—', '.']:
            system.append([])
        else:
            system.append(_u_signs_from_text(tok))
    return system, ""


def build_pct_favorite_u_system(prob_vector, antal_matcher):
    """Utgångssystem med högsta aktuella procent/streck per match."""
    out = []
    for m in range(int(antal_matcher)):
        idx = m * 3
        vals = list(prob_vector[idx:idx+3]) if prob_vector is not None else []
        if len(vals) < 3:
            out.append(['1'])
            continue
        out.append([['1', 'X', '2'][int(np.argmax(vals))]])
    return out


def build_pct_second_u_system(prob_vector, antal_matcher):
    """Utgångssystem med näst högsta aktuella procent/streck per match."""
    out = []
    for m in range(int(antal_matcher)):
        idx = m * 3
        vals = list(prob_vector[idx:idx+3]) if prob_vector is not None else []
        if len(vals) < 3:
            out.append(['X'])
            continue
        order = sorted(range(3), key=lambda i: float(vals[i]), reverse=True)
        out.append([['1', 'X', '2'][order[1]]])
    return out


def build_history_ai_u_system(v_m, filter_vec, antal_matcher):
    """Utgångssystem med historiskt starkaste tecken bland liknande omgångar."""
    match_stats = build_match_ai_stats(v_m, filter_vec, antal_matcher)
    return [[ms.get('Top1', '1')] for ms in match_stats]


def u_system_counts(row_str, u_system):
    """Räknar Helgardering-liknande Antal tecken mot ett utgångssystem.

    total = antalet utgångstips som sitter.
    ones/draws/twos = träffar uppdelade på faktiskt tecken.
    any_1x2 = högsta av ones/draws/twos, dvs filtret Antal 1/X/2 som
    fångar om minst en teckentyp ligger i valt intervall i en kontrollerbar form.
    """
    row_str = normalize_single_row_text(row_str)
    total = ones = draws = twos = 0
    for i, c in enumerate(row_str):
        if i >= len(u_system):
            break
        allowed = set(_u_signs_from_text(u_system[i]))
        if c in allowed:
            total += 1
            if c == '1':
                ones += 1
            elif c == 'X':
                draws += 1
            elif c == '2':
                twos += 1
    return {
        'utips': int(total),
        'ones': int(ones),
        'draws': int(draws),
        'twos': int(twos),
        'any_1x2': int(max(ones, draws, twos)),
    }


def u_row_hit_count(row_str, u_row):
    """Bakåtkompatibel helper: antal träffar mot en enkel U-rad."""
    return u_system_counts(row_str, u_system_from_single_row(u_row, len(normalize_single_row_text(u_row))))['utips']


def _u_system_metric(row_str, u_system, metric):
    return u_system_counts(row_str, u_system).get(metric, 0)


def _u_system_signature(u_systems):
    parts = []
    for u in u_systems or []:
        if not isinstance(u, dict):
            continue
        frame_txt = u_system_to_text(u.get('system', []))
        parts.append((str(u.get('id','')), str(u.get('name','')), frame_txt))
    return tuple(parts)


def _collect_u_rows_from_session(antal_matcher):
    """Sparar utgångssystem/U-filter i spelfil och filterpaket.

    Namnet behålls avsiktligt för bakåtkompatibilitet med tidigare v12.0aj-spelfiler,
    men innehållet är nu Helgardering-liknande utgångssystem i stället för enkla U-rader.
    """
    out = {'version': 'utgangssystem_v2', 'slots': []}
    choices = ['', '1', 'X', '2', '1X', '12', 'X2', '1X2']
    default_modes = {1: 'Favorit/procent', 2: 'Historik/AI', 3: '5 bästa spikar', 4: '5 bästa halvor', 5: '5 bästa skrällar', 6: 'Manuell'}
    for slot in range(1, 7):
        enabled = bool(st.session_state.get(f'v12_us_{slot}_enabled', slot in [1, 2]))
        mode = str(st.session_state.get(f'v12_us_{slot}_source', default_modes.get(slot, 'Manuell')) or 'Manuell')
        name = str(st.session_state.get(f'v12_us_{slot}_name', f'Utgångssystem {slot}') or f'Utgångssystem {slot}').strip()
        signs = []
        for m in range(int(antal_matcher)):
            val = str(st.session_state.get(f'v12_us_{slot}_m{m}', '') or '')
            if val not in choices:
                val = _u_signs_display(val)
            signs.append(_u_signs_from_text(val))
        raw_text = str(st.session_state.get(f'v12_us_{slot}_text', '') or '')
        out['slots'].append({
            'slot': slot,
            'enabled': enabled,
            'source': mode,
            'name': name,
            'manual_system': signs,
            'raw_text': raw_text,
            # Direktstyrning från Utgångssystem-panelen för huvudfiltret
            # "Antalet utgångstips". Filtercentralen speglar dessa värden.
            'utips_mode': str(st.session_state.get(f'v12_us_{slot}_utips_mode', 'Av') or 'Av'),
            'utips_interval': list(st.session_state.get(f'v12_us_{slot}_utips_interval', (0, int(antal_matcher))) or (0, int(antal_matcher))),
        })
    return out


def _apply_u_rows_to_session(u_payload):
    """Återställer utgångssystem/U-filter innan filtercentralens widgets byggs."""
    if not isinstance(u_payload, dict):
        return
    # Nytt format från v12.0ak.
    if u_payload.get('version') == 'utgangssystem_v2' or u_payload.get('slots'):
        for item in u_payload.get('slots', []) or []:
            if not isinstance(item, dict):
                continue
            try:
                slot = int(item.get('slot', 0))
            except Exception:
                slot = 0
            if not (1 <= slot <= 6):
                continue
            st.session_state[f'v12_us_{slot}_enabled'] = bool(item.get('enabled', False))
            st.session_state[f'v12_us_{slot}_source'] = str(item.get('source') or 'Manuell')
            st.session_state[f'v12_us_{slot}_name'] = str(item.get('name') or f'Utgångssystem {slot}')
            if item.get('raw_text') is not None:
                st.session_state[f'v12_us_{slot}_text'] = str(item.get('raw_text') or '')
            if item.get('utips_mode') is not None:
                st.session_state[f'v12_us_{slot}_utips_mode'] = str(item.get('utips_mode') or 'Av')
            if item.get('utips_interval') is not None:
                try:
                    _iv = item.get('utips_interval') or []
                    if len(_iv) >= 2:
                        st.session_state[f'v12_us_{slot}_utips_interval'] = (int(float(_iv[0])), int(float(_iv[1])))
                except Exception:
                    pass
            manual = item.get('manual_system') or []
            for m, signs in enumerate(manual[:13]):
                st.session_state[f'v12_us_{slot}_m{m}'] = _u_signs_display(signs)
        return

    # Bakåtkompatibilitet: tidigare v12.0aj sparade include_favorite/history/second + manuella U-rader.
    if 'include_favorite' in u_payload:
        st.session_state['v12_us_1_enabled'] = bool(u_payload.get('include_favorite'))
        st.session_state['v12_us_1_source'] = 'Favorit/procent'
        st.session_state['v12_us_1_name'] = 'Utgångssystem 1'
    if 'include_history' in u_payload:
        st.session_state['v12_us_2_enabled'] = bool(u_payload.get('include_history'))
        st.session_state['v12_us_2_source'] = 'Historik/AI'
        st.session_state['v12_us_2_name'] = 'Utgångssystem 2'
    if 'include_second' in u_payload:
        st.session_state['v12_us_3_enabled'] = bool(u_payload.get('include_second'))
        st.session_state['v12_us_3_source'] = 'Andrahand'
        st.session_state['v12_us_3_name'] = 'Utgångssystem 3'
    # Lägg första gamla manuella U-raden i första lediga manuella slot.
    for item in u_payload.get('manual', []) or []:
        if not isinstance(item, dict):
            continue
        row = parse_u_row_text(item.get('row') or item.get('raw') or '', 13)
        if not row:
            continue
        for slot in range(1, 7):
            if not bool(st.session_state.get(f'v12_us_{slot}_enabled', False)):
                st.session_state[f'v12_us_{slot}_enabled'] = True
                st.session_state[f'v12_us_{slot}_source'] = 'Manuell'
                st.session_state[f'v12_us_{slot}_name'] = str(item.get('name') or f'Utgångssystem {slot}')
                for m, c in enumerate(row):
                    st.session_state[f'v12_us_{slot}_m{m}'] = c
                break


def build_u_rows_for_filtercentral(v_m, filter_vec, antal_matcher, hist_df=None, max_shock_pct=22):
    """Bygger aktiva utgångssystem som sedan blir fem filter per system."""
    systems = []
    hist_df = hist_df if isinstance(hist_df, pd.DataFrame) and not hist_df.empty else v_m
    default_modes = {1: 'Favorit/procent', 2: 'Historik/AI', 3: '5 bästa spikar', 4: '5 bästa halvor', 5: '5 bästa skrällar', 6: 'Manuell'}
    for slot in range(1, 7):
        enabled = bool(st.session_state.get(f'v12_us_{slot}_enabled', slot in [1, 2]))
        if not enabled:
            continue
        mode = str(st.session_state.get(f'v12_us_{slot}_source', default_modes.get(slot, 'Manuell')) or 'Manuell')
        name = str(st.session_state.get(f'v12_us_{slot}_name', f'Utgångssystem {slot}') or f'Utgångssystem {slot}').strip()
        source = mode
        if mode == 'Favorit/procent':
            system = build_pct_favorite_u_system(filter_vec, antal_matcher)
            source = 'Högsta aktuella procent per match'
        elif mode == 'Historik/AI':
            system = build_history_ai_u_system(v_m, filter_vec, antal_matcher)
            source = 'Vanligaste historiska tecken bland liknande omgångar'
        elif mode == 'Andrahand':
            system = build_pct_second_u_system(filter_vec, antal_matcher)
            source = 'Näst högsta aktuella procent per match'
        elif mode in STRECK_U_SOURCES:
            system, _rec_df = _u_system_from_streck_source(mode, filter_vec, hist_df, v_m, antal_matcher, max_shock_pct=max_shock_pct)
            source = f'{mode} – baserat på aktuell streckbild och verifierat mot historiken'
        else:
            system = []
            for m in range(int(antal_matcher)):
                system.append(_u_signs_from_text(st.session_state.get(f'v12_us_{slot}_m{m}', '')))
            source = 'Manuellt utgångssystem'
        if len(system) != int(antal_matcher):
            continue
        if u_system_marked_count(system) <= 0:
            continue
        systems.append({
            'id': f'us{slot}',
            'slot': slot,
            'name': name or f'Utgångssystem {slot}',
            'system': system,
            'source': source,
            'mode': mode,
            'marked': u_system_marked_count(system),
        })
    return systems


def u_row_diag_df(u_systems, hist_rows, antal_matcher, target_pct=90.0):
    """Diagnos för utgångssystem: visar fem Helgardering-liknande Antal tecken-värden."""
    rows = []
    metric_labels = {
        'utips': 'Antalet utgångstips',
        'ones': 'Antalet ettor',
        'draws': 'Antalet kryss',
        'twos': 'Antalet tvåor',
        'any_1x2': 'Antal 1/X/2',
    }
    clean_hist = [normalize_single_row_text(r) for r in (hist_rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    for u in u_systems or []:
        system = u.get('system', [])
        for metric, label in metric_labels.items():
            vals = [_u_system_metric(r, system, metric) for r in clean_hist]
            if vals:
                rec = get_best_interval(vals, target_pct)
                hp, ht, pct = _hist_pass_count(vals, rec)
                dist_txt = f"min {min(vals)} · median {float(np.median(vals)):.1f} · max {max(vals)}"
                rec_txt = f"{int(rec[0])}–{int(rec[1])}"
            else:
                hp = ht = 0; pct = 0.0; dist_txt = '—'; rec_txt = '—'
            rows.append({
                'System': u.get('name', ''),
                'Filter': label,
                'Systemtecken': u_system_to_text(system),
                'Markerade matcher': u.get('marked', u_system_marked_count(system)),
                'Källa': u.get('source', ''),
                'Rek. intervall': rec_txt,
                'Historisk träff': f"{hp}/{ht}",
                'Träffbild': dist_txt,
            })
    return pd.DataFrame(rows)

def pass_super_macro_row(row_str, prob_vector, bounds, req_groups):
    c1, cx, c2 = row_str.count('1'), row_str.count('X'), row_str.count('2')
    s1_c, sx_c, s2_c, _ = get_streaks(row_str)
    g1_c, gx_c, g2_c, _ = get_gaps(row_str)
    si1_c, six_c, si2_c, _, _ = get_singles(row_str)
    d1_c, dx_c, d2_c, _, _ = get_doublets(row_str)
    t1_c, tx_c, t2_c, _, _ = get_triplets(row_str)
    o1_c, ox_c, o2_c, _, _ = get_occurrences(row_str)
    f_c, a_c, t_c, _ = get_fat(row_str, prob_vector)
    g_pass = 0
    b = bounds
    if sum([b['1'][0] <= c1 <= b['1'][1], b['X'][0] <= cx <= b['X'][1], b['2'][0] <= c2 <= b['2'][1]]) >= 2: g_pass += 1
    if sum([b['s1'][0] <= s1_c <= b['s1'][1], b['sx'][0] <= sx_c <= b['sx'][1], b['s2'][0] <= s2_c <= b['s2'][1]]) >= 2: g_pass += 1
    if sum([b['g1'][0] <= g1_c <= b['g1'][1], b['gx'][0] <= gx_c <= b['gx'][1], b['g2'][0] <= g2_c <= b['g2'][1]]) >= 2: g_pass += 1
    if sum([b['si1'][0] <= si1_c <= b['si1'][1], b['six'][0] <= six_c <= b['six'][1], b['si2'][0] <= si2_c <= b['si2'][1]]) >= 2: g_pass += 1
    if sum([b['d1'][0] <= d1_c <= b['d1'][1], b['dx'][0] <= dx_c <= b['dx'][1], b['d2'][0] <= d2_c <= b['d2'][1]]) >= 2: g_pass += 1
    if sum([b['t1'][0] <= t1_c <= b['t1'][1], b['tx'][0] <= tx_c <= b['tx'][1], b['t2'][0] <= t2_c <= b['t2'][1]]) >= 2: g_pass += 1
    if sum([b['o1'][0] <= o1_c <= b['o1'][1], b['ox'][0] <= ox_c <= b['ox'][1], b['o2'][0] <= o2_c <= b['o2'][1]]) >= 2: g_pass += 1
    if sum([b['f'][0] <= f_c <= b['f'][1], b['a'][0] <= a_c <= b['a'][1], b['t'][0] <= t_c <= b['t'][1]]) >= 2: g_pass += 1
    return g_pass >= req_groups

def classify_filter(hist_pct, keep_pct):
    if keep_pct is None or pd.isna(keep_pct):
        return "Info", None
    lift = hist_pct - keep_pct
    if lift >= 35 and keep_pct <= 65:
        return "Mycket stark", lift
    if lift >= 20 and keep_pct <= 75:
        return "Stark", lift
    if lift >= 10:
        return "OK", lift
    if lift >= 0:
        return "Svag", lift
    return "Irrationell", lift

def filter_strength_row(name, interval_text, hist_pct, keep_pct, module, hg_text, group="", active=True):
    cls, lift = classify_filter(hist_pct, keep_pct)
    reduction = None if keep_pct is None or pd.isna(keep_pct) else 100 - keep_pct
    return {
        "Aktiv": "Ja" if active else "Nej",
        "Filter": name,
        "Intervall": interval_text,
        "Grupp": group,
        "Historisk träff %": round(hist_pct, 1),
        "Kvar rad %": None if keep_pct is None or pd.isna(keep_pct) else round(keep_pct, 1),
        "Reducerar %": None if reduction is None else round(reduction, 1),
        "Rationell faktor": None if lift is None else round(lift, 1),
        "Klass": cls,
        "Helgardering-modul": module,
        "Helgardering-rad": hg_text
    }



def get_filter_family(filter_name, group_name=""):
    """Grov filterfamilj för revision/rensning."""
    name = str(filter_name).lower()
    grp = str(group_name).lower()
    if any(k in name for k in ["ai-matrix", "sft", "100-minus", "skrälltryck", "rank summa", "total diff", "poängfilter"]):
        return "Risk/sannolikhet"
    if any(k in name for k in ["fat", "favorittryck", "topp", "favorit-delta"]):
        return "FAT/favorit"
    if any(k in name for k in ["skrällstyrka"]):
        return "Skrällnivå"
    if any(k in name for k in ["u-rad", "utgångsrad", "utgångssystem", "utgångstips"]):
        return "Utgångssystem"
    if grp == "struktur" or any(k in name for k in ["tecken 1x2", "sviter", "följder", "teckenföljd", "teckenlucka", "luckor", "singlar", "dubbletter", "tripplar", "uppkomster"]):
        return "Struktur"
    if "super" in name or "grupp" in str(group_name).lower():
        return "Grupp/soft"
    return "Övrigt"


def get_filter_overlap_note(filter_name, family):
    name = str(filter_name).lower()
    if "favorit-delta" in name:
        return "Överlappar FAT/favoritbild. Använd helst som diagnos, inte som extra hårt filter."
    if family == "Risk/sannolikhet":
        return "Riskfilter mäter delvis samma sak. Kör helst som pro-grupp/softkrav i stället för många hårda AND-filter."
    if family == "FAT/favorit":
        return "Överlappar favorit-/andrahandsprofil. Välj bästa FAT/pro-grupp före flera separata hårda krav."
    if family == "Struktur":
        return "Strukturfilter är främst stöd. Bra i mjuk grupp, sällan som ensam huvudmotor."
    if family == "Skrällnivå":
        return "Ska balanseras mot SFT/Skrälltryck Log så att inte skrällkravet dubbelräknas."
    return ""


def build_filter_revision_df(diag_df):
    """Skapar beslutsvy: Använd / Reserv / Diagnos / Pausa."""
    if diag_df is None or diag_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in diag_df.iterrows():
        name = r.get("Filter", "")
        family = get_filter_family(name, r.get("Grupp", ""))
        klass = str(r.get("Klass", ""))
        hist = pd.to_numeric(r.get("Historisk träff %", None), errors="coerce")
        keep = pd.to_numeric(r.get("Kvar rad %", None), errors="coerce")
        rf = pd.to_numeric(r.get("Rationell faktor", None), errors="coerce")
        red = pd.to_numeric(r.get("Reducerar %", None), errors="coerce")

        lname = str(name).lower()
        if "favorit-delta" in lname:
            decision = "Diagnos"
            reason = "Dubblerar i praktiken favorit-/FAT-bilden. Behåll bara om den slår övriga tydligt."
        elif klass in ["Mycket stark", "Stark"]:
            decision = "Använd"
            reason = "Hög historisk träff i förhållande till kvarvarande radmassa."
        elif klass == "OK":
            decision = "Reserv"
            reason = "Kan användas som stödfilter eller i grupp, men inte förstaval."
        elif klass == "Svag":
            decision = "Diagnos"
            reason = "Reducerar för lite i förhållande till träff. Kör inte hårt utan stöd."
        elif klass == "Irrationell":
            decision = "Pausa"
            reason = "Sämre än radmassan. Filtret sparar för många/fel rader i denna körning."
        else:
            decision = "Info"
            reason = "Informationsfilter eller saknar radmasseberäkning."

        if pd.isna(rf):
            score = -999
        else:
            score = float(rf)
            if not pd.isna(hist):
                score += max(-10, min(10, float(hist) - 90)) * 0.35
            if not pd.isna(keep):
                score += max(-10, min(10, 60 - float(keep))) * 0.15
            if decision == "Pausa":
                score -= 50
            elif decision == "Diagnos":
                score -= 10
            elif decision == "Använd":
                score += 5

        rows.append({
            "Prioritetspoäng": round(score, 1) if score != -999 else None,
            "Beslut": decision,
            "Filterfamilj": family,
            "Filter": name,
            "Klass": klass,
            "Historisk träff %": None if pd.isna(hist) else round(float(hist), 1),
            "Kvar rad %": None if pd.isna(keep) else round(float(keep), 1),
            "Reducerar %": None if pd.isna(red) else round(float(red), 1),
            "Rationell faktor": None if pd.isna(rf) else round(float(rf), 1),
            "Dubblett-/risknotis": get_filter_overlap_note(name, family),
            "Kommentar": reason,
            "Helgardering-rad": r.get("Helgardering-rad", ""),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Beslut", "Prioritetspoäng"], ascending=[True, False], na_position="last")
    return out


def build_family_summary(revision_df):
    if revision_df is None or revision_df.empty:
        return pd.DataFrame()
    rows = []
    for fam, g in revision_df.groupby("Filterfamilj"):
        g2 = g.copy()
        g2["Rationell faktor"] = pd.to_numeric(g2["Rationell faktor"], errors="coerce")
        best = g2.sort_values("Rationell faktor", ascending=False, na_position="last").iloc[0]
        use_count = int((g2["Beslut"] == "Använd").sum())
        pause_count = int((g2["Beslut"] == "Pausa").sum())
        if use_count >= 2:
            action = "Bygg grupp / välj 1-2"
        elif use_count == 1:
            action = "Använd bästa"
        elif pause_count == len(g2):
            action = "Pausa familjen"
        else:
            action = "Diagnos/reserv"
        rows.append({
            "Filterfamilj": fam,
            "Antal filter": len(g2),
            "Använd": use_count,
            "Bäst filter": best.get("Filter", ""),
            "Bäst RF": best.get("Rationell faktor", None),
            "Rekommendation": action,
        })
    return pd.DataFrame(rows).sort_values("Bäst RF", ascending=False, na_position="last")


def build_starter_package(revision_df, max_filters=6):
    """Plockar ett litet startpaket: först bästa risk/FAT, därefter kompletterande stöd."""
    if revision_df is None or revision_df.empty:
        return pd.DataFrame()
    df = revision_df.copy()
    df["Rationell faktor"] = pd.to_numeric(df["Rationell faktor"], errors="coerce")
    usable = df[df["Beslut"].isin(["Använd", "Reserv"])].copy()
    if usable.empty:
        return pd.DataFrame()
    picks = []
    # Max 2 riskfilter, max 2 FAT/favorit, max 1 skräll, max 1 struktur/grupp.
    caps = {"Risk/sannolikhet": 2, "FAT/favorit": 2, "Skrällnivå": 1, "Struktur": 1, "Grupp/soft": 1, "Övrigt": 1}
    for fam, cap in caps.items():
        sub = usable[usable["Filterfamilj"] == fam].sort_values("Rationell faktor", ascending=False, na_position="last").head(cap)
        for _, r in sub.iterrows():
            picks.append(r)
    out = pd.DataFrame(picks).sort_values("Rationell faktor", ascending=False, na_position="last").head(max_filters)
    return out

def recommend_group_requirement(group_name, hist_scores, cand_scores, rule_names, rule_details=None, target_hist_pct=90.0):
    """
    Bakåtkompatibel grupprekommendation. Behålls för äldre delar av koden.
    V9 använder optimize_pro_group() för nya, tightare pro-grupper.
    """
    n_rules = len(rule_names)
    if n_rules == 0:
        return None, pd.DataFrame()
    if rule_details is None:
        rule_details = [f"{name}: -" for name in rule_names]
    detail_text = "\n".join([f"- {d}" for d in rule_details])
    detail_one_line = " | ".join(rule_details)

    hist_total = len(hist_scores)
    cand_total = len(cand_scores)
    rows = []
    max_req = max(1, n_rules - 1) if n_rules > 1 else 1
    for req in range(1, max_req + 1):
        hist_pass = sum(1 for s in hist_scores if s >= req)
        cand_pass = sum(1 for s in cand_scores if s >= req)
        hist_pct = (hist_pass / hist_total) * 100 if hist_total else 0.0
        keep_pct = (cand_pass / cand_total) * 100 if cand_total else 0.0
        cls, lift = classify_filter(hist_pct, keep_pct)
        rows.append({
            "Grupp": group_name,
            "Krav": f"{req} av {n_rules}",
            "Kravtal": req,
            "Antal filter": n_rules,
            "Tighthet %": None,
            "Historisk träff %": round(hist_pct, 1),
            "Kvar rad %": round(keep_pct, 1),
            "Reducerar %": round(100 - keep_pct, 1),
            "Rationell faktor": round(lift, 1) if lift is not None else None,
            "Klass": cls,
            "Ingående filter": ", ".join(rule_names),
            "Filterintervall": detail_one_line,
            "Helgardering-detaljer": detail_text,
            "Helgardering-rad": f"{group_name}: minst {req} av {n_rules} filter ska vara sanna"
        })

    valid = [r for r in rows if r["Historisk träff %"] >= target_hist_pct]
    if valid:
        best = max(valid, key=lambda r: (r["Reducerar %"], r["Rationell faktor"] or -999, r["Historisk träff %"]))
    else:
        best = max(rows, key=lambda r: (r["Historisk träff %"], r["Reducerar %"]))

    return best, pd.DataFrame(rows)


def _is_multi_value(v):
    return isinstance(v, (tuple, list, np.ndarray)) and not isinstance(v, (str, bytes))


def _build_value_interval(values, coverage):
    """Bygger ett eller flera intervall. Scalar => (min,max), tuple/list => [(min,max), ...]."""
    if not values:
        return (0, 0)
    first = values[0]
    if _is_multi_value(first):
        width = len(first)
        intervals = []
        for j in range(width):
            intervals.append(get_best_interval([v[j] for v in values], coverage))
        return intervals
    return get_best_interval(values, coverage)


def _value_in_interval(value, interval):
    if isinstance(interval, list):
        return all(in_range(value[j], interval[j]) for j in range(len(interval)))
    return in_range(value, interval)


def _fmt_value_interval(interval, decimals=0):
    if isinstance(interval, list):
        return " | ".join(fmt_interval(x, decimals) for x in interval)
    return fmt_interval(interval, decimals)


def make_group_spec(name, hist_values, cand_getter, formatter=None, decimals=0):
    """
    Spec för ett internt gruppfilter.
    hist_values: lista med historiska värden, scalar eller tuple per historisk rad.
    cand_getter: funktion(row_str) -> scalar eller tuple för radmassan.
    formatter: funktion(interval) -> Helgardering-text.
    """
    return {
        "name": name,
        "hist_values": list(hist_values),
        "cand_getter": cand_getter,
        "formatter": formatter,
        "decimals": decimals,
    }


def optimize_pro_group(group_name, specs, candidate_rows, target_hist_pct=90.0, coverage_steps=None):
    """
    V9: optimerar pro-grupper genom att testa tightare intervall + mjukare gruppkrav.
    Målet är hög reducering tillsammans med hög historisk träff.

    coverage_steps styr hur mycket av de liknande historiska vinnarraderna varje internt
    filter ensamt ska täcka. Lägre värde = tightare intervall.
    """
    specs = [s for s in specs if s and s.get("hist_values")]
    n_rules = len(specs)
    if n_rules == 0:
        return None, pd.DataFrame()

    if coverage_steps is None:
        # 50-85 räcker oftast bättre för gruppfilter än 90-100.
        coverage_steps = [50, 55, 60, 65, 70, 75, 80, 85, 90]

    hist_total = len(specs[0]["hist_values"])
    cand_total = len(candidate_rows)
    cand_values_by_spec = []
    for spec in specs:
        cand_values_by_spec.append([spec["cand_getter"](tr) for tr in candidate_rows])

    rows = []
    # Gruppfilter ska normalt inte bli "alla av alla". Då är det hårda filter.
    max_req = max(1, n_rules - 1) if n_rules > 1 else 1

    for coverage in coverage_steps:
        intervals_by_spec = [_build_value_interval(spec["hist_values"], coverage) for spec in specs]

        hist_scores = []
        for i in range(hist_total):
            pts = 0
            for spec, interval in zip(specs, intervals_by_spec):
                if _value_in_interval(spec["hist_values"][i], interval):
                    pts += 1
            hist_scores.append(pts)

        cand_scores = []
        for row_idx in range(cand_total):
            pts = 0
            for spec_idx, interval in enumerate(intervals_by_spec):
                if _value_in_interval(cand_values_by_spec[spec_idx][row_idx], interval):
                    pts += 1
            cand_scores.append(pts)

        detail_parts = []
        for spec, interval in zip(specs, intervals_by_spec):
            formatter = spec.get("formatter")
            if formatter:
                detail_parts.append(formatter(interval))
            else:
                detail_parts.append(f"{spec['name']}: {_fmt_value_interval(interval, spec.get('decimals', 0))}")
        detail_text = "\n".join([f"- {d}" for d in detail_parts])
        detail_one_line = " | ".join(detail_parts)

        for req in range(1, max_req + 1):
            hist_pass = sum(1 for s in hist_scores if s >= req)
            cand_pass = sum(1 for s in cand_scores if s >= req)
            hist_pct = (hist_pass / hist_total) * 100 if hist_total else 0.0
            keep_pct = (cand_pass / cand_total) * 100 if cand_total else 0.0
            reduction = 100 - keep_pct
            cls, lift = classify_filter(hist_pct, keep_pct)

            # Optimeringspoäng: reducering viktigast, men bara efter historisk träff.
            # Straffa extremt låga krav som ofta släpper igenom för mycket.
            req_ratio = req / n_rules if n_rules else 0
            balance_bonus = 8 if 0.45 <= req_ratio <= 0.80 else 0
            score = reduction + max(lift or 0, 0) * 0.45 + max(hist_pct - target_hist_pct, 0) * 0.08 + balance_bonus

            rows.append({
                "Grupp": group_name,
                "Krav": f"{req} av {n_rules}",
                "Kravtal": req,
                "Antal filter": n_rules,
                "Tighthet %": coverage,
                "Historisk träff %": round(hist_pct, 1),
                "Kvar rad %": round(keep_pct, 1),
                "Reducerar %": round(reduction, 1),
                "Rationell faktor": round(lift, 1) if lift is not None else None,
                "Klass": cls,
                "Optimeringspoäng": round(score, 2),
                "Ingående filter": ", ".join([s["name"] for s in specs]),
                "Filterintervall": detail_one_line,
                "Helgardering-detaljer": detail_text,
                "Helgardering-rad": f"{group_name}: minst {req} av {n_rules} filter ska vara sanna. Tighthet {coverage}%"
            })

    valid = [r for r in rows if r["Historisk träff %"] >= target_hist_pct]
    if valid:
        # Välj bästa reducering/rationell faktor, inte hårdaste krav.
        best = max(valid, key=lambda r: (r["Optimeringspoäng"], r["Reducerar %"], r["Rationell faktor"] or -999))
    else:
        # Om målet är för hårt: välj högst historisk träff, sedan bäst reducering.
        best = max(rows, key=lambda r: (r["Historisk träff %"], r["Reducerar %"], r["Rationell faktor"] or -999))

    return best, pd.DataFrame(rows)


def score_bool_columns(bool_columns, n_rows):
    """Summerar True/False-kolumner radvis."""
    if not bool_columns:
        return []
    scores = []
    for i in range(n_rows):
        scores.append(sum(1 for col in bool_columns if col[i]))
    return scores


def score_candidate_group(candidate_rows, predicates):
    """Räknar hur många villkor varje kandidat-rad klarar i en pro-grupp."""
    scores = []
    for tr in candidate_rows:
        scores.append(sum(1 for _, pred in predicates if pred(tr)))
    return scores

def build_database_quality_report(df, antal_matcher):
    report = []
    n = len(df)
    report.append(("Antal omgångar", n))
    if 'Datum' in df.columns:
        dates = pd.to_datetime(df['Datum'], errors='coerce')
        report.append(("Datum min", dates.min().date() if dates.notna().any() else "-"))
        report.append(("Datum max", dates.max().date() if dates.notna().any() else "-"))
        report.append(("Felaktiga datum", int(dates.isna().sum())))
    valid_row = df['Correct_Row'].astype(str).str.fullmatch(f"[1X2]{{{antal_matcher}}}") if 'Correct_Row' in df.columns else pd.Series([False]*n)
    report.append(("Ogiltiga Correct_Row", int((~valid_row).sum())))
    good_vec = df['Prob_Vector'].apply(lambda v: isinstance(v, list) and len(v) == antal_matcher * 3) if 'Prob_Vector' in df.columns else pd.Series([False]*n)
    report.append(("Ogiltiga procentvektorer", int((~good_vec).sum())))
    if 'True_Rank' in df.columns:
        ranks = pd.to_numeric(df['True_Rank'], errors='coerce')
        max_rank = 3**antal_matcher
        bad = ranks.isna() | (ranks < 1) | (ranks > max_rank)
        report.append(("Ogiltiga True_Rank", int(bad.sum())))
    if 'Payout' in df.columns:
        report.append(("Utdelning min", int(df['Payout'].min())))
        report.append(("Utdelning max", int(df['Payout'].max())))
    return pd.DataFrame(report, columns=["Kontroll", "Värde"])


# ==========================================
# 2. AUTO-LADDNING & DATABAS
# ==========================================

def find_local_database(spelform):
    mapp = str(Path(__file__).resolve().parent) if '__file__' in globals() else str(Path.cwd())
    alla_filer = [f for f in os.listdir(mapp) if f.endswith('.csv') or f.endswith('.xlsx')]
    def match(f, inc, exc):
        f_lower = f.lower()
        return all(w in f_lower for w in inc) and not any(w in f_lower for w in exc)
    kandidater = []
    if spelform == "Stryktips": kandidater = [f for f in alla_filer if match(f, ["stryk"], ["topp"])]
    elif spelform == "Europatips": kandidater = [f for f in alla_filer if match(f, ["europa"], ["topp"])]
    elif spelform == "Topptips ST": kandidater = [f for f in alla_filer if match(f, ["topp", "st"], [])]
    elif spelform == "Topptips EU": kandidater = [f for f in alla_filer if match(f, ["topp", "eu"], [])]
    elif spelform == "Topptips Övrigt": kandidater = [f for f in alla_filer if match(f, ["topp", "övrig"], [])]
    elif spelform == "Powerplay": kandidater = [f for f in alla_filer if match(f, ["powerplay"], [])]

    if not kandidater: return None
    for k in kandidater:
        if "med_rank" in k.lower(): return os.path.join(mapp, k)
    return os.path.join(mapp, kandidater[0])

@st.cache_data
def load_database(filepath, antal_matcher):
    if filepath.endswith('.xlsx'): global_db = pd.read_excel(filepath)
    else:
        try:
            with open(filepath, 'r', encoding='utf-8') as f: first_line = f.readline()
            sep = ';' if ';' in first_line else ','
            if '\t' in first_line and ';' not in first_line and ',' not in first_line: sep = '\t'
            global_db = pd.read_csv(filepath, sep=sep, encoding='utf-8', on_bad_lines='skip')
        except:
            with open(filepath, 'r', encoding='latin-1') as f: first_line = f.readline()
            sep = ';' if ';' in first_line else ','
            if '\t' in first_line and ';' not in first_line and ',' not in first_line: sep = '\t'
            global_db = pd.read_csv(filepath, sep=sep, encoding='latin-1', on_bad_lines='skip')

    global_db.columns = [str(c).strip() for c in global_db.columns]

    col_m = [f'M{i}' for i in range(1, antal_matcher + 1)]
    if all(c in global_db.columns for c in col_m):
        def safe_join(row):
            try: return ''.join([str(x).replace('.0', '').strip().upper() for x in row])
            except: return ""
        global_db['Correct_Row'] = global_db[col_m].apply(safe_join, axis=1)

    prob_vectors = []
    valid_rows = []
    for idx, row in global_db.iterrows():
        try:
            p_vec = []
            for m in range(1, antal_matcher + 1):
                p1 = float(str(row[f'M{m}-1']).replace(',', '.'))
                px = float(str(row[f'M{m}-X']).replace(',', '.'))
                p2 = float(str(row[f'M{m}-2']).replace(',', '.'))
                p_vec.extend([p1, px, p2])
            prob_vectors.append(p_vec)
            valid_rows.append(True)
        except:
            prob_vectors.append([])
            valid_rows.append(False)

    global_db['Prob_Vector'] = prob_vectors
    
    payout_col = None
    for col in global_db.columns:
        clean_col = str(col).lower().replace(' ', '').replace('_', '').replace('-', '')
        if f"{antal_matcher}rätt" in clean_col or clean_col == str(antal_matcher) or "utdelning" in clean_col or "vinst" in clean_col:
            payout_col = col
            break
            
    if payout_col: 
        raw_payout = global_db[payout_col].astype(str)
        clean_payout = raw_payout.str.replace(r'[,.]00', '', regex=True)
        clean_payout = clean_payout.str.replace(r'[^\d]', '', regex=True).replace('', '0')
        global_db['Payout'] = pd.to_numeric(clean_payout, errors='coerce').fillna(0)
    else: 
        global_db['Payout'] = 0
        
    return global_db[valid_rows]


# ==========================================
# 2B. STRECKBASERADE REKOMMENDATIONER
# ==========================================

SIGN_ORDER = ['1', 'X', '2']
PAIR_ORDER = ['1X', '12', 'X2']


def _current_match_pct_dict(filter_vec, match_idx):
    """Aktuella streck/procent för en match som dict 1/X/2."""
    idx = int(match_idx) * 3
    vals = list(filter_vec[idx:idx+3]) if filter_vec is not None else []
    while len(vals) < 3:
        vals.append(0.0)
    return {'1': float(vals[0]), 'X': float(vals[1]), '2': float(vals[2])}


def _pct_rank(vals, sign):
    order = sorted(SIGN_ORDER, key=lambda s: float(vals.get(s, 0.0)), reverse=True)
    try:
        return order.index(sign) + 1
    except Exception:
        return 3


def _empirical_streck_rate(df, signs, target_pct, antal_matcher, min_samples=25, start_band=4, require_same_rank=None):
    """Historisk träfffrekvens för ett tecken/par vid ungefär samma strecknivå.

    signs kan vara ['1'] eller ['1','X']. För varje historisk match jämförs
    den historiska strecknivån för samma tecken/par med aktuell target_pct.
    Bandet breddas stegvis tills stickprovet är tillräckligt stort.
    """
    signs = normalize_signs(signs)
    if not signs or df is None or len(df) == 0:
        return {'hit_pct': 0.0, 'samples': 0, 'expected_pct': 0.0, 'lift_pct': 0.0, 'band': None}
    try:
        target_pct = float(target_pct)
    except Exception:
        target_pct = 0.0

    # Bredare fallback behövs främst för liknande-urvalet, eftersom 30 omgångar
    # bara ger 390 matchobservationer på Stryktipset.
    bands = []
    for b in [start_band, 6, 8, 10, 12, 15, 20, 30, 100]:
        if b not in bands:
            bands.append(b)

    best = {'hit_pct': 0.0, 'samples': 0, 'expected_pct': 0.0, 'lift_pct': 0.0, 'band': None}
    for band in bands:
        hits = 0
        samples = 0
        exp_sum = 0.0
        for _, row in df.iterrows():
            corr = normalize_single_row_text(row.get('Correct_Row', ''))
            pvec = row.get('Prob_Vector', [])
            if len(corr) != int(antal_matcher) or not isinstance(pvec, list) or len(pvec) < int(antal_matcher) * 3:
                continue
            for m in range(int(antal_matcher)):
                vals = {'1': float(pvec[m*3]), 'X': float(pvec[m*3+1]), '2': float(pvec[m*3+2])}
                hist_pct = float(sum(vals.get(s, 0.0) for s in signs))
                if abs(hist_pct - target_pct) > float(band):
                    continue
                if require_same_rank is not None and len(signs) == 1:
                    if _pct_rank(vals, signs[0]) != int(require_same_rank):
                        continue
                samples += 1
                exp_sum += hist_pct
                if corr[m] in signs:
                    hits += 1
        if samples > 0:
            hit_pct = 100.0 * hits / samples
            expected_pct = exp_sum / samples
            best = {
                'hit_pct': round(hit_pct, 1),
                'samples': int(samples),
                'expected_pct': round(expected_pct, 1),
                'lift_pct': round(hit_pct - expected_pct, 1),
                'band': int(band),
            }
        if samples >= int(min_samples) or band == 100:
            return best
    return best


def _blend_hist_rate(sim_meta, all_meta):
    """Vägd historikträff: liknande omgångar väger mest, full historik stabiliserar."""
    s_n = float(sim_meta.get('samples', 0) or 0)
    a_n = float(all_meta.get('samples', 0) or 0)
    if s_n <= 0 and a_n <= 0:
        return 0.0
    # Liknande historik får hög vikt men begränsas av stickprovsstorlek.
    sim_weight = 0.65 * min(1.0, s_n / 20.0)
    all_weight = 0.35 * min(1.0, a_n / 60.0)
    if sim_weight + all_weight <= 0:
        return 0.0
    return (float(sim_meta.get('hit_pct', 0.0)) * sim_weight + float(all_meta.get('hit_pct', 0.0)) * all_weight) / (sim_weight + all_weight)


def _recommendation_label(kind, current_pct, sim_meta, all_meta):
    """Kort manuell varnings-/styrtext, inte ett automatiskt beslut."""
    hit = _blend_hist_rate(sim_meta, all_meta)
    lift = float(all_meta.get('lift_pct', 0.0) or 0.0)
    if kind == 'spik':
        if hit >= current_pct - 3 and float(sim_meta.get('hit_pct', 0.0)) >= current_pct - 6:
            return 'Stark'
        if hit >= current_pct - 8:
            return 'OK'
        return 'Varning'
    if kind == 'halv':
        if hit >= current_pct - 2:
            return 'Stark'
        if hit >= current_pct - 7:
            return 'OK'
        return 'Varning'
    # Skräll: positiv historisk lift är viktigare än absolut träff.
    if lift >= 4 or hit >= current_pct + 4:
        return 'Värde'
    if lift >= 0:
        return 'Möjlig'
    return 'Tunn'


def _recommendation_score(kind, current_pct, sim_meta, all_meta):
    hist = _blend_hist_rate(sim_meta, all_meta)
    lift = float(all_meta.get('lift_pct', 0.0) or 0.0)
    sim_n = min(1.0, float(sim_meta.get('samples', 0) or 0) / 20.0)
    all_n = min(1.0, float(all_meta.get('samples', 0) or 0) / 60.0)
    reliability = 0.55 * sim_n + 0.45 * all_n
    if kind == 'spik':
        return hist * 0.55 + float(current_pct) * 0.35 + max(lift, -20) * 0.10 + reliability * 5
    if kind == 'halv':
        return hist * 0.58 + float(current_pct) * 0.32 + max(lift, -20) * 0.10 + reliability * 5
    # Skrällar rankas inte bara på högst sannolikhet; historisk överprestation/lift ska synas.
    return hist * 0.42 + max(lift, -20) * 1.35 + float(current_pct) * 0.12 + reliability * 4


def _format_band(meta):
    b = meta.get('band')
    if b is None:
        return '-'
    if int(b) >= 100:
        return 'alla'
    return f'±{int(b)} pp'


def build_streck_recommendation_tables(filter_vec, hist_df, similar_df, antal_matcher, max_shock_pct=22):
    """Bygger 5 bästa spikar, halvor och skrällar mot aktuell streckbild.

    Princip:
    - Kandidaterna skapas från aktuell streck/procent.
    - Träfffrekvensen kontrolleras historiskt mot liknande strecknivåer.
    - Både de mest liknande omgångarna och hela statistikfilen visas.
    """
    if filter_vec is None or len(filter_vec) < int(antal_matcher) * 3:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    hist_df = hist_df if hist_df is not None else pd.DataFrame()
    similar_df = similar_df if similar_df is not None else pd.DataFrame()

    spik_rows = []
    half_rows = []
    shock_rows = []
    for m in range(int(antal_matcher)):
        vals = _current_match_pct_dict(filter_vec, m)
        order = sorted(SIGN_ORDER, key=lambda s: vals[s], reverse=True)

        # Spik: bästa enkeltecken enligt aktuell streckbild, verifierat historiskt.
        fav = order[0]
        fav_pct = vals[fav]
        sim = _empirical_streck_rate(similar_df, [fav], fav_pct, antal_matcher, min_samples=8, start_band=4, require_same_rank=1)
        allh = _empirical_streck_rate(hist_df, [fav], fav_pct, antal_matcher, min_samples=35, start_band=3, require_same_rank=1)
        spik_rows.append({
            '_score': _recommendation_score('spik', fav_pct, sim, allh),
            'Match': f'M{m+1}',
            'Spik': fav,
            'Streck %': round(fav_pct, 1),
            'Liknande träff %': sim['hit_pct'],
            'Liknande n': sim['samples'],
            'Liknande band': _format_band(sim),
            'Historik träff %': allh['hit_pct'],
            'Historik n': allh['samples'],
            'Hist. lift %': allh['lift_pct'],
            'Rek': _recommendation_label('spik', fav_pct, sim, allh),
        })

        # Halva: testa alla 3 halvgarderingar, men visa bara bästa per match
        # så listan inte fylls av flera nästan identiska alternativ från samma match.
        match_halves = []
        for pair in PAIR_ORDER:
            signs = list(pair)
            pair_pct = sum(vals[s] for s in signs)
            sim = _empirical_streck_rate(similar_df, signs, pair_pct, antal_matcher, min_samples=10, start_band=4)
            allh = _empirical_streck_rate(hist_df, signs, pair_pct, antal_matcher, min_samples=45, start_band=3)
            match_halves.append({
                '_score': _recommendation_score('halv', pair_pct, sim, allh),
                'Match': f'M{m+1}',
                'Halv': pair,
                'Strecksumma %': round(pair_pct, 1),
                'Liknande träff %': sim['hit_pct'],
                'Liknande n': sim['samples'],
                'Liknande band': _format_band(sim),
                'Historik träff %': allh['hit_pct'],
                'Historik n': allh['samples'],
                'Hist. lift %': allh['lift_pct'],
                'Rek': _recommendation_label('halv', pair_pct, sim, allh),
            })
        if match_halves:
            half_rows.append(max(match_halves, key=lambda r: r['_score']))

        # Skräll: icke-favorittecken under vald maxgräns.
        for s in order[1:]:
            p = vals[s]
            if p > float(max_shock_pct):
                continue
            rank = _pct_rank(vals, s)
            sim = _empirical_streck_rate(similar_df, [s], p, antal_matcher, min_samples=6, start_band=4, require_same_rank=rank)
            allh = _empirical_streck_rate(hist_df, [s], p, antal_matcher, min_samples=25, start_band=3, require_same_rank=rank)
            shock_rows.append({
                '_score': _recommendation_score('skrall', p, sim, allh),
                'Match': f'M{m+1}',
                'Skräll': s,
                'Rank': int(rank),
                'Streck %': round(p, 1),
                'Liknande träff %': sim['hit_pct'],
                'Liknande n': sim['samples'],
                'Liknande band': _format_band(sim),
                'Historik träff %': allh['hit_pct'],
                'Historik n': allh['samples'],
                'Hist. lift %': allh['lift_pct'],
                'Rek': _recommendation_label('skrall', p, sim, allh),
            })

    def _finalize(rows, n=5):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values('_score', ascending=False).head(int(n)).reset_index(drop=True)
        return df.drop(columns=['_score'], errors='ignore')

    return _finalize(spik_rows, 5), _finalize(half_rows, 5), _finalize(shock_rows, 5)


def _recommendation_frame_text(spik_df, half_df, shock_df, antal_matcher):
    """Bygger en enkel text som kan användas som stöd för grundramen."""
    frame = {i: set() for i in range(1, int(antal_matcher) + 1)}
    if isinstance(spik_df, pd.DataFrame) and not spik_df.empty:
        for _, r in spik_df.iterrows():
            try:
                m = int(str(r['Match']).replace('M', ''))
                frame[m].add(str(r['Spik']))
            except Exception:
                pass
    if isinstance(half_df, pd.DataFrame) and not half_df.empty:
        for _, r in half_df.iterrows():
            try:
                m = int(str(r['Match']).replace('M', ''))
                for s in str(r['Halv']):
                    if s in SIGN_ORDER:
                        frame[m].add(s)
            except Exception:
                pass
    if isinstance(shock_df, pd.DataFrame) and not shock_df.empty:
        for _, r in shock_df.iterrows():
            try:
                m = int(str(r['Match']).replace('M', ''))
                frame[m].add(str(r['Skräll']))
            except Exception:
                pass
    parts = []
    for m in range(1, int(antal_matcher) + 1):
        signs = normalize_signs(list(frame[m]))
        parts.append(_sort_signs_display(signs) if signs else '-')
    return ' / '.join(parts)


STRECK_U_SOURCES = ['5 bästa spikar', '5 bästa halvor', '5 bästa skrällar']


def _build_streck_filter_systems(filter_vec, hist_df, similar_df, antal_matcher, max_shock_pct=22):
    """Skapar tre färdiga filter från aktuell streckbild, verifierade mot historiken.

    Filtren används direkt i Favorit & skräll. De är inte U-system i UI:t:
    värdet är antal av de fem föreslagna markeringarna som enkelraden klarar.
    """
    hist_src = hist_df if isinstance(hist_df, pd.DataFrame) and not hist_df.empty else similar_df
    spik_df, half_df, shock_df = build_streck_recommendation_tables(
        filter_vec,
        hist_src,
        similar_df,
        int(antal_matcher),
        max_shock_pct=int(max_shock_pct or 22),
    )
    items = []
    for name, df, col, key in [
        ('5 bästa spikar', spik_df, 'Spik', 'streck_5_basta_spikar'),
        ('5 bästa halvor', half_df, 'Halv', 'streck_5_basta_halvor'),
        ('5 bästa skrällar', shock_df, 'Skräll', 'streck_5_basta_skrallar'),
    ]:
        system = _system_from_recommendation_df(df, col, antal_matcher)
        marked = u_system_marked_count(system)
        if marked <= 0:
            continue
        details = u_system_to_text(system)
        compact = _compact_rec_preview_df(df, name)
        if isinstance(compact, pd.DataFrame) and not compact.empty:
            preview = '; '.join([f"{r.get('Match','')} {r.get('Val','')}" for _, r in compact.iterrows()])
        else:
            preview = details
        items.append({
            'name': name,
            'key': key,
            'system': system,
            'marked': marked,
            'details': details,
            'preview': preview,
            'table': df,
        })
    return items


# --- MANUELLA TECKENGRUPPER / TECKENBLOCK ---
MANUAL_GROUP_SLOTS = 8


def _manual_choice_sort_key(choice):
    try:
        return int(choice.get('match', 999))
    except Exception:
        return 999


def _normalize_sign_list(signs):
    """Normaliserar en lista/sträng med tecken till Helgardering-ordning."""
    if signs is None:
        return []
    if isinstance(signs, str):
        raw = [ch.upper() for ch in signs if ch.upper() in {'1', 'X', '2'}]
    else:
        raw = [str(s).strip().upper() for s in signs]
    return [s for s in ['1', 'X', '2'] if s in raw]


def _manual_group_choices_from_group(group, antal_matcher=None):
    """Returnerar gruppens matchval: [{'match': 3, 'signs': ['1','X']}].

    Bakåtkompatibel med gamla formatet `picks` där varje post var M:tecken.
    Ny tolkning: ett matchval kan innehålla flera tecken, men räknas max som
    1 träff eftersom en match bara kan sluta på ett tecken.
    """
    by_match = {}
    raw_choices = group.get('choices', None) if isinstance(group, dict) else None
    if raw_choices:
        for c in raw_choices or []:
            try:
                m = int(c.get('match', 0))
            except Exception:
                continue
            signs = _normalize_sign_list(c.get('signs', []))
            if not signs:
                continue
            if antal_matcher is not None and not (1 <= m <= int(antal_matcher)):
                continue
            if m not in by_match:
                by_match[m] = set()
            by_match[m].update(signs)
    else:
        for p in (group.get('picks', []) if isinstance(group, dict) else []) or []:
            try:
                m = int(p.get('match', 0))
                sign = str(p.get('sign', '')).upper()
            except Exception:
                continue
            if sign not in {'1', 'X', '2'}:
                continue
            if antal_matcher is not None and not (1 <= m <= int(antal_matcher)):
                continue
            if m not in by_match:
                by_match[m] = set()
            by_match[m].add(sign)
    out = []
    for m in sorted(by_match):
        signs = [s for s in ['1', 'X', '2'] if s in by_match[m]]
        if signs:
            out.append({'match': int(m), 'signs': signs})
    return out


def _manual_group_choices_to_picks(choices):
    picks = []
    for c in choices or []:
        try:
            m = int(c.get('match', 0))
        except Exception:
            continue
        for s in _normalize_sign_list(c.get('signs', [])):
            picks.append({'match': m, 'sign': s})
    return picks


def _manual_group_choices_to_text(choices):
    parts = []
    for c in sorted(choices or [], key=_manual_choice_sort_key):
        try:
            m = int(c.get('match', 0))
        except Exception:
            continue
        signs = ''.join(_normalize_sign_list(c.get('signs', [])))
        if m > 0 and signs:
            parts.append(f"M{m}:{signs}")
    return ", ".join(parts)


def _manual_sign_groups_signature(groups):
    """Stabil signatur så paketmotorn kan se om teckengrupper ändrats."""
    clean = []
    for g in groups or []:
        if not isinstance(g, dict):
            continue
        choices = _manual_group_choices_from_group(g)
        clean.append({
            'active': bool(g.get('active', False)),
            'name': str(g.get('name', '')),
            'choices': [
                {'match': int(c.get('match', 0)), 'signs': _normalize_sign_list(c.get('signs', []))}
                for c in choices
            ],
            'min': int(g.get('min', 0) or 0),
            'max': int(g.get('max', len(choices)) if g.get('max', None) is not None else len(choices)),
        })
    return json.dumps(clean, ensure_ascii=False, sort_keys=True)


def _manual_group_picks_to_text(picks):
    """Gammal textvisning, kvar för bakåtkompatibilitet."""
    parts = []
    for p in picks or []:
        try:
            m = int(p.get('match', 0))
            sg = str(p.get('sign', '')).upper()
            if m > 0 and sg in {'1', 'X', '2'}:
                parts.append(f"M{m}:{sg}")
        except Exception:
            continue
    return ", ".join(parts)


def _parse_manual_group_picks(text, antal_matcher):
    """Läser t.ex. 'M3:2, M10:1' eller '3:2 10=1'.

    Stöder även flera tecken per match, t.ex. M3:1X eller M10:X2.
    Returnerar fortfarande `picks` för gammal kompatibilitet.
    """
    raw = str(text or '').upper().replace('×', 'X')
    tokens = re.findall(r'(?:MATCH\s*)?M?\s*(\d{1,2})\s*[:=\-]?\s*([1X2]{1,3})', raw)
    picks = []
    warnings = []
    used_pairs = set()
    for m_txt, signs_txt in tokens:
        try:
            m = int(m_txt)
        except Exception:
            continue
        if not (1 <= m <= int(antal_matcher)):
            warnings.append(f"M{m} finns inte på kupongen.")
            continue
        signs = _normalize_sign_list(signs_txt)
        for sign in signs:
            pair = (m, sign)
            if pair in used_pairs:
                continue
            used_pairs.add(pair)
            picks.append({'match': int(m), 'sign': sign})
    return picks, warnings


def _normalize_manual_sign_groups(groups, antal_matcher):
    out = []
    for idx, g in enumerate(groups or [], 1):
        if not isinstance(g, dict):
            continue
        choices = _manual_group_choices_from_group(g, antal_matcher)
        n = len(choices)
        mn = int(g.get('min', 0) or 0)
        mx = int(g.get('max', n) if g.get('max', None) is not None else n)
        mn = max(0, min(n, mn))
        mx = max(mn, min(n, mx))
        picks = _manual_group_choices_to_picks(choices)
        out.append({
            'name': str(g.get('name') or f'Teckengrupp {idx}'),
            'active': bool(g.get('active', False)) and n > 0,
            'text': _manual_group_choices_to_text(choices),
            'choices': choices,
            'picks': picks,
            'min': mn,
            'max': mx,
        })
    return out


def _manual_group_choice_count(group):
    return len(_manual_group_choices_from_group(group))


def _manual_group_hit_count(row_str, group):
    row_str = normalize_single_row_text(row_str)
    hits = 0
    for c in _manual_group_choices_from_group(group):
        try:
            m = int(c.get('match', 0))
            signs = set(_normalize_sign_list(c.get('signs', [])))
            if 1 <= m <= len(row_str) and row_str[m - 1] in signs:
                hits += 1
        except Exception:
            continue
    return int(hits)


def _manual_group_pass(row_str, group):
    if not group.get('active') or not _manual_group_choices_from_group(group):
        return True
    h = _manual_group_hit_count(row_str, group)
    n = _manual_group_choice_count(group)
    return int(group.get('min', 0)) <= h <= int(group.get('max', n))


def _manual_sign_groups_pass(row_str, groups):
    for g in groups or []:
        if g.get('active') and not _manual_group_pass(row_str, g):
            return False
    return True


def _manual_sign_groups_hist_mask(v_m, groups, antal_matcher):
    rows = []
    try:
        rows = [normalize_single_row_text(r) for r in list(v_m.get('Correct_Row', []))]
    except Exception:
        rows = []
    mask = []
    active = [g for g in (groups or []) if g.get('active') and _manual_group_choices_from_group(g)]
    for r in rows:
        if len(r) != int(antal_matcher):
            mask.append(False)
        elif not active:
            mask.append(True)
        else:
            mask.append(_manual_sign_groups_pass(r, active))
    return np.array(mask, dtype=bool)


def _apply_manual_sign_groups_to_rows(rows, groups, antal_matcher):
    active = [g for g in (groups or []) if g.get('active') and _manual_group_choices_from_group(g)]
    clean = [normalize_single_row_text(r) for r in (rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    if not active:
        return clean
    return [r for r in clean if _manual_sign_groups_pass(r, active)]


def _manual_group_probability_distribution(group, filter_vec):
    """Poisson-binomial över valda matchval, baserat på dagens streck.

    Om ett matchval innehåller flera tecken, t.ex. M3:1X, blir sannolikheten
    summan av dessa tecken i matchen. Matchvalet räknas max som 1 träff.
    """
    probs = []
    for c in _manual_group_choices_from_group(group):
        try:
            m = int(c.get('match', 0))
            p_sum = 0.0
            for sign in _normalize_sign_list(c.get('signs', [])):
                idx = (m - 1) * 3 + {'1': 0, 'X': 1, '2': 2}[sign]
                if 0 <= idx < len(filter_vec):
                    p_sum += max(0.0, min(1.0, float(filter_vec[idx]) / 100.0))
            probs.append(max(0.0, min(1.0, p_sum)))
        except Exception:
            continue
    dist = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for k, val in enumerate(dist):
            nxt[k] += val * (1.0 - p)
            nxt[k + 1] += val * p
        dist = nxt
    return dist


def _manual_group_probability_pass_pct(group, filter_vec):
    dist = _manual_group_probability_distribution(group, filter_vec)
    mn = int(group.get('min', 0) or 0)
    mx = int(group.get('max', len(dist) - 1) or 0)
    return round(100.0 * sum(dist[k] for k in range(max(0, mn), min(len(dist) - 1, mx) + 1)), 1) if dist else 0.0


def _manual_group_hist_summary(group, v_m, antal_matcher):
    vals = []
    try:
        rows = [normalize_single_row_text(r) for r in list(v_m.get('Correct_Row', []))]
    except Exception:
        rows = []
    for r in rows:
        if len(r) == int(antal_matcher):
            vals.append(_manual_group_hit_count(r, group))
    if not vals:
        return 0, 0, '—'
    n = _manual_group_choice_count(group)
    mn = int(group.get('min', 0) or 0)
    mx = int(group.get('max', n) or 0)
    hit = sum(1 for v in vals if mn <= v <= mx)
    parts = [f"{i}:{vals.count(i)}" for i in range(0, max(n, max(vals)) + 1)]
    return int(hit), int(len(vals)), ' | '.join(parts)


def _manual_group_frame_diagnostics(group, frame, frame_rows, antal_matcher):
    choices = _manual_group_choices_from_group(group, antal_matcher)
    possible = []
    missing = []
    for c in choices:
        try:
            m = int(c.get('match', 0))
            signs = _normalize_sign_list(c.get('signs', []))
            frame_signs = normalize_signs(frame[m - 1]) if frame and 1 <= m <= len(frame) else []
            possible_signs = [s for s in signs if s in frame_signs]
            missing_signs = [s for s in signs if s not in frame_signs]
            if possible_signs:
                possible.append(f"M{m}:{''.join(possible_signs)}")
            if missing_signs:
                missing.append(f"M{m}:{''.join(missing_signs)}")
        except Exception:
            continue
    mn = int(group.get('min', 0) or 0)
    notes = []
    if missing:
        notes.append("saknas i grundram: " + ', '.join(missing))
    if len(possible) < mn:
        notes.append(f"omöjligt: bara {len(possible)} möjliga matchval, men min är {mn}")
    elif mn > 0 and len(possible) == mn:
        if len(possible) == 1:
            notes.append("dolt tvång: " + possible[0])
        else:
            notes.append("dolt tvång: alla dessa matchval måste sitta: " + ', '.join(possible))
    keep = None
    total = None
    if frame_rows is not None:
        total = len(frame_rows)
        try:
            keep = sum(1 for r in frame_rows if _manual_group_pass(r, group))
        except Exception:
            keep = None
    return possible, missing, notes, keep, total


def _manual_sign_groups_summary_df(groups, v_m, filter_vec, frame, frame_rows, antal_matcher):
    rows = []
    for idx, g in enumerate(groups or [], 1):
        if not g.get('active') or not _manual_group_choices_from_group(g):
            continue
        h, t, dist = _manual_group_hist_summary(g, v_m, antal_matcher)
        prob = _manual_group_probability_pass_pct(g, filter_vec)
        possible, missing, notes, keep, total = _manual_group_frame_diagnostics(g, frame, frame_rows, antal_matcher)
        n = _manual_group_choice_count(g)
        rows.append({
            'Grupp': g.get('name') or f'Teckengrupp {idx}',
            'Tecken': _manual_group_choices_to_text(_manual_group_choices_from_group(g)),
            'Krav': f"{int(g.get('min',0))}–{int(g.get('max',0))} av {n}",
            'Historikträff': f"{h}/{t}" if t else '—',
            'Strecksannolikhet': f"{prob:.1f}%",
            'Grundram': (f"{keep}/{total}" if keep is not None and total is not None else '—'),
            'Fördelning hist': dist,
            'Diagnos': '; '.join(notes) if notes else 'OK',
        })
    return pd.DataFrame(rows)


def _render_manual_choice_row(group_idx, choice_idx, saved_choice, antal_matcher):
    """Renderar ett matchval: rullgardin för match + kryssrutor 1/X/2."""
    c_match, c1, cx, c2 = st.columns([1.6, 0.6, 0.6, 0.6])
    try:
        saved_m = int((saved_choice or {}).get('match', 0) or 0)
    except Exception:
        saved_m = 0
    saved_signs = set(_normalize_sign_list((saved_choice or {}).get('signs', [])))
    with c_match:
        m = st.selectbox(
            f'Val {choice_idx + 1}: match',
            options=list(range(0, int(antal_matcher) + 1)),
            index=max(0, min(int(antal_matcher), saved_m)),
            format_func=lambda x: '—' if int(x) == 0 else f'M{x}',
            key=f'v12_mtg_choice_match_{group_idx}_{choice_idx}',
        )
    with c1:
        s1 = st.checkbox('1', value=('1' in saved_signs), key=f'v12_mtg_choice_1_{group_idx}_{choice_idx}')
    with cx:
        sx = st.checkbox('X', value=('X' in saved_signs), key=f'v12_mtg_choice_x_{group_idx}_{choice_idx}')
    with c2:
        s2 = st.checkbox('2', value=('2' in saved_signs), key=f'v12_mtg_choice_2_{group_idx}_{choice_idx}')
    signs = []
    if s1: signs.append('1')
    if sx: signs.append('X')
    if s2: signs.append('2')
    return {'match': int(m), 'signs': signs}


def _render_manual_sign_groups_panel(v_m, filter_vec, frame, frame_rows, antal_matcher):
    """Renderar manuella teckengrupper som en minimerbar panel.

    Teckengrupperna är fortfarande ett förfilter före paketmotorn, men UI:t
    är stängt som standard så filtercentralen inte blir onödigt lång.
    """
    saved = _normalize_manual_sign_groups(st.session_state.get('v12_manual_sign_groups', []), antal_matcher)
    # Säkerställ 6 synliga slots när panelen öppnas.
    while len(saved) < 6:
        saved.append({'name': f'Teckengrupp {len(saved)+1}', 'active': False, 'text': '', 'choices': [], 'picks': [], 'min': 0, 'max': 0})

    active_groups_preview = [g for g in saved if g.get('active') and _manual_group_choices_from_group(g, antal_matcher)]
    if active_groups_preview:
        label = f"🎯 Manuella teckengrupper före paket · {len(active_groups_preview)} aktiva"
    else:
        label = "🎯 Manuella teckengrupper före paket · inga aktiva"

    with st.expander(label, expanded=False):
        st.markdown("<div class='v12-card'>", unsafe_allow_html=True)
        st.markdown("<div class='v12-step'>Steg 3A</div><div class='v12-title'>Manuella teckengrupper före paket</div>", unsafe_allow_html=True)
        st.caption("Används som förfilter före rekommenderade filterpaket. Välj match i rullgardinen och kryssa i 1/X/2. Ett matchval räknas max som 1 träff, även om du kryssar flera tecken på samma match.")

        with st.form('v12_manual_sign_groups_form'):
            draft_rows = []
            for i in range(6):
                g = saved[i]
                choices = _manual_group_choices_from_group(g, antal_matcher)
                while len(choices) < MANUAL_GROUP_SLOTS:
                    choices.append({'match': 0, 'signs': []})
                expanded = bool(g.get('active')) or bool(_manual_group_choices_from_group(g, antal_matcher))
                with st.expander(f"Teckengrupp {i+1}: {g.get('name') or f'Teckengrupp {i+1}'}", expanded=expanded):
                    c0, c1, c2, c3 = st.columns([0.8, 2.2, 0.8, 0.8])
                    with c0:
                        active = st.checkbox('Aktiv', value=bool(g.get('active', False)), key=f'v12_mtg_active_{i}')
                    with c1:
                        name = st.text_input('Namn', value=str(g.get('name') or f'Teckengrupp {i+1}'), key=f'v12_mtg_name_{i}')
                    with c2:
                        mn = st.number_input('Min', min_value=0, max_value=int(antal_matcher), value=int(g.get('min', 0) or 0), step=1, key=f'v12_mtg_min_{i}')
                    with c3:
                        mx_default = int(g.get('max', max(0, _manual_group_choice_count(g))) or 0)
                        mx = st.number_input('Max', min_value=0, max_value=int(antal_matcher), value=max(0, min(int(antal_matcher), mx_default)), step=1, key=f'v12_mtg_max_{i}')

                    st.caption('Välj upp till 8 matchval. Exempel: M3 med 2 ikryssad + M10 med 1 ikryssad och krav 1–2.')
                    group_choices = []
                    for j in range(MANUAL_GROUP_SLOTS):
                        ch = _render_manual_choice_row(i, j, choices[j], antal_matcher)
                        group_choices.append(ch)
                    draft_rows.append({'active': active, 'name': name, 'choices': group_choices, 'min': mn, 'max': mx})
            submitted = st.form_submit_button('💾 Spara manuella teckengrupper', use_container_width=True)

        if submitted:
            new_groups = []
            all_warnings = []
            for i, row in enumerate(draft_rows, 1):
                by_match = {}
                for ch in row.get('choices', []) or []:
                    try:
                        m = int(ch.get('match', 0) or 0)
                    except Exception:
                        m = 0
                    signs = _normalize_sign_list(ch.get('signs', []))
                    if m <= 0 and signs:
                        all_warnings.append(f"Teckengrupp {i}: tecken är ikryssade utan vald match och ignoreras.")
                        continue
                    if m <= 0 or not signs:
                        continue
                    if not (1 <= m <= int(antal_matcher)):
                        all_warnings.append(f"Teckengrupp {i}: M{m} finns inte på kupongen.")
                        continue
                    if m not in by_match:
                        by_match[m] = set()
                    by_match[m].update(signs)
                choices = [{'match': int(m), 'signs': [s for s in ['1','X','2'] if s in signs]} for m, signs in sorted(by_match.items())]
                n = len(choices)
                mn = max(0, min(n, int(row.get('min', 0) or 0)))
                mx = max(mn, min(n, int(row.get('max', n) if row.get('max', None) is not None else n)))
                picks = _manual_group_choices_to_picks(choices)
                new_groups.append({
                    'active': bool(row.get('active')) and n > 0,
                    'name': str(row.get('name') or f'Teckengrupp {i}'),
                    'text': _manual_group_choices_to_text(choices),
                    'choices': choices,
                    'picks': picks,
                    'min': mn,
                    'max': mx,
                })
            old_sig = _manual_sign_groups_signature(st.session_state.get('v12_manual_sign_groups', []))
            new_sig = _manual_sign_groups_signature(new_groups)
            st.session_state['v12_manual_sign_groups'] = new_groups
            if old_sig != new_sig:
                # Gamla paket gäller inte längre, eftersom paketmotorn ska räknas efter teckengrupperna.
                for _k in ['v12_recommended_packages', 'v12_recommended_candidate_audit', 'v12_recommended_meta', 'v12_applied_package_meta', 'v12_applied_package_snapshot']:
                    st.session_state.pop(_k, None)
                st.session_state['v12_last_result_stale'] = True
                st.success('Manuella teckengrupper sparade. Rekommenderade paket behöver beräknas om.')
            else:
                st.success('Manuella teckengrupper sparade.')
            for w in all_warnings:
                st.warning(w)
            saved = _normalize_manual_sign_groups(new_groups, antal_matcher)

        active_groups = [g for g in saved if g.get('active') and _manual_group_choices_from_group(g, antal_matcher)]
        if active_groups:
            summary_df = _manual_sign_groups_summary_df(active_groups, v_m, filter_vec, frame, frame_rows, antal_matcher)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            base_after = _apply_manual_sign_groups_to_rows(frame_rows, active_groups, antal_matcher)
            hist_mask = _manual_sign_groups_hist_mask(v_m, active_groups, antal_matcher)
            st.info(f"Manuella teckengrupper aktiva: grundram {len(frame_rows):,} → {len(base_after):,} rader · manuell historik {int(hist_mask.sum())}/{len(hist_mask)}. Paketmotorn räknar radreducering på denna förfiltrerade radmassa, men rekommenderade filters träffbild visas separat mot alla {len(hist_mask)} liknande omgångar.".replace(',', ' '))
        else:
            st.info('Inga manuella teckengrupper är aktiva. Paketmotorn räknar på hela grundramen.')
        st.markdown("</div>", unsafe_allow_html=True)

    return _normalize_manual_sign_groups(st.session_state.get('v12_manual_sign_groups', []), antal_matcher)



FILTER_CATEGORY_ORDER = [
    'Struktur',
    'Värde & svårighet',
    'FAT',
    'FAT-sekvenser',
    'Favorit & skräll',
    'Manuella teckengrupper',
    'Super-Makro',
]


def _ordered_categories_from_specs(specs):
    present = []
    for s in specs or []:
        cat = s.get('category')
        if cat not in present:
            present.append(cat)
    ordered = [c for c in FILTER_CATEGORY_ORDER if c in present]
    ordered.extend([c for c in present if c not in ordered])
    return ordered

def _match_no_from_label(value):
    try:
        return int(str(value).upper().replace('MATCH', '').replace('M', '').strip())
    except Exception:
        return None


def _system_from_recommendation_df(df, value_col, antal_matcher):
    """Gör rekommendationstabell till ett Helgardering-liknande utgångssystem."""
    system = [[] for _ in range(int(antal_matcher))]
    if not isinstance(df, pd.DataFrame) or df.empty or value_col not in df.columns:
        return system
    for _, r in df.iterrows():
        m = _match_no_from_label(r.get('Match'))
        if m is None or not (1 <= m <= int(antal_matcher)):
            continue
        system[m - 1] = normalize_signs(str(r.get(value_col, '')))
    return system


def _streck_rec_tables_for_u(filter_vec, hist_df, similar_df, antal_matcher, max_shock_pct=22):
    hist_df = hist_df if isinstance(hist_df, pd.DataFrame) and not hist_df.empty else pd.DataFrame()
    similar_df = similar_df if isinstance(similar_df, pd.DataFrame) and not similar_df.empty else pd.DataFrame()
    return build_streck_recommendation_tables(
        filter_vec,
        hist_df,
        similar_df,
        int(antal_matcher),
        max_shock_pct=int(max_shock_pct or 22),
    )


def _u_system_from_streck_source(source, filter_vec, hist_df, similar_df, antal_matcher, max_shock_pct=22):
    spik_df, half_df, shock_df = _streck_rec_tables_for_u(filter_vec, hist_df, similar_df, antal_matcher, max_shock_pct=max_shock_pct)
    if source == '5 bästa spikar':
        return _system_from_recommendation_df(spik_df, 'Spik', antal_matcher), spik_df
    if source == '5 bästa halvor':
        return _system_from_recommendation_df(half_df, 'Halv', antal_matcher), half_df
    if source == '5 bästa skrällar':
        return _system_from_recommendation_df(shock_df, 'Skräll', antal_matcher), shock_df
    return [[] for _ in range(int(antal_matcher))], pd.DataFrame()


def _short_u_interval_summary(system, hist_rows, antal_matcher, target_pct=90.0):
    clean_hist = [normalize_single_row_text(r) for r in (hist_rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    vals = [_u_system_metric(r, system, 'utips') for r in clean_hist]
    if not vals:
        return {'text': 'Rek. intervall saknas', 'interval': None, 'hits': 0, 'total': 0, 'pct': 0.0}
    rec = get_best_interval(vals, target_pct)
    hp, ht, pct = _hist_pass_count(vals, rec)
    marked = u_system_marked_count(system)
    return {
        'text': f"Rek. spela {int(rec[0])}–{int(rec[1])} av {marked} utgångstips · historik {hp}/{ht} ({pct:.1f}%)",
        'interval': rec,
        'hits': hp,
        'total': ht,
        'pct': pct,
    }




def _u_total_filter_key(slot):
    """Filtercentralnyckel för ett utgångssystems huvudfilter: Antalet utgångstips."""
    return f"u_sys_us{int(slot)}_utips"


def _u_total_filter_mode_key(slot):
    return f"filter_mode_{_u_total_filter_key(slot)}"


def _u_total_filter_range_key(slot, target_pct, top_fav_count):
    return f"filter_range_{_u_total_filter_key(slot)}_h{int(target_pct)}_tf{int(top_fav_count)}"


def _normalize_int_interval(interval, low, high, fallback):
    try:
        a, b = interval
        a = int(round(float(a)))
        b = int(round(float(b)))
    except Exception:
        a, b = fallback
    a = max(int(low), min(int(a), int(high)))
    b = max(int(low), min(int(b), int(high)))
    if a > b:
        a, b = b, a
    return (a, b)


def _render_u_total_filter_controls(slot, system, rec_interval, filter_hist_target_pct, top_fav_count):
    """Direktkontroll i U-panelen för "x-y av utgångstips ska sitta".

    Samma värden skrivs till Filtercentralens session_state-nycklar, så användaren
    slipper leta upp raden "Utgångssystem N – Antalet utgångstips" längre ner.
    """
    marked = max(0, int(u_system_marked_count(system)))
    if marked <= 0:
        return {'mode': 'Av', 'interval': (0, 0)}

    mode_options = ['Av', 'Tvingat'] + [f'Grupp {i}' for i in range(1, 7)]
    u_mode_key = f'v12_us_{int(slot)}_utips_mode'
    u_range_key = f'v12_us_{int(slot)}_utips_interval'

    default_interval = _normalize_int_interval(rec_interval or (0, marked), 0, marked, (0, marked))
    if u_mode_key not in st.session_state:
        st.session_state[u_mode_key] = 'Av'
    if u_range_key not in st.session_state:
        st.session_state[u_range_key] = default_interval
    else:
        st.session_state[u_range_key] = _normalize_int_interval(st.session_state[u_range_key], 0, marked, default_interval)

    c_mode, c_range, c_apply = st.columns([0.9, 2.0, 0.8])
    with c_mode:
        mode = st.selectbox(
            "Filterläge",
            mode_options,
            index=mode_options.index(st.session_state.get(u_mode_key, 'Av')) if st.session_state.get(u_mode_key, 'Av') in mode_options else 0,
            key=u_mode_key,
            help="Av = bara visning. Tvingat = detta intervall måste sitta. Grupp = räknas i valt gruppkrav.",
        )
    # Streamlit tillåter inte att man ändrar session_state för en widgetnyckel
    # efter att widgeten med samma key har skapats i samma körning. Därför ligger
    # Rek-knappen före slidern i körordning, även om den visuellt visas i högerkolumnen.
    with c_apply:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("Rek.", key=f'v12_us_{int(slot)}_utips_use_rec', help="Återställ till rekommenderat historikintervall", use_container_width=True):
            st.session_state[u_range_key] = default_interval
            st.rerun()
    with c_range:
        interval = st.slider(
            f"Spela antal utgångstips som ska sitta, 0–{marked}",
            0,
            marked,
            st.session_state[u_range_key],
            step=1,
            key=u_range_key,
            help="Exempel 1–4 betyder att en rad får passera om 1 till 4 av de markerade utgångstipsen sitter.",
        )

    # Spegla direkt till Filtercentralen. Den byggs senare i samma körning.
    st.session_state[_u_total_filter_mode_key(slot)] = mode
    st.session_state[_u_total_filter_range_key(slot, filter_hist_target_pct, top_fav_count)] = interval

    return {'mode': mode, 'interval': interval}


def _compact_rec_preview_df(df, source):
    """Visar rekommendationer inne i U-panelen utan separat modul."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    cols = ['Match']
    if source == '5 bästa spikar':
        cols += ['Spik', 'Streck %']
    elif source == '5 bästa halvor':
        cols += ['Halv', 'Strecksumma %']
    elif source == '5 bästa skrällar':
        cols += ['Skräll', 'Rank', 'Streck %']
    for c in ['Liknande träff %', 'Liknande n', 'Historik träff %', 'Historik n', 'Hist. lift %', 'Rek']:
        if c in df.columns:
            cols.append(c)
    return df[[c for c in cols if c in df.columns]].copy()


@st.cache_data(show_spinner=False)
def run_core_analysis(input_text, spelform, antal_matcher, krav_odds, cb_structure, slider_top_n, cb_payout, pay_min, pay_max):
    fil_sökväg = find_local_database(spelform)
    if not fil_sökväg:
        return None, None, None, f"❌ Hittade ingen fil för {spelform} i mappen!"
    
    try:
        input_vec = parse_input_string(input_text, krav_odds)
    except:
        return None, None, None, "⚠️ Fel vid inläsning av odds."
        
    if len(input_vec) != krav_odds: 
        return None, None, None, f"⚠️ Fel: {spelform} kräver exakt {krav_odds} värden. Hittade {len(input_vec)}."

    global_db = load_database(fil_sökväg, antal_matcher)

    # Viktigt: similarity_vec används bara för att hitta liknande historiska kuponger.
    # input_vec behåller riktig matchordning och används i alla faktiska filterberäkningar.
    similarity_vec = get_structural_vector(input_vec) if cb_structure else input_vec
    weights_arr = np.array([w for i in range(0, krav_odds, 3) for w in [(max(similarity_vec[i:i+3])/100.0)**2]*3])

    df_s = global_db.copy()

    # Utdelningskravet ska ligga före top-N, annars kan top-N först plockas fram och därefter rasa bort.
    if cb_payout:
        df_s = df_s[(df_s['Payout'] >= pay_min) & (df_s['Payout'] <= pay_max)]
    if len(df_s) == 0:
        return df_s, input_vec, os.path.basename(fil_sökväg), None

    df_s['Sim'] = [
        weighted_euclidean(
            similarity_vec,
            get_structural_vector(r['Prob_Vector']) if cb_structure else r['Prob_Vector'],
            weights_arr
        ) if len(r['Prob_Vector']) == krav_odds else 9999
        for _, r in df_s.iterrows()
    ]
    
    v_m = df_s.sort_values('Sim').head(slider_top_n)
    return v_m, input_vec, os.path.basename(fil_sökväg), None


# ==========================================
# STREAMLIT UI – v12.0 REN HELGARDERING-FILTERCENTRAL
# ==========================================

st.markdown("""
<style>
    .main .block-container {max-width: 1480px; padding-top: 1rem;}
    .v12-hero {border:1px solid rgba(120,120,120,.25); border-radius:18px; padding:18px 22px; margin-bottom:16px; background:linear-gradient(135deg, rgba(64,120,255,.12), rgba(0,0,0,.02));}
    .v12-step {font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; opacity:.70; font-weight:800; margin-bottom:.2rem;}
    .v12-title {font-size:1.45rem; font-weight:900; margin-bottom:.15rem;}
    .v12-muted {opacity:.75; font-size:.93rem;}
    .v12-card {border:1px solid rgba(120,120,120,.22); border-radius:16px; padding:14px 16px; margin:10px 0 14px 0; background:rgba(128,128,128,.035);} 
    .v12-pill {display:inline-block; padding:.18rem .56rem; border:1px solid rgba(120,120,120,.35); border-radius:999px; font-size:.82rem; margin-right:.28rem; margin-top:.25rem;}
    .v12-info-grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:12px; margin:.65rem 0 1.1rem 0;}
    .v12-info-card {border:1px solid rgba(120,120,120,.24); border-radius:14px; padding:12px 14px; background:rgba(128,128,128,.045); min-height:86px;}
    .v12-info-label {font-size:.82rem; opacity:.86; font-weight:800; margin-bottom:.34rem;}
    .v12-info-value {font-size:1.38rem; font-weight:850; line-height:1.15; white-space:normal; overflow-wrap:anywhere; word-break:break-word;}
    .v12-info-sub {font-size:.82rem; opacity:.76; margin-top:.32rem; line-height:1.25;}

    .v12-preview-grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:.55rem 0 .85rem 0;}
    .v12-preview-card {border:1px solid rgba(120,120,120,.24); border-radius:12px; padding:8px 11px; background:rgba(128,128,128,.04); min-height:58px;}
    .v12-preview-label {font-size:.76rem; opacity:.82; font-weight:800; margin-bottom:.18rem;}
    .v12-preview-value {font-size:1.08rem; font-weight:850; line-height:1.15; white-space:normal; overflow-wrap:anywhere; word-break:break-word;}
</style>
""", unsafe_allow_html=True)


def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _slug(s):
    return re.sub(r'[^a-zA-Z0-9_]+', '_', str(s)).strip('_').lower()


FILTER_HIST_TARGET_DEFAULTS_BY_CATEGORY = {
    'Värde & svårighet': 95,
    'FAT': 95,
    'FAT-sekvenser': 95,
    'Favorit & skräll': 95,
    'Struktur': 100,
    'Super-Makro': 95,
}


def _filter_target_key_for_category(category):
    return f"v12_filter_hist_target_cat_{_slug(category)}"


def _clamp_filter_hist_target(value, default=95):
    try:
        return int(max(50, min(100, int(value))))
    except Exception:
        return int(default)


def _get_filter_hist_target_pct_by_category():
    """Returnerar kategori-specifika träffmål för rekommenderade filterintervall.

    Den gamla globala slidern finns kvar som bakåtkompatibel fallback för gamla
    sparfiler, men nya sessioner använder separata värden per filterkategori.
    """
    legacy = st.session_state.get('v12_filter_hist_target_pct', None)
    out = {}
    for category, default in FILTER_HIST_TARGET_DEFAULTS_BY_CATEGORY.items():
        key = _filter_target_key_for_category(category)
        if key not in st.session_state:
            st.session_state[key] = _clamp_filter_hist_target(legacy, default) if legacy is not None else int(default)
        out[category] = _clamp_filter_hist_target(st.session_state.get(key), default)
    return out


def _set_filter_hist_target_pct_by_category(targets):
    if not isinstance(targets, dict):
        return
    for category, default in FILTER_HIST_TARGET_DEFAULTS_BY_CATEGORY.items():
        if category in targets:
            st.session_state[_filter_target_key_for_category(category)] = _clamp_filter_hist_target(targets.get(category), default)


def _hist_target_for_spec(spec, fallback=95):
    if isinstance(spec, dict) and spec.get('target_hist_pct') is not None:
        return _clamp_filter_hist_target(spec.get('target_hist_pct'), fallback)
    if isinstance(spec, dict) and spec.get('category') in FILTER_HIST_TARGET_DEFAULTS_BY_CATEGORY:
        targets = _get_filter_hist_target_pct_by_category()
        return _clamp_filter_hist_target(targets.get(spec.get('category')), FILTER_HIST_TARGET_DEFAULTS_BY_CATEGORY.get(spec.get('category'), fallback))
    return _clamp_filter_hist_target(fallback, fallback)


def _hist_pass_label_from_pct(pct, total):
    try:
        return int(math.ceil(float(pct) * float(total) / 100.0))
    except Exception:
        return 0


def _hist_pass_count(values, interval):
    ok = [v for v in values if in_range(v, interval)]
    return len(ok), len(values), (100.0 * len(ok) / len(values) if len(values) else 0.0)


def _display_interval(interval, decimals=0):
    return fmt_interval(interval, decimals)


def _bounds_from_values(values, interval=None, decimals=0, hard_min=None, hard_max=None):
    vals = [float(v) for v in values if not pd.isna(v)]
    if interval is not None:
        vals += [float(interval[0]), float(interval[1])]
    if not vals:
        vals = [0.0, 1.0]
    lo = min(vals) if hard_min is None else float(hard_min)
    hi = max(vals) if hard_max is None else float(hard_max)
    if decimals == 0:
        lo = int(np.floor(lo)); hi = int(np.ceil(hi))
        if lo == hi:
            hi = lo + 1
    else:
        pad = 0.0
        lo = round(lo - pad, decimals); hi = round(hi + pad, decimals)
        if lo == hi:
            hi = round(lo + (0.1 if decimals == 1 else 0.01), decimals)
    return lo, hi


def _current_interval_for_spec(spec):
    k = spec['key']
    interval = st.session_state.get(f'filter_range_{k}', spec['default_interval'])
    try:
        return (interval[0], interval[1])
    except Exception:
        return spec['default_interval']


def _spec_value(row, spec):
    return spec['getter'](row)


def _spec_pass(row, spec, interval=None):
    if interval is None:
        interval = _current_interval_for_spec(spec)
    return in_range(_spec_value(row, spec), interval)


def _make_freq_df(values, decimals=0):
    """Bygger en frekvenstabell som alltid är numeriskt sorterad.

    Tidigare användes strängar som index i diagrammet. Då kunde negativa tal
    visas i lexikografisk ordning, t.ex. -13 före -83. Tabellen returnerar
    därför även en intern numerisk sorteringskolumn.
    """
    clean_vals = []
    for v in values:
        if pd.isna(v):
            continue
        try:
            clean_vals.append(float(v))
        except Exception:
            continue
    vals = pd.Series(clean_vals, dtype='float64')
    if vals.empty:
        return pd.DataFrame({'Intervall/Värde': [], 'Antal': [], 'Andel': [], '_sort': []})

    total = max(1, len(vals))
    unique_count = int(vals.nunique())

    # Exakta värden när det är få historiska observationer, vilket det oftast är
    # för 30 liknande omgångar. Detta gör tabellen lätt att läsa.
    if unique_count <= 30:
        rounded = vals.round(decimals)
        if decimals == 0:
            rounded = rounded.astype(int)
        vc = rounded.value_counts().reset_index()
        vc.columns = ['_value', 'Antal']
        vc = vc.sort_values('_value', ascending=True).reset_index(drop=True)
        def fmt_value(x):
            try:
                if decimals == 0:
                    return str(int(x))
                return f"{float(x):.{decimals}f}"
            except Exception:
                return str(x)
        vc['Intervall/Värde'] = vc['_value'].map(fmt_value)
        vc['_sort'] = vc['_value'].astype(float)
        vc['Andel'] = vc['Antal'].map(lambda n: f"{100*n/total:.1f}%")
        return vc[['Intervall/Värde', 'Antal', 'Andel', '_sort']]

    # Om det skulle finnas fler värden än 30 grupperas de i numeriskt sorterade intervall.
    bins = min(10, max(5, int(np.sqrt(unique_count))))
    try:
        cats = pd.cut(vals, bins=bins, include_lowest=True)
        vc = cats.value_counts().sort_index().reset_index()
        vc.columns = ['_bin', 'Antal']
        def fmt_bin(iv):
            left = float(iv.left); right = float(iv.right)
            if decimals == 0:
                return f"{int(np.floor(left))} – {int(np.ceil(right))}"
            return f"{left:.{decimals}f} – {right:.{decimals}f}"
        vc['Intervall/Värde'] = vc['_bin'].map(fmt_bin)
        vc['_sort'] = vc['_bin'].map(lambda iv: float(iv.left))
        vc['Andel'] = vc['Antal'].map(lambda n: f"{100*n/total:.1f}%")
        vc = vc.sort_values('_sort', ascending=True).reset_index(drop=True)
        return vc[['Intervall/Värde', 'Antal', 'Andel', '_sort']]
    except Exception:
        rounded = vals.round(decimals)
        vc = rounded.value_counts().reset_index()
        vc.columns = ['_value', 'Antal']
        vc = vc.sort_values('_value', ascending=True).reset_index(drop=True)
        vc['Intervall/Värde'] = vc['_value'].astype(str)
        vc['_sort'] = vc['_value'].astype(float)
        vc['Andel'] = vc['Antal'].map(lambda n: f"{100*n/total:.1f}%")
        return vc[['Intervall/Värde', 'Antal', 'Andel', '_sort']]


def _top_sequences_from_fat(fat_strings, length=2, top_n=5):
    counts = {}
    for fs in fat_strings:
        seen = set(fs[i:i+length] for i in range(0, max(0, len(fs)-length+1)))
        for s in seen:
            if len(s) == length:
                counts[s] = counts.get(s, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]


def _top_sequence_pairs_from_fat(fat_strings, length=3, top_n=5):
    counts = {}
    for fs in fat_strings:
        seqs = sorted(set(fs[i:i+length] for i in range(0, max(0, len(fs)-length+1)) if len(fs[i:i+length]) == length))
        for a, b in itertools.combinations(seqs, 2):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]


def _super_macro_count(row_str, prob_vector, bounds):
    """Returnerar antal makrogrupper där minst 2 av 3 interna delvärden sitter."""
    score = 0
    ones, draws, twos = row_str.count('1'), row_str.count('X'), row_str.count('2')
    s1, sx, s2, _ = get_streaks(row_str)
    g1, gx, g2, _ = get_gaps(row_str)
    si1, six, si2, _, _ = get_singles(row_str)
    d1, dx, d2, _, _ = get_doublets(row_str)
    t1, tx, t2, _, _ = get_triplets(row_str)
    o1, ox, o2, _, _ = get_occurrences(row_str)
    f, a, t, _ = get_fat(row_str, prob_vector)
    groups = [
        [in_range(ones, bounds['ones']), in_range(draws, bounds['draws']), in_range(twos, bounds['twos'])],
        [in_range(s1, bounds['s1']), in_range(sx, bounds['sx']), in_range(s2, bounds['s2'])],
        [in_range(g1, bounds['g1']), in_range(gx, bounds['gx']), in_range(g2, bounds['g2'])],
        [in_range(si1, bounds['si1']), in_range(six, bounds['six']), in_range(si2, bounds['si2'])],
        [in_range(d1, bounds['d1']), in_range(dx, bounds['dx']), in_range(d2, bounds['d2'])],
        [in_range(t1, bounds['t1']), in_range(tx, bounds['tx']), in_range(t2, bounds['t2'])],
        [in_range(o1, bounds['o1']), in_range(ox, bounds['ox']), in_range(o2, bounds['o2'])],
        [in_range(f, bounds['fat_f']), in_range(a, bounds['fat_a']), in_range(t, bounds['fat_t'])],
    ]
    for g in groups:
        if sum(bool(x) for x in g) >= 2:
            score += 1
    return score


def _sample_rows_for_macro(rows, max_items=3000):
    """Deterministiskt urval för Super-Makro-rekommendation när grundramen är stor."""
    rows = list(rows or [])
    n = len(rows)
    if n <= int(max_items):
        return rows, False
    step = n / float(max_items)
    idxs = [min(n - 1, int(i * step)) for i in range(int(max_items))]
    return [rows[i] for i in idxs], True


def _recommend_super_macro_interval(hist_values, candidate_rows, filter_vec, macro_bounds, target_hist_pct=90):
    """Väljer Super-Makro-nivå efter rationell effekt, inte bara högsta blockträff.

    Testar 0-8, 1-8, ..., 8-8. För varje nivå räknas historisk träff och
    kvarvarande andel av grundramen. Nivå 4-6 premieras eftersom Super-Makro
    är tänkt som ett mjukt makrofilter; 7-8/8-8 ska bara väljas när de faktiskt
    ger tydligt bättre rationell profil utan att bli ett överhårt normalitetslås.
    """
    hist_values = [int(v) for v in (hist_values or []) if not pd.isna(v)]
    if not hist_values:
        return (0, 8), pd.DataFrame(), False

    cand_sample, sampled = _sample_rows_for_macro(candidate_rows or [], max_items=12000)
    if cand_sample:
        cand_values = [_super_macro_count(r, filter_vec, macro_bounds) for r in cand_sample]
    else:
        cand_values = []

    hist_total = len(hist_values)
    cand_total = len(cand_values)
    rows_out = []
    for req in range(0, 9):
        interval = (req, 8)
        hist_pass = sum(1 for v in hist_values if req <= v <= 8)
        hist_pct = (hist_pass / hist_total) * 100.0 if hist_total else 0.0
        if cand_total:
            cand_pass = sum(1 for v in cand_values if req <= v <= 8)
            keep_pct = (cand_pass / cand_total) * 100.0
        else:
            keep_pct = 100.0
        reduction = 100.0 - keep_pct
        lift = hist_pct - keep_pct

        # Balanspremie: 4-6/8 är ofta den rationella zonen för makrofilter.
        # Högre krav kan fortfarande vinna, men måste kompensera för att de
        # lätt blir ett normalitetslås med låg robusthet.
        if req == 4:
            balance_bonus = 30.0
        elif req == 5:
            balance_bonus = 24.0
        elif req == 6:
            balance_bonus = 12.0
        elif req == 3:
            balance_bonus = 10.0
        elif req == 7:
            balance_bonus = -15.0
        elif req == 8:
            balance_bonus = -30.0
        else:
            balance_bonus = -8.0

        score = lift + reduction * 0.25 + max(hist_pct - float(target_hist_pct), 0.0) * 0.10 + balance_bonus
        rows_out.append({
            'Intervall': f'{req}–8',
            'Min block': int(req),
            'Max block': 8,
            'Historisk träff': f'{hist_pass}/{hist_total}',
            'Historisk träff %': round(hist_pct, 1),
            'Kvar rad %': round(keep_pct, 1),
            'Reducerar %': round(reduction, 1),
            'Rationell faktor': round(lift, 1),
            'Balanspoäng': round(score, 2),
        })

    target = float(target_hist_pct or 90)
    valid = [r for r in rows_out if r['Historisk träff %'] + 1e-9 >= target]
    if not valid:
        # Om användaren kräver t.ex. 100% och inget nivåintervall når dit väljer vi högsta
        # historik först, därefter bästa balanspoäng.
        valid = rows_out
        best = max(valid, key=lambda r: (r['Historisk träff %'], r['Balanspoäng'], r['Reducerar %']))
    else:
        best = max(valid, key=lambda r: (r['Balanspoäng'], r['Rationell faktor'], r['Reducerar %']))

    return (int(best['Min block']), 8), pd.DataFrame(rows_out), sampled



def _macro_count_from_specs_row(row, macro_specs):
    """Räknar hur många filter/specar som klaras med respektive rekommenderat intervall."""
    count = 0
    for spec in macro_specs or []:
        try:
            if in_range(_spec_value(row, spec), spec.get('default_interval', (0, 0))):
                count += 1
        except Exception:
            continue
    return int(count)


def _macro_hist_values_from_specs(macro_specs):
    """Bygger historiska makrovärden: antal underfilter som sitter per historisk omgång."""
    macro_specs = list(macro_specs or [])
    if not macro_specs:
        return []
    lengths = [len(s.get('hist_values', [])) for s in macro_specs if s.get('hist_values') is not None]
    if not lengths:
        return []
    n = min(lengths)
    vals = []
    for i in range(n):
        c = 0
        for spec in macro_specs:
            try:
                hv = spec.get('hist_values', [])[i]
                if in_range(hv, spec.get('default_interval', (0, 0))):
                    c += 1
            except Exception:
                continue
        vals.append(int(c))
    return vals


def _recommend_count_macro_interval(hist_values, candidate_rows, row_getter, max_score, target_hist_pct=90, preferred_low=None, preferred_high=None):
    """Generisk rationell rekommendation för makron som räknar antal block/filter som sitter.

    Testar 0-max, 1-max, ..., max-max. Väljer nivån efter historisk träff,
    reducering och rationell faktor, inte bara efter högsta normalitetsnivå.
    """
    hist_values = [int(v) for v in (hist_values or []) if not pd.isna(v)]
    max_score = int(max(0, max_score or 0))
    if max_score <= 0:
        return (0, 0), pd.DataFrame(), False
    if not hist_values:
        return (0, max_score), pd.DataFrame(), False

    if preferred_low is None:
        preferred_low = int(max(0, round(max_score * 0.45)))
    if preferred_high is None:
        preferred_high = int(max(preferred_low, round(max_score * 0.75)))
    preferred_low = max(0, min(max_score, int(preferred_low)))
    preferred_high = max(preferred_low, min(max_score, int(preferred_high)))

    cand_sample, sampled = _sample_rows_for_macro(candidate_rows or [], max_items=12000)
    cand_values = []
    if cand_sample:
        for r in cand_sample:
            try:
                cand_values.append(int(row_getter(r)))
            except Exception:
                continue

    hist_total = len(hist_values)
    cand_total = len(cand_values)
    rows_out = []
    target = float(target_hist_pct or 90)
    for req in range(0, max_score + 1):
        hist_pass = sum(1 for v in hist_values if req <= v <= max_score)
        hist_pct = (hist_pass / hist_total) * 100.0 if hist_total else 0.0
        if cand_total:
            cand_pass = sum(1 for v in cand_values if req <= v <= max_score)
            keep_pct = (cand_pass / cand_total) * 100.0
        else:
            keep_pct = 100.0
        reduction = 100.0 - keep_pct
        lift = hist_pct - keep_pct

        if preferred_low <= req <= preferred_high:
            # Störst bonus i mitten av den rationella zonen.
            mid = (preferred_low + preferred_high) / 2.0
            span = max(1.0, (preferred_high - preferred_low) / 2.0)
            balance_bonus = 22.0 - 8.0 * abs(req - mid) / span
        elif req > preferred_high:
            balance_bonus = -10.0 * (req - preferred_high)
        else:
            balance_bonus = -4.0 * (preferred_low - req)

        score = lift + reduction * 0.25 + max(hist_pct - target, 0.0) * 0.10 + balance_bonus
        rows_out.append({
            'Intervall': f'{req}–{max_score}',
            'Min block': int(req),
            'Max block': int(max_score),
            'Historisk träff': f'{hist_pass}/{hist_total}',
            'Historisk träff %': round(hist_pct, 1),
            'Kvar rad %': round(keep_pct, 1),
            'Reducerar %': round(reduction, 1),
            'Rationell faktor': round(lift, 1),
            'Balanspoäng': round(score, 2),
        })

    valid = [r for r in rows_out if r['Historisk träff %'] + 1e-9 >= target]
    if not valid:
        valid = rows_out
        best = max(valid, key=lambda r: (r['Historisk träff %'], r['Balanspoäng'], r['Reducerar %']))
    else:
        best = max(valid, key=lambda r: (r['Balanspoäng'], r['Rationell faktor'], r['Reducerar %']))
    return (int(best['Min block']), int(best['Max block'])), pd.DataFrame(rows_out), sampled


@st.cache_data(show_spinner=False)
def _cached_rows_from_frame(frame_tuple, antal_matcher, max_rows=None):
    frame = [list(x) for x in frame_tuple]
    return generate_rows_from_frame(frame, max_rows=max_rows)


def _frame_cache_tuple(frame):
    return tuple(tuple(x) for x in frame)


def build_clean_filter_specs(v_m, filter_vec, antal_matcher, slider_u_count=3, target_hist_pct=90, u_rows=None, hist_df=None, max_shock_pct=22, candidate_rows=None, include_supermakro=False, category_hist_target_pct=None):
    """Bygger exakt en spec per filter. Inga AutoHard-varianter/dubbletter.

    Rekommenderat intervall/startvärde styrs kategori för kategori.
    Exempel: Struktur kan ligga på 100% medan Värde & svårighet ligger på 95%.
    target_hist_pct finns kvar som bakåtkompatibel fallback.
    """
    rows = list(v_m['Correct_Row'])
    probs = list(v_m['Prob_Vector'])
    match_odds_filter = [filter_vec[j:j+3] for j in range(0, len(filter_vec), 3)]
    cur_matrix, cur_scores_asc, cur_tot = calculate_ai_matrix_from_values(filter_vec)
    specs = []

    target_hist_pct = int(max(50, min(100, target_hist_pct)))
    category_hist_target_pct = category_hist_target_pct or {}

    def _category_target_pct(category, fallback=None):
        if fallback is None:
            fallback = target_hist_pct
        default = FILTER_HIST_TARGET_DEFAULTS_BY_CATEGORY.get(str(category), fallback)
        return _clamp_filter_hist_target(category_hist_target_pct.get(str(category), fallback), default)

    def add_interval(name, category, values, getter, decimals=0, coverage=85, hard_min=None, hard_max=None, help_text="", key_override=None, meta=None, default_interval_override=None):
        vals = [v for v in values if not pd.isna(v)]
        if not vals:
            return
        # v12.0cn: träffmålet styrs per filterkategori. Struktur kan t.ex.
        # sättas till 100% medan Värde & svårighet/FAT/Favorit & skräll ligger
        # på 95%. Det gamla globala värdet används bara som fallback.
        coverage = _category_target_pct(category, coverage)
        if default_interval_override is None:
            default_interval = get_best_interval(vals, coverage)
        else:
            default_interval = default_interval_override
        bounds = _bounds_from_values(vals, default_interval, decimals=decimals, hard_min=hard_min, hard_max=hard_max)
        hist_pass, hist_total, hist_pct = _hist_pass_count(vals, default_interval)
        key_base = key_override or _slug(name)
        # Undvik nyckelkrockar om ett namn återkommer.
        key = key_base
        i = 2
        existing = {s['key'] for s in specs}
        while key in existing:
            key = f"{key_base}_{i}"; i += 1
        specs.append({
            'key': key,
            'name': name,
            'category': category,
            'type': 'interval',
            'hist_values': list(vals),
            'getter': getter,
            'decimals': decimals,
            'default_interval': default_interval,
            'bounds': bounds,
            'hist_pass': hist_pass,
            'hist_total': hist_total,
            'hist_pct': hist_pct,
            'target_hist_pct': int(coverage),
            'help': help_text,
            'meta': meta or {},
        })

    # Historiska värden
    ai_ranks, delta_vals, total_diff_vals = [], [], []
    rank_sums, minus_sums, log_sums, sft_sums, points_vals = [], [], [], [], []
    fat_zone_a, fat_zone_b, fat_zone_c = [], [], []
    fat_f, fat_a, fat_t, fat_sum = [], [], [], []
    abc_1, abc_2, abc_3, abc_sum = [], [], [], []
    fat_rise1, fat_rise2, fat_fall1, fat_fall2 = [], [], [], []
    fav_top_vals_by_n = {n: [] for n in range(3, 7)}
    fav_delta_vals = []
    fav70, fav60, fav50 = [], [], []
    sh10, sh15, sh20, sh_low = [], [], [], []
    ones, draws, twos = [], [], []
    s1, sx, s2, stot = [], [], [], []
    g1, gx, g2, gtot = [], [], [], []
    si1, six, si2, sit = [], [], [], []
    d1, dx, d2, dt = [], [], [], []
    tr1, trx, tr2, trt = [], [], [], []
    o1, ox, o2, ot = [], [], [], []
    fat_strings = []

    # Utgångssystem/U-rader är avstängda i denna version.
    # Streckbaserade val läggs i stället in som vanliga filter under Favorit & skräll.

    for row_str, p in zip(rows, probs):
        if len(str(row_str)) != antal_matcher:
            continue
        try:
            src_row = v_m[v_m['Correct_Row'] == row_str].iloc[0]
            if 'True_Rank' in src_row and pd.notna(src_row['True_Rank']) and src_row['True_Rank'] > 0:
                ai_ranks.append(int(src_row['True_Rank']))
            else:
                hmat, hscores, htot = calculate_ai_matrix_from_values(p)
                rr, _ = get_exact_rank(row_str, hmat, hscores, htot)
                ai_ranks.append(rr)
        except Exception:
            hmat, hscores, htot = calculate_ai_matrix_from_values(p)
            rr, _ = get_exact_rank(row_str, hmat, hscores, htot)
            ai_ranks.append(rr)
        delta_vals.append(calculate_delta(row_str, p))
        total_diff_vals.append(calculate_total_diff([p[j:j+3] for j in range(0, len(p), 3)], list(row_str)))
        rank_sums.append(get_rank_sum(row_str, p))
        minus_sums.append(get_100_minus_sum(row_str, p))
        log_sums.append(get_log_surprise_sum(row_str, p))
        sft_sums.append(get_sft_sum(row_str, p))
        points_vals.append(get_rank_points(row_str, p))
        fat_zone_a.append(get_fat_zone_points(row_str, p, 'A', antal_matcher))
        fat_zone_b.append(get_fat_zone_points(row_str, p, 'B', antal_matcher))
        fat_zone_c.append(get_fat_zone_points(row_str, p, 'C', antal_matcher))
        f, a, t, fs = get_fat(row_str, p); fat_f.append(f); fat_a.append(a); fat_t.append(t); fat_sum.append(fs)
        _fr1, _fr2, _ff1, _ff2 = get_fat_step_counts(row_str, p, antal_matcher); fat_rise1.append(_fr1); fat_rise2.append(_fr2); fat_fall1.append(_ff1); fat_fall2.append(_ff2)
        _abc1, _abc2, _abc3, _abcs = get_abc_counts(row_str, p, antal_matcher=antal_matcher)
        abc_1.append(_abc1); abc_2.append(_abc2); abc_3.append(_abc3); abc_sum.append(_abcs)
        for _n in range(3, 7):
            fav_top_vals_by_n[_n].append(get_top_n_favs_wins(row_str, p, _n))
        fav_delta_vals.append(get_favorite_delta(row_str, p))
        fp = get_favorite_pressure(row_str, p); fav70.append(fp['F70_Wins']); fav60.append(fp['F60_Wins']); fav50.append(fp['F50_Wins'])
        sh = get_shock_strength(row_str, p); sh10.append(sh['U10_Wins']); sh15.append(sh['U15_Wins']); sh20.append(sh['U20_Wins']); sh_low.append(sh['Lowest_Win_Pct'])
        ones.append(row_str.count('1')); draws.append(row_str.count('X')); twos.append(row_str.count('2'))
        _s1, _sx, _s2, _stot = get_streaks(row_str); s1.append(_s1); sx.append(_sx); s2.append(_s2); stot.append(_stot)
        _g1, _gx, _g2, _gtot = get_gaps(row_str); g1.append(_g1); gx.append(_gx); g2.append(_g2); gtot.append(_gtot)
        _si1, _six, _si2, _sit, _ = get_singles(row_str); si1.append(_si1); six.append(_six); si2.append(_si2); sit.append(_sit)
        _d1, _dx, _d2, _dt, _ = get_doublets(row_str); d1.append(_d1); dx.append(_dx); d2.append(_d2); dt.append(_dt)
        _tr1, _trx, _tr2, _trt, _ = get_triplets(row_str); tr1.append(_tr1); trx.append(_trx); tr2.append(_tr2); trt.append(_trt)
        _o1, _ox, _o2, _ot, _ = get_occurrences(row_str); o1.append(_o1); ox.append(_ox); o2.append(_o2); ot.append(_ot)
        fat_strings.append(get_fat_string(row_str, p))

    # Värde / svårighet
    add_interval('AI-Rank', 'Värde & svårighet', ai_ranks, lambda r: get_exact_rank(r, cur_matrix, cur_scores_asc, cur_tot)[0], 0, 85, 1, cur_tot)
    add_interval('Delta / Avvikelse', 'Värde & svårighet', delta_vals, lambda r: calculate_delta(r, filter_vec), 1, 85)
    add_interval('Total Diff', 'Värde & svårighet', total_diff_vals, lambda r: calculate_total_diff(match_odds_filter, list(r)), 0, 85)
    add_interval('Rank Summa', 'Värde & svårighet', rank_sums, lambda r: get_rank_sum(r, filter_vec), 1, 85)
    add_interval('100-minus Summa', 'Värde & svårighet', minus_sums, lambda r: get_100_minus_sum(r, filter_vec), 1, 85)
    add_interval('Skrälltryck Log Summa', 'Värde & svårighet', log_sums, lambda r: get_log_surprise_sum(r, filter_vec), 0, 85)
    add_interval('SFT Summa', 'Värde & svårighet', sft_sums, lambda r: get_sft_sum(r, filter_vec), 1, 85)
    if sft_sums:
        sft_hist_median = float(np.median([v for v in sft_sums if not pd.isna(v)]))
        sft_dist_vals = [abs(float(v) - sft_hist_median) for v in sft_sums if not pd.isna(v)]
        add_interval(
            'SFT-avstånd från historik',
            'Värde & svårighet',
            sft_dist_vals,
            lambda r, _med=sft_hist_median: abs(float(get_sft_sum(r, filter_vec)) - _med),
            1,
            85,
            0,
            None,
            f"Avstånd från median-SFT i de liknande historiska omgångarna ({sft_hist_median:.1f}). Lägre värde = närmare historikens normala svårighetsnivå.",
            key_override='sft_avstand_historik',
        )
    add_interval('Poängfilter', 'Värde & svårighet', points_vals, lambda r: get_rank_points(r, filter_vec), 0, 85)

    _cur_zones = get_fat_zone_indices(filter_vec, antal_matcher=antal_matcher)
    _zone_defs = [
        ('A', 'Poängzon A – toppfavoriter', fat_zone_a, 'Toppfavoritzon'),
        ('B', 'Poängzon B – mittenmatcher', fat_zone_b, 'Mittenzon'),
        ('C', 'Poängzon C – öppna matcher', fat_zone_c, 'Öppen/svår zon'),
    ]
    for _zone, _name, _vals, _label in _zone_defs:
        _zsize = len(_cur_zones.get(_zone, []))
        if _zsize <= 0:
            continue
        add_interval(
            _name,
            'Värde & svårighet',
            _vals,
            lambda r, _z=_zone: get_fat_zone_points(r, filter_vec, _z, antal_matcher),
            0,
            85,
            _zsize,
            _zsize * 3,
            f"FAT-poäng i {_label}. Favorit=1, andratecken=2, tredjetecken=3. Aktuell zon: {describe_fat_zone(filter_vec, _zone, antal_matcher)}.",
            key_override=f'fat_poangzon_{_zone.lower()}',
        )

    # FAT
    add_interval('FAT F', 'FAT', fat_f, lambda r: get_fat(r, filter_vec)[0], 0, 85, 0, antal_matcher)
    add_interval('FAT A', 'FAT', fat_a, lambda r: get_fat(r, filter_vec)[1], 0, 85, 0, antal_matcher)
    add_interval('FAT T', 'FAT', fat_t, lambda r: get_fat(r, filter_vec)[2], 0, 85, 0, antal_matcher)
    add_interval('FAT Summa', 'FAT', fat_sum, lambda r: get_fat(r, filter_vec)[3], 0, 85)
    if fat_sum:
        fat_hist_median = float(np.median([v for v in fat_sum if not pd.isna(v)]))
        fat_dist_vals = [abs(float(v) - fat_hist_median) for v in fat_sum if not pd.isna(v)]
        add_interval(
            'FAT-avstånd från historik',
            'FAT',
            fat_dist_vals,
            lambda r, _med=fat_hist_median: abs(float(get_fat(r, filter_vec)[3]) - _med),
            1,
            85,
            0,
            None,
            f"Avstånd från median-FAT-poängsumma i de liknande historiska omgångarna ({fat_hist_median:.1f}). Lägre värde = närmare historikens normala FAT-nivå.",
            key_override='fat_avstand_historik',
        )

    _fat_step_help = "Stigande/fallande poängtabell med FAT-poäng per match: favorit=1, andratecken=2, tredjetecken=3. Räknar rörelser mellan intilliggande matcher i kupongordning; lika värden räknas inte."
    add_interval('FAT stigande 1 steg', 'FAT', fat_rise1, lambda r: get_fat_step_counts(r, filter_vec, antal_matcher)[0], 0, 85, 0, max(0, antal_matcher - 1), _fat_step_help, key_override='fat_stigande_1_steg')
    add_interval('FAT stigande 2 steg', 'FAT', fat_rise2, lambda r: get_fat_step_counts(r, filter_vec, antal_matcher)[1], 0, 85, 0, max(0, antal_matcher - 1), _fat_step_help, key_override='fat_stigande_2_steg')
    add_interval('FAT fallande 1 steg', 'FAT', fat_fall1, lambda r: get_fat_step_counts(r, filter_vec, antal_matcher)[2], 0, 85, 0, max(0, antal_matcher - 1), _fat_step_help, key_override='fat_fallande_1_steg')
    add_interval('FAT fallande 2 steg', 'FAT', fat_fall2, lambda r: get_fat_step_counts(r, filter_vec, antal_matcher)[3], 0, 85, 0, max(0, antal_matcher - 1), _fat_step_help, key_override='fat_fallande_2_steg')

    _abc_current_map = get_abc_class_map(filter_vec, antal_matcher)
    _abc_help = "Global ABC-rank: alla tecken rankas 1–39 efter streck. Rank 1–13 = klass 1, 14–26 = klass 2, 27–39 = klass 3. Matchkarta 1-X-2: " + describe_abc_match_patterns(filter_vec, antal_matcher)
    add_interval('ABC 1', 'FAT', abc_1, lambda r, _m=_abc_current_map: get_abc_counts(r, filter_vec, _m, antal_matcher)[0], 0, 85, 0, antal_matcher, _abc_help, key_override='abc_1')
    add_interval('ABC 2', 'FAT', abc_2, lambda r, _m=_abc_current_map: get_abc_counts(r, filter_vec, _m, antal_matcher)[1], 0, 85, 0, antal_matcher, _abc_help, key_override='abc_2')
    add_interval('ABC 3', 'FAT', abc_3, lambda r, _m=_abc_current_map: get_abc_counts(r, filter_vec, _m, antal_matcher)[2], 0, 85, 0, antal_matcher, _abc_help, key_override='abc_3')
    add_interval('ABC Summa', 'FAT', abc_sum, lambda r, _m=_abc_current_map: get_abc_counts(r, filter_vec, _m, antal_matcher)[3], 0, 85, antal_matcher, antal_matcher * 3, _abc_help, key_override='abc_summa')

    seq2 = _top_sequences_from_fat(fat_strings, 2, 5)
    if seq2:
        vals = [count_fat_sequence_hits(fs, seq2) for fs in fat_strings]
        add_interval('FAT 2-sekvenser', 'FAT-sekvenser', vals, lambda r, seqs=seq2: count_fat_sequence_hits(get_fat_string(r, filter_vec), seqs), 0, 85, 0, len(seq2), f"Toppar: {', '.join(seq2)}")
    seq3 = _top_sequences_from_fat(fat_strings, 3, 5)
    if seq3:
        vals = [count_fat_sequence_hits(fs, seq3) for fs in fat_strings]
        add_interval('FAT 3-sekvenser', 'FAT-sekvenser', vals, lambda r, seqs=seq3: count_fat_sequence_hits(get_fat_string(r, filter_vec), seqs), 0, 85, 0, len(seq3), f"Toppar: {', '.join(seq3)}")
    pairs = _top_sequence_pairs_from_fat(fat_strings, 3, 5)
    if pairs:
        vals = [count_fat_pair_hits(fs, pairs) for fs in fat_strings]
        add_interval('FAT dubbelchans', 'FAT-sekvenser', vals, lambda r, prs=pairs: count_fat_pair_hits(get_fat_string(r, filter_vec), prs), 0, 85, 0, len(pairs), f"Par: {', '.join([str(p) for p in pairs])}")

    # Favorit / skräll
    streck_filter_items = _build_streck_filter_systems(filter_vec, hist_df, v_m, antal_matcher, max_shock_pct=max_shock_pct)
    for _item in streck_filter_items:
        _system = _item.get('system', [])
        _marked = int(_item.get('marked', u_system_marked_count(_system)))
        if _marked <= 0:
            continue
        _vals = [_u_system_metric(str(r), _system, 'utips') for r in rows if len(str(r)) == int(antal_matcher)]
        _help = (
            f"{_item.get('name')}: {_item.get('details')}. "
            f"Räknar hur många av de {_marked} rekommenderade markeringarna som enkelraden klarar. "
            f"Val: {_item.get('preview', '')}."
        )
        add_interval(
            _item.get('name'),
            'Favorit & skräll',
            _vals,
            lambda r, _sys=_system: _u_system_metric(r, _sys, 'utips'),
            0,
            90,
            0,
            _marked,
            _help,
            key_override=_item.get('key'),
        )

    for _n in range(3, 7):
        add_interval(
            f'Topp {_n} favoriter',
            'Favorit & skräll',
            fav_top_vals_by_n.get(_n, []),
            lambda r, _nn=_n: get_top_n_favs_wins(r, filter_vec, _nn),
            0,
            90,
            0,
            _n,
            f'Antal av kupongens {_n} högst streckade favoriter som vinner.',
            key_override=f'topp_{_n}_favoriter',
        )
    _fav_counts = get_favorite_threshold_counts(filter_vec)
    add_interval('70%+ favoriter som sitter', 'Favorit & skräll', fav70, lambda r: get_favorite_pressure(r, filter_vec)['F70_Wins'], 0, 85, 0, _fav_counts.get(70, 0), 'Räknar hur många av kupongens favoriter på minst 70 % som vinner på raden. Historiken räknas på samma antal/favorittecken för respektive historisk omgång.', key_override='favorittryck_70', meta={'diag_type': 'favorite_threshold', 'threshold': 70})
    add_interval('60%+ favoriter som sitter', 'Favorit & skräll', fav60, lambda r: get_favorite_pressure(r, filter_vec)['F60_Wins'], 0, 85, 0, _fav_counts.get(60, 0), 'Räknar hur många av kupongens favoriter på minst 60 % som vinner på raden. Historiken räknas på samma antal/favorittecken för respektive historisk omgång.', key_override='favorittryck_60', meta={'diag_type': 'favorite_threshold', 'threshold': 60})
    add_interval('50%+ favoriter som sitter', 'Favorit & skräll', fav50, lambda r: get_favorite_pressure(r, filter_vec)['F50_Wins'], 0, 85, 0, _fav_counts.get(50, 0), 'Räknar hur många av kupongens favoriter på minst 50 % som vinner på raden. Historiken räknas på samma antal/favorittecken för respektive historisk omgång.', key_override='favorittryck_50', meta={'diag_type': 'favorite_threshold', 'threshold': 50})
    _shock_caps = get_shock_capacity(filter_vec)
    add_interval('Vinnare under 10%', 'Favorit & skräll', sh10, lambda r: get_shock_strength(r, filter_vec)['U10_Wins'], 0, 85, 0, _shock_caps.get(10, 0), 'Räknar hur många valda vinnartecken på raden som är streckade under 10 %. Grundramsdiagnosen varnar om intervallet blir en dold spik.', key_override='skrallstyrka_u10', meta={'diag_type': 'shock_threshold', 'threshold': 10})
    add_interval('Vinnare under 15%', 'Favorit & skräll', sh15, lambda r: get_shock_strength(r, filter_vec)['U15_Wins'], 0, 85, 0, _shock_caps.get(15, 0), 'Räknar hur många valda vinnartecken på raden som är streckade under 15 %. Grundramsdiagnosen varnar om intervallet blir en dold spik.', key_override='skrallstyrka_u15', meta={'diag_type': 'shock_threshold', 'threshold': 15})
    add_interval('Vinnare under 20%', 'Favorit & skräll', sh20, lambda r: get_shock_strength(r, filter_vec)['U20_Wins'], 0, 85, 0, _shock_caps.get(20, 0), 'Räknar hur många valda vinnartecken på raden som är streckade under 20 %. Grundramsdiagnosen varnar om intervallet blir en dold spik.', key_override='skrallstyrka_u20', meta={'diag_type': 'shock_threshold', 'threshold': 20})
    add_interval('Lägsta vinnande %', 'Favorit & skräll', sh_low, lambda r: get_shock_strength(r, filter_vec)['Lowest_Win_Pct'], 1, 85, 0, 100)
    add_interval('Favorit-delta', 'Favorit & skräll', fav_delta_vals, lambda r: get_favorite_delta(r, filter_vec), 2, 85)

    # Struktur
    add_interval('Tecken 1', 'Struktur', ones, lambda r: r.count('1'), 0, 100, 0, antal_matcher)
    add_interval('Tecken X', 'Struktur', draws, lambda r: r.count('X'), 0, 100, 0, antal_matcher)
    add_interval('Tecken 2', 'Struktur', twos, lambda r: r.count('2'), 0, 100, 0, antal_matcher)
    add_interval('Följder 1', 'Struktur', s1, lambda r: get_streaks(r)[0], 0, 100, 0, antal_matcher, key_override='sviter_1')
    add_interval('Följder X', 'Struktur', sx, lambda r: get_streaks(r)[1], 0, 100, 0, antal_matcher, key_override='sviter_x')
    add_interval('Följder 2', 'Struktur', s2, lambda r: get_streaks(r)[2], 0, 100, 0, antal_matcher, key_override='sviter_2')
    add_interval('Längsta följd totalt', 'Struktur', stot, lambda r: get_streaks(r)[3], 0, 100, 0, antal_matcher)
    add_interval('Längsta lucka totalt', 'Struktur', gtot, lambda r: get_gaps(r)[3], 0, 100, 0, antal_matcher)
    add_interval('Luckor 1', 'Struktur', g1, lambda r: get_gaps(r)[0], 0, 100, 0, antal_matcher)
    add_interval('Luckor X', 'Struktur', gx, lambda r: get_gaps(r)[1], 0, 100, 0, antal_matcher)
    add_interval('Luckor 2', 'Struktur', g2, lambda r: get_gaps(r)[2], 0, 100, 0, antal_matcher)
    add_interval('Singlar 1', 'Struktur', si1, lambda r: get_singles(r)[0], 0, 100, 0, antal_matcher)
    add_interval('Singlar X', 'Struktur', six, lambda r: get_singles(r)[1], 0, 100, 0, antal_matcher)
    add_interval('Singlar 2', 'Struktur', si2, lambda r: get_singles(r)[2], 0, 100, 0, antal_matcher)
    add_interval('Singlar total', 'Struktur', sit, lambda r: get_singles(r)[3], 0, 100, 0, antal_matcher)
    add_interval('Dubbletter 1', 'Struktur', d1, lambda r: get_doublets(r)[0], 0, 100, 0, antal_matcher)
    add_interval('Dubbletter X', 'Struktur', dx, lambda r: get_doublets(r)[1], 0, 100, 0, antal_matcher)
    add_interval('Dubbletter 2', 'Struktur', d2, lambda r: get_doublets(r)[2], 0, 100, 0, antal_matcher)
    add_interval('Dubbletter total', 'Struktur', dt, lambda r: get_doublets(r)[3], 0, 100, 0, antal_matcher)
    add_interval('Tripplar 1', 'Struktur', tr1, lambda r: get_triplets(r)[0], 0, 100, 0, antal_matcher)
    add_interval('Tripplar X', 'Struktur', trx, lambda r: get_triplets(r)[1], 0, 100, 0, antal_matcher)
    add_interval('Tripplar 2', 'Struktur', tr2, lambda r: get_triplets(r)[2], 0, 100, 0, antal_matcher)
    add_interval('Tripplar total', 'Struktur', trt, lambda r: get_triplets(r)[3], 0, 100, 0, antal_matcher)
    add_interval('Uppkomster 1', 'Struktur', o1, lambda r: get_occurrences(r)[0], 0, 100, 0, antal_matcher)
    add_interval('Uppkomster X', 'Struktur', ox, lambda r: get_occurrences(r)[1], 0, 100, 0, antal_matcher)
    add_interval('Uppkomster 2', 'Struktur', o2, lambda r: get_occurrences(r)[2], 0, 100, 0, antal_matcher)
    add_interval('Uppkomster total', 'Struktur', ot, lambda r: get_occurrences(r)[3], 0, 100, 0, antal_matcher)

    # v12.0bb: Super-Makro är dyrt att rekommendera live eftersom det räknar många
    # underfilter per rad. Därför byggs Super-Makro-filter bara när användaren
    # aktivt slår på dem i sidomenyn. Alla övriga filter påverkas inte.
    if not include_supermakro:
        return specs

    # Super-Makro: bygg flera synliga makron. Strukturmakrot är gamla Super-Makro-logiken.
    super_macro_target_pct = _category_target_pct('Super-Makro', target_hist_pct)
    value_macro_specs = [sp for sp in specs if sp.get('category') == 'Värde & svårighet']
    favorite_macro_specs = [sp for sp in specs if sp.get('category') == 'Favorit & skräll']

    macro_bounds = {
        'ones': get_best_interval(ones, 90), 'draws': get_best_interval(draws, 90), 'twos': get_best_interval(twos, 90),
        's1': get_best_interval(s1, 90), 'sx': get_best_interval(sx, 90), 's2': get_best_interval(s2, 90),
        'g1': get_best_interval(g1, 90), 'gx': get_best_interval(gx, 90), 'g2': get_best_interval(g2, 90),
        'si1': get_best_interval(si1, 90), 'six': get_best_interval(six, 90), 'si2': get_best_interval(si2, 90),
        'd1': get_best_interval(d1, 90), 'dx': get_best_interval(dx, 90), 'd2': get_best_interval(d2, 90),
        't1': get_best_interval(tr1, 90), 'tx': get_best_interval(trx, 90), 't2': get_best_interval(tr2, 90),
        'o1': get_best_interval(o1, 90), 'ox': get_best_interval(ox, 90), 'o2': get_best_interval(o2, 90),
        'fat_f': get_best_interval(fat_f, 90), 'fat_a': get_best_interval(fat_a, 90), 'fat_t': get_best_interval(fat_t, 90),
    }
    macro_vals = [_super_macro_count(r, p, macro_bounds) for r, p in zip(rows, probs)]
    _macro_rec_interval, _macro_rec_table, _macro_sampled = _recommend_super_macro_interval(
        macro_vals,
        candidate_rows or [],
        filter_vec,
        macro_bounds,
        target_hist_pct=super_macro_target_pct,
    )
    _macro_help = (
        'Gamla Super-Makro-logiken, nu tydligare som Strukturmakro. Räknar 8 struktur/FAT-block: '
        'teckenfördelning, följder, luckor, singlar, dubbletter, tripplar, uppkomster och FAT. '
        'Varje block räknas om minst 2 av 3 interna delar sitter. Rekommendationen testar alla min-nivåer '
        'och väljer rationell nivå utifrån historisk träff, kvarvarande grundram och balans.'
    )
    if _macro_sampled:
        _macro_help += ' Kvar rad % i rekommendationstabellen bygger på ett deterministiskt urval av grundramen för att hålla appen snabb.'
    add_interval(
        'Super-Makro Struktur',
        'Super-Makro',
        macro_vals,
        lambda r, b=macro_bounds: _super_macro_count(r, filter_vec, b),
        0,
        super_macro_target_pct,
        0,
        8,
        _macro_help,
        key_override='super_makro_grupper',
        default_interval_override=_macro_rec_interval,
        meta={
            'diag_type': 'super_macro_rational',
            'recommendation_table': _macro_rec_table.to_dict('records') if isinstance(_macro_rec_table, pd.DataFrame) else [],
            'sampled_candidate_rows': bool(_macro_sampled),
            'max_blocks': 8,
        },
    )

    # Super-Makro Värde/Svårighet: räknar hur många av värde-/svårighetsfiltren som klarar sitt rekommenderade intervall.
    value_macro_vals = _macro_hist_values_from_specs(value_macro_specs)
    value_macro_max = len(value_macro_specs)
    if value_macro_max > 0 and value_macro_vals:
        _value_row_getter = lambda r, _specs=value_macro_specs: _macro_count_from_specs_row(r, _specs)
        _value_rec_interval, _value_rec_table, _value_sampled = _recommend_count_macro_interval(
            value_macro_vals,
            candidate_rows or [],
            _value_row_getter,
            value_macro_max,
            target_hist_pct=super_macro_target_pct,
        )
        _value_help = (
            f'Räknar hur många av {value_macro_max} filter i Värde & svårighet som klarar sina rekommenderade intervall. '
            'Detta är ett mjukt makrofilter: appen testar alla min-nivåer och väljer nivån med bäst rationell effekt, '
            'inte bara högsta antal träffade filter.'
        )
        if _value_sampled:
            _value_help += ' Kvar rad % bygger på ett deterministiskt urval av grundramen.'
        add_interval(
            'Super-Makro Värde/Svårighet',
            'Super-Makro',
            value_macro_vals,
            _value_row_getter,
            0,
            super_macro_target_pct,
            0,
            value_macro_max,
            _value_help,
            key_override='super_makro_varde_svarighet',
            default_interval_override=_value_rec_interval,
            meta={
                'diag_type': 'super_macro_rational',
                'recommendation_table': _value_rec_table.to_dict('records') if isinstance(_value_rec_table, pd.DataFrame) else [],
                'sampled_candidate_rows': bool(_value_sampled),
                'max_blocks': value_macro_max,
                'included_filters': [sp.get('name', '') for sp in value_macro_specs],
            },
        )
    else:
        _value_rec_interval = (0, 0)
        _value_row_getter = lambda r: 0

    # Super-Makro Favorit/Skräll: räknar hur många favorit-/skrällfilter som klarar sina rekommenderade intervall.
    favorite_macro_vals = _macro_hist_values_from_specs(favorite_macro_specs)
    favorite_macro_max = len(favorite_macro_specs)
    if favorite_macro_max > 0 and favorite_macro_vals:
        _fav_row_getter = lambda r, _specs=favorite_macro_specs: _macro_count_from_specs_row(r, _specs)
        _fav_rec_interval, _fav_rec_table, _fav_sampled = _recommend_count_macro_interval(
            favorite_macro_vals,
            candidate_rows or [],
            _fav_row_getter,
            favorite_macro_max,
            target_hist_pct=super_macro_target_pct,
        )
        _fav_help = (
            f'Räknar hur många av {favorite_macro_max} filter i Favorit & skräll som klarar sina rekommenderade intervall. '
            'Bra för att använda favorit-/skrällbilden mjukt i stället för att tvinga många överlappande filter samtidigt.'
        )
        if _fav_sampled:
            _fav_help += ' Kvar rad % bygger på ett deterministiskt urval av grundramen.'
        add_interval(
            'Super-Makro Favorit/Skräll',
            'Super-Makro',
            favorite_macro_vals,
            _fav_row_getter,
            0,
            super_macro_target_pct,
            0,
            favorite_macro_max,
            _fav_help,
            key_override='super_makro_favorit_skrall',
            default_interval_override=_fav_rec_interval,
            meta={
                'diag_type': 'super_macro_rational',
                'recommendation_table': _fav_rec_table.to_dict('records') if isinstance(_fav_rec_table, pd.DataFrame) else [],
                'sampled_candidate_rows': bool(_fav_sampled),
                'max_blocks': favorite_macro_max,
                'included_filters': [sp.get('name', '') for sp in favorite_macro_specs],
            },
        )
    else:
        _fav_rec_interval = (0, 0)
        _fav_row_getter = lambda r: 0

    # v12.0bc: Super-Makro Total är borttaget.
    # Total-makrot blev ett meta-på-meta-filter som dubbelräknade Struktur,
    # Värde/Svårighet och Favorit/Skräll och gjorde sidan onödigt tung.
    # De tre konkreta makrona ovan behålls som vanliga, synliga filter.


    return specs


def _apply_manual_filters(rows, specs, settings, group_reqs):
    forced_specs = []
    group_specs = {f'Grupp {i}': [] for i in range(1, 7)}
    for spec in specs:
        mode = settings.get(spec['key'], {}).get('mode', 'Av')
        if mode == 'Tvingat':
            forced_specs.append(spec)
        elif mode in group_specs:
            group_specs[mode].append(spec)

    filtered = []
    for r in rows:
        ok = True
        for spec in forced_specs:
            if not _spec_pass(r, spec, settings[spec['key']]['interval']):
                ok = False; break
        if not ok:
            continue
        for gname, gs in group_specs.items():
            if not gs:
                continue
            min_req, max_req = _group_req_bounds(group_reqs, gname, len(gs))
            if min_req <= 0 and max_req >= len(gs):
                continue
            hits = 0
            for spec in gs:
                if _spec_pass(r, spec, settings[spec['key']]['interval']):
                    hits += 1
            if hits < min_req or hits > max_req:
                ok = False; break
        if ok:
            filtered.append(r)
    return filtered


def _hist_package_passes(v_m, specs, settings, group_reqs, manual_sign_groups=None, antal_matcher=None):
    """Samlad historisk träff för aktiva filter.

    Viktigt: detta ska räknas på samma historiska filtervärden som används i
    frekvenstabellerna och i rekommenderade paket. Vissa filter beror på
    veckans procentvektor i sin getter, t.ex. AI-Rank/Delta/Poängfilter. Om vi
    kör getter direkt på historiska rättsrader med veckans filter_vec blir
    samlad träff fel och kan visa t.ex. 9/30 fast paketmotorn räknade 22/30.

    Därför räknar denna funktion med spec['hist_values'][i] för varje av de
    liknande historiska omgångarna. Det gör Filtercentralens samlade träff och
    paketmotorns träff jämförbara.

    v12.0bs: manuella teckengrupper ingår inte i denna träff. De är spelarens
    egna förfilter och visas med egen historikträff. Filtercentralens samlade
    träff ska därför kunna jämföras direkt med rekommenderat paket.
    Parametrarna manual_sign_groups/antal_matcher finns kvar för bakåtkompatibilitet
    men används inte här.
    """
    try:
        htot = int(max([len(s.get('hist_values', [])) for s in specs] + [len(v_m)]))
    except Exception:
        htot = len(v_m)
    if htot <= 0:
        return 0, 0

    forced_specs = []
    group_specs = {f'Grupp {i}': [] for i in range(1, 7)}
    for spec in specs:
        mode = settings.get(spec.get('key'), {}).get('mode', 'Av')
        if mode == 'Tvingat':
            forced_specs.append(spec)
        elif mode in group_specs:
            group_specs[mode].append(spec)

    def hist_value_pass(spec, idx):
        vals = spec.get('hist_values', []) or []
        if idx >= len(vals):
            return False
        interval = settings.get(spec.get('key'), {}).get('interval', spec.get('default_interval'))
        return in_range(vals[idx], interval)

    # Manuella teckengrupper ska inte sänka Filtercentralens/paketets
    # historikträff. De påverkar bara aktuell radmassa före paketmotorn och
    # visas separat i panelen för manuella teckengrupper.
    passed = 0
    for i in range(htot):
        ok = True
        for spec in forced_specs:
            if not hist_value_pass(spec, i):
                ok = False
                break
        if not ok:
            continue
        for gname, gs in group_specs.items():
            if not gs:
                continue
            min_req, max_req = _group_req_bounds(group_reqs, gname, len(gs))
            if min_req <= 0 and max_req >= len(gs):
                continue
            hits = 0
            for spec in gs:
                if hist_value_pass(spec, i):
                    hits += 1
            if hits < min_req or hits > max_req:
                ok = False
                break
        if ok:
            passed += 1
    return int(passed), int(htot)




def _hist_total_for_specs(v_m, specs):
    """Antal historiska observationer som filterpaketet ska räknas mot."""
    try:
        return int(max([len(s.get('hist_values', [])) for s in (specs or [])] + [len(v_m)]))
    except Exception:
        try:
            return int(len(v_m))
        except Exception:
            return 0


def _hist_value_pass_for_spec(spec, idx, settings):
    vals = spec.get('hist_values', []) or []
    if idx >= len(vals):
        return False, None
    val = vals[idx]
    interval = settings.get(spec.get('key'), {}).get('interval', spec.get('default_interval'))
    try:
        return bool(in_range(val, interval)), val
    except Exception:
        return False, val


def _active_specs_by_mode(specs, settings):
    forced_specs = []
    group_specs = {f'Grupp {i}': [] for i in range(1, 7)}
    for spec in specs or []:
        mode = settings.get(spec.get('key'), {}).get('mode', 'Av')
        if mode == 'Tvingat':
            forced_specs.append(spec)
        elif mode in group_specs:
            group_specs[mode].append(spec)
    return forced_specs, group_specs


def _active_package_diagnostic_df(v_m, specs, settings, group_reqs, antal_matcher, max_rows=100):
    """Rad-för-rad-diagnos för samlad historikträff.

    Detta är kontrollvyn som visar varför många starka individuella filter kan
    ge låg samlad träff när de kombineras med AND och gruppkrav.
    """
    htot = _hist_total_for_specs(v_m, specs)
    if htot <= 0:
        return pd.DataFrame()
    forced_specs, group_specs = _active_specs_by_mode(specs, settings)
    out = []
    for idx in range(min(int(htot), int(max_rows))):
        try:
            hist_row = normalize_single_row_text(v_m.iloc[idx].get('Correct_Row', '')) if idx < len(v_m) else ''
        except Exception:
            hist_row = ''
        ok = True
        forced_hits = 0
        forced_misses = []
        for spec in forced_specs:
            passed, val = _hist_value_pass_for_spec(spec, idx, settings)
            if passed:
                forced_hits += 1
            else:
                ok = False
                try:
                    val_txt = _format_filter_value(val, spec.get('decimals', 0))
                except Exception:
                    val_txt = '-'
                forced_misses.append(f"{spec.get('name','')}={val_txt}")

        group_parts = []
        group_misses = []
        for gname, gs in group_specs.items():
            if not gs:
                continue
            mn, mx = _group_req_bounds(group_reqs, gname, len(gs))
            if mn <= 0 and mx >= len(gs):
                continue
            hits = 0
            for spec in gs:
                passed, _ = _hist_value_pass_for_spec(spec, idx, settings)
                if passed:
                    hits += 1
            gok = (mn <= hits <= mx)
            if not gok:
                ok = False
                group_misses.append(f"{gname} {hits}/{len(gs)}")
            group_parts.append(f"{gname}: {hits}/{len(gs)} krav {mn}–{mx} {'OK' if gok else 'MISS'}")

        miss_txt = '; '.join(forced_misses[:6] + group_misses[:6])
        if len(forced_misses) + len(group_misses) > 12:
            miss_txt += ' …'
        out.append({
            'Historik #': idx + 1,
            'Rad': hist_row if hist_row else '-',
            'Paketstatus': '✅ Träff' if ok else '❌ Miss',
            'Tvingade': f"{forced_hits}/{len(forced_specs)}",
            'Grupper': ' | '.join(group_parts) if group_parts else '—',
            'Missar': miss_txt if miss_txt else '—',
        })
    return pd.DataFrame(out)


def _active_group_diagnostic_df(specs, settings, group_reqs, frame_rows=None):
    """Synlig gruppdiagnos för aktuell filtercentral."""
    _, group_specs = _active_specs_by_mode(specs, settings)
    htot = _hist_total_for_specs(pd.DataFrame(), specs)
    rows = []
    for gname, gs in group_specs.items():
        n = len(gs)
        if n <= 0:
            continue
        mn, mx = _group_req_bounds(group_reqs, gname, n)
        hist_scores = []
        hist_pass = 0
        for idx in range(htot):
            hits = 0
            for spec in gs:
                passed, _ = _hist_value_pass_for_spec(spec, idx, settings)
                if passed:
                    hits += 1
            hist_scores.append(hits)
            if mn <= hits <= mx:
                hist_pass += 1

        frame_keep_txt = '—'
        frame_red_txt = '—'
        if frame_rows is not None and len(frame_rows) <= 30000:
            try:
                keep = 0
                for r in frame_rows:
                    hits = 0
                    for spec in gs:
                        interval = settings.get(spec.get('key'), {}).get('interval', spec.get('default_interval'))
                        if _spec_pass(r, spec, interval):
                            hits += 1
                    if mn <= hits <= mx:
                        keep += 1
                frame_keep_txt = f"{keep}/{len(frame_rows)}"
                frame_red_txt = f"{100.0 - 100.0 * keep / max(1, len(frame_rows)):.1f}%"
            except Exception:
                pass

        if hist_scores:
            score_txt = f"min {min(hist_scores)} · median {float(np.median(hist_scores)):.1f} · max {max(hist_scores)}"
        else:
            score_txt = '—'
        rows.append({
            'Grupp': gname,
            'Filter i grupp': int(n),
            'Krav': _group_req_label(group_reqs, gname, n),
            'Historisk gruppträff': f"{hist_pass}/{htot}" if htot else '0/0',
            'Träffar i historik': score_txt,
            'Grundram kvar': frame_keep_txt,
            'Reducerar grundram': frame_red_txt,
            'Status': 'Aktiv' if not (mn <= 0 and mx >= n) else 'Påverkar inte',
        })
    return pd.DataFrame(rows)

def _settings_package_signature(settings, group_reqs=None):
    """Stabil signatur för aktiv filtercentral.

    v12.0ai: Gruppkrav ingår i signaturen. Annars kunde ett applicerat
    hårt grupppaket visa falsk mismatch: filterlägena stämde, men aktuell
    signatur saknade Grupp min/max som paketmotorn räknade med.
    """
    sig = []
    active_group_counts = {f'Grupp {i}': 0 for i in range(1, 7)}
    for k, v in sorted((settings or {}).items()):
        mode = v.get('mode', 'Av')
        if mode == 'Av':
            continue
        interval = v.get('interval')
        try:
            iv = (round(float(interval[0]), 6), round(float(interval[1]), 6))
        except Exception:
            iv = tuple(interval) if isinstance(interval, (list, tuple)) else interval
        sig.append((k, mode, iv))
        if mode in active_group_counts:
            active_group_counts[mode] += 1

    for gname, n in active_group_counts.items():
        if n <= 0:
            continue
        mn, mx = _group_req_bounds(group_reqs or {}, gname, n)
        sig.append((gname, 'REQ', int(mn), int(mx), int(n)))
    return tuple(sorted(sig))


def _package_signature(package):
    sig = []
    for c in package.get('filters', []) or []:
        interval = c.get('interval')
        try:
            iv = (round(float(interval[0]), 6), round(float(interval[1]), 6))
        except Exception:
            iv = tuple(interval) if isinstance(interval, (list, tuple)) else interval
        mode = c.get('package_mode', 'Tvingat')
        sig.append((c.get('key'), mode, iv))
    for g in package.get('groups', []) or []:
        sig.append((g.get('name'), 'REQ', int(g.get('req', 0)), int(g.get('max_req', g.get('n', 0))), int(g.get('n', 0))))
    return tuple(sorted(sig))



def _group_req_bounds(group_reqs, gname, n_filters=0):
    """Returnerar (min, max) för en grupp, bakåtkompatibelt med gamla heltalskrav.

    Gamla versioner sparade bara `Grupp 1: 4`, vilket betydde minst 4 av N.
    v12.0af sparar `{'min': 4, 'max': 6}` så gruppen kan styras som min/max.
    """
    n_filters = int(max(0, n_filters or 0))
    raw = (group_reqs or {}).get(gname, None)
    if isinstance(raw, dict):
        mn = int(raw.get('min', raw.get('req', 0)) or 0)
        mx = int(raw.get('max', raw.get('max_req', n_filters)) or n_filters)
    else:
        mn = int(raw or 0)
        mx = n_filters
    mn = max(0, min(mn, n_filters))
    mx = max(0, min(mx, n_filters))
    if mx < mn:
        mx = mn
    return mn, mx


def _group_req_label(group_reqs, gname, n_filters=0):
    mn, mx = _group_req_bounds(group_reqs, gname, n_filters)
    if n_filters <= 0:
        return '—'
    if mn <= 0 and mx >= n_filters:
        return '0 = påverkar inte'
    if mx >= n_filters:
        return f'minst {mn} av {n_filters}'
    return f'{mn}–{mx} av {n_filters}'


def _json_safe_value(x):
    """Gör numpy/pandas-värden JSON-vänliga utan att förstöra listor/dicts."""
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return [_json_safe_value(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_json_safe_value(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_safe_value(v) for k, v in x.items()}
    return x


def _json_safe_dataframe(df):
    """Returnerar DataFrame som JSON-vänliga poster. Tom lista om inget går att spara."""
    try:
        if isinstance(df, pd.DataFrame) and not df.empty:
            return _json_safe_value(df.to_dict(orient='records'))
    except Exception:
        pass
    return []


def _sync_frame_widget_token():
    """Tvingar Streamlit att bygga om grundrams-checkboxar med sparade värden.

    Checkboxar med fasta key-värden behåller annars gammalt widget-state även
    efter att en spelfil har lästs in. Då kan alla rutor se ikryssade ut medan
    kolumnen Val visar den sparade ramen. Token i key löser det utan att ändra
    själva grundramslogiken.
    """
    try:
        st.session_state['v12_frame_widget_token'] = int(st.session_state.get('v12_frame_widget_token', 0) or 0) + 1
    except Exception:
        st.session_state['v12_frame_widget_token'] = 1


def _sync_frame_widget_state_from_frame(frame, antal_matcher):
    """Sätter även kända widgetnycklar så äldre sessioner blir stabila direkt."""
    try:
        token = int(st.session_state.get('v12_frame_widget_token', 0) or 0)
    except Exception:
        token = 0
    for i in range(int(antal_matcher)):
        signs = set(normalize_signs(frame[i] if i < len(frame) else ['1', 'X', '2']))
        # Gamla fasta nycklar + nya token-nycklar. Sätts före widgetarna skapas.
        for prefix in (f'v12_frame_{i}', f'v12_frame_{token}_{i}'):
            st.session_state[f'{prefix}_1'] = ('1' in signs)
            st.session_state[f'{prefix}_x'] = ('X' in signs)
            st.session_state[f'{prefix}_2'] = ('2' in signs)



# v12.0co: Filtercentralen renderar bara en kategori åt gången. Streamlit kan då
# städa bort widget-state för filterrader som inte längre renderas. Därför
# speglar vi alla filterval till en separat, persistent store som inte är
# widget-bunden. Utan detta kan Struktur-filter försvinna när användaren öppnar
# FAT-fliken, vilket var den buggen Niklas hittade.
def _filtercentral_persist_store():
    store = st.session_state.get('v12_filtercentral_persisted_settings')
    if not isinstance(store, dict):
        store = {}
        st.session_state['v12_filtercentral_persisted_settings'] = store
    return store


def _filtercentral_range_key_for_spec(spec, filter_hist_target_pct=95, top_fav_count=3):
    try:
        _fhp_spec = _hist_target_for_spec(spec, filter_hist_target_pct)
    except Exception:
        _fhp_spec = filter_hist_target_pct
    return f"filter_range_{spec.get('key')}_h{int(_fhp_spec)}_tf{int(top_fav_count)}"


def _filtercentral_store_set(k, mode=None, interval=None, spec=None, target_hist_pct=None):
    if not k:
        return
    store = _filtercentral_persist_store()
    item = store.get(k)
    if not isinstance(item, dict):
        item = {}
    if spec is not None:
        item['name'] = spec.get('name', k)
        item['category'] = spec.get('category', '')
        try:
            item['target_hist_pct'] = int(_hist_target_for_spec(spec, target_hist_pct or 95))
        except Exception:
            pass
    if mode is not None:
        item['mode'] = str(mode)
    if interval is not None:
        try:
            item['interval'] = (interval[0], interval[1])
        except Exception:
            item['interval'] = interval
    store[k] = item
    st.session_state['v12_filtercentral_persisted_settings'] = store


def _sync_rendered_filter_widgets_to_store(specs, filter_hist_target_pct=95, top_fav_count=3):
    """Kopiera renderade widgetvärden till persistent store.

    Körs innan kategoribyte hinner städa bort gamla widgetnycklar, och efter
    formulärsubmit så nya värden sparas. Läser bara widgetnycklar som faktiskt
    finns i session_state.
    """
    for spec in specs or []:
        k = spec.get('key')
        if not k:
            continue
        mode_key = f'filter_mode_{k}'
        range_key = _filtercentral_range_key_for_spec(spec, filter_hist_target_pct, top_fav_count)
        mode = st.session_state.get(mode_key) if mode_key in st.session_state else None
        interval = st.session_state.get(range_key) if range_key in st.session_state else None
        if mode is not None or interval is not None:
            _filtercentral_store_set(k, mode=mode, interval=interval, spec=spec, target_hist_pct=filter_hist_target_pct)


def _filtercentral_get_persisted_mode(k, default='Av'):
    item = _filtercentral_persist_store().get(k, {})
    if isinstance(item, dict):
        return item.get('mode', default)
    return default


def _filtercentral_get_persisted_interval(spec, default_interval=None):
    k = spec.get('key') if isinstance(spec, dict) else None
    item = _filtercentral_persist_store().get(k, {}) if k else {}
    if isinstance(item, dict) and isinstance(item.get('interval'), (list, tuple)) and len(item.get('interval')) >= 2:
        return (item.get('interval')[0], item.get('interval')[1])
    return default_interval if default_interval is not None else (spec.get('default_interval') if isinstance(spec, dict) else None)


def _filtercentral_clear_persistent_store():
    st.session_state['v12_filtercentral_persisted_settings'] = {}



def _clear_filtercentral_widget_state_for_load():
    """Rensar gamla filterwidgets innan spelfil/filterpaket läses in.

    Streamlit behåller session_state mellan uppladdningar. Om en tidigare kupong
    hade aktiva filter kan de annars ligga kvar om den nya sparfilen saknar en
    nyckel, eller om ett intervall har samma widgetnamn men annan innebörd.
    """
    for _k in list(st.session_state.keys()):
        _ks = str(_k)
        if _ks.startswith('filter_mode_') or _ks.startswith('filter_range_'):
            st.session_state.pop(_k, None)
    _filtercentral_clear_persistent_store()


def _apply_saved_applied_package_snapshot(package, saved_filter_keys=None, filter_hist_target_pct=90, top_fav_count=3):
    """Återställer exakt det rekommenderade paket som var applicerat när filen sparades.

    v12.0bx: Sparfilen innehåller både en vanlig filtercentral och en snapshot
    av valt rekommenderat paket. Vid uppladdning ska paket-snapshoten vara
    auktoritativ när paketet fortfarande var aktivt vid sparning; annars kan
    filterintervall tappas när träffmålets slider bygger nya nycklar.
    """
    if not isinstance(package, dict):
        return False
    filters = package.get('filters') or []
    groups = package.get('groups') or []
    if not filters and not groups:
        return False
    fhp = int(filter_hist_target_pct or 90)
    tfc = int(top_fav_count or 3)

    # Stäng av alla filter som sparfilen känner till innan paketets egna filter sätts.
    for _k in set(saved_filter_keys or []):
        if _k:
            st.session_state[f'filter_mode_{_k}'] = 'Av'
            _filtercentral_store_set(_k, mode='Av')

    for c in filters:
        if not isinstance(c, dict):
            continue
        k = c.get('key')
        if not k:
            continue
        _mode = c.get('package_mode', 'Tvingat')
        st.session_state[f'filter_mode_{k}'] = _mode
        interval = c.get('interval')
        if isinstance(interval, (list, tuple)) and len(interval) >= 2:
            _fhp_filter = _clamp_filter_hist_target(c.get('target_hist_pct', fhp), fhp)
            _interval = (interval[0], interval[1])
            st.session_state[f'filter_range_{k}_h{_fhp_filter}_tf{tfc}'] = _interval
            _filtercentral_store_set(k, mode=_mode, interval=_interval, target_hist_pct=_fhp_filter)
        else:
            _filtercentral_store_set(k, mode=_mode, target_hist_pct=fhp)

    # Gruppkrav från paketet ska ersätta gamla sidomenykrav.
    for i in range(1, 7):
        st.session_state[f'group_req_{i}'] = 0
        st.session_state[f'group_req_min_{i}'] = 0
        st.session_state[f'group_req_max_{i}'] = 40
    for g in groups:
        if not isinstance(g, dict):
            continue
        try:
            gi = int(str(g.get('name', 'Grupp 0')).split()[-1])
            if 1 <= gi <= 6:
                mn = int(g.get('req', g.get('min_req', 0)) or 0)
                mx = int(g.get('max_req', g.get('n', 40)) or 40)
                st.session_state[f'group_req_{gi}'] = mn
                st.session_state[f'group_req_min_{gi}'] = mn
                st.session_state[f'group_req_max_{gi}'] = mx
        except Exception:
            pass

    st.session_state['v12_applied_package_meta'] = {
        'hist_hit': int(package.get('hist_hit', 0) or 0),
        'hist_total': int(package.get('hist_total', 0) or 0),
        'frame_after': int(package.get('frame_after', 0) or 0),
        'num_filters': int(package.get('num_filters', len(filters)) or 0),
        'signature': _package_signature(package),
        'package_type': package.get('package_type', ''),
        'reduction_pct': float(package.get('reduction_pct', 0.0) or 0.0),
    }
    st.session_state['v12_applied_package_snapshot'] = _json_safe_value(package)
    return True

def _package_for_spelfil(package):
    """Sparar paket utan stora mask-arrayer.

    Maskerna behövs bara under själva paketberäkningen. För att visa och
    applicera ett sparat paket räcker filterkey, intervall, steg och metadata.
    """
    if not isinstance(package, dict):
        return package
    out = {}
    for k, v in package.items():
        if k in {'hist_mask', 'frame_mask'}:
            continue
        if k == 'filters' and isinstance(v, list):
            clean_filters = []
            for c in v:
                if isinstance(c, dict):
                    clean_filters.append({kk: _json_safe_value(vv) for kk, vv in c.items() if kk not in {'hist_mask', 'frame_mask'}})
                else:
                    clean_filters.append(_json_safe_value(c))
            out[k] = clean_filters
        else:
            out[k] = _json_safe_value(v)
    return out


def _packages_for_spelfil(packages):
    return [_package_for_spelfil(p) for p in (packages or []) if isinstance(p, dict)]


def _last_result_for_spelfil(result):
    """Sparar senast filtrerade/reducerade rader i spelfilen.

    Spelfilen kan bli större, men användaren slipper köra om reduceringen när
    systemet öppnas igen. Raderna är redan beräknat resultat och används inte
    för nya filterbeslut.
    """
    if not isinstance(result, dict):
        return {}
    out = {}
    for k in ['filtered_rows', 'reduced_rows']:
        rows = result.get(k) or []
        if isinstance(rows, tuple) and len(rows) == 2 and isinstance(rows[0], list):
            rows = rows[0]
        out[k] = [str(r).strip().upper() for r in rows if isinstance(r, str)]
    for k in ['settings', 'group_reqs', 'hist_package', 'tm_meta']:
        if k in result:
            out[k] = _json_safe_value(result.get(k))
    return out


def _apply_last_result_from_spelfil(payload):
    lr = (payload or {}).get('last_result') or {}
    if not isinstance(lr, dict) or not lr:
        return
    restored = {
        'filtered_rows': [str(r).strip().upper() for r in (lr.get('filtered_rows') or []) if isinstance(r, str)],
        'reduced_rows': [str(r).strip().upper() for r in (lr.get('reduced_rows') or []) if isinstance(r, str)],
        'settings': lr.get('settings') or {},
        'group_reqs': lr.get('group_reqs') or {},
        'hist_package': lr.get('hist_package') or {},
        'tm_meta': lr.get('tm_meta') or {},
    }
    st.session_state['v12_last_result'] = restored
    st.session_state['v12_last_result_stale'] = bool((payload or {}).get('last_result_stale', False))


def _history_records_for_spelfil(v_m):
    """Sparar bara det som behövs för att återskapa filterstatistik utan ny databasläsning."""
    records = []
    try:
        for _, row in v_m.iterrows():
            records.append({
                'Correct_Row': normalize_single_row_text(row.get('Correct_Row', '')),
                'Prob_Vector': _json_safe_value(row.get('Prob_Vector', [])),
            })
    except Exception:
        pass
    return records


def _history_df_from_records(records):
    rows = []
    for r in records or []:
        cr = normalize_single_row_text((r or {}).get('Correct_Row', ''))
        pv = (r or {}).get('Prob_Vector', []) or []
        try:
            pv = [float(x) for x in pv]
        except Exception:
            pv = []
        if cr and pv:
            rows.append({'Correct_Row': cr, 'Prob_Vector': pv})
    return pd.DataFrame(rows)


def _collect_filter_settings_for_save(specs, filter_hist_target_pct, top_fav_count, settings_override=None):
    """Samlar filterläge/intervall för sparfil.

    Viktigt: om användaren precis har applicerat ett rekommenderat paket eller
    ändrat filter i UI:t ska sparfilen få exakt den filtercentral som visas nu.
    Därför kan funktionen läsa från det renderade `settings`-objektet i stället
    för enbart session_state. Session_state används som fallback för
    bakåtkompatibilitet och tidiga sparlägen.
    """
    out = {}
    settings_override = settings_override or {}
    for spec in specs or []:
        k = spec.get('key')
        if not k:
            continue
        if isinstance(settings_override.get(k), dict):
            mode = settings_override[k].get('mode', 'Av')
            interval = settings_override[k].get('interval', spec.get('default_interval'))
        else:
            mode = _filtercentral_get_persisted_mode(k, st.session_state.get(f'filter_mode_{k}', 'Av'))
            _fhp_spec = _hist_target_for_spec(spec, filter_hist_target_pct)
            range_key = f'filter_range_{k}_h{int(_fhp_spec)}_tf{int(top_fav_count)}'
            interval = st.session_state.get(range_key, _filtercentral_get_persisted_interval(spec, spec.get('default_interval')))
        out[k] = {
            'name': spec.get('name', k),
            'category': spec.get('category', ''),
            'target_hist_pct': int(_hist_target_for_spec(spec, filter_hist_target_pct)),
            'mode': mode,
            'interval': _json_safe_value(interval),
        }
    return out


def _collect_recommended_package_state_for_save(specs=None, settings=None, group_reqs=None):
    """Sparar paketmotorns styrvärden, obligatoriska filter och paketlista.

    v12.0bv: obligatoriska paketfilter har en egen lista i session_state
    (`v12_required_pkg_keys`). Tidigare sparläget läste gamla widgetnycklar,
    vilket gjorde att kryssade "måste ingå"-filter kunde tappas vid spara/öppna.
    Nu sparas även beräknade paket och kandidatanalys så en spelfil öppnar med
    samma rekommenderade paketlista som när den sparades.
    """
    specs = specs or []
    valid_keys = {s.get('key') for s in specs if s.get('key')}
    required_keys = [k for k in (st.session_state.get('v12_required_pkg_keys', []) or []) if k in valid_keys]
    candidate_audit = st.session_state.get('v12_recommended_candidate_audit')
    applied_meta = st.session_state.get('v12_applied_package_meta') or {}
    applied_is_active = None
    if isinstance(settings, dict) and isinstance(applied_meta, dict) and applied_meta.get('signature'):
        try:
            applied_is_active = (_settings_package_signature(settings, group_reqs or {}) == applied_meta.get('signature'))
        except Exception:
            applied_is_active = None
    out = {
        'rec_min_step': _json_safe_value(st.session_state.get('v12_rec_min_step')),
        'rec_max_filters': _json_safe_value(st.session_state.get('v12_rec_max_filters')),
        'rec_min_hit': _json_safe_value(st.session_state.get('v12_rec_min_hit')),
        'rec_display_max_rows': _json_safe_value(st.session_state.get('v12_rec_display_max_rows')),
        'rec_frame_adapt': _json_safe_value(st.session_state.get('v12_rec_frame_adapt')),
        'rec_min_value_filters': _json_safe_value(st.session_state.get('v12_rec_min_value_filters')),
        'required_keys': _json_safe_value(required_keys),
        'recommended_meta': _json_safe_value(st.session_state.get('v12_recommended_meta', {})),
        'recommended_packages': _packages_for_spelfil(st.session_state.get('v12_recommended_packages', []) or []),
        'candidate_audit_records': _json_safe_dataframe(candidate_audit),
        'applied_package_meta': _json_safe_value(st.session_state.get('v12_applied_package_meta', {})),
        'applied_package_snapshot': _package_for_spelfil(st.session_state.get('v12_applied_package_snapshot', {})),
        'applied_package_is_active': _json_safe_value(applied_is_active),
    }
    return out


def _current_group_reqs_from_session():
    out = {}
    for i in range(1, 7):
        # Bakåtkompatibel default: om gamla group_req_i finns används den som min.
        old = int(st.session_state.get(f'group_req_{i}', 0) or 0)
        mn = int(st.session_state.get(f'group_req_min_{i}', old) or 0)
        mx = int(st.session_state.get(f'group_req_max_{i}', 40) or 40)
        out[f'Grupp {i}'] = {'min': mn, 'max': mx}
    return out


def _apply_filter_settings_to_session(saved_settings, filter_hist_target_pct, top_fav_count, category_hist_target_pct=None):
    category_hist_target_pct = category_hist_target_pct or {}
    for k, v in (saved_settings or {}).items():
        if not isinstance(v, dict):
            continue
        _mode = v.get('mode', 'Av')
        st.session_state[f'filter_mode_{k}'] = _mode
        _filtercentral_store_set(k, mode=_mode)
        interval = v.get('interval')
        if isinstance(interval, (list, tuple)) and len(interval) >= 2:
            _cat = v.get('category', '')
            _fhp_filter = v.get('target_hist_pct', category_hist_target_pct.get(_cat, filter_hist_target_pct))
            _fhp_filter = _clamp_filter_hist_target(_fhp_filter, filter_hist_target_pct)
            range_key = f'filter_range_{k}_h{int(_fhp_filter)}_tf{int(top_fav_count)}'
            # Behåll int/float-typen från filen. Streamlits int-sliders kan annars
            # bli griniga om de får float-värden i session_state.
            _interval = (interval[0], interval[1])
            st.session_state[range_key] = _interval
            _filtercentral_store_set(k, mode=_mode, interval=_interval, target_hist_pct=_fhp_filter)


def _apply_group_reqs_to_session(group_reqs):
    for i in range(1, 7):
        gname = f'Grupp {i}'
        raw = (group_reqs or {}).get(gname, {})
        if isinstance(raw, dict):
            mn = int(raw.get('min', raw.get('req', 0)) or 0)
            mx = int(raw.get('max', raw.get('max_req', 40)) or 40)
        else:
            mn = int(raw or 0)
            mx = 40
        st.session_state[f'group_req_min_{i}'] = mn
        st.session_state[f'group_req_max_{i}'] = mx
        st.session_state[f'group_req_{i}'] = mn


def _build_filterpaket_payload(specs, group_reqs, filter_hist_target_pct, top_fav_count, spelform, antal_matcher, settings_override=None, package_state=None, manual_sign_groups=None):
    package_state = package_state if package_state is not None else _collect_recommended_package_state_for_save(specs)
    return {
        'file_type': 'tipset_filterpaket',
        'app_version': APP_VERSION,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'spelform': spelform,
        'antal_matcher': int(antal_matcher),
        'filter_hist_target_pct': int(filter_hist_target_pct),
        'filter_hist_target_pct_by_category': _json_safe_value(_get_filter_hist_target_pct_by_category()),
        'top_fav_filters': 'Topp 3/4/5/6',
        'filters': _collect_filter_settings_for_save(specs, filter_hist_target_pct, top_fav_count, settings_override=settings_override),
        'group_reqs': _json_safe_value(group_reqs),
        'recommended_package_state': _json_safe_value(package_state or {}),
        'manual_sign_groups': _json_safe_value(manual_sign_groups if manual_sign_groups is not None else st.session_state.get('v12_manual_sign_groups', [])),
        'u_rows': {},
    }


def _build_spelfil_payload(specs, group_reqs, filter_hist_target_pct, top_fav_count, spelform, antal_matcher, input_text, top_n, pay_min, pay_max, frame, v_m, filter_vec, reducer_settings=None, settings_override=None, package_state=None, manual_sign_groups=None):
    payload = _build_filterpaket_payload(specs, group_reqs, filter_hist_target_pct, top_fav_count, spelform, antal_matcher, settings_override=settings_override, package_state=package_state, manual_sign_groups=manual_sign_groups)
    payload.update({
        'file_type': 'tipset_spelfil',
        'input_text': input_text or '',
        'top_n': int(top_n),
        'pay_min': int(pay_min),
        'pay_max': int(pay_max),
        'frame': _json_safe_value(frame),
        'filter_vec': _json_safe_value(filter_vec or []),
        'history_records': _history_records_for_spelfil(v_m),
        'reducer_settings': _json_safe_value(reducer_settings or {}),
        'last_result': _last_result_for_spelfil(st.session_state.get('v12_last_result')),
        'last_result_stale': bool(st.session_state.get('v12_last_result_stale', False)),
    })
    return payload


def _payload_to_json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')


def _load_payload_from_uploaded_file(uploaded):
    raw = uploaded.read()
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return json.loads(raw)


def _apply_spelfil_payload(payload):
    """Läser in spelfil/filterpaket i session_state. Körs helst före huvudwidgets skapas."""
    ftype = str((payload or {}).get('file_type', '')).lower()
    # Rensa beräknade objekt från tidigare session innan filens objekt läggs in.
    # Annars kan gamla rekommenderade paket eller reduceringsresultat synas efter
    # att en annan spelfil/filterpaket öppnats.
    for _k in ['v12_recommended_packages', 'v12_recommended_candidate_audit', 'v12_recommended_meta', 'v12_applied_package_meta', 'v12_applied_package_snapshot']:
        st.session_state.pop(_k, None)
    st.session_state['v12_last_result'] = None
    st.session_state['v12_last_result_stale'] = False
    # Grundinställningar: sätts tidigt i sidebar innan selectbox/text_area byggs.
    if payload.get('spelform'):
        st.session_state['v12_spelform'] = payload.get('spelform')
    if payload.get('input_text') is not None:
        st.session_state['v12_input_text'] = payload.get('input_text', '')
    if payload.get('top_n') is not None:
        st.session_state['v12_top_n'] = int(payload.get('top_n') or 30)
    if payload.get('pay_min') is not None:
        st.session_state['v12_pay_min'] = int(payload.get('pay_min') or 0)
    if payload.get('pay_max') is not None:
        st.session_state['v12_pay_max'] = int(payload.get('pay_max') or 2500000)
    elif payload.get('file_type') == 'tipset_spelfil':
        st.session_state['v12_pay_max'] = 2500000
    if payload.get('filter_hist_target_pct') is not None:
        st.session_state['v12_filter_hist_target_pct'] = int(payload.get('filter_hist_target_pct') or 90)
    if isinstance(payload.get('filter_hist_target_pct_by_category'), dict):
        _set_filter_hist_target_pct_by_category(payload.get('filter_hist_target_pct_by_category') or {})
    elif payload.get('filter_hist_target_pct') is not None:
        # Bakåtkompatibilitet: gamla sparfiler hade ett globalt värde. Använd det
        # i alla kategorier så gamla intervallnycklar kan läsas tillbaka.
        _legacy_fhp = int(payload.get('filter_hist_target_pct') or 90)
        _set_filter_hist_target_pct_by_category({cat: _legacy_fhp for cat in FILTER_HIST_TARGET_DEFAULTS_BY_CATEGORY})
    # top_fav_count är borttaget i v12.0ar. Toppfavoriter finns som fasta filter 3/4/5/6.
    # Använd fast bakåtkompatibel slidernyckel så gamla sparade intervall kan läsas in stabilt.
    fhp = int(st.session_state.get('v12_filter_hist_target_pct', payload.get('filter_hist_target_pct', 90)) or 90)
    category_fhp = _get_filter_hist_target_pct_by_category()
    tfc = 3
    # Viktigt vid uppladdning: gamla filter/slidervärden från tidigare kupong
    # ska bort, och träffmålets prev-nyckel ska synkas. Annars raderas de
    # nyss laddade intervallen direkt i filtercentralen och paketet kan visas
    # som t.ex. 11/30 fast sparad snapshot var 29/30.
    _clear_filtercentral_widget_state_for_load()
    st.session_state['v12_filter_hist_target_prev'] = int(fhp)
    st.session_state['v12_filter_hist_target_prev_by_category'] = dict(category_fhp)
    _apply_u_rows_to_session(payload.get('u_rows', {}))
    if payload.get('manual_sign_groups') is not None:
        st.session_state['v12_manual_sign_groups'] = _normalize_manual_sign_groups(payload.get('manual_sign_groups') or [], int(payload.get('antal_matcher') or 13))
    _apply_filter_settings_to_session(payload.get('filters', {}), fhp, tfc, category_fhp)
    _apply_group_reqs_to_session(payload.get('group_reqs', {}))

    pkg_state = payload.get('recommended_package_state') or {}
    if isinstance(pkg_state, dict):
        # Återställ paketmotorns styrvärden och obligatoriska filter. Detta gör
        # att en sparad spelfil/filterpaket inte bara återställer filterlägena,
        # utan även vilken rekommenderad paketkonfiguration filen byggdes från.
        _map = {
            'rec_min_step': 'v12_rec_min_step',
            'rec_max_filters': 'v12_rec_max_filters',
            'rec_min_hit': 'v12_rec_min_hit',
            'rec_display_max_rows': 'v12_rec_display_max_rows',
            'rec_frame_adapt': 'v12_rec_frame_adapt',
            'rec_min_value_filters': 'v12_rec_min_value_filters',
        }
        _pkg_settings_loaded = False
        for src, dst in _map.items():
            if pkg_state.get(src) is not None:
                st.session_state[dst] = pkg_state.get(src)
                _pkg_settings_loaded = True
        if _pkg_settings_loaded:
            # En sparad spelfil/filterpaket ska få behålla sina egna paketmotorvärden
            # och inte skrivas över av nya standardvärden längre ner i gränssnittet.
            st.session_state['v12_pkg_defaults_version'] = 'v12.0bw'
        req_loaded = [str(rk) for rk in (pkg_state.get('required_keys', []) or []) if rk]
        st.session_state['v12_required_pkg_keys'] = list(dict.fromkeys(req_loaded))
        # Bakåtkompatibilitet för äldre sessionsnycklar/formulär.
        for rk in req_loaded:
            st.session_state[f'v12_reqpkg_{rk}'] = True
            st.session_state[f'v12_reqpkg_form_{rk}'] = True
        if isinstance(pkg_state.get('recommended_meta'), dict):
            st.session_state['v12_recommended_meta'] = pkg_state.get('recommended_meta') or {}
        if isinstance(pkg_state.get('recommended_packages'), list):
            st.session_state['v12_recommended_packages'] = pkg_state.get('recommended_packages') or []
        if isinstance(pkg_state.get('candidate_audit_records'), list) and pkg_state.get('candidate_audit_records'):
            st.session_state['v12_recommended_candidate_audit'] = pd.DataFrame(pkg_state.get('candidate_audit_records') or [])
        elif 'candidate_audit_records' in pkg_state:
            st.session_state['v12_recommended_candidate_audit'] = pd.DataFrame()
        if isinstance(pkg_state.get('applied_package_meta'), dict):
            st.session_state['v12_applied_package_meta'] = pkg_state.get('applied_package_meta') or {}
        if isinstance(pkg_state.get('applied_package_snapshot'), dict):
            st.session_state['v12_applied_package_snapshot'] = pkg_state.get('applied_package_snapshot') or {}
        # Om sparfilen hade ett applicerat rekommenderat paket ska samma paket
        # läggas in i filtercentralen igen. Nya v12.0bx-filer anger om paketet
        # fortfarande var aktivt vid sparning; äldre filer saknar flaggan och då
        # väljer vi paket-snapshoten framför en potentiellt stale filterlista.
        _restore_applied = pkg_state.get('applied_package_is_active', None)
        _snapshot = pkg_state.get('applied_package_snapshot') or {}
        if _restore_applied is not False and isinstance(_snapshot, dict) and (_snapshot.get('filters') or _snapshot.get('groups')):
            _saved_filter_keys = list((payload.get('filters') or {}).keys())
            if _apply_saved_applied_package_snapshot(_snapshot, _saved_filter_keys, fhp, tfc):
                st.session_state['v12_loaded_applied_package_restored'] = True

    if ftype == 'tipset_spelfil':
        frame = payload.get('frame') or []
        if frame:
            clean_frame = [[s for s in normalize_signs(x)] for x in frame]
            st.session_state['v12_saved_frame'] = clean_frame
            st.session_state['v12_frame_saved'] = True
            st.session_state['v12_frame_defaults'] = clean_frame
            st.session_state['v12_frame_spelform'] = payload.get('spelform', st.session_state.get('v12_spelform'))
            _sync_frame_widget_token()
            _sync_frame_widget_state_from_frame(clean_frame, int(payload.get('antal_matcher') or 13))
        hist_df = _history_df_from_records(payload.get('history_records', []))
        filter_vec = payload.get('filter_vec', []) or []
        if not hist_df.empty and filter_vec:
            st.session_state['v12_v_m'] = hist_df
            st.session_state['v12_filter_vec'] = [float(x) for x in filter_vec]
            st.session_state['v12_db_name'] = 'spelfil'
            st.session_state['v12_analysis_ready'] = True
        rs = payload.get('reducer_settings') or {}
        if isinstance(rs, dict):
            for k, v in rs.items():
                st.session_state[k] = v
        _apply_last_result_from_spelfil(payload)
    return ftype


def _fmt_file_stem(prefix='tipset'):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"



def _hit_distribution_rows(label, rows, result_row, antal_matcher):
    """Räknar antal rader med 13/12/11/10-rätt-liknande nivåer.

    För 13 matcher visas exakt 13, 12, 11, 10 och Under 10.
    För andra spelformer visas antal_matcher, -1, -2, -3 och under.
    """
    result_row = normalize_single_row_text(result_row)
    clean_rows = [normalize_single_row_text(r) for r in (rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    total = len(clean_rows)
    if not result_row or len(result_row) != int(antal_matcher):
        return []
    levels = [int(antal_matcher), int(antal_matcher)-1, int(antal_matcher)-2, int(antal_matcher)-3]
    levels = [x for x in levels if x >= 0]
    counts = {lvl: 0 for lvl in levels}
    under = 0
    floor = levels[-1] if levels else 0
    for r in clean_rows:
        hit = sum(1 for a, b in zip(result_row, r) if a == b)
        if hit in counts:
            counts[hit] += 1
        elif hit < floor:
            under += 1
    out = []
    for lvl in levels:
        n = counts.get(lvl, 0)
        out.append({
            'Steg': label,
            'Rätt': f'{lvl} rätt',
            'Antal': int(n),
            'Andel': f'{100*n/max(1,total):.2f}%' if total else '0.00%',
        })
    out.append({
        'Steg': label,
        'Rätt': f'Under {floor} rätt',
        'Antal': int(under),
        'Andel': f'{100*under/max(1,total):.2f}%' if total else '0.00%',
    })
    return out


def build_correction_hit_distribution_df(result_row, base_rows, filtered_rows, reduced_rows, antal_matcher):
    rows = []
    rows += _hit_distribution_rows('Grundram', base_rows, result_row, antal_matcher)
    rows += _hit_distribution_rows('Efter filter', filtered_rows, result_row, antal_matcher)
    if reduced_rows:
        rows += _hit_distribution_rows('Efter TipsetMatrix', reduced_rows, result_row, antal_matcher)
    return pd.DataFrame(rows)



def parse_live_result_row(text, antal_matcher):
    """Läser pågående facit. 1/X/2 = känd match, -/?/_ = ej rättad ännu."""
    raw = str(text or "").strip().upper()
    if not raw:
        return None, ""
    chars = []
    invalid = []
    for ch in raw:
        if ch in {'1', 'X', '2'}:
            chars.append(ch)
        elif ch in {'-', '?', '_', '*'}:
            chars.append('-')
        elif ch in {' ', ',', ';', '/', '|', ':'}:
            continue
        else:
            invalid.append(ch)
            chars.append(ch)
    if invalid:
        bad = ', '.join(sorted(set(invalid)))
        return None, f"Live-raden innehåller ogiltiga tecken: {bad}. Använd bara 1, X, 2 eller - för orättade matcher."
    if len(chars) != int(antal_matcher):
        return None, f"Live-raden måste innehålla exakt {int(antal_matcher)} positioner. Använd 1/X/2 för kända matcher och - för ej rättade. Hittade {len(chars)}."
    if not any(c in {'1', 'X', '2'} for c in chars):
        return None, "Minst en match måste vara rättad/påbörjad."
    return ''.join(chars), ""


def count_live_result_input(text):
    """Räknar live-facitpositioner utan att kräva full giltighet.

    Returnerar antal positioner, antal rättade positioner, normaliserad rad
    och ogiltiga tecken. Separatorer ignoreras. Ogiltiga tecken räknas
    som positioner i räknaren så användaren ser faktisk längd, men de stoppas vid rättning.
    """
    chars = []
    known = 0
    invalid = []
    for ch in str(text or '').strip().upper():
        if ch in {'1', 'X', '2'}:
            chars.append(ch)
            known += 1
        elif ch in {'-', '?', '_', '*'}:
            chars.append('-')
        elif ch in {' ', ',', ';', '/', '|', ':'}:
            continue
        else:
            chars.append(ch)
            invalid.append(ch)
    return len(chars), known, ''.join(chars), sorted(set(invalid))


def count_result_row_input(text):
    """Räknar 1/X/2-tecken i en sluträttningsrad. Separatorer ignoreras."""
    return len(re.findall(r'[1X2]', str(text or '').upper()))


def live_counter_status(pos_count, antal_matcher):
    if pos_count == int(antal_matcher):
        return "OK"
    if pos_count < int(antal_matcher):
        return f"Saknar {int(antal_matcher) - pos_count}"
    return f"För många: +{pos_count - int(antal_matcher)}"


def render_counter_caption(kind, pos_count, known_count=None, antal_matcher=13, normalized='', invalid_chars=None):
    invalid_chars = invalid_chars or []
    status = live_counter_status(pos_count, antal_matcher)
    if known_count is None:
        msg = f"{kind}: {pos_count}/{int(antal_matcher)} tecken · {status}"
    else:
        msg = f"{kind}: {pos_count}/{int(antal_matcher)} positioner · rättade {known_count} · {status}"
    if normalized and pos_count == int(antal_matcher):
        msg += f" · {normalized}"
    if invalid_chars:
        st.error(msg + f" · Ogiltigt: {', '.join(invalid_chars)}", icon="⛔")
    elif pos_count == int(antal_matcher):
        st.success(msg, icon="✅")
    elif pos_count == 0:
        st.caption(msg)
    else:
        st.warning(msg, icon="⚠️")


def live_known_positions(live_row):
    return [i for i, c in enumerate(str(live_row or '')) if c in {'1', 'X', '2'}]


def live_hits_misses(row, live_row):
    row = normalize_single_row_text(row)
    known = live_known_positions(live_row)
    hits = sum(1 for i in known if i < len(row) and row[i] == live_row[i])
    misses = len(known) - hits
    return int(hits), int(misses), int(len(known))


def build_live_pool_summary_df(live_row, base_rows, filtered_rows, reduced_rows, antal_matcher):
    """Sammanfattar hur många rader som fortfarande lever för 13/12/11/10 vid live-rättning."""
    pools = [('Grundram', base_rows), ('Efter filter', filtered_rows)]
    if reduced_rows:
        pools.append(('Efter TipsetMatrix', reduced_rows))
    levels = [int(antal_matcher), int(antal_matcher)-1, int(antal_matcher)-2, int(antal_matcher)-3]
    levels = [lvl for lvl in levels if lvl >= 0]
    out = []
    for label, rows in pools:
        clean = [normalize_single_row_text(r) for r in (rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
        total = len(clean)
        miss_counts = []
        best_hits = 0
        best_max = 0
        for r in clean:
            hits, misses, known = live_hits_misses(r, live_row)
            miss_counts.append(misses)
            best_hits = max(best_hits, hits)
            best_max = max(best_max, int(antal_matcher) - misses)
        for lvl in levels:
            allowed_misses = int(antal_matcher) - lvl
            n = sum(1 for m in miss_counts if m <= allowed_misses)
            label_lvl = f"Lever för {lvl}" if lvl == int(antal_matcher) else f"Lever för {lvl}+"
            out.append({
                'Steg': label,
                'Nivå': label_lvl,
                'Antal rader': int(n),
                'Andel': f"{100*n/max(1,total):.2f}%" if total else '0.00%',
                'Bästa live-träff': f"{best_hits}/{len(live_known_positions(live_row))}",
                'Bästa maxnivå': f"{best_max} rätt" if total else '—',
            })
    return pd.DataFrame(out)


def live_row_cells_html(row, live_row):
    row = normalize_single_row_text(row)
    cells = []
    for i, ch in enumerate(row):
        live = live_row[i] if i < len(str(live_row)) else '-'
        if live not in {'1', 'X', '2'}:
            style = "background:rgba(128,128,128,.14);border:1px solid rgba(128,128,128,.28);color:inherit;"
        elif ch == live:
            style = "background:rgba(40,167,69,.28);border:1px solid rgba(40,167,69,.65);font-weight:800;"
        else:
            style = "background:rgba(220,53,69,.25);border:1px solid rgba(220,53,69,.65);font-weight:800;"
        cells.append(f"<span style='display:inline-block;min-width:1.35rem;text-align:center;margin:.05rem;padding:.12rem .18rem;border-radius:.35rem;{style}'>{html.escape(ch)}</span>")
    return ''.join(cells)


def best_live_rows(rows, live_row, prob_vector, antal_matcher, limit=50):
    """Returnerar bästa live-rader. Sortering: minst missar, högst träff, högst maxnivå, högst log-probability."""
    import heapq
    clean = [normalize_single_row_text(r) for r in (rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    items = []
    for idx, r in enumerate(clean):
        hits, misses, known = live_hits_misses(r, live_row)
        max_possible = int(antal_matcher) - misses
        try:
            score = row_log_probability(r, prob_vector or [])
        except Exception:
            score = 0.0
        # nlargest: högre är bättre. Missar inverteras.
        key = (-misses, hits, max_possible, score, -idx)
        items.append((key, r, hits, misses, max_possible, score))
    best = heapq.nlargest(int(limit), items, key=lambda x: x[0]) if items else []
    out = []
    known_count = len(live_known_positions(live_row))
    for _, r, hits, misses, max_possible, score in best:
        out.append({
            'Rad': r,
            'Träffbild': live_row_cells_html(r, live_row),
            'Live-träff': f"{hits}/{known_count}",
            'Missar': int(misses),
            'Max möjligt': f"{max_possible} rätt",
            'Radscore': round(float(score), 4),
        })
    return out


def render_best_live_rows_table(rows_info, title="Bästa levande rader"):
    if not rows_info:
        st.info("Inga rader att visa.")
        return
    html_rows = []
    for i, item in enumerate(rows_info, 1):
        html_rows.append(
            "<tr>"
            f"<td style='white-space:nowrap;text-align:right;padding:.35rem .5rem;'>{i}</td>"
            f"<td style='white-space:nowrap;padding:.35rem .5rem;font-family:monospace;'>{html.escape(item.get('Rad',''))}</td>"
            f"<td style='white-space:nowrap;padding:.35rem .5rem;'>{item.get('Träffbild','')}</td>"
            f"<td style='white-space:nowrap;text-align:center;padding:.35rem .5rem;'>{html.escape(str(item.get('Live-träff','')))}</td>"
            f"<td style='white-space:nowrap;text-align:center;padding:.35rem .5rem;'>{html.escape(str(item.get('Missar','')))}</td>"
            f"<td style='white-space:nowrap;text-align:center;padding:.35rem .5rem;'>{html.escape(str(item.get('Max möjligt','')))}</td>"
            "</tr>"
        )
    table = (
        f"<div style='overflow-x:auto;'>"
        f"<table style='border-collapse:collapse;width:100%;font-size:.92rem;'>"
        f"<caption style='caption-side:top;text-align:left;font-weight:800;margin:.4rem 0;'>{html.escape(title)}</caption>"
        "<thead><tr>"
        "<th style='text-align:right;padding:.35rem .5rem;border-bottom:1px solid rgba(128,128,128,.35);'>#</th>"
        "<th style='text-align:left;padding:.35rem .5rem;border-bottom:1px solid rgba(128,128,128,.35);'>Rad</th>"
        "<th style='text-align:left;padding:.35rem .5rem;border-bottom:1px solid rgba(128,128,128,.35);'>Träffbild</th>"
        "<th style='text-align:center;padding:.35rem .5rem;border-bottom:1px solid rgba(128,128,128,.35);'>Live</th>"
        "<th style='text-align:center;padding:.35rem .5rem;border-bottom:1px solid rgba(128,128,128,.35);'>Missar</th>"
        "<th style='text-align:center;padding:.35rem .5rem;border-bottom:1px solid rgba(128,128,128,.35);'>Max</th>"
        "</tr></thead><tbody>"
        + ''.join(html_rows) +
        "</tbody></table></div>"
    )
    st.markdown(table, unsafe_allow_html=True)


def _format_filter_value(v, decimals=0):
    try:
        if int(decimals) == 0:
            return str(int(round(float(v))))
        return f"{float(v):.{int(decimals)}f}"
    except Exception:
        return str(v)


def build_filter_correction_df(result_row, specs, settings, group_reqs):
    """Visar vilka aktiva filter facitraden träffar/missar."""
    result_row = normalize_single_row_text(result_row)
    rows = []
    for spec in specs:
        setting = settings.get(spec.get('key'), {})
        mode = setting.get('mode', 'Av')
        if mode == 'Av':
            continue
        interval = setting.get('interval', spec.get('default_interval'))
        try:
            val = _spec_value(result_row, spec)
            ok = bool(_spec_pass(result_row, spec, interval))
            val_txt = _format_filter_value(val, spec.get('decimals', 0))
        except Exception as e:
            ok = False
            val_txt = 'kunde inte räknas'
        hp, ht, pct = _hist_pass_count(spec.get('hist_values', []), interval)
        rows.append({
            'Status': '✅ Träff' if ok else '❌ Miss',
            'Läge': mode,
            'Kategori': spec.get('category', ''),
            'Filter': spec.get('name', ''),
            'Facitvärde': val_txt,
            'Intervall/regler': _display_interval(interval, spec.get('decimals', 0)),
            'Historisk träff med intervallet': f'{hp}/{ht}',
        })
    return pd.DataFrame(rows)


def build_group_correction_df(result_row, specs, settings, group_reqs):
    """Summerar gruppfilter: hur många filter i gruppen facitraden klarar mot gruppkravet."""
    result_row = normalize_single_row_text(result_row)
    out = []
    for gi in range(1, 7):
        gname = f'Grupp {gi}'
        group_specs = [s for s in specs if settings.get(s.get('key'), {}).get('mode') == gname]
        if not group_specs:
            continue
        min_req, max_req = _group_req_bounds(group_reqs, gname, len(group_specs))
        hits = 0
        miss_names = []
        for spec in group_specs:
            interval = settings.get(spec.get('key'), {}).get('interval', spec.get('default_interval'))
            try:
                ok = bool(_spec_pass(result_row, spec, interval))
            except Exception:
                ok = False
            if ok:
                hits += 1
            else:
                miss_names.append(spec.get('name', ''))
        inactive = (min_req <= 0 and max_req >= len(group_specs))
        ok_group = True if inactive else (min_req <= hits <= max_req)
        out.append({
            'Grupp': gname,
            'Krav': _group_req_label(group_reqs, gname, len(group_specs)),
            'Facit träffar': f'{hits}/{len(group_specs)}',
            'Status': '✅ Träff' if ok_group else '❌ Miss',
            'Missade filter': ', '.join(miss_names[:8]) + (' …' if len(miss_names) > 8 else ''),
        })
    return pd.DataFrame(out)


def _forced_quality_rows(specs, settings, rows=None, min_hit_pct=90.0, min_reduction_pct=5.0):
    """Returnerar kvalitetsrad per tvingat filter.

    Detta ändrar inte användarens filterval. Det ger bara tydlig kontroll:
    - individuell historisk träff på de valda liknande omgångarna
    - faktisk ensamseffekt på sparad grundram
    - om filtret ligger under rekommenderad spärr
    """
    out = []
    for spec in specs:
        setting = settings.get(spec['key'], {})
        if setting.get('mode') != 'Tvingat':
            continue
        interval = setting.get('interval', spec['default_interval'])
        hp, ht, pct = _hist_pass_count(spec.get('hist_values', []), interval)
        kept = None
        red_pct = None
        if rows is not None and len(rows) <= 30000:
            try:
                kept = sum(1 for r in rows if _spec_pass(r, spec, interval))
                red_pct = 100.0 - 100.0 * kept / max(1, len(rows))
            except Exception:
                kept = None
                red_pct = None
        issues = []
        if pct < float(min_hit_pct):
            issues.append(f"träff {pct:.1f}% < {float(min_hit_pct):.0f}%")
        if red_pct is not None and red_pct < float(min_reduction_pct):
            issues.append(f"reducering {red_pct:.1f}% < {float(min_reduction_pct):.1f}%")
        out.append({
            'Filter': spec['name'],
            'Kategori': spec.get('category', ''),
            'Intervall/krav': _display_interval(interval, spec.get('decimals', 0)),
            'Hist träff': f"{hp}/{ht} ({pct:.1f}%)",
            'Reducerar ensamt': ('—' if red_pct is None else f"{red_pct:.1f}%"),
            'Kvalitet': 'OK' if not issues else '⚠️ ' + '; '.join(issues),
            '_pct': pct,
            '_red_pct': (-1 if red_pct is None else red_pct),
            '_ok': not issues,
        })
    return out

def _build_filter_summary_df(specs, settings, group_reqs, rows=None):
    data = []
    for spec in specs:
        mode = settings.get(spec['key'], {}).get('mode', 'Av')
        interval = settings.get(spec['key'], {}).get('interval', spec['default_interval'])
        hp, ht, pct = _hist_pass_count(spec['hist_values'], interval)
        exact_keep_txt = '—'
        red_txt = '—'
        if rows is not None and len(rows) <= 30000:
            try:
                kept = sum(1 for r in rows if _spec_pass(r, spec, interval))
                exact_keep_txt = f"{kept}/{len(rows)}"
                red_txt = f"{(100 - 100*kept/max(1,len(rows))):.1f}%"
            except Exception:
                pass
        data.append({
            'Filter': spec['name'],
            'Kategori': spec['category'],
            'Läge': mode,
            'Intervall/krav': _display_interval(interval, spec['decimals']),
            'Hist träff': f"{hp}/{ht} ({pct:.1f}%)",
            'Exakt på grundram': exact_keep_txt,
            'Reducerar ensamt': red_txt,
        })
    return pd.DataFrame(data)



def _render_info_cards(cards):
    """Visar filterinfo som egna HTML-kort så långa intervall inte kapas med ellips."""
    parts = ["<div class='v12-info-grid'>"]
    for label, value, sub in cards:
        parts.append(
            "<div class='v12-info-card'>"
            f"<div class='v12-info-label'>{html.escape(str(label))}</div>"
            f"<div class='v12-info-value'>{html.escape(str(value))}</div>"
            f"<div class='v12-info-sub'>{html.escape(str(sub))}</div>"
            "</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)

def _render_favorite_shock_diagnostics(spec, interval, frame, filter_vec, antal_matcher):
    """Kort grundramsdiagnos för favorit-/skrällfilter så de inte blir dolda spikar."""
    meta = spec.get('meta') or {}
    dtype = meta.get('diag_type')
    if dtype not in {'favorite_threshold', 'shock_threshold'}:
        return
    try:
        low = float(interval[0]); high = float(interval[1])
    except Exception:
        return

    threshold = float(meta.get('threshold', 0))
    if dtype == 'favorite_threshold':
        details = get_favorite_threshold_details(filter_vec, threshold)
        covered = _details_covered_by_frame(details, frame)
        txt = f"Kupongen har {len(details)} favoriter ≥{threshold:.0f} %. Grundramen täcker {len(covered)} av dem."
        if details:
            txt += " " + _format_match_sign_details(details)
        st.caption(txt)
        if low > len(covered):
            st.warning(f"Det valda intervallet kräver minst {int(low)} träffar, men grundramen kan maximalt ge {len(covered)}. Filtret är omöjligt med nuvarande grundram.")
        elif len(covered) > 0 and low >= len(covered):
            st.warning(f"Intervallet kräver att alla {len(covered)} täckta favorittecken sitter. Det fungerar, men blir ett hårt favoritlås i grundramen.")
        return

    if dtype == 'shock_threshold':
        details = get_shock_sign_details(filter_vec, threshold)
        covered = _details_covered_by_frame(details, frame)
        txt = f"Kupongen har {len(details)} tecken under {threshold:.0f} %. Grundramen innehåller {len(covered)} av dem."
        if covered:
            txt += " " + _format_match_sign_details(covered)
        st.caption(txt)
        if low > len(covered):
            st.warning(f"Det valda intervallet kräver minst {int(low)} skrälltecken under {threshold:.0f} %, men grundramen kan maximalt ge {len(covered)}. Filtret är omöjligt med nuvarande grundram.")
        elif len(covered) == 1 and low >= 1:
            only = _format_match_sign_details(covered, max_items=1)
            st.warning(f"Detta blir i praktiken en dold spik: för att få minst {int(low)} vinnare under {threshold:.0f} % måste {only} vara med på varje rad som passerar filtret.")
        elif len(covered) > 1 and low >= len(covered):
            st.warning(f"Intervallet kräver att alla {len(covered)} skrälltecken under {threshold:.0f} % i grundramen sitter. Det är ett mycket hårt grundramslås.")


def _render_inline_filter_info(spec, interval, frame_rows, frame, antal_matcher):
    """Renderar statistik för ett filter.

    Viktigt: historisk träff i rutan gäller alltid 30 liknande omgångar.
    Vi visar både rekommenderat intervall/träff och nuvarande slider/träff så att
    det blir tydligt om användaren ändrat intervallet.
    """
    rec_interval = spec['default_interval']
    rhp, rht, rpct = _hist_pass_count(spec['hist_values'], rec_interval)
    hp, ht, pct = _hist_pass_count(spec['hist_values'], interval)

    row_values = []
    pass_rows = []
    sampled_info = False
    rows_for_info = []
    if frame_rows is not None:
        if len(frame_rows) > 3000:
            rows_for_info, sampled_info = _sample_rows_for_macro(frame_rows, max_items=3000)
        else:
            rows_for_info = frame_rows
        for r in rows_for_info:
            try:
                val = _spec_value(r, spec)
                row_values.append(val)
                if in_range(val, interval):
                    pass_rows.append(r)
            except Exception:
                pass

    st.markdown(f"**ℹ️ {spec['name']} — statistik**")
    st.caption("Historisk träff räknas på de 30 mest liknande omgångarna efter valt utdelningskrav. Rek. träff gäller rekommenderat intervall; nuvarande träff gäller dina sliders.")
    if spec.get('help'):
        st.caption(spec.get('help'))

    rec_txt = _display_interval(rec_interval, spec['decimals'])
    cur_txt = _display_interval(interval, spec['decimals'])
    if row_values:
        red_pct = 100 - 100 * len(pass_rows) / max(1, len(rows_for_info))
        if sampled_info:
            est_pass = int(round(len(frame_rows) * len(pass_rows) / max(1, len(rows_for_info))))
            frame_txt = f"≈ {len(frame_rows):,} → {est_pass:,}".replace(',', ' ')
            frame_sub = f"Uppskattar reducering {red_pct:.1f}% på 3 000-raders urval"
        else:
            frame_txt = f"{len(frame_rows):,} → {len(pass_rows):,}".replace(',', ' ')
            frame_sub = f"Reducerar {red_pct:.1f}% av grundramen"
    else:
        frame_txt = "—"
        frame_sub = "Spara grundram för exakt effekt"

    _render_info_cards([
        ("Rek. intervall", rec_txt, "Rekommenderat från historiken"),
        ("Rek. träff", f"{rhp}/{rht}", f"{rpct:.1f}% av 30 liknande"),
        ("Nu valt", cur_txt, "Dina aktuella sliders"),
        ("Nuvarande träff", f"{hp}/{ht}", f"{pct:.1f}% av 30 liknande"),
        ("Grundram → filter", frame_txt, frame_sub),
    ])

    meta = spec.get('meta') or {}
    if meta.get('diag_type') == 'super_macro_rational':
        rec_rows = meta.get('recommendation_table') or []
        if rec_rows:
            st.markdown("**Super-Makro nivåtest**")
            st.caption("Tabellen visar varför rekommenderat intervall valdes. Den testar alla min-nivåer mot historiken och grundramen. Balanspoäng premierar rationell mjukzon så filtret inte automatiskt blir ett överhårt normalitetslås.")
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
        included = meta.get('included_filters') or []
        if included:
            with st.expander('Ingår i detta Super-Makro', expanded=False):
                st.write(', '.join([str(x) for x in included]))

    freq_df = _make_freq_df(spec['hist_values'], spec['decimals'])
    left, right = st.columns([1.0, 1.0])
    with left:
        st.markdown("**Frekvenstabell, 30 liknande omgångar**")
        if not freq_df.empty:
            table_df = freq_df.drop(columns=['_sort'], errors='ignore')
            st.dataframe(table_df, use_container_width=True, hide_index=True, height=230)
            st.caption("Sorterad numeriskt från lägsta till högsta värde. Andel avser de 30 liknande omgångarna.")
        else:
            st.info("Ingen frekvensdata hittades för detta filter.")
    with right:
        st.markdown("**Fördelning**")
        if not freq_df.empty:
            plot_df = freq_df.sort_values('_sort', ascending=True).copy()
            chart_df = plot_df.rename(columns={'Intervall/Värde': 'Värde / intervall'})[['Värde / intervall', 'Antal']]
            try:
                st.bar_chart(chart_df, x='Värde / intervall', y='Antal', use_container_width=True)
            except Exception:
                st.bar_chart(chart_df.set_index('Värde / intervall')['Antal'])

    if pass_rows:
        with st.expander("Teckenfördelning efter detta filter", expanded=False):
            st.dataframe(build_sign_distribution_df(pass_rows, frame, antal_matcher), use_container_width=True, hide_index=True)


def _rows_from_mask(rows, mask):
    """Returnerar rader där mask är True. Separat helper för läsbarhet i paketoptimeringen."""
    try:
        return [r for r, ok in zip(rows, mask) if bool(ok)]
    except Exception:
        return []


def _fmt_elapsed(seconds):
    """Formaterar sekunder som mm:ss eller h:mm:ss för progressraden."""
    try:
        seconds = int(max(0, float(seconds)))
    except Exception:
        seconds = 0
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _frame_row_matrix(frame_rows, antal_matcher):
    """Preberäknar grundrader som teckenmatris för snabb teckenskyddskontroll."""
    try:
        clean = [normalize_single_row_text(r) for r in (frame_rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
        if not clean:
            return None
        return np.array([list(r) for r in clean], dtype='<U1')
    except Exception:
        return None


def _selected_signs_missing_from_mask(row_matrix, mask, frame, antal_matcher):
    """Snabb variant av selected_signs_missing när vi redan har en boolean-mask."""
    if row_matrix is None or not frame:
        return []
    try:
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0 or not mask.any():
            # Om inga rader finns kvar saknas alla manuellt valda tecken.
            return [(mi + 1, sign) for mi in range(int(antal_matcher)) for sign in (normalize_signs(frame[mi]) if mi < len(frame) else [])]
        sub = row_matrix[mask]
        missing = []
        for mi in range(int(antal_matcher)):
            selected = normalize_signs(frame[mi]) if mi < len(frame) else []
            if not selected:
                continue
            col = sub[:, mi]
            for sign in selected:
                if not np.any(col == sign):
                    missing.append((mi + 1, sign))
        return missing
    except Exception:
        return []


def _mask_keeps_teckenskydd(row_matrix, mask, frame, antal_matcher):
    """True om maskad radmassa fortfarande innehåller alla manuellt valda tecken."""
    miss = _selected_signs_missing_from_mask(row_matrix, mask, frame, antal_matcher)
    return not bool(miss)


def _prune_candidate_ladder(spec_candidates, max_levels=18):
    """Behåller en nivåtrappa per filter i stället för bara säkrast/hårdast.

    För varje möjlig historisk träffnivå behålls den variant som reducerar mest.
    Det ger en riktig trade-off-kurva: 30/30, 29/30, 28/30 ... ner till
    mer aggressiva nivåer, utan att paketmotorn drunknar i dubbletter.
    """
    by_hit = {}
    for c in spec_candidates or []:
        h = int(c.get('hist_hit', 0))
        cur = by_hit.get(h)
        score = (float(c.get('red_pct', 0.0)), -int(c.get('frame_keep', 10**12)), float(c.get('hist_pct', 0.0)))
        if cur is None or score > cur[0]:
            by_hit[h] = (score, c)
    ladder = [v[1] for v in by_hit.values()]
    ladder.sort(key=lambda c: (-int(c.get('hist_hit', 0)), int(c.get('frame_keep', 10**12)), -float(c.get('red_pct', 0.0))))
    if len(ladder) <= int(max_levels):
        return ladder
    # Bevara säkra änden, aggressiva änden och jämnt fördelade mellanlägen.
    keep_idx = {0, 1, len(ladder)-2, len(ladder)-1}
    slots = max(0, int(max_levels) - len(keep_idx))
    if slots > 0 and len(ladder) > 4:
        for idx in np.linspace(2, len(ladder)-3, slots):
            keep_idx.add(int(round(idx)))
    return [ladder[i] for i in sorted(i for i in keep_idx if 0 <= i < len(ladder))]


def _candidate_ladder_text(cand_list, htot, max_items=9):
    """Kort text med filtertrappan för kandidatanalysen."""
    if not cand_list:
        return '-'
    ordered = sorted(cand_list, key=lambda c: (-int(c.get('hist_hit', 0)), -float(c.get('red_pct', 0.0))))
    parts = []
    for c in ordered[:int(max_items)]:
        parts.append(f"{c.get('interval_txt','-')} {int(c.get('hist_hit',0))}/{int(htot)} {float(c.get('red_pct',0.0)):.1f}%")
    if len(ordered) > int(max_items):
        parts.append(f"+{len(ordered)-int(max_items)} nivåer")
    return ' | '.join(parts)


def _balanced_candidate_for_audit(cand_list, htot, min_hit_floor=None):
    """Väljer ett illustrativt mellanläge: hög reducering men med rimlig historisk säkerhet."""
    if not cand_list:
        return None
    if min_hit_floor is None:
        min_hit_floor = max(1, int(round(float(htot) * 0.70)))
    viable = [c for c in cand_list if int(c.get('hist_hit', 0)) >= int(min_hit_floor)] or list(cand_list)
    def score(c):
        hp = int(c.get('hist_hit', 0)) / max(1.0, float(htot))
        return (float(c.get('red_pct', 0.0)) * (hp ** 1.8), hp, -int(c.get('frame_keep', 10**12)))
    return max(viable, key=score)



def _frame_capacity_pressure(spec, interval, frame, frame_rows, antal_matcher):
    """Bedömer om ett paketfilter är för hårt relativt den manuella grundramen.

    Problemet vi skyddar mot: historiken kan gilla t.ex. Tecken 1 = 5–8,
    men om spelarens grundram bara har 8 möjliga ettor totalt blir filtret
    väldigt toppstyrt. Då kan paketmotorn välja något som är historiskt starkt
    men opraktiskt/obalanserat i just veckans manuella ram.

    Returnerar (blockera: bool, varningstext: str).
    Detta påverkar bara rekommenderade paket; manuella filterval är fortfarande fria.
    """
    try:
        name = str(spec.get('name', ''))
        low = float(interval[0]); high = float(interval[1])
        if not frame:
            return False, ''

        # Direkt kapacitetskontroll för teckenbalans 1/X/2.
        sign = None
        if name == 'Tecken 1': sign = '1'
        elif name == 'Tecken X': sign = 'X'
        elif name == 'Tecken 2': sign = '2'
        if sign is not None:
            norm_frame = [normalize_signs(x) for x in frame]
            min_possible = sum(1 for signs in norm_frame if len(signs) == 1 and sign in signs)
            max_possible = sum(1 for signs in norm_frame if sign in signs)
            span = max(1, max_possible - min_possible)
            lower_pressure = max(0.0, (low - min_possible) / span)
            upper_pressure = max(0.0, (max_possible - high) / span)
            # Vid få möjliga tecken är intervall nära ytterkant extra farliga.
            if max_possible <= 9 and lower_pressure >= 0.38:
                return True, f"Grundramsanpassning: {name} {int(low)}–{int(high)} kräver för många {sign} relativt din ram ({min_possible}–{max_possible} möjliga)."
            if max_possible <= 9 and upper_pressure >= 0.45:
                return True, f"Grundramsanpassning: {name} {int(low)}–{int(high)} tillåter för få {sign} relativt din ram ({min_possible}–{max_possible} möjliga)."
            return False, ''

        # Mjukare generell kontroll för diskreta struktur-/kapacitetsfilter.
        # Används bara om filtret ligger väldigt nära övre/nedre kanten av de
        # faktiska värden som kan uppstå i den sparade grundramen.
        category = str(spec.get('category', ''))
        if category not in {'Struktur', 'FAT', 'FAT-sekvenser', 'Favorit & skräll'}:
            return False, ''
        dec = int(spec.get('decimals', 0))
        if dec != 0:
            return False, ''
        if not frame_rows:
            return False, ''
        vals = []
        getter = spec.get('getter')
        for r in frame_rows[:30000]:
            try:
                vals.append(float(getter(r)))
            except Exception:
                pass
        if len(vals) < 10:
            return False, ''
        vmin = min(vals); vmax = max(vals)
        span = max(1.0, vmax - vmin)
        lower_pressure = max(0.0, (low - vmin) / span)
        upper_pressure = max(0.0, (vmax - high) / span)

        # FAT-sekvenser behöver samma typ av ramkoll som teckenbalans.
        # Om spelarens ram bara ger få möjliga sekvensträffar ska paketmotorn
        # inte välja ett intervall som pressar mot ytterkanten, t.ex. 4–5 av 5,
        # bara för att det historiskt ser starkt ut.
        if category == 'FAT-sekvenser':
            # FAT-sekvenser kan bli missvisande när spelarens manuella ram redan
            # låser in ett visst antal sekvensträffar via spikar/halvor.
            # Exempel: om grundramens möjliga spann redan är 3–5 och filtret är 3–4
            # reducerar det hårt genom att kapa max, men det kräver egentligen inte
            # någon "egen" sekvensträff utöver ramen. Då ska paketmotorn inte
            # behandla det som ett starkt rekommenderat filter.
            small_span = span <= 3.0
            locks_to_frame_min = (vmin >= 1 and low <= vmin + 0.01 and high < vmax - 0.01)
            near_upper_cap = upper_pressure >= (0.45 if small_span else 0.65)
            near_lower_req = lower_pressure >= (0.45 if small_span else 0.65)
            if locks_to_frame_min:
                return True, f"Grundramsanpassning: {name} {int(low)}–{int(high)} är för beroende av din grundram. Ramen har redan minst {vmin:.0f} sekvensträffar och filtret kapar bara spannet {vmin:.0f}–{vmax:.0f}."
            if near_lower_req:
                return True, f"Grundramsanpassning: {name} {int(low)}–{int(high)} kräver hög FAT-sekvensträff relativt din ram ({vmin:.0f}–{vmax:.0f} möjliga i grundramen)."
            if near_upper_cap:
                return True, f"Grundramsanpassning: {name} {int(low)}–{int(high)} kapar FAT-sekvenser för hårt relativt din ram ({vmin:.0f}–{vmax:.0f} möjliga i grundramen)."
            return False, ''

        # Blockera bara extrema ytterkantsfilter. Vanliga intervall får passera.
        if lower_pressure >= 0.72:
            return True, f"Grundramsanpassning: {name} ligger högt i grundramens möjliga värden ({vmin:.0f}–{vmax:.0f})."
        if upper_pressure >= 0.72:
            return True, f"Grundramsanpassning: {name} ligger lågt i grundramens möjliga värden ({vmin:.0f}–{vmax:.0f})."
    except Exception:
        return False, ''
    return False, ''

def _candidate_intervals_for_spec(spec, coverages=None):
    """Skapar flera intervallvarianter för ett filter utan att skapa dubbletter i UI.

    Detta används bara av rekommenderade paket. Den manuella filtercentralens
    sliderstart styrs fortfarande av "Minsta historiska träff på filterintervall".
    Paketmotorn ska däremot själv få testa flera träffnivåer per filter.
    """
    if coverages is None:
        coverages = [100, 97, 95, 93, 90, 87, 85, 80, 75, 70, 65, 60, 55, 50]
    vals = [v for v in spec.get('hist_values', []) if not pd.isna(v)]
    out = []
    seen = set()
    for cov in coverages:
        try:
            interval = get_best_interval(vals, float(cov))
            # Normalisera nyckeln så samma intervall inte testas många gånger.
            dec = int(spec.get('decimals', 0))
            key = (round(float(interval[0]), max(dec, 0)), round(float(interval[1]), max(dec, 0)))
            if key in seen:
                continue
            seen.add(key)
            hp, ht, pct = _hist_pass_count(vals, interval)
            out.append({
                'coverage': float(cov),
                'interval': interval,
                'hist_pass': int(hp),
                'hist_total': int(ht),
                'hist_pct': float(pct),
                'interval_txt': _display_interval(interval, spec.get('decimals', 0)),
            })
        except Exception:
            continue
    # Bredaste/säkraste först om två intervall får samma träff; smalare intervall kan ändå vinna på radreducering senare.
    out.sort(key=lambda x: (x['hist_pass'], -abs(float(x['interval'][1])-float(x['interval'][0]))), reverse=True)
    return out


def _pareto_reduce_packages(packages):
    """Returnerar bara paket som inte domineras av ett annat paket.

    Ett paket domineras om ett annat paket har minst lika hög samlad historisk
    träff och minst lika låg kvarvarande radmassa, med förbättring på minst en
    av dimensionerna.
    """
    clean = []
    for p in packages:
        if p.get('frame_after', 10**18) >= p.get('frame_start', 0):
            # Visa inte tomma paket som inte reducerar alls, om det finns andra paket.
            continue
        dominated = False
        for q in packages:
            if q is p:
                continue
            if q.get('frame_after', 10**18) >= q.get('frame_start', 0):
                continue
            if (q['hist_hit'] >= p['hist_hit'] and q['frame_after'] <= p['frame_after'] and
                (q['hist_hit'] > p['hist_hit'] or q['frame_after'] < p['frame_after'])):
                dominated = True
                break
        if not dominated:
            clean.append(p)
    # Om pareto råkar bli tomt, fall tillbaka på de bästa vi hittade.
    if not clean:
        clean = list(packages)
    # Sortera från högst träffbild till mest aggressiva.
    clean.sort(key=lambda x: (-x['hist_hit'], x['frame_after'], -x['reduction_pct']))
    # Ta bort exakta dubbletter i resultatnivå.
    dedup = []
    seen = set()
    for p in clean:
        k = (p.get('hist_hit'), p.get('frame_after'))
        if k in seen:
            continue
        seen.add(k)
        dedup.append(p)
    return dedup


def _is_hard_group_package(p):
    """Sant om paketet innehåller hårda gruppvillkor.

    Grupppaket ska visas som egen väg. Annars kan de lätt försvinna i Pareto-
    reduceringen eftersom ett rent tvingat paket ofta reducerar fler rader på
    samma historiska träffnivå. Det betyder inte att grupppaketet är ointressant;
    det är en annan riskprofil.
    """
    try:
        return bool(p.get('groups')) or ('grupp' in str(p.get('package_type', '')).lower())
    except Exception:
        return False


def _dedupe_package_list(packages):
    """Tar bort identiska paket men behåller olika pakettyper/riskprofiler."""
    out = []
    seen = set()
    for p in list(packages or []):
        try:
            sig = _package_signature(p)
        except Exception:
            sig = (p.get('package_type'), p.get('hist_hit'), p.get('frame_after'), p.get('num_filters'))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(p)
    return out


def _append_group_representatives(pareto_packages, all_packages, max_group_packages=5):
    """Lägg tillbaka representativa grupppaket efter Pareto.

    Normal Pareto-logik är matematisk korrekt för två axlar: högre träff och
    färre rader. Men hårda grupper har en annan funktion: de kan ge högre
    robusthet och fler filter utan att alla måste sitta samtidigt. Därför ska
    minst några grupppaket överleva som egen jämförelseväg även om ett tvingat
    paket råkar dominera dem i ren radmängd.
    """
    pareto_packages = list(pareto_packages or [])
    group_candidates = [p for p in list(all_packages or []) if _is_hard_group_package(p) and int(p.get('frame_after', 10**18)) < int(p.get('frame_start', 0))]
    if not group_candidates:
        return pareto_packages

    # Bästa grupppaket per samlad träffnivå. Detta ger både säkra och mer
    # reducerande gruppalternativ, i stället för att bara visa ett enda.
    best_by_hit = {}
    for p in group_candidates:
        h = int(p.get('hist_hit', 0))
        cur = best_by_hit.get(h)
        if cur is None:
            best_by_hit[h] = p
            continue
        score_p = (float(p.get('reduction_pct', 0.0)), len(p.get('groups', []) or []), -int(p.get('frame_after', 10**12)), int(p.get('num_filters', 0)))
        score_c = (float(cur.get('reduction_pct', 0.0)), len(cur.get('groups', []) or []), -int(cur.get('frame_after', 10**12)), int(cur.get('num_filters', 0)))
        if score_p > score_c:
            best_by_hit[h] = p

    reps = list(best_by_hit.values())
    reps.sort(key=lambda p: (-int(p.get('hist_hit', 0)), int(p.get('frame_after', 10**12)), -float(p.get('reduction_pct', 0.0))))
    # Lägg även till bästa rena reducering bland grupppaketen om den inte redan
    # finns bland träffnivå-representanterna.
    strongest = sorted(group_candidates, key=lambda p: (float(p.get('reduction_pct', 0.0)), int(p.get('hist_hit', 0)), -int(p.get('frame_after', 10**12))), reverse=True)[:2]
    merged = _dedupe_package_list(pareto_packages + reps[:int(max_group_packages)] + strongest)
    merged.sort(key=lambda x: (-int(x.get('hist_hit', 0)), int(x.get('frame_after', 10**12)), 0 if _is_hard_group_package(x) else 1, -float(x.get('reduction_pct', 0.0))))
    return merged



def _combine_forced_candidate_masks(chosen, initial_hist_mask=None):
    """Kombinerar tvingade filter till samlad historik-/grundramsmask."""
    chosen = list(chosen or [])
    if not chosen:
        return None, None
    try:
        htot = len(chosen[0]['hist_mask'])
        ftot = len(chosen[0]['frame_mask'])
        if initial_hist_mask is not None:
            hist = np.array(initial_hist_mask, dtype=bool)
            if len(hist) != htot:
                hist = np.ones(htot, dtype=bool)
        else:
            hist = np.ones(htot, dtype=bool)
        frame = np.ones(ftot, dtype=bool)
        for c in chosen:
            hist = hist & np.asarray(c['hist_mask'], dtype=bool)
            frame = frame & np.asarray(c['frame_mask'], dtype=bool)
        return hist, frame
    except Exception:
        return None, None


def _rebuild_forced_package_steps(chosen, htot, ftot, initial_hist_mask=None):
    """Bygger om steg-tabellen efter att nivåer bytts i eftertrimningen."""
    if initial_hist_mask is not None:
        cur_hist = np.array(initial_hist_mask, dtype=bool)
        if len(cur_hist) != int(htot):
            cur_hist = np.ones(int(htot), dtype=bool)
    else:
        cur_hist = np.ones(int(htot), dtype=bool)
    cur_frame = np.ones(int(ftot), dtype=bool)
    steps = []
    for cand in list(chosen or []):
        prev_count = int(cur_frame.sum())
        cur_hist = cur_hist & np.asarray(cand['hist_mask'], dtype=bool)
        cur_frame = cur_frame & np.asarray(cand['frame_mask'], dtype=bool)
        new_count = int(cur_frame.sum())
        step_red_pct = 100.0 * (prev_count - new_count) / max(1, prev_count)
        test_level = f"{float(cand.get('coverage', 0.0)):.0f}%"
        if cand.get('required_in_package'):
            test_level += " · måste ingå"
        steps.append({
            'Filter': cand.get('name', ''),
            'Kategori': cand.get('category', ''),
            'Intervall': cand.get('interval_txt', '-'),
            'Intervallträff': f"{int(cand.get('hist_hit', 0))}/{int(htot)}",
            'Testnivå': test_level,
            'Stegreducering': f"{step_red_pct:.1f}%",
            'Efter filter': int(new_count),
            'Samlad träff efter steg': f"{int(cur_hist.sum())}/{int(htot)}",
        })
    return steps, int(cur_hist.sum()), int(cur_frame.sum()), cur_hist, cur_frame


def _post_trim_forced_package_levels(chosen, candidates_by_key, frame, antal_matcher, row_matrix=None, max_passes=2, initial_hist_mask=None):
    """Snävar åt valda filter efter paketbygget utan att sänka samlad träff.

    Exempel: om paketet har FAT Summa 19–30 och hela paketet ändå har 23/30,
    testas hårdare FAT Summa-nivåer. Om 19–26 fortfarande ger 23/30 men färre
    rader, byts nivån automatiskt. Detta görs bara för tvingade paket där varje
    filter är ett AND-villkor; grupppaket hanteras separat av gruppkravet.
    """
    chosen = [dict(c) for c in list(chosen or [])]
    if not chosen:
        return chosen, [], None, None
    base_hist, base_frame = _combine_forced_candidate_masks(chosen, initial_hist_mask=initial_hist_mask)
    if base_hist is None or base_frame is None:
        return chosen, [], None, None
    preserve_hit = int(base_hist.sum())
    htot = len(base_hist)
    notes = []

    for _pass in range(int(max_passes)):
        changed = False
        for idx, cur in enumerate(list(chosen)):
            cur_key = cur.get('key')
            if cur_key is None:
                continue
            cur_hist_mask, cur_frame_mask = _combine_forced_candidate_masks(chosen, initial_hist_mask=initial_hist_mask)
            if cur_hist_mask is None or cur_frame_mask is None:
                continue
            cur_rows = int(cur_frame_mask.sum())
            best_alt = None
            best_alt_frame = None
            best_score = None
            for alt in candidates_by_key.get(cur_key, []) or []:
                # Samma nivå ger ingen nytta.
                if str(alt.get('interval_txt')) == str(cur.get('interval_txt')):
                    continue
                trial = list(chosen)
                alt2 = dict(alt)
                # Bevara metadata som behövs i UI/applicering.
                if cur.get('required_in_package'):
                    alt2['required_in_package'] = True
                trial[idx] = alt2
                new_hist, new_frame = _combine_forced_candidate_masks(trial, initial_hist_mask=initial_hist_mask)
                if new_hist is None or new_frame is None:
                    continue
                new_hit = int(new_hist.sum())
                if new_hit < preserve_hit:
                    continue
                new_rows = int(new_frame.sum())
                if new_rows >= cur_rows:
                    continue
                if row_matrix is not None:
                    if not _mask_keeps_teckenskydd(row_matrix, new_frame, frame, antal_matcher):
                        continue
                # Välj lägst slutradantal. Vid samma radantal: högre individuell
                # träff först så eftertrimmen inte väljer onödigt riskig nivå.
                score = (new_rows, -int(alt.get('hist_hit', 0)), -float(alt.get('hist_pct', 0.0)), -float(alt.get('red_pct', 0.0)))
                if best_alt is None or score < best_score:
                    best_alt = alt2
                    best_alt_frame = new_frame
                    best_score = score
            if best_alt is not None:
                before = str(cur.get('interval_txt', '-'))
                after = str(best_alt.get('interval_txt', '-'))
                before_hit = int(cur.get('hist_hit', 0))
                after_hit = int(best_alt.get('hist_hit', 0))
                before_rows = int(cur_rows)
                after_rows = int(best_score[0]) if best_score else int(best_alt_frame.sum())
                notes.append({
                    'Filter': cur.get('name', ''),
                    'Före': before,
                    'Efter': after,
                    'Filterträff före': f"{before_hit}/{int(htot)}",
                    'Filterträff efter': f"{after_hit}/{int(htot)}",
                    'Paket-rader före': int(before_rows),
                    'Paket-rader efter': int(after_rows),
                    'Samlad träff bevarad': f"{preserve_hit}/{int(htot)}",
                })
                chosen[idx] = best_alt
                changed = True
        if not changed:
            break
    final_hist, final_frame = _combine_forced_candidate_masks(chosen, initial_hist_mask=initial_hist_mask)
    return chosen, notes, final_hist, final_frame




def _pick_best_variant_per_filter_for_group(candidates, category_filter, target, htot, min_hist_floor=None, max_items=10, required_keys=None):
    """Väljer en rimlig kandidatvariant per filter för en hård grupp.

    Grupppaket ska inte bara ta den aggressivaste varianten. Därför balanseras
    historisk träff och reducering. För gruppfilter tillåts lägre individuell
    träff än målträffen eftersom gruppkravet (t.ex. 5 av 7) skyddar helheten.
    """
    required_keys = set(required_keys or [])
    if min_hist_floor is None:
        min_hist_floor = max(1, min(htot, int(target) - 6))
    by_key = {}
    for c in candidates:
        if not category_filter(c):
            continue
        is_required = c.get('key') in required_keys
        if int(c.get('hist_hit', 0)) < int(min_hist_floor) and not is_required:
            continue
        # Undvik extremt svaga filter i gruppkärnan. Obligatoriska filter får dock
        # följa med om de reducerar mätbart och paketet som helhet håller träffbilden.
        if float(c.get('red_pct', 0.0)) <= 0.05 and not is_required:
            continue
        cur = by_key.get(c.get('key'))
        # Gruppscore: historisk säkerhet först, sedan reducering. Det hindrar att
        # t.ex. AI-Rank 3/30 väljs bara för att den reducerar brutalt.
        score = (
            min(int(c.get('hist_hit', 0)), int(target) + 3),
            float(c.get('hist_pct', 0.0)),
            float(c.get('red_pct', 0.0)),
            -int(c.get('frame_keep', 10**9)),
        )
        if cur is None or score > cur[0]:
            by_key[c.get('key')] = (score, c)
    picked = [v[1] for v in by_key.values()]
    # Sortera för att få med de starkaste komponenterna först, men obligatoriska
    # filter ska aldrig råka kapas bort av max_items.
    picked.sort(key=lambda c: (float(c.get('hist_pct',0.0))*0.60 + float(c.get('red_pct',0.0))*0.40, int(c.get('hist_hit',0))), reverse=True)
    req = [c for c in picked if c.get('key') in required_keys]
    opt = [c for c in picked if c.get('key') not in required_keys]
    out = req + opt[:max(0, int(max_items) - len(req))]
    # Bevara ordningen men ta bort ev. dubbletter.
    dedup = []
    seen = set()
    for c in out:
        if c.get('key') in seen:
            continue
        seen.add(c.get('key'))
        dedup.append(c)
    return dedup


def _best_hard_group_from_candidates(group_label, group_no, group_candidates, cur_hist, cur_frame, target, frame_rows, frame, antal_matcher, min_members=3, row_matrix=None):
    """Testar hårda gruppkrav med både min och max, t.ex. 5-7 av 8.

    Gruppen måste hålla samlad historikträff efter att kombineras med redan valt paket.
    v12.0ag testar inte bara "minst X" utan även övre spärr, eftersom vissa
    filterfamiljer kan bli för extrema om alla delvillkor träffar samtidigt.
    """
    group_candidates = list(group_candidates or [])
    if len(group_candidates) < int(min_members):
        return None
    htot = len(cur_hist); ftot = len(cur_frame)
    cur_frame_count = int(cur_frame.sum())
    if cur_frame_count <= 0:
        return None
    hist_stack = np.vstack([c['hist_mask'] for c in group_candidates]).astype(int)
    frame_stack = np.vstack([c['frame_mask'] for c in group_candidates]).astype(int)
    hist_scores = hist_stack.sum(axis=0)
    frame_scores = frame_stack.sum(axis=0)
    n = len(group_candidates)
    # Hårda grupper: börja högt. En grupp på 8 filter testar t.ex. 8-8, 7-8, 6-8,
    # men även 5-7 osv. Max=n betyder vanlig "minst"-grupp. Max<n är en äkta
    # Helgardering-lik min/max-grupp.
    min_req_floor = max(1, int(np.ceil(n * 0.58)))
    best = None
    best_score = None
    for req in range(n, min_req_floor - 1, -1):
        for max_req in range(n, req - 1, -1):
            g_hist_mask = (hist_scores >= req) & (hist_scores <= max_req)
            g_frame_mask = (frame_scores >= req) & (frame_scores <= max_req)
            new_hist = cur_hist & g_hist_mask
            hist_hit = int(new_hist.sum())
            if hist_hit < int(target):
                continue
            new_frame = cur_frame & g_frame_mask
            new_count = int(new_frame.sum())
            if new_count >= cur_frame_count:
                continue
            if row_matrix is not None:
                if not _mask_keeps_teckenskydd(row_matrix, new_frame, frame, antal_matcher):
                    continue
            else:
                test_rows = _rows_from_mask(frame_rows, new_frame)
                if selected_signs_missing(test_rows, frame, antal_matcher):
                    continue
            red_pct = 100.0 * (cur_frame_count - new_count) / max(1, cur_frame_count)
            req_ratio = req / max(1, n)
            max_tightness = (n - max_req) / max(1, n)
            # Poäng: först samlad träff, därefter faktisk reducering. Vid jämnt
            # läge premieras hård min-nivå och en meningsfull max-spärr.
            score = (hist_hit, red_pct, req_ratio, max_tightness, -new_count, req, -max_req)
            if best is None or score > best_score:
                best = {
                    'group_label': group_label,
                    'group_no': int(group_no),
                    'req': int(req),
                    'max_req': int(max_req),
                    'n': int(n),
                    'candidates': group_candidates,
                    'hist_mask': g_hist_mask,
                    'frame_mask': g_frame_mask,
                    'hist_hit': hist_hit,
                    'frame_after': new_count,
                    'step_red_pct': red_pct,
                }
                best_score = score
    return best


def _build_grouped_package_for_target(candidates, target, frame_rows, frame, antal_matcher, max_filters=18, min_value_filters=3, required_keys=None, row_matrix=None, initial_hist_mask=None):
    """Bygger ett paket med hårda grupper i stället för att tvinga varje filter.

    Målet är: fler viktiga filter, särskilt värde-/poängfilter, men med hårda gruppkrav
    så träffbilden inte rasar lika hårt som när allt är tvingat.
    """
    if not candidates:
        return None
    required_keys = set(required_keys or [])
    htot = len(candidates[0]['hist_mask'])
    ftot = len(candidates[0]['frame_mask'])
    if initial_hist_mask is not None:
        try:
            cur_hist = np.array(initial_hist_mask, dtype=bool)
            if len(cur_hist) != htot:
                cur_hist = np.ones(htot, dtype=bool)
        except Exception:
            cur_hist = np.ones(htot, dtype=bool)
    else:
        cur_hist = np.ones(htot, dtype=bool)
    cur_frame = np.ones(ftot, dtype=bool)
    groups = []
    steps = []
    used_keys = set()

    # 1) Värde-/poängkärna. Detta är obligatoriskt om min_value_filters > 0.
    value_cands = _pick_best_variant_per_filter_for_group(
        candidates,
        lambda c: str(c.get('category','')) == 'Värde & svårighet',
        target,
        htot,
        min_hist_floor=max(1, int(target) - 6),
        max_items=max(int(min_value_filters), min(9, int(max_filters))),
        required_keys=required_keys,
    )
    value_required_keys = {k for k in required_keys if any(c.get('key') == k and str(c.get('category','')) == 'Värde & svårighet' for c in candidates)}
    if int(min_value_filters) > 0 and len(value_cands) < int(min_value_filters):
        return None
    if value_required_keys and not value_required_keys.issubset({c.get('key') for c in value_cands}):
        return None
    if value_cands:
        # Använd minst min_value_filters, men ta alltid med manuellt obligatoriska värdefilter.
        req_val = [c for c in value_cands if c.get('key') in value_required_keys]
        opt_val = [c for c in value_cands if c.get('key') not in value_required_keys]
        n_val = min(len(value_cands), max(int(min_value_filters), len(req_val), min(8, int(max_filters))))
        group_cands = req_val + opt_val[:max(0, n_val - len(req_val))]
        block = _best_hard_group_from_candidates('Värde-/poänggrupp', 1, group_cands, cur_hist, cur_frame, target, frame_rows, frame, antal_matcher, min_members=max(1, min(max(2, int(min_value_filters)), len(group_cands))), row_matrix=row_matrix)
        if block is None:
            return None if (int(min_value_filters) > 0 or value_required_keys) else None
        cur_hist = cur_hist & block['hist_mask']
        cur_frame = cur_frame & block['frame_mask']
        groups.append(block)
        used_keys.update(c['key'] for c in block['candidates'])
        steps.append({
            'Steg': 'Grupp',
            'Filter/Grupp': block['group_label'],
            'Kategori': 'Värde & svårighet',
            'Intervall': f"{block['req']}–{block.get('max_req', block['n'])} av {block['n']}",
            'Intervallträff': f"{block['hist_hit']}/{htot}",
            'Testnivå': 'hård grupp',
            'Stegreducering': f"{block['step_red_pct']:.1f}%",
            'Efter filter': int(block['frame_after']),
            'Samlad träff efter steg': f"{int(cur_hist.sum())}/{htot}",
        })

    # 2) Komplettera med FAT/Favorit/Struktur som hårda grupper, om de faktiskt förbättrar.
    family_defs = [
        ('FAT-/sekvensgrupp', 2, lambda c: str(c.get('category','')) in {'FAT', 'FAT-sekvenser'}, 3, 7),
        ('Favorit-/skrällgrupp', 3, lambda c: str(c.get('category','')) == 'Favorit & skräll', 2, 6),
        ('Strukturgrupp', 4, lambda c: str(c.get('category','')) == 'Struktur', 3, 8),
        ('Super-Makrogrupp', 5, lambda c: str(c.get('category','')) == 'Super-Makro', 1, 3),
    ]

    # Först: om användaren har kryssat i obligatoriska filter i en familj ska
    # den familjen försöka byggas in direkt. Det gör "måste ingå" till en riktig
    # spärr i rekommenderade paket. Om den inte går att få in utan att rasera
    # träffbild/teckenskydd, faller paketet bort på den träffnivån.
    for label, gno, pred, min_members, max_items in family_defs:
        if any(g.get('group_no') == gno for g in groups):
            continue
        family_required = {k for k in required_keys if any(c.get('key') == k and pred(c) for c in candidates)}
        if not family_required:
            continue
        fam_cands = [c for c in _pick_best_variant_per_filter_for_group(
            candidates,
            lambda c, pred=pred: pred(c) and c.get('key') not in used_keys,
            target,
            htot,
            min_hist_floor=max(1, int(target) - 10),
            max_items=max(max_items, len(family_required) + 3),
            required_keys=family_required,
        ) if c.get('key') not in used_keys]
        if not family_required.issubset({c.get('key') for c in fam_cands}):
            return None
        block = _best_hard_group_from_candidates(label, gno, fam_cands, cur_hist, cur_frame, target, frame_rows, frame, antal_matcher, min_members=1, row_matrix=row_matrix)
        if block is None:
            return None
        cur_hist = cur_hist & block['hist_mask']
        cur_frame = cur_frame & block['frame_mask']
        groups.append(block)
        used_keys.update(c['key'] for c in block['candidates'])
        steps.append({
            'Steg': 'Grupp',
            'Filter/Grupp': block['group_label'],
            'Kategori': 'Obligatorisk grupp',
            'Intervall': f"{block['req']}–{block.get('max_req', block['n'])} av {block['n']}",
            'Intervallträff': f"{block['hist_hit']}/{htot}",
            'Testnivå': 'måste ingå',
            'Stegreducering': f"{block['step_red_pct']:.1f}%",
            'Efter filter': int(block['frame_after']),
            'Samlad träff efter steg': f"{int(cur_hist.sum())}/{htot}",
        })

    while sum(len(g['candidates']) for g in groups) < int(max_filters):
        best_block = None
        best_score = None
        for label, gno, pred, min_members, max_items in family_defs:
            if any(g.get('group_no') == gno for g in groups):
                continue
            fam_cands = [c for c in _pick_best_variant_per_filter_for_group(
                candidates,
                lambda c, pred=pred: pred(c) and c.get('key') not in used_keys,
                target,
                htot,
                min_hist_floor=max(1, int(target) - 8),
                max_items=max_items,
                required_keys=required_keys,
            ) if c.get('key') not in used_keys]
            if len(fam_cands) < min_members:
                continue
            block = _best_hard_group_from_candidates(label, gno, fam_cands, cur_hist, cur_frame, target, frame_rows, frame, antal_matcher, min_members=min_members, row_matrix=row_matrix)
            if block is None:
                continue
            score = (block['step_red_pct'], block['hist_hit'], -block['frame_after'], block['req'] / max(1, block['n']))
            if best_block is None or score > best_score:
                best_block = block
                best_score = score
        if best_block is None:
            break
        # Kräv att kompletterande grupper ger åtminstone tydlig extra effekt.
        if best_block['step_red_pct'] < 2.0:
            break
        cur_hist = cur_hist & best_block['hist_mask']
        cur_frame = cur_frame & best_block['frame_mask']
        groups.append(best_block)
        used_keys.update(c['key'] for c in best_block['candidates'])
        steps.append({
            'Steg': 'Grupp',
            'Filter/Grupp': best_block['group_label'],
            'Kategori': 'Gruppfilter',
            'Intervall': f"{best_block['req']}–{best_block.get('max_req', best_block['n'])} av {best_block['n']}",
            'Intervallträff': f"{best_block['hist_hit']}/{htot}",
            'Testnivå': 'hård grupp',
            'Stegreducering': f"{best_block['step_red_pct']:.1f}%",
            'Efter filter': int(best_block['frame_after']),
            'Samlad träff efter steg': f"{int(cur_hist.sum())}/{htot}",
        })

    if not groups:
        return None
    final_hit = int(cur_hist.sum())
    final_keep = int(cur_frame.sum())
    if final_hit < int(target) or final_keep >= ftot:
        return None

    filters = []
    group_meta = []
    for g in groups:
        gname = f"Grupp {int(g['group_no'])}"
        group_meta.append({
            'name': gname,
            'label': g['group_label'],
            'req': int(g['req']),
            'max_req': int(g.get('max_req', g['n'])),
            'n': int(g['n']),
        })
        for c in g['candidates']:
            c2 = dict(c)
            c2['package_mode'] = gname
            c2['package_group_label'] = g['group_label']
            c2['package_group_req'] = int(g['req'])
            c2['package_group_max_req'] = int(g.get('max_req', g['n']))
            c2['package_group_n'] = int(g['n'])
            filters.append(c2)

    value_filters = sum(1 for c in filters if str(c.get('category','')) == 'Värde & svårighet')
    fat_filters = sum(1 for c in filters if str(c.get('category','')) in {'FAT', 'FAT-sekvenser'})
    structure_filters = sum(1 for c in filters if str(c.get('category','')) == 'Struktur')
    return {
        'target': int(target),
        'target_label': f"minst {int(target)}/{htot}",
        'hist_hit': final_hit,
        'hist_total': htot,
        'frame_start': ftot,
        'frame_after': final_keep,
        'reduction_pct': 100.0 - 100.0 * final_keep / max(1, ftot),
        'num_filters': len(filters),
        'filters': filters,
        'steps': steps,
        'groups': group_meta,
        'package_type': 'Hårda grupper',
        'min_value_filters': int(min_value_filters),
        'value_filters': int(value_filters),
        'fat_filters': int(fat_filters),
        'structure_filters': int(structure_filters),
    }


def _best_filter_pair_lift(candidates, cur_hist, cur_frame, used_keys, target, htot, frame_rows, frame, antal_matcher, row_matrix=None, min_pair_reduction_pct=1.0, require_value_if_needed=False, max_keys=48, variants_per_key=2):
    """Hittar ett filterpar som ger tydligt lyft trots att filtren var för svaga var för sig.

    Detta är ett exakt test mot aktuell radmassa och de 30 liknande historiska
    omgångarna. Det används av paketmotorn efter den vanliga greedy-trappan.
    Syftet är att fånga filter A+B där A och B var för små var för sig, men
    tillsammans ger ett rationellt extra steg utan att tappa målträffen.
    """
    try:
        cur_hist = np.asarray(cur_hist, dtype=bool)
        cur_frame = np.asarray(cur_frame, dtype=bool)
        cur_frame_count = int(cur_frame.sum())
        if cur_frame_count <= 0:
            return None
        used_keys = set(used_keys or [])
        target = int(target)
        min_pair_reduction_pct = float(min_pair_reduction_pct)

        by_key = {}
        for cand in candidates or []:
            key = cand.get('key')
            if key is None or key in used_keys:
                continue
            if int(cand.get('hist_hit', 0)) < target:
                continue
            try:
                new_hist = cur_hist & np.asarray(cand['hist_mask'], dtype=bool)
                hist_hit = int(new_hist.sum())
                if hist_hit < target:
                    continue
                new_frame = cur_frame & np.asarray(cand['frame_mask'], dtype=bool)
                new_count = int(new_frame.sum())
                if new_count >= cur_frame_count:
                    continue
                step_pct = 100.0 * (cur_frame_count - new_count) / max(1, cur_frame_count)
                if step_pct <= 0:
                    continue
            except Exception:
                continue
            is_value = str(cand.get('category', '')) == 'Värde & svårighet'
            score = (hist_hit, float(step_pct), float(cand.get('red_pct', 0.0)), -new_count, 1 if is_value else 0)
            by_key.setdefault(key, []).append({
                'cand': cand,
                'hist': new_hist,
                'frame': new_frame,
                'hist_hit': hist_hit,
                'frame_count': new_count,
                'step_pct': float(step_pct),
                'is_value': bool(is_value),
                'score': score,
            })

        if len(by_key) < 2:
            return None

        # Behåll bara de starkaste nivåerna per filter och de mest lovande filternycklarna.
        per_key = {}
        key_scores = []
        for key, rows in by_key.items():
            rows = sorted(rows, key=lambda r: r['score'], reverse=True)[:int(max(1, variants_per_key))]
            per_key[key] = rows
            best = rows[0]
            key_scores.append((best['step_pct'], best['hist_hit'], best['score'], key))
        key_scores.sort(reverse=True)
        keys = [k for *_rest, k in key_scores[:int(max(2, max_keys))]]

        best_pair = None
        best_score = None
        for i, key_a in enumerate(keys):
            for key_b in keys[i+1:]:
                for a in per_key.get(key_a, []):
                    for b in per_key.get(key_b, []):
                        try:
                            pair_hist = cur_hist & np.asarray(a['cand']['hist_mask'], dtype=bool) & np.asarray(b['cand']['hist_mask'], dtype=bool)
                            pair_hit = int(pair_hist.sum())
                            if pair_hit < target:
                                continue
                            pair_frame = cur_frame & np.asarray(a['cand']['frame_mask'], dtype=bool) & np.asarray(b['cand']['frame_mask'], dtype=bool)
                            pair_count = int(pair_frame.sum())
                            if pair_count >= cur_frame_count:
                                continue
                            pair_step = 100.0 * (cur_frame_count - pair_count) / max(1, cur_frame_count)
                            if pair_step < min_pair_reduction_pct:
                                continue
                        except Exception:
                            continue
                        value_add = int(bool(a['is_value'])) + int(bool(b['is_value']))
                        if require_value_if_needed and value_add <= 0:
                            continue
                        if row_matrix is not None:
                            if not _mask_keeps_teckenskydd(row_matrix, pair_frame, frame, antal_matcher):
                                continue
                        else:
                            test_rows = _rows_from_mask(frame_rows, pair_frame)
                            if selected_signs_missing(test_rows, frame, antal_matcher):
                                continue
                        # Synergi = parets lyft utöver bästa enskilda lyftet. Ett positivt värde
                        # visar att kombinationen gör något som filtren inte gör ensamma.
                        synergy = float(pair_step) - max(float(a['step_pct']), float(b['step_pct']))
                        min_individual_hit = min(int(a['hist_hit']), int(b['hist_hit']))
                        score = (
                            value_add if require_value_if_needed else min(value_add, 1),
                            int(pair_hit),
                            float(pair_step),
                            float(synergy),
                            int(min_individual_hit),
                            -int(pair_count),
                        )
                        if best_pair is None or score > best_score:
                            best_pair = (a, b, pair_hist, pair_frame, pair_hit, pair_count, pair_step, synergy)
                            best_score = score
        return best_pair
    except Exception:
        return None

def _build_recommended_filter_packages(v_m, specs, frame_rows, frame, antal_matcher, hit_levels=None, min_step_reduction_pct=5.0, max_filters=14, min_hit_count=15, frame_adapt=True, min_value_filters=3, required_keys=None, target_frame_after=None, progress_cb=None, manual_hist_mask=None):
    """Bygger Pareto-rekommenderade filterpaket.

    Viktigt från v12.0n:
    - Paketmotorn ignorerar den manuella slidern "Minsta historiska träff på filterintervall".
    - Varje filter finns fortfarande bara en gång i UI, men motorn testar flera intervallnivåer
      bakom kulisserna, t.ex. 100/95/90/85/.../50%.
    - Historisk träff i paketmotorn räknas på filtervärdena från de 30 liknande
      omgångarna, inte genom att återräkna dem med veckans filter_vec.
    - Den visar bästa paket längs träffbild/reducering-skalan, ner till valt min-hit, t.ex. 15/30.
    """
    hist_rows = [normalize_single_row_text(r) for r in list(v_m['Correct_Row']) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    frame_rows = [normalize_single_row_text(r) for r in (frame_rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
    htot = len(hist_rows)
    ftot = len(frame_rows)
    if htot == 0 or ftot == 0:
        return []
    # v12.0br: manuella teckengrupper är spelarens eget förfilter och ska inte
    # sänka träffbilden i rekommenderade filterpaket. Paketmotorn räknar därför
    # historisk filterträff mot alla liknande historiska omgångar, medan reducering
    # och radantal fortfarande räknas på radmassan EFTER manuella teckengrupper.
    # manual_hist_mask accepteras bara bakåtkompatibelt/diagnostiskt men används
    # inte som startmask i paketens historikträff.
    pre_hist_mask = np.ones(htot, dtype=bool)
    pre_hist_hit = int(pre_hist_mask.sum())
    if pre_hist_hit <= 0:
        return [], pd.DataFrame()
    required_keys = set(required_keys or [])
    try:
        target_frame_after = int(target_frame_after) if target_frame_after is not None else None
    except Exception:
        target_frame_after = None
    min_hit_count = int(max(0, min(pre_hist_hit, min_hit_count)))
    min_value_filters = int(max(0, min(12, min_value_filters)))
    if hit_levels is None:
        # Gå hela vägen ner till vald lägstanivå. Manuella teckengrupper påverkar
        # inte denna träffskala; de visas separat och påverkar bara aktuell radmassa.
        hit_levels = list(range(pre_hist_hit, min_hit_count - 1, -1))
    hit_levels = sorted({int(max(0, min(pre_hist_hit, h))) for h in hit_levels if int(h) >= min_hit_count}, reverse=True)

    # Progress/ETA: totalen är en uppskattning men ger en stabil klocka i UI.
    progress_total = max(1, len(specs) + (2 * len(hit_levels)) + 4)
    progress_done = 0

    def _progress(label, best=None, bump=1):
        nonlocal progress_done
        progress_done = min(progress_total, progress_done + int(max(0, bump)))
        if progress_cb is not None:
            try:
                progress_cb(progress_done, progress_total, label, best)
            except Exception:
                pass

    row_matrix = _frame_row_matrix(frame_rows, antal_matcher)

    # Bygg kandidatvarianter. Varje filter får en riktig nivåtrappa, inte bara
    # säkrast + aggressivast. För hastighet räknas grundramens filtervärden
    # en gång per filter och återanvänds för alla intervallnivåer.
    candidates = []
    rejected_by_frame_adapt = []
    # Tätare grid ger mellanlägen, t.ex. FAT Summa 20–28 / 21–27 / 22–26,
    # medan _prune_candidate_ladder tar bort dubbletter och svaga nivåer.
    coverage_grid = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50]
    for si, spec in enumerate(specs, 1):
        spec_cands = []
        try:
            getter = spec.get('getter')
            frame_vals = np.array([float(getter(r)) for r in frame_rows], dtype=float)
        except Exception:
            frame_vals = None
        hist_vals = list(spec.get('hist_values', []))
        hist_arr = np.array([np.nan if pd.isna(v) else float(v) for v in hist_vals], dtype=float)
        for iv in _candidate_intervals_for_spec(spec, coverage_grid):
            interval = iv['interval']
            if frame_adapt:
                block, reason = _frame_capacity_pressure(spec, interval, frame, frame_rows, antal_matcher)
                if block:
                    rejected_by_frame_adapt.append({
                        'Kategori': spec.get('category', ''),
                        'Filter': spec.get('name', ''),
                        'Intervall': iv.get('interval_txt', _display_interval(interval, spec.get('decimals', 0))),
                        'Orsak': reason + (' · Överkörd eftersom filtret är markerat som måste ingå.' if spec.get('key') in required_keys else ''),
                    })
                    if spec.get('key') not in required_keys:
                        continue
            try:
                lo, hi = float(interval[0]), float(interval[1])
                hist_mask = np.isfinite(hist_arr) & (hist_arr >= lo) & (hist_arr <= hi)
                if frame_vals is not None:
                    frame_mask = np.isfinite(frame_vals) & (frame_vals >= lo) & (frame_vals <= hi)
                else:
                    frame_mask = np.array([_spec_pass(r, spec, interval) for r in frame_rows], dtype=bool)
            except Exception:
                continue
            hist_hit = int((pre_hist_mask & hist_mask).sum())
            frame_keep = int(frame_mask.sum())
            red_pct = 100.0 - 100.0 * frame_keep / max(1, ftot)
            if frame_keep >= ftot and spec.get('key') not in required_keys:
                continue
            spec_cands.append({
                'key': spec['key'],
                'name': spec['name'],
                'category': spec.get('category', ''),
                'coverage': iv['coverage'],
                'interval': interval,
                'interval_txt': iv['interval_txt'],
                'hist_mask': hist_mask.astype(bool),
                'frame_mask': frame_mask.astype(bool),
                'hist_hit': hist_hit,
                'hist_total': htot,
                'hist_pct': 100.0 * hist_hit / max(1, htot),
                'frame_keep': frame_keep,
                'red_pct': red_pct,
            })
        pruned = _prune_candidate_ladder(spec_cands, max_levels=18)
        candidates.extend(pruned)
        if si == 1 or si == len(specs) or si % 3 == 0:
            _progress(f"Steg 1/4: bygger nivåtrappor · filter {si}/{len(specs)} · kandidater {len(candidates)}")
        else:
            progress_done = min(progress_total, progress_done + 1)

    packages = []
    candidates_by_key_for_trim = {}
    for _c in candidates:
        candidates_by_key_for_trim.setdefault(_c.get('key'), []).append(_c)
    min_step_reduction_pct = float(min_step_reduction_pct)
    for target_idx, target in enumerate(hit_levels, 1):
        cur_hist = pre_hist_mask.copy()
        cur_frame = np.ones(ftot, dtype=bool)
        chosen = []
        used_keys = set()
        steps = []

        # Obligatoriska filter ska fungera som startkärna, inte som passiv efterkontroll.
        # För varje kryssat filter försöker vi lägga in den bredaste/säkraste variant
        # som fortfarande håller målträffen och teckenskyddet. Därefter får motorn
        # fylla på med kompletterande filter tills paketet blir spelbart eller stoppas
        # av historik/teckenskydd.
        forced_failed = False
        for req_key in list(required_keys):
            if req_key in used_keys:
                continue
            req_cands = [c for c in candidates if c.get('key') == req_key and int(c.get('hist_hit', 0)) >= int(target)]
            req_cands.sort(key=lambda c: (int(c.get('hist_hit', 0)), float(c.get('hist_pct', 0.0)), float(c.get('red_pct', 0.0)), -int(c.get('frame_keep', 10**9))), reverse=True)
            best_req = None
            best_req_score = None
            cur_frame_count = int(cur_frame.sum())
            for cand in req_cands:
                new_hist = cur_hist & cand['hist_mask']
                hist_hit = int(new_hist.sum())
                if hist_hit < int(target):
                    continue
                new_frame = cur_frame & cand['frame_mask']
                new_frame_count = int(new_frame.sum())
                # Obligatoriska filter får följa med även om de inte reducerar
                # just den manuella radmassan. De är användarens "måste ingå"-val
                # och ska inte försvinna bara för att ramen redan uppfyller intervallet.
                if new_frame_count > cur_frame_count:
                    continue
                if row_matrix is not None:
                    if not _mask_keeps_teckenskydd(row_matrix, new_frame, frame, antal_matcher):
                        continue
                else:
                    test_rows = _rows_from_mask(frame_rows, new_frame)
                    if selected_signs_missing(test_rows, frame, antal_matcher):
                        continue
                step_red_pct = 100.0 * (cur_frame_count - new_frame_count) / max(1, cur_frame_count)
                # För obligatorisk kärna prioriteras historisk säkerhet först.
                score = (hist_hit, cand['hist_hit'], step_red_pct, -new_frame_count)
                if best_req is None or score > best_req_score:
                    best_req = (cand, new_hist, new_frame, hist_hit, new_frame_count, step_red_pct)
                    best_req_score = score
            if best_req is None:
                forced_failed = True
                break
            cand, cur_hist, cur_frame, hist_hit, new_frame_count, step_red_pct = best_req
            cand = dict(cand)
            cand['required_in_package'] = True
            chosen.append(cand)
            used_keys.add(cand['key'])
            steps.append({
                'Filter': cand['name'],
                'Kategori': cand['category'],
                'Intervall': cand['interval_txt'],
                'Intervallträff': f"{cand['hist_hit']}/{htot}",
                'Testnivå': f"{cand['coverage']:.0f}% · måste ingå",
                'Stegreducering': f"{step_red_pct:.1f}%",
                'Efter filter': int(new_frame_count),
                'Samlad träff efter steg': f"{hist_hit}/{htot}",
            })
        if forced_failed:
            continue

        while len(chosen) < int(max_filters):
            best = None
            best_score = None
            cur_frame_count = int(cur_frame.sum())
            if cur_frame_count <= 0:
                break
            value_count_now = sum(1 for c in chosen if str(c.get('category', '')) == 'Värde & svårighet')
            need_value = value_count_now < int(min_value_filters)
            eligible_value_exists = False
            eligible_items = []
            for cand in candidates:
                if cand['key'] in used_keys:
                    continue
                # En variant som inte ens själv når målet kan aldrig ingå i ett AND-paket som når målet.
                if cand['hist_hit'] < target:
                    continue
                new_hist = cur_hist & cand['hist_mask']
                hist_hit = int(new_hist.sum())
                if hist_hit < target:
                    continue
                new_frame = cur_frame & cand['frame_mask']
                new_frame_count = int(new_frame.sum())
                if new_frame_count >= cur_frame_count:
                    continue
                step_red_pct = 100.0 * (cur_frame_count - new_frame_count) / max(1, cur_frame_count)
                is_value = str(cand.get('category', '')) == 'Värde & svårighet'

                # Värde-/poängfilter ska kunna bilda en värdekärna.
                # Om användaren kräver minst t.ex. 3 värdefilter men varje filter måste ge 5%
                # NY marginalreducering, kan motorn fastna efter första värdefiltret eftersom
                # många värdefilter överlappar varandra. Därför gäller normal stegreducering
                # för FAT/struktur/övriga filter, men värdefilter som behövs för kvoten får
                # läggas till om de ger mätbar reducering och håller samlad historisk träff.
                min_step_for_cand = float(min_step_reduction_pct)
                if need_value and is_value and int(min_value_filters) > 0:
                    min_step_for_cand = min(min_step_for_cand, 0.5)
                # Om användaren har valt obligatoriska filter och paketet fortfarande
                # ligger över radgränsen, fortsätt leta mer aggressivt efter små men
                # verkliga kompletteringar. Annars kan motorn fastna med ett enda
                # dolt paket över gränsen.
                if required_keys and target_frame_after is not None and cur_frame_count > int(target_frame_after):
                    min_step_for_cand = min(min_step_for_cand, 0.20)
                if step_red_pct < min_step_for_cand:
                    continue

                if row_matrix is not None:
                    if not _mask_keeps_teckenskydd(row_matrix, new_frame, frame, antal_matcher):
                        continue
                else:
                    test_rows = _rows_from_mask(frame_rows, new_frame)
                    if selected_signs_missing(test_rows, frame, antal_matcher):
                        continue
                if is_value:
                    eligible_value_exists = True
                eligible_items.append((cand, new_hist, new_frame, hist_hit, new_frame_count, step_red_pct, is_value))

            # För rekommenderade paket vill vi inte få paket som bara är struktur/FAT.
            # Så länge det finns värde-/poängfilter som klarar kraven prioriteras de tills kvoten är fylld.
            if need_value and eligible_value_exists:
                eligible_items = [x for x in eligible_items if x[-1]]

            for cand, new_hist, new_frame, hist_hit, new_frame_count, step_red_pct, is_value in eligible_items:
                # Score: välj ett balanserat mellanläge. Målet är inte bara maxslakt
                # per filter, utan bästa reducering MED så hög samlad träff som möjligt.
                value_bonus = 1 if (need_value and is_value) else 0
                hit_buffer = int(hist_hit) - int(target)
                cand_safety = int(cand.get('hist_hit', 0))
                abs_gain = int(cur_frame_count - new_frame_count)
                # Om vi ligger över användarens radgräns premieras faktisk väg mot gränsen,
                # annars premieras extra reducering försiktigare så träffen inte offras i onödan.
                if target_frame_after is not None and cur_frame_count > int(target_frame_after):
                    need_rows = max(1, cur_frame_count - int(target_frame_after))
                    target_progress = min(1.0, abs_gain / need_rows)
                else:
                    target_progress = 0.0
                safe_floor = max(int(target), min(int(htot), max(int(round(float(htot) * 0.87)), int(target) + 2)))
                risk_cost = max(0, int(safe_floor) - int(cand_safety))
                heavy_low_hit = 1 if (float(cand.get('red_pct', 0.0)) >= 45.0 and risk_cost > 0) else 0
                if need_value and is_value:
                    # Värdekärnan ska byggas med starka filter först. Ett filter som
                    # reducerar brutalt men bara har låg individuell historikträff får
                    # inte vinna enbart på reducering om ett stabilare värdefilter finns.
                    score = (value_bonus, hist_hit, -risk_cost, -heavy_low_hit, cand_safety, step_red_pct, -new_frame_count, cand['red_pct'])
                else:
                    # När vi fortfarande ligger över radgränsen får faktisk framdrift väga
                    # tungt, men riskkostnad ligger före ren stegreducering. Detta minskar
                    # risken att t.ex. Delta/Avvikelse väljs extremt hårt i onödan.
                    score = (target_progress, hist_hit, hit_buffer, -risk_cost, -heavy_low_hit, cand_safety, step_red_pct, abs_gain, -new_frame_count)
                if best is None or score > best_score:
                    best = (cand, new_hist, new_frame, hist_hit, new_frame_count, step_red_pct)
                    best_score = score
            if best is None:
                break
            cand, cur_hist, cur_frame, hist_hit, new_frame_count, step_red_pct = best
            chosen.append(cand)
            used_keys.add(cand['key'])
            steps.append({
                'Filter': cand['name'],
                'Kategori': cand['category'],
                'Intervall': cand['interval_txt'],
                'Intervallträff': f"{cand['hist_hit']}/{htot}",
                'Testnivå': f"{cand['coverage']:.0f}%",
                'Stegreducering': f"{step_red_pct:.1f}%",
                'Efter filter': int(new_frame_count),
                'Samlad träff efter steg': f"{hist_hit}/{htot}",
            })

        # Kombinationslyft: testa filterpar som var för svaga var för sig men
        # som tillsammans ger tydlig extra reducering utan att tappa målträffen.
        # Detta gör paketmotorn mindre girig och bättre på överlappande småfilter.
        combo_notes = []
        combo_round = 0
        while len(chosen) + 2 <= int(max_filters) and combo_round < 2:
            cur_frame_count = int(cur_frame.sum())
            if cur_frame_count <= 0:
                break
            value_count_now = sum(1 for c in chosen if str(c.get('category', '')) == 'Värde & svårighet')
            need_value_pair = value_count_now < int(min_value_filters)
            pair_min_step = max(1.0, float(min_step_reduction_pct) * 1.10)
            if required_keys and target_frame_after is not None and cur_frame_count > int(target_frame_after):
                pair_min_step = min(pair_min_step, 0.75)
            pair = _best_filter_pair_lift(
                candidates,
                cur_hist,
                cur_frame,
                used_keys,
                int(target),
                int(htot),
                frame_rows,
                frame,
                antal_matcher,
                row_matrix=row_matrix,
                min_pair_reduction_pct=pair_min_step,
                require_value_if_needed=need_value_pair,
                max_keys=42,
                variants_per_key=2,
            )
            if pair is None:
                break
            a, b, pair_hist, pair_frame, pair_hit, pair_count, pair_step, synergy = pair
            cand_a = dict(a['cand'])
            cand_b = dict(b['cand'])
            combo_round += 1
            combo_label = f"Kombinationslyft {combo_round}"
            cand_a['combo_lift'] = combo_label
            cand_b['combo_lift'] = combo_label
            chosen.extend([cand_a, cand_b])
            used_keys.update([cand_a.get('key'), cand_b.get('key')])
            cur_hist = pair_hist
            cur_frame = pair_frame
            steps.append({
                'Filter': f"{cand_a.get('name','')} + {cand_b.get('name','')}",
                'Kategori': 'Kombinationslyft',
                'Intervall': f"{cand_a.get('interval_txt','-')} + {cand_b.get('interval_txt','-')}",
                'Intervallträff': f"{int(cand_a.get('hist_hit',0))}/{htot} + {int(cand_b.get('hist_hit',0))}/{htot}",
                'Testnivå': 'filterpar',
                'Stegreducering': f"{float(pair_step):.1f}%",
                'Efter filter': int(pair_count),
                'Samlad träff efter steg': f"{int(pair_hit)}/{htot}",
            })
            combo_notes.append({
                'Typ': 'Kombinationslyft',
                'Filter': f"{cand_a.get('name','')} + {cand_b.get('name','')}",
                'Intervall': f"{cand_a.get('interval_txt','-')} + {cand_b.get('interval_txt','-')}",
                'Enskild extra reducering': f"{float(a.get('step_pct',0.0)):.1f}% / {float(b.get('step_pct',0.0)):.1f}%",
                'Parreducering': f"{float(pair_step):.1f}%",
                'Synergi': f"{float(synergy):+.1f}%",
                'Efter filter': int(pair_count),
                'Samlad träff bevarad': f"{int(pair_hit)}/{int(htot)}",
            })


        # Eftertrimma valda nivåer: byt till hårdare intervall om hela paketets
        # samlade historikträff inte försämras. Detta fångar t.ex. FAT Summa
        # 19–30 -> 19–26 när båda ger samma 23/30 i färdigt paket.
        chosen, post_trim_notes, trim_hist, trim_frame = _post_trim_forced_package_levels(
            chosen,
            candidates_by_key_for_trim,
            frame,
            antal_matcher,
            row_matrix=row_matrix,
            max_passes=2,
            initial_hist_mask=pre_hist_mask,
        )
        if trim_hist is not None and trim_frame is not None:
            cur_hist, cur_frame = trim_hist, trim_frame
            steps, _, _, _, _ = _rebuild_forced_package_steps(chosen, htot, ftot, initial_hist_mask=pre_hist_mask)
        else:
            post_trim_notes = []
        if combo_notes:
            post_trim_notes = list(combo_notes) + list(post_trim_notes or [])

        final_hit = int(cur_hist.sum())
        final_keep = int(cur_frame.sum())
        value_filters = sum(1 for c in chosen if str(c.get('category', '')) == 'Värde & svårighet')
        fat_filters = sum(1 for c in chosen if str(c.get('category', '')) in {'FAT', 'FAT-sekvenser'})
        structure_filters = sum(1 for c in chosen if str(c.get('category', '')) == 'Struktur')
        # Hård spärr: om användaren kräver t.ex. minst 3 värde-/poängfilter
        # ska paketet inte visas om det bara råkar få en bonus från ett värdefilter.
        # Tidigare var detta mer en prioritering, vilket kunde ge topppaket med bara
        # 1 värdefilter och resten struktur/FAT.
        if int(min_value_filters) > 0 and int(value_filters) < int(min_value_filters):
            continue

        packages.append({
            'target': int(target),
            'target_label': f"minst {int(target)}/{htot}",
            'hist_hit': final_hit,
            'hist_total': htot,
            'frame_start': ftot,
            'frame_after': final_keep,
            'reduction_pct': 100.0 - 100.0 * final_keep / max(1, ftot),
            'num_filters': len(chosen),
            'filters': chosen,
            'steps': steps,
            'min_step_reduction_pct': min_step_reduction_pct,
            'min_hit_count': min_hit_count,
            'frame_adapt': bool(frame_adapt),
            'min_value_filters': int(min_value_filters),
            'value_filters': int(value_filters),
            'fat_filters': int(fat_filters),
            'structure_filters': int(structure_filters),
            'post_trim_notes': post_trim_notes,
        })
        _progress(f"Steg 2/4: bygger tvingade paket · träffnivå {target_idx}/{len(hit_levels)} · bästa hittills {final_hit}/{htot}, {final_keep:,} rader".replace(',', ' '), best={'hit': final_hit, 'total': htot, 'rows': final_keep})

    # Testa även hårda grupppaket. De bygger på samma kandidatvarianter men
    # använder t.ex. 6 av 8 i stället för att tvinga alla 8. Det kan ge fler
    # värde-/poängfilter utan att samlad historikträff faller lika hårt.
    group_packages = []
    for group_idx, target in enumerate(hit_levels, 1):
        try:
            gp = _build_grouped_package_for_target(
                candidates, int(target), frame_rows, frame, antal_matcher,
                max_filters=int(max_filters),
                min_value_filters=int(min_value_filters),
                required_keys=required_keys,
                row_matrix=row_matrix,
                initial_hist_mask=pre_hist_mask,
            )
            if gp is not None:
                group_packages.append(gp)
                _progress(f"Steg 3/4: bygger hårda grupper · träffnivå {group_idx}/{len(hit_levels)} · bästa grupp {gp.get('hist_hit',0)}/{htot}, {int(gp.get('frame_after',0)):,} rader".replace(',', ' '), best={'hit': gp.get('hist_hit',0), 'total': htot, 'rows': gp.get('frame_after',0)})
            else:
                _progress(f"Steg 3/4: bygger hårda grupper · träffnivå {group_idx}/{len(hit_levels)}")
        except Exception:
            _progress(f"Steg 3/4: bygger hårda grupper · träffnivå {group_idx}/{len(hit_levels)}")
            pass
    packages.extend(group_packages)

    if required_keys:
        packages = [p for p in packages if required_keys.issubset({c.get('key') for c in (p.get('filters', []) or [])})]

    _progress("Steg 4/4: väljer Pareto-bästa paket och bygger kandidatanalys")
    pareto_packages = _pareto_reduce_packages(packages)
    # v12.0ad: Grupppaket ska inte bara vara fallback. De visas som egen
    # risk-/reduceringsväg även om rena tvingade paket dominerar i radantal.
    final_packages = _append_group_representatives(pareto_packages, packages, max_group_packages=6)

    # Kandidatanalys: visa att alla filter faktiskt testades, även om de inte hamnade
    # i ett Pareto-paket. Särskilt viktigt för värde-/poängfilter som ofta överlappar
    # andra filter och därför kan bli utkonkurrerade i greedy-urvalet.
    selected_by_key = {}
    selected_detail_by_key = {}
    for pi, pkg in enumerate(final_packages, 1):
        for c in pkg.get('filters', []) or []:
            label = f"P{pi}"
            selected_by_key.setdefault(c.get('key'), []).append(label)
            mode = c.get('package_mode') or pkg.get('package_type', 'Tvingat')
            selected_detail_by_key.setdefault(c.get('key'), []).append(f"{label}: {c.get('interval_txt','-')} ({mode})")

    candidates_by_key = {}
    for c in candidates:
        candidates_by_key.setdefault(c.get('key'), []).append(c)

    audit_rows = []
    for spec in specs:
        key = spec.get('key')
        cand_list = candidates_by_key.get(key, [])
        if not cand_list:
            frame_reasons = [r.get('Orsak', '') for r in rejected_by_frame_adapt if r.get('Filter') == spec.get('name')]
            if frame_reasons:
                comment = 'Blockerat av grundramsanpassning. Exempel: ' + frame_reasons[0]
            else:
                comment = 'Ingen kandidat gav reducering på grundramen eller kunde inte testas.'
            audit_rows.append({
                'Kategori': spec.get('category', ''),
                'Filter': spec.get('name', ''),
                'Nivåer testade': 0,
                'Nivåtrappa': '-',
                'Bästa säkra intervall': '-',
                'Säker träff': '-',
                'Säker reducering': '-',
                'Balanserat intervall': '-',
                'Balanserad träff': '-',
                'Balanserad reducering': '-',
                'Mest reducerande intervall': '-',
                'Mest reducerande träff': '-',
                'Mest reducerar': '-',
                'Valt i paket': 'Nej',
                'Valt intervall': '-',
                'Kommentar': comment,
            })
            continue

        # Visa både bästa historiskt säkra variant och mest reducerande variant.
        # Tidigare visades bara max reducering, vilket gjorde att värdefilter kunde
        # se ut som 3/30 eller 4/30 trots att samma filter också hade bredare och
        # spelbara kandidater.
        best_safe = max(cand_list, key=lambda c: (int(c.get('hist_hit', 0)), float(c.get('red_pct', 0.0)), -int(c.get('frame_keep', 10**9))))
        best_red = max(cand_list, key=lambda c: (float(c.get('red_pct', 0.0)), int(c.get('hist_hit', 0)), -int(c.get('frame_keep', 10**9))))
        best_bal = _balanced_candidate_for_audit(cand_list, htot, min_hit_floor=max(int(min_hit_count), int(round(htot * 0.70))))
        selected = selected_by_key.get(key, [])
        selected_details = selected_detail_by_key.get(key, [])
        if selected:
            comment = 'Valt i ' + ', '.join(selected)
        else:
            comment = 'Testat men ej valt: gav inte bästa Pareto-paketet. Ofta beror detta på överlapp med andra filter eller för låg extra reducering efter tidigare filter.'
        audit_rows.append({
            'Kategori': best_safe.get('category', spec.get('category', '')),
            'Filter': best_safe.get('name', spec.get('name', '')),
            'Nivåer testade': int(len(cand_list)),
            'Nivåtrappa': _candidate_ladder_text(cand_list, htot, max_items=8),
            'Bästa säkra intervall': best_safe.get('interval_txt', '-'),
            'Säker träff': f"{int(best_safe.get('hist_hit', 0))}/{htot}",
            'Säker reducering': f"{float(best_safe.get('red_pct', 0.0)):.1f}%",
            'Balanserat intervall': best_bal.get('interval_txt', '-') if best_bal else '-',
            'Balanserad träff': f"{int(best_bal.get('hist_hit', 0))}/{htot}" if best_bal else '-',
            'Balanserad reducering': f"{float(best_bal.get('red_pct', 0.0)):.1f}%" if best_bal else '-',
            'Mest reducerande intervall': best_red.get('interval_txt', '-'),
            'Mest reducerande träff': f"{int(best_red.get('hist_hit', 0))}/{htot}",
            'Mest reducerar': f"{float(best_red.get('red_pct', 0.0)):.1f}%",
            'Valt i paket': ', '.join(selected) if selected else 'Nej',
            'Valt intervall': ' | '.join(selected_details) if selected_details else '-',
            'Kommentar': comment,
        })

    audit_df = pd.DataFrame(audit_rows)
    _progress(f"Klar: {len(final_packages)} Pareto-paket · {len(candidates)} filterkandidater testade", bump=progress_total)
    return final_packages, audit_df



def _similar_history_for_backtest(global_db, input_vec, antal_matcher, top_n=30, pay_min=0, pay_max=10000000, exclude_index=None, mode='leave-one-out', test_date=None):
    """Hämtar liknande historik för ett backtestfall.

    Backtestet ska inte låta testomgången själv ingå i sin liknande historik.
    Leave-one-out använder alla andra omgångar. Kronologiskt använder bara
    omgångar före testdatumet när datum finns.
    """
    try:
        df = global_db.copy()
    except Exception:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    try:
        if exclude_index is not None and exclude_index in df.index:
            df = df.drop(index=exclude_index)
    except Exception:
        pass
    try:
        if 'Payout' in df.columns:
            df = df[(pd.to_numeric(df['Payout'], errors='coerce').fillna(0) >= int(pay_min)) & (pd.to_numeric(df['Payout'], errors='coerce').fillna(0) <= int(pay_max))]
    except Exception:
        pass
    if str(mode).lower().startswith('krono') and test_date is not None and 'Datum' in df.columns:
        try:
            dates = pd.to_datetime(df['Datum'], errors='coerce')
            td = pd.to_datetime(test_date, errors='coerce')
            if pd.notna(td):
                df = df[dates < td]
        except Exception:
            pass
    if df.empty:
        return df
    krav_odds = int(antal_matcher) * 3
    try:
        input_vec = [float(x) for x in list(input_vec)]
    except Exception:
        return pd.DataFrame()
    if len(input_vec) != krav_odds:
        return pd.DataFrame()
    similarity_vec = get_structural_vector(input_vec)
    weights_arr = np.array([w for i in range(0, krav_odds, 3) for w in [(max(similarity_vec[i:i+3])/100.0)**2]*3])
    sims = []
    for _, r in df.iterrows():
        pv = r.get('Prob_Vector', [])
        if not isinstance(pv, list) or len(pv) != krav_odds:
            sims.append(999999.0)
            continue
        try:
            sims.append(weighted_euclidean(similarity_vec, get_structural_vector(pv), weights_arr))
        except Exception:
            sims.append(999999.0)
    out = df.copy()
    out['Sim'] = sims
    return out.sort_values('Sim').head(int(top_n))


def _package_passes_row(row_str, specs, package):
    """Kontrollerar om en facitrad klarar ett rekommenderat paket.

    Tvingade paket kräver att alla filter passerar. Hårda grupppaket räknar
    antal passerade filter per grupp och jämför mot gruppens min/max.
    """
    row_str = normalize_single_row_text(row_str)
    if not row_str or not isinstance(package, dict):
        return False, 'saknar rad eller paket'
    spec_by_key = {sp.get('key'): sp for sp in (specs or [])}
    filters = list(package.get('filters', []) or [])
    if not filters:
        return True, 'paket utan filter'

    def cand_pass(cand):
        sp = spec_by_key.get(cand.get('key'))
        if sp is None:
            return False
        try:
            val = float(sp.get('getter')(row_str))
            lo, hi = cand.get('interval', (None, None))
            return float(lo) <= val <= float(hi)
        except Exception:
            return False

    if package.get('groups'):
        passed_by_key = {c.get('key'): bool(cand_pass(c)) for c in filters}
        for gm in package.get('groups', []) or []:
            gname = gm.get('name')
            gfilters = [c for c in filters if c.get('package_mode') == gname]
            if not gfilters:
                continue
            hits = sum(1 for c in gfilters if passed_by_key.get(c.get('key'), False))
            mn = int(gm.get('req', 0) or 0)
            mx = int(gm.get('max_req', gm.get('n', len(gfilters))) or len(gfilters))
            if not (mn <= hits <= mx):
                return False, f"{gname} {hits}/{len(gfilters)} utanför {mn}-{mx}"
        forced_left = [c for c in filters if not c.get('package_mode')]
        for c in forced_left:
            if not passed_by_key.get(c.get('key'), False):
                return False, str(c.get('name', 'filter'))
        return True, 'OK'

    for c in filters:
        if not cand_pass(c):
            return False, str(c.get('name', 'filter'))
    return True, 'OK'


def _choose_backtest_package(packages, max_rows):
    """Väljer paket i backtest med samma praktiska princip som diskussionen:
    högsta samlad träff inom radbudget först, sedan lägre radantal.
    """
    packages = list(packages or [])
    if not packages:
        return None, 'inget paket'
    try:
        max_rows = int(max_rows)
    except Exception:
        max_rows = 10**12
    under = [p for p in packages if int(p.get('frame_after', 10**12)) <= max_rows]
    base = under if under else packages
    base = _dedupe_package_list(base)
    base.sort(key=lambda p: (
        -int(p.get('hist_hit', 0)),
        int(p.get('frame_after', 10**12)),
        -float(_package_value_score(p)),
        0 if not _is_hard_group_package(p) else 1,
    ))
    note = 'under radgräns' if under else 'över radgräns'
    return (base[0] if base else None), note


def _ranked_frame_like_widths(base_frame, prob_vector, antal_matcher):
    """Bygger en historiskt rimlig backtestram från testkupongens streck.

    Den använder INTE dagens valda tecken. Den använder bara bredden i dagens
    grundram som mall: har du 1 tecken i en match väljs testkupongens största
    tecken, har du 2 tecken väljs testkupongens två största, och har du 3 tecken
    väljs helgardering. Det gör backtestet relevant utan att testa gamla facit
    mot dagens specifika matchval.
    """
    signs = ['1', 'X', '2']
    out = []
    try:
        pv = [float(x) for x in list(prob_vector)]
    except Exception:
        pv = []
    for i in range(int(antal_matcher)):
        try:
            width = len(normalize_signs((base_frame or [])[i]))
        except Exception:
            width = 3
        width = int(max(1, min(3, width)))
        try:
            vals = pv[i*3:i*3+3]
            order = sorted(range(3), key=lambda j: (-float(vals[j]), j))
            picked = [signs[j] for j in order[:width]]
            out.append(normalize_signs(picked))
        except Exception:
            out.append(signs[:width])
    return out


def _run_package_engine_backtest(global_db, frame, manual_sign_groups, antal_matcher, top_n, pay_min, pay_max, filter_hist_target_pct, rec_settings, required_keys=None, max_tests=10, mode='leave-one-out', progress_cb=None, test_scope='package_only', frame_mode='ranked_widths'):
    """Kör backtest av paketmotorn på historiska omgångar.

    Varje testomgång låtsas vara dagens kupong: dess procentvektor blir input,
    liknande historik väljs utan testomgången, filterdefinitioner byggs om och
    paketmotorn körs med aktuella paketinställningar.

    Viktigt i v12.0bz: standardläget testar paketfiltren, inte dagens manuella
    grundram. Dagens specifika teckenval hör bara till aktuell kupong och ska
    inte döma historiska facit. Därför används en automatisk backtestram som
    behåller samma bredd per match som din grundram men väljer testkupongens
    högst streckade tecken i respektive match. Facitmåttet är primärt:
    "Paket klarar facit".
    """
    if not isinstance(global_db, pd.DataFrame) or global_db.empty:
        return pd.DataFrame(), {'error': 'Ingen historikdatabas laddad.'}
    try:
        db = global_db.copy()
        if 'Payout' in db.columns:
            db = db[(pd.to_numeric(db['Payout'], errors='coerce').fillna(0) >= int(pay_min)) & (pd.to_numeric(db['Payout'], errors='coerce').fillna(0) <= int(pay_max))]
        db['_bt_row_ok'] = db['Correct_Row'].apply(lambda r: len(normalize_single_row_text(r)) == int(antal_matcher))
        db['_bt_vec_ok'] = db['Prob_Vector'].apply(lambda v: isinstance(v, list) and len(v) == int(antal_matcher) * 3)
        db = db[db['_bt_row_ok'] & db['_bt_vec_ok']].copy()
    except Exception as e:
        return pd.DataFrame(), {'error': f'Kunde inte förbereda historik: {e}'}
    if db.empty:
        return pd.DataFrame(), {'error': 'Inga giltiga historiska omgångar att testa.'}

    if 'Datum' in db.columns:
        try:
            db['_bt_date'] = pd.to_datetime(db['Datum'], errors='coerce')
            db = db.sort_values('_bt_date', ascending=False, na_position='last')
        except Exception:
            pass
    else:
        db = db.sort_index(ascending=False)

    package_only = str(test_scope or 'package_only') == 'package_only'
    active_manual_groups_current = _normalize_manual_sign_groups(manual_sign_groups or [], antal_matcher)
    test_rows = list(db.iterrows())[:int(max_tests)]
    rows = []
    skipped = 0
    total = len(test_rows)
    for ti, (idx, test_row) in enumerate(test_rows, 1):
        if progress_cb is not None:
            try:
                progress_cb(ti - 1, max(1, total), f"Backtest {ti}/{total}: väljer liknande historik")
            except Exception:
                pass
        correct = normalize_single_row_text(test_row.get('Correct_Row', ''))
        input_vec = test_row.get('Prob_Vector', [])
        test_date = test_row.get('Datum', None)
        sim_df = _similar_history_for_backtest(
            global_db,
            input_vec,
            antal_matcher,
            top_n=int(top_n),
            pay_min=int(pay_min),
            pay_max=int(pay_max),
            exclude_index=idx,
            mode=mode,
            test_date=test_date,
        )
        if len(sim_df) < max(10, min(int(top_n), 20)):
            skipped += 1
            rows.append({
                'Datum': str(test_date)[:10] if test_date is not None else str(idx),
                'Facit': correct,
                'Status': 'Hoppad över',
                'Orsak': f'För få liknande/priorhistoriska omgångar ({len(sim_df)})',
            })
            continue
        try:
            if package_only:
                # Historiskt rättvisare än dagens tecken: använd bara breddmallen.
                engine_frame = _ranked_frame_like_widths(frame, input_vec, antal_matcher)
                active_manual_groups = []
                frame_label = 'Auto-rankad breddram'
            else:
                # Gammalt/felsökningsläge: exakt dagens grundram och manuella grupper.
                engine_frame = frame
                active_manual_groups = active_manual_groups_current
                frame_label = 'Aktuell grundram'

            engine_frame_rows, _, ok, msg = generate_rows_from_frame(engine_frame, max_rows=None)
            if not ok:
                rows.append({
                    'Datum': str(test_date)[:10] if test_date is not None else str(idx),
                    'Facit': correct,
                    'Status': 'Hoppad över',
                    'Orsak': f'Kunde inte skapa backtestram: {msg}',
                })
                skipped += 1
                continue

            engine_rows_after_manual = _apply_manual_sign_groups_to_rows(engine_frame_rows, active_manual_groups, antal_matcher)
            if not engine_rows_after_manual:
                rows.append({
                    'Datum': str(test_date)[:10] if test_date is not None else str(idx),
                    'Facit': correct,
                    'Status': 'Hoppad över',
                    'Orsak': 'Backtestram/manuella teckengrupper lämnar 0 rader',
                })
                skipped += 1
                continue

            specs_bt = build_clean_filter_specs(
                sim_df,
                input_vec,
                int(antal_matcher),
                slider_u_count=3,
                target_hist_pct=int(filter_hist_target_pct),
                u_rows=None,
                hist_df=global_db,
                max_shock_pct=22,
                candidate_rows=engine_rows_after_manual,
                include_supermakro=True,
            )
            if progress_cb is not None:
                try:
                    progress_cb(ti - 1, max(1, total), f"Backtest {ti}/{total}: kör paketmotor")
                except Exception:
                    pass
            rec_result = _build_recommended_filter_packages(
                sim_df,
                specs_bt,
                engine_rows_after_manual,
                engine_frame,
                int(antal_matcher),
                min_step_reduction_pct=float(rec_settings.get('min_step', 1.0)),
                max_filters=int(rec_settings.get('max_filters', 30)),
                min_hit_count=min(int(rec_settings.get('min_hit', 28)), int(len(sim_df))),
                frame_adapt=bool(rec_settings.get('frame_adapt', True)),
                min_value_filters=int(rec_settings.get('min_value_filters', 3)),
                required_keys=required_keys or [],
                target_frame_after=int(rec_settings.get('display_max_rows', 5000)),
                progress_cb=None,
                manual_hist_mask=None,
            )
            packages_bt = rec_result[0] if isinstance(rec_result, tuple) else rec_result
            pkg, pkg_note = _choose_backtest_package(packages_bt, int(rec_settings.get('display_max_rows', 5000)))

            in_backtest_frame = result_in_frame(correct, engine_frame)
            after_manual = bool(_manual_sign_groups_pass(correct, active_manual_groups)) if (in_backtest_frame and active_manual_groups) else (True if not active_manual_groups else False)
            if pkg is None:
                pkg_pass = False
                fail_reason = 'Inget paket hittades'
                pkg_hit = pkg_total = pkg_rows = None
                pkg_type = '-'
            else:
                pkg_pass, fail_reason = _package_passes_row(correct, specs_bt, pkg)
                pkg_hit = int(pkg.get('hist_hit', 0))
                pkg_total = int(pkg.get('hist_total', len(sim_df)))
                pkg_rows = int(pkg.get('frame_after', 0))
                pkg_type = str(pkg.get('package_type', 'Tvingade filter'))

            # Standardläget ska mäta paketfiltren. Fullflödesläge finns bara som diagnos.
            survives = bool(pkg_pass) if package_only else bool(in_backtest_frame and after_manual and pkg_pass)
            rows.append({
                'Datum': str(test_date)[:10] if test_date is not None else str(idx),
                'Facit': correct,
                'Status': 'OK',
                'Liknande': int(len(sim_df)),
                'Backtestram': frame_label,
                'Facit i backtestram': 'Ja' if in_backtest_frame else 'Nej',
                'Efter manuell': 'Ej använd' if package_only else ('Ja' if after_manual else 'Nej'),
                'Paket klarar facit': 'Ja' if pkg_pass else 'Nej',
                'Huvudresultat': 'Ja' if survives else 'Nej',
                'Paketträff': f"{pkg_hit}/{pkg_total}" if pkg is not None else '-',
                'Paketrader': pkg_rows if pkg_rows is not None else '-',
                'Pakettyp': pkg_type,
                'Valprincip': pkg_note,
                'Orsak': 'OK' if survives else fail_reason,
            })
        except Exception as e:
            rows.append({
                'Datum': str(test_date)[:10] if test_date is not None else str(idx),
                'Facit': correct,
                'Status': 'Fel',
                'Orsak': str(e)[:220],
            })
        if progress_cb is not None:
            try:
                progress_cb(ti, max(1, total), f"Backtest {ti}/{total}: klar")
            except Exception:
                pass

    out = pd.DataFrame(rows)
    ok_df = out[out.get('Status', '') == 'OK'] if not out.empty and 'Status' in out.columns else pd.DataFrame()
    def _count_yes(col):
        try:
            return int((ok_df[col] == 'Ja').sum())
        except Exception:
            return 0
    meta = {
        'tested': int(len(ok_df)),
        'requested': int(total),
        'skipped': int(skipped),
        'ground_hits': _count_yes('Facit i backtestram'),
        'manual_hits': _count_yes('Efter manuell'),
        'package_hits': _count_yes('Paket klarar facit'),
        'survivors': _count_yes('Huvudresultat'),
        'mode': str(mode),
        'top_n': int(top_n),
        'pay_min': int(pay_min),
        'pay_max': int(pay_max),
        'test_scope': 'package_only' if package_only else 'current_full_flow',
    }
    return out, meta

def _package_value_score(p):
    """Spelvärdespoäng för rekommenderade paket.

    Poängen är en effektivitetspoäng: hög samlad historisk träff OCH hög
    reducering. Värde-/poängfilter får en liten bonus eftersom de bättre styr
    utdelningsprofilen än rena strukturfilter. Poängen används för sortering,
    men alla spelbara paket under radgränsen ska kunna väljas manuellt.
    """
    try:
        hit_pct = 100.0 * float(p.get('hist_hit', 0)) / max(1.0, float(p.get('hist_total', 1)))
        red_pct = float(p.get('reduction_pct', 0.0))
        value_bonus = min(8.0, 1.5 * float(p.get('value_filters', 0)))
        group_bonus = 10.0 if _is_hard_group_package(p) else 0.0
        return (hit_pct * red_pct) + value_bonus + group_bonus
    except Exception:
        return 0.0

def _top_playable_packages(packages, n=3):
    pkgs = [p for p in (packages or []) if int(p.get('num_filters', 0)) > 0]
    return sorted(pkgs, key=lambda p: (_package_value_score(p), int(p.get('hist_hit', 0)), -int(p.get('frame_after', 10**12))), reverse=True)[:int(n)]

def _recommended_packages_summary_df(packages, package_index_map=None):
    rows = []
    for i, p in enumerate(packages, 1):
        group_txt = ''
        if p.get('groups'):
            group_txt = ' | '.join([f"{g.get('name')}: {g.get('req')}-{g.get('max_req', g.get('n'))}/{g.get('n')}" for g in p.get('groups', [])])
        try:
            p_label = f"P{int(package_index_map.get(id(p), i))}" if package_index_map else f"P{i}"
        except Exception:
            p_label = f"P{i}"
        rows.append({
            'Paket': p_label,
            'Typ': p.get('package_type', 'Tvingade filter'),
            'Samlad träff': f"{p['hist_hit']}/{p['hist_total']}",
            'Träff %': f"{100.0 * p['hist_hit'] / max(1, p['hist_total']):.1f}%",
            'Grundram → filter': f"{p['frame_start']:,} → {p['frame_after']:,}".replace(',', ' '),
            'Reducerar': f"{p['reduction_pct']:.1f}%",
            'Filter': int(p['num_filters']),
            'Eftertrim': int(len(p.get('post_trim_notes', []) or [])),
            'Gruppkrav': group_txt or '—',
            'Värde-/poängfilter': int(p.get('value_filters', 0)),
            'FAT/sekvens': int(p.get('fat_filters', 0)),
            'Struktur': int(p.get('structure_filters', 0)),
            'Spelvärde': f"{_package_value_score(p):.0f}",
            'Sökmål': p.get('target_label', ''),
        })
    return pd.DataFrame(rows)


def _hidden_packages_reference_df(packages, max_after_rows, max_rows=8):
    """Visar bästa dolda paket över radgränsen som referens.

    Ett paket över radgränsen är ofta för brett för spel, men det är ändå
    användbart att se att säkrare paket fanns och varför de inte visas i
    huvudlistan.
    """
    hidden = [p for p in (packages or []) if int(p.get('frame_after', 10**12)) > int(max_after_rows)]
    if not hidden:
        return pd.DataFrame()
    # Behåll bästa/lägsta radantal per samlad träffnivå.
    best_by_hit = {}
    for p in hidden:
        h = int(p.get('hist_hit', 0))
        cur = best_by_hit.get(h)
        if cur is None or int(p.get('frame_after', 10**12)) < int(cur.get('frame_after', 10**12)):
            best_by_hit[h] = p
    rows = []
    for p in sorted(best_by_hit.values(), key=lambda x: (-int(x.get('hist_hit', 0)), int(x.get('frame_after', 10**12))))[:int(max_rows)]:
        rows.append({
            'Samlad träff': f"{p['hist_hit']}/{p['hist_total']}",
            'Grundram → filter': f"{p['frame_start']:,} → {p['frame_after']:,}".replace(',', ' '),
            'Reducerar': f"{p['reduction_pct']:.1f}%",
            'Filter': int(p.get('num_filters', 0)),
            'Orsak': f"Över radgränsen {int(max_after_rows):,}".replace(',', ' '),
        })
    return pd.DataFrame(rows)




def _group_packages_status_df(packages, max_after_rows, max_rows=6):
    """Sammanfattar grupppaket oavsett om de ligger under radgränsen.

    UI:t ska aldrig kännas som att gruppmotorn saknas. Om grupppaket finns men
    inte är spelbara under vald radgräns visas bästa över gränsen med tydlig
    status. Om inga grupppaket alls finns returneras tom DataFrame och UI:t
    visar en förklaring.
    """
    group_packages = [p for p in (packages or []) if _is_hard_group_package(p)]
    if not group_packages:
        return pd.DataFrame()

    # Visa alltid de bästa synliga först, sedan bästa dolda över gränsen.
    visible = [p for p in group_packages if int(p.get('frame_after', 10**12)) <= int(max_after_rows)]
    hidden = [p for p in group_packages if int(p.get('frame_after', 10**12)) > int(max_after_rows)]
    chosen = []
    chosen.extend(_top_playable_packages(visible, max_rows))

    # För dolda grupppaket: välj bästa per träffnivå så man ser trade-offen.
    best_hidden_by_hit = {}
    for p in hidden:
        h = int(p.get('hist_hit', 0))
        cur = best_hidden_by_hit.get(h)
        if cur is None or int(p.get('frame_after', 10**12)) < int(cur.get('frame_after', 10**12)):
            best_hidden_by_hit[h] = p
    hidden_best = sorted(best_hidden_by_hit.values(), key=lambda x: (-int(x.get('hist_hit',0)), int(x.get('frame_after', 10**12))))[:max(0, int(max_rows) - len(chosen))]
    chosen.extend(hidden_best)
    chosen = _dedupe_package_list(chosen)[:int(max_rows)]

    rows = []
    for i, p in enumerate(chosen, 1):
        after = int(p.get('frame_after', 0))
        group_txt = ' | '.join([f"{g.get('label', g.get('name','Grupp'))}: {g.get('req')}-{g.get('max_req', g.get('n'))}/{g.get('n')}" for g in p.get('groups', [])]) or '—'
        status = 'Under radgränsen' if after <= int(max_after_rows) else f"Över radgränsen {int(max_after_rows):,}".replace(',', ' ')
        rows.append({
            'Grupppaket': f"G{i}",
            'Status': status,
            'Samlad träff': f"{int(p.get('hist_hit',0))}/{int(p.get('hist_total',0))}",
            'Grundram → filter': f"{int(p.get('frame_start',0)):,} → {after:,}".replace(',', ' '),
            'Reducerar': f"{float(p.get('reduction_pct',0.0)):.1f}%",
            'Filter': int(p.get('num_filters',0)),
            'Värde-/poängfilter': int(p.get('value_filters',0)),
            'Gruppkrav': group_txt,
            'Spelvärde': f"{_package_value_score(p):.0f}",
        })
    return pd.DataFrame(rows)


def _group_packages_status_text(packages, max_after_rows):
    group_packages = [p for p in (packages or []) if _is_hard_group_package(p)]
    if not group_packages:
        return "Inga hårda grupppaket skapades på de valda kraven. Vanliga orsaker är för hög minsta träff, för få lämpliga filter i samma familj, teckenskydd eller att gruppen inte gav någon faktisk extra reducering."
    visible = [p for p in group_packages if int(p.get('frame_after', 10**12)) <= int(max_after_rows)]
    hidden = [p for p in group_packages if int(p.get('frame_after', 10**12)) > int(max_after_rows)]
    if visible:
        return f"{len(visible)} hårda grupppaket ligger under radgränsen."
    if hidden:
        best = min(hidden, key=lambda p: int(p.get('frame_after', 10**12)))
        return f"Grupppaket finns, men inget ligger under {int(max_after_rows):,} rader. Närmast är {int(best.get('frame_after',0)):,} rader med {int(best.get('hist_hit',0))}/{int(best.get('hist_total',0))} samlad träff.".replace(',', ' ')
    return "Inga hårda grupppaket att visa."


def _parse_spelvarde_value(v):
    """Robust konvertering av visat spelvärde till tal för färgkodning."""
    try:
        return float(str(v).replace(' ', '').replace(',', '.'))
    except Exception:
        return None


def _spelvarde_band(value):
    """Klassar spelvärde enligt appens praktiska tolkningszoner.

    5 000-7 000 är bästa normalzonen: bra balans mellan samlad historikträff
    och faktisk reducering. Färgningen är bara beslutsstöd; paket ska fortfarande
    bedömas på samlad träff, radantal och filterinnehåll.
    """
    v = _parse_spelvarde_value(value)
    if v is None:
        return ""
    if 5000 <= v <= 7000:
        return "background-color: rgba(46, 204, 113, 0.28); font-weight: 700;"
    if 4000 <= v < 5000 or 7000 < v <= 8000:
        return "background-color: rgba(241, 196, 15, 0.24);"
    if v > 8000:
        return "background-color: rgba(230, 126, 34, 0.26); font-weight: 700;"
    return "color: rgba(128,128,128,0.95);"


def _style_spelvarde_df(df):
    """Markerar spelvärdeskolumnen utan att ändra själva dataframen."""
    if df is None or getattr(df, 'empty', True) or 'Spelvärde' not in df.columns:
        return df
    try:
        return df.style.map(_spelvarde_band, subset=['Spelvärde'])
    except Exception:
        try:
            return df.style.applymap(_spelvarde_band, subset=['Spelvärde'])
        except Exception:
            return df


def _spelvarde_caption():
    return "Spelvärde färgkodas som beslutsstöd: grönt 5 000–7 000, gult 4 000–5 000 eller 7 000–8 000, orange över 8 000. Samlad historikträff och radantal väger fortfarande tyngst."


def _apply_recommended_package_to_session(package, specs, filter_hist_target_pct, top_fav_count):
    chosen_keys = {c['key'] for c in package.get('filters', [])}
    chosen_by_key = {c['key']: c for c in package.get('filters', [])}
    for spec in specs:
        k = spec['key']
        _fhp_spec = _hist_target_for_spec(spec, filter_hist_target_pct)
        range_key = f'filter_range_{k}_h{int(_fhp_spec)}_tf{int(top_fav_count)}'
        if k in chosen_by_key:
            chosen = chosen_by_key[k]
            _mode = chosen.get('package_mode', 'Tvingat')
            _interval = chosen['interval']
            st.session_state[f'filter_mode_{k}'] = _mode
            st.session_state[range_key] = _interval
            _filtercentral_store_set(k, mode=_mode, interval=_interval, spec=spec, target_hist_pct=_fhp_spec)
        else:
            _interval = spec.get('default_interval')
            st.session_state[f'filter_mode_{k}'] = 'Av'
            # Låt avstängda filter ligga på rekommenderat intervall för tydlighet.
            st.session_state[range_key] = _interval
            _filtercentral_store_set(k, mode='Av', interval=_interval, spec=spec, target_hist_pct=_fhp_spec)
    for i in range(1, 7):
        st.session_state[f'group_req_{i}'] = 0
        st.session_state[f'group_req_min_{i}'] = 0
        st.session_state[f'group_req_max_{i}'] = 40
    for g in package.get('groups', []) or []:
        try:
            gi = int(str(g.get('name','Grupp 0')).split()[-1])
            if 1 <= gi <= 6:
                mn = int(g.get('req', g.get('min_req', 0)) or 0)
                mx = int(g.get('max_req', g.get('n', 40)) or 40)
                st.session_state[f'group_req_{gi}'] = mn
                st.session_state[f'group_req_min_{gi}'] = mn
                st.session_state[f'group_req_max_{gi}'] = mx
        except Exception:
            pass

    # Spara vad paketmotorn räknade, så vi kan verifiera att Filtercentralen
    # visar samma samlade historiska träff efter applicering.
    st.session_state['v12_applied_package_meta'] = {
        'hist_hit': int(package.get('hist_hit', 0)),
        'hist_total': int(package.get('hist_total', 0)),
        'frame_after': int(package.get('frame_after', 0)),
        'num_filters': int(package.get('num_filters', len(package.get('filters', []) or []))),
        'signature': _package_signature(package),
        'package_type': package.get('package_type', ''),
        'reduction_pct': float(package.get('reduction_pct', 0.0) or 0.0),
    }
    # Sparas i spelfil/filterpaket så exakt vilket rekommenderat paket som
    # applicerades kan återskapas och jämföras efter uppladdning.
    st.session_state['v12_applied_package_snapshot'] = _json_safe_value(package)





# =============================================================================
# v12.0dd – MÖNSTERMOTOR 2K AKTIV / TOPP 3
# =============================================================================
# Skapar egna dynamiska regler från de 30 liknande rätta raderna.
# Motorn kan nu aktiveras som ett riktigt dynamiskt filterpaket, synas i aktiva
# filter/rättning och visa topp 3 spelbara alternativ med valbar maxradgräns.


def _pm2k_row_str(row):
    try:
        if isinstance(row, str):
            return row.strip().upper()
        if isinstance(row, (list, tuple)):
            return ''.join(str(x).strip().upper() for x in row)
        return str(row).strip().upper()
    except Exception:
        return ''


def _pm2k_prob_groups(prob_vec, n=13):
    try:
        vals = [float(x) for x in list(prob_vec or [])]
    except Exception:
        vals = []
    out = []
    for i in range(int(n)):
        base = 3*i
        tri = vals[base:base+3]
        if len(tri) < 3:
            tri = [0.0, 0.0, 0.0]
        out.append((float(tri[0]), float(tri[1]), float(tri[2])))
    return out


def _pm2k_selected_pcts(row, prob_vec, n=13):
    s = _pm2k_row_str(row)
    groups = _pm2k_prob_groups(prob_vec, n)
    idx_map = {'1': 0, 'X': 1, '2': 2}
    vals = []
    for i in range(min(int(n), len(s), len(groups))):
        idx = idx_map.get(s[i], None)
        if idx is None:
            vals.append(0.0)
        else:
            vals.append(float(groups[i][idx]))
    while len(vals) < int(n):
        vals.append(0.0)
    return vals


def _pm2k_fav_idxs(prob_vec, n=13):
    groups = _pm2k_prob_groups(prob_vec, n)
    out = []
    for i, tri in enumerate(groups):
        try:
            mi = int(np.argmax(list(tri)))
            out.append((i, mi, float(tri[mi]), tuple(float(x) for x in tri)))
        except Exception:
            out.append((i, 0, 0.0, (0.0,0.0,0.0)))
    return out


def _pm2k_sign_idx(ch):
    return {'1':0,'X':1,'2':2}.get(str(ch).upper(), -1)


def _pm2k_longest_run(row, target=None):
    s = _pm2k_row_str(row)
    best = cur = 0
    prev = None
    for ch in s:
        if target is not None and ch != target:
            cur = 0
            prev = None
            continue
        if target is None:
            if ch == prev:
                cur += 1
            else:
                cur = 1
                prev = ch
        else:
            cur += 1
        best = max(best, cur)
    return int(best)


def _pm2k_changes(row):
    s = _pm2k_row_str(row)
    return int(sum(1 for i in range(1, len(s)) if s[i] != s[i-1]))


def _pm2k_zone(row, start, end):
    s = _pm2k_row_str(row)
    return s[int(start):int(end)]


def _pm2k_make_feature_defs(filter_vec, antal_matcher=13):
    """Skapar egna dynamiska featuredefinitioner för mönstermotorn.

    Varje feature räknas både på historiska facitrader med respektive historisk
    procentvektor och på dagens grundram med dagens procentvektor.
    """
    n = int(antal_matcher or 13)
    defs = []
    def add(name, group, fn, desc='', risk='Normal'):
        defs.append({'name': str(name), 'group': str(group), 'fn': fn, 'desc': str(desc), 'risk': str(risk)})

    # Grundläggande streckvärde på vald rad.
    add('Egen poäng: strecksumma vald rad', 'Egen poäng/värde', lambda row,pv: sum(_pm2k_selected_pcts(row,pv,n)), 'Summa av streckprocent på de tecken raden väljer.')
    add('Egen poäng: snittstreck vald rad', 'Egen poäng/värde', lambda row,pv: float(np.mean(_pm2k_selected_pcts(row,pv,n))), 'Medelstreck på valda tecken.')
    add('Egen poäng: streckspridning', 'Egen poäng/värde', lambda row,pv: float(np.std(_pm2k_selected_pcts(row,pv,n))), 'Spridning mellan höga/låga valda streck.')
    add('Egen poäng: lågstreck under 10', 'Egen poäng/värde', lambda row,pv: sum(1 for x in _pm2k_selected_pcts(row,pv,n) if x < 10), 'Antal valda tecken under 10%.')
    add('Egen poäng: lågstreck under 15', 'Egen poäng/värde', lambda row,pv: sum(1 for x in _pm2k_selected_pcts(row,pv,n) if x < 15), 'Antal valda tecken under 15%.')
    add('Egen poäng: mellanskräll 10–25', 'Egen poäng/värde', lambda row,pv: sum(1 for x in _pm2k_selected_pcts(row,pv,n) if 10 <= x <= 25), 'Antal valda tecken 10–25%.')
    add('Egen poäng: värdetecken 20–45', 'Egen poäng/värde', lambda row,pv: sum(1 for x in _pm2k_selected_pcts(row,pv,n) if 20 <= x <= 45), 'Antal valda tecken i mellanzonen 20–45%.')
    add('Egen poäng: högstreck 60+', 'Egen poäng/värde', lambda row,pv: sum(1 for x in _pm2k_selected_pcts(row,pv,n) if x >= 60), 'Antal valda tecken över 60%.')
    add('Egen poäng: överfavoriter 70+', 'Egen poäng/värde', lambda row,pv: sum(1 for x in _pm2k_selected_pcts(row,pv,n) if x >= 70), 'Antal valda tecken över 70%.')
    add('Egen värdebalans: mellan minus extrem', 'Egen poäng/värde', lambda row,pv: sum(1 for x in _pm2k_selected_pcts(row,pv,n) if 20 <= x <= 45) - sum(1 for x in _pm2k_selected_pcts(row,pv,n) if x < 10 or x >= 70), 'Mellantecken minus extrema tecken.')
    add('Egen skrällbalans: små + mellan', 'Egen skräll', lambda row,pv: sum(2 for x in _pm2k_selected_pcts(row,pv,n) if 10 <= x < 25) + sum(1 for x in _pm2k_selected_pcts(row,pv,n) if 25 <= x <= 40) - sum(2 for x in _pm2k_selected_pcts(row,pv,n) if x < 8), 'Premierar spelbara skrällar 10–40%, straffar extrema under 8%.')

    # Favorit-/skrällrelationer.
    def fav_count_top(row, pv, topn):
        s = _pm2k_row_str(row)
        favs = sorted(_pm2k_fav_idxs(pv,n), key=lambda t: t[2], reverse=True)[:int(topn)]
        return sum(1 for i, mi, pct, tri in favs if i < len(s) and _pm2k_sign_idx(s[i]) == mi)
    def fav_miss_top(row, pv, topn):
        s = _pm2k_row_str(row)
        favs = sorted(_pm2k_fav_idxs(pv,n), key=lambda t: t[2], reverse=True)[:int(topn)]
        return sum(1 for i, mi, pct, tri in favs if i < len(s) and _pm2k_sign_idx(s[i]) != mi)
    for topn in [3,4,5,6,8]:
        add(f'Egen favoritregel: topp {topn} favoriter sitter', 'Egen favorit/skräll', lambda row,pv,topn=topn: fav_count_top(row,pv,topn), f'Antal av topp {topn} favoriter som raden tar.')
        add(f'Egen favoritregel: topp {topn} favoritmissar', 'Egen favorit/skräll', lambda row,pv,topn=topn: fav_miss_top(row,pv,topn), f'Antal av topp {topn} favoriter som raden fäller.')

    def fav_count_threshold(row, pv, thr):
        s = _pm2k_row_str(row)
        c=0
        for i,mi,pct,tri in _pm2k_fav_idxs(pv,n):
            if pct >= float(thr) and i < len(s) and _pm2k_sign_idx(s[i]) == mi:
                c += 1
        return c
    def fav_miss_threshold(row, pv, thr):
        s = _pm2k_row_str(row)
        c=0
        for i,mi,pct,tri in _pm2k_fav_idxs(pv,n):
            if pct >= float(thr) and i < len(s) and _pm2k_sign_idx(s[i]) != mi:
                c += 1
        return c
    for thr in [50,60,70]:
        add(f'Egen favoritregel: {thr}%+ favoriter sitter', 'Egen favorit/skräll', lambda row,pv,thr=thr: fav_count_threshold(row,pv,thr), f'Antal favoriter över {thr}% som sitter.')
        add(f'Egen favoritregel: {thr}%+ favoriter faller', 'Egen favorit/skräll', lambda row,pv,thr=thr: fav_miss_threshold(row,pv,thr), f'Antal favoriter över {thr}% som faller.')

    # Tecken och zoner. Detta är egna strukturregler, inte gamla färdiga specs.
    for sign in ['1','X','2']:
        add(f'Egen struktur: antal {sign}', 'Egen struktur', lambda row,pv,sign=sign: _pm2k_row_str(row).count(sign), f'Totalt antal {sign}.')
        add(f'Egen struktur: {sign} första 6', 'Egen struktur', lambda row,pv,sign=sign: _pm2k_zone(row,0,6).count(sign), f'Antal {sign} i match 1–6.')
        add(f'Egen struktur: {sign} sista 7', 'Egen struktur', lambda row,pv,sign=sign: _pm2k_zone(row,6,n).count(sign), f'Antal {sign} i match 7–{n}.')
        add(f'Egen sekvens: längsta följd {sign}', 'Egen struktur', lambda row,pv,sign=sign: _pm2k_longest_run(row,sign), f'Längsta följd med {sign}.')
    add('Egen sekvens: längsta följd totalt', 'Egen struktur', lambda row,pv: _pm2k_longest_run(row,None), 'Längsta följd av samma tecken.')
    add('Egen sekvens: antal teckenbyten', 'Egen struktur', lambda row,pv: _pm2k_changes(row), 'Antal byten mellan tecken i raden.')

    # Zonvärde: streckprofil i första/sista delen.
    def selected_zone_pcts(row,pv,start,end):
        vals = _pm2k_selected_pcts(row,pv,n)
        return vals[int(start):int(end)]
    for label,start,end in [('första 6',0,6),('sista 7',6,n),('mittzon 4–10',3,min(n,10))]:
        add(f'Egen zonpoäng: strecksumma {label}', 'Egen poäng/värde', lambda row,pv,start=start,end=end: sum(selected_zone_pcts(row,pv,start,end)), f'Strecksumma i {label}.')
        add(f'Egen zonpoäng: lågstreck {label}', 'Egen skräll', lambda row,pv,start=start,end=end: sum(1 for x in selected_zone_pcts(row,pv,start,end) if x < 20), f'Antal valda tecken under 20% i {label}.')
        add(f'Egen zonpoäng: värdetecken {label}', 'Egen poäng/värde', lambda row,pv,start=start,end=end: sum(1 for x in selected_zone_pcts(row,pv,start,end) if 20 <= x <= 45), f'Antal valda tecken 20–45% i {label}.')

    return defs


def _pm2k_hist_pairs(v_m, antal_matcher=13):
    pairs = []
    try:
        for _, row in v_m.iterrows():
            pairs.append((_pm2k_row_str(row.get('Correct_Row','')), list(row.get('Prob_Vector', []) or [])))
    except Exception:
        pass
    return pairs


def _pm2k_feature_values(feature, hist_pairs, frame_rows, filter_vec, antal_matcher=13):
    hist_vals = []
    for r, pv in hist_pairs:
        try:
            hist_vals.append(float(feature['fn'](r, pv)))
        except Exception:
            hist_vals.append(np.nan)
    frame_vals = []
    for r in frame_rows or []:
        try:
            frame_vals.append(float(feature['fn'](r, filter_vec)))
        except Exception:
            frame_vals.append(np.nan)
    return hist_vals, frame_vals


def _pm2k_candidate_intervals_for_feature(feature, hist_vals, frame_vals, frame_total, min_hit_floor=27, max_rules_per_feature=6, frame_adapt=False):
    clean = [float(v) for v in hist_vals if not pd.isna(v)]
    if not clean or not frame_vals:
        return []
    htot = len(clean)
    arr = sorted(clean)
    frame_clean = []
    try:
        frame_clean = [float(v) for v in frame_vals if not pd.isna(v)]
    except Exception:
        frame_clean = []

    def _pm2k_fmt_interval_value(x):
        try:
            fx = float(x)
            if abs(fx - round(fx)) < 1e-9:
                return int(round(fx))
            return round(fx, 3)
        except Exception:
            return x

    def _frame_effective_interval(lo2, hi2):
        """Returnerar intervallets verkliga spann på aktuell grundram.

        Exempel: regeln 1 sista 7 = 2–7, men om grundramen bara kan ge max
        fem ettor i match 7–13 är filtret i praktiken 2–5. Detta ska visas
        tydligt utan att ändra historikträffen; 2–7 och 2–5 filtrerar nämligen
        exakt samma radmängd i just den grundramen.
        """
        if not frame_clean:
            return lo2, hi2, None, None, False
        try:
            fmin = float(min(frame_clean)); fmax = float(max(frame_clean))
            eff_lo = max(float(lo2), fmin)
            eff_hi = min(float(hi2), fmax)
            int_like = all(abs(float(v) - round(float(v))) < 1e-9 for v in frame_clean[:500])
            if int_like:
                fmin2 = int(round(fmin)); fmax2 = int(round(fmax))
                eff_lo2 = int(np.ceil(eff_lo)); eff_hi2 = int(np.floor(eff_hi))
                lo_cmp = int(round(float(lo2))) if abs(float(lo2)-round(float(lo2))) < 1e-9 else float(lo2)
                hi_cmp = int(round(float(hi2))) if abs(float(hi2)-round(float(hi2))) < 1e-9 else float(hi2)
            else:
                fmin2 = round(fmin, 3); fmax2 = round(fmax, 3)
                eff_lo2 = round(eff_lo, 3); eff_hi2 = round(eff_hi, 3)
                lo_cmp = round(float(lo2), 3); hi_cmp = round(float(hi2), 3)
            changed = bool(eff_lo2 != lo_cmp or eff_hi2 != hi_cmp)
            return eff_lo2, eff_hi2, fmin2, fmax2, changed
        except Exception:
            return lo2, hi2, None, None, False

    rules = []

    def _maybe_frame_adapt_interval(lo2, hi2):
        """Mjuk grundramsanpassning för dynamiska PM2K-intervall.

        Historiken sätter originalintervallet, men om intervallet ligger nära en
        kant i veckans manuella grundram breddas det försiktigt. Syftet är inte
        att rädda facit i efterhand, utan att undvika att PM2K väljer filter som
        bara kapar en grundramsdriven ytterkant, t.ex. strecksumma första 6 när
        ramen redan har flera spikar/halvor i toppdelen.
        """
        if not frame_adapt or len(frame_clean) < 20:
            return lo2, hi2, False, ''
        try:
            name = str(feature.get('name', ''))
            group = str(feature.get('group', ''))
            fmin = float(min(frame_clean)); fmax = float(max(frame_clean))
            if fmax <= fmin:
                return lo2, hi2, False, ''
            fspan = max(1.0, fmax - fmin)
            q25 = float(np.quantile(frame_clean, 0.25))
            q75 = float(np.quantile(frame_clean, 0.75))
            old_lo, old_hi = lo2, hi2
            is_score_like = any(x in (name + ' ' + group).lower() for x in ['poäng', 'värde', 'skräll', 'favorit', 'strecksumma', 'balans'])
            is_structure_like = 'struktur' in group.lower() or 'sekvens' in name.lower()
            if not (is_score_like or is_structure_like):
                return lo2, hi2, False, ''

            # Större marginal på strecksumma/poäng, mindre på rena antal/struktur.
            if 'strecksumma' in name.lower():
                margin = max(6.0, round(0.04 * fspan, 3))
            elif is_score_like:
                margin = max(2.0, round(0.04 * fspan, 3))
            else:
                margin = max(1.0, round(0.03 * fspan, 3))

            upper_cut = float(hi2) < fmax and (float(hi2) >= q75 or (fmax - float(hi2)) <= 0.35 * fspan)
            lower_cut = float(lo2) > fmin and (float(lo2) <= q25 or (float(lo2) - fmin) <= 0.35 * fspan)
            if upper_cut:
                hi2 = min(fmax, float(hi2) + margin)
            if lower_cut:
                lo2 = max(fmin, float(lo2) - margin)

            # Bevara heltalskänsla där featurevärdena är heltal.
            if all(abs(float(v) - round(float(v))) < 1e-9 for v in frame_clean[:200]):
                lo2 = int(np.floor(float(lo2)))
                hi2 = int(np.ceil(float(hi2)))
            else:
                lo2 = round(float(lo2), 3)
                hi2 = round(float(hi2), 3)

            if lo2 != old_lo or hi2 != old_hi:
                note = f'Grundramsanpassad från {old_lo}–{old_hi} till {lo2}–{hi2}'
                return lo2, hi2, True, note
        except Exception:
            pass
        return lo2, hi2, False, ''

    # Skapa fönster som täcker 30/29/28/27 av 30, och låt de mest reducerande vinna.
    for hit_target in range(htot, max(0, int(min_hit_floor)-1), -1):
        if hit_target <= 0 or hit_target > htot:
            continue
        for i in range(0, htot - hit_target + 1):
            lo = arr[i]
            hi = arr[i + hit_target - 1]
            # Numerisk padding för decimalvärden, men inte mer än nödvändigt.
            if abs(lo - round(lo)) < 1e-9 and abs(hi - round(hi)) < 1e-9:
                lo2, hi2 = int(round(lo)), int(round(hi))
            else:
                lo2, hi2 = round(lo, 3), round(hi, 3)
            original_lo2, original_hi2 = lo2, hi2
            lo2, hi2, frame_adapted, adapt_note = _maybe_frame_adapt_interval(lo2, hi2)
            hist_hit = sum(1 for v in clean if lo2 <= float(v) <= hi2)
            if hist_hit < int(min_hit_floor):
                continue
            keep = 0
            for v in frame_vals:
                try:
                    if lo2 <= float(v) <= hi2:
                        keep += 1
                except Exception:
                    pass
            if keep <= 0 or keep >= int(frame_total):
                continue
            red = 100.0 - 100.0 * keep / max(1, int(frame_total))
            if red < 2.0:
                continue
            width = float(hi2) - float(lo2)
            eff_lo, eff_hi, fposs_lo, fposs_hi, frame_clamped = _frame_effective_interval(lo2, hi2)
            rules.append({
                'feature': feature,
                'name': feature.get('name',''),
                'group': feature.get('group','Mönster'),
                'lo': lo2,
                'hi': hi2,
                'hist_hit': int(hist_hit),
                'hist_total': int(htot),
                'frame_after_single': int(keep),
                'single_reduction_pct': float(red),
                'width': float(width),
                'original_lo': original_lo2,
                'original_hi': original_hi2,
                'frame_adapted': bool(frame_adapted),
                'adapt_note': adapt_note,
                'frame_possible_lo': fposs_lo,
                'frame_possible_hi': fposs_hi,
                'effective_lo': eff_lo,
                'effective_hi': eff_hi,
                'frame_effective_clamped': bool(frame_clamped),
                'desc': feature.get('desc',''),
                'risk': feature.get('risk','Normal'),
                'key': 'PM2K::' + str(abs(hash((feature.get('name',''), lo2, hi2, bool(frame_adapted)))) % 10**12),
            })
    # Dedupe per exakt intervall.
    best = {}
    for r in rules:
        k = (r['name'], r['lo'], r['hi'])
        cur = best.get(k)
        if cur is None or (r['hist_hit'], r['single_reduction_pct']) > (cur['hist_hit'], cur['single_reduction_pct']):
            best[k] = r
    rules = list(best.values())
    # Enskilda regler: historik först, sedan reduktion. För paketbyggaren behåller vi bredd.
    rules.sort(key=lambda r: (-int(r['hist_hit']), -float(r['single_reduction_pct']), float(r['width'])))
    return rules[:int(max_rules_per_feature)]


def _pm2k_build_rule_pool(v_m, frame_rows, filter_vec, antal_matcher=13, min_hit_floor=27, frame_adapt=False):
    features = _pm2k_make_feature_defs(filter_vec, antal_matcher)
    hist_pairs = _pm2k_hist_pairs(v_m, antal_matcher)
    frame_total = len(frame_rows or [])
    all_rules = []
    feature_diag = []
    for feat in features:
        hv, fv = _pm2k_feature_values(feat, hist_pairs, frame_rows, filter_vec, antal_matcher)
        rules = _pm2k_candidate_intervals_for_feature(feat, hv, fv, frame_total, min_hit_floor=min_hit_floor, max_rules_per_feature=5, frame_adapt=bool(frame_adapt))
        all_rules.extend(rules)
        if rules:
            best = rules[0]
            feature_diag.append({
                'Feature': feat.get('name',''), 'Grupp': feat.get('group',''),
                'Bästa intervall': f"{best['lo']}–{best['hi']}",
                'Träff': f"{best['hist_hit']}/{best['hist_total']}",
                'Kvar rader': int(best['frame_after_single']),
                'Reducerar %': round(float(best['single_reduction_pct']), 1),
                'Justerad': 'Ja' if best.get('frame_adapted') else 'Nej',
                'Original': f"{best.get('original_lo')}–{best.get('original_hi')}" if best.get('frame_adapted') else '',
                'Effektiv i grundram': f"{best.get('effective_lo')}–{best.get('effective_hi')}" if best.get('frame_effective_clamped') else '',
                'Möjligt spann i grundram': f"{best.get('frame_possible_lo')}–{best.get('frame_possible_hi')}" if best.get('frame_possible_lo') is not None else '',
                'Beskrivning': feat.get('desc',''),
            })
    # Prioritera egna profil/poängregler i poolen men behåll struktur som möjligt stöd.
    def rule_score(r):
        grp = str(r.get('group',''))
        profile_bonus = 10.0 if any(x in grp for x in ['poäng','värde','skräll','favorit']) else 0.0
        structure_penalty = -4.0 if 'struktur' in grp.lower() else 0.0
        return (int(r.get('hist_hit',0))*100.0 + float(r.get('single_reduction_pct',0))*2.0 + profile_bonus + structure_penalty)
    all_rules = sorted(all_rules, key=rule_score, reverse=True)
    return all_rules, pd.DataFrame(feature_diag)


def _pm2k_rule_masks(rules, v_m, frame_rows, filter_vec, antal_matcher=13):
    hist_pairs = _pm2k_hist_pairs(v_m, antal_matcher)
    out = []
    for r in rules:
        fn = r['feature']['fn']
        lo, hi = r['lo'], r['hi']
        hm = []
        for hr, hpv in hist_pairs:
            try:
                hm.append(bool(lo <= float(fn(hr, hpv)) <= hi))
            except Exception:
                hm.append(False)
        fm = []
        for row in frame_rows or []:
            try:
                fm.append(bool(lo <= float(fn(row, filter_vec)) <= hi))
            except Exception:
                fm.append(False)
        rr = dict(r)
        rr['_hist_mask'] = np.array(hm, dtype=bool)
        rr['_frame_mask'] = np.array(fm, dtype=bool)
        out.append(rr)
    return out


def _pm2k_search_package(v_m, frame_rows, filter_vec, antal_matcher=13, target_rows=2200, min_rows=1850, max_rows=2500, min_hit_floor=27, top_n=3, frame_adapt=False, progress_cb=None):
    """Sök dynamiska mönsterpaket och returnera toppalternativ.

    v12.0dd: funktionen returnerar fortfarande bästa paket som `chosen`, men
    lägger även `_pm2k_alternatives` i paketet och `options` i meta. Detta gör
    att användaren kan välja bland topp 3 spelbara paket i UI:t.
    """
    frame_rows = list(frame_rows or [])
    frame_total = int(len(frame_rows))
    htot = int(len(v_m))
    try:
        if progress_cb:
            progress_cb(3, 'Startar Mönstermotor 2K...')
    except Exception:
        pass
    rules, feat_diag = _pm2k_build_rule_pool(v_m, frame_rows, filter_vec, antal_matcher, min_hit_floor=min_hit_floor, frame_adapt=bool(frame_adapt))
    try:
        if progress_cb:
            progress_cb(18, f'Skapade {len(rules)} dynamiska regler. Förbereder kandidatpool...')
    except Exception:
        pass

    selected = []
    by_group = {}
    for r in rules:
        by_group.setdefault(str(r.get('group','')), []).append(r)
    for g, rs in by_group.items():
        selected.extend(rs[:45])
    selected.extend(rules[:120])

    seen = set(); cand = []
    for r in selected:
        k = (r['name'], r['lo'], r['hi'])
        if k in seen:
            continue
        seen.add(k); cand.append(r)
    cand = cand[:170]
    try:
        if progress_cb:
            progress_cb(28, f'Räknar masker för {len(cand)} kandidater...')
    except Exception:
        pass
    cand = _pm2k_rule_masks(cand, v_m, frame_rows, filter_vec, antal_matcher)

    diag_rows = []
    floors = list(range(htot, max(0, int(min_hit_floor)-1), -1))
    all_under = []
    all_near = []
    # v12.0dg: samla bästa paket per faktisk träffnivå (30/30, 29/30, 28/30, 27/30).
    # Tidigare sparades i praktiken bara bästa paket per sökgolv. När maxrader höjdes
    # kunde ett hårdpressat 28/30-paket inom gränsen försvinna eftersom 30/30-paketet
    # dominerade sökgolvet. Maxrader ska bara vara en visningsspärr, inte stoppa
    # insamlingen av billigare paket inom valda gränser.
    best_under_by_hit = {}

    def _state_summary(state, floor, status):
        rows_left = int(state['frame_mask'].sum())
        hh = int(state['hist_mask'].sum())
        pc = int(state.get('profile_count',0)); sc = int(state.get('structure_count',0))
        return {
            'floor': int(floor),
            'rows': rows_left,
            'hit': hh,
            'total': int(htot),
            'filters': len(state.get('rules',[]) or []),
            'profile_filters': pc,
            'structure_filters': sc,
            'status': status,
            'score': (-hh, rows_left, len(state.get('rules',[]) or []), -pc, sc, abs(rows_left-int(target_rows))),
        }

    def _rule_signature(state):
        return tuple(sorted((str(r.get('name','')), str(r.get('lo')), str(r.get('hi'))) for r in (state.get('rules') or [])))

    def _remember_under_candidate(state, floor):
        try:
            rows_left = int(state['frame_mask'].sum())
            if not (int(min_rows) <= rows_left <= int(max_rows)):
                return
            summ = _state_summary(state, floor, 'UNDER_MAX')
            hh = int(summ.get('hit', 0))
            # Inom samma faktiska träffnivå vill vi ha lägst radantal = bäst reducering.
            rank = (
                int(summ.get('rows', 10**9)),
                int(summ.get('filters', 10**9)),
                -int(summ.get('profile_filters', 0)),
                int(summ.get('structure_filters', 10**9)),
                abs(int(summ.get('rows', 10**9)) - int(target_rows)),
            )
            old = best_under_by_hit.get(hh)
            if old is None or rank < old[0]:
                best_under_by_hit[hh] = (rank, state, summ)
        except Exception:
            return

    for floor_i, floor in enumerate(floors):
        try:
            if progress_cb:
                progress_cb(32 + int(55 * floor_i / max(1, len(floors))), f'Söker paket med träffgolv {floor}/{htot}...')
        except Exception:
            pass
        beam = [{
            'rules': [],
            'used_features': set(),
            'hist_mask': np.ones(htot, dtype=bool),
            'frame_mask': np.ones(frame_total, dtype=bool),
            'profile_count': 0,
            'structure_count': 0,
        }]
        best_under = None
        best_any = None
        max_depth = 14
        beam_width = 240
        for depth in range(max_depth):
            nxt = []
            for state in beam:
                cur_rows = int(state['frame_mask'].sum())
                if cur_rows < int(min_rows):
                    continue
                for rule in cand:
                    fname = str(rule.get('name',''))
                    if fname in state['used_features']:
                        continue
                    new_hmask = state['hist_mask'] & rule['_hist_mask']
                    hh = int(new_hmask.sum())
                    if hh < int(floor):
                        continue
                    new_fmask = state['frame_mask'] & rule['_frame_mask']
                    rows_left = int(new_fmask.sum())
                    if rows_left <= 0 or rows_left >= cur_rows:
                        continue
                    grp = str(rule.get('group','')).lower()
                    is_struct = 'struktur' in grp
                    pc = int(state['profile_count']) + (0 if is_struct else 1)
                    sc = int(state['structure_count']) + (1 if is_struct else 0)
                    # Struktur får vara stöd, men inte äta upp paketet.
                    if sc > max(4, pc + 1):
                        continue
                    ns = {
                        'rules': state['rules'] + [rule],
                        'used_features': set(state['used_features']) | {fname},
                        'hist_mask': new_hmask,
                        'frame_mask': new_fmask,
                        'profile_count': pc,
                        'structure_count': sc,
                    }
                    val = (-hh, rows_left, len(ns['rules']), -pc, sc, abs(rows_left-int(target_rows)))
                    if int(min_rows) <= rows_left <= int(max_rows):
                        if best_under is None or val < best_under[0]:
                            best_under = (val, ns)
                        _remember_under_candidate(ns, floor)
                    if rows_left >= int(min_rows):
                        any_val = (max(0, rows_left-int(max_rows)), -hh, rows_left, len(ns['rules']), -pc, sc, abs(rows_left-int(target_rows)))
                        if best_any is None or any_val < best_any[0]:
                            best_any = (any_val, ns)
                    nxt.append(ns)
            if not nxt:
                break
            try:
                if progress_cb and depth % 2 == 0:
                    progress_cb(32 + int(55 * (floor_i + (depth+1)/max(1, max_depth)) / max(1, len(floors))), f'Träffgolv {floor}/{htot}: djup {depth+1}/{max_depth}, beam {len(nxt)}...')
            except Exception:
                pass
            def state_rank(s):
                rows_left = int(s['frame_mask'].sum())
                hh = int(s['hist_mask'].sum())
                pc = int(s.get('profile_count',0)); sc = int(s.get('structure_count',0))
                in_band = 0 if int(min_rows) <= rows_left <= int(max_rows) else 1
                # v12.0dg: på varje träffgolv ska beamet leta fram bästa reducering
                # för just den nivån. Annars dominerar 30/30-stater även när vi söker
                # 28/30 och billiga paket kan försvinna när maxrader höjs.
                return (in_band, max(0, rows_left-int(max_rows)), rows_left, abs(hh-int(floor)), len(s['rules']), -pc, sc, -hh, abs(rows_left-int(target_rows)))
            nxt.sort(key=state_rank)
            beam = nxt[:beam_width]
            # Fortsätt söka även efter första träff, så topp 3 kan bli bättre.
            if best_under is not None and depth >= 8:
                break

        chosen_for_diag = best_under[1] if best_under is not None else (best_any[1] if best_any is not None else None)
        if chosen_for_diag is not None:
            rows_left = int(chosen_for_diag['frame_mask'].sum())
            hh = int(chosen_for_diag['hist_mask'].sum())
            status = 'UNDER_MAX' if int(min_rows) <= rows_left <= int(max_rows) else 'ÖVER_BUDGET'
            diag_rows.append({
                'Variant': f'MÖNSTERMOTOR2K {floor}/{htot}',
                'Typ': 'Mönstermotor2K',
                'Status': status,
                'Orsak': f'egna dynamiska regler · kandidater={len(cand)} · feature-regler={len(rules)} · profil={int(chosen_for_diag.get("profile_count",0))} · struktur={int(chosen_for_diag.get("structure_count",0))}',
                'Paketträff': f'{hh}/{htot}',
                'Paketrader': rows_left,
                'Filter totalt': len(chosen_for_diag['rules']),
                'Budgetstatus': 'JA' if status == 'UNDER_MAX' else 'NEJ',
            })
            if best_under is not None:
                all_under.append((best_under[0], chosen_for_diag, _state_summary(chosen_for_diag, floor, 'UNDER_MAX')))
            elif best_any is not None:
                all_near.append((best_any[0], chosen_for_diag, _state_summary(chosen_for_diag, floor, 'ÖVER_BUDGET')))
        else:
            diag_rows.append({
                'Variant': f'MÖNSTERMOTOR2K {floor}/{htot}', 'Typ': 'Mönstermotor2K', 'Status': 'INGEN_KANDIDAT',
                'Orsak': f'kandidater={len(cand)} · feature-regler={len(rules)}', 'Paketträff': f'0/{htot}', 'Paketrader': 0, 'Filter totalt': 0, 'Budgetstatus':'NEJ'
            })

    def _dedupe_options(items):
        best = {}
        for score, state, summ in items:
            sig = _rule_signature(state)
            old = best.get(sig)
            if old is None or score < old[0]:
                best[sig] = (score, state, summ)
        return list(best.values())

    # v12.0dg: bygg toppalternativ från både sökgolvets vinnare och bästa paket
    # per faktisk träffnivå. Därmed syns t.ex. ett 1 861-raders 28/30-paket även
    # när maxgränsen är 5 000 och 30/30/29/30 också finns inom gränsen.
    under_seed = []
    under_seed.extend(all_under)
    under_seed.extend(list(best_under_by_hit.values()))
    under = _dedupe_options(under_seed)
    under.sort(key=lambda x: (-int(x[2].get('hit', 0)), int(x[2].get('rows', 10**9)), int(x[2].get('filters', 10**9)), -int(x[2].get('profile_filters', 0)), int(x[2].get('structure_filters', 10**9))))
    if under:
        top = under[:max(1, int(top_n or 3))]
        option_states = []
        option_summaries = []
        for i, (_, state, summ) in enumerate(top, 1):
            s2 = dict(state)
            s2['_pm2k_option_summary'] = dict(summ, option=i)
            option_states.append(s2)
            option_summaries.append(dict(summ, option=i))
            diag_rows.append({
                'Variant': f'MÖNSTERMOTOR2K TOPP {i}',
                'Typ': 'Mönstermotor2K',
                'Status': 'SPELBAR',
                'Orsak': f'toppalternativ {i} · profil={summ.get("profile_filters")} · struktur={summ.get("structure_filters")}',
                'Paketträff': f'{summ.get("hit")}/{summ.get("total")}',
                'Paketrader': int(summ.get('rows', 0)),
                'Filter totalt': int(summ.get('filters', 0)),
                'Budgetstatus': 'JA',
            })
        chosen = option_states[0]
        chosen['_pm2k_alternatives'] = option_states
        meta = {
            'status': 'MÖNSTERMOTOR2K_VALD',
            'rows': int(option_summaries[0]['rows']),
            'hit': int(option_summaries[0]['hit']),
            'total': int(htot),
            'candidate_count': len(cand),
            'feature_rule_count': len(rules),
            'options': option_summaries,
            'target_rows': int(target_rows),
            'min_rows': int(min_rows),
            'max_rows': int(max_rows),
            'frame_adapt': bool(frame_adapt),
        }
        try:
            if progress_cb:
                progress_cb(100, f'Mönstermotor klar: {len(option_states)} spelbara alternativ. Bäst: {meta["rows"]} rader · {meta["hit"]}/{htot}.')
        except Exception:
            pass
        return chosen, meta, pd.DataFrame(diag_rows), feat_diag

    # Inget under budget: returnera diagnos med bästa nära.
    near = _dedupe_options(all_near)
    near.sort(key=lambda x: x[0])
    best_summary = near[0][2] if near else None
    meta = {'status':'MÖNSTERMOTOR2K_INGET_UNDER_MAX','candidate_count':len(cand),'feature_rule_count':len(rules), 'target_rows': int(target_rows), 'min_rows': int(min_rows), 'max_rows': int(max_rows), 'frame_adapt': bool(frame_adapt)}
    if best_summary:
        meta.update({'best_rows': int(best_summary.get('rows',0)), 'best_hit': f"{best_summary.get('hit')}/{best_summary.get('total')}", 'best_filters': int(best_summary.get('filters',0))})
    try:
        if progress_cb:
            progress_cb(100, f'Mönstermotor klar: inget paket under {int(max_rows)}.')
    except Exception:
        pass
    return None, meta, pd.DataFrame(diag_rows), feat_diag

def _pm2k_rule_value(rule, row, filter_vec):
    try:
        fn = rule.get('feature', {}).get('fn')
        if fn is None:
            return None
        return float(fn(row, filter_vec))
    except Exception:
        return None


def _pm2k_rule_passes(rule, row, filter_vec):
    try:
        val = _pm2k_rule_value(rule, row, filter_vec)
        if val is None:
            return False
        return bool(float(rule.get('lo')) <= float(val) <= float(rule.get('hi')))
    except Exception:
        return False


def _pm2k_rules_to_rows(chosen, base_rows=None, filter_vec=None, antal_matcher=13):
    rows = []
    if not isinstance(chosen, dict):
        return pd.DataFrame(rows)
    current_rows = list(base_rows or []) if base_rows is not None else None
    for i, r in enumerate(chosen.get('rules', []) or [], 1):
        before_n = len(current_rows) if current_rows is not None else None
        after_n = None
        step_red = None
        if current_rows is not None and filter_vec is not None:
            try:
                next_rows = [row for row in current_rows if _pm2k_rule_passes(r, row, filter_vec)]
                after_n = len(next_rows)
                if before_n and before_n > 0:
                    step_red = 100.0 - 100.0 * after_n / before_n
                current_rows = next_rows
            except Exception:
                pass
        rows.append({
            'Regel': f'MÖNSTERREGEL – {i:02d}',
            'Grupp': r.get('group',''),
            'Namn': r.get('name',''),
            'Intervall': f"{r.get('lo')}–{r.get('hi')}",
            'Effektiv i grundram': f"{r.get('effective_lo')}–{r.get('effective_hi')}" if r.get('frame_effective_clamped') else '',
            'Möjligt spann i grundram': f"{r.get('frame_possible_lo')}–{r.get('frame_possible_hi')}" if r.get('frame_possible_lo') is not None else '',
            'Original': f"{r.get('original_lo')}–{r.get('original_hi')}" if r.get('frame_adapted') else '',
            'Justerad': 'Ja' if r.get('frame_adapted') else 'Nej',
            'Träff ensam': f"{r.get('hist_hit')}/{r.get('hist_total')}",
            'Kvar ensam': int(r.get('frame_after_single',0)),
            'Reducerar ensam %': round(float(r.get('single_reduction_pct',0.0)), 1),
            'Före steg': '' if before_n is None else int(before_n),
            'Efter steg': '' if after_n is None else int(after_n),
            'Stegreducering %': '' if step_red is None else round(float(step_red), 1),
            'Beskrivning': r.get('desc',''),
        })
    return pd.DataFrame(rows)


def _pm2k_options_to_df(options):
    rows = []
    for i, opt in enumerate(options or [], 1):
        summ = opt.get('_pm2k_option_summary') if isinstance(opt, dict) else None
        if not isinstance(summ, dict):
            continue
        rows.append({
            'Alternativ': i,
            'Rader': int(summ.get('rows', 0)),
            'Träff': f"{summ.get('hit')}/{summ.get('total')}",
            'Filter': int(summ.get('filters', 0)),
            'Profilfilter': int(summ.get('profile_filters', 0)),
            'Strukturfilter': int(summ.get('structure_filters', 0)),
            'Status': summ.get('status', ''),
        })
    return pd.DataFrame(rows)


def _pm2k_meta_from_chosen(chosen, fallback_meta=None):
    base = dict(fallback_meta or {}) if isinstance(fallback_meta, dict) else {}
    if isinstance(chosen, dict):
        summ = chosen.get('_pm2k_option_summary') or {}
        if isinstance(summ, dict) and summ:
            base.update({
                'status': 'MÖNSTERMOTOR2K_VALD',
                'rows': int(summ.get('rows', base.get('rows', 0) or 0)),
                'hit': int(summ.get('hit', base.get('hit', 0) or 0)),
                'total': int(summ.get('total', base.get('total', 0) or 0)),
                'filters': int(summ.get('filters', len(chosen.get('rules', []) or []))),
                'profile_filters': int(summ.get('profile_filters', 0)),
                'structure_filters': int(summ.get('structure_filters', 0)),
            })
        else:
            base.update({'status':'MÖNSTERMOTOR2K_VALD', 'filters': len(chosen.get('rules', []) or [])})
    return base


def _pm2k_hist_pass_count(chosen, v_m, antal_matcher=13):
    if not isinstance(chosen, dict) or not (chosen.get('rules') or []):
        return (0, int(len(v_m)) if hasattr(v_m, '__len__') else 0)
    pairs = _pm2k_hist_pairs(v_m, antal_matcher)
    passed = 0
    for row, pv in pairs:
        ok = True
        for rule in chosen.get('rules') or []:
            if not _pm2k_rule_passes(rule, row, pv):
                ok = False
                break
        if ok:
            passed += 1
    return int(passed), int(len(pairs))


def _pm2k_correction_df(corr_row, chosen, filter_vec):
    """Rättningstabell för aktiva Mönstermotor2K-regler.

    v12.0dj: returnerar samma huvudkolumner som vanliga filtercentralen
    (Status, Läge, Kategori, Filter, Facitvärde, Intervall/regler,
    Historisk träff med intervallet). Tidigare hamnade Träff/Miss i en separat
    kolumn längst till höger, vilket gjorde att Mönstermotor-raderna såg
    tomma ut i den sammanslagna rättningstabellen.
    """
    rows = []
    if not isinstance(chosen, dict) or not (chosen.get('rules') or []):
        return pd.DataFrame(rows)
    for i, r in enumerate(chosen.get('rules') or [], 1):
        val = _pm2k_rule_value(r, corr_row, filter_vec)
        lo, hi = r.get('lo'), r.get('hi')
        ok = False if val is None else bool(float(lo) <= float(val) <= float(hi))
        if val is None:
            val_txt = ''
        else:
            try:
                fv = float(val)
                val_txt = str(int(round(fv))) if abs(fv - round(fv)) < 1e-9 else str(round(fv, 3))
            except Exception:
                val_txt = str(val)
        hist_hit = r.get('hist_hit', '')
        hist_total = r.get('hist_total', '')
        hist_txt = f'{hist_hit}/{hist_total}' if hist_hit != '' and hist_total != '' else ''
        interval_txt = f'{lo}–{hi}'
        effective_txt = f"{r.get('effective_lo')}–{r.get('effective_hi')}" if r.get('frame_effective_clamped') else ''
        rows.append({
            'Status': '✅ Träff' if ok else '❌ Miss',
            'Läge': 'Mönstermotor2K',
            'Kategori': r.get('group','Mönstermotor2K'),
            'Filter': f'Mönstermotor2K – {i:02d} {r.get("name", "")}',
            'Facitvärde': val_txt,
            'Intervall/regler': interval_txt,
            'Effektiv i grundram': effective_txt,
            'Historisk träff med intervallet': hist_txt,
            'Original': f"{r.get('original_lo')}–{r.get('original_hi')}" if r.get('frame_adapted') else '',
            'Justerad': 'Ja' if r.get('frame_adapted') else 'Nej',
            'Typ': 'Dynamisk regel',
        })
    return pd.DataFrame(rows)

def _pm2k_apply_chosen_to_rows(rows, chosen, filter_vec, antal_matcher=13):
    """Applicerar valda Mönstermotor2K-regler på en radlista.

    Reglerna är dynamiska och finns inte som vanliga filtercentral-specs. Därför
    används de direkt på raden vid körning i stället för att skrivas in i
    filtercentralens fasta filterlista.
    """
    if not isinstance(chosen, dict) or not (chosen.get('rules') or []):
        return list(rows or [])
    out = []
    for row in rows or []:
        ok = True
        for r in chosen.get('rules') or []:
            try:
                fn = r.get('feature', {}).get('fn')
                if fn is None:
                    ok = False
                    break
                val = float(fn(row, filter_vec))
                if not (float(r.get('lo')) <= val <= float(r.get('hi'))):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            out.append(row)
    return out


def _pm2k_active_label(meta):
    if not isinstance(meta, dict):
        return ''
    try:
        return f"Mönstermotor2K aktiv: {int(meta.get('rows', 0)):,} rader · {int(meta.get('hit', 0))}/{int(meta.get('total', 0))} träff".replace(',', ' ')
    except Exception:
        return 'Mönstermotor2K aktiv'


# =============================================================================
# v12.0cu – EXAKT COLAB V46-PORT
# =============================================================================
# Viktigt: denna sektion ska inte vara en ny motor.
# Den använder samma V46-modul som Colabfilen: motorn.py.
# Den kör samma råvarianter som V46-finalen:
#   B52200_TRUE, B52200_BS29_TIGHT, B52200_OLD29_CV, REF_SUPER
# Därefter används Colabfunktionens egen _v46_make_micro_detail för att skapa
# MICRO_TIGHT_CLOSE / MICRO_TIGHT_SOFT / MICRO_CV_CLOSE / MICRO_BEST_CLOSE.
# Appen väljer sedan B52200_MICRO_TIGHT_CLOSE, dvs samma huvudbeslut som V46.


V46_COLAB_LOCKED_VARIANTS = ['B52200_TRUE', 'B52200_BS29_TIGHT', 'B52200_OLD29_CV', 'REF_SUPER']
V46_COLAB_PRIMARY_SYNTH = 'B52200_MICRO_TIGHT_CLOSE'


def _v46_engine_module_path():
    try:
        base = Path(__file__).resolve().parent
    except Exception:
        base = Path.cwd()
    return base / 'motorn.py'


@st.cache_resource(show_spinner=False)
def _load_v46_engine_module_cached(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Hittar inte V46-Colabmotorn: {path.name}. Lägg motorn.py i samma mapp som app.py.")
    spec = importlib.util.spec_from_file_location('tipset_final_motor_v46_exact_colab', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError('Kunde inte ladda V46-Colabmotorn som Python-modul.')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['tipset_final_motor_v46_exact_colab'] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_v46_engine_module():
    return _load_v46_engine_module_cached(str(_v46_engine_module_path()))


def _v46_final_args(top_n, pay_min, pay_max, filter_hist_target_pct=95):
    """Exakta V46-finalargument från Colab, med bara liveparametrarna top_n/pay_min/pay_max."""
    return argparse.Namespace(
        variants=','.join(V46_COLAB_LOCKED_VARIANTS),
        max_tests=1, test_offset=0, random_seed=20260720, sample_mode='current_coupon',
        top_n=int(top_n), wide_n=max(35, int(top_n)), pay_min=int(pay_min), pay_max=int(pay_max),
        filter_hist_target_pct=int(filter_hist_target_pct), frame_profile='manual', frames='manual', mode='leave-one-out',
        candidate_min_hit=24, min_candidate_val_pct=52.0, min_structure_val_pct=64.0,
        min_gap_score=0.45, variants_per_key=10, max_candidates=340,
        fast_no_supermakro=True, min_hit=23, min_package_val_pct=56.0,
        min_structure_package_val_pct=74.0, min_unique_rows=8,
        beam_width=80, archive_width=950, structure_seed_count=2,
        hit_power=2.40, validation_power=0.70, reduction_power=4.50,
        payout_weight=0.02, cluster_weight=0.05, payout_direction_weight=0.03,
        bundle_pool_size=8, triple_pool_size=0, max_bundle_trials_per_state=18,
        max_bundle_keep_per_state=6, row_bucket_size=200, per_bucket_keep=9,
        enable_triples=False, v15_group_max_filters=18, v15_max_group_candidates=75,
        v15_min_group_size=3, v15_cross_per_family=2, v15_cross_max_filters=12,
    )


def _v46_pkg_hit_number(x, default=0):
    try:
        if isinstance(x, str) and '/' in x:
            return int(str(x).split('/')[0])
        return int(x)
    except Exception:
        return int(default)


def _v46_exact_detail_row(mod, variant_id, pkg=None, error=''):
    """Skapa en Colab-liknande detailrad för aktuell kupong utan facitkolumner."""
    label = variant_id
    try:
        label = str(mod.V15_VARIANT_DEFS.get(variant_id, {}).get('label', variant_id))
    except Exception:
        pass
    if not isinstance(pkg, dict):
        return {
            'Datum': 'AKTUELL_KUPONG',
            'Variant': variant_id,
            'Variantnamn': label,
            'Status': 'Fel' if error else 'Saknas',
            'Orsak': error or 'Paket saknas',
            'Paketträff': '-', 'Valideringsträff': '-', 'Paketrader': '-',
            'Grundram rader': '-', 'Reducerar %': '-', 'Filter totalt': '-',
            'Strukturfilter': '-', 'Profilfilter': '-', 'FAT/ABC-filter': '-',
            'Värde/favorit/skräll': '-', 'Budgetstatus': '-',
        }
    counts = {}
    try:
        counts = mod._v15_group_counts(pkg)
    except Exception:
        counts = {}
    return {
        'Datum': 'AKTUELL_KUPONG',
        'Variant': variant_id,
        'Variantnamn': label,
        'Status': 'OK',
        'Orsak': 'OK',
        'Paketträff': f"{int(pkg.get('hist_hit', 0) or 0)}/{int(pkg.get('hist_total', 0) or 0)}",
        'Valideringsträff': f"{int(pkg.get('val_hit', 0) or 0)}/{int(pkg.get('val_total', 0) or 0)}",
        'Grundram rader': int(pkg.get('frame_before', 0) or 0),
        'Paketrader': int(pkg.get('frame_after', 0) or 0),
        'Reducerar %': round(float(pkg.get('reduction_pct', 0.0) or 0.0), 2),
        'Gemensamt score': round(float(pkg.get('joint_score', 0.0) or 0.0), 4),
        'Utdelningsriktning %': round(float(pkg.get('payout_direction_pct', 0.0) or 0.0), 2),
        'Borttagna hist omgångar': int(pkg.get('removed_hist_count', 0) or 0),
        'Medel spret-gap': round(float(pkg.get('cluster_mean', 0.0) or 0.0), 3),
        'Filter totalt': int(pkg.get('num_filters', len(pkg.get('filters', []) or [])) or 0),
        'Strukturfilter': int(pkg.get('structure_filters', 0) or 0),
        'Profilfilter': int(pkg.get('profile_filters', 0) or 0),
        'FAT/ABC-filter': int(pkg.get('fat_filters', 0) or 0),
        'Värde/favorit/skräll': int(pkg.get('edge_filters', 0) or 0),
        'Budgetstatus': str(pkg.get('v31_budget_status', '')),
        **counts,
    }


def _v46_synth_diag_rows_from_detail(detail_micro):
    if not isinstance(detail_micro, pd.DataFrame) or detail_micro.empty or 'Variant' not in detail_micro.columns:
        return []
    rows = []
    synth = detail_micro[detail_micro['Variant'].astype(str).str.startswith('B52200_MICRO')].copy()
    for _, r in synth.iterrows():
        rows.append({
            'Variant': str(r.get('Variant', '')),
            'Variantnamn': str(r.get('Variantnamn', '')),
            'Status': str(r.get('Status', 'OK')),
            'Orsak': str(r.get('Orsak', 'OK')),
            'Paketträff': str(r.get('Paketträff', '-')),
            'Valideringsträff': str(r.get('Valideringsträff', '-')),
            'Grundram rader': r.get('Grundram rader', '-'),
            'Paketrader': r.get('Paketrader', '-'),
            'Reducerar %': r.get('Reducerar %', '-'),
            'Filter totalt': r.get('Filter totalt', '-'),
            'Strukturfilter': r.get('Strukturfilter', '-'),
            'Profilfilter': r.get('Profilfilter', '-'),
            'FAT/ABC-filter': r.get('FAT/ABC-filter', '-'),
            'Värde/favorit/skräll': r.get('Värde/favorit/skräll', '-'),
            'Micro rescue aktiv': str(r.get('Micro rescue aktiv', '')),
            'Micro källa': str(r.get('Micro källa', '')),
            'Micro beslut': str(r.get('Micro beslut', '')),
            'Budgetstatus': str(r.get('Budgetstatus', '')),
        })
    return rows


def _normalize_v46_package_for_app(pkg, variant_label, source_label, micro_row=None):
    """Gör Colab-V46-paketet kompatibelt med appens paketlista/appliceringslogik."""
    if not isinstance(pkg, dict):
        return pkg
    out = dict(pkg)
    out.setdefault('groups', [])
    out['package_type'] = 'Slutmotor V46 – EXAKT Colab-port'
    out['variant'] = str(variant_label or V46_COLAB_PRIMARY_SYNTH)
    out['variant_label'] = f"{variant_label} via {source_label}"
    out['target_label'] = f"V46 Colab {int(out.get('hist_hit', 0))}/{int(out.get('hist_total', 0))}"
    out['value_filters'] = int(out.get('value_filters', out.get('profile_filters', 0)) or 0)
    out['fat_filters'] = int(out.get('fat_filters', 0) or 0)
    out['structure_filters'] = int(out.get('structure_filters', 0) or 0)
    out['num_filters'] = int(out.get('num_filters', len(out.get('filters', []) or [])) or 0)
    meta = dict(out.get('meta') or {})
    meta.update({
        'engine': 'v46_exact_colab_port',
        'colab_variant': str(variant_label or V46_COLAB_PRIMARY_SYNTH),
        'selected_source_package': str(source_label or ''),
        'micro_rescue_active': str((micro_row or {}).get('Micro rescue aktiv', '')).lower() in ['ja','true','1'],
        'micro_reason': str((micro_row or {}).get('Micro beslut', '')),
    })
    out['meta'] = meta
    return out


def _select_v46_colab_primary_from_detail(detail_micro, built):
    """Välj exakt V46:s syntetiska huvudvariant MICRO_TIGHT_CLOSE ur Colab-detail."""
    if not isinstance(detail_micro, pd.DataFrame) or detail_micro.empty:
        raise RuntimeError('V46-Colab skapade ingen detailtabell.')
    rr = detail_micro[detail_micro['Variant'].astype(str).eq(V46_COLAB_PRIMARY_SYNTH)].copy()
    if rr.empty:
        # Ska normalt inte hända. För livekupong måste rådetail innehålla Datum,
        # annars skapar V46:s Colab-funktion inga MICRO-rader. Falla tillbaka
        # till baspaketet i stället för att krascha, men märk tydligt i metadata.
        if 'B52200_TRUE' not in built or not isinstance(built.get('B52200_TRUE'), dict):
            raise RuntimeError(f'V46-Colab skapade inte {V46_COLAB_PRIMARY_SYNTH} och B52200_TRUE saknas.')
        row = {
            'Variant': V46_COLAB_PRIMARY_SYNTH,
            'Micro källa': 'B52200_TRUE',
            'Micro rescue aktiv': 'Nej',
            'Micro beslut': 'MICRO-rad saknades; fallback till B52200_TRUE',
        }
        return built.get('B52200_TRUE'), row, 'B52200_TRUE'
    row = rr.iloc[0].to_dict()
    source = str(row.get('Micro källa', '') or 'B52200_TRUE')
    if source not in built or not isinstance(built.get(source), dict):
        # Safety: synthraden är klonad från basen om ingen rescue används.
        source = 'B52200_TRUE'
    pkg = built.get(source)
    if not isinstance(pkg, dict):
        raise RuntimeError(f'V46-Colab huvudvariant pekade på {source}, men paketet saknas.')
    return pkg, row, source



def _v46_to_int_safe(x, default=0):
    try:
        if isinstance(x, str) and '/' in x:
            return int(str(x).split('/')[0])
        if pd.isna(x):
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)


def _v46_budget_hunt_current(mod, ns, candidates, htot, vtot, ftot, hist_payout, frame_rows, frame, antal_matcher, *, target_rows=2200, min_rows=1700, max_rows=2500, min_hit_floor=27):
    """Sista spelbarhetssteg ovanpå exakt V46-Colab.

    Viktigt: detta bygger INTE nya filterkandidater. Den återanvänder samma
    kandidatpool som V46 redan skapade/transfomerade och gör bara ett separat
    budgeturval om MICRO_TIGHT_CLOSE hamnar för högt.
    """
    if not candidates:
        return None, {'status': 'BUDGETJAKT_INGA_KANDIDATER'}, pd.DataFrame()
    try:
        raw_row_matrix = ns['_frame_row_matrix'](frame_rows, int(antal_matcher))
        sign_bits = mod._build_teckenskydd_bits(ns, raw_row_matrix, frame, int(antal_matcher))
    except Exception:
        sign_bits = tuple()

    # Dedupe: behåll flera nivåer per filter, men inte identisk key+intervall.
    by_sig = {}
    for c in candidates:
        if not isinstance(c, dict) or not c.get('key'):
            continue
        hb = int(c.get('hist_bits', 0) or 0)
        fb = int(c.get('frame_bits', 0) or 0)
        if hb <= 0 or fb <= 0:
            continue
        sig = (str(c.get('key')), str(c.get('interval_txt', c.get('interval', ''))))
        old = by_sig.get(sig)
        def ckey(x):
            return (int(x.get('hist_hit',0) or 0), float(x.get('val_pct',100.0) or 0.0), float(x.get('red_pct',0.0) or 0.0), -int(x.get('frame_keep', 10**12) or 10**12))
        if old is None or ckey(c) > ckey(old):
            by_sig[sig] = dict(c)
    cands = list(by_sig.values())
    if not cands:
        return None, {'status': 'BUDGETJAKT_INGA_GILTIGA_KANDIDATER'}, pd.DataFrame()

    def is_structure(c):
        try:
            return bool(mod._is_structure_candidate(c))
        except Exception:
            return str(c.get('category','')) == 'Struktur'

    def is_profile(c):
        try:
            return bool(mod._is_profile_candidate(c))
        except Exception:
            return not is_structure(c)

    def is_fat(c):
        return str(c.get('category','')) in {'FAT', 'FAT-sekvenser'}

    def is_edge(c):
        return str(c.get('category','')) in {'Värde & svårighet', 'Favorit & skräll'}

    # Viktigt: kategori låses inte. Sortering gynnar däremot filter som kan
    # reducera, håller träff/validering och inte bara är kosmetiska småfilter.
    def cand_sort_key(c):
        return (
            int(c.get('hist_hit',0) or 0),
            float(c.get('val_pct',100.0) or 0.0),
            float(c.get('red_pct',0.0) or 0.0),
            1 if is_profile(c) else 0,
            float(c.get('payout_direction_pct', c.get('payout_lift_pct', 0.0)) or 0.0),
            -int(c.get('frame_keep', 10**12) or 10**12),
        )
    cands.sort(key=cand_sort_key, reverse=True)
    # Begränsa inte för hårt; beam gör jobbet men Streamlit måste orka.
    cands = cands[:420]

    def full_hist_bits(): return (1 << int(htot)) - 1
    def full_val_bits(): return (1 << int(vtot)) - 1 if int(vtot) > 0 else 0
    def full_frame_bits(): return (1 << int(ftot)) - 1

    def state_counts(st):
        return int(st['hist_bits'].bit_count()), (int(st['val_bits'].bit_count()) if int(vtot) > 0 else 0), int(st['frame_bits'].bit_count())

    def state_quality(st, hfloor, vfloor):
        # Budgetjaktens mål är INTE högsta möjliga 30/30-paket. När ett
        # träffgolv provas (t.ex. 29/30) är alla paket >= golvet giltiga, men
        # sorteringen måste prioritera spelbart radantal först. Annars väljs
        # alltid 30/30 på 4k-5k rader och lägre träffgolv används aldrig.
        hh, vv, rr = state_counts(st)
        in_band = 1 if int(min_rows) <= rr <= int(max_rows) else 0
        under = 1 if rr <= int(max_rows) else 0
        over_pen = max(0, rr - int(max_rows))
        under_pen = max(0, int(min_rows) - rr)
        chosen = st.get('chosen', ())
        prof = sum(1 for c in chosen if is_profile(c))
        fat = sum(1 for c in chosen if is_fat(c))
        edge = sum(1 for c in chosen if is_edge(c))
        structure = sum(1 for c in chosen if is_structure(c))
        payout = sum(float(c.get('payout_direction_pct', c.get('payout_lift_pct', 0.0)) or 0.0) for c in chosen)
        # Fashit 1: nå budgetbandet. Fashit 2: om inte budget nås, kom så nära
        # under/över som möjligt. Först därefter rankas träff/validering.
        return (
            in_band,
            under,
            -abs(rr - int(target_rows)),
            -over_pen,
            -under_pen,
            hh,
            vv,
            prof + edge + fat,
            payout,
            -structure,
            -len(chosen),
        )

    def apply_one(st, c, hfloor, vfloor):
        key = str(c.get('key',''))
        if not key or key in st['used_keys']:
            return None
        new_h = int(st['hist_bits']) & int(c.get('hist_bits', 0) or 0)
        if int(new_h.bit_count()) < int(hfloor):
            return None
        if int(vtot) > 0:
            new_v = int(st['val_bits']) & int(c.get('val_bits', 0) or 0)
            if int(new_v.bit_count()) < int(vfloor):
                return None
        else:
            new_v = 0
        new_f = int(st['frame_bits']) & int(c.get('frame_bits', 0) or 0)
        new_rows = int(new_f.bit_count())
        cur_rows = int(st['frame_bits'].bit_count())
        if new_rows <= 0 or new_rows >= cur_rows:
            return None
        # Stoppa extrem överpress. Under 1500 blir det sällan rätt pipeline för 20% 13-chans.
        if new_rows < 1500:
            return None
        try:
            if sign_bits and any((new_f & int(bits)) == 0 for bits in sign_bits):
                return None
        except Exception:
            pass
        removed = cur_rows - new_rows
        c2 = dict(c)
        c2['step_red_pct'] = 100.0 * removed / max(1, cur_rows)
        c2['budget_hunt_source_variant'] = str(c2.get('budget_hunt_source_variant', c2.get('variant', '')))
        step = {
            'Filter': c2.get('name',''),
            'Kategori': c2.get('category',''),
            'Intervall': c2.get('interval_txt','-'),
            'Intervallträff': f"{int(c2.get('hist_hit',0) or 0)}/{int(c2.get('hist_total', htot) or htot)}",
            'Stegreducering': f"{float(c2['step_red_pct']):.1f}%",
            'Efter filter': int(new_rows),
            'Samlad träff efter steg': f"{int(new_h.bit_count())}/{int(htot)}",
            'Samlad validering': f"{int(new_v.bit_count())}/{int(vtot)}" if int(vtot)>0 else '-',
            'Fas': 'budgetjakt',
        }
        return {
            'hist_bits': new_h,
            'val_bits': new_v,
            'frame_bits': new_f,
            'used_keys': frozenset(set(st['used_keys']) | {key}),
            'chosen': tuple(list(st.get('chosen', ())) + [c2]),
            'steps': tuple(list(st.get('steps', ())) + [step]),
        }

    diag_rows = []
    best_any = None
    for hfloor in range(int(htot), int(min_hit_floor)-1, -1):
        # Budgetjakt ska kunna offra träff stegvis. Därför får även
        # valideringsgolvet följa med ned. Gammal formel låste 29/30 nästan
        # lika hårt som 30/30 och gjorde sökningen tandlös.
        if int(vtot) > 0:
            miss = max(0, int(htot) - int(hfloor))
            vfloor = max(0, int(vtot) - (2 + 2 * miss))
        else:
            vfloor = 0
        initial = {'hist_bits': full_hist_bits(), 'val_bits': full_val_bits(), 'frame_bits': full_frame_bits(), 'used_keys': frozenset(), 'chosen': tuple(), 'steps': tuple()}
        states = [initial]
        archive = []
        seen = set()
        max_depth = min(26, len({str(c.get('key')) for c in cands if c.get('key')}))
        beam_width = 260
        archive_width = 2500
        for depth in range(max_depth):
            expanded = []
            for st0 in states:
                for cand in cands:
                    ns2 = apply_one(st0, cand, hfloor, vfloor)
                    if ns2 is None:
                        continue
                    sig = (ns2['hist_bits'], ns2['val_bits'], ns2['frame_bits'])
                    if sig in seen:
                        continue
                    seen.add(sig)
                    expanded.append(ns2)
            if not expanded:
                break
            expanded.sort(key=lambda st: state_quality(st, hfloor, vfloor), reverse=True)
            states = expanded[:beam_width]
            archive.extend(states)
            if len(archive) > archive_width:
                archive.sort(key=lambda st: state_quality(st, hfloor, vfloor), reverse=True)
                archive = archive[:archive_width]
        if archive:
            archive.sort(key=lambda st: state_quality(st, hfloor, vfloor), reverse=True)
            best_level = archive[0]
            bh, bv, br = state_counts(best_level)
            diag_rows.append({
                'Variant': f'BUDGETJAKT {hfloor}/{htot}',
                'Typ': 'Budgetjakt',
                'Status': 'Diagnos',
                'Paketträff': f'{bh}/{htot}',
                'Valideringsträff': f'{bv}/{vtot}' if int(vtot)>0 else '-',
                'Paketrader': int(br),
                'Filter totalt': int(len(best_level.get('chosen', ()))),
                'Budgetstatus': 'JA' if int(br) <= int(max_rows) else 'NEJ',
                'Orsak': f'Kandidater={len(cands)}, maxdjup={max_depth}, golv={hfloor}/{htot}, valgolv={vfloor}/{vtot}',
            })
            if best_any is None or state_quality(best_level, hfloor, vfloor) > state_quality(best_any, hfloor, vfloor):
                best_any = best_level
        winners = [st for st in archive if int(min_rows) <= int(st['frame_bits'].bit_count()) <= int(max_rows)]
        if not winners:
            winners = [st for st in archive if int(st['frame_bits'].bit_count()) <= int(max_rows)]
        if winners:
            winners.sort(key=lambda st: state_quality(st, hfloor, vfloor), reverse=True)
            best = winners[0]
            bh, bv, br = state_counts(best)
            chosen = list(best.get('chosen', ()))
            structure_filters = sum(1 for c in chosen if is_structure(c))
            profile_filters = sum(1 for c in chosen if is_profile(c))
            fat_filters = sum(1 for c in chosen if is_fat(c))
            edge_filters = sum(1 for c in chosen if is_edge(c))
            pkg = {
                'variant': 'V46_BUDGET_HUNT',
                'variant_label': f'V46 budgetjakt {bh}/{htot}',
                'target': int(bh),
                'target_label': f'Budgetjakt {bh}/{htot}',
                'hist_hit': int(bh), 'hist_total': int(htot),
                'val_hit': int(bv), 'val_total': int(vtot),
                'frame_start': int(ftot), 'frame_before': int(ftot), 'frame_after': int(br),
                'reduction_pct': 100.0 - 100.0 * int(br) / max(1, int(ftot)),
                'num_filters': int(len(chosen)),
                'filters': chosen,
                'steps': list(best.get('steps', ())),
                'package_type': 'Slutmotor V46 – budgetjakt spelbar',
                'structure_filters': int(structure_filters),
                'profile_filters': int(profile_filters),
                'fat_filters': int(fat_filters),
                'edge_filters': int(edge_filters),
                'value_filters': int(profile_filters),
                'v31_budget_status': 'BUDGETJAKT_UNDER_2500',
                'groups': [],
                'meta': {
                    'engine': 'v46_colab_plus_budget_hunt',
                    'source': 'same_v46_candidate_pool',
                    'hist_floor_used': int(hfloor),
                    'val_floor_used': int(vfloor),
                    'target_rows': int(target_rows),
                    'min_rows': int(min_rows),
                    'max_rows': int(max_rows),
                    'candidate_count': int(len(cands)),
                },
            }
            meta = dict(pkg['meta'])
            meta['status'] = 'BUDGETJAKT_VALD'
            meta['rows'] = int(br)
            return pkg, meta, pd.DataFrame(diag_rows)

    meta = {'status': 'BUDGETJAKT_INGET_UNDER_2500', 'candidate_count': int(len(cands)), 'target_rows': int(target_rows), 'max_rows': int(max_rows), 'frame_adapt': bool(frame_adapt)}
    if best_any is not None:
        bh, bv, br = state_counts(best_any)
        meta.update({'best_rows': int(br), 'best_hit': int(bh), 'best_val': int(bv), 'best_filters': int(len(best_any.get('chosen', ())))})
    return None, meta, pd.DataFrame(diag_rows)



def _budgetmotor2k_profile_first_current(mod, ns, candidates, htot, vtot, ftot, hist_payout, frame_rows, frame, antal_matcher, *, target_rows=2200, min_rows=1850, max_rows=2500, min_hit_floor=27):
    """Budgetmotor 2K: separat spelbarhetsmotor efter V46.

    Den här är avsiktligt inte en V46/B52200-fallback. Den använder samma
    kandidatpool som appen redan skapat, men styr sökningen efter spelbarhet:
    1) först profilfilter/FAT/värde/favorit-skräll,
    2) därefter struktur som trim,
    3) mål 1850-2500 rader, helst nära 2200,
    4) historikgolv 30->29->28->27.

    Validering används som ranking, inte som hård spärr. Det är medvetet:
    tidigare budgetjakter dog runt 3200-5300 rader eftersom valideringsgolvet
    låste bort filter som behövdes för att nå spelbart radantal.
    """
    if not candidates:
        return None, {'status': 'BUDGETMOTOR2K_INGA_KANDIDATER'}, pd.DataFrame()
    try:
        raw_row_matrix = ns['_frame_row_matrix'](frame_rows, int(antal_matcher))
        sign_bits = mod._build_teckenskydd_bits(ns, raw_row_matrix, frame, int(antal_matcher))
    except Exception:
        sign_bits = tuple()

    def cat(c):
        return str((c or {}).get('category','') or '')

    def is_structure(c):
        try:
            return bool(mod._is_structure_candidate(c))
        except Exception:
            return cat(c) == 'Struktur'

    def is_profile(c):
        return not is_structure(c)

    def is_fat(c):
        return cat(c) in {'FAT', 'FAT-sekvenser'}

    def is_edge(c):
        return cat(c) in {'Värde & svårighet', 'Favorit & skräll'}

    def red(c):
        try: return float(c.get('red_pct', 0.0) or 0.0)
        except Exception: return 0.0

    def val_hit(c):
        try: return int(float(c.get('val_hit', 0) or 0))
        except Exception: return 0

    def hist_hit(c):
        try: return int(float(c.get('hist_hit', 0) or 0))
        except Exception: return 0

    # Dedupe hårt på filter+intervall. Dubbletter från RAW/B52200/REF ska inte
    # få beam-sökningen att tro att samma filter är flera alternativa vägar.
    by_sig = {}
    for c in candidates or []:
        if not isinstance(c, dict) or not c.get('key'):
            continue
        hb = int(c.get('hist_bits', 0) or 0)
        fb = int(c.get('frame_bits', 0) or 0)
        if hb <= 0 or fb <= 0:
            continue
        sig = (str(c.get('key')), str(c.get('interval_txt', c.get('interval', ''))))
        old = by_sig.get(sig)
        def keep_key(x):
            # Egen reducering först, därefter historik/validering. Detta gör att
            # riktiga profilfilter som Total Diff/FAT/Favorittryck överlever.
            return (
                red(x),
                1 if is_profile(x) else 0,
                hist_hit(x),
                val_hit(x),
                -int(x.get('frame_keep', 10**12) or 10**12),
            )
        if old is None or keep_key(c) > keep_key(old):
            by_sig[sig] = dict(c)
    cands = list(by_sig.values())
    if not cands:
        return None, {'status': 'BUDGETMOTOR2K_INGA_GILTIGA_KANDIDATER'}, pd.DataFrame()

    # Behåll svagare profilfilter också, men kasta nästan helt kosmetiska
    # strukturfilter om de inte används som sluttrim. Kandidatpoolen var full av
    # små strukturfilter med 0.2-2%, vilket drog V46 åt fel håll.
    profile_cands = [c for c in cands if is_profile(c) and red(c) >= 2.0]
    structure_cands = [c for c in cands if is_structure(c) and red(c) >= 2.0]
    # Om något viktigt profilfilter har red_pct under 2 men hög träff kan det få
    # vara med, men rankas lågt. Säkerhetsnät:
    if len(profile_cands) < 8:
        profile_cands = [c for c in cands if is_profile(c) and red(c) > 0.0]
    if not profile_cands:
        return None, {'status': 'BUDGETMOTOR2K_INGA_PROFILKANDIDATER', 'candidate_count': len(cands)}, pd.DataFrame()

    def cand_rank(c):
        # För kandidatordning: profil och egen reducering före ren träff.
        return (
            1 if is_profile(c) else 0,
            1 if is_edge(c) else 0,
            1 if is_fat(c) else 0,
            red(c),
            hist_hit(c),
            val_hit(c),
            -int(c.get('frame_keep', 10**12) or 10**12),
        )
    profile_cands.sort(key=cand_rank, reverse=True)
    structure_cands.sort(key=cand_rank, reverse=True)
    # Begränsa för fart men inte så hårt att profilen tappas.
    profile_cands = profile_cands[:120]
    structure_cands = structure_cands[:80]

    full_h = (1 << int(htot)) - 1
    full_v = (1 << int(vtot)) - 1 if int(vtot) > 0 else 0
    full_f = (1 << int(ftot)) - 1

    def counts(st):
        return int(st['hist_bits'].bit_count()), (int(st['val_bits'].bit_count()) if int(vtot)>0 else 0), int(st['frame_bits'].bit_count())

    def profile_count(st):
        return sum(1 for c in st.get('chosen', ()) if is_profile(c))
    def structure_count(st):
        return sum(1 for c in st.get('chosen', ()) if is_structure(c))
    def fat_count(st):
        return sum(1 for c in st.get('chosen', ()) if is_fat(c))
    def edge_count(st):
        return sum(1 for c in st.get('chosen', ()) if is_edge(c))

    def quality(st, hfloor, min_prof):
        hh, vv, rr = counts(st)
        pc = profile_count(st); sc = structure_count(st); fc = fat_count(st); ec = edge_count(st)
        in_band = 1 if int(min_rows) <= rr <= int(max_rows) else 0
        under = 1 if rr <= int(max_rows) else 0
        # Hård prioritet: spelbart radantal. Sedan närhet till 2200.
        # Profilkravet gör att 11 struktur + 1 favorit inte kan vinna om en
        # rimlig profilväg existerar.
        profile_ok = 1 if pc >= int(min_prof) else 0
        return (
            in_band,
            under,
            profile_ok,
            -abs(rr - int(target_rows)),
            -max(0, rr - int(max_rows)),
            -max(0, int(min_rows) - rr),
            hh,
            vv,
            pc,
            ec + fc,
            -sc,
            -len(st.get('chosen', ())),
        )

    def apply_one(st, c, hfloor):
        key = str(c.get('key',''))
        if not key or key in st['used_keys']:
            return None
        new_h = int(st['hist_bits']) & int(c.get('hist_bits', 0) or 0)
        if int(new_h.bit_count()) < int(hfloor):
            return None
        new_v = (int(st['val_bits']) & int(c.get('val_bits', 0) or 0)) if int(vtot)>0 else 0
        new_f = int(st['frame_bits']) & int(c.get('frame_bits', 0) or 0)
        new_rows = int(new_f.bit_count())
        cur_rows = int(st['frame_bits'].bit_count())
        if new_rows <= 0 or new_rows >= cur_rows:
            return None
        # Låt den gå något under bandet i sökningen, men inte döda allt.
        if new_rows < 1400:
            return None
        try:
            if sign_bits and any((new_f & int(bits)) == 0 for bits in sign_bits):
                return None
        except Exception:
            pass
        c2 = dict(c)
        c2['step_red_pct'] = 100.0 * (cur_rows - new_rows) / max(1, cur_rows)
        c2['budget_hunt_source_variant'] = str(c2.get('budget_hunt_source_variant', c2.get('variant', '')))
        step = {
            'Filter': c2.get('name',''),
            'Kategori': c2.get('category',''),
            'Intervall': c2.get('interval_txt','-'),
            'Intervallträff': f"{int(c2.get('hist_hit',0) or 0)}/{int(c2.get('hist_total', htot) or htot)}",
            'Stegreducering': f"{float(c2['step_red_pct']):.1f}%",
            'Efter filter': int(new_rows),
            'Samlad träff efter steg': f"{int(new_h.bit_count())}/{int(htot)}",
            'Samlad validering': f"{int(new_v.bit_count())}/{int(vtot)}" if int(vtot)>0 else '-',
            'Fas': 'budgetmotor2k',
        }
        return {
            'hist_bits': new_h,
            'val_bits': new_v,
            'frame_bits': new_f,
            'used_keys': frozenset(set(st['used_keys']) | {key}),
            'chosen': tuple(list(st.get('chosen', ())) + [c2]),
            'steps': tuple(list(st.get('steps', ())) + [step]),
        }

    diag_rows = []
    best_any = None

    # Kravstege: prova högt profilkrav först. Om det inte finns någon väg,
    # sänk. Detta är sista chansen men fortfarande kontrollerat.
    for hfloor in range(int(htot), int(min_hit_floor)-1, -1):
        for min_prof in [6, 5, 4, 3, 2]:
            initial = {'hist_bits': full_h, 'val_bits': full_v, 'frame_bits': full_f, 'used_keys': frozenset(), 'chosen': tuple(), 'steps': tuple()}
            states = [initial]
            archive = []
            seen = set()
            max_depth = 24
            beam_width = 420
            archive_width = 6000
            for depth in range(max_depth):
                expanded = []
                for st0 in states:
                    pc = profile_count(st0)
                    rr0 = int(st0['frame_bits'].bit_count())
                    # Profilfas: innan min_prof är uppnått får den bara ta profil.
                    # Efteråt får struktur trimma, men profilen ligger redan som bas.
                    if pc < int(min_prof):
                        pool = profile_cands
                    else:
                        # När vi är nära budget, struktur får hjälpa. Profil får alltid fortsätta.
                        pool = profile_cands + structure_cands
                    for cand in pool:
                        ns2 = apply_one(st0, cand, hfloor)
                        if ns2 is None:
                            continue
                        sig = (ns2['hist_bits'], ns2['val_bits'], ns2['frame_bits'], profile_count(ns2), structure_count(ns2))
                        if sig in seen:
                            continue
                        seen.add(sig)
                        expanded.append(ns2)
                if not expanded:
                    break
                expanded.sort(key=lambda st: quality(st, hfloor, min_prof), reverse=True)
                states = expanded[:beam_width]
                archive.extend(states)
                if len(archive) > archive_width:
                    archive.sort(key=lambda st: quality(st, hfloor, min_prof), reverse=True)
                    archive = archive[:archive_width]
            if archive:
                archive.sort(key=lambda st: quality(st, hfloor, min_prof), reverse=True)
                best_level = archive[0]
                bh, bv, br = counts(best_level)
                diag_rows.append({
                    'Variant': f'BUDGETMOTOR2K {hfloor}/{htot} prof>={min_prof}',
                    'Typ': 'Budgetmotor2K',
                    'Status': 'Diagnos',
                    'Orsak': f'profilförst, kandidater profil={len(profile_cands)}, strukturtrim={len(structure_cands)}, min_prof={min_prof}',
                    'Paketträff': f'{bh}/{htot}',
                    'Valideringsträff': f'{bv}/{vtot}' if int(vtot)>0 else '-',
                    'Paketrader': int(br),
                    'Filter totalt': int(len(best_level.get('chosen', ()))),
                    'Budgetstatus': 'JA' if int(min_rows) <= int(br) <= int(max_rows) else ('UNDER' if int(br) < int(min_rows) else 'NEJ'),
                })
                if best_any is None or quality(best_level, hfloor, min_prof) > quality(best_any, hfloor, min_prof):
                    best_any = best_level
            winners = [st for st in archive if int(min_rows) <= int(st['frame_bits'].bit_count()) <= int(max_rows) and profile_count(st) >= int(min_prof)]
            if winners:
                winners.sort(key=lambda st: quality(st, hfloor, min_prof), reverse=True)
                best = winners[0]
                bh, bv, br = counts(best)
                chosen = list(best.get('chosen', ()))
                structure_filters = sum(1 for c in chosen if is_structure(c))
                profile_filters = sum(1 for c in chosen if is_profile(c))
                fat_filters = sum(1 for c in chosen if is_fat(c))
                edge_filters = sum(1 for c in chosen if is_edge(c))
                pkg = {
                    'variant': 'BUDGETMOTOR_2K',
                    'variant_label': f'Budgetmotor 2K {bh}/{htot}',
                    'target': int(bh),
                    'target_label': f'Budgetmotor 2K {bh}/{htot}',
                    'hist_hit': int(bh), 'hist_total': int(htot),
                    'val_hit': int(bv), 'val_total': int(vtot),
                    'frame_start': int(ftot), 'frame_before': int(ftot), 'frame_after': int(br),
                    'reduction_pct': 100.0 - 100.0 * int(br) / max(1, int(ftot)),
                    'num_filters': int(len(chosen)),
                    'filters': chosen,
                    'steps': list(best.get('steps', ())),
                    'package_type': 'Budgetmotor 2K – profilförst',
                    'structure_filters': int(structure_filters),
                    'profile_filters': int(profile_filters),
                    'fat_filters': int(fat_filters),
                    'edge_filters': int(edge_filters),
                    'value_filters': int(profile_filters),
                    'v31_budget_status': 'BUDGETMOTOR2K_UNDER_2500',
                    'groups': [],
                    'meta': {
                        'engine': 'budgetmotor_2k_profile_first',
                        'source': 'same_candidate_pool_but_profile_first',
                        'hist_floor_used': int(hfloor),
                        'min_profile_used': int(min_prof),
                        'target_rows': int(target_rows),
                        'min_rows': int(min_rows),
                        'max_rows': int(max_rows),
                        'candidate_count': int(len(cands)),
                        'profile_candidates': int(len(profile_cands)),
                        'structure_candidates': int(len(structure_cands)),
                    },
                }
                meta = dict(pkg['meta'])
                meta.update({'status': 'BUDGETMOTOR2K_VALD', 'rows': int(br), 'hit': int(bh), 'val': int(bv), 'filters': int(len(chosen))})
                return pkg, meta, pd.DataFrame(diag_rows)

    meta = {'status': 'BUDGETMOTOR2K_INGET_UNDER_2500', 'candidate_count': int(len(cands)), 'profile_candidates': int(len(profile_cands)), 'structure_candidates': int(len(structure_cands)), 'target_rows': int(target_rows), 'max_rows': int(max_rows), 'frame_adapt': bool(frame_adapt)}
    if best_any is not None:
        bh, bv, br = counts(best_any)
        meta.update({'best_rows': int(br), 'best_hit': int(bh), 'best_val': int(bv), 'best_filters': int(len(best_any.get('chosen', ()))), 'best_profile_filters': int(profile_count(best_any)), 'best_structure_filters': int(structure_count(best_any))})
    return None, meta, pd.DataFrame(diag_rows)


# === v12.0cz: Kandidatdiagnos för filteruniversum ===
def _cz_norm_category(x):
    try:
        s = str(x or '').strip()
    except Exception:
        s = ''
    return s if s else 'Okänd'


def _cz_category_order():
    return ['Struktur', 'Värde & svårighet', 'FAT', 'FAT-sekvenser', 'Favorit & skräll', 'Super-Makro', 'Okänd']


def _cz_category_counts(items, specs_mode=False):
    counts = {c: 0 for c in _cz_category_order()}
    total = 0
    for it in items or []:
        if not isinstance(it, dict):
            continue
        cat = _cz_norm_category(it.get('category' if not specs_mode else 'category'))
        if cat not in counts:
            counts[cat] = 0
        counts[cat] += 1
        total += 1
    counts['Totalt'] = total
    return counts


def _cz_counts_text(counts):
    parts = []
    for cat in _cz_category_order():
        if cat in counts:
            parts.append(f"{cat}={int(counts.get(cat, 0) or 0)}")
    parts.append(f"Totalt={int(counts.get('Totalt', 0) or 0)}")
    return ' · '.join(parts)


def _cz_universe_summary_row(label, items, status='Diagnos', detail='', specs_mode=False):
    counts = _cz_category_counts(items, specs_mode=specs_mode)
    non_structure = int(counts.get('Värde & svårighet', 0) or 0) + int(counts.get('FAT', 0) or 0) + int(counts.get('FAT-sekvenser', 0) or 0) + int(counts.get('Favorit & skräll', 0) or 0) + int(counts.get('Super-Makro', 0) or 0)
    if int(counts.get('Totalt', 0) or 0) > 0 and non_structure <= 0:
        status = 'VARNING_BARA_STRUKTUR'
    return {
        'Variant': f'KANDIDATPOOL – {label}',
        'Typ': 'Kandidatdiagnos',
        'Status': status,
        'Orsak': (_cz_counts_text(counts) + (f" · {detail}" if detail else '')),
        'Paketträff': '-',
        'Valideringsträff': '-',
        'Paketrader': '',
        'Filter totalt': int(counts.get('Totalt', 0) or 0),
        'Budgetstatus': '',
        'Vald': '',
        'Gruppkandidater': '',
    }


def _cz_candidate_metric(c, key, default=0.0):
    try:
        v = c.get(key, default)
        if v is None or (hasattr(pd, 'isna') and pd.isna(v)):
            return default
        return float(v)
    except Exception:
        return default


def _cz_candidate_int(c, key, default=0):
    try:
        v = c.get(key, default)
        if v is None or (hasattr(pd, 'isna') and pd.isna(v)):
            return default
        return int(float(v))
    except Exception:
        return default


def _cz_top_candidate_rows(label, items, htot=0, vtot=0, top_per_cat=3):
    rows = []
    by_cat = {}
    for c in items or []:
        if not isinstance(c, dict):
            continue
        by_cat.setdefault(_cz_norm_category(c.get('category')), []).append(c)
    for cat in _cz_category_order():
        cand_list = by_cat.get(cat, [])
        if not cand_list:
            continue
        # Viktigt: detta är fristående kandidatstyrka, inte paketval.
        # Sortera på egen reducering först, sedan historik/validering.
        cand_list = sorted(
            cand_list,
            key=lambda c: (
                _cz_candidate_metric(c, 'red_pct', 0.0),
                _cz_candidate_int(c, 'hist_hit', 0),
                _cz_candidate_metric(c, 'val_pct', 0.0),
                -_cz_candidate_int(c, 'frame_keep', 10**12),
            ),
            reverse=True,
        )[:int(max(1, top_per_cat))]
        for i, c in enumerate(cand_list, 1):
            hh = _cz_candidate_int(c, 'hist_hit', 0)
            ht = _cz_candidate_int(c, 'hist_total', htot or 0) or int(htot or 0)
            vh = _cz_candidate_int(c, 'val_hit', -1)
            val_txt = f"{vh}/{int(vtot)}" if vh >= 0 and int(vtot or 0) > 0 else f"{_cz_candidate_metric(c, 'val_pct', 0.0):.1f}%"
            rows.append({
                'Variant': f'TOPPKANDIDAT – {label} – {cat} #{i}',
                'Typ': 'Kandidatdiagnos',
                'Status': 'Fristående filterkandidat',
                'Orsak': (
                    f"{c.get('name','')} · intervall {c.get('interval_txt', c.get('interval','-'))} · "
                    f"egen reducering {_cz_candidate_metric(c, 'red_pct', 0.0):.1f}% · "
                    f"kvar {_cz_candidate_int(c, 'frame_keep', 0)} rader · "
                    f"träff {hh}/{ht if ht else '?'} · val {val_txt} · key={c.get('key','')} · källa={c.get('budget_hunt_source_variant', c.get('variant',''))}"
                ),
                'Paketträff': f"{hh}/{ht}" if ht else '-',
                'Valideringsträff': val_txt,
                'Paketrader': _cz_candidate_int(c, 'frame_keep', 0),
                'Filter totalt': 1,
                'Budgetstatus': '',
                'Vald': '',
                'Gruppkandidater': '',
            })
    return rows


def _cz_selected_filter_rows(pkg, label='VALT PAKET'):
    rows = []
    if not isinstance(pkg, dict):
        return rows
    filters = pkg.get('filters', []) or []
    rows.append(_cz_universe_summary_row(label, [c for c in filters if isinstance(c, dict)], status='Valt paket', detail='kategoriuppdelning av valda filter'))
    for i, c in enumerate(filters, 1):
        if not isinstance(c, dict):
            continue
        rows.append({
            'Variant': f'VALT FILTER – {i:02d}',
            'Typ': 'Kandidatdiagnos',
            'Status': _cz_norm_category(c.get('category')),
            'Orsak': f"{c.get('name','')} · {c.get('interval_txt', c.get('interval','-'))} · stegreducering={c.get('step_red_pct','')} · egen red={c.get('red_pct','')} · källa={c.get('budget_hunt_source_variant', c.get('variant',''))}",
            'Paketträff': '',
            'Valideringsträff': '',
            'Paketrader': '',
            'Filter totalt': 1,
            'Budgetstatus': '',
            'Vald': '',
            'Gruppkandidater': '',
        })
    return rows

def _run_v46_final_motor_current(sim_df, specs, frame_rows, frame, filter_vec, antal_matcher, global_db=None, top_n=30, pay_min=100000, pay_max=2500000, filter_hist_target_pct=95):
    """Kör samma V46-Colabflöde för aktuell kupong/grundram, men utan facittest."""
    if int(antal_matcher) != 13:
        raise ValueError('V46-Colabmotorn är endast låst/testad för 13 matcher.')
    mod = _load_v46_engine_module()
    ns = globals()
    try:
        mod._ACTIVE_V9_NS = ns
    except Exception:
        pass

    args = _v46_final_args(top_n=top_n, pay_min=pay_min, pay_max=pay_max, filter_hist_target_pct=filter_hist_target_pct)
    if isinstance(global_db, pd.DataFrame) and not global_db.empty:
        try:
            wide_df = _similar_history_for_backtest(
                global_db, filter_vec, int(antal_matcher),
                top_n=max(int(top_n), 35), pay_min=int(pay_min), pay_max=int(pay_max),
                exclude_index=None, mode='leave-one-out', test_date=None,
            )
            if not isinstance(wide_df, pd.DataFrame) or wide_df.empty:
                wide_df = sim_df
        except Exception:
            wide_df = sim_df
    else:
        wide_df = sim_df

    # Detta är exakt samma kandidatbygge som _run_backtest_v15 i V46:
    # först en bred B5-kandidatpool, därefter variantspecifik transformering.
    base_args = mod._v15_variant_args(args, 'B5')
    global_settings = {
        'profile_min_hit': int(getattr(base_args, 'candidate_min_hit', getattr(base_args, 'min_hit', 29))),
        'min_candidate_val_pct': float(getattr(base_args, 'min_candidate_val_pct', 85.0)),
        'min_structure_val_pct': float(getattr(base_args, 'min_structure_val_pct', 95.0)),
        'min_gap_score': float(getattr(base_args, 'min_gap_score', 0.75)),
    }
    candidates, htot, vtot, ftot, hist_payout = mod._build_dynamic_candidates_v9(
        ns, specs, sim_df, frame_rows, frame, int(antal_matcher),
        profile_min_hit=int(global_settings['profile_min_hit']),
        variants_per_key=int(args.variants_per_key),
        max_candidates=int(args.max_candidates),
        validation_df=wide_df,
        min_candidate_val_pct=float(global_settings['min_candidate_val_pct']),
        min_structure_val_pct=float(global_settings['min_structure_val_pct']),
        min_gap_score=float(global_settings['min_gap_score']),
        frame_adapt=True,
    )
    candidates = mod._v13_enrich_candidates(candidates, hist_payout, htot)
    if not candidates:
        raise RuntimeError('V46-Colab kunde inte bygga några giltiga kandidater.')

    cz_diag_rows = []
    try:
        cz_diag_rows.append(_cz_universe_summary_row('SPECS I APPEN', specs, detail='alla filter som appen skapade före V46', specs_mode=True))
        cz_diag_rows.extend(_cz_top_candidate_rows('SPECS I APPEN', [], htot, vtot, top_per_cat=0))
        cz_diag_rows.append(_cz_universe_summary_row('V46 RÅKANDIDATER', candidates, detail='efter _build_dynamic_candidates_v9 + enrich'))
        cz_diag_rows.extend(_cz_top_candidate_rows('V46 RÅKANDIDATER', candidates, htot, vtot, top_per_cat=3))
    except Exception as _cz_e:
        cz_diag_rows.append({'Variant': 'KANDIDATPOOL – DIAGNOSFEL', 'Typ': 'Kandidatdiagnos', 'Status': 'FEL', 'Orsak': str(_cz_e)})

    built = {}
    errors = {}
    group_counts = {}
    raw_rows = []
    # Samlad kandidatpool för sista budgetjakten. Nu tar vi med både
    # råkandidaterna från V46-kandidatbygget och de variantspecifika
    # transformerade kandidaterna. Tidigare användes i praktiken för smal pool,
    # vilket gjorde att budgetjakten fastnade i samma 30/30-strukturpaket.
    all_budget_candidates = []
    try:
        for _bc in list(candidates or []):
            if isinstance(_bc, dict):
                _b2 = dict(_bc)
                _b2['budget_hunt_source_variant'] = 'RAW_DYNAMIC'
                all_budget_candidates.append(_b2)
    except Exception:
        pass
    for variant_id in V46_COLAB_LOCKED_VARIANTS:
        try:
            vargs = mod._v15_variant_args(args, variant_id)
            cand0 = mod._v15_filter_candidates_for_variant(candidates, vargs, htot)
            cand_v, group_cands = mod._v15_transform_candidates(cand0, htot, vtot, ftot, hist_payout, vargs, variant_id)
            group_counts[variant_id] = int(len(group_cands))
            try:
                cz_diag_rows.append(_cz_universe_summary_row(f'{variant_id} EFTER VARIANTFILTER', cand0, detail='efter _v15_filter_candidates_for_variant'))
                cz_diag_rows.append(_cz_universe_summary_row(f'{variant_id} EFTER TRANSFORM', cand_v, detail='efter _v15_transform_candidates'))
                if group_cands:
                    cz_diag_rows.append(_cz_universe_summary_row(f'{variant_id} GRUPPKANDIDATER', group_cands, detail='hårda gruppkandidater från transform'))
            except Exception:
                pass
            try:
                for _bc in list(cand_v or []) + list(group_cands or []):
                    if isinstance(_bc, dict):
                        _b2 = dict(_bc)
                        _b2['budget_hunt_source_variant'] = str(variant_id)
                        all_budget_candidates.append(_b2)
            except Exception:
                pass
            if not cand_v:
                raise RuntimeError('Inga kandidater efter V15-transformering.')
            pkg, meta = mod._build_cluster_payout_package_v13_from_candidates(
                ns, cand_v, htot, vtot, ftot, hist_payout, frame_rows, frame, int(antal_matcher), vargs, variant_id,
            )
            if pkg is None:
                raise RuntimeError((meta or {}).get('error', 'Inget paket'))
            built[variant_id] = pkg
            raw_rows.append(_v46_exact_detail_row(mod, variant_id, pkg=pkg))
        except Exception as e:
            errors[variant_id] = str(e)
            raw_rows.append(_v46_exact_detail_row(mod, variant_id, pkg=None, error=str(e)))

    detail_raw = pd.DataFrame(raw_rows)
    # Använd V46-Colabs egen syntetiska microfunktion, inte en app-omskrivning.
    try:
        detail_micro = mod._v46_make_micro_detail(detail_raw)
    except Exception as e:
        raise RuntimeError(f'V46-Colab micro-detail misslyckades: {e}')

    try:
        cz_diag_rows.append(_cz_universe_summary_row('BUDGETJAKT TOTAL POOL', all_budget_candidates, detail='råkandidater + variantspecifika kandidater + gruppkandidater'))
        cz_diag_rows.extend(_cz_top_candidate_rows('BUDGETJAKT TOTAL POOL', all_budget_candidates, htot, vtot, top_per_cat=4))
    except Exception:
        pass

    selected_pkg, micro_row, source = _select_v46_colab_primary_from_detail(detail_micro, built)
    selected_app = _normalize_v46_package_for_app(selected_pkg, V46_COLAB_PRIMARY_SYNTH, source, micro_row)

    budget_pkg = None
    budget_meta = {'status': 'BUDGETMOTOR2K_EJ_KÖRD'}
    budget_diag = pd.DataFrame()
    try:
        selected_rows_now = int(selected_app.get('frame_after', 10**12) or 10**12)
        selected_budget_status = str(selected_app.get('v31_budget_status', ''))
        # V46 är nu bara referens. Om den inte är spelbar kör vi en separat
        # Budgetmotor 2K som startar profilförst och använder struktur som trim.
        if selected_rows_now > 2500 or 'INGET_2K' in selected_budget_status.upper():
            budget_pkg, budget_meta, budget_diag = _budgetmotor2k_profile_first_current(
                mod, ns, all_budget_candidates, htot, vtot, ftot, hist_payout,
                frame_rows, frame, int(antal_matcher),
                target_rows=2200, min_rows=1850, max_rows=2500, min_hit_floor=27,
            )
            if isinstance(budget_pkg, dict) and int(budget_pkg.get('frame_after', 10**12) or 10**12) <= 2500:
                selected_app = _normalize_v46_package_for_app(budget_pkg, 'BUDGETMOTOR_2K', 'BUDGETMOTOR2K', {'Micro rescue aktiv': 'Nej', 'Micro beslut': 'V46 över 2500/inget 2k-band; Budgetmotor 2K vald.'})
                selected_app['package_type'] = 'Budgetmotor 2K – profilförst'
                source = 'BUDGETMOTOR2K'
    except Exception as _budget_e:
        budget_meta = {'status': 'BUDGETMOTOR2K_FEL', 'error': str(_budget_e)}

    # Diagnostik: visa både kandidatpool, råpaket, syntetiska microvarianter och budgetjakt.
    diag_rows = []
    try:
        diag_rows.extend(cz_diag_rows)
        diag_rows.extend(_cz_selected_filter_rows(selected_app, 'VALT PAKET EFTER V46/BUDGET'))
    except Exception as _cz_e:
        diag_rows.append({'Variant': 'KANDIDATPOOL – SAMMANSTÄLLNINGSFEL', 'Typ': 'Kandidatdiagnos', 'Status': 'FEL', 'Orsak': str(_cz_e)})
    for r in raw_rows:
        rr = dict(r)
        rr['Typ'] = 'Råpaket'
        rr['Vald'] = 'JA' if str(rr.get('Variant')) == str(source) else ''
        rr['Gruppkandidater'] = group_counts.get(str(rr.get('Variant')), 0)
        diag_rows.append(rr)
    for r in _v46_synth_diag_rows_from_detail(detail_micro):
        rr = dict(r)
        rr['Typ'] = 'Micro/syntetisk'
        rr['Vald'] = 'JA' if (str(rr.get('Variant')) == V46_COLAB_PRIMARY_SYNTH and source != 'BUDGETJAKT') else ''
        rr['Gruppkandidater'] = ''
        diag_rows.append(rr)
    if isinstance(budget_diag, pd.DataFrame) and not budget_diag.empty:
        for _, _br in budget_diag.iterrows():
            rr = _br.to_dict()
            rr['Vald'] = 'JA' if (source in ['BUDGETJAKT','BUDGETMOTOR2K'] and str(rr.get('Budgetstatus','')).upper() in ['JA','UNDER']) else ''
            rr['Gruppkandidater'] = ''
            diag_rows.append(rr)
    if isinstance(budget_meta, dict) and str(budget_meta.get('status','')) not in ['', 'BUDGETJAKT_EJ_KÖRD']:
        diag_rows.append({
            'Variant': 'V46_BUDGET_HUNT',
            'Typ': 'Budgetjakt',
            'Status': str(budget_meta.get('status','')),
            'Orsak': str(budget_meta),
            'Paketträff': f"{int(budget_meta.get('best_hit', 0) or 0)}/{int(htot)}" if budget_meta.get('best_hit') is not None else '-',
            'Valideringsträff': f"{int(budget_meta.get('best_val', 0) or 0)}/{int(vtot)}" if budget_meta.get('best_val') is not None and int(vtot)>0 else '-',
            'Paketrader': int(budget_meta.get('rows', budget_meta.get('best_rows', 0)) or 0),
            'Filter totalt': int(budget_meta.get('best_filters', 0) or 0),
            'Budgetstatus': 'JA' if str(budget_meta.get('status')) in ['BUDGETJAKT_VALD','BUDGETMOTOR2K_VALD'] else 'NEJ',
            'Vald': 'JA' if source in ['BUDGETJAKT','BUDGETMOTOR2K'] else '',
            'Gruppkandidater': '',
        })
    diag = pd.DataFrame(diag_rows)

    meta = {
        'package_engine': 'v46_exact_colab_port',
        'selected_variant': 'BUDGETMOTOR_2K' if source == 'BUDGETMOTOR2K' else ('V46_BUDGET_HUNT' if source == 'BUDGETJAKT' else V46_COLAB_PRIMARY_SYNTH),
        'selected_source': source,
        'micro_rescue_active': str(micro_row.get('Micro rescue aktiv', '')).lower() in ['ja','true','1'],
        'micro_reason': str(micro_row.get('Micro beslut', '')),
        'budget_hunt_status': str((budget_meta or {}).get('status', '')),
        'budget_hunt_meta': budget_meta,
        'candidates': int(len(candidates)),
        'budget_candidates': int(len(all_budget_candidates)),
        'hist_total': int(htot),
        'validation_total': int(vtot),
        'frame_rows': int(ftot),
        'top_n': int(top_n),
        'pay_min': int(pay_min),
        'pay_max': int(pay_max),
        'module_path': str(_v46_engine_module_path()),
        'raw_variants': ','.join(V46_COLAB_LOCKED_VARIANTS),
    }
    return selected_app, meta, diag


# Init state
for k, v in {
    'v12_analysis_ready': False,
    'v12_frame_saved': False,
    'v12_filter_saved': False,
    'v12_last_result': None,
    'v12_frame_widget_token': 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Header
st.markdown(f"""
<div class='v12-hero'>
  <div class='v12-step'>Ren omstart</div>
  <div class='v12-title'>🎯 Tipset AI — Helgardering-lik filtercentral</div>
  <div class='v12-muted'>En grundram. Ett filter per rad. Av / Tvingat / Grupp. Statistik på varje filter när du öppnar info.</div>
  <div><span class='v12-pill'>{APP_VERSION}</span><span class='v12-pill'>Manuell filtercentral</span><span class='v12-pill'>Mönstermotor2K</span></div>
</div>
""", unsafe_allow_html=True)

# Sidebar deliberately minimal
with st.sidebar:
    st.header("Kontroll")
    st.caption("Sidan är rensad. Filter och grupper styrs manuellt.")
    st.markdown("**Öppna spelfil/filterpaket**")
    uploaded_setup = st.file_uploader(
        "JSON-fil",
        type=["json"],
        key="v12_open_spelfil_filterpaket",
        help="Spelfil återställer kupong, grundram och filter. Filterpaket återställer bara filter/grupper.",
    )
    if uploaded_setup is not None:
        upload_sig = f"{getattr(uploaded_setup, 'name', '')}:{getattr(uploaded_setup, 'size', 0)}"
        if st.session_state.get('v12_last_loaded_upload_sig') != upload_sig:
            try:
                payload = _load_payload_from_uploaded_file(uploaded_setup)
                ftype = _apply_spelfil_payload(payload)
                st.session_state['v12_last_loaded_upload_sig'] = upload_sig
                st.success("Spelfil öppnad." if ftype == 'tipset_spelfil' else "Filterpaket öppnat.")
                st.rerun()
            except Exception as e:
                st.error(f"Kunde inte öppna filen: {e}")
        else:
            st.caption("Filen är redan öppnad. Ladda upp en annan fil om du vill byta.")

    sidebar_save_slot = st.empty()
    with sidebar_save_slot.container():
        st.markdown("**Spara spelfil/filterpaket**")
        st.caption("Spara blir aktivt när kupong, grundram och filtercentral är laddade.")

    st.checkbox("Visa tidsprofil", value=bool(st.session_state.get('v12_show_perf_profile', False)), key='v12_show_perf_profile', help="Visar vilka huvudsteg som tar tid vid varje sidkörning.")
    if st.button("🧹 Rensa cache", use_container_width=True):
        st.cache_data.clear()
        for _k in ['v12_specs_cache', 'v12_specs_cache_sig']:
            st.session_state.pop(_k, None)
        st.success("Cache tömd.")

# v12.0bh performance profile resets per rerun
if _perf_enabled():
    st.session_state['v12_perf_marks'] = []
    _perf_start = time.perf_counter()
else:
    _perf_start = time.perf_counter()

# Step 1 – kupongdata / historik
st.markdown("<div class='v12-card'>", unsafe_allow_html=True)
st.markdown("<div class='v12-step'>Steg 1</div><div class='v12-title'>Kupongdata och historik</div>", unsafe_allow_html=True)
col_a, col_b, col_c, col_d = st.columns([1.2, 1, 1, 1])
with col_a:
    spelform = st.selectbox("Spelform", ["Stryktips", "Europatips", "Topptips ST", "Topptips EU", "Topptips Övrigt", "Powerplay"], key="v12_spelform")
antal_matcher = 13 if spelform in ["Stryktips", "Europatips"] else 8
krav_odds = antal_matcher * 3
with col_b:
    top_n = st.number_input("Historikbas – liknande omgångar", min_value=20, max_value=100, value=30, step=5, key="v12_top_n", help="Rekommenderat: 30. 20 kan testas, men 30 ger stabilare filterstatistik.")
with col_c:
    pay_min = st.number_input("Min utdelning i historik", min_value=0, max_value=10000000, value=100000, step=50000, key="v12_pay_min")
with col_d:
    pay_max = st.number_input("Max utdelning i historik", min_value=0, max_value=50000000, value=2500000, step=50000, key="v12_pay_max")
if int(pay_max) < int(pay_min):
    st.warning("Max utdelning är lägre än min utdelning. Appen använder min som max tills du ändrar.")
    pay_max = int(pay_min)

input_text = st.text_area(
    f"Klistra in {krav_odds} procent/odds-värden",
    height=90,
    key="v12_input_text",
    placeholder="Exempel: 62 23 15 48 29 23 ..."
)
col_run, col_status = st.columns([1, 3])
with col_run:
    run_analysis = st.button("📥 Läs in kupong", use_container_width=True)
with col_status:
    if st.session_state.get('v12_analysis_ready'):
        _hist_n = len(st.session_state['v12_v_m'])
        _db_name = st.session_state.get('v12_db_name', 'databas')
        _db_total = st.session_state.get('v12_db_total_rows')
        _db_after = st.session_state.get('v12_db_after_payout_rows')
        if _db_total is not None:
            if _db_after is not None and int(_db_after) != int(_db_total):
                st.success(f"Historik klar: {_hist_n} liknande omgångar från {_db_name}. Läste igenom {int(_db_total):,} giltiga omgångar · {int(_db_after):,} kvar efter utdelningsintervall.".replace(',', ' '))
            else:
                st.success(f"Historik klar: {_hist_n} liknande omgångar från {_db_name}. Läste igenom {int(_db_total):,} giltiga omgångar i statistikfilen.".replace(',', ' '))
        else:
            st.success(f"Historik klar: {_hist_n} liknande omgångar från {_db_name}.")

if run_analysis:
    with st.spinner("Läser databas och hittar liknande omgångar..."):
        v_m, filter_vec, db_name, err = run_core_analysis(
            input_text,
            spelform,
            antal_matcher,
            krav_odds,
            True,
            int(top_n),
            True,
            int(pay_min),
            int(pay_max),
        )
    if err:
        st.error(err)
    elif v_m is None or len(v_m) == 0:
        st.error("Hittade inga historiska omgångar med valda krav.")
    else:
        st.session_state['v12_analysis_ready'] = True
        st.session_state['v12_v_m'] = v_m
        st.session_state['v12_filter_vec'] = filter_vec
        st.session_state['v12_db_name'] = db_name
        # Visa hur stor historikbas appen faktiskt läste igenom innan top-N valdes.
        try:
            _db_path_now = find_local_database(spelform)
            _db_all_now = load_database(_db_path_now, antal_matcher) if _db_path_now else pd.DataFrame()
            _db_total_now = int(len(_db_all_now))
            _db_after_now = _db_total_now
            if isinstance(_db_all_now, pd.DataFrame) and not _db_all_now.empty and 'Payout' in _db_all_now.columns:
                _db_after_now = int(len(_db_all_now[(_db_all_now['Payout'] >= int(pay_min)) & (_db_all_now['Payout'] <= int(pay_max))]))
            st.session_state['v12_db_total_rows'] = _db_total_now
            st.session_state['v12_db_after_payout_rows'] = _db_after_now
        except Exception:
            st.session_state['v12_db_total_rows'] = None
            st.session_state['v12_db_after_payout_rows'] = None
        st.session_state['v12_filter_saved'] = False
        _db_total_msg = st.session_state.get('v12_db_total_rows')
        if _db_total_msg is not None:
            st.success(f"Klart: {len(v_m)} liknande omgångar hittades. Läste igenom {int(_db_total_msg):,} giltiga omgångar i statistikfilen.".replace(',', ' '))
        else:
            st.success(f"Klart: {len(v_m)} liknande omgångar hittades.")
st.markdown("</div>", unsafe_allow_html=True)

# Streckrekommendationer används nu som vanliga intervallfilter under Favorit & skräll.

# Step 2 – ground frame
st.markdown("<div class='v12-card'>", unsafe_allow_html=True)
st.markdown("<div class='v12-step'>Steg 2</div><div class='v12-title'>Manuell grundram</div>", unsafe_allow_html=True)
st.caption("Tecknen uppdateras direkt: du ser radantalet innan du sparar grundramen. Själva filtercentralen räknas först när du sparar/kör vidare.")

if 'v12_frame_defaults' not in st.session_state or st.session_state.get('v12_frame_spelform') != spelform:
    st.session_state['v12_frame_defaults'] = [['1', 'X', '2'] for _ in range(antal_matcher)]
    st.session_state['v12_frame_spelform'] = spelform
    _sync_frame_widget_token()

_frame_token = int(st.session_state.get('v12_frame_widget_token', 0) or 0)

header_cols = st.columns([0.55, 0.8, 0.8, 0.8, 1.2])
header_cols[0].markdown("**Match**"); header_cols[1].markdown("**1**"); header_cols[2].markdown("**X**"); header_cols[3].markdown("**2**"); header_cols[4].markdown("**Val**")
frame_new = []
prev = st.session_state.get('v12_saved_frame', st.session_state['v12_frame_defaults'])
for i in range(antal_matcher):
    signs_prev = prev[i] if i < len(prev) else ['1','X','2']
    c0, c1, c2, c3, c4 = st.columns([0.55, 0.8, 0.8, 0.8, 1.2])
    c0.write(f"M{i+1}")
    b1 = c1.checkbox("", value=('1' in signs_prev), key=f"v12_frame_{_frame_token}_{i}_1")
    bx = c2.checkbox("", value=('X' in signs_prev), key=f"v12_frame_{_frame_token}_{i}_x")
    b2 = c3.checkbox("", value=('2' in signs_prev), key=f"v12_frame_{_frame_token}_{i}_2")
    signs = []
    if b1: signs.append('1')
    if bx: signs.append('X')
    if b2: signs.append('2')
    c4.write(''.join(signs) if signs else '—')
    frame_new.append(signs)

_missing_frame_matches = [f"M{i+1}" for i, signs in enumerate(frame_new) if len(signs) == 0]
if _missing_frame_matches:
    st.warning("Grundram ej komplett: " + ", ".join(_missing_frame_matches) + " saknar tecken.")
else:
    _preview_rows = frame_row_count(frame_new)
    _spikar = sum(1 for signs in frame_new if len(signs) == 1)
    _halv = sum(1 for signs in frame_new if len(signs) == 2)
    _hel = sum(1 for signs in frame_new if len(signs) == 3)
    _sign_counts = {s: sum(1 for signs in frame_new if s in signs) for s in ['1','X','2']}
    st.markdown(
        f"""
        <div class='v12-preview-grid'>
          <div class='v12-preview-card'>
            <div class='v12-preview-label'>Förhandsvisning rader</div>
            <div class='v12-preview-value'>{f'{_preview_rows:,}'.replace(',', ' ')}</div>
          </div>
          <div class='v12-preview-card'>
            <div class='v12-preview-label'>Ramtyp</div>
            <div class='v12-preview-value'>{_spikar} spik · {_hel} hel · {_halv} halv</div>
          </div>
          <div class='v12-preview-card'>
            <div class='v12-preview-label'>Markerade tecken</div>
            <div class='v12-preview-value'>1:{_sign_counts['1']} · X:{_sign_counts['X']} · 2:{_sign_counts['2']}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

save_frame = st.button("💾 Spara grundram", use_container_width=True, key=f"v12_save_frame_btn_{_frame_token}")

if save_frame:
    if any(len(x) == 0 for x in frame_new):
        st.error("Minst ett matchfält är tomt. Välj minst ett tecken per match.")
    else:
        st.session_state['v12_saved_frame'] = frame_new
        st.session_state['v12_frame_saved'] = True
        st.session_state['v12_filter_saved'] = False
        rc = frame_row_count(frame_new)
        st.success(f"Grundram sparad: {rc:,} rader.".replace(',', ' '))

if st.session_state.get('v12_frame_saved'):
    frame = st.session_state['v12_saved_frame']
    rc = frame_row_count(frame)
    st.info(f"Sparad grundram: **{rc:,} rader** · {frame_compact_string(frame)}".replace(',', ' '))
st.markdown("</div>", unsafe_allow_html=True)

# Step 3 – filtercentral
if st.session_state.get('v12_analysis_ready') and st.session_state.get('v12_frame_saved'):
    v_m = st.session_state['v12_v_m']
    filter_vec = st.session_state['v12_filter_vec']
    frame = st.session_state['v12_saved_frame']
    frame_rows, frame_n_rows, frame_ok, frame_msg = _cached_rows_from_frame(_frame_cache_tuple(frame), antal_matcher)
    if not frame_ok:
        st.error(frame_msg)
        st.stop()

    manual_sign_groups = _render_manual_sign_groups_panel(v_m, filter_vec, frame, frame_rows, antal_matcher)
    manual_sign_groups_sig = _manual_sign_groups_signature(manual_sign_groups)
    manual_hist_mask = _manual_sign_groups_hist_mask(v_m, manual_sign_groups, antal_matcher)
    manual_frame_rows = _apply_manual_sign_groups_to_rows(frame_rows, manual_sign_groups, antal_matcher)
    if manual_sign_groups and any(g.get('active') for g in manual_sign_groups):
        if len(manual_frame_rows) == 0:
            st.error('De manuella teckengrupperna lämnar inga rader i grundramen. Justera kraven innan du fortsätter.')
            st.stop()

    # Toppfavoritfiltret finns nu som fasta filter: Topp 3, 4, 5 och 6 favoriter.
    # 5 bästa skrällar använder fast maxstreck 22% enligt standardvalet.
    top_fav_count = 3  # behålls bara för bakåtkompatibla sparnycklar
    max_shock_pct = 22
    # v12.0cn: träffmål för rekommenderade filterintervall styrs per kategori.
    # Det gamla globala värdet finns kvar som fallback för gamla sparfiler, men
    # visas inte längre som en styrande global slider.
    filter_hist_target_pct_by_category = _get_filter_hist_target_pct_by_category()
    filter_hist_target_pct = int(st.session_state.get('v12_filter_hist_target_pct', 95))
    filter_hist_target_pct = max(50, min(100, filter_hist_target_pct))

    st.markdown("<div class='v12-card'>", unsafe_allow_html=True)
    st.markdown("<div class='v12-step'>Steg 3</div><div class='v12-title'>Filtercentral</div>", unsafe_allow_html=True)
    st.caption("Ett filter finns bara en gång. Välj Av, Tvingat eller Grupp. Varje filterkategori har nu eget träffmål för rekommenderat startintervall. 5 bästa skrällar använder maxstreck 22%.")

    ctrl_a, ctrl_b = st.columns([1.4, 1.0])
    with ctrl_a:
        _target_summary_txt = " · ".join([f"{cat}: {pct}%" for cat, pct in filter_hist_target_pct_by_category.items()])
        st.caption(f"Träffmål per kategori: {_target_summary_txt}")

    # Statistikfilen används för de färdiga streckfiltren under Favorit & skräll.
    _streck_db_path = find_local_database(spelform)
    _streck_hist_db = load_database(_streck_db_path, antal_matcher) if _streck_db_path else pd.DataFrame()

    # Varje intervallslider har en nyckel som innehåller kategorins träffmål.
    # Annars återanvänder Streamlit gamla slider-värden och ignorerar nya defaultintervall.
    prev_targets = st.session_state.get('v12_filter_hist_target_prev_by_category')
    if prev_targets != filter_hist_target_pct_by_category:
        if prev_targets is not None:
            for _k in list(st.session_state.keys()):
                if str(_k).startswith('filter_range_'):
                    del st.session_state[_k]
            st.session_state['v12_filter_hist_target_prev_by_category'] = dict(filter_hist_target_pct_by_category)
            st.info("Filterintervallen uppdateras efter nya kategorivisa träffmål…")
            st.rerun()
        else:
            st.session_state['v12_filter_hist_target_prev_by_category'] = dict(filter_hist_target_pct_by_category)

    # Prestandafix v12.0bc:
    # Super-Makro Total är borttaget, så de tre konkreta Super-Makrona kan visas
    # direkt igen. Rekommendationernas kvar-rad-procent beräknas fortfarande på
    # ett stabilt urval när grundramen är stor. Filtrering/reducering längre ned
    # använder hela frame_rows.
    enable_supermakro = True

    # v12.0bi: Beslutskritiska filterrekommendationer räknas på hela grundramen.
    # Prestandaoptimeringen får inte ändra vilka intervall/paket som rekommenderas.
    # Urval används bara i separata visningsdiagnoser/popovers, inte för att välja filterintervall.
    spec_candidate_rows = manual_frame_rows
    spec_sampled_for_specs = False

    _t_specs = time.perf_counter()
    _spec_sig = (
        int(antal_matcher),
        tuple(sorted((str(k), int(v)) for k, v in filter_hist_target_pct_by_category.items())),
        int(max_shock_pct),
        bool(enable_supermakro),
        tuple(round(float(x), 4) for x in list(filter_vec or [])),
        tuple(str(x) for x in list(v_m.get('Correct_Row', []))[:120]),
        int(len(v_m)),
        int(len(frame_rows or [])),
        int(len(manual_frame_rows or [])),
        manual_sign_groups_sig,
        _short_hash_items(spec_candidate_rows),
        int(len(_streck_hist_db)) if isinstance(_streck_hist_db, pd.DataFrame) else 0,
    )
    if st.session_state.get('v12_specs_cache_sig') == _spec_sig and st.session_state.get('v12_specs_cache') is not None:
        specs = st.session_state['v12_specs_cache']
        specs_from_cache = True
    else:
        specs = build_clean_filter_specs(
            v_m,
            filter_vec,
            antal_matcher,
            slider_u_count=top_fav_count,
            target_hist_pct=filter_hist_target_pct,
            u_rows=None,
            hist_df=_streck_hist_db,
            max_shock_pct=max_shock_pct,
            candidate_rows=spec_candidate_rows,
            include_supermakro=enable_supermakro,
            category_hist_target_pct=filter_hist_target_pct_by_category,
        )
        st.session_state['v12_specs_cache_sig'] = _spec_sig
        st.session_state['v12_specs_cache'] = specs
        specs_from_cache = False
    st.session_state['v12_specs'] = specs
    # v12.0co: spegla befintliga renderade filterwidgetar till persistent store
    # innan Streamlit hinner städa bort filter i kategorier som inte visas.
    _sync_rendered_filter_widgets_to_store(specs, filter_hist_target_pct, top_fav_count)
    _t_perf = _perf_mark('Bygga/läsa filterdefinitioner', _t_specs)
    cache_txt = 'cache' if specs_from_cache else 'ny beräkning'
    st.caption(f"Exakt läge: rekommenderade filters historikträff räknas mot alla liknande omgångar. Radreducering och paketens kvarvarande rader räknas på aktuell radmassa efter manuella teckengrupper ({len(manual_frame_rows):,} av {len(frame_rows):,} rader, {cache_txt}). Snabburval används endast i vissa informationsrutor/diagnoser, aldrig för slutlig filtrering eller reducering.".replace(',', ' '))
    with ctrl_b:
        if st.button("Återställ alla filter till Av", use_container_width=True, key="v12_reset_all_filters_off"):
            for _k in list(st.session_state.keys()):
                if str(_k).startswith('filter_mode_'):
                    st.session_state[_k] = 'Av'
            for _spec in specs:
                st.session_state[f"filter_mode_{_spec.get('key')}"] = 'Av'
                _filtercentral_store_set(_spec.get('key'), mode='Av', spec=_spec, target_hist_pct=filter_hist_target_pct)
            st.success("Alla filter är satta till Av.")
            st.rerun()
    st.caption("När du ändrar träffmål i en filterkategori får filtren i kategorin nya startintervall. Struktur kan t.ex. sättas till 100% medan Värde & svårighet kan ligga på 95%.")

    with st.expander("🧠 Rekommenderade filterpaket", expanded=False):
        st.caption("Mönstermotor 2K är huvudspåret: den skapar egna dynamiska regler från de 30 liknande facitraderna och kan aktiveras som ett riktigt filterpaket i nästa filtrering/rättning.")

        st.divider()
        st.markdown("**🧬 Mönstermotor 2K – valbar filtermotor**")
        st.caption("Skapar egna regler för poäng, värde, skräll, favorit, zon och struktur. Maxrader är bara ett tak: topp 3 sorteras efter högsta träffsäkerhet och därefter lägst radantal/bäst reducering. Grundramsanpassning kan mjuka upp regler som ligger för nära veckans ram.")
        c_pm_a, c_pm_b, c_pm_c, c_pm_d = st.columns([1, 1, 1, 1])
        with c_pm_a:
            pm_target_rows = st.number_input("Mål rader (diagnos)", min_value=1000, max_value=6000, value=int(st.session_state.get('v12_pm2k_target_rows', 2200)), step=50, key='v12_pm2k_target_rows')
        with c_pm_b:
            pm_min_rows = st.number_input("Min spelbara rader", min_value=500, max_value=5000, value=int(st.session_state.get('v12_pm2k_min_rows', 1850)), step=50, key='v12_pm2k_min_rows')
        with c_pm_c:
            pm_max_rows = st.number_input("Max spelbara rader", min_value=1000, max_value=8000, value=int(st.session_state.get('v12_pm2k_max_rows', 2500)), step=50, key='v12_pm2k_max_rows')
        with c_pm_d:
            pm_frame_adapt = st.checkbox(
                "Anpassa mot grundram",
                value=bool(st.session_state.get('v12_pm2k_frame_adapt', True)),
                key='v12_pm2k_frame_adapt',
                help="Mjukar upp dynamiska mönsterintervall som ligger nära grundramens ytterkanter, t.ex. strecksumma första 6 när ramen redan är toppstyrd av spikar/halvor.",
            )
        run_pm2k = st.button("🧬 Beräkna Mönstermotor 2K", use_container_width=True, key="v12_run_pm2k_diag")
        if run_pm2k:
            try:
                pm_progress = st.progress(0, text="Startar Mönstermotor 2K...")
                pm_status = st.empty()
                pm_start = time.time()
                def _pm_progress(pct, label):
                    try:
                        pct = int(max(0, min(100, pct)))
                        elapsed = time.time() - pm_start
                        eta_txt = ""
                        if pct > 3 and pct < 100:
                            eta = elapsed * (100 - pct) / max(1, pct)
                            eta_txt = f" · beräknas klar om ca {_fmt_elapsed(eta)}"
                        pm_progress.progress(pct, text=f"{label} · {pct}%{eta_txt}")
                        pm_status.caption(f"Förfluten tid: {_fmt_elapsed(elapsed)}{eta_txt}")
                    except Exception:
                        pass
                pm_chosen, pm_meta, pm_diag, pm_feat_diag = _pm2k_search_package(
                    v_m, manual_frame_rows, filter_vec, int(antal_matcher),
                    target_rows=int(pm_target_rows), min_rows=int(pm_min_rows), max_rows=int(pm_max_rows), min_hit_floor=27, top_n=3,
                    frame_adapt=bool(pm_frame_adapt),
                    progress_cb=_pm_progress,
                )
                pm_progress.progress(100, text=f"Mönstermotor 2K klar på {_fmt_elapsed(time.time() - pm_start)}")
                st.session_state['v12_pm2k_meta'] = pm_meta
                st.session_state['v12_pm2k_diag'] = pm_diag
                st.session_state['v12_pm2k_feature_diag'] = pm_feat_diag
                st.session_state['v12_pm2k_chosen'] = pm_chosen
                st.session_state['v12_pm2k_options'] = (pm_chosen.get('_pm2k_alternatives') if isinstance(pm_chosen, dict) else []) or ([pm_chosen] if isinstance(pm_chosen, dict) else [])
                st.session_state['v12_pm2k_option_idx'] = 0
                st.session_state['v12_pm2k_rules'] = _pm2k_rules_to_rows(pm_chosen, manual_frame_rows, filter_vec, int(antal_matcher))
                status = str(pm_meta.get('status',''))
                if status == 'MÖNSTERMOTOR2K_VALD':
                    st.success(f"Mönstermotor hittade paket: {int(pm_meta.get('rows',0)):,} rader · {int(pm_meta.get('hit',0))}/{int(pm_meta.get('total',0))} träff. Du kan aktivera paketet med knappen nedanför.".replace(',', ' '))
                else:
                    st.warning(f"Mönstermotor hittade inget under 2 500. Bästa info: {pm_meta}")
            except Exception as e:
                st.error(f"Mönstermotor kunde inte köras: {e}")
                with st.expander("Visa tekniskt fel", expanded=False):
                    st.code(traceback.format_exc(), language="text")
        pm_diag_show = st.session_state.get('v12_pm2k_diag')
        pm_rules_show = st.session_state.get('v12_pm2k_rules')
        pm_feat_show = st.session_state.get('v12_pm2k_feature_diag')
        pm_meta_show = st.session_state.get('v12_pm2k_meta') or {}
        pm_options_show = st.session_state.get('v12_pm2k_options') or []
        if str(pm_meta_show.get('status','')) == 'MÖNSTERMOTOR2K_VALD' and pm_options_show:
            opt_df = _pm2k_options_to_df(pm_options_show)
            if isinstance(opt_df, pd.DataFrame) and not opt_df.empty:
                st.caption("Topp 3 spelbara Mönstermotor2K-alternativ – sorterat på högst träff och därefter lägst radantal")
                st.dataframe(opt_df, use_container_width=True, hide_index=True)
            def _fmt_pm_opt(i):
                try:
                    summ = pm_options_show[int(i)].get('_pm2k_option_summary', {})
                    return f"Alternativ {int(i)+1}: {int(summ.get('rows',0))} rader · {int(summ.get('hit',0))}/{int(summ.get('total',0))} · {int(summ.get('filters',0))} regler"
                except Exception:
                    return f"Alternativ {int(i)+1}"
            opt_idx = st.radio(
                "Välj Mönstermotor2K-alternativ att aktivera",
                options=list(range(len(pm_options_show))),
                format_func=_fmt_pm_opt,
                horizontal=False,
                key='v12_pm2k_option_idx',
            )
            try:
                selected_pm_option = pm_options_show[int(opt_idx)]
                st.session_state['v12_pm2k_chosen'] = selected_pm_option
                st.session_state['v12_pm2k_meta'] = _pm2k_meta_from_chosen(selected_pm_option, pm_meta_show)
                st.session_state['v12_pm2k_rules'] = _pm2k_rules_to_rows(selected_pm_option, manual_frame_rows, filter_vec, int(antal_matcher))
                pm_meta_show = st.session_state.get('v12_pm2k_meta') or pm_meta_show
            except Exception:
                selected_pm_option = st.session_state.get('v12_pm2k_chosen')
            c_use_pm, c_clear_pm = st.columns([2, 1])
            with c_use_pm:
                if st.button('✅ Använd valt Mönstermotor2K-paket i nästa filtrering', use_container_width=True, key='v12_use_pm2k_as_filter'):
                    st.session_state['v12_pm2k_active'] = True
                    st.session_state['v12_last_result_stale'] = True
                    st.success(_pm2k_active_label(st.session_state.get('v12_pm2k_meta') or pm_meta_show))
            with c_clear_pm:
                if st.button('Stäng av Mönstermotor2K', use_container_width=True, key='v12_clear_pm2k_as_filter'):
                    st.session_state['v12_pm2k_active'] = False
                    st.session_state['v12_last_result_stale'] = True
                    st.info('Mönstermotor2K är avstängd.')
        if bool(st.session_state.get('v12_pm2k_active')):
            st.success(_pm2k_active_label(st.session_state.get('v12_pm2k_meta') or {}))

        # Uppdatera lokala visningsobjekt efter att användaren bytt toppalternativ.
        pm_rules_show = st.session_state.get('v12_pm2k_rules')
        pm_meta_show = st.session_state.get('v12_pm2k_meta') or pm_meta_show

        if isinstance(pm_diag_show, pd.DataFrame) and not pm_diag_show.empty:
            st.caption("Mönstermotor 2K – sökresultat per träffgolv")
            st.dataframe(pm_diag_show, use_container_width=True, hide_index=True)
            st.caption(str(pm_meta_show))
        if isinstance(pm_rules_show, pd.DataFrame) and not pm_rules_show.empty:
            st.caption("Valda egna regler i mönsterpaketet")
            st.dataframe(pm_rules_show, use_container_width=True, hide_index=True)
        if isinstance(pm_feat_show, pd.DataFrame) and not pm_feat_show.empty:
            with st.expander("Visa toppfeatures som mönstermotorn skapade", expanded=False):
                st.dataframe(pm_feat_show.head(80), use_container_width=True, hide_index=True)

        with st.expander("Välj filter som måste ingå i rekommenderade paket", expanded=False):
            st.caption("Kryssa i filter du vill att paketmotorn ska använda. Ändringar ligger i ett formulär och sparas först när du trycker på knappen, så sidan laddar inte om för varje kryss.")
            saved_required_keys = set(st.session_state.get('v12_required_pkg_keys', []))
            required_keys_draft = []
            with st.form("v12_required_pkg_form"):
                req_cats = _ordered_categories_from_specs(specs)
                for _cat in req_cats:
                    st.markdown(f"**{_cat}**")
                    _cat_specs = [_s for _s in specs if _s.get('category') == _cat]
                    _cols = st.columns(3)
                    for _i, _spec in enumerate(_cat_specs):
                        with _cols[_i % 3]:
                            _key = _spec.get('key')
                            if st.checkbox(_spec.get('name',''), value=(_key in saved_required_keys), key=f"v12_reqpkg_form_{_key}"):
                                required_keys_draft.append(_key)
                req_save = st.form_submit_button("Spara obligatoriska filter", use_container_width=True)
            if req_save:
                st.session_state['v12_required_pkg_keys'] = list(dict.fromkeys(required_keys_draft))
                # Paket som redan är beräknade bygger på gamla obligatoriska val.
                # Rensa dem så användaren inte råkar applicera ett paket där de
                # nya "måste ingå"-filtren saknas.
                for _k in ['v12_recommended_packages', 'v12_recommended_candidate_audit', 'v12_recommended_meta', 'v12_applied_package_meta', 'v12_applied_package_snapshot']:
                    st.session_state.pop(_k, None)
                st.success(f"{len(required_keys_draft)} obligatoriska filter sparade. Beräkna paket igen.")
            required_keys_saved = st.session_state.get('v12_required_pkg_keys', [])
            if required_keys_saved:
                st.success(f"{len(required_keys_saved)} filter är markerade som måste ingå i paketmotorn.")
            else:
                st.info("Inga obligatoriska filter valda. Paketmotorn söker fritt.")
        required_keys_now = list(st.session_state.get('v12_required_pkg_keys', []))

        # Standard för paketmotorn. Dessa värden gäller bara när inget redan är sparat
        # i session/spelfil. För att uppdatera gamla öppna sessioner byts det gamla
        # standardvärdet 22/30 till nya 28/30 en gång, men egna val lämnas i fred.
        default_rec_min_step = 1.0
        default_rec_max_filters = 30
        default_rec_min_hit = min(28, int(len(v_m)))
        default_rec_display_max_rows = int(st.session_state.get("v12_matrix_limit", 5000))
        default_rec_frame_adapt = True
        default_rec_min_value_filters = 3
        if st.session_state.get("v12_pkg_defaults_version") != "v12.0bw":
            if "v12_rec_min_hit" not in st.session_state or int(st.session_state.get("v12_rec_min_hit", 22)) <= 22:
                st.session_state["v12_rec_min_hit"] = default_rec_min_hit
            st.session_state.setdefault("v12_rec_min_step", default_rec_min_step)
            st.session_state.setdefault("v12_rec_max_filters", default_rec_max_filters)
            st.session_state.setdefault("v12_rec_display_max_rows", default_rec_display_max_rows)
            st.session_state.setdefault("v12_rec_frame_adapt", default_rec_frame_adapt)
            st.session_state.setdefault("v12_rec_min_value_filters", default_rec_min_value_filters)
            st.session_state["v12_pkg_defaults_version"] = "v12.0bw"

        with st.form("v12_recommended_package_engine_form"):
            st.caption("Paketmotorns standard är nu: sök ner till 28/30, max 30 filter, minsta extra reducering 1,00, anpassa mot grundram och minst 3 värde-/poängfilter. Ändra flera saker och tryck Beräkna paket en gång.")
            rp_c1, rp_c2, rp_c3, rp_c4, rp_c5, rp_c6 = st.columns([1, 1, 1, 1, 1, 1])
            with rp_c1:
                rec_min_step = st.number_input(
                    "Minsta extra reducering",
                    min_value=0.5,
                    max_value=20.0,
                    value=float(st.session_state.get("v12_rec_min_step", default_rec_min_step)),
                    step=0.25,
                    format="%.2f",
                    key="v12_rec_min_step",
                    help="Gäller främst efter att värdekärnan är byggd. Värde-/poängfilter som behövs för kvoten får läggas till med lägre marginalkrav om de håller samlad träff.",
                )
            with rp_c2:
                rec_max_filters = st.number_input("Max filter i paket", min_value=1, max_value=40, value=int(st.session_state.get("v12_rec_max_filters", default_rec_max_filters)), step=1, key="v12_rec_max_filters")
            with rp_c3:
                rec_min_hit = st.number_input("Sök ner till träff", min_value=1, max_value=int(len(v_m)), value=min(int(st.session_state.get("v12_rec_min_hit", default_rec_min_hit)), int(len(v_m))), step=1, key="v12_rec_min_hit", help="Standard 28/30. Höj till 29–30 för tryggare paket, sänk bara om du behöver pressa radantalet hårdare.")
            with rp_c4:
                rec_display_max_rows = st.number_input("Visa paket under", min_value=100, max_value=75000, value=int(st.session_state.get("v12_rec_display_max_rows", default_rec_display_max_rows)), step=100, key="v12_rec_display_max_rows", help="Sätt detta till din riktiga radbudget före TipsetMatrix. Standard följer TipsetMatrix-spärren om den finns, annars 5 000.")
            with rp_c5:
                rec_frame_adapt = st.checkbox(
                    "Anpassa mot grundram",
                    value=bool(st.session_state.get("v12_rec_frame_adapt", default_rec_frame_adapt)),
                    key="v12_rec_frame_adapt",
                    help="Undviker paketfilter som ligger för nära grundramens yttergränser. Gäller även FAT-sekvenser som blir grundramsdrivna, t.ex. när ramen redan låser minst 3 sekvensträffar och filtret bara kapar max.",
                )
            with rp_c6:
                rec_min_value_filters = st.number_input(
                    "Min värde-/poängfilter",
                    min_value=0,
                    max_value=10,
                    value=int(st.session_state.get("v12_rec_min_value_filters", default_rec_min_value_filters)),
                    step=1,
                    key="v12_rec_min_value_filters",
                    help="Hård spärr: ett rekommenderat paket måste innehålla minst detta antal filter från Värde & svårighet. Annars visas paketet inte i listan.",
                )
            build_recs = st.form_submit_button("Beräkna paket", use_container_width=True)
        if build_recs:
            start_clock = time.time()
            progress_bar = st.progress(0, text="Startar paketmotor...")
            progress_status = st.empty()
            progress_best = st.empty()

            def _ui_package_progress(done, total, label, best=None):
                try:
                    done_i = int(max(0, done)); total_i = int(max(1, total))
                    pct = int(max(0, min(100, round(100 * done_i / total_i))))
                    elapsed = time.time() - start_clock
                    if done_i > 0 and pct > 0:
                        eta = elapsed * (total_i - done_i) / max(1, done_i)
                        eta_txt = _fmt_elapsed(eta)
                    else:
                        eta_txt = "beräknas..."
                    progress_bar.progress(pct, text=f"{label} · {pct}%")
                    progress_status.caption(f"Förfluten tid: {_fmt_elapsed(elapsed)} · uppskattad kvar: {eta_txt}")
                    if isinstance(best, dict) and best:
                        progress_best.info(f"Bästa hittills: {int(best.get('hit', 0))}/{int(best.get('total', 0))} · {int(best.get('rows', 0)):,} rader".replace(',', ' '))
                except Exception:
                    pass

            rec_result = _build_recommended_filter_packages(
                v_m, specs, manual_frame_rows, frame, antal_matcher,
                min_step_reduction_pct=float(rec_min_step),
                max_filters=int(rec_max_filters),
                min_hit_count=int(rec_min_hit),
                frame_adapt=bool(rec_frame_adapt),
                min_value_filters=int(rec_min_value_filters),
                required_keys=required_keys_now,
                target_frame_after=int(rec_display_max_rows),
                progress_cb=_ui_package_progress,
                manual_hist_mask=None,
            )
            if isinstance(rec_result, tuple):
                packages, candidate_audit = rec_result
            else:
                packages, candidate_audit = rec_result, pd.DataFrame()
            elapsed_total = time.time() - start_clock
            progress_bar.progress(100, text=f"Klar på {_fmt_elapsed(elapsed_total)}")
            progress_status.success(f"Paketberäkningen är klar. Total tid: {_fmt_elapsed(elapsed_total)}.")
            st.session_state['v12_recommended_packages'] = packages
            st.session_state['v12_recommended_candidate_audit'] = candidate_audit
            st.session_state['v12_recommended_meta'] = {
                'package_engine': 'pareto_multilevel_progress_posttrim_group_visible_diagnostics_filter_hist_independent_of_manual_groups',
                'manual_hist_target_pct': int(filter_hist_target_pct),
                'top_fav_filters': 'Topp 3/4/5/6',
                'frame_rows': int(len(manual_frame_rows)),
                'frame_rows_before_manual': int(len(frame_rows)),
                'manual_sign_groups_sig': manual_sign_groups_sig,
                'manual_sign_groups_active': int(sum(1 for g in manual_sign_groups if g.get('active'))),
                'min_step': float(rec_min_step),
                'max_filters': int(rec_max_filters),
                'min_hit': int(rec_min_hit),
                'display_max_rows': int(rec_display_max_rows),
                'frame_adapt': bool(rec_frame_adapt),
                'min_value_filters': int(rec_min_value_filters),
                'required_keys': list(required_keys_now),
            }
        packages = st.session_state.get('v12_recommended_packages') or []
        _pkg_meta = st.session_state.get('v12_recommended_meta') or {}
        if packages and _pkg_meta.get('manual_sign_groups_sig') != manual_sign_groups_sig:
            st.warning('Manuella teckengrupper har ändrats sedan paketen beräknades. Beräkna rekommenderade paket igen innan du använder ett paket.')
            packages = []
        rec_display_max_rows = int(st.session_state.get('v12_rec_display_max_rows', 5000))
        visible_packages = [p for p in packages if int(p.get('frame_after', 10**12)) <= rec_display_max_rows]
        hidden_packages = [p for p in packages if int(p.get('frame_after', 10**12)) > rec_display_max_rows]
        visible_package_index_map = {id(p): i for i, p in enumerate(visible_packages, 1)}
        hidden_package_index_map = {id(p): i for i, p in enumerate(hidden_packages, 1)}
        if packages:
            st.caption(f"Visar bara paket där filtermassan är högst {rec_display_max_rows:,} rader. {len(hidden_packages)} paket är dolda över gränsen.".replace(',', ' '))
            if visible_packages:
                top_packages = _top_playable_packages(visible_packages, 3)
                st.markdown("**3 högst spelvärde**")
                st.caption("Spelvärde premierar effektiv reducering. Det är inte samma sak som tryggast paket. Alla spelbara paket under radgränsen kan väljas i dropdownen nedanför.")
                st.dataframe(_style_spelvarde_df(_recommended_packages_summary_df(top_packages, visible_package_index_map)), use_container_width=True, hide_index=True)
                st.caption(_spelvarde_caption())

                # v12.0ae: visa grupppaketsektionen alltid, även om inga grupppaket
                # ligger under radgränsen. Annars ser funktionen ut att saknas.
                st.markdown("**Bästa hårda grupppaket**")
                st.caption("Visas separat eftersom grupppaket är en annan riskprofil: fler filter kan ingå men alla måste inte sitta samtidigt. Sektionen visas även när bästa grupppaket ligger över radgränsen.")
                group_status_df = _group_packages_status_df(packages, rec_display_max_rows, max_rows=6)
                if not group_status_df.empty:
                    st.dataframe(_style_spelvarde_df(group_status_df), use_container_width=True, hide_index=True)
                    st.caption(_group_packages_status_text(packages, rec_display_max_rows))
                else:
                    st.info(_group_packages_status_text(packages, rec_display_max_rows))

                best_group_packages = _top_playable_packages([p for p in visible_packages if _is_hard_group_package(p)], 3)
                rest_packages = [p for p in visible_packages if p not in top_packages and p not in best_group_packages]
                if rest_packages:
                    with st.expander("Visa övriga spelbara paket under radgränsen", expanded=False):
                        st.caption("Dessa paket är också valbara i dropdownen nedanför. De kan vara tryggare även om spelvärdepoängen är lägre.")
                        st.dataframe(_style_spelvarde_df(_recommended_packages_summary_df(rest_packages, visible_package_index_map)), use_container_width=True, hide_index=True)
            else:
                top_packages = []
                st.warning("Inga paket hamnade under vald radgräns. Visar därför bästa paketet över gränsen som referens så att listan inte blir tom.")
                if hidden_packages:
                    best_over = _top_playable_packages(hidden_packages, 1)
                    if best_over:
                        st.markdown("**Bästa paket över radgränsen**")
                        st.dataframe(_style_spelvarde_df(_recommended_packages_summary_df(best_over, hidden_package_index_map)), use_container_width=True, hide_index=True)
                        st.caption(_spelvarde_caption())
                        st.caption("Detta paket klarar dina krav bäst men lämnar fler rader än vald gräns. Höj radgränsen eller kryssa i fler/lämpligare filter om du vill pressa vidare.")
                st.markdown("**Bästa hårda grupppaket**")
                st.caption("Sektionen visas även när inga grupppaket är spelbara under radgränsen, så du ser om gruppmotorn faktiskt byggde något.")
                group_status_df = _group_packages_status_df(packages, rec_display_max_rows, max_rows=6)
                if not group_status_df.empty:
                    st.dataframe(_style_spelvarde_df(group_status_df), use_container_width=True, hide_index=True)
                    st.caption(_group_packages_status_text(packages, rec_display_max_rows))
                else:
                    st.info(_group_packages_status_text(packages, rec_display_max_rows))
            if hidden_packages:
                with st.expander("Visa bästa dolda paket över radgränsen", expanded=False):
                    hidden_ref = _hidden_packages_reference_df(packages, rec_display_max_rows)
                    if not hidden_ref.empty:
                        st.dataframe(hidden_ref, use_container_width=True, hide_index=True)
                    st.caption("Dessa paket har ofta högre träffbild men är för breda för huvudlistan eftersom de lämnar för många rader före TipsetMatrix.")
            candidate_audit = st.session_state.get('v12_recommended_candidate_audit')
            if isinstance(candidate_audit, pd.DataFrame) and not candidate_audit.empty:
                with st.expander('Visa kandidatanalys – alla filter som paketmotorn testade', expanded=False):
                    st.caption('Här ser du hela nivåtrappan per filter: säkert, balanserat, aggressivt och valt intervall i paket. Det gör det lättare att se varför t.ex. FAT Summa valdes på en viss nivå.')
                    st.dataframe(candidate_audit, use_container_width=True, hide_index=True)
            with st.expander('🧪 Backtest av paketmotorn', expanded=False):
                st.caption('Tungt men ärligt test: varje historisk omgång låtsas vara dagens kupong. Testomgången tas bort från sin egen liknande historik. Standardläget testar paketfiltren – inte dagens specifika grundram.')
                st.warning('Backtest kör paketmotorn flera gånger och kan ta tid. Börja med 3–5 omgångar. Det används ingen sampling i själva paketbeslutet.')
                bt_c1, bt_c2, bt_c3 = st.columns([1, 1, 1])
                with bt_c1:
                    bt_cases = st.number_input('Antal omgångar att testa', min_value=3, max_value=50, value=int(st.session_state.get('v12_bt_cases', 5)), step=1, key='v12_bt_cases', help='Backtestar de senaste giltiga historiska omgångarna efter utdelningskravet. Högre antal kan bli mycket långsamt.')
                with bt_c2:
                    bt_mode = st.selectbox('Historikläge', ['Leave-one-out', 'Kronologiskt'], index=0 if st.session_state.get('v12_bt_mode', 'Leave-one-out') == 'Leave-one-out' else 1, key='v12_bt_mode', help='Leave-one-out använder alla historiska omgångar utom testomgången. Kronologiskt använder bara omgångar före testdatumet, vilket är striktare men kan ge färre testfall.')
                with bt_c3:
                    bt_require_budget = st.checkbox('Välj paket under radgräns', value=bool(st.session_state.get('v12_bt_require_budget', True)), key='v12_bt_require_budget', help='På: backtestet väljer högsta träffpaket under Visa paket under. Av: kan välja bästa paket även över gränsen.')
                bt_scope = st.radio(
                    'Backtestmått',
                    ['Paketfilter endast', 'Fullt flöde med aktuell grundram'],
                    index=0 if st.session_state.get('v12_bt_scope', 'Paketfilter endast') == 'Paketfilter endast' else 1,
                    key='v12_bt_scope',
                    horizontal=True,
                    help='Paketfilter endast är rekommenderat: historiska facit testas mot paketets filter, inte mot dagens specifika grundram/manuella tecken. Fullt flöde är bara felsökningsdiagnos för aktuell kupong.'
                )
                run_bt = st.button('Kör backtest av paketmotorn', use_container_width=True, key='v12_run_package_backtest')
                if run_bt:
                    db_path_bt = find_local_database(spelform)
                    if not db_path_bt:
                        st.error('Hittade ingen historikfil för backtest.')
                    else:
                        db_bt = load_database(db_path_bt, antal_matcher)
                        bt_progress = st.progress(0, text='Startar backtest...')
                        bt_status = st.empty()
                        bt_start = time.time()
                        def _bt_progress(done, total, label):
                            try:
                                pct = int(max(0, min(100, round(100 * int(done) / max(1, int(total))))))
                                bt_progress.progress(pct, text=f'{label} · {pct}%')
                                elapsed = time.time() - bt_start
                                bt_status.caption(f'Förfluten tid: {_fmt_elapsed(elapsed)}')
                            except Exception:
                                pass
                        bt_display_rows = int(rec_display_max_rows) if bool(bt_require_budget) else 10**12
                        bt_settings = {
                            'min_step': float(st.session_state.get('v12_rec_min_step', rec_min_step)),
                            'max_filters': int(st.session_state.get('v12_rec_max_filters', rec_max_filters)),
                            'min_hit': int(st.session_state.get('v12_rec_min_hit', rec_min_hit)),
                            'display_max_rows': int(bt_display_rows),
                            'frame_adapt': bool(st.session_state.get('v12_rec_frame_adapt', rec_frame_adapt)),
                            'min_value_filters': int(st.session_state.get('v12_rec_min_value_filters', rec_min_value_filters)),
                        }
                        bt_df, bt_meta = _run_package_engine_backtest(
                            db_bt,
                            frame,
                            manual_sign_groups,
                            int(antal_matcher),
                            int(top_n),
                            int(pay_min),
                            int(pay_max),
                            int(filter_hist_target_pct),
                            bt_settings,
                            required_keys=required_keys_now,
                            max_tests=int(bt_cases),
                            mode='kronologiskt' if str(bt_mode).lower().startswith('krono') else 'leave-one-out',
                            progress_cb=_bt_progress,
                            test_scope='current_full_flow' if str(bt_scope).startswith('Fullt') else 'package_only',
                        )
                        bt_progress.progress(100, text=f'Backtest klart på {_fmt_elapsed(time.time() - bt_start)}')
                        st.session_state['v12_package_backtest_df'] = bt_df
                        st.session_state['v12_package_backtest_meta'] = bt_meta
                bt_df_show = st.session_state.get('v12_package_backtest_df')
                bt_meta_show = st.session_state.get('v12_package_backtest_meta') or {}
                if isinstance(bt_df_show, pd.DataFrame) and not bt_df_show.empty:
                    if bt_meta_show.get('error'):
                        st.error(bt_meta_show.get('error'))
                    tested = int(bt_meta_show.get('tested', 0) or 0)
                    if tested > 0:
                        package_hits = int(bt_meta_show.get('package_hits', 0) or 0)
                        main_hits = int(bt_meta_show.get('survivors', 0) or 0)
                        scope_now = str(bt_meta_show.get('test_scope', 'package_only'))
                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric('Testade paketfall', f"{tested}/{int(bt_meta_show.get('requested', 0) or 0)}")
                        b2.metric('Facit klarar paket', f"{package_hits}/{tested}")
                        b3.metric('Facit i backtestram', f"{int(bt_meta_show.get('ground_hits', 0))}/{tested}")
                        b4.metric('Huvudresultat', f"{main_hits}/{tested}")
                        main_pct = 100.0 * main_hits / max(1, tested)
                        if scope_now == 'package_only':
                            st.info(f"Backtestresultat: {package_hits}/{tested} ({100.0 * package_hits / max(1, tested):.1f}%) historiska facit klarade valt rekommenderat paket. Dagens grundram och manuella teckengrupper ingår inte i huvudmåttet.")
                        else:
                            st.info(f"Backtestresultat: {main_hits}/{tested} ({main_pct:.1f}%) historiska facit överlevde aktuell grundram + manuell teckengrupp + valt rekommenderat paket. Detta läge är bara diagnostiskt för dagens kupong.")
                    st.dataframe(bt_df_show, use_container_width=True, hide_index=True)
                    st.caption('Standardläget testar paketfiltren. Backtestramen är auto-rankad efter testkupongens streck och samma breddmönster som din grundram, så gamla facit testas inte mot dagens specifika matchtecken.')

            if visible_packages:
                # v12.0bu: alla spelbara paket under radgränsen ska kunna väljas,
                # inte bara topp 3 på spelvärde och bästa grupppaket. Annars kan
                # tryggare 30/30- eller 29/30-paket synas i "övriga" men inte gå att använda.
                selectable_packages = _dedupe_package_list(visible_packages)
            else:
                _sel_base = _top_playable_packages(hidden_packages, 1)
                _sel_groups = _top_playable_packages([p for p in hidden_packages if _is_hard_group_package(p)], 2)
                selectable_packages = _dedupe_package_list(_sel_base + _sel_groups)
            if not visible_packages and selectable_packages:
                st.info("Du kan använda bästa paketet över radgränsen, men filtermassan blir bredare än din valda gräns.")
            apply_selected_package = False
            if not selectable_packages:
                sel_pkg = None
            else:
                _label_index_map = visible_package_index_map if visible_packages else hidden_package_index_map
                def _pkg_select_label(_p):
                    try:
                        _pno = int(_label_index_map.get(id(_p), 0))
                        _prefix = f"P{_pno} · " if _pno else ""
                    except Exception:
                        _prefix = ""
                    return (f"{_prefix}{_p['hist_hit']}/{_p['hist_total']} · "
                            f"{_p['frame_start']:,}→{_p['frame_after']:,} · "
                            f"{_p['reduction_pct']:.1f}% · spelvärde {_package_value_score(_p):.0f} · "
                            f"{_p['num_filters']} filter · {_p.get('value_filters',0)} värde · "
                            f"{_p.get('package_type','Tvingat')}").replace(',', ' ')
                labels = [_pkg_select_label(p) for p in selectable_packages]
                with st.form("v12_select_recommended_package_form"):
                    pick_idx = st.selectbox("Välj paket att använda", list(range(len(selectable_packages))), format_func=lambda i: labels[i], key="v12_recommended_pick")
                    apply_selected_package = st.form_submit_button("✅ Använd valt paket i filtercentralen", use_container_width=True)
                if visible_packages:
                    st.caption("Dropdownen innehåller alla paket under radgränsen, även de som visas under övriga spelbara paket.")
                sel_pkg = selectable_packages[int(pick_idx)]
            if sel_pkg is not None:
                with st.expander("Visa filter och intervall i valt paket", expanded=False):
                    if sel_pkg.get('steps'):
                        st.dataframe(pd.DataFrame(sel_pkg['steps']), use_container_width=True, hide_index=True)
                        st.caption("Intervallträff är hur många av de liknande historiska omgångarna det enskilda filterintervallet klarar. Samlad träff efter steg är hela paketets träff efter att filtret lagts till.")
                        if sel_pkg.get('post_trim_notes'):
                            st.markdown("**Eftertrimning utan tappad samlad träff**")
                            st.dataframe(pd.DataFrame(sel_pkg.get('post_trim_notes') or []), use_container_width=True, hide_index=True)
                            st.caption("Eftertrim betyder att appen hittade ett snävare intervall som gav färre rader men behöll samma samlade historikträff. Kombinationslyft betyder att två små filter tillsammans gav ett tydligt extra steg trots att de var svaga var för sig.")
                    else:
                        st.info("Detta paket hittade inga filter som gav tillräcklig extra reducering inom träffmålet.")
                if apply_selected_package:
                    _apply_recommended_package_to_session(sel_pkg, specs, filter_hist_target_pct, top_fav_count)
                    st.session_state['v12_last_result_stale'] = True
                    st.success("Paketet har lagts in i filtercentralen. Kontrollera filtercentralen nedanför och kör filtrering när du är nöjd.")
        else:
            candidate_audit = st.session_state.get('v12_recommended_candidate_audit')
            if st.session_state.get('v12_recommended_meta'):
                st.warning("Paketmotorn körde men hittade inga paket som klarade alla spärrar. Testa att höja radgränsen, sänka Min värde-/poängfilter eller sänka Minsta extra reducering.")
                if isinstance(candidate_audit, pd.DataFrame) and not candidate_audit.empty:
                    with st.expander('Visa kandidatanalys – varför inga paket visades', expanded=True):
                        st.dataframe(candidate_audit, use_container_width=True, hide_index=True)
            else:
                st.info("Inga rekommenderade paket beräknade ännu.")

    mode_options = ['Av', 'Tvingat'] + [f'Grupp {i}' for i in range(1, 7)]
    cats = _ordered_categories_from_specs(specs)

    def _range_key_for_spec(_spec):
        _fhp_spec = _hist_target_for_spec(_spec, filter_hist_target_pct)
        return f"filter_range_{_spec.get('key')}_h{int(_fhp_spec)}_tf{top_fav_count}"

    def _default_interval_for_spec(_spec):
        lo, hi = _spec['bounds']
        default_interval = _current_interval_for_spec(_spec)
        try:
            return (max(lo, float(default_interval[0])), min(hi, float(default_interval[1])))
        except Exception:
            return _spec['default_interval']

    # v12.0bj: rendera bara vald filterkategori – och tillåt att ingen kategori är öppen.
    # Det minskar scroll och gör att filtercentralen kan vara helt hopfälld utan att förlora aktiva filter.
    cat_closed_label = "Stäng alla filterkategorier"
    cat_options = [cat_closed_label] + cats
    _prev_cat = st.session_state.get('v12_selected_filter_category', cat_closed_label)
    selected_cat = st.radio(
        "Visa filterkategori",
        cat_options,
        index=cat_options.index(_prev_cat) if _prev_cat in cat_options else 0,
        horizontal=True,
        key='v12_selected_filter_category',
    )

    settings = {}
    filter_apply = False
    if selected_cat == cat_closed_label:
        st.info("Alla filterkategorier är stängda. Aktiva filter ligger kvar och används fortfarande när du kör filtrering/reducering.")
    else:
        _cat_target_key = _filter_target_key_for_category(selected_cat)
        _cat_default_target = FILTER_HIST_TARGET_DEFAULTS_BY_CATEGORY.get(selected_cat, 95)
        _cat_target_val = st.slider(
            f"Minsta historiska träff på filterintervall – {selected_cat}",
            min_value=50,
            max_value=100,
            value=_clamp_filter_hist_target(st.session_state.get(_cat_target_key, _cat_default_target), _cat_default_target),
            step=1,
            key=_cat_target_key,
            help=(
                "Styr rekommenderat startintervall bara för denna filterkategori. "
                "Exempel: Struktur 100% = intervallen täcker alla liknande historiska omgångar; "
                "Värde & svårighet 95% ger snävare profilfilter men släpper viss historisk variation."
            ),
        )
        _cat_target_hits = _hist_pass_label_from_pct(_cat_target_val, len(v_m))
        st.caption(f"{selected_cat}: startintervallen byggs på minst cirka {_cat_target_hits}/{len(v_m)} historiska träffar.")
        with st.form("v12_filtercentral_form"):
            st.caption("Ändra flera filter i vald kategori och tryck Applicera filterändringar. Övriga filter behåller sina sparade lägen/intervall.")
            st.markdown(f"### 📂 {selected_cat}")
            cat_specs = [s for s in specs if s['category'] == selected_cat]
            for spec in cat_specs:
                k = spec['key']
                mode_key = f'filter_mode_{k}'
                range_key = _range_key_for_spec(spec)
                default_mode = _filtercentral_get_persisted_mode(k, st.session_state.get(mode_key, 'Av'))
                # Sätt widgetvärdet från persistent store innan widgeten skapas.
                if mode_key not in st.session_state:
                    st.session_state[mode_key] = default_mode
                lo, hi = spec['bounds']
                dec = spec['decimals']
                val = _filtercentral_get_persisted_interval(spec, _default_interval_for_spec(spec))
                if range_key not in st.session_state and isinstance(val, (list, tuple)) and len(val) >= 2:
                    st.session_state[range_key] = val
                if range_key in st.session_state:
                    try:
                        cur = st.session_state[range_key]
                        if float(cur[0]) < float(lo) or float(cur[1]) > float(hi):
                            st.session_state[range_key] = val
                    except Exception:
                        st.session_state[range_key] = val
                c1, c2, c3, c4 = st.columns([1.35, 1.0, 3.0, 0.45])
                with c1:
                    st.markdown(f"**{spec['name']}**")
                    hp, ht, pct = _hist_pass_count(spec['hist_values'], spec['default_interval'])
                    _spec_target = int(spec.get('target_hist_pct', _cat_target_val))
                    st.caption(f"Rek: {_display_interval(spec['default_interval'], dec)} · {hp}/{ht} · mål {_spec_target}%")
                with c2:
                    mode = st.selectbox(
                        "Läge",
                        mode_options,
                        index=mode_options.index(default_mode) if default_mode in mode_options else 0,
                        key=mode_key,
                        label_visibility="collapsed",
                    )
                with c3:
                    step = 1 if dec == 0 else (0.01 if dec >= 2 else 0.1)
                    if dec == 0:
                        lo_i, hi_i = int(lo), int(hi)
                        val_i = (int(round(val[0])), int(round(val[1])))
                        rng = st.slider("Intervall", lo_i, hi_i, val_i, step=1, key=range_key, label_visibility="collapsed")
                    else:
                        rng = st.slider("Intervall", float(lo), float(hi), (float(val[0]), float(val[1])), step=step, key=range_key, label_visibility="collapsed")
                with c4:
                    if hasattr(st, "popover"):
                        with st.popover("ℹ️", help="Visa frekvenstabell, träff och reducering för detta filter"):
                            _render_inline_filter_info(spec, rng, frame_rows, frame, antal_matcher)
                    else:
                        with st.expander("ℹ️", expanded=False):
                            _render_inline_filter_info(spec, rng, frame_rows, frame, antal_matcher)
                _render_favorite_shock_diagnostics(spec, rng, frame, filter_vec, antal_matcher)
                st.divider()
            filter_apply = st.form_submit_button("✅ Applicera filterändringar", use_container_width=True)

        if filter_apply:
            # Spara den renderade kategorins widgetvärden till persistent store.
            # Då ligger de kvar när användaren byter filterkategori.
            _sync_rendered_filter_widgets_to_store(cat_specs, filter_hist_target_pct, top_fav_count)

    # Samla inställningar för ALLA filter, inte bara den kategori som renderades.
    for spec in specs:
        k = spec['key']
        mode_key = f'filter_mode_{k}'
        range_key = _range_key_for_spec(spec)
        settings[k] = {
            'mode': _filtercentral_get_persisted_mode(k, st.session_state.get(mode_key, 'Av')),
            'interval': st.session_state.get(range_key, _filtercentral_get_persisted_interval(spec, _default_interval_for_spec(spec))),
        }
    if filter_apply:
        st.session_state['v12_last_result_stale'] = True
        st.success("Filterändringar applicerade. Kör filtrering + reducering när du vill räkna om systemet.")
    # Gruppkrav ligger i sidomenyn så de alltid är lätta att hitta när filter flyttas till Grupp 1–6.
    group_reqs = {}
    with st.sidebar:
        st.markdown("---")
        st.subheader("Gruppkrav – min/max")
        st.caption("Styr hur många filter i varje grupp som måste träffa. Exempel: 5–7 av 8 betyder minst 5 och högst 7 träffade filter.")
        for i in range(1, 7):
            gname = f'Grupp {i}'
            n_in_group = int(sum(1 for v in settings.values() if v.get('mode') == gname))
            st.markdown(f"**{gname}** · {n_in_group} filter")
            if n_in_group <= 0:
                st.caption("Inga filter i gruppen")
                group_reqs[gname] = {'min': 0, 'max': 0}
                st.session_state[f'group_req_{i}'] = 0
                st.session_state[f'group_req_min_{i}'] = 0
                st.session_state[f'group_req_max_{i}'] = 0
                continue
            old_req = int(st.session_state.get(f'group_req_{i}', 0) or 0)
            default_min = int(st.session_state.get(f'group_req_min_{i}', old_req) or 0)
            default_max = int(st.session_state.get(f'group_req_max_{i}', n_in_group) or n_in_group)
            default_min = max(0, min(default_min, n_in_group))
            default_max = max(default_min, min(default_max, n_in_group))
            cmin, cmax = st.columns(2)
            with cmin:
                mn = st.number_input(
                    "Min",
                    min_value=0,
                    max_value=n_in_group,
                    value=default_min,
                    step=1,
                    key=f"group_req_min_{i}",
                    help="0 = inget min-krav.",
                )
            with cmax:
                mx = st.number_input(
                    "Max",
                    min_value=0,
                    max_value=n_in_group,
                    value=default_max,
                    step=1,
                    key=f"group_req_max_{i}",
                    help="Sätt max till antal filter i gruppen om du inte vill ha övre spärr.",
                )
            if int(mx) < int(mn):
                st.warning("Max < min")
            group_reqs[gname] = {'min': int(mn), 'max': int(mx)}
            # Bakåtkompatibel nyckel, så äldre delar/metadata fortfarande kan läsa min-kravet.
            st.session_state[f'group_req_{i}'] = int(mn)
    active_preview = sum(1 for v in settings.values() if v['mode'] != 'Av')
    if active_preview > 0:
        st.success(f"{active_preview} filter är aktiva direkt. Du behöver inte spara filtercentralen separat — körknappen använder nuvarande val.")
    else:
        st.info("Inga aktiva filter ännu. Välj Tvingat eller Grupp på filterraderna innan du kör filtrering.")

    active_count_base = sum(1 for v in settings.values() if v['mode'] != 'Av')
    forced_count = sum(1 for v in settings.values() if v['mode'] == 'Tvingat')
    group_count = active_count_base - forced_count
    pm2k_active_metrics = bool(st.session_state.get('v12_pm2k_active')) and isinstance(st.session_state.get('v12_pm2k_chosen'), dict)
    pm2k_rule_count = len((st.session_state.get('v12_pm2k_chosen') or {}).get('rules', []) or []) if pm2k_active_metrics else 0
    active_count = active_count_base + pm2k_rule_count
    hpkg, htot = _hist_package_passes(v_m, specs, settings, group_reqs)
    hist_metric_txt = f"{hpkg}/{htot}"
    if pm2k_active_metrics:
        pmh, pmt = _pm2k_hist_pass_count(st.session_state.get('v12_pm2k_chosen'), v_m, int(antal_matcher))
        hist_metric_txt = f"{pmh}/{pmt}" if active_count_base == 0 else f"{hpkg}/{htot} + M2K {pmh}/{pmt}"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Aktiva filter", active_count)
    m2.metric("Tvingade", forced_count)
    m3.metric("Grupp/Mönster", group_count + pm2k_rule_count)
    m4.metric("Samlad historikträff", hist_metric_txt)

    applied_meta = st.session_state.get('v12_applied_package_meta')
    if applied_meta and active_count:
        current_sig = _settings_package_signature(settings, group_reqs)
        if current_sig == applied_meta.get('signature'):
            pkg_txt = f"{int(applied_meta.get('hist_hit', 0))}/{int(applied_meta.get('hist_total', 0))}"
            cur_txt = f"{hpkg}/{htot}"
            if int(applied_meta.get('hist_hit', -1)) == int(hpkg) and int(applied_meta.get('hist_total', -1)) == int(htot):
                st.success(f"Paketkontroll OK: paketet och aktiv filtercentral visar samma samlade träff ({cur_txt}).")
            else:
                snapshot = st.session_state.get('v12_applied_package_snapshot', {})
                debug_payload = {
                    'feltyp': 'paketkontroll_mismatch',
                    'förklaring': 'Paketets sparade samlade historikträff skiljer sig från aktiv filtercentrals samlade historikträff trots att signaturen matchar.',
                    'paket_räknades_som': pkg_txt,
                    'aktiv_filtercentral_visar': cur_txt,
                    'paket_meta': {
                        'hist_hit': int(applied_meta.get('hist_hit', 0)),
                        'hist_total': int(applied_meta.get('hist_total', 0)),
                        'frame_after': int(applied_meta.get('frame_after', 0)),
                        'num_filters': int(applied_meta.get('num_filters', 0)),
                        'package_type': applied_meta.get('package_type', ''),
                        'reduction_pct': applied_meta.get('reduction_pct', None),
                    },
                    'aktiv_filtercentral': {
                        'hist_hit': int(hpkg),
                        'hist_total': int(htot),
                        'aktiva_filter': int(active_count),
                        'tvingade_filter': int(forced_count),
                        'gruppfilter': int(group_count),
                    },
                    'signatur_matchar': bool(current_sig == applied_meta.get('signature')),
                    'signatur_diff': _signature_debug_diff(applied_meta.get('signature'), current_sig),
                    'gruppkrav': _json_safe_value(group_reqs),
                    'aktiva_filter_detalj': _debug_active_filters(specs, settings),
                    'paket_filter_detalj': _debug_package_filters_from_snapshot(snapshot),
                    'paket_grupper': _json_safe_value(snapshot.get('groups', []) if isinstance(snapshot, dict) else []),
                    'senaste_applicerade_paket_namn': snapshot.get('package_type', '') if isinstance(snapshot, dict) else '',
                }
                _render_copyable_error_report(
                    f"Paketkontroll mismatch: paketet räknades som {pkg_txt}, men aktiv filtercentral visar {cur_txt}",
                    'PKG-MISMATCH',
                    debug_payload,
                    expanded=True,
                )
        else:
            st.caption("Filtercentralen har ändrats efter att paketet applicerades; paketkontrollen är därför inte längre aktiv.")

    if active_count:
        st.caption("Samlad historikträff är hela aktiva paketet räknat mot de liknande historiska omgångarna: alla Tvingade filter måste sitta och varje Grupp måste klara sitt min/max-krav. Det är inte ett snitt av individuella filterträffar.")
        active_u_count = sum(1 for s in specs if s.get('category') == 'Utgångssystem – Antal tecken' and settings.get(s.get('key'), {}).get('mode') != 'Av')
        if active_u_count:
            st.info(f"{active_u_count} utgångssystemfilter är aktiva. Reduceringsgarantin är villkorad på att facit klarar utgångssystemens intervall och övriga filter innan TipsetMatrix körs.")
    if active_count and hpkg <= max(3, int(0.25 * max(1, htot))):
        st.warning("Samlad historikträff är mycket låg. Det betyder oftast att för många filter är Tvingade samtidigt. Lägg över närliggande filter i Grupp eller bredda intervallen.")

    with st.expander("🔬 Diagnoser och översikter", expanded=False):
        st.caption("Prestandaläge: tunga diagnoser räknas bara när du aktivt väljer dem här.")
        show_hist_diag = st.checkbox("Visa samlad historikträff rad-för-rad", value=False, key="v12_show_hist_diag")
        show_group_diag = st.checkbox("Visa gruppdiagnos", value=False, key="v12_show_group_diag")
        show_filter_overview = st.checkbox("Visa filteröversikt med reducering", value=False, key="v12_show_filter_overview")
        if show_hist_diag:
            if active_count:
                diag_df = _active_package_diagnostic_df(v_m, specs, settings, group_reqs, antal_matcher, max_rows=htot)
                if not diag_df.empty:
                    st.dataframe(diag_df, use_container_width=True, hide_index=True)
                    st.caption("✅ Träff betyder att den historiska vinnarraden klarade hela aktiva filterpaketet. ❌ Miss visar vilket tvingat filter eller gruppkrav som stoppade raden.")
                else:
                    st.info("Ingen historikdiagnos kunde byggas.")
            else:
                st.info("Aktivera minst ett filter för att se samlad historikdiagnos.")
        if show_group_diag:
            group_diag_df = _active_group_diagnostic_df(specs, settings, group_reqs, frame_rows=frame_rows)
            if not group_diag_df.empty:
                st.dataframe(group_diag_df, use_container_width=True, hide_index=True)
                st.caption("Gruppdiagnosen räknar gruppens min/max-krav separat från tvingade filter.")
            else:
                st.info("Inga filter ligger i Grupp 1–6 just nu.")
        if show_filter_overview:
            overview_rows = frame_rows
            if frame_rows is not None and len(frame_rows) > 5000:
                overview_rows, _ = _sample_rows_for_macro(frame_rows, max_items=5000)
                st.caption("Filteröversiktens reducering uppskattas på 5 000 rader. Kör filtrering för exakt resultat.")
            st.dataframe(_build_filter_summary_df(specs, settings, group_reqs, rows=overview_rows), use_container_width=True, hide_index=True)

    # Filterinfo visas nu i popover direkt på respektive filter. Att öppna/stänga popover triggar normalt inte rerun.
    st.markdown("</div>", unsafe_allow_html=True)

    # Step 4 – run filters + TipsetMatrix
    st.markdown("<div class='v12-card'>", unsafe_allow_html=True)
    st.markdown("<div class='v12-step'>Steg 4</div><div class='v12-title'>Kör filtrering och TipsetMatrix</div>", unsafe_allow_html=True)
    col_r1, col_r2, col_r3, col_r4 = st.columns([1, 1, 1, 1])
    with col_r1:
        matrix_limit = st.number_input(
            "Max filtermassa till TipsetMatrix",
            min_value=500,
            max_value=50000,
            value=int(st.session_state.get('v12_matrix_limit', 5000)),
            step=500,
            key="v12_matrix_limit",
            help="Om fler rader återstår körs filtreringen ändå, men TipsetMatrix stoppas tills du höjer spärren eller filtrerar hårdare.",
        )
    with col_r2:
        run_matrix = st.checkbox("Kör TipsetMatrix 12", value=bool(st.session_state.get('v12_run_matrix', True)), key="v12_run_matrix")
    with col_r3:
        reducer_modes = ["Minsta rader", "Balans", "Favoriter", "Skräll"]
        saved_mode = st.session_state.get('v12_reducer_mode', "Minsta rader")
        reducer_mode = st.selectbox("Motor", reducer_modes, index=reducer_modes.index(saved_mode) if saved_mode in reducer_modes else 0, key="v12_reducer_mode")
    with col_r4:
        target_13_pct = st.number_input(
            "Höj 13-chans till minst %",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.get('v12_target_13_pct', 0.0)),
            step=1.0,
            key="v12_target_13_pct",
            help="0 = minsta radantal för full 12-rättsgaranti. Om systemet känns för billigt kan du höja procenten; appen lägger då till extra rader med högst procentvikt för att öka chansen till 13 rätt.",
        )

    reducer_save_settings = {
        'v12_matrix_limit': int(matrix_limit),
        'v12_run_matrix': bool(run_matrix),
        'v12_reducer_mode': reducer_mode,
        'v12_target_13_pct': float(target_13_pct),
    }
    package_save_state = _collect_recommended_package_state_for_save(specs, settings=settings, group_reqs=group_reqs)
    filter_payload = _build_filterpaket_payload(
        specs, group_reqs, filter_hist_target_pct, top_fav_count, spelform, antal_matcher,
        settings_override=settings, package_state=package_save_state, manual_sign_groups=manual_sign_groups,
    )
    game_payload = _build_spelfil_payload(
        specs, group_reqs, filter_hist_target_pct, top_fav_count, spelform, antal_matcher,
        input_text, top_n, pay_min, pay_max, frame, v_m, filter_vec, reducer_settings=reducer_save_settings,
        settings_override=settings, package_state=package_save_state, manual_sign_groups=manual_sign_groups,
    )
    with sidebar_save_slot.container():
        st.markdown("**Spara spelfil/filterpaket**")
        st.caption("Spelfil sparar kupong, grundram, filter, grupper och reduceringsinställningar. Filterpaket sparar bara filter/intervall/gruppkrav.")
        tm_download_button(
            "⬇️ Spara spelfil",
            _payload_to_json_bytes(game_payload),
            f"{_fmt_file_stem('spelfil')}.json",
            "application/json",
            use_container_width=True,
            key="v12_sidebar_save_spelfil",
        )
        tm_download_button(
            "⬇️ Spara filterpaket",
            _payload_to_json_bytes(filter_payload),
            f"{_fmt_file_stem('filterpaket')}.json",
            "application/json",
            use_container_width=True,
            key="v12_sidebar_save_filterpaket",
        )

    st.caption("Spara/öppna spelfil och filterpaket ligger nu i sidomenyn.")

    if st.session_state.get('v12_last_result_stale') and st.session_state.get('v12_last_result'):
        st.warning("Filter/paketinställningar har ändrats sedan senaste körningen. Tryck Kör filtrering + reducering för att räkna om resultatet.")
    go = st.button("🚀 Kör filtrering + reducering", use_container_width=True)

    if go:
        st.session_state['v12_last_result_stale'] = False
        with st.spinner("Filtrerar exakt grundram..."):
            filtered_rows = _apply_manual_filters(manual_frame_rows, specs, settings, group_reqs)
            pm2k_before = len(filtered_rows)
            pm2k_active = bool(st.session_state.get('v12_pm2k_active'))
            pm2k_chosen = st.session_state.get('v12_pm2k_chosen') if pm2k_active else None
            if pm2k_active and isinstance(pm2k_chosen, dict):
                filtered_rows = _pm2k_apply_chosen_to_rows(filtered_rows, pm2k_chosen, filter_vec, int(antal_matcher))
        st.session_state['v12_last_result'] = {'filtered_rows': filtered_rows, 'reduced_rows': [], 'settings': settings, 'group_reqs': group_reqs, 'hist_package': {'hit': int(hpkg), 'total': int(htot)}, 'pm2k_active': bool(st.session_state.get('v12_pm2k_active')), 'pm2k_before': int(pm2k_before) if 'pm2k_before' in locals() else None}
        if bool(st.session_state.get('v12_pm2k_active')):
            st.success(f"Filtrering klar: {len(frame_rows):,} → {len(manual_frame_rows):,} efter manuella teckengrupper → {pm2k_before:,} efter filtercentral → {len(filtered_rows):,} efter Mönstermotor2K.".replace(',', ' '))
        else:
            st.success(f"Filtrering klar: {len(frame_rows):,} → {len(manual_frame_rows):,} efter manuella teckengrupper → {len(filtered_rows):,} rader.".replace(',', ' '))
        miss = selected_signs_missing(filtered_rows, frame, antal_matcher)
        if miss:
            st.warning("Vissa markerade tecken saknas efter filter: " + format_missing_signs(miss) + " · Varningsläge endast: appen stoppar inte detta ännu, men paketet har låst bort tecknet helt.")
        if run_matrix:
            if len(filtered_rows) == 0:
                st.error("Inga rader kvar efter filter.")
            elif len(filtered_rows) > int(matrix_limit):
                st.warning(f"Filtermassan är {len(filtered_rows):,} rader, över spärren {int(matrix_limit):,}. Höj spärren eller filtrera hårdare innan TipsetMatrix.".replace(',', ' '))
            else:
                with st.spinner("Kör TipsetMatrix 12-garanti..."):
                    if len(filter_vec) < antal_matcher * 3:
                        st.warning("Kupongvektorn är kortare än antal matcher. TipsetMatrix körs med neutral viktning för att undvika krasch.")
                    clean_filtered_rows = [str(r).strip().upper() for r in filtered_rows if isinstance(r, str) and len(str(r).strip()) == antal_matcher]
                    if len(clean_filtered_rows) != len(filtered_rows):
                        st.warning("Några felaktiga radobjekt sorterades bort före TipsetMatrix.")
                    filtered_rows = clean_filtered_rows
                    st.session_state['v12_last_result']['filtered_rows'] = filtered_rows
                    scores = [row_log_probability(r, filter_vec) for r in filtered_rows]
                    reduced_rows_12, tm_meta = tipsetmatrix12_reduce(filtered_rows, row_scores=scores, mode=reducer_mode, max_output_rows=None, seed=42)
                    reduced_rows, boost_meta = add_rows_for_13_chance(filtered_rows, reduced_rows_12, row_scores=scores, target_13_pct=target_13_pct)
                    tm_meta['base_12_rows'] = int(boost_meta.get('base_12_rows', len(reduced_rows_12)))
                    tm_meta['extra_13_rows'] = int(boost_meta.get('extra_13_rows', 0))
                    tm_meta['target_13_pct'] = float(boost_meta.get('target_13_pct', 0.0))
                    tm_meta['final_13_pct'] = float(boost_meta.get('final_13_pct', 0.0))
                st.session_state['v12_last_result']['reduced_rows'] = reduced_rows
                st.session_state['v12_last_result']['tm_meta'] = tm_meta
                if tm_meta.get('extra_13_rows', 0):
                    st.success(f"TipsetMatrix klar: 12-garanti {len(filtered_rows):,} → {tm_meta.get('base_12_rows', len(reduced_rows)):,} rader. 13-chans höjd med {tm_meta.get('extra_13_rows',0):,} extra rader → {len(reduced_rows):,} slutrader.".replace(',', ' '))
                else:
                    st.success(f"TipsetMatrix klar: {len(filtered_rows):,} → {len(reduced_rows):,} rader.".replace(',', ' '))
                if not tm_meta.get('complete', False):
                    st.warning(f"TipsetMatrix täckte {tm_meta.get('covered_pct', 0)}% av filtermassan. Höj spärren eller filtrera hårdare om du vill kräva full 12-garanti.")

    res = st.session_state.get('v12_last_result')
    if res and res.get('filtered_rows') is not None:
        filtered_rows = res['filtered_rows']
        reduced_rows = res.get('reduced_rows') or []
        # Skydd mot äldre session/resultat där reducerfunktionen felaktigt sparades som (rader, metadata).
        if isinstance(reduced_rows, tuple) and len(reduced_rows) == 2 and isinstance(reduced_rows[0], list):
            reduced_rows = reduced_rows[0]
            st.session_state['v12_last_result']['reduced_rows'] = reduced_rows
        reduced_rows = [normalize_single_row_text(r) for r in (reduced_rows or []) if len(normalize_single_row_text(r)) == int(antal_matcher)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Grundram", f"{len(frame_rows):,}".replace(',', ' '))
        c2.metric("Efter filter", f"{len(filtered_rows):,}".replace(',', ' '), f"-{(100-100*len(filtered_rows)/max(1,len(frame_rows))):.1f}%")
        if reduced_rows:
            tm_meta_display = res.get('tm_meta', {}) or {}
            base_12_rows = int(tm_meta_display.get('base_12_rows', len(reduced_rows)) or len(reduced_rows))
            extra_13_rows = int(tm_meta_display.get('extra_13_rows', 0) or 0)
            c3.metric("Efter TipsetMatrix", f"{len(reduced_rows):,}".replace(',', ' '), f"-{(100-100*len(reduced_rows)/max(1,len(filtered_rows))):.1f}%")
            weighted_13_pct = weighted_13_share(filtered_rows, reduced_rows, filter_vec)
            c4.metric("13-chans", f"{100*len(reduced_rows)/max(1,len(filtered_rows)):.2f}%", f"viktad {weighted_13_pct:.2f}%")
            if extra_13_rows > 0:
                st.info(f"Reduceringsval: minsta hittade 12-garanti gav {base_12_rows:,} rader. Därefter lades {extra_13_rows:,} extra rader till för att höja 13-chansen.".replace(',', ' '))
            else:
                st.caption(f"Reduceringsval: minsta hittade 12-garantimassa används ({base_12_rows:,} rader). Höj 13-chansmålet om systemet känns för billigt.".replace(',', ' '))
            guarantee_df, guarantee_meta = build_tipsetmatrix_guarantee_table(filtered_rows, reduced_rows, antal_matcher, prob_vector=filter_vec)
            if float(guarantee_meta.get('12plus', 0.0)) < 100.0:
                st.error(f"Kontrollvarning: garantitabellen visar bara {guarantee_meta.get('12plus', 0)}% 12+. Använd inte inlämningsfilen förrän filtret/reduceringen har körts om eller spärren justerats.")
            st.dataframe(guarantee_df, use_container_width=True, hide_index=True)
            with st.expander("Teckenfördelning efter filter och slutrader"):
                st.dataframe(build_combined_sign_distribution_df(filtered_rows, reduced_rows, frame, antal_matcher), use_container_width=True, hide_index=True)
            sub_text = rows_to_submission_text(reduced_rows, spelform, antal_matcher)
            tm_download_button("⬇️ Ladda ner inlämningsfil TXT", sub_text, f"{submission_file_stem(spelform)}_v12_inlamning.txt", "text/plain", use_container_width=True)
        else:
            c3.metric("Efter TipsetMatrix", "Ej körd")
            c4.metric("13-chans", "—")
            with st.expander("Teckenfördelning efter filter"):
                st.dataframe(build_sign_distribution_df(filtered_rows, frame, antal_matcher), use_container_width=True, hide_index=True)
            if filtered_rows:
                tm_download_button("⬇️ Ladda ner filtrerade rader TXT", rows_to_submission_text(filtered_rows, spelform, antal_matcher), f"{submission_file_stem(spelform)}_v12_filtrerade.txt", "text/plain", use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧾 Rättningsmodul")
        st.caption("Rättningen använder senast körda grundram, filtermassa och TipsetMatrix-rader — den kör inte om filter eller reducering.")
        correction_mode = st.radio("Rättningsläge", ["Live-rättning / endast påbörjade matcher", "Sluträttning"], horizontal=True, key="v12_correction_mode")

        if correction_mode.startswith("Live"):
            st.caption("Skriv 1/X/2 för matcher som ska rättas och - för matcher som inte ska räknas ännu. Exempel: XX2--1---X---")
            if 'v12_live_correction_draft' not in st.session_state:
                st.session_state['v12_live_correction_draft'] = st.session_state.get('v12_live_correction_input', '')

            # v12.0bq: Rättningsraden måste vara stabil. Tidigare kunde Streamlit-reruns
            # visa senast sparade rättningsrad igen när man ändrade texten, vilket gjorde att
            # användaren ibland behövde skriva/rätta två gånger. Formen gör att textändringen
            # och knappen skickas som en enda transaktion, medan tabellerna nedan alltid bygger
            # på senast uttryckligen rättad rad.
            with st.form("v12_live_correction_form", clear_on_submit=False):
                live_txt_draft = st.text_input("Live-facit", key='v12_live_correction_draft', placeholder="Exempel: XX2--1---X---")
                live_pos_count, live_known_count, live_normalized, live_invalid_chars = count_live_result_input(live_txt_draft)
                render_counter_caption("Teckenräknare live-facit", live_pos_count, live_known_count, antal_matcher, live_normalized, live_invalid_chars)
                live_submit = st.form_submit_button("Rätta live", use_container_width=True)

            if live_submit:
                st.session_state['v12_live_correction_input'] = live_txt_draft

            live_submitted_txt = st.session_state.get('v12_live_correction_input', '')
            live_dirty = (str(st.session_state.get('v12_live_correction_draft', '') or '') != str(live_submitted_txt or ''))
            if live_dirty and live_submitted_txt:
                st.info("Live-facit har ändrats. Tabellerna nedan visar senast rättade rad. Klicka **Rätta live** för att uppdatera.")
            live_txt = live_submitted_txt
            live_row, live_err = parse_live_result_row(live_txt, antal_matcher)
            if live_txt and live_err:
                st.error(live_err)
            if live_row:
                known_count = len(live_known_positions(live_row))
                base_rows_for_corr = frame_rows
                filtered_rows_for_corr = filtered_rows
                reduced_rows_for_corr = reduced_rows
                cA, cB, cC, cD = st.columns(4)
                cA.metric("Rättade matcher", f"{known_count}/{antal_matcher}")
                # Bästa rad efter filter/slutrader
                best_pool = reduced_rows_for_corr if reduced_rows_for_corr else filtered_rows_for_corr
                top_one = best_live_rows(best_pool, live_row, filter_vec, antal_matcher, limit=1)
                if top_one:
                    cB.metric("Bästa live-träff", top_one[0].get('Live-träff', '—'))
                    cC.metric("Bästa maxnivå", top_one[0].get('Max möjligt', '—'))
                    cD.metric("Missar", top_one[0].get('Missar', '—'))
                else:
                    cB.metric("Bästa live-träff", "—")
                    cC.metric("Bästa maxnivå", "—")
                    cD.metric("Missar", "—")

                st.markdown("**Rader som fortfarande lever**")
                live_summary_df = build_live_pool_summary_df(live_row, base_rows_for_corr, filtered_rows_for_corr, reduced_rows_for_corr, antal_matcher)
                st.dataframe(live_summary_df, use_container_width=True, hide_index=True)

                top_limit = st.slider("Visa antal bästa rader", 10, 200, 50, 10, key="v12_live_top_limit")
                if reduced_rows_for_corr:
                    render_best_live_rows_table(
                        best_live_rows(reduced_rows_for_corr, live_row, filter_vec, antal_matcher, limit=top_limit),
                        title="Bästa inlämnade/TipsetMatrix-rader"
                    )
                    with st.expander("Visa bästa rader efter filtermassan", expanded=False):
                        render_best_live_rows_table(
                            best_live_rows(filtered_rows_for_corr, live_row, filter_vec, antal_matcher, limit=top_limit),
                            title="Bästa rader efter filter"
                        )
                else:
                    render_best_live_rows_table(
                        best_live_rows(filtered_rows_for_corr, live_row, filter_vec, antal_matcher, limit=top_limit),
                        title="Bästa rader efter filter"
                    )
                st.caption("Grön = träff på rättad match. Röd = miss på rättad match. Grå = matchen räknas inte ännu.")

        else:
            if 'v12_correction_draft' not in st.session_state:
                st.session_state['v12_correction_draft'] = st.session_state.get('v12_correction_input', '')
            with st.form("v12_final_correction_form", clear_on_submit=False):
                corr_txt = st.text_input("Rätt rad", key='v12_correction_draft', placeholder="Exempel: 1X2X1122X... eller 1,X,2,X,...")
                corr_count = count_result_row_input(corr_txt)
                render_counter_caption("Teckenräknare rätt rad", corr_count, None, antal_matcher, '')
                corr_submit = st.form_submit_button("Rätta system")
            if corr_submit:
                rr, err = parse_result_row(corr_txt, antal_matcher)
                if err:
                    st.error(err)
                    st.session_state['v12_correction_row'] = ''
                else:
                    st.session_state['v12_correction_input'] = corr_txt
                    st.session_state['v12_correction_row'] = rr
            corr_dirty = (str(corr_txt or '') != str(st.session_state.get('v12_correction_input', '') or ''))
            if corr_dirty and st.session_state.get('v12_correction_row'):
                st.info("Rätt rad har ändrats. Klicka **Rätta system** för att uppdatera rättningen.")
            corr_row = '' if corr_dirty else st.session_state.get('v12_correction_row')
            if corr_row:
                base_rows_for_corr = frame_rows
                filtered_rows_for_corr = filtered_rows
                reduced_rows_for_corr = reduced_rows
                corr = build_facit_check(corr_row, frame, base_rows_for_corr, filtered_rows_for_corr, reduced_rows_for_corr, antal_matcher)
                cA, cB, cC, cD, cE = st.columns(5)
                cA.metric("I grundram", yes_no(corr.get('I grundram')))
                cB.metric("Efter filter", yes_no(corr.get('Efter filter')))
                cC.metric("Efter TipsetMatrix", yes_no(corr.get('Efter TipsetMatrix')) if reduced_rows_for_corr else "Ej körd")
                cD.metric("Bästa rätt", corr.get('Bästa rätt efter TipsetMatrix', 0) if reduced_rows_for_corr else "—")
                cE.metric("12+", yes_no(corr.get('12+ uppnått')) if reduced_rows_for_corr else "—")

                st.markdown("**Antal rader med 13/12/11/10 rätt**")
                dist_df = build_correction_hit_distribution_df(corr_row, base_rows_for_corr, filtered_rows_for_corr, reduced_rows_for_corr, antal_matcher)
                st.dataframe(dist_df, use_container_width=True, hide_index=True)

                if reduced_rows_for_corr and corr.get('Närmaste reducerade rader'):
                    with st.expander("Visa närmaste reducerade rader", expanded=False):
                        nearest_df = pd.DataFrame({
                            'Rad': corr.get('Närmaste reducerade rader'),
                            'Rätt': [sum(1 for a, b in zip(corr_row, r) if a == b) for r in corr.get('Närmaste reducerade rader')],
                        })
                        st.dataframe(nearest_df, use_container_width=True, hide_index=True)

                st.markdown("**Filterträff/miss på rätt rad**")
                filter_corr_df = build_filter_correction_df(corr_row, specs, res.get('settings', settings), res.get('group_reqs', group_reqs))
                pm_corr_df = pd.DataFrame()
                if bool(st.session_state.get('v12_pm2k_active')):
                    pm_corr_df = _pm2k_correction_df(corr_row, st.session_state.get('v12_pm2k_chosen'), filter_vec)
                if not pm_corr_df.empty:
                    if filter_corr_df.empty:
                        filter_corr_df = pm_corr_df
                    else:
                        filter_corr_df = pd.concat([filter_corr_df, pm_corr_df], ignore_index=True, sort=False)
                if filter_corr_df.empty:
                    st.info("Inga aktiva filter att rätta mot.")
                else:
                    st.dataframe(filter_corr_df, use_container_width=True, hide_index=True)

                group_corr_df = build_group_correction_df(corr_row, specs, res.get('settings', settings), res.get('group_reqs', group_reqs))
                if not group_corr_df.empty:
                    st.markdown("**Gruppträffar**")
                    st.dataframe(group_corr_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Läs in kupongdata och spara grundramen för att öppna filtercentralen.")


# v12.0bh: tidsprofil för felsökning av långsamma reruns
try:
    if _perf_enabled():
        _perf_mark('Total sidkörning', _perf_start)
        _marks = st.session_state.get('v12_perf_marks', [])
        if _marks:
            with st.sidebar.expander('⏱ Tidsprofil senaste körning', expanded=False):
                st.dataframe(pd.DataFrame(_marks), use_container_width=True, hide_index=True)
except Exception:
    pass
