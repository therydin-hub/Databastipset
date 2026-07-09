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
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Tipset AI-Analys", layout="wide", page_icon="🎯")
APP_VERSION = "v12.0ak – Utgångssystem / U-filter"


st.markdown("""
<style>
    .main .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {background: rgba(255,255,255,0.04); border: 1px solid rgba(128,128,128,0.22); padding: 0.75rem; border-radius: 0.9rem;}
    .tm-hero {border: 1px solid rgba(128,128,128,0.25); border-radius: 1rem; padding: 1rem 1.1rem; background: linear-gradient(135deg, rgba(60,120,255,0.10), rgba(20,20,20,0.02)); margin-bottom: 1rem;}
    .tm-step {font-size: 0.82rem; letter-spacing: .06em; text-transform: uppercase; opacity: .72; font-weight: 700; margin-bottom: .15rem;}
    .tm-title {font-size: 1.25rem; font-weight: 800; margin-bottom: .2rem;}
    .tm-muted {opacity: .72; font-size: .92rem;}
    .tm-pill {display:inline-block; padding: .16rem .50rem; border:1px solid rgba(128,128,128,.35); border-radius:999px; font-size:.82rem; margin-right:.25rem; margin-top:.2rem;}
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


def generate_rows_from_frame(frame, max_rows=50000):
    """Genererar alla enkelrader från en egen grundram, med säkerhetsspärr."""
    if not frame:
        return [], 0, False, "Ingen grundram finns."
    frame = [normalize_signs(s) for s in frame]
    empty_matches = [i + 1 for i, signs in enumerate(frame) if len(signs) == 0]
    if empty_matches:
        return [], 0, False, "Alla matcher måste ha minst ett tecken. Saknar tecken i match: " + ", ".join(map(str, empty_matches))
    n_rows = frame_row_count(frame)
    if n_rows > int(max_rows):
        return [], n_rows, False, f"Grundramen är {n_rows:,} rader och överskrider spärren {int(max_rows):,}. Smalna av ramen eller höj spärren."
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

    Formatet är t.ex. Europatipset:
    EUROPATIPSET
    E,2,2,1,X,2,...
    """
    sf = str(spelform or "").strip().lower()
    if "europa" in sf:
        return "EUROPATIPSET", "E"
    if "stryk" in sf:
        return "STRYKTIPSET", "S"
    if "topp" in sf:
        return "TOPPTIPSET", "T"
    if "power" in sf:
        return "POWERPLAY", "P"
    return str(spelform or "TIPSET").upper(), "R"


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
    default_modes = {1: 'Favorit/procent', 2: 'Historik/AI', 3: 'Manuell'}
    for slot in range(1, 4):
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
            if not (1 <= slot <= 3):
                continue
            st.session_state[f'v12_us_{slot}_enabled'] = bool(item.get('enabled', False))
            st.session_state[f'v12_us_{slot}_source'] = str(item.get('source') or 'Manuell')
            st.session_state[f'v12_us_{slot}_name'] = str(item.get('name') or f'Utgångssystem {slot}')
            if item.get('raw_text') is not None:
                st.session_state[f'v12_us_{slot}_text'] = str(item.get('raw_text') or '')
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
        for slot in range(1, 4):
            if not bool(st.session_state.get(f'v12_us_{slot}_enabled', False)):
                st.session_state[f'v12_us_{slot}_enabled'] = True
                st.session_state[f'v12_us_{slot}_source'] = 'Manuell'
                st.session_state[f'v12_us_{slot}_name'] = str(item.get('name') or f'Utgångssystem {slot}')
                for m, c in enumerate(row):
                    st.session_state[f'v12_us_{slot}_m{m}'] = c
                break


def build_u_rows_for_filtercentral(v_m, filter_vec, antal_matcher):
    """Bygger aktiva utgångssystem som sedan blir fem filter per system."""
    systems = []
    default_modes = {1: 'Favorit/procent', 2: 'Historik/AI', 3: 'Manuell'}
    for slot in range(1, 4):
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
    if grp == "struktur" or any(k in name for k in ["tecken 1x2", "sviter", "teckenföljd", "teckenlucka", "singlar", "dubbletter", "tripplar", "uppkomster"]):
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


@st.cache_data(show_spinner=False)
def _cached_rows_from_frame(frame_tuple, antal_matcher, max_rows=75000):
    frame = [list(x) for x in frame_tuple]
    return generate_rows_from_frame(frame, max_rows=max_rows)


def _frame_cache_tuple(frame):
    return tuple(tuple(x) for x in frame)


def build_clean_filter_specs(v_m, filter_vec, antal_matcher, slider_u_count=3, target_hist_pct=90, u_rows=None):
    """Bygger exakt en spec per filter. Inga AutoHard-varianter/dubbletter.

    target_hist_pct styr rekommenderat intervall/startvärde för varje filter.
    Exempel: 90% på 30 liknande omgångar ger normalt minst 27/30 i rek. träff.
    """
    rows = list(v_m['Correct_Row'])
    probs = list(v_m['Prob_Vector'])
    match_odds_filter = [filter_vec[j:j+3] for j in range(0, len(filter_vec), 3)]
    cur_matrix, cur_scores_asc, cur_tot = calculate_ai_matrix_from_values(filter_vec)
    specs = []

    target_hist_pct = int(max(50, min(100, target_hist_pct)))

    def add_interval(name, category, values, getter, decimals=0, coverage=85, hard_min=None, hard_max=None, help_text="", key_override=None):
        vals = [v for v in values if not pd.isna(v)]
        if not vals:
            return
        # I v12.0j styr användaren själv minsta historiska träff för rekommenderat
        # intervall/startvärde. Tidigare låg olika filter på 85/90/100, vilket gjorde
        # att vissa filter fortfarande kunde visa t.ex. 26/30 trots att användaren ville ha 100%.
        coverage = target_hist_pct
        default_interval = get_best_interval(vals, coverage)
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
            'help': help_text,
        })

    # Historiska värden
    ai_ranks, delta_vals, total_diff_vals = [], [], []
    rank_sums, minus_sums, log_sums, sft_sums, points_vals = [], [], [], [], []
    fat_f, fat_a, fat_t, fat_sum = [], [], [], []
    fav_top_vals, fav_delta_vals = [], []
    fav70, fav60, fav50 = [], [], []
    sh10, sh15, sh20, sh_low = [], [], [], []
    ones, draws, twos = [], [], []
    s1, sx, s2 = [], [], []
    g1, gx, g2 = [], [], []
    si1, six, si2, sit = [], [], [], []
    d1, dx, d2, dt = [], [], [], []
    tr1, trx, tr2, trt = [], [], [], []
    o1, ox, o2, ot = [], [], [], []
    fat_strings = []

    # Utgångssystem / U-filter: varje system blir fem Helgardering-liknande
    # Antal tecken-filter: utgångstips, ettor, kryss, tvåor och 1/X/2.
    # Garantin är villkorad: TipsetMatrix-garantin gäller bara om facit
    # överlever dessa filter precis som övriga filter.
    metric_defs = [
        ('utips', 'Antalet utgångstips', 'antal matchande tecken mot hela utgångssystemet'),
        ('ones', 'Antalet ettor', 'antal ettor i enkelraden som också finns markerade i utgångssystemet'),
        ('draws', 'Antalet kryss', 'antal kryss i enkelraden som också finns markerade i utgångssystemet'),
        ('twos', 'Antalet tvåor', 'antal tvåor i enkelraden som också finns markerade i utgångssystemet'),
        ('any_1x2', 'Antal 1/X/2', 'högsta av ettor/kryss/tvåor; motsvarar minst en teckentyp i intervallet'),
    ]
    for u in (u_rows or []):
        system = u.get('system', [])
        if len(system) != int(antal_matcher) or u_system_marked_count(system) <= 0:
            continue
        uname = str(u.get('name', 'Utgångssystem')).strip() or 'Utgångssystem'
        usource = str(u.get('source', '') or '')
        system_txt = u_system_to_text(system)
        for metric, label, metric_help in metric_defs:
            vals = [_u_system_metric(str(r), system, metric) for r in rows if len(str(r)) == int(antal_matcher)]
            help_txt = f"{uname}: {system_txt}. {metric_help}." + (f" Källa: {usource}." if usource else "")
            add_interval(
                f"{uname} – {label}",
                'Utgångssystem – Antal tecken',
                vals,
                lambda r, _sys=system, _metric=metric: _u_system_metric(r, _sys, _metric),
                0,
                90,
                0,
                antal_matcher,
                help_txt,
                key_override=f"u_sys_{_slug(u.get('id', uname))}_{metric}",
            )

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
        f, a, t, fs = get_fat(row_str, p); fat_f.append(f); fat_a.append(a); fat_t.append(t); fat_sum.append(fs)
        fav_top_vals.append(get_top_n_favs_wins(row_str, p, slider_u_count))
        fav_delta_vals.append(get_favorite_delta(row_str, p))
        fp = get_favorite_pressure(row_str, p); fav70.append(fp['F70_Wins']); fav60.append(fp['F60_Wins']); fav50.append(fp['F50_Wins'])
        sh = get_shock_strength(row_str, p); sh10.append(sh['U10_Wins']); sh15.append(sh['U15_Wins']); sh20.append(sh['U20_Wins']); sh_low.append(sh['Lowest_Win_Pct'])
        ones.append(row_str.count('1')); draws.append(row_str.count('X')); twos.append(row_str.count('2'))
        _s1, _sx, _s2, _ = get_streaks(row_str); s1.append(_s1); sx.append(_sx); s2.append(_s2)
        _g1, _gx, _g2, _ = get_gaps(row_str); g1.append(_g1); gx.append(_gx); g2.append(_g2)
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
    add_interval('Poängfilter', 'Värde & svårighet', points_vals, lambda r: get_rank_points(r, filter_vec), 0, 85)

    # FAT
    add_interval('FAT F', 'FAT', fat_f, lambda r: get_fat(r, filter_vec)[0], 0, 85, 0, antal_matcher)
    add_interval('FAT A', 'FAT', fat_a, lambda r: get_fat(r, filter_vec)[1], 0, 85, 0, antal_matcher)
    add_interval('FAT T', 'FAT', fat_t, lambda r: get_fat(r, filter_vec)[2], 0, 85, 0, antal_matcher)
    add_interval('FAT Summa', 'FAT', fat_sum, lambda r: get_fat(r, filter_vec)[3], 0, 85)

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
    add_interval(f'Topp {slider_u_count} favoriter', 'Favorit & skräll', fav_top_vals, lambda r: get_top_n_favs_wins(r, filter_vec, slider_u_count), 0, 90, 0, slider_u_count, key_override='topp_favoriter')
    add_interval('Favorittryck ≥70%', 'Favorit & skräll', fav70, lambda r: get_favorite_pressure(r, filter_vec)['F70_Wins'], 0, 85, 0, get_favorite_threshold_counts(filter_vec).get(70, 0))
    add_interval('Favorittryck ≥60%', 'Favorit & skräll', fav60, lambda r: get_favorite_pressure(r, filter_vec)['F60_Wins'], 0, 85, 0, get_favorite_threshold_counts(filter_vec).get(60, 0))
    add_interval('Favorittryck ≥50%', 'Favorit & skräll', fav50, lambda r: get_favorite_pressure(r, filter_vec)['F50_Wins'], 0, 85, 0, get_favorite_threshold_counts(filter_vec).get(50, 0))
    add_interval('Skrällstyrka U10', 'Favorit & skräll', sh10, lambda r: get_shock_strength(r, filter_vec)['U10_Wins'], 0, 85, 0, get_shock_capacity(filter_vec).get(10, 0))
    add_interval('Skrällstyrka U15', 'Favorit & skräll', sh15, lambda r: get_shock_strength(r, filter_vec)['U15_Wins'], 0, 85, 0, get_shock_capacity(filter_vec).get(15, 0))
    add_interval('Skrällstyrka U20', 'Favorit & skräll', sh20, lambda r: get_shock_strength(r, filter_vec)['U20_Wins'], 0, 85, 0, get_shock_capacity(filter_vec).get(20, 0))
    add_interval('Lägsta vinnande %', 'Favorit & skräll', sh_low, lambda r: get_shock_strength(r, filter_vec)['Lowest_Win_Pct'], 1, 85, 0, 100)
    add_interval('Favorit-delta', 'Favorit & skräll', fav_delta_vals, lambda r: get_favorite_delta(r, filter_vec), 2, 85)

    # Struktur
    add_interval('Tecken 1', 'Struktur', ones, lambda r: r.count('1'), 0, 100, 0, antal_matcher)
    add_interval('Tecken X', 'Struktur', draws, lambda r: r.count('X'), 0, 100, 0, antal_matcher)
    add_interval('Tecken 2', 'Struktur', twos, lambda r: r.count('2'), 0, 100, 0, antal_matcher)
    add_interval('Sviter 1', 'Struktur', s1, lambda r: get_streaks(r)[0], 0, 100, 0, antal_matcher)
    add_interval('Sviter X', 'Struktur', sx, lambda r: get_streaks(r)[1], 0, 100, 0, antal_matcher)
    add_interval('Sviter 2', 'Struktur', s2, lambda r: get_streaks(r)[2], 0, 100, 0, antal_matcher)
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

    # Super-makro som ett enda manuellt filter.
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
    add_interval('Super-Makro grupper', 'Super-Makro', macro_vals, lambda r, b=macro_bounds: _super_macro_count(r, filter_vec, b), 0, 90, 0, 8, 'Antal makrogrupper som klaras. Varje grupp räknas om minst 2 av 3 interna delar sitter.')

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


def _hist_package_passes(v_m, specs, settings, group_reqs):
    """Samlad historisk träff för aktiva filter.

    Viktigt: detta ska räknas på samma historiska filtervärden som används i
    frekvenstabellerna och i rekommenderade paket. Vissa filter beror på
    veckans procentvektor i sin getter, t.ex. AI-Rank/Delta/Poängfilter. Om vi
    kör getter direkt på historiska rättsrader med veckans filter_vec blir
    samlad träff fel och kan visa t.ex. 9/30 fast paketmotorn räknade 22/30.

    Därför räknar denna funktion med spec['hist_values'][i] för varje av de
    liknande historiska omgångarna. Det gör Filtercentralens samlade träff och
    paketmotorns träff jämförbara.
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


def _collect_filter_settings_for_save(specs, filter_hist_target_pct, top_fav_count):
    out = {}
    for spec in specs or []:
        k = spec.get('key')
        if not k:
            continue
        mode = st.session_state.get(f'filter_mode_{k}', 'Av')
        range_key = f'filter_range_{k}_h{int(filter_hist_target_pct)}_tf{int(top_fav_count)}'
        interval = st.session_state.get(range_key, spec.get('default_interval'))
        out[k] = {
            'name': spec.get('name', k),
            'category': spec.get('category', ''),
            'mode': mode,
            'interval': _json_safe_value(interval),
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


def _apply_filter_settings_to_session(saved_settings, filter_hist_target_pct, top_fav_count):
    for k, v in (saved_settings or {}).items():
        if not isinstance(v, dict):
            continue
        st.session_state[f'filter_mode_{k}'] = v.get('mode', 'Av')
        interval = v.get('interval')
        if isinstance(interval, (list, tuple)) and len(interval) >= 2:
            range_key = f'filter_range_{k}_h{int(filter_hist_target_pct)}_tf{int(top_fav_count)}'
            # Behåll int/float-typen från filen. Streamlits int-sliders kan annars
            # bli griniga om de får float-värden i session_state.
            st.session_state[range_key] = (interval[0], interval[1])


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


def _build_filterpaket_payload(specs, group_reqs, filter_hist_target_pct, top_fav_count, spelform, antal_matcher):
    return {
        'file_type': 'tipset_filterpaket',
        'app_version': APP_VERSION,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'spelform': spelform,
        'antal_matcher': int(antal_matcher),
        'filter_hist_target_pct': int(filter_hist_target_pct),
        'top_fav_count': int(top_fav_count),
        'filters': _collect_filter_settings_for_save(specs, filter_hist_target_pct, top_fav_count),
        'group_reqs': _json_safe_value(group_reqs),
        'u_rows': _json_safe_value(_collect_u_rows_from_session(antal_matcher)),
    }


def _build_spelfil_payload(specs, group_reqs, filter_hist_target_pct, top_fav_count, spelform, antal_matcher, input_text, top_n, pay_min, frame, v_m, filter_vec, reducer_settings=None):
    payload = _build_filterpaket_payload(specs, group_reqs, filter_hist_target_pct, top_fav_count, spelform, antal_matcher)
    payload.update({
        'file_type': 'tipset_spelfil',
        'input_text': input_text or '',
        'top_n': int(top_n),
        'pay_min': int(pay_min),
        'frame': _json_safe_value(frame),
        'filter_vec': _json_safe_value(filter_vec or []),
        'history_records': _history_records_for_spelfil(v_m),
        'reducer_settings': _json_safe_value(reducer_settings or {}),
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
    # Grundinställningar: sätts tidigt i sidebar innan selectbox/text_area byggs.
    if payload.get('spelform'):
        st.session_state['v12_spelform'] = payload.get('spelform')
    if payload.get('input_text') is not None:
        st.session_state['v12_input_text'] = payload.get('input_text', '')
    if payload.get('top_n') is not None:
        st.session_state['v12_top_n'] = int(payload.get('top_n') or 30)
    if payload.get('pay_min') is not None:
        st.session_state['v12_pay_min'] = int(payload.get('pay_min') or 0)
    if payload.get('filter_hist_target_pct') is not None:
        st.session_state['v12_filter_hist_target_pct'] = int(payload.get('filter_hist_target_pct') or 90)
    if payload.get('top_fav_count') is not None:
        st.session_state['v12_top_fav_count'] = int(payload.get('top_fav_count') or 3)

    fhp = int(st.session_state.get('v12_filter_hist_target_pct', payload.get('filter_hist_target_pct', 90)) or 90)
    tfc = int(st.session_state.get('v12_top_fav_count', payload.get('top_fav_count', 3)) or 3)
    _apply_u_rows_to_session(payload.get('u_rows', {}))
    _apply_filter_settings_to_session(payload.get('filters', {}), fhp, tfc)
    _apply_group_reqs_to_session(payload.get('group_reqs', {}))

    if ftype == 'tipset_spelfil':
        frame = payload.get('frame') or []
        if frame:
            st.session_state['v12_saved_frame'] = [[s for s in normalize_signs(x)] for x in frame]
            st.session_state['v12_frame_saved'] = True
            st.session_state['v12_frame_defaults'] = st.session_state['v12_saved_frame']
            st.session_state['v12_frame_spelform'] = payload.get('spelform', st.session_state.get('v12_spelform'))
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
    if frame_rows is not None and len(frame_rows) <= 30000:
        for r in frame_rows:
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
        red_pct = 100 - 100 * len(pass_rows) / max(1, len(frame_rows))
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



def _combine_forced_candidate_masks(chosen):
    """Kombinerar tvingade filter till samlad historik-/grundramsmask."""
    chosen = list(chosen or [])
    if not chosen:
        return None, None
    try:
        htot = len(chosen[0]['hist_mask'])
        ftot = len(chosen[0]['frame_mask'])
        hist = np.ones(htot, dtype=bool)
        frame = np.ones(ftot, dtype=bool)
        for c in chosen:
            hist = hist & np.asarray(c['hist_mask'], dtype=bool)
            frame = frame & np.asarray(c['frame_mask'], dtype=bool)
        return hist, frame
    except Exception:
        return None, None


def _rebuild_forced_package_steps(chosen, htot, ftot):
    """Bygger om steg-tabellen efter att nivåer bytts i eftertrimningen."""
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


def _post_trim_forced_package_levels(chosen, candidates_by_key, frame, antal_matcher, row_matrix=None, max_passes=2):
    """Snävar åt valda filter efter paketbygget utan att sänka samlad träff.

    Exempel: om paketet har FAT Summa 19–30 och hela paketet ändå har 23/30,
    testas hårdare FAT Summa-nivåer. Om 19–26 fortfarande ger 23/30 men färre
    rader, byts nivån automatiskt. Detta görs bara för tvingade paket där varje
    filter är ett AND-villkor; grupppaket hanteras separat av gruppkravet.
    """
    chosen = [dict(c) for c in list(chosen or [])]
    if not chosen:
        return chosen, [], None, None
    base_hist, base_frame = _combine_forced_candidate_masks(chosen)
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
            cur_hist_mask, cur_frame_mask = _combine_forced_candidate_masks(chosen)
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
                new_hist, new_frame = _combine_forced_candidate_masks(trial)
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
    final_hist, final_frame = _combine_forced_candidate_masks(chosen)
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


def _build_grouped_package_for_target(candidates, target, frame_rows, frame, antal_matcher, max_filters=18, min_value_filters=3, required_keys=None, row_matrix=None):
    """Bygger ett paket med hårda grupper i stället för att tvinga varje filter.

    Målet är: fler viktiga filter, särskilt värde-/poängfilter, men med hårda gruppkrav
    så träffbilden inte rasar lika hårt som när allt är tvingat.
    """
    if not candidates:
        return None
    required_keys = set(required_keys or [])
    htot = len(candidates[0]['hist_mask'])
    ftot = len(candidates[0]['frame_mask'])
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

def _build_recommended_filter_packages(v_m, specs, frame_rows, frame, antal_matcher, hit_levels=None, min_step_reduction_pct=5.0, max_filters=14, min_hit_count=15, frame_adapt=True, min_value_filters=3, required_keys=None, target_frame_after=None, progress_cb=None):
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
    required_keys = set(required_keys or [])
    try:
        target_frame_after = int(target_frame_after) if target_frame_after is not None else None
    except Exception:
        target_frame_after = None
    min_hit_count = int(max(0, min(htot, min_hit_count)))
    min_value_filters = int(max(0, min(12, min_value_filters)))
    if hit_levels is None:
        # Gå hela vägen ner till vald lägstanivå, standard 15/30 när htot=30.
        hit_levels = list(range(htot, min_hit_count - 1, -1))
    hit_levels = sorted({int(max(0, min(htot, h))) for h in hit_levels if int(h) >= min_hit_count}, reverse=True)

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
                        'Orsak': reason,
                    })
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
            hist_hit = int(hist_mask.sum())
            frame_keep = int(frame_mask.sum())
            red_pct = 100.0 - 100.0 * frame_keep / max(1, ftot)
            if frame_keep >= ftot:
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
        cur_hist = np.ones(htot, dtype=bool)
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
                if new_frame_count >= cur_frame_count:
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
        )
        if trim_hist is not None and trim_frame is not None:
            cur_hist, cur_frame = trim_hist, trim_frame
            steps, _, _, _, _ = _rebuild_forced_package_steps(chosen, htot, ftot)
        else:
            post_trim_notes = []

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


def _package_value_score(p):
    """Spelvärdespoäng för rekommenderade paket.

    Poängen används bara för att välja de 3 mest intressanta paketen i listan:
    hög samlad historisk träff OCH hög reducering. Värde-/poängfilter får en liten bonus
    eftersom de bättre styr utdelningsprofilen än rena strukturfilter.
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

def _recommended_packages_summary_df(packages):
    rows = []
    for i, p in enumerate(packages, 1):
        group_txt = ''
        if p.get('groups'):
            group_txt = ' | '.join([f"{g.get('name')}: {g.get('req')}-{g.get('max_req', g.get('n'))}/{g.get('n')}" for g in p.get('groups', [])])
        rows.append({
            'Paket': f"P{i}",
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

def _apply_recommended_package_to_session(package, specs, filter_hist_target_pct, top_fav_count):
    chosen_keys = {c['key'] for c in package.get('filters', [])}
    chosen_by_key = {c['key']: c for c in package.get('filters', [])}
    for spec in specs:
        k = spec['key']
        range_key = f'filter_range_{k}_h{int(filter_hist_target_pct)}_tf{int(top_fav_count)}'
        if k in chosen_by_key:
            chosen = chosen_by_key[k]
            st.session_state[f'filter_mode_{k}'] = chosen.get('package_mode', 'Tvingat')
            st.session_state[range_key] = chosen['interval']
        else:
            st.session_state[f'filter_mode_{k}'] = 'Av'
            # Låt avstängda filter ligga på rekommenderat intervall för tydlighet.
            st.session_state[range_key] = spec.get('default_interval')
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
    }



# Init state
for k, v in {
    'v12_analysis_ready': False,
    'v12_frame_saved': False,
    'v12_filter_saved': False,
    'v12_last_result': None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Header
st.markdown(f"""
<div class='v12-hero'>
  <div class='v12-step'>Ren omstart</div>
  <div class='v12-title'>🎯 Tipset AI — Helgardering-lik filtercentral</div>
  <div class='v12-muted'>En grundram. Ett filter per rad. Av / Tvingat / Grupp. Statistik på varje filter när du öppnar info.</div>
  <div><span class='v12-pill'>{APP_VERSION}</span><span class='v12-pill'>Manuell filtercentral</span><span class='v12-pill'>Pareto-paket</span></div>
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

    if st.button("🧹 Rensa cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache tömd.")

# Step 1 – kupongdata / historik
st.markdown("<div class='v12-card'>", unsafe_allow_html=True)
st.markdown("<div class='v12-step'>Steg 1</div><div class='v12-title'>Kupongdata och historik</div>", unsafe_allow_html=True)
col_a, col_b, col_c = st.columns([1.2, 1, 1])
with col_a:
    spelform = st.selectbox("Spelform", ["Stryktips", "Europatips", "Topptips ST", "Topptips EU", "Topptips Övrigt", "Powerplay"], key="v12_spelform")
antal_matcher = 13 if spelform in ["Stryktips", "Europatips"] else 8
krav_odds = antal_matcher * 3
with col_b:
    top_n = st.number_input("Historikbas – liknande omgångar", min_value=20, max_value=100, value=30, step=5, key="v12_top_n", help="Rekommenderat: 30. 20 kan testas, men 30 ger stabilare filterstatistik.")
with col_c:
    pay_min = st.number_input("Min utdelning i historik", min_value=0, max_value=10000000, value=100000, step=50000, key="v12_pay_min")

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
        st.success(f"Historik klar: {len(st.session_state['v12_v_m'])} liknande omgångar från {st.session_state.get('v12_db_name','databas')}.")

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
            10000000,
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
        st.session_state['v12_filter_saved'] = False
        st.success(f"Klart: {len(v_m)} liknande omgångar hittades.")
st.markdown("</div>", unsafe_allow_html=True)

# Step 2 – ground frame
st.markdown("<div class='v12-card'>", unsafe_allow_html=True)
st.markdown("<div class='v12-step'>Steg 2</div><div class='v12-title'>Manuell grundram</div>", unsafe_allow_html=True)
st.caption("Tecknen ligger i ett formulär. Appen räknar inte om när du klickar 1/X/2, utan först när du sparar grundramen.")

if 'v12_frame_defaults' not in st.session_state or st.session_state.get('v12_frame_spelform') != spelform:
    st.session_state['v12_frame_defaults'] = [['1', 'X', '2'] for _ in range(antal_matcher)]
    st.session_state['v12_frame_spelform'] = spelform

with st.form("v12_frame_form"):
    header_cols = st.columns([0.55, 0.8, 0.8, 0.8, 1.2])
    header_cols[0].markdown("**Match**"); header_cols[1].markdown("**1**"); header_cols[2].markdown("**X**"); header_cols[3].markdown("**2**"); header_cols[4].markdown("**Val**")
    frame_new = []
    for i in range(antal_matcher):
        prev = st.session_state.get('v12_saved_frame', st.session_state['v12_frame_defaults'])
        signs_prev = prev[i] if i < len(prev) else ['1','X','2']
        c0, c1, c2, c3, c4 = st.columns([0.55, 0.8, 0.8, 0.8, 1.2])
        c0.write(f"M{i+1}")
        b1 = c1.checkbox("", value=('1' in signs_prev), key=f"v12_frame_{i}_1")
        bx = c2.checkbox("", value=('X' in signs_prev), key=f"v12_frame_{i}_x")
        b2 = c3.checkbox("", value=('2' in signs_prev), key=f"v12_frame_{i}_2")
        signs = []
        if b1: signs.append('1')
        if bx: signs.append('X')
        if b2: signs.append('2')
        c4.write(''.join(signs) if signs else '—')
        frame_new.append(signs)
    save_frame = st.form_submit_button("💾 Spara grundram", use_container_width=True)

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
    frame_rows, frame_n_rows, frame_ok, frame_msg = _cached_rows_from_frame(_frame_cache_tuple(frame), antal_matcher, max_rows=75000)
    if not frame_ok:
        st.error(frame_msg)
        st.stop()

    top_fav_count = int(st.session_state.get('v12_top_fav_count', 3))
    top_fav_count = max(1, min(6, top_fav_count))
    filter_hist_target_pct = int(st.session_state.get('v12_filter_hist_target_pct', 90))
    filter_hist_target_pct = max(50, min(100, filter_hist_target_pct))

    st.markdown("<div class='v12-card'>", unsafe_allow_html=True)
    st.markdown("<div class='v12-step'>Steg 3</div><div class='v12-title'>Filtercentral</div>", unsafe_allow_html=True)
    st.caption("Ett filter finns bara en gång. Välj Av, Tvingat eller Grupp. Du styr intervallen själv med sliders.")

    ctrl_a, ctrl_b = st.columns([1.2, 1.0])
    with ctrl_a:
        filter_hist_target_pct = st.slider(
            "Minsta historiska träff på filterintervall",
            min_value=50,
            max_value=100,
            value=filter_hist_target_pct,
            step=1,
            key="v12_filter_hist_target_pct",
            help="Styr rekommenderat intervall och startvärde för varje filter. 90% = ungefär 27/30 vid 30 historiska omgångar. 100% = intervallet täcker alla 30.",
        )
    with ctrl_b:
        top_fav_count = st.number_input("Toppfavoriter-filter: antal favoriter", min_value=1, max_value=6, value=top_fav_count, step=1, key="v12_top_fav_count", help="Styr filtret Topp N favoriter. Du kan välja 1–6.")
    filter_hist_target_pct = int(max(50, min(100, filter_hist_target_pct)))
    top_fav_count = int(max(1, min(6, top_fav_count)))

    # Utgångssystem byggs innan specs, så varje system kan bli fem filterrader
    # med Av/Tvingat/Grupp och min/max-intervall.
    for _slot, _defaults in {
        1: {'enabled': True, 'source': 'Favorit/procent', 'name': 'Utgångssystem 1'},
        2: {'enabled': True, 'source': 'Historik/AI', 'name': 'Utgångssystem 2'},
        3: {'enabled': False, 'source': 'Manuell', 'name': 'Utgångssystem 3'},
    }.items():
        st.session_state.setdefault(f'v12_us_{_slot}_enabled', _defaults['enabled'])
        st.session_state.setdefault(f'v12_us_{_slot}_source', _defaults['source'])
        st.session_state.setdefault(f'v12_us_{_slot}_name', _defaults['name'])
        st.session_state.setdefault(f'v12_us_{_slot}_text', '')
        for _m in range(antal_matcher):
            st.session_state.setdefault(f'v12_us_{_slot}_m{_m}', '')

    with st.expander("🎯 Utgångssystem – Antal tecken", expanded=False):
        st.caption("Helgardering-liknande U-filter: mata in ett utgångssystem med 1/X/2 per match. Appen skapar fem transparenta filter i filtercentralen: Antalet utgångstips, Antalet ettor, Antalet kryss, Antalet tvåor och Antal 1/X/2.")
        st.info("Filtervärdena väljs inte här. De visas under kategorin 'Utgångssystem – Antal tecken' i filtercentralen där varje rad kan sättas Av, Tvingat eller Grupp med min/max-intervall.")
        sign_choices = ['', '1', 'X', '2', '1X', '12', 'X2', '1X2']
        source_choices = ['Favorit/procent', 'Historik/AI', 'Andrahand', 'Manuell']
        for _slot in range(1, 4):
            with st.container(border=True):
                h1, h2, h3 = st.columns([0.9, 1.25, 1.5])
                with h1:
                    st.checkbox(f"Aktivera system {_slot}", key=f'v12_us_{_slot}_enabled')
                with h2:
                    st.selectbox("Källa", source_choices, key=f'v12_us_{_slot}_source', label_visibility='collapsed')
                with h3:
                    st.text_input("Namn", key=f'v12_us_{_slot}_name', label_visibility='collapsed')

                _src = st.session_state.get(f'v12_us_{_slot}_source', 'Manuell')
                if _src == 'Manuell':
                    st.caption("Manuellt system: tomt fält = matchen räknas inte i utgångssystemet. Klistra gärna som 13 grupper, t.ex. 1X / 1 / 12 / - / X2 ...")
                    txt_col, btn_col = st.columns([3.0, 1.0])
                    with txt_col:
                        st.text_input("Textgrundram", key=f'v12_us_{_slot}_text', placeholder='Exempel: 1X / 1 / 12 / - / X2 / 2 / 1 / 1X2 / X / 12 / 1 / X2 / 2', label_visibility='collapsed')
                    with btn_col:
                        if st.button("Läs text", key=f'v12_us_{_slot}_parse_text', use_container_width=True):
                            parsed, msg = parse_u_system_text(st.session_state.get(f'v12_us_{_slot}_text', ''), antal_matcher)
                            if parsed is None:
                                st.warning(msg)
                            else:
                                for _m, _signs in enumerate(parsed):
                                    st.session_state[f'v12_us_{_slot}_m{_m}'] = _u_signs_display(_signs)
                                st.rerun()
                    grid_cols = st.columns(antal_matcher)
                    for _m in range(antal_matcher):
                        with grid_cols[_m]:
                            st.selectbox(f"M{_m+1}", sign_choices, key=f'v12_us_{_slot}_m{_m}')
                else:
                    _preview_single = build_u_rows_for_filtercentral(v_m, filter_vec, antal_matcher)
                    _this = next((x for x in _preview_single if int(x.get('slot', 0)) == _slot), None)
                    if _this:
                        st.code(u_system_to_text(_this.get('system', [])), language=None)
                    else:
                        st.caption("Systemet visas när det är aktivt.")

        _preview_u_rows = build_u_rows_for_filtercentral(v_m, filter_vec, antal_matcher)
        if _preview_u_rows:
            st.dataframe(u_row_diag_df(_preview_u_rows, list(v_m['Correct_Row']), antal_matcher, target_pct=filter_hist_target_pct), use_container_width=True, hide_index=True)
            st.caption("Rek. intervall blir startvärde i filtercentralen. Garantin är villkorad på att facit överlever utgångsfilter och övriga filter.")
        else:
            st.info("Inga aktiva utgångssystem finns.")

    u_rows = build_u_rows_for_filtercentral(v_m, filter_vec, antal_matcher)

    # Viktigt: kontrollen ovan måste läsas INNAN specs/sliders byggs.
    # Dessutom har varje intervallslider en nyckel som innehåller träffmålet.
    # Annars återanvänder Streamlit gamla slider-värden och ignorerar nya defaultintervall.
    prev_target = st.session_state.get('v12_filter_hist_target_prev')
    prev_topfav = st.session_state.get('v12_top_fav_count_prev')
    if prev_target != filter_hist_target_pct or prev_topfav != top_fav_count:
        for _k in list(st.session_state.keys()):
            if str(_k).startswith('filter_range_'):
                del st.session_state[_k]
        st.session_state['v12_filter_hist_target_prev'] = filter_hist_target_pct
        st.session_state['v12_top_fav_count_prev'] = top_fav_count
        st.info("Filterintervallen uppdateras efter nytt träffmål/toppfavoritval…")
        st.rerun()

    specs = build_clean_filter_specs(v_m, filter_vec, antal_matcher, slider_u_count=top_fav_count, target_hist_pct=filter_hist_target_pct, u_rows=u_rows)
    st.session_state['v12_specs'] = specs
    st.caption("När du ändrar minsta historiska träff får varje filter nya slider-nycklar och startar på sitt rekommenderade intervall. 100% ska därför ge startintervall med 30/30 där det är möjligt.")

    with st.expander("🧠 Rekommenderade filterpaket", expanded=False):
        st.caption("Testar Pareto-bästa paket på din exakta grundram. Paketmotorn bygger nivåtrappa per filter, visar progressklocka/ETA, eftertrimmar valt paket och visar nu hårda grupppaket som egen pakettyp. Gruppresultatet visas alltid, även när det ligger över radgränsen eller inte kunde byggas.")

        with st.expander("Välj filter som måste ingå i rekommenderade paket", expanded=False):
            st.caption("Kryssa i filter du vill att paketmotorn ska använda. Motorn får själv avgöra om de passar bäst som tvingade filter eller i hårda grupper. Om ett ikryssat filter inte kan användas utan att träffbild/teckenskydd rasar visas inget paket på den nivån.")
            required_keys_now = []
            req_cats = []
            for _s in specs:
                if _s.get('category') not in req_cats:
                    req_cats.append(_s.get('category'))
            for _cat in req_cats:
                st.markdown(f"**{_cat}**")
                _cat_specs = [_s for _s in specs if _s.get('category') == _cat]
                _cols = st.columns(3)
                for _i, _spec in enumerate(_cat_specs):
                    with _cols[_i % 3]:
                        if st.checkbox(_spec.get('name',''), key=f"v12_reqpkg_{_spec.get('key')}"):
                            required_keys_now.append(_spec.get('key'))
            if required_keys_now:
                st.success(f"{len(required_keys_now)} filter är markerade som måste ingå i paketmotorn.")
            else:
                st.info("Inga obligatoriska filter valda. Paketmotorn söker fritt.")
        required_keys_now = [s.get('key') for s in specs if st.session_state.get(f"v12_reqpkg_{s.get('key')}", False)]

        rp_c1, rp_c2, rp_c3, rp_c4, rp_c5, rp_c6, rp_c7 = st.columns([1, 1, 1, 1, 1, 1, 1])
        with rp_c1:
            rec_min_step = st.number_input(
                "Minsta extra reducering",
                min_value=0.5,
                max_value=20.0,
                value=5.0,
                step=0.5,
                key="v12_rec_min_step",
                help="Gäller främst efter att värdekärnan är byggd. Värde-/poängfilter som behövs för kvoten får läggas till med lägre marginalkrav om de håller samlad träff.",
            )
        with rp_c2:
            rec_max_filters = st.number_input("Max filter i paket", min_value=1, max_value=40, value=18, step=1, key="v12_rec_max_filters")
        with rp_c3:
            rec_min_hit = st.number_input("Sök ner till träff", min_value=1, max_value=int(len(v_m)), value=min(15, int(len(v_m))), step=1, key="v12_rec_min_hit", help="Exempel: 15 betyder att paketmotorn söker ända ner till 15/30 när historikbasen är 30.")
        with rp_c4:
            rec_display_max_rows = st.number_input("Visa paket under", min_value=100, max_value=75000, value=5000, step=100, key="v12_rec_display_max_rows", help="Paket över denna filtermassa döljs i huvudlistan. Standard 5 000 eftersom bredare paket oftast är opraktiska inför TipsetMatrix.")
        with rp_c5:
            rec_frame_adapt = st.checkbox(
                "Anpassa mot grundram",
                value=True,
                key="v12_rec_frame_adapt",
                help="Undviker paketfilter som ligger för nära grundramens yttergränser. Gäller även FAT-sekvenser som blir grundramsdrivna, t.ex. när ramen redan låser minst 3 sekvensträffar och filtret bara kapar max.",
            )
        with rp_c6:
            rec_min_value_filters = st.number_input(
                "Min värde-/poängfilter",
                min_value=0,
                max_value=10,
                value=3,
                step=1,
                key="v12_rec_min_value_filters",
                help="Hård spärr: ett rekommenderat paket måste innehålla minst detta antal filter från Värde & svårighet. Annars visas paketet inte i listan.",
            )
        with rp_c7:
            build_recs = st.button("Beräkna paket", use_container_width=True, key="v12_build_recommended_packages")
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
                v_m, specs, frame_rows, frame, antal_matcher,
                min_step_reduction_pct=float(rec_min_step),
                max_filters=int(rec_max_filters),
                min_hit_count=int(rec_min_hit),
                frame_adapt=bool(rec_frame_adapt),
                min_value_filters=int(rec_min_value_filters),
                required_keys=required_keys_now,
                target_frame_after=int(rec_display_max_rows),
                progress_cb=_ui_package_progress,
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
                'package_engine': 'pareto_multilevel_progress_posttrim_group_visible_diagnostics',
                'manual_hist_target_pct': int(filter_hist_target_pct),
                'top_fav_count': int(top_fav_count),
                'frame_rows': int(len(frame_rows)),
                'min_step': float(rec_min_step),
                'max_filters': int(rec_max_filters),
                'min_hit': int(rec_min_hit),
                'display_max_rows': int(rec_display_max_rows),
                'frame_adapt': bool(rec_frame_adapt),
                'min_value_filters': int(rec_min_value_filters),
                'required_keys': list(required_keys_now),
            }
        packages = st.session_state.get('v12_recommended_packages') or []
        rec_display_max_rows = int(st.session_state.get('v12_rec_display_max_rows', 5000))
        visible_packages = [p for p in packages if int(p.get('frame_after', 10**12)) <= rec_display_max_rows]
        hidden_packages = [p for p in packages if int(p.get('frame_after', 10**12)) > rec_display_max_rows]
        if packages:
            st.caption(f"Visar bara paket där filtermassan är högst {rec_display_max_rows:,} rader. {len(hidden_packages)} paket är dolda över gränsen.".replace(',', ' '))
            if visible_packages:
                top_packages = _top_playable_packages(visible_packages, 3)
                st.markdown("**3 mest spelvärda paket**")
                st.caption("Sorterat på kombinationen samlad historisk träff + faktisk reducering. Paket med fler värde-/poängfilter får en liten bonus eftersom de styr utdelningsprofilen bättre.")
                st.dataframe(_recommended_packages_summary_df(top_packages), use_container_width=True, hide_index=True)

                # v12.0ae: visa grupppaketsektionen alltid, även om inga grupppaket
                # ligger under radgränsen. Annars ser funktionen ut att saknas.
                st.markdown("**Bästa hårda grupppaket**")
                st.caption("Visas separat eftersom grupppaket är en annan riskprofil: fler filter kan ingå men alla måste inte sitta samtidigt. Sektionen visas även när bästa grupppaket ligger över radgränsen.")
                group_status_df = _group_packages_status_df(packages, rec_display_max_rows, max_rows=6)
                if not group_status_df.empty:
                    st.dataframe(group_status_df, use_container_width=True, hide_index=True)
                    st.caption(_group_packages_status_text(packages, rec_display_max_rows))
                else:
                    st.info(_group_packages_status_text(packages, rec_display_max_rows))

                best_group_packages = _top_playable_packages([p for p in visible_packages if _is_hard_group_package(p)], 3)
                rest_packages = [p for p in visible_packages if p not in top_packages and p not in best_group_packages]
                if rest_packages:
                    with st.expander("Visa övriga spelbara paket under radgränsen", expanded=False):
                        st.dataframe(_recommended_packages_summary_df(rest_packages), use_container_width=True, hide_index=True)
            else:
                top_packages = []
                st.warning("Inga paket hamnade under vald radgräns. Visar därför bästa paketet över gränsen som referens så att listan inte blir tom.")
                if hidden_packages:
                    best_over = _top_playable_packages(hidden_packages, 1)
                    if best_over:
                        st.markdown("**Bästa paket över radgränsen**")
                        st.dataframe(_recommended_packages_summary_df(best_over), use_container_width=True, hide_index=True)
                        st.caption("Detta paket klarar dina krav bäst men lämnar fler rader än vald gräns. Höj radgränsen eller kryssa i fler/lämpligare filter om du vill pressa vidare.")
                st.markdown("**Bästa hårda grupppaket**")
                st.caption("Sektionen visas även när inga grupppaket är spelbara under radgränsen, så du ser om gruppmotorn faktiskt byggde något.")
                group_status_df = _group_packages_status_df(packages, rec_display_max_rows, max_rows=6)
                if not group_status_df.empty:
                    st.dataframe(group_status_df, use_container_width=True, hide_index=True)
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
            if visible_packages:
                _sel_base = _top_playable_packages(visible_packages, 3)
                _sel_groups = _top_playable_packages([p for p in visible_packages if _is_hard_group_package(p)], 3)
                selectable_packages = _dedupe_package_list(_sel_base + _sel_groups)
            else:
                _sel_base = _top_playable_packages(hidden_packages, 1)
                _sel_groups = _top_playable_packages([p for p in hidden_packages if _is_hard_group_package(p)], 2)
                selectable_packages = _dedupe_package_list(_sel_base + _sel_groups)
            if not visible_packages and selectable_packages:
                st.info("Du kan använda bästa paketet över radgränsen, men filtermassan blir bredare än din valda gräns.")
            if not selectable_packages:
                sel_pkg = None
            else:
                labels = [f"{p['hist_hit']}/{p['hist_total']} · {p['frame_start']:,}→{p['frame_after']:,} · {p['reduction_pct']:.1f}% · {p['num_filters']} filter · {p.get('value_filters',0)} värde · {p.get('package_type','Tvingat')}".replace(',', ' ') for p in selectable_packages]
                pick_idx = st.selectbox("Välj toppaket att använda", list(range(len(selectable_packages))), format_func=lambda i: labels[i], key="v12_recommended_pick")
                sel_pkg = selectable_packages[int(pick_idx)]
            if sel_pkg is not None:
                with st.expander("Visa filter och intervall i valt paket", expanded=False):
                    if sel_pkg.get('steps'):
                        st.dataframe(pd.DataFrame(sel_pkg['steps']), use_container_width=True, hide_index=True)
                        st.caption("Intervallträff är hur många av de liknande historiska omgångarna det enskilda filterintervallet klarar. Samlad träff efter steg är hela paketets träff efter att filtret lagts till.")
                        if sel_pkg.get('post_trim_notes'):
                            st.markdown("**Eftertrimning utan tappad samlad träff**")
                            st.dataframe(pd.DataFrame(sel_pkg.get('post_trim_notes') or []), use_container_width=True, hide_index=True)
                            st.caption("Eftertrim betyder att appen hittade ett snävare intervall som gav färre rader men behöll samma samlade historikträff för paketet.")
                    else:
                        st.info("Detta paket hittade inga filter som gav tillräcklig extra reducering inom träffmålet.")
                if st.button("✅ Använd valt paket i filtercentralen", use_container_width=True, key="v12_apply_recommended_package"):
                    _apply_recommended_package_to_session(sel_pkg, specs, filter_hist_target_pct, top_fav_count)
                    st.success("Paketet har lagts in i filtercentralen.")
                    st.rerun()
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
    cats = []
    for s in specs:
        if s['category'] not in cats:
            cats.append(s['category'])

    settings = {}
    for ci, cat in enumerate(cats):
        with st.expander(f"📂 {cat}", expanded=(ci == 0)):
            cat_specs = [s for s in specs if s['category'] == cat]
            for spec in cat_specs:
                k = spec['key']
                mode_key = f'filter_mode_{k}'
                # Viktigt: range_key innehåller träffmål/toppfavoritval.
                # Streamlit återanvänder annars gamla slider-värden från session_state och ignorerar nytt default-value.
                # Det var därför "Rek. intervall" kunde vara 95/100%, medan "Nu valt" låg kvar på gammalt 90%-intervall.
                range_key = f'filter_range_{k}_h{filter_hist_target_pct}_tf{top_fav_count}'
                default_mode = st.session_state.get(mode_key, 'Av')
                lo, hi = spec['bounds']
                dec = spec['decimals']
                default_interval = _current_interval_for_spec(spec)
                try:
                    val = (max(lo, float(default_interval[0])), min(hi, float(default_interval[1])))
                except Exception:
                    val = spec['default_interval']
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
                    st.caption(f"Rek: {_display_interval(spec['default_interval'], dec)} · {hp}/{ht}")
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
                    # Popover öppnas/stängs i klienten och triggar normalt inte en full sidkörning,
                    # till skillnad från en vanlig knapp. Statistik visas därmed nära filtret utan
                    # separat scrollsektion och utan att Auto/TipsetMatrix körs om.
                    if hasattr(st, "popover"):
                        with st.popover("ℹ️", help="Visa frekvenstabell, träff och reducering för detta filter"):
                            _render_inline_filter_info(spec, rng, frame_rows, frame, antal_matcher)
                    else:
                        with st.expander("ℹ️", expanded=False):
                            _render_inline_filter_info(spec, rng, frame_rows, frame, antal_matcher)
                settings[k] = {'mode': mode, 'interval': rng}
                st.divider()
    st.markdown("---")
    st.markdown("**Gruppkrav – min/max**")
    st.caption("Min/max styr hur många filter i respektive grupp som måste träffa. Exempel: 5–7 av 8 betyder att raden ska klara minst 5 men inte nödvändigtvis alla 8. Max kan även användas för att stoppa för extrema rader som klarar för många filter i samma familj.")
    gc = st.columns(6)
    group_reqs = {}
    for i in range(1, 7):
        gname = f'Grupp {i}'
        n_in_group = sum(1 for v in settings.values() if v.get('mode') == gname)
        old_req = int(st.session_state.get(f'group_req_{i}', 0) or 0)
        default_min = int(st.session_state.get(f'group_req_min_{i}', old_req) or 0)
        default_max = int(st.session_state.get(f'group_req_max_{i}', max(1, n_in_group)) or max(1, n_in_group))
        default_min = max(0, min(default_min, max(40, n_in_group)))
        default_max = max(default_min, min(default_max, max(40, n_in_group)))
        with gc[i-1]:
            st.caption(f"{gname} · {n_in_group} filter")
            mn = st.number_input(
                "Min",
                min_value=0,
                max_value=40,
                value=default_min,
                step=1,
                key=f"group_req_min_{i}",
                help="0 = inget min-krav. Gruppen påverkar bara om min eller max faktiskt begränsar.",
            )
            mx = st.number_input(
                "Max",
                min_value=0,
                max_value=40,
                value=default_max,
                step=1,
                key=f"group_req_max_{i}",
                help="Max antal filter som får träffa i gruppen. Sätt max till antal filter i gruppen om du inte vill ha övre spärr.",
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

    active_count = sum(1 for v in settings.values() if v['mode'] != 'Av')
    forced_count = sum(1 for v in settings.values() if v['mode'] == 'Tvingat')
    group_count = active_count - forced_count
    hpkg, htot = _hist_package_passes(v_m, specs, settings, group_reqs)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Aktiva filter", active_count)
    m2.metric("Tvingade", forced_count)
    m3.metric("Gruppfilter", group_count)
    m4.metric("Samlad historikträff", f"{hpkg}/{htot}")

    applied_meta = st.session_state.get('v12_applied_package_meta')
    if applied_meta and active_count:
        current_sig = _settings_package_signature(settings, group_reqs)
        if current_sig == applied_meta.get('signature'):
            pkg_txt = f"{int(applied_meta.get('hist_hit', 0))}/{int(applied_meta.get('hist_total', 0))}"
            cur_txt = f"{hpkg}/{htot}"
            if int(applied_meta.get('hist_hit', -1)) == int(hpkg) and int(applied_meta.get('hist_total', -1)) == int(htot):
                st.success(f"Paketkontroll OK: paketet och aktiv filtercentral visar samma samlade träff ({cur_txt}).")
            else:
                st.error(f"Paketkontroll mismatch: paketet räknades som {pkg_txt}, men aktiv filtercentral visar {cur_txt}.")
        else:
            st.caption("Filtercentralen har ändrats efter att paketet applicerades; paketkontrollen är därför inte längre aktiv.")

    if active_count:
        st.caption("Samlad historikträff är hela aktiva paketet räknat mot de liknande historiska omgångarna: alla Tvingade filter måste sitta och varje Grupp måste klara sitt min/max-krav. Det är inte ett snitt av individuella filterträffar.")
        active_u_count = sum(1 for s in specs if s.get('category') == 'Utgångssystem – Antal tecken' and settings.get(s.get('key'), {}).get('mode') != 'Av')
        if active_u_count:
            st.info(f"{active_u_count} utgångssystemfilter är aktiva. Reduceringsgarantin är villkorad på att facit klarar utgångssystemens intervall och övriga filter innan TipsetMatrix körs.")
    if active_count and hpkg <= max(3, int(0.25 * max(1, htot))):
        st.warning("Samlad historikträff är mycket låg. Det betyder oftast att för många filter är Tvingade samtidigt. Lägg över närliggande filter i Grupp eller bredda intervallen.")

    with st.expander("🔬 Samlad historikträff – rad-för-rad diagnos", expanded=False):
        if active_count:
            diag_df = _active_package_diagnostic_df(v_m, specs, settings, group_reqs, antal_matcher, max_rows=htot)
            if not diag_df.empty:
                st.dataframe(diag_df, use_container_width=True, hide_index=True)
                st.caption("✅ Träff betyder att den historiska vinnarraden klarade hela aktiva filterpaketet. ❌ Miss visar vilket tvingat filter eller gruppkrav som stoppade raden.")
            else:
                st.info("Ingen historikdiagnos kunde byggas.")
        else:
            st.info("Aktivera minst ett filter för att se samlad historikdiagnos.")

    with st.expander("🧩 Gruppdiagnos – min/max och faktisk effekt", expanded=False):
        group_diag_df = _active_group_diagnostic_df(specs, settings, group_reqs, frame_rows=frame_rows)
        if not group_diag_df.empty:
            st.dataframe(group_diag_df, use_container_width=True, hide_index=True)
            st.caption("Gruppdiagnosen räknar gruppens min/max-krav separat från tvingade filter. Grundramseffekt visas exakt när grundramen är högst 30 000 rader.")
        else:
            st.info("Inga filter ligger i Grupp 1–6 just nu.")

    with st.expander("📋 Filteröversikt", expanded=False):
        st.dataframe(_build_filter_summary_df(specs, settings, group_reqs, rows=frame_rows), use_container_width=True, hide_index=True)

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
    filter_payload = _build_filterpaket_payload(specs, group_reqs, filter_hist_target_pct, top_fav_count, spelform, antal_matcher)
    game_payload = _build_spelfil_payload(specs, group_reqs, filter_hist_target_pct, top_fav_count, spelform, antal_matcher, input_text, top_n, pay_min, frame, v_m, filter_vec, reducer_settings=reducer_save_settings)
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

    go = st.button("🚀 Kör filtrering + reducering", use_container_width=True)

    if go:
        with st.spinner("Filtrerar exakt grundram..."):
            filtered_rows = _apply_manual_filters(frame_rows, specs, settings, group_reqs)
        st.session_state['v12_last_result'] = {'filtered_rows': filtered_rows, 'reduced_rows': [], 'settings': settings, 'group_reqs': group_reqs, 'hist_package': {'hit': int(hpkg), 'total': int(htot)}}
        st.success(f"Filtrering klar: {len(frame_rows):,} → {len(filtered_rows):,} rader.".replace(',', ' '))
        miss = selected_signs_missing(filtered_rows, frame, antal_matcher)
        if miss:
            st.warning("Vissa markerade tecken saknas efter filter: " + format_missing_signs(miss))
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
        st.caption("Skriv in rätt rad manuellt. Rättningen använder senast körda grundram, filtermassa och TipsetMatrix-rader — den kör inte om filter eller reducering.")
        with st.form("v12_correction_form", clear_on_submit=False):
            corr_txt = st.text_input("Rätt rad", value=st.session_state.get('v12_correction_input', ''), placeholder="Exempel: 1X2X1122X... eller 1,X,2,X,...")
            corr_submit = st.form_submit_button("Rätta system")
        if corr_submit:
            rr, err = parse_result_row(corr_txt, antal_matcher)
            if err:
                st.error(err)
                st.session_state['v12_correction_row'] = ''
            else:
                st.session_state['v12_correction_input'] = corr_txt
                st.session_state['v12_correction_row'] = rr
        corr_row = st.session_state.get('v12_correction_row')
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
