import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import itertools
import bisect
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Tipset AI-Analys", layout="wide", page_icon="🎯")
APP_VERSION = "v11.1w – Alla poängfilter som spetsfilter"


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
    """Log-sannolikhet enligt dagens procent. Används bara som vikt, inte som facit."""
    total = 0.0
    for i, c in enumerate(row_str):
        idx = i * 3
        if c == '1': p = prob_vector[idx]
        elif c == 'X': p = prob_vector[idx + 1]
        else: p = prob_vector[idx + 2]
        total += np.log(max(float(p), 0.05) / 100.0)
    return float(total)


def weighted_13_share(all_rows, selected_rows, prob_vector):
    """Viktad 13-chans inom filtrerad radmassa, baserad på dagens procentvektor."""
    if not all_rows or not selected_rows:
        return 0.0
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


def build_tipsetmatrix_guarantee_table(filtered_rows, reduced_rows, antal_matcher, prob_vector=None):
    """Bygger garantitabell: varje filtrerad rad testas som möjlig facitrad."""
    filtered_rows = list(dict.fromkeys(filtered_rows))
    reduced_rows = list(dict.fromkeys(reduced_rows))
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
# STREAMLIT UI & SIDEBAR
# ==========================================

col_header1, col_header2 = st.columns([2, 1])
with col_header1:
    st.title("🎯 Tipset AI-Analys")
    st.caption(f"Version: {APP_VERSION}")
with col_header2:
    spelform = st.selectbox("⚽ Välj Spelform:", [
        "Stryktips", "Europatips", "Topptips ST", "Topptips EU", "Topptips Övrigt", "Powerplay"
    ])

antal_matcher = 13 if spelform in ["Stryktips", "Europatips"] else 8
krav_odds = antal_matcher * 3

st.markdown(f"**Läge:** Du har valt **{spelform}** ({antal_matcher} matcher).")
st.markdown("---")

if 'aktuell_spelform' not in st.session_state:
    st.session_state['aktuell_spelform'] = spelform
if st.session_state['aktuell_spelform'] != spelform:
    st.session_state['har_kort_analys'] = False
    st.session_state['aktuell_spelform'] = spelform

# --- SIDEBAR (REN KONTROLLPANEL) ---
with st.sidebar:
    st.header("🎛️ Kontrollpanel")
    st.caption("Normal-läget styrs av utdelningsnivå och samlad mallträff. Appen väljer intervaller, basfilter, spetsfilter och TipsetMatrix-inställningar själv.")

    if st.button("🧹 Töm minne / rensa cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cachen tömd. Ladda om sidan vid behov.")

    profile_defaults = {
        "Försiktig": {"top_n": 40, "core_val": 95, "core_str": 100, "automin": 90, "group": 90, "pass_ratio": 0.80, "super": 3},
        "Rekommenderad": {"top_n": 30, "core_val": 90, "core_str": 100, "automin": 85, "group": 90, "pass_ratio": 0.88, "super": 4},
        "Hård": {"top_n": 30, "core_val": 85, "core_str": 95, "automin": 80, "group": 85, "pass_ratio": 0.97, "super": 5},
        "Expert": {"top_n": 30, "core_val": 90, "core_str": 100, "automin": 85, "group": 90, "pass_ratio": 0.90, "super": 4},
    }
    p_opts = [0, 500, 1000, 2500, 5000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000, 1000000, 2500000, 5000000, 10000000]

    st.markdown("---")
    expert_mode = st.toggle("🔧 Expertläge", value=False, help="Av = normalspel med en huvudkontroll. På = visa alla tekniska reglage.")

    # Standard: allt på. Expertläget kan användas för felsökning.
    cb_payout = True
    cb_u_favs = True
    cb_sft = True
    cb_fat = True
    cb_fat_sequences = True
    cb_points = True
    cb_100minus = True
    cb_log_surprise = True
    cb_rank24 = True
    cb_totaldiff = True
    cb_fav_pressure = True
    cb_shock_strength = True
    cb_fav_delta = True
    cb_base = True
    cb_streak = True
    cb_gap = True
    cb_single = True
    cb_doublet = True
    cb_triplet = True
    cb_occur = True
    cb_super_macro = True
    cb_aimatrix = True
    cb_manual_ai_rank = False
    cb_structure = True
    cb_pro_groups = True
    cb_filter_revision = True
    cb_group_risk = True
    cb_group_shock = True
    cb_group_fat = True
    cb_group_structure = True
    cb_ai_frame = False

    max_rank = 3**antal_matcher

    if not expert_mode:
        # Normalspel: Niklas-spåret.
        # En historisk mall: hög utdelning + liknande procentbild + samlad mallträff.
        defaults = profile_defaults["Rekommenderad"]
        slider_top_n = 30
        st.subheader("Spelprofil")
        payout_min_normal = st.select_slider(
            "Historik: minsta utdelning",
            options=p_opts,
            value=100000,
            help="Appen jämför bara mot historiska omgångar där 13-rättsutdelningen är minst denna nivå. Standard: 100 000 kr+."
        )
        slider_payout = (int(payout_min_normal), p_opts[-1])
        target_hits_normal = st.slider(
            "Samlad mallträff av 30",
            15, 30, 20, step=1,
            help="Appen väljer bara filterpaket där minst detta antal av de 30 mest lika historiska högutdelningsomgångarna överlever."
        )
        slider_combined_min_hist_pct = (100.0 * float(target_hits_normal) / float(slider_top_n)) if slider_top_n else 67.0
        st.caption(f"Mål: minst {target_hits_normal}/{slider_top_n} historiska mallrader inom {int(payout_min_normal):,} kr+.")
        tm_min_13_chance_pct = st.slider(
            "Minsta 13-chans efter reducering",
            0, 35, 20, step=1,
            help="Om TipsetMatrix 12-garanti ger lägre 13-chans än detta lägger appen till högst rankade rader från den filtrerade radmassan tills målet nås. 0 = avstängt."
        )

        # Fasta rationella standardvärden i normal-läge.
        slider_core_val = int(defaults["core_val"])
        slider_core_str = int(defaults["core_str"])
        cb_autotrim = True
        slider_autotrim_min_hist = int(defaults["automin"])
        slider_macro_target = 90
        slider_super_groups = int(defaults["super"])
        slider_group_target = int(defaults["group"])
        slider_u_count = min(3, antal_matcher)
        slider_frame_budget = 25000
        slider_frame_max_full = min(4, antal_matcher)
        slider_u_target = 90

        # TipsetMatrix standard i normal-läge: klickbar ram och full 12-garanti.
        cb_tipsetmatrix = True
        tm_frame_source = "Klickbar grundram"
        tm_base_limit = 25000
        tm_filter_limit = 5000
        tm_output_limit = 300
        tm_guarantee_mode = True
        tm_mode = "Balans"
        tm_weighting = "Filterpoäng + sannolikhet"
        tm_seed = 42
        cb_tm_backtest = False

        st.markdown("---")
        st.success("Normal-läge: appen väljer bästa intervaller, hårda basfilter och hårda spetsfilter automatiskt utifrån träffbilden.")

    else:
        st.subheader("1) AutoFilter")
        filter_profile = st.selectbox(
            "Filterläge",
            ["Rekommenderad", "Försiktig", "Hård", "Expert"],
            index=0,
            help="Expertläge visar de gamla tekniska reglagen. Normal-läge använder bara utdelning + samlad mallträff."
        )
        defaults = profile_defaults[filter_profile]

        slider_top_n = st.slider("Liknande historiska omgångar", 5, 100, int(defaults["top_n"]), step=5)
        slider_combined_min_hist_pct = st.slider(
            "Min samlad mallträff %",
            50, 95, 67, step=1,
            help="AutoHard Spetsfilter väljer hårda basfilter och lägger bara till spetsfilter som både klarar denna historiska träff och reducerar veckans radmassa tydligt."
        )
        slider_core_val = st.slider("Värdeintervall / kärna %", 40, 100, int(defaults["core_val"]), step=5)
        slider_core_str = st.slider("Strukturintervall / kärna %", 40, 100, int(defaults["core_str"]), step=5)
        cb_autotrim = st.toggle("AutoTrim: välj rationella intervall", value=True)
        slider_autotrim_min_hist = st.slider("AutoTrim: min historisk träff %", 70, 100, int(defaults["automin"]), step=5)
        slider_payout = st.select_slider("Utdelningsspann i historik", options=p_opts, value=(100000, p_opts[-1]))

        st.markdown("---")
        st.subheader("2) Reducering")
        slider_macro_target = st.slider("Super-Makro historisk träff %", 40, 100, 90, step=5)
        slider_super_groups = st.slider("Super-Makro krav", 1, 8, int(defaults["super"]), step=1, help="Kravet är minst X av 8 struktur-/FAT-grupper.")
        slider_group_target = st.slider("Pro-gruppmål historisk träff %", 70, 100, int(defaults["group"]), step=5)
        slider_u_count = st.slider("Toppfavoriter / U-tecken", 1, antal_matcher, min(3, antal_matcher), step=1)
        frame_budget_options = [864, 1728, 3456, 6912, 10368, 15552, 20736, 25000, 31104, 50000, 75000]
        slider_frame_budget = st.select_slider("AI-Balansram max radantal", options=frame_budget_options, value=25000)
        slider_frame_max_full = st.slider("AI-Balansram max helgarderingar", 0, antal_matcher, min(4, antal_matcher), step=1)
        slider_u_target = st.slider("U-rad mål historisk träff %", 70, 100, 90, step=5)

        st.markdown("---")
        st.subheader("3) TipsetMatrix 12")
        cb_tipsetmatrix = True
        tm_frame_source = st.selectbox("Grundram", ["Klickbar grundram", "AI-Balansram", "Textgrundram"], index=0)
        tm_base_limit = st.select_slider("Max rader i grundram", options=[1000, 2500, 5000, 10000, 15000, 25000, 50000, 75000], value=25000)
        tm_filter_limit = st.select_slider("Max rader efter filter", options=[500, 1000, 1500, 2500, 5000, 10000, 20000], value=5000)
        tm_output_limit = st.select_slider("Max reducerade rader i budgetläge", options=[50, 100, 150, 200, 300, 500, 750, 1000], value=300)
        tm_guarantee_mode = st.toggle("Kör till full 12-garanti", value=True, help="På = motorn stannar inte vid maxrader, utan kör tills hela filtrerade radmassan är 12-täckt. Av = budgetläge med maxrader.")
        tm_min_13_chance_pct = st.slider(
            "Minsta 13-chans efter reducering %",
            0, 50, 20, step=1,
            help="Lägger till högst rankade rader från filtermassan om 13-chansen efter TipsetMatrix blir lägre än målet. 0 = avstängt."
        )
        tm_mode = st.selectbox("Motorläge", ["Snabb", "Balans", "Max"], index=1)
        tm_weighting = st.selectbox("Viktning", ["Filterpoäng + sannolikhet", "Filterpoäng", "Neutral"], index=0)
        tm_seed = st.number_input("Seed", min_value=1, max_value=999999, value=42, step=1)
        cb_tm_backtest = st.toggle("Visa rättning/facitpanel", value=False)

        with st.expander("Aktiva filter", expanded=False):
            cb_payout = st.checkbox("Utdelning", value=cb_payout)
            cb_u_favs = st.checkbox("Topp-Favoriter", value=cb_u_favs)
            cb_sft = st.checkbox("SFT Summa", value=cb_sft)
            cb_fat = st.checkbox("FAT-tabell", value=cb_fat)
            cb_fat_sequences = st.checkbox("FAT-sekvenser", value=cb_fat_sequences)
            cb_points = st.checkbox("Poängfilter", value=cb_points)
            cb_100minus = st.checkbox("100-minus", value=cb_100minus)
            cb_log_surprise = st.checkbox("Skrälltryck Log", value=cb_log_surprise)
            cb_rank24 = st.checkbox(f"Rank 1-{krav_odds} Summa", value=cb_rank24)
            cb_totaldiff = st.checkbox("Total Diff", value=cb_totaldiff)
            cb_fav_pressure = st.checkbox("Favorittryck", value=cb_fav_pressure)
            cb_shock_strength = st.checkbox("Skrällstyrka", value=cb_shock_strength)
            cb_fav_delta = st.checkbox("Favorit-delta", value=cb_fav_delta)
            cb_aimatrix = st.checkbox("AI-Matrix Rank", value=cb_aimatrix)
            cb_manual_ai_rank = st.checkbox("Styr AI-Rank manuellt", value=False)
            if cb_manual_ai_rank:
                slider_ai_rank = st.slider("AI-Rank Slider", 1, max_rank, (1, max_rank))
            cb_structure = st.checkbox("Matcha struktur viktat", value=cb_structure)
        with st.expander("Strukturfilter", expanded=False):
            cb_base = st.checkbox("Tecken 1X2", value=cb_base)
            cb_streak = st.checkbox("Sviter", value=cb_streak)
            cb_gap = st.checkbox("Luckor", value=cb_gap)
            cb_single = st.checkbox("Singlar", value=cb_single)
            cb_doublet = st.checkbox("Dubbletter", value=cb_doublet)
            cb_triplet = st.checkbox("Tripplar", value=cb_triplet)
            cb_occur = st.checkbox("Uppkomster", value=cb_occur)
            cb_super_macro = st.checkbox("Super-Makro", value=cb_super_macro)
        with st.expander("Visning", expanded=False):
            cb_pro_groups = st.checkbox("Visa pro-grupper", value=cb_pro_groups)
            cb_filter_revision = st.checkbox("Visa filterrevision", value=cb_filter_revision)
            cb_ai_frame = st.checkbox("Visa AI-Ram & U-filter", value=cb_ai_frame)
            cb_group_risk = st.checkbox("Riskgrupp", value=cb_group_risk)
            cb_group_shock = st.checkbox("Skrällgrupp", value=cb_group_shock)
            cb_group_fat = st.checkbox("FAT-profilgrupp", value=cb_group_fat)
            cb_group_structure = st.checkbox("Strukturgrupp", value=cb_group_structure)

    active_filters_list = [
        cb_u_favs, cb_sft, cb_fat, cb_fat_sequences, cb_points, cb_100minus, cb_log_surprise, cb_rank24, cb_totaldiff,
        cb_fav_pressure, cb_shock_strength, cb_fav_delta,
        cb_base, cb_streak, cb_gap, cb_single, cb_doublet, cb_triplet, cb_occur,
        cb_super_macro, cb_aimatrix
    ]
    total_active = sum(active_filters_list)
    if total_active > 0:
        auto_pass_req = max(1, min(total_active, int(np.ceil(total_active * float(defaults["pass_ratio"])))) )
    else:
        auto_pass_req = 0

    # Diagnostiskt poängkrav används bara för analys/viktning. Aktiv filtrering görs med AutoHard Spetsfilter.
    manual_soft_req = False
    if expert_mode and total_active > 0:
        manual_soft_req = st.checkbox("Styr diagnostiskt poängkrav manuellt", value=False)
        if manual_soft_req:
            slider_pass_req = st.slider("Diagnostisk filterpoäng: minsta antal uppfyllda krav", 1, total_active, auto_pass_req)
        else:
            slider_pass_req = auto_pass_req
            st.metric("Diagnostisk filterpoäng", f"{slider_pass_req} av {total_active}")
    else:
        slider_pass_req = auto_pass_req
# --- MAIN AREA FÖR INMATNING ---
st.markdown(
    f"""
    <div class="tm-hero">
        <div class="tm-step">Steg 1 · Kupongdata</div>
        <div class="tm-title">{spelform} · {antal_matcher} matcher</div>
        <div class="tm-muted">Klistra in startoddsprocent/streckprocent i matchordning: M1 1-X-2, M2 1-X-2 osv. Appen kräver {krav_odds} värden.</div>
    </div>
    """,
    unsafe_allow_html=True
)
input_text = st.text_area(
    f"Klistra in procentvärden ({krav_odds} värden)",
    height=120,
    placeholder="Exempel: 52 25 23 48 28 24 ...",
    help="Du kan klistra in från tabell. Appen plockar ut siffror automatiskt och accepterar svensk decimal-komma."
)
_input_count = len(parse_input_string(input_text, krav_odds)) if input_text else 0
if input_text:
    if _input_count == krav_odds:
        st.success(f"✅ {_input_count} av {krav_odds} värden hittades. Kupongdata är komplett.")
    else:
        st.warning(f"⚠️ {_input_count} av {krav_odds} värden hittades. Analysen kräver exakt {krav_odds} värden.")

st.markdown(
    """
    <div class="tm-hero">
        <div class="tm-step">Steg 2 · Manuell grundram</div>
        <div class="tm-title">Klicka i tecknen du själv vill spela</div>
        <div class="tm-muted">Grundramen är din kupong. AutoFilter och TipsetMatrix arbetar sedan bara på rader som ryms i din ram.</div>
    </div>
    """,
    unsafe_allow_html=True
)


tm_click_frame = None
tm_text_frame_input = ""
if 'cb_tipsetmatrix' in globals() and cb_tipsetmatrix:
    with st.expander("🧮 TipsetMatrix grundram", expanded=(tm_frame_source == "Klickbar grundram")):
        st.caption("Välj egen ram här om du vill reducera exakt de tecken du själv tänker spela. Full 13-hel spärras normalt av radgränsen.")
        if tm_frame_source == "Klickbar grundram":
            frame_state_key = f"tm_saved_frame_{spelform}_{antal_matcher}"
            if frame_state_key not in st.session_state:
                st.session_state[frame_state_key] = [['1', 'X', '2'] for _ in range(antal_matcher)]
            saved_frame = [normalize_signs(x) for x in st.session_state.get(frame_state_key, [])]
            if len(saved_frame) != int(antal_matcher):
                saved_frame = [['1', 'X', '2'] for _ in range(antal_matcher)]
                st.session_state[frame_state_key] = saved_frame

            tm_click_frame = saved_frame
            st.markdown("**Manuell teckenram** — ändra tecken och tryck sedan **Spara grundram**.")
            st.caption(
                "Tecknen ligger nu i ett formulär. Appen kör inte om AutoFilter/TipsetMatrix för varje klick på 1/X/2, "
                "utan först när du sparar ramen och sedan trycker Kör."
            )

            with st.form(key=f"tm_manual_frame_form_{spelform}_{antal_matcher}"):
                draft_frame = []
                header = st.columns([0.9, 0.8, 0.8, 0.8, 1.7])
                header[0].markdown("**Match**")
                header[1].markdown("**1**")
                header[2].markdown("**X**")
                header[3].markdown("**2**")
                header[4].markdown("**Sparat val**")

                for mi in range(antal_matcher):
                    saved_signs = normalize_signs(saved_frame[mi]) if mi < len(saved_frame) else ['1', 'X', '2']
                    row_cols = st.columns([0.9, 0.8, 0.8, 0.8, 1.7])
                    row_cols[0].markdown(f"**M{mi+1}**")
                    selected = []
                    for col_idx, sign in enumerate(['1', 'X', '2'], start=1):
                        checked = row_cols[col_idx].checkbox(
                            sign,
                            value=(sign in saved_signs),
                            key=f"tm_manual_draft_{spelform}_{mi}_{sign}",
                            label_visibility="collapsed"
                        )
                        if checked:
                            selected.append(sign)
                    selected = normalize_signs(selected)
                    draft_frame.append(selected)
                    if saved_signs:
                        row_cols[4].markdown(f"`{_sort_signs_display(saved_signs)}`")
                    else:
                        row_cols[4].markdown("⚠️ minst ett tecken")

                submitted_frame = st.form_submit_button("💾 Spara grundram", use_container_width=True)

            if submitted_frame:
                empty_draft_matches = [i + 1 for i, signs in enumerate(draft_frame) if len(signs) == 0]
                if empty_draft_matches:
                    st.warning("Grundramen sparades inte. Minst ett tecken saknas i match: " + ", ".join(map(str, empty_draft_matches)))
                else:
                    st.session_state[frame_state_key] = draft_frame
                    tm_click_frame = draft_frame
                    st.session_state['har_kort_analys'] = False
                    st.success("Grundramen är sparad. Tryck Kör AutoFilter + TipsetMatrix när du vill räkna om systemet.")

            empty_matches = [i + 1 for i, signs in enumerate(tm_click_frame) if len(signs) == 0]
            tm_manual_rows = frame_row_count(tm_click_frame)
            if empty_matches:
                st.warning("Alla matcher måste ha minst ett tecken. Saknar tecken i match: " + ", ".join(map(str, empty_matches)))
            st.caption(f"Sparad grundram: {tm_manual_rows:,} rader".replace(',', ' '))

            with st.expander("📊 Sparad grundram – tecken per match", expanded=False):
                frame_rows = []
                for mi, signs in enumerate(tm_click_frame, start=1):
                    frame_rows.append({
                        "Match/lag": f"M{mi}",
                        "Tecken": _sort_signs_display(signs),
                        "Antal tecken": len(signs),
                    })
                st.dataframe(pd.DataFrame(frame_rows), use_container_width=True, hide_index=True)

        elif tm_frame_source == "Textgrundram":
            tm_text_frame_input = st.text_area(
                "Textgrundram, en grupp per match",
                value=" / ".join(["1X2"] * antal_matcher),
                height=70,
                help="Exempel: 1X2 / 1X / 12 / 1 / X2 ..."
            )
        else:
            st.info("AI-Balansram använder de liknande historiska omgångarna och din valda radbudget/max antal helgarderingar.")

if 'har_kort_analys' not in st.session_state:
    st.session_state['har_kort_analys'] = False

st.markdown(
    """
    <div class="tm-hero">
        <div class="tm-step">Steg 3 · Kör</div>
        <div class="tm-title">Bygg mall, filtrera och reducera</div>
        <div class="tm-muted">När kupongdata och grundram är klara kör appen hela kedjan: liknande historik → AutoTrim → filterrevision → TipsetMatrix.</div>
    </div>
    """,
    unsafe_allow_html=True
)
if st.button("🚀 Kör AutoFilter + TipsetMatrix", use_container_width=True):
    if not input_text:
        st.error("⚠️ Klistra in procentvärden först.")
    elif _input_count != krav_odds:
        st.error(f"⚠️ Fel antal värden: {_input_count} av {krav_odds}. Klistra in komplett kupongdata.")
    else:
        st.session_state['har_kort_analys'] = True

# ==========================================
# RESULTAT-VISNING (Laddas från Cache-Motorn)
# ==========================================
if st.session_state.get('har_kort_analys') and input_text:
    
    pay_min, pay_max = slider_payout
    v_m, input_vec, filnamn, err = run_core_analysis(
        input_text, spelform, antal_matcher, krav_odds, cb_structure, 
        slider_top_n, cb_payout, pay_min, pay_max
    )
    
    if err:
        st.error(err)
    elif v_m is None or len(v_m) == 0:
        st.error("❌ Inga matcher kvar efter filtrering. Testa att lätta på Utdelningskravet.")
    else:
        filter_vec = input_vec  # Faktisk matchordning. Används för alla filterberäkningar.
        
        # --- DATABASKONTROLL ---
        db_full = load_database(find_local_database(spelform), antal_matcher)
        with st.expander("🧪 Databaskontroll", expanded=False):
            st.dataframe(build_database_quality_report(db_full, antal_matcher), use_container_width=True, hide_index=True)

        # --- BERÄKNA DELTA FÖR ALLA HISTORISKA RADER ---
        delta_list = []
        for _, row in v_m.iterrows():
            if len(row['Correct_Row']) == antal_matcher and len(row['Prob_Vector']) == krav_odds:
                delta_list.append(calculate_delta(row['Correct_Row'], row['Prob_Vector']))
            else:
                delta_list.append(0)
        v_m['Delta'] = delta_list

        visnings_kolumner = [c for c in ['Datum', 'ID_Omg'] if c in v_m.columns] + ['Payout', 'Delta', 'Sim']
        
        if 'ID_Omg' in v_m.columns:
            v_m['ID_Omg'] = pd.to_numeric(v_m['ID_Omg'], errors='coerce').fillna(0).astype(int).astype(str)
            v_m['ID_Omg'] = v_m['ID_Omg'].replace('0', '')
            
        st.success(f"✅ Auto-laddade: **{filnamn}**. Exakt {len(v_m)} liknande omgångar hittades.")
        
        with st.expander(f"📋 Liknande historiska omgångar ({len(v_m)} st)", expanded=False):
            st.dataframe(v_m[visnings_kolumner].rename(columns={'Payout':f'Utdelning ({antal_matcher}r)', 'Sim':'Likhet'}).style.format({f'Utdelning ({antal_matcher}r)': '{:.0f} kr', 'Likhet': '{:.2f}', 'Delta': '{:.1f}'}), use_container_width=True)

        ones, draws, twos = [], [], []
        s1, sx, s2, g1, gx, g2 = [], [], [], [], [], []
        sing1, singx, sing2, sing_tot = [], [], [], []
        dub1, dubx, dub2, dub_tot = [], [], [], []
        trip1, tripx, trip2, trip_tot = [], [], [], []
        occ1, occx, occ2, occ_tot = [], [], [], []
        sft_sums, fat_f, fat_a, fat_t, fat_sums = [], [], [], [], []
        points_vals, minus_sums, log_surprise_sums, rank24_sums, total_diff_vals, u_wins, ai_ranks = [], [], [], [], [], [], []
        fav70_wins, fav60_wins, fav50_wins = [], [], []
        shock_u10, shock_u15, shock_u20, shock_lowest = [], [], [], []
        fav_delta_vals = []
        
        for _, row in v_m.iterrows():
            r, p = row['Correct_Row'], row['Prob_Vector']
            if 'True_Rank' in row and pd.notna(row['True_Rank']) and row['True_Rank'] > 0:
                ai_ranks.append(row['True_Rank'])
            else:
                h_matrix, h_scores_asc, h_tot = calculate_ai_matrix_from_values(p)
                rank_c, _ = get_exact_rank(r, h_matrix, h_scores_asc, h_tot)
                ai_ranks.append(rank_c)

            ones.append(r.count('1')); draws.append(r.count('X')); twos.append(r.count('2'))
            _s1, _sx, _s2, _ = get_streaks(r); s1.append(_s1); sx.append(_sx); s2.append(_s2)
            _g1, _gx, _g2, _ = get_gaps(r); g1.append(_g1); gx.append(_gx); g2.append(_g2)
            _si1, _six, _si2, _sitot, _ = get_singles(r); sing1.append(_si1); singx.append(_six); sing2.append(_si2); sing_tot.append(_sitot)
            _d1, _dx, _d2, _dtot, _ = get_doublets(r); dub1.append(_d1); dubx.append(_dx); dub2.append(_d2); dub_tot.append(_dtot)
            _t1, _tx, _t2, _ttot, _ = get_triplets(r); trip1.append(_t1); tripx.append(_tx); trip2.append(_t2); trip_tot.append(_ttot)
            _o1, _ox, _o2, _otot, _ = get_occurrences(r); occ1.append(_o1); occx.append(_ox); occ2.append(_o2); occ_tot.append(_otot)
            sft_sums.append(get_sft_sum(r, p))
            _f, _a, _t, _fs = get_fat(r, p); fat_f.append(_f); fat_a.append(_a); fat_t.append(_t); fat_sums.append(_fs)
            points_vals.append(get_rank_points(r, p))
            minus_sums.append(get_100_minus_sum(r, p))
            log_surprise_sums.append(get_log_surprise_sum(r, p))
            rank24_sums.append(get_rank_sum(r, p))
            u_wins.append(get_top_n_favs_wins(r, p, slider_u_count))
            
            match_odds_list = [p[j:j+3] for j in range(0, len(p), 3)]
            total_diff_vals.append(calculate_total_diff(match_odds_list, list(r)))

            fp = get_favorite_pressure(r, p)
            fav70_wins.append(fp['F70_Wins']); fav60_wins.append(fp['F60_Wins']); fav50_wins.append(fp['F50_Wins'])
            sh = get_shock_strength(r, p)
            shock_u10.append(sh['U10_Wins']); shock_u15.append(sh['U15_Wins']); shock_u20.append(sh['U20_Wins']); shock_lowest.append(sh['Lowest_Win_Pct'])
            fav_delta_vals.append(get_favorite_delta(r, p))
            


        c_v, c_s = slider_core_val, slider_core_str
        
        # --- STANDARD STRUKTUR ---
        c_ones = get_best_interval(ones, c_s); c_draws = get_best_interval(draws, c_s); c_twos = get_best_interval(twos, c_s)
        c_s1 = get_best_interval(s1, c_s); c_sx = get_best_interval(sx, c_s); c_s2 = get_best_interval(s2, c_s)
        c_g1 = get_best_interval(g1, c_s); c_gx = get_best_interval(gx, c_s); c_g2 = get_best_interval(g2, c_s)
        c_sing1 = get_best_interval(sing1, c_s); c_singx = get_best_interval(singx, c_s); c_sing2 = get_best_interval(sing2, c_s); c_singtot = get_best_interval(sing_tot, c_s)
        c_dub1 = get_best_interval(dub1, c_s); c_dubx = get_best_interval(dubx, c_s); c_dub2 = get_best_interval(dub2, c_s); c_dubtot = get_best_interval(dub_tot, c_s)
        c_trip1 = get_best_interval(trip1, c_s); c_tripx = get_best_interval(tripx, c_s); c_trip2 = get_best_interval(trip2, c_s); c_triptot = get_best_interval(trip_tot, c_s)
        c_occ1 = get_best_interval(occ1, c_s); c_occx = get_best_interval(occx, c_s); c_occ2 = get_best_interval(occ2, c_s); c_occtot = get_best_interval(occ_tot, c_s)
        
        # --- AUTOTRIM: rationella värdeintervall ---
        # Kandidatrader används för att se vad varje intervall faktiskt kostar i radmassa.
        candidate_rows, exact_universe = make_candidate_rows(antal_matcher)
        total_candidates = len(candidate_rows)
        match_odds_filter = [filter_vec[j:j+3] for j in range(0, len(filter_vec), 3)]
        autotrim_meta = {}

        def _auto_interval(key, hist_values, cand_getter, decimals=0):
            safety = get_best_interval(hist_values, c_v)
            if cb_autotrim:
                interval, meta = choose_rational_interval(
                    hist_values,
                    candidate_rows,
                    cand_getter,
                    target_coverage_percent=c_v,
                    min_hist_pct=slider_autotrim_min_hist
                )
            else:
                hist_pct, _, misses = pct_values_in_interval(hist_values, safety)
                keep = candidate_keep_pct(candidate_rows, cand_getter, safety)
                meta = {
                    'AutoTrim': False,
                    'Vald tighthet %': c_v,
                    'Säkerhetsintervall': safety,
                    'Historisk träff %': round(hist_pct, 1),
                    'Kvar rad %': round(keep, 1) if keep is not None else None,
                    'Reducerar %': round(100 - keep, 1) if keep is not None else None,
                    'Rationell faktor': round(hist_pct - keep, 1) if keep is not None else None,
                    'Kapade outliers': misses,
                    'Testade intervall': 1,
                }
                interval = safety
            meta['Visning'] = fmt_interval(interval, decimals)
            meta['Rekommenderat intervall'] = interval
            meta['Säkerhet visning'] = fmt_interval(meta.get('Säkerhetsintervall'), decimals)
            autotrim_meta[key] = meta
            return interval

        c_sft = _auto_interval('SFT Summa', sft_sums, lambda tr: get_sft_sum(tr, filter_vec), 1)
        c_fatf = get_best_interval(fat_f, c_v); c_fata = get_best_interval(fat_a, c_v); c_fatt = get_best_interval(fat_t, c_v); c_fatsum = get_best_interval(fat_sums, c_v)
        c_points = _auto_interval('Poängfilter', points_vals, lambda tr: get_rank_points(tr, filter_vec), 0)
        c_minus = _auto_interval('100-minus Summa', minus_sums, lambda tr: get_100_minus_sum(tr, filter_vec), 1)
        c_log_surprise = _auto_interval('Skrälltryck Log Summa', log_surprise_sums, lambda tr: get_log_surprise_sum(tr, filter_vec), 0)
        c_rank24 = _auto_interval('Rank Summa', rank24_sums, lambda tr: get_rank_sum(tr, filter_vec), 1)
        c_totaldiff = _auto_interval('Total Diff', total_diff_vals, lambda tr: calculate_total_diff(match_odds_filter, list(tr)), 0)
        c_u = _auto_interval(f'Topp {slider_u_count} favoriter', u_wins, lambda tr: get_top_n_favs_wins(tr, filter_vec, slider_u_count), 0)
        c_delta = _auto_interval('Delta', list(v_m['Delta']), lambda tr: calculate_delta(tr, filter_vec), 1)

        c_fav70_raw = get_best_interval(fav70_wins, c_v)
        c_fav60_raw = get_best_interval(fav60_wins, c_v)
        c_fav50_raw = get_best_interval(fav50_wins, c_v)
        todays_fav_counts = get_favorite_threshold_counts(filter_vec)
        c_fav70 = clamp_interval(c_fav70_raw, 0, todays_fav_counts.get(70, 0))
        c_fav60 = clamp_interval(c_fav60_raw, 0, todays_fav_counts.get(60, 0))
        c_fav50 = clamp_interval(c_fav50_raw, 0, todays_fav_counts.get(50, 0))

        c_shock10_raw = get_best_interval(shock_u10, c_v)
        c_shock15_raw = get_best_interval(shock_u15, c_v)
        c_shock20_raw = get_best_interval(shock_u20, c_v)
        todays_shock_capacity = get_shock_capacity(filter_vec)
        c_shock10 = clamp_interval(c_shock10_raw, 0, todays_shock_capacity.get(10, 0))
        c_shock15 = clamp_interval(c_shock15_raw, 0, todays_shock_capacity.get(15, 0))
        c_shock20 = clamp_interval(c_shock20_raw, 0, todays_shock_capacity.get(20, 0))
        c_shock_lowest = get_best_interval(shock_lowest, c_v)
        c_fav_delta = _auto_interval('Favorit-delta', fav_delta_vals, lambda tr: get_favorite_delta(tr, filter_vec), 2)
        # --- SPETSFILTERVARIANTER ---
        # Alla poäng-/värdefilter som visas i VECKANS MALL ska även kunna bli hårda spetsfilter:
        # AI-Rank, Delta, Total Diff, Rank Summa, 100-minus, Skrälltryck Log, SFT, Poängfilter,
        # Favorit-delta, Toppfavoriter och FAT-delar. Varje filter testas i flera hårdhetsgrader.
        # Varianten används bara om hela paketet klarar samlad mallträff och minst 5% faktisk stegreducering.
        spets_variant_defs = {}

        def _add_interval_variants(label, hist_values, cand_getter, decimals=0, coverages=(95, 90, 85, 80, 75)):
            seen_intervals = set()
            vals = [v for v in hist_values if not pd.isna(v)]
            if not vals:
                return
            for cov in coverages:
                try:
                    interval = get_best_interval(vals, cov)
                except Exception:
                    continue
                key = (round(float(interval[0]), 6), round(float(interval[1]), 6))
                if key in seen_intervals:
                    continue
                seen_intervals.add(key)
                # Undvik att skapa en variant som är identisk med huvudregeln men ge ändå 85/80/75 utrymme.
                name = f"{label} spets {int(cov)}%"
                spets_variant_defs[name] = {
                    "label": label,
                    "coverage": int(cov),
                    "interval": interval,
                    "hist_values": hist_values,
                    "getter": cand_getter,
                    "decimals": decimals,
                }

        if cb_points:
            _add_interval_variants('Poängfilter', points_vals, lambda tr: get_rank_points(tr, filter_vec), 0)
        if cb_rank24:
            _add_interval_variants('Ranksumma', rank24_sums, lambda tr: get_rank_sum(tr, filter_vec), 1)
        if cb_log_surprise:
            _add_interval_variants('Skrälltryck Log', log_surprise_sums, lambda tr: get_log_surprise_sum(tr, filter_vec), 0)
        if cb_sft:
            _add_interval_variants('SFT Summa', sft_sums, lambda tr: get_sft_sum(tr, filter_vec), 1)
        if cb_100minus:
            _add_interval_variants('100-minus', minus_sums, lambda tr: get_100_minus_sum(tr, filter_vec), 1)
        if cb_totaldiff:
            _add_interval_variants('Total Diff', total_diff_vals, lambda tr: calculate_total_diff(match_odds_filter, list(tr)), 0)
        if cb_fav_delta:
            _add_interval_variants('Favorit-delta', fav_delta_vals, lambda tr: get_favorite_delta(tr, filter_vec), 2)
        # Delta visas alltid i VECKANS MALL och ska därför också kunna väljas som hårt spetsfilter.
        _add_interval_variants('Delta', list(v_m['Delta']), lambda tr: calculate_delta(tr, filter_vec), 1)
        if cb_u_favs:
            _add_interval_variants(f'Topp {slider_u_count} favoriter', u_wins, lambda tr: get_top_n_favs_wins(tr, filter_vec, slider_u_count), 0)
        if cb_fat:
            _add_interval_variants('FAT F', fat_f, lambda tr: get_fat(tr, filter_vec)[0], 0)
            _add_interval_variants('FAT A', fat_a, lambda tr: get_fat(tr, filter_vec)[1], 0)
            _add_interval_variants('FAT T', fat_t, lambda tr: get_fat(tr, filter_vec)[2], 0)
            _add_interval_variants('FAT Summa', fat_sums, lambda tr: get_fat(tr, filter_vec)[3], 0)
        # --- FAT-SEKVENSER: aktiva reduceringsbyggklossar ---
        fat_strings_hist = [get_fat_string(row['Correct_Row'], row['Prob_Vector']) for _, row in v_m.iterrows() if len(str(row['Correct_Row'])) == antal_matcher]
        base_fat_strings = [get_fat_string(row['Correct_Row'], row['Prob_Vector']) for _, row in db_full.iterrows() if len(str(row['Correct_Row'])) == antal_matcher]
        fat_seq2_rows = choose_fat_sequences(fat_strings_hist, base_fat_strings, length=2, top_n=5, min_lift=0.0)
        fat_seq3_rows = choose_fat_sequences(fat_strings_hist, base_fat_strings, length=3, top_n=5, min_lift=0.0)
        fat_pair_rows = choose_fat_sequence_pairs(fat_strings_hist, base_fat_strings, top_n=5, min_lift=0.0)
        fat_seq2_list = [r['Sekvens'] for r in fat_seq2_rows]
        fat_seq3_list = [r['Sekvens'] for r in fat_seq3_rows]
        fat_pair_list = [r['Par'] for r in fat_pair_rows]
        fat_seq2_hits = [count_fat_sequence_hits(f, fat_seq2_list) for f in fat_strings_hist]
        fat_seq3_hits = [count_fat_sequence_hits(f, fat_seq3_list) for f in fat_strings_hist]
        fat_pair_hits = [count_fat_pair_hits(f, fat_pair_list) for f in fat_strings_hist]

        if fat_seq2_list:
            c_fat_seq2 = _auto_interval('FAT 2-sekvenser', fat_seq2_hits, lambda tr: count_fat_sequence_hits(get_fat_string(tr, filter_vec), fat_seq2_list), 0)
        else:
            c_fat_seq2 = (0, 0)
        if fat_seq3_list:
            c_fat_seq3 = _auto_interval('FAT 3-sekvenser', fat_seq3_hits, lambda tr: count_fat_sequence_hits(get_fat_string(tr, filter_vec), fat_seq3_list), 0)
        else:
            c_fat_seq3 = (0, 0)
        if fat_pair_list:
            c_fat_pairs = _auto_interval('FAT dubbelchans', fat_pair_hits, lambda tr: count_fat_pair_hits(get_fat_string(tr, filter_vec), fat_pair_list), 0)
        else:
            c_fat_pairs = (0, 0)

        if cb_fat_sequences:
            if fat_seq2_list:
                _add_interval_variants('FAT 2-sekvenser', fat_seq2_hits, lambda tr: count_fat_sequence_hits(get_fat_string(tr, filter_vec), fat_seq2_list), 0, coverages=(95, 90, 85, 80, 75))
            if fat_seq3_list:
                _add_interval_variants('FAT 3-sekvenser', fat_seq3_hits, lambda tr: count_fat_sequence_hits(get_fat_string(tr, filter_vec), fat_seq3_list), 0, coverages=(95, 90, 85, 80, 75))
            if fat_pair_list:
                _add_interval_variants('FAT dubbelchans', fat_pair_hits, lambda tr: count_fat_pair_hits(get_fat_string(tr, filter_vec), fat_pair_list), 0, coverages=(95, 90, 85, 80, 75))

        fat_sequence_active_parts = sum([bool(fat_seq2_list), bool(fat_seq3_list), bool(fat_pair_list)])
        fat_sequence_group_req = 2 if fat_sequence_active_parts >= 2 else 1

        def fat_sequence_subscore_from_fat_string(fstr):
            pts = 0
            if fat_seq2_list and in_range(count_fat_sequence_hits(fstr, fat_seq2_list), c_fat_seq2): pts += 1
            if fat_seq3_list and in_range(count_fat_sequence_hits(fstr, fat_seq3_list), c_fat_seq3): pts += 1
            if fat_pair_list and in_range(count_fat_pair_hits(fstr, fat_pair_list), c_fat_pairs): pts += 1
            return pts

        def fat_sequence_group_ok_from_fat_string(fstr):
            if fat_sequence_active_parts == 0:
                return False
            return fat_sequence_subscore_from_fat_string(fstr) >= fat_sequence_group_req

        def fat_sequence_group_ok_row(row_str):
            return fat_sequence_group_ok_from_fat_string(get_fat_string(row_str, filter_vec))

        
        c_ai_rank = get_best_interval(ai_ranks, c_v) if len(ai_ranks) > 0 else (1, max_rank)
        active_ai_min, active_ai_max = slider_ai_rank if cb_manual_ai_rank else c_ai_rank
        ai_txt = "AI-Rank (MANUELL)" if cb_manual_ai_rank else f"AI-Rank (AUTO {c_v}%)"

        # AI-Rank beräknas från veckans aktuella procentmatris för kandidat-/grundrader.
        # Den ska behandlas som samma typ av hårt poängfilter som övriga värdefilter.
        spets_ai_matrix = spets_ai_scores_asc = spets_ai_tot = None
        if cb_aimatrix and not cb_manual_ai_rank and len(ai_ranks) > 0:
            try:
                spets_ai_matrix, spets_ai_scores_asc, spets_ai_tot = calculate_ai_matrix_from_values(filter_vec)
                _add_interval_variants(
                    'AI-Rank',
                    ai_ranks,
                    lambda tr: get_exact_rank(tr, spets_ai_matrix, spets_ai_scores_asc, spets_ai_tot)[0],
                    0,
                    coverages=(95, 90, 85, 80, 75, 70)
                )
            except Exception:
                pass

        # --- SUPER-MAKRO ---
        def get_super_macro_bounds(total_rows, target_prob, req_groups):
            lists_dict = {
                '1': ones, 'X': draws, '2': twos,
                's1': s1, 'sx': sx, 's2': s2,
                'g1': g1, 'gx': gx, 'g2': g2,
                'si1': sing1, 'six': singx, 'si2': sing2,
                'd1': dub1, 'dx': dubx, 'd2': dub2,
                't1': trip1, 'tx': tripx, 't2': trip2,
                'o1': occ1, 'ox': occx, 'o2': occ2,
                'f': fat_f, 'a': fat_a, 't': fat_t
            }
            if total_rows == 0: return {k: (0,0) for k in lists_dict}, 0.0
            
            for cov in range(0, 101, 1):
                b = {}
                for k, lst in lists_dict.items():
                    if not lst: 
                        b[k] = (0,0)
                        continue
                    if cov == 0:
                        v = int(np.median(lst))
                        b[k] = (v, v)
                    else:
                        low, upp = 50 - cov/2.0, 50 + cov/2.0
                        b[k] = (int(round(np.percentile(lst, low))), int(round(np.percentile(lst, upp))))
                
                hits = 0
                for i in range(total_rows):
                    g_pass = 0
                    if sum([b['1'][0] <= ones[i] <= b['1'][1], b['X'][0] <= draws[i] <= b['X'][1], b['2'][0] <= twos[i] <= b['2'][1]]) >= 2: g_pass += 1
                    if sum([b['s1'][0] <= s1[i] <= b['s1'][1], b['sx'][0] <= sx[i] <= b['sx'][1], b['s2'][0] <= s2[i] <= b['s2'][1]]) >= 2: g_pass += 1
                    if sum([b['g1'][0] <= g1[i] <= b['g1'][1], b['gx'][0] <= gx[i] <= b['gx'][1], b['g2'][0] <= g2[i] <= b['g2'][1]]) >= 2: g_pass += 1
                    if sum([b['si1'][0] <= sing1[i] <= b['si1'][1], b['six'][0] <= singx[i] <= b['six'][1], b['si2'][0] <= sing2[i] <= b['si2'][1]]) >= 2: g_pass += 1
                    if sum([b['d1'][0] <= dub1[i] <= b['d1'][1], b['dx'][0] <= dubx[i] <= b['dx'][1], b['d2'][0] <= dub2[i] <= b['d2'][1]]) >= 2: g_pass += 1
                    if sum([b['t1'][0] <= trip1[i] <= b['t1'][1], b['tx'][0] <= tripx[i] <= b['tx'][1], b['t2'][0] <= trip2[i] <= b['t2'][1]]) >= 2: g_pass += 1
                    if sum([b['o1'][0] <= occ1[i] <= b['o1'][1], b['ox'][0] <= occx[i] <= b['ox'][1], b['o2'][0] <= occ2[i] <= b['o2'][1]]) >= 2: g_pass += 1
                    if sum([b['f'][0] <= fat_f[i] <= b['f'][1], b['a'][0] <= fat_a[i] <= b['a'][1], b['t'][0] <= fat_t[i] <= b['t'][1]]) >= 2: g_pass += 1
                    
                    if g_pass >= req_groups: hits += 1
                    
                prob = (hits / total_rows) * 100
                if prob >= target_prob:
                    return b, prob
            return b, 100.0

        n_rows = len(v_m)
        sm_bounds, sm_prob = get_super_macro_bounds(n_rows, slider_macro_target, slider_super_groups)

        st.markdown("---")
        st.header(f"📋 VECKANS MALL ({spelform}) – snabb inmatning")
        
        def _show_value_interval(label, key, interval, decimals=0, suffix=""):
            st.write(f"**{label}:** {fmt_interval(interval, decimals)}{suffix}")
            meta = autotrim_meta.get(key)
            note = autotrim_caption(meta, decimals)
            if note and (cb_autotrim or key in autotrim_meta):
                st.caption(note)

        col_v, col_s = st.columns(2)
        with col_v:
            st.subheader(f"💰 VÄRDE & SVÅRIGHET ({c_v}%)")
            if cb_payout: st.write(f"**Utdelning:** {v_m['Payout'].min():.0f} - {v_m['Payout'].max():.0f} kr")
            _show_value_interval("Delta (Avvikelse)", "Delta", c_delta, 1)
            if cb_aimatrix: st.write(f"**{ai_txt}:** {active_ai_min:.0f} - {active_ai_max:.0f}")
            if cb_totaldiff: _show_value_interval("Total Diff", "Total Diff", c_totaldiff, 0)
            if cb_rank24: _show_value_interval("Rank Summa", "Rank Summa", c_rank24, 1)
            if cb_100minus: _show_value_interval("100-minus Summa", "100-minus Summa", c_minus, 1)
            if cb_log_surprise: _show_value_interval("Skrälltryck Log Summa", "Skrälltryck Log Summa", c_log_surprise, 0)
            if cb_sft: _show_value_interval("SFT Summa", "SFT Summa", c_sft, 1)
            if cb_points: _show_value_interval("Poängfilter", "Poängfilter", c_points, 0)
            if cb_fat: st.write(f"**FAT (Standard):** F:{c_fatf[0]}-{c_fatf[1]} | A:{c_fata[0]}-{c_fata[1]} | T:{c_fatt[0]}-{c_fatt[1]} (Summa: {c_fatsum[0]}-{c_fatsum[1]})")
            if cb_fat_sequences:
                st.markdown("**FAT-sekvenser (aktivt reduceringsfilter):**")
                st.caption(f"2-seq {', '.join(fat_seq2_list) if fat_seq2_list else '-'} · krav {fmt_interval(c_fat_seq2)} av 5 | 3-seq {', '.join(fat_seq3_list) if fat_seq3_list else '-'} · krav {fmt_interval(c_fat_seq3)} av 5 | dubbelchans {fmt_interval(c_fat_pairs)} av 5 | gruppkrav {fat_sequence_group_req} av {fat_sequence_active_parts}")
            if cb_u_favs: _show_value_interval(f"Topp {slider_u_count} Favoriter", f"Topp {slider_u_count} favoriter", c_u, 0, " st vinner")
            if cb_fav_pressure: st.write(f"**Favorittryck:** ≥70% {fmt_interval(c_fav70)} av {todays_fav_counts.get(70,0)} | ≥60% {fmt_interval(c_fav60)} av {todays_fav_counts.get(60,0)} | ≥50% {fmt_interval(c_fav50)} av {todays_fav_counts.get(50,0)}")
            if cb_shock_strength: st.write(f"**Skrällstyrka:** <10% {fmt_interval(c_shock10)} | <15% {fmt_interval(c_shock15)} | <20% {fmt_interval(c_shock20)} | Lägsta vinnande % {fmt_interval(c_shock_lowest, 1)}")
            if cb_fav_delta: _show_value_interval("Favorit-delta", "Favorit-delta", c_fav_delta, 2)

        with col_s:
            st.subheader(f"⚽ STRUKTUR ({c_s}%)")
            if cb_base: st.write(f"**1X2:** 1: {c_ones[0]}-{c_ones[1]} | X: {c_draws[0]}-{c_draws[1]} | 2: {c_twos[0]}-{c_twos[1]}")
            if cb_streak: st.write(f"**Sviter:** 1: {c_s1[0]}-{c_s1[1]} | X: {c_sx[0]}-{c_sx[1]} | 2: {c_s2[0]}-{c_s2[1]}")
            if cb_gap: st.write(f"**Luckor:** 1: {c_g1[0]}-{c_g1[1]} | X: {c_gx[0]}-{c_gx[1]} | 2: {c_g2[0]}-{c_g2[1]}")
            if cb_single: st.write(f"**Singlar:** 1: {c_sing1[0]}-{c_sing1[1]} | X: {c_singx[0]}-{c_singx[1]} | 2: {c_sing2[0]}-{c_sing2[1]} | Tot: {c_singtot[0]}-{c_singtot[1]}")
            if cb_doublet: st.write(f"**Dubbletter:** 1: {c_dub1[0]}-{c_dub1[1]} | X: {c_dubx[0]}-{c_dubx[1]} | 2: {c_dub2[0]}-{c_dub2[1]} | Tot: {c_dubtot[0]}-{c_dubtot[1]}")
            if cb_triplet: st.write(f"**Tripplar:** 1: {c_trip1[0]}-{c_trip1[1]} | X: {c_tripx[0]}-{c_tripx[1]} | 2: {c_trip2[0]}-{c_trip2[1]} | Tot: {c_triptot[0]}-{c_triptot[1]}")
            if cb_occur: st.write(f"**Uppkomster:** 1: {c_occ1[0]}-{c_occ1[1]} | X: {c_occx[0]}-{c_occx[1]} | 2: {c_occ2[0]}-{c_occ2[1]} | Tot: {c_occtot[0]}-{c_occtot[1]}")
            
            st.markdown("---")
            st.subheader(f"🧩 SUPER-MAKRO (Krav: Minst {slider_super_groups} av 8 grupper)")
            st.markdown(f"*Minst 2 av 3 interna tecken i en grupp måste sitta för att gruppen ska räknas som 'träffad'. Överlever historiskt **{sm_prob:.1f}%**. Super-Makro används som stödfilter och bedöms i Filterrevisionen.*")
            if cb_super_macro:
                b = sm_bounds
                st.write(f"⚽ **Grp 1 (1X2):** 1: {b['1'][0]}-{b['1'][1]} | X: {b['X'][0]}-{b['X'][1]} | 2: {b['2'][0]}-{b['2'][1]}")
                st.write(f"🔥 **Grp 2 (Sviter):** 1: {b['s1'][0]}-{b['s1'][1]} | X: {b['sx'][0]}-{b['sx'][1]} | 2: {b['s2'][0]}-{b['s2'][1]}")
                st.write(f"🕳️ **Grp 3 (Luckor):** 1: {b['g1'][0]}-{b['g1'][1]} | X: {b['gx'][0]}-{b['gx'][1]} | 2: {b['g2'][0]}-{b['g2'][1]}")
                st.write(f"🎯 **Grp 4 (Singlar):** 1: {b['si1'][0]}-{b['si1'][1]} | X: {b['six'][0]}-{b['six'][1]} | 2: {b['si2'][0]}-{b['si2'][1]}")
                st.write(f"👯 **Grp 5 (Dubbletter):** 1: {b['d1'][0]}-{b['d1'][1]} | X: {b['dx'][0]}-{b['dx'][1]} | 2: {b['d2'][0]}-{b['d2'][1]}")
                st.write(f"📐 **Grp 6 (Tripplar):** 1: {b['t1'][0]}-{b['t1'][1]} | X: {b['tx'][0]}-{b['tx'][1]} | 2: {b['t2'][0]}-{b['t2'][1]}")
                st.write(f"💥 **Grp 7 (Uppkomster):** 1: {b['o1'][0]}-{b['o1'][1]} | X: {b['ox'][0]}-{b['ox'][1]} | 2: {b['o2'][0]}-{b['o2'][1]}")
                st.write(f"🧬 **Grp 8 (FAT):** F: {b['f'][0]}-{b['f'][1]} | A: {b['a'][0]}-{b['a'][1]} | T: {b['t'][0]}-{b['t'][1]}")
                
                if antal_matcher == 8:
                    test_rows = [''.join(tup) for tup in itertools.product(['1','X','2'], repeat=8)]
                    total_test = 6561
                    is_exact = True
                else:
                    mc_matrix = np.random.choice(['1', 'X', '2'], size=(10000, 13))
                    test_rows = [''.join(row) for row in mc_matrix]
                    total_test = 10000
                    is_exact = False
                    
                sm_survivors = 0
                for tr in test_rows:
                    g_pass = 0
                    c1, cx, c2 = tr.count('1'), tr.count('X'), tr.count('2')
                    s1_c, sx_c, s2_c, _ = get_streaks(tr)
                    g1_c, gx_c, g2_c, _ = get_gaps(tr)
                    si1_c, six_c, si2_c, _, _ = get_singles(tr)
                    d1_c, dx_c, d2_c, _, _ = get_doublets(tr)
                    t1_c, tx_c, t2_c, _, _ = get_triplets(tr)
                    o1_c, ox_c, o2_c, _, _ = get_occurrences(tr)
                    f_c, a_c, t_c, _ = get_fat(tr, filter_vec)
                    
                    if sum([b['1'][0] <= c1 <= b['1'][1], b['X'][0] <= cx <= b['X'][1], b['2'][0] <= c2 <= b['2'][1]]) >= 2: g_pass += 1
                    if sum([b['s1'][0] <= s1_c <= b['s1'][1], b['sx'][0] <= sx_c <= b['sx'][1], b['s2'][0] <= s2_c <= b['s2'][1]]) >= 2: g_pass += 1
                    if sum([b['g1'][0] <= g1_c <= b['g1'][1], b['gx'][0] <= gx_c <= b['gx'][1], b['g2'][0] <= g2_c <= b['g2'][1]]) >= 2: g_pass += 1
                    if sum([b['si1'][0] <= si1_c <= b['si1'][1], b['six'][0] <= six_c <= b['six'][1], b['si2'][0] <= si2_c <= b['si2'][1]]) >= 2: g_pass += 1
                    if sum([b['d1'][0] <= d1_c <= b['d1'][1], b['dx'][0] <= dx_c <= b['dx'][1], b['d2'][0] <= d2_c <= b['d2'][1]]) >= 2: g_pass += 1
                    if sum([b['t1'][0] <= t1_c <= b['t1'][1], b['tx'][0] <= tx_c <= b['tx'][1], b['t2'][0] <= t2_c <= b['t2'][1]]) >= 2: g_pass += 1
                    if sum([b['o1'][0] <= o1_c <= b['o1'][1], b['ox'][0] <= ox_c <= b['ox'][1], b['o2'][0] <= o2_c <= b['o2'][1]]) >= 2: g_pass += 1
                    if sum([b['f'][0] <= f_c <= b['f'][1], b['a'][0] <= a_c <= b['a'][1], b['t'][0] <= t_c <= b['t'][1]]) >= 2: g_pass += 1
                    
                    if g_pass >= slider_super_groups:
                        sm_survivors += 1
                        
                red_pct = 100 - ((sm_survivors / total_test) * 100)
                if is_exact:
                    st.success(f"✂️ **Omedelbar effekt:** Bara detta Super-Makro ensamt slaktar bort **{red_pct:.1f}%** av de matematiska raderna! (Kvar: {sm_survivors} av 6 561)")
                else:
                    est_rader = int(1594323 * (sm_survivors / total_test))
                    st.success(f"✂️ **Omedelbar effekt (AI-Estimat):** Bara detta Super-Makro ensamt slaktar bort ca **{red_pct:.1f}%** av de matematiska raderna! (Kvar: ca {est_rader:,} av 1 594 323 rader)".replace(',', ' '))

        # --- AUTOHARD: breda hårda basfilter ---
        # Hårda filter använder säkerhetsintervall/breda intervall. De ska inte vara knivskarpa,
        # utan rensa bort uppenbart felaktiga rader innan hårda spetsfilter väljs.
        def _hard_interval(key, fallback):
            meta = autotrim_meta.get(key, {}) if isinstance(autotrim_meta, dict) else {}
            safe = meta.get('Säkerhetsintervall')
            return safe if safe is not None else fallback

        h_sft = _hard_interval('SFT Summa', c_sft)
        h_points = _hard_interval('Poängfilter', c_points)
        h_minus = _hard_interval('100-minus Summa', c_minus)
        h_log_surprise = _hard_interval('Skrälltryck Log Summa', c_log_surprise)
        h_rank24 = _hard_interval('Rank Summa', c_rank24)
        h_totaldiff = _hard_interval('Total Diff', c_totaldiff)
        h_u = _hard_interval(f'Topp {slider_u_count} favoriter', c_u)
        h_fav_delta = _hard_interval('Favorit-delta', c_fav_delta)
        h_fat_seq2 = _hard_interval('FAT 2-sekvenser', c_fat_seq2)
        h_fat_seq3 = _hard_interval('FAT 3-sekvenser', c_fat_seq3)
        h_fat_pairs = _hard_interval('FAT dubbelchans', c_fat_pairs)

        def _gate_req(n, ratio=0.67):
            return max(1, int(np.ceil(max(1, n) * ratio)))

        def _super_macro_hist_ok(i):
            if not cb_super_macro:
                return False
            g_pass = 0
            b = sm_bounds
            if sum([b['1'][0] <= ones[i] <= b['1'][1], b['X'][0] <= draws[i] <= b['X'][1], b['2'][0] <= twos[i] <= b['2'][1]]) >= 2: g_pass += 1
            if sum([b['s1'][0] <= s1[i] <= b['s1'][1], b['sx'][0] <= sx[i] <= b['sx'][1], b['s2'][0] <= s2[i] <= b['s2'][1]]) >= 2: g_pass += 1
            if sum([b['g1'][0] <= g1[i] <= b['g1'][1], b['gx'][0] <= gx[i] <= b['gx'][1], b['g2'][0] <= g2[i] <= b['g2'][1]]) >= 2: g_pass += 1
            if sum([b['si1'][0] <= sing1[i] <= b['si1'][1], b['six'][0] <= singx[i] <= b['six'][1], b['si2'][0] <= sing2[i] <= b['si2'][1]]) >= 2: g_pass += 1
            if sum([b['d1'][0] <= dub1[i] <= b['d1'][1], b['dx'][0] <= dubx[i] <= b['dx'][1], b['d2'][0] <= dub2[i] <= b['d2'][1]]) >= 2: g_pass += 1
            if sum([b['t1'][0] <= trip1[i] <= b['t1'][1], b['tx'][0] <= tripx[i] <= b['tx'][1], b['t2'][0] <= trip2[i] <= b['t2'][1]]) >= 2: g_pass += 1
            if sum([b['o1'][0] <= occ1[i] <= b['o1'][1], b['ox'][0] <= occx[i] <= b['ox'][1], b['o2'][0] <= occ2[i] <= b['o2'][1]]) >= 2: g_pass += 1
            if sum([b['f'][0] <= fat_f[i] <= b['f'][1], b['a'][0] <= fat_a[i] <= b['a'][1], b['t'][0] <= fat_t[i] <= b['t'][1]]) >= 2: g_pass += 1
            return g_pass >= slider_super_groups

        def hard_history_bools(i):
            gates = []
            # Gate 1: Risk/värde – breda säkerhetsintervall, krav på majoritet.
            risk_checks = []
            if cb_sft: risk_checks.append(in_range(sft_sums[i], h_sft))
            if cb_points: risk_checks.append(in_range(points_vals[i], h_points))
            if cb_100minus: risk_checks.append(in_range(minus_sums[i], h_minus))
            if cb_log_surprise: risk_checks.append(in_range(log_surprise_sums[i], h_log_surprise))
            if cb_rank24: risk_checks.append(in_range(rank24_sums[i], h_rank24))
            if cb_totaldiff: risk_checks.append(in_range(total_diff_vals[i], h_totaldiff))
            if risk_checks:
                gates.append(sum(risk_checks) >= _gate_req(len(risk_checks), 0.67))

            # Gate 2: FAT-profil – bred men fortfarande hård.
            if cb_fat:
                gates.append(c_fatf[0] <= fat_f[i] <= c_fatf[1] and c_fata[0] <= fat_a[i] <= c_fata[1] and c_fatt[0] <= fat_t[i] <= c_fatt[1] and c_fatsum[0] <= fat_sums[i] <= c_fatsum[1])

            # Gate 3: FAT-sekvenser – bredare säkerhetsintervall än softvarianten.
            seq_checks = []
            if cb_fat_sequences and i < len(fat_strings_hist):
                fstr = fat_strings_hist[i]
                if fat_seq2_list: seq_checks.append(in_range(count_fat_sequence_hits(fstr, fat_seq2_list), h_fat_seq2))
                if fat_seq3_list: seq_checks.append(in_range(count_fat_sequence_hits(fstr, fat_seq3_list), h_fat_seq3))
                if fat_pair_list: seq_checks.append(in_range(count_fat_pair_hits(fstr, fat_pair_list), h_fat_pairs))
            if seq_checks:
                gates.append(sum(seq_checks) >= max(1, min(2, len(seq_checks))))

            # Gate 4: Struktur – bred strukturmajoritet, inte alla strukturfilter som AND.
            struct_checks = []
            if cb_base: struct_checks.append(c_ones[0] <= ones[i] <= c_ones[1] and c_draws[0] <= draws[i] <= c_draws[1] and c_twos[0] <= twos[i] <= c_twos[1])
            if cb_streak: struct_checks.append(c_s1[0] <= s1[i] <= c_s1[1] and c_sx[0] <= sx[i] <= c_sx[1] and c_s2[0] <= s2[i] <= c_s2[1])
            if cb_gap: struct_checks.append(c_g1[0] <= g1[i] <= c_g1[1] and c_gx[0] <= gx[i] <= c_gx[1] and c_g2[0] <= g2[i] <= c_g2[1])
            if cb_single: struct_checks.append(c_sing1[0] <= sing1[i] <= c_sing1[1] and c_singx[0] <= singx[i] <= c_singx[1] and c_sing2[0] <= sing2[i] <= c_sing2[1] and c_singtot[0] <= sing_tot[i] <= c_singtot[1])
            if cb_doublet: struct_checks.append(c_dub1[0] <= dub1[i] <= c_dub1[1] and c_dubx[0] <= dubx[i] <= c_dubx[1] and c_dub2[0] <= dub2[i] <= c_dub2[1] and c_dubtot[0] <= dub_tot[i] <= c_dubtot[1])
            if cb_triplet: struct_checks.append(c_trip1[0] <= trip1[i] <= c_trip1[1] and c_tripx[0] <= tripx[i] <= c_tripx[1] and c_trip2[0] <= trip2[i] <= c_trip2[1] and c_triptot[0] <= trip_tot[i] <= c_triptot[1])
            if cb_occur: struct_checks.append(c_occ1[0] <= occ1[i] <= c_occ1[1] and c_occx[0] <= occx[i] <= c_occx[1] and c_occ2[0] <= occ2[i] <= c_occ2[1] and c_occtot[0] <= occ_tot[i] <= c_occtot[1])
            if struct_checks:
                gates.append(sum(struct_checks) >= _gate_req(len(struct_checks), 0.60))

            # Gate 5: Super-Makro som eget bredare stödfilter.
            if cb_super_macro:
                gates.append(_super_macro_hist_ok(i))

            # Gate 6: Favorit/skrällbalans – minst hälften av aktiva delbilder.
            fs_checks = []
            if cb_u_favs: fs_checks.append(in_range(u_wins[i], h_u))
            if cb_fav_pressure: fs_checks.append(in_range(fav70_wins[i], c_fav70) and in_range(fav60_wins[i], c_fav60) and in_range(fav50_wins[i], c_fav50))
            if cb_shock_strength: fs_checks.append(in_range(shock_u10[i], c_shock10) and in_range(shock_u15[i], c_shock15) and in_range(shock_u20[i], c_shock20) and in_range(shock_lowest[i], c_shock_lowest))
            if cb_fav_delta: fs_checks.append(in_range(fav_delta_vals[i], h_fav_delta))
            if fs_checks:
                gates.append(sum(fs_checks) >= _gate_req(len(fs_checks), 0.50))
            return gates

        def hard_candidate_bools(tr):
            gates = []
            c1, cx, c2 = tr.count('1'), tr.count('X'), tr.count('2')
            s1_c, sx_c, s2_c, _ = get_streaks(tr)
            g1_c, gx_c, g2_c, _ = get_gaps(tr)
            si1_c, six_c, si2_c, singtot_c, _ = get_singles(tr)
            d1_c, dx_c, d2_c, dubtot_c, _ = get_doublets(tr)
            t1_c, tx_c, t2_c, triptot_c, _ = get_triplets(tr)
            o1_c, ox_c, o2_c, occtot_c, _ = get_occurrences(tr)
            f_c, a_c, t_c, fsum_c = get_fat(tr, filter_vec)

            risk_checks = []
            if cb_sft: risk_checks.append(in_range(get_sft_sum(tr, filter_vec), h_sft))
            if cb_points: risk_checks.append(in_range(get_rank_points(tr, filter_vec), h_points))
            if cb_100minus: risk_checks.append(in_range(get_100_minus_sum(tr, filter_vec), h_minus))
            if cb_log_surprise: risk_checks.append(in_range(get_log_surprise_sum(tr, filter_vec), h_log_surprise))
            if cb_rank24: risk_checks.append(in_range(get_rank_sum(tr, filter_vec), h_rank24))
            if cb_totaldiff: risk_checks.append(in_range(calculate_total_diff(match_odds_filter, list(tr)), h_totaldiff))
            if risk_checks:
                gates.append(sum(risk_checks) >= _gate_req(len(risk_checks), 0.67))
            if cb_fat:
                gates.append(in_range(f_c, c_fatf) and in_range(a_c, c_fata) and in_range(t_c, c_fatt) and in_range(fsum_c, c_fatsum))
            seq_checks = []
            if cb_fat_sequences:
                fstr = get_fat_string(tr, filter_vec)
                if fat_seq2_list: seq_checks.append(in_range(count_fat_sequence_hits(fstr, fat_seq2_list), h_fat_seq2))
                if fat_seq3_list: seq_checks.append(in_range(count_fat_sequence_hits(fstr, fat_seq3_list), h_fat_seq3))
                if fat_pair_list: seq_checks.append(in_range(count_fat_pair_hits(fstr, fat_pair_list), h_fat_pairs))
            if seq_checks:
                gates.append(sum(seq_checks) >= max(1, min(2, len(seq_checks))))
            struct_checks = []
            if cb_base: struct_checks.append(in_range(c1, c_ones) and in_range(cx, c_draws) and in_range(c2, c_twos))
            if cb_streak: struct_checks.append(in_range(s1_c, c_s1) and in_range(sx_c, c_sx) and in_range(s2_c, c_s2))
            if cb_gap: struct_checks.append(in_range(g1_c, c_g1) and in_range(gx_c, c_gx) and in_range(g2_c, c_g2))
            if cb_single: struct_checks.append(in_range(si1_c, c_sing1) and in_range(six_c, c_singx) and in_range(si2_c, c_sing2) and in_range(singtot_c, c_singtot))
            if cb_doublet: struct_checks.append(in_range(d1_c, c_dub1) and in_range(dx_c, c_dubx) and in_range(d2_c, c_dub2) and in_range(dubtot_c, c_dubtot))
            if cb_triplet: struct_checks.append(in_range(t1_c, c_trip1) and in_range(tx_c, c_tripx) and in_range(t2_c, c_trip2) and in_range(triptot_c, c_triptot))
            if cb_occur: struct_checks.append(in_range(o1_c, c_occ1) and in_range(ox_c, c_occx) and in_range(o2_c, c_occ2) and in_range(occtot_c, c_occtot))
            if struct_checks:
                gates.append(sum(struct_checks) >= _gate_req(len(struct_checks), 0.60))
            if cb_super_macro:
                gates.append(pass_super_macro_row(tr, filter_vec, sm_bounds, slider_super_groups))
            fs_checks = []
            if cb_u_favs: fs_checks.append(in_range(get_top_n_favs_wins(tr, filter_vec, slider_u_count), h_u))
            if cb_fav_pressure:
                fp_c = get_favorite_pressure(tr, filter_vec)
                fs_checks.append(in_range(fp_c['F70_Wins'], c_fav70) and in_range(fp_c['F60_Wins'], c_fav60) and in_range(fp_c['F50_Wins'], c_fav50))
            if cb_shock_strength:
                sh_c = get_shock_strength(tr, filter_vec)
                fs_checks.append(in_range(sh_c['U10_Wins'], c_shock10) and in_range(sh_c['U15_Wins'], c_shock15) and in_range(sh_c['U20_Wins'], c_shock20) and in_range(sh_c['Lowest_Win_Pct'], c_shock_lowest))
            if cb_fav_delta: fs_checks.append(in_range(get_favorite_delta(tr, filter_vec), h_fav_delta))
            if fs_checks:
                gates.append(sum(fs_checks) >= _gate_req(len(fs_checks), 0.50))
            return gates

        def hard_history_score(i):
            return sum(1 for x in hard_history_bools(i) if x)

        def hard_candidate_score(tr):
            return sum(1 for x in hard_candidate_bools(tr) if x)

        hard_gate_total = len(hard_history_bools(0)) if len(v_m) else 0
        history_hard_scores = [hard_history_score(i) for i in range(len(v_m))]

        mall_hits = 0
        hard_all_hits = 0
        history_filter_scores = []
        for i in range(len(v_m)):
            pts = 0
            if cb_base and (c_ones[0] <= ones[i] <= c_ones[1] and c_draws[0] <= draws[i] <= c_draws[1] and c_twos[0] <= twos[i] <= c_twos[1]): pts += 1
            if cb_streak and (c_s1[0] <= s1[i] <= c_s1[1] and c_sx[0] <= sx[i] <= c_sx[1] and c_s2[0] <= s2[i] <= c_s2[1]): pts += 1
            if cb_gap and (c_g1[0] <= g1[i] <= c_g1[1] and c_gx[0] <= gx[i] <= c_gx[1] and c_g2[0] <= g2[i] <= c_g2[1]): pts += 1
            if cb_single and (c_sing1[0] <= sing1[i] <= c_sing1[1] and c_singx[0] <= singx[i] <= c_singx[1] and c_sing2[0] <= sing2[i] <= c_sing2[1] and c_singtot[0] <= sing_tot[i] <= c_singtot[1]): pts += 1
            if cb_doublet and (c_dub1[0] <= dub1[i] <= c_dub1[1] and c_dubx[0] <= dubx[i] <= c_dubx[1] and c_dub2[0] <= dub2[i] <= c_dub2[1] and c_dubtot[0] <= dub_tot[i] <= c_dubtot[1]): pts += 1
            if cb_triplet and (c_trip1[0] <= trip1[i] <= c_trip1[1] and c_tripx[0] <= tripx[i] <= c_tripx[1] and c_trip2[0] <= trip2[i] <= c_trip2[1] and c_triptot[0] <= trip_tot[i] <= c_triptot[1]): pts += 1
            if cb_occur and (c_occ1[0] <= occ1[i] <= c_occ1[1] and c_occx[0] <= occx[i] <= c_occx[1] and c_occ2[0] <= occ2[i] <= c_occ2[1] and c_occtot[0] <= occ_tot[i] <= c_occtot[1]): pts += 1

            if cb_super_macro:
                g_pass = 0
                b = sm_bounds
                if sum([b['1'][0] <= ones[i] <= b['1'][1], b['X'][0] <= draws[i] <= b['X'][1], b['2'][0] <= twos[i] <= b['2'][1]]) >= 2: g_pass += 1
                if sum([b['s1'][0] <= s1[i] <= b['s1'][1], b['sx'][0] <= sx[i] <= b['sx'][1], b['s2'][0] <= s2[i] <= b['s2'][1]]) >= 2: g_pass += 1
                if sum([b['g1'][0] <= g1[i] <= b['g1'][1], b['gx'][0] <= gx[i] <= b['gx'][1], b['g2'][0] <= g2[i] <= b['g2'][1]]) >= 2: g_pass += 1
                if sum([b['si1'][0] <= sing1[i] <= b['si1'][1], b['six'][0] <= singx[i] <= b['six'][1], b['si2'][0] <= sing2[i] <= b['si2'][1]]) >= 2: g_pass += 1
                if sum([b['d1'][0] <= dub1[i] <= b['d1'][1], b['dx'][0] <= dubx[i] <= b['dx'][1], b['d2'][0] <= dub2[i] <= b['d2'][1]]) >= 2: g_pass += 1
                if sum([b['t1'][0] <= trip1[i] <= b['t1'][1], b['tx'][0] <= tripx[i] <= b['tx'][1], b['t2'][0] <= trip2[i] <= b['t2'][1]]) >= 2: g_pass += 1
                if sum([b['o1'][0] <= occ1[i] <= b['o1'][1], b['ox'][0] <= occx[i] <= b['ox'][1], b['o2'][0] <= occ2[i] <= b['o2'][1]]) >= 2: g_pass += 1
                if sum([b['f'][0] <= fat_f[i] <= b['f'][1], b['a'][0] <= fat_a[i] <= b['a'][1], b['t'][0] <= fat_t[i] <= b['t'][1]]) >= 2: g_pass += 1
                if g_pass >= slider_super_groups: pts += 1
            
            if cb_fat and (c_fatf[0] <= fat_f[i] <= c_fatf[1] and c_fata[0] <= fat_a[i] <= c_fata[1] and c_fatt[0] <= fat_t[i] <= c_fatt[1] and c_fatsum[0] <= fat_sums[i] <= c_fatsum[1]): pts += 1
            if cb_fat_sequences and i < len(fat_strings_hist) and fat_sequence_group_ok_from_fat_string(fat_strings_hist[i]): pts += 1
            if cb_u_favs and (c_u[0] <= u_wins[i] <= c_u[1]): pts += 1
            if cb_sft and (c_sft[0] <= sft_sums[i] <= c_sft[1]): pts += 1
            if cb_points and (c_points[0] <= points_vals[i] <= c_points[1]): pts += 1
            if cb_100minus and (c_minus[0] <= minus_sums[i] <= c_minus[1]): pts += 1
            if cb_log_surprise and in_range(log_surprise_sums[i], c_log_surprise): pts += 1
            if cb_rank24 and (c_rank24[0] <= rank24_sums[i] <= c_rank24[1]): pts += 1
            if cb_totaldiff and (c_totaldiff[0] <= total_diff_vals[i] <= c_totaldiff[1]): pts += 1
            if cb_fav_pressure and (in_range(fav70_wins[i], c_fav70) and in_range(fav60_wins[i], c_fav60) and in_range(fav50_wins[i], c_fav50)): pts += 1
            if cb_shock_strength and (in_range(shock_u10[i], c_shock10) and in_range(shock_u15[i], c_shock15) and in_range(shock_u20[i], c_shock20) and in_range(shock_lowest[i], c_shock_lowest)): pts += 1
            if cb_fav_delta and in_range(fav_delta_vals[i], c_fav_delta): pts += 1
            if cb_aimatrix and (active_ai_min <= ai_ranks[i] <= active_ai_max): pts += 1
            
            history_filter_scores.append(pts)
            if pts == total_active: hard_all_hits += 1
            if pts >= slider_pass_req: mall_hits += 1

        # Historiskt styrt diagnostiskt poängkrav:
        # välj hårdaste krav X av N som fortfarande klarar minsta samlade mallträff.
        # Standard i sidomenyn är 67 %, dvs ungefär 20 av 30 omgångar.
        hist_total_for_soft = len(history_filter_scores)
        target_soft_hits = int(np.ceil(hist_total_for_soft * (float(slider_combined_min_hist_pct) / 100.0))) if hist_total_for_soft else 0
        target_soft_hits = max(1, min(hist_total_for_soft, target_soft_hits)) if hist_total_for_soft else 0
        soft_req_options = []
        if total_active > 0 and hist_total_for_soft > 0:
            for req in range(total_active, 0, -1):
                hits_req = sum(1 for s in history_filter_scores if s >= req)
                soft_req_options.append({
                    "Softkrav": req,
                    "Historiska träffar": hits_req,
                    "Historisk träff %": round((hits_req / hist_total_for_soft) * 100, 1),
                })
            if not manual_soft_req:
                valid_reqs = [r for r in soft_req_options if r["Historiska träffar"] >= target_soft_hits]
                if valid_reqs:
                    # Listan går från hårdast till mjukast, så första godkända är rätt val.
                    slider_pass_req = int(valid_reqs[0]["Softkrav"])
                else:
                    # Om inget krav klarar målet: använd mjukaste möjliga och visa varning.
                    slider_pass_req = 1

        # Slutligt krav väljs efter att både hårda gates och softpoäng har jämförts mot radmassan.
        # Vi väntar med visning tills AutoHard+AutoSoft har valt bästa kombination.
        mall_hits = sum(1 for s in history_filter_scores if s >= slider_pass_req)
        hard_all_pct = (hard_all_hits / len(v_m) * 100) if len(v_m) else 0.0
        soft_hit_pct = (mall_hits / len(v_m) * 100) if len(v_m) else 0.0

        # --- FILTERSTYRKA, FILTERREGLER OCH HELGARDERING-EXPORT ---
        # candidate_rows / total_candidates / match_odds_filter skapades redan ovan för AutoTrim.

        # Sammanlagd reducering: räkna hela mallen tillsammans, inte filter för filter.
        # För Topptips är detta exakt. För 13 matcher används ett stabilt Monte Carlo-estimat.
        cand_ai_matrix = cand_ai_scores_asc = cand_ai_tot = None
        if cb_aimatrix:
            cand_ai_matrix, cand_ai_scores_asc, cand_ai_tot = calculate_ai_matrix_from_values(filter_vec)

        def score_candidate_row(tr):
            pts = 0
            c1, cx, c2 = tr.count('1'), tr.count('X'), tr.count('2')
            s1_c, sx_c, s2_c, _ = get_streaks(tr)
            g1_c, gx_c, g2_c, _ = get_gaps(tr)
            si1_c, six_c, si2_c, singtot_c, _ = get_singles(tr)
            d1_c, dx_c, d2_c, dubtot_c, _ = get_doublets(tr)
            t1_c, tx_c, t2_c, triptot_c, _ = get_triplets(tr)
            o1_c, ox_c, o2_c, occtot_c, _ = get_occurrences(tr)
            f_c, a_c, t_c, fsum_c = get_fat(tr, filter_vec)

            if cb_base and (in_range(c1, c_ones) and in_range(cx, c_draws) and in_range(c2, c_twos)): pts += 1
            if cb_streak and (in_range(s1_c, c_s1) and in_range(sx_c, c_sx) and in_range(s2_c, c_s2)): pts += 1
            if cb_gap and (in_range(g1_c, c_g1) and in_range(gx_c, c_gx) and in_range(g2_c, c_g2)): pts += 1
            if cb_single and (in_range(si1_c, c_sing1) and in_range(six_c, c_singx) and in_range(si2_c, c_sing2) and in_range(singtot_c, c_singtot)): pts += 1
            if cb_doublet and (in_range(d1_c, c_dub1) and in_range(dx_c, c_dubx) and in_range(d2_c, c_dub2) and in_range(dubtot_c, c_dubtot)): pts += 1
            if cb_triplet and (in_range(t1_c, c_trip1) and in_range(tx_c, c_tripx) and in_range(t2_c, c_trip2) and in_range(triptot_c, c_triptot)): pts += 1
            if cb_occur and (in_range(o1_c, c_occ1) and in_range(ox_c, c_occx) and in_range(o2_c, c_occ2) and in_range(occtot_c, c_occtot)): pts += 1
            if cb_super_macro and pass_super_macro_row(tr, filter_vec, sm_bounds, slider_super_groups): pts += 1
            if cb_fat and (in_range(f_c, c_fatf) and in_range(a_c, c_fata) and in_range(t_c, c_fatt) and in_range(fsum_c, c_fatsum)): pts += 1
            if cb_fat_sequences and fat_sequence_group_ok_row(tr): pts += 1
            if cb_u_favs and in_range(get_top_n_favs_wins(tr, filter_vec, slider_u_count), c_u): pts += 1
            if cb_sft and in_range(get_sft_sum(tr, filter_vec), c_sft): pts += 1
            if cb_points and in_range(get_rank_points(tr, filter_vec), c_points): pts += 1
            if cb_100minus and in_range(get_100_minus_sum(tr, filter_vec), c_minus): pts += 1
            if cb_log_surprise and in_range(get_log_surprise_sum(tr, filter_vec), c_log_surprise): pts += 1
            if cb_rank24 and in_range(get_rank_sum(tr, filter_vec), c_rank24): pts += 1
            if cb_totaldiff and in_range(calculate_total_diff(match_odds_filter, list(tr)), c_totaldiff): pts += 1
            if cb_fav_pressure:
                fp_c = get_favorite_pressure(tr, filter_vec)
                if in_range(fp_c['F70_Wins'], c_fav70) and in_range(fp_c['F60_Wins'], c_fav60) and in_range(fp_c['F50_Wins'], c_fav50): pts += 1
            if cb_shock_strength:
                sh_c = get_shock_strength(tr, filter_vec)
                if in_range(sh_c['U10_Wins'], c_shock10) and in_range(sh_c['U15_Wins'], c_shock15) and in_range(sh_c['U20_Wins'], c_shock20) and in_range(sh_c['Lowest_Win_Pct'], c_shock_lowest): pts += 1
            if cb_fav_delta and in_range(get_favorite_delta(tr, filter_vec), c_fav_delta): pts += 1
            if cb_aimatrix and cand_ai_matrix is not None:
                rank_c, _ = get_exact_rank(tr, cand_ai_matrix, cand_ai_scores_asc, cand_ai_tot)
                if active_ai_min <= rank_c <= active_ai_max: pts += 1
            return pts

        # --- AUTOHARD EXAKT SPETSFILTER ---
        # Soft som aktivt filtersteg är borttaget. Vi använder i stället:
        # 1) breda hårda bas-gates
        # 2) enskilda hårda spetsfilter som bara väljs om de både klarar historikmålet
        #    och reducerar den kvarvarande radmassan tydligt.

        def _pass_ratio(bools, ratio=0.70):
            vals = [bool(x) for x in bools if x is not None]
            if not vals:
                return False
            req = max(1, int(np.ceil(len(vals) * ratio)))
            return sum(vals) >= req

        def _spets_history_checks(i):
            checks = {}

            # FAT-sekvenser och sekvenspaket
            fat_seq_checks = []
            if cb_fat_sequences and i < len(fat_strings_hist):
                fstr = fat_strings_hist[i]
                if fat_seq2_list:
                    v = in_range(count_fat_sequence_hits(fstr, fat_seq2_list), c_fat_seq2)
                    checks["FAT 2-sekvenser"] = v
                    fat_seq_checks.append(v)
                if fat_seq3_list:
                    v = in_range(count_fat_sequence_hits(fstr, fat_seq3_list), c_fat_seq3)
                    checks["FAT 3-sekvenser"] = v
                    fat_seq_checks.append(v)
                if fat_pair_list:
                    v = in_range(count_fat_pair_hits(fstr, fat_pair_list), c_fat_pairs)
                    checks["FAT dubbelchans"] = v
                    fat_seq_checks.append(v)
                if len(fat_seq_checks) >= 2:
                    checks["FAT-sekvenspaket"] = _pass_ratio(fat_seq_checks, 0.67)

            # Värde-/riskfilter, både enskilt och som paket.
            risk_checks = []
            if cb_points:
                v = in_range(points_vals[i], c_points)
                checks["Poängfilter smal"] = v
                risk_checks.append(v)
            if cb_rank24:
                v = in_range(rank24_sums[i], c_rank24)
                checks["Ranksumma smal"] = v
                risk_checks.append(v)
            if cb_log_surprise:
                v = in_range(log_surprise_sums[i], c_log_surprise)
                checks["Skrälltryck Log smal"] = v
                risk_checks.append(v)
            if cb_sft:
                v = in_range(sft_sums[i], c_sft)
                checks["SFT Summa smal"] = v
                risk_checks.append(v)
            if cb_100minus:
                v = in_range(minus_sums[i], c_minus)
                checks["100-minus smal"] = v
                risk_checks.append(v)
            if cb_totaldiff:
                v = in_range(total_diff_vals[i], c_totaldiff)
                checks["Total Diff smal"] = v
                risk_checks.append(v)
            if 'Delta' in v_m.columns:
                try:
                    v = in_range(list(v_m['Delta'])[i], c_delta)
                    checks["Delta smal"] = v
                    risk_checks.append(v)
                except Exception:
                    pass
            if cb_aimatrix and i < len(ai_ranks):
                v = in_range(ai_ranks[i], c_ai_rank)
                checks["AI-Rank smal"] = v
                risk_checks.append(v)
            if cb_fav_delta:
                v = in_range(fav_delta_vals[i], c_fav_delta)
                checks["Favorit-delta smal"] = v
                risk_checks.append(v)
            if len(risk_checks) >= 3:
                checks["Risk-/värdepaket smal"] = _pass_ratio(risk_checks, 0.70)

            # FAT-profil som helhet och som delkomponenter.
            fat_profile_checks = []
            if cb_fat:
                vf = in_range(fat_f[i], c_fatf)
                va = in_range(fat_a[i], c_fata)
                vt = in_range(fat_t[i], c_fatt)
                vs = in_range(fat_sums[i], c_fatsum)
                checks["FAT F smal"] = vf
                checks["FAT A smal"] = va
                checks["FAT T smal"] = vt
                checks["FAT Summa smal"] = vs
                checks["FAT-profil smal"] = vf and va and vt and vs
                fat_profile_checks.extend([vf, va, vt, vs])
                checks["FAT-profilpaket"] = _pass_ratio(fat_profile_checks, 0.75)

            # Favorit-/skrällbalans, både enskilt och paket.
            fav_shock_checks = []
            if cb_u_favs:
                v = in_range(u_wins[i], c_u)
                checks[f"Topp {slider_u_count} favoriter smal"] = v
                fav_shock_checks.append(v)
            if cb_fav_pressure:
                v70 = in_range(fav70_wins[i], c_fav70)
                v60 = in_range(fav60_wins[i], c_fav60)
                v50 = in_range(fav50_wins[i], c_fav50)
                checks["Favorittryck 70 smal"] = v70
                checks["Favorittryck 60 smal"] = v60
                checks["Favorittryck 50 smal"] = v50
                checks["Favorittryck smal"] = v70 and v60 and v50
                fav_shock_checks.extend([v70, v60, v50])
            if cb_shock_strength:
                vu10 = in_range(shock_u10[i], c_shock10)
                vu15 = in_range(shock_u15[i], c_shock15)
                vu20 = in_range(shock_u20[i], c_shock20)
                vlow = in_range(shock_lowest[i], c_shock_lowest)
                checks["Skrällstyrka U10 smal"] = vu10
                checks["Skrällstyrka U15 smal"] = vu15
                checks["Skrällstyrka U20 smal"] = vu20
                checks["Lägsta vinnare smal"] = vlow
                checks["Skrällstyrka smal"] = vu10 and vu15 and vu20 and vlow
                fav_shock_checks.extend([vu10, vu15, vu20, vlow])
            if len(fav_shock_checks) >= 3:
                checks["Favorit-/skrällpaket smal"] = _pass_ratio(fav_shock_checks, 0.70)

            # Strukturfilter: nu används även delkomponenter som hårda spetsfilter.
            struct_checks = []
            if cb_base:
                v1 = in_range(ones[i], c_ones)
                vx = in_range(draws[i], c_draws)
                v2 = in_range(twos[i], c_twos)
                checks["Antal 1 smal"] = v1
                checks["Antal X smal"] = vx
                checks["Antal 2 smal"] = v2
                checks["Teckenbalans 1X2 smal"] = v1 and vx and v2
                struct_checks.extend([v1, vx, v2])
            if cb_streak:
                v1 = in_range(s1[i], c_s1)
                vx = in_range(sx[i], c_sx)
                v2 = in_range(s2[i], c_s2)
                checks["Sviter 1 smal"] = v1
                checks["Sviter X smal"] = vx
                checks["Sviter 2 smal"] = v2
                checks["Sviter paket smal"] = v1 and vx and v2
                struct_checks.extend([v1, vx, v2])
            if cb_gap:
                v1 = in_range(g1[i], c_g1)
                vx = in_range(gx[i], c_gx)
                v2 = in_range(g2[i], c_g2)
                checks["Luckor 1 smal"] = v1
                checks["Luckor X smal"] = vx
                checks["Luckor 2 smal"] = v2
                checks["Luckor paket smal"] = v1 and vx and v2
                struct_checks.extend([v1, vx, v2])
            if cb_single:
                v1 = in_range(sing1[i], c_sing1)
                vx = in_range(singx[i], c_singx)
                v2 = in_range(sing2[i], c_sing2)
                vt = in_range(sing_tot[i], c_singtot)
                checks["Singlar 1 smal"] = v1
                checks["Singlar X smal"] = vx
                checks["Singlar 2 smal"] = v2
                checks["Singlar total smal"] = vt
                checks["Singlar paket smal"] = v1 and vx and v2 and vt
                struct_checks.extend([v1, vx, v2, vt])
            if cb_doublet:
                v1 = in_range(dub1[i], c_dub1)
                vx = in_range(dubx[i], c_dubx)
                v2 = in_range(dub2[i], c_dub2)
                vt = in_range(dub_tot[i], c_dubtot)
                checks["Dubbletter 1 smal"] = v1
                checks["Dubbletter X smal"] = vx
                checks["Dubbletter 2 smal"] = v2
                checks["Dubbletter total smal"] = vt
                checks["Dubbletter paket smal"] = v1 and vx and v2 and vt
                struct_checks.extend([v1, vx, v2, vt])
            if cb_triplet:
                v1 = in_range(trip1[i], c_trip1)
                vx = in_range(tripx[i], c_tripx)
                v2 = in_range(trip2[i], c_trip2)
                vt = in_range(trip_tot[i], c_triptot)
                checks["Tripplar 1 smal"] = v1
                checks["Tripplar X smal"] = vx
                checks["Tripplar 2 smal"] = v2
                checks["Tripplar total smal"] = vt
                checks["Tripplar paket smal"] = v1 and vx and v2 and vt
                struct_checks.extend([v1, vx, v2, vt])
            if cb_occur:
                v1 = in_range(occ1[i], c_occ1)
                vx = in_range(occx[i], c_occx)
                v2 = in_range(occ2[i], c_occ2)
                vt = in_range(occ_tot[i], c_occtot)
                checks["Uppkomster 1 smal"] = v1
                checks["Uppkomster X smal"] = vx
                checks["Uppkomster 2 smal"] = v2
                checks["Uppkomster total smal"] = vt
                checks["Uppkomster paket smal"] = v1 and vx and v2 and vt
                struct_checks.extend([v1, vx, v2, vt])
            if len(struct_checks) >= 5:
                checks["Strukturpaket smal"] = _pass_ratio(struct_checks, 0.72)

            for _vname, _vdef in spets_variant_defs.items():
                try:
                    if i < len(_vdef.get("hist_values", [])):
                        checks[_vname] = in_range(_vdef["hist_values"][i], _vdef["interval"])
                except Exception:
                    pass

            if cb_super_macro:
                checks["Super-Makro"] = _super_macro_hist_ok(i)
            return checks

        def _spets_candidate_checks(tr):
            checks = {}
            c1, cx, c2 = tr.count('1'), tr.count('X'), tr.count('2')
            s1_c, sx_c, s2_c, _ = get_streaks(tr)
            g1_c, gx_c, g2_c, _ = get_gaps(tr)
            si1_c, six_c, si2_c, singtot_c, _ = get_singles(tr)
            d1_c, dx_c, d2_c, dubtot_c, _ = get_doublets(tr)
            t1_c, tx_c, t2_c, triptot_c, _ = get_triplets(tr)
            o1_c, ox_c, o2_c, occtot_c, _ = get_occurrences(tr)
            f_c, a_c, t_c, fsum_c = get_fat(tr, filter_vec)

            fat_seq_checks = []
            if cb_fat_sequences:
                fstr = get_fat_string(tr, filter_vec)
                if fat_seq2_list:
                    v = in_range(count_fat_sequence_hits(fstr, fat_seq2_list), c_fat_seq2)
                    checks["FAT 2-sekvenser"] = v
                    fat_seq_checks.append(v)
                if fat_seq3_list:
                    v = in_range(count_fat_sequence_hits(fstr, fat_seq3_list), c_fat_seq3)
                    checks["FAT 3-sekvenser"] = v
                    fat_seq_checks.append(v)
                if fat_pair_list:
                    v = in_range(count_fat_pair_hits(fstr, fat_pair_list), c_fat_pairs)
                    checks["FAT dubbelchans"] = v
                    fat_seq_checks.append(v)
                if len(fat_seq_checks) >= 2:
                    checks["FAT-sekvenspaket"] = _pass_ratio(fat_seq_checks, 0.67)

            risk_checks = []
            if cb_points:
                v = in_range(get_rank_points(tr, filter_vec), c_points)
                checks["Poängfilter smal"] = v
                risk_checks.append(v)
            if cb_rank24:
                v = in_range(get_rank_sum(tr, filter_vec), c_rank24)
                checks["Ranksumma smal"] = v
                risk_checks.append(v)
            if cb_log_surprise:
                v = in_range(get_log_surprise_sum(tr, filter_vec), c_log_surprise)
                checks["Skrälltryck Log smal"] = v
                risk_checks.append(v)
            if cb_sft:
                v = in_range(get_sft_sum(tr, filter_vec), c_sft)
                checks["SFT Summa smal"] = v
                risk_checks.append(v)
            if cb_100minus:
                v = in_range(get_100_minus_sum(tr, filter_vec), c_minus)
                checks["100-minus smal"] = v
                risk_checks.append(v)
            if cb_totaldiff:
                v = in_range(calculate_total_diff(match_odds_filter, list(tr)), c_totaldiff)
                checks["Total Diff smal"] = v
                risk_checks.append(v)
            try:
                v = in_range(calculate_delta(tr, filter_vec), c_delta)
                checks["Delta smal"] = v
                risk_checks.append(v)
            except Exception:
                pass
            if cb_aimatrix and spets_ai_matrix is not None:
                try:
                    rank_c, _ = get_exact_rank(tr, spets_ai_matrix, spets_ai_scores_asc, spets_ai_tot)
                    v = in_range(rank_c, c_ai_rank)
                    checks["AI-Rank smal"] = v
                    risk_checks.append(v)
                except Exception:
                    pass
            if cb_fav_delta:
                v = in_range(get_favorite_delta(tr, filter_vec), c_fav_delta)
                checks["Favorit-delta smal"] = v
                risk_checks.append(v)
            if len(risk_checks) >= 3:
                checks["Risk-/värdepaket smal"] = _pass_ratio(risk_checks, 0.70)

            if cb_fat:
                vf = in_range(f_c, c_fatf)
                va = in_range(a_c, c_fata)
                vt = in_range(t_c, c_fatt)
                vs = in_range(fsum_c, c_fatsum)
                checks["FAT F smal"] = vf
                checks["FAT A smal"] = va
                checks["FAT T smal"] = vt
                checks["FAT Summa smal"] = vs
                checks["FAT-profil smal"] = vf and va and vt and vs
                checks["FAT-profilpaket"] = _pass_ratio([vf, va, vt, vs], 0.75)

            fav_shock_checks = []
            if cb_u_favs:
                v = in_range(get_top_n_favs_wins(tr, filter_vec, slider_u_count), c_u)
                checks[f"Topp {slider_u_count} favoriter smal"] = v
                fav_shock_checks.append(v)
            if cb_fav_pressure:
                fp_c = get_favorite_pressure(tr, filter_vec)
                v70 = in_range(fp_c['F70_Wins'], c_fav70)
                v60 = in_range(fp_c['F60_Wins'], c_fav60)
                v50 = in_range(fp_c['F50_Wins'], c_fav50)
                checks["Favorittryck 70 smal"] = v70
                checks["Favorittryck 60 smal"] = v60
                checks["Favorittryck 50 smal"] = v50
                checks["Favorittryck smal"] = v70 and v60 and v50
                fav_shock_checks.extend([v70, v60, v50])
            if cb_shock_strength:
                sh_c = get_shock_strength(tr, filter_vec)
                vu10 = in_range(sh_c['U10_Wins'], c_shock10)
                vu15 = in_range(sh_c['U15_Wins'], c_shock15)
                vu20 = in_range(sh_c['U20_Wins'], c_shock20)
                vlow = in_range(sh_c['Lowest_Win_Pct'], c_shock_lowest)
                checks["Skrällstyrka U10 smal"] = vu10
                checks["Skrällstyrka U15 smal"] = vu15
                checks["Skrällstyrka U20 smal"] = vu20
                checks["Lägsta vinnare smal"] = vlow
                checks["Skrällstyrka smal"] = vu10 and vu15 and vu20 and vlow
                fav_shock_checks.extend([vu10, vu15, vu20, vlow])
            if len(fav_shock_checks) >= 3:
                checks["Favorit-/skrällpaket smal"] = _pass_ratio(fav_shock_checks, 0.70)

            struct_checks = []
            if cb_base:
                v1 = in_range(c1, c_ones)
                vx = in_range(cx, c_draws)
                v2 = in_range(c2, c_twos)
                checks["Antal 1 smal"] = v1
                checks["Antal X smal"] = vx
                checks["Antal 2 smal"] = v2
                checks["Teckenbalans 1X2 smal"] = v1 and vx and v2
                struct_checks.extend([v1, vx, v2])
            if cb_streak:
                v1 = in_range(s1_c, c_s1)
                vx = in_range(sx_c, c_sx)
                v2 = in_range(s2_c, c_s2)
                checks["Sviter 1 smal"] = v1
                checks["Sviter X smal"] = vx
                checks["Sviter 2 smal"] = v2
                checks["Sviter paket smal"] = v1 and vx and v2
                struct_checks.extend([v1, vx, v2])
            if cb_gap:
                v1 = in_range(g1_c, c_g1)
                vx = in_range(gx_c, c_gx)
                v2 = in_range(g2_c, c_g2)
                checks["Luckor 1 smal"] = v1
                checks["Luckor X smal"] = vx
                checks["Luckor 2 smal"] = v2
                checks["Luckor paket smal"] = v1 and vx and v2
                struct_checks.extend([v1, vx, v2])
            if cb_single:
                v1 = in_range(si1_c, c_sing1)
                vx = in_range(six_c, c_singx)
                v2 = in_range(si2_c, c_sing2)
                vt = in_range(singtot_c, c_singtot)
                checks["Singlar 1 smal"] = v1
                checks["Singlar X smal"] = vx
                checks["Singlar 2 smal"] = v2
                checks["Singlar total smal"] = vt
                checks["Singlar paket smal"] = v1 and vx and v2 and vt
                struct_checks.extend([v1, vx, v2, vt])
            if cb_doublet:
                v1 = in_range(d1_c, c_dub1)
                vx = in_range(dx_c, c_dubx)
                v2 = in_range(d2_c, c_dub2)
                vt = in_range(dubtot_c, c_dubtot)
                checks["Dubbletter 1 smal"] = v1
                checks["Dubbletter X smal"] = vx
                checks["Dubbletter 2 smal"] = v2
                checks["Dubbletter total smal"] = vt
                checks["Dubbletter paket smal"] = v1 and vx and v2 and vt
                struct_checks.extend([v1, vx, v2, vt])
            if cb_triplet:
                v1 = in_range(t1_c, c_trip1)
                vx = in_range(tx_c, c_tripx)
                v2 = in_range(t2_c, c_trip2)
                vt = in_range(triptot_c, c_triptot)
                checks["Tripplar 1 smal"] = v1
                checks["Tripplar X smal"] = vx
                checks["Tripplar 2 smal"] = v2
                checks["Tripplar total smal"] = vt
                checks["Tripplar paket smal"] = v1 and vx and v2 and vt
                struct_checks.extend([v1, vx, v2, vt])
            if cb_occur:
                v1 = in_range(o1_c, c_occ1)
                vx = in_range(ox_c, c_occx)
                v2 = in_range(o2_c, c_occ2)
                vt = in_range(occtot_c, c_occtot)
                checks["Uppkomster 1 smal"] = v1
                checks["Uppkomster X smal"] = vx
                checks["Uppkomster 2 smal"] = v2
                checks["Uppkomster total smal"] = vt
                checks["Uppkomster paket smal"] = v1 and vx and v2 and vt
                struct_checks.extend([v1, vx, v2, vt])
            if len(struct_checks) >= 5:
                checks["Strukturpaket smal"] = _pass_ratio(struct_checks, 0.72)

            for _vname, _vdef in spets_variant_defs.items():
                try:
                    checks[_vname] = in_range(_vdef["getter"](tr), _vdef["interval"])
                except Exception:
                    pass

            if cb_super_macro:
                checks["Super-Makro"] = pass_super_macro_row(tr, filter_vec, sm_bounds, slider_super_groups)
            return checks

        # --- VISNING: filterregler/intervall för valda bas- och spetsfilter ---
        def _short_join(items, max_items=6):
            items = [str(x) for x in items if str(x)]
            if not items:
                return "-"
            if len(items) <= max_items:
                return ", ".join(items)
            return ", ".join(items[:max_items]) + f" +{len(items)-max_items} till"

        def _parts(*items):
            return " | ".join([str(x) for x in items if str(x)])

        def _maybe_interval(label, interval, decimals=0):
            return f"{label}: {fmt_interval(interval, decimals)}"

        def _spets_rule_text(name):
            """Kort Helgardering-liknande regeltext för valt spetsfilter."""
            if name in spets_variant_defs:
                _v = spets_variant_defs[name]
                return f"{_v['label']}: {fmt_interval(_v['interval'], _v.get('decimals', 0))} | variant {int(_v.get('coverage', 0))}%"
            dynamic_top = f"Topp {slider_u_count} favoriter smal"
            rules = {
                "FAT 2-sekvenser": f"Träffar {fmt_interval(c_fat_seq2)} av toppsekvenser: {_short_join(fat_seq2_list)}",
                "FAT 3-sekvenser": f"Träffar {fmt_interval(c_fat_seq3)} av toppsekvenser: {_short_join(fat_seq3_list)}",
                "FAT dubbelchans": f"Träffar {fmt_interval(c_fat_pairs)} av topp-par: {_short_join(fat_pair_list)}",
                "FAT-sekvenspaket": _parts(
                    f"Minst 2 av aktiva FAT-sekvensdelar",
                    f"2-seq {fmt_interval(c_fat_seq2)}: {_short_join(fat_seq2_list, 4)}",
                    f"3-seq {fmt_interval(c_fat_seq3)}: {_short_join(fat_seq3_list, 4)}",
                    f"dubbel {fmt_interval(c_fat_pairs)}: {_short_join(fat_pair_list, 4)}",
                ),
                "Poängfilter smal": _maybe_interval("Poängfilter", c_points, 0),
                "Ranksumma smal": _maybe_interval("Rank Summa", c_rank24, 1),
                "Skrälltryck Log smal": _maybe_interval("Skrälltryck Log", c_log_surprise, 0),
                "SFT Summa smal": _maybe_interval("SFT Summa", c_sft, 1),
                "100-minus smal": _maybe_interval("100-minus", c_minus, 1),
                "Total Diff smal": _maybe_interval("Total Diff", c_totaldiff, 0),
                "Delta smal": _maybe_interval("Delta", c_delta, 1),
                "AI-Rank smal": _maybe_interval("AI-Rank", c_ai_rank, 0),
                "Favorit-delta smal": _maybe_interval("Favorit-delta", c_fav_delta, 2),
                "Risk-/värdepaket smal": _parts(
                    "Minst ca 70% av aktiva risk/värde-delar",
                    _maybe_interval("Poäng", c_points, 0),
                    _maybe_interval("Rank", c_rank24, 1),
                    _maybe_interval("SFT", c_sft, 1),
                    _maybe_interval("Log", c_log_surprise, 0),
                    _maybe_interval("100-minus", c_minus, 1),
                    _maybe_interval("TotalDiff", c_totaldiff, 0),
                    _maybe_interval("Delta", c_delta, 1),
                    _maybe_interval("AI-Rank", c_ai_rank, 0),
                ),
                "FAT F smal": _maybe_interval("F", c_fatf, 0),
                "FAT A smal": _maybe_interval("A", c_fata, 0),
                "FAT T smal": _maybe_interval("T", c_fatt, 0),
                "FAT Summa smal": _maybe_interval("FAT Summa", c_fatsum, 0),
                "FAT-profil smal": _parts(_maybe_interval("F", c_fatf, 0), _maybe_interval("A", c_fata, 0), _maybe_interval("T", c_fatt, 0), _maybe_interval("Summa", c_fatsum, 0)),
                "FAT-profilpaket": _parts("Minst 3 av 4 FAT-delar", _maybe_interval("F", c_fatf, 0), _maybe_interval("A", c_fata, 0), _maybe_interval("T", c_fatt, 0), _maybe_interval("Summa", c_fatsum, 0)),
                dynamic_top: f"{fmt_interval(c_u)} vinster bland topp {slider_u_count} favoriter",
                "Favorittryck 70 smal": _maybe_interval("F70-vinster", c_fav70, 0),
                "Favorittryck 60 smal": _maybe_interval("F60-vinster", c_fav60, 0),
                "Favorittryck 50 smal": _maybe_interval("F50-vinster", c_fav50, 0),
                "Favorittryck smal": _parts(_maybe_interval("F70", c_fav70, 0), _maybe_interval("F60", c_fav60, 0), _maybe_interval("F50", c_fav50, 0)),
                "Skrällstyrka U10 smal": _maybe_interval("U10-vinster", c_shock10, 0),
                "Skrällstyrka U15 smal": _maybe_interval("U15-vinster", c_shock15, 0),
                "Skrällstyrka U20 smal": _maybe_interval("U20-vinster", c_shock20, 0),
                "Lägsta vinnare smal": _maybe_interval("Lägsta vinnarprocent", c_shock_lowest, 1),
                "Skrällstyrka smal": _parts(_maybe_interval("U10", c_shock10, 0), _maybe_interval("U15", c_shock15, 0), _maybe_interval("U20", c_shock20, 0), _maybe_interval("Lägsta", c_shock_lowest, 1)),
                "Favorit-/skrällpaket smal": _parts(
                    "Minst ca 70% av aktiva favorit-/skrälldelar",
                    _maybe_interval("Toppfav", c_u, 0),
                    _maybe_interval("F70", c_fav70, 0),
                    _maybe_interval("F60", c_fav60, 0),
                    _maybe_interval("F50", c_fav50, 0),
                    _maybe_interval("U10", c_shock10, 0),
                    _maybe_interval("U15", c_shock15, 0),
                    _maybe_interval("U20", c_shock20, 0),
                ),
                "Antal 1 smal": _maybe_interval("Antal 1", c_ones, 0),
                "Antal X smal": _maybe_interval("Antal X", c_draws, 0),
                "Antal 2 smal": _maybe_interval("Antal 2", c_twos, 0),
                "Teckenbalans 1X2 smal": _parts(_maybe_interval("1", c_ones, 0), _maybe_interval("X", c_draws, 0), _maybe_interval("2", c_twos, 0)),
                "Sviter 1 smal": _maybe_interval("Sviter 1", c_s1, 0),
                "Sviter X smal": _maybe_interval("Sviter X", c_sx, 0),
                "Sviter 2 smal": _maybe_interval("Sviter 2", c_s2, 0),
                "Sviter paket smal": _parts(_maybe_interval("S1", c_s1, 0), _maybe_interval("SX", c_sx, 0), _maybe_interval("S2", c_s2, 0)),
                "Luckor 1 smal": _maybe_interval("Luckor 1", c_g1, 0),
                "Luckor X smal": _maybe_interval("Luckor X", c_gx, 0),
                "Luckor 2 smal": _maybe_interval("Luckor 2", c_g2, 0),
                "Luckor paket smal": _parts(_maybe_interval("G1", c_g1, 0), _maybe_interval("GX", c_gx, 0), _maybe_interval("G2", c_g2, 0)),
                "Singlar 1 smal": _maybe_interval("Singlar 1", c_sing1, 0),
                "Singlar X smal": _maybe_interval("Singlar X", c_singx, 0),
                "Singlar 2 smal": _maybe_interval("Singlar 2", c_sing2, 0),
                "Singlar total smal": _maybe_interval("Singlar total", c_singtot, 0),
                "Singlar paket smal": _parts(_maybe_interval("Si1", c_sing1, 0), _maybe_interval("SiX", c_singx, 0), _maybe_interval("Si2", c_sing2, 0), _maybe_interval("Tot", c_singtot, 0)),
                "Dubbletter 1 smal": _maybe_interval("Dubbletter 1", c_dub1, 0),
                "Dubbletter X smal": _maybe_interval("Dubbletter X", c_dubx, 0),
                "Dubbletter 2 smal": _maybe_interval("Dubbletter 2", c_dub2, 0),
                "Dubbletter total smal": _maybe_interval("Dubbletter total", c_dubtot, 0),
                "Dubbletter paket smal": _parts(_maybe_interval("D1", c_dub1, 0), _maybe_interval("DX", c_dubx, 0), _maybe_interval("D2", c_dub2, 0), _maybe_interval("Tot", c_dubtot, 0)),
                "Tripplar 1 smal": _maybe_interval("Tripplar 1", c_trip1, 0),
                "Tripplar X smal": _maybe_interval("Tripplar X", c_tripx, 0),
                "Tripplar 2 smal": _maybe_interval("Tripplar 2", c_trip2, 0),
                "Tripplar total smal": _maybe_interval("Tripplar total", c_triptot, 0),
                "Tripplar paket smal": _parts(_maybe_interval("T1", c_trip1, 0), _maybe_interval("TX", c_tripx, 0), _maybe_interval("T2", c_trip2, 0), _maybe_interval("Tot", c_triptot, 0)),
                "Uppkomster 1 smal": _maybe_interval("Uppkomster 1", c_occ1, 0),
                "Uppkomster X smal": _maybe_interval("Uppkomster X", c_occx, 0),
                "Uppkomster 2 smal": _maybe_interval("Uppkomster 2", c_occ2, 0),
                "Uppkomster total smal": _maybe_interval("Uppkomster total", c_occtot, 0),
                "Uppkomster paket smal": _parts(_maybe_interval("O1", c_occ1, 0), _maybe_interval("OX", c_occx, 0), _maybe_interval("O2", c_occ2, 0), _maybe_interval("Tot", c_occtot, 0)),
                "Strukturpaket smal": "Minst ca 72% av aktiva strukturdelar: 1/X/2, sviter, luckor, singlar, dubbletter, tripplar och uppkomster",
                "Super-Makro": f"Super-Makro: minst {slider_super_groups} av strukturgrupper inom breda intervall",
            }
            return rules.get(name, name)

        def _base_gate_rule_rows():
            rows = []
            risk_parts = []
            if cb_sft: risk_parts.append(_maybe_interval("SFT", h_sft, 1))
            if cb_points: risk_parts.append(_maybe_interval("Poäng", h_points, 0))
            if cb_100minus: risk_parts.append(_maybe_interval("100-minus", h_minus, 1))
            if cb_log_surprise: risk_parts.append(_maybe_interval("Log", h_log_surprise, 0))
            if cb_rank24: risk_parts.append(_maybe_interval("Rank", h_rank24, 1))
            if cb_totaldiff: risk_parts.append(_maybe_interval("TotalDiff", h_totaldiff, 0))
            if risk_parts:
                rows.append({"Bas-gate": "Risk/värde bred", "Krav": f"Minst {_gate_req(len(risk_parts), 0.67)} av {len(risk_parts)} delar", "Intervall/regler": _parts(*risk_parts)})
            if cb_fat:
                rows.append({"Bas-gate": "FAT-profil bred", "Krav": "Alla FAT-delar inom intervall", "Intervall/regler": _parts(_maybe_interval("F", c_fatf, 0), _maybe_interval("A", c_fata, 0), _maybe_interval("T", c_fatt, 0), _maybe_interval("Summa", c_fatsum, 0))})
            seq_parts = []
            if cb_fat_sequences:
                if fat_seq2_list: seq_parts.append(f"2-seq {fmt_interval(h_fat_seq2)}: {_short_join(fat_seq2_list, 4)}")
                if fat_seq3_list: seq_parts.append(f"3-seq {fmt_interval(h_fat_seq3)}: {_short_join(fat_seq3_list, 4)}")
                if fat_pair_list: seq_parts.append(f"dubbel {fmt_interval(h_fat_pairs)}: {_short_join(fat_pair_list, 4)}")
            if seq_parts:
                rows.append({"Bas-gate": "FAT-sekvenser bred", "Krav": f"Minst {max(1, min(2, len(seq_parts)))} av {len(seq_parts)} delar", "Intervall/regler": _parts(*seq_parts)})
            struct_parts = []
            if cb_base: struct_parts.append(_parts(_maybe_interval("1", c_ones, 0), _maybe_interval("X", c_draws, 0), _maybe_interval("2", c_twos, 0)))
            if cb_streak: struct_parts.append(_parts(_maybe_interval("S1", c_s1, 0), _maybe_interval("SX", c_sx, 0), _maybe_interval("S2", c_s2, 0)))
            if cb_gap: struct_parts.append(_parts(_maybe_interval("G1", c_g1, 0), _maybe_interval("GX", c_gx, 0), _maybe_interval("G2", c_g2, 0)))
            if cb_single: struct_parts.append(_parts(_maybe_interval("SingTot", c_singtot, 0)))
            if cb_doublet: struct_parts.append(_parts(_maybe_interval("DubTot", c_dubtot, 0)))
            if cb_triplet: struct_parts.append(_parts(_maybe_interval("TripTot", c_triptot, 0)))
            if cb_occur: struct_parts.append(_parts(_maybe_interval("OccTot", c_occtot, 0)))
            if struct_parts:
                rows.append({"Bas-gate": "Struktur bred", "Krav": f"Minst {_gate_req(len(struct_parts), 0.60)} av {len(struct_parts)} strukturgrupper", "Intervall/regler": _parts(*struct_parts)})
            if cb_super_macro:
                rows.append({"Bas-gate": "Super-Makro bred", "Krav": f"Minst {slider_super_groups} grupper", "Intervall/regler": "Breda strukturintervall från Super-Makro"})
            fs_parts = []
            if cb_u_favs: fs_parts.append(_maybe_interval(f"Topp {slider_u_count} fav", h_u, 0))
            if cb_fav_pressure: fs_parts.append(_parts(_maybe_interval("F70", c_fav70, 0), _maybe_interval("F60", c_fav60, 0), _maybe_interval("F50", c_fav50, 0)))
            if cb_shock_strength: fs_parts.append(_parts(_maybe_interval("U10", c_shock10, 0), _maybe_interval("U15", c_shock15, 0), _maybe_interval("U20", c_shock20, 0), _maybe_interval("Lägsta", c_shock_lowest, 1)))
            if cb_fav_delta: fs_parts.append(_maybe_interval("Fav-delta", h_fav_delta, 2))
            if fs_parts:
                rows.append({"Bas-gate": "Favorit/skräll bred", "Krav": f"Minst {_gate_req(len(fs_parts), 0.50)} av {len(fs_parts)} delar", "Intervall/regler": _parts(*fs_parts)})
            return rows

        selected_spets_names = []
        def spets_candidate_passes(tr, names=None):
            names = selected_spets_names if names is None else names
            if not names:
                return True
            checks = _spets_candidate_checks(tr)
            return all(checks.get(name, False) for name in names)

        combined_soft_survivors = combined_hard_survivors = 0
        combined_score_counts = {}
        combined_hard_score_counts = {}
        combined_matrix_counts = {}
        candidate_hard_scores = []
        if total_candidates > 0:
            for tr in candidate_rows:
                pts_c = score_candidate_row(tr) if total_active > 0 else 0
                hpts_c = hard_candidate_score(tr) if hard_gate_total > 0 else 0
                candidate_hard_scores.append(hpts_c)
                combined_score_counts[pts_c] = combined_score_counts.get(pts_c, 0) + 1
                combined_hard_score_counts[hpts_c] = combined_hard_score_counts.get(hpts_c, 0) + 1
                combined_matrix_counts[(hpts_c, pts_c)] = combined_matrix_counts.get((hpts_c, pts_c), 0) + 1
        combined_est_label = "exakt" if exact_universe else "estimat"

        # Välj breda bas-gates först. Detta är fortfarande ett hårt krav, men byggt som N av M gates.
        hard_gate_decision_rows = []
        selected_hard_req = 0
        if hard_gate_total > 0 and hist_total_for_soft > 0:
            valid_hard = []
            for hreq in range(hard_gate_total, 0, -1):
                hist_hits_h = sum(1 for hs in history_hard_scores if hs >= hreq)
                keep_rows_h = sum(cnt for score, cnt in combined_hard_score_counts.items() if score >= hreq)
                keep_pct_h = (keep_rows_h / total_candidates * 100) if total_candidates else 0.0
                row = {
                    "Hårdkrav": hreq,
                    "Historiska träffar": hist_hits_h,
                    "Historisk träff %": round((hist_hits_h / hist_total_for_soft) * 100, 1),
                    "Kvar rader": int(keep_rows_h),
                    "Kvar rad %": round(keep_pct_h, 1),
                    "Reducerar %": round(100 - keep_pct_h, 1),
                    "Godkänd mot mål": hist_hits_h >= target_soft_hits,
                }
                hard_gate_decision_rows.append(row)
                if row["Godkänd mot mål"]:
                    valid_hard.append(row)
            if valid_hard:
                chosen_hard = min(valid_hard, key=lambda r: (r["Kvar rader"], -r["Hårdkrav"], -r["Historiska träffar"]))
                selected_hard_req = int(chosen_hard["Hårdkrav"])
            else:
                selected_hard_req = 1

        hard_only_hist_mask = [(hs >= selected_hard_req if hard_gate_total > 0 else True) for hs in history_hard_scores]
        hard_only_cand_mask = [(hs >= selected_hard_req if hard_gate_total > 0 else True) for hs in candidate_hard_scores]
        hard_stage_survivors = int(sum(hard_only_cand_mask)) if total_candidates else 0
        hard_stage_keep_pct = (hard_stage_survivors / total_candidates * 100) if total_candidates else 0.0

        # Precomputera spetsfilter en gång på kandidat-estimatet och historiken.
        all_spets_names = []
        for i in range(len(v_m)):
            for name in _spets_history_checks(i).keys():
                if name not in all_spets_names:
                    all_spets_names.append(name)
        hist_spets_map = {name: [] for name in all_spets_names}
        for i in range(len(v_m)):
            checks = _spets_history_checks(i)
            for name in all_spets_names:
                hist_spets_map[name].append(bool(checks.get(name, False)))
        cand_spets_map = {name: [] for name in all_spets_names}
        for tr in candidate_rows:
            checks = _spets_candidate_checks(tr)
            for name in all_spets_names:
                cand_spets_map[name].append(bool(checks.get(name, False)))

        # v11.1u: Exakt filteroptimering på manuell grundram. Först görs en begränsad kombinationssökning,
        # därefter fortsätter appen greedigt att lägga till varje hårt spetsfilter som
        # fortfarande klarar samlad mallträff, reducerar radmassan och senare klarar teckenskydd.
        min_spets_reduction_pct = 5.0
        relaxed_spets_reduction_pct = 5.0
        # Normal-läget ska inte stoppa vid 3–8 filter. Använd så många hårda filter som går.
        max_spets_filters = max(1, min(40, len(all_spets_names)))
        beam_width = 20
        beam_search_depth = min(12, max_spets_filters)
        desired_estimated_after_spets = min(int(tm_filter_limit), 1500) if 'tm_filter_limit' in locals() else 1500

        current_hist_mask = list(hard_only_hist_mask)
        current_cand_mask = list(hard_only_cand_mask)
        selected_spets_rows = []
        skipped_spets_rows = []
        available_spets = list(all_spets_names)

        # Gör snabba bool-arrayer. De används bara här i urvalet, inte i UI.
        hist_np = {name: np.array(hist_spets_map.get(name, []), dtype=bool) for name in all_spets_names}
        cand_np = {name: np.array(cand_spets_map.get(name, []), dtype=bool) for name in all_spets_names}
        base_hist_arr = np.array(hard_only_hist_mask, dtype=bool)
        base_cand_arr = np.array(hard_only_cand_mask, dtype=bool)

        # Förhandsfiltrera kandidater som aldrig kan vara relevanta ens direkt efter basfilter.
        initial_options = []
        base_rows = int(base_cand_arr.sum())
        for name in available_spets:
            h_arr = base_hist_arr & hist_np[name]
            c_arr = base_cand_arr & cand_np[name]
            hist_hits = int(h_arr.sum())
            keep_rows = int(c_arr.sum())
            red_pct = ((base_rows - keep_rows) / base_rows * 100) if base_rows else 0.0
            row = {
                "Filter": name,
                "Intervall/regler": _spets_rule_text(name),
                "Före rader": base_rows,
                "Efter rader": keep_rows,
                "Reducerar steg %": round(red_pct, 1),
                "Kvar total %": round((keep_rows / total_candidates * 100), 1) if total_candidates else 0.0,
                "Historisk träff": hist_hits,
                "Historisk träff %": round((hist_hits / hist_total_for_soft) * 100, 1) if hist_total_for_soft else 0.0,
                "Godkänd historik": hist_hits >= target_soft_hits,
                "Godkänd reduktion": keep_rows < base_rows,
            }
            if row["Godkänd historik"] and keep_rows > 0 and keep_rows < base_rows:
                initial_options.append((name, keep_rows, red_pct, hist_hits))
            else:
                skipped_spets_rows.append(row)

        # Behåll de mest intressanta filtren för kombinationssökningen.
        # Sortering: få rader kvar, hög reducering, hög historisk träff.
        candidate_filter_names = [x[0] for x in sorted(initial_options, key=lambda x: (x[1], -x[2], -x[3]))[:60]]

        # state = (names_tuple, hist_arr, cand_arr, rows_detail_list)
        beam = [(tuple(), base_hist_arr, base_cand_arr, [])]
        best_state = beam[0]
        all_examined_rows = []

        for depth in range(beam_search_depth):
            next_states = []
            seen = set()
            for names_tuple, hist_arr, cand_arr, rows_detail in beam:
                current_rows = int(cand_arr.sum())
                if current_rows <= 0:
                    continue
                # När vi redan är nere under målnivån kräver vi normal effekt; ovanför målnivån
                # tillåts även mindre, men fortfarande mätbar, reducering för att komma vidare.
                step_min_red = relaxed_spets_reduction_pct if current_rows > desired_estimated_after_spets else min_spets_reduction_pct
                for name in candidate_filter_names:
                    if name in names_tuple:
                        continue
                    new_hist_arr = hist_arr & hist_np[name]
                    hist_hits = int(new_hist_arr.sum())
                    if hist_hits < target_soft_hits:
                        continue
                    new_cand_arr = cand_arr & cand_np[name]
                    keep_rows = int(new_cand_arr.sum())
                    if keep_rows <= 0 or keep_rows >= current_rows:
                        continue
                    red_pct = ((current_rows - keep_rows) / current_rows * 100) if current_rows else 0.0
                    if red_pct < step_min_red:
                        continue
                    new_names = tuple(list(names_tuple) + [name])
                    sig = tuple(sorted(new_names))
                    if sig in seen:
                        continue
                    seen.add(sig)
                    detail_row = {
                        "Filter": name,
                        "Intervall/regler": _spets_rule_text(name),
                        "Före rader": current_rows,
                        "Efter rader": keep_rows,
                        "Reducerar steg %": round(red_pct, 1),
                        "Kvar total %": round((keep_rows / total_candidates * 100), 1) if total_candidates else 0.0,
                        "Historisk träff": hist_hits,
                        "Historisk träff %": round((hist_hits / hist_total_for_soft) * 100, 1) if hist_total_for_soft else 0.0,
                        "Godkänd historik": True,
                        "Godkänd reduktion": True,
                    }
                    all_examined_rows.append(detail_row)
                    next_states.append((new_names, new_hist_arr, new_cand_arr, rows_detail + [detail_row]))
            if not next_states:
                break
            # Håll bara de bästa grenarna. Prioritera lägst radantal, men straffa inte flera filter
            # så länge historikmålet är uppfyllt; målet är hög reducering med bibehållen träffbild.
            next_states = sorted(
                next_states,
                key=lambda stt: (int(stt[2].sum()), -int(stt[1].sum()), len(stt[0]))
            )[:beam_width]
            beam = next_states
            candidate_best = beam[0]
            if int(candidate_best[2].sum()) < int(best_state[2].sum()):
                best_state = candidate_best
            # Om vi är klart under målnivån och nästa steg inte förbättrar mycket stoppar sökningen senare via valid_options.

        # Maxfyllning: efter kombinationssökningen fortsätter vi lägga till alla filter
        # som fortfarande förbättrar paketet utan att bryta mallträffsmålet. Detta är den
        # viktiga skillnaden mot tidigare versioner där appen kunde stanna på t.ex. 3 spets.
        fill_names = tuple(best_state[0])
        fill_hist_arr = np.array(best_state[1], dtype=bool)
        fill_cand_arr = np.array(best_state[2], dtype=bool)
        fill_rows_detail = list(best_state[3])
        fill_safety = 0
        while len(fill_names) < int(max_spets_filters) and fill_safety < int(max_spets_filters) * 2:
            fill_safety += 1
            current_rows = int(fill_cand_arr.sum())
            if current_rows <= 0:
                break
            best_add = None
            for name in candidate_filter_names:
                if name in fill_names:
                    continue
                new_hist_arr = fill_hist_arr & hist_np[name]
                hist_hits = int(new_hist_arr.sum())
                if hist_hits < target_soft_hits:
                    continue
                new_cand_arr = fill_cand_arr & cand_np[name]
                keep_rows = int(new_cand_arr.sum())
                if keep_rows <= 0 or keep_rows >= current_rows:
                    continue
                red_pct = ((current_rows - keep_rows) / current_rows * 100.0) if current_rows else 0.0
                # En liten men verklig förbättring räcker. Huvudspärren är samlad mallträff.
                if red_pct < float(relaxed_spets_reduction_pct):
                    continue
                detail_row = {
                    "Filter": name,
                    "Intervall/regler": _spets_rule_text(name),
                    "Före rader": current_rows,
                    "Efter rader": keep_rows,
                    "Reducerar steg %": round(red_pct, 1),
                    "Kvar total %": round((keep_rows / total_candidates * 100), 1) if total_candidates else 0.0,
                    "Historisk träff": hist_hits,
                    "Historisk träff %": round((hist_hits / hist_total_for_soft) * 100, 1) if hist_total_for_soft else 0.0,
                    "Godkänd historik": True,
                    "Godkänd reduktion": True,
                }
                # Välj den kandidat som ger lägst radantal; historik är sekundärt så länge målet uppfylls.
                key = (keep_rows, -red_pct, -hist_hits)
                if best_add is None or key < best_add[0]:
                    best_add = (key, name, new_hist_arr, new_cand_arr, detail_row)
            if best_add is None:
                break
            _, add_name, fill_hist_arr, fill_cand_arr, detail_row = best_add
            fill_names = tuple(list(fill_names) + [add_name])
            fill_rows_detail.append(detail_row)
            all_examined_rows.append(detail_row)

        best_state = (fill_names, fill_hist_arr, fill_cand_arr, fill_rows_detail)

        selected_spets_names = list(best_state[0])
        selected_spets_rows = list(best_state[3])
        current_hist_mask = list(np.array(best_state[1], dtype=bool))
        current_cand_mask = list(np.array(best_state[2], dtype=bool))

        # Visa även några bra men ej valda kandidater i beslutsloggen.
        selected_set = set(selected_spets_names)
        for r in all_examined_rows:
            if r.get("Filter") not in selected_set:
                skipped_spets_rows.append(r)

        mall_hits = int(sum(current_hist_mask)) if hist_total_for_soft else 0
        soft_hit_pct = (mall_hits / len(v_m) * 100) if len(v_m) else 0.0
        combined_soft_survivors = int(sum(current_cand_mask)) if total_candidates else 0
        combined_soft_keep_pct = (combined_soft_survivors / total_candidates * 100) if total_candidates else 0.0
        combined_hard_survivors = hard_stage_survivors
        combined_hard_keep_pct = hard_stage_keep_pct
        hard_all_hits = int(sum(hard_only_hist_mask)) if hist_total_for_soft else 0
        hard_all_pct = (hard_all_hits / len(v_m) * 100) if len(v_m) else 0.0
        spets_status = "✅ Godkänd" if mall_hits >= target_soft_hits else "⚠️ Under mål"

        st.info(
            f"📈 **AUTOHARD EXAKT SPETSFILTER:** {spets_status}. "
            f"Valt basfilter **{selected_hard_req} av {hard_gate_total}** breda gates. "
            f"Valda spetsfilter: **{len(selected_spets_names)}**. "
            f"Historiskt klarar {mall_hits} av {len(v_m)} rader ({soft_hit_pct:.1f}%). "
            f"Mål: minst {target_soft_hits} av {len(v_m)} ({float(slider_combined_min_hist_pct):.0f}%)."
        )
        st.caption(
            "Soft som aktivt filtersteg används inte längre. Spetsfilter väljs bara om de klarar historikmålet "
            f"och Exakt filteroptimering väljer spetsfilter på den faktiska manuella grundramen och behåller bara filter som minskar exakt radmassa minst 5% utan att bryta mallträff/teckenskydd. Alla poäng-/värdefilter från VECKANS MALL testas som spetskandidater. "
            f"Estimat: efter basfilter {hard_stage_survivors}/{total_candidates} ({hard_stage_keep_pct:.1f}%), "
            f"efter bas+spets {combined_soft_survivors}/{total_candidates} ({combined_soft_keep_pct:.1f}%)."
        )

        soft_req_decision_rows = []
        if hard_gate_decision_rows:
            for r in sorted(hard_gate_decision_rows, key=lambda x: (-x["Godkänd mot mål"], x["Kvar rader"], -x["Hårdkrav"]))[:20]:
                soft_req_decision_rows.append({
                    "Typ": "Basfilter",
                    "Vald": "✅" if int(r["Hårdkrav"]) == int(selected_hard_req) else "",
                    "Krav/filter": f"{int(r['Hårdkrav'])} av {hard_gate_total}",
                    "Historisk träff": f"{int(r['Historiska träffar'])}/{hist_total_for_soft}",
                    "Historisk träff %": r["Historisk träff %"],
                    "Före rader": total_candidates,
                    "Efter rader": int(r["Kvar rader"]),
                    "Reducerar steg %": r["Reducerar %"],
                    "Godkänd": "Ja" if r["Godkänd mot mål"] else "Nej",
                })
        for r in selected_spets_rows:
            soft_req_decision_rows.append({
                "Typ": "Spetsfilter",
                "Vald": "✅",
                "Krav/filter": r["Filter"],
                "Intervall/regler": r.get("Intervall/regler", _spets_rule_text(r["Filter"])),
                "Historisk träff": f"{int(r['Historisk träff'])}/{hist_total_for_soft}",
                "Historisk träff %": r["Historisk träff %"],
                "Före rader": int(r["Före rader"]),
                "Efter rader": int(r["Efter rader"]),
                "Reducerar steg %": r["Reducerar steg %"],
                "Godkänd": "Ja",
            })
        for r in sorted(skipped_spets_rows, key=lambda x: (-x["Godkänd historik"], -x["Reducerar steg %"], -x["Historisk träff"]))[:30]:
            soft_req_decision_rows.append({
                "Typ": "Ej valt",
                "Vald": "",
                "Krav/filter": r["Filter"],
                "Intervall/regler": r.get("Intervall/regler", _spets_rule_text(r["Filter"])),
                "Historisk träff": f"{int(r['Historisk träff'])}/{hist_total_for_soft}",
                "Historisk träff %": r["Historisk träff %"],
                "Före rader": int(r["Före rader"]),
                "Efter rader": int(r["Efter rader"]),
                "Reducerar steg %": r["Reducerar steg %"],
                "Godkänd": "Nej",
            })

        def pct_count(predicate, n_items):
            return (sum(1 for i in range(n_items) if predicate(i)) / n_items) * 100 if n_items else 0.0

        def keep_pct_rows(predicate):
            return (sum(1 for tr in candidate_rows if predicate(tr)) / total_candidates) * 100 if total_candidates else 0.0

        rule_rows = []
        if cb_payout:
            rule_rows.append(filter_strength_row(
                "Utdelning", f"{v_m['Payout'].min():.0f}-{v_m['Payout'].max():.0f} kr", 100.0, None,
                "Föranalys", "Utdelning används före historisk matchning, inte som Helgardering-filter", "Värde"
            ))
        if cb_aimatrix:
            ai_keep = ((active_ai_max - active_ai_min + 1) / max_rank) * 100
            hist_pct = pct_count(lambda i: active_ai_min <= ai_ranks[i] <= active_ai_max, n_rows)
            rule_rows.append(filter_strength_row("AI-Matrix Rank", fmt_interval((active_ai_min, active_ai_max)), hist_pct, ai_keep, "AI-Matrix", f"AI-Matrix Rank: {active_ai_min:.0f}-{active_ai_max:.0f}", "Värde"))
        if cb_totaldiff:
            hist_pct = pct_count(lambda i: in_range(total_diff_vals[i], c_totaldiff), n_rows)
            keep_pct = keep_pct_rows(lambda tr: in_range(calculate_total_diff(match_odds_filter, list(tr)), c_totaldiff))
            rule_rows.append(filter_strength_row("Total Diff", fmt_interval(c_totaldiff), hist_pct, keep_pct, "Poängdifferens", f"Total Diff: {fmt_interval(c_totaldiff)}", "Värde"))
        if cb_rank24:
            hist_pct = pct_count(lambda i: in_range(rank24_sums[i], c_rank24), n_rows)
            keep_pct = keep_pct_rows(lambda tr: in_range(get_rank_sum(tr, filter_vec), c_rank24))
            rule_rows.append(filter_strength_row("Rank Summa", fmt_interval(c_rank24, 1), hist_pct, keep_pct, "Poängsumma", f"Rank Summa: {fmt_interval(c_rank24, 1)}", "Värde"))
        if cb_100minus:
            hist_pct = pct_count(lambda i: in_range(minus_sums[i], c_minus), n_rows)
            keep_pct = keep_pct_rows(lambda tr: in_range(get_100_minus_sum(tr, filter_vec), c_minus))
            rule_rows.append(filter_strength_row("100-minus Summa", fmt_interval(c_minus, 1), hist_pct, keep_pct, "Poängsumma", f"100-minus Summa: {fmt_interval(c_minus, 1)}", "Värde"))
        if cb_log_surprise:
            hist_pct = pct_count(lambda i: in_range(log_surprise_sums[i], c_log_surprise), n_rows)
            keep_pct = keep_pct_rows(lambda tr: in_range(get_log_surprise_sum(tr, filter_vec), c_log_surprise))
            rule_rows.append(filter_strength_row("Skrälltryck Log Summa", fmt_interval(c_log_surprise), hist_pct, keep_pct, "Poängsumma", f"Skrälltryck Log Summa: {fmt_interval(c_log_surprise)}", "Värde"))
        if cb_sft:
            hist_pct = pct_count(lambda i: in_range(sft_sums[i], c_sft), n_rows)
            keep_pct = keep_pct_rows(lambda tr: in_range(get_sft_sum(tr, filter_vec), c_sft))
            rule_rows.append(filter_strength_row("SFT Summa", fmt_interval(c_sft, 1), hist_pct, keep_pct, "SFT/Poängsumma", f"SFT Summa: {fmt_interval(c_sft, 1)}", "Värde"))
        if cb_points:
            hist_pct = pct_count(lambda i: in_range(points_vals[i], c_points), n_rows)
            keep_pct = keep_pct_rows(lambda tr: in_range(get_rank_points(tr, filter_vec), c_points))
            rule_rows.append(filter_strength_row("Poängfilter", fmt_interval(c_points), hist_pct, keep_pct, "Poängsumma", f"Poängfilter: {fmt_interval(c_points)}", "Värde"))
        if cb_fat:
            hist_pct = pct_count(lambda i: in_range(fat_f[i], c_fatf) and in_range(fat_a[i], c_fata) and in_range(fat_t[i], c_fatt) and in_range(fat_sums[i], c_fatsum), n_rows)
            def _fat_ok(tr):
                f0, a0, t0, fs0 = get_fat(tr, filter_vec)
                return in_range(f0, c_fatf) and in_range(a0, c_fata) and in_range(t0, c_fatt) and in_range(fs0, c_fatsum)
            keep_pct = keep_pct_rows(_fat_ok)
            rule_rows.append(filter_strength_row("FAT", f"F {fmt_interval(c_fatf)} | A {fmt_interval(c_fata)} | T {fmt_interval(c_fatt)} | Summa {fmt_interval(c_fatsum)}", hist_pct, keep_pct, "FAT", f"FAT: F {fmt_interval(c_fatf)}, A {fmt_interval(c_fata)}, T {fmt_interval(c_fatt)}, Summa {fmt_interval(c_fatsum)}", "Värde"))
        if cb_fat_sequences and fat_sequence_active_parts > 0:
            if fat_seq2_list:
                hist_pct = pct_count(lambda i: i < len(fat_seq2_hits) and in_range(fat_seq2_hits[i], c_fat_seq2), n_rows)
                keep_pct = keep_pct_rows(lambda tr: in_range(count_fat_sequence_hits(get_fat_string(tr, filter_vec), fat_seq2_list), c_fat_seq2))
                rule_rows.append(filter_strength_row("FAT 2-sekvenser", fmt_interval(c_fat_seq2), hist_pct, keep_pct, "Sekvenser", f"FAT 2-sekvenser: {', '.join(fat_seq2_list)} | krav {fmt_interval(c_fat_seq2)} av 5", "FAT-sekvens"))
            if fat_seq3_list:
                hist_pct = pct_count(lambda i: i < len(fat_seq3_hits) and in_range(fat_seq3_hits[i], c_fat_seq3), n_rows)
                keep_pct = keep_pct_rows(lambda tr: in_range(count_fat_sequence_hits(get_fat_string(tr, filter_vec), fat_seq3_list), c_fat_seq3))
                rule_rows.append(filter_strength_row("FAT 3-sekvenser", fmt_interval(c_fat_seq3), hist_pct, keep_pct, "Sekvenser", f"FAT 3-sekvenser: {', '.join(fat_seq3_list)} | krav {fmt_interval(c_fat_seq3)} av 5", "FAT-sekvens"))
            if fat_pair_list:
                pair_text = ', '.join([f"{a}/{b}" for a, b in fat_pair_list])
                hist_pct = pct_count(lambda i: i < len(fat_pair_hits) and in_range(fat_pair_hits[i], c_fat_pairs), n_rows)
                keep_pct = keep_pct_rows(lambda tr: in_range(count_fat_pair_hits(get_fat_string(tr, filter_vec), fat_pair_list), c_fat_pairs))
                rule_rows.append(filter_strength_row("FAT dubbelchans", fmt_interval(c_fat_pairs), hist_pct, keep_pct, "Sekvenser", f"FAT dubbelchans: {pair_text} | krav {fmt_interval(c_fat_pairs)} av 5", "FAT-sekvens"))
            hist_pct = pct_count(lambda i: i < len(fat_strings_hist) and fat_sequence_group_ok_from_fat_string(fat_strings_hist[i]), n_rows)
            keep_pct = keep_pct_rows(lambda tr: fat_sequence_group_ok_row(tr))
            rule_rows.append(filter_strength_row("FAT-sekvensgrupp", f"Minst {fat_sequence_group_req} av {fat_sequence_active_parts}", hist_pct, keep_pct, "Gruppmodul/Sekvenser", f"FAT-sekvensgrupp: minst {fat_sequence_group_req} av {fat_sequence_active_parts} delgrupper", "FAT-sekvens"))
        if cb_u_favs:
            hist_pct = pct_count(lambda i: in_range(u_wins[i], c_u), n_rows)
            keep_pct = keep_pct_rows(lambda tr: in_range(get_top_n_favs_wins(tr, filter_vec, slider_u_count), c_u))
            rule_rows.append(filter_strength_row(f"Topp {slider_u_count} favoriter", f"{fmt_interval(c_u)} st vinner", hist_pct, keep_pct, "Utgång/U-tecken", f"Topp {slider_u_count} favoriter: {fmt_interval(c_u)} st", "Värde"))
        if cb_fav_pressure:
            hist_pct = pct_count(lambda i: in_range(fav70_wins[i], c_fav70) and in_range(fav60_wins[i], c_fav60) and in_range(fav50_wins[i], c_fav50), n_rows)
            def _fav_pressure_ok(tr):
                fp0 = get_favorite_pressure(tr, filter_vec)
                return in_range(fp0['F70_Wins'], c_fav70) and in_range(fp0['F60_Wins'], c_fav60) and in_range(fp0['F50_Wins'], c_fav50)
            keep_pct = keep_pct_rows(_fav_pressure_ok)
            rule_rows.append(filter_strength_row(
                "Favorittryck", f"≥70 {fmt_interval(c_fav70)} | ≥60 {fmt_interval(c_fav60)} | ≥50 {fmt_interval(c_fav50)}",
                hist_pct, keep_pct, "Tecken/FAT", f"Favorittryck: ≥70% {fmt_interval(c_fav70)} av {todays_fav_counts.get(70,0)}, ≥60% {fmt_interval(c_fav60)} av {todays_fav_counts.get(60,0)}, ≥50% {fmt_interval(c_fav50)} av {todays_fav_counts.get(50,0)}", "Värde"
            ))
        if cb_shock_strength:
            hist_pct = pct_count(lambda i: in_range(shock_u10[i], c_shock10) and in_range(shock_u15[i], c_shock15) and in_range(shock_u20[i], c_shock20) and in_range(shock_lowest[i], c_shock_lowest), n_rows)
            def _shock_ok(tr):
                sh0 = get_shock_strength(tr, filter_vec)
                return in_range(sh0['U10_Wins'], c_shock10) and in_range(sh0['U15_Wins'], c_shock15) and in_range(sh0['U20_Wins'], c_shock20) and in_range(sh0['Lowest_Win_Pct'], c_shock_lowest)
            keep_pct = keep_pct_rows(_shock_ok)
            rule_rows.append(filter_strength_row(
                "Skrällstyrka", f"<10 {fmt_interval(c_shock10)} | <15 {fmt_interval(c_shock15)} | <20 {fmt_interval(c_shock20)} | Lägsta {fmt_interval(c_shock_lowest,1)}%",
                hist_pct, keep_pct, "Poängsumma/Tecken", f"Skrällstyrka: <10% {fmt_interval(c_shock10)}, <15% {fmt_interval(c_shock15)}, <20% {fmt_interval(c_shock20)}, lägsta vinnande % {fmt_interval(c_shock_lowest,1)}", "Värde"
            ))
        if cb_fav_delta:
            hist_pct = pct_count(lambda i: in_range(fav_delta_vals[i], c_fav_delta), n_rows)
            keep_pct = keep_pct_rows(lambda tr: in_range(get_favorite_delta(tr, filter_vec), c_fav_delta))
            rule_rows.append(filter_strength_row("Favorit-delta", fmt_interval(c_fav_delta, 2), hist_pct, keep_pct, "Poängdifferens", f"Favorit-delta: {fmt_interval(c_fav_delta, 2)}", "Värde"))
        if cb_base:
            hist_pct = pct_count(lambda i: in_range(ones[i], c_ones) and in_range(draws[i], c_draws) and in_range(twos[i], c_twos), n_rows)
            keep_pct = keep_pct_rows(lambda tr: in_range(tr.count('1'), c_ones) and in_range(tr.count('X'), c_draws) and in_range(tr.count('2'), c_twos))
            rule_rows.append(filter_strength_row("Tecken 1X2", f"1 {fmt_interval(c_ones)} | X {fmt_interval(c_draws)} | 2 {fmt_interval(c_twos)}", hist_pct, keep_pct, "Tecken", f"Tecken: 1 {fmt_interval(c_ones)}, X {fmt_interval(c_draws)}, 2 {fmt_interval(c_twos)}", "Struktur"))
        if cb_streak:
            hist_pct = pct_count(lambda i: in_range(s1[i], c_s1) and in_range(sx[i], c_sx) and in_range(s2[i], c_s2), n_rows)
            def _streak_ok(tr):
                x = get_streaks(tr)
                return in_range(x[0], c_s1) and in_range(x[1], c_sx) and in_range(x[2], c_s2)
            keep_pct = keep_pct_rows(_streak_ok)
            rule_rows.append(filter_strength_row("Teckenföljd/Sviter", f"1 {fmt_interval(c_s1)} | X {fmt_interval(c_sx)} | 2 {fmt_interval(c_s2)}", hist_pct, keep_pct, "Teckenföljd", f"Teckenföljd: 1 {fmt_interval(c_s1)}, X {fmt_interval(c_sx)}, 2 {fmt_interval(c_s2)}", "Struktur"))
        if cb_gap:
            hist_pct = pct_count(lambda i: in_range(g1[i], c_g1) and in_range(gx[i], c_gx) and in_range(g2[i], c_g2), n_rows)
            def _gap_ok(tr):
                x = get_gaps(tr)
                return in_range(x[0], c_g1) and in_range(x[1], c_gx) and in_range(x[2], c_g2)
            keep_pct = keep_pct_rows(_gap_ok)
            rule_rows.append(filter_strength_row("Teckenlucka", f"1 {fmt_interval(c_g1)} | X {fmt_interval(c_gx)} | 2 {fmt_interval(c_g2)}", hist_pct, keep_pct, "Teckenlucka", f"Teckenlucka: 1 {fmt_interval(c_g1)}, X {fmt_interval(c_gx)}, 2 {fmt_interval(c_g2)}", "Struktur"))
        if cb_single:
            hist_pct = pct_count(lambda i: in_range(sing1[i], c_sing1) and in_range(singx[i], c_singx) and in_range(sing2[i], c_sing2) and in_range(sing_tot[i], c_singtot), n_rows)
            def _single_ok(tr):
                x = get_singles(tr)
                return in_range(x[0], c_sing1) and in_range(x[1], c_singx) and in_range(x[2], c_sing2) and in_range(x[3], c_singtot)
            keep_pct = keep_pct_rows(_single_ok)
            rule_rows.append(filter_strength_row("Singlar", f"1 {fmt_interval(c_sing1)} | X {fmt_interval(c_singx)} | 2 {fmt_interval(c_sing2)} | Tot {fmt_interval(c_singtot)}", hist_pct, keep_pct, "Singlar", f"Singlar: 1 {fmt_interval(c_sing1)}, X {fmt_interval(c_singx)}, 2 {fmt_interval(c_sing2)}, Tot {fmt_interval(c_singtot)}", "Struktur"))
        if cb_doublet:
            hist_pct = pct_count(lambda i: in_range(dub1[i], c_dub1) and in_range(dubx[i], c_dubx) and in_range(dub2[i], c_dub2) and in_range(dub_tot[i], c_dubtot), n_rows)
            def _doublet_ok(tr):
                x = get_doublets(tr)
                return in_range(x[0], c_dub1) and in_range(x[1], c_dubx) and in_range(x[2], c_dub2) and in_range(x[3], c_dubtot)
            keep_pct = keep_pct_rows(_doublet_ok)
            rule_rows.append(filter_strength_row("Dubbletter", f"1 {fmt_interval(c_dub1)} | X {fmt_interval(c_dubx)} | 2 {fmt_interval(c_dub2)} | Tot {fmt_interval(c_dubtot)}", hist_pct, keep_pct, "Dubbletter", f"Dubbletter: 1 {fmt_interval(c_dub1)}, X {fmt_interval(c_dubx)}, 2 {fmt_interval(c_dub2)}, Tot {fmt_interval(c_dubtot)}", "Struktur"))
        if cb_triplet:
            hist_pct = pct_count(lambda i: in_range(trip1[i], c_trip1) and in_range(tripx[i], c_tripx) and in_range(trip2[i], c_trip2) and in_range(trip_tot[i], c_triptot), n_rows)
            def _triplet_ok(tr):
                x = get_triplets(tr)
                return in_range(x[0], c_trip1) and in_range(x[1], c_tripx) and in_range(x[2], c_trip2) and in_range(x[3], c_triptot)
            keep_pct = keep_pct_rows(_triplet_ok)
            rule_rows.append(filter_strength_row("Tripplar", f"1 {fmt_interval(c_trip1)} | X {fmt_interval(c_tripx)} | 2 {fmt_interval(c_trip2)} | Tot {fmt_interval(c_triptot)}", hist_pct, keep_pct, "Tripplar", f"Tripplar: 1 {fmt_interval(c_trip1)}, X {fmt_interval(c_tripx)}, 2 {fmt_interval(c_trip2)}, Tot {fmt_interval(c_triptot)}", "Struktur"))
        if cb_occur:
            hist_pct = pct_count(lambda i: in_range(occ1[i], c_occ1) and in_range(occx[i], c_occx) and in_range(occ2[i], c_occ2) and in_range(occ_tot[i], c_occtot), n_rows)
            def _occur_ok(tr):
                x = get_occurrences(tr)
                return in_range(x[0], c_occ1) and in_range(x[1], c_occx) and in_range(x[2], c_occ2) and in_range(x[3], c_occtot)
            keep_pct = keep_pct_rows(_occur_ok)
            rule_rows.append(filter_strength_row("Uppkomster", f"1 {fmt_interval(c_occ1)} | X {fmt_interval(c_occx)} | 2 {fmt_interval(c_occ2)} | Tot {fmt_interval(c_occtot)}", hist_pct, keep_pct, "Uppkomster", f"Uppkomster: 1 {fmt_interval(c_occ1)}, X {fmt_interval(c_occx)}, 2 {fmt_interval(c_occ2)}, Tot {fmt_interval(c_occtot)}", "Struktur"))
        if cb_super_macro:
            hist_pct = pct_count(lambda i: pass_super_macro_row(v_m.iloc[i]['Correct_Row'], v_m.iloc[i]['Prob_Vector'], sm_bounds, slider_super_groups), n_rows)
            keep_pct = keep_pct_rows(lambda tr: pass_super_macro_row(tr, filter_vec, sm_bounds, slider_super_groups))
            rule_rows.append(filter_strength_row("Super-Makro", f"Minst {slider_super_groups} av 8 grupper", hist_pct, keep_pct, "Gruppmodul", f"Super-Makro: minst {slider_super_groups} av 8 grupper, minst 2 av 3 interna villkor", "Super-Makro"))

        filter_rules_df = pd.DataFrame(rule_rows)

        # --- PRO-GRUPPER V9: tightare intervall + mjukare gruppkrav ---
        pro_group_rows = []
        pro_group_detail_tables = {}

        def add_optimized_group(group_name, specs):
            if not specs:
                return
            best_g, detail_g = optimize_pro_group(
                group_name,
                specs,
                candidate_rows,
                target_hist_pct=slider_group_target,
                coverage_steps=[50, 55, 60, 65, 70, 75, 80, 85, 90]
            )
            if best_g:
                pro_group_rows.append(best_g)
                pro_group_detail_tables[group_name] = detail_g

        if cb_pro_groups:
            # Riskgrupp: värde-/riskmått. Intervallen görs tightare internt och gruppkravet mjukare.
            if cb_group_risk:
                specs = []
                if cb_sft:
                    specs.append(make_group_spec(
                        "SFT Summa", sft_sums,
                        lambda tr: get_sft_sum(tr, filter_vec),
                        formatter=lambda iv: f"SFT Summa: {fmt_interval(iv, 1)}",
                        decimals=1
                    ))
                if cb_log_surprise:
                    specs.append(make_group_spec(
                        "Skrälltryck Log", log_surprise_sums,
                        lambda tr: get_log_surprise_sum(tr, filter_vec),
                        formatter=lambda iv: f"Skrälltryck Log: {fmt_interval(iv)}"
                    ))
                if cb_100minus:
                    specs.append(make_group_spec(
                        "100-minus", minus_sums,
                        lambda tr: get_100_minus_sum(tr, filter_vec),
                        formatter=lambda iv: f"100-minus: {fmt_interval(iv, 1)}",
                        decimals=1
                    ))
                if cb_rank24:
                    specs.append(make_group_spec(
                        "Rank Summa", rank24_sums,
                        lambda tr: get_rank_sum(tr, filter_vec),
                        formatter=lambda iv: f"Rank Summa: {fmt_interval(iv, 1)}",
                        decimals=1
                    ))
                if cb_totaldiff:
                    specs.append(make_group_spec(
                        "Total Diff", total_diff_vals,
                        lambda tr: calculate_total_diff(match_odds_filter, list(tr)),
                        formatter=lambda iv: f"Total Diff: {fmt_interval(iv)}"
                    ))
                if cb_aimatrix and cand_ai_matrix is not None:
                    specs.append(make_group_spec(
                        "AI-Rank", ai_ranks,
                        lambda tr: get_exact_rank(tr, cand_ai_matrix, cand_ai_scores_asc, cand_ai_tot)[0],
                        formatter=lambda iv: f"AI-Rank: {fmt_interval(iv)}"
                    ))
                add_optimized_group("Riskgrupp", specs)

            # Skrällgrupp: lågprocentare och skrällnivå.
            if cb_group_shock:
                specs = []
                if cb_shock_strength:
                    specs.extend([
                        make_group_spec("<10%", shock_u10, lambda tr: get_shock_strength(tr, filter_vec)['U10_Wins'], formatter=lambda iv: f"Vinnande tecken <10%: {fmt_interval(iv)}"),
                        make_group_spec("<15%", shock_u15, lambda tr: get_shock_strength(tr, filter_vec)['U15_Wins'], formatter=lambda iv: f"Vinnande tecken <15%: {fmt_interval(iv)}"),
                        make_group_spec("<20%", shock_u20, lambda tr: get_shock_strength(tr, filter_vec)['U20_Wins'], formatter=lambda iv: f"Vinnande tecken <20%: {fmt_interval(iv)}"),
                        make_group_spec("Lägsta vinnande %", shock_lowest, lambda tr: get_shock_strength(tr, filter_vec)['Lowest_Win_Pct'], formatter=lambda iv: f"Lägsta vinnande procent: {fmt_interval(iv, 1)}", decimals=1),
                    ])
                if cb_log_surprise:
                    specs.append(make_group_spec("Skrälltryck Log", log_surprise_sums, lambda tr: get_log_surprise_sum(tr, filter_vec), formatter=lambda iv: f"Skrälltryck Log: {fmt_interval(iv)}"))
                if cb_100minus:
                    specs.append(make_group_spec("100-minus", minus_sums, lambda tr: get_100_minus_sum(tr, filter_vec), formatter=lambda iv: f"100-minus: {fmt_interval(iv, 1)}", decimals=1))
                add_optimized_group("Skrällgrupp", specs)

            # FAT-profilgrupp: favorit/andrahands/skräll-profilen som mjuk grupp.
            if cb_group_fat:
                specs = []
                if cb_fat:
                    specs.extend([
                        make_group_spec("FAT F", fat_f, lambda tr: get_fat(tr, filter_vec)[0], formatter=lambda iv: f"FAT F: {fmt_interval(iv)}"),
                        make_group_spec("FAT A", fat_a, lambda tr: get_fat(tr, filter_vec)[1], formatter=lambda iv: f"FAT A: {fmt_interval(iv)}"),
                        make_group_spec("FAT T", fat_t, lambda tr: get_fat(tr, filter_vec)[2], formatter=lambda iv: f"FAT T: {fmt_interval(iv)}"),
                        make_group_spec("FAT Summa", fat_sums, lambda tr: get_fat(tr, filter_vec)[3], formatter=lambda iv: f"FAT Summa: {fmt_interval(iv)}"),
                    ])
                if cb_u_favs:
                    specs.append(make_group_spec(
                        f"Topp {slider_u_count} favoriter", u_wins,
                        lambda tr: get_top_n_favs_wins(tr, filter_vec, slider_u_count),
                        formatter=lambda iv: f"Topp {slider_u_count} favoriter: {fmt_interval(iv)}"
                    ))
                if cb_fav_pressure:
                    fav_pressure_hist = list(zip(fav70_wins, fav60_wins, fav50_wins))
                    def _fmt_fav_pressure(iv):
                        parts = []
                        if todays_fav_counts.get(70, 0) > 0:
                            parts.append(f"≥70% {fmt_interval(iv[0])} av {todays_fav_counts.get(70,0)}")
                        if todays_fav_counts.get(60, 0) > 0:
                            parts.append(f"≥60% {fmt_interval(iv[1])} av {todays_fav_counts.get(60,0)}")
                        if todays_fav_counts.get(50, 0) > 0:
                            parts.append(f"≥50% {fmt_interval(iv[2])} av {todays_fav_counts.get(50,0)}")
                        return "Favorittryck: " + (" | ".join(parts) if parts else "ej relevant")
                    specs.append(make_group_spec(
                        "Favorittryck", fav_pressure_hist,
                        lambda tr: (lambda fp: (fp['F70_Wins'], fp['F60_Wins'], fp['F50_Wins']))(get_favorite_pressure(tr, filter_vec)),
                        formatter=_fmt_fav_pressure
                    ))
                add_optimized_group("FAT-profilgrupp", specs)

            # Strukturgrupp: klassiska grundramsfilter med tightare interna intervall.
            if cb_group_structure:
                specs = []
                if cb_base:
                    specs.append(make_group_spec(
                        "Tecken 1X2", list(zip(ones, draws, twos)),
                        lambda tr: (tr.count('1'), tr.count('X'), tr.count('2')),
                        formatter=lambda iv: f"Tecken 1X2: 1 {fmt_interval(iv[0])}, X {fmt_interval(iv[1])}, 2 {fmt_interval(iv[2])}"
                    ))
                if cb_streak:
                    specs.append(make_group_spec(
                        "Sviter", list(zip(s1, sx, s2)),
                        lambda tr: get_streaks(tr)[:3],
                        formatter=lambda iv: f"Sviter: 1 {fmt_interval(iv[0])}, X {fmt_interval(iv[1])}, 2 {fmt_interval(iv[2])}"
                    ))
                if cb_gap:
                    specs.append(make_group_spec(
                        "Luckor", list(zip(g1, gx, g2)),
                        lambda tr: get_gaps(tr)[:3],
                        formatter=lambda iv: f"Luckor: 1 {fmt_interval(iv[0])}, X {fmt_interval(iv[1])}, 2 {fmt_interval(iv[2])}"
                    ))
                if cb_single:
                    specs.append(make_group_spec(
                        "Singlar", list(zip(sing1, singx, sing2, sing_tot)),
                        lambda tr: get_singles(tr)[:4],
                        formatter=lambda iv: f"Singlar: 1 {fmt_interval(iv[0])}, X {fmt_interval(iv[1])}, 2 {fmt_interval(iv[2])}, Tot {fmt_interval(iv[3])}"
                    ))
                if cb_doublet:
                    specs.append(make_group_spec(
                        "Dubbletter", list(zip(dub1, dubx, dub2, dub_tot)),
                        lambda tr: get_doublets(tr)[:4],
                        formatter=lambda iv: f"Dubbletter: 1 {fmt_interval(iv[0])}, X {fmt_interval(iv[1])}, 2 {fmt_interval(iv[2])}, Tot {fmt_interval(iv[3])}"
                    ))
                if cb_triplet:
                    specs.append(make_group_spec(
                        "Tripplar", list(zip(trip1, tripx, trip2, trip_tot)),
                        lambda tr: get_triplets(tr)[:4],
                        formatter=lambda iv: f"Tripplar: 1 {fmt_interval(iv[0])}, X {fmt_interval(iv[1])}, 2 {fmt_interval(iv[2])}, Tot {fmt_interval(iv[3])}"
                    ))
                if cb_occur:
                    specs.append(make_group_spec(
                        "Uppkomster", list(zip(occ1, occx, occ2, occ_tot)),
                        lambda tr: get_occurrences(tr)[:4],
                        formatter=lambda iv: f"Uppkomster: 1 {fmt_interval(iv[0])}, X {fmt_interval(iv[1])}, 2 {fmt_interval(iv[2])}, Tot {fmt_interval(iv[3])}"
                    ))
                add_optimized_group("Strukturgrupp", specs)

        pro_groups_df = pd.DataFrame(pro_group_rows)

        st.markdown("---")
        st.subheader("🧪 Filterdiagnos")
        st.caption("Här visas bara värderingen av filtren. Själva inmatningsmallen ligger ovanför i VECKANS MALL.")

        if cb_pro_groups and not pro_groups_df.empty:
            st.markdown("### 🧩 Pro-grupper")
            st.caption(
                "V9 testar tightare interna intervall och flera gruppkrav. "
                "Målet är hög reducering ihop med hög historisk träffsäkerhet, utan att grupperna blir 6 av 6."
            )
            show_group_cols = [
                "Grupp", "Krav", "Tighthet %", "Historisk träff %", "Kvar rad %", "Reducerar %",
                "Rationell faktor", "Klass", "Filterintervall"
            ]
            show_group_cols = [c for c in show_group_cols if c in pro_groups_df.columns]
            st.dataframe(pro_groups_df[show_group_cols], use_container_width=True, hide_index=True)

            with st.expander("Visa pro-grupper med exakta filterintervall"):
                for _, gr in pro_groups_df.iterrows():
                    st.markdown(f"**{gr['Grupp']} – krav {gr['Krav']}**")
                    st.text(gr.get('Helgardering-detaljer', ''))

            with st.expander("Visa testade gruppkrav"):
                for gname, gdf in pro_group_detail_tables.items():
                    st.markdown(f"**{gname}**")
                    cols = ["Krav", "Tighthet %", "Historisk träff %", "Kvar rad %", "Reducerar %", "Rationell faktor", "Klass"]
                    cols = [c for c in cols if c in gdf.columns]
                    st.dataframe(gdf[cols], use_container_width=True, hide_index=True)

        if not filter_rules_df.empty:
            diag_df = filter_rules_df.copy()
            if 'Rationell faktor' in diag_df.columns:
                diag_df = diag_df.sort_values('Rationell faktor', ascending=False, na_position='last')

            numeric_lift = pd.to_numeric(diag_df['Rationell faktor'], errors='coerce')
            numeric_keep = pd.to_numeric(diag_df['Kvar rad %'], errors='coerce')
            numeric_red = pd.to_numeric(diag_df['Reducerar %'], errors='coerce')

            if cb_filter_revision:
                st.markdown("### 🧪 Filterrevision")
                st.caption(
                    "Målet här är att rensa: hitta vilka filter som ska användas, vilka som bara ska vara diagnos, "
                    "och vilka som dubblerar varandra. Rangordningen bygger på historisk träff jämfört med kvarvarande radmassa."
                )
                revision_df = build_filter_revision_df(diag_df)
                family_df = build_family_summary(revision_df)
                starter_df = build_starter_package(revision_df, max_filters=6)

                if not revision_df.empty:
                    rev_cols = [
                        "Beslut", "Filterfamilj", "Filter", "Klass", "Historisk träff %",
                        "Kvar rad %", "Reducerar %", "Rationell faktor", "Dubblett-/risknotis"
                    ]
                    rev_cols = [c for c in rev_cols if c in revision_df.columns]

                    cfr1, cfr2, cfr3, cfr4 = st.columns(4)
                    with cfr1:
                        st.metric("Använd", int((revision_df["Beslut"] == "Använd").sum()))
                    with cfr2:
                        st.metric("Reserv", int((revision_df["Beslut"] == "Reserv").sum()))
                    with cfr3:
                        st.metric("Diagnos", int((revision_df["Beslut"] == "Diagnos").sum()))
                    with cfr4:
                        st.metric("Pausa", int((revision_df["Beslut"] == "Pausa").sum()))

                    st.markdown("**Filter att prioritera / rensa**")
                    decision_order = ["Använd", "Reserv", "Diagnos", "Pausa", "Info"]
                    for decision in decision_order:
                        sub = revision_df[revision_df["Beslut"] == decision]
                        if len(sub) == 0:
                            continue
                        with st.expander(f"{decision} ({len(sub)} filter)", expanded=(decision in ["Använd", "Pausa"])):
                            st.dataframe(sub[rev_cols], use_container_width=True, hide_index=True)

                    if not family_df.empty:
                        st.markdown("**Filterfamiljer – vad ska vi göra med varje grupp?**")
                        st.dataframe(family_df, use_container_width=True, hide_index=True)

                    if not starter_df.empty:
                        st.markdown("**Föreslaget startpaket denna körning**")
                        st.caption(
                            "Detta är inte en slutlig systemmall. Det är en rensad startlista: få filter, hög RF, mindre dubblering. "
                            "Pro-grupperna bör fortfarande vara huvudvägen när de är starkare än enskilda filter."
                        )
                        pkg_cols = ["Filterfamilj", "Filter", "Klass", "Historisk träff %", "Kvar rad %", "Rationell faktor", "Helgardering-rad"]
                        pkg_cols = [c for c in pkg_cols if c in starter_df.columns]
                        st.dataframe(starter_df[pkg_cols], use_container_width=True, hide_index=True)

                    tm_download_button(
                        "⬇️ Ladda ner filterrevision CSV",
                        revision_df.to_csv(index=False, sep=';').encode('utf-8-sig'),
                        file_name=f"filterrevision_{spelform.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

            st.markdown("**Sammanlagd mallträff**")
            csum1, csum2, csum3, csum4, csum5 = st.columns(5)
            with csum1:
                st.metric("Mål", f"{target_soft_hits}/{len(v_m)}", f"{float(slider_combined_min_hist_pct):.0f}%")
            with csum2:
                st.metric("Historisk totalträff", f"{soft_hit_pct:.1f}%", f"{mall_hits}/{len(v_m)}")
            with csum3:
                st.metric("Historisk hård träff", f"{hard_all_pct:.1f}%", f"{hard_all_hits}/{len(v_m)}")
            with csum4:
                st.metric(f"Kvar efter bas+spets ({combined_est_label})", f"{combined_soft_keep_pct:.1f}%", f"{combined_soft_survivors}/{total_candidates}")
            with csum5:
                st.metric(f"Kvar rader hårt ({combined_est_label})", f"{combined_hard_keep_pct:.1f}%", f"{combined_hard_survivors}/{total_candidates}")

            if mall_hits >= target_soft_hits:
                st.success(
                    f"AutoHard Spetsfilter valde **{selected_hard_req} av {hard_gate_total}** hårda bas-gates och **{len(selected_spets_names)}** spetsfilter eftersom paketet klarar minst "
                    f"{target_soft_hits} av {len(v_m)} historiska omgångar."
                )
            else:
                st.warning(
                    f"Inget bas+spets-paket klarade målet {target_soft_hits}/{len(v_m)} fullt ut. Appen använder basfilter och du bör bredda intervall/filter."
                )

            st.caption(
                "Individuella filter kan vara starka var för sig men tillsammans bli för hårda. "
                "AutoHard Spetsfilter använder breda bas-gates och lägger bara till hårda spetsfilter som både klarar historikmålet och ger tydlig extra reducering."
            )

            base_gate_rows = _base_gate_rule_rows()
            if base_gate_rows:
                with st.expander("Visa hårda basfilter och breda intervaller", expanded=False):
                    st.caption(f"Basfiltret är ett N-av-M-krav: vald nivå **{selected_hard_req} av {hard_gate_total}**. En rad behöver alltså inte klara varje bas-gate, utan minst vald nivå.")
                    st.dataframe(pd.DataFrame(base_gate_rows), use_container_width=True, hide_index=True)

            if selected_spets_rows:
                st.markdown("**Valda hårda spetsfilter och intervaller**")
                selected_view_rows = []
                for idx, r in enumerate(selected_spets_rows, start=1):
                    selected_view_rows.append({
                        "#": idx,
                        "Filter": r["Filter"],
                        "Intervall/regler": r.get("Intervall/regler", _spets_rule_text(r["Filter"])),
                        "Historisk träff": f"{int(r['Historisk träff'])}/{hist_total_for_soft}",
                        "Före": int(r["Före rader"]),
                        "Efter": int(r["Efter rader"]),
                        "Reducerar": f"{float(r['Reducerar steg %']):.1f}%",
                    })
                st.dataframe(pd.DataFrame(selected_view_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Inga spetsfilter valdes. Basfiltret används ensamt eftersom inga spetsfilter klarade både historikmålet och minsta reducering.")

            if soft_req_decision_rows:
                with st.expander("Visa hur AutoHard Spetsfilter valde filter", expanded=False):
                    st.dataframe(pd.DataFrame(soft_req_decision_rows), use_container_width=True, hide_index=True)

            cdiag1, cdiag2, cdiag3, cdiag4 = st.columns(4)
            with cdiag1:
                st.metric("Aktiva filter", len(diag_df))
            with cdiag2:
                st.metric("Snitt reducering/filter", f"{numeric_red.mean():.1f}%" if numeric_red.notna().any() else "-")
            with cdiag3:
                st.metric("Snitt rationell faktor", f"{numeric_lift.mean():+.1f}" if numeric_lift.notna().any() else "-")
            with cdiag4:
                strong_count = int(diag_df['Klass'].isin(['Mycket stark', 'Stark']).sum())
                st.metric("Starka filter", strong_count)

            ctop, cweak = st.columns(2)
            with ctop:
                st.markdown("**Bäst filter just nu**")
                top_cols = ['Filter', 'Klass', 'Rationell faktor', 'Kvar rad %', 'Reducerar %']
                st.dataframe(diag_df[top_cols].head(6), use_container_width=True, hide_index=True)
            with cweak:
                st.markdown("**Filter att granska**")
                weak = diag_df[diag_df['Klass'].isin(['Svag', 'Irrationell'])]
                if len(weak) > 0:
                    st.dataframe(weak[['Filter', 'Klass', 'Rationell faktor', 'Kvar rad %']].head(8), use_container_width=True, hide_index=True)
                else:
                    st.success("Inga filter är klassade som svaga/irrationella i denna körning.")

            with st.expander("Visa full filterdiagnos och exakta regelrader"):
                st.markdown("**Diagnoskolumner**")
                metric_cols = [
                    'Filter', 'Grupp', 'Helgardering-modul', 'Historisk träff %',
                    'Kvar rad %', 'Reducerar %', 'Rationell faktor', 'Klass'
                ]
                metric_cols = [c for c in metric_cols if c in diag_df.columns]
                st.dataframe(diag_df[metric_cols], use_container_width=True, hide_index=True)

                st.markdown("**Poängfördelning historiskt**")
                score_dist = pd.DataFrame([
                    {
                        "Antal träffade filter": score,
                        "Historiska rader": history_filter_scores.count(score),
                        "Historisk %": round((history_filter_scores.count(score) / len(history_filter_scores)) * 100, 1) if history_filter_scores else 0.0
                    }
                    for score in range(total_active, -1, -1)
                    if history_filter_scores.count(score) > 0
                ])
                st.dataframe(score_dist, use_container_width=True, hide_index=True)

                st.markdown("**Poängfördelning radmassa**")
                cand_score_dist = pd.DataFrame([
                    {
                        "Antal träffade filter": score,
                        "Rader": count,
                        "Rad %": round((count / total_candidates) * 100, 2) if total_candidates else 0.0
                    }
                    for score, count in sorted(combined_score_counts.items(), reverse=True)
                ])
                st.dataframe(cand_score_dist, use_container_width=True, hide_index=True)

                st.markdown("**Exakta regelrader / Helgardering-text**")
                st.dataframe(
                    diag_df[['Filter', 'Intervall', 'Helgardering-modul', 'Helgardering-rad']],
                    use_container_width=True,
                    hide_index=True
                )

            helg_lines = [
                f"HELGARDERING-EXPORT - {spelform}",
                f"Skapad: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Historiska omgångar: {len(v_m)} | Basfilter: {selected_hard_req} av {hard_gate_total} | Spetsfilter: {len(selected_spets_names)}",
                "",
                "GRUNDRAMSFILTER / TABELLFILTER"
            ]
            for _, rr in filter_rules_df.iterrows():
                if rr['Helgardering-modul'] not in ['Föranalys', 'Gruppmodul']:
                    keep_txt = '-' if pd.isna(rr['Kvar rad %']) else f"{rr['Kvar rad %']:.1f}%"
                    helg_lines.append(f"- {rr['Helgardering-rad']}  [{rr['Klass']}, hist {rr['Historisk träff %']:.1f}%, kvar {keep_txt}]")
            helg_lines += ["", "GRUPPER / SUPER-MAKRO"]
            if cb_super_macro:
                b = sm_bounds
                helg_lines.append(f"- Super-Makro: minst {slider_super_groups} av 8 grupper")
                helg_lines.append(f"  Grp 1 Tecken: 1 {fmt_interval(b['1'])}, X {fmt_interval(b['X'])}, 2 {fmt_interval(b['2'])}")
                helg_lines.append(f"  Grp 2 Sviter: 1 {fmt_interval(b['s1'])}, X {fmt_interval(b['sx'])}, 2 {fmt_interval(b['s2'])}")
                helg_lines.append(f"  Grp 3 Luckor: 1 {fmt_interval(b['g1'])}, X {fmt_interval(b['gx'])}, 2 {fmt_interval(b['g2'])}")
                helg_lines.append(f"  Grp 4 Singlar: 1 {fmt_interval(b['si1'])}, X {fmt_interval(b['six'])}, 2 {fmt_interval(b['si2'])}")
                helg_lines.append(f"  Grp 5 Dubbletter: 1 {fmt_interval(b['d1'])}, X {fmt_interval(b['dx'])}, 2 {fmt_interval(b['d2'])}")
                helg_lines.append(f"  Grp 6 Tripplar: 1 {fmt_interval(b['t1'])}, X {fmt_interval(b['tx'])}, 2 {fmt_interval(b['t2'])}")
                helg_lines.append(f"  Grp 7 Uppkomster: 1 {fmt_interval(b['o1'])}, X {fmt_interval(b['ox'])}, 2 {fmt_interval(b['o2'])}")
                helg_lines.append(f"  Grp 8 FAT: F {fmt_interval(b['f'])}, A {fmt_interval(b['a'])}, T {fmt_interval(b['t'])}")
            if cb_pro_groups and not pro_groups_df.empty:
                helg_lines += ["", "PRO-GRUPPER / REKOMMENDERADE GRUPPKRAV"]
                for _, gr in pro_groups_df.iterrows():
                    helg_lines.append(f"- {gr['Helgardering-rad']}  [{gr['Klass']}, hist {gr['Historisk träff %']:.1f}%, kvar {gr['Kvar rad %']:.1f}%]")
                    helg_lines.append("  Intervall/filter som ska ingå i gruppen:")
                    for line in str(gr.get('Helgardering-detaljer', '')).splitlines():
                        helg_lines.append(f"  {line}")
            helg_text = "\n".join(helg_lines)

            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                tm_download_button("⬇️ Ladda ner filterregler CSV", filter_rules_df.to_csv(index=False, sep=';').encode('utf-8-sig'), file_name=f"filterregler_{spelform.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
            with col_dl2:
                tm_download_button("⬇️ Ladda ner Helgardering TXT", helg_text.encode('utf-8'), file_name=f"helgardering_export_{spelform.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", mime="text/plain")
            with col_dl3:
                if st.button("💾 Spara analys lokalt"):
                    outdir = Path("analysis_exports")
                    outdir.mkdir(exist_ok=True)
                    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filter_rules_df.to_csv(outdir / f"filterregler_{spelform}_{stamp}.csv", index=False, sep=';', encoding='utf-8-sig')
                    (outdir / f"helgardering_export_{spelform}_{stamp}.txt").write_text(helg_text, encoding='utf-8')
                    st.success(f"Sparat i {outdir.resolve()}")

        # --- TIPSETMATRIX 12 ---
        if cb_tipsetmatrix:
            st.markdown("---")
            st.subheader("🧮 TipsetMatrix 12-rätts reducering")
            st.caption(
                "Motorn reducerar din exakta manuella grundram efter hårda basfilter och valda spetsfilter och bygger garantitabell mot den filtrerade radmassan. "
                "Garantin gäller bara om 13-rättsraden finns kvar efter grundram och filter."
            )

            tm_frame = None
            tm_frame_note = ""
            if tm_frame_source == "Klickbar grundram":
                tm_frame = tm_click_frame
                tm_frame_note = "Klickbar grundram"
            elif tm_frame_source == "Textgrundram":
                tm_frame, tm_err = parse_frame_text(tm_text_frame_input, antal_matcher)
                tm_frame_note = "Textgrundram"
                if tm_err:
                    st.error(tm_err)
            else:
                hist_rows_for_tm = [str(r).strip().upper() for r in v_m['Correct_Row'].astype(str) if len(str(r).strip()) == antal_matcher]
                match_tm_stats = build_match_ai_stats(v_m, filter_vec, antal_matcher)
                tm_frame, tm_frame_eval = optimize_frame_budgeted(
                    hist_rows_for_tm,
                    match_tm_stats,
                    antal_matcher,
                    max_rows=int(tm_base_limit),
                    max_full=int(slider_frame_max_full)
                )
                tm_frame_note = "AI-Balansram"

            if tm_frame:
                tm_rows, tm_base_count, tm_ok, tm_msg = generate_rows_from_frame(tm_frame, max_rows=int(tm_base_limit))
                ctm1, ctm2, ctm3, ctm4 = st.columns(4)
                with ctm1:
                    st.metric("Grundram", f"{tm_base_count:,}".replace(',', ' '), tm_frame_note)
                with ctm2:
                    st.metric("Bas + Spets", f"{selected_hard_req}/{hard_gate_total} + {len(selected_spets_names)} spets")
                with ctm3:
                    st.metric("Motor", tm_mode)
                with ctm4:
                    st.metric("Reduceringsläge", "Full garanti" if tm_guarantee_mode else f"Max {int(tm_output_limit):,}".replace(',', ' '))

                with st.expander("Visa grundram som reduceras"):
                    st.code(frame_export_text(tm_frame), language="text")

                tm_facit_text = ""
                tm_test_label = ""
                if cb_tm_backtest:
                    with st.expander("🧾 Facitkontroll / manuell backtestlogg", expanded=True):
                        st.caption(
                            "Klistra in facit/vinstrad från en historisk omgång. Appen visar exakt var raden dör: "
                            "i grundramen, i filtret eller i TipsetMatrix-reduceringen."
                        )
                        cfc1, cfc2 = st.columns([2, 1])
                        with cfc1:
                            # Streamlit Cloud kan ibland få cache-/frontendfel på st.text_input efter deploy.
                            # st.text_area använder annan frontend-komponent och är stabilare här.
                            tm_facit_text = st.text_area(
                                f"Facitrad ({antal_matcher} tecken, valfritt)",
                                value="",
                                placeholder="Exempel: 1X2112X21X...",
                                key=f"tm_facit_{spelform}_{antal_matcher}",
                                height=68
                            )
                        with cfc2:
                            tm_test_label = st.text_area(
                                "Testnamn/datum",
                                value="",
                                placeholder="t.ex. 2024-10-12",
                                key=f"tm_testlabel_{spelform}_{antal_matcher}",
                                height=68
                            )

                if not tm_ok:
                    st.warning(tm_msg)
                else:
                    with st.spinner("Filtrerar exakt grundram och kör TipsetMatrix 12..."):
                        # Softfiltrera exakt grundram med samma poängfunktion som filterrevisionen.
                        tm_scored = []
                        for rr in tm_rows:
                            pts = score_candidate_row(rr) if total_active > 0 else 0
                            hpts = hard_candidate_score(rr) if hard_gate_total > 0 else 0
                            tm_scored.append((rr, pts, hpts, row_log_probability(rr, filter_vec)))
                        # Teckenskydd: normal-läget ska inte omedvetet blanka ett tecken som du markerat i grundramen.
                        # Om ett valt spetsfilter gör att t.ex. M5:1 blir 0 rader kvar, tas senaste spetsfiltret bort runtime.
                        tm_runtime_hard_req = int(selected_hard_req)
                        tm_active_spets_names = list(selected_spets_names)
                        tm_active_spets_rows = list(selected_spets_rows)
                        tm_sign_guard_removed_spets = []
                        tm_sign_guard_lowered_hard = False
                        tm_sign_guard_missing_after_filter = []

                        def _runtime_filter_rows(hreq, names):
                            hard_rows = [x for x in tm_scored if (x[2] >= int(hreq) if hard_gate_total > 0 else True)]
                            filtered_rows = [x for x in hard_rows if spets_candidate_passes(x[0], names)]
                            return hard_rows, filtered_rows

                        def _runtime_history_hits(hreq, names):
                            hits = 0
                            for ii in range(len(v_m)):
                                base_ok = (history_hard_scores[ii] >= int(hreq) if hard_gate_total > 0 else True)
                                if not base_ok:
                                    continue
                                checks = _spets_history_checks(ii)
                                if all(bool(checks.get(n, False)) for n in names):
                                    hits += 1
                            return hits

                        def _try_add_runtime_replacement_spets(hreq, names, removed_names):
                            """Försöker ersätta spetsfilter som teckenskyddet tog bort.

                            Ersättaren måste:
                            - klara samlad historisk mallträff
                            - ge minsta reducering på exakt veckans filtermassa
                            - inte nolla något manuellt markerat tecken
                            """
                            added_rows = []
                            active = list(names)
                            blocked = set(removed_names or [])
                            safety = 0
                            while len(active) < int(max_spets_filters) and safety < int(max_spets_filters) * 2:
                                safety += 1
                                _, current_filtered = _runtime_filter_rows(hreq, active)
                                current_count = len(current_filtered)
                                if current_count <= 0:
                                    break
                                best = None
                                for cand_name in all_spets_names:
                                    if cand_name in active or cand_name in blocked:
                                        continue
                                    trial_names = active + [cand_name]
                                    hist_hits_trial = _runtime_history_hits(hreq, trial_names)
                                    if hist_hits_trial < target_soft_hits:
                                        continue
                                    _, trial_filtered = _runtime_filter_rows(hreq, trial_names)
                                    after_count = len(trial_filtered)
                                    if after_count <= 0 or after_count >= current_count:
                                        continue
                                    missing_trial = selected_signs_missing([x[0] for x in trial_filtered], tm_frame, antal_matcher)
                                    if missing_trial:
                                        continue
                                    red_pct = ((current_count - after_count) / current_count * 100.0) if current_count else 0.0
                                    if red_pct < float(relaxed_spets_reduction_pct):
                                        continue
                                    row = {
                                        "Filter": cand_name,
                                        "Intervall/regler": _spets_rule_text(cand_name),
                                        "Före rader": current_count,
                                        "Efter rader": after_count,
                                        "Reducerar steg %": round(red_pct, 1),
                                        "Historisk träff": hist_hits_trial,
                                        "Historisk träff %": round((hist_hits_trial / hist_total_for_soft) * 100, 1) if hist_total_for_soft else 0.0,
                                    }
                                    key = (red_pct, hist_hits_trial, -after_count)
                                    if best is None or key > best[0]:
                                        best = (key, cand_name, row)
                                if best is None:
                                    break
                                _, chosen_name, chosen_row = best
                                active.append(chosen_name)
                                added_rows.append(chosen_row)
                            return active, added_rows

                        # v11.1u: Exakt filteroptimering på den faktiska manuella grundramen.
                        # Tidigare valdes många spetsfilter först via kandidat-estimat. Det kunde ge 3→12 filter
                        # utan att den exakta radmassan minskade särskilt mycket. Här optimeras bas+spets om
                        # direkt på tm_rows/tm_scored, med samlad mallträff och teckenskydd som hårda spärrar.
                        tm_prelim_spets_names = list(tm_active_spets_names)

                        def _mask_missing_selected_signs(mask_arr):
                            """Snabb teckenskyddskontroll direkt på en bool-mask över tm_scored."""
                            missing = []
                            if mask_arr is None or len(mask_arr) == 0 or not bool(np.any(mask_arr)):
                                return selected_signs_missing([], tm_frame, antal_matcher)
                            present = [set() for _ in range(antal_matcher)]
                            idxs = np.flatnonzero(mask_arr)
                            for idx in idxs:
                                rr = tm_scored[int(idx)][0]
                                for j, ch in enumerate(rr):
                                    if j < antal_matcher:
                                        present[j].add(ch)
                            for j, allowed in enumerate(tm_frame):
                                for ch in allowed:
                                    if ch not in present[j]:
                                        missing.append((j + 1, ch))
                            return missing

                        def _rows_from_mask(mask_arr):
                            return [tm_scored[int(idx)] for idx in np.flatnonzero(mask_arr)]

                        def _build_exact_spets_np():
                            exact = {name: np.zeros(len(tm_scored), dtype=bool) for name in all_spets_names}
                            for idx, item in enumerate(tm_scored):
                                checks = _spets_candidate_checks(item[0])
                                for name in all_spets_names:
                                    exact[name][idx] = bool(checks.get(name, False))
                            return exact

                        def _optimize_exact_manual_filters():
                            exact_spets_np = _build_exact_spets_np()
                            hist_np_local = {name: np.array(hist_spets_map.get(name, []), dtype=bool) for name in all_spets_names}
                            n_exact = len(tm_scored)
                            if n_exact == 0:
                                return int(selected_hard_req), [], [], [], []

                            # Högsta bas-krav testas först, men slutvalet styrs av lägst exakt radantal.
                            # Om två paket ger samma radantal väljer vi högre historisk träff och därefter högre bas-krav.
                            hard_candidates = list(range(int(hard_gate_total), -1, -1)) if hard_gate_total > 0 else [0]
                            global_best = None
                            global_log = []
                            exact_min_reduction_pct = 5.0  # minsta faktiska stegreducering; filter under 5% tas inte med i aktivt spetsfilterpaket
                            exact_beam_width = 25
                            exact_beam_depth = min(18, max(1, len(all_spets_names)))
                            exact_max_filters = max(1, min(40, len(all_spets_names)))

                            for hreq in hard_candidates:
                                base_hist_arr = np.array([(history_hard_scores[ii] >= int(hreq) if hard_gate_total > 0 else True) for ii in range(len(v_m))], dtype=bool)
                                base_hist_hits = int(base_hist_arr.sum())
                                if base_hist_hits < int(target_soft_hits):
                                    global_log.append({"Basfilter": hreq, "Status": "Nekad", "Orsak": f"Historik {base_hist_hits}/{hist_total_for_soft}"})
                                    continue

                                base_exact_arr = np.array([(x[2] >= int(hreq) if hard_gate_total > 0 else True) for x in tm_scored], dtype=bool)
                                base_count = int(base_exact_arr.sum())
                                if base_count <= 0:
                                    global_log.append({"Basfilter": hreq, "Status": "Nekad", "Orsak": "0 rader efter bas"})
                                    continue
                                miss_base = _mask_missing_selected_signs(base_exact_arr)
                                if miss_base:
                                    global_log.append({"Basfilter": hreq, "Status": "Nekad", "Orsak": "Teckenskydd: " + format_missing_signs(miss_base)})
                                    continue

                                # Förhandslista på filter som åtminstone kan reducera från basläget och klara historiken.
                                viable_names = []
                                rejected_preview = []
                                for name in all_spets_names:
                                    h_arr = base_hist_arr & hist_np_local[name]
                                    hist_hits = int(h_arr.sum())
                                    if hist_hits < int(target_soft_hits):
                                        rejected_preview.append((name, "historik", hist_hits, base_count, base_count, 0.0))
                                        continue
                                    e_arr = base_exact_arr & exact_spets_np[name]
                                    keep_count = int(e_arr.sum())
                                    if keep_count <= 0 or keep_count >= base_count:
                                        rejected_preview.append((name, "ingen extra reducering", hist_hits, base_count, keep_count, 0.0))
                                        continue
                                    miss = _mask_missing_selected_signs(e_arr)
                                    if miss:
                                        rejected_preview.append((name, "teckenskydd", hist_hits, base_count, keep_count, 0.0))
                                        continue
                                    red_pct = ((base_count - keep_count) / base_count * 100.0) if base_count else 0.0
                                    viable_names.append((name, keep_count, red_pct, hist_hits))

                                viable_names = [x[0] for x in sorted(viable_names, key=lambda x: (x[1], -x[2], -x[3]))[:70]]

                                beam = [(tuple(), base_hist_arr, base_exact_arr, [])]
                                best_for_h = (tuple(), base_hist_arr, base_exact_arr, [])
                                examined = []
                                for _depth in range(exact_beam_depth):
                                    next_states = []
                                    seen = set()
                                    for names_tuple, hist_arr, exact_arr, details in beam:
                                        current_count = int(exact_arr.sum())
                                        if current_count <= 0 or len(names_tuple) >= exact_max_filters:
                                            continue
                                        for name in viable_names:
                                            if name in names_tuple:
                                                continue
                                            new_hist_arr = hist_arr & hist_np_local[name]
                                            hist_hits = int(new_hist_arr.sum())
                                            if hist_hits < int(target_soft_hits):
                                                continue
                                            new_exact_arr = exact_arr & exact_spets_np[name]
                                            keep_count = int(new_exact_arr.sum())
                                            if keep_count <= 0 or keep_count >= current_count:
                                                continue
                                            red_pct = ((current_count - keep_count) / current_count * 100.0) if current_count else 0.0
                                            if red_pct < exact_min_reduction_pct:
                                                continue
                                            miss = _mask_missing_selected_signs(new_exact_arr)
                                            if miss:
                                                continue
                                            new_names = tuple(list(names_tuple) + [name])
                                            sig = tuple(sorted(new_names))
                                            if sig in seen:
                                                continue
                                            seen.add(sig)
                                            row = {
                                                "Filter": name,
                                                "Intervall/regler": _spets_rule_text(name),
                                                "Före rader": current_count,
                                                "Efter rader": keep_count,
                                                "Reducerar steg %": round(red_pct, 2),
                                                "Kvar total %": round((keep_count / n_exact * 100), 1) if n_exact else 0.0,
                                                "Historisk träff": hist_hits,
                                                "Historisk träff %": round((hist_hits / hist_total_for_soft) * 100, 1) if hist_total_for_soft else 0.0,
                                                "Godkänd historik": True,
                                                "Godkänd reduktion": True,
                                            }
                                            examined.append(row)
                                            next_states.append((new_names, new_hist_arr, new_exact_arr, details + [row]))
                                    if not next_states:
                                        break
                                    next_states = sorted(next_states, key=lambda stt: (int(stt[2].sum()), -int(stt[1].sum()), len(stt[0])))[:exact_beam_width]
                                    beam = next_states
                                    candidate = beam[0]
                                    if int(candidate[2].sum()) < int(best_for_h[2].sum()):
                                        best_for_h = candidate

                                # Maxfyll exakt: fortsätt lägga till varje filter som faktiskt minskar den exakta manuella radmassan.
                                fill_names = tuple(best_for_h[0])
                                fill_hist_arr = np.array(best_for_h[1], dtype=bool)
                                fill_exact_arr = np.array(best_for_h[2], dtype=bool)
                                fill_details = list(best_for_h[3])
                                safety = 0
                                while len(fill_names) < exact_max_filters and safety < exact_max_filters * 2:
                                    safety += 1
                                    current_count = int(fill_exact_arr.sum())
                                    best_add = None
                                    for name in viable_names:
                                        if name in fill_names:
                                            continue
                                        new_hist_arr = fill_hist_arr & hist_np_local[name]
                                        hist_hits = int(new_hist_arr.sum())
                                        if hist_hits < int(target_soft_hits):
                                            continue
                                        new_exact_arr = fill_exact_arr & exact_spets_np[name]
                                        keep_count = int(new_exact_arr.sum())
                                        if keep_count <= 0 or keep_count >= current_count:
                                            continue
                                        red_pct = ((current_count - keep_count) / current_count * 100.0) if current_count else 0.0
                                        if red_pct < exact_min_reduction_pct:
                                            continue
                                        miss = _mask_missing_selected_signs(new_exact_arr)
                                        if miss:
                                            continue
                                        row = {
                                            "Filter": name,
                                            "Intervall/regler": _spets_rule_text(name),
                                            "Före rader": current_count,
                                            "Efter rader": keep_count,
                                            "Reducerar steg %": round(red_pct, 2),
                                            "Kvar total %": round((keep_count / n_exact * 100), 1) if n_exact else 0.0,
                                            "Historisk träff": hist_hits,
                                            "Historisk träff %": round((hist_hits / hist_total_for_soft) * 100, 1) if hist_total_for_soft else 0.0,
                                            "Godkänd historik": True,
                                            "Godkänd reduktion": True,
                                        }
                                        key = (keep_count, -red_pct, -hist_hits)
                                        if best_add is None or key < best_add[0]:
                                            best_add = (key, name, new_hist_arr, new_exact_arr, row)
                                    if best_add is None:
                                        break
                                    _, add_name, fill_hist_arr, fill_exact_arr, add_row = best_add
                                    fill_names = tuple(list(fill_names) + [add_name])
                                    fill_details.append(add_row)
                                    examined.append(add_row)

                                final_count = int(fill_exact_arr.sum())
                                final_hits = int(fill_hist_arr.sum())
                                pack = {
                                    "hreq": int(hreq),
                                    "names": list(fill_names),
                                    "hist_arr": fill_hist_arr,
                                    "exact_arr": fill_exact_arr,
                                    "rows": fill_details,
                                    "final_count": final_count,
                                    "hist_hits": final_hits,
                                    "base_count": base_count,
                                    "examined": examined,
                                }
                                global_log.append({
                                    "Basfilter": hreq,
                                    "Status": "Testad",
                                    "Orsak": f"{base_count} → {final_count} rader, {final_hits}/{hist_total_for_soft} hist, {len(fill_names)} spets",
                                })
                                key = (final_count, -final_hits, -int(hreq), len(fill_names))
                                if global_best is None or key < global_best[0]:
                                    global_best = (key, pack)

                            if global_best is None:
                                # Fallback till tidigare kandidatval om exakt sökning inte hittar något paket.
                                hreq = int(selected_hard_req)
                                hard_rows, filtered_rows = _runtime_filter_rows(hreq, tm_prelim_spets_names)
                                return hreq, list(tm_prelim_spets_names), list(tm_active_spets_rows), global_log, ["fallback"]

                            pack = global_best[1]
                            return int(pack["hreq"]), list(pack["names"]), list(pack["rows"]), global_log, []

                        tm_runtime_hard_req, tm_active_spets_names, tm_active_spets_rows, tm_exact_optimizer_log, tm_exact_optimizer_fallback = _optimize_exact_manual_filters()
                        tm_sign_guard_lowered_hard = int(tm_runtime_hard_req) < int(selected_hard_req)
                        tm_sign_guard_removed_spets = [n for n in tm_prelim_spets_names if n not in set(tm_active_spets_names)]
                        tm_sign_guard_added_spets_rows = [r for r in tm_active_spets_rows if r.get("Filter") not in set(tm_prelim_spets_names)]

                        tm_hard_scored, tm_filtered_scored = _runtime_filter_rows(tm_runtime_hard_req, tm_active_spets_names)
                        tm_sign_guard_missing_after_filter = selected_signs_missing([x[0] for x in tm_filtered_scored], tm_frame, antal_matcher)

                        truncated = False
                        if len(tm_filtered_scored) > int(tm_filter_limit):
                            truncated = True
                            # Säkerhet för Streamlit: behåll bästa raderna om filtermassan är för stor.
                            # Garantitabellen gäller då mot toppurvalet, inte hela filtrerade massan.
                            tm_filtered_scored = sorted(tm_filtered_scored, key=lambda x: (x[1], x[2]), reverse=True)[:int(tm_filter_limit)]

                        tm_filtered_rows = [x[0] for x in tm_filtered_scored]
                        if tm_weighting == "Neutral":
                            tm_scores = [0.0 for _ in tm_filtered_scored]
                        elif tm_weighting == "Filterpoäng":
                            tm_scores = [float(x[1]) + float(x[2]) * 2.0 for x in tm_filtered_scored]
                        else:
                            logs = [float(x[3]) for x in tm_filtered_scored]
                            if logs:
                                mn, mx = min(logs), max(logs)
                                span = max(mx - mn, 1e-9)
                                tm_scores = [(float(x[1]) * 10.0 + float(x[2]) * 15.0 + ((float(x[3]) - mn) / span) * 5.0) for x in tm_filtered_scored]
                            else:
                                tm_scores = []

                        tm_reduced_rows, tm_meta = tipsetmatrix12_reduce(
                            tm_filtered_rows,
                            row_scores=tm_scores,
                            mode=tm_mode,
                            max_output_rows=(None if tm_guarantee_mode else int(tm_output_limit)),
                            seed=int(tm_seed)
                        )

                        # Teckenskydd även efter själva TipsetMatrix-valet.
                        # 12-garantin kan tekniskt täcka en saknad 13-rad via ett annat tecken, men för Egna rader
                        # vill vi inte ha 0 finalrader på ett tecken som spelaren aktivt markerat.
                        tm_reduced_rows, tm_sign_guard_added_rows, tm_sign_guard_missing_after_reduce = add_rows_for_sign_coverage(
                            tm_filtered_rows,
                            tm_reduced_rows,
                            tm_frame,
                            row_scores=tm_scores,
                            antal_matcher=antal_matcher
                        )

                        tm_reduced_rows, tm_13chance_added_rows = add_rows_for_min_13_chance(
                            tm_filtered_rows,
                            tm_reduced_rows,
                            target_pct=float(tm_min_13_chance_pct),
                            row_scores=tm_scores,
                            antal_matcher=antal_matcher
                        )

                        tm_gtable, tm_gsum = build_tipsetmatrix_guarantee_table(
                            tm_filtered_rows,
                            tm_reduced_rows,
                            antal_matcher,
                            prob_vector=filter_vec
                        )

                    if truncated:
                        st.warning(
                            f"Efter bas+spetsfilter fanns fler än spärren {int(tm_filter_limit):,} rader. "
                            "TipsetMatrix kördes på de högst rankade raderna. Garantitabellen gäller därför mot detta toppurval. "
                            "Höj spärren om du vill räkna på hela filtermassan."
                        )

                    if len(tm_filtered_rows) == 0:
                        st.error("Inga rader överlevde bas+spetsfilter i den valda grundramen. Bredda ramen eller sänk historikmålet.")
                    else:
                        if tm_sign_guard_removed_spets or tm_sign_guard_lowered_hard or tm_sign_guard_added_rows or tm_sign_guard_added_spets_rows or tm_13chance_added_rows:
                            parts = []
                            if tm_sign_guard_removed_spets:
                                parts.append("tog bort spetsfilter: " + ", ".join(tm_sign_guard_removed_spets))
                            if tm_sign_guard_added_spets_rows:
                                parts.append("lade till/ersatte med spetsfilter: " + ", ".join([r.get("Filter", "") for r in tm_sign_guard_added_spets_rows]))
                            if tm_sign_guard_lowered_hard:
                                parts.append(f"sänkte basfilter från {selected_hard_req}/{hard_gate_total} till {tm_runtime_hard_req}/{hard_gate_total}")
                            if tm_sign_guard_added_rows:
                                parts.append(f"lade till {len(tm_sign_guard_added_rows)} slutrad(er) för teckenskydd")
                            if tm_13chance_added_rows:
                                parts.append(f"lade till {len(tm_13chance_added_rows)} slutrad(er) för att nå minst {float(tm_min_13_chance_pct):.0f}% 13-chans")
                            st.warning("🛡️ Teckenskydd/13-chans aktivt: " + "; ".join(parts) + ".")
                        if tm_sign_guard_missing_after_filter:
                            st.error("Teckenskydd kunde inte bevara dessa tecken efter filter: " + format_missing_signs(tm_sign_guard_missing_after_filter))
                        if tm_sign_guard_missing_after_reduce:
                            st.error("Teckenskydd kunde inte bevara dessa tecken efter TipsetMatrix: " + format_missing_signs(tm_sign_guard_missing_after_reduce))

                        ctm5, ctm6, ctm7, ctm8 = st.columns(4)
                        with ctm5:
                            st.metric("Efter bas → spets", f"{len(tm_hard_scored):,} → {len(tm_filtered_rows):,}".replace(',', ' '))
                        with ctm6:
                            st.metric("Efter TipsetMatrix", f"{len(tm_reduced_rows):,}".replace(',', ' '), f"-{tm_gsum['reduceringsgrad']:.1f}%")
                        with ctm7:
                            st.metric("12+ garanti", f"{tm_gsum['12plus']:.2f}%", f"min {tm_gsum['min_garanti']} rätt")
                        with ctm8:
                            st.metric("13-chans", f"{tm_gsum['13_oviktad']:.2f}%", f"viktad {tm_gsum['13_viktad']:.2f}%")
                        st.caption(f"Mål för 13-chans: minst {float(tm_min_13_chance_pct):.0f}% i normal/expert-inställningen. Om målet är högre än grundreduceringen läggs extra högst rankade rader till från filtermassan.")

                        if True:
                            st.caption("v11.1u: Bas + spets som används i TipsetMatrix är nu omoptimerat exakt på den sparade manuella grundramen, inte bara på kandidat-estimatet.")

                        if tm_active_spets_names != selected_spets_names:
                            st.markdown("**Faktiskt använda spetsfilter efter exakt optimering / teckenskydd**")
                            actual_rows = []
                            for idx, name in enumerate(tm_active_spets_names, start=1):
                                row_info = next((r for r in tm_active_spets_rows if r.get("Filter") == name), None)
                                actual_rows.append({
                                    "#": idx,
                                    "Filter": name,
                                    "Intervall/regler": (row_info or {}).get("Intervall/regler", _spets_rule_text(name)),
                                    "Historisk träff": (f"{int((row_info or {}).get('Historisk träff', 0))}/{hist_total_for_soft}" if row_info else "-"),
                                    "Före": int((row_info or {}).get("Före rader", 0)) if row_info else "-",
                                    "Efter": int((row_info or {}).get("Efter rader", 0)) if row_info else "-",
                                    "Reducerar": (f"{float((row_info or {}).get('Reducerar steg %', 0.0)):.1f}%" if row_info else "-"),
                                })
                            st.dataframe(pd.DataFrame(actual_rows), use_container_width=True, hide_index=True)

                        if not tm_meta.get('complete', False):
                            st.error(
                                f"Reduceraren täckte {tm_meta.get('covered_pct', 0):.2f}% av filtermassan med max {int(tm_output_limit)} rader. "
                                "Höj max slutrader eller kör Balans/Max för full 12-garanti."
                            )
                        else:
                            st.success("TipsetMatrix hittade full 12-rättsgaranti inom den filtrerade radmassan.")

                        # Facitkontroll för historiska omgångar.
                        facit_row, facit_err = parse_result_row(tm_facit_text, antal_matcher) if cb_tm_backtest else (None, "")
                        facit_report = None
                        if cb_tm_backtest and tm_facit_text:
                            if facit_err:
                                st.error(facit_err)
                            elif facit_row:
                                facit_report = build_facit_check(
                                    facit_row,
                                    tm_frame,
                                    tm_rows,
                                    tm_filtered_rows,
                                    tm_reduced_rows,
                                    antal_matcher
                                )
                                st.markdown("### 🧾 Facitkontroll")
                                f1, f2, f3, f4, f5 = st.columns(5)
                                with f1:
                                    st.metric("I grundram", yes_no(facit_report["I grundram"]))
                                with f2:
                                    st.metric("Efter filter", yes_no(facit_report["Efter filter"]))
                                with f3:
                                    st.metric("13 efter TM", yes_no(facit_report["Efter TipsetMatrix"]))
                                with f4:
                                    st.metric("Bästa rätt", f"{facit_report['Bästa rätt efter TipsetMatrix']} rätt")
                                with f5:
                                    st.metric("12+", yes_no(facit_report["12+ uppnått"]))

                                if facit_report["Efter filter"] and not facit_report["12+ uppnått"]:
                                    st.error("VARNING: Facitraden fanns efter filter men TipsetMatrix gav inte 12+. Det tyder på fel i reduceringen eller för låg maxgräns.")
                                elif facit_report["Efter filter"] and facit_report["12+ uppnått"] and not facit_report["Efter TipsetMatrix"]:
                                    st.info("Facitraden överlevde filtret men valdes inte som 13-rad. 12-rättsgarantin fungerade ändå.")
                                elif facit_report["Efter TipsetMatrix"]:
                                    st.success("Facitraden finns exakt bland de reducerade raderna: 13-rättsträff i detta test.")
                                elif not facit_report["I grundram"]:
                                    st.warning("Facitraden fanns inte i grundramen. Då kan inget filter eller reducering rädda 13 rätt.")
                                elif not facit_report["Efter filter"]:
                                    st.warning("Facitraden fanns i grundramen men filtrerades bort. Då är bas-/spetspaketet för hårt för just denna omgång.")

                                with st.expander("Visa närmaste reducerade rad/rader"):
                                    if facit_report["Närmaste reducerade rader"]:
                                        st.code("\n".join(facit_report["Närmaste reducerade rader"]), language="text")
                                    else:
                                        st.caption("Inga reducerade rader att jämföra med.")

                                if "tm_manual_backtest_log" not in st.session_state:
                                    st.session_state["tm_manual_backtest_log"] = []
                                if st.button("➕ Lägg till i manuell backtestlogg", key="tm_add_manual_backtest"):
                                    st.session_state["tm_manual_backtest_log"].append({
                                        "Tid": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        "Spelform": spelform,
                                        "Test": tm_test_label,
                                        "Facitrad": facit_row,
                                        "Grundram rader": tm_base_count,
                                        "Efter filter": len(tm_filtered_rows),
                                        "Efter TipsetMatrix": len(tm_reduced_rows),
                                        "I grundram": yes_no(facit_report["I grundram"]),
                                        "Efter filter facit": yes_no(facit_report["Efter filter"]),
                                        "13 efter TM": yes_no(facit_report["Efter TipsetMatrix"]),
                                        "Bästa rätt": facit_report["Bästa rätt efter TipsetMatrix"],
                                        "12+": yes_no(facit_report["12+ uppnått"]),
                                        "13-chans %": tm_gsum['13_oviktad'],
                                        "13-chans viktad %": tm_gsum['13_viktad'],
                                        "12+ garanti %": tm_gsum['12plus'],
                                        "Spetsfilter": ", ".join(tm_active_spets_names) if tm_active_spets_names else "Inga",
                                        "Motor": tm_mode,
                                        "Viktning": tm_weighting,
                                    })
                                    st.success("Tillagd i manuell backtestlogg.")

                        if cb_tm_backtest and st.session_state.get("tm_manual_backtest_log"):
                            st.markdown("### 📘 Manuell backtestlogg")
                            log_df = pd.DataFrame(st.session_state["tm_manual_backtest_log"])
                            st.dataframe(log_df, use_container_width=True, hide_index=True)
                            tm_download_button(
                                "⬇️ Ladda ner manuell backtestlogg CSV",
                                log_df.to_csv(index=False, sep=';').encode('utf-8-sig'),
                                file_name=f"tipsetmatrix_manual_backtest_{spelform.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                            if st.button("🧹 Rensa manuell backtestlogg", key="tm_clear_manual_backtest"):
                                st.session_state["tm_manual_backtest_log"] = []
                                st.success("Backtestloggen rensades.")

                        with st.expander("📊 Teckenfördelning per match/lag", expanded=True):
                            st.caption(
                                "Visar hur varje match/lag får sina 1/X/2-tecken efter bas+spetsfilter och i de faktiska inlämningsraderna. "
                                "Status ska vara OK för alla tecken du markerat i grundramen."
                            )
                            st.dataframe(
                                build_combined_sign_distribution_df(tm_filtered_rows, tm_reduced_rows, tm_frame, antal_matcher),
                                use_container_width=True,
                                hide_index=True
                            )
                            with st.expander("Detaljtabeller med separata kolumner", expanded=False):
                                st.markdown("**Efter bas+spetsfilter**")
                                st.dataframe(build_sign_distribution_df(tm_filtered_rows, tm_frame, antal_matcher), use_container_width=True, hide_index=True)
                                st.markdown("**Efter TipsetMatrix / inlämningsrader**")
                                st.dataframe(build_sign_distribution_df(tm_reduced_rows, tm_frame, antal_matcher), use_container_width=True, hide_index=True)

                        st.markdown("### 📊 Garantitabell – TipsetMatrix 12")
                        st.dataframe(tm_gtable, use_container_width=True, hide_index=True)

                        tm_submission_text = rows_to_submission_text(tm_reduced_rows, spelform, antal_matcher)

                        with st.expander("Visa reducerade rader / inlämningsformat"):
                            st.markdown("**Inlämningsformat för Egna rader**")
                            st.caption("Första raden är spelformen. Varje rad börjar med spelformens prefix, t.ex. E för Europatipset.")
                            st.code("\n".join(tm_submission_text.splitlines()[:1000]), language="text")
                            if len(tm_reduced_rows) > 999:
                                st.caption("Visar första 999 spelraderna plus rubrik i appen. Nedladdningen innehåller alla.")
                            st.markdown("**Råa reducerade rader utan prefix**")
                            st.code("\n".join(tm_reduced_rows[:1000]), language="text")

                        tm_export_lines = [
                            f"TIPSETMATRIX 12 - {spelform}",
                            f"Skapad: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            f"Grundram: {frame_export_text(tm_frame)}",
                            f"Grundram rader: {tm_base_count}",
                            f"Efter filter: {len(tm_filtered_rows)}",
                            f"Basfilter: {tm_runtime_hard_req} av {hard_gate_total}",
                            f"Spetsfilter: {', '.join(tm_active_spets_names) if tm_active_spets_names else 'Inga'}",
                            "",
                            "VALDA SPETSFILTER OCH INTERVALL",
                            *[f"{idx}. {r['Filter']} | {r.get('Intervall/regler', _spets_rule_text(r['Filter']))} | {int(r['Före rader'])} -> {int(r['Efter rader'])} | hist {int(r['Historisk träff'])}/{hist_total_for_soft}" for idx, r in enumerate(tm_active_spets_rows, start=1)],
                            "",
                            f"Efter TipsetMatrix: {len(tm_reduced_rows)}",
                            f"12+ garanti: {tm_gsum['12plus']:.2f}%",
                            f"Minsta 13-chans mål: {float(tm_min_13_chance_pct):.0f}%",
                            f"13-chans oviktad: {tm_gsum['13_oviktad']:.2f}%",
                            f"13-chans viktad: {tm_gsum['13_viktad']:.2f}%",
                            f"Extra rader för 13-chans: {len(tm_13chance_added_rows)}",
                        ]
                        if facit_report:
                            tm_export_lines.extend([
                                "",
                                "FACITKONTROLL",
                                f"Test: {tm_test_label}",
                                f"Facitrad: {facit_report['Facitrad']}",
                                f"I grundram: {yes_no(facit_report['I grundram'])}",
                                f"Efter filter: {yes_no(facit_report['Efter filter'])}",
                                f"Efter TipsetMatrix: {yes_no(facit_report['Efter TipsetMatrix'])}",
                                f"Bästa rätt efter TipsetMatrix: {facit_report['Bästa rätt efter TipsetMatrix']}",
                                f"12+ uppnått: {yes_no(facit_report['12+ uppnått'])}",
                            ])
                        tm_export_lines.extend([
                            "",
                            "INLÄMNINGSFORMAT EGNA RADER",
                            tm_submission_text,
                            "",
                            "REDUCERADE RADER UTAN PREFIX",
                            *tm_reduced_rows,
                        ])
                        tm_export_text = "\n".join(tm_export_lines)
                        dl_tm1, dl_tm2, dl_tm3 = st.columns(3)
                        with dl_tm1:
                            tm_download_button(
                                "⬇️ Ladda ner inlämningsfil TXT",
                                tm_submission_text.encode('utf-8'),
                                file_name=f"egna_rader_{submission_file_stem(spelform)}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain"
                            )
                        with dl_tm2:
                            tm_download_button(
                                "⬇️ Ladda ner råa rader TXT",
                                "\n".join(tm_reduced_rows).encode('utf-8'),
                                file_name=f"tipsetmatrix12_rader_{spelform.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain"
                            )
                        with dl_tm3:
                            tm_download_button(
                                "⬇️ Ladda ner rapport TXT",
                                tm_export_text.encode('utf-8'),
                                file_name=f"tipsetmatrix12_rapport_{spelform.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain"
                            )

            if cb_tm_backtest:
                st.markdown("---")
                st.subheader("📈 Nästa backteststeg")
                st.info(
                    "Denna version har facitkontroll och manuell backtestlogg. "
                    "Nästa större steg är automatisk walk-forward-backtest där varje historisk omgång bara får använda äldre omgångar."
                )

        # --- AI-RAM & U-FILTER ---
        if cb_ai_frame:
            st.markdown("---")
            st.subheader("🎯 AI-Ram & U-filter")
            st.caption(
                "Rammotorn använder de liknande historiska omgångarna för att föreslå spikar/halvor/hela. "
                "Den backtestar ramen mot historiken och optimerar bästa täckning inom en radbudget, så ramen inte fylls med för många helgarderingar."
            )

            hist_rows_for_frame = [str(r).strip().upper() for r in v_m['Correct_Row'].astype(str) if len(str(r).strip()) == antal_matcher]
            match_ai_stats = build_match_ai_stats(v_m, filter_vec, antal_matcher)

            frame_profiles = [
                ("Aggressiv", max(200, int(slider_frame_budget * 0.35)), max(0, slider_frame_max_full - 2)),
                ("Balans", int(slider_frame_budget), int(slider_frame_max_full)),
                ("Trygg", int(slider_frame_budget * 2), min(antal_matcher, int(slider_frame_max_full + 1))),
            ]
            frame_results = []
            frame_detail_texts = []
            for label, max_rows, max_full in frame_profiles:
                frame, ev = optimize_frame_budgeted(
                    hist_rows_for_frame,
                    match_ai_stats,
                    antal_matcher,
                    max_rows=max_rows,
                    max_full=max_full
                )
                frame_txt = frame_to_string(frame)
                frame_results.append({
                    "Ram": label,
                    "Max rader": max_rows,
                    "Max hela": max_full,
                    "Radantal": ev['Radantal'],
                    "Spikar": ev['Spikar'],
                    "Halvor": ev['Halvor'],
                    "Hela": ev['Hela'],
                    "13 inom ram %": ev['13'],
                    "12+ inom ram %": ev['12+'],
                    "11+ inom ram %": ev['11+'],
                    "10+ inom ram %": ev['10+'],
                    "Ramtecken": frame_txt,
                })
                frame_detail_texts.append(
                    f"{label}: {frame_txt}\n"
                    f"Radantal: {ev['Radantal']} av max {max_rows} | Spikar {ev['Spikar']} | Halvor {ev['Halvor']} | Hela {ev['Hela']} av max {max_full}\n"
                    f"Historisk täckning: 13 {ev['13']}%, 12+ {ev['12+']}%, 11+ {ev['11+']}%, 10+ {ev['10+']}%"
                )

            frame_df = pd.DataFrame(frame_results)
            st.dataframe(frame_df, use_container_width=True, hide_index=True)

            # AI-U-rad: bästa historiska tecken per match bland de liknande omgångarna.
            ai_u_row = build_ai_u_row(match_ai_stats)
            u_eval = evaluate_u_row(ai_u_row, hist_rows_for_frame, antal_matcher, target_pct=slider_u_target)

            c_ai1, c_ai2, c_ai3 = st.columns(3)
            with c_ai1:
                st.metric("AI-U-rad", ai_u_row)
            with c_ai2:
                st.metric("Rekommenderat U-filter", f"{u_eval['recommended_min']}-{antal_matcher}")
            with c_ai3:
                st.metric("Historisk täckning", f"{u_eval['coverage']:.1f}%", f"mål {slider_u_target}%")

            st.info(
                "Praktiskt: använd AI-ramen som grundram om du vill att appen ska välja spik/halv/hel. "
                "Eller använd AI-U-raden som utgångsfilter, t.ex. "
                f"{u_eval['recommended_min']}-{antal_matcher} rätt, om du vill styra systemet utan att låsa ramen för hårt."
            )

            with st.expander("Visa matchvis AI-bedömning för ramval"):
                match_df = pd.DataFrame(match_ai_stats)
                cols = ['Match', 'Top1', 'Top2', '1 hist %', 'X hist %', '2 hist %', '1 idag %', 'X idag %', '2 idag %']
                st.dataframe(match_df[cols], use_container_width=True, hide_index=True)

            with st.expander("Visa U-radens historiska träfffördelning"):
                st.dataframe(u_eval['dist'], use_container_width=True, hide_index=True)

            ai_frame_export_lines = [
                f"AI-RAM & U-FILTER - {spelform}",
                f"Skapad: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Liknande historiska omgångar: {len(hist_rows_for_frame)}",
                "",
                "RAMFÖRSLAG"
            ]
            for txt in frame_detail_texts:
                ai_frame_export_lines += [txt, ""]
            ai_frame_export_lines += [
                "AI-U-RAD",
                f"U-rad: {ai_u_row}",
                f"Rekommenderat U-filter: {u_eval['recommended_min']}-{antal_matcher} rätt",
                f"Historisk täckning: {u_eval['coverage']:.1f}% vid mål {slider_u_target}%",
                "",
                "MATCHVIS HISTORIK"
            ]
            for ms in match_ai_stats:
                ai_frame_export_lines.append(
                    f"M{ms['Match']}: Top1 {ms['Top1']} | Top2 {ms['Top2']} | "
                    f"Hist 1 {ms['1 hist %']}%, X {ms['X hist %']}%, 2 {ms['2 hist %']}% | "
                    f"Idag 1 {ms['1 idag %']}%, X {ms['X idag %']}%, 2 {ms['2 idag %']}%"
                )
            ai_frame_export = "\n".join(ai_frame_export_lines)
            tm_download_button(
                "⬇️ Ladda ner AI-Ram & U-filter TXT",
                ai_frame_export.encode('utf-8'),
                file_name=f"ai_ram_u_filter_{spelform.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

        st.markdown("---")
        st.subheader("🧬 Dagens Bästa FAT-Sekvenser (Byggklossar)")
        st.markdown("Här analyserar AI:n vilka specifika mönster (1=Fav, 2=Andrahand, 3=Skräll) som bäst täcker in **exakt denna typ av omgång**. Statistiken visar hur många av de 5 sekvenserna som brukar dyka upp i en och samma vinnarrad.")
        st.success("FAT-sekvenser används nu som aktivt mjukt reduceringsfilter och syns i Filterrevisionen. Urvalet prioriterar positiv lift och rationell effekt.")

        fat_strings = [get_fat_string(row['Correct_Row'], row['Prob_Vector']) for _, row in v_m.iterrows() if len(row['Correct_Row']) == antal_matcher]
        base_fat_strings = [get_fat_string(row['Correct_Row'], row['Prob_Vector']) for _, row in db_full.iterrows() if len(str(row['Correct_Row'])) == antal_matcher]
        total_twins = len(fat_strings)
        total_base_fat = len(base_fat_strings)
        
        if total_twins > 0:
            col_seq2, col_seq3, col_combo = st.columns(3)
            
            def calculate_top_seqs(fat_list, length, top_n=5):
                seqs = [''.join(p) for p in itertools.product('123', repeat=length)]
                counts = {s: sum(1 for r in fat_list if s in r) for s in seqs}
                return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

            with col_seq2:
                st.markdown("**Två i rad (Längd 2)**")
                top2 = calculate_top_seqs(fat_strings, 2)
                for seq, count in top2:
                    chans = (count/total_twins)*100
                    base_pct = (sum(1 for r in base_fat_strings if seq in r) / total_base_fat) * 100 if total_base_fat else 0
                    st.write(f"**{seq}** ➡️ **{chans:.1f}%** ({count} st) | Bas {base_pct:.1f}% | Lift {chans-base_pct:+.1f}")
                
                if top2:
                    top2_seqs = [x[0] for x in top2]
                    hits2 = [sum(1 for seq in top2_seqs if seq in r) for r in fat_strings]
                    st.markdown("---")
                    st.markdown(get_stat_strings(hits2, len(top2_seqs)))
                    
            with col_seq3:
                st.markdown("**Tre i rad (Längd 3)**")
                top3 = calculate_top_seqs(fat_strings, 3)
                for seq, count in top3:
                    chans = (count/total_twins)*100
                    base_pct = (sum(1 for r in base_fat_strings if seq in r) / total_base_fat) * 100 if total_base_fat else 0
                    st.write(f"**{seq}** ➡️ **{chans:.1f}%** ({count} st) | Bas {base_pct:.1f}% | Lift {chans-base_pct:+.1f}")
                
                if top3:
                    top3_seqs = [x[0] for x in top3]
                    hits3 = [sum(1 for seq in top3_seqs if seq in r) for r in fat_strings]
                    st.markdown("---")
                    st.markdown(get_stat_strings(hits3, len(top3_seqs)))
                    
            with col_combo:
                st.markdown("**Dubbelchans (Minst 1 av 2)**")
                st.caption("Kombinationer av Längd 3")
                
                seqs3 = [''.join(p) for p in itertools.product('123', repeat=3)]
                pair_counts = []
                for s1, s2 in itertools.combinations(seqs3, 2):
                    covered = sum(1 for r in fat_strings if s1 in r or s2 in r)
                    pair_counts.append(((s1, s2), covered))
                
                best_pairs = sorted(pair_counts, key=lambda x: x[1], reverse=True)[:5]
                for (s1, s2), count in best_pairs:
                    chans = (count/total_twins)*100
                    base_pct = (sum(1 for r in base_fat_strings if s1 in r or s2 in r) / total_base_fat) * 100 if total_base_fat else 0
                    st.write(f"**{s1}** / **{s2}** ➡️ **{chans:.1f}%** ({count} st) | Bas {base_pct:.1f}% | Lift {chans-base_pct:+.1f}")
                
                if best_pairs:
                    # En Dubbelchans räknas som "satt" om minst en av de två sekvenserna finns i raden
                    hits_pairs = [sum(1 for pair in best_pairs if pair[0][0] in r or pair[0][1] in r) for r in fat_strings]
                    st.markdown("---")
                    st.markdown(get_stat_strings(hits_pairs, len(best_pairs)))
                    
        st.markdown("---")
        st.subheader("🧠 AI:ns Strategiska Byggklossar (För reducering)")
        
        match_data = []
        for m in range(antal_matcher):
            hist_outcomes = [row['Correct_Row'][m] for _, row in v_m.iterrows() if len(row['Correct_Row']) == antal_matcher]
            if not hist_outcomes: continue
                
            tot = len(hist_outcomes)
            c1, cx, c2 = hist_outcomes.count('1'), hist_outcomes.count('X'), hist_outcomes.count('2')
            p1, px, p2 = (c1/tot)*100, (cx/tot)*100, (c2/tot)*100
            utfall = [('1', p1), ('X', px), ('2', p2)]
            utfall.sort(key=lambda x: x[1], reverse=True)
            
            odds_idag = {'1': input_vec[m*3], 'X': input_vec[m*3+1], '2': input_vec[m*3+2]}
            
            best_single = utfall[0]
            value_single = best_single[1] - odds_idag[best_single[0]]
            
            best_double_signs = sorted([utfall[0][0], utfall[1][0]])
            best_double_str = "1X" if best_double_signs==['1','X'] else "12" if best_double_signs==['1','2'] else "X2"
            best_double_pct = utfall[0][1] + utfall[1][1]
            
            skrällar = []
            for sign, hist_pct in utfall:
                if odds_idag[sign] < 20:
                    skrällar.append({
                        'match': m+1, 'sign': sign, 'hist_pct': hist_pct, 'odds_idag': odds_idag[sign]
                    })
                    
            match_data.append({
                'match': m+1,
                'odds_idag': odds_idag,
                'best_single_sign': best_single[0],
                'best_single_pct': best_single[1],
                'value_single': value_single,
                'best_double_str': best_double_str,
                'best_double_pct': best_double_pct,
                'skrällar': skrällar
            })

        spikar_sorted = sorted(match_data, key=lambda x: (x['best_single_pct'], x['value_single']), reverse=True)
        top_5_spikar = spikar_sorted[:5]
        
        las_sorted = sorted(match_data, key=lambda x: x['best_double_pct'], reverse=True)
        top_5_las = las_sorted[:5]
        
        all_skrallar = []
        for md in match_data:
            all_skrallar.extend(md['skrällar'])
        top_5_skrallar = sorted(all_skrallar, key=lambda x: x['hist_pct'], reverse=True)[:5]
        
        # --- BERÄKNA HISTORISKT UTFALL FÖR BYGGKLOSSARNA ---
        spik_hits, las_hits, skrall_hits = [], [], []
        valid_rows_for_stats = [row['Correct_Row'] for _, row in v_m.iterrows() if len(row['Correct_Row']) == antal_matcher]
        
        for r_str in valid_rows_for_stats:
            spik_hits.append(sum(1 for s in top_5_spikar if r_str[s['match']-1] == s['best_single_sign']))
            las_hits.append(sum(1 for l in top_5_las if r_str[l['match']-1] in l['best_double_str']))
            skrall_hits.append(sum(1 for sk in top_5_skrallar if r_str[sk['match']-1] == sk['sign']))

        # --- RITA UPP BYGGKLOSSARNA ---
        col_spik, col_las, col_skrall = st.columns(3)
        
        with col_spik:
            st.markdown("🔥 **Topp 5 Spikarna**")
            for s in top_5_spikar:
                st.write(f"**M{s['match']}: {s['best_single_sign']}** (Vinner {s['best_single_pct']:.0f}%, Streck {s['odds_idag'][s['best_single_sign']]:.0f}%)")
            st.markdown("---")
            st.markdown(get_stat_strings(spik_hits, len(top_5_spikar)))

        with col_las:
            st.markdown("🔒 **Topp 5 Låsen**")
            for l in top_5_las:
                st.write(f"**M{l['match']}: {l['best_double_str']}** (Täcker {l['best_double_pct']:.0f}%)")
            st.markdown("---")
            st.markdown(get_stat_strings(las_hits, len(top_5_las)))

        with col_skrall:
            st.markdown("💣 **Topp 5 Skrälldrag (<20%)**")
            if not top_5_skrallar:
                st.write("Hittade inga skrällar under 20% idag.")
            else:
                for sk in top_5_skrallar:
                    st.write(f"**M{sk['match']}: {sk['sign']}** (Vinner {sk['hist_pct']:.0f}%, Streck {sk['odds_idag']:.0f}%)")
                st.markdown("---")
                st.markdown(get_stat_strings(skrall_hits, len(top_5_skrallar)))

        if antal_matcher == 8:
            st.markdown("---")
            st.subheader("🎲 EXAKT UTRÄKNING (6 561 rader)")
            all_possible_rows = [''.join(tup) for tup in itertools.product(['1','X','2'], repeat=8)]
            ai_matrix, ai_scores_asc, ai_tot = calculate_ai_matrix_from_values(filter_vec)
            
            valid_exact_rows = [] 
            match_odds_input = [filter_vec[j:j+3] for j in range(0, len(filter_vec), 3)]
            
            for tr in all_possible_rows:
                pts = 0
                c1, cx, c2 = tr.count('1'), tr.count('X'), tr.count('2')
                s1_c, sx_c, s2_c, _ = get_streaks(tr)
                g1_c, gx_c, g2_c, _ = get_gaps(tr)
                si1_c, six_c, si2_c, singtot_c, _ = get_singles(tr)
                d1_c, dx_c, d2_c, dubtot_c, _ = get_doublets(tr)
                t1_c, tx_c, t2_c, triptot_c, _ = get_triplets(tr)
                o1_c, ox_c, o2_c, occtot_c, _ = get_occurrences(tr)
                f_c, a_c, t_c, fsum_c = get_fat(tr, filter_vec)
                
                if cb_base and (c_ones[0] <= c1 <= c_ones[1] and c_draws[0] <= cx <= c_draws[1] and c_twos[0] <= c2 <= c_twos[1]): pts += 1
                if cb_streak and (c_s1[0] <= s1_c <= c_s1[1] and c_sx[0] <= sx_c <= c_sx[1] and c_s2[0] <= s2_c <= c_s2[1]): pts += 1
                if cb_gap and (c_g1[0] <= g1_c <= c_g1[1] and c_gx[0] <= gx_c <= c_gx[1] and c_g2[0] <= g2_c <= c_g2[1]): pts += 1
                if cb_single and (c_sing1[0] <= si1_c <= c_sing1[1] and c_singx[0] <= six_c <= c_singx[1] and c_sing2[0] <= si2_c <= c_sing2[1] and c_singtot[0] <= singtot_c <= c_singtot[1]): pts += 1
                if cb_doublet and (c_dub1[0] <= d1_c <= c_dub1[1] and c_dubx[0] <= dx_c <= c_dubx[1] and c_dub2[0] <= d2_c <= c_dub2[1] and c_dubtot[0] <= dubtot_c <= c_dubtot[1]): pts += 1
                if cb_triplet and (c_trip1[0] <= t1_c <= c_trip1[1] and c_tripx[0] <= tx_c <= c_tripx[1] and c_trip2[0] <= t2_c <= c_trip2[1] and c_triptot[0] <= triptot_c <= c_triptot[1]): pts += 1
                if cb_occur and (c_occ1[0] <= o1_c <= c_occ1[1] and c_occx[0] <= ox_c <= c_occx[1] and c_occ2[0] <= o2_c <= c_occ2[1] and c_occtot[0] <= occtot_c <= c_occtot[1]): pts += 1
                
                if cb_super_macro:
                    g_pass = 0
                    b = sm_bounds
                    if sum([b['1'][0] <= c1 <= b['1'][1], b['X'][0] <= cx <= b['X'][1], b['2'][0] <= c2 <= b['2'][1]]) >= 2: g_pass += 1
                    if sum([b['s1'][0] <= s1_c <= b['s1'][1], b['sx'][0] <= sx_c <= b['sx'][1], b['s2'][0] <= s2_c <= b['s2'][1]]) >= 2: g_pass += 1
                    if sum([b['g1'][0] <= g1_c <= b['g1'][1], b['gx'][0] <= gx_c <= b['gx'][1], b['g2'][0] <= g2_c <= b['g2'][1]]) >= 2: g_pass += 1
                    if sum([b['si1'][0] <= si1_c <= b['si1'][1], b['six'][0] <= six_c <= b['six'][1], b['si2'][0] <= si2_c <= b['si2'][1]]) >= 2: g_pass += 1
                    if sum([b['d1'][0] <= d1_c <= b['d1'][1], b['dx'][0] <= dx_c <= b['dx'][1], b['d2'][0] <= d2_c <= b['d2'][1]]) >= 2: g_pass += 1
                    if sum([b['t1'][0] <= t1_c <= b['t1'][1], b['tx'][0] <= tx_c <= b['tx'][1], b['t2'][0] <= t2_c <= b['t2'][1]]) >= 2: g_pass += 1
                    if sum([b['o1'][0] <= o1_c <= b['o1'][1], b['ox'][0] <= ox_c <= b['ox'][1], b['o2'][0] <= o2_c <= b['o2'][1]]) >= 2: g_pass += 1
                    if sum([b['f'][0] <= f_c <= b['f'][1], b['a'][0] <= a_c <= b['a'][1], b['t'][0] <= t_c <= b['t'][1]]) >= 2: g_pass += 1
                    if g_pass >= slider_super_groups: pts += 1

                if cb_fat and (c_fatf[0] <= f_c <= c_fatf[1] and c_fata[0] <= a_c <= c_fata[1] and c_fatt[0] <= t_c <= c_fatt[1] and c_fatsum[0] <= fsum_c <= c_fatsum[1]): pts += 1
                if cb_u_favs and (c_u[0] <= get_top_n_favs_wins(tr, filter_vec, slider_u_count) <= c_u[1]): pts += 1
                if cb_sft and (c_sft[0] <= get_sft_sum(tr, filter_vec) <= c_sft[1]): pts += 1
                if cb_points and (c_points[0] <= get_rank_points(tr, filter_vec) <= c_points[1]): pts += 1
                if cb_100minus and (c_minus[0] <= get_100_minus_sum(tr, filter_vec) <= c_minus[1]): pts += 1
                if cb_log_surprise and in_range(get_log_surprise_sum(tr, filter_vec), c_log_surprise): pts += 1
                if cb_rank24 and (c_rank24[0] <= get_rank_sum(tr, filter_vec) <= c_rank24[1]): pts += 1
                if cb_totaldiff:
                    td_c = calculate_total_diff(match_odds_input, list(tr))
                    if (c_totaldiff[0] <= td_c <= c_totaldiff[1]): pts += 1
                if cb_fav_pressure:
                    fp_c = get_favorite_pressure(tr, filter_vec)
                    if in_range(fp_c['F70_Wins'], c_fav70) and in_range(fp_c['F60_Wins'], c_fav60) and in_range(fp_c['F50_Wins'], c_fav50): pts += 1
                if cb_shock_strength:
                    sh_c = get_shock_strength(tr, filter_vec)
                    if in_range(sh_c['U10_Wins'], c_shock10) and in_range(sh_c['U15_Wins'], c_shock15) and in_range(sh_c['U20_Wins'], c_shock20) and in_range(sh_c['Lowest_Win_Pct'], c_shock_lowest): pts += 1
                if cb_fav_delta and in_range(get_favorite_delta(tr, filter_vec), c_fav_delta): pts += 1
                if cb_aimatrix:
                    rank_c, _ = get_exact_rank(tr, ai_matrix, ai_scores_asc, ai_tot)
                    if (active_ai_min <= rank_c <= active_ai_max): pts += 1
                
                if pts >= slider_pass_req: 
                    valid_exact_rows.append(tr)
            
            sim_hits = len(valid_exact_rows)
            st.success(f"🎯 **EXAKT KVARVARANDE RADANTAL:** {sim_hits} st (Skurit bort {100 - ((sim_hits / 6561) * 100):.2f}%) med Soft Filtering.")
            
            if sim_hits > 0:
                st.markdown("### 📊 Frekvenstabell (Mallen)")
                freq_data = []
                for m in range(antal_matcher):
                    c1 = sum(1 for r in valid_exact_rows if r[m] == '1')
                    cx = sum(1 for r in valid_exact_rows if r[m] == 'X')
                    c2 = sum(1 for r in valid_exact_rows if r[m] == '2')
                    freq_data.append({
                        "Match": f"M{m+1}",
                        "1 (%)": f"{(c1/sim_hits)*100:.1f} %",
                        "X (%)": f"{(cx/sim_hits)*100:.1f} %",
                        "2 (%)": f"{(c2/sim_hits)*100:.1f} %"
                    })
                st.dataframe(pd.DataFrame(freq_data).set_index("Match"), use_container_width=True)

        else:
            st.info("💡 Exakt uträkning är avstängd för 13 matcher (för snabbhetens skull). Mallen är dock helt redo för reducering!")

        st.markdown("---")
        st.subheader("📊 Datadistribution")
        fig = plt.figure(figsize=(18, 16)) 
        def smart_plot(data_list, col_idx, color, title, xlabel, is_active, val_min, val_max, meta_key=None):
            plt.subplot(3, 3, col_idx) 
            valid_data = [d for d in data_list if not pd.isna(d)]
            if not valid_data: plt.text(0.5, 0.5, 'Ingen data', ha='center', va='center'); plt.title(title); return
            d_min, d_max = min(valid_data), max(valid_data)
            d_range = d_max - d_min
            bins = np.arange(np.floor(d_min)-0.5, np.ceil(d_max)+1.5, 1) if d_range <= 40 else int(d_range) if d_range <= 150 else 25
            plt.hist(valid_data, bins=bins, color=color, edgecolor='black', alpha=0.8)
            plt.title(title, fontweight='bold'); plt.xlabel(xlabel); plt.ylabel('Antal')
            
            if d_range == 0: ticks = [d_min]
            elif d_range <= 20: ticks = np.arange(np.floor(d_min), np.ceil(d_max) + 1, 1)
            elif d_range <= 60: ticks = np.arange(np.floor(d_min), np.ceil(d_max) + 2, 2)
            elif d_range <= 150: ticks = np.arange(np.floor(d_min), np.ceil(d_max) + 5, 5)
            elif d_range <= 400: ticks = np.arange(np.floor(d_min), np.ceil(d_max) + 10, 10)
            else: ticks = np.linspace(d_min, d_max, 10).astype(int)
            
            plt.xticks(ticks, rotation=45); plt.grid(axis='y', linestyle='--', alpha=0.5)
            if is_active:
                if meta_key and meta_key in autotrim_meta:
                    safe = autotrim_meta[meta_key].get('Säkerhetsintervall')
                    if safe is not None and tuple(safe) != (val_min, val_max):
                        plt.axvline(safe[0], color='orange', linestyle='dotted', linewidth=1.5, label='Säkerhet Min')
                        plt.axvline(safe[1], color='orange', linestyle='dotted', linewidth=1.5, label='Säkerhet Max')
                plt.axvline(val_min, color='red', linestyle='dashed', linewidth=2, label='Rek Min')
                plt.axvline(val_max, color='darkred', linestyle='dashed', linewidth=2, label='Rek Max')
                plt.legend()

        smart_plot([r for r in ai_ranks if r > 0], 1, 'skyblue', 'AI-Rank', 'AI-Rank', cb_aimatrix, active_ai_min, active_ai_max)
        smart_plot(sft_sums, 2, 'coral', 'SFT Summa', 'SFT Summa', cb_sft, c_sft[0], c_sft[1], 'SFT Summa')
        smart_plot(fat_sums, 3, 'gold', 'FAT Summa', 'FAT Summa', cb_fat, c_fatsum[0], c_fatsum[1]) 
        smart_plot(points_vals, 4, 'mediumpurple', 'Poängfilter', 'Poäng', cb_points, c_points[0], c_points[1], 'Poängfilter')
        smart_plot(minus_sums, 5, 'tan', '100-minus Summa', '100-minus', cb_100minus, c_minus[0], c_minus[1], '100-minus Summa')
        smart_plot(rank24_sums, 6, 'lightpink', 'Rank Summa', 'Rank Summa', cb_rank24, c_rank24[0], c_rank24[1], 'Rank Summa')
        smart_plot(total_diff_vals, 7, 'lightgreen', 'Total Diff (T1-T2)', 'Differens', cb_totaldiff, c_totaldiff[0], c_totaldiff[1], 'Total Diff')
        smart_plot(list(v_m['Delta']), 8, 'lightblue', 'Delta (Avvikelse)', 'Delta Poäng', True, c_delta[0], c_delta[1], 'Delta')
        
        plt.tight_layout(pad=2.0, h_pad=2.0)
        
        st.pyplot(fig)
        plt.close(fig)
