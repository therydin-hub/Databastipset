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
APP_VERSION = "v10.0 – AI-Ram & U-filter"

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
    n = 1
    for signs in frame:
        n *= max(1, len(signs))
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

# --- SIDEBAR (INSTÄLLNINGAR) ---
with st.sidebar:
    st.header("⚙️ Inställningar")
    
    if st.button("🧹 Töm Minne / Rensa Cache"):
        st.cache_data.clear()
        st.success("Cachen tömd! Tryck F5 för att ladda om.")

    st.subheader("Matchning & Kärna")
    slider_top_n = st.slider("Antal historiska matcher att hämta", 5, 100, 30, step=5)
    slider_core_val = st.slider("Kärna % (Värde & Svårighet)", 40, 100, 90, step=5)
    slider_core_str = st.slider("Kärna % (Struktur & Tecken)", 40, 100, 100, step=5)
    
    st.subheader("Makro & Super-Makro")
    slider_macro_target = st.slider("Målsättning Procent %", 40, 100, 90, step=5)
    slider_super_groups = st.slider("Super-Makro: Minst antal grupper (av 8)", 1, 8, 4, step=1)
    
    st.subheader("Avancerade Filter")
    slider_u_count = st.slider("Antal Topp-Favoriter (U-tecken)", 1, antal_matcher, min(3, antal_matcher), step=1)
    p_opts = [0, 500, 1000, 2500, 5000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000, 1000000, 2500000, 5000000, 10000000]
    slider_payout = st.select_slider("Utdelnings-krav (kr)", options=p_opts, value=(0, p_opts[-1]))
    
    st.subheader("Mallen - Aktiva Filter")
    cb_payout = st.checkbox("Utdelning", value=True)
    cb_u_favs = st.checkbox("Topp-Favoriter (U-tecken)", value=True)
    cb_sft = st.checkbox("SFT Summa", value=True)
    cb_fat = st.checkbox("FAT-Tabell & Summa (Standard)", value=True)
    cb_points = st.checkbox("POÄNGFILTER (Eget)", value=True)
    cb_100minus = st.checkbox("100-minus Summa", value=True)
    cb_log_surprise = st.checkbox("Skrälltryck Log Summa", value=True)
    cb_rank24 = st.checkbox(f"Rank 1-{krav_odds} Summa", value=True)
    cb_totaldiff = st.checkbox("Total Diff (T1 - T2)", value=True)
    cb_fav_pressure = st.checkbox("Favorittryck (70/60/50%)", value=True)
    cb_shock_strength = st.checkbox("Skrällstyrka (<10/<15/<20%)", value=True)
    cb_fav_delta = st.checkbox("Favorit-delta", value=True)
    
    st.markdown("**Struktur (Standard):**")
    cb_base = st.checkbox("Grundfilter (1, X, 2)", value=True)
    cb_streak = st.checkbox("Sviter", value=True)
    cb_gap = st.checkbox("Luckor", value=True)
    cb_single = st.checkbox("Singlar", value=True)
    cb_doublet = st.checkbox("Dubbletter", value=True)
    cb_triplet = st.checkbox("Tripplar", value=True)
    cb_occur = st.checkbox("Uppkomster", value=True)
    
    st.markdown("**Super-Makro:**")
    cb_super_macro = st.checkbox(f"Super-Makro (Minst {slider_super_groups} av 8 Grupper)", value=True)
    
    cb_aimatrix = st.checkbox("AI-Matrix Rank", value=True)
    cb_manual_ai_rank = st.checkbox("Styr AI-Rank manuellt", value=False)
    max_rank = 3**antal_matcher
    if cb_manual_ai_rank:
        slider_ai_rank = st.slider("AI-Rank Slider", 1, max_rank, (1, max_rank))
    
    cb_structure = st.checkbox("Matcha Struktur (Viktad)", value=True)

    st.markdown("---")
    st.subheader("🧩 Pro-grupper")
    cb_pro_groups = st.checkbox("Visa pro-grupper", value=True)
    slider_group_target = st.slider("Gruppmål historisk träff %", 70, 100, 90, step=5)
    cb_group_risk = st.checkbox("Riskgrupp", value=True)
    cb_group_shock = st.checkbox("Skrällgrupp", value=True)
    cb_group_fat = st.checkbox("FAT-profilgrupp", value=True)
    cb_group_structure = st.checkbox("Strukturgrupp", value=True)

    st.markdown("---")
    st.subheader("🎯 AI-Ram & U-filter")
    cb_ai_frame = st.checkbox("Visa AI-Ram & U-filter", value=True)
    slider_frame_target = st.slider("Ram-mål: minst 12 rätt inom ram %", 70, 100, 95, step=5)
    slider_u_target = st.slider("U-rad mål historisk träff %", 70, 100, 90, step=5)

    st.markdown("---")
    st.subheader("🎯 Soft Filtering")
    # Uppdaterad lista för poängberäkningen
    active_filters_list = [
        cb_u_favs, cb_sft, cb_fat, cb_points, cb_100minus, cb_log_surprise, cb_rank24, cb_totaldiff,
        cb_fav_pressure, cb_shock_strength, cb_fav_delta,
        cb_base, cb_streak, cb_gap, cb_single, cb_doublet, cb_triplet, cb_occur,
        cb_super_macro, cb_aimatrix
    ]
    total_active = sum(active_filters_list)
    slider_pass_req = st.slider("Minsta antal uppfyllda krav", 1, total_active, total_active) if total_active > 0 else 0


# --- MAIN AREA FÖR INMATNING ---
input_text = st.text_area(f"Klistra in VÄRDEN ({krav_odds} st oddsprocent):", height=120)

if 'har_kort_analys' not in st.session_state:
    st.session_state['har_kort_analys'] = False

if st.button("🚀 KÖR ANALYS", use_container_width=True):
    if input_text:
        st.session_state['har_kort_analys'] = True
    else:
        st.error("⚠️ Klistra in oddsen först!")

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
        
        st.subheader(f"📋 Historiska Omgångar ({len(v_m)} st)")
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
        
        c_sft = get_best_interval(sft_sums, c_v)
        c_fatf = get_best_interval(fat_f, c_v); c_fata = get_best_interval(fat_a, c_v); c_fatt = get_best_interval(fat_t, c_v); c_fatsum = get_best_interval(fat_sums, c_v)
        c_points = get_best_interval(points_vals, c_v)
        c_minus = get_best_interval(minus_sums, c_v)
        c_log_surprise = get_best_interval(log_surprise_sums, c_v)
        c_rank24 = get_best_interval(rank24_sums, c_v)
        c_totaldiff = get_best_interval(total_diff_vals, c_v)
        c_u = get_best_interval(u_wins, c_v)
        c_delta = get_best_interval(list(v_m['Delta']), c_v)

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
        c_fav_delta = get_best_interval(fav_delta_vals, c_v)
        
        c_ai_rank = get_best_interval(ai_ranks, c_v) if len(ai_ranks) > 0 else (1, max_rank)
        active_ai_min, active_ai_max = slider_ai_rank if cb_manual_ai_rank else c_ai_rank
        ai_txt = "AI-Rank (MANUELL)" if cb_manual_ai_rank else f"AI-Rank (AUTO {c_v}%)"

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
        
        col_v, col_s = st.columns(2)
        with col_v:
            st.subheader(f"💰 VÄRDE & SVÅRIGHET ({c_v}%)")
            if cb_payout: st.write(f"**Utdelning:** {v_m['Payout'].min():.0f} - {v_m['Payout'].max():.0f} kr")
            st.write(f"**Delta (Avvikelse):** {c_delta[0]:.1f} - {c_delta[1]:.1f}")
            if cb_aimatrix: st.write(f"**{ai_txt}:** {active_ai_min:.0f} - {active_ai_max:.0f}")
            if cb_totaldiff: st.write(f"**Total Diff:** {c_totaldiff[0]} - {c_totaldiff[1]}")
            if cb_rank24: st.write(f"**Rank Summa:** {c_rank24[0]:.1f} - {c_rank24[1]:.1f}")
            if cb_100minus: st.write(f"**100-minus Summa:** {c_minus[0]} - {c_minus[1]}")
            if cb_log_surprise: st.write(f"**Skrälltryck Log Summa:** {c_log_surprise[0]} - {c_log_surprise[1]}")
            if cb_sft: st.write(f"**SFT Summa:** {c_sft[0]} - {c_sft[1]}")
            if cb_points: st.write(f"**Poängfilter:** {c_points[0]} - {c_points[1]}")
            if cb_fat: st.write(f"**FAT (Standard):** F:{c_fatf[0]}-{c_fatf[1]} | A:{c_fata[0]}-{c_fata[1]} | T:{c_fatt[0]}-{c_fatt[1]} (Summa: {c_fatsum[0]}-{c_fatsum[1]})")
            if cb_u_favs: st.write(f"**Topp {slider_u_count} Favoriter:** {c_u[0]} - {c_u[1]} st vinner")
            if cb_fav_pressure: st.write(f"**Favorittryck:** ≥70% {fmt_interval(c_fav70)} av {todays_fav_counts.get(70,0)} | ≥60% {fmt_interval(c_fav60)} av {todays_fav_counts.get(60,0)} | ≥50% {fmt_interval(c_fav50)} av {todays_fav_counts.get(50,0)}")
            if cb_shock_strength: st.write(f"**Skrällstyrka:** <10% {fmt_interval(c_shock10)} | <15% {fmt_interval(c_shock15)} | <20% {fmt_interval(c_shock20)} | Lägsta vinnande % {fmt_interval(c_shock_lowest, 1)}")
            if cb_fav_delta: st.write(f"**Favorit-delta:** {fmt_interval(c_fav_delta, 2)}")

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
            st.markdown(f"*Minst 2 av 3 interna tecken i en grupp måste sitta för att gruppen ska räknas som 'träffad'. Överlever historiskt **{sm_prob:.1f}%**.*")
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

        hard_all_pct = (hard_all_hits / len(v_m) * 100) if len(v_m) else 0.0
        soft_hit_pct = (mall_hits / len(v_m) * 100) if len(v_m) else 0.0
        st.info(f"📈 **SAMMANLAGD HISTORISK TRÄFF:** Softfilter {mall_hits} av {len(v_m)} rader ({soft_hit_pct:.1f}%) klarade minst {slider_pass_req} av {total_active} aktiva filter.")
        if total_active > 0:
            st.caption(f"Om samma filter hade körts som helt hårda AND-filter hade {hard_all_hits} av {len(v_m)} historiska rader klarat alla {total_active} filter ({hard_all_pct:.1f}%).")

        # --- FILTERSTYRKA, FILTERREGLER OCH HELGARDERING-EXPORT ---
        candidate_rows, exact_universe = make_candidate_rows(antal_matcher)
        total_candidates = len(candidate_rows)
        match_odds_filter = [filter_vec[j:j+3] for j in range(0, len(filter_vec), 3)]

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

        combined_soft_survivors = combined_hard_survivors = 0
        combined_score_counts = {}
        if total_active > 0 and total_candidates > 0:
            for tr in candidate_rows:
                pts_c = score_candidate_row(tr)
                combined_score_counts[pts_c] = combined_score_counts.get(pts_c, 0) + 1
                if pts_c == total_active:
                    combined_hard_survivors += 1
                if pts_c >= slider_pass_req:
                    combined_soft_survivors += 1
        combined_hard_keep_pct = (combined_hard_survivors / total_candidates * 100) if total_candidates else 0.0
        combined_soft_keep_pct = (combined_soft_survivors / total_candidates * 100) if total_candidates else 0.0
        combined_est_label = "exakt" if exact_universe else "estimat"

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

            st.markdown("**Sammanlagd mallträff**")
            csum1, csum2, csum3, csum4 = st.columns(4)
            with csum1:
                st.metric("Historisk soft-träff", f"{soft_hit_pct:.1f}%", f"{mall_hits}/{len(v_m)}")
            with csum2:
                st.metric("Historisk hård träff", f"{hard_all_pct:.1f}%", f"{hard_all_hits}/{len(v_m)}")
            with csum3:
                st.metric(f"Kvar rader soft ({combined_est_label})", f"{combined_soft_keep_pct:.1f}%", f"{combined_soft_survivors}/{total_candidates}")
            with csum4:
                st.metric(f"Kvar rader hårt ({combined_est_label})", f"{combined_hard_keep_pct:.1f}%", f"{combined_hard_survivors}/{total_candidates}")

            st.caption(
                "Individuella filter på 90-95 % kan tillsammans få mycket lägre träff om de körs som hårda AND-filter. "
                "Softfiltret räknar i stället hur många filter raden klarar, och är därför huvudvärdet att styra på."
            )

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
                f"Historiska omgångar: {len(v_m)} | Softfilterkrav: {slider_pass_req} av {total_active}",
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
                st.download_button("⬇️ Ladda ner filterregler CSV", filter_rules_df.to_csv(index=False, sep=';').encode('utf-8-sig'), file_name=f"filterregler_{spelform.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
            with col_dl2:
                st.download_button("⬇️ Ladda ner Helgardering TXT", helg_text.encode('utf-8'), file_name=f"helgardering_export_{spelform.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", mime="text/plain")
            with col_dl3:
                if st.button("💾 Spara analys lokalt"):
                    outdir = Path("analysis_exports")
                    outdir.mkdir(exist_ok=True)
                    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filter_rules_df.to_csv(outdir / f"filterregler_{spelform}_{stamp}.csv", index=False, sep=';', encoding='utf-8-sig')
                    (outdir / f"helgardering_export_{spelform}_{stamp}.txt").write_text(helg_text, encoding='utf-8')
                    st.success(f"Sparat i {outdir.resolve()}")

        # --- AI-RAM & U-FILTER ---
        if cb_ai_frame:
            st.markdown("---")
            st.subheader("🎯 AI-Ram & U-filter")
            st.caption(
                "Rammotorn använder de liknande historiska omgångarna för att föreslå spikar/halvor/hela. "
                "Den backtestar ramen mot historiken och visar hur många av vinnarraderna som ryms inom ramen."
            )

            hist_rows_for_frame = [str(r).strip().upper() for r in v_m['Correct_Row'].astype(str) if len(str(r).strip()) == antal_matcher]
            match_ai_stats = build_match_ai_stats(v_m, filter_vec, antal_matcher)

            frame_targets = [
                ("Aggressiv", max(70, slider_frame_target - 10)),
                ("Balans", slider_frame_target),
                ("Trygg", min(100, slider_frame_target + 5)),
            ]
            frame_results = []
            frame_detail_texts = []
            for label, target in frame_targets:
                frame, ev = optimize_frame_greedy(
                    hist_rows_for_frame,
                    match_ai_stats,
                    antal_matcher,
                    target_pct=target,
                    metric='12+'
                )
                frame_txt = frame_to_string(frame)
                frame_results.append({
                    "Ram": label,
                    "Mål 12+ %": target,
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
                    f"Radantal: {ev['Radantal']} | Spikar {ev['Spikar']} | Halvor {ev['Halvor']} | Hela {ev['Hela']}\n"
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
            st.download_button(
                "⬇️ Ladda ner AI-Ram & U-filter TXT",
                ai_frame_export.encode('utf-8'),
                file_name=f"ai_ram_u_filter_{spelform.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

        st.markdown("---")
        st.subheader("🧬 Dagens Bästa FAT-Sekvenser (Byggklossar)")
        st.markdown("Här analyserar AI:n vilka specifika mönster (1=Fav, 2=Andrahand, 3=Skräll) som bäst täcker in **exakt denna typ av omgång**. Statistiken visar hur många av de 5 sekvenserna som brukar dyka upp i en och samma vinnarrad.")

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
        def smart_plot(data_list, col_idx, color, title, xlabel, is_active, val_min, val_max):
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
                plt.axvline(val_min, color='red', linestyle='dashed', linewidth=2, label='Mallen Min')
                plt.axvline(val_max, color='darkred', linestyle='dashed', linewidth=2, label='Mallen Max')
                plt.legend()

        smart_plot([r for r in ai_ranks if r > 0], 1, 'skyblue', 'AI-Rank', 'AI-Rank', cb_aimatrix, active_ai_min, active_ai_max)
        smart_plot(sft_sums, 2, 'coral', 'SFT Summa', 'SFT Summa', cb_sft, c_sft[0], c_sft[1])
        smart_plot(fat_sums, 3, 'gold', 'FAT Summa', 'FAT Summa', cb_fat, c_fatsum[0], c_fatsum[1]) 
        smart_plot(points_vals, 4, 'mediumpurple', 'Poängfilter', 'Poäng', cb_points, c_points[0], c_points[1])
        smart_plot(minus_sums, 5, 'tan', '100-minus Summa', '100-minus', cb_100minus, c_minus[0], c_minus[1])
        smart_plot(rank24_sums, 6, 'lightpink', 'Rank Summa', 'Rank Summa', cb_rank24, c_rank24[0], c_rank24[1])
        smart_plot(total_diff_vals, 7, 'lightgreen', 'Total Diff (T1-T2)', 'Differens', cb_totaldiff, c_totaldiff[0], c_totaldiff[1])
        smart_plot(list(v_m['Delta']), 8, 'lightblue', 'Delta (Avvikelse)', 'Delta Poäng', False, c_delta[0], c_delta[1])
        
        plt.tight_layout(pad=2.0, h_pad=2.0)
        
        st.pyplot(fig)
        plt.close(fig)
