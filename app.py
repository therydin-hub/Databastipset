import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import itertools
import bisect
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tipset AI-Analys", layout="wide", page_icon="🎯")

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
    return [float(p) for p in re.sub(r'[^\d\.\s]', '', text).split()[:max_vals]]

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


# ==========================================
# 2. AUTO-LADDNING & DATABAS
# ==========================================

def find_local_database(spelform):
    mapp = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
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
            global_db = pd.read_csv(filepath, sep=sep, encoding='utf-8', on_bad_lines='skip')
        except:
            with open(filepath, 'r', encoding='latin-1') as f: first_line = f.readline()
            sep = ';' if ';' in first_line else ','
            global_db = pd.read_csv(filepath, sep=sep, encoding='latin-1', on_bad_lines='skip')

    col_m = [f'M{i}' for i in range(1, antal_matcher + 1)]
    if all(c in global_db.columns for c in col_m):
         global_db['Correct_Row'] = global_db[col_m].astype(str).agg(''.join, axis=1)

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
    target_payout = f"{antal_matcher} rätt".lower()
    payout_col = next((col for col in global_db.columns if str(col).lower() == target_payout), None)
    
    if payout_col: global_db['Payout'] = pd.to_numeric(global_db[payout_col].astype(str).str.replace(' ', '').str.replace(',', '.'), errors='coerce').fillna(0)
    else: global_db['Payout'] = 0
        
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
    
    input_compare = get_structural_vector(input_vec) if cb_structure else input_vec
    weights_arr = np.array([w for i in range(0, krav_odds, 3) for w in [(max(input_compare[i:i+3])/100.0)**2]*3])

    df_s = global_db.copy()
    df_s['Sim'] = [weighted_euclidean(input_compare, get_structural_vector(r['Prob_Vector']) if cb_structure else r['Prob_Vector'], weights_arr) if len(r['Prob_Vector'])==krav_odds else 9999 for _, r in df_s.iterrows()]
    
    v_m = df_s.sort_values('Sim').head(slider_top_n)
    
    if cb_payout: 
        v_m = v_m[(v_m['Payout'] >= pay_min) & (v_m['Payout'] <= pay_max)]
        
    return v_m, input_vec, os.path.basename(fil_sökväg), None


# ==========================================
# STREAMLIT UI & SIDEBAR
# ==========================================

col_header1, col_header2 = st.columns([2, 1])
with col_header1:
    st.title("🎯 Tipset AI-Analys")
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
    
    st.subheader("Matchning & Kärna")
    slider_top_n = st.slider("Antal historiska matcher att hämta", 5, 100, 30, step=5)
    slider_core_val = st.slider("Kärna % (Värde & Svårighet)", 40, 100, 90, step=5)
    slider_core_str = st.slider("Kärna % (Struktur & Tecken)", 40, 100, 100, step=5)
    slider_u_count = st.slider("Antal Topp-Favoriter (U-tecken)", 1, antal_matcher, min(3, antal_matcher), step=1)
    
    st.subheader("Avancerade Filter")
    p_opts = [0, 500, 1000, 2500, 5000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000, 1000000, 2500000, 5000000, 10000000]
    slider_payout = st.select_slider("Utdelnings-krav (kr)", options=p_opts, value=(0, p_opts[-1]))
    
    st.subheader("Mallen - Aktiva Filter")
    cb_payout = st.checkbox("Utdelning", value=True)
    cb_u_favs = st.checkbox("Topp-Favoriter (U-tecken)", value=True)
    cb_sft = st.checkbox("SFT Summa", value=True)
    cb_fat = st.checkbox("FAT-Tabell & Summa", value=True)
    cb_points = st.checkbox("POÄNGFILTER (Eget)", value=True)
    cb_100minus = st.checkbox("100-minus Summa", value=True)
    cb_rank24 = st.checkbox(f"Rank 1-{krav_odds} Summa", value=True)
    cb_totaldiff = st.checkbox("Total Diff (T1 - T2)", value=True)
    
    cb_base = st.checkbox("Grundfilter (1, X, 2)", value=True)
    cb_streak = st.checkbox("Sviter", value=True)
    cb_gap = st.checkbox("Luckor", value=True)
    cb_single = st.checkbox("Singlar", value=True)
    cb_doublet = st.checkbox("Dubbletter", value=True)
    cb_triplet = st.checkbox("Tripplar", value=True)
    cb_occur = st.checkbox("Uppkomster", value=True)
    
    cb_aimatrix = st.checkbox("AI-Matrix Rank", value=True)
    cb_manual_ai_rank = st.checkbox("Styr AI-Rank manuellt", value=False)
    max_rank = 3**antal_matcher
    if cb_manual_ai_rank:
        slider_ai_rank = st.slider("AI-Rank Slider", 1, max_rank, (1, max_rank))
    
    cb_structure = st.checkbox("Matcha Struktur (Viktad)", value=True)

    st.markdown("---")
    st.subheader("🎯 Soft Filtering")
    active_filters_list = [cb_u_favs, cb_sft, cb_fat, cb_points, cb_100minus, cb_rank24, cb_totaldiff, cb_base, cb_streak, cb_gap, cb_single, cb_doublet, cb_triplet, cb_occur, cb_aimatrix]
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
        input_compare = get_structural_vector(input_vec) if cb_structure else input_vec
        
        visnings_kolumner = [c for c in ['Datum', 'ID_Omg'] if c in v_m.columns] + ['Payout', 'Sim']
        st.success(f"✅ Auto-laddade: **{filnamn}**. Exakt {len(v_m)} liknande omgångar hittades.")
        
        st.subheader(f"📋 Historiska Omgångar ({len(v_m)} st)")
        st.dataframe(v_m[visnings_kolumner].rename(columns={'Payout':f'Utdelning ({antal_matcher}r)', 'Sim':'Likhet'}).style.format({f'Utdelning ({antal_matcher}r)': '{:.0f} kr', 'Likhet': '{:.2f}'}), use_container_width=True)

        ones, draws, twos = [], [], []
        s1, sx, s2, g1, gx, g2 = [], [], [], [], [], []
        sing1, singx, sing2, sing_tot = [], [], [], []
        dub1, dubx, dub2, dub_tot = [], [], [], []
        trip1, tripx, trip2, trip_tot = [], [], [], []
        occ1, occx, occ2, occ_tot = [], [], [], []
        sft_sums, fat_f, fat_a, fat_t, fat_sums = [], [], [], [], []
        points_vals, minus_sums, rank24_sums, total_diff_vals, u_wins, ai_ranks = [], [], [], [], [], []

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
            rank24_sums.append(get_rank_sum(r, p))
            u_wins.append(get_top_n_favs_wins(r, p, slider_u_count))
            
            match_odds_list = [p[j:j+3] for j in range(0, len(p), 3)]
            total_diff_vals.append(calculate_total_diff(match_odds_list, list(r)))

        c_v, c_s = slider_core_val, slider_core_str
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
        c_rank24 = get_best_interval(rank24_sums, c_v)
        c_totaldiff = get_best_interval(total_diff_vals, c_v)
        c_u = get_best_interval(u_wins, c_v)
        
        c_ai_rank = get_best_interval(ai_ranks, c_v) if len(ai_ranks) > 0 else (1, max_rank)
        active_ai_min, active_ai_max = slider_ai_rank if cb_manual_ai_rank else c_ai_rank
        ai_txt = "AI-Rank (MANUELL)" if cb_manual_ai_rank else f"AI-Rank (AUTO {c_v}%)"

        # --- Dynamisk sökning för Struktur Grupper (1X2) - Minst 90% träff ---
        t_ones, t_draws, t_twos = (0,0), (0,0), (0,0)
        macro_prob = 0
        for cov in range(30, 101, 2): 
            t_ones = get_best_interval(ones, cov)
            t_draws = get_best_interval(draws, cov)
            t_twos = get_best_interval(twos, cov)
            
            macro_hits_count = 0
            for i in range(len(v_m)):
                req1 = 1 if (t_ones[0] <= ones[i] <= t_ones[1]) else 0
                reqx = 1 if (t_draws[0] <= draws[i] <= t_draws[1]) else 0
                req2 = 1 if (t_twos[0] <= twos[i] <= t_twos[1]) else 0
                if (req1 + reqx + req2) >= 2: macro_hits_count += 1
                    
            macro_prob = (macro_hits_count / len(v_m)) * 100 if len(v_m) > 0 else 0
            if macro_prob >= 90.0: break

        st.markdown("---")
        st.header(f"📋 VECKANS MALL ({spelform})")
        
        col_v, col_s = st.columns(2)
        with col_v:
            st.subheader(f"💰 VÄRDE & SVÅRIGHET ({c_v}%)")
            if cb_payout: st.write(f"**Utdelning:** {v_m['Payout'].min():.0f} - {v_m['Payout'].max():.0f} kr")
            if cb_aimatrix: st.write(f"**{ai_txt}:** {active_ai_min:.0f} - {active_ai_max:.0f}")
            if cb_totaldiff: st.write(f"**Total Diff:** {c_totaldiff[0]} - {c_totaldiff[1]}")
            if cb_rank24: st.write(f"**Rank Summa:** {c_rank24[0]:.1f} - {c_rank24[1]:.1f}")
            if cb_100minus: st.write(f"**100-minus Summa:** {c_minus[0]} - {c_minus[1]}")
            if cb_sft: st.write(f"**SFT Summa:** {c_sft[0]} - {c_sft[1]}")
            if cb_points: st.write(f"**Poängfilter:** {c_points[0]} - {c_points[1]}")
            if cb_fat: st.write(f"**FAT:** F:{c_fatf[0]}-{c_fatf[1]} | A:{c_fata[0]}-{c_fata[1]} | T:{c_fatt[0]}-{c_fatt[1]} (Summa: {c_fatsum[0]}-{c_fatsum[1]})")
            if cb_u_favs: st.write(f"**Topp {slider_u_count} Favoriter:** {c_u[0]} - {c_u[1]} st vinner")

        with col_s:
            st.subheader(f"⚽ STRUKTUR ({c_s}%)")
            if cb_streak: st.write(f"**Sviter:** 1: {c_s1[0]}-{c_s1[1]} | X: {c_sx[0]}-{c_sx[1]} | 2: {c_s2[0]}-{c_s2[1]}")
            if cb_gap: st.write(f"**Luckor:** 1: {c_g1[0]}-{c_g1[1]} | X: {c_gx[0]}-{c_gx[1]} | 2: {c_g2[0]}-{c_g2[1]}")
            if cb_single: st.write(f"**Singlar:** 1: {c_sing1[0]}-{c_sing1[1]} | X: {c_singx[0]}-{c_singx[1]} | 2: {c_sing2[0]}-{c_sing2[1]} | Tot: {c_singtot[0]}-{c_singtot[1]}")
            if cb_doublet: st.write(f"**Dubbletter:** 1: {c_dub1[0]}-{c_dub1[1]} | X: {c_dubx[0]}-{c_dubx[1]} | 2: {c_dub2[0]}-{c_dub2[1]} | Tot: {c_dubtot[0]}-{c_dubtot[1]}")
            if cb_triplet: st.write(f"**Tripplar:** 1: {c_trip1[0]}-{c_trip1[1]} | X: {c_tripx[0]}-{c_tripx[1]} | 2: {c_trip2[0]}-{c_trip2[1]} | Tot: {c_triptot[0]}-{c_triptot[1]}")
            if cb_occur: st.write(f"**Uppkomster:** 1: {c_occ1[0]}-{c_occ1[1]} | X: {c_occx[0]}-{c_occx[1]} | 2: {c_occ2[0]}-{c_occ2[1]} | Tot: {c_occtot[0]}-{c_occtot[1]}")
            
            st.markdown("---")
            st.subheader("🧩 STRUKTUR GRUPPER (Krav: Minst 2 av 3)")
            if cb_base: st.write(f"**1X2 (Överlever {macro_prob:.1f}%):** 1: {t_ones[0]}-{t_ones[1]} | X: {t_draws[0]}-{t_draws[1]} | 2: {t_twos[0]}-{t_twos[1]}")

        mall_hits = 0
        for i in range(len(v_m)):
            pts = 0
            if cb_base and (c_ones[0] <= ones[i] <= c_ones[1] and c_draws[0] <= draws[i] <= c_draws[1] and c_twos[0] <= twos[i] <= c_twos[1]): pts += 1
            if cb_u_favs and (c_u[0] <= u_wins[i] <= c_u[1]): pts += 1
            if cb_sft and (c_sft[0] <= sft_sums[i] <= c_sft[1]): pts += 1
            if cb_fat and (c_fatf[0] <= fat_f[i] <= c_fatf[1] and c_fata[0] <= fat_a[i] <= c_fata[1] and c_fatt[0] <= fat_t[i] <= c_fatt[1] and c_fatsum[0] <= fat_sums[i] <= c_fatsum[1]): pts += 1
            if cb_streak and (c_s1[0] <= s1[i] <= c_s1[1] and c_sx[0] <= sx[i] <= c_sx[1] and c_s2[0] <= s2[i] <= c_s2[1]): pts += 1
            if cb_gap and (c_g1[0] <= g1[i] <= c_g1[1] and c_gx[0] <= gx[i] <= c_gx[1] and c_g2[0] <= g2[i] <= c_g2[1]): pts += 1
            if cb_single and (c_sing1[0] <= sing1[i] <= c_sing1[1] and c_singx[0] <= singx[i] <= c_singx[1] and c_sing2[0] <= sing2[i] <= c_sing2[1] and c_singtot[0] <= sing_tot[i] <= c_singtot[1]): pts += 1
            if cb_doublet and (c_dub1[0] <= dub1[i] <= c_dub1[1] and c_dubx[0] <= dubx[i] <= c_dubx[1] and c_dub2[0] <= dub2[i] <= c_dub2[1] and c_dubtot[0] <= dub_tot[i] <= c_dubtot[1]): pts += 1
            if cb_triplet and (c_trip1[0] <= trip1[i] <= c_trip1[1] and c_tripx[0] <= tripx[i] <= c_tripx[1] and c_trip2[0] <= trip2[i] <= c_trip2[1] and c_triptot[0] <= trip_tot[i] <= c_triptot[1]): pts += 1
            if cb_occur and (c_occ1[0] <= occ1[i] <= c_occ1[1] and c_occx[0] <= occx[i] <= c_occx[1] and c_occ2[0] <= occ2[i] <= c_occ2[1] and c_occtot[0] <= occ_tot[i] <= c_occtot[1]): pts += 1
            if cb_points and (c_points[0] <= points_vals[i] <= c_points[1]): pts += 1
            if cb_100minus and (c_minus[0] <= minus_sums[i] <= c_minus[1]): pts += 1
            if cb_rank24 and (c_rank24[0] <= rank24_sums[i] <= c_rank24[1]): pts += 1
            if cb_totaldiff and (c_totaldiff[0] <= total_diff_vals[i] <= c_totaldiff[1]): pts += 1
            if cb_aimatrix and (active_ai_min <= ai_ranks[i] <= active_ai_max): pts += 1
            if pts >= slider_pass_req: mall_hits += 1

        st.markdown("---")
        st.info(f"📈 **HISTORISK TRÄFFSÄKERHET:** {mall_hits} av {len(v_m)} rader ({mall_hits/len(v_m)*100:.1f}%) fick tillräckligt med poäng ({slider_pass_req} poäng) för att passera Soft-filtret.")

        st.markdown("---")
        st.subheader("🧬 Dagens Bästa FAT-Sekvenser (Byggklossar)")
        st.markdown("Här analyserar AI:n vilka specifika mönster (1=Fav, 2=Andrahand, 3=Skräll) som bäst täcker in **exakt denna typ av omgång**.")
        
        def get_fat_string(row_str, prob_vector):
            fat_str = ""
            for i, char in enumerate(row_str):
                idx = i * 3
                ranked = sorted([('1', prob_vector[idx]), ('X', prob_vector[idx+1]), ('2', prob_vector[idx+2])], key=lambda x: x[1], reverse=True)
                if char == ranked[0][0]: fat_str += '1'
                elif char == ranked[1][0]: fat_str += '2'
                else: fat_str += '3'
            return fat_str

        fat_strings = [get_fat_string(row['Correct_Row'], row['Prob_Vector']) for _, row in v_m.iterrows() if len(row['Correct_Row']) == antal_matcher]
        total_twins = len(fat_strings)
        
        if total_twins > 0:
            col_seq2, col_seq3, col_combo = st.columns(3)
            
            def calculate_top_seqs(fat_list, length, top_n=5):
                seqs = [''.join(p) for p in itertools.product('123', repeat=length)]
                counts = {s: sum(1 for r in fat_list if s in r) for s in seqs}
                return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

            with col_seq2:
                st.markdown("**Två i rad (Längd 2)**")
                for seq, count in calculate_top_seqs(fat_strings, 2):
                    chans = (count/total_twins)*100
                    st.write(f"**{seq}** ➡️ **{chans:.1f}% chans** ({count} st)")
                    
            with col_seq3:
                st.markdown("**Tre i rad (Längd 3)**")
                for seq, count in calculate_top_seqs(fat_strings, 3):
                    chans = (count/total_twins)*100
                    st.write(f"**{seq}** ➡️ **{chans:.1f}% chans** ({count} st)")
                    
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
                    st.write(f"**{s1}** / **{s2}** ➡️ **{chans:.1f}%** ({count} st)")
                    
        # ==========================================
        # NYA AI-STRATEGIN: BYGGKLOSSAR FÖR REDUCERING
        # ==========================================
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
            
            # Beräkna Bästa Spik
            best_single = utfall[0]
            value_single = best_single[1] - odds_idag[best_single[0]]
            
            # Beräkna Bästa Lås (Två tecken)
            best_double_signs = sorted([utfall[0][0], utfall[1][0]])
            best_double_str = "1X" if best_double_signs==['1','X'] else "12" if best_double_signs==['1','2'] else "X2"
            best_double_pct = utfall[0][1] + utfall[1][1]
            
            # Beräkna Skrällar (Dagens odds < 20%)
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

        # --- SORTERA FRAM KATEGORIERNA (NU TOPP 5) ---
        spikar_sorted = sorted(match_data, key=lambda x: (x['best_single_pct'], x['value_single']), reverse=True)
        top_5_spikar = spikar_sorted[:5]
        
        las_sorted = sorted(match_data, key=lambda x: x['best_double_pct'], reverse=True)
        top_5_las = las_sorted[:5]
        
        all_skrallar = []
        for md in match_data:
            all_skrallar.extend(md['skrällar'])
        top_5_skrallar = sorted(all_skrallar, key=lambda x: x['hist_pct'], reverse=True)[:5]
        
        kv_spikar = spikar_sorted[:2]
        used_matches_for_kv = [m['match'] for m in kv_spikar]
        
        kv_las_candidates = [m for m in las_sorted if m['match'] not in used_matches_for_kv]
        kv_las = kv_las_candidates[:4]
        
        group_a = [kv_spikar[0], kv_las[0], kv_las[1]] if len(kv_las) >= 2 else []
        group_b = [kv_spikar[1], kv_las[2], kv_las[3]] if len(kv_las) >= 4 else []

        # --- RITA UPP ALLT ---
        col_spik, col_las, col_skrall = st.columns(3)
        
        with col_spik:
            st.markdown("🔥 **Topp 5 Spikarna**")
            for s in top_5_spikar:
                st.write(f"**M{s['match']}: {s['best_single_sign']}** (Vinner {s['best_single_pct']:.0f}%, Streck {s['odds_idag'][s['best_single_sign']]:.0f}%)")

        with col_las:
            st.markdown("🔒 **Topp 5 Låsen**")
            for l in top_5_las:
                st.write(f"**M{l['match']}: {l['best_double_str']}** (Täcker {l['best_double_pct']:.0f}%)")

        with col_skrall:
            st.markdown("💣 **Topp 5 Skrälldrag (<20%)**")
            if not top_5_skrallar:
                st.write("Hittade inga skrällar under 20% idag.")
            for sk in top_5_skrallar:
                st.write(f"**M{sk['match']}: {sk['sign']}** (Vinner {sk['hist_pct']:.0f}%, Streck {sk['odds_idag']:.0f}%)")

        if group_a and group_b:
            st.markdown("---")
            st.markdown("🎯 **Dina Dubbla Kärnvillkor (Reducerings-förslag)**")
            st.markdown("Ställ in dessa som *'U-tecken/Utgångsrad'* i ditt program med kravet att **exakt 2 eller 3 måste sitta**. Genom att kräva detta slipper du helgardera allt, rensar bort massor av skräprader, och behåller ändå en oerhört hög slagkraft.")
            
            def calc_2_of_3_prob(p1, p2, p3):
                p1, p2, p3 = p1/100.0, p2/100.0, p3/100.0
                q1, q2, q3 = 1-p1, 1-p2, 1-p3
                exakt_2 = (p1 * p2 * q3) + (p1 * p3 * q2) + (p2 * p3 * q1)
                alla_3 = p1 * p2 * p3
                return (exakt_2 + alla_3) * 100

            prob_a = calc_2_of_3_prob(group_a[0]['best_single_pct'], group_a[1]['best_double_pct'], group_a[2]['best_double_pct'])
            prob_b = calc_2_of_3_prob(group_b[0]['best_single_pct'], group_b[1]['best_double_pct'], group_b[2]['best_double_pct'])

            col_kva, col_kvb = st.columns(2)
            with col_kva:
                st.info(f"🛡️ **Grupp A (Krav: Minst 2 av 3 ska sitta)**\n\n"
                        f"✅ **M{group_a[0]['match']}**: Spik {group_a[0]['best_single_sign']} *(Vinner {group_a[0]['best_single_pct']:.0f}%)*\n\n"
                        f"✅ **M{group_a[1]['match']}**: Lås {group_a[1]['best_double_str']} *(Täcker {group_a[1]['best_double_pct']:.0f}%)*\n\n"
                        f"✅ **M{group_a[2]['match']}**: Lås {group_a[2]['best_double_str']} *(Täcker {group_a[2]['best_double_pct']:.0f}%)*\n\n"
                        f"📊 **Chans att villkoret överlever: {prob_a:.1f}%**")
            with col_kvb:
                st.info(f"⚔️ **Grupp B (Krav: Minst 2 av 3 ska sitta)**\n\n"
                        f"✅ **M{group_b[0]['match']}**: Spik {group_b[0]['best_single_sign']} *(Vinner {group_b[0]['best_single_pct']:.0f}%)*\n\n"
                        f"✅ **M{group_b[1]['match']}**: Lås {group_b[1]['best_double_str']} *(Täcker {group_b[1]['best_double_pct']:.0f}%)*\n\n"
                        f"✅ **M{group_b[2]['match']}**: Lås {group_b[2]['best_double_str']} *(Täcker {group_b[2]['best_double_pct']:.0f}%)*\n\n"
                        f"📊 **Chans att villkoret överlever: {prob_b:.1f}%**")
# ==========================================
        # NYHET: MAKRO-VILLKOR (TECKENFÖRDELNING 2 AV 3)
        # ==========================================
        st.markdown("---")
        st.subheader("🧩 Makro-Villkor: Teckenfördelning")
        st.markdown("Här har AI:n dynamiskt letat upp de tajtaste möjliga intervallerna som **garanterar minst 90% historisk överlevnad** när du spelar med villkoret att 2 av 3 måste sitta.")
        
        # Dynamisk sökning efter perfekta intervaller (mål: minst 90% träff)
        t_ones, t_draws, t_twos = (0,0), (0,0), (0,0)
        macro_prob = 0
        macro_hits = 0
        
        # AI:n testar gradvis bredare täckning (från 30% upp till 100%) tills 2-av-3-villkoret når 90%
        for cov in range(30, 101, 2): 
            t_ones = get_best_interval(ones, cov)
            t_draws = get_best_interval(draws, cov)
            t_twos = get_best_interval(twos, cov)
            
            macro_hits = 0
            for i in range(len(v_m)):
                req1 = 1 if (t_ones[0] <= ones[i] <= t_ones[1]) else 0
                reqx = 1 if (t_draws[0] <= draws[i] <= t_draws[1]) else 0
                req2 = 1 if (t_twos[0] <= twos[i] <= t_twos[1]) else 0
                
                if (req1 + reqx + req2) >= 2:
                    macro_hits += 1
                    
            macro_prob = (macro_hits / len(v_m)) * 100 if len(v_m) > 0 else 0
            
            if macro_prob >= 90.0:
                break # Målet nått! Stanna på dessa optimalt snäva intervaller.

        st.info(f"⚖️ **Krav: Minst 2 av följande 3 påståenden måste stämma på raden:**\n\n"
                f"1️⃣ **Antal 1:or** ska vara exakt **{t_ones[0]} till {t_ones[1]}** st.\n\n"
                f"✖️ **Antal X** ska vara exakt **{t_draws[0]} till {t_draws[1]}** st.\n\n"
                f"2️⃣ **Antal 2:or** ska vara exakt **{t_twos[0]} till {t_twos[1]}** st.\n\n"
                f"📊 **Historisk chans att kupongen överlever detta villkor: {macro_prob:.1f}%**")
        
        if antal_matcher == 8:
            st.markdown("---")
            st.subheader("🎲 EXAKT UTRÄKNING (6 561 rader)")
            all_possible_rows = [''.join(tup) for tup in itertools.product(['1','X','2'], repeat=8)]
            ai_matrix, ai_scores_asc, ai_tot = calculate_ai_matrix_from_values(input_compare)
            
            valid_exact_rows = [] 
            
            match_odds_input = [input_compare[j:j+3] for j in range(0, len(input_compare), 3)]
            
            for tr in all_possible_rows:
                pts = 0
                if cb_base and (c_ones[0] <= tr.count('1') <= c_ones[1] and c_draws[0] <= tr.count('X') <= c_draws[1] and c_twos[0] <= tr.count('2') <= c_twos[1]): pts += 1
                if cb_u_favs and (c_u[0] <= get_top_n_favs_wins(tr, input_compare, slider_u_count) <= c_u[1]): pts += 1
                if cb_sft and (c_sft[0] <= get_sft_sum(tr, input_compare) <= c_sft[1]): pts += 1
                if cb_fat:
                    f_c, a_c, t_c, fsum_c = get_fat(tr, input_compare)
                    if (c_fatf[0] <= f_c <= c_fatf[1] and c_fata[0] <= a_c <= c_fata[1] and c_fatt[0] <= t_c <= c_fatt[1] and c_fatsum[0] <= fsum_c <= c_fatsum[1]): pts += 1
                if cb_streak:
                    s1_c, sx_c, s2_c, _ = get_streaks(tr)
                    if (c_s1[0] <= s1_c <= c_s1[1] and c_sx[0] <= sx_c <= c_sx[1] and c_s2[0] <= s2_c <= c_s2[1]): pts += 1
                if cb_gap:
                    g1_c, gx_c, g2_c, _ = get_gaps(tr)
                    if (c_g1[0] <= g1_c <= c_g1[1] and c_gx[0] <= gx_c <= c_gx[1] and c_g2[0] <= g2_c <= c_g2[1]): pts += 1
                if cb_single:
                    si1_c, six_c, si2_c, singtot_c, _ = get_singles(tr)
                    if (c_sing1[0] <= si1_c <= c_sing1[1] and c_singx[0] <= six_c <= c_singx[1] and c_sing2[0] <= si2_c <= c_sing2[1] and c_singtot[0] <= singtot_c <= c_singtot[1]): pts += 1
                if cb_doublet:
                    d1_c, dx_c, d2_c, dubtot_c, _ = get_doublets(tr)
                    if (c_dub1[0] <= d1_c <= c_dub1[1] and c_dubx[0] <= dx_c <= c_dubx[1] and c_dub2[0] <= d2_c <= c_dub2[1] and c_dubtot[0] <= dubtot_c <= c_dubtot[1]): pts += 1
                if cb_triplet:
                    t1_c, tx_c, t2_c, triptot_c, _ = get_triplets(tr)
                    if (c_trip1[0] <= t1_c <= c_trip1[1] and c_tripx[0] <= tx_c <= c_tripx[1] and c_trip2[0] <= t2_c <= c_trip2[1] and c_triptot[0] <= triptot_c <= c_triptot[1]): pts += 1
                if cb_occur:
                    o1_c, ox_c, o2_c, occtot_c, _ = get_occurrences(tr)
                    if (c_occ1[0] <= o1_c <= c_occ1[1] and c_occx[0] <= ox_c <= c_occx[1] and c_occ2[0] <= o2_c <= c_occ2[1] and c_occtot[0] <= occtot_c <= c_occtot[1]): pts += 1
                if cb_points and (c_points[0] <= get_rank_points(tr, input_compare) <= c_points[1]): pts += 1
                if cb_100minus and (c_minus[0] <= get_100_minus_sum(tr, input_compare) <= c_minus[1]): pts += 1
                if cb_rank24 and (c_rank24[0] <= get_rank_sum(tr, input_compare) <= c_rank24[1]): pts += 1
                if cb_totaldiff:
                    td_c = calculate_total_diff(match_odds_input, list(tr))
                    if (c_totaldiff[0] <= td_c <= c_totaldiff[1]): pts += 1
                    
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
        
        plt.tight_layout(pad=2.0, h_pad=2.0)
        
        st.pyplot(fig)
        plt.close(fig)
