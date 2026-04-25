import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import itertools
import bisect
import matplotlib.pyplot as plt

st.set_page_config(page_title="Topptipset AI-Analys", layout="wide", page_icon="⚽")

# ==========================================
# 1. FUNKTIONER
# ==========================================

def calculate_ai_matrix_from_values(values_24):
    matrix = [values_24[i:i+3] for i in range(0, 24, 3)]
    all_scores = sorted([sum(combo) for combo in itertools.product(*matrix)], reverse=True)
    return matrix, all_scores[::-1], len(all_scores)

def get_exact_rank(row_str, matrix, scores_ascending, total_count):
    score = sum(matrix[i][0] if c == '1' else matrix[i][1] if c == 'X' else matrix[i][2] for i, c in enumerate(row_str))
    rank = total_count - bisect.bisect_left(scores_ascending, score)
    return max(rank, 1), score

def get_100_minus_sum(row_str, prob_vector):
    return sum((100 - (prob_vector[i*3] if c == '1' else prob_vector[i*3+1] if c == 'X' else prob_vector[i*3+2])) for i, c in enumerate(row_str))

def get_rank_points_1_16(row_str, prob_vector):
    pm = [[0, 0, 0] for _ in range(8)]
    nf = []
    for m in range(8):
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

def parse_input_string(text):
    return [float(p) for p in re.sub(r'[^\d\.\s]', '', text).split()[:24]]

def get_structural_vector(vec_24):
    matches = [sorted(vec_24[i:i+3], reverse=True) for i in range(0, 24, 3)]
    matches_sorted = sorted(matches, key=lambda m: m[0], reverse=True)
    return [val for m in matches_sorted for val in m]

def get_rank_1_24_sum(row_str, prob_vector):
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
        total_rank += rank_dict.get(p, 24)
    return total_rank

def weighted_euclidean(u, v, w):
    return np.sqrt(np.sum(w * (np.array(u) - np.array(v))**2))

def get_top_n_favs_wins(row_str, prob_vector, top_n):
    match_favs = []
    for m in range(8):
        idx = m * 3
        probs = [(prob_vector[idx], '1'), (prob_vector[idx+1], 'X'), (prob_vector[idx+2], '2')]
        probs.sort(key=lambda x: x[0], reverse=True)
        match_favs.append({'pct': probs[0][0], 'sign': probs[0][1], 'match_idx': m})
    match_favs.sort(key=lambda x: x['pct'], reverse=True)
    top_n_list = match_favs[:top_n]
    wins = sum(1 for fav in top_n_list if row_str[fav['match_idx']] == fav['sign'])
    return wins

@st.cache_data
def load_database(uploaded_file):
    raw_bytes = uploaded_file.read()
    if raw_bytes.startswith(b'PK'):
        global_db = pd.read_excel(io.BytesIO(raw_bytes))
    else:
        try:
            text = raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = raw_bytes.decode('latin-1')
        sep = ';' if ';' in text.split('\n')[0] else ','
        global_db = pd.read_csv(io.StringIO(text), sep=sep, on_bad_lines='skip')

    col_m = [f'M{i}' for i in range(1, 9)]
    if all(c in global_db.columns for c in col_m):
         global_db['Correct_Row'] = global_db[col_m].astype(str).agg(''.join, axis=1)

    prob_vectors = []
    valid_rows = []
    bank_counts = []

    for idx, row in global_db.iterrows():
        try:
            p_vec = []
            n_bank = 0
            for m in range(1, 9):
                p1 = float(str(row[f'M{m}-1']).replace(',', '.'))
                px = float(str(row[f'M{m}-X']).replace(',', '.'))
                p2 = float(str(row[f'M{m}-2']).replace(',', '.'))
                p_vec.extend([p1, px, p2])
                if p1 >= 70 or px >= 70 or p2 >= 70: n_bank += 1
            prob_vectors.append(p_vec)
            bank_counts.append(n_bank)
            valid_rows.append(True)
        except:
            prob_vectors.append([])
            bank_counts.append(0)
            valid_rows.append(False)

    global_db['Prob_Vector'] = prob_vectors
    global_db['Antal_Bankar'] = bank_counts
    
    if '8 rätt' not in global_db.columns: 
        global_db['8 rätt'] = 0
        
    global_db['Payout'] = pd.to_numeric(global_db['8 rätt'].astype(str).str.replace(' ', '').str.replace(',', '.'), errors='coerce').fillna(0)
    return global_db[valid_rows]


# ==========================================
# STREAMLIT UI
# ==========================================

st.title("🎯 Topptipset AI-Analys & Reducering")
st.markdown("Ladda upp din historik-fil, skriv in dina odds och låt matematiken bygga ditt system!")

# --- SIDEBAR (INSTÄLLNINGAR) ---
with st.sidebar:
    st.header("⚙️ Inställningar")
    
    st.subheader("Matchning & Kärna")
    slider_top_n = st.slider("Antal historiska matcher att hämta", 5, 100, 30, step=5)
    slider_core_val = st.slider("Kärna % (Värde & Svårighet)", 40, 100, 90, step=5)
    slider_core_str = st.slider("Kärna % (Struktur & Tecken)", 40, 100, 100, step=5)
    slider_u_count = st.slider("Antal Topp-Favoriter (U-tecken)", 1, 8, 3, step=1)
    
    st.subheader("Avancerade Filter")
    p_opts = [0, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
    slider_payout = st.select_slider("Utdelnings-krav (kr)", options=p_opts, value=(0, 500000))
    slider_bankar = st.slider("Antal Bankar i omgången (>=70%)", 0, 8, (0, 8))
    
    st.subheader("Mallen - Aktiva Filter")
    cb_payout = st.checkbox("Utdelning (8 rätt)", value=True)
    cb_u_favs = st.checkbox("Topp-Favoriter (U-tecken)", value=True)
    cb_sft = st.checkbox("SFT Summa", value=True)
    cb_fat = st.checkbox("FAT-Tabell & Summa", value=True)
    cb_points = st.checkbox("POÄNGFILTER (Eget)", value=True)
    cb_100minus = st.checkbox("100-minus Summa", value=True)
    cb_rank24 = st.checkbox("Rank 1-24 Summa", value=True)
    
    cb_base = st.checkbox("Grundfilter (1, X, 2)", value=True)
    cb_streak = st.checkbox("Sviter", value=True)
    cb_gap = st.checkbox("Luckor", value=True)
    cb_single = st.checkbox("Singlar", value=True)
    cb_doublet = st.checkbox("Dubbletter", value=True)
    cb_triplet = st.checkbox("Tripplar", value=True)
    cb_occur = st.checkbox("Uppkomster", value=True)
    
    cb_aimatrix = st.checkbox("AI-Matrix Rank", value=True)
    cb_manual_ai_rank = st.checkbox("Styr AI-Rank manuellt", value=False)
    if cb_manual_ai_rank:
        slider_ai_rank = st.slider("AI-Rank Slider", 1, 6561, (1, 6561))
    
    cb_structure = st.checkbox("Matcha Struktur (Viktad)", value=True)

# --- MAIN AREA ---
uploaded_file = st.file_uploader("Ladda upp din CSV/XLSX Databas", type=["csv", "xlsx"])

input_text = st.text_area("Klistra in VÄRDEN (24 st oddsprocent för Topptipset):", height=100)

if st.button("🚀 KÖR ANALYS", use_container_width=True):
    if uploaded_file is None:
        st.error("⚠️ Vänligen ladda upp en databas-fil först!")
    elif not input_text:
        st.error("⚠️ Vänligen klistra in 24 oddsvärden!")
    else:
        with st.spinner("Analyserar..."):
            global_db = load_data(uploaded_file)
            
            input_vec = parse_input_string(input_text)
            if len(input_vec) != 24: 
                st.error(f"⚠️ Fel: Topptipset kräver exakt 24 värden. Hittade {len(input_vec)}.")
                st.stop()

            current_bankers = sum(1 for v in input_vec if v >= 70)
            st.success(f"✅ Databas inladdad ({len(global_db)} omgångar). Hittade {current_bankers} st bankar (>=70%) i din rad.")

            input_compare = get_structural_vector(input_vec) if cb_structure else input_vec
            weights_arr = np.array([w for i in range(0, 24, 3) for w in [(max(input_compare[i:i+3])/100.0)**2]*3])

            df_s = global_db.copy()
            df_s['Sim'] = [weighted_euclidean(input_compare, get_structural_vector(r['Prob_Vector']) if cb_structure else r['Prob_Vector'], weights_arr) if len(r['Prob_Vector'])==24 else 9999 for _, r in df_s.iterrows()]
            
            matches = df_s.sort_values('Sim').head(slider_top_n)

            v_m = matches.copy()
            pay_min, pay_max = slider_payout
            
            if cb_payout: v_m = v_m[(v_m['Payout'] >= pay_min) & (v_m['Payout'] <= pay_max)]
            v_m = v_m[(v_m['Antal_Bankar'] >= slider_bankar[0]) & (v_m['Antal_Bankar'] <= slider_bankar[1])]

            if len(v_m) == 0: 
                st.error("❌ Inga matcher kvar efter filtrering. Testa att lätta på Utdelningskravet i menyn till vänster.")
                st.stop()

            st.subheader(f"📋 Inkluderade Historiska Omgångar ({len(v_m)} st)")
            st.dataframe(v_m[['Datum', 'ID_Omg', 'Payout', 'Sim']].rename(columns={'Payout':'Utdelning', 'Sim':'Likhet'}).style.format({'Utdelning': '{:.0f} kr', 'Likhet': '{:.2f}'}), use_container_width=True)

            db_ranks = v_m['True_Rank'].tolist() if 'True_Rank' in v_m.columns else [0]*len(v_m)

            ones, draws, twos = [], [], []
            s1, sx, s2, g1, gx, g2 = [], [], [], [], [], []
            sing1, singx, sing2, sing_tot = [], [], [], []
            dub1, dubx, dub2, dub_tot = [], [], [], []
            trip1, tripx, trip2, trip_tot = [], [], [], []
            occ1, occx, occ2, occ_tot = [], [], [], []
            sft_sums, fat_f, fat_a, fat_t, fat_sums = [], [], [], [], []
            points_vals, minus_sums, rank24_sums, u_wins, ai_ranks = [], [], [], [], []

            for _, row in v_m.iterrows():
                r, p = row['Correct_Row'], row['Prob_Vector']
                
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
                points_vals.append(get_rank_points_1_16(r, p))
                minus_sums.append(get_100_minus_sum(r, p))
                rank24_sums.append(get_rank_1_24_sum(r, p))
                u_wins.append(get_top_n_favs_wins(r, p, slider_u_count))

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
            c_u = get_best_interval(u_wins, c_v)
            
            c_ai_rank = get_best_interval(ai_ranks, c_v) if len(ai_ranks) > 0 else (1, 6561)
            active_ai_min, active_ai_max = slider_ai_rank if cb_manual_ai_rank else c_ai_rank
            ai_txt = "AI-Rank (MANUELL SLIDER)" if cb_manual_ai_rank else f"AI-Rank (AUTO {c_v}%)"

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🤖 AI:ns Råd")
                if np.std(rank24_sums) <= 12: st.success(f"✅ Rank 1-24 Summa: Extremt enad historik!")
                elif np.std(rank24_sums) >= 18: st.warning(f"⚠️ Rank 1-24 Summa: Sprider sig för mycket, undvik tajta krav")
                
                if np.std(u_wins) <= 0.8: st.success(f"✅ Topp {slider_u_count} Favoriter: MAGISKT reduceringstips!")
                elif np.std(u_wins) >= 1.2: st.warning(f"⚠️ Topp {slider_u_count} Favoriter: Sviker helt oförutsägbart, undvik.")
            with col2:
                st.write("")
                if np.std(ones) <= 1.0: st.success(f"✅ Antal 1:or: Mycket starkt mönster")
                elif np.std(ones) >= 1.5: st.warning(f"⚠️ Antal 1:or: Mycket ojämnt historiskt, lämna öppet")
                if np.std(fat_f) <= 0.8: st.success(f"✅ FAT (Favoriter): Hög tillförlitlighet!")
                elif np.std(fat_f) >= 1.3: st.warning(f"⚠️ FAT (Favoriter): Varierar kraftigt, undvik filter")

            st.markdown("---")
            st.header("📋 VECKANS MALL")
            
            col_v, col_s = st.columns(2)
            with col_v:
                st.subheader(f"💰 VÄRDE & SVÅRIGHET ({c_v}%)")
                if cb_payout: st.write(f"**Utdelning:** {v_m['Payout'].min():.0f} - {v_m['Payout'].max():.0f} kr")
                if cb_aimatrix: st.write(f"**{ai_txt}:** {active_ai_min:.0f} - {active_ai_max:.0f}")
                if cb_rank24: st.write(f"**Rank 1-24 Summa:** {c_rank24[0]:.1f} - {c_rank24[1]:.1f}")
                if cb_100minus: st.write(f"**100-minus Summa:** {c_minus[0]} - {c_minus[1]}")
                if cb_sft: st.write(f"**SFT Summa:** {c_sft[0]} - {c_sft[1]}")
                if cb_points: st.write(f"**Poängfilter:** {c_points[0]} - {c_points[1]}")
                if cb_fat: st.write(f"**FAT:** F:{c_fatf[0]}-{c_fatf[1]} | A:{c_fata[0]}-{c_fata[1]} | T:{c_fatt[0]}-{c_fatt[1]} (Summa: {c_fatsum[0]}-{c_fatsum[1]})")
                if cb_u_favs: st.write(f"**Topp {slider_u_count} Favoriter:** {c_u[0]} - {c_u[1]} st vinner")

            with col_s:
                st.subheader(f"⚽ STRUKTUR ({c_s}%)")
                if cb_base: st.write(f"**1X2:** 1: {c_ones[0]}-{c_ones[1]} | X: {c_draws[0]}-{c_draws[1]} | 2: {c_twos[0]}-{c_twos[1]}")
                if cb_streak: st.write(f"**Sviter:** 1: {c_s1[0]}-{c_s1[1]} | X: {c_sx[0]}-{c_sx[1]} | 2: {c_s2[0]}-{c_s2[1]}")
                if cb_gap: st.write(f"**Luckor:** 1: {c_g1[0]}-{c_g1[1]} | X: {c_gx[0]}-{c_gx[1]} | 2: {c_g2[0]}-{c_g2[1]}")
                if cb_single: st.write(f"**Singlar:** 1: {c_sing1[0]}-{c_sing1[1]} | X: {c_singx[0]}-{c_singx[1]} | 2: {c_sing2[0]}-{c_sing2[1]} | Tot: {c_singtot[0]}-{c_singtot[1]}")
                if cb_doublet: st.write(f"**Dubbletter:** 1: {c_dub1[0]}-{c_dub1[1]} | X: {c_dubx[0]}-{c_dubx[1]} | 2: {c_dub2[0]}-{c_dub2[1]} | Tot: {c_dubtot[0]}-{c_dubtot[1]}")
                if cb_triplet: st.write(f"**Tripplar:** 1: {c_trip1[0]}-{c_trip1[1]} | X: {c_tripx[0]}-{c_tripx[1]} | 2: {c_trip2[0]}-{c_trip2[1]} | Tot: {c_triptot[0]}-{c_triptot[1]}")
                if cb_occur: st.write(f"**Uppkomster:** 1: {c_occ1[0]}-{c_occ1[1]} | X: {c_occx[0]}-{c_occx[1]} | 2: {c_occ2[0]}-{c_occ2[1]} | Tot: {c_occtot[0]}-{c_occtot[1]}")

            mall_hits = 0
            for i in range(len(v_m)):
                ok = True
                if cb_base and not (c_ones[0] <= ones[i] <= c_ones[1] and c_draws[0] <= draws[i] <= c_draws[1] and c_twos[0] <= twos[i] <= c_twos[1]): ok = False
                if cb_u_favs and not (c_u[0] <= u_wins[i] <= c_u[1]): ok = False
                if cb_sft and not (c_sft[0] <= sft_sums[i] <= c_sft[1]): ok = False
                if cb_fat and not (c_fatf[0] <= fat_f[i] <= c_fatf[1] and c_fata[0] <= fat_a[i] <= c_fata[1] and c_fatt[0] <= fat_t[i] <= c_fatt[1] and c_fatsum[0] <= fat_sums[i] <= c_fatsum[1]): ok = False
                if cb_streak and not (c_s1[0] <= s1[i] <= c_s1[1] and c_sx[0] <= sx[i] <= c_sx[1] and c_s2[0] <= s2[i] <= c_s2[1]): ok = False
                if cb_gap and not (c_g1[0] <= g1[i] <= c_g1[1] and c_gx[0] <= gx[i] <= c_gx[1] and c_g2[0] <= g2[i] <= c_g2[1]): ok = False
                if cb_single and not (c_sing1[0] <= sing1[i] <= c_sing1[1] and c_singx[0] <= singx[i] <= c_singx[1] and c_sing2[0] <= sing2[i] <= c_sing2[1] and c_singtot[0] <= sing_tot[i] <= c_singtot[1]): ok = False
                if cb_doublet and not (c_dub1[0] <= dub1[i] <= c_dub1[1] and c_dubx[0] <= dubx[i] <= c_dubx[1] and c_dub2[0] <= dub2[i] <= c_dub2[1] and c_dubtot[0] <= dub_tot[i] <= c_dubtot[1]): ok = False
                if cb_triplet and not (c_trip1[0] <= trip1[i] <= c_trip1[1] and c_tripx[0] <= tripx[i] <= c_tripx[1] and c_trip2[0] <= trip2[i] <= c_trip2[1] and c_triptot[0] <= trip_tot[i] <= c_triptot[1]): ok = False
                if cb_occur and not (c_occ1[0] <= occ1[i] <= c_occ1[1] and c_occx[0] <= occx[i] <= c_occx[1] and c_occ2[0] <= occ2[i] <= c_occ2[1] and c_occtot[0] <= occ_tot[i] <= c_occtot[1]): ok = False 
                if cb_points and not (c_points[0] <= points_vals[i] <= c_points[1]): ok = False
                if cb_100minus and not (c_minus[0] <= minus_sums[i] <= c_minus[1]): ok = False
                if cb_rank24 and not (c_rank24[0] <= rank24_sums[i] <= c_rank24[1]): ok = False
                if cb_aimatrix and not (active_ai_min <= ai_ranks[i] <= active_ai_max): ok = False
                if ok: mall_hits += 1

            st.markdown("---")
            st.info(f"📈 **HISTORISK TRÄFFSÄKERHET:** {mall_hits} av {len(v_m)} rader ({mall_hits/len(v_m)*100:.1f}%) klarade alla ovanstående krav i kombination.")

            # --- EXAKT UTRÄKNING FÖR TOPPTIPSET ---
            sim_hits = 0
            all_possible_rows = [''.join(tup) for tup in itertools.product(['1','X','2'], repeat=8)]
            ai_matrix, ai_scores_asc, ai_tot = calculate_ai_matrix_from_values(input_compare)
            
            for tr in all_possible_rows:
                if cb_base and not (c_ones[0] <= tr.count('1') <= c_ones[1] and c_draws[0] <= tr.count('X') <= c_draws[1] and c_twos[0] <= tr.count('2') <= c_twos[1]): continue
                if cb_u_favs and not (c_u[0] <= get_top_n_favs_wins(tr, input_compare, slider_u_count) <= c_u[1]): continue
                if cb_sft and not (c_sft[0] <= get_sft_sum(tr, input_compare) <= c_sft[1]): continue
                if cb_fat:
                    f_c, a_c, t_c, fsum_c = get_fat(tr, input_compare)
                    if not (c_fatf[0] <= f_c <= c_fatf[1] and c_fata[0] <= a_c <= c_fata[1] and c_fatt[0] <= t_c <= c_fatt[1] and c_fatsum[0] <= fsum_c <= c_fatsum[1]): continue
                if cb_streak:
                    s1_c, sx_c, s2_c, _ = get_streaks(tr)
                    if not (c_s1[0] <= s1_c <= c_s1[1] and c_sx[0] <= sx_c <= c_sx[1] and c_s2[0] <= s2_c <= c_s2[1]): continue
                if cb_gap:
                    g1_c, gx_c, g2_c, _ = get_gaps(tr)
                    if not (c_g1[0] <= g1_c <= c_g1[1] and c_gx[0] <= gx_c <= c_gx[1] and c_g2[0] <= g2_c <= c_g2[1]): continue
                if cb_single:
                    si1_c, six_c, si2_c, singtot_c, _ = get_singles(tr)
                    if not (c_sing1[0] <= si1_c <= c_sing1[1] and c_singx[0] <= six_c <= c_singx[1] and c_sing2[0] <= si2_c <= c_sing2[1] and c_singtot[0] <= singtot_c <= c_singtot[1]): continue
                if cb_doublet:
                    d1_c, dx_c, d2_c, dubtot_c, _ = get_doublets(tr)
                    if not (c_dub1[0] <= d1_c <= c_dub1[1] and c_dubx[0] <= dx_c <= c_dubx[1] and c_dub2[0] <= d2_c <= c_dub2[1] and c_dubtot[0] <= dubtot_c <= c_dubtot[1]): continue
                if cb_triplet:
                    t1_c, tx_c, t2_c, triptot_c, _ = get_triplets(tr)
                    if not (c_trip1[0] <= t1_c <= c_trip1[1] and c_tripx[0] <= tx_c <= c_tripx[1] and c_trip2[0] <= t2_c <= c_trip2[1] and c_triptot[0] <= triptot_c <= c_triptot[1]): continue
                if cb_occur:
                    o1_c, ox_c, o2_c, occtot_c, _ = get_occurrences(tr)
                    if not (c_occ1[0] <= o1_c <= c_occ1[1] and c_occx[0] <= ox_c <= c_occx[1] and c_occ2[0] <= o2_c <= c_occ2[1] and c_occtot[0] <= occtot_c <= c_occtot[1]): continue
                if cb_points and not (c_points[0] <= get_rank_points_1_16(tr, input_compare) <= c_points[1]): continue
                if cb_100minus and not (c_minus[0] <= get_100_minus_sum(tr, input_compare) <= c_minus[1]): continue
                if cb_rank24 and not (c_rank24[0] <= get_rank_1_24_sum(tr, input_compare) <= c_rank24[1]): continue
                if cb_aimatrix:
                    rank_c, _ = get_exact_rank(tr, ai_matrix, ai_scores_asc, ai_tot)
                    if not (active_ai_min <= rank_c <= active_ai_max): continue

                sim_hits += 1

            st.success(f"🎯 **EXAKT KVARVARANDE RADANTAL:** {sim_hits} st (Skurit bort {100 - ((sim_hits / 6561) * 100):.2f}%)")

            # --- GRAF-MOTOR ---
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
            smart_plot(v_m['Payout'].tolist(), 2, 'lightgreen', 'Utdelning', 'Utdelning (kr)', cb_payout, pay_min, pay_max)
            smart_plot(sft_sums, 3, 'coral', 'SFT Summa', 'SFT Summa', cb_sft, c_sft[0], c_sft[1])
            smart_plot(fat_sums, 4, 'gold', 'FAT Summa', 'FAT Summa', cb_fat, c_fatsum[0], c_fatsum[1])
            smart_plot(points_vals, 5, 'mediumpurple', 'Poängfilter', 'Poäng', cb_points, c_points[0], c_points[1])
            smart_plot(minus_sums, 6, 'tan', '100-minus Summa', '100-minus', cb_100minus, c_minus[0], c_minus[1])
            smart_plot(rank24_sums, 7, 'lightpink', 'Rank 1-24 Summa', 'Rank 1-24', cb_rank24, c_rank24[0], c_rank24[1])
            plt.tight_layout(pad=2.0, h_pad=2.0)
            
            st.pyplot(fig)
