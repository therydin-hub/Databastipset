# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGHT-version: standard kör bara 3 omgångar, P29/P27/P24, smalare beam och inga tripplar. Används för snabb iteration, inte slutbetyg.

Tipset AI V9b – kluster/spret, utdelningsriktning och exakt paketsamverkan.

Detta är ett externt Colab-backtest. Streamlit-appen ändras inte.

Standardtest:
- 10 historiska omgångar
- utdelning 100 000–2 500 000 kr
- 30 mest lika omgångar
- strukturpaket 30/30
- övriga individuella filter och slutpaket minst 29/30
- inget hårt maxantal filter
- minst 1 unik borttagen rad per filter
- ett enda vinnande paket per testomgång

Kör i Colab:
    %run package_cluster_payout_synergy_v9b_colab_upload.py
"""

import argparse
import ast
import json
import math
import sys
import subprocess
import time
import traceback
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Colab/app-laddning
# ---------------------------------------------------------------------------

class _DummyStreamlit:
    session_state = {}

    def __getattr__(self, name):
        if name in {"cache_data", "cache_resource"}:
            def deco(func=None, **kwargs):
                if func is None:
                    return lambda f: f
                return func
            return deco

        def f(*args, **kwargs):
            return None
        return f


def _is_colab_runtime() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def _looks_like_app_file(path: Path) -> bool:
    try:
        p = Path(path)
        if not p.exists() or p.is_dir() or p.suffix.lower() not in {'.txt', '.py'}:
            return False
        name = p.name.lower()
        if 'package_' in name and 'app_py' not in name:
            return False
        txt = p.read_text(encoding='utf-8', errors='replace')[:700000]
        return all(x in txt for x in ['APP_VERSION', 'load_database', 'build_clean_filter_specs', '_similar_history_for_backtest'])
    except Exception:
        return False


def _looks_like_db_file(path: Path) -> bool:
    try:
        p = Path(path)
        if not p.exists() or p.is_dir() or p.suffix.lower() != '.csv':
            return False
        name = p.name.lower()
        if any(bad in name for bad in ['summary', 'detail', 'report', 'external_', 'test']):
            return False
        sample = p.read_text(encoding='utf-8', errors='replace')[:20000]
        return (('M1-1' in sample and 'M13-2' in sample) or 'Correct_Row' in sample or 'True_Rank' in sample or 'med_rank' in name)
    except Exception:
        return False


def _candidate_score(path: Path, kind: str) -> int:
    name = path.name.lower()
    score = 0
    if kind == 'app':
        if not _looks_like_app_file(path):
            return -999
        score += 20
        if 'app_py' in name:
            score += 50
        if 'v12_0ce' in name or 'utdelningsintervall' in name:
            score += 40
        if name == 'app.py':
            score += 35
        if any(x in name for x in ['runner', 'colab', 'strategy_compare', 'structure_trim']):
            score -= 60
    else:
        if not _looks_like_db_file(path):
            return -999
        score += 20
        if 'med_rank' in name or 'med rank' in name:
            score += 50
        if 'stryktips' in name or 'stryk' in name:
            score += 30
    return score


def _find_best_file(kind: str, search_dirs=None, uploaded_paths=None):
    if search_dirs is None:
        search_dirs = [Path.cwd(), Path('/content'), Path('/mnt/data')]
    exts = ['*.txt', '*.py'] if kind == 'app' else ['*.csv']
    source_paths = list(uploaded_paths or [])
    for d in search_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for pat in exts:
            source_paths.extend(d.glob(pat))
    candidates = []
    seen = set()
    for p in source_paths:
        try:
            p = Path(p)
            if not p.exists() or p.is_dir():
                continue
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            sc = _candidate_score(p, kind)
            if sc > 0:
                candidates.append((sc, p.stat().st_mtime, p))
        except Exception:
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _colab_upload_files(prompt_title: str, wanted_text: str):
    from google.colab import files  # type: ignore
    print('\n' + prompt_title)
    print(wanted_text)
    print('Välj filen i filväljaren.\n')
    uploaded = files.upload()
    saved = []
    for name, data in uploaded.items():
        p = Path('/content') / name if Path('/content').exists() else Path(name)
        try:
            if not p.exists() or p.stat().st_size != len(data):
                p.write_bytes(data)
        except Exception:
            pass
        saved.append(p)
    return saved


def _resolve_required_files(args):
    def as_existing(value):
        if not value:
            return None
        p = Path(str(value))
        return p if p.exists() else None

    force = bool(getattr(args, 'force_upload', False))
    uploaded_paths = []
    app_file = None if force else as_existing(args.app_file)
    db_file = None if force else as_existing(args.db_file)

    if app_file is not None and not _looks_like_app_file(app_file):
        print(f'Angiven appfil ser inte ut som rätt appbas och ignoreras: {app_file}', flush=True)
        app_file = None
    if db_file is not None and not _looks_like_db_file(db_file):
        print(f'Angiven databasfil ser inte ut som rätt historikfil och ignoreras: {db_file}', flush=True)
        db_file = None

    if app_file is None and not force:
        app_file = _find_best_file('app')
    if db_file is None and not force:
        db_file = _find_best_file('db')

    if _is_colab_runtime():
        if app_file is None or force:
            uploaded_paths += _colab_upload_files(
                'STEG 1/2: Ladda upp APPFILEN som ska testas',
                'Exempel: app_py_v12_0ce_utdelningsintervall_minmax(1).txt eller din app.py.'
            )
            app_file = _find_best_file('app', uploaded_paths=uploaded_paths) or _find_best_file('app')
            if app_file is None:
                raise SystemExit('Hittar ingen appfil efter uppladdning.')

        if db_file is None or force:
            uploaded_paths += _colab_upload_files(
                'STEG 2/2: Ladda upp DATABASFILEN',
                'Exempel: Stryktips _Med_Rank(4).csv. Filnamnet behöver inte vara exakt; innehållet avgör.'
            )
            db_file = _find_best_file('db', uploaded_paths=uploaded_paths) or _find_best_file('db')
            if db_file is None:
                raise SystemExit('Hittar ingen databasfil efter uppladdning.')

    if app_file is None or not Path(app_file).exists():
        raise SystemExit('Hittar ingen appfil. Kör med --force-upload eller ange --app-file.')
    if db_file is None or not Path(db_file).exists():
        raise SystemExit('Hittar ingen databasfil. Kör med --force-upload eller ange --db-file.')
    return Path(app_file), Path(db_file)


def _load_app_functions(app_file: Path, fast_no_supermakro: bool = False):
    sys.modules.setdefault('streamlit', _DummyStreamlit())
    source = app_file.read_text(encoding='utf-8', errors='replace')
    mod = ast.parse(source, filename=str(app_file))
    keep_nodes = []
    for node in mod.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep_nodes.append(node)
        elif isinstance(node, ast.Assign):
            # Behåll modulkonstanter före UI-blocket.
            if getattr(node, 'lineno', 999999) < 8130:
                keep_nodes.append(node)
    new_mod = ast.Module(body=keep_nodes, type_ignores=[])
    ast.fix_missing_locations(new_mod)
    ns = {'__name__': 'frequency_interval_synergy_runner'}
    exec(compile(new_mod, str(app_file), 'exec'), ns)

    if fast_no_supermakro and 'build_clean_filter_specs' in ns:
        orig = ns['build_clean_filter_specs']

        def fast_build_clean_filter_specs(*args, **kwargs):
            kwargs['include_supermakro'] = False
            return orig(*args, **kwargs)

        ns['build_clean_filter_specs'] = fast_build_clean_filter_specs
    return ns


# ---------------------------------------------------------------------------
# Optimal intervall- och samverkansmotor
# ---------------------------------------------------------------------------

PROFILE_CATEGORIES = {'Värde & svårighet', 'FAT', 'FAT-sekvenser', 'Favorit & skräll'}
STRUCTURE_CATEGORIES = {'Struktur'}


def _is_profile_candidate(c: dict) -> bool:
    return str(c.get('category', '')) in PROFILE_CATEGORIES


def _is_structure_candidate(c: dict) -> bool:
    return str(c.get('category', '')) in STRUCTURE_CATEGORIES


def _safe_category(c: dict) -> str:
    return str(c.get('category', '') or '')


def _is_edge_candidate(c: dict) -> bool:
    # Niklas manuella spretfilter: värde/poäng/svårighet + favorit/skräll.
    # I V8c får de komplettera, men inte dominera paketet.
    return _safe_category(c) in {'Värde & svårighet', 'Favorit & skräll'}


def _is_fat_abc_candidate(c: dict) -> bool:
    """FAT, FAT-sekvenser och ABC-filter (ABC ligger i FAT-kategorin)."""
    txt = f"{c.get('name', '')} {c.get('key', '')}".lower()
    return _safe_category(c) in {'FAT', 'FAT-sekvenser'} or 'abc' in txt


def _row_band_parts(rows: int, target_min_rows: int, target_max_rows: int):
    """Rangkomponenter för ett önskat radintervall, inte ett ensidigt tak."""
    rows = int(rows)
    lo = max(0, int(target_min_rows))
    hi = max(lo, int(target_max_rows))
    mid = (lo + hi) / 2.0
    if lo <= rows <= hi:
        zone = 2
        distance = 0
    elif rows > hi:
        zone = 1
        distance = rows - hi
    else:
        zone = 0
        distance = lo - rows
    return zone, -int(distance), -abs(float(rows) - mid)


def _candidate_text(c: dict) -> str:
    return f"{c.get('name','')} {c.get('key','')} {c.get('category','')}".lower()


def _is_blocked_auto_candidate(c: dict) -> bool:
    """V8c: blockera filter som visade sig kunna döda facit i snabbtestet.

    SFT-avstånd från historik föll på testfallet 2026-05-17 trots 29/30
    paketträff. Det kan fortfarande visas/användas manuellt i appen, men ska
    inte väljas automatiskt i den här konservativa paketmotorn.
    """
    txt = _candidate_text(c)
    dangerous = [
        'sft-avstånd', 'sft avstånd', 'sft_avstånd', 'sft distance',
    ]
    return any(d in txt for d in dangerous)


def _coverage_grid_dense():
    return [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50]


def _fast_frame_capacity_block(ns: dict, spec: dict, interval, frame, frame_vals) -> bool:
    """Samma ytterkantskontroll som appen, men återanvänder redan beräknade ramvärden."""
    try:
        name = str(spec.get('name', ''))
        low, high = float(interval[0]), float(interval[1])
        if not frame:
            return False

        sign = {'Tecken 1': '1', 'Tecken X': 'X', 'Tecken 2': '2'}.get(name)
        if sign is not None:
            normalize_signs = ns.get('normalize_signs', lambda x: list(x or []))
            norm_frame = [normalize_signs(x) for x in frame]
            min_possible = sum(1 for signs in norm_frame if len(signs) == 1 and sign in signs)
            max_possible = sum(1 for signs in norm_frame if sign in signs)
            span = max(1, max_possible - min_possible)
            lower_pressure = max(0.0, (low - min_possible) / span)
            upper_pressure = max(0.0, (max_possible - high) / span)
            return bool((max_possible <= 9 and lower_pressure >= 0.38) or (max_possible <= 9 and upper_pressure >= 0.45))

        category = str(spec.get('category', ''))
        if category not in {'Struktur', 'FAT', 'FAT-sekvenser', 'Favorit & skräll'}:
            return False
        if int(spec.get('decimals', 0)) != 0:
            return False
        vals = np.asarray(frame_vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 10:
            return False
        vmin, vmax = float(vals.min()), float(vals.max())
        span = max(1.0, vmax - vmin)
        lower_pressure = max(0.0, (low - vmin) / span)
        upper_pressure = max(0.0, (vmax - high) / span)
        if category == 'FAT-sekvenser':
            small_span = span <= 3.0
            locks_to_frame_min = (vmin >= 1 and low <= vmin + 0.01 and high < vmax - 0.01)
            near_upper_cap = upper_pressure >= (0.45 if small_span else 0.65)
            near_lower_req = lower_pressure >= (0.45 if small_span else 0.65)
            return bool(locks_to_frame_min or near_upper_cap or near_lower_req)
        return bool(lower_pressure >= 0.72 or upper_pressure >= 0.72)
    except Exception:
        return False


def _build_dynamic_candidates(ns: dict, specs: list, sim_df: pd.DataFrame, frame_rows: list, frame: list, antal_matcher: int, *, frame_adapt: bool = True, target_floor: int = 27, variants_per_key: int = 8, max_candidates: int = 220, structure_safe_only: bool = True, val_df: Optional[pd.DataFrame] = None, min_val_pct: float = 70.0, min_structure_val_pct: float = 85.0):
    """Bygger dynamiska intervallkandidater med samma grundidé som appens motor.

    Varje filter får många intervallnivåer, men vi beskär till de bästa varianterna
    per filternyckel så beam-sökningen blir rimligt snabb i Colab.
    """
    _candidate_intervals_for_spec = ns['_candidate_intervals_for_spec']
    _frame_capacity_pressure = ns['_frame_capacity_pressure']
    _spec_pass = ns['_spec_pass']
    _prune_candidate_ladder = ns['_prune_candidate_ladder']

    hist_rows = [ns['normalize_single_row_text'](r) for r in list(sim_df['Correct_Row']) if len(ns['normalize_single_row_text'](r)) == int(antal_matcher)]
    htot = len(hist_rows)
    ftot = len(frame_rows)
    val_rows = []
    if val_df is not None and isinstance(val_df, pd.DataFrame) and 'Correct_Row' in val_df.columns:
        val_rows = [ns['normalize_single_row_text'](r) for r in list(val_df['Correct_Row']) if len(ns['normalize_single_row_text'](r)) == int(antal_matcher)]
    vtot = len(val_rows)
    if htot <= 0 or ftot <= 0:
        return [], htot, ftot

    candidates = []
    grid = _coverage_grid_dense()
    pre_hist_mask = np.ones(htot, dtype=bool)

    for spec in specs or []:
        # V8c: vissa värde/svårighetsfilter är för vassa som auto-filter.
        # Framför allt SFT-avstånd från historik; det dödade facit i diagnosen.
        if _is_blocked_auto_candidate(spec):
            continue
        try:
            getter = spec.get('getter')
            if getter is None:
                continue
            frame_vals = np.array([float(getter(r)) for r in frame_rows], dtype=float)
            hist_vals = list(spec.get('hist_values', []))
            hist_arr = np.array([np.nan if pd.isna(v) else float(v) for v in hist_vals], dtype=float)
            if len(hist_arr) != htot:
                continue
            val_arr = np.array([float(getter(r)) for r in val_rows], dtype=float) if vtot > 0 else np.ones(0, dtype=float)
        except Exception:
            continue

        spec_cands = []
        for iv in _candidate_intervals_for_spec(spec, grid):
            interval = iv.get('interval')
            if interval is None:
                continue
            if frame_adapt and _fast_frame_capacity_block(ns, spec, interval, frame, frame_vals):
                continue
            try:
                lo, hi = float(interval[0]), float(interval[1])
                hist_mask = np.isfinite(hist_arr) & (hist_arr >= lo) & (hist_arr <= hi)
                frame_mask = np.isfinite(frame_vals) & (frame_vals >= lo) & (frame_vals <= hi)
                if vtot > 0:
                    val_mask = np.isfinite(val_arr) & (val_arr >= lo) & (val_arr <= hi)
                else:
                    val_mask = np.ones(0, dtype=bool)
            except Exception:
                try:
                    hist_mask = np.array([_spec_pass(r, spec, interval) for r in hist_rows], dtype=bool)
                    frame_mask = np.array([_spec_pass(r, spec, interval) for r in frame_rows], dtype=bool)
                    val_mask = np.array([_spec_pass(r, spec, interval) for r in val_rows], dtype=bool) if vtot > 0 else np.ones(0, dtype=bool)
                except Exception:
                    continue

            hist_hit = int((pre_hist_mask & hist_mask).sum())
            frame_keep = int(frame_mask.sum())
            if frame_keep >= ftot:
                continue
            if hist_hit < int(target_floor):
                continue
            red_pct = 100.0 - 100.0 * frame_keep / max(1, ftot)
            val_hit = int(val_mask.sum()) if vtot > 0 else 0
            val_pct = 100.0 * val_hit / max(1, vtot) if vtot > 0 else 100.0
            if vtot > 0:
                if str(spec.get('category', '')) in STRUCTURE_CATEGORIES and val_pct < float(min_structure_val_pct):
                    continue
                if str(spec.get('category', '')) not in STRUCTURE_CATEGORIES and val_pct < float(min_val_pct):
                    continue
            # Niklas manuella princip: struktur används bara som 30/30-säker ram.
            # Profilfilter får däremot kapa spretiga yttertal ned till target_floor.
            if structure_safe_only and str(spec.get('category', '')) in STRUCTURE_CATEGORIES and hist_hit < htot:
                continue
            spec_cands.append({
                'key': spec.get('key'),
                'name': spec.get('name'),
                'category': spec.get('category', ''),
                'coverage': float(iv.get('coverage', 0)),
                'interval': interval,
                'interval_txt': iv.get('interval_txt') or _display_interval_local(interval, spec.get('decimals', 0)),
                'hist_mask': hist_mask.astype(bool),
                'frame_mask': frame_mask.astype(bool),
                'hist_hit': hist_hit,
                'hist_total': htot,
                'hist_pct': 100.0 * hist_hit / max(1, htot),
                'val_hit': int(val_hit),
                'val_total': int(vtot),
                'val_pct': float(val_pct),
                'frame_keep': frame_keep,
                'red_pct': red_pct,
            })

        try:
            pruned = _prune_candidate_ladder(spec_cands, max_levels=max(variants_per_key, 6))
        except Exception:
            pruned = spec_cands
        # Per filter: behåll både säkra och reducerande nivåer.
        pruned = sorted(pruned, key=lambda c: (int(c.get('hist_hit', 0)), float(c.get('red_pct', 0.0)), -int(c.get('frame_keep', 10**12))), reverse=True)[:int(variants_per_key)]
        candidates.extend(pruned)

    # Global beskärning V4: behåll en balanserad kandidatpool.
    # V3 vann på FAT + säker struktur, men värde/skräll-kandidaterna fick ofta
    # aldrig en ärlig chans. V4 kvoterar därför in kandidater per kategori innan
    # global fyllnad görs. Det är fortfarande samverkanssökningen som avgör om
    # de ska in i slutpaketet.
    def cand_rank(c):
        edge = 2 if _is_edge_candidate(c) else 0
        profile = 1 if _is_profile_candidate(c) else 0
        structure_bonus = 1 if _is_structure_candidate(c) else 0
        return (int(c.get('hist_hit', 0)), float(c.get('red_pct', 0.0)), edge, profile, structure_bonus, -int(c.get('frame_keep', 10**12)))

    candidates = [c for c in candidates if c.get('key')]
    candidates.sort(key=cand_rank, reverse=True)

    maxc = int(max_candidates)
    value = [c for c in candidates if _safe_category(c) == 'Värde & svårighet']
    favshock = [c for c in candidates if _safe_category(c) == 'Favorit & skräll']
    fat = [c for c in candidates if _safe_category(c) in {'FAT', 'FAT-sekvenser'}]
    structure = [c for c in candidates if _is_structure_candidate(c)]
    other = [c for c in candidates if c not in value and c not in favshock and c not in fat and c not in structure]

    keep = []
    # Kvoterna är avsiktligt generösa. Dedupe + beam tar hand om resten.
    keep += value[:max(25, maxc // 4)]
    keep += favshock[:max(25, maxc // 4)]
    keep += fat[:max(45, maxc // 3)]
    keep += structure[:max(35, maxc // 5)]
    keep += other[:max(10, maxc // 10)]
    # Fyll med bästa globalt så vi inte tappar starka kandidater.
    for c in candidates:
        if len(keep) >= maxc + max(40, maxc // 4):
            break
        keep.append(c)

    # Dedup exakt samma key+interval.
    seen = set()
    out = []
    for c in keep:
        ident = (c.get('key'), c.get('interval_txt'))
        if ident in seen:
            continue
        seen.add(ident)
        c = dict(c)
        c['hist_bits'] = _bool_mask_to_bits(c.get('hist_mask'))
        c['frame_bits'] = _bool_mask_to_bits(c.get('frame_mask'))
        out.append(c)
        if len(out) >= maxc:
            break
    return out, htot, ftot


def _bool_mask_to_bits(mask) -> int:
    """Packar en boolean-mask till ett exakt Python-heltal för snabb AND/bit_count."""
    arr = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        return 0
    packed = np.packbits(arr, bitorder='little')
    return int.from_bytes(packed.tobytes(), byteorder='little', signed=False)


def _build_teckenskydd_bits(ns: dict, row_matrix, frame, antal_matcher: int):
    """En bitmängd per manuellt valt tecken. Exakt teckenskydd utan radkopior."""
    if row_matrix is None or not frame:
        return tuple()
    normalize_signs = ns.get('normalize_signs', lambda x: list(x or []))
    out = []
    try:
        for mi in range(int(antal_matcher)):
            selected = normalize_signs(frame[mi]) if mi < len(frame) else []
            for sign in selected:
                out.append(_bool_mask_to_bits(row_matrix[:, mi] == str(sign)))
    except Exception:
        return tuple()
    return tuple(out)


def _display_interval_local(interval, decimals=0):
    try:
        lo, hi = interval
        if int(decimals) <= 0:
            return f"{round(float(lo)):.0f}–{round(float(hi)):.0f}"
        return f"{float(lo):.{int(decimals)}f}–{float(hi):.{int(decimals)}f}"
    except Exception:
        return str(interval)


@dataclass
class _State:
    hist_mask: int
    frame_mask: int
    hist_size: int
    frame_size: int
    chosen: Tuple[dict, ...]
    used_keys: frozenset
    steps: Tuple[dict, ...]

    @property
    def hist_hit(self) -> int:
        return int(int(self.hist_mask).bit_count())

    @property
    def frame_count(self) -> int:
        return int(int(self.frame_mask).bit_count())

    @property
    def value_filters(self) -> int:
        return sum(1 for c in self.chosen if _safe_category(c) == 'Värde & svårighet')

    @property
    def favshock_filters(self) -> int:
        return sum(1 for c in self.chosen if _safe_category(c) == 'Favorit & skräll')

    @property
    def edge_filters(self) -> int:
        return sum(1 for c in self.chosen if _is_edge_candidate(c))

    @property
    def fat_filters(self) -> int:
        return sum(1 for c in self.chosen if _is_fat_abc_candidate(c))

    @property
    def structure_filters(self) -> int:
        return sum(1 for c in self.chosen if _is_structure_candidate(c))


def _state_sort_key(state: _State, *, target_rows: int, target_min_rows: int, target_hit: int, min_value_filters: int, min_edge_filters: int, phase: str):
    rows = state.frame_count
    hit = state.hist_hit
    zone, band_distance, mid_distance = _row_band_parts(rows, target_min_rows, target_rows)
    value_ok = 1 if state.value_filters >= int(min_value_filters) else 0
    edge_ok = 1 if state.edge_filters >= int(min_edge_filters) else 0
    structure_pen = state.structure_filters
    filters = len(state.chosen)
    return (
        zone,
        hit,
        value_ok,
        edge_ok,
        band_distance,
        mid_distance,
        -structure_pen,
        -filters,
    )


def _final_sort_key(state: _State, *, target_rows: int, target_min_rows: int, min_value_filters: int, min_edge_filters: int, choose_policy: str = 'target_band_then_hit'):
    rows = state.frame_count
    zone, band_distance, mid_distance = _row_band_parts(rows, target_min_rows, target_rows)
    value_ok = 1 if state.value_filters >= int(min_value_filters) else 0
    edge_ok = 1 if state.edge_filters >= int(min_edge_filters) else 0

    if str(choose_policy) == 'hit_then_rows':
        return (
            state.hist_hit, zone, band_distance, mid_distance,
            value_ok, edge_ok, -state.structure_filters, -len(state.chosen),
        )
    if str(choose_policy) == 'budget_then_hit':
        # Bakåtkompatibelt namn, men V8c tolkar budget som ett målband.
        return (
            zone, state.hist_hit, band_distance, mid_distance,
            value_ok, edge_ok, -state.structure_filters, -len(state.chosen),
        )
    return (
        zone, state.hist_hit, band_distance, mid_distance,
        value_ok, edge_ok, -state.structure_filters, -len(state.chosen),
    )


def _dedupe_states(states: List[_State], *, target_rows: int, target_min_rows: int, target_hit: int, min_value_filters: int, min_edge_filters: int, phase: str, beam_width: int) -> List[_State]:
    best_by_sig: Dict[Tuple[str, ...], _State] = {}
    for st in states:
        sig = tuple(sorted(st.used_keys))
        old = best_by_sig.get(sig)
        key = _state_sort_key(st, target_rows=target_rows, target_min_rows=target_min_rows, target_hit=target_hit, min_value_filters=min_value_filters, min_edge_filters=min_edge_filters, phase=phase)
        if old is None:
            best_by_sig[sig] = st
        else:
            old_key = _state_sort_key(old, target_rows=target_rows, target_min_rows=target_min_rows, target_hit=target_hit, min_value_filters=min_value_filters, min_edge_filters=min_edge_filters, phase=phase)
            if key > old_key:
                best_by_sig[sig] = st
    out = list(best_by_sig.values())
    out.sort(key=lambda st: _state_sort_key(st, target_rows=target_rows, target_min_rows=target_min_rows, target_hit=target_hit, min_value_filters=min_value_filters, min_edge_filters=min_edge_filters, phase=phase), reverse=True)
    return out[:int(beam_width)]


def _candidate_step_min(c: dict, *, phase: str, min_step_profile: float, min_step_structure: float, absolute_min_step: float = 5.0) -> float:
    category_min = float(min_step_structure) if _is_structure_candidate(c) else float(min_step_profile)
    return max(float(absolute_min_step), category_min)


def _apply_candidate_to_state(ns: dict, st: _State, cand: dict, *, target_hit: int, target_rows: int, target_min_rows: int, frame_rows: list, frame: list, antal_matcher: int, row_matrix, min_step_profile: float, min_step_structure: float, absolute_min_step: float, min_value_filters: int, min_edge_filters: int, max_structure_filters: int, max_edge_auto_filters: int, max_fat_abc_filters: int, structure_max_reduction_pct: float, phase: str, post_target_min_step: float = 0.0, low_val_floor: float = 0.0, low_val_min_step: float = 0.0) -> Optional[_State]:
    if cand.get('key') in st.used_keys or _is_blocked_auto_candidate(cand):
        return None

    if phase == 'trim':
        if not _is_structure_candidate(cand):
            return None
        if st.structure_filters >= int(max_structure_filters):
            return None
    elif phase == 'profil':
        if not _is_profile_candidate(cand):
            return None
        if _is_edge_candidate(cand) and st.edge_filters >= int(max_edge_auto_filters):
            return None
        if _is_fat_abc_candidate(cand) and st.fat_filters >= int(max_fat_abc_filters):
            return None

    try:
        cur_rows = st.frame_count
        if cur_rows <= 0:
            return None
        new_hist = int(st.hist_mask) & int(cand.get('hist_bits', 0))
        hist_hit = int(new_hist.bit_count())
        # Exakt samverkan: varje profilpaket måste fortsatt hålla minst target_hit.
        if hist_hit < int(target_hit):
            return None
        new_frame = int(st.frame_mask) & int(cand.get('frame_bits', 0))
        new_rows = int(new_frame.bit_count())
        if new_rows <= 0 or new_rows >= cur_rows:
            return None
        step_pct = 100.0 * (cur_rows - new_rows) / max(1, cur_rows)

        # Inga kosmetiska småfilter: minst 5 % verklig stegreduktion på aktuell paketmask.
        min_step = _candidate_step_min(
            cand, phase=phase, min_step_profile=min_step_profile,
            min_step_structure=min_step_structure, absolute_min_step=absolute_min_step,
        )
        if step_pct + 1e-12 < min_step:
            return None

        # Struktur är skyddsram, inte budgetkniv. Begränsa både antal och samlad kapning.
        if phase == 'trim':
            cumulative_red = 100.0 * (st.frame_size - new_rows) / max(1, st.frame_size)
            if cumulative_red > float(structure_max_reduction_pct):
                return None

        # När paketet ligger i målbandet får endast ett tydligt FAT/ABC-steg fortsätta.
        in_band_now = int(target_min_rows) <= cur_rows <= int(target_rows)
        if in_band_now:
            if not _is_fat_abc_candidate(cand):
                return None
            if float(post_target_min_step) > 0 and step_pct < float(post_target_min_step):
                return None

        try:
            cand_val_pct = float(cand.get('val_pct', 100.0))
            cand_val_total = int(cand.get('val_total', 0))
        except Exception:
            cand_val_pct, cand_val_total = 100.0, 0
        if cand_val_total > 0 and float(low_val_floor) > 0 and cand_val_pct < float(low_val_floor) and step_pct < float(low_val_min_step):
            return None

        try:
            # row_matrix-parametern innehåller här förberäknade tecken-bitmängder.
            if row_matrix and any((new_frame & int(sign_bits)) == 0 for sign_bits in row_matrix):
                return None
        except Exception:
            pass

        c2 = dict(cand)
        c2['step_red_pct'] = float(step_pct)
        step = {
            'Filter': c2.get('name', ''),
            'Kategori': c2.get('category', ''),
            'Intervall': c2.get('interval_txt', '-'),
            'Intervallträff': f"{int(c2.get('hist_hit', 0))}/{int(c2.get('hist_total', 0))}",
            'Stegreducering': f"{float(step_pct):.1f}%",
            'Efter filter': int(new_rows),
            'Samlad träff efter steg': f"{hist_hit}/{int(c2.get('hist_total', 0))}",
            'Fas': phase,
        }
        return _State(
            hist_mask=new_hist, frame_mask=new_frame, hist_size=st.hist_size, frame_size=st.frame_size, chosen=st.chosen + (c2,),
            used_keys=frozenset(set(st.used_keys) | {c2.get('key')}), steps=st.steps + (step,),
        )
    except Exception:
        return None


def _beam_expand(ns: dict, start_states: List[_State], candidates: List[dict], *, target_hit: int, target_rows: int, target_min_rows: int, frame_rows: list, frame: list, antal_matcher: int, row_matrix, min_step_profile: float, min_step_structure: float, absolute_min_step: float, min_value_filters: int, min_edge_filters: int, max_structure_filters: int, max_edge_auto_filters: int, max_fat_abc_filters: int, structure_max_reduction_pct: float, max_steps: int, beam_width: int, phase: str, post_target_min_step: float = 0.0, low_val_floor: float = 0.0, low_val_min_step: float = 0.0) -> List[_State]:
    states = list(start_states)
    all_states = list(start_states)
    for _depth in range(int(max_steps)):
        expanded = []
        for st in states:
            for cand in candidates:
                ns2 = _apply_candidate_to_state(
                    ns, st, cand, target_hit=target_hit, target_rows=target_rows,
                    target_min_rows=target_min_rows, frame_rows=frame_rows, frame=frame,
                    antal_matcher=antal_matcher, row_matrix=row_matrix,
                    min_step_profile=min_step_profile, min_step_structure=min_step_structure,
                    absolute_min_step=absolute_min_step, min_value_filters=min_value_filters,
                    min_edge_filters=min_edge_filters, max_structure_filters=max_structure_filters,
                    max_edge_auto_filters=max_edge_auto_filters, max_fat_abc_filters=max_fat_abc_filters,
                    structure_max_reduction_pct=structure_max_reduction_pct, phase=phase,
                    post_target_min_step=post_target_min_step, low_val_floor=low_val_floor,
                    low_val_min_step=low_val_min_step,
                )
                if ns2 is not None:
                    expanded.append(ns2)
        if not expanded:
            break
        states = _dedupe_states(
            expanded, target_rows=target_rows, target_min_rows=target_min_rows,
            target_hit=target_hit, min_value_filters=min_value_filters,
            min_edge_filters=min_edge_filters, phase=phase, beam_width=beam_width,
        )
        all_states.extend(states)
        in_band = sum(
            1 for st in states
            if int(target_min_rows) <= st.frame_count <= int(target_rows)
            and st.value_filters >= int(min_value_filters)
            and st.edge_filters >= int(min_edge_filters)
        )
        if phase == 'trim' and in_band >= max(5, int(beam_width) // 15):
            break
    return _dedupe_states(
        all_states, target_rows=target_rows, target_min_rows=target_min_rows,
        target_hit=target_hit, min_value_filters=min_value_filters,
        min_edge_filters=min_edge_filters, phase=phase, beam_width=beam_width,
    )


def _choose_best_state(states: List[_State], *, target_rows: int, target_min_rows: int, min_value_filters: int, min_edge_filters: int, choose_policy: str = 'target_band_then_hit') -> Optional[_State]:
    valid = [st for st in states if st.chosen and st.value_filters >= int(min_value_filters) and st.edge_filters >= int(min_edge_filters)]
    if not valid:
        valid = [st for st in states if st.chosen]
    if not valid:
        return None
    valid.sort(
        key=lambda st: _final_sort_key(
            st, target_rows=target_rows, target_min_rows=target_min_rows,
            min_value_filters=min_value_filters, min_edge_filters=min_edge_filters,
            choose_policy=choose_policy,
        ),
        reverse=True,
    )
    return valid[0]


def _choose_structure_seeds(states: List[_State], *, target_rows: int, target_min_rows: int, max_seeds: int = 16) -> List[_State]:
    """V8c: flera lätta, exakt 30/30-säkra strukturgrunder."""
    valid = [st for st in (states or []) if st.chosen and st.hist_hit == st.hist_size]
    if not valid:
        return []

    def key(st: _State):
        red = 100.0 * (st.frame_size - st.frame_count) / max(1, st.frame_size)
        # Färre strukturfilter först; därefter rimlig (inte maximal) skyddsram.
        return (-len(st.chosen), -abs(red - 20.0), -st.frame_count)

    valid.sort(key=key, reverse=True)
    out, seen = [], set()
    for st in valid:
        sig = tuple(sorted(st.used_keys))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(st)
        if len(out) >= int(max_seeds):
            break
    return out


def _build_frequency_interval_synergy_package(ns: dict, sim_df: pd.DataFrame, specs: list, frame_rows: list, frame: list, antal_matcher: int, *, min_hit_count: int = 28, target_rows: int = 2000, target_min_rows: int = 1700, max_filters: int = 6, min_value_filters: int = 0, min_edge_filters: int = 0, min_step_profile: float = 5.0, min_step_structure: float = 5.0, absolute_min_step: float = 5.0, max_structure_filters: int = 2, max_edge_auto_filters: int = 1, max_fat_abc_filters: int = 4, structure_max_reduction_pct: float = 35.0, beam_width: int = 30, variants_per_key: int = 3, max_candidates: int = 70, frame_adapt: bool = True, structure_safe_only: bool = True, choose_policy: str = 'target_band_then_hit', validation_df: Optional[pd.DataFrame] = None, min_val_pct: float = 70.0, min_structure_val_pct: float = 85.0, post_target_min_step: float = 5.0, low_val_floor: float = 70.0, low_val_min_step: float = 25.0) -> Tuple[Optional[dict], dict]:
    """V8c: 30/30-strukturgrund, därefter exakt profilsynergi mot ett radmålband."""
    hist_rows = [ns['normalize_single_row_text'](r) for r in list(sim_df['Correct_Row']) if len(ns['normalize_single_row_text'](r)) == int(antal_matcher)]
    htot = len(hist_rows)
    ftot = len(frame_rows)
    if htot <= 0 or ftot <= 0:
        return None, {'error': 'tom historik eller ram'}

    target_min_rows = max(1, min(int(target_min_rows), int(target_rows)))
    _tp = time.time()
    candidates, _, _ = _build_dynamic_candidates(
        ns, specs, sim_df, frame_rows, frame, int(antal_matcher),
        frame_adapt=bool(frame_adapt), target_floor=int(min_hit_count),
        variants_per_key=int(variants_per_key), max_candidates=int(max_candidates),
        structure_safe_only=True, val_df=validation_df,
        min_val_pct=float(min_val_pct), min_structure_val_pct=float(min_structure_val_pct),
    )
    print(f'    kandidatbygge: {time.time()-_tp:.2f}s · kandidater={len(candidates)}', flush=True)
    if not candidates:
        return None, {'error': 'inga kandidater'}

    try:
        raw_row_matrix = ns['_frame_row_matrix'](frame_rows, antal_matcher)
        row_matrix = _build_teckenskydd_bits(ns, raw_row_matrix, frame, antal_matcher)
    except Exception:
        row_matrix = tuple()

    initial = _State(
        hist_mask=(1 << htot) - 1, frame_mask=(1 << ftot) - 1,
        hist_size=htot, frame_size=ftot, chosen=tuple(), used_keys=frozenset(), steps=tuple(),
    )
    profile_cands = [c for c in candidates if _is_profile_candidate(c)]
    structure_cands = [c for c in candidates if _is_structure_candidate(c) and int(c.get('hist_hit', 0)) == int(htot)]
    trace = []

    _tp = time.time()
    structure_states = []
    if structure_cands and int(max_structure_filters) > 0:
        structure_states = _beam_expand(
            ns, [initial], structure_cands, target_hit=int(htot),
            target_rows=int(target_rows), target_min_rows=int(target_min_rows),
            frame_rows=frame_rows, frame=frame, antal_matcher=int(antal_matcher),
            row_matrix=row_matrix, min_step_profile=float(min_step_profile),
            min_step_structure=float(min_step_structure), absolute_min_step=float(absolute_min_step),
            min_value_filters=0, min_edge_filters=0, max_structure_filters=int(max_structure_filters),
            max_edge_auto_filters=int(max_edge_auto_filters), max_fat_abc_filters=int(max_fat_abc_filters),
            structure_max_reduction_pct=float(structure_max_reduction_pct),
            max_steps=min(int(max_structure_filters), int(max_filters)),
            beam_width=max(10, int(beam_width)), phase='trim',
            post_target_min_step=float(post_target_min_step), low_val_floor=0.0, low_val_min_step=0.0,
        )

    print(f'    strukturbeam: {time.time()-_tp:.2f}s · states={len(structure_states)}', flush=True)
    _tp = time.time()
    seeds = [initial] + _choose_structure_seeds(
        structure_states, target_rows=int(target_rows), target_min_rows=int(target_min_rows),
        max_seeds=max(5, int(beam_width) // 6),
    )
    dedup, seen = [], set()
    for seed in seeds:
        sig = tuple(sorted(seed.used_keys))
        if sig not in seen:
            seen.add(sig)
            dedup.append(seed)
    seeds = dedup

    if structure_states:
        best_struct = _choose_best_state(
            structure_states, target_rows=int(target_rows), target_min_rows=int(target_min_rows),
            min_value_filters=0, min_edge_filters=0, choose_policy='hit_then_rows',
        )
        if best_struct is not None:
            trace.append({'target': htot, 'fas': 'struktur_100', 'bästa_rader': best_struct.frame_count, 'bästa_träff': best_struct.hist_hit, 'filter': len(best_struct.chosen), 'struktur': best_struct.structure_filters})

    print(f'    strukturfrön: {time.time()-_tp:.2f}s · seeds={len(seeds)}', flush=True)
    _tp = time.time()
    all_final_states: List[_State] = [seed for seed in seeds if seed.chosen]
    # En exakt beam vid min_hit räcker: states behåller sina faktiska 28–30 träffar
    # och rangordnas med högre samlad träff inom samma radzon. Det kapar körtid ca 3x.
    for target in [int(min_hit_count)]:
        states_for_target = []
        for seed in seeds:
            remaining_steps = max(0, int(max_filters) - len(seed.chosen))
            if remaining_steps <= 0:
                continue
            states_for_target.extend(_beam_expand(
                ns, [seed], profile_cands, target_hit=int(target),
                target_rows=int(target_rows), target_min_rows=int(target_min_rows),
                frame_rows=frame_rows, frame=frame, antal_matcher=int(antal_matcher),
                row_matrix=row_matrix, min_step_profile=float(min_step_profile),
                min_step_structure=float(min_step_structure), absolute_min_step=float(absolute_min_step),
                min_value_filters=int(min_value_filters), min_edge_filters=int(min_edge_filters),
                max_structure_filters=int(max_structure_filters), max_edge_auto_filters=int(max_edge_auto_filters),
                max_fat_abc_filters=int(max_fat_abc_filters), structure_max_reduction_pct=float(structure_max_reduction_pct),
                max_steps=remaining_steps, beam_width=max(10, int(beam_width)), phase='profil',
                post_target_min_step=float(post_target_min_step), low_val_floor=float(low_val_floor),
                low_val_min_step=float(low_val_min_step),
            ))
        if states_for_target:
            states_for_target = _dedupe_states(
                states_for_target, target_rows=int(target_rows), target_min_rows=int(target_min_rows),
                target_hit=int(target), min_value_filters=int(min_value_filters),
                min_edge_filters=int(min_edge_filters), phase='profil', beam_width=int(beam_width),
            )
            best_t = _choose_best_state(
                states_for_target, target_rows=int(target_rows), target_min_rows=int(target_min_rows),
                min_value_filters=int(min_value_filters), min_edge_filters=int(min_edge_filters),
                choose_policy=str(choose_policy),
            )
            if best_t is not None:
                trace.append({'target': target, 'fas': 'profil_ovanpå_struktur', 'bästa_rader': best_t.frame_count, 'bästa_träff': best_t.hist_hit, 'filter': len(best_t.chosen), 'struktur': best_t.structure_filters, 'fat_abc': best_t.fat_filters, 'edge': best_t.edge_filters})
            all_final_states.extend(states_for_target)

    print(f'    profilbeam: {time.time()-_tp:.2f}s · finalstates={len(all_final_states)}', flush=True)
    best = _choose_best_state(
        all_final_states, target_rows=int(target_rows), target_min_rows=int(target_min_rows),
        min_value_filters=int(min_value_filters), min_edge_filters=int(min_edge_filters),
        choose_policy=str(choose_policy),
    )
    if best is None:
        return None, {'error': 'ingen state vald', 'kandidater': len(candidates), 'trace': trace}

    chosen = list(best.chosen)
    package = {
        'target': int(best.hist_hit),
        'target_label': f"V8c struktur100+profil {best.hist_hit}/{htot}",
        'hist_hit': int(best.hist_hit), 'hist_total': int(htot),
        'frame_start': int(ftot), 'frame_after': int(best.frame_count),
        'reduction_pct': 100.0 - 100.0 * best.frame_count / max(1, ftot),
        'num_filters': len(chosen), 'filters': chosen, 'steps': list(best.steps),
        'package_type': 'V8c struktur100 + målstyrd profilsynergi',
        'min_value_filters': int(min_value_filters), 'value_filters': int(best.value_filters),
        'fat_filters': int(best.fat_filters), 'favshock_filters': int(best.favshock_filters),
        'edge_filters': int(best.edge_filters), 'structure_filters': int(best.structure_filters),
        'meta': {
            'engine': 'v8c_structure100_targetband_synergy',
            'target_min_rows': int(target_min_rows), 'target_rows': int(target_rows),
            'min_hit_count': int(min_hit_count), 'absolute_min_step': float(absolute_min_step),
            'min_step_profile': float(min_step_profile), 'min_step_structure': float(min_step_structure),
            'max_filters': int(max_filters), 'max_structure_filters': int(max_structure_filters),
            'max_edge_auto_filters': int(max_edge_auto_filters), 'max_fat_abc_filters': int(max_fat_abc_filters),
            'structure_max_reduction_pct': float(structure_max_reduction_pct),
            'beam_width': int(beam_width), 'post_target_min_step': float(post_target_min_step),
            'low_val_floor': float(low_val_floor), 'low_val_min_step': float(low_val_min_step),
            'choose_policy': str(choose_policy), 'structure_safe_only': True,
            'candidates': int(len(candidates)), 'profile_candidates': int(len(profile_cands)),
            'structure_candidates': int(len(structure_cands)), 'structure_candidates_30_30': int(len(structure_cands)),
            'structure_seed_count': int(len(seeds)), 'trace': trace[:30],
        },
    }
    return package, package['meta']


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def _release_test_memory():
    """Frigör cykliska closure-/NumPy-objekt mellan Colab-testfall."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def _parse_frame_profile(text: str) -> Tuple[int, int, int]:
    try:
        spik, halv, hel = [int(x) for x in str(text).split('-')]
    except Exception:
        raise SystemExit('frame-profile måste vara t.ex. 4-6-3')
    if spik + halv + hel != 13:
        raise SystemExit('frame-profile måste summera till 13 matcher.')
    return spik, halv, hel


def _base_frame(spik: int, halv: int, hel: int):
    return [['1'] for _ in range(spik)] + [['1', 'X'] for _ in range(halv)] + [['1', 'X', '2'] for _ in range(hel)]


def _numeric_text_series(series: pd.Series) -> pd.Series:
    txt = series.astype(str).str.replace(r'[,.]00', '', regex=True)
    txt = txt.str.replace(r'[^\d-]', '', regex=True).replace('', '0')
    return pd.to_numeric(txt, errors='coerce').fillna(0)


def _prepare_supersvar_payout(global_db: pd.DataFrame, pay_max: int):
    """0 kr + 0 vinnare = ingen 13-vinnare, inte låg utdelning.

    För själva utdelningsurvalet placeras dessa omgångar exakt på vald maxgräns.
    Originalvärdet sparas; proxyn används bara för urvalet i detta backtest.
    """
    db = global_db.copy()
    winner_col = None
    for col in db.columns:
        clean = str(col).lower().replace(' ', '').replace('_', '').replace('-', '')
        if clean in {'vinnare13', '13vinnare'}:
            winner_col = col
            break
    if 'Payout' not in db.columns or winner_col is None:
        db['_BT_No13Winner'] = False
        return db, 0
    payout = pd.to_numeric(db['Payout'], errors='coerce').fillna(0)
    winners = _numeric_text_series(db[winner_col])
    no_winner = (payout == 0) & (winners == 0)
    db['_BT_Payout_Original'] = payout
    db['_BT_No13Winner'] = no_winner
    db.loc[no_winner, 'Payout'] = int(pay_max)
    return db, int(no_winner.sum())


def _prepare_db(ns: dict, global_db: pd.DataFrame, antal_matcher: int, pay_min: int, pay_max: int) -> pd.DataFrame:
    db = global_db.copy()
    if 'Payout' in db.columns:
        db = db[(pd.to_numeric(db['Payout'], errors='coerce').fillna(0) >= int(pay_min)) & (pd.to_numeric(db['Payout'], errors='coerce').fillna(0) <= int(pay_max))]
    norm = ns['normalize_single_row_text']
    db['_bt_row_ok'] = db['Correct_Row'].apply(lambda r: len(norm(r)) == int(antal_matcher))
    db['_bt_vec_ok'] = db['Prob_Vector'].apply(lambda v: isinstance(v, list) and len(v) == int(antal_matcher) * 3)
    db = db[db['_bt_row_ok'] & db['_bt_vec_ok']].copy()
    if 'Datum' in db.columns:
        try:
            db['_bt_date'] = pd.to_datetime(db['Datum'], errors='coerce')
            db = db.sort_values('_bt_date', ascending=False, na_position='last')
        except Exception:
            db = db.sort_index(ascending=False)
    else:
        db = db.sort_index(ascending=False)
    return db


def _run_backtest(ns: dict, global_db: pd.DataFrame, *, max_tests: int, test_offset: int, top_n: int, pay_min: int, pay_max: int, min_hit: int, target_rows: int, target_min_rows: int, min_step_profile: float, min_step_structure: float, absolute_min_step: float, max_filters: int, max_structure_filters: int, max_edge_auto_filters: int, max_fat_abc_filters: int, structure_max_reduction_pct: float, min_value_filters: int, min_edge_filters: int, filter_hist_target_pct: int, mode: str, frame_profile: str, beam_width: int, variants_per_key: int, max_candidates: int, structure_safe_only: bool, choose_policy: str, wide_n: int, min_val_pct: float, min_structure_val_pct: float, post_target_min_step: float = 0.0, low_val_floor: float = 0.0, low_val_min_step: float = 0.0) -> Tuple[pd.DataFrame, dict]:
    antal_matcher = 13
    norm = ns['normalize_single_row_text']
    similar_history = ns['_similar_history_for_backtest']
    ranked_frame = ns['_ranked_frame_like_widths']
    generate_rows_from_frame = ns['generate_rows_from_frame']
    build_clean_filter_specs = ns['build_clean_filter_specs']
    package_passes_row = ns['_package_passes_row']

    global_db, supersvar_count = _prepare_supersvar_payout(global_db, pay_max)
    db = _prepare_db(ns, global_db, antal_matcher, pay_min, pay_max)
    if db.empty:
        return pd.DataFrame(), {'error': 'Inga giltiga historiska omgångar.'}

    spik, halv, hel = _parse_frame_profile(frame_profile)
    frame_template = _base_frame(spik, halv, hel)
    test_rows = list(db.iterrows())[int(test_offset):int(test_offset)+int(max_tests)]
    rows = []
    skipped = 0
    t_all = time.time()

    for ti, (idx, test_row) in enumerate(test_rows, 1):
        _release_test_memory()
        t0 = time.time()
        correct = norm(test_row.get('Correct_Row', ''))
        input_vec = test_row.get('Prob_Vector', [])
        test_date = test_row.get('Datum', None)
        print(f'Testfall {ti}/{len(test_rows)} · {str(test_date)[:10]}', flush=True)

        try:
            t_stage = time.time()
            sim_df = similar_history(
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
            sim_wide_df = similar_history(
                global_db,
                input_vec,
                antal_matcher,
                top_n=max(int(top_n), int(wide_n)),
                pay_min=int(pay_min),
                pay_max=int(pay_max),
                exclude_index=idx,
                mode=mode,
                test_date=test_date,
            )
            print(f'  likhet: {time.time()-t_stage:.2f}s · N={len(sim_df)}/{len(sim_wide_df)}', flush=True)
            t_stage = time.time()
            if len(sim_df) < max(10, min(int(top_n), 20)):
                skipped += 1
                rows.append({
                    'Datum': str(test_date)[:10] if test_date is not None else str(idx),
                    'Status': 'Hoppad över',
                    'Orsak': f'För få liknande historiska omgångar ({len(sim_df)})',
                })
                continue

            engine_frame = ranked_frame(frame_template, input_vec, antal_matcher)
            frame_rows, _, ok, msg = generate_rows_from_frame(engine_frame, max_rows=None)
            if not ok or not frame_rows:
                skipped += 1
                rows.append({
                    'Datum': str(test_date)[:10] if test_date is not None else str(idx),
                    'Status': 'Hoppad över',
                    'Orsak': f'Kunde inte skapa backtestram: {msg}',
                })
                continue

            print(f'  ram: {time.time()-t_stage:.2f}s · rader={len(frame_rows)}', flush=True)
            t_stage = time.time()
            specs = build_clean_filter_specs(
                sim_df,
                input_vec,
                antal_matcher,
                slider_u_count=3,
                target_hist_pct=int(filter_hist_target_pct),
                u_rows=None,
                hist_df=global_db,
                max_shock_pct=22,
                candidate_rows=frame_rows,
                include_supermakro=True,
            )

            print(f'  filterspecs: {time.time()-t_stage:.2f}s · specs={len(specs)}', flush=True)
            t_stage = time.time()
            pkg, meta = _build_frequency_interval_synergy_package(
                ns,
                sim_df,
                specs,
                frame_rows,
                engine_frame,
                antal_matcher,
                min_hit_count=int(min_hit),
                target_rows=int(target_rows),
                target_min_rows=int(target_min_rows),
                max_filters=int(max_filters),
                absolute_min_step=float(absolute_min_step),
                max_structure_filters=int(max_structure_filters),
                max_edge_auto_filters=int(max_edge_auto_filters),
                max_fat_abc_filters=int(max_fat_abc_filters),
                structure_max_reduction_pct=float(structure_max_reduction_pct),
                min_value_filters=int(min_value_filters),
                min_edge_filters=int(min_edge_filters),
                min_step_profile=float(min_step_profile),
                min_step_structure=float(min_step_structure),
                beam_width=int(beam_width),
                variants_per_key=int(variants_per_key),
                max_candidates=int(max_candidates),
                frame_adapt=True,
                structure_safe_only=bool(structure_safe_only),
                choose_policy=str(choose_policy),
                validation_df=sim_wide_df,
                min_val_pct=float(min_val_pct),
                min_structure_val_pct=float(min_structure_val_pct),
                post_target_min_step=float(post_target_min_step),
                low_val_floor=float(low_val_floor),
                low_val_min_step=float(low_val_min_step),
            )
            print(f'  paketmotor: {time.time()-t_stage:.2f}s', flush=True)
            if pkg is None:
                rows.append({
                    'Datum': str(test_date)[:10] if test_date is not None else str(idx),
                    'Facit': correct,
                    'Status': 'OK',
                    'Paket klarar facit': 'Nej',
                    'Huvudresultat': 'Nej',
                    'Orsak': meta.get('error', 'Inget paket'),
                    'Liknande': int(len(sim_df)),
                    'Backtestram rader': int(len(frame_rows)),
                    'Sekunder': round(time.time() - t0, 2),
                })
                continue

            pkg_pass, fail_reason = package_passes_row(correct, specs, pkg)
            rows.append({
                'Datum': str(test_date)[:10] if test_date is not None else str(idx),
                'Facit': correct,
                'Status': 'OK',
                'Liknande': int(len(sim_df)),
                'Backtestram rader': int(len(frame_rows)),
                'Paket klarar facit': 'Ja' if pkg_pass else 'Nej',
                'Huvudresultat': 'Ja' if pkg_pass else 'Nej',
                'Paketträff': f"{int(pkg.get('hist_hit', 0))}/{int(pkg.get('hist_total', len(sim_df)))}",
                'Paketrader': int(pkg.get('frame_after', 0)),
                'I målband': 'Ja' if int(target_min_rows) <= int(pkg.get('frame_after', 0)) <= int(target_rows) else 'Nej',
                'Supersvår testomgång': 'Ja' if bool(test_row.get('_BT_No13Winner', False)) else 'Nej',
                'Reducerar %': round(float(pkg.get('reduction_pct', 0.0)), 1),
                'Pakettyp': pkg.get('package_type', 'Optimal'),
                'Orsak': 'OK' if pkg_pass else fail_reason,
                'Filter totalt': int(pkg.get('num_filters', 0)),
                'Värdefilter': int(pkg.get('value_filters', 0)),
                'FAT-filter': int(pkg.get('fat_filters', 0)),
                'Favorit/skräll-filter': int(pkg.get('favshock_filters', 0)),
                'Spretfilter': int(pkg.get('edge_filters', 0)),
                'Strukturfilter': int(pkg.get('structure_filters', 0)),
                'Kandidater': int((pkg.get('meta') or {}).get('candidates', 0)),
                'Profilkandidater': int((pkg.get('meta') or {}).get('profile_candidates', 0)),
                'Strukturkandidater': int((pkg.get('meta') or {}).get('structure_candidates', 0)),
                'Validering N': int(len(sim_wide_df)),
                'Valpolicy': str((pkg.get('meta') or {}).get('choose_policy', '')),
                'Struktur 30/30': bool((pkg.get('meta') or {}).get('structure_safe_only', True)),
                'Första filter': ' | '.join([f"{c.get('category','')}: {c.get('name','')} {c.get('interval_txt','')}" for c in list(pkg.get('filters', []) or [])[:4]]),
                'Sekunder': round(time.time() - t0, 2),
            })
        except Exception as e:
            rows.append({
                'Datum': str(test_date)[:10] if test_date is not None else str(idx),
                'Facit': correct,
                'Status': 'Fel',
                'Orsak': str(e)[:400],
                'Sekunder': round(time.time() - t0, 2),
            })

    detail = pd.DataFrame(rows)
    meta = {
        'tested_rows': int(max_tests),
        'skipped': int(skipped),
        'top_n': int(top_n),
        'pay_min': int(pay_min),
        'pay_max': int(pay_max),
        'mode': str(mode),
        'frame_profile': str(frame_profile),
        'target_rows': int(target_rows),
        'target_min_rows': int(target_min_rows),
        'supersvar_rows_in_pool': int(supersvar_count),
        'absolute_min_step': float(absolute_min_step),
        'max_structure_filters': int(max_structure_filters),
        'max_edge_auto_filters': int(max_edge_auto_filters),
        'max_fat_abc_filters': int(max_fat_abc_filters),
        'min_hit': int(min_hit),
        'min_step_profile': float(min_step_profile),
            'min_edge_filters': int(min_edge_filters),
        'min_step_structure': float(min_step_structure),
        'post_target_min_step': float(post_target_min_step),
        'low_val_floor': float(low_val_floor),
        'low_val_min_step': float(low_val_min_step),
        'beam_width': int(beam_width),
        'seconds_total': round(time.time() - t_all, 1),
    }
    return detail, meta


def _summarize_detail(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame()
    ok = detail_df[detail_df.get('Status', '') == 'OK'] if 'Status' in detail_df.columns else pd.DataFrame()
    tested = int(len(ok))
    hit = int((ok.get('Paket klarar facit', pd.Series(dtype=str)) == 'Ja').sum()) if tested else 0
    no_pkg = int(ok.get('Orsak', pd.Series(dtype=str)).astype(str).str.contains('Inget paket|inga kandidater|ingen state', na=False).sum()) if tested else 0
    paketrader = pd.to_numeric(ok.get('Paketrader', pd.Series(dtype=float)), errors='coerce') if tested else pd.Series(dtype=float)
    filt = pd.to_numeric(ok.get('Filter totalt', pd.Series(dtype=float)), errors='coerce') if tested else pd.Series(dtype=float)
    vfil = pd.to_numeric(ok.get('Värdefilter', pd.Series(dtype=float)), errors='coerce') if tested else pd.Series(dtype=float)
    ffil = pd.to_numeric(ok.get('FAT-filter', pd.Series(dtype=float)), errors='coerce') if tested else pd.Series(dtype=float)
    favfil = pd.to_numeric(ok.get('Favorit/skräll-filter', pd.Series(dtype=float)), errors='coerce') if tested else pd.Series(dtype=float)
    edgefil = pd.to_numeric(ok.get('Spretfilter', pd.Series(dtype=float)), errors='coerce') if tested else pd.Series(dtype=float)
    sfil = pd.to_numeric(ok.get('Strukturfilter', pd.Series(dtype=float)), errors='coerce') if tested else pd.Series(dtype=float)
    red = pd.to_numeric(ok.get('Reducerar %', pd.Series(dtype=float)), errors='coerce') if tested else pd.Series(dtype=float)
    band_hit = int((ok.get('I målband', pd.Series(dtype=str)) == 'Ja').sum()) if tested else 0
    hit_dist = {}
    for x in ok.get('Paketträff', pd.Series(dtype=str)).astype(str).tolist() if tested else []:
        try:
            h = int(str(x).split('/')[0])
            hit_dist[str(h)] = hit_dist.get(str(h), 0) + 1
        except Exception:
            pass
    row = {
        'Motor': 'Frekvensintervall/samverkan V8c',
        'Testade OK': tested,
        'Facit klarar paket': hit,
        'Facit %': round(100.0 * hit / max(1, tested), 1),
        'Median paketrader': int(paketrader.median()) if len(paketrader.dropna()) else None,
        'I målband': band_hit,
        'I målband %': round(100.0 * band_hit / max(1, tested), 1),
        'Medel paketrader': round(float(paketrader.mean()), 1) if len(paketrader.dropna()) else None,
        'Median reducering %': round(float(red.median()), 1) if len(red.dropna()) else None,
        'Median filter': round(float(filt.median()), 1) if len(filt.dropna()) else None,
        'Median värdefilter': round(float(vfil.median()), 1) if len(vfil.dropna()) else None,
        'Median FAT-filter': round(float(ffil.median()), 1) if len(ffil.dropna()) else None,
        'Median favorit/skräll-filter': round(float(favfil.median()), 1) if len(favfil.dropna()) else None,
        'Median spretfilter': round(float(edgefil.median()), 1) if len(edgefil.dropna()) else None,
        'Median strukturfilter': round(float(sfil.median()), 1) if len(sfil.dropna()) else None,
        'Inget paket': no_pkg,
        'Paketträff-fördelning': json.dumps(hit_dist, ensure_ascii=False, sort_keys=True),
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main – V3 tuner
# ---------------------------------------------------------------------------

def _profile_slug(label: str) -> str:
    s = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(label))
    while '__' in s:
        s = s.replace('__', '_')
    return s.strip('_')[:80] or 'profile'


def _build_test_profiles(args):
    """V8c: ett huvudläge eller tre små känslighetsprofiler."""
    base_common = dict(
        min_step_profile=float(args.min_step_profile),
        min_step_structure=float(args.min_step_structure),
        absolute_min_step=float(args.absolute_min_step),
        min_val_pct=float(args.min_val_pct),
        min_structure_val_pct=float(args.min_structure_val_pct),
        target_ratio=float(args.target_ratio),
        target_min_ratio=float(args.target_min_ratio),
        structure_safe_only=True, min_value_filters=int(args.min_value_filters),
        min_edge_filters=int(args.min_edge_filters), post_target_min_step=float(args.post_target_min_step),
        low_val_floor=float(args.low_val_floor), low_val_min_step=float(args.low_val_min_step),
        max_structure_filters=int(args.max_structure_filters),
        max_edge_auto_filters=int(args.max_edge_auto_filters),
        max_fat_abc_filters=int(args.max_fat_abc_filters),
        structure_max_reduction_pct=float(args.structure_max_reduction_pct),
    )
    if args.profiles == 'single':
        return [dict(
            label='V8c målband 1700–2000 struktur100 + profilsynergi',
            choose_policy=str(args.choose_policy), min_hit=int(args.min_hit), **base_common,
        )]

    p = dict(base_common)
    p['min_hit'] = 28 if int(args.top_n) >= 30 else max(1, int(args.top_n) - 2)
    profiles = [dict(label='V8c huvudmålband', choose_policy='target_band_then_hit', **p)]

    safer = dict(p)
    safer['min_hit'] = 29 if int(args.top_n) >= 30 else max(1, int(args.top_n) - 1)
    profiles.append(dict(label='V8c 29/30 säkrare', choose_policy='target_band_then_hit', **safer))

    fat4 = dict(p)
    fat4['max_fat_abc_filters'] = max(4, int(args.max_fat_abc_filters))
    fat4['max_edge_auto_filters'] = min(1, int(args.max_edge_auto_filters))
    profiles.append(dict(label='V8c FAT/ABC-bonus', choose_policy='target_band_then_hit', **fat4))
    return profiles

def _aggregate_worker_results(args, out_dir: Path, worker_dirs: List[Path]):
    details = []
    for wd in worker_dirs:
        p = wd / f'{args.output_prefix}_detail.csv'
        if p.exists():
            details.append(pd.read_csv(p))
    if not details:
        raise SystemExit('Inga worker-resultat kunde läsas.')

    combined_detail = pd.concat(details, ignore_index=True)
    summary_parts = []
    group_cols = [c for c in ['Motor', 'Policy', 'Min träff krav', 'Radmål min', 'Radmål max'] if c in combined_detail.columns]
    grouped = combined_detail.groupby(group_cols, dropna=False, sort=False) if group_cols else [((), combined_detail)]
    for key, group in grouped:
        summary = _summarize_detail(group)
        if summary.empty:
            continue
        values = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(group_cols, values))
        if 'Motor' in meta:
            summary.loc[:, 'Motor'] = meta['Motor']
        insert_at = 1
        for col in ['Policy', 'Min träff krav', 'Radmål min', 'Radmål max']:
            if col in meta:
                summary.insert(insert_at, col, meta[col])
                insert_at += 1
        summary_parts.append(summary)

    combined_summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    summary_path = out_dir / f'{args.output_prefix}_summary.csv'
    detail_path = out_dir / f'{args.output_prefix}_detail.csv'
    report_path = out_dir / f'{args.output_prefix}_report.txt'
    combined_summary.to_csv(summary_path, index=False)
    combined_detail.to_csv(detail_path, index=False)

    lines = [
        'FREKVENSINTERVALL/SAMVERKAN V8c – ISOLERADE TESTPROCESSER',
        '=' * 76,
        f'Testfall: {len(combined_detail)}',
        f'Worker-processer: {len(worker_dirs)}',
        'Varje testfall kördes i en ren Python-process för stabil Colab-körtid.',
        '',
        'SAMMANFATTNING',
        combined_summary.to_string(index=False) if not combined_summary.empty else '(tom)',
    ]
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – SAMMANFATTNING FRÅN ISOLERADE TESTPROCESSER', flush=True)
    print(combined_summary.to_string(index=False), flush=True)
    print(f'Rapport: {report_path}', flush=True)
    print(f'Summary: {summary_path}', flush=True)
    print(f'Detail: {detail_path}', flush=True)


def _run_fresh_test_workers(args, app_file: Path, db_file: Path, out_dir: Path):
    worker_root = out_dir / f'_{args.output_prefix}_workers'
    worker_root.mkdir(parents=True, exist_ok=True)
    worker_dirs = []

    def add(cmd, flag, value):
        cmd.extend([flag, str(value)])

    for i in range(int(args.max_tests)):
        offset = int(args.test_offset) + i
        wd = worker_root / f'case_{offset:03d}'
        wd.mkdir(parents=True, exist_ok=True)
        worker_dirs.append(wd)
        cmd = [sys.executable, str(Path(__file__).resolve())]
        add(cmd, '--app-file', app_file)
        add(cmd, '--db-file', db_file)
        add(cmd, '--out-dir', wd)
        add(cmd, '--output-prefix', args.output_prefix)
        add(cmd, '--max-tests', 1)
        add(cmd, '--test-offset', offset)
        add(cmd, '--profiles', args.profiles)
        add(cmd, '--top-n', args.top_n)
        add(cmd, '--wide-n', args.wide_n)
        add(cmd, '--pay-min', args.pay_min)
        add(cmd, '--pay-max', args.pay_max)
        add(cmd, '--min-hit', args.min_hit)
        add(cmd, '--target-rows', args.target_rows)
        add(cmd, '--target-min-rows', args.target_min_rows)
        add(cmd, '--target-ratio', args.target_ratio)
        add(cmd, '--target-min-ratio', args.target_min_ratio)
        add(cmd, '--choose-policy', args.choose_policy)
        add(cmd, '--min-val-pct', args.min_val_pct)
        add(cmd, '--min-structure-val-pct', args.min_structure_val_pct)
        add(cmd, '--post-target-min-step', args.post_target_min_step)
        add(cmd, '--low-val-floor', args.low_val_floor)
        add(cmd, '--low-val-min-step', args.low_val_min_step)
        add(cmd, '--max-filters', args.max_filters)
        add(cmd, '--absolute-min-step', args.absolute_min_step)
        add(cmd, '--max-structure-filters', args.max_structure_filters)
        add(cmd, '--max-edge-auto-filters', args.max_edge_auto_filters)
        add(cmd, '--max-fat-abc-filters', args.max_fat_abc_filters)
        add(cmd, '--structure-max-reduction-pct', args.structure_max_reduction_pct)
        add(cmd, '--min-value-filters', args.min_value_filters)
        add(cmd, '--min-edge-filters', args.min_edge_filters)
        add(cmd, '--min-step-profile', args.min_step_profile)
        add(cmd, '--min-step-structure', args.min_step_structure)
        add(cmd, '--filter-hist-target-pct', args.filter_hist_target_pct)
        add(cmd, '--mode', args.mode)
        add(cmd, '--frame-profile', args.frame_profile)
        add(cmd, '--beam-width', args.beam_width)
        add(cmd, '--variants-per-key', args.variants_per_key)
        add(cmd, '--max-candidates', args.max_candidates)
        cmd.append('--internal-worker')
        if args.allow_unsafe_structure:
            cmd.append('--allow-unsafe-structure')
        if args.fast_no_supermakro:
            cmd.append('--fast-no-supermakro')
        print('\n' + '=' * 76, flush=True)
        print(f'Isolerat testfall {i+1}/{args.max_tests} · offset {offset}', flush=True)
        print('=' * 76, flush=True)
        subprocess.run(cmd, check=True)

    _aggregate_worker_results(args, out_dir, worker_dirs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax(1).txt')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='frequency_interval_synergy_v8c_targetband')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0, help='Hoppa över detta antal senaste giltiga testfall.')
    parser.add_argument('--profiles', default='single', choices=['core', 'full', 'single'], help='core=3 V8-profiler. full=inkluderar 30/30-maxsäker. single=kör bara argumentens policy.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=45, help='Bredare valideringspool, normalt 60 när top-n är 30.')
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--min-hit', type=int, default=28, help='Lägsta tillåtna samlade historikträff för single/V3 30-radpress.')
    parser.add_argument('--target-rows', type=int, default=2000, help='Övre radmål efter paket. 0 = räkna från --target-ratio.')
    parser.add_argument('--target-min-rows', type=int, default=1700, help='Nedre radmål. Paket i intervallet target-min-rows..target-rows premieras.')
    parser.add_argument('--target-ratio', type=float, default=0.289, help='Övre radmål som andel av ramen när --target-rows=0.')
    parser.add_argument('--target-min-ratio', type=float, default=0.246, help='Nedre radmål som andel av ramen när --target-min-rows=0. 0.246 ≈ 1700/6912.')
    parser.add_argument('--choose-policy', default='target_band_then_hit', choices=['target_band_then_hit', 'hit_then_budget', 'budget_then_hit', 'hit_then_rows'], help='single-läge: hit_then_rows är V3 träff-först/radpress-policy.')
    parser.add_argument('--allow-unsafe-structure', action='store_true', help='Stäng av 30/30-kravet för struktur. Bör normalt INTE användas.')
    parser.add_argument('--min-val-pct', type=float, default=80.0, help='Min träff i bredare valideringspool för profilfilter.')
    parser.add_argument('--min-structure-val-pct', type=float, default=85.0, help='Min träff i bredare valideringspool för strukturfilter.')
    parser.add_argument('--post-target-min-step', type=float, default=5.0, help='V8: när paketet redan är på/under radmål krävs minst denna extra reducering för ytterligare filter.')
    parser.add_argument('--low-val-floor', type=float, default=75.0, help='V8: valideringsnivå under vilken filter betraktas som lågvaliderade.')
    parser.add_argument('--low-val-min-step', type=float, default=25.0, help='V8: lågvaliderade filter måste reducera minst detta per steg.')
    parser.add_argument('--max-filters', type=int, default=6)
    parser.add_argument('--absolute-min-step', type=float, default=5.0, help='Absolut minsta verkliga stegreduktion för alla auto-filter.')
    parser.add_argument('--max-structure-filters', type=int, default=2, help='Struktur är skyddsram: högst detta antal strukturfilter.')
    parser.add_argument('--max-edge-auto-filters', type=int, default=1, help='Högst detta antal värde/favorit/skräll-spretfilter.')
    parser.add_argument('--max-fat-abc-filters', type=int, default=4, help='Tillåt upp till fyra FAT/ABC-filter inom max-filters.')
    parser.add_argument('--structure-max-reduction-pct', type=float, default=35.0, help='Max samlad reducering från den 30/30-säkra strukturgrunden.')
    parser.add_argument('--min-value-filters', type=int, default=0, help='Minst detta antal Värde & svårighet-filter när det finns giltigt paket.')
    parser.add_argument('--min-edge-filters', type=int, default=0, help='Minst detta antal spretfilter: Värde & svårighet eller Favorit & skräll.')
    parser.add_argument('--min-step-profile', type=float, default=5.0, help='Profilfilter: minsta extra reducering per steg.')
    parser.add_argument('--min-step-structure', type=float, default=5.0, help='Strukturtrim: minsta extra reducering per steg.')
    parser.add_argument('--filter-hist-target-pct', type=int, default=90)
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out', 'kronologiskt'])
    parser.add_argument('--frame-profile', default='2-8-3', help='Format spikar-halvor-hel, t.ex. 4-6-3')
    parser.add_argument('--beam-width', type=int, default=12)
    parser.add_argument('--variants-per-key', type=int, default=2)
    parser.add_argument('--max-candidates', type=int, default=50)
    parser.add_argument('--fast-no-supermakro', action='store_true', default=True)
    parser.add_argument('--force-upload', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-fresh-workers', action='store_true', help='Kör flera testfall i samma process. Standard är isolerad process per testfall.')
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print('Ignorerar notebook-/miljöargument:', ' '.join(unknown_args), flush=True)

    app_file, db_file = _resolve_required_files(args)
    print(f'Appfil vald: {app_file}', flush=True)
    print(f'Databas vald: {db_file}', flush=True)

    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)

    if int(args.max_tests) > 1 and not bool(args.internal_worker) and not bool(args.no_fresh_workers):
        _run_fresh_test_workers(args, app_file, db_file, out_dir)
        return

    print('Laddar appfunktioner...', flush=True)
    ns = _load_app_functions(app_file, fast_no_supermakro=bool(args.fast_no_supermakro))
    print('Laddar historik...', flush=True)
    db = ns['load_database'](str(db_file), 13)
    print(f'Historik laddad: {len(db)} giltiga omgångar', flush=True)

    spik, halv, hel = _parse_frame_profile(str(args.frame_profile))
    frame_rows_count = (2 ** halv) * (3 ** hel)
    base_target_rows = int(args.target_rows) if int(args.target_rows) > 0 else max(1, int(round(frame_rows_count * float(args.target_ratio))))
    base_target_min_rows = int(args.target_min_rows) if int(args.target_min_rows) > 0 else max(1, int(round(frame_rows_count * float(args.target_min_ratio))))
    base_target_min_rows = min(base_target_min_rows, base_target_rows)
    args._frame_rows_count = frame_rows_count
    profiles = _build_test_profiles(args)

    print('\n' + '=' * 76, flush=True)
    print('Kör FREKVENSINTERVALL/SAMVERKAN V8c – struktur 100 % + målband + profilsynergi', flush=True)
    print(f'Backtestram: {args.frame_profile} = {frame_rows_count} rader', flush=True)
    print(f'Basradmål: {base_target_min_rows}–{base_target_rows} rader', flush=True)
    print(f'Profiler: {len(profiles)} · testfall/profil: {args.max_tests}', flush=True)
    print(f'Strukturfilter kräver 30/30: {not bool(args.allow_unsafe_structure)}', flush=True)
    print('=' * 76, flush=True)

    all_summaries = []
    all_details = []
    report_lines = []
    start_all = time.time()

    for pi, prof in enumerate(profiles, start=1):
        label = prof['label']
        target_rows = int(args.target_rows) if int(args.target_rows) > 0 else max(1, int(round(frame_rows_count * float(prof.get('target_ratio', args.target_ratio)))))
        target_min_rows = int(args.target_min_rows) if int(args.target_min_rows) > 0 else max(1, int(round(frame_rows_count * float(prof.get('target_min_ratio', args.target_min_ratio)))))
        target_min_rows = min(target_min_rows, target_rows)
        print('\n' + '-' * 76, flush=True)
        print(f'Profil {pi}/{len(profiles)}: {label}', flush=True)
        print(f"Policy {prof['choose_policy']} · min träff {prof['min_hit']}/30 · radmål {target_min_rows}–{target_rows} · minsteg {prof['absolute_min_step']}% · validering {prof['min_val_pct']}%", flush=True)
        print('-' * 76, flush=True)
        t0 = time.time()
        detail_df, meta = _run_backtest(
            ns,
            db,
            max_tests=int(args.max_tests),
            test_offset=int(args.test_offset),
            top_n=int(args.top_n),
            pay_min=int(args.pay_min),
            pay_max=int(args.pay_max),
            min_hit=int(prof['min_hit']),
            target_rows=int(target_rows),
            target_min_rows=int(target_min_rows),
            min_step_profile=float(prof['min_step_profile']),
            absolute_min_step=float(prof['absolute_min_step']),
            min_step_structure=float(prof['min_step_structure']),
            max_filters=int(args.max_filters),
            max_structure_filters=int(prof['max_structure_filters']),
            max_edge_auto_filters=int(prof['max_edge_auto_filters']),
            max_fat_abc_filters=int(prof['max_fat_abc_filters']),
            structure_max_reduction_pct=float(prof['structure_max_reduction_pct']),
            min_value_filters=int(prof.get('min_value_filters', args.min_value_filters)),
            min_edge_filters=int(prof.get('min_edge_filters', args.min_edge_filters)),
            filter_hist_target_pct=int(args.filter_hist_target_pct),
            mode=str(args.mode),
            frame_profile=str(args.frame_profile),
            beam_width=int(args.beam_width),
            variants_per_key=int(args.variants_per_key),
            max_candidates=int(args.max_candidates),
            structure_safe_only=bool(prof['structure_safe_only']),
            choose_policy=str(prof['choose_policy']),
            wide_n=int(args.wide_n),
            min_val_pct=float(prof['min_val_pct']),
            min_structure_val_pct=float(prof['min_structure_val_pct']),
            post_target_min_step=float(prof.get('post_target_min_step', args.post_target_min_step)),
            low_val_floor=float(prof.get('low_val_floor', args.low_val_floor)),
            low_val_min_step=float(prof.get('low_val_min_step', args.low_val_min_step)),
        )
        detail_df.insert(0, 'Motor', label)
        detail_df.insert(1, 'Policy', str(prof['choose_policy']))
        detail_df.insert(2, 'Min träff krav', int(prof['min_hit']))
        detail_df.insert(3, 'Radmål min', int(target_min_rows))
        detail_df.insert(4, 'Radmål max', int(target_rows))
        summary_df = _summarize_detail(detail_df)
        if not summary_df.empty:
            summary_df.loc[:, 'Motor'] = label
            summary_df.insert(1, 'Policy', str(prof['choose_policy']))
            summary_df.insert(2, 'Min träff krav', int(prof['min_hit']))
            summary_df.insert(3, 'Radmål min', int(target_min_rows))
            summary_df.insert(4, 'Radmål max', int(target_rows))
        all_summaries.append(summary_df)
        all_details.append(detail_df)
        slug = _profile_slug(label)
        detail_path = out_dir / f'{args.output_prefix}_{slug}_detail.csv'
        detail_df.to_csv(detail_path, index=False)
        print('\nDelresultat', flush=True)
        print(summary_df.to_string(index=False), flush=True)
        print(f'Detail: {detail_path}', flush=True)
        report_lines.append(f"{label}: {time.time()-t0:.1f} sekunder")

    combined_summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    combined_detail = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()

    summary_path = out_dir / f'{args.output_prefix}_summary.csv'
    detail_path = out_dir / f'{args.output_prefix}_detail.csv'
    report_path = out_dir / f'{args.output_prefix}_report.txt'
    combined_summary.to_csv(summary_path, index=False)
    combined_detail.to_csv(detail_path, index=False)

    lines = []
    lines.append('FREKVENSINTERVALL/SAMVERKAN V8c – MÅLBAND OCH EXAKT PAKETSAMVERKAN')
    lines.append('=' * 76)
    lines.append(f'Appbas: {app_file.name}')
    lines.append(f'Databas: {db_file.name}')
    lines.append(f'Historik: {len(db)} giltiga omgångar')
    lines.append(f'Utdelningsintervall: {args.pay_min:,} – {args.pay_max:,} kr'.replace(',', ' '))
    lines.append(f'Liknande historik: {args.top_n} + bred validering {args.wide_n}')
    lines.append(f'Testfall per profil: {args.max_tests}')
    lines.append(f'Profiler: {args.profiles}')
    lines.append(f'Breddmall: {args.frame_profile} = {frame_rows_count} rader')
    lines.append(f'Basradmål: {base_target_min_rows}–{base_target_rows} rader')
    lines.append(f'Struktur 30/30-krav: {not bool(args.allow_unsafe_structure)}')
    lines.append(f'Beam width: {args.beam_width}, varianter/filter: {args.variants_per_key}, max kandidater: {args.max_candidates}')
    lines.append(f'Super-Makro: {"avstängt" if args.fast_no_supermakro else "på / enligt appen"}')
    lines.append('')
    lines.append('V8c-FÖRBÄTTRING')
    lines.append('V8c använder ett radmålband, minst 5 % verklig stegreduktion, högst sex filter och exakt paketträff minst 28/30.')
    lines.append('Struktur är 30/30-säker skyddsram; FAT/ABC får stå för merparten av den aktiva reduceringen.')
    lines.append('')
    lines.append('SAMMANFATTNING')
    lines.append(combined_summary.to_string(index=False) if not combined_summary.empty else '(tom)')
    lines.append('')
    lines.append('KÖRTIDER')
    lines.extend(report_lines)
    lines.append(f'Total körtid: {time.time() - start_all:.1f} sekunder')
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print('\nKLART – SAMMANFATTNING', flush=True)
    print(combined_summary.to_string(index=False), flush=True)
    print(f'Rapport: {report_path}', flush=True)
    print(f'Summary: {summary_path}', flush=True)
    print(f'Detail: {detail_path}', flush=True)


# =============================================================================
# V9 OVERRIDE – KLUSTER/SPRET + UTDELNINGSRIKTNING + ETT SLUTPAKET
# =============================================================================

@dataclass
class _V9State:
    hist_mask: int
    val_mask: int
    frame_mask: int
    hist_size: int
    val_size: int
    frame_size: int
    chosen: Tuple[dict, ...]
    used_keys: frozenset
    steps: Tuple[dict, ...]

    @property
    def hist_hit(self) -> int:
        return int(int(self.hist_mask).bit_count())

    @property
    def val_hit(self) -> int:
        return int(int(self.val_mask).bit_count()) if self.val_size > 0 else 0

    @property
    def frame_count(self) -> int:
        return int(int(self.frame_mask).bit_count())

    @property
    def structure_filters(self) -> int:
        return sum(1 for c in self.chosen if _is_structure_candidate(c))

    @property
    def profile_filters(self) -> int:
        return sum(1 for c in self.chosen if _is_profile_candidate(c))

    @property
    def fat_filters(self) -> int:
        return sum(1 for c in self.chosen if _is_fat_abc_candidate(c))

    @property
    def edge_filters(self) -> int:
        return sum(1 for c in self.chosen if _is_edge_candidate(c))


def _v9_clean_payout_values(df: Optional[pd.DataFrame], n: int) -> np.ndarray:
    if not isinstance(df, pd.DataFrame) or n <= 0 or 'Payout' not in df.columns:
        return np.ones(max(0, int(n)), dtype=float)
    vals = pd.to_numeric(df['Payout'], errors='coerce').fillna(0).to_numpy(dtype=float)
    vals = vals[:int(n)]
    if len(vals) < int(n):
        vals = np.pad(vals, (0, int(n)-len(vals)), constant_values=0)
    vals = np.maximum(vals, 1.0)
    return vals


def _v9_robust_scale(values: np.ndarray, decimals: int = 0) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return 1.0 if int(decimals) <= 0 else 0.1
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    q25, q75 = np.percentile(vals, [25, 75])
    iqr_scale = float(q75 - q25) / 1.349 if q75 > q25 else 0.0
    gaps = np.diff(np.sort(np.unique(vals)))
    pos = gaps[gaps > 1e-12]
    gap_scale = float(np.median(pos)) if pos.size else 0.0
    floor = 1.0 if int(decimals) <= 0 else 10.0 ** (-max(1, int(decimals)))
    return max(floor, mad, iqr_scale, gap_scale, 1e-9)


def _v9_cluster_meta(values: np.ndarray, interval, decimals: int = 0) -> dict:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {'gap_score': 0.0, 'density_score': 0.0, 'excluded': 0, 'width_scaled': 0.0}
    lo, hi = float(interval[0]), float(interval[1])
    inside = vals[(vals >= lo) & (vals <= hi)]
    below = vals[vals < lo]
    above = vals[vals > hi]
    scale = _v9_robust_scale(vals, decimals)
    gaps = []
    if below.size:
        gaps.append(max(0.0, lo - float(np.max(below))) / scale)
    if above.size:
        gaps.append(max(0.0, float(np.min(above)) - hi) / scale)
    gap_score = min(gaps) if gaps else 0.0
    width_scaled = max(0.0, hi - lo) / scale
    density_score = float(inside.size) / max(1.0, 1.0 + width_scaled)
    return {
        'gap_score': float(gap_score),
        'density_score': float(density_score),
        'excluded': int(vals.size - inside.size),
        'width_scaled': float(width_scaled),
    }


def _v9_interval_variants(spec: dict, hist_arr: np.ndarray, min_hit: int) -> List[dict]:
    vals = np.asarray(hist_arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return []
    vals = np.sort(vals)
    n = int(vals.size)
    min_hit = max(1, min(int(min_hit), n))
    decimals = int(spec.get('decimals', 0) or 0)
    raw = []

    def add(lo, hi, source):
        try:
            lo, hi = float(lo), float(hi)
            if lo > hi:
                lo, hi = hi, lo
            raw.append({'interval': (lo, hi), 'source': source})
        except Exception:
            pass

    # Fullt säkert intervall.
    add(vals[0], vals[-1], 'full_100')

    # Täta fönster på exakt tillåtet träffgolv. Vid 29/30 ger detta både
    # lågtrim, högtrim och det smalaste huvudklustret utan att anta riktning.
    for k in sorted(set([n, min_hit]), reverse=True):
        if k <= 0 or k > n:
            continue
        for i in range(0, n-k+1):
            add(vals[i], vals[i+k-1], f'window_{k}_{i}')

    # Appens egna intervallvarianter läggs också in som komplettering.
    try:
        cov_floor = 100.0 * min_hit / max(1, n)
        coverages = [100, 99, 98, 97, 96, 95]
        coverages = [c for c in coverages if c + 1e-9 >= cov_floor]
        for iv in globals().get('_ACTIVE_V9_NS', {}).get('_candidate_intervals_for_spec', lambda *_a, **_k: [])(spec, coverages):
            interval = iv.get('interval')
            if interval is not None:
                add(interval[0], interval[1], f"app_{iv.get('coverage', '')}")
    except Exception:
        pass

    seen, out = set(), []
    for item in raw:
        lo, hi = item['interval']
        key = (round(lo, max(0, decimals)), round(hi, max(0, decimals)))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _v9_candidate_rank(c: dict):
    return (
        int(c.get('hist_hit', 0)),
        float(c.get('val_pct', 0.0)),
        float(c.get('red_pct', 0.0)),
        float(c.get('gap_score', 0.0)),
        float(c.get('payout_lift_pct', 0.0)),
        float(c.get('density_score', 0.0)),
        -int(c.get('frame_keep', 10**12)),
    )


def _build_dynamic_candidates_v9(
    ns: dict,
    specs: list,
    sim_df: pd.DataFrame,
    frame_rows: list,
    frame: list,
    antal_matcher: int,
    *,
    profile_min_hit: int = 29,
    variants_per_key: int = 4,
    max_candidates: int = 180,
    validation_df: Optional[pd.DataFrame] = None,
    min_candidate_val_pct: float = 90.0,
    min_structure_val_pct: float = 95.0,
    min_gap_score: float = 1.25,
    frame_adapt: bool = True,
):
    norm = ns['normalize_single_row_text']
    hist_rows = [norm(r) for r in list(sim_df['Correct_Row']) if len(norm(r)) == int(antal_matcher)]
    val_rows = []
    if isinstance(validation_df, pd.DataFrame) and 'Correct_Row' in validation_df.columns:
        val_rows = [norm(r) for r in list(validation_df['Correct_Row']) if len(norm(r)) == int(antal_matcher)]
    htot, vtot, ftot = len(hist_rows), len(val_rows), len(frame_rows)
    if htot <= 0 or ftot <= 0:
        return [], htot, vtot, ftot, np.ones(htot, dtype=float)

    hist_payout = _v9_clean_payout_values(sim_df, htot)
    all_log_median = float(np.median(np.log(hist_payout))) if hist_payout.size else 0.0
    candidates = []

    for spec in specs or []:
        if _is_blocked_auto_candidate(spec):
            continue
        getter = spec.get('getter')
        if getter is None:
            continue
        try:
            frame_vals = np.array([float(getter(r)) for r in frame_rows], dtype=float)
            hist_vals = list(spec.get('hist_values', []))
            hist_arr = np.array([np.nan if pd.isna(v) else float(v) for v in hist_vals], dtype=float)
            if len(hist_arr) != htot:
                continue
            val_arr = np.array([float(getter(r)) for r in val_rows], dtype=float) if vtot else np.ones(0, dtype=float)
        except Exception:
            continue

        is_structure = _is_structure_candidate(spec)
        indiv_floor = htot if is_structure else min(int(profile_min_hit), htot)
        spec_candidates = []
        for iv in _v9_interval_variants(spec, hist_arr, indiv_floor):
            interval = iv.get('interval')
            if interval is None:
                continue
            if frame_adapt and _fast_frame_capacity_block(ns, spec, interval, frame, frame_vals):
                continue
            lo, hi = float(interval[0]), float(interval[1])
            hist_mask = np.isfinite(hist_arr) & (hist_arr >= lo) & (hist_arr <= hi)
            frame_mask = np.isfinite(frame_vals) & (frame_vals >= lo) & (frame_vals <= hi)
            val_mask = (np.isfinite(val_arr) & (val_arr >= lo) & (val_arr <= hi)) if vtot else np.ones(0, dtype=bool)
            hist_hit = int(hist_mask.sum())
            val_hit = int(val_mask.sum()) if vtot else 0
            frame_keep = int(frame_mask.sum())
            if frame_keep <= 0 or frame_keep >= ftot:
                continue
            if hist_hit < indiv_floor:
                continue
            val_pct = 100.0 * val_hit / max(1, vtot) if vtot else 100.0
            if is_structure and vtot and val_pct + 1e-9 < float(min_structure_val_pct):
                continue
            if not is_structure and vtot and val_pct + 1e-9 < float(min_candidate_val_pct):
                continue

            cluster = _v9_cluster_meta(hist_arr, interval, int(spec.get('decimals', 0) or 0))
            # När ett historiskt värde tas bort ska det verkligen ligga separerat
            # från huvudklungan. 30/30-intervall behöver ingen gapkontroll.
            if hist_hit < htot and float(cluster['gap_score']) + 1e-12 < float(min_gap_score):
                continue

            kept_payout = hist_payout[hist_mask]
            removed_payout = hist_payout[~hist_mask]
            keep_log_med = float(np.median(np.log(kept_payout))) if kept_payout.size else all_log_median
            payout_lift_pct = 100.0 * (math.exp(keep_log_med - all_log_median) - 1.0)
            removed_median = float(np.median(removed_payout)) if removed_payout.size else float('nan')
            red_pct = 100.0 - 100.0 * frame_keep / max(1, ftot)
            c = {
                'key': spec.get('key'),
                'name': spec.get('name'),
                'category': spec.get('category', ''),
                'interval': interval,
                'interval_txt': _display_interval_local(interval, spec.get('decimals', 0)),
                'source': iv.get('source', ''),
                'hist_mask': hist_mask.astype(bool),
                'val_mask': val_mask.astype(bool),
                'frame_mask': frame_mask.astype(bool),
                'hist_hit': hist_hit,
                'hist_total': htot,
                'hist_pct': 100.0 * hist_hit / max(1, htot),
                'val_hit': val_hit,
                'val_total': vtot,
                'val_pct': val_pct,
                'frame_keep': frame_keep,
                'red_pct': red_pct,
                'gap_score': float(cluster['gap_score']),
                'density_score': float(cluster['density_score']),
                'excluded_hist': int(cluster['excluded']),
                'payout_lift_pct': float(payout_lift_pct),
                'kept_payout_median': float(np.median(kept_payout)) if kept_payout.size else 0.0,
                'removed_payout_median': removed_median,
            }
            spec_candidates.append(c)

        spec_candidates.sort(key=_v9_candidate_rank, reverse=True)
        candidates.extend(spec_candidates[:max(1, int(variants_per_key))])

    # Först minst en stark kandidat per filternyckel, därefter extra varianter.
    by_key: Dict[str, List[dict]] = {}
    for c in candidates:
        if c.get('key'):
            by_key.setdefault(str(c['key']), []).append(c)
    for vals in by_key.values():
        vals.sort(key=_v9_candidate_rank, reverse=True)
    first = [vals[0] for vals in by_key.values() if vals]
    first.sort(key=_v9_candidate_rank, reverse=True)
    selected = list(first[:int(max_candidates)])
    if len(selected) < int(max_candidates):
        extras = []
        for vals in by_key.values():
            extras.extend(vals[1:])
        extras.sort(key=_v9_candidate_rank, reverse=True)
        selected.extend(extras[:max(0, int(max_candidates)-len(selected))])

    out, seen = [], set()
    for c in selected:
        ident = (c.get('key'), c.get('interval_txt'))
        if ident in seen:
            continue
        seen.add(ident)
        c = dict(c)
        c['hist_bits'] = _bool_mask_to_bits(c['hist_mask'])
        c['val_bits'] = _bool_mask_to_bits(c['val_mask']) if vtot else 0
        c['frame_bits'] = _bool_mask_to_bits(c['frame_mask'])
        out.append(c)
    return out, htot, vtot, ftot, hist_payout


def _v9_state_metrics(st: _V9State, payout_values: np.ndarray, *, hit_power: float, val_power: float, reduction_power: float, payout_weight: float, cluster_weight: float) -> dict:
    hist_frac = st.hist_hit / max(1, st.hist_size)
    val_frac = st.val_hit / max(1, st.val_size) if st.val_size > 0 else 1.0
    red_frac = 1.0 - st.frame_count / max(1, st.frame_size)
    red_frac = max(red_frac, 1e-12)
    core = 100.0 * (hist_frac ** float(hit_power)) * (val_frac ** float(val_power)) * (red_frac ** float(reduction_power))

    payout_lift_pct = 0.0
    if payout_values.size and st.hist_size > 0:
        mask = np.array([(int(st.hist_mask) >> i) & 1 for i in range(st.hist_size)], dtype=bool)
        if mask.any():
            all_med = float(np.median(np.log(np.maximum(payout_values, 1.0))))
            keep_med = float(np.median(np.log(np.maximum(payout_values[mask], 1.0))))
            payout_lift_pct = 100.0 * (math.exp(keep_med - all_med) - 1.0)
    payout_bonus = float(payout_weight) * max(-50.0, min(50.0, payout_lift_pct))

    gap_values = [float(c.get('gap_score', 0.0)) for c in st.chosen if int(c.get('excluded_hist', 0)) > 0]
    cluster_mean = float(np.mean(gap_values)) if gap_values else 0.0
    cluster_bonus = float(cluster_weight) * min(5.0, max(0.0, cluster_mean))
    return {
        'joint_score': float(core + payout_bonus + cluster_bonus),
        'core_score': float(core),
        'hist_pct': 100.0 * hist_frac,
        'val_pct': 100.0 * val_frac,
        'reduction_pct': 100.0 * red_frac,
        'payout_lift_pct': float(payout_lift_pct),
        'cluster_mean': float(cluster_mean),
    }


def _v9_state_sort_key(st: _V9State, payout_values: np.ndarray, args) -> tuple:
    m = _v9_state_metrics(
        st, payout_values,
        hit_power=args.hit_power,
        val_power=args.validation_power,
        reduction_power=args.reduction_power,
        payout_weight=args.payout_weight,
        cluster_weight=args.cluster_weight,
    )
    return (
        m['joint_score'],
        st.hist_hit,
        st.val_hit,
        m['reduction_pct'],
        m['payout_lift_pct'],
        m['cluster_mean'],
        -st.frame_count,
        -len(st.chosen),
    )


def _v9_dedupe(states: List[_V9State], payout_values: np.ndarray, args, beam_width: int) -> List[_V9State]:
    best: Dict[Tuple[int, int, int], _V9State] = {}
    for st in states:
        sig = (int(st.hist_mask), int(st.val_mask), int(st.frame_mask))
        old = best.get(sig)
        if old is None or _v9_state_sort_key(st, payout_values, args) > _v9_state_sort_key(old, payout_values, args):
            best[sig] = st
    out = list(best.values())
    out.sort(key=lambda s: _v9_state_sort_key(s, payout_values, args), reverse=True)
    return out[:max(1, int(beam_width))]


def _v9_apply_candidate(st: _V9State, cand: dict, *, phase: str, hist_floor: int, val_floor_hit: int, sign_bits: tuple, min_unique_rows: int) -> Optional[_V9State]:
    key = cand.get('key')
    if not key or key in st.used_keys or _is_blocked_auto_candidate(cand):
        return None
    if phase == 'structure' and not _is_structure_candidate(cand):
        return None
    if phase == 'profile' and not _is_profile_candidate(cand):
        return None

    new_hist = int(st.hist_mask) & int(cand.get('hist_bits', 0))
    hist_hit = int(new_hist.bit_count())
    if hist_hit < int(hist_floor):
        return None
    new_val = int(st.val_mask) & int(cand.get('val_bits', 0)) if st.val_size > 0 else 0
    val_hit = int(new_val.bit_count()) if st.val_size > 0 else 0
    if st.val_size > 0 and val_hit < int(val_floor_hit):
        return None
    new_frame = int(st.frame_mask) & int(cand.get('frame_bits', 0))
    new_rows = int(new_frame.bit_count())
    removed = st.frame_count - new_rows
    if new_rows <= 0 or removed < max(1, int(min_unique_rows)):
        return None
    if sign_bits and any((new_frame & int(bits)) == 0 for bits in sign_bits):
        return None

    c2 = dict(cand)
    c2['step_removed_rows'] = int(removed)
    c2['step_red_pct'] = 100.0 * removed / max(1, st.frame_count)
    step = {
        'Filter': c2.get('name', ''),
        'Kategori': c2.get('category', ''),
        'Intervall': c2.get('interval_txt', '-'),
        'Efter filter': int(new_rows),
        'Borttagna unika rader': int(removed),
        'Stegreducering %': round(float(c2['step_red_pct']), 3),
        'Samlad historikträff': f'{hist_hit}/{st.hist_size}',
        'Samlad validering': f'{val_hit}/{st.val_size}' if st.val_size else '-',
        'Spret-gap': round(float(c2.get('gap_score', 0.0)), 3),
        'Utdelningslift %': round(float(c2.get('payout_lift_pct', 0.0)), 2),
        'Fas': phase,
    }
    return _V9State(
        hist_mask=new_hist,
        val_mask=new_val,
        frame_mask=new_frame,
        hist_size=st.hist_size,
        val_size=st.val_size,
        frame_size=st.frame_size,
        chosen=st.chosen + (c2,),
        used_keys=frozenset(set(st.used_keys) | {key}),
        steps=st.steps + (step,),
    )


def _v9_beam_search(start_states: List[_V9State], candidates: List[dict], *, phase: str, hist_floor: int, val_floor_hit: int, sign_bits: tuple, min_unique_rows: int, payout_values: np.ndarray, args) -> List[_V9State]:
    states = list(start_states)
    archive = _v9_dedupe(list(start_states), payout_values, args, max(args.beam_width, len(start_states)))
    unique_keys = {str(c.get('key')) for c in candidates if c.get('key')}
    # Ingen godtycklig filtergräns: varje filternyckel kan användas högst en gång,
    # så antalet unika nycklar är det matematiska maxdjupet.
    max_depth = len(unique_keys)
    for depth in range(max_depth):
        expanded = []
        for st in states:
            for cand in candidates:
                ns2 = _v9_apply_candidate(
                    st, cand,
                    phase=phase,
                    hist_floor=hist_floor,
                    val_floor_hit=val_floor_hit,
                    sign_bits=sign_bits,
                    min_unique_rows=min_unique_rows,
                )
                if ns2 is not None:
                    expanded.append(ns2)
        if not expanded:
            break
        states = _v9_dedupe(expanded, payout_values, args, args.beam_width)
        archive = _v9_dedupe(archive + states, payout_values, args, args.archive_width)
        best = states[0]
        print(f'      {phase} djup {depth+1}: states={len(states)} · filter={len(best.chosen)} · träff={best.hist_hit}/{best.hist_size} · val={best.val_hit}/{best.val_size if best.val_size else 0} · rader={best.frame_count}', flush=True)
    return archive


def _v9_redundancy_cleanup(st: _V9State, *, hist_floor: int, val_floor_hit: int, sign_bits: tuple) -> _V9State:
    chosen = list(st.chosen)
    changed = True
    while changed and chosen:
        changed = False
        for i in range(len(chosen)-1, -1, -1):
            trial = chosen[:i] + chosen[i+1:]
            hist_mask = (1 << st.hist_size) - 1
            val_mask = (1 << st.val_size) - 1 if st.val_size > 0 else 0
            frame_mask = (1 << st.frame_size) - 1
            for c in trial:
                hist_mask &= int(c.get('hist_bits', 0))
                if st.val_size > 0:
                    val_mask &= int(c.get('val_bits', 0))
                frame_mask &= int(c.get('frame_bits', 0))
            if frame_mask != st.frame_mask:
                continue
            if int(hist_mask.bit_count()) < int(hist_floor):
                continue
            if st.val_size > 0 and int(val_mask.bit_count()) < int(val_floor_hit):
                continue
            if sign_bits and any((frame_mask & int(bits)) == 0 for bits in sign_bits):
                continue
            chosen = trial
            changed = True
            break
    if len(chosen) == len(st.chosen):
        return st

    hist_mask = (1 << st.hist_size) - 1
    val_mask = (1 << st.val_size) - 1 if st.val_size > 0 else 0
    frame_mask = (1 << st.frame_size) - 1
    steps = []
    prev_rows = st.frame_size
    for c in chosen:
        hist_mask &= int(c.get('hist_bits', 0))
        if st.val_size > 0:
            val_mask &= int(c.get('val_bits', 0))
        frame_mask &= int(c.get('frame_bits', 0))
        rows = int(frame_mask.bit_count())
        steps.append({
            'Filter': c.get('name', ''), 'Kategori': c.get('category', ''),
            'Intervall': c.get('interval_txt', '-'), 'Efter filter': rows,
            'Borttagna unika rader': prev_rows-rows,
            'Samlad historikträff': f'{int(hist_mask.bit_count())}/{st.hist_size}',
            'Samlad validering': f'{int(val_mask.bit_count())}/{st.val_size}' if st.val_size else '-',
            'Fas': 'rensad',
        })
        prev_rows = rows
    return _V9State(
        hist_mask=hist_mask, val_mask=val_mask, frame_mask=frame_mask,
        hist_size=st.hist_size, val_size=st.val_size, frame_size=st.frame_size,
        chosen=tuple(chosen), used_keys=frozenset(c.get('key') for c in chosen), steps=tuple(steps),
    )


def _build_cluster_payout_package_v9(ns: dict, sim_df: pd.DataFrame, specs: list, frame_rows: list, frame: list, antal_matcher: int, validation_df: Optional[pd.DataFrame], args):
    global _ACTIVE_V9_NS
    _ACTIVE_V9_NS = ns
    candidates, htot, vtot, ftot, hist_payout = _build_dynamic_candidates_v9(
        ns, specs, sim_df, frame_rows, frame, antal_matcher,
        profile_min_hit=int(args.min_hit),
        variants_per_key=int(args.variants_per_key),
        max_candidates=int(args.max_candidates),
        validation_df=validation_df,
        min_candidate_val_pct=float(args.min_candidate_val_pct),
        min_structure_val_pct=float(args.min_structure_val_pct),
        min_gap_score=float(args.min_gap_score),
        frame_adapt=True,
    )
    print(f'    V9 kandidatbygge: kandidater={len(candidates)} · historik={htot} · validering={vtot}', flush=True)
    if not candidates:
        return None, {'error': 'Inga giltiga V9-kandidater.'}

    raw_row_matrix = ns['_frame_row_matrix'](frame_rows, antal_matcher)
    sign_bits = _build_teckenskydd_bits(ns, raw_row_matrix, frame, antal_matcher)
    initial = _V9State(
        hist_mask=(1 << htot)-1,
        val_mask=(1 << vtot)-1 if vtot else 0,
        frame_mask=(1 << ftot)-1,
        hist_size=htot, val_size=vtot, frame_size=ftot,
        chosen=tuple(), used_keys=frozenset(), steps=tuple(),
    )
    val_floor_profile = int(math.ceil(vtot * float(args.min_package_val_pct) / 100.0)) if vtot else 0
    val_floor_structure = int(math.ceil(vtot * float(args.min_structure_package_val_pct) / 100.0)) if vtot else 0

    structure_cands = [c for c in candidates if _is_structure_candidate(c) and int(c.get('hist_hit', 0)) == htot]
    profile_cands = [c for c in candidates if _is_profile_candidate(c) and int(c.get('hist_hit', 0)) >= int(args.min_hit)]

    structure_states = [initial]
    if structure_cands:
        structure_states = _v9_beam_search(
            [initial], structure_cands,
            phase='structure', hist_floor=htot,
            val_floor_hit=val_floor_structure,
            sign_bits=sign_bits, min_unique_rows=int(args.min_unique_rows),
            payout_values=hist_payout, args=args,
        )
    structure_states.sort(key=lambda s: _v9_state_sort_key(s, hist_payout, args), reverse=True)
    seeds = [initial]
    seen_masks = {(initial.hist_mask, initial.val_mask, initial.frame_mask)}
    for st in structure_states:
        sig = (st.hist_mask, st.val_mask, st.frame_mask)
        if sig in seen_masks:
            continue
        seen_masks.add(sig)
        seeds.append(st)
        if len(seeds) >= int(args.structure_seed_count):
            break
    print(f'    V9 strukturfrön: {len(seeds)} · strukturkandidater={len(structure_cands)}', flush=True)

    finals = []
    for si, seed in enumerate(seeds, 1):
        print(f'    Profilbeam från strukturfrö {si}/{len(seeds)} · startrader={seed.frame_count}', flush=True)
        finals.extend(_v9_beam_search(
            [seed], profile_cands,
            phase='profile', hist_floor=int(args.min_hit),
            val_floor_hit=val_floor_profile,
            sign_bits=sign_bits, min_unique_rows=int(args.min_unique_rows),
            payout_values=hist_payout, args=args,
        ))
    finals.extend([st for st in structure_states if st.chosen])
    valid = [st for st in finals if st.chosen and st.hist_hit >= int(args.min_hit) and (vtot == 0 or st.val_hit >= val_floor_profile)]
    if not valid:
        return None, {'error': 'Ingen V9-state klarade paketkraven.', 'candidates': len(candidates)}
    valid = _v9_dedupe(valid, hist_payout, args, args.archive_width)
    best = max(valid, key=lambda s: _v9_state_sort_key(s, hist_payout, args))
    best = _v9_redundancy_cleanup(best, hist_floor=int(args.min_hit), val_floor_hit=val_floor_profile, sign_bits=sign_bits)
    metrics = _v9_state_metrics(
        best, hist_payout,
        hit_power=args.hit_power, val_power=args.validation_power,
        reduction_power=args.reduction_power, payout_weight=args.payout_weight,
        cluster_weight=args.cluster_weight,
    )
    package = {
        'target': int(best.hist_hit),
        'target_label': f'V9 kluster+utdelning {best.hist_hit}/{htot}',
        'hist_hit': int(best.hist_hit), 'hist_total': int(htot),
        'val_hit': int(best.val_hit), 'val_total': int(vtot),
        'frame_start': int(ftot), 'frame_after': int(best.frame_count),
        'reduction_pct': float(metrics['reduction_pct']),
        'joint_score': float(metrics['joint_score']),
        'core_score': float(metrics['core_score']),
        'payout_lift_pct': float(metrics['payout_lift_pct']),
        'cluster_mean': float(metrics['cluster_mean']),
        'num_filters': len(best.chosen),
        'filters': list(best.chosen), 'steps': list(best.steps),
        'package_type': 'V9 struktur100 + kluster/spret + utdelningsriktning',
        'structure_filters': best.structure_filters,
        'profile_filters': best.profile_filters,
        'fat_filters': best.fat_filters,
        'edge_filters': best.edge_filters,
        'meta': {
            'engine': 'v9_cluster_payout_joint_synergy',
            'no_filter_cap': True,
            'min_hit': int(args.min_hit),
            'min_package_val_pct': float(args.min_package_val_pct),
            'min_unique_rows': int(args.min_unique_rows),
            'candidate_count': len(candidates),
            'structure_candidates': len(structure_cands),
            'profile_candidates': len(profile_cands),
            'structure_seeds': len(seeds),
            'beam_width': int(args.beam_width),
            'archive_width': int(args.archive_width),
            'min_gap_score': float(args.min_gap_score),
        },
    }
    return package, package['meta']


def _v9_json_filters(pkg: dict) -> str:
    rows = []
    for c in pkg.get('filters', []) or []:
        rows.append({
            'key': c.get('key'), 'name': c.get('name'), 'category': c.get('category'),
            'interval': c.get('interval_txt'), 'hist': f"{c.get('hist_hit')}/{c.get('hist_total')}",
            'val_pct': round(float(c.get('val_pct', 0.0)), 1),
            'frame_red_pct_alone': round(float(c.get('red_pct', 0.0)), 1),
            'step_removed_rows': int(c.get('step_removed_rows', 0)),
            'gap_score': round(float(c.get('gap_score', 0.0)), 3),
            'payout_lift_pct': round(float(c.get('payout_lift_pct', 0.0)), 2),
        })
    return json.dumps(rows, ensure_ascii=False)


def _run_backtest_v9(ns: dict, global_db: pd.DataFrame, args) -> Tuple[pd.DataFrame, dict]:
    antal_matcher = 13
    norm = ns['normalize_single_row_text']
    similar_history = ns['_similar_history_for_backtest']
    ranked_frame = ns['_ranked_frame_like_widths']
    generate_rows_from_frame = ns['generate_rows_from_frame']
    build_clean_filter_specs = ns['build_clean_filter_specs']
    package_passes_row = ns['_package_passes_row']

    global_db, supersvar_count = _prepare_supersvar_payout(global_db, int(args.pay_max))
    db = _prepare_db(ns, global_db, antal_matcher, int(args.pay_min), int(args.pay_max))
    if db.empty:
        return pd.DataFrame(), {'error': 'Inga giltiga omgångar i utdelningsintervallet.'}

    spik, halv, hel = _parse_frame_profile(args.frame_profile)
    template = _base_frame(spik, halv, hel)
    test_rows = list(db.iterrows())[int(args.test_offset):int(args.test_offset)+int(args.max_tests)]
    out = []
    for ti, (idx, test_row) in enumerate(test_rows, 1):
        _release_test_memory()
        t0 = time.time()
        correct = norm(test_row.get('Correct_Row', ''))
        input_vec = test_row.get('Prob_Vector', [])
        test_date = test_row.get('Datum', None)
        payout = float(pd.to_numeric(pd.Series([test_row.get('Payout', 0)]), errors='coerce').fillna(0).iloc[0])
        print(f'V9 testfall {ti}/{len(test_rows)} · {str(test_date)[:10]} · utdelning {payout:,.0f} kr'.replace(',', ' '), flush=True)
        try:
            sim_df = similar_history(
                global_db, input_vec, antal_matcher,
                top_n=int(args.top_n), pay_min=int(args.pay_min), pay_max=int(args.pay_max),
                exclude_index=idx, mode=str(args.mode), test_date=test_date,
            )
            wide_df = similar_history(
                global_db, input_vec, antal_matcher,
                top_n=max(int(args.top_n), int(args.wide_n)), pay_min=int(args.pay_min), pay_max=int(args.pay_max),
                exclude_index=idx, mode=str(args.mode), test_date=test_date,
            )
            if len(sim_df) < int(args.top_n):
                raise RuntimeError(f'För få liknande omgångar: {len(sim_df)} av önskade {args.top_n}.')
            engine_frame = ranked_frame(template, input_vec, antal_matcher)
            frame_rows, _, ok, msg = generate_rows_from_frame(engine_frame, max_rows=None)
            if not ok or not frame_rows:
                raise RuntimeError(f'Kunde inte skapa ram: {msg}')
            specs = build_clean_filter_specs(
                sim_df, input_vec, antal_matcher,
                slider_u_count=3, target_hist_pct=int(args.filter_hist_target_pct),
                u_rows=None, hist_df=global_db, max_shock_pct=22,
                candidate_rows=frame_rows, include_supermakro=not bool(args.fast_no_supermakro),
            )
            pkg, meta = _build_cluster_payout_package_v9(
                ns, sim_df, specs, frame_rows, engine_frame, antal_matcher, wide_df, args,
            )
            if pkg is None:
                raise RuntimeError(meta.get('error', 'Inget paket'))
            pkg_pass, fail_reason = package_passes_row(correct, specs, pkg)
            out.append({
                'Datum': str(test_date)[:10],
                'Utdelning': int(round(payout)),
                'Facit': correct,
                'Status': 'OK',
                'Paket klarar facit': 'Ja' if pkg_pass else 'Nej',
                'Orsak': 'OK' if pkg_pass else fail_reason,
                'Liknande historik': len(sim_df),
                'Bred validering': len(wide_df),
                'Paketträff': f"{pkg['hist_hit']}/{pkg['hist_total']}",
                'Valideringsträff': f"{pkg['val_hit']}/{pkg['val_total']}",
                'Grundram rader': len(frame_rows),
                'Paketrader': int(pkg['frame_after']),
                'Reducerar %': round(float(pkg['reduction_pct']), 2),
                'Gemensamt score': round(float(pkg['joint_score']), 4),
                'Utdelningslift %': round(float(pkg['payout_lift_pct']), 2),
                'Medel spret-gap': round(float(pkg['cluster_mean']), 3),
                'Filter totalt': int(pkg['num_filters']),
                'Strukturfilter': int(pkg['structure_filters']),
                'Profilfilter': int(pkg['profile_filters']),
                'FAT/ABC-filter': int(pkg['fat_filters']),
                'Värde/favorit/skräll': int(pkg['edge_filters']),
                'Filter JSON': _v9_json_filters(pkg),
                'Steg JSON': json.dumps(pkg.get('steps', []), ensure_ascii=False),
                'Supersvår testomgång': 'Ja' if bool(test_row.get('_BT_No13Winner', False)) else 'Nej',
                'Sekunder': round(time.time()-t0, 2),
            })
        except Exception as e:
            out.append({
                'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct,
                'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e),
                'Sekunder': round(time.time()-t0, 2),
            })
            print(f'  FEL: {e}', flush=True)
    return pd.DataFrame(out), {'supersvar_count': supersvar_count, 'eligible_rounds': len(db)}


def _summarize_v9(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    ok = detail[detail['Status'].astype(str) == 'OK'].copy() if 'Status' in detail.columns else detail.copy()
    hits = int((ok.get('Paket klarar facit', pd.Series(dtype=str)) == 'Ja').sum()) if not ok.empty else 0
    n = len(ok)
    rows = pd.to_numeric(ok.get('Paketrader'), errors='coerce').dropna() if not ok.empty else pd.Series(dtype=float)
    red = pd.to_numeric(ok.get('Reducerar %'), errors='coerce').dropna() if not ok.empty else pd.Series(dtype=float)
    filters = pd.to_numeric(ok.get('Filter totalt'), errors='coerce').dropna() if not ok.empty else pd.Series(dtype=float)
    lift = pd.to_numeric(ok.get('Utdelningslift %'), errors='coerce').dropna() if not ok.empty else pd.Series(dtype=float)
    return pd.DataFrame([{
        'Testade omgångar': n,
        'Träffar': hits,
        'Träff %': round(100.0*hits/max(1,n), 1),
        'Median paketrader': round(float(rows.median()), 1) if len(rows) else None,
        'Medel paketrader': round(float(rows.mean()), 1) if len(rows) else None,
        'Median reducering %': round(float(red.median()), 1) if len(red) else None,
        'Median filter': round(float(filters.median()), 1) if len(filters) else None,
        'Max filter': int(filters.max()) if len(filters) else None,
        'Median utdelningslift %': round(float(lift.median()), 2) if len(lift) else None,
        'Fel/hoppade': int(len(detail)-n),
    }])


def _write_v9_outputs(detail: pd.DataFrame, args, out_dir: Path, app_file: Path, db_file: Path, worker_count: int = 0):
    summary = _summarize_v9(detail)
    detail_path = out_dir / f'{args.output_prefix}_detail.csv'
    summary_path = out_dir / f'{args.output_prefix}_summary.csv'
    report_path = out_dir / f'{args.output_prefix}_report.txt'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    lines = [
        'TIPSET AI – V9 KLUSTER/SPRET + UTDELNINGSRIKTNING',
        '='*76,
        f'Appbas: {app_file.name}',
        f'Databas: {db_file.name}',
        f'Testomgångar: {args.max_tests}',
        f'Utdelningsintervall: {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '),
        f'Liknande historik: {args.top_n}; bred validering: {args.wide_n}',
        f'Strukturpaket: 100 % på {args.top_n} liknande omgångar',
        f'Övriga individuella filter och slutpaket: minst {args.min_hit}/{args.top_n}',
        f'Gemensam bred validering: minst {args.min_package_val_pct:.1f} %',
        'Max antal filter: ingen hård gräns',
        f'Minsta marginalbidrag: {args.min_unique_rows} unik rad',
        f'Minsta spret-gap: {args.min_gap_score}',
        f'Beam width: {args.beam_width}; arkiv: {args.archive_width}; kandidater: {args.max_candidates}',
        f'Isolerade worker-processer: {worker_count}',
        '', 'SAMMANFATTNING',
        summary.to_string(index=False) if not summary.empty else '(tom)',
        '',
        'MOTORPRINCIP',
        'Ett enda paket väljs genom ett gemensamt score där historikträff, bred validering och exakt reducering multipliceras.',
        'Utdelningsriktning och tydligt spret-gap används som mindre preferenser, inte som ersättning för träffsäkerhet.',
        'Alla radbeslut räknas exakt på hela grundramen; ingen sampling används.',
    ]
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V9 SAMMANFATTNING', flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f'Rapport: {report_path}', flush=True)
    print(f'Summary: {summary_path}', flush=True)
    print(f'Detail: {detail_path}', flush=True)



def _resolve_v9_script_path() -> Optional[Path]:
    """Hitta själva V9-runnerfilen även när Jupyter saknar ``__file__``.

    ``%run`` brukar sätta ``__file__``, men vissa Colab/Jupyter-sätt att köra
    koden gör inte det. Worker-processerna behöver en verklig .py-fil att
    starta, så vi söker även i aktuell katalog, /content och /mnt/data.
    """
    candidates = []

    raw_file = globals().get('__file__')
    if raw_file:
        candidates.append(Path(str(raw_file)))

    try:
        argv0 = Path(str(sys.argv[0]))
        if argv0.suffix.lower() == '.py':
            candidates.append(argv0)
    except Exception:
        pass

    preferred_names = [
        'package_cluster_payout_synergy_v9_colab_upload.py',
        'package_cluster_payout_synergy_v9b_colab_upload.py',
        'package_cluster_payout_synergy_v9_fixed_colab_upload.py',
    ]
    for directory in [Path.cwd(), Path('/content'), Path('/mnt/data')]:
        if not directory.exists():
            continue
        for name in preferred_names:
            candidates.append(directory / name)
        candidates.extend(sorted(directory.glob('package_cluster_payout_synergy_v9*.py')))

    seen = set()
    valid = []
    for candidate in candidates:
        try:
            candidate = Path(candidate)
            if not candidate.exists() or candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if str(resolved) in seen:
                continue
            seen.add(str(resolved))
            sample = resolved.read_text(encoding='utf-8', errors='replace')
            if ('def main_v9' not in sample or
                    'def _run_fresh_workers_v9' not in sample or
                    'def _resolve_v9_script_path' not in sample):
                continue
            score = 0
            name = resolved.name.lower()
            if name == 'package_cluster_payout_synergy_v9_colab_upload.py':
                score += 100
            if '/content/' in str(resolved):
                score += 30
            if 'fixed' in name or 'v9b' in name:
                score += 10
            valid.append((score, resolved.stat().st_mtime, resolved))
        except Exception:
            continue

    if not valid:
        return None
    valid.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return valid[0][2]


def _run_v9_workers_inline(args, app_file: Path, db_file: Path, out_dir: Path):
    """Reservväg när koden körts som en notebook-cell utan tillgänglig .py-fil.

    Beräkningarna är fortfarande exakta. Skillnaden är endast att testfallen
    körs sekventiellt i samma notebook-process i stället för i delprocesser.
    """
    root = out_dir / f'_{args.output_prefix}_workers_inline'
    root.mkdir(parents=True, exist_ok=True)
    details = []
    for i in range(int(args.max_tests)):
        offset = int(args.test_offset) + i
        print('\n' + '=' * 76, flush=True)
        print(f'V9 sekventiellt testfall {i+1}/{args.max_tests} · offset {offset}', flush=True)
        print('=' * 76, flush=True)

        worker_args = argparse.Namespace(**vars(args))
        worker_args.max_tests = 1
        worker_args.test_offset = offset
        worker_args.internal_worker = True
        worker_args.no_fresh_workers = True

        ns = _load_app_functions(app_file, fast_no_supermakro=False)
        db = ns['load_database'](str(db_file), 13)
        detail, _meta = _run_backtest_v9(ns, db, worker_args)
        if isinstance(detail, pd.DataFrame) and not detail.empty:
            details.append(detail)
        del ns, db, detail
        gc.collect()

    if not details:
        raise SystemExit('Inga V9-resultat skapades i den sekventiella reservkörningen.')
    combined = pd.concat(details, ignore_index=True)
    _write_v9_outputs(combined, args, out_dir, app_file, db_file, worker_count=0)

def _run_fresh_workers_v9(args, app_file: Path, db_file: Path, out_dir: Path):
    script_path = _resolve_v9_script_path()
    if script_path is None:
        print(
            'Varning: Jupyter saknar __file__ och ingen uppladdad V9 .py-fil hittades. '
            'Kör testfallen sekventiellt med exakt beräkning.',
            flush=True,
        )
        _run_v9_workers_inline(args, app_file, db_file, out_dir)
        return

    print(f'Worker-runner: {script_path}', flush=True)
    root = out_dir / f'_{args.output_prefix}_workers'
    root.mkdir(parents=True, exist_ok=True)
    worker_dirs = []
    pass_flags = [
        'top_n','wide_n','pay_min','pay_max','min_hit','min_candidate_val_pct',
        'min_package_val_pct','min_structure_val_pct','min_structure_package_val_pct',
        'min_gap_score','min_unique_rows','filter_hist_target_pct','frame_profile',
        'beam_width','archive_width','structure_seed_count','variants_per_key','max_candidates',
        'hit_power','validation_power','reduction_power','payout_weight','cluster_weight','mode',
    ]
    for i in range(int(args.max_tests)):
        offset = int(args.test_offset)+i
        wd = root / f'case_{offset:03d}'
        wd.mkdir(parents=True, exist_ok=True)
        worker_dirs.append(wd)
        cmd = [sys.executable, str(script_path),
               '--app-file', str(app_file), '--db-file', str(db_file),
               '--out-dir', str(wd), '--output-prefix', args.output_prefix,
               '--max-tests', '1', '--test-offset', str(offset), '--internal-worker']
        flag_map = {
            'top_n':'--top-n','wide_n':'--wide-n','pay_min':'--pay-min','pay_max':'--pay-max',
            'min_hit':'--min-hit','min_candidate_val_pct':'--min-candidate-val-pct',
            'min_package_val_pct':'--min-package-val-pct','min_structure_val_pct':'--min-structure-val-pct',
            'min_structure_package_val_pct':'--min-structure-package-val-pct','min_gap_score':'--min-gap-score',
            'min_unique_rows':'--min-unique-rows','filter_hist_target_pct':'--filter-hist-target-pct',
            'frame_profile':'--frame-profile','beam_width':'--beam-width','archive_width':'--archive-width',
            'structure_seed_count':'--structure-seed-count','variants_per_key':'--variants-per-key',
            'max_candidates':'--max-candidates','hit_power':'--hit-power','validation_power':'--validation-power',
            'reduction_power':'--reduction-power','payout_weight':'--payout-weight','cluster_weight':'--cluster-weight',
            'mode':'--mode',
        }
        for name in pass_flags:
            cmd.extend([flag_map[name], str(getattr(args, name))])
        if args.fast_no_supermakro:
            cmd.append('--fast-no-supermakro')
        print('\n'+'='*76, flush=True)
        print(f'V9 isolerat testfall {i+1}/{args.max_tests} · offset {offset}', flush=True)
        print('='*76, flush=True)
        subprocess.run(cmd, check=True)

    details = []
    for wd in worker_dirs:
        p = wd / f'{args.output_prefix}_detail.csv'
        if p.exists():
            details.append(pd.read_csv(p))
    if not details:
        raise SystemExit('Inga V9-workerresultat hittades.')
    combined = pd.concat(details, ignore_index=True)
    _write_v9_outputs(combined, args, out_dir, app_file, db_file, worker_count=len(worker_dirs))


def main_v9():
    parser = argparse.ArgumentParser(description='Tipset AI V9 – kluster/spret, utdelningsriktning och exakt paketsamverkan.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_cluster_payout_synergy_v9_test10')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--min-hit', type=int, default=29)
    parser.add_argument('--min-candidate-val-pct', type=float, default=90.0)
    parser.add_argument('--min-package-val-pct', type=float, default=90.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=95.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=95.0)
    parser.add_argument('--min-gap-score', type=float, default=1.25)
    parser.add_argument('--min-unique-rows', type=int, default=1)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='2-8-3')
    parser.add_argument('--beam-width', type=int, default=80)
    parser.add_argument('--archive-width', type=int, default=400)
    parser.add_argument('--structure-seed-count', type=int, default=12)
    parser.add_argument('--variants-per-key', type=int, default=4)
    parser.add_argument('--max-candidates', type=int, default=180)
    parser.add_argument('--hit-power', type=float, default=3.0)
    parser.add_argument('--validation-power', type=float, default=1.5)
    parser.add_argument('--reduction-power', type=float, default=1.0)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.25)
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--fast-no-supermakro', action='store_true', default=True)
    parser.add_argument('--force-upload', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-fresh-workers', action='store_true')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)

    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)

    if args.max_tests > 1 and not args.internal_worker and not args.no_fresh_workers:
        _run_fresh_workers_v9(args, app_file, db_file, out_dir)
        return

    ns = _load_app_functions(app_file, fast_no_supermakro=False)
    db = ns['load_database'](str(db_file), 13)
    print('\n'+'='*76, flush=True)
    print('TIPSET AI V9 – ETT PAKET · EXAKT SAMVERKAN · KLUSTER/SPRET · UTDELNINGSRIKTNING', flush=True)
    print(f'Test: {args.max_tests} omgångar inom {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '), flush=True)
    print(f'Struktur: 30/30 · profil/slutpaket: minst {args.min_hit}/30 · inget filtertak', flush=True)
    print(f'Ram: {args.frame_profile} · beam {args.beam_width} · kandidater {args.max_candidates}', flush=True)
    print('='*76, flush=True)
    detail, meta = _run_backtest_v9(ns, db, args)
    _write_v9_outputs(detail, args, out_dir, app_file, db_file, worker_count=1 if args.internal_worker else 0)




# =============================================================================
# V13 OVERRIDE – V9-PLUS KONTROLLERAD VARIANTOPTIMERING
# =============================================================================

V13_VARIANT_DEFS = {
    'A': {
        'label': 'A V9-baslinje',
        'fixed_payout': False,
        'bundle_search': False,
        'description': 'Oförändrad V9-logik som kontroll.',
    },
    'B': {
        'label': 'B V9 + rättad utdelningsriktning',
        'fixed_payout': True,
        'bundle_search': False,
        'description': 'V9-sökning men utdelningsriktning räknas på uteslutna historikomgångar.',
    },
    'C': {
        'label': 'C V9 + paket-/parsynergi',
        'fixed_payout': False,
        'bundle_search': True,
        'description': 'V9 med singlar plus selektiva filterpar/tripplar i beam-expansionen.',
    },
    'D': {
        'label': 'D V9 + utdelning + paket-/parsynergi',
        'fixed_payout': True,
        'bundle_search': True,
        'description': 'Kombinerar rättad utdelningsriktning med selektiv paket-/parsynergi.',
    },
}


def _v13_parse_variants(value: str) -> List[str]:
    raw = str(value or 'A,B,C,D').replace(';', ',').replace('+', ',').split(',')
    out = []
    for x in raw:
        k = x.strip().upper()
        if not k:
            continue
        if k not in V13_VARIANT_DEFS:
            raise ValueError(f'Okänd V13-variant: {k}. Tillåtna: {", ".join(V13_VARIANT_DEFS)}')
        if k not in out:
            out.append(k)
    return out or ['A', 'B', 'C', 'D']


def _v13_bits_to_bool(mask_bits: int, n: int) -> np.ndarray:
    return np.array([(int(mask_bits) >> i) & 1 for i in range(int(n))], dtype=bool)


def _v13_payout_direction_from_removed(keep_bits: int, n: int, payout_values: np.ndarray, *, clip: float = 150.0) -> dict:
    """Riktning från uteslutna historikomgångar.

    Positivt värde betyder att paketet/intervallet främst tog bort lägre
    utdelningar än klungan som helhet. Negativt värde betyder att det främst
    tog bort högre/supersvåra omgångar. Det här reagerar även när bara 1 av
    30 omgångar tas bort, vilket medianbaserad lift inte gjorde.
    """
    n = int(n)
    if n <= 0 or payout_values is None or len(payout_values) <= 0:
        return {'direction_pct': 0.0, 'removed_count': 0, 'removed_mean': 0.0, 'kept_mean': 0.0}
    vals = np.asarray(payout_values[:n], dtype=float)
    vals = np.maximum(vals, 1.0)
    keep = _v13_bits_to_bool(int(keep_bits), n)
    if keep.size != vals.size:
        keep = keep[:vals.size]
    removed = ~keep
    if not removed.any():
        return {'direction_pct': 0.0, 'removed_count': 0, 'removed_mean': 0.0, 'kept_mean': float(np.mean(vals))}
    all_log = float(np.mean(np.log(vals)))
    rem_log = float(np.mean(np.log(vals[removed])))
    kept_mean = float(np.mean(vals[keep])) if keep.any() else 0.0
    removed_mean = float(np.mean(vals[removed]))
    # Positiv när borttagna omgångar har lägre log-utdelning än helheten.
    direction_pct = 100.0 * (math.exp(all_log - rem_log) - 1.0)
    direction_pct = max(-float(clip), min(float(clip), float(direction_pct)))
    return {
        'direction_pct': float(direction_pct),
        'removed_count': int(removed.sum()),
        'removed_mean': removed_mean,
        'kept_mean': kept_mean,
        'removed_min': float(np.min(vals[removed])) if removed.any() else 0.0,
        'removed_max': float(np.max(vals[removed])) if removed.any() else 0.0,
    }


def _v13_enrich_candidates(candidates: List[dict], hist_payout: np.ndarray, htot: int) -> List[dict]:
    out = []
    for c in candidates:
        c2 = dict(c)
        d = _v13_payout_direction_from_removed(int(c2.get('hist_bits', 0)), int(htot), hist_payout)
        c2['payout_direction_pct'] = float(d['direction_pct'])
        c2['removed_hist_count'] = int(d['removed_count'])
        c2['removed_payout_mean'] = float(d['removed_mean'])
        c2['kept_payout_mean'] = float(d['kept_mean'])
        out.append(c2)
    return out


def _v13_state_metrics(st: _V9State, payout_values: np.ndarray, args, variant_id: str) -> dict:
    cfg = V13_VARIANT_DEFS.get(str(variant_id).upper(), V13_VARIANT_DEFS['A'])
    base = _v9_state_metrics(
        st, payout_values,
        hit_power=args.hit_power,
        val_power=args.validation_power,
        reduction_power=args.reduction_power,
        payout_weight=0.0 if cfg.get('fixed_payout') else args.payout_weight,
        cluster_weight=args.cluster_weight,
    )
    direction = _v13_payout_direction_from_removed(int(st.hist_mask), int(st.hist_size), payout_values)
    payout_direction_pct = float(direction['direction_pct'])
    if cfg.get('fixed_payout'):
        # Liten preferens, inte ett sätt att köpa sämre träff. Säkerhetsgolven
        # ligger i apply-funktionen; här används den bara för att välja mellan
        # jämnstarka paket.
        base['joint_score'] = float(base['core_score']) + float(args.payout_direction_weight) * payout_direction_pct + float(args.cluster_weight) * min(5.0, max(0.0, base.get('cluster_mean', 0.0)))
        base['payout_lift_pct'] = payout_direction_pct
    base['payout_direction_pct'] = payout_direction_pct
    base['removed_hist_count'] = int(direction['removed_count'])
    base['removed_payout_mean'] = float(direction['removed_mean'])
    base['kept_payout_mean'] = float(direction['kept_mean'])
    return base


def _v13_state_sort_key(st: _V9State, payout_values: np.ndarray, args, variant_id: str) -> tuple:
    m = _v13_state_metrics(st, payout_values, args, variant_id)
    # Fortsätt premiera kombinationen av säkerhet och reducering, men håll kvar
    # träff/validering i nyckeln så beam-arkivet inte fylls av för riskabla paket.
    return (
        float(m['joint_score']),
        int(st.hist_hit),
        int(st.val_hit),
        float(m['reduction_pct']),
        float(m.get('payout_direction_pct', 0.0)),
        float(m.get('cluster_mean', 0.0)),
        -int(st.frame_count),
        -int(len(st.chosen)),
    )


def _v13_dedupe(states: List[_V9State], payout_values: np.ndarray, args, beam_width: int, variant_id: str) -> List[_V9State]:
    best: Dict[Tuple[int, int, int], _V9State] = {}
    for st in states:
        sig = (int(st.hist_mask), int(st.val_mask), int(st.frame_mask))
        old = best.get(sig)
        if old is None or _v13_state_sort_key(st, payout_values, args, variant_id) > _v13_state_sort_key(old, payout_values, args, variant_id):
            best[sig] = st
    pool = list(best.values())
    pool.sort(key=lambda s: _v13_state_sort_key(s, payout_values, args, variant_id), reverse=True)

    # Extra diversitet: behåll inte bara toppscoren utan även starka stater per
    # faktisk historikträff och radband. Det gör att 29/30-vägar inte försvinner
    # bara för att 30/30-vägar råkar få högre tidigt score.
    selected, seen = [], set()
    def add(st):
        sig = (int(st.hist_mask), int(st.val_mask), int(st.frame_mask))
        if sig not in seen:
            seen.add(sig)
            selected.append(st)
    for st in pool[:max(1, int(beam_width))]:
        add(st)
    buckets: Dict[Tuple[int, int, int], List[_V9State]] = {}
    for st in pool:
        rows_band = int(st.frame_count // max(1, int(args.row_bucket_size)))
        buckets.setdefault((int(st.hist_hit), int(st.val_hit), rows_band), []).append(st)
    for vals in buckets.values():
        vals.sort(key=lambda s: _v13_state_sort_key(s, payout_values, args, variant_id), reverse=True)
        for st in vals[:max(1, int(args.per_bucket_keep))]:
            add(st)
            if len(selected) >= max(int(beam_width), int(args.archive_width)):
                break
        if len(selected) >= max(int(beam_width), int(args.archive_width)):
            break
    selected.sort(key=lambda s: _v13_state_sort_key(s, payout_values, args, variant_id), reverse=True)
    return selected[:max(1, int(beam_width))]


def _v13_apply_sequence(st: _V9State, seq: Sequence[dict], *, phase: str, hist_floor: int, val_floor_hit: int, sign_bits: tuple, min_unique_rows: int) -> Optional[_V9State]:
    cur = st
    for cand in seq:
        cur = _v9_apply_candidate(
            cur, cand,
            phase=phase,
            hist_floor=hist_floor,
            val_floor_hit=val_floor_hit,
            sign_bits=sign_bits,
            min_unique_rows=min_unique_rows,
        )
        if cur is None:
            return None
    if cur.frame_count >= st.frame_count:
        return None
    return cur


def _v13_sequence_key(seq: Sequence[dict], st: _V9State) -> tuple:
    # Billig försortering av bundle-kombinationer innan exakt state-sort.
    step_red = 0.0
    gap = 0.0
    payout = 0.0
    hist = 10**9
    val = 10**9
    for c in seq:
        step_red += float(c.get('red_pct', 0.0))
        gap += float(c.get('gap_score', 0.0))
        payout += float(c.get('payout_direction_pct', c.get('payout_lift_pct', 0.0)))
        hist = min(hist, int(c.get('hist_hit', 0)))
        val = min(val, int(c.get('val_hit', 0)))
    return (hist, val, step_red, gap, payout, -len(seq))


def _v13_beam_search(start_states: List[_V9State], candidates: List[dict], *, phase: str, hist_floor: int, val_floor_hit: int, sign_bits: tuple, min_unique_rows: int, payout_values: np.ndarray, args, variant_id: str) -> List[_V9State]:
    cfg = V13_VARIANT_DEFS.get(str(variant_id).upper(), V13_VARIANT_DEFS['A'])
    states = list(start_states)
    archive = _v13_dedupe(list(start_states), payout_values, args, max(args.beam_width, len(start_states)), variant_id)
    unique_keys = {str(c.get('key')) for c in candidates if c.get('key')}
    max_depth = len(unique_keys)

    base_sorted = sorted(candidates, key=_v9_candidate_rank, reverse=True)
    for depth in range(max_depth):
        expanded = []
        for st in states:
            # Singlar: alltid.
            for cand in candidates:
                ns2 = _v13_apply_sequence(
                    st, [cand], phase=phase, hist_floor=hist_floor,
                    val_floor_hit=val_floor_hit, sign_bits=sign_bits,
                    min_unique_rows=min_unique_rows,
                )
                if ns2 is not None:
                    expanded.append(ns2)

            # Bundle-sökning: bara för profilfasen och bara i C/D.
            if cfg.get('bundle_search') and phase == 'profile':
                available = [c for c in base_sorted if c.get('key') and c.get('key') not in st.used_keys]
                pool = available[:max(2, int(args.bundle_pool_size))]
                seqs = []
                for i in range(len(pool)):
                    for j in range(i + 1, len(pool)):
                        if pool[i].get('key') == pool[j].get('key'):
                            continue
                        seqs.append((pool[i], pool[j]))
                if bool(args.enable_triples):
                    tpool = pool[:max(3, int(args.triple_pool_size))]
                    for i in range(len(tpool)):
                        for j in range(i + 1, len(tpool)):
                            for k in range(j + 1, len(tpool)):
                                keys = {tpool[i].get('key'), tpool[j].get('key'), tpool[k].get('key')}
                                if len(keys) < 3:
                                    continue
                                seqs.append((tpool[i], tpool[j], tpool[k]))
                seqs.sort(key=lambda s: _v13_sequence_key(s, st), reverse=True)
                bundle_added = []
                for seq in seqs[:max(0, int(args.max_bundle_trials_per_state))]:
                    ns2 = _v13_apply_sequence(
                        st, seq, phase=phase, hist_floor=hist_floor,
                        val_floor_hit=val_floor_hit, sign_bits=sign_bits,
                        min_unique_rows=min_unique_rows,
                    )
                    if ns2 is not None:
                        bundle_added.append(ns2)
                if bundle_added:
                    bundle_added.sort(key=lambda s: _v13_state_sort_key(s, payout_values, args, variant_id), reverse=True)
                    expanded.extend(bundle_added[:max(0, int(args.max_bundle_keep_per_state))])
        if not expanded:
            break
        states = _v13_dedupe(expanded, payout_values, args, args.beam_width, variant_id)
        archive = _v13_dedupe(archive + states, payout_values, args, args.archive_width, variant_id)
        best = states[0]
        print(f'      V13 {variant_id} {phase} djup {depth+1}: states={len(states)} · filter={len(best.chosen)} · träff={best.hist_hit}/{best.hist_size} · val={best.val_hit}/{best.val_size if best.val_size else 0} · rader={best.frame_count}', flush=True)
    return archive


def _build_cluster_payout_package_v13_from_candidates(ns: dict, candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, frame_rows: list, frame: list, antal_matcher: int, args, variant_id: str):
    raw_row_matrix = ns['_frame_row_matrix'](frame_rows, antal_matcher)
    sign_bits = _build_teckenskydd_bits(ns, raw_row_matrix, frame, antal_matcher)
    initial = _V9State(
        hist_mask=(1 << htot)-1,
        val_mask=(1 << vtot)-1 if vtot else 0,
        frame_mask=(1 << ftot)-1,
        hist_size=htot, val_size=vtot, frame_size=ftot,
        chosen=tuple(), used_keys=frozenset(), steps=tuple(),
    )
    val_floor_profile = int(math.ceil(vtot * float(args.min_package_val_pct) / 100.0)) if vtot else 0
    val_floor_structure = int(math.ceil(vtot * float(args.min_structure_package_val_pct) / 100.0)) if vtot else 0

    structure_cands = [c for c in candidates if _is_structure_candidate(c) and int(c.get('hist_hit', 0)) == htot]
    profile_cands = [c for c in candidates if _is_profile_candidate(c) and int(c.get('hist_hit', 0)) >= int(args.min_hit)]

    structure_states = [initial]
    if structure_cands:
        structure_states = _v13_beam_search(
            [initial], structure_cands,
            phase='structure', hist_floor=htot,
            val_floor_hit=val_floor_structure,
            sign_bits=sign_bits, min_unique_rows=int(args.min_unique_rows),
            payout_values=hist_payout, args=args, variant_id=variant_id,
        )
    structure_states.sort(key=lambda s: _v13_state_sort_key(s, hist_payout, args, variant_id), reverse=True)

    seeds = [initial]
    seen_masks = {(initial.hist_mask, initial.val_mask, initial.frame_mask)}
    for st in structure_states:
        sig = (st.hist_mask, st.val_mask, st.frame_mask)
        if sig in seen_masks:
            continue
        seen_masks.add(sig)
        seeds.append(st)
        if len(seeds) >= int(args.structure_seed_count):
            break

    finals = []
    for si, seed in enumerate(seeds, 1):
        print(f'    V13 {variant_id} profilbeam från strukturfrö {si}/{len(seeds)} · startrader={seed.frame_count}', flush=True)
        finals.extend(_v13_beam_search(
            [seed], profile_cands,
            phase='profile', hist_floor=int(args.min_hit),
            val_floor_hit=val_floor_profile,
            sign_bits=sign_bits, min_unique_rows=int(args.min_unique_rows),
            payout_values=hist_payout, args=args, variant_id=variant_id,
        ))
    finals.extend([st for st in structure_states if st.chosen])
    valid = [st for st in finals if st.chosen and st.hist_hit >= int(args.min_hit) and (vtot == 0 or st.val_hit >= val_floor_profile)]
    if not valid:
        return None, {'error': f'Ingen V13-state klarade paketkraven för variant {variant_id}.', 'candidates': len(candidates)}
    valid = _v13_dedupe(valid, hist_payout, args, args.archive_width, variant_id)
    best = max(valid, key=lambda s: _v13_state_sort_key(s, hist_payout, args, variant_id))
    best = _v9_redundancy_cleanup(best, hist_floor=int(args.min_hit), val_floor_hit=val_floor_profile, sign_bits=sign_bits)
    metrics = _v13_state_metrics(best, hist_payout, args, variant_id)
    package = {
        'variant': str(variant_id),
        'variant_label': V13_VARIANT_DEFS.get(str(variant_id), {}).get('label', str(variant_id)),
        'target': int(best.hist_hit),
        'target_label': f'V13 {variant_id} {best.hist_hit}/{htot}',
        'hist_hit': int(best.hist_hit), 'hist_total': int(htot),
        'val_hit': int(best.val_hit), 'val_total': int(vtot),
        'frame_start': int(ftot), 'frame_after': int(best.frame_count),
        'reduction_pct': float(metrics['reduction_pct']),
        'joint_score': float(metrics['joint_score']),
        'core_score': float(metrics['core_score']),
        'payout_lift_pct': float(metrics.get('payout_lift_pct', 0.0)),
        'payout_direction_pct': float(metrics.get('payout_direction_pct', metrics.get('payout_lift_pct', 0.0))),
        'removed_hist_count': int(metrics.get('removed_hist_count', 0)),
        'removed_payout_mean': float(metrics.get('removed_payout_mean', 0.0)),
        'cluster_mean': float(metrics['cluster_mean']),
        'num_filters': len(best.chosen),
        'filters': list(best.chosen), 'steps': list(best.steps),
        'package_type': f'V13 {variant_id} – {V13_VARIANT_DEFS.get(str(variant_id), {}).get("label", str(variant_id))}',
        'structure_filters': best.structure_filters,
        'profile_filters': best.profile_filters,
        'fat_filters': best.fat_filters,
        'edge_filters': best.edge_filters,
        'meta': {
            'engine': 'v13_v9_plus_controlled_optimizer',
            'variant': str(variant_id),
            'variant_label': V13_VARIANT_DEFS.get(str(variant_id), {}).get('label', str(variant_id)),
            'fixed_payout': bool(V13_VARIANT_DEFS.get(str(variant_id), {}).get('fixed_payout')),
            'bundle_search': bool(V13_VARIANT_DEFS.get(str(variant_id), {}).get('bundle_search')),
            'no_filter_cap': True,
            'min_hit': int(args.min_hit),
            'min_package_val_pct': float(args.min_package_val_pct),
            'min_unique_rows': int(args.min_unique_rows),
            'candidate_count': len(candidates),
            'structure_candidates': len(structure_cands),
            'profile_candidates': len(profile_cands),
            'structure_seeds': len(seeds),
            'beam_width': int(args.beam_width),
            'archive_width': int(args.archive_width),
            'min_gap_score': float(args.min_gap_score),
        },
    }
    return package, package['meta']


def _v13_json_filters(pkg: dict) -> str:
    rows = []
    for c in pkg.get('filters', []) or []:
        rows.append({
            'key': c.get('key'), 'name': c.get('name'), 'category': c.get('category'),
            'interval': c.get('interval_txt'), 'hist': f"{c.get('hist_hit')}/{c.get('hist_total')}",
            'val_pct': round(float(c.get('val_pct', 0.0)), 1),
            'frame_red_pct_alone': round(float(c.get('red_pct', 0.0)), 1),
            'step_removed_rows': int(c.get('step_removed_rows', 0)),
            'gap_score': round(float(c.get('gap_score', 0.0)), 3),
            'payout_lift_pct_v9': round(float(c.get('payout_lift_pct', 0.0)), 2),
            'payout_direction_pct': round(float(c.get('payout_direction_pct', 0.0)), 2),
            'removed_hist_count': int(c.get('removed_hist_count', 0)),
            'removed_payout_mean': round(float(c.get('removed_payout_mean', 0.0)), 0),
        })
    return json.dumps(rows, ensure_ascii=False)


def _v13_diagnose_facit(ns: dict, correct: str, specs: list, pkg: dict) -> str:
    """Exakt första dödande filter för facit, om paketet missar."""
    try:
        by_key = {s.get('key'): s for s in specs or []}
        active = correct
        for ix, c in enumerate(pkg.get('filters', []) or [], start=1):
            spec = by_key.get(c.get('key'), c)
            interval = c.get('interval')
            ok = True
            val = None
            try:
                getter = spec.get('getter')
                if getter is not None:
                    val = getter(correct)
                    ok = (float(val) >= float(interval[0]) and float(val) <= float(interval[1]))
                else:
                    ok = ns.get('_spec_pass', lambda *_: True)(correct, spec, interval)
            except Exception:
                try:
                    ok = ns.get('_spec_pass', lambda *_: True)(correct, spec, interval)
                except Exception:
                    ok = False
            if not ok:
                payload = {
                    'dödande_steg': ix,
                    'filter': c.get('name'),
                    'kategori': c.get('category'),
                    'intervall': c.get('interval_txt'),
                    'facitvärde': None if val is None else float(val) if isinstance(val, (int, float, np.number)) else str(val),
                    'hist_filter': f"{c.get('hist_hit')}/{c.get('hist_total')}",
                    'step_removed_rows': int(c.get('step_removed_rows', 0)),
                    'payout_direction_pct': round(float(c.get('payout_direction_pct', 0.0)), 2),
                    'gap_score': round(float(c.get('gap_score', 0.0)), 3),
                }
                return json.dumps(payload, ensure_ascii=False)
        return ''
    except Exception as e:
        return json.dumps({'diagnosfel': str(e)}, ensure_ascii=False)


def _run_backtest_v13(ns: dict, global_db: pd.DataFrame, args) -> Tuple[pd.DataFrame, dict]:
    antal_matcher = 13
    norm = ns['normalize_single_row_text']
    similar_history = ns['_similar_history_for_backtest']
    ranked_frame = ns['_ranked_frame_like_widths']
    generate_rows_from_frame = ns['generate_rows_from_frame']
    build_clean_filter_specs = ns['build_clean_filter_specs']
    package_passes_row = ns['_package_passes_row']

    global_db, supersvar_count = _prepare_supersvar_payout(global_db, int(args.pay_max))
    db = _prepare_db(ns, global_db, antal_matcher, int(args.pay_min), int(args.pay_max))
    if db.empty:
        return pd.DataFrame(), {'error': 'Inga giltiga omgångar i utdelningsintervallet.'}

    variants = _v13_parse_variants(args.variants)
    spik, halv, hel = _parse_frame_profile(args.frame_profile)
    template = _base_frame(spik, halv, hel)
    test_rows = list(db.iterrows())[int(args.test_offset):int(args.test_offset)+int(args.max_tests)]
    out = []
    for ti, (idx, test_row) in enumerate(test_rows, 1):
        _release_test_memory()
        t0_case = time.time()
        correct = norm(test_row.get('Correct_Row', ''))
        input_vec = test_row.get('Prob_Vector', [])
        test_date = test_row.get('Datum', None)
        payout = float(pd.to_numeric(pd.Series([test_row.get('Payout', 0)]), errors='coerce').fillna(0).iloc[0])
        print(f'V13 testfall {ti}/{len(test_rows)} · {str(test_date)[:10]} · utdelning {payout:,.0f} kr'.replace(',', ' '), flush=True)
        try:
            sim_df = similar_history(
                global_db, input_vec, antal_matcher,
                top_n=int(args.top_n), pay_min=int(args.pay_min), pay_max=int(args.pay_max),
                exclude_index=idx, mode=str(args.mode), test_date=test_date,
            )
            wide_df = similar_history(
                global_db, input_vec, antal_matcher,
                top_n=max(int(args.top_n), int(args.wide_n)), pay_min=int(args.pay_min), pay_max=int(args.pay_max),
                exclude_index=idx, mode=str(args.mode), test_date=test_date,
            )
            if len(sim_df) < int(args.top_n):
                raise RuntimeError(f'För få liknande omgångar: {len(sim_df)} av önskade {args.top_n}.')
            engine_frame = ranked_frame(template, input_vec, antal_matcher)
            frame_rows, _, ok, msg = generate_rows_from_frame(engine_frame, max_rows=None)
            if not ok or not frame_rows:
                raise RuntimeError(f'Kunde inte skapa ram: {msg}')
            specs = build_clean_filter_specs(
                sim_df, input_vec, antal_matcher,
                slider_u_count=3, target_hist_pct=int(args.filter_hist_target_pct),
                u_rows=None, hist_df=global_db, max_shock_pct=22,
                candidate_rows=frame_rows, include_supermakro=not bool(args.fast_no_supermakro),
            )
            global _ACTIVE_V9_NS
            _ACTIVE_V9_NS = ns
            candidates, htot, vtot, ftot, hist_payout = _build_dynamic_candidates_v9(
                ns, specs, sim_df, frame_rows, engine_frame, antal_matcher,
                profile_min_hit=int(args.min_hit),
                variants_per_key=int(args.variants_per_key),
                max_candidates=int(args.max_candidates),
                validation_df=wide_df,
                min_candidate_val_pct=float(args.min_candidate_val_pct),
                min_structure_val_pct=float(args.min_structure_val_pct),
                min_gap_score=float(args.min_gap_score),
                frame_adapt=True,
            )
            candidates = _v13_enrich_candidates(candidates, hist_payout, htot)
            print(f'    V13 kandidatbygge: kandidater={len(candidates)} · historik={htot} · validering={vtot}', flush=True)
            if not candidates:
                raise RuntimeError('Inga giltiga V13-kandidater.')

            for variant_id in variants:
                t0 = time.time()
                try:
                    print(f'  Variant {variant_id}: {V13_VARIANT_DEFS[variant_id]["label"]}', flush=True)
                    pkg, meta = _build_cluster_payout_package_v13_from_candidates(
                        ns, candidates, htot, vtot, ftot, hist_payout,
                        frame_rows, engine_frame, antal_matcher, args, variant_id,
                    )
                    if pkg is None:
                        raise RuntimeError(meta.get('error', 'Inget paket'))
                    pkg_pass, fail_reason = package_passes_row(correct, specs, pkg)
                    diag = '' if pkg_pass else _v13_diagnose_facit(ns, correct, specs, pkg)
                    out.append({
                        'Variant': variant_id,
                        'Variantnamn': V13_VARIANT_DEFS[variant_id]['label'],
                        'Datum': str(test_date)[:10],
                        'Utdelning': int(round(payout)),
                        'Facit': correct,
                        'Status': 'OK',
                        'Paket klarar facit': 'Ja' if pkg_pass else 'Nej',
                        'Orsak': 'OK' if pkg_pass else fail_reason,
                        'Missdiagnos JSON': diag,
                        'Liknande historik': len(sim_df),
                        'Bred validering': len(wide_df),
                        'Paketträff': f"{pkg['hist_hit']}/{pkg['hist_total']}",
                        'Valideringsträff': f"{pkg['val_hit']}/{pkg['val_total']}",
                        'Grundram rader': len(frame_rows),
                        'Paketrader': int(pkg['frame_after']),
                        'V31 budget target': int(pkg.get('v31_budget_target', 0) or 0),
                        'V31 budget min': int(pkg.get('v31_budget_min', 0) or 0),
                        'V31 budget max': int(pkg.get('v31_budget_max', 0) or 0),
                        'V31 budget status': str(pkg.get('v31_budget_status', '')),
                        'V31 budget inom band': str(pkg.get('v31_budget_ok', '')),
                        'V31 budgetavvikelse': int(pkg.get('v31_budget_delta', 0) or 0),
                        'Reducerar %': round(float(pkg['reduction_pct']), 2),
                        'Gemensamt score': round(float(pkg['joint_score']), 4),
                        'Utdelningslift V9 %': round(float(pkg.get('payout_lift_pct', 0.0)), 2),
                        'Utdelningsriktning %': round(float(pkg.get('payout_direction_pct', 0.0)), 2),
                        'Borttagna hist omgångar': int(pkg.get('removed_hist_count', 0)),
                        'Borttagna hist medelutdelning': round(float(pkg.get('removed_payout_mean', 0.0)), 0),
                        'Medel spret-gap': round(float(pkg['cluster_mean']), 3),
                        'Filter totalt': int(pkg['num_filters']),
                        'Strukturfilter': int(pkg['structure_filters']),
                        'Profilfilter': int(pkg['profile_filters']),
                        'FAT/ABC-filter': int(pkg['fat_filters']),
                        'Värde/favorit/skräll': int(pkg['edge_filters']),
                        'Filter JSON': _v13_json_filters(pkg),
                        'Steg JSON': json.dumps(pkg.get('steps', []), ensure_ascii=False),
                        'Supersvår testomgång': 'Ja' if bool(test_row.get('_BT_No13Winner', False)) else 'Nej',
                        'Sekunder': round(time.time()-t0, 2),
                    })
                except Exception as e:
                    out.append({
                        'Variant': variant_id,
                        'Variantnamn': V13_VARIANT_DEFS.get(variant_id, {}).get('label', variant_id),
                        'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct,
                        'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e),
                        'Traceback': traceback.format_exc(),
                        'Sekunder': round(time.time()-t0, 2),
                    })
                    print(f'  FEL variant {variant_id}: {e}', flush=True)
        except Exception as e:
            for variant_id in variants:
                out.append({
                    'Variant': variant_id,
                    'Variantnamn': V13_VARIANT_DEFS.get(variant_id, {}).get('label', variant_id),
                    'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct,
                    'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e),
                    'Traceback': traceback.format_exc(),
                    'Sekunder': round(time.time()-t0_case, 2),
                })
            print(f'  FEL testfall: {e}', flush=True)
    return pd.DataFrame(out), {'supersvar_count': supersvar_count, 'eligible_rounds': len(db), 'variants': variants}


def _summarize_v13(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    group_key = 'Variant' if 'Variant' in detail.columns else None
    groups = detail.groupby(group_key, dropna=False) if group_key else [('ALLA', detail)]
    for variant, grp in groups:
        ok = grp[grp['Status'].astype(str) == 'OK'].copy() if 'Status' in grp.columns else grp.copy()
        hits = int((ok.get('Paket klarar facit', pd.Series(dtype=str)) == 'Ja').sum()) if not ok.empty else 0
        n = len(ok)
        pak = pd.to_numeric(ok.get('Paketrader'), errors='coerce').dropna() if not ok.empty else pd.Series(dtype=float)
        red = pd.to_numeric(ok.get('Reducerar %'), errors='coerce').dropna() if not ok.empty else pd.Series(dtype=float)
        filters = pd.to_numeric(ok.get('Filter totalt'), errors='coerce').dropna() if not ok.empty else pd.Series(dtype=float)
        direction = pd.to_numeric(ok.get('Utdelningsriktning %'), errors='coerce').dropna() if not ok.empty else pd.Series(dtype=float)
        name = ''
        if not grp.empty and 'Variantnamn' in grp.columns:
            name = str(grp['Variantnamn'].dropna().iloc[0]) if len(grp['Variantnamn'].dropna()) else ''
        rows.append({
            'Variant': variant,
            'Variantnamn': name,
            'Testade omgångar': n,
            'Träffar': hits,
            'Träff %': round(100.0*hits/max(1,n), 1),
            'Median paketrader': round(float(pak.median()), 1) if len(pak) else None,
            'Medel paketrader': round(float(pak.mean()), 1) if len(pak) else None,
            'Median reducering %': round(float(red.median()), 1) if len(red) else None,
            'Median filter': round(float(filters.median()), 1) if len(filters) else None,
            'Max filter': int(filters.max()) if len(filters) else None,
            'Median utdelningsriktning %': round(float(direction.median()), 2) if len(direction) else None,
            'Fel/hoppade': int(len(grp)-n),
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    # Markerar vinnaren enligt utvecklingstestets fasta ranking.
    def win_key(r):
        hit = float(r.get('Träff %', 0.0) or 0.0)
        hits = int(r.get('Träffar', 0) or 0)
        med_rows = r.get('Median paketrader')
        med_rows = float(med_rows) if pd.notna(med_rows) else 10**12
        direction = r.get('Median utdelningsriktning %')
        direction = float(direction) if pd.notna(direction) else 0.0
        return (hits, hit, -med_rows, direction)
    best_idx = max(summary.index, key=lambda i: win_key(summary.loc[i]))
    summary['Vinnare'] = ''
    summary.loc[best_idx, 'Vinnare'] = 'JA'
    return summary.sort_values(by=['Vinnare','Träffar','Median paketrader'], ascending=[False,False,True])


def _v13_winner_variant(summary: pd.DataFrame) -> Optional[str]:
    if summary is None or summary.empty or 'Vinnare' not in summary.columns:
        return None
    w = summary[summary['Vinnare'].astype(str) == 'JA']
    if w.empty:
        return None
    return str(w.iloc[0].get('Variant'))


def _write_v13_outputs(detail: pd.DataFrame, args, out_dir: Path, app_file: Path, db_file: Path, worker_count: int = 0):
    summary = _summarize_v13(detail)
    winner_variant = _v13_winner_variant(summary)
    detail_path = out_dir / f'{args.output_prefix}_detail.csv'
    summary_path = out_dir / f'{args.output_prefix}_summary.csv'
    report_path = out_dir / f'{args.output_prefix}_report.txt'
    winner_detail_path = out_dir / f'{args.output_prefix}_winner_detail.csv'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    if winner_variant and 'Variant' in detail.columns:
        detail[detail['Variant'].astype(str) == str(winner_variant)].to_csv(winner_detail_path, index=False)
    else:
        pd.DataFrame().to_csv(winner_detail_path, index=False)

    variant_lines = []
    for k in _v13_parse_variants(args.variants):
        cfg = V13_VARIANT_DEFS[k]
        variant_lines.append(f'{k}: {cfg["label"]} – {cfg["description"]}')

    lines = [
        'TIPSET AI – V13 V9-PLUS KONTROLLERAD VARIANTOPTIMERING',
        '='*76,
        f'Appbas: {app_file.name}',
        f'Databas: {db_file.name}',
        f'Testomgångar: {args.max_tests}',
        f'Utdelningsintervall: {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '),
        f'Liknande historik: {args.top_n}; bred validering: {args.wide_n}',
        f'Strukturpaket: 100 % på {args.top_n} liknande omgångar',
        f'Profil/slutpaket: minst {args.min_hit}/{args.top_n}; bred paketvalidering minst {args.min_package_val_pct:.1f} %',
        'Max antal filter: ingen hård gräns',
        f'Minsta marginalbidrag: {args.min_unique_rows} unik rad',
        f'Minsta spret-gap: {args.min_gap_score}',
        f'Beam width: {args.beam_width}; arkiv: {args.archive_width}; kandidater: {args.max_candidates}',
        f'Bundle-sökning: pool {args.bundle_pool_size}; tripplar {"på" if args.enable_triples else "av"}',
        f'Super-Makro: {"avstängt" if args.fast_no_supermakro else "på / enligt appen"}',
        f'Isolerade worker-processer: {worker_count}',
        '',
        'TESTADE INTERNA VARIANTER',
        *variant_lines,
        '',
        'VINNANDE KONFIGURATION',
        str(winner_variant) + (' – ' + V13_VARIANT_DEFS.get(str(winner_variant), {}).get('label', '') if winner_variant else ''),
        '',
        'SAMMANFATTNING',
        summary.to_string(index=False) if not summary.empty else '(tom)',
        '',
        'MOTORPRINCIP',
        'V13 är ett utvecklingstest byggt på V9. Det jämför fyra interna varianter, men appen ska bara få vinnaren.',
        'Rättad utdelningsriktning räknas på faktiskt uteslutna historikomgångar.',
        'Paket-/parsynergi testar singlar samt selektiva par/tripplar utan att lägga ett filtertak.',
        'Alla radbeslut räknas exakt på hela grundramen; ingen sampling används.',
        '',
        f'Detail alla varianter: {detail_path}',
        f'Detail vinnare: {winner_detail_path}',
    ]
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V13 SAMMANFATTNING', flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f'Vinnare: {winner_variant}', flush=True)
    print(f'Rapport: {report_path}', flush=True)
    print(f'Summary: {summary_path}', flush=True)
    print(f'Detail: {detail_path}', flush=True)
    print(f'Winner detail: {winner_detail_path}', flush=True)


def _resolve_v13_script_path() -> Optional[Path]:
    candidates = []
    raw_file = globals().get('__file__')
    if raw_file:
        candidates.append(Path(str(raw_file)))
    try:
        argv0 = Path(str(sys.argv[0]))
        if argv0.suffix.lower() == '.py':
            candidates.append(argv0)
    except Exception:
        pass
    preferred_names = [
        'package_cluster_payout_synergy_v13_v9plus_optimizer_colab_upload.py',
        'package_cluster_payout_synergy_v13_colab_upload.py',
    ]
    for directory in [Path.cwd(), Path('/content'), Path('/mnt/data')]:
        if not directory.exists():
            continue
        for name in preferred_names:
            candidates.append(directory / name)
        candidates.extend(sorted(directory.glob('package_cluster_payout_synergy_v13*.py')))
    valid, seen = [], set()
    for candidate in candidates:
        try:
            candidate = Path(candidate)
            if not candidate.exists() or candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if str(resolved) in seen:
                continue
            seen.add(str(resolved))
            sample = resolved.read_text(encoding='utf-8', errors='replace')
            if ('def main_v13' not in sample or
                    'def _run_fresh_workers_v13' not in sample or
                    'V13_VARIANT_DEFS' not in sample):
                continue
            score = 0
            name = resolved.name.lower()
            if name == 'package_cluster_payout_synergy_v13_v9plus_optimizer_colab_upload.py':
                score += 100
            if '/content/' in str(resolved):
                score += 30
            valid.append((score, resolved.stat().st_mtime, resolved))
        except Exception:
            continue
    if not valid:
        return None
    valid.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return valid[0][2]


def _run_v13_workers_inline(args, app_file: Path, db_file: Path, out_dir: Path):
    details = []
    for i in range(int(args.max_tests)):
        offset = int(args.test_offset) + i
        print('\n' + '=' * 76, flush=True)
        print(f'V13 sekventiellt testfall {i+1}/{args.max_tests} · offset {offset}', flush=True)
        print('=' * 76, flush=True)
        worker_args = argparse.Namespace(**vars(args))
        worker_args.max_tests = 1
        worker_args.test_offset = offset
        worker_args.internal_worker = True
        worker_args.no_fresh_workers = True
        ns = _load_app_functions(app_file, fast_no_supermakro=False)
        db = ns['load_database'](str(db_file), 13)
        detail, _meta = _run_backtest_v13(ns, db, worker_args)
        if isinstance(detail, pd.DataFrame) and not detail.empty:
            details.append(detail)
        del ns, db, detail
        gc.collect()
    if not details:
        raise SystemExit('Inga V13-resultat skapades i den sekventiella reservkörningen.')
    combined = pd.concat(details, ignore_index=True)
    _write_v13_outputs(combined, args, out_dir, app_file, db_file, worker_count=0)


def _run_fresh_workers_v13(args, app_file: Path, db_file: Path, out_dir: Path):
    script_path = _resolve_v13_script_path()
    if script_path is None:
        print('Varning: ingen uppladdad V13 .py-fil hittades. Kör sekventiellt med exakt beräkning.', flush=True)
        _run_v13_workers_inline(args, app_file, db_file, out_dir)
        return
    print(f'Worker-runner: {script_path}', flush=True)
    root = out_dir / f'_{args.output_prefix}_workers'
    root.mkdir(parents=True, exist_ok=True)
    worker_dirs = []
    pass_flags = [
        'variants','top_n','wide_n','pay_min','pay_max','min_hit','min_candidate_val_pct',
        'min_package_val_pct','min_structure_val_pct','min_structure_package_val_pct',
        'min_gap_score','min_unique_rows','filter_hist_target_pct','frame_profile',
        'beam_width','archive_width','structure_seed_count','variants_per_key','max_candidates',
        'hit_power','validation_power','reduction_power','payout_weight','cluster_weight','mode',
        'payout_direction_weight','bundle_pool_size','triple_pool_size','max_bundle_trials_per_state',
        'max_bundle_keep_per_state','row_bucket_size','per_bucket_keep',
    ]
    flag_map = {
        'variants':'--variants','top_n':'--top-n','wide_n':'--wide-n','pay_min':'--pay-min','pay_max':'--pay-max',
        'min_hit':'--min-hit','min_candidate_val_pct':'--min-candidate-val-pct',
        'min_package_val_pct':'--min-package-val-pct','min_structure_val_pct':'--min-structure-val-pct',
        'min_structure_package_val_pct':'--min-structure-package-val-pct','min_gap_score':'--min-gap-score',
        'min_unique_rows':'--min-unique-rows','filter_hist_target_pct':'--filter-hist-target-pct',
        'frame_profile':'--frame-profile','beam_width':'--beam-width','archive_width':'--archive-width',
        'structure_seed_count':'--structure-seed-count','variants_per_key':'--variants-per-key',
        'max_candidates':'--max-candidates','hit_power':'--hit-power','validation_power':'--validation-power',
        'reduction_power':'--reduction-power','payout_weight':'--payout-weight','cluster_weight':'--cluster-weight',
        'mode':'--mode','payout_direction_weight':'--payout-direction-weight','bundle_pool_size':'--bundle-pool-size',
        'triple_pool_size':'--triple-pool-size','max_bundle_trials_per_state':'--max-bundle-trials-per-state',
        'max_bundle_keep_per_state':'--max-bundle-keep-per-state','row_bucket_size':'--row-bucket-size',
        'per_bucket_keep':'--per-bucket-keep',
    }
    for i in range(int(args.max_tests)):
        offset = int(args.test_offset)+i
        wd = root / f'case_{offset:03d}'
        wd.mkdir(parents=True, exist_ok=True)
        worker_dirs.append(wd)
        cmd = [sys.executable, str(script_path),
               '--app-file', str(app_file), '--db-file', str(db_file),
               '--out-dir', str(wd), '--output-prefix', args.output_prefix,
               '--max-tests', '1', '--test-offset', str(offset), '--internal-worker']
        for name in pass_flags:
            cmd.extend([flag_map[name], str(getattr(args, name))])
        if args.fast_no_supermakro:
            cmd.append('--fast-no-supermakro')
        else:
            cmd.append('--include-supermakro')
        if args.enable_triples:
            cmd.append('--enable-triples')
        print('\n'+'='*76, flush=True)
        print(f'V13 isolerat testfall {i+1}/{args.max_tests} · offset {offset}', flush=True)
        print('='*76, flush=True)
        subprocess.run(cmd, check=True)
    details = []
    for wd in worker_dirs:
        p = wd / f'{args.output_prefix}_detail.csv'
        if p.exists():
            details.append(pd.read_csv(p))
    if not details:
        raise SystemExit('Inga V13-workerresultat hittades.')
    combined = pd.concat(details, ignore_index=True)
    _write_v13_outputs(combined, args, out_dir, app_file, db_file, worker_count=len(worker_dirs))


def main_v13():
    parser = argparse.ArgumentParser(description='Tipset AI V13 – V9-plus, kontrollerad intern variantoptimering.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_cluster_payout_synergy_v13_v9plus_test10')
    parser.add_argument('--variants', default='A,B,C,D')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--min-hit', type=int, default=29)
    parser.add_argument('--min-candidate-val-pct', type=float, default=90.0)
    parser.add_argument('--min-package-val-pct', type=float, default=90.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=95.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=95.0)
    parser.add_argument('--min-gap-score', type=float, default=1.25)
    parser.add_argument('--min-unique-rows', type=int, default=1)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='2-8-3')
    parser.add_argument('--beam-width', type=int, default=80)
    parser.add_argument('--archive-width', type=int, default=500)
    parser.add_argument('--structure-seed-count', type=int, default=12)
    parser.add_argument('--variants-per-key', type=int, default=4)
    parser.add_argument('--max-candidates', type=int, default=220)
    parser.add_argument('--hit-power', type=float, default=3.0)
    parser.add_argument('--validation-power', type=float, default=1.5)
    parser.add_argument('--reduction-power', type=float, default=1.0)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.25)
    parser.add_argument('--payout-direction-weight', type=float, default=0.08)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=50)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=350)
    parser.add_argument('--per-bucket-keep', type=int, default=1)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    # V13 standard: behåll V9-basens snabbhet och jämför kontrollerat mot V9.
    # Kör med --include-supermakro för att testa hela filteruppsättningen.
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    parser.add_argument('--force-upload', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-fresh-workers', action='store_true')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    _v13_parse_variants(args.variants)

    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)

    if args.max_tests > 1 and not args.internal_worker and not args.no_fresh_workers:
        _run_fresh_workers_v13(args, app_file, db_file, out_dir)
        return

    ns = _load_app_functions(app_file, fast_no_supermakro=False)
    db = ns['load_database'](str(db_file), 13)
    print('\n'+'='*76, flush=True)
    print('TIPSET AI V13 – V9-PLUS · FYRA INTERNA VARIANTER · ETT VINNANDE PAKET', flush=True)
    print(f'Test: {args.max_tests} omgångar inom {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '), flush=True)
    print(f'Varianter: {args.variants} · struktur 30/30 · profil/slutpaket minst {args.min_hit}/30', flush=True)
    print(f'Ram: {args.frame_profile} · beam {args.beam_width} · kandidater {args.max_candidates}', flush=True)
    print('='*76, flush=True)
    detail, meta = _run_backtest_v13(ns, db, args)
    _write_v13_outputs(detail, args, out_dir, app_file, db_file, worker_count=1 if args.internal_worker else 0)




# ---------------------------------------------------------------------------
# V14 – B-baserad parameteroptimerare ovanpå V13/V9
# ---------------------------------------------------------------------------

V14_VARIANT_DEFS = {
    'B0': {
        'label': 'B0 V13-B låst',
        'description': 'Ren V13-B som kontroll: V9 + rättad utdelningsriktning.',
        'fixed_payout': True, 'bundle_search': False,
        'overrides': {},
    },
    'B1': {
        'label': 'B1 reducering+',
        'description': 'Mer reduceringshungrig B: något mjukare bredkontroll och lägre gapkrav.',
        'fixed_payout': True, 'bundle_search': False,
        'overrides': {
            'hit_power': 2.5, 'validation_power': 1.1, 'reduction_power': 1.35,
            'min_candidate_val_pct': 88.0, 'min_package_val_pct': 88.0,
            'min_gap_score': 1.00, 'payout_direction_weight': 0.05,
        },
    },
    'B2': {
        'label': 'B2 träff+',
        'description': 'Mer konservativ B: högre vikt på träff och bred kontroll.',
        'fixed_payout': True, 'bundle_search': False,
        'overrides': {
            'hit_power': 4.5, 'validation_power': 2.25, 'reduction_power': 1.0,
            'min_candidate_val_pct': 92.0, 'min_package_val_pct': 92.0,
            'min_gap_score': 1.25, 'payout_direction_weight': 0.08,
        },
    },
    'B3': {
        'label': 'B3 lägre spretgap',
        'description': 'Samma B-princip men tillåter mindre tydliga klusterglapp.',
        'fixed_payout': True, 'bundle_search': False,
        'overrides': {
            'min_gap_score': 0.75, 'min_candidate_val_pct': 90.0, 'min_package_val_pct': 90.0,
        },
    },
    'B4': {
        'label': 'B4 mjuk bredkontroll',
        'description': 'Samma B-princip men bredkontrollen sänks för att se om den blockerar reducering.',
        'fixed_payout': True, 'bundle_search': False,
        'overrides': {
            'min_candidate_val_pct': 85.0, 'min_package_val_pct': 85.0,
            'min_gap_score': 1.00, 'validation_power': 1.0,
        },
    },
    'B5': {
        'label': 'B5 radpress',
        'description': 'Aggressiv radpress inom 29/30: lägre gap/bredkrav och högre reduktionsvikt.',
        'fixed_payout': True, 'bundle_search': False,
        'overrides': {
            'hit_power': 2.2, 'validation_power': 0.9, 'reduction_power': 1.65,
            'min_candidate_val_pct': 85.0, 'min_package_val_pct': 85.0,
            'min_gap_score': 0.75, 'payout_direction_weight': 0.04,
        },
    },
    'B6': {
        'label': 'B6 utdelning+',
        'description': 'B med tydligare preferens för att kasta historiskt låga utdelningsspret.',
        'fixed_payout': True, 'bundle_search': False,
        'overrides': {
            'payout_direction_weight': 0.22, 'min_gap_score': 1.00,
            'min_candidate_val_pct': 90.0, 'min_package_val_pct': 90.0,
        },
    },
    'B7': {
        'label': 'B7 tvåspret final 28/30',
        'description': 'Kandidater måste fortfarande vara 29/30, men slutpaketet får gemensamt gå till 28/30.',
        'fixed_payout': True, 'bundle_search': False,
        'overrides': {
            'min_hit': 28, 'candidate_min_hit': 29,
            'hit_power': 3.2, 'validation_power': 1.5, 'reduction_power': 1.35,
            'min_candidate_val_pct': 88.0, 'min_package_val_pct': 88.0,
            'min_gap_score': 1.00, 'payout_direction_weight': 0.06,
        },
    },
}

# Låt V13:s interna sortering läsa V14-varianterna utan att behöva duplicera hela maskmotorn.
V13_VARIANT_DEFS.update(V14_VARIANT_DEFS)


def _v14_parse_variants(value: str) -> List[str]:
    raw = str(value or 'B0,B1,B2,B3,B4,B5,B6,B7').replace(';', ',').replace('+', ',').split(',')
    out = []
    for x in raw:
        k = x.strip().upper()
        if not k:
            continue
        if k not in V14_VARIANT_DEFS:
            raise ValueError(f'Okänd V14-variant: {k}. Tillåtna: {", ".join(V14_VARIANT_DEFS)}')
        if k not in out:
            out.append(k)
    return out or list(V14_VARIANT_DEFS.keys())


def _v14_variant_args(args, variant_id: str):
    vargs = argparse.Namespace(**vars(args))
    cfg = V14_VARIANT_DEFS.get(str(variant_id).upper(), V14_VARIANT_DEFS['B0'])
    for k, v in (cfg.get('overrides') or {}).items():
        setattr(vargs, k, v)
    # Om variant har separat kandidatgolv används det bara i kandidatfiltreringen.
    if not hasattr(vargs, 'candidate_min_hit'):
        setattr(vargs, 'candidate_min_hit', int(getattr(vargs, 'min_hit', 29)))
    return vargs


def _v14_filter_candidates_for_variant(candidates: List[dict], vargs, htot: int) -> List[dict]:
    out = []
    cand_floor = int(getattr(vargs, 'candidate_min_hit', getattr(vargs, 'min_hit', 29)))
    for c in candidates:
        if _is_blocked_auto_candidate(c):
            continue
        is_structure = _is_structure_candidate(c)
        if is_structure:
            if int(c.get('hist_hit', 0)) < int(htot):
                continue
            if float(c.get('val_pct', 100.0)) + 1e-9 < float(getattr(vargs, 'min_structure_val_pct', 95.0)):
                continue
        elif _is_profile_candidate(c):
            if int(c.get('hist_hit', 0)) < cand_floor:
                continue
            if float(c.get('val_pct', 100.0)) + 1e-9 < float(getattr(vargs, 'min_candidate_val_pct', 90.0)):
                continue
            if int(c.get('hist_hit', 0)) < int(htot) and float(c.get('gap_score', 0.0)) + 1e-12 < float(getattr(vargs, 'min_gap_score', 1.25)):
                continue
        else:
            continue
        out.append(c)
    return out


def _v14_global_candidate_settings(args, variants: List[str]) -> dict:
    # Bygg kandidatpoolen så brett att alla V14-varianter får chans. Varje variant
    # filtrerar sedan poolen enligt sina egna regler.
    vals = []
    for vid in variants:
        va = _v14_variant_args(args, vid)
        vals.append(va)
    return {
        'profile_min_hit': min(int(getattr(v, 'candidate_min_hit', getattr(v, 'min_hit', 29))) for v in vals),
        'min_candidate_val_pct': min(float(getattr(v, 'min_candidate_val_pct', 90.0)) for v in vals),
        'min_structure_val_pct': min(float(getattr(v, 'min_structure_val_pct', 95.0)) for v in vals),
        'min_gap_score': min(float(getattr(v, 'min_gap_score', 1.25)) for v in vals),
    }


def _run_backtest_v14(ns: dict, global_db: pd.DataFrame, args) -> Tuple[pd.DataFrame, dict]:
    antal_matcher = 13
    norm = ns['normalize_single_row_text']
    similar_history = ns['_similar_history_for_backtest']
    ranked_frame = ns['_ranked_frame_like_widths']
    generate_rows_from_frame = ns['generate_rows_from_frame']
    build_clean_filter_specs = ns['build_clean_filter_specs']
    package_passes_row = ns['_package_passes_row']

    global_db, supersvar_count = _prepare_supersvar_payout(global_db, int(args.pay_max))
    db = _prepare_db(ns, global_db, antal_matcher, int(args.pay_min), int(args.pay_max))
    if db.empty:
        return pd.DataFrame(), {'error': 'Inga giltiga omgångar i utdelningsintervallet.'}

    variants = _v14_parse_variants(args.variants)
    global_settings = _v14_global_candidate_settings(args, variants)
    spik, halv, hel = _parse_frame_profile(args.frame_profile)
    template = _base_frame(spik, halv, hel)
    test_rows = list(db.iterrows())[int(args.test_offset):int(args.test_offset)+int(args.max_tests)]
    out = []
    for ti, (idx, test_row) in enumerate(test_rows, 1):
        _release_test_memory()
        t0_case = time.time()
        correct = norm(test_row.get('Correct_Row', ''))
        input_vec = test_row.get('Prob_Vector', [])
        test_date = test_row.get('Datum', None)
        payout = float(pd.to_numeric(pd.Series([test_row.get('Payout', 0)]), errors='coerce').fillna(0).iloc[0])
        print(f'V14 testfall {ti}/{len(test_rows)} · {str(test_date)[:10]} · utdelning {payout:,.0f} kr'.replace(',', ' '), flush=True)
        try:
            sim_df = similar_history(
                global_db, input_vec, antal_matcher,
                top_n=int(args.top_n), pay_min=int(args.pay_min), pay_max=int(args.pay_max),
                exclude_index=idx, mode=str(args.mode), test_date=test_date,
            )
            wide_df = similar_history(
                global_db, input_vec, antal_matcher,
                top_n=max(int(args.top_n), int(args.wide_n)), pay_min=int(args.pay_min), pay_max=int(args.pay_max),
                exclude_index=idx, mode=str(args.mode), test_date=test_date,
            )
            if len(sim_df) < int(args.top_n):
                raise RuntimeError(f'För få liknande omgångar: {len(sim_df)} av önskade {args.top_n}.')
            engine_frame = ranked_frame(template, input_vec, antal_matcher)
            frame_rows, _, ok, msg = generate_rows_from_frame(engine_frame, max_rows=None)
            if not ok or not frame_rows:
                raise RuntimeError(f'Kunde inte skapa ram: {msg}')
            specs = build_clean_filter_specs(
                sim_df, input_vec, antal_matcher,
                slider_u_count=3, target_hist_pct=int(args.filter_hist_target_pct),
                u_rows=None, hist_df=global_db, max_shock_pct=22,
                candidate_rows=frame_rows, include_supermakro=not bool(args.fast_no_supermakro),
            )
            global _ACTIVE_V9_NS
            _ACTIVE_V9_NS = ns
            candidates, htot, vtot, ftot, hist_payout = _build_dynamic_candidates_v9(
                ns, specs, sim_df, frame_rows, engine_frame, antal_matcher,
                profile_min_hit=int(global_settings['profile_min_hit']),
                variants_per_key=int(args.variants_per_key),
                max_candidates=int(args.max_candidates),
                validation_df=wide_df,
                min_candidate_val_pct=float(global_settings['min_candidate_val_pct']),
                min_structure_val_pct=float(global_settings['min_structure_val_pct']),
                min_gap_score=float(global_settings['min_gap_score']),
                frame_adapt=True,
            )
            candidates = _v13_enrich_candidates(candidates, hist_payout, htot)
            print(f'    V14 kandidatbygge: kandidater={len(candidates)} · historik={htot} · validering={vtot}', flush=True)
            if not candidates:
                raise RuntimeError('Inga giltiga V14-kandidater.')

            for variant_id in variants:
                t0 = time.time()
                try:
                    vargs = _v14_variant_args(args, variant_id)
                    cand_v = _v14_filter_candidates_for_variant(candidates, vargs, htot)
                    print(f'  Variant {variant_id}: {V14_VARIANT_DEFS[variant_id]["label"]} · kandidater={len(cand_v)}', flush=True)
                    if not cand_v:
                        raise RuntimeError('Inga kandidater efter variantfiltrering.')
                    pkg, meta = _build_cluster_payout_package_v13_from_candidates(
                        ns, cand_v, htot, vtot, ftot, hist_payout,
                        frame_rows, engine_frame, antal_matcher, vargs, variant_id,
                    )
                    if pkg is None:
                        raise RuntimeError(meta.get('error', 'Inget paket'))
                    pkg_pass, fail_reason = package_passes_row(correct, specs, pkg)
                    diag = '' if pkg_pass else _v13_diagnose_facit(ns, correct, specs, pkg)
                    out.append({
                        'Variant': variant_id,
                        'Variantnamn': V14_VARIANT_DEFS[variant_id]['label'],
                        'Datum': str(test_date)[:10],
                        'Utdelning': int(round(payout)),
                        'Facit': correct,
                        'Status': 'OK',
                        'Paket klarar facit': 'Ja' if pkg_pass else 'Nej',
                        'Orsak': 'OK' if pkg_pass else fail_reason,
                        'Missdiagnos JSON': diag,
                        'Liknande historik': len(sim_df),
                        'Bred validering': len(wide_df),
                        'Variant min_hit': int(getattr(vargs, 'min_hit', 29)),
                        'Variant kandidat_min_hit': int(getattr(vargs, 'candidate_min_hit', getattr(vargs, 'min_hit', 29))),
                        'Variant bredkrav %': float(getattr(vargs, 'min_package_val_pct', 90.0)),
                        'Variant gapkrav': float(getattr(vargs, 'min_gap_score', 1.25)),
                        'Paketträff': f"{pkg['hist_hit']}/{pkg['hist_total']}",
                        'Valideringsträff': f"{pkg['val_hit']}/{pkg['val_total']}",
                        'Grundram rader': len(frame_rows),
                        'Paketrader': int(pkg['frame_after']),
                        'Reducerar %': round(float(pkg['reduction_pct']), 2),
                        'Gemensamt score': round(float(pkg['joint_score']), 4),
                        'Utdelningslift V9 %': round(float(pkg.get('payout_lift_pct', 0.0)), 2),
                        'Utdelningsriktning %': round(float(pkg.get('payout_direction_pct', 0.0)), 2),
                        'Borttagna hist omgångar': int(pkg.get('removed_hist_count', 0)),
                        'Borttagna hist medelutdelning': round(float(pkg.get('removed_payout_mean', 0.0)), 0),
                        'Medel spret-gap': round(float(pkg['cluster_mean']), 3),
                        'Filter totalt': int(pkg['num_filters']),
                        'Strukturfilter': int(pkg['structure_filters']),
                        'Profilfilter': int(pkg['profile_filters']),
                        'FAT/ABC-filter': int(pkg['fat_filters']),
                        'Värde/favorit/skräll': int(pkg['edge_filters']),
                        'Filter JSON': _v13_json_filters(pkg),
                        'Steg JSON': json.dumps(pkg.get('steps', []), ensure_ascii=False),
                        'Supersvår testomgång': 'Ja' if bool(test_row.get('_BT_No13Winner', False)) else 'Nej',
                        'Sekunder': round(time.time()-t0, 2),
                    })
                except Exception as e:
                    out.append({
                        'Variant': variant_id,
                        'Variantnamn': V14_VARIANT_DEFS.get(variant_id, {}).get('label', variant_id),
                        'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct,
                        'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e),
                        'Traceback': traceback.format_exc(),
                        'Sekunder': round(time.time()-t0, 2),
                    })
                    print(f'  FEL variant {variant_id}: {e}', flush=True)
        except Exception as e:
            for variant_id in variants:
                out.append({
                    'Variant': variant_id,
                    'Variantnamn': V14_VARIANT_DEFS.get(variant_id, {}).get('label', variant_id),
                    'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct,
                    'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e),
                    'Traceback': traceback.format_exc(),
                    'Sekunder': round(time.time()-t0_case, 2),
                })
            print(f'  FEL testfall: {e}', flush=True)
    return pd.DataFrame(out), {'supersvar_count': supersvar_count, 'eligible_rounds': len(db), 'variants': variants}


def _summarize_v14(detail: pd.DataFrame) -> pd.DataFrame:
    summary = _summarize_v13(detail)
    if summary is None or summary.empty:
        return summary
    # V14-vinnare: verklig träff först, därefter medianrader, därefter medelrader,
    # därefter utdelningsriktning. Detta är striktare än intern joint_score.
    def val(r, col, default=0.0):
        x = r.get(col, default)
        try:
            return float(x) if pd.notna(x) else float(default)
        except Exception:
            return float(default)
    def win_key(r):
        med = val(r, 'Median paketrader', 10**12)
        mean = val(r, 'Medel paketrader', 10**12)
        return (
            int(val(r, 'Träffar', 0)),
            val(r, 'Träff %', 0.0),
            -med,
            -mean,
            val(r, 'Median reducering %', 0.0),
            val(r, 'Median utdelningsriktning %', 0.0),
        )
    summary['Vinnare'] = ''
    best_idx = max(summary.index, key=lambda i: win_key(summary.loc[i]))
    summary.loc[best_idx, 'Vinnare'] = 'JA'
    return summary.sort_values(by=['Vinnare','Träffar','Median paketrader'], ascending=[False,False,True])


def _write_v14_outputs(detail: pd.DataFrame, args, out_dir: Path, app_file: Path, db_file: Path, worker_count: int = 0):
    summary = _summarize_v14(detail)
    winner_variant = _v13_winner_variant(summary)
    detail_path = out_dir / f'{args.output_prefix}_detail.csv'
    summary_path = out_dir / f'{args.output_prefix}_summary.csv'
    report_path = out_dir / f'{args.output_prefix}_report.txt'
    winner_detail_path = out_dir / f'{args.output_prefix}_winner_detail.csv'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    if winner_variant and 'Variant' in detail.columns:
        detail[detail['Variant'].astype(str) == str(winner_variant)].to_csv(winner_detail_path, index=False)
    else:
        pd.DataFrame().to_csv(winner_detail_path, index=False)

    variant_lines = []
    for k in _v14_parse_variants(args.variants):
        cfg = V14_VARIANT_DEFS[k]
        variant_lines.append(f'{k}: {cfg["label"]} – {cfg["description"]}')

    lines = [
        'TIPSET AI – V14 B-BASERAD PARAMETEROPTIMERARE',
        '='*76,
        f'Appbas: {app_file.name}',
        f'Databas: {db_file.name}',
        f'Testomgångar: {args.max_tests}',
        f'Utdelningsintervall: {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '),
        f'Liknande historik: {args.top_n}; bred validering: {args.wide_n}',
        'Bas: V13-B / V9 + rättad utdelningsriktning',
        'Synlig slutmotor: en vinnare; varianterna är bara Colab-utvecklingstest.',
        f'Strukturpaket: 100 % på {args.top_n} liknande omgångar',
        f'Grundkrav profil/slutpaket: minst {args.min_hit}/{args.top_n}; varianter kan skärpa/mjuka detta internt',
        'Max antal filter: ingen hård gräns',
        f'Minsta marginalbidrag: {args.min_unique_rows} unik rad',
        f'Beam width: {args.beam_width}; arkiv: {args.archive_width}; kandidater: {args.max_candidates}',
        f'Färska workers: {worker_count}',
        '',
        'VARIANTER',
        '-'*76,
        *variant_lines,
        '',
        'SAMMANFATTNING',
        '-'*76,
        summary.to_string(index=False) if isinstance(summary, pd.DataFrame) and not summary.empty else '(tom)',
        '',
        f'Detail: {detail_path}',
        f'Summary: {summary_path}',
        f'Winner detail: {winner_detail_path}',
    ]
    if winner_variant:
        lines.append(f'Vinnare: {winner_variant}')
        lines.append('')
        lines.append('TOLKNING V36: detta är snabb slumpkontroll. En variant är intressant om den klarar minst 6/8 och median helst 2400–2800. Kör flera seeds innan 20/30-test.')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V14 SAMMANFATTNING', flush=True)
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        print(summary.to_string(index=False), flush=True)
    if winner_variant:
        print(f'Vinnare: {winner_variant}', flush=True)
    print(f'Rapport: {report_path}', flush=True)
    print(f'Summary: {summary_path}', flush=True)
    print(f'Detail: {detail_path}', flush=True)


def _resolve_v14_script_path() -> Optional[Path]:
    candidates = []
    raw_file = globals().get('__file__')
    if raw_file:
        candidates.append(Path(str(raw_file)))
    try:
        argv0 = Path(str(sys.argv[0]))
        if argv0.suffix.lower() == '.py':
            candidates.append(argv0)
    except Exception:
        pass
    preferred_names = [
        'package_cluster_payout_synergy_v14_b_optimizer_colab_upload.py',
        'package_cluster_payout_synergy_v14_colab_upload.py',
    ]
    for directory in [Path.cwd(), Path('/content'), Path('/mnt/data')]:
        if not directory.exists():
            continue
        for name in preferred_names:
            candidates.append(directory / name)
        candidates.extend(sorted(directory.glob('package_cluster_payout_synergy_v14*.py')))
    valid, seen = [], set()
    for candidate in candidates:
        try:
            candidate = Path(candidate)
            if not candidate.exists() or candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if str(resolved) in seen:
                continue
            seen.add(str(resolved))
            sample = resolved.read_text(encoding='utf-8', errors='replace')
            if ('def main_v14' not in sample or 'V14_VARIANT_DEFS' not in sample or 'def _run_fresh_workers_v14' not in sample):
                continue
            score = 0
            name = resolved.name.lower()
            if name == 'package_cluster_payout_synergy_v14_b_optimizer_colab_upload.py':
                score += 100
            if '/content/' in str(resolved):
                score += 30
            valid.append((score, resolved.stat().st_mtime, resolved))
        except Exception:
            continue
    if not valid:
        return None
    valid.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return valid[0][2]


def _run_v14_workers_inline(args, app_file: Path, db_file: Path, out_dir: Path):
    details = []
    for i in range(int(args.max_tests)):
        offset = int(args.test_offset) + i
        print('\n' + '=' * 76, flush=True)
        print(f'V14 sekventiellt testfall {i+1}/{args.max_tests} · offset {offset}', flush=True)
        print('=' * 76, flush=True)
        worker_args = argparse.Namespace(**vars(args))
        worker_args.max_tests = 1
        worker_args.test_offset = offset
        worker_args.internal_worker = True
        worker_args.no_fresh_workers = True
        ns = _load_app_functions(app_file, fast_no_supermakro=False)
        db = ns['load_database'](str(db_file), 13)
        detail, _meta = _run_backtest_v14(ns, db, worker_args)
        if isinstance(detail, pd.DataFrame) and not detail.empty:
            details.append(detail)
        del ns, db, detail
        gc.collect()
    if not details:
        raise SystemExit('Inga V14-resultat skapades i den sekventiella reservkörningen.')
    combined = pd.concat(details, ignore_index=True)
    _write_v14_outputs(combined, args, out_dir, app_file, db_file, worker_count=0)


def _run_fresh_workers_v14(args, app_file: Path, db_file: Path, out_dir: Path):
    script_path = _resolve_v14_script_path()
    if script_path is None:
        print('Varning: ingen uppladdad V14 .py-fil hittades. Kör sekventiellt med exakt beräkning.', flush=True)
        _run_v14_workers_inline(args, app_file, db_file, out_dir)
        return
    print(f'Worker-runner: {script_path}', flush=True)
    root = out_dir / f'_{args.output_prefix}_workers'
    root.mkdir(parents=True, exist_ok=True)
    worker_dirs = []
    pass_flags = [
        'variants','top_n','wide_n','pay_min','pay_max','min_hit','min_candidate_val_pct',
        'min_package_val_pct','min_structure_val_pct','min_structure_package_val_pct',
        'min_gap_score','min_unique_rows','filter_hist_target_pct','frame_profile',
        'beam_width','archive_width','structure_seed_count','variants_per_key','max_candidates',
        'hit_power','validation_power','reduction_power','payout_weight','cluster_weight','mode',
        'payout_direction_weight','bundle_pool_size','triple_pool_size','max_bundle_trials_per_state',
        'max_bundle_keep_per_state','row_bucket_size','per_bucket_keep',
    ]
    flag_map = {
        'variants':'--variants','top_n':'--top-n','wide_n':'--wide-n','pay_min':'--pay-min','pay_max':'--pay-max',
        'min_hit':'--min-hit','min_candidate_val_pct':'--min-candidate-val-pct',
        'min_package_val_pct':'--min-package-val-pct','min_structure_val_pct':'--min-structure-val-pct',
        'min_structure_package_val_pct':'--min-structure-package-val-pct','min_gap_score':'--min-gap-score',
        'min_unique_rows':'--min-unique-rows','filter_hist_target_pct':'--filter-hist-target-pct',
        'frame_profile':'--frame-profile','beam_width':'--beam-width','archive_width':'--archive-width',
        'structure_seed_count':'--structure-seed-count','variants_per_key':'--variants-per-key',
        'max_candidates':'--max-candidates','hit_power':'--hit-power','validation_power':'--validation-power',
        'reduction_power':'--reduction-power','payout_weight':'--payout-weight','cluster_weight':'--cluster-weight',
        'mode':'--mode','payout_direction_weight':'--payout-direction-weight','bundle_pool_size':'--bundle-pool-size',
        'triple_pool_size':'--triple-pool-size','max_bundle_trials_per_state':'--max-bundle-trials-per-state',
        'max_bundle_keep_per_state':'--max-bundle-keep-per-state','row_bucket_size':'--row-bucket-size',
        'per_bucket_keep':'--per-bucket-keep',
    }
    for i in range(int(args.max_tests)):
        offset = int(args.test_offset)+i
        wd = root / f'case_{offset:03d}'
        wd.mkdir(parents=True, exist_ok=True)
        worker_dirs.append(wd)
        cmd = [sys.executable, str(script_path),
               '--app-file', str(app_file), '--db-file', str(db_file),
               '--out-dir', str(wd), '--output-prefix', args.output_prefix,
               '--max-tests', '1', '--test-offset', str(offset), '--internal-worker']
        for name in pass_flags:
            cmd.extend([flag_map[name], str(getattr(args, name))])
        if args.fast_no_supermakro:
            cmd.append('--fast-no-supermakro')
        else:
            cmd.append('--include-supermakro')
        if args.enable_triples:
            cmd.append('--enable-triples')
        print('\n'+'='*76, flush=True)
        print(f'V14 isolerat testfall {i+1}/{args.max_tests} · offset {offset}', flush=True)
        print('='*76, flush=True)
        subprocess.run(cmd, check=True)
    details = []
    for wd in worker_dirs:
        p = wd / f'{args.output_prefix}_detail.csv'
        if p.exists():
            details.append(pd.read_csv(p))
    if not details:
        raise SystemExit('Inga V14-workerresultat hittades.')
    combined = pd.concat(details, ignore_index=True)
    _write_v14_outputs(combined, args, out_dir, app_file, db_file, worker_count=len(worker_dirs))


def main_v14():
    parser = argparse.ArgumentParser(description='Tipset AI V14 – B-baserad parameteroptimerare.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_cluster_payout_synergy_v14_b_optimizer_test10')
    parser.add_argument('--variants', default='B0,B1,B2,B3,B4,B5,B6,B7')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--min-hit', type=int, default=29)
    parser.add_argument('--min-candidate-val-pct', type=float, default=90.0)
    parser.add_argument('--min-package-val-pct', type=float, default=90.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=95.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=95.0)
    parser.add_argument('--min-gap-score', type=float, default=1.25)
    parser.add_argument('--min-unique-rows', type=int, default=1)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='2-8-3')
    parser.add_argument('--beam-width', type=int, default=24)
    parser.add_argument('--archive-width', type=int, default=120)
    parser.add_argument('--structure-seed-count', type=int, default=4)
    parser.add_argument('--variants-per-key', type=int, default=3)
    parser.add_argument('--max-candidates', type=int, default=100)
    parser.add_argument('--hit-power', type=float, default=3.0)
    parser.add_argument('--validation-power', type=float, default=1.5)
    parser.add_argument('--reduction-power', type=float, default=1.0)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.25)
    parser.add_argument('--payout-direction-weight', type=float, default=0.08)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=50)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=350)
    parser.add_argument('--per-bucket-keep', type=int, default=1)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    parser.add_argument('--force-upload', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-fresh-workers', action='store_true')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    _v14_parse_variants(args.variants)

    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)

    if args.max_tests > 1 and not args.internal_worker and not args.no_fresh_workers:
        _run_fresh_workers_v14(args, app_file, db_file, out_dir)
        return

    ns = _load_app_functions(app_file, fast_no_supermakro=False)
    db = ns['load_database'](str(db_file), 13)
    print('\n'+'='*76, flush=True)
    print('TIPSET AI V14 – B-BASERAD PARAMETEROPTIMERARE · ETT VINNANDE PAKET', flush=True)
    print(f'Test: {args.max_tests} omgångar inom {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '), flush=True)
    print(f'Varianter: {args.variants} · bas V13-B · struktur 30/30', flush=True)
    print(f'Ram: {args.frame_profile} · beam {args.beam_width} · kandidater {args.max_candidates}', flush=True)
    print('='*76, flush=True)
    detail, meta = _run_backtest_v14(ns, db, args)
    _write_v14_outputs(detail, args, out_dir, app_file, db_file, worker_count=1 if args.internal_worker else 0)


# V14 main disabled; V15 main is appended below.


# =============================================================================
# V15 – GRUPP-/SUPERFILTEROPTIMERARE ovanpå V14-B5
# =============================================================================

V15_VARIANT_DEFS = {
    'B5': {
        'label': 'B5 kontroll',
        'description': 'Oförändrad V14-B5 radpress som kontroll.',
        'mode': 'b5',
        'overrides': {},
    },
    'SG2': {
        'label': 'SG2 strukturgrupp max 2',
        'description': 'Tar bort individuella strukturfilter och lägger in struktur som gruppregel max 2 missar.',
        'mode': 'structure_group',
        'structure_miss': [2],
        'drop_individual_structure': True,
        'overrides': {},
    },
    'SG3': {
        'label': 'SG3 strukturgrupp max 3',
        'description': 'Tar bort individuella strukturfilter och lägger in struktur som gruppregel max 3 missar.',
        'mode': 'structure_group',
        'structure_miss': [3],
        'drop_individual_structure': True,
        'overrides': {},
    },
    'FAM': {
        'label': 'FAM familjegrupper',
        'description': 'B5 + mjuka gruppregler per filterfamilj. Behåller individuella profilfilter men ersätter individuell struktur.',
        'mode': 'family_groups',
        'structure_miss': [2, 3],
        'family_miss': [0, 1, 2, 3],
        'cross_family': False,
        'drop_individual_structure': True,
        'overrides': {},
    },
    'SUPER': {
        'label': 'SUPER familj + motfilter',
        'description': 'B5 + familjegrupper + tvärfamiljegrupper som kan fånga filter som delvis jobbar emot varandra.',
        'mode': 'super_groups',
        'structure_miss': [2, 3],
        'family_miss': [0, 1, 2, 3, 4],
        'cross_family': True,
        'drop_individual_structure': True,
        'overrides': {
            'beam_width': 80,
            'archive_width': 600,
        },
    },
    'GONLY': {
        'label': 'GONLY endast grupper',
        'description': 'Testar superfilter-idén rent: struktur/familje/tvärfamiljegrupper utan individuella profilfilter.',
        'mode': 'groups_only',
        'structure_miss': [2, 3],
        'family_miss': [0, 1, 2, 3, 4],
        'cross_family': True,
        'drop_individual_structure': True,
        'drop_individual_profile': True,
        'overrides': {
            'beam_width': 80,
            'archive_width': 600,
        },
    },
}

# Låt V13/V14-byggaren känna igen V15-varianterna vid score/sortering.
for _k, _cfg in V15_VARIANT_DEFS.items():
    if _k not in V13_VARIANT_DEFS:
        V13_VARIANT_DEFS[_k] = {
            'label': _cfg['label'],
            'fixed_payout': True,
            'bundle_search': False,
            'description': _cfg['description'],
        }


def _v15_parse_variants(value: str) -> List[str]:
    raw = str(value or 'B5,SG2,SG3,FAM,SUPER,GONLY').replace(';', ',').replace('+', ',').split(',')
    out = []
    for x in raw:
        k = x.strip().upper()
        if not k:
            continue
        if k not in V15_VARIANT_DEFS:
            raise ValueError(f'Okänd V15-variant: {k}. Tillåtna: {", ".join(V15_VARIANT_DEFS)}')
        if k not in out:
            out.append(k)
    return out or ['B5', 'SG2', 'SG3', 'FAM', 'SUPER', 'GONLY']


def _v15_variant_args(args, variant_id: str):
    # Alla V15-varianter utgår från V14-B5, därefter läggs variantens egna override på.
    vargs = argparse.Namespace(**vars(args))
    b5_cfg = V14_VARIANT_DEFS['B5']
    for k, v in (b5_cfg.get('overrides') or {}).items():
        setattr(vargs, k, v)
    cfg = V15_VARIANT_DEFS.get(str(variant_id).upper(), V15_VARIANT_DEFS['B5'])
    for k, v in (cfg.get('overrides') or {}).items():
        setattr(vargs, k, v)
    if not hasattr(vargs, 'candidate_min_hit'):
        setattr(vargs, 'candidate_min_hit', int(getattr(vargs, 'min_hit', 29)))
    return vargs


def _v15_bits_to_bool(bits: int, n: int) -> np.ndarray:
    return np.array([(int(bits) >> i) & 1 for i in range(int(n))], dtype=bool)


def _v15_group_bits(cands: List[dict], n: int, required: int, field: str) -> int:
    if n <= 0:
        return 0
    if not cands:
        return 0
    counts = np.zeros(int(n), dtype=np.int16)
    for c in cands:
        counts += _v15_bits_to_bool(int(c.get(field, 0)), int(n)).astype(np.int16)
    mask = counts >= int(required)
    return _bool_mask_to_bits(mask)


def _v15_group_family(c: dict) -> str:
    cat = str(c.get('category', '') or '')
    text = f"{c.get('name','')} {c.get('key','')} {cat}".lower()
    if cat in STRUCTURE_CATEGORIES or 'struktur' in text or 'uppkomst' in text or 'dubblett' in text or 'tripp' in text or 'luck' in text or 'svit' in text:
        return 'Struktur'
    if 'abc' in text:
        return 'ABC'
    if 'sekvens' in text or 'sequence' in text:
        return 'FAT-sekvens'
    if 'fat' in text or cat in {'FAT', 'FAT-sekvenser'}:
        return 'FAT'
    if 'favorit' in text or 'fav' in text:
        return 'Favorit'
    if 'skräll' in text or 'shock' in text or 'u10' in text or 'u15' in text or 'u20' in text:
        return 'Skräll'
    if 'rank' in text or 'poäng' in text or 'point' in text:
        return 'Poäng/rank'
    if 'värde' in text or 'svår' in text or 'utdel' in text or 'sft' in text:
        return 'Värde/svårighet'
    return cat or 'Övrigt'


def _v15_category_for_group(family: str, hist_hit: int, htot: int) -> str:
    # Strukturgrupp som är 30/30 får gå i strukturfasen. Strukturgrupp som är 29/30
    # behandlas som profilregel, annars skulle V13-byggaren kasta bort den.
    if str(family) == 'Struktur' and int(hist_hit) >= int(htot):
        return 'Struktur'
    if str(family) in {'FAT', 'ABC', 'FAT-sekvens'}:
        return 'FAT'
    if str(family) in {'Favorit', 'Skräll'}:
        return 'Favorit & skräll'
    return 'Värde & svårighet'


def _v15_unique_best_per_key(cands: List[dict], limit: int) -> List[dict]:
    by_key: Dict[str, dict] = {}
    for c in sorted(cands, key=lambda x: _v13_state_sort_proxy_candidate(x), reverse=True):
        k = str(c.get('key', ''))
        if not k:
            continue
        if k not in by_key:
            by_key[k] = c
    vals = list(by_key.values())
    vals.sort(key=lambda x: _v13_state_sort_proxy_candidate(x), reverse=True)
    return vals[:max(1, int(limit))]


def _v13_state_sort_proxy_candidate(c: dict) -> tuple:
    return (
        int(c.get('hist_hit', 0)),
        float(c.get('val_pct', 0.0)),
        float(c.get('red_pct', 0.0)),
        float(c.get('payout_direction_pct', c.get('payout_lift_pct', 0.0))),
        float(c.get('gap_score', 0.0)),
        -int(c.get('frame_keep', 10**12)),
    )


def _v15_make_group_candidate(
    family: str,
    label: str,
    cands: List[dict],
    *,
    max_miss: int,
    htot: int,
    vtot: int,
    ftot: int,
    hist_payout: np.ndarray,
    min_hist_hit: int,
    min_val_pct: float,
) -> Optional[dict]:
    cands = [c for c in cands if c.get('key')]
    m = len(cands)
    if m <= 0:
        return None
    miss = max(0, min(int(max_miss), m - 1))
    required = max(1, m - miss)
    hist_bits = _v15_group_bits(cands, htot, required, 'hist_bits')
    val_bits = _v15_group_bits(cands, vtot, required, 'val_bits') if vtot else 0
    frame_bits = _v15_group_bits(cands, ftot, required, 'frame_bits')
    hist_hit = int(hist_bits.bit_count())
    val_hit = int(val_bits.bit_count()) if vtot else 0
    frame_keep = int(frame_bits.bit_count())
    if frame_keep <= 0 or frame_keep >= int(ftot):
        return None
    if hist_hit < int(min_hist_hit):
        return None
    val_pct = 100.0 * val_hit / max(1, int(vtot)) if vtot else 100.0
    if vtot and val_pct + 1e-9 < float(min_val_pct):
        return None
    direction = _v13_payout_direction_from_removed(int(hist_bits), int(htot), hist_payout)
    child_keys = [str(c.get('key')) for c in cands]
    child_names = [str(c.get('name', c.get('key', ''))) for c in cands]
    category = _v15_category_for_group(family, hist_hit, htot)
    red_pct = 100.0 - 100.0 * frame_keep / max(1, int(ftot))
    avg_gap = float(np.mean([float(c.get('gap_score', 0.0)) for c in cands])) if cands else 0.0
    avg_payout = float(np.mean([float(c.get('payout_direction_pct', c.get('payout_lift_pct', 0.0))) for c in cands])) if cands else 0.0
    key = f'GROUP::{label}::{m}::{miss}::{hash(tuple(child_keys)) & 0xfffffff}'
    return {
        'key': key,
        'name': f'Grupp {label}: minst {required}/{m} sitter',
        'category': category,
        'interval': (float(required), float(m)),
        'interval_txt': f'minst {required}/{m} · max {miss} miss',
        'source': 'v15_group_rule',
        'hist_bits': int(hist_bits),
        'val_bits': int(val_bits),
        'frame_bits': int(frame_bits),
        'hist_hit': int(hist_hit),
        'hist_total': int(htot),
        'hist_pct': 100.0 * hist_hit / max(1, int(htot)),
        'val_hit': int(val_hit),
        'val_total': int(vtot),
        'val_pct': float(val_pct),
        'frame_keep': int(frame_keep),
        'red_pct': float(red_pct),
        'gap_score': float(avg_gap),
        'density_score': 0.0,
        'excluded_hist': int(htot - hist_hit),
        'payout_lift_pct': float(direction.get('direction_pct', 0.0)),
        'payout_direction_pct': float(direction.get('direction_pct', 0.0)),
        'removed_hist_count': int(direction.get('removed_count', 0)),
        'removed_payout_mean': float(direction.get('removed_mean', 0.0)),
        'is_v15_group': True,
        'group_family': str(family),
        'group_label': str(label),
        'group_size': int(m),
        'group_required': int(required),
        'group_max_miss': int(miss),
        'group_child_keys': child_keys,
        'group_child_names': child_names,
        'group_children': [
            {
                'key': ch.get('key'),
                'name': ch.get('name'),
                'category': ch.get('category'),
                'interval': ch.get('interval'),
                'interval_txt': ch.get('interval_txt'),
            }
            for ch in cands
        ],
        'child_avg_payout_direction_pct': float(avg_payout),
    }


def _v15_make_group_candidates(candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, args, variant_id: str) -> List[dict]:
    cfg = V15_VARIANT_DEFS.get(str(variant_id).upper(), V15_VARIANT_DEFS['B5'])
    if cfg.get('mode') == 'b5':
        return []
    fams: Dict[str, List[dict]] = {}
    for c in candidates:
        fams.setdefault(_v15_group_family(c), []).append(c)
    groups = []
    # Struktur: använd topplistan av 30/30-strukturfilter, en kandidat per key.
    if 'Struktur' in fams:
        sc = [c for c in fams['Struktur'] if int(c.get('hist_hit', 0)) >= int(htot)]
        sc = _v15_unique_best_per_key(sc, int(args.v15_group_max_filters))
        for miss in cfg.get('structure_miss', []):
            g = _v15_make_group_candidate(
                'Struktur', f'Struktur max {miss}', sc,
                max_miss=int(miss), htot=htot, vtot=vtot, ftot=ftot,
                hist_payout=hist_payout,
                min_hist_hit=int(args.min_hit),
                min_val_pct=float(getattr(args, 'min_package_val_pct', 85.0)),
            )
            if g is not None:
                groups.append(g)
    if cfg.get('mode') in {'family_groups', 'super_groups', 'groups_only'}:
        for fam, vals in sorted(fams.items()):
            if fam == 'Struktur':
                continue
            vals = _v15_unique_best_per_key(vals, int(args.v15_group_max_filters))
            if len(vals) < int(args.v15_min_group_size):
                continue
            sizes = sorted(set([int(args.v15_min_group_size), 4, 6, 8, 10, 12, min(len(vals), int(args.v15_group_max_filters))]))
            sizes = [s for s in sizes if int(args.v15_min_group_size) <= s <= len(vals)]
            for s in sizes:
                sub = vals[:s]
                for miss in cfg.get('family_miss', [0, 1, 2]):
                    if int(miss) >= s:
                        continue
                    g = _v15_make_group_candidate(
                        fam, f'{fam} max {miss}', sub,
                        max_miss=int(miss), htot=htot, vtot=vtot, ftot=ftot,
                        hist_payout=hist_payout,
                        min_hist_hit=int(args.min_hit),
                        min_val_pct=float(getattr(args, 'min_package_val_pct', 85.0)),
                    )
                    if g is not None:
                        groups.append(g)
        if cfg.get('cross_family'):
            # Tvärfamilj: välj de bästa 1–2 kandidaterna från varje icke-strukturfamilj.
            cross = []
            for fam, vals in sorted(fams.items()):
                if fam == 'Struktur':
                    continue
                best = _v15_unique_best_per_key(vals, max(1, int(args.v15_cross_per_family)))
                cross.extend(best)
            cross = _v15_unique_best_per_key(cross, int(args.v15_cross_max_filters))
            if len(cross) >= int(args.v15_min_group_size):
                for miss in cfg.get('family_miss', [1, 2, 3, 4]):
                    if int(miss) >= len(cross):
                        continue
                    g = _v15_make_group_candidate(
                        'Tvärfamilj', f'Tvärfamilj max {miss}', cross,
                        max_miss=int(miss), htot=htot, vtot=vtot, ftot=ftot,
                        hist_payout=hist_payout,
                        min_hist_hit=int(args.min_hit),
                        min_val_pct=float(getattr(args, 'min_package_val_pct', 85.0)),
                    )
                    if g is not None:
                        groups.append(g)
    # Dedupe på exakt mask + gruppnamn.
    out, seen = [], set()
    for g in sorted(groups, key=lambda x: _v13_state_sort_proxy_candidate(x), reverse=True):
        sig = (g.get('group_label'), int(g.get('hist_bits', 0)), int(g.get('val_bits', 0)), int(g.get('frame_bits', 0)))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(g)
    return out[:max(0, int(args.v15_max_group_candidates))]


def _v15_transform_candidates(candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, args, variant_id: str) -> Tuple[List[dict], List[dict]]:
    cfg = V15_VARIANT_DEFS.get(str(variant_id).upper(), V15_VARIANT_DEFS['B5'])
    group_cands = _v15_make_group_candidates(candidates, htot, vtot, ftot, hist_payout, args, variant_id)
    base = []
    for c in candidates:
        if cfg.get('drop_individual_structure') and _is_structure_candidate(c):
            continue
        if cfg.get('drop_individual_profile') and _is_profile_candidate(c):
            continue
        base.append(c)
    final = base + group_cands
    # Ingen hård filterlimit i slutpaketet, men kandidatlistan behöver dedupas.
    out, seen = [], set()
    for c in sorted(final, key=lambda x: _v13_state_sort_proxy_candidate(x), reverse=True):
        sig = (c.get('key'), c.get('interval_txt'))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out, group_cands


def _v15_filter_candidates_for_variant(candidates: List[dict], vargs, htot: int) -> List[dict]:
    # B5-filtering som bas, men låt group candidates som skapats senare slippa denna passeringsrunda.
    return _v14_filter_candidates_for_variant(candidates, vargs, htot)


def _v15_child_passes_row(ns: dict, row: str, specs_by_key: dict, child: dict) -> bool:
    key = child.get('key')
    spec = specs_by_key.get(key)
    interval = child.get('interval')
    if spec is None or interval is None:
        return True
    try:
        getter = spec.get('getter')
        if getter is not None:
            v = getter(row)
            return float(v) >= float(interval[0]) and float(v) <= float(interval[1])
    except Exception:
        pass
    try:
        return bool(globals().get('_ACTIVE_V9_NS', ns).get('_spec_pass', lambda *_: True)(row, spec, interval))
    except Exception:
        return False


def _v15_package_passes_row(ns: dict, correct: str, specs: list, pkg: dict) -> Tuple[bool, str]:
    specs_by_key = {s.get('key'): s for s in specs or []}
    for ix, c in enumerate(pkg.get('filters', []) or [], start=1):
        if c.get('is_v15_group'):
            children = c.get('group_children', []) or []
            ok_count = sum(1 for ch in children if _v15_child_passes_row(ns, correct, specs_by_key, ch))
            required = int(c.get('group_required', len(children)))
            if ok_count < required:
                return False, f'V15 gruppfilter miss steg {ix}: {c.get("name")} · facit klarade {ok_count}/{len(children)}, krävde {required}'
        else:
            key = c.get('key')
            spec = specs_by_key.get(key, c)
            interval = c.get('interval')
            try:
                getter = spec.get('getter')
                if getter is not None:
                    v = getter(correct)
                    ok = float(v) >= float(interval[0]) and float(v) <= float(interval[1])
                else:
                    ok = bool(ns.get('_spec_pass', lambda *_: True)(correct, spec, interval))
            except Exception:
                try:
                    ok = bool(ns.get('_spec_pass', lambda *_: True)(correct, spec, interval))
                except Exception:
                    ok = False
            if not ok:
                return False, f'Filter miss steg {ix}: {c.get("name")} {c.get("interval_txt")}'
    return True, 'OK'


def _v15_diagnose_facit(ns: dict, correct: str, specs: list, pkg: dict) -> str:
    specs_by_key = {s.get('key'): s for s in specs or []}
    for ix, c in enumerate(pkg.get('filters', []) or [], start=1):
        if c.get('is_v15_group'):
            children = c.get('group_children', []) or []
            failed = []
            passed = 0
            for ch in children:
                ok = _v15_child_passes_row(ns, correct, specs_by_key, ch)
                if ok:
                    passed += 1
                else:
                    failed.append({'key': ch.get('key'), 'namn': ch.get('name'), 'intervall': ch.get('interval_txt')})
            required = int(c.get('group_required', len(children)))
            if passed < required:
                return json.dumps({
                    'dödande_steg': ix,
                    'typ': 'V15 gruppfilter',
                    'filter': c.get('name'),
                    'familj': c.get('group_family'),
                    'krav': f'{required}/{len(children)}',
                    'facit_klarade': passed,
                    'facit_missade': len(children) - passed,
                    'första_missade_barn': failed[:12],
                }, ensure_ascii=False)
        else:
            ok, _reason = _v15_package_passes_row(ns, correct, specs, {'filters': [c]})
            if not ok:
                try:
                    return _v13_diagnose_facit(ns, correct, specs, {'filters': [c]})
                except Exception:
                    return json.dumps({'dödande_steg': ix, 'filter': c.get('name'), 'intervall': c.get('interval_txt')}, ensure_ascii=False)
    return ''


def _v15_group_json(pkg: dict) -> str:
    rows = []
    for c in pkg.get('filters', []) or []:
        if c.get('is_v15_group'):
            rows.append({
                'namn': c.get('name'),
                'familj': c.get('group_family'),
                'kategori': c.get('category'),
                'krav': f"{c.get('group_required')}/{c.get('group_size')}",
                'max_miss': c.get('group_max_miss'),
                'hist': f"{c.get('hist_hit')}/{c.get('hist_total')}",
                'val_pct': round(float(c.get('val_pct', 0.0)), 1),
                'frame_red_pct_alone': round(float(c.get('red_pct', 0.0)), 1),
                'barn': c.get('group_child_names', [])[:25],
            })
    return json.dumps(rows, ensure_ascii=False)


def _v15_group_counts(pkg: dict) -> dict:
    filters = list(pkg.get('filters', []) or [])
    groups = [c for c in filters if c.get('is_v15_group')]
    out = {
        'Gruppfilter totalt': len(groups),
        'Individuella filter': len(filters) - len(groups),
        'Strukturgruppfilter': sum(1 for c in groups if c.get('group_family') == 'Struktur'),
        'Familjegruppfilter': sum(1 for c in groups if c.get('group_family') not in {'Struktur', 'Tvärfamilj'}),
        'Tvärfamiljgruppfilter': sum(1 for c in groups if c.get('group_family') == 'Tvärfamilj'),
    }
    return out


def _run_backtest_v15(ns: dict, global_db: pd.DataFrame, args) -> Tuple[pd.DataFrame, dict]:
    antal_matcher = 13
    norm = ns['normalize_single_row_text']
    similar_history = ns['_similar_history_for_backtest']
    ranked_frame = ns['_ranked_frame_like_widths']
    generate_rows_from_frame = ns['generate_rows_from_frame']
    build_clean_filter_specs = ns['build_clean_filter_specs']

    global _ACTIVE_V9_NS
    _ACTIVE_V9_NS = ns

    global_db, supersvar_count = _prepare_supersvar_payout(global_db, int(args.pay_max))
    db = _prepare_db(ns, global_db, antal_matcher, int(args.pay_min), int(args.pay_max))
    if db.empty:
        return pd.DataFrame(), {'error': 'Inga giltiga omgångar i utdelningsintervallet.'}

    variants = _v15_parse_variants(args.variants)
    spik, halv, hel = _parse_frame_profile(args.frame_profile)
    template = _base_frame(spik, halv, hel)
    test_rows = list(db.iterrows())[int(args.test_offset):int(args.test_offset)+int(args.max_tests)]
    out = []
    for ti, (idx, test_row) in enumerate(test_rows, 1):
        _release_test_memory()
        t0_case = time.time()
        correct = norm(test_row.get('Correct_Row', ''))
        input_vec = test_row.get('Prob_Vector', [])
        test_date = test_row.get('Datum', None)
        payout = float(pd.to_numeric(pd.Series([test_row.get('Payout', 0)]), errors='coerce').fillna(0).iloc[0])
        print(f'V15 testfall {ti}/{len(test_rows)} · {str(test_date)[:10]} · utdelning {payout:,.0f} kr'.replace(',', ' '), flush=True)
        try:
            sim_df = similar_history(
                global_db, input_vec, antal_matcher,
                top_n=int(args.top_n), pay_min=int(args.pay_min), pay_max=int(args.pay_max),
                exclude_index=idx, mode=str(args.mode), test_date=test_date,
            )
            wide_df = similar_history(
                global_db, input_vec, antal_matcher,
                top_n=max(int(args.top_n), int(args.wide_n)), pay_min=int(args.pay_min), pay_max=int(args.pay_max),
                exclude_index=idx, mode=str(args.mode), test_date=test_date,
            )
            if len(sim_df) < int(args.top_n):
                raise RuntimeError(f'För få liknande omgångar: {len(sim_df)} av önskade {args.top_n}.')
            engine_frame = ranked_frame(template, input_vec, antal_matcher)
            frame_rows, _, ok, msg = generate_rows_from_frame(engine_frame, max_rows=None)
            if not ok or not frame_rows:
                raise RuntimeError(f'Kunde inte skapa ram: {msg}')
            specs = build_clean_filter_specs(
                sim_df, input_vec, antal_matcher,
                slider_u_count=3, target_hist_pct=int(args.filter_hist_target_pct),
                u_rows=None, hist_df=global_db, max_shock_pct=22,
                candidate_rows=frame_rows, include_supermakro=not bool(args.fast_no_supermakro),
            )
            # Bygg en bred B5-kandidatpool en gång per omgång.
            base_args = _v15_variant_args(args, 'B5')
            global_settings = {
                'profile_min_hit': int(getattr(base_args, 'candidate_min_hit', getattr(base_args, 'min_hit', 29))),
                'min_candidate_val_pct': float(getattr(base_args, 'min_candidate_val_pct', 85.0)),
                'min_structure_val_pct': float(getattr(base_args, 'min_structure_val_pct', 95.0)),
                'min_gap_score': float(getattr(base_args, 'min_gap_score', 0.75)),
            }
            candidates, htot, vtot, ftot, hist_payout = _build_dynamic_candidates_v9(
                ns, specs, sim_df, frame_rows, engine_frame, antal_matcher,
                profile_min_hit=int(global_settings['profile_min_hit']),
                variants_per_key=int(args.variants_per_key),
                max_candidates=int(args.max_candidates),
                validation_df=wide_df,
                min_candidate_val_pct=float(global_settings['min_candidate_val_pct']),
                min_structure_val_pct=float(global_settings['min_structure_val_pct']),
                min_gap_score=float(global_settings['min_gap_score']),
                frame_adapt=True,
            )
            candidates = _v13_enrich_candidates(candidates, hist_payout, htot)
            print(f'    V15 kandidatbygge: kandidater={len(candidates)} · historik={htot} · validering={vtot}', flush=True)
            if not candidates:
                raise RuntimeError('Inga giltiga V15-kandidater.')
            for variant_id in variants:
                t0 = time.time()
                try:
                    vargs = _v15_variant_args(args, variant_id)
                    cand0 = _v15_filter_candidates_for_variant(candidates, vargs, htot)
                    cand_v, group_cands = _v15_transform_candidates(cand0, htot, vtot, ftot, hist_payout, vargs, variant_id)
                    print(f'  Variant {variant_id}: {V15_VARIANT_DEFS[variant_id]["label"]} · kandidater={len(cand_v)} · grupper={len(group_cands)}', flush=True)
                    if not cand_v:
                        raise RuntimeError('Inga kandidater efter V15-gruppering.')
                    pkg, meta = _build_cluster_payout_package_v13_from_candidates(
                        ns, cand_v, htot, vtot, ftot, hist_payout,
                        frame_rows, engine_frame, antal_matcher, vargs, variant_id,
                    )
                    if pkg is None:
                        raise RuntimeError(meta.get('error', 'Inget paket'))
                    pkg_pass, fail_reason = _v15_package_passes_row(ns, correct, specs, pkg)
                    diag = '' if pkg_pass else _v15_diagnose_facit(ns, correct, specs, pkg)
                    counts = _v15_group_counts(pkg)
                    out.append({
                        'Variant': variant_id,
                        'Variantnamn': V15_VARIANT_DEFS[variant_id]['label'],
                        'Datum': str(test_date)[:10],
                        'Utdelning': int(round(payout)),
                        'Facit': correct,
                        'Status': 'OK',
                        'Paket klarar facit': 'Ja' if pkg_pass else 'Nej',
                        'Orsak': 'OK' if pkg_pass else fail_reason,
                        'Missdiagnos JSON': diag,
                        'Liknande historik': len(sim_df),
                        'Bred validering': len(wide_df),
                        'Paketträff': f"{pkg['hist_hit']}/{pkg['hist_total']}",
                        'Valideringsträff': f"{pkg['val_hit']}/{pkg['val_total']}",
                        'Grundram rader': len(frame_rows),
                        'Paketrader': int(pkg['frame_after']),
                        'Reducerar %': round(float(pkg['reduction_pct']), 2),
                        'Gemensamt score': round(float(pkg['joint_score']), 4),
                        'Utdelningsriktning %': round(float(pkg.get('payout_direction_pct', 0.0)), 2),
                        'Borttagna hist omgångar': int(pkg.get('removed_hist_count', 0)),
                        'Medel spret-gap': round(float(pkg['cluster_mean']), 3),
                        'Filter totalt': int(pkg['num_filters']),
                        'Strukturfilter': int(pkg['structure_filters']),
                        'Profilfilter': int(pkg['profile_filters']),
                        'FAT/ABC-filter': int(pkg['fat_filters']),
                        'Värde/favorit/skräll': int(pkg['edge_filters']),
                        **counts,
                        'Tillgängliga gruppkandidater': int(len(group_cands)),
                        'Filter JSON': _v13_json_filters(pkg),
                        'Gruppfilter JSON': _v15_group_json(pkg),
                        'Steg JSON': json.dumps(pkg.get('steps', []), ensure_ascii=False),
                        'Supersvår testomgång': 'Ja' if bool(test_row.get('_BT_No13Winner', False)) else 'Nej',
                        'Sekunder': round(time.time()-t0, 2),
                    })
                except Exception as e:
                    out.append({
                        'Variant': variant_id,
                        'Variantnamn': V15_VARIANT_DEFS.get(variant_id, {}).get('label', variant_id),
                        'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct,
                        'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e),
                        'Traceback': traceback.format_exc(),
                        'Sekunder': round(time.time()-t0, 2),
                    })
                    print(f'  FEL variant {variant_id}: {e}', flush=True)
        except Exception as e:
            for variant_id in variants:
                out.append({
                    'Variant': variant_id,
                    'Variantnamn': V15_VARIANT_DEFS.get(variant_id, {}).get('label', variant_id),
                    'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct,
                    'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e),
                    'Traceback': traceback.format_exc(),
                    'Sekunder': round(time.time()-t0_case, 2),
                })
            print(f'  FEL testfall: {e}', flush=True)
    return pd.DataFrame(out), {'supersvar_count': supersvar_count, 'eligible_rounds': len(db), 'variants': variants}


def _summarize_v15(detail: pd.DataFrame) -> pd.DataFrame:
    summary = _summarize_v13(detail)
    if summary is None or summary.empty:
        return summary
    # Lägg till gruppmedianer om de finns.
    ok = detail[detail['Status'].astype(str) == 'OK'].copy() if 'Status' in detail.columns else detail.copy()
    extra_rows = []
    for variant, grp in ok.groupby('Variant') if 'Variant' in ok.columns else []:
        def med(col):
            return float(pd.to_numeric(grp.get(col), errors='coerce').dropna().median()) if col in grp.columns and len(pd.to_numeric(grp.get(col), errors='coerce').dropna()) else 0.0
        extra_rows.append({
            'Variant': variant,
            'Median gruppfilter': med('Gruppfilter totalt'),
            'Median individuella filter': med('Individuella filter'),
            'Median tvärfamiljgrupper': med('Tvärfamiljgruppfilter'),
        })
    if extra_rows:
        summary = summary.merge(pd.DataFrame(extra_rows), on='Variant', how='left')
    def val(r, col, default=0.0):
        try:
            x = r.get(col, default)
            return float(x) if pd.notna(x) else float(default)
        except Exception:
            return float(default)
    def win_key(r):
        return (
            int(val(r, 'Träffar', 0)),
            val(r, 'Träff %', 0.0),
            -val(r, 'Median paketrader', 10**12),
            -val(r, 'Medel paketrader', 10**12),
            val(r, 'Median reducering %', 0.0),
            val(r, 'Median utdelningsriktning %', 0.0),
        )
    summary['Vinnare'] = ''
    best_idx = max(summary.index, key=lambda i: win_key(summary.loc[i]))
    summary.loc[best_idx, 'Vinnare'] = 'JA'
    return summary.sort_values(by=['Vinnare','Träffar','Median paketrader'], ascending=[False,False,True])


def _write_v15_outputs(detail: pd.DataFrame, args, out_dir: Path, app_file: Path, db_file: Path, worker_count: int = 0):
    summary = _summarize_v15(detail)
    winner_variant = _v13_winner_variant(summary)
    detail_path = out_dir / f'{args.output_prefix}_detail.csv'
    summary_path = out_dir / f'{args.output_prefix}_summary.csv'
    report_path = out_dir / f'{args.output_prefix}_report.txt'
    winner_detail_path = out_dir / f'{args.output_prefix}_winner_detail.csv'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    if winner_variant and 'Variant' in detail.columns:
        detail[detail['Variant'].astype(str) == str(winner_variant)].to_csv(winner_detail_path, index=False)
    else:
        pd.DataFrame().to_csv(winner_detail_path, index=False)
    variant_lines = []
    for k in _v15_parse_variants(args.variants):
        cfg = V15_VARIANT_DEFS[k]
        variant_lines.append(f'{k}: {cfg["label"]} – {cfg["description"]}')
    lines = [
        'TIPSET AI – V15 GRUPP-/SUPERFILTEROPTIMERARE',
        '='*76,
        f'Appbas: {app_file.name}',
        f'Databas: {db_file.name}',
        f'Testomgångar: {args.max_tests}',
        f'Utdelningsintervall: {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '),
        f'Liknande historik: {args.top_n}; bred validering: {args.wide_n}',
        'Bas: V14-B5 radpress. V15 testar struktur-/familje-/tvärfamiljegrupper.',
        'Synlig slutmotor: en vinnare; varianterna är bara Colab-utvecklingstest.',
        'Gruppregel: minst X av Y filter i familjen ska sitta. Detta kan fånga filter som motverkar varandra men tillsammans blir ett superfilter.',
        f'Max antal filter i slutpaket: ingen hård gräns. Gruppkandidater max: {args.v15_max_group_candidates}',
        f'Beam width: {args.beam_width}; arkiv: {args.archive_width}; baskandidater: {args.max_candidates}',
        f'Färska workers: {worker_count}',
        '',
        'VARIANTER',
        '-'*76,
        *variant_lines,
        '',
        'SAMMANFATTNING',
        '-'*76,
        summary.to_string(index=False) if isinstance(summary, pd.DataFrame) and not summary.empty else '(tom)',
        '',
        f'Detail: {detail_path}',
        f'Summary: {summary_path}',
        f'Winner detail: {winner_detail_path}',
    ]
    if winner_variant:
        lines.append(f'Vinnare: {winner_variant}')
        lines.append('')
        lines.append('TOLKNING V36: detta är snabb slumpkontroll. En variant är intressant om den klarar minst 6/8 och median helst 2400–2800. Kör flera seeds innan 20/30-test.')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V15 SAMMANFATTNING', flush=True)
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        print(summary.to_string(index=False), flush=True)
    if winner_variant:
        print(f'Vinnare: {winner_variant}', flush=True)
    print(f'Rapport: {report_path}', flush=True)
    print(f'Summary: {summary_path}', flush=True)
    print(f'Detail: {detail_path}', flush=True)


def _resolve_v15_script_path() -> Optional[Path]:
    candidates = []
    raw_file = globals().get('__file__')
    if raw_file:
        candidates.append(Path(str(raw_file)))
    try:
        argv0 = Path(str(sys.argv[0]))
        if argv0.suffix.lower() == '.py':
            candidates.append(argv0)
    except Exception:
        pass
    preferred_names = [
        'package_group_optimizer_v15_colab_upload.py',
        'package_cluster_payout_synergy_v15_group_optimizer_colab_upload.py',
    ]
    for directory in [Path.cwd(), Path('/content'), Path('/mnt/data')]:
        if not directory.exists():
            continue
        for name in preferred_names:
            candidates.append(directory / name)
        candidates.extend(sorted(directory.glob('package*group*v15*.py')))
        candidates.extend(sorted(directory.glob('package_cluster_payout_synergy_v15*.py')))
    valid, seen = [], set()
    for candidate in candidates:
        try:
            candidate = Path(candidate)
            if not candidate.exists() or candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if str(resolved) in seen:
                continue
            seen.add(str(resolved))
            sample = resolved.read_text(encoding='utf-8', errors='replace')
            if ('def main_v15' not in sample or 'V15_VARIANT_DEFS' not in sample or 'def _run_fresh_workers_v15' not in sample):
                continue
            score = 0
            name = resolved.name.lower()
            if name == 'package_group_optimizer_v15_colab_upload.py':
                score += 100
            if '/content/' in str(resolved):
                score += 30
            valid.append((score, resolved.stat().st_mtime, resolved))
        except Exception:
            continue
    if not valid:
        return None
    valid.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return valid[0][2]


def _run_v15_workers_inline(args, app_file: Path, db_file: Path, out_dir: Path):
    details = []
    for i in range(int(args.max_tests)):
        offset = int(args.test_offset) + i
        print('\n' + '=' * 76, flush=True)
        print(f'V15 sekventiellt testfall {i+1}/{args.max_tests} · offset {offset}', flush=True)
        print('=' * 76, flush=True)
        worker_args = argparse.Namespace(**vars(args))
        worker_args.max_tests = 1
        worker_args.test_offset = offset
        worker_args.internal_worker = True
        worker_args.no_fresh_workers = True
        ns = _load_app_functions(app_file, fast_no_supermakro=False)
        db = ns['load_database'](str(db_file), 13)
        detail, _meta = _run_backtest_v15(ns, db, worker_args)
        if isinstance(detail, pd.DataFrame) and not detail.empty:
            details.append(detail)
        del ns, db, detail
        gc.collect()
    if not details:
        raise SystemExit('Inga V15-resultat skapades i den sekventiella reservkörningen.')
    combined = pd.concat(details, ignore_index=True)
    _write_v15_outputs(combined, args, out_dir, app_file, db_file, worker_count=0)


def _run_fresh_workers_v15(args, app_file: Path, db_file: Path, out_dir: Path):
    script_path = _resolve_v15_script_path()
    if script_path is None:
        print('Varning: ingen uppladdad V15 .py-fil hittades. Kör sekventiellt med exakt beräkning.', flush=True)
        _run_v15_workers_inline(args, app_file, db_file, out_dir)
        return
    print(f'Worker-runner: {script_path}', flush=True)
    root = out_dir / f'_{args.output_prefix}_workers'
    root.mkdir(parents=True, exist_ok=True)
    worker_dirs = []
    pass_flags = [
        'variants','top_n','wide_n','pay_min','pay_max','min_hit','min_candidate_val_pct',
        'min_package_val_pct','min_structure_val_pct','min_structure_package_val_pct',
        'min_gap_score','min_unique_rows','filter_hist_target_pct','frame_profile',
        'beam_width','archive_width','structure_seed_count','variants_per_key','max_candidates',
        'hit_power','validation_power','reduction_power','payout_weight','cluster_weight','mode',
        'payout_direction_weight','bundle_pool_size','triple_pool_size','max_bundle_trials_per_state',
        'max_bundle_keep_per_state','row_bucket_size','per_bucket_keep',
        'v15_group_max_filters','v15_max_group_candidates','v15_min_group_size','v15_cross_per_family','v15_cross_max_filters',
    ]
    flag_map = {
        'variants':'--variants','top_n':'--top-n','wide_n':'--wide-n','pay_min':'--pay-min','pay_max':'--pay-max',
        'min_hit':'--min-hit','min_candidate_val_pct':'--min-candidate-val-pct',
        'min_package_val_pct':'--min-package-val-pct','min_structure_val_pct':'--min-structure-val-pct',
        'min_structure_package_val_pct':'--min-structure-package-val-pct','min_gap_score':'--min-gap-score',
        'min_unique_rows':'--min-unique-rows','filter_hist_target_pct':'--filter-hist-target-pct',
        'frame_profile':'--frame-profile','beam_width':'--beam-width','archive_width':'--archive-width',
        'structure_seed_count':'--structure-seed-count','variants_per_key':'--variants-per-key',
        'max_candidates':'--max-candidates','hit_power':'--hit-power','validation_power':'--validation-power',
        'reduction_power':'--reduction-power','payout_weight':'--payout-weight','cluster_weight':'--cluster-weight',
        'mode':'--mode','payout_direction_weight':'--payout-direction-weight','bundle_pool_size':'--bundle-pool-size',
        'triple_pool_size':'--triple-pool-size','max_bundle_trials_per_state':'--max-bundle-trials-per-state',
        'max_bundle_keep_per_state':'--max-bundle-keep-per-state','row_bucket_size':'--row-bucket-size',
        'per_bucket_keep':'--per-bucket-keep','v15_group_max_filters':'--v15-group-max-filters',
        'v15_max_group_candidates':'--v15-max-group-candidates','v15_min_group_size':'--v15-min-group-size',
        'v15_cross_per_family':'--v15-cross-per-family','v15_cross_max_filters':'--v15-cross-max-filters',
    }
    for i in range(int(args.max_tests)):
        offset = int(args.test_offset)+i
        wd = root / f'case_{offset:03d}'
        wd.mkdir(parents=True, exist_ok=True)
        worker_dirs.append(wd)
        cmd = [sys.executable, str(script_path),
               '--app-file', str(app_file), '--db-file', str(db_file),
               '--out-dir', str(wd), '--output-prefix', args.output_prefix,
               '--max-tests', '1', '--test-offset', str(offset), '--internal-worker']
        for name in pass_flags:
            cmd.extend([flag_map[name], str(getattr(args, name))])
        if args.fast_no_supermakro:
            cmd.append('--fast-no-supermakro')
        else:
            cmd.append('--include-supermakro')
        if args.enable_triples:
            cmd.append('--enable-triples')
        print('\n'+'='*76, flush=True)
        print(f'V15 isolerat testfall {i+1}/{args.max_tests} · offset {offset}', flush=True)
        print('='*76, flush=True)
        subprocess.run(cmd, check=True)
    details = []
    for wd in worker_dirs:
        p = wd / f'{args.output_prefix}_detail.csv'
        if p.exists():
            details.append(pd.read_csv(p))
    if not details:
        raise SystemExit('Inga V15-workerresultat hittades.')
    combined = pd.concat(details, ignore_index=True)
    _write_v15_outputs(combined, args, out_dir, app_file, db_file, worker_count=len(worker_dirs))


def main_v15():
    parser = argparse.ArgumentParser(description='Tipset AI V15 – grupp-/superfilteroptimerare.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_group_optimizer_v15_test30')
    parser.add_argument('--variants', default='B5,SG2,SG3,FAM,SUPER,GONLY')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--min-hit', type=int, default=29)
    parser.add_argument('--min-candidate-val-pct', type=float, default=90.0)
    parser.add_argument('--min-package-val-pct', type=float, default=90.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=95.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=95.0)
    parser.add_argument('--min-gap-score', type=float, default=1.25)
    parser.add_argument('--min-unique-rows', type=int, default=1)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='2-8-3')
    parser.add_argument('--beam-width', type=int, default=24)
    parser.add_argument('--archive-width', type=int, default=120)
    parser.add_argument('--structure-seed-count', type=int, default=4)
    parser.add_argument('--variants-per-key', type=int, default=3)
    parser.add_argument('--max-candidates', type=int, default=100)
    parser.add_argument('--hit-power', type=float, default=3.0)
    parser.add_argument('--validation-power', type=float, default=1.5)
    parser.add_argument('--reduction-power', type=float, default=1.0)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.25)
    parser.add_argument('--payout-direction-weight', type=float, default=0.08)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=50)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=350)
    parser.add_argument('--per-bucket-keep', type=int, default=1)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--v15-group-max-filters', type=int, default=10)
    parser.add_argument('--v15-max-group-candidates', type=int, default=30)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=1)
    parser.add_argument('--v15-cross-max-filters', type=int, default=12)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    parser.add_argument('--force-upload', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-fresh-workers', action='store_true')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    _v15_parse_variants(args.variants)

    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)

    if args.max_tests > 1 and not args.internal_worker and not args.no_fresh_workers:
        _run_fresh_workers_v15(args, app_file, db_file, out_dir)
        return

    ns = _load_app_functions(app_file, fast_no_supermakro=False)
    db = ns['load_database'](str(db_file), 13)
    print('\n'+'='*76, flush=True)
    print('TIPSET AI V15 – GRUPP-/SUPERFILTEROPTIMERARE · ETT VINNANDE PAKET', flush=True)
    print(f'Test: {args.max_tests} omgångar inom {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '), flush=True)
    print(f'Varianter: {args.variants} · bas V14-B5 · gruppfilter minst X av Y', flush=True)
    print(f'Ram: {args.frame_profile} · beam {args.beam_width} · baskandidater {args.max_candidates} · gruppkandidater {args.v15_max_group_candidates}', flush=True)
    print('='*76, flush=True)
    detail, meta = _run_backtest_v15(ns, db, args)
    _write_v15_outputs(detail, args, out_dir, app_file, db_file, worker_count=1 if args.internal_worker else 0)




# ---------------------------------------------------------------------------
# V17 – SKALTEST FÖR NIKLAS VANLIGA SYSTEMRAMAR
# ---------------------------------------------------------------------------

def _v17_parse_frames(value: str):
    raw = str(value or '2-7-4,3-4-6,3-5-5').replace(';', ',').replace('+', ',').split(',')
    frames = []
    for x in raw:
        x = x.strip()
        if not x:
            continue
        _parse_frame_profile(x)  # validerar spik-halv-hel och summa 13
        frames.append(x)
    return frames or ['2-7-4','3-4-6','3-5-5']


def _v17_frame_rows_count(frame_profile: str) -> int:
    spik, halv, hel = _parse_frame_profile(frame_profile)
    return int((2 ** int(halv)) * (3 ** int(hel)))


def _v17_frame_label(frame_profile: str) -> str:
    spik, halv, hel = _parse_frame_profile(frame_profile)
    return f'{spik} spik / {hel} hel / {halv} halv'


def _v17_safe_prefix_part(text: str) -> str:
    return str(text).strip().replace('-', '_').replace(' ', '_').replace('/', '_')


def _v17_collect_combined_outputs(out_dir: Path, base_prefix: str, frames: list, variants: str):
    summary_parts = []
    detail_parts = []
    winner_parts = []
    for frame in frames:
        fp = f'{base_prefix}_{_v17_safe_prefix_part(frame)}'
        label = _v17_frame_label(frame)
        rows = _v17_frame_rows_count(frame)
        for kind, bucket in [('summary', summary_parts), ('detail', detail_parts), ('winner_detail', winner_parts)]:
            p = Path(out_dir) / f'{fp}_{kind}.csv'
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
                df.insert(0, 'Ramprofil', frame)
                df.insert(1, 'Ramtyp', label)
                df.insert(2, 'Teoretisk grundram', rows)
                bucket.append(df)
            except Exception as e:
                print(f'Kunde inte läsa {p}: {e}', flush=True)
    out = {}
    if summary_parts:
        comb = pd.concat(summary_parts, ignore_index=True)
        # Välj total vinnare över alla ramvarianter/varianter: träff först, sedan medianrader.
        try:
            comb['Vinnare alla ramar'] = ''
            sort_cols = ['Träffar', 'Median paketrader', 'Medel paketrader']
            asc = [False, True, True]
            best_idx = comb.sort_values(sort_cols, ascending=asc).index[0]
            comb.loc[best_idx, 'Vinnare alla ramar'] = 'JA'
        except Exception:
            pass
        p = Path(out_dir) / f'{base_prefix}_ALL_FRAMES_summary.csv'
        comb.to_csv(p, index=False)
        out['summary'] = p
    if detail_parts:
        comb = pd.concat(detail_parts, ignore_index=True)
        p = Path(out_dir) / f'{base_prefix}_ALL_FRAMES_detail.csv'
        comb.to_csv(p, index=False)
        out['detail'] = p
    if winner_parts:
        comb = pd.concat(winner_parts, ignore_index=True)
        p = Path(out_dir) / f'{base_prefix}_ALL_FRAMES_winner_detail.csv'
        comb.to_csv(p, index=False)
        out['winner_detail'] = p
    return out


def _v17_print_scale_report(out_dir: Path, base_prefix: str, combined_paths: dict, frames: list, variants: str):
    report_path = Path(out_dir) / f'{base_prefix}_ALL_FRAMES_report.txt'
    lines = []
    lines.append('TIPSET AI – V17 SKALTEST FÖR VANLIGA SPELRAMAR')
    lines.append('')
    lines.append('Syfte: testa V15/V16-SUPER/FAM/SG2/SG3/B5 på Niklas vanligaste större systemramar.')
    lines.append('Ramprofil anges som spik-halv-hel. Exempel 2-7-4 = 2 spik, 7 halva, 4 hela.')
    lines.append('')
    lines.append('Testade ramar:')
    for f in frames:
        lines.append(f'  - {f}: {_v17_frame_label(f)} = {_v17_frame_rows_count(f)} grundrader')
    lines.append('')
    lines.append(f'Varianter: {variants}')
    lines.append('')
    if combined_paths.get('summary') and Path(combined_paths['summary']).exists():
        try:
            sdf = pd.read_csv(combined_paths['summary'])
            cols = [c for c in ['Ramprofil','Ramtyp','Teoretisk grundram','Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Median reducering %','Median filter','Max filter','Vinnare','Vinnare alla ramar'] if c in sdf.columns]
            lines.append('SAMMANFATTNING')
            lines.append(sdf[cols].to_string(index=False))
            lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa sammanfattning: {e}')
    lines.append('Resultatfiler:')
    for k, p in combined_paths.items():
        lines.append(f'  - {k}: {p}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V17 SKALTEST', flush=True)
    if combined_paths.get('summary'):
        try:
            sdf = pd.read_csv(combined_paths['summary'])
            show_cols = [c for c in ['Ramprofil','Teoretisk grundram','Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Median reducering %','Median filter','Max filter','Vinnare','Vinnare alla ramar'] if c in sdf.columns]
            print(sdf[show_cols].to_string(index=False), flush=True)
        except Exception:
            pass
    print(f'Rapport: {report_path}', flush=True)
    for k, p in combined_paths.items():
        print(f'{k}: {p}', flush=True)


def main_v17():
    parser = argparse.ArgumentParser(description='V17 skaltest: V15/V16 gruppmotor på Niklas vanliga systemramar.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_group_optimizer_v17_scale_realframes_test30')
    parser.add_argument('--frames', default='2-7-4,3-4-6,3-5-5', help='Kommalista i format spik-halv-hel. Ex: 2-7-4,3-4-6,3-5-5')
    parser.add_argument('--variants', default='SUPER,FAM,SG3,SG2,B5')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--min-hit', type=int, default=29)
    parser.add_argument('--min-candidate-val-pct', type=float, default=90.0)
    parser.add_argument('--min-package-val-pct', type=float, default=90.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=95.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=95.0)
    parser.add_argument('--min-gap-score', type=float, default=1.25)
    parser.add_argument('--min-unique-rows', type=int, default=1)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='2-7-4', help=argparse.SUPPRESS)
    parser.add_argument('--beam-width', type=int, default=24)
    parser.add_argument('--archive-width', type=int, default=120)
    parser.add_argument('--structure-seed-count', type=int, default=4)
    parser.add_argument('--variants-per-key', type=int, default=3)
    parser.add_argument('--max-candidates', type=int, default=100)
    parser.add_argument('--hit-power', type=float, default=3.0)
    parser.add_argument('--validation-power', type=float, default=1.5)
    parser.add_argument('--reduction-power', type=float, default=1.0)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.25)
    parser.add_argument('--payout-direction-weight', type=float, default=0.08)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=50)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=350)
    parser.add_argument('--per-bucket-keep', type=int, default=1)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--v15-group-max-filters', type=int, default=10)
    parser.add_argument('--v15-max-group-candidates', type=int, default=30)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=1)
    parser.add_argument('--v15-cross-max-filters', type=int, default=12)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    parser.add_argument('--force-upload', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-fresh-workers', action='store_true')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    frames = _v17_parse_frames(args.frames)
    _v15_parse_variants(args.variants)

    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)
    print('\n'+'='*76, flush=True)
    print('TIPSET AI V17 – SKALTEST · V15/V16-SUPER PÅ VERKLIGA RAMAR', flush=True)
    print(f'Test: {args.max_tests} omgångar inom {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '), flush=True)
    print(f'Ramar: {", ".join(frames)}', flush=True)
    print(f'Testomgångar: {args.max_tests} (hårdlåst)', flush=True)
    print(f'Test-offset: {args.test_offset} (hårdlåst)', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('='*76, flush=True)

    # Om detta är en worker kör vi exakt en ram/testoffset.
    if args.internal_worker:
        ns = _load_app_functions(app_file, fast_no_supermakro=False)
        db = ns['load_database'](str(db_file), 13)
        detail, meta = _run_backtest_v15(ns, db, args)
        _write_v15_outputs(detail, args, out_dir, app_file, db_file, worker_count=1)
        return

    # Huvudläge: kör varje ram separat. För varje ram återanvänds V15:s workerlogik.
    for frame in frames:
        frame_args = argparse.Namespace(**vars(args))
        frame_args.frames = frame
        frame_args.frame_profile = frame
        frame_args.output_prefix = f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n' + '-'*76, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*76, flush=True)
        if int(frame_args.max_tests) > 1 and not bool(frame_args.no_fresh_workers):
            _run_fresh_workers_v15(frame_args, app_file, db_file, out_dir)
        else:
            ns = _load_app_functions(app_file, fast_no_supermakro=False)
            db = ns['load_database'](str(db_file), 13)
            detail, meta = _run_backtest_v15(ns, db, frame_args)
            _write_v15_outputs(detail, frame_args, out_dir, app_file, db_file, worker_count=0)

    combined = _v17_collect_combined_outputs(out_dir, args.output_prefix, frames, args.variants)
    _v17_print_scale_report(out_dir, args.output_prefix, combined, frames, args.variants)



# =============================================================================
# V18 – SAMVERKANSMOTOR: paketträffgolv 29→20 med starka individuella filter
# =============================================================================

V18_VARIANT_DEFS = {}
for _floor in range(29, 19, -1):
    _key = f'P{_floor}'
    V18_VARIANT_DEFS[_key] = {
        'label': f'P{_floor} SUPER · paketgolv {_floor}/30',
        'description': (
            f'V15/V16-SUPER som bas. Enskilda profilkandidater måste fortfarande vara minst 29/30, '
            f'men hela paketets gemensamma historikträff får gå ned till {_floor}/30 om reduceringen vinner.'
        ),
        'mode': 'super_groups',
        'structure_miss': [2, 3],
        'family_miss': [0, 1, 2, 3, 4, 5],
        'cross_family': True,
        'drop_individual_structure': True,
        'overrides': {
            'min_hit': int(_floor),
            'candidate_min_hit': 29,
            # Behåll B5/SUPER radpress men låt lägre paketgolv arbeta lite mer reducerande.
            'hit_power': 2.0 if _floor >= 27 else 1.6 if _floor >= 24 else 1.25,
            'validation_power': 0.9 if _floor >= 27 else 0.75 if _floor >= 24 else 0.6,
            'reduction_power': 1.75 if _floor >= 27 else 1.95 if _floor >= 24 else 2.15,
            'min_candidate_val_pct': 85.0,
            # Bred kontroll är ranknings-/säkerhetsgolv, inte absolut historikperfektion.
            'min_package_val_pct': 85.0 if _floor >= 27 else 82.0 if _floor >= 24 else 78.0,
            'min_gap_score': 0.75,
            'payout_direction_weight': 0.05,
            'beam_width': 90,
            'archive_width': 750,
        },
    }

# Registrera V18-varianter i V15/V13-maskineriet så vi kan återanvända den beprövade
# grupp-/superfiltermotorn utan att duplicera hela sökmotorn.
for _k, _cfg in V18_VARIANT_DEFS.items():
    V15_VARIANT_DEFS[_k] = {
        'label': _cfg['label'],
        'description': _cfg['description'],
        'mode': _cfg['mode'],
        'structure_miss': _cfg['structure_miss'],
        'family_miss': _cfg['family_miss'],
        'cross_family': _cfg['cross_family'],
        'drop_individual_structure': _cfg['drop_individual_structure'],
        'overrides': _cfg['overrides'],
    }
    V13_VARIANT_DEFS[_k] = {
        'label': _cfg['label'],
        'fixed_payout': True,
        'bundle_search': False,
        'description': _cfg['description'],
    }


def _v18_parse_variants(value: str) -> List[str]:
    raw = str(value or ','.join(V18_VARIANT_DEFS.keys())).replace(';', ',').replace('+', ',').split(',')
    out = []
    for x in raw:
        k = x.strip().upper()
        if not k:
            continue
        if k not in V18_VARIANT_DEFS:
            raise ValueError(f'Okänd V18-variant: {k}. Tillåtna: {", ".join(V18_VARIANT_DEFS)}')
        if k not in out:
            out.append(k)
    return out or list(V18_VARIANT_DEFS.keys())


# V15-workerlogiken återanvänds. Den här resolvern gör att den hittar V18-filen
# när Colab kör varje testfall i en ny process.
def _resolve_v15_script_path() -> Optional[Path]:  # noqa: F811 - avsiktlig override av V15-resolvern
    candidates = []
    raw_file = globals().get('__file__')
    if raw_file:
        candidates.append(Path(str(raw_file)))
    try:
        argv0 = Path(str(sys.argv[0]))
        if argv0.suffix.lower() == '.py':
            candidates.append(argv0)
    except Exception:
        pass
    preferred_names = [
        'package_relationship_optimizer_v18_light_colab_upload.py',
        'package_group_optimizer_v18_package_floor_colab_upload.py',
    ]
    for directory in [Path.cwd(), Path('/content'), Path('/mnt/data')]:
        if not directory.exists():
            continue
        for name in preferred_names:
            candidates.append(directory / name)
        candidates.extend(sorted(directory.glob('package*relationship*v18*.py')))
        candidates.extend(sorted(directory.glob('package*group*v18*.py')))
        candidates.extend(sorted(directory.glob('package*optimizer*v18*.py')))
    valid, seen = [], set()
    for candidate in candidates:
        try:
            candidate = Path(candidate)
            if not candidate.exists() or candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if str(resolved) in seen:
                continue
            seen.add(str(resolved))
            sample = resolved.read_text(encoding='utf-8', errors='replace')
            if ('def main_v18' not in sample or 'V18_VARIANT_DEFS' not in sample or 'def _run_fresh_workers_v15' not in sample):
                continue
            valid.append((resolved.stat().st_mtime, resolved))
        except Exception:
            pass
    if not valid:
        return None
    valid.sort(reverse=True)
    return valid[0][1]


def _v18_collect_combined_outputs(out_dir: Path, base_prefix: str, frames: list):
    summary_parts, detail_parts, winner_parts = [], [], []
    for frame in frames:
        fp = f'{base_prefix}_{_v17_safe_prefix_part(frame)}'
        label = _v17_frame_label(frame)
        rows = _v17_frame_rows_count(frame)
        for kind, bucket in [('summary', summary_parts), ('detail', detail_parts), ('winner_detail', winner_parts)]:
            p = Path(out_dir) / f'{fp}_{kind}.csv'
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
                df.insert(0, 'Ramprofil', frame)
                df.insert(1, 'Ramtyp', label)
                df.insert(2, 'Teoretisk grundram', rows)
                bucket.append(df)
            except Exception as e:
                print(f'Kunde inte läsa {p}: {e}', flush=True)
    out = {}
    if summary_parts:
        comb = pd.concat(summary_parts, ignore_index=True)
        # Robust vinnare: träff först, därefter medianrader, därefter medelrader.
        comb['Vinnare alla ramar'] = ''
        try:
            best_idx = comb.sort_values(
                ['Träffar', 'Median paketrader', 'Medel paketrader'],
                ascending=[False, True, True]
            ).index[0]
            comb.loc[best_idx, 'Vinnare alla ramar'] = 'JA'
        except Exception:
            pass
        p = Path(out_dir) / f'{base_prefix}_ALL_FRAMES_summary.csv'
        comb.to_csv(p, index=False)
        out['summary'] = p
    if detail_parts:
        comb = pd.concat(detail_parts, ignore_index=True)
        p = Path(out_dir) / f'{base_prefix}_ALL_FRAMES_detail.csv'
        comb.to_csv(p, index=False)
        out['detail'] = p
    if winner_parts:
        comb = pd.concat(winner_parts, ignore_index=True)
        p = Path(out_dir) / f'{base_prefix}_ALL_FRAMES_winner_detail.csv'
        comb.to_csv(p, index=False)
        out['winner_detail'] = p
    return out


def _v18_print_report(out_dir: Path, base_prefix: str, combined_paths: dict, frames: list, variants: list):
    report_path = Path(out_dir) / f'{base_prefix}_ALL_FRAMES_report.txt'
    lines = []
    lines.append('TIPSET AI – V18 SAMVERKANSMOTOR / PAKETTRÄFFGOLV')
    lines.append('')
    lines.append('Syfte: testa om starka individuella filter kan kombineras hårdare genom att tillåta lägre gemensam paketträff på de 30 liknande omgångarna.')
    lines.append('Viktigt: kandidatfilter är fortfarande höga, normalt minst 29/30. Det som varieras är HELA paketets gemensamma historikgolv P29–P20.')
    lines.append('')
    lines.append('Testade ramar:')
    for f in frames:
        lines.append(f'  - {f}: {_v17_frame_label(f)} = {_v17_frame_rows_count(f)} grundrader')
    lines.append('')
    lines.append('Testade paketgolv: ' + ', '.join(variants))
    lines.append('')
    if combined_paths.get('summary') and Path(combined_paths['summary']).exists():
        try:
            sdf = pd.read_csv(combined_paths['summary'])
            cols = [c for c in ['Ramprofil','Teoretisk grundram','Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Median reducering %','Median filter','Max filter','Median gruppfilter','Median individuella filter','Median tvärfamiljgrupper','Vinnare','Vinnare alla ramar'] if c in sdf.columns]
            lines.append('SAMMANFATTNING')
            lines.append('-'*76)
            lines.append(sdf[cols].to_string(index=False))
            lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa sammanfattning: {e}')
    lines.append('Tolkning: om P24/P23/P22 ger nästan samma testträff som P29 men mycket färre rader, har vi bevis för att gemensamt 29/30 var för konservativt.')
    lines.append('')
    lines.append('Resultatfiler:')
    for k, p in combined_paths.items():
        lines.append(f'  - {k}: {p}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V18 PAKETGOLV/SAMVERKAN', flush=True)
    if combined_paths.get('summary'):
        try:
            sdf = pd.read_csv(combined_paths['summary'])
            show_cols = [c for c in ['Ramprofil','Teoretisk grundram','Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Median reducering %','Median filter','Max filter','Vinnare','Vinnare alla ramar'] if c in sdf.columns]
            print(sdf[show_cols].to_string(index=False), flush=True)
        except Exception:
            pass
    print(f'Rapport: {report_path}', flush=True)
    for k, p in combined_paths.items():
        print(f'{k}: {p}', flush=True)


def main_v18():
    parser = argparse.ArgumentParser(description='V18 samverkansmotor: starka kandidatfilter men varierat gemensamt paketträffgolv P29–P20.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_relationship_optimizer_v18_light_test3')
    parser.add_argument('--frames', default='3-5-5', help='Kommalista i format spik-halv-hel. Standard: 3-5-5 för snabb första kurva. Ex alla: 2-7-4,3-4-6,3-5-5')
    parser.add_argument('--variants', default='P29,P27,P24')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--min-hit', type=int, default=29, help='Basvärde. Varianterna Pxx skriver över detta per paketgolv.')
    parser.add_argument('--min-candidate-val-pct', type=float, default=85.0)
    parser.add_argument('--min-package-val-pct', type=float, default=85.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=95.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=95.0)
    parser.add_argument('--min-gap-score', type=float, default=0.75)
    parser.add_argument('--min-unique-rows', type=int, default=1)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='3-5-5', help=argparse.SUPPRESS)
    parser.add_argument('--beam-width', type=int, default=24)
    parser.add_argument('--archive-width', type=int, default=140)
    parser.add_argument('--structure-seed-count', type=int, default=4)
    parser.add_argument('--variants-per-key', type=int, default=3)
    parser.add_argument('--max-candidates', type=int, default=110)
    parser.add_argument('--hit-power', type=float, default=2.0)
    parser.add_argument('--validation-power', type=float, default=0.9)
    parser.add_argument('--reduction-power', type=float, default=1.75)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.25)
    parser.add_argument('--payout-direction-weight', type=float, default=0.05)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=50)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=350)
    parser.add_argument('--per-bucket-keep', type=int, default=1)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--v15-group-max-filters', type=int, default=10)
    parser.add_argument('--v15-max-group-candidates', type=int, default=35)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=1)
    parser.add_argument('--v15-cross-max-filters', type=int, default=12)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    parser.add_argument('--force-upload', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-fresh-workers', action='store_true')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)

    frames = _v17_parse_frames(args.frames)
    variants = _v18_parse_variants(args.variants)
    # V15-motorn läser args.variants. Använd explicit normaliserad lista.
    args.variants = ','.join(variants)

    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)
    print('\n'+'='*76, flush=True)
    print('TIPSET AI V18 – SAMVERKANSMOTOR · PAKETTRÄFFGOLV 29→20', flush=True)
    print(f'Test: {args.max_tests} omgångar inom {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '), flush=True)
    print(f'Ramar: {", ".join(frames)}', flush=True)
    print(f'Testomgångar: {args.max_tests} (hårdlåst)', flush=True)
    print(f'Test-offset: {args.test_offset} (hårdlåst)', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('Enskilda profilkandidater: minst 29/30. Varierar bara paketets gemensamma golv.', flush=True)
    print('='*76, flush=True)

    if args.internal_worker:
        ns = _load_app_functions(app_file, fast_no_supermakro=False)
        db = ns['load_database'](str(db_file), 13)
        detail, meta = _run_backtest_v15(ns, db, args)
        _write_v15_outputs(detail, args, out_dir, app_file, db_file, worker_count=1)
        return

    for frame in frames:
        frame_args = argparse.Namespace(**vars(args))
        frame_args.frames = frame
        frame_args.frame_profile = frame
        frame_args.output_prefix = f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n' + '-'*76, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*76, flush=True)
        if int(frame_args.max_tests) > 1 and not bool(frame_args.no_fresh_workers):
            _run_fresh_workers_v15(frame_args, app_file, db_file, out_dir)
        else:
            ns = _load_app_functions(app_file, fast_no_supermakro=False)
            db = ns['load_database'](str(db_file), 13)
            detail, meta = _run_backtest_v15(ns, db, frame_args)
            _write_v15_outputs(detail, frame_args, out_dir, app_file, db_file, worker_count=0)

    combined = _v18_collect_combined_outputs(out_dir, args.output_prefix, frames)
    _v18_print_report(out_dir, args.output_prefix, combined, frames, variants)




# =============================================================================
# V19 – LIGHT FAMILY FOCUS / FILTER-AUDIT
# =============================================================================
# Syfte: snabb iteration där motorn tvingas ge FAT, värde/svårighet och
# favorit/skräll en ärlig chans. Detta är inte slutlig 30-testmotor utan en
# snabb felsöknings-/idétestare för 1–3 omgångar.

V19_VARIANT_DEFS = {
    'BASE': {
        'label': 'BASE SUPER kontroll',
        'description': 'V15/V16-SUPER som light-kontroll. Visar vad nuvarande motor väljer utan familjstyrning.',
        'mode': 'super_groups',
        'structure_miss': [2, 3],
        'family_miss': [0, 1, 2, 3, 4],
        'cross_family': True,
        'drop_individual_structure': True,
        'overrides': {
            'min_hit': 27,
            'candidate_min_hit': 29,
            'beam_width': 24,
            'archive_width': 160,
            'min_package_val_pct': 84.0,
            'min_candidate_val_pct': 84.0,
            'reduction_power': 1.85,
            'payout_direction_weight': 0.07,
        },
    },
    'FVFS': {
        'label': 'FVFS FAT + värde + favorit/skräll',
        'description': 'Fokuserar kandidatpoolen på FAT/ABC, värde/svårighet, poäng/rank och favorit/skräll. Struktur får endast vara gruppskydd.',
        'mode': 'super_groups',
        'structure_miss': [2, 3],
        'family_miss': [0, 1, 2, 3, 4, 5],
        'cross_family': True,
        'drop_individual_structure': True,
        'v19_allowed_families': {'FAT', 'ABC', 'FAT-sekvens', 'Värde/svårighet', 'Poäng/rank', 'Favorit', 'Skräll', 'Tvärfamilj', 'Struktur'},
        'v19_family_bonus': {'FAT': 0.60, 'ABC': 0.55, 'FAT-sekvens': 0.45, 'Värde/svårighet': 0.70, 'Poäng/rank': 0.45, 'Favorit': 0.45, 'Skräll': 0.50, 'Tvärfamilj': 0.25},
        'v19_target_rows': 2400,
        'v19_min_active_families': 3,
        'overrides': {
            'min_hit': 27,
            'candidate_min_hit': 28,
            'beam_width': 28,
            'archive_width': 180,
            'min_unique_rows': 25,
            'min_package_val_pct': 82.0,
            'min_candidate_val_pct': 80.0,
            'reduction_power': 2.20,
            'validation_power': 0.75,
            'payout_direction_weight': 0.14,
            'v15_group_max_filters': 12,
            'v15_max_group_candidates': 44,
            'v15_cross_per_family': 2,
        },
    },
    'RADPRESS': {
        'label': 'RADPRESS offensiv familjemotor',
        'description': 'Som FVFS men med hårdare radpress och krav på större unik radnytta per nytt filter.',
        'mode': 'super_groups',
        'structure_miss': [2, 3],
        'family_miss': [0, 1, 2, 3, 4, 5],
        'cross_family': True,
        'drop_individual_structure': True,
        'v19_allowed_families': {'FAT', 'ABC', 'FAT-sekvens', 'Värde/svårighet', 'Poäng/rank', 'Favorit', 'Skräll', 'Tvärfamilj', 'Struktur'},
        'v19_family_bonus': {'FAT': 0.75, 'ABC': 0.70, 'FAT-sekvens': 0.55, 'Värde/svårighet': 0.85, 'Poäng/rank': 0.50, 'Favorit': 0.55, 'Skräll': 0.60, 'Tvärfamilj': 0.35},
        'v19_target_rows': 2300,
        'v19_min_active_families': 3,
        'overrides': {
            'min_hit': 26,
            'candidate_min_hit': 28,
            'beam_width': 30,
            'archive_width': 200,
            'min_unique_rows': 60,
            'min_package_val_pct': 80.0,
            'min_candidate_val_pct': 78.0,
            'reduction_power': 2.65,
            'validation_power': 0.62,
            'payout_direction_weight': 0.16,
            'v15_group_max_filters': 14,
            'v15_max_group_candidates': 50,
            'v15_cross_per_family': 2,
        },
    },
    'NOSTRUCT': {
        'label': 'NOSTRUCT profil utan strukturkniv',
        'description': 'Tar bort strukturfilter även som grupp för att se om FAT/värde/favorit-skräll ensamma kan reducera bättre.',
        'mode': 'super_groups',
        'structure_miss': [],
        'family_miss': [0, 1, 2, 3, 4, 5],
        'cross_family': True,
        'drop_individual_structure': True,
        'v19_drop_structure_all': True,
        'v19_allowed_families': {'FAT', 'ABC', 'FAT-sekvens', 'Värde/svårighet', 'Poäng/rank', 'Favorit', 'Skräll', 'Tvärfamilj'},
        'v19_family_bonus': {'FAT': 0.85, 'ABC': 0.75, 'FAT-sekvens': 0.60, 'Värde/svårighet': 0.90, 'Poäng/rank': 0.55, 'Favorit': 0.60, 'Skräll': 0.65, 'Tvärfamilj': 0.35},
        'v19_target_rows': 2400,
        'v19_min_active_families': 3,
        'overrides': {
            'min_hit': 26,
            'candidate_min_hit': 28,
            'beam_width': 30,
            'archive_width': 200,
            'min_unique_rows': 40,
            'min_package_val_pct': 80.0,
            'min_candidate_val_pct': 78.0,
            'reduction_power': 2.35,
            'validation_power': 0.65,
            'payout_direction_weight': 0.18,
            'v15_group_max_filters': 14,
            'v15_max_group_candidates': 46,
            'v15_cross_per_family': 2,
        },
    },
    'FATCORE': {
        'label': 'FATCORE FAT/ABC-kärna',
        'description': 'Fokuserar hårdare på FAT/ABC/FAT-sekvens plus värde som sekundär signal.',
        'mode': 'super_groups',
        'structure_miss': [3],
        'family_miss': [0, 1, 2, 3, 4],
        'cross_family': True,
        'drop_individual_structure': True,
        'v19_allowed_families': {'FAT', 'ABC', 'FAT-sekvens', 'Värde/svårighet', 'Poäng/rank', 'Tvärfamilj', 'Struktur'},
        'v19_family_bonus': {'FAT': 1.05, 'ABC': 0.90, 'FAT-sekvens': 0.75, 'Värde/svårighet': 0.45, 'Poäng/rank': 0.25, 'Tvärfamilj': 0.25},
        'v19_target_rows': 2350,
        'v19_min_active_families': 2,
        'overrides': {
            'min_hit': 27,
            'candidate_min_hit': 28,
            'beam_width': 26,
            'archive_width': 180,
            'min_unique_rows': 45,
            'min_package_val_pct': 81.0,
            'min_candidate_val_pct': 78.0,
            'reduction_power': 2.45,
            'validation_power': 0.68,
            'payout_direction_weight': 0.10,
            'v15_group_max_filters': 14,
            'v15_max_group_candidates': 46,
            'v15_cross_per_family': 2,
        },
    },
}

# Registrera V19-varianter i V15/V13-maskineriet.
for _k, _cfg in V19_VARIANT_DEFS.items():
    V15_VARIANT_DEFS[_k] = {
        'label': _cfg['label'],
        'description': _cfg['description'],
        'mode': _cfg['mode'],
        'structure_miss': _cfg.get('structure_miss', []),
        'family_miss': _cfg.get('family_miss', [0, 1, 2, 3]),
        'cross_family': _cfg.get('cross_family', False),
        'drop_individual_structure': _cfg.get('drop_individual_structure', True),
        'drop_individual_profile': _cfg.get('drop_individual_profile', False),
        'v19_allowed_families': _cfg.get('v19_allowed_families'),
        'v19_drop_structure_all': _cfg.get('v19_drop_structure_all', False),
        'v19_family_bonus': _cfg.get('v19_family_bonus', {}),
        'v19_target_rows': _cfg.get('v19_target_rows'),
        'v19_min_active_families': _cfg.get('v19_min_active_families', 0),
        'overrides': _cfg.get('overrides', {}),
    }
    V13_VARIANT_DEFS[_k] = {
        'label': _cfg['label'],
        'fixed_payout': True,
        'bundle_search': False,
        'description': _cfg['description'],
    }


def _v19_parse_variants(value: str) -> List[str]:
    raw = str(value or ','.join(V19_VARIANT_DEFS.keys())).replace(';', ',').replace('+', ',').split(',')
    out = []
    for x in raw:
        k = x.strip().upper()
        if not k:
            continue
        if k not in V19_VARIANT_DEFS:
            raise ValueError(f'Okänd V19-variant: {k}. Tillåtna: {", ".join(V19_VARIANT_DEFS)}')
        if k not in out:
            out.append(k)
    return out or list(V19_VARIANT_DEFS.keys())


def _v19_family_of_candidate(c: dict) -> str:
    if bool(c.get('is_v15_group')):
        return str(c.get('group_family') or _v15_group_family(c))
    return _v15_group_family(c)


_ORIG_V15_TRANSFORM_CANDIDATES = _v15_transform_candidates

def _v15_transform_candidates(candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, args, variant_id: str) -> Tuple[List[dict], List[dict]]:  # noqa: F811
    """V19-override: återanvänd V15-grupper men kan fokusera på offensiva filterfamiljer."""
    final, group_cands = _ORIG_V15_TRANSFORM_CANDIDATES(candidates, htot, vtot, ftot, hist_payout, args, variant_id)
    cfg = V15_VARIANT_DEFS.get(str(variant_id).upper(), {})
    if str(variant_id).upper() not in V19_VARIANT_DEFS:
        return final, group_cands

    allowed = cfg.get('v19_allowed_families')
    drop_structure_all = bool(cfg.get('v19_drop_structure_all', False))

    def keep(c: dict) -> bool:
        fam = _v19_family_of_candidate(c)
        if drop_structure_all and fam == 'Struktur':
            return False
        if allowed is not None and fam not in allowed:
            return False
        return True

    final2 = [c for c in final if keep(c)]
    group2 = [g for g in group_cands if keep(g)]

    # Om filtreringen blev för hård faller vi tillbaka till ursprunglig final så att testet inte dör.
    # Men behåll struktur-borttagningen i NOSTRUCT.
    if not final2:
        final2 = [c for c in final if not (drop_structure_all and _v19_family_of_candidate(c) == 'Struktur')]
    return final2, group2


_ORIG_V13_STATE_METRICS = _v13_state_metrics

def _v13_state_metrics(st: _V9State, payout_values: np.ndarray, args, variant_id: str) -> dict:  # noqa: F811
    """V19-override: familjebonus + radmål så FAT/värde/favorit-skräll inte drunknar i trygghetsscore."""
    m = _ORIG_V13_STATE_METRICS(st, payout_values, args, variant_id)
    cfg = V15_VARIANT_DEFS.get(str(variant_id).upper(), {})
    if str(variant_id).upper() not in V19_VARIANT_DEFS:
        return m

    fam_bonus = cfg.get('v19_family_bonus') or {}
    fams = [_v19_family_of_candidate(c) for c in (st.chosen or tuple())]
    fam_set = set(fams)
    bonus = 0.0
    for fam in fam_set:
        bonus += float(fam_bonus.get(fam, 0.0))
    min_active = int(cfg.get('v19_min_active_families') or 0)
    non_structure_fams = {f for f in fam_set if f not in {'Struktur', 'Tvärfamilj'}}
    if min_active > 0:
        # Mjuk bonus/straff: tidiga states kan fortfarande växa, men slutpaket med fler relevanta
        # familjer får tydlig fördel.
        bonus += 0.35 * min(len(non_structure_fams), min_active)
        if len(non_structure_fams) < min_active and len(st.chosen) >= min_active:
            bonus -= 0.75 * (min_active - len(non_structure_fams))

    target_rows = cfg.get('v19_target_rows')
    if target_rows:
        target = max(1, int(target_rows))
        # Belöna när vi närmar oss radmålet, men gör inte för små system automatiskt bäst.
        if st.frame_count > target:
            pressure = min(2.0, (st.frame_count - target) / max(1.0, st.frame_size))
            bonus -= 0.80 * pressure
        else:
            bonus += 0.20

    m['v19_family_bonus'] = float(bonus)
    m['joint_score'] = float(m.get('joint_score', 0.0)) + float(bonus)
    return m


def _resolve_v15_script_path() -> Optional[Path]:  # noqa: F811
    candidates = []
    raw_file = globals().get('__file__')
    if raw_file:
        candidates.append(Path(str(raw_file)))
    try:
        argv0 = Path(str(sys.argv[0]))
        if argv0.suffix.lower() == '.py':
            candidates.append(argv0)
    except Exception:
        pass
    preferred_names = [
        'package_family_focus_v19_light_colab_upload.py',
        'package_relationship_optimizer_v18_light_colab_upload.py',
    ]
    for directory in [Path.cwd(), Path('/content'), Path('/mnt/data')]:
        if not directory.exists():
            continue
        for name in preferred_names:
            candidates.append(directory / name)
        candidates.extend(sorted(directory.glob('package*family*focus*v19*.py')))
        candidates.extend(sorted(directory.glob('package*relationship*v18*.py')))
    valid, seen = [], set()
    for candidate in candidates:
        try:
            candidate = Path(candidate)
            if not candidate.exists() or candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if str(resolved) in seen:
                continue
            seen.add(str(resolved))
            sample = resolved.read_text(encoding='utf-8', errors='replace')
            if ('def main_v19' not in sample or 'V19_VARIANT_DEFS' not in sample or 'def _run_fresh_workers_v15' not in sample):
                continue
            valid.append((resolved.stat().st_mtime, resolved))
        except Exception:
            pass
    if not valid:
        return None
    valid.sort(reverse=True)
    return valid[0][1]


def _v19_collect_combined_outputs(out_dir: Path, base_prefix: str, frames: list):
    return _v18_collect_combined_outputs(out_dir, base_prefix, frames)


def _v19_print_report(out_dir: Path, base_prefix: str, combined_paths: dict, frames: list, variants: list):
    report_path = Path(out_dir) / f'{base_prefix}_ALL_FRAMES_report.txt'
    lines = []
    lines.append('TIPSET AI – V19 LIGHT FAMILY FOCUS')
    lines.append('')
    lines.append('Syfte: snabbtesta om motorn börjar använda rätt filterfamiljer: FAT/ABC, värde/svårighet, poäng/rank och favorit/skräll.')
    lines.append('Detta är en LIGHT-runner. Resultatet visar riktning och filterval, inte slutlig 30-omgångsbevisning.')
    lines.append('')
    lines.append('Testade ramar:')
    for f in frames:
        lines.append(f'  - {f}: {_v17_frame_label(f)} = {_v17_frame_rows_count(f)} grundrader')
    lines.append('')
    lines.append('Testade varianter: ' + ', '.join(variants))
    lines.append('')
    if combined_paths.get('summary') and Path(combined_paths['summary']).exists():
        try:
            sdf = pd.read_csv(combined_paths['summary'])
            cols = [c for c in ['Ramprofil','Teoretisk grundram','Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Median reducering %','Median filter','Max filter','Median gruppfilter','Median individuella filter','Median tvärfamiljgrupper','Vinnare','Vinnare alla ramar'] if c in sdf.columns]
            lines.append('SAMMANFATTNING')
            lines.append('-'*76)
            lines.append(sdf[cols].to_string(index=False))
            lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa sammanfattning: {e}')
    if combined_paths.get('winner_detail') and Path(combined_paths['winner_detail']).exists():
        try:
            wdf = pd.read_csv(combined_paths['winner_detail'])
            if 'Valda filter JSON' in wdf.columns:
                lines.append('VINNARFILTER – första testfallen')
                lines.append('-'*76)
                for _, r in wdf.head(3).iterrows():
                    lines.append(f"{str(r.get('Datum',''))[:10]} · {r.get('Variant','')} · träff={r.get('Facit överlever','') or r.get('Hit','')}")
                    raw = r.get('Valda filter JSON', '')
                    try:
                        items = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
                    except Exception:
                        items = []
                    for it in items[:12]:
                        fam = _v15_group_family({'name': it.get('name',''), 'key': it.get('key',''), 'category': it.get('category','')})
                        lines.append(f"  - {it.get('category','')} / {fam}: {it.get('name','')} {it.get('interval','')} · red_alone={it.get('frame_red_pct_alone','')} · step={it.get('step_removed_rows','')}")
                    lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa vinnarfilter: {e}')
    lines.append('Tolkning: BASE ska jämföras mot FVFS/RADPRESS/NOSTRUCT/FATCORE. Om familjevarianterna inte minskar rader tydligt eller tappar facit direkt är filterfamiljscoren fortfarande fel.')
    lines.append('')
    lines.append('Resultatfiler:')
    for k, p in combined_paths.items():
        lines.append(f'  - {k}: {p}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V19 LIGHT FAMILY FOCUS', flush=True)
    if combined_paths.get('summary'):
        try:
            sdf = pd.read_csv(combined_paths['summary'])
            show_cols = [c for c in ['Ramprofil','Teoretisk grundram','Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Median reducering %','Median filter','Max filter','Vinnare','Vinnare alla ramar'] if c in sdf.columns]
            print(sdf[show_cols].to_string(index=False), flush=True)
        except Exception:
            pass
    print(f'Rapport: {report_path}', flush=True)
    for k, p in combined_paths.items():
        print(f'{k}: {p}', flush=True)


def main_v19():
    parser = argparse.ArgumentParser(description='V21c filterguard: RADPRESS-patch med Delta-/favoritguard och auditoutput.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_filter_guard_v21c_test10')
    parser.add_argument('--frames', default='3-5-5', help='Kommalista i format spik-halv-hel. Standard: 3-5-5. Ex alla: 2-7-4,3-4-6,3-5-5')
    parser.add_argument('--variants', default='BASE,RADPRESS,V21A,V21B,V21C,V21D')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--min-hit', type=int, default=27)
    parser.add_argument('--min-candidate-val-pct', type=float, default=82.0)
    parser.add_argument('--min-package-val-pct', type=float, default=82.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=95.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=95.0)
    parser.add_argument('--min-gap-score', type=float, default=0.65)
    parser.add_argument('--min-unique-rows', type=int, default=25)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='3-5-5', help=argparse.SUPPRESS)
    parser.add_argument('--beam-width', type=int, default=26)
    parser.add_argument('--archive-width', type=int, default=180)
    parser.add_argument('--structure-seed-count', type=int, default=2)
    parser.add_argument('--variants-per-key', type=int, default=3)
    parser.add_argument('--max-candidates', type=int, default=130)
    parser.add_argument('--hit-power', type=float, default=1.65)
    parser.add_argument('--validation-power', type=float, default=0.70)
    parser.add_argument('--reduction-power', type=float, default=2.20)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.20)
    parser.add_argument('--payout-direction-weight', type=float, default=0.12)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=50)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=300)
    parser.add_argument('--per-bucket-keep', type=int, default=2)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--v15-group-max-filters', type=int, default=14)
    parser.add_argument('--v15-max-group-candidates', type=int, default=50)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=2)
    parser.add_argument('--v15-cross-max-filters', type=int, default=14)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    parser.add_argument('--force-upload', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-fresh-workers', action='store_true')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)

    frames = _v17_parse_frames(args.frames)
    variants = _v19_parse_variants(args.variants)
    args.variants = ','.join(variants)

    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)
    print('\n'+'='*76, flush=True)
    print('TIPSET AI V21c – FILTERGUARD TEST10', flush=True)
    print(f'Test: {args.max_tests} omgångar inom {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '), flush=True)
    print(f'Ramar: {", ".join(frames)}', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('Fokus: BASE/RADPRESS mot V21A–D med Delta-/favoritguard och auditoutput.', flush=True)
    print('='*76, flush=True)

    if args.internal_worker:
        ns = _load_app_functions(app_file, fast_no_supermakro=False)
        db = ns['load_database'](str(db_file), 13)
        detail, meta = _run_backtest_v15(ns, db, args)
        _write_v15_outputs(detail, args, out_dir, app_file, db_file, worker_count=1)
        return

    for frame in frames:
        frame_args = argparse.Namespace(**vars(args))
        frame_args.frames = frame
        frame_args.frame_profile = frame
        frame_args.output_prefix = f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n' + '-'*76, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*76, flush=True)
        if int(frame_args.max_tests) > 1 and not bool(frame_args.no_fresh_workers):
            _run_fresh_workers_v15(frame_args, app_file, db_file, out_dir)
        else:
            ns = _load_app_functions(app_file, fast_no_supermakro=False)
            db = ns['load_database'](str(db_file), 13)
            detail, meta = _run_backtest_v15(ns, db, frame_args)
            _write_v15_outputs(detail, frame_args, out_dir, app_file, db_file, worker_count=0)

    combined = _v19_collect_combined_outputs(out_dir, args.output_prefix, frames)
    _v19_print_report(out_dir, args.output_prefix, combined, frames, variants)



# =============================================================================
# V20 – LIGHT FILTER AUDIT
# =============================================================================
# Syfte: kör samma snabba familjevarianter som V19, men bygg extra audit-filer
# som visar filter för filter: unik radnytta, träff/miss, dödande filter och
# familjer som är röda/gula/gröna. Detta är en diagnosrunner, inte slutmotor.


def _v20_json_load_cell(x, default=None):
    if default is None:
        default = []
    if x is None:
        return default
    try:
        if isinstance(x, float) and np.isnan(x):
            return default
    except Exception:
        pass
    if isinstance(x, (list, dict)):
        return x
    s = str(x).strip()
    if not s or s.lower() in {'nan', 'none', 'null'}:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _v20_is_hit_value(x) -> bool:
    s = str(x).strip().lower()
    return s in {'ja', 'true', '1', 'yes', 'y'}


def _v20_num(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.replace('%','').replace(' ', '').replace(',', '.')
        if x == '':
            return default
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def _v20_filter_family(name: str, category: str = '', key: str = '') -> str:
    try:
        return _v15_group_family({'name': name or '', 'category': category or '', 'key': key or ''})
    except Exception:
        txt = f'{name} {category} {key}'.lower()
        if 'struktur' in txt or 'följd' in txt or 'uppkomst' in txt or 'luck' in txt or 'dubbl' in txt or 'tripp' in txt:
            return 'Struktur'
        if 'fat' in txt:
            return 'FAT'
        if 'abc' in txt:
            return 'ABC'
        if 'värde' in txt or 'svår' in txt or 'rank' in txt or 'poäng' in txt or 'delta' in txt or 'avvik' in txt:
            return 'Värde/svårighet'
        if 'favorit' in txt:
            return 'Favorit'
        if 'skräll' in txt or 'skrall' in txt:
            return 'Skräll'
        return str(category or 'Okänd')


def _v20_step_name(step: dict) -> str:
    return str(step.get('Filter') or step.get('filter') or step.get('name') or step.get('namn') or '').strip()


def _v20_step_category(step: dict) -> str:
    return str(step.get('Kategori') or step.get('kategori') or step.get('category') or '').strip()


def _v20_write_audit_outputs(out_dir: Path, base_prefix: str, frames: list) -> dict:
    """Bygger auditfiler från kombinerad detailfil, utan att köra om backtest."""
    out_dir = Path(out_dir)
    detail_path = out_dir / f'{base_prefix}_ALL_FRAMES_detail.csv'
    if not detail_path.exists():
        # Fallback: enskild ram om combined inte hann skapas.
        for frame in frames:
            p = out_dir / f'{base_prefix}_{_v17_safe_prefix_part(frame)}_detail.csv'
            if p.exists():
                detail_path = p
                break
    paths = {}
    if not detail_path.exists():
        return paths

    df = pd.read_csv(detail_path)
    step_rows = []
    killer_rows = []
    package_rows = []

    for ridx, r in df.iterrows():
        variant = str(r.get('Variant', ''))
        variant_name = str(r.get('Variantnamn', ''))
        date = str(r.get('Datum', ''))[:10]
        frame = str(r.get('Ramprofil', '') or getattr(r, 'Ramprofil', '') or '')
        hit = _v20_is_hit_value(r.get('Paket klarar facit', r.get('Facit överlever', r.get('Hit', ''))))
        base_rows = int(_v20_num(r.get('Grundram rader', 0), 0))
        pack_rows = int(_v20_num(r.get('Paketrader', 0), 0))
        package_rows.append({
            'Radindex': ridx,
            'Ramprofil': frame,
            'Variant': variant,
            'Variantnamn': variant_name,
            'Datum': date,
            'Utdelning': r.get('Utdelning', ''),
            'Facit överlever': 'Ja' if hit else 'Nej',
            'Grundram rader': base_rows,
            'Paketrader': pack_rows,
            'Reducerar %': _v20_num(r.get('Reducerar %', 0), 0),
            'Paketträff': r.get('Paketträff', ''),
            'Valideringsträff': r.get('Valideringsträff', ''),
            'Orsak': r.get('Orsak', ''),
        })

        miss = _v20_json_load_cell(r.get('Missdiagnos JSON', ''), default={})
        killer_name = ''
        killer_cat = ''
        if isinstance(miss, dict) and miss:
            killer_name = str(miss.get('filter') or miss.get('Filter') or '')
            killer_cat = str(miss.get('kategori') or miss.get('Kategori') or '')
            killer_rows.append({
                'Ramprofil': frame,
                'Variant': variant,
                'Variantnamn': variant_name,
                'Datum': date,
                'Utdelning': r.get('Utdelning', ''),
                'Dödande filter': killer_name,
                'Kategori': killer_cat,
                'Familj': _v20_filter_family(killer_name, killer_cat),
                'Intervall': miss.get('intervall', ''),
                'Facitvärde': miss.get('facitvärde', ''),
                'Hist filter': miss.get('hist_filter', ''),
                'Step removed rows': miss.get('step_removed_rows', ''),
                'Payout direction %': miss.get('payout_direction_pct', ''),
                'Gap score': miss.get('gap_score', ''),
                'Orsak': r.get('Orsak', ''),
            })

        steps = _v20_json_load_cell(r.get('Steg JSON', ''), default=[])
        if not isinstance(steps, list) or not steps:
            filters = _v20_json_load_cell(r.get('Filter JSON', ''), default=[])
            if isinstance(filters, list):
                # Fallback med ungefärliga kolumnnamn.
                steps = []
                after = base_rows
                for f in filters:
                    steps.append({
                        'Filter': f.get('name',''),
                        'Kategori': f.get('category',''),
                        'Intervall': f.get('interval',''),
                        'Efter filter': after,
                        'Borttagna unika rader': f.get('step_removed_rows', 0),
                        'Stegreducering %': '',
                        'Samlad historikträff': f.get('hist',''),
                        'Samlad validering': f.get('val_pct',''),
                        'Fas': 'unknown',
                    })
        for sidx, step in enumerate(steps if isinstance(steps, list) else []):
            if not isinstance(step, dict):
                continue
            name = _v20_step_name(step)
            cat = _v20_step_category(step)
            interval = str(step.get('Intervall') or step.get('intervall') or '')
            removed = int(round(_v20_num(step.get('Borttagna unika rader', step.get('step_removed_rows', 0)), 0)))
            after_rows = int(round(_v20_num(step.get('Efter filter', step.get('after_rows', 0)), 0)))
            fam = _v20_filter_family(name, cat)
            is_killer = bool((not hit) and killer_name and name == killer_name and (not killer_cat or cat == killer_cat))
            step_rows.append({
                'Radindex': ridx,
                'Ramprofil': frame,
                'Variant': variant,
                'Variantnamn': variant_name,
                'Datum': date,
                'Utdelning': r.get('Utdelning', ''),
                'Facit överlever': 'Ja' if hit else 'Nej',
                'Steg': sidx + 1,
                'Filter': name,
                'Kategori': cat,
                'Familj': fam,
                'Intervall': interval,
                'Fas': step.get('Fas', step.get('phase', '')),
                'Efter filter': after_rows,
                'Borttagna unika rader': removed,
                'Stegreducering %': _v20_num(step.get('Stegreducering %', ''), np.nan),
                'Samlad historikträff': step.get('Samlad historikträff', ''),
                'Samlad validering': step.get('Samlad validering', ''),
                'Spret-gap': _v20_num(step.get('Spret-gap', 0), 0),
                'Utdelningslift %': _v20_num(step.get('Utdelningslift %', 0), 0),
                'Dödande filter': 'Ja' if is_killer else 'Nej',
                'Orsak': r.get('Orsak', ''),
            })

    step_df = pd.DataFrame(step_rows)
    killer_df = pd.DataFrame(killer_rows)
    package_df = pd.DataFrame(package_rows)

    step_path = out_dir / f'{base_prefix}_FILTER_STEP_AUDIT.csv'
    killer_path = out_dir / f'{base_prefix}_KILLER_FILTERS.csv'
    agg_path = out_dir / f'{base_prefix}_FILTER_AGG_AUDIT.csv'
    family_path = out_dir / f'{base_prefix}_FAMILY_AUDIT.csv'
    package_path = out_dir / f'{base_prefix}_PACKAGE_AUDIT.csv'

    if not step_df.empty:
        step_df.to_csv(step_path, index=False)
        paths['step_audit'] = str(step_path)
        group_cols = ['Ramprofil','Variant','Filter','Kategori','Familj']
        agg = step_df.groupby(group_cols, dropna=False).agg(
            Antal_val=('Filter','size'),
            Valda_i_träff=('Facit överlever', lambda x: int((x == 'Ja').sum())),
            Valda_i_miss=('Facit överlever', lambda x: int((x != 'Ja').sum())),
            Dödade_facit=('Dödande filter', lambda x: int((x == 'Ja').sum())),
            Median_unik_reducering=('Borttagna unika rader','median'),
            Medel_unik_reducering=('Borttagna unika rader','mean'),
            Max_unik_reducering=('Borttagna unika rader','max'),
            Median_efter_filter=('Efter filter','median'),
            Median_stegreducering_pct=('Stegreducering %','median'),
            Median_spret_gap=('Spret-gap','median'),
            Median_utdelningslift_pct=('Utdelningslift %','median'),
        ).reset_index()
        agg['Träff när vald %'] = np.where(agg['Antal_val'] > 0, 100.0 * agg['Valda_i_träff'] / agg['Antal_val'], np.nan).round(1)
        agg['Risk per 1000 rader'] = np.where(agg['Medel_unik_reducering'] > 0, 1000.0 * agg['Valda_i_miss'] / agg['Medel_unik_reducering'], np.nan).round(3)
        def flag_row(x):
            if int(x.get('Dödade_facit', 0)) > 0 and float(x.get('Median_unik_reducering', 0) or 0) < 600:
                return 'RÖD: dödar facit med låg/mellan radnytta'
            if int(x.get('Dödade_facit', 0)) > 0:
                return 'GUL/RÖD: dödar facit, kontrollera om radnyttan är värd risken'
            if float(x.get('Median_unik_reducering', 0) or 0) >= 1000 and float(x.get('Träff när vald %', 0) or 0) >= 99:
                return 'GRÖN: hög radnytta utan miss i testet'
            if float(x.get('Median_unik_reducering', 0) or 0) < 150:
                return 'GUL: låg unik radnytta'
            return 'OK'
        agg['Audit flagga'] = agg.apply(flag_row, axis=1)
        agg = agg.sort_values(['Dödade_facit','Valda_i_miss','Median_unik_reducering'], ascending=[False, False, False])
        agg.to_csv(agg_path, index=False)
        paths['filter_agg_audit'] = str(agg_path)

        fam = step_df.groupby(['Ramprofil','Variant','Familj'], dropna=False).agg(
            Antal_filtersteg=('Filter','size'),
            Unika_filter=('Filter','nunique'),
            Valda_i_träff=('Facit överlever', lambda x: int((x == 'Ja').sum())),
            Valda_i_miss=('Facit överlever', lambda x: int((x != 'Ja').sum())),
            Dödade_facit=('Dödande filter', lambda x: int((x == 'Ja').sum())),
            Median_unik_reducering=('Borttagna unika rader','median'),
            Medel_unik_reducering=('Borttagna unika rader','mean'),
            Total_unik_reducering=('Borttagna unika rader','sum'),
        ).reset_index()
        fam['Träff när vald %'] = np.where(fam['Antal_filtersteg'] > 0, 100.0 * fam['Valda_i_träff'] / fam['Antal_filtersteg'], np.nan).round(1)
        fam = fam.sort_values(['Dödade_facit','Valda_i_miss','Total_unik_reducering'], ascending=[False, False, False])
        fam.to_csv(family_path, index=False)
        paths['family_audit'] = str(family_path)

    if not killer_df.empty:
        killer_df.to_csv(killer_path, index=False)
        paths['killer_filters'] = str(killer_path)
    else:
        pd.DataFrame(columns=['Ramprofil','Variant','Datum','Dödande filter','Kategori','Familj','Intervall','Facitvärde','Step removed rows']).to_csv(killer_path, index=False)
        paths['killer_filters'] = str(killer_path)
    if not package_df.empty:
        package_df.to_csv(package_path, index=False)
        paths['package_audit'] = str(package_path)

    return paths


def _v20_print_audit_report(out_dir: Path, base_prefix: str, audit_paths: dict):
    out_dir = Path(out_dir)
    report_path = out_dir / f'{base_prefix}_AUDIT_REPORT.txt'
    lines = []
    lines.append('TIPSET AI – V20 LIGHT FILTER AUDIT')
    lines.append('')
    lines.append('Syfte: hitta vilka filter som faktiskt hjälper, vilka som är redundanta och vilka som dödar facit i snabba 1–10-test.')
    lines.append('')
    if audit_paths.get('package_audit') and Path(audit_paths['package_audit']).exists():
        try:
            pdf = pd.read_csv(audit_paths['package_audit'])
            cols = [c for c in ['Ramprofil','Variant','Datum','Facit överlever','Grundram rader','Paketrader','Reducerar %','Paketträff','Valideringsträff','Orsak'] if c in pdf.columns]
            lines.append('PAKET PER TESTFALL')
            lines.append('-'*88)
            lines.append(pdf[cols].to_string(index=False))
            lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa package_audit: {e}')
    if audit_paths.get('killer_filters') and Path(audit_paths['killer_filters']).exists():
        try:
            kdf = pd.read_csv(audit_paths['killer_filters'])
            lines.append('DÖDANDE FILTER')
            lines.append('-'*88)
            if kdf.empty:
                lines.append('Inga dödande filter i detta light-test.')
            else:
                cols = [c for c in ['Ramprofil','Variant','Datum','Dödande filter','Kategori','Familj','Intervall','Facitvärde','Step removed rows','Orsak'] if c in kdf.columns]
                lines.append(kdf[cols].to_string(index=False))
            lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa killer_filters: {e}')
    if audit_paths.get('family_audit') and Path(audit_paths['family_audit']).exists():
        try:
            fdf = pd.read_csv(audit_paths['family_audit'])
            lines.append('FAMILJEAUDIT')
            lines.append('-'*88)
            cols = [c for c in ['Ramprofil','Variant','Familj','Antal_filtersteg','Unika_filter','Valda_i_träff','Valda_i_miss','Dödade_facit','Median_unik_reducering','Total_unik_reducering','Träff när vald %'] if c in fdf.columns]
            lines.append(fdf[cols].head(80).to_string(index=False))
            lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa family_audit: {e}')
    if audit_paths.get('filter_agg_audit') and Path(audit_paths['filter_agg_audit']).exists():
        try:
            adf = pd.read_csv(audit_paths['filter_agg_audit'])
            lines.append('FILTER MED HÖGST RISK / STÖRST NYTTA')
            lines.append('-'*88)
            cols = [c for c in ['Ramprofil','Variant','Filter','Kategori','Familj','Antal_val','Valda_i_träff','Valda_i_miss','Dödade_facit','Median_unik_reducering','Träff när vald %','Audit flagga'] if c in adf.columns]
            lines.append(adf[cols].head(100).to_string(index=False))
            lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa filter_agg_audit: {e}')
    lines.append('Auditfiler:')
    for k, p in audit_paths.items():
        lines.append(f'  - {k}: {p}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V20 FILTER-AUDIT', flush=True)
    print(f'Auditrapport: {report_path}', flush=True)
    for k, p in audit_paths.items():
        print(f'{k}: {p}', flush=True)


_ORIG_V19_PRINT_REPORT_FOR_V20 = _v19_print_report

def _v19_print_report(out_dir: Path, base_prefix: str, combined_paths: dict, frames: list, variants: list):  # noqa: F811
    _ORIG_V19_PRINT_REPORT_FOR_V20(out_dir, base_prefix, combined_paths, frames, variants)
    audit_paths = _v20_write_audit_outputs(Path(out_dir), base_prefix, frames)
    _v20_print_audit_report(Path(out_dir), base_prefix, audit_paths)


def main_v20():
    # Återanvänder V19-runnern men sätter V20-prefix och en snabbare standard om användaren inte anger annat.
    argv = list(sys.argv)
    if not any(a == '--output-prefix' or a.startswith('--output-prefix=') for a in argv[1:]):
        argv.extend(['--output-prefix', 'package_filter_audit_v20_test10'])
    if not any(a == '--variants' or a.startswith('--variants=') for a in argv[1:]):
        argv.extend(['--variants', 'BASE,FVFS,RADPRESS,FATCORE'])
    if not any(a == '--max-tests' or a.startswith('--max-tests=') for a in argv[1:]):
        argv.extend(['--max-tests', '10'])
    sys.argv = argv
    print('TIPSET AI V20 – LIGHT FILTER AUDIT', flush=True)
    print('Kör V19-familjevarianter + efteranalys av varje filtersteg.', flush=True)
    main_v19()




# =============================================================================
# V21 – RADPRESS PATCH / FILTERGUARD
# =============================================================================
# Syfte: bygg vidare på V20-auditen.
# Auditfynd:
#   1) Delta / Avvikelse dödade offensiva varianter nära intervallkant.
#   2) Grupp Favorit max 0 kunde döda facit med mycket låg unik radnytta.
# V21 testar därför små patchar, inte en helt ny motor.

V21_VARIANT_DEFS = {
    'V21A': {
        'label': 'V21A RADPRESS utan hård Delta',
        'description': 'RADPRESS-bas men Delta / Avvikelse får inte väljas som individuellt hårt filter. Kan fortfarande ingå mjukt i grupp.',
        'base': 'RADPRESS',
        'block_delta_individual': True,
        'block_delta_groups': False,
        'soft_favorite_max0': False,
        'green_name_bonus': True,
        'overrides': {
            'min_hit': 26,
            'candidate_min_hit': 28,
            'beam_width': 30,
            'archive_width': 220,
            'min_unique_rows': 60,
            'min_package_val_pct': 80.0,
            'min_candidate_val_pct': 78.0,
            'reduction_power': 2.70,
            'validation_power': 0.64,
            'payout_direction_weight': 0.16,
            'v15_group_max_filters': 14,
            'v15_max_group_candidates': 52,
            'v15_cross_per_family': 2,
        },
    },
    'V21B': {
        'label': 'V21B RADPRESS + Delta/favoritguard',
        'description': 'RADPRESS-bas med hård Delta spärrad och favoritgrupper max 0 spärrade när de kräver perfekt träff i grupp.',
        'base': 'RADPRESS',
        'block_delta_individual': True,
        'block_delta_groups': True,
        'soft_favorite_max0': True,
        'green_name_bonus': True,
        'overrides': {
            'min_hit': 26,
            'candidate_min_hit': 28,
            'beam_width': 32,
            'archive_width': 240,
            'min_unique_rows': 70,
            'min_package_val_pct': 80.0,
            'min_candidate_val_pct': 78.0,
            'reduction_power': 2.85,
            'validation_power': 0.66,
            'payout_direction_weight': 0.15,
            'v15_group_max_filters': 14,
            'v15_max_group_candidates': 54,
            'v15_cross_per_family': 2,
        },
    },
    'V21C': {
        'label': 'V21C grön kärna FAT/Poäng/ABC',
        'description': 'Som V21B men med extra bonus för auditgröna familjer och filter: FAT Summa, Poäng/rank, AI-Rank, Poängfilter, ABC Summa, Topp 6 favoriter.',
        'base': 'RADPRESS',
        'block_delta_individual': True,
        'block_delta_groups': True,
        'soft_favorite_max0': True,
        'green_name_bonus': True,
        'core_mode': True,
        'overrides': {
            'min_hit': 26,
            'candidate_min_hit': 28,
            'beam_width': 34,
            'archive_width': 260,
            'min_unique_rows': 75,
            'min_package_val_pct': 79.0,
            'min_candidate_val_pct': 77.0,
            'reduction_power': 3.00,
            'validation_power': 0.62,
            'payout_direction_weight': 0.14,
            'v15_group_max_filters': 14,
            'v15_max_group_candidates': 56,
            'v15_cross_per_family': 2,
        },
    },
    'V21D': {
        'label': 'V21D BASE med favoritguard',
        'description': 'BASE/SUPER-kontroll men tar bort perfekt favoritgrupp max 0 för att testa om BASE-missen 2026-04-11 räddas.',
        'base': 'BASE',
        'block_delta_individual': False,
        'block_delta_groups': False,
        'soft_favorite_max0': True,
        'green_name_bonus': False,
        'overrides': {
            'min_hit': 27,
            'candidate_min_hit': 29,
            'beam_width': 24,
            'archive_width': 180,
            'min_package_val_pct': 84.0,
            'min_candidate_val_pct': 84.0,
            'reduction_power': 1.90,
            'payout_direction_weight': 0.07,
        },
    },
}


def _v21_clone_variant(base_key: str, new_key: str, cfg: dict):
    base = V19_VARIANT_DEFS.get(base_key, V15_VARIANT_DEFS.get(base_key, {}))
    if not base:
        raise RuntimeError(f'Kan inte hitta basvariant för V21: {base_key}')
    new = dict(base)
    new['label'] = cfg['label']
    new['description'] = cfg['description']
    # Säkra att V21 arbetar med samma familjefokus som RADPRESS/BASE, men små patchar ovanpå.
    new['v21_cfg'] = cfg
    over = dict(base.get('overrides', {}) or {})
    over.update(cfg.get('overrides', {}) or {})
    new['overrides'] = over
    if bool(cfg.get('core_mode')):
        fam_bonus = dict(new.get('v19_family_bonus', {}) or {})
        fam_bonus.update({'FAT': 1.05, 'ABC': 0.95, 'Poäng/rank': 0.90, 'Favorit': 0.65, 'Skräll': 0.60, 'Värde/svårighet': 0.45, 'Tvärfamilj': 0.30})
        new['v19_family_bonus'] = fam_bonus
        new['v19_target_rows'] = 2250
        new['v19_min_active_families'] = 3
    V19_VARIANT_DEFS[new_key] = new
    V15_VARIANT_DEFS[new_key] = {
        'label': new['label'],
        'description': new['description'],
        'mode': new.get('mode', 'super_groups'),
        'structure_miss': new.get('structure_miss', [2, 3]),
        'family_miss': new.get('family_miss', [0, 1, 2, 3, 4, 5]),
        'cross_family': new.get('cross_family', True),
        'drop_individual_structure': new.get('drop_individual_structure', True),
        'drop_individual_profile': new.get('drop_individual_profile', False),
        'v19_allowed_families': new.get('v19_allowed_families'),
        'v19_drop_structure_all': new.get('v19_drop_structure_all', False),
        'v19_family_bonus': new.get('v19_family_bonus', {}),
        'v19_target_rows': new.get('v19_target_rows'),
        'v19_min_active_families': new.get('v19_min_active_families', 0),
        'v21_cfg': cfg,
        'overrides': new.get('overrides', {}),
    }
    V13_VARIANT_DEFS[new_key] = {
        'label': new['label'],
        'fixed_payout': True,
        'bundle_search': False,
        'description': new['description'],
    }


for _v21_key, _v21_cfg in V21_VARIANT_DEFS.items():
    _v21_clone_variant(_v21_cfg.get('base', 'RADPRESS'), _v21_key, _v21_cfg)


def _v21_is_delta_avvikelse(c: dict) -> bool:
    txt = f"{c.get('name','')} {c.get('key','')} {c.get('category','')} {c.get('interval_txt','')}".lower()
    return ('delta' in txt and ('avvik' in txt or 'avvikelse' in txt)) or 'delta / avvikelse' in txt


def _v21_group_has_delta(c: dict) -> bool:
    if not bool(c.get('is_v15_group')):
        return False
    for ch in c.get('group_children', []) or []:
        if _v21_is_delta_avvikelse(ch):
            return True
    return False


def _v21_is_bad_favorite_max0(c: dict) -> bool:
    if not bool(c.get('is_v15_group')):
        return False
    fam = str(c.get('group_family') or _v19_family_of_candidate(c))
    if fam != 'Favorit':
        return False
    try:
        max_miss = int(c.get('group_max_miss', -1))
        size = int(c.get('group_size', 0))
        required = int(c.get('group_required', 0))
    except Exception:
        return False
    # Audit visade att max 0 / 5 av 5 kunde döda för 49 rader. Spärra perfekt favoritgrupp
    # när gruppen är minst 4 filter och kräver alla.
    return max_miss == 0 and size >= 4 and required >= size


def _v21_filter_guard(c: dict, cfg: dict) -> bool:
    if bool(cfg.get('block_delta_individual')) and (not bool(c.get('is_v15_group'))) and _v21_is_delta_avvikelse(c):
        return False
    if bool(cfg.get('block_delta_groups')) and _v21_group_has_delta(c) and int(c.get('group_max_miss', 0)) == 0:
        return False
    if bool(cfg.get('soft_favorite_max0')) and _v21_is_bad_favorite_max0(c):
        return False
    return True


_ORIG_V21_TRANSFORM_CANDIDATES = _v15_transform_candidates

def _v15_transform_candidates(candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, args, variant_id: str) -> Tuple[List[dict], List[dict]]:  # noqa: F811
    final, group_cands = _ORIG_V21_TRANSFORM_CANDIDATES(candidates, htot, vtot, ftot, hist_payout, args, variant_id)
    vid = str(variant_id).upper()
    cfg = (V15_VARIANT_DEFS.get(vid, {}) or {}).get('v21_cfg')
    if not cfg:
        return final, group_cands
    final2 = [c for c in final if _v21_filter_guard(c, cfg)]
    group2 = [g for g in group_cands if _v21_filter_guard(g, cfg)]
    # Om spärren råkar tömma allt, falla tillbaka men behåll blockerad individuell Delta.
    if not final2:
        final2 = [c for c in final if not ((not bool(c.get('is_v15_group'))) and _v21_is_delta_avvikelse(c))]
    return final2, group2


_ORIG_V21_STATE_METRICS = _v13_state_metrics

def _v13_state_metrics(st: _V9State, payout_values: np.ndarray, args, variant_id: str) -> dict:  # noqa: F811
    m = _ORIG_V21_STATE_METRICS(st, payout_values, args, variant_id)
    vid = str(variant_id).upper()
    cfg = (V15_VARIANT_DEFS.get(vid, {}) or {}).get('v21_cfg')
    if not cfg:
        return m
    if bool(cfg.get('green_name_bonus')):
        bonus = 0.0
        names = [str(c.get('name', '')).lower() for c in (st.chosen or tuple())]
        for n in names:
            if 'fat summa' in n:
                bonus += 0.25
            if 'poäng/rank' in n or 'rank summa' in n or 'ai-rank' in n or 'poängfilter' in n:
                bonus += 0.22
            if 'abc summa' in n or 'grupp abc' in n:
                bonus += 0.18
            if 'topp 6 favoriter' in n:
                bonus += 0.14
        # Extra hård straff om en förbjuden kandidat ändå skulle slinka igenom.
        for c in (st.chosen or tuple()):
            if _v21_is_delta_avvikelse(c) and not bool(c.get('is_v15_group')):
                bonus -= 5.0
            if _v21_is_bad_favorite_max0(c):
                bonus -= 5.0
        m['v21_green_bonus'] = float(bonus)
        m['joint_score'] = float(m.get('joint_score', 0.0)) + float(bonus)
    return m


def _v21_parse_variants(value: str) -> List[str]:
    raw = str(value or 'BASE,RADPRESS,V21A,V21B,V21C,V21D').replace(';', ',').replace('+', ',').split(',')
    out = []
    allowed = set(V19_VARIANT_DEFS.keys())
    for x in raw:
        k = x.strip().upper()
        if not k:
            continue
        if k not in allowed:
            raise ValueError(f'Okänd V21-variant: {k}. Tillåtna: {", ".join(sorted(allowed))}')
        if k not in out:
            out.append(k)
    return out or ['BASE', 'RADPRESS', 'V21A', 'V21B', 'V21C', 'V21D']


def _v21_print_report(out_dir: Path, base_prefix: str, combined_paths: dict, frames: list, variants: list):
    _ORIG_V19_PRINT_REPORT_FOR_V20(out_dir, base_prefix, combined_paths, frames, variants)
    audit_paths = _v20_write_audit_outputs(Path(out_dir), base_prefix, frames)
    # Skriv en liten V21-specifik rapport ovanpå auditfilerna.
    report_path = Path(out_dir) / f'{base_prefix}_V21_PATCH_REPORT.txt'
    lines = []
    lines.append('TIPSET AI – V21 RADPRESS PATCH / FILTERGUARD')
    lines.append('')
    lines.append('Patchar från V20-audit:')
    lines.append('  - Delta / Avvikelse spärras som individuellt hårt filter i V21A/V21B/V21C.')
    lines.append('  - Perfekta favoritgrupper max 0 spärras i V21B/V21C/V21D när gruppen kräver alla filter.')
    lines.append('  - V21C ger extra bonus till auditgröna filterfamiljer: FAT, Poäng/rank, ABC och Topp 6 favoriter.')
    lines.append('')
    if combined_paths.get('summary') and Path(combined_paths['summary']).exists():
        try:
            sdf = pd.read_csv(combined_paths['summary'])
            cols = [c for c in ['Ramprofil','Teoretisk grundram','Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Median reducering %','Median filter','Max filter','Vinnare','Vinnare alla ramar'] if c in sdf.columns]
            lines.append('SAMMANFATTNING')
            lines.append('-'*88)
            lines.append(sdf[cols].to_string(index=False))
            lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa summary: {e}')
    if audit_paths.get('killer_filters') and Path(audit_paths['killer_filters']).exists():
        try:
            kdf = pd.read_csv(audit_paths['killer_filters'])
            lines.append('DÖDANDE FILTER')
            lines.append('-'*88)
            if kdf.empty:
                lines.append('Inga dödande filter.')
            else:
                cols = [c for c in ['Ramprofil','Variant','Datum','Dödande filter','Kategori','Familj','Intervall','Facitvärde','Step removed rows','Orsak'] if c in kdf.columns]
                lines.append(kdf[cols].to_string(index=False))
            lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa killer_filters: {e}')
    lines.append('Auditfiler:')
    for k, p in audit_paths.items():
        lines.append(f'  - {k}: {p}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V21 PATCH/FILTERGUARD', flush=True)
    print(f'V21-rapport: {report_path}', flush=True)
    for k, p in audit_paths.items():
        print(f'{k}: {p}', flush=True)


# Ersätt V19/V20-rapporthooken med V21-rapporten när main_v21 körs.
def main_v21():
    # V21c är medvetet låst. Jupyter skickar ofta -f <kernel>.json,
    # och tidigare versioner kunde råka falla tillbaka till gamla V19/V20-workerkod.
    # Här behåller vi bara användarens explicita godkända flaggor och tvingar inline-körning.
    original = list(sys.argv[1:])
    allowed_with_value = {
        '--app-file', '--db-file', '--out-dir', '--frames', '--max-tests', '--test-offset',
        '--top-n', '--wide-n', '--pay-min', '--pay-max', '--output-prefix'
    }
    allowed_bool = {'--include-supermakro', '--fast-no-supermakro', '--enable-triples'}
    cleaned = []
    i = 0
    while i < len(original):
        a = original[i]
        if a in allowed_with_value and i + 1 < len(original):
            cleaned.extend([a, original[i+1]])
            i += 2
            continue
        if any(a.startswith(k + '=') for k in allowed_with_value):
            cleaned.append(a)
            i += 1
            continue
        if a in allowed_bool:
            cleaned.append(a)
            i += 1
            continue
        # ignorera kernelargument och gamla variant-/prefixflaggor
        i += 1

    argv = [sys.argv[0]] + cleaned
    def hasflag(flag):
        return any(a == flag or a.startswith(flag + '=') for a in argv[1:])
    if not hasflag('--output-prefix'):
        argv.extend(['--output-prefix', 'package_filter_guard_v21c_test10'])
    # LÅST variantlista: gamla FVFS/FATCORE får inte smyga in.
    argv.extend(['--variants', 'BASE,RADPRESS,V21A,V21B,V21C,V21D'])
    if not hasflag('--max-tests'):
        argv.extend(['--max-tests', '10'])
    # Viktigt: kör inline i denna fil så V21-patcharna verkligen används.
    argv.append('--no-fresh-workers')
    sys.argv = argv
    globals()['_v19_print_report'] = _v21_print_report
    print('TIPSET AI V21c – RADPRESS PATCH / FILTERGUARD', flush=True)
    print('LÅST variantlista: BASE,RADPRESS,V21A,V21B,V21C,V21D', flush=True)
    print('Tvingar inline-körning: inga externa V19/V20-workers används.', flush=True)
    main_v19()



# =============================================================================
# V22 – CONSTRUCT + REPAIR + GROUPIFY
# =============================================================================
# Idé från Niklas:
#   1) Plocka först in filter som är starka var för sig och faktiskt reducerar.
#   2) Pressa paketet mot budgetområdet 2 300–2 500 rader.
#   3) Reparera träffen genom att blocka/mjuka riskfilter och premiera gruppregler
#      framför små hårda filter där reduceringen är låg.
#
# Detta är fortfarande en snabb test10-runner, inte slutmotor i appen.

V22_VARIANT_DEFS = {
    'V21A': V21_VARIANT_DEFS['V21A'],  # kontroll från bästa V21
    'BUILD5_RAW': {
        'label': 'BUILD5 rå aggressiv bank',
        'description': 'Aggressiv konstruktion: kandidatbank med minst cirka 5 % egen reducering. Används som stress-test, inte som slutval.',
        'base': 'RADPRESS',
        'block_delta_individual': True,
        'block_delta_groups': False,
        'soft_favorite_max0': False,
        'green_name_bonus': True,
        'v22_mode': 'construct_raw',
        'v22_min_candidate_red_pct': 5.0,
        'v22_target_rows': 2400,
        'v22_budget_rows': 2500,
        'v22_prefer_groups': False,
        'v22_penalize_low_step': True,
        'overrides': {
            'min_hit': 24,
            'candidate_min_hit': 28,
            'beam_width': 26,
            'archive_width': 180,
            'min_unique_rows': 320,
            'min_package_val_pct': 76.0,
            'min_candidate_val_pct': 76.0,
            'reduction_power': 3.35,
            'validation_power': 0.55,
            'payout_direction_weight': 0.12,
            'v15_group_max_filters': 14,
            'v15_max_group_candidates': 48,
            'v15_cross_per_family': 2,
        },
    },
    'REPAIR27': {
        'label': 'REPAIR27 bygg hårt + reparera till 27/30',
        'description': 'Bygger offensivt men håller gemensam paketträff minst 27/30 och blockerar kända riskfilter.',
        'base': 'RADPRESS',
        'block_delta_individual': True,
        'block_delta_groups': True,
        'soft_favorite_max0': True,
        'green_name_bonus': True,
        'v22_mode': 'repair',
        'v22_min_candidate_red_pct': 3.5,
        'v22_target_rows': 2450,
        'v22_budget_rows': 2500,
        'v22_prefer_groups': False,
        'v22_penalize_low_step': True,
        'overrides': {
            'min_hit': 27,
            'candidate_min_hit': 28,
            'beam_width': 30,
            'archive_width': 220,
            'min_unique_rows': 180,
            'min_package_val_pct': 79.0,
            'min_candidate_val_pct': 77.0,
            'reduction_power': 3.10,
            'validation_power': 0.68,
            'payout_direction_weight': 0.14,
            'v15_group_max_filters': 14,
            'v15_max_group_candidates': 56,
            'v15_cross_per_family': 2,
        },
    },
    'GROUPIFY27': {
        'label': 'GROUPIFY27 reparera med gruppregler',
        'description': 'Som REPAIR27 men premierar gruppfilter/max-miss-regler framför hårda singlar. Målet är samma reducering men högre träff.',
        'base': 'RADPRESS',
        'block_delta_individual': True,
        'block_delta_groups': True,
        'soft_favorite_max0': True,
        'green_name_bonus': True,
        'v22_mode': 'groupify',
        'v22_min_candidate_red_pct': 3.0,
        'v22_target_rows': 2450,
        'v22_budget_rows': 2500,
        'v22_prefer_groups': True,
        'v22_penalize_low_step': True,
        'overrides': {
            'min_hit': 27,
            'candidate_min_hit': 28,
            'beam_width': 32,
            'archive_width': 240,
            'min_unique_rows': 140,
            'min_package_val_pct': 80.0,
            'min_candidate_val_pct': 77.0,
            'reduction_power': 3.00,
            'validation_power': 0.72,
            'payout_direction_weight': 0.13,
            'v15_group_max_filters': 16,
            'v15_max_group_candidates': 64,
            'v15_cross_per_family': 2,
            'v15_cross_max_filters': 16,
        },
    },
    'BUDGET2500': {
        'label': 'BUDGET2500 aktivt mål ≤2500',
        'description': 'Budgetstyrd variant som ger stark score för paket nära eller under 2500 rader, men med filterguard.',
        'base': 'RADPRESS',
        'block_delta_individual': True,
        'block_delta_groups': True,
        'soft_favorite_max0': True,
        'green_name_bonus': True,
        'v22_mode': 'budget2500',
        'v22_min_candidate_red_pct': 3.0,
        'v22_target_rows': 2350,
        'v22_budget_rows': 2500,
        'v22_prefer_groups': True,
        'v22_penalize_low_step': True,
        'overrides': {
            'min_hit': 26,
            'candidate_min_hit': 28,
            'beam_width': 34,
            'archive_width': 260,
            'min_unique_rows': 120,
            'min_package_val_pct': 78.0,
            'min_candidate_val_pct': 76.0,
            'reduction_power': 3.55,
            'validation_power': 0.60,
            'payout_direction_weight': 0.12,
            'v15_group_max_filters': 16,
            'v15_max_group_candidates': 66,
            'v15_cross_per_family': 2,
            'v15_cross_max_filters': 16,
        },
    },
    'BUDGET2300': {
        'label': 'BUDGET2300 hård budgetpress',
        'description': 'Mer aggressivt mål runt 2300 rader. Accepterar något lägre intern paketträff för att se var riskgränsen går.',
        'base': 'RADPRESS',
        'block_delta_individual': True,
        'block_delta_groups': True,
        'soft_favorite_max0': True,
        'green_name_bonus': True,
        'v22_mode': 'budget2300',
        'v22_min_candidate_red_pct': 2.5,
        'v22_target_rows': 2200,
        'v22_budget_rows': 2350,
        'v22_prefer_groups': True,
        'v22_penalize_low_step': True,
        'overrides': {
            'min_hit': 25,
            'candidate_min_hit': 28,
            'beam_width': 34,
            'archive_width': 260,
            'min_unique_rows': 100,
            'min_package_val_pct': 76.0,
            'min_candidate_val_pct': 75.0,
            'reduction_power': 3.85,
            'validation_power': 0.54,
            'payout_direction_weight': 0.10,
            'v15_group_max_filters': 16,
            'v15_max_group_candidates': 66,
            'v15_cross_per_family': 2,
            'v15_cross_max_filters': 16,
        },
    },
}


def _v22_clone_variant(base_key: str, new_key: str, cfg: dict):
    # V21A finns redan registrerad via V21-kloningen. För övriga V22-varianter
    # återanvänd V21-klonaren så Delta-/favoritguard fungerar i samma pipeline.
    if new_key == 'V21A':
        base = V15_VARIANT_DEFS.get('V21A', {})
        if base:
            base['v22_cfg'] = {
                'mode': 'control',
                'target_rows': 2500,
                'budget_rows': 2500,
                'min_candidate_red_pct': 0.0,
                'prefer_groups': False,
                'penalize_low_step': False,
            }
        return
    _v21_clone_variant(base_key, new_key, cfg)
    # Lägg V22-specifika inställningar i V15-konfigen.
    v22_cfg = {
        'mode': cfg.get('v22_mode', ''),
        'target_rows': int(cfg.get('v22_target_rows') or 2500),
        'budget_rows': int(cfg.get('v22_budget_rows') or cfg.get('v22_target_rows') or 2500),
        'min_candidate_red_pct': float(cfg.get('v22_min_candidate_red_pct') or 0.0),
        'prefer_groups': bool(cfg.get('v22_prefer_groups', False)),
        'penalize_low_step': bool(cfg.get('v22_penalize_low_step', False)),
    }
    V15_VARIANT_DEFS[new_key]['v22_cfg'] = v22_cfg
    # Öka familjebonus för familjer som V20-auditen visade grönt.
    fam_bonus = dict(V15_VARIANT_DEFS[new_key].get('v19_family_bonus', {}) or {})
    fam_bonus.update({
        'FAT': max(float(fam_bonus.get('FAT', 0.0)), 1.05),
        'ABC': max(float(fam_bonus.get('ABC', 0.0)), 0.95),
        'Poäng/rank': max(float(fam_bonus.get('Poäng/rank', 0.0)), 0.95),
        'Favorit': max(float(fam_bonus.get('Favorit', 0.0)), 0.65),
        'Skräll': max(float(fam_bonus.get('Skräll', 0.0)), 0.65),
        # Värde/svårighet får mindre bonus eftersom Delta/Avvikelse visade kant-risk.
        'Värde/svårighet': min(max(float(fam_bonus.get('Värde/svårighet', 0.0)), 0.35), 0.65),
        'Tvärfamilj': max(float(fam_bonus.get('Tvärfamilj', 0.0)), 0.40),
    })
    V15_VARIANT_DEFS[new_key]['v19_family_bonus'] = fam_bonus
    V19_VARIANT_DEFS[new_key]['v19_family_bonus'] = fam_bonus
    V19_VARIANT_DEFS[new_key]['v22_cfg'] = v22_cfg


for _v22_key, _v22_cfg in V22_VARIANT_DEFS.items():
    if _v22_key == 'V21A':
        _v22_clone_variant('V21A', 'V21A', _v22_cfg)
    else:
        _v22_clone_variant(_v22_cfg.get('base', 'RADPRESS'), _v22_key, _v22_cfg)


def _v22_cfg_for_variant(variant_id: str) -> dict:
    return (V15_VARIANT_DEFS.get(str(variant_id).upper(), {}) or {}).get('v22_cfg') or {}


def _v22_is_low_value_risk_candidate(c: dict) -> bool:
    # Filter som tidigare audit visade som farliga när de är perfekta och små.
    # Exakt radnytta per steg vet vi först i apply, men vissa typer ska hindras redan i kandidatpoolen.
    if _v21_is_bad_favorite_max0(c):
        return True
    if (not bool(c.get('is_v15_group'))) and _v21_is_delta_avvikelse(c):
        return True
    return False


_ORIG_V22_TRANSFORM_CANDIDATES = _v15_transform_candidates

def _v15_transform_candidates(candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, args, variant_id: str) -> Tuple[List[dict], List[dict]]:  # noqa: F811
    final, group_cands = _ORIG_V22_TRANSFORM_CANDIDATES(candidates, htot, vtot, ftot, hist_payout, args, variant_id)
    vid = str(variant_id).upper()
    cfg = _v22_cfg_for_variant(vid)
    if not cfg:
        return final, group_cands

    min_red = float(cfg.get('min_candidate_red_pct') or 0.0)
    prefer_groups = bool(cfg.get('prefer_groups', False))

    def keep(c: dict) -> bool:
        # V21-guard har redan blockat Delta/favorit där relevant, men säkra igen för V22.
        if _v22_is_low_value_risk_candidate(c):
            return False
        fam = _v19_family_of_candidate(c)
        red = float(c.get('red_pct', 0.0) or 0.0)
        # Strukturgrupper får vara skydd även om egen reducering är låg.
        if fam == 'Struktur':
            return True
        # Gruppregler kan ibland ha lägre egen red_pct men bättre samlad riskprofil.
        if bool(c.get('is_v15_group')) and prefer_groups:
            return red >= max(1.0, min_red * 0.55)
        return red >= min_red

    f2 = [c for c in final if keep(c)]
    g2 = [g for g in group_cands if keep(g)]

    # Sortera så att gruppifierade varianter ser grupper tidigare, men behåll engine-sökning.
    def srt(c: dict):
        fam = _v19_family_of_candidate(c)
        group_bonus = 1.0 if bool(c.get('is_v15_group')) and prefer_groups else 0.0
        green_bonus = 0.0
        name = str(c.get('name','')).lower()
        if 'fat summa' in name or 'abc summa' in name or 'ai-rank' in name or 'poängfilter' in name or 'rank summa' in name or 'topp 6 favoriter' in name:
            green_bonus += 1.0
        return (group_bonus, green_bonus, int(c.get('hist_hit',0)), float(c.get('val_pct',0.0)), float(c.get('red_pct',0.0)), -int(c.get('frame_keep',10**12)))

    f2 = sorted(f2, key=srt, reverse=True)
    g2 = sorted(g2, key=srt, reverse=True)
    if not f2:
        # Fallback: behåll guardad lista om 5%-banken blev för tom.
        f2 = [c for c in final if not _v22_is_low_value_risk_candidate(c)]
        g2 = [g for g in group_cands if not _v22_is_low_value_risk_candidate(g)]
    return f2, g2


_ORIG_V22_STATE_METRICS = _v13_state_metrics

def _v13_state_metrics(st: _V9State, payout_values: np.ndarray, args, variant_id: str) -> dict:  # noqa: F811
    m = _ORIG_V22_STATE_METRICS(st, payout_values, args, variant_id)
    vid = str(variant_id).upper()
    cfg = _v22_cfg_for_variant(vid)
    if not cfg:
        return m

    target = int(cfg.get('target_rows') or 2500)
    budget = int(cfg.get('budget_rows') or target)
    prefer_groups = bool(cfg.get('prefer_groups', False))
    penalize_low_step = bool(cfg.get('penalize_low_step', False))

    rows = int(st.frame_count)
    frame_size = max(1, int(st.frame_size))
    bonus = 0.0

    # Budgetscore: sök aktivt mot budgetintervallet, inte bara högsta säkerhet.
    if rows > budget:
        # Tydligare penalty när vi ligger kvar på 3k+.
        bonus -= 4.5 * min(1.5, (rows - budget) / frame_size * 5.0)
    else:
        # Belöna att nå budget, men straffa inte om paketet är något över target men under budget.
        bonus += 0.75
        if rows <= target:
            bonus += 0.35
        # För små paket kan vara överrisk; liten penalty under ca 80 % av target.
        if rows < int(0.80 * target):
            bonus -= 0.60 * ((0.80 * target - rows) / max(1.0, target))

    # Groupify: premierar gruppregler som ersätter farliga hårda filter.
    chosen = list(st.chosen or tuple())
    group_count = sum(1 for c in chosen if bool(c.get('is_v15_group')))
    indiv_count = max(0, len(chosen) - group_count)
    if prefer_groups:
        bonus += 0.22 * min(group_count, 5)
        if len(chosen) >= 4 and group_count == 0:
            bonus -= 0.60
        # Max0-grupper som kräver allt är farliga om de inte har stor effekt; straffa generellt.
        for c in chosen:
            if bool(c.get('is_v15_group')):
                try:
                    if int(c.get('group_max_miss', 0)) == 0 and int(c.get('group_size', 0)) >= 4:
                        bonus -= 0.12
                except Exception:
                    pass

    # Straffa filtersteg som bara gör kosmetisk reducering.
    if penalize_low_step:
        for c in chosen:
            step_removed = int(c.get('step_removed_rows', 0) or 0)
            if step_removed and step_removed < max(60, int(0.012 * frame_size)):
                bonus -= 0.18
            if step_removed and step_removed >= int(0.05 * frame_size):
                bonus += 0.12

    # Namn-/familjebonus från V20-auditen.
    for c in chosen:
        name = str(c.get('name','')).lower()
        fam = _v19_family_of_candidate(c)
        if 'fat summa' in name:
            bonus += 0.28
        if 'abc summa' in name or 'grupp abc' in name:
            bonus += 0.22
        if 'ai-rank' in name or 'poängfilter' in name or 'rank summa' in name or fam == 'Poäng/rank':
            bonus += 0.20
        if 'topp 6 favoriter' in name:
            bonus += 0.16
        if 'fat 2-sekvenser' in name:
            bonus -= 0.10
        if _v21_is_delta_avvikelse(c) and not bool(c.get('is_v15_group')):
            bonus -= 6.0
        if _v21_is_bad_favorite_max0(c):
            bonus -= 6.0

    m['v22_budget_bonus'] = float(bonus)
    m['joint_score'] = float(m.get('joint_score', 0.0)) + float(bonus)
    return m


def _v22_parse_variants(value: str) -> List[str]:
    # V22b: robust parser. Gamla notebook-celler kan fortfarande skicka V21B/V21C/V21D
    # eller V19-varianter. Det ska inte krascha, och om listan bara är gammal körs
    # V22-standard i stället.
    default = ['V21A','BUILD5_RAW','REPAIR27','GROUPIFY27','BUDGET2500','BUDGET2300']
    raw_items = [x.strip().upper() for x in str(value or ','.join(default)).replace(';', ',').replace('+', ',').split(',') if x.strip()]
    allowed = set(V22_VARIANT_DEFS.keys()) | {'BASE','RADPRESS'}
    v22_core = {'BUILD5_RAW','REPAIR27','GROUPIFY27','BUDGET2500','BUDGET2300'}
    legacy = {'V21B','V21C','V21D','FVFS','FATCORE'}

    # Om användaren råkar köra gammal V21/V19-variantlista: kör V22-standard.
    if raw_items and any(k in legacy for k in raw_items) and not any(k in v22_core for k in raw_items):
        print('V22b: gammal variantlista upptäckt (' + ','.join(raw_items) + '). Kör V22-standard i stället.', flush=True)
        return default

    out = []
    for k in raw_items:
        if k in {'V21B','V21C','V21D'}:
            # V21A är kontrollen i V22. V21B/C/D hör hemma i V21-testet.
            k = 'V21A'
        if k in {'FVFS','FATCORE'}:
            print(f'V22b: ignorerar gammal V19-variant {k}.', flush=True)
            continue
        if k not in allowed:
            print(f'V22b: ignorerar okänd variant {k}. Tillåtna V22: {", ".join(default)}', flush=True)
            continue
        if k not in out:
            out.append(k)
    if not any(k in v22_core for k in out):
        # Undvik att av misstag bara köra kontroller och missa själva V22-testet.
        return default
    return out or default


def _v22_collect_combined_outputs(out_dir: Path, base_prefix: str, frames: list) -> dict:
    return _v19_collect_combined_outputs(out_dir, base_prefix, frames)


def _v22_print_report(out_dir: Path, base_prefix: str, combined_paths: dict, frames: list, variants: list):
    # Skriv V22-rapport utan att återanropa V19-rapportbannern.
    audit_paths = _v20_write_audit_outputs(Path(out_dir), base_prefix, frames)
    report_path = Path(out_dir) / f'{base_prefix}_V22_CONSTRUCT_REPAIR_GROUPIFY_REPORT.txt'
    lines = []
    lines.append('TIPSET AI – V22 CONSTRUCT + REPAIR + GROUPIFY')
    lines.append('')
    lines.append('Syfte: börja aggressivt med filter som reducerar, reparera träff genom guard/breddning/gruppifiering och jämför mot V21A-kontrollen.')
    lines.append('')
    lines.append('V22-varianter:')
    lines.append('  - V21A: bästa V21-kontroll.')
    lines.append('  - BUILD5_RAW: rå aggressiv kandidatbank, cirka 5 % reduceringskrav.')
    lines.append('  - REPAIR27: hård byggfas men paketträff minst 27/30 och filterguard.')
    lines.append('  - GROUPIFY27: som REPAIR27 men premierar gruppregler/max-miss.')
    lines.append('  - BUDGET2500/BUDGET2300: aktiv radbudgetpress.')
    lines.append('')
    if combined_paths.get('summary') and Path(combined_paths['summary']).exists():
        try:
            sdf = pd.read_csv(combined_paths['summary'])
            cols = [c for c in ['Ramprofil','Teoretisk grundram','Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Median reducering %','Median filter','Max filter','Vinnare','Vinnare alla ramar'] if c in sdf.columns]
            lines.append('SAMMANFATTNING')
            lines.append('-'*100)
            lines.append(sdf[cols].to_string(index=False))
            lines.append('')
            # Extra enkel målbedömning.
            if 'Median paketrader' in sdf.columns and 'Träffar' in sdf.columns:
                sdf2 = sdf.copy()
                sdf2['Budget OK ≤2500'] = pd.to_numeric(sdf2['Median paketrader'], errors='coerce') <= 2500
                cols2 = [c for c in ['Variant','Träffar','Median paketrader','Budget OK ≤2500'] if c in sdf2.columns]
                lines.append('BUDGETKOLL')
                lines.append('-'*100)
                lines.append(sdf2[cols2].to_string(index=False))
                lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa summary: {e}')
    if audit_paths.get('killer_filters') and Path(audit_paths['killer_filters']).exists():
        try:
            kdf = pd.read_csv(audit_paths['killer_filters'])
            lines.append('DÖDANDE FILTER')
            lines.append('-'*100)
            if kdf.empty:
                lines.append('Inga dödande filter.')
            else:
                cols = [c for c in ['Ramprofil','Variant','Datum','Dödande filter','Kategori','Familj','Intervall','Facitvärde','Step removed rows','Orsak'] if c in kdf.columns]
                lines.append(kdf[cols].to_string(index=False))
            lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa killer_filters: {e}')
    lines.append('Auditfiler:')
    for k, p in audit_paths.items():
        lines.append(f'  - {k}: {p}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V22b CONSTRUCT + REPAIR + GROUPIFY', flush=True)
    if combined_paths.get('summary') and Path(combined_paths['summary']).exists():
        try:
            sdf = pd.read_csv(combined_paths['summary'])
            show_cols = [c for c in ['Ramprofil','Teoretisk grundram','Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Median reducering %','Median filter','Max filter','Vinnare','Vinnare alla ramar'] if c in sdf.columns]
            print(sdf[show_cols].to_string(index=False), flush=True)
        except Exception:
            pass
    print(f'V22b-rapport: {report_path}', flush=True)
    for k, p in audit_paths.items():
        print(f'{k}: {p}', flush=True)


def main_v22():
    # Låst snabbtest: Colab/Jupyter kan skicka -f kernel.json, så vi filtrerar argument.
    parser = argparse.ArgumentParser(description='V22 Construct + Repair + Groupify: snabb test10-runner.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_construct_repair_groupify_v22b_test10')
    parser.add_argument('--frames', default='3-5-5', help='Kommalista i format spik-halv-hel. Ex: 2-7-4,3-4-6,3-5-5')
    parser.add_argument('--variants', default='V21A,BUILD5_RAW,REPAIR27,GROUPIFY27,BUDGET2500,BUDGET2300')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--min-hit', type=int, default=27)
    parser.add_argument('--min-candidate-val-pct', type=float, default=78.0)
    parser.add_argument('--min-package-val-pct', type=float, default=79.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=95.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=95.0)
    parser.add_argument('--min-gap-score', type=float, default=0.60)
    parser.add_argument('--min-unique-rows', type=int, default=120)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='3-5-5', help=argparse.SUPPRESS)
    parser.add_argument('--beam-width', type=int, default=30)
    parser.add_argument('--archive-width', type=int, default=230)
    parser.add_argument('--structure-seed-count', type=int, default=2)
    parser.add_argument('--variants-per-key', type=int, default=3)
    parser.add_argument('--max-candidates', type=int, default=150)
    parser.add_argument('--hit-power', type=float, default=1.45)
    parser.add_argument('--validation-power', type=float, default=0.66)
    parser.add_argument('--reduction-power', type=float, default=3.00)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.18)
    parser.add_argument('--payout-direction-weight', type=float, default=0.12)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=50)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=275)
    parser.add_argument('--per-bucket-keep', type=int, default=2)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--v15-group-max-filters', type=int, default=16)
    parser.add_argument('--v15-max-group-candidates', type=int, default=66)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=2)
    parser.add_argument('--v15-cross-max-filters', type=int, default=16)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    parser.add_argument('--force-upload', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-fresh-workers', action='store_true', default=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)

    frames = _v17_parse_frames(args.frames)
    variants = _v22_parse_variants(args.variants)
    args.variants = ','.join(variants)
    # Säkerställ inline-körning så V22-patcharna används.
    args.no_fresh_workers = True

    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)
    print('\n' + '='*86, flush=True)
    print('TIPSET AI V22b – CONSTRUCT + REPAIR + GROUPIFY', flush=True)
    print(f'Test: {args.max_tests} omgångar inom {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '), flush=True)
    print(f'Ramar: {", ".join(frames)}', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('V22b robust variantparser: gamla V21B/V21C/V21D/FVFS/FATCORE-argument kraschar inte.', flush=True)
    print('Låst inline-körning. Fokus: aggressiv filterbank → reparera träff → gruppifiera riskfilter.', flush=True)
    print('='*86, flush=True)

    # Hooka rapporten till V22-rapport, inte V19/V21.
    globals()['_v19_print_report'] = _v22_print_report

    for frame in frames:
        frame_args = argparse.Namespace(**vars(args))
        frame_args.frames = frame
        frame_args.frame_profile = frame
        frame_args.output_prefix = f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n' + '-'*86, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*86, flush=True)
        ns = _load_app_functions(app_file, fast_no_supermakro=False)
        db = ns['load_database'](str(db_file), 13)
        detail, meta = _run_backtest_v15(ns, db, frame_args)
        _write_v15_outputs(detail, frame_args, out_dir, app_file, db_file, worker_count=0)

    combined = _v22_collect_combined_outputs(out_dir, args.output_prefix, frames)
    _v22_print_report(out_dir, args.output_prefix, combined, frames, variants)


# =============================================================================
# V23 – STRICT BUDGET DIAGNOSTIC
# =============================================================================
# Syfte: testa om nuvarande kandidatpoolen faktiskt kan komma under 2 500/2 300
# rader. Till skillnad från V22 är budget här inte bara en bonus i score.
# Sorteringen premierar lågt radantal inom säkerhetsgolven, så beam-arkivet inte
# automatiskt fastnar på trygga paket runt 3 000 rader.

V23_VARIANT_DEFS = {
    'V21A': V21_VARIANT_DEFS['V21A'],
    'STRICT2500': {
        'label': 'STRICT2500 hårt budgettak 2500',
        'description': 'Tvingar sökningen att prioritera paket under/kring 2500 rader, med Delta-guard och mjukare riskgrupper.',
        'base': 'RADPRESS',
        'block_delta_individual': True,
        'block_delta_groups': True,
        'soft_favorite_max0': True,
        'green_name_bonus': True,
        'v22_mode': 'strict2500',
        'v22_min_candidate_red_pct': 1.5,
        'v22_target_rows': 2350,
        'v22_budget_rows': 2500,
        'v22_prefer_groups': True,
        'v22_penalize_low_step': True,
        'v23_strict_budget': True,
        'overrides': {
            'min_hit': 25,
            'candidate_min_hit': 27,
            'beam_width': 64,
            'archive_width': 520,
            'min_unique_rows': 35,
            'min_package_val_pct': 72.0,
            'min_candidate_val_pct': 70.0,
            'reduction_power': 5.2,
            'validation_power': 0.35,
            'hit_power': 0.9,
            'payout_direction_weight': 0.06,
            'v15_group_max_filters': 24,
            'v15_max_group_candidates': 120,
            'v15_cross_per_family': 3,
            'v15_cross_max_filters': 24,
            'bundle_pool_size': 14,
            'max_bundle_trials_per_state': 120,
            'max_bundle_keep_per_state': 18,
        },
    },
    'STRICT2300': {
        'label': 'STRICT2300 hårt budgettak 2300',
        'description': 'Ännu hårdare budgetdiagnos mot cirka 2300 rader. Får hellre visa träfftapp än gömma att budget inte nås.',
        'base': 'RADPRESS',
        'block_delta_individual': True,
        'block_delta_groups': True,
        'soft_favorite_max0': True,
        'green_name_bonus': True,
        'v22_mode': 'strict2300',
        'v22_min_candidate_red_pct': 1.0,
        'v22_target_rows': 2200,
        'v22_budget_rows': 2300,
        'v22_prefer_groups': True,
        'v22_penalize_low_step': True,
        'v23_strict_budget': True,
        'overrides': {
            'min_hit': 24,
            'candidate_min_hit': 27,
            'beam_width': 72,
            'archive_width': 650,
            'min_unique_rows': 25,
            'min_package_val_pct': 68.0,
            'min_candidate_val_pct': 68.0,
            'reduction_power': 5.8,
            'validation_power': 0.25,
            'hit_power': 0.75,
            'payout_direction_weight': 0.04,
            'v15_group_max_filters': 26,
            'v15_max_group_candidates': 140,
            'v15_cross_per_family': 3,
            'v15_cross_max_filters': 26,
            'bundle_pool_size': 16,
            'max_bundle_trials_per_state': 160,
            'max_bundle_keep_per_state': 22,
        },
    },
    'STRICT2100': {
        'label': 'STRICT2100 stress-test',
        'description': 'Stress-test: försöker tvinga ner radantalet mot 2100 för att se var facit börjar dö.',
        'base': 'RADPRESS',
        'block_delta_individual': True,
        'block_delta_groups': True,
        'soft_favorite_max0': True,
        'green_name_bonus': True,
        'v22_mode': 'strict2100',
        'v22_min_candidate_red_pct': 0.8,
        'v22_target_rows': 2050,
        'v22_budget_rows': 2150,
        'v22_prefer_groups': True,
        'v22_penalize_low_step': True,
        'v23_strict_budget': True,
        'overrides': {
            'min_hit': 23,
            'candidate_min_hit': 26,
            'beam_width': 72,
            'archive_width': 700,
            'min_unique_rows': 20,
            'min_package_val_pct': 65.0,
            'min_candidate_val_pct': 65.0,
            'reduction_power': 6.4,
            'validation_power': 0.20,
            'hit_power': 0.65,
            'payout_direction_weight': 0.02,
            'v15_group_max_filters': 28,
            'v15_max_group_candidates': 160,
            'v15_cross_per_family': 4,
            'v15_cross_max_filters': 28,
            'bundle_pool_size': 18,
            'max_bundle_trials_per_state': 180,
            'max_bundle_keep_per_state': 24,
        },
    },
}

# Registrera V23-varianter i V15/V19/V13-pipelinen.
for _v23_key, _v23_cfg in V23_VARIANT_DEFS.items():
    if _v23_key == 'V21A':
        if 'V21A' in V15_VARIANT_DEFS:
            V15_VARIANT_DEFS['V21A']['v23_cfg'] = {'strict_budget': False, 'budget_rows': 2500, 'target_rows': 2350}
        continue
    _v21_clone_variant(_v23_cfg.get('base', 'RADPRESS'), _v23_key, _v23_cfg)
    v22_cfg = {
        'mode': _v23_cfg.get('v22_mode', ''),
        'target_rows': int(_v23_cfg.get('v22_target_rows') or 2500),
        'budget_rows': int(_v23_cfg.get('v22_budget_rows') or _v23_cfg.get('v22_target_rows') or 2500),
        'min_candidate_red_pct': float(_v23_cfg.get('v22_min_candidate_red_pct') or 0.0),
        'prefer_groups': bool(_v23_cfg.get('v22_prefer_groups', False)),
        'penalize_low_step': bool(_v23_cfg.get('v22_penalize_low_step', False)),
    }
    v23_cfg = {
        'strict_budget': bool(_v23_cfg.get('v23_strict_budget', False)),
        'target_rows': v22_cfg['target_rows'],
        'budget_rows': v22_cfg['budget_rows'],
    }
    V15_VARIANT_DEFS[_v23_key]['v22_cfg'] = v22_cfg
    V15_VARIANT_DEFS[_v23_key]['v23_cfg'] = v23_cfg
    V19_VARIANT_DEFS[_v23_key] = V15_VARIANT_DEFS[_v23_key]
    V13_VARIANT_DEFS[_v23_key] = {'label': _v23_cfg['label'], 'fixed_payout': True, 'bundle_search': True, 'description': _v23_cfg['description']}

_ORIG_V23_TRANSFORM_CANDIDATES = _v15_transform_candidates

def _v15_transform_candidates(candidates, htot, vtot, ftot, hist_payout, args, variant_id):  # noqa: F811
    final, group_cands = _ORIG_V23_TRANSFORM_CANDIDATES(candidates, htot, vtot, ftot, hist_payout, args, variant_id)
    vid = str(variant_id).upper()
    cfg = (V15_VARIANT_DEFS.get(vid, {}) or {}).get('v23_cfg') or {}
    if not cfg or not bool(cfg.get('strict_budget', False)):
        return final, group_cands
    # V23: släpp fram fler budgetkandidater men behåll riskguard mot kända mördare.
    # Lägre tröskel här betyder inte att alla blir valda; beam-sorteringen väljer sedan.
    def ok(c):
        if _v22_is_low_value_risk_candidate(c):
            return False
        try:
            red = float(c.get('red_pct', 0.0) or 0.0)
            val = float(c.get('val_pct', 100.0) or 100.0)
        except Exception:
            red, val = 0.0, 100.0
        if bool(c.get('is_v15_group')):
            return red >= 0.4 and val >= 60.0
        return red >= 0.6 and val >= 60.0
    f = [c for c in final if ok(c)]
    g = [c for c in group_cands if ok(c)]
    def srt(c):
        name = str(c.get('name','')).lower()
        fam = _v19_family_of_candidate(c)
        green = 0
        if any(x in name for x in ['fat summa','abc summa','ai-rank','poängfilter','rank summa','topp 6 favoriter']): green += 2
        if fam in ['FAT','ABC','Poäng/rank','Favorit','Skräll']: green += 1
        grp = 1 if bool(c.get('is_v15_group')) else 0
        return (green, grp, float(c.get('red_pct',0.0)), int(c.get('hist_hit',0)), float(c.get('val_pct',0.0)), -int(c.get('frame_keep',10**12)))
    f = sorted(f, key=srt, reverse=True)[:220]
    g = sorted(g, key=srt, reverse=True)[:220]
    return f, g

_ORIG_V23_STATE_SORT_KEY = _v13_state_sort_key

def _v13_state_sort_key(st, payout_values, args, variant_id):  # noqa: F811
    vid = str(variant_id).upper()
    cfg = (V15_VARIANT_DEFS.get(vid, {}) or {}).get('v23_cfg') or {}
    if not cfg or not bool(cfg.get('strict_budget', False)):
        return _ORIG_V23_STATE_SORT_KEY(st, payout_values, args, variant_id)
    rows = int(st.frame_count)
    budget = int(cfg.get('budget_rows') or 2500)
    target = int(cfg.get('target_rows') or budget)
    chosen = list(st.chosen or tuple())
    group_count = sum(1 for c in chosen if bool(c.get('is_v15_group')))
    # Hård budgetsort: inom tillåtna hist/val-golv prioriteras radantal mycket mer.
    # Om inget paket når budget ska den åtminstone leta lägsta möjliga rader, inte stanna vid trygg 3k.
    under_budget = 1 if rows <= budget else 0
    under_target = 1 if rows <= target else 0
    over = max(0, rows - budget)
    close_target = -abs(rows - target)
    # Extra skydd: 0/0 och teckenskydd hanteras redan i apply; här är bara ranking.
    return (
        under_budget,
        under_target,
        -over,
        -rows,
        int(st.hist_hit),
        int(st.val_hit),
        group_count,
        len(chosen),
    )


def _v23_parse_variants(value):
    default = ['V21A','STRICT2500','STRICT2300','STRICT2100']
    raw = str(value or ','.join(default)).replace(';', ',').replace('+', ',').split(',')
    allowed = set(default) | {'BASE','RADPRESS'}
    out = []
    for x in raw:
        k = x.strip().upper()
        if not k:
            continue
        if k not in allowed:
            continue
        if k not in out:
            out.append(k)
    if not any(k.startswith('STRICT') for k in out):
        out = default
    return out


def _v23_read_csv_auto(path: Path) -> pd.DataFrame:
    # _write_v15_outputs har varierat mellan semikolon och komma i olika runners.
    # Auto-detektera så ALL_FRAMES inte blir en enda hopklistrad kolumn.
    try:
        return pd.read_csv(path, sep=None, engine='python')
    except Exception:
        try:
            return pd.read_csv(path, sep=';')
        except Exception:
            return pd.read_csv(path)


def _v23_collect_combined_outputs(out_dir: Path, base_prefix: str, frames: List[str]) -> Dict[str, pd.DataFrame]:
    out = {}
    parts = []
    for frame in frames:
        p = Path(out_dir) / f'{base_prefix}_{_v17_safe_prefix_part(frame)}_summary.csv'
        if p.exists():
            try:
                df = _v23_read_csv_auto(p)
                df.insert(0, 'Ramprofil', frame)
                df.insert(1, 'Teoretisk grundram', _v17_frame_rows_count(frame))
                parts.append(df)
            except Exception as e:
                print(f'V23b: kunde inte läsa summary {p}: {e}', flush=True)
    if parts:
        out['summary'] = pd.concat(parts, ignore_index=True)
        out['summary'].to_csv(Path(out_dir) / f'{base_prefix}_ALL_FRAMES_summary.csv', sep=';', index=False)
    return out


def main_v23():
    parser = argparse.ArgumentParser(description='V23 Strict Budget Diagnostic: försöker visa om <2500/<2300 är möjligt med nuvarande kandidatpool.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_strict_budget_v23b_test10')
    parser.add_argument('--frames', default='3-5-5')
    parser.add_argument('--variants', default='V21A,STRICT2500,STRICT2300,STRICT2100')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--min-hit', type=int, default=24)
    parser.add_argument('--min-candidate-val-pct', type=float, default=68.0)
    parser.add_argument('--min-package-val-pct', type=float, default=68.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=90.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=90.0)
    parser.add_argument('--min-gap-score', type=float, default=0.55)
    parser.add_argument('--min-unique-rows', type=int, default=25)
    parser.add_argument('--filter-hist-target-pct', type=int, default=92)
    parser.add_argument('--frame-profile', default='3-5-5', help=argparse.SUPPRESS)
    parser.add_argument('--beam-width', type=int, default=72)
    parser.add_argument('--archive-width', type=int, default=650)
    parser.add_argument('--structure-seed-count', type=int, default=4)
    parser.add_argument('--variants-per-key', type=int, default=5)
    parser.add_argument('--max-candidates', type=int, default=240)
    parser.add_argument('--hit-power', type=float, default=0.75)
    parser.add_argument('--validation-power', type=float, default=0.25)
    parser.add_argument('--reduction-power', type=float, default=5.8)
    parser.add_argument('--payout-weight', type=float, default=0.0)
    parser.add_argument('--cluster-weight', type=float, default=0.05)
    parser.add_argument('--payout-direction-weight', type=float, default=0.04)
    parser.add_argument('--bundle-pool-size', type=int, default=18)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=160)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=22)
    parser.add_argument('--row-bucket-size', type=int, default=175)
    parser.add_argument('--per-bucket-keep', type=int, default=4)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--v15-group-max-filters', type=int, default=28)
    parser.add_argument('--v15-max-group-candidates', type=int, default=160)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=4)
    parser.add_argument('--v15-cross-max-filters', type=int, default=28)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    parser.add_argument('--force-upload', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-fresh-workers', action='store_true', default=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    frames = _v17_parse_frames(args.frames)
    variants = _v23_parse_variants(args.variants)
    args.variants = ','.join(variants)
    args.no_fresh_workers = True
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n' + '='*90, flush=True)
    print('TIPSET AI V23b – STRICT BUDGET DIAGNOSTIC', flush=True)
    print('Budget är hård ranking här: V23 ska visa om 2500/2300/2100 ens går att nå.', flush=True)
    print(f'Ramar: {", ".join(frames)}', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('='*90, flush=True)
    globals()['_v19_print_report'] = _v22_print_report
    for frame in frames:
        frame_args = argparse.Namespace(**vars(args))
        frame_args.frames = frame
        frame_args.frame_profile = frame
        frame_args.output_prefix = f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n' + '-'*90, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*90, flush=True)
        ns = _load_app_functions(app_file, fast_no_supermakro=False)
        db = ns['load_database'](str(db_file), 13)
        detail, meta = _run_backtest_v15(ns, db, frame_args)
        _write_v15_outputs(detail, frame_args, out_dir, app_file, db_file, worker_count=0)
    combined = _v23_collect_combined_outputs(out_dir, args.output_prefix, frames)
    if combined.get('summary') is not None:
        print('\nKLART – V23b STRICT BUDGET DIAGNOSTIC', flush=True)
        print(combined['summary'].to_string(index=False), flush=True)
        print('Summary:', str(Path(out_dir) / f'{args.output_prefix}_ALL_FRAMES_summary.csv'), flush=True)


# =============================================================================
# V24 – TRUE STACK + PRUNE / CONSTRUCT THEN REPAIR
# =============================================================================
# Denna version gör det Niklas beskrev mer bokstavligt:
# 1) Bygg först en aggressiv stack av alla starka filter som ger verklig MARGINALreducering.
# 2) När paketet är för hårt: reparera genom att ta bort det filter som ger mest träff tillbaka
#    för minst radkostnad.
# 3) Försök därefter återpressa rader med kvarvarande filter utan att falla under träffgolvet.
#
# Viktigt: detta är en diagnos/utvecklingsrunner. Den använder inte facit i valet; facit används först
# efteråt för backtest, precis som tidigare.

V24_VARIANT_DEFS = {
    'V21A': V21_VARIANT_DEFS['V21A'],
    'STACK5_PRUNE27': {
        'label': 'STACK5→PRUNE27 alla 5%-filter, reparera till 27/30',
        'description': 'Bygger ett överhårt paket av filter med minst 5 % marginalnytta, reparerar sedan till 27/30.',
        'base': 'RADPRESS',
        'target_hist': 27,
        'raw_hist_floor': 18,
        'target_val_pct': 76.0,
        'raw_val_pct': 56.0,
        'min_individual_hit': 28,
        'min_val_pct': 70.0,
        'construct_step_pct': 5.0,
        'repress_step_pct': 2.0,
        'budget_rows': 2500,
        'prefer_groups': False,
        'block_delta_hard': True,
        'max_construct_filters': 40,
        'overrides': {
            'min_hit': 24, 'candidate_min_hit': 28,
            'beam_width': 20, 'archive_width': 160,
            'min_unique_rows': 20,
            'min_candidate_val_pct': 70.0,
            'min_package_val_pct': 70.0,
            'v15_group_max_filters': 24,
            'v15_max_group_candidates': 150,
            'v15_cross_per_family': 4,
            'v15_cross_max_filters': 24,
            'max_candidates': 260,
            'variants_per_key': 5,
        },
    },
    'STACK5_PRUNE28': {
        'label': 'STACK5→PRUNE28 alla 5%-filter, reparera till 28/30',
        'description': 'Samma som STACK5_PRUNE27 men reparerar hårdare till 28/30.',
        'base': 'RADPRESS',
        'target_hist': 28,
        'raw_hist_floor': 18,
        'target_val_pct': 78.0,
        'raw_val_pct': 56.0,
        'min_individual_hit': 28,
        'min_val_pct': 70.0,
        'construct_step_pct': 5.0,
        'repress_step_pct': 2.0,
        'budget_rows': 2500,
        'prefer_groups': False,
        'block_delta_hard': True,
        'max_construct_filters': 40,
        'overrides': {
            'min_hit': 24, 'candidate_min_hit': 28,
            'beam_width': 20, 'archive_width': 160,
            'min_unique_rows': 20,
            'min_candidate_val_pct': 70.0,
            'min_package_val_pct': 70.0,
            'v15_group_max_filters': 24,
            'v15_max_group_candidates': 150,
            'v15_cross_per_family': 4,
            'v15_cross_max_filters': 24,
            'max_candidates': 260,
            'variants_per_key': 5,
        },
    },
    'STACK3_PRUNE27': {
        'label': 'STACK3→PRUNE27 alla 3%-filter, reparera till 27/30',
        'description': 'Mer aggressiv bank: marginalkrav 3 %, sedan reparation till 27/30.',
        'base': 'RADPRESS',
        'target_hist': 27,
        'raw_hist_floor': 17,
        'target_val_pct': 75.0,
        'raw_val_pct': 54.0,
        'min_individual_hit': 28,
        'min_val_pct': 68.0,
        'construct_step_pct': 3.0,
        'repress_step_pct': 1.5,
        'budget_rows': 2500,
        'prefer_groups': False,
        'block_delta_hard': True,
        'max_construct_filters': 45,
        'overrides': {
            'min_hit': 23, 'candidate_min_hit': 28,
            'beam_width': 20, 'archive_width': 180,
            'min_unique_rows': 15,
            'min_candidate_val_pct': 68.0,
            'min_package_val_pct': 68.0,
            'v15_group_max_filters': 26,
            'v15_max_group_candidates': 170,
            'v15_cross_per_family': 4,
            'v15_cross_max_filters': 26,
            'max_candidates': 280,
            'variants_per_key': 5,
        },
    },
    'GROUPSTACK5_PRUNE27': {
        'label': 'GROUPSTACK5→PRUNE27 gruppförst + reparera',
        'description': 'Bygger stacken med gruppkandidater högre prioriterade; målet är samma reducering men färre hårda singlar.',
        'base': 'RADPRESS',
        'target_hist': 27,
        'raw_hist_floor': 18,
        'target_val_pct': 77.0,
        'raw_val_pct': 56.0,
        'min_individual_hit': 28,
        'min_val_pct': 70.0,
        'construct_step_pct': 5.0,
        'repress_step_pct': 2.0,
        'budget_rows': 2500,
        'prefer_groups': True,
        'block_delta_hard': True,
        'max_construct_filters': 40,
        'overrides': {
            'min_hit': 24, 'candidate_min_hit': 28,
            'beam_width': 20, 'archive_width': 180,
            'min_unique_rows': 20,
            'min_candidate_val_pct': 70.0,
            'min_package_val_pct': 70.0,
            'v15_group_max_filters': 28,
            'v15_max_group_candidates': 190,
            'v15_cross_per_family': 5,
            'v15_cross_max_filters': 28,
            'max_candidates': 280,
            'variants_per_key': 5,
        },
    },
    'STACK5_PRUNE29': {
        'label': 'STACK5→PRUNE29 säkrare reparation',
        'description': 'Aggressiv stack men reparerar tillbaka till 29/30 för att se radkostnad för hög säkerhet.',
        'base': 'RADPRESS',
        'target_hist': 29,
        'raw_hist_floor': 18,
        'target_val_pct': 80.0,
        'raw_val_pct': 56.0,
        'min_individual_hit': 28,
        'min_val_pct': 70.0,
        'construct_step_pct': 5.0,
        'repress_step_pct': 2.0,
        'budget_rows': 2800,
        'prefer_groups': True,
        'block_delta_hard': True,
        'max_construct_filters': 40,
        'overrides': {
            'min_hit': 24, 'candidate_min_hit': 28,
            'beam_width': 20, 'archive_width': 160,
            'min_unique_rows': 20,
            'min_candidate_val_pct': 70.0,
            'min_package_val_pct': 70.0,
            'v15_group_max_filters': 24,
            'v15_max_group_candidates': 150,
            'v15_cross_per_family': 4,
            'v15_cross_max_filters': 24,
            'max_candidates': 260,
            'variants_per_key': 5,
        },
    },
}

# Registrera V24-varianter i befintlig pipeline.
for _v24_key, _v24_cfg in V24_VARIANT_DEFS.items():
    if _v24_key == 'V21A':
        if 'V21A' in V15_VARIANT_DEFS:
            V15_VARIANT_DEFS['V21A']['v24_cfg'] = {'control': True}
        continue
    _v21_clone_variant(_v24_cfg.get('base', 'RADPRESS'), _v24_key, _v24_cfg)
    V15_VARIANT_DEFS[_v24_key]['v24_cfg'] = {k: v for k, v in _v24_cfg.items() if k not in {'overrides','label','description','base'}}
    V19_VARIANT_DEFS[_v24_key] = V15_VARIANT_DEFS[_v24_key]
    V13_VARIANT_DEFS[_v24_key] = {'label': _v24_cfg['label'], 'fixed_payout': True, 'bundle_search': True, 'description': _v24_cfg['description']}


def _v24_is_stack_variant(variant_id: str) -> bool:
    return str(variant_id).upper() in {k for k in V24_VARIANT_DEFS if k != 'V21A'}


def _v24_candidate_family_priority(c: dict) -> int:
    fam = _v19_family_of_candidate(c)
    name = str(c.get('name','')).lower()
    pri = 0
    if fam in ['FAT', 'ABC', 'Poäng/rank']:
        pri += 4
    if fam in ['Favorit', 'Skräll']:
        pri += 3
    if fam in ['Värde/svårighet']:
        pri += 2
    if fam == 'Struktur':
        pri += 1
    if any(x in name for x in ['fat summa','abc summa','ai-rank','poängfilter','rank summa','topp 6 favoriter']):
        pri += 3
    if 'delta / avvikelse' in name:
        pri -= 8
    if 'fat 2-sekvenser' in name or 'sekvenser' in name:
        pri -= 1
    return pri


def _v24_is_allowed_candidate(c: dict, cfg: dict) -> bool:
    if not c or not c.get('key'):
        return False
    name = str(c.get('name','')).lower()
    fam = _v19_family_of_candidate(c)
    # Känd farlig hård variant från audit. Gruppversioner kan tillåtas om de finns.
    if bool(cfg.get('block_delta_hard', True)) and 'delta / avvikelse' in name and not bool(c.get('is_v15_group')):
        return False
    # Lågnyttiga små riskregler från V20/V21-audit.
    if _v22_is_low_value_risk_candidate(c):
        return False
    hist = int(c.get('hist_hit', 0) or 0)
    valp = float(c.get('val_pct', 0.0) or 0.0)
    min_hit = int(cfg.get('min_individual_hit', 28))
    min_val = float(cfg.get('min_val_pct', 68.0))
    # Struktur får bara vara gruppskydd, inte hårda individuella strukturknivar.
    if fam == 'Struktur' and not bool(c.get('is_v15_group')):
        return False
    if bool(c.get('is_v15_group')):
        return hist >= max(26, min_hit-1) and valp >= max(60.0, min_val-8.0)
    return hist >= min_hit and valp >= min_val


def _v24_sort_candidates(cands: list, cfg: dict):
    prefer_groups = bool(cfg.get('prefer_groups', False))
    def key(c):
        group_bonus = 3 if (prefer_groups and bool(c.get('is_v15_group'))) else 0
        return (
            group_bonus,
            _v24_candidate_family_priority(c),
            float(c.get('red_pct', 0.0) or 0.0),
            int(c.get('hist_hit', 0) or 0),
            float(c.get('val_pct', 0.0) or 0.0),
            float(c.get('payout_direction_pct', c.get('payout_lift_pct', 0.0)) or 0.0),
            -int(c.get('frame_keep', 10**12) or 10**12),
        )
    return sorted(cands, key=key, reverse=True)


def _v24_apply_candidate(st: _V9State, cand: dict, *, sign_bits: tuple, hist_floor: int, val_floor_hit: int, min_step_rows: int, min_step_pct_current: float, phase: str) -> Optional[_V9State]:
    key = cand.get('key')
    if not key or key in st.used_keys:
        return None
    new_hist = int(st.hist_mask) & int(cand.get('hist_bits', 0))
    hist_hit = int(new_hist.bit_count())
    if hist_hit < int(hist_floor):
        return None
    new_val = int(st.val_mask) & int(cand.get('val_bits', 0)) if st.val_size > 0 else 0
    val_hit = int(new_val.bit_count()) if st.val_size > 0 else 0
    if st.val_size > 0 and val_hit < int(val_floor_hit):
        return None
    new_frame = int(st.frame_mask) & int(cand.get('frame_bits', 0))
    new_rows = int(new_frame.bit_count())
    removed = int(st.frame_count - new_rows)
    if new_rows <= 0:
        return None
    if removed < max(1, int(min_step_rows)):
        return None
    if 100.0 * removed / max(1, int(st.frame_count)) < float(min_step_pct_current):
        return None
    if sign_bits and any((new_frame & int(bits)) == 0 for bits in sign_bits):
        return None
    c2 = dict(cand)
    c2['step_removed_rows'] = int(removed)
    c2['step_red_pct'] = 100.0 * removed / max(1, st.frame_count)
    step = {
        'Filter': c2.get('name',''),
        'Kategori': c2.get('category',''),
        'Intervall': c2.get('interval_txt','-'),
        'Efter filter': int(new_rows),
        'Borttagna unika rader': int(removed),
        'Stegreducering %': round(float(c2['step_red_pct']), 3),
        'Samlad historikträff': f'{hist_hit}/{st.hist_size}',
        'Samlad validering': f'{val_hit}/{st.val_size}' if st.val_size else '-',
        'Spret-gap': round(float(c2.get('gap_score',0.0)),3),
        'Utdelningslift %': round(float(c2.get('payout_direction_pct', c2.get('payout_lift_pct', 0.0)) or 0.0),2),
        'Fas': phase,
    }
    return _V9State(
        hist_mask=new_hist, val_mask=new_val, frame_mask=new_frame,
        hist_size=st.hist_size, val_size=st.val_size, frame_size=st.frame_size,
        chosen=st.chosen + (c2,), used_keys=frozenset(set(st.used_keys) | {key}),
        steps=st.steps + (step,),
    )


def _v24_state_from_sequence(seq: list, initial: _V9State, sign_bits: tuple) -> Optional[_V9State]:
    st = initial
    for c in seq:
        ns = _v24_apply_candidate(st, c, sign_bits=sign_bits, hist_floor=0, val_floor_hit=0, min_step_rows=1, min_step_pct_current=0.0, phase='rebuild')
        if ns is None:
            return None
        st = ns
    return st


def _v24_construct_stack(initial: _V9State, candidates: list, cfg: dict, sign_bits: tuple) -> _V9State:
    st = initial
    raw_hist = int(cfg.get('raw_hist_floor', 18))
    raw_val = int(math.ceil(st.val_size * float(cfg.get('raw_val_pct', 55.0)) / 100.0)) if st.val_size else 0
    step_pct = float(cfg.get('construct_step_pct', 5.0))
    max_filters = int(cfg.get('max_construct_filters', 40))
    # Första kravet är 5 % av aktuell radmängd. Mot slutet får filtret fortfarande vara relevant.
    added = True
    while added and len(st.chosen) < max_filters:
        added = False
        best = None
        best_key = None
        for c in candidates:
            if c.get('key') in st.used_keys:
                continue
            ns = _v24_apply_candidate(st, c, sign_bits=sign_bits, hist_floor=raw_hist, val_floor_hit=raw_val,
                                      min_step_rows=1, min_step_pct_current=step_pct, phase='construct')
            if ns is None:
                continue
            removed = st.frame_count - ns.frame_count
            fam_pri = _v24_candidate_family_priority(c)
            # Välj störst verklig marginalnytta, med familjeprioritet som tie-break.
            cand_key = (removed, 100.0*removed/max(1,st.frame_count), fam_pri, int(c.get('hist_hit',0)), float(c.get('val_pct',0.0)))
            if best is None or cand_key > best_key:
                best = ns
                best_key = cand_key
        if best is not None:
            st = best
            added = True
            print(f'      V24 construct: filter={len(st.chosen)} · träff={st.hist_hit}/{st.hist_size} · val={st.val_hit}/{st.val_size if st.val_size else 0} · rader={st.frame_count}', flush=True)
    return st


def _v24_prune_to_target(st: _V9State, initial: _V9State, sign_bits: tuple, cfg: dict) -> _V9State:
    target_hist = int(cfg.get('target_hist', 27))
    target_val = int(math.ceil(st.val_size * float(cfg.get('target_val_pct', 76.0)) / 100.0)) if st.val_size else 0
    chosen = list(st.chosen)
    cur = st
    # Ta bort filter tills paketträff och validering når målet. Varje borttagning ökar träff/rader.
    guard = 0
    while chosen and (cur.hist_hit < target_hist or (cur.val_size and cur.val_hit < target_val)) and guard < 100:
        guard += 1
        options = []
        for i, c in enumerate(chosen):
            trial_seq = chosen[:i] + chosen[i+1:]
            trial = _v24_state_from_sequence(trial_seq, initial, sign_bits)
            if trial is None:
                continue
            hist_gain = trial.hist_hit - cur.hist_hit
            val_gain = trial.val_hit - cur.val_hit if cur.val_size else 0
            rows_added = trial.frame_count - cur.frame_count
            # Mest återvunnen träff per radkostnad. Straffa om borttagningen blåser upp radantalet stort.
            score = (hist_gain * 100000 + val_gain * 2500) - max(0, rows_added)
            # Om fortfarande under mål: bonus för att nå mål direkt.
            if trial.hist_hit >= target_hist and (not trial.val_size or trial.val_hit >= target_val):
                score += 50000
            # Om filtret är känt svagt/riskigt: bonus att ta bort.
            name = str(c.get('name','')).lower()
            if 'delta / avvikelse' in name or 'favorit max 0' in name:
                score += 25000
            options.append((score, -max(0, rows_added), i, trial))
        if not options:
            break
        options.sort(reverse=True)
        _, _, remove_i, trial = options[0]
        removed_filter = chosen[remove_i]
        print(f'      V24 prune: tar bort "{removed_filter.get("name","")}" → träff={trial.hist_hit}/{trial.hist_size} · val={trial.val_hit}/{trial.val_size if trial.val_size else 0} · rader={trial.frame_count}', flush=True)
        chosen = chosen[:remove_i] + chosen[remove_i+1:]
        cur = trial
    return cur


def _v24_repress_after_prune(st: _V9State, initial: _V9State, candidates: list, cfg: dict, sign_bits: tuple) -> _V9State:
    target_hist = int(cfg.get('target_hist', 27))
    target_val = int(math.ceil(st.val_size * float(cfg.get('target_val_pct', 76.0)) / 100.0)) if st.val_size else 0
    step_pct = float(cfg.get('repress_step_pct', 2.0))
    budget = int(cfg.get('budget_rows', 2500))
    cur = st
    # Lägg tillbaka/addera kandidater som fortfarande håller mål och ger marginalnytta.
    for _ in range(30):
        if cur.frame_count <= budget:
            break
        best = None
        best_key = None
        for c in candidates:
            if c.get('key') in cur.used_keys:
                continue
            ns = _v24_apply_candidate(cur, c, sign_bits=sign_bits, hist_floor=target_hist, val_floor_hit=target_val,
                                      min_step_rows=1, min_step_pct_current=step_pct, phase='repress')
            if ns is None:
                continue
            removed = cur.frame_count - ns.frame_count
            fam_pri = _v24_candidate_family_priority(c)
            key = (removed, fam_pri, ns.hist_hit, ns.val_hit)
            if best is None or key > best_key:
                best = ns
                best_key = key
        if best is None:
            break
        cur = best
        print(f'      V24 repress: filter={len(cur.chosen)} · träff={cur.hist_hit}/{cur.hist_size} · val={cur.val_hit}/{cur.val_size if cur.val_size else 0} · rader={cur.frame_count}', flush=True)
    return cur


_ORIG_V24_BUILD_PACKAGE = _build_cluster_payout_package_v13_from_candidates

def _build_cluster_payout_package_v13_from_candidates(ns: dict, candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, frame_rows: list, frame: list, antal_matcher: int, args, variant_id: str):  # noqa: F811
    vid = str(variant_id).upper()
    if not _v24_is_stack_variant(vid):
        return _ORIG_V24_BUILD_PACKAGE(ns, candidates, htot, vtot, ftot, hist_payout, frame_rows, frame, antal_matcher, args, variant_id)
    cfg = (V15_VARIANT_DEFS.get(vid, {}) or {}).get('v24_cfg') or {}
    raw_row_matrix = ns['_frame_row_matrix'](frame_rows, antal_matcher)
    sign_bits = _build_teckenskydd_bits(ns, raw_row_matrix, frame, antal_matcher)
    initial = _V9State(
        hist_mask=(1 << htot)-1,
        val_mask=(1 << vtot)-1 if vtot else 0,
        frame_mask=(1 << ftot)-1,
        hist_size=htot, val_size=vtot, frame_size=ftot,
        chosen=tuple(), used_keys=frozenset(), steps=tuple(),
    )
    bank = [c for c in candidates if _v24_is_allowed_candidate(c, cfg)]
    bank = _v24_sort_candidates(bank, cfg)
    # Dedupa per nyckel men tillåt grupp/singel av olika nycklar.
    seen = set(); deduped = []
    for c in bank:
        sig = (c.get('key'), c.get('interval_txt'))
        if sig in seen:
            continue
        seen.add(sig); deduped.append(c)
    bank = deduped[:260]
    if not bank:
        return None, {'error': f'V24 bank tom för {vid}', 'candidates': len(candidates)}
    print(f'    V24 {vid}: kandidatbank={len(bank)} · construct först, prune sen', flush=True)
    raw = _v24_construct_stack(initial, bank, cfg, sign_bits)
    pruned = _v24_prune_to_target(raw, initial, sign_bits, cfg)
    final = _v24_repress_after_prune(pruned, initial, bank, cfg, sign_bits)
    target_hist = int(cfg.get('target_hist', 27))
    target_val = int(math.ceil(vtot * float(cfg.get('target_val_pct', 76.0)) / 100.0)) if vtot else 0
    # Cleanup endast om exakt samma radmask bibehålls.
    final = _v9_redundancy_cleanup(final, hist_floor=min(final.hist_hit, target_hist), val_floor_hit=min(final.val_hit, target_val), sign_bits=sign_bits)
    metrics = _v13_state_metrics(final, hist_payout, args, variant_id)
    package = {
        'variant': vid,
        'variant_label': V13_VARIANT_DEFS.get(vid, {}).get('label', vid),
        'target': int(final.hist_hit),
        'target_label': f'V24 {vid} {final.hist_hit}/{htot}',
        'hist_hit': int(final.hist_hit), 'hist_total': int(htot),
        'val_hit': int(final.val_hit), 'val_total': int(vtot),
        'frame_start': int(ftot), 'frame_after': int(final.frame_count),
        'reduction_pct': float(metrics['reduction_pct']),
        'joint_score': float(metrics['joint_score']),
        'core_score': float(metrics['core_score']),
        'payout_lift_pct': float(metrics.get('payout_lift_pct', 0.0)),
        'payout_direction_pct': float(metrics.get('payout_direction_pct', metrics.get('payout_lift_pct', 0.0))),
        'removed_hist_count': int(metrics.get('removed_hist_count', 0)),
        'removed_payout_mean': float(metrics.get('removed_payout_mean', 0.0)),
        'cluster_mean': float(metrics['cluster_mean']),
        'num_filters': len(final.chosen),
        'filters': list(final.chosen), 'steps': list(final.steps),
        'package_type': f'V24 {vid} – construct/prune',
        'structure_filters': final.structure_filters,
        'profile_filters': final.profile_filters,
        'fat_filters': final.fat_filters,
        'edge_filters': final.edge_filters,
        'meta': {
            'engine': 'v24_true_stack_prune_groupify',
            'variant': vid,
            'candidate_bank': len(bank),
            'raw_construct_filters': len(raw.chosen),
            'raw_construct_rows': raw.frame_count,
            'raw_construct_hist': raw.hist_hit,
            'raw_construct_val': raw.val_hit,
            'final_filters': len(final.chosen),
            'target_hist': target_hist,
            'target_val': target_val,
            'budget_rows': int(cfg.get('budget_rows', 2500)),
            'construct_step_pct': float(cfg.get('construct_step_pct', 5.0)),
        },
    }
    return package, package['meta']


def _v24_parse_variants(value):
    default = ['V21A','STACK5_PRUNE27','STACK5_PRUNE28','STACK3_PRUNE27','GROUPSTACK5_PRUNE27','STACK5_PRUNE29']
    raw = str(value or ','.join(default)).replace(';', ',').replace('+', ',').split(',')
    allowed = set(default) | {'BASE','RADPRESS'}
    out = []
    for x in raw:
        k = x.strip().upper()
        if not k:
            continue
        # Om gammalt V21/V22-argument råkar ligga kvar: ignorera, kör V24-default.
        if k not in allowed:
            continue
        if k not in out:
            out.append(k)
    if not any(k.startswith('STACK') or k.startswith('GROUPSTACK') for k in out):
        out = default
    return out


def _v24_collect_combined_outputs(out_dir: Path, base_prefix: str, frames: List[str]) -> Dict[str, pd.DataFrame]:
    return _v23_collect_combined_outputs(out_dir, base_prefix, frames)


def _v24_print_report(out_dir: Path, base_prefix: str, app_file: Path, db_file: Path, variants: str):
    lines = []
    lines.append('TIPSET AI V24 – TRUE STACK + PRUNE')
    lines.append('Bygger först aggressiv filterstack av alla marginalstarka filter, reparerar sedan genom att ta bort dyra riskfilter.')
    lines.append(f'Appfil: {app_file}')
    lines.append(f'Databas: {db_file}')
    lines.append(f'Varianter: {variants}')
    lines.append('')
    for p in sorted(Path(out_dir).glob(f'{base_prefix}_*_summary.csv')):
        if 'ALL_FRAMES' in p.name:
            continue
        try:
            df = _v23_read_csv_auto(p)
            lines.append(f'--- {p.name} ---')
            cols = [c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Median reducering %','Median filter','Max filter','Vinnare'] if c in df.columns]
            lines.append(df[cols].to_string(index=False) if cols else df.head(20).to_string(index=False))
            lines.append('')
        except Exception as e:
            lines.append(f'Kunde inte läsa {p}: {e}')
    report_path = Path(out_dir) / f'{base_prefix}_V24_TRUE_STACK_PRUNE_REPORT.txt'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    return report_path


def main_v24():
    parser = argparse.ArgumentParser(description='V24 True Stack + Prune: lägg in alla bästa filter först, sålla sedan för högre träff.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_true_stack_prune_v24_test10')
    parser.add_argument('--frames', default='3-5-5')
    parser.add_argument('--variants', default='V21A,STACK5_PRUNE27,STACK5_PRUNE28,STACK3_PRUNE27,GROUPSTACK5_PRUNE27,STACK5_PRUNE29')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--min-hit', type=int, default=24)
    parser.add_argument('--min-candidate-val-pct', type=float, default=68.0)
    parser.add_argument('--min-package-val-pct', type=float, default=68.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=90.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=90.0)
    parser.add_argument('--min-gap-score', type=float, default=0.55)
    parser.add_argument('--min-unique-rows', type=int, default=15)
    parser.add_argument('--filter-hist-target-pct', type=int, default=92)
    parser.add_argument('--frame-profile', default='3-5-5', help=argparse.SUPPRESS)
    parser.add_argument('--beam-width', type=int, default=20)
    parser.add_argument('--archive-width', type=int, default=180)
    parser.add_argument('--structure-seed-count', type=int, default=2)
    parser.add_argument('--variants-per-key', type=int, default=5)
    parser.add_argument('--max-candidates', type=int, default=280)
    parser.add_argument('--hit-power', type=float, default=0.75)
    parser.add_argument('--validation-power', type=float, default=0.25)
    parser.add_argument('--reduction-power', type=float, default=4.8)
    parser.add_argument('--payout-weight', type=float, default=0.0)
    parser.add_argument('--cluster-weight', type=float, default=0.05)
    parser.add_argument('--payout-direction-weight', type=float, default=0.04)
    parser.add_argument('--bundle-pool-size', type=int, default=12)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=80)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=175)
    parser.add_argument('--per-bucket-keep', type=int, default=4)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--v15-group-max-filters', type=int, default=28)
    parser.add_argument('--v15-max-group-candidates', type=int, default=190)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=5)
    parser.add_argument('--v15-cross-max-filters', type=int, default=28)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    parser.add_argument('--force-upload', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-fresh-workers', action='store_true', default=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    frames = _v17_parse_frames(args.frames)
    variants = _v24_parse_variants(args.variants)
    args.variants = ','.join(variants)
    args.no_fresh_workers = True
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n' + '='*94, flush=True)
    print('TIPSET AI V24 – TRUE STACK + PRUNE', flush=True)
    print('LÅST LOGIK: construct först med alla marginalstarka filter → prune/repair sedan.', flush=True)
    print(f'Ramar: {", ".join(frames)}', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('='*94, flush=True)
    globals()['_v19_print_report'] = _v22_print_report
    for frame in frames:
        frame_args = argparse.Namespace(**vars(args))
        frame_args.frames = frame
        frame_args.frame_profile = frame
        frame_args.output_prefix = f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n' + '-'*94, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*94, flush=True)
        ns = _load_app_functions(app_file, fast_no_supermakro=False)
        db = ns['load_database'](str(db_file), 13)
        detail, meta = _run_backtest_v15(ns, db, frame_args)
        _write_v15_outputs(detail, frame_args, out_dir, app_file, db_file, worker_count=0)
    combined = _v24_collect_combined_outputs(out_dir, args.output_prefix, frames)
    report_path = _v24_print_report(out_dir, args.output_prefix, app_file, db_file, args.variants)
    if combined.get('summary') is not None:
        print('\nKLART – V24 TRUE STACK + PRUNE', flush=True)
        print(combined['summary'].to_string(index=False), flush=True)
        print('Summary:', str(Path(out_dir) / f'{args.output_prefix}_ALL_FRAMES_summary.csv'), flush=True)
        print('V24-rapport:', str(report_path), flush=True)


# =============================================================================
# V25 – ROW SCORE BUDGET: filter som röster, inte veto
# =============================================================================
# Syfte: bryta 3k-platån. I stället för hårda OCH-filter får varje kandidat
# poängsätta varje rad. Slutpaketet är exakt TOP N rader efter score.
# Facit används först efteråt i backtestet.

V25_VARIANT_DEFS = {
    'V21A': {
        'label': 'V21A kontroll hårt paket',
        'description': 'Kontroll från V21A/RADPRESS utan hård Delta.',
        'mode': 'hard_control',
        'top_rows': None,
    },
    'TOP2500': {
        'label': 'TOP2500 radscore budget',
        'description': 'Alla starka filter röstar, välj topp 2500 rader.',
        'mode': 'row_score', 'top_rows': 2500,
        'min_hit': 28, 'min_val_pct': 72.0, 'min_red_pct': 3.0,
        'family_balance': False, 'safe': False, 'block_delta': False,
    },
    'TOP2300': {
        'label': 'TOP2300 radscore budget',
        'description': 'Alla starka filter röstar, välj topp 2300 rader.',
        'mode': 'row_score', 'top_rows': 2300,
        'min_hit': 28, 'min_val_pct': 72.0, 'min_red_pct': 3.0,
        'family_balance': False, 'safe': False, 'block_delta': False,
    },
    'TOP2100': {
        'label': 'TOP2100 radscore budget',
        'description': 'Alla starka filter röstar, välj topp 2100 rader.',
        'mode': 'row_score', 'top_rows': 2100,
        'min_hit': 28, 'min_val_pct': 72.0, 'min_red_pct': 3.0,
        'family_balance': False, 'safe': False, 'block_delta': False,
    },
    'TOP1800': {
        'label': 'TOP1800 radscore stress',
        'description': 'Alla starka filter röstar, välj topp 1800 rader.',
        'mode': 'row_score', 'top_rows': 1800,
        'min_hit': 28, 'min_val_pct': 72.0, 'min_red_pct': 3.0,
        'family_balance': False, 'safe': False, 'block_delta': False,
    },

    'SOFT2500': {
        'label': 'SOFT2500 mjuk radscore',
        'description': 'Topp 2500 med delpoäng för nära kantmissar i stället för binära veto.',
        'mode': 'row_score', 'top_rows': 2500,
        'min_hit': 28, 'min_val_pct': 72.0, 'min_red_pct': 3.0,
        'family_balance': False, 'safe': False, 'block_delta': False, 'soft_distance': True,
    },
    'SOFT2300': {
        'label': 'SOFT2300 mjuk radscore',
        'description': 'Topp 2300 med delpoäng för nära kantmissar.',
        'mode': 'row_score', 'top_rows': 2300,
        'min_hit': 28, 'min_val_pct': 72.0, 'min_red_pct': 3.0,
        'family_balance': False, 'safe': False, 'block_delta': False, 'soft_distance': True,
    },
    'SOFT2100': {
        'label': 'SOFT2100 mjuk radscore',
        'description': 'Topp 2100 med delpoäng för nära kantmissar.',
        'mode': 'row_score', 'top_rows': 2100,
        'min_hit': 28, 'min_val_pct': 72.0, 'min_red_pct': 3.0,
        'family_balance': False, 'safe': False, 'block_delta': False, 'soft_distance': True,
    },

    'TOP2500_SAFE': {
        'label': 'TOP2500 safe radscore',
        'description': 'Topp 2500 men starkare validering och lägre riskfiltervikt.',
        'mode': 'row_score', 'top_rows': 2500,
        'min_hit': 29, 'min_val_pct': 78.0, 'min_red_pct': 3.0,
        'family_balance': True, 'safe': True, 'block_delta': False,
    },
    'TOP2300_SAFE': {
        'label': 'TOP2300 safe radscore',
        'description': 'Topp 2300 med familjebalans och starkare säkerhet.',
        'mode': 'row_score', 'top_rows': 2300,
        'min_hit': 29, 'min_val_pct': 78.0, 'min_red_pct': 3.0,
        'family_balance': True, 'safe': True, 'block_delta': False,
    },
    'TOP2500_NODELTA': {
        'label': 'TOP2500 utan Delta',
        'description': 'Topp 2500 där Delta / Avvikelse inte får rösta alls.',
        'mode': 'row_score', 'top_rows': 2500,
        'min_hit': 28, 'min_val_pct': 72.0, 'min_red_pct': 3.0,
        'family_balance': False, 'safe': False, 'block_delta': True,
    },
}


def _v25_parse_variants(value):
    default = ['V21A','SOFT2500','SOFT2300','SOFT2100','TOP2500','TOP2300','TOP2100','TOP1800','TOP2500_SAFE','TOP2300_SAFE','TOP2500_NODELTA']
    allowed = set(V25_VARIANT_DEFS)
    raw = str(value or ','.join(default)).replace(';', ',').replace('+', ',').split(',')
    out = []
    for x in raw:
        k = x.strip().upper()
        if not k:
            continue
        # Mappa gamla varianter till relevant kontroll i stället för att krascha.
        if k in {'BASE','RADPRESS','V21B','V21C','V21D'}:
            k = 'V21A'
        if k not in allowed:
            print(f'V25: ignorerar okänd variant {k}', flush=True)
            continue
        if k not in out:
            out.append(k)
    return out or default


def _v25_family(c):
    try:
        return _v20_filter_family(str(c.get('name','')), str(c.get('category','')), str(c.get('key','')))
    except Exception:
        try:
            return _v15_group_family(c)
        except Exception:
            return str(c.get('category','') or 'Övrigt')


def _v25_is_delta(c):
    name = str(c.get('name','')).lower()
    key = str(c.get('key','')).lower()
    return ('delta' in name and 'avvik' in name) or ('delta' in key and 'avvik' in key)


def _v25_candidate_allowed(c, cfg):
    if not c.get('key'):
        return False
    if _is_blocked_auto_candidate(c):
        return False
    if bool(cfg.get('block_delta')) and _v25_is_delta(c):
        return False
    if bool(cfg.get('safe')) and _v25_is_delta(c):
        # Safe-varianterna får fortfarande använda Delta, men mycket svagare; låt inte bortfiltrera här.
        pass
    if int(c.get('hist_hit', 0)) < int(cfg.get('min_hit', 28)):
        return False
    if float(c.get('val_pct', 100.0)) + 1e-9 < float(cfg.get('min_val_pct', 72.0)):
        return False
    if float(c.get('red_pct', 0.0)) + 1e-9 < float(cfg.get('min_red_pct', 3.0)):
        return False
    # Hårda favoritgrupper med perfektkrav och låg radnytta är bara brus/risk.
    if _v21_is_bad_favorite_max0(c):
        return False
    return True


def _v25_weight(c, cfg, family_counts=None):
    fam = _v25_family(c)
    hist_frac = max(0.0, min(1.0, float(c.get('hist_hit', 0)) / max(1.0, float(c.get('hist_total', 30) or 30))))
    val_frac = max(0.0, min(1.0, float(c.get('val_pct', 100.0)) / 100.0))
    red_frac = max(0.0, min(0.98, float(c.get('red_pct', 0.0)) / 100.0))
    # Selectivity är viktig men får inte dominera så att riskfilter styr allt.
    w = (0.15 + math.sqrt(max(0.0, red_frac))) * (hist_frac ** 3.2) * (val_frac ** 1.6)
    # Familjeprioritet från audit: FAT/ABC/Poäng/rank/Favorit/Skräll ska få rösta tydligt.
    fam_bonus = {
        'FAT': 1.18,
        'ABC': 1.14,
        'Poäng/rank': 1.22,
        'Favorit': 1.10,
        'Skräll': 1.08,
        'Värde/svårighet': 0.92,
        'Struktur': 0.72,
        'FAT-sekvens': 0.72,
    }.get(fam, 1.0)
    w *= fam_bonus
    # Utdelningsriktning som liten bonus. Negativ riktning straffas mildare än veto.
    direction = float(c.get('payout_direction_pct', c.get('payout_lift_pct', 0.0)) or 0.0)
    if direction > 0:
        w *= (1.0 + min(0.20, direction / 400.0))
    elif direction < -20:
        w *= 0.94
    # Kända kant-/riskfilter ska inte dominera, men kan rösta i osäkra fall.
    if _v25_is_delta(c):
        w *= 0.35 if not bool(cfg.get('safe')) else 0.20
    if _v21_is_bad_favorite_max0(c):
        w *= 0.20
    if fam == 'Struktur' and not c.get('is_v15_group'):
        w *= 0.75
    if bool(cfg.get('family_balance')) and family_counts:
        w /= math.sqrt(max(1.0, float(family_counts.get(fam, 1))))
    return float(max(0.0001, w))


def _v25_bits_bool(bits, n):
    return _v15_bits_to_bool(int(bits), int(n))


def _v25_row_passes_candidate(ns, row, specs_by_key, c):
    if c.get('is_v15_group'):
        children = c.get('group_children', []) or []
        if not children:
            return False
        ok_count = sum(1 for ch in children if _v15_child_passes_row(ns, row, specs_by_key, ch))
        return ok_count >= int(c.get('group_required', len(children)))
    spec = specs_by_key.get(c.get('key'), c)
    interval = c.get('interval')
    try:
        getter = spec.get('getter')
        if getter is not None:
            v = getter(row)
            return float(v) >= float(interval[0]) and float(v) <= float(interval[1])
        return bool(ns.get('_spec_pass', lambda *_: True)(row, spec, interval))
    except Exception:
        try:
            return bool(ns.get('_spec_pass', lambda *_: True)(row, spec, interval))
        except Exception:
            return False



def _v25_candidate_row_multipliers(ns, frame_rows, specs_by_key, c, cfg):
    """Returnerar 0..1-multiplikator per rad. Soft-varianter ger delpoäng nära intervallkant."""
    n = len(frame_rows)
    bits = int(c.get('frame_bits', 0))
    hard = _v25_bits_bool(bits, n) if bits > 0 else np.zeros(n, dtype=bool)
    if not bool(cfg.get('soft_distance')) or c.get('is_v15_group'):
        return hard.astype(float)
    spec = specs_by_key.get(c.get('key'), c)
    getter = spec.get('getter') if isinstance(spec, dict) else None
    interval = c.get('interval')
    if getter is None or interval is None:
        return hard.astype(float)
    try:
        lo, hi = float(interval[0]), float(interval[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            return hard.astype(float)
        if lo > hi:
            lo, hi = hi, lo
    except Exception:
        return hard.astype(float)
    vals = np.empty(n, dtype=float)
    vals[:] = np.nan
    for i, row in enumerate(frame_rows):
        try:
            vals[i] = float(getter(row))
        except Exception:
            pass
    inside = np.isfinite(vals) & (vals >= lo) & (vals <= hi)
    mult = np.zeros(n, dtype=float)
    mult[inside] = 1.0
    width = max(1e-9, abs(hi - lo))
    # Kantbuffert: 15 % av intervallbredden, minst 3 enheter. Nära missar som 90 mot 95 får hög delpoäng.
    scale = max(3.0, width * 0.15)
    below = np.isfinite(vals) & (vals < lo)
    above = np.isfinite(vals) & (vals > hi)
    if below.any():
        mult[below] = np.exp(-(lo - vals[below]) / scale)
    if above.any():
        mult[above] = np.exp(-(vals[above] - hi) / scale)
    # Långt utanför ska inte få mycket poäng.
    mult[mult < 0.05] = 0.0
    return mult


def _v25_correct_multiplier(ns, correct, specs_by_key, c, cfg):
    ok = _v25_row_passes_candidate(ns, correct, specs_by_key, c)
    if ok:
        return 1.0
    if not bool(cfg.get('soft_distance')) or c.get('is_v15_group'):
        return 0.0
    spec = specs_by_key.get(c.get('key'), c)
    getter = spec.get('getter') if isinstance(spec, dict) else None
    interval = c.get('interval')
    if getter is None or interval is None:
        return 0.0
    try:
        lo, hi = float(interval[0]), float(interval[1])
        val = float(getter(correct))
        if lo > hi:
            lo, hi = hi, lo
        width = max(1e-9, abs(hi - lo))
        scale = max(3.0, width * 0.15)
        if val < lo:
            m = math.exp(-(lo - val) / scale)
        elif val > hi:
            m = math.exp(-(val - hi) / scale)
        else:
            m = 1.0
        return float(m if m >= 0.05 else 0.0)
    except Exception:
        return 0.0

def _v25_score_rows(ns, frame_rows, specs, candidates, cfg):
    n = len(frame_rows)
    scores = np.zeros(n, dtype=float)
    fam_scores = {}
    family_counts = {}
    for c in candidates:
        family_counts[_v25_family(c)] = family_counts.get(_v25_family(c), 0) + 1
    specs_by_key = {s.get('key'): s for s in specs or []}
    used = []
    for c in candidates:
        if not _v25_candidate_allowed(c, cfg):
            continue
        mult = _v25_candidate_row_multipliers(ns, frame_rows, specs_by_key, c, cfg)
        if mult.size != n or not np.any(mult > 0):
            continue
        w = _v25_weight(c, cfg, family_counts)
        scores += mult * w
        fam = _v25_family(c)
        if fam not in fam_scores:
            fam_scores[fam] = np.zeros(n, dtype=float)
        fam_scores[fam] += mult * w
        d = dict(c)
        d['_v25_weight'] = w
        d['_v25_family'] = fam
        used.append(d)
    return scores, fam_scores, used


def _v25_score_correct(ns, correct, specs, candidates, cfg):
    specs_by_key = {s.get('key'): s for s in specs or []}
    score = 0.0
    fam = {}
    passed = 0
    for c in candidates:
        if not _v25_candidate_allowed(c, cfg):
            continue
        mult = _v25_correct_multiplier(ns, correct, specs_by_key, c, cfg)
        if mult > 0:
            w = float(c.get('_v25_weight', _v25_weight(c, cfg, None)))
            score += w * float(mult)
            f = _v25_family(c)
            fam[f] = fam.get(f, 0.0) + w * float(mult)
            if mult >= 0.999999:
                passed += 1
    return float(score), fam, int(passed)


def _v25_build_row_score_package(ns, frame_rows, specs, candidates, htot, vtot, ftot, hist_payout, cfg, variant_id, correct=None):
    top_n = int(cfg.get('top_rows') or 2500)
    top_n = max(1, min(top_n, len(frame_rows)))
    scores, fam_scores, used = _v25_score_rows(ns, frame_rows, specs, candidates, cfg)
    if len(used) <= 0 or not np.isfinite(scores).any():
        return None, {'error': 'Inga V25-scorekandidater.'}
    # Tie-break: score först, sedan hög familybredd, sedan lägre index stabilt.
    family_breadth = np.zeros(len(frame_rows), dtype=float)
    for arr in fam_scores.values():
        family_breadth += (arr > 0).astype(float)
    order = np.lexsort((np.arange(len(frame_rows)), -family_breadth, -scores))
    chosen_idx = order[:top_n]
    frame_mask_bool = np.zeros(len(frame_rows), dtype=bool)
    frame_mask_bool[chosen_idx] = True
    frame_bits = _bool_mask_to_bits(frame_mask_bool)
    cutoff_score = float(scores[chosen_idx[-1]]) if len(chosen_idx) else 0.0
    selected_scores = scores[chosen_idx]
    # Historik/validering: scorea omgångarnas facitrader mot samma filterröster.
    hist_scores = []
    val_scores = []
    # För historik/val kan vi approximera TOP N-medlemskap med cutoffscore; facitranking mot exakt ram är bara för testfacit.
    # Detta är inte läckage: cutoff kommer från dagens ram och kandidater, inte från testfacit.
    specs_by_key = {s.get('key'): s for s in specs or []}
    # Vi har inte direkt historikraderna här; kandidaternas hist_bits anger pass per kandidat.
    # Beräkna historikscore genom summering av kandidatbits.
    h_scores = np.zeros(int(htot), dtype=float)
    v_scores = np.zeros(int(vtot), dtype=float) if int(vtot) > 0 else np.zeros(0, dtype=float)
    for c in used:
        w = float(c.get('_v25_weight', 0.0))
        if int(c.get('hist_bits', 0)):
            h_scores[_v25_bits_bool(int(c.get('hist_bits', 0)), int(htot))] += w
        if int(vtot) > 0 and int(c.get('val_bits', 0)):
            v_scores[_v25_bits_bool(int(c.get('val_bits', 0)), int(vtot))] += w
    hist_hit = int((h_scores >= cutoff_score - 1e-12).sum()) if len(h_scores) else 0
    val_hit = int((v_scores >= cutoff_score - 1e-12).sum()) if len(v_scores) else 0
    hist_mask_bool = (h_scores >= cutoff_score - 1e-12) if len(h_scores) else np.zeros(int(htot), dtype=bool)
    val_mask_bool = (v_scores >= cutoff_score - 1e-12) if len(v_scores) else np.zeros(int(vtot), dtype=bool)
    hist_bits = _bool_mask_to_bits(hist_mask_bool)
    val_bits = _bool_mask_to_bits(val_mask_bool) if int(vtot) > 0 else 0

    # Facitdiagnos/rank om korrekt rad finns i ramen.
    correct_in_frame = False
    correct_rank = None
    correct_score = None
    correct_selected = False
    correct_family_json = ''
    if correct:
        try:
            idx_map = {r: i for i, r in enumerate(frame_rows)}
            if correct in idx_map:
                ci = idx_map[correct]
                correct_in_frame = True
                correct_score = float(scores[ci])
                # rank 1 = bäst; exakt sorterad ordning.
                pos = np.where(order == ci)[0]
                correct_rank = int(pos[0]) + 1 if len(pos) else None
                correct_selected = bool(frame_mask_bool[ci])
                fam_payload = {fam: round(float(arr[ci]), 4) for fam, arr in fam_scores.items() if float(arr[ci]) > 0}
                correct_family_json = json.dumps(fam_payload, ensure_ascii=False)
            else:
                # Facit kan ligga utanför proxygrundramen eftersom dessa Colab-tester främst
                # utvärderar filter/radreducering, inte kupongens teckenval. För rowscore
                # gör vi därför samma typ av filterproxy som tidigare motorer: facit får en
                # score från filterrösterna och jämförs med TOP N-cutoffen i ramen.
                correct_score, fam_payload, passed = _v25_score_correct(ns, correct, specs, used, cfg)
                correct_rank = int((scores > float(correct_score) + 1e-12).sum()) + 1
                correct_selected = bool(correct_rank <= top_n)
                correct_family_json = json.dumps({k: round(v,4) for k,v in fam_payload.items()}, ensure_ascii=False)
        except Exception:
            pass

    family_weight_totals = []
    for fam, arr in fam_scores.items():
        family_weight_totals.append({'familj': fam, 'total_row_weight': round(float(arr.sum()), 3), 'median_row_weight': round(float(np.median(arr)), 4)})
    family_weight_totals.sort(key=lambda x: x['total_row_weight'], reverse=True)
    top_filters = sorted(used, key=lambda c: float(c.get('_v25_weight', 0.0)), reverse=True)[:40]
    filters_json = json.dumps([
        {
            'name': c.get('name'), 'category': c.get('category'), 'family': c.get('_v25_family'),
            'interval': c.get('interval_txt'), 'hist': f"{c.get('hist_hit')}/{c.get('hist_total')}",
            'val_pct': round(float(c.get('val_pct', 0.0)), 1),
            'red_pct': round(float(c.get('red_pct', 0.0)), 1),
            'weight': round(float(c.get('_v25_weight', 0.0)), 5),
            'group': bool(c.get('is_v15_group')),
        } for c in top_filters
    ], ensure_ascii=False)
    package = {
        'variant': str(variant_id),
        'variant_label': V25_VARIANT_DEFS[str(variant_id)]['label'],
        'target': int(top_n),
        'target_label': f'V25 {variant_id} TOP {top_n}',
        'hist_hit': int(hist_hit), 'hist_total': int(htot),
        'val_hit': int(val_hit), 'val_total': int(vtot),
        'frame_start': int(ftot), 'frame_after': int(top_n),
        'reduction_pct': 100.0 - 100.0 * top_n / max(1, int(ftot)),
        'joint_score': float(np.mean(selected_scores)) if len(selected_scores) else 0.0,
        'core_score': float(np.mean(selected_scores)) if len(selected_scores) else 0.0,
        'payout_lift_pct': 0.0,
        'payout_direction_pct': 0.0,
        'removed_hist_count': int(htot - hist_hit),
        'removed_payout_mean': 0.0,
        'cluster_mean': 0.0,
        'num_filters': int(len(used)),
        'filters': top_filters,
        'steps': [],
        'package_type': f'V25 {variant_id} – {V25_VARIANT_DEFS[str(variant_id)]["label"]}',
        'structure_filters': sum(1 for c in used if _v25_family(c) == 'Struktur'),
        'profile_filters': sum(1 for c in used if _v25_family(c) != 'Struktur'),
        'fat_filters': sum(1 for c in used if _v25_family(c) in {'FAT','ABC','FAT-sekvens'}),
        'edge_filters': sum(1 for c in used if _v25_family(c) in {'Värde/svårighet','Favorit','Skräll','Poäng/rank'}),
        'v25_meta': {
            'top_rows': int(top_n), 'cutoff_score': round(float(cutoff_score), 6),
            'score_min_selected': round(float(np.min(selected_scores)), 6) if len(selected_scores) else 0.0,
            'score_med_selected': round(float(np.median(selected_scores)), 6) if len(selected_scores) else 0.0,
            'score_max_selected': round(float(np.max(selected_scores)), 6) if len(selected_scores) else 0.0,
            'candidate_votes': int(len(used)),
            'correct_in_frame': bool(correct_in_frame),
            'correct_selected': bool(correct_selected),
            'correct_rank': correct_rank,
            'correct_score': correct_score,
            'correct_family_json': correct_family_json,
            'family_weight_json': json.dumps(family_weight_totals, ensure_ascii=False),
            'top_filters_json': filters_json,
        },
    }
    return package, package['v25_meta']


def _v25_run_backtest(ns: dict, global_db: pd.DataFrame, args):
    antal_matcher = 13
    norm = ns['normalize_single_row_text']
    similar_history = ns['_similar_history_for_backtest']
    ranked_frame = ns['_ranked_frame_like_widths']
    generate_rows_from_frame = ns['generate_rows_from_frame']
    build_clean_filter_specs = ns['build_clean_filter_specs']
    global _ACTIVE_V9_NS
    _ACTIVE_V9_NS = ns
    global_db, supersvar_count = _prepare_supersvar_payout(global_db, int(args.pay_max))
    db = _prepare_db(ns, global_db, antal_matcher, int(args.pay_min), int(args.pay_max))
    if db.empty:
        return pd.DataFrame(), {'error': 'Inga giltiga omgångar i utdelningsintervallet.'}
    variants = _v25_parse_variants(args.variants)
    spik, halv, hel = _parse_frame_profile(args.frame_profile)
    template = _base_frame(spik, halv, hel)
    test_rows = list(db.iterrows())[int(args.test_offset):int(args.test_offset)+int(args.max_tests)]
    out = []
    filter_rows = []
    for ti, (idx, test_row) in enumerate(test_rows, 1):
        _release_test_memory()
        t0_case = time.time()
        correct = norm(test_row.get('Correct_Row', ''))
        input_vec = test_row.get('Prob_Vector', [])
        test_date = test_row.get('Datum', None)
        payout = float(pd.to_numeric(pd.Series([test_row.get('Payout', 0)]), errors='coerce').fillna(0).iloc[0])
        print(f'V25 testfall {ti}/{len(test_rows)} · {str(test_date)[:10]} · utdelning {payout:,.0f} kr'.replace(',', ' '), flush=True)
        try:
            sim_df = similar_history(global_db, input_vec, antal_matcher, top_n=int(args.top_n), pay_min=int(args.pay_min), pay_max=int(args.pay_max), exclude_index=idx, mode=str(args.mode), test_date=test_date)
            wide_df = similar_history(global_db, input_vec, antal_matcher, top_n=max(int(args.top_n), int(args.wide_n)), pay_min=int(args.pay_min), pay_max=int(args.pay_max), exclude_index=idx, mode=str(args.mode), test_date=test_date)
            if len(sim_df) < int(args.top_n):
                raise RuntimeError(f'För få liknande omgångar: {len(sim_df)} av önskade {args.top_n}.')
            engine_frame = ranked_frame(template, input_vec, antal_matcher)
            frame_rows, _, ok, msg = generate_rows_from_frame(engine_frame, max_rows=None)
            if not ok or not frame_rows:
                raise RuntimeError(f'Kunde inte skapa ram: {msg}')
            specs = build_clean_filter_specs(sim_df, input_vec, antal_matcher, slider_u_count=3, target_hist_pct=int(args.filter_hist_target_pct), u_rows=None, hist_df=global_db, max_shock_pct=22, candidate_rows=frame_rows, include_supermakro=not bool(args.fast_no_supermakro))
            # Breda kandidater, inte försiktig score. Rowscore kan hantera att vissa filter är halvriskabla.
            candidates, htot, vtot, ftot, hist_payout = _build_dynamic_candidates_v9(
                ns, specs, sim_df, frame_rows, engine_frame, antal_matcher,
                profile_min_hit=int(args.candidate_min_hit),
                variants_per_key=int(args.variants_per_key),
                max_candidates=int(args.max_candidates),
                validation_df=wide_df,
                min_candidate_val_pct=float(args.min_candidate_val_pct),
                min_structure_val_pct=float(args.min_structure_val_pct),
                min_gap_score=float(args.min_gap_score),
                frame_adapt=True,
            )
            candidates = _v13_enrich_candidates(candidates, hist_payout, htot)
            # Lägg till V21/SUPER-grupper som extra röster.
            vargs = _v15_variant_args(args, 'V21A') if 'V21A' in V15_VARIANT_DEFS else args
            cand0 = _v15_filter_candidates_for_variant(candidates, vargs, htot)
            cand_v, group_cands = _v15_transform_candidates(cand0, htot, vtot, ftot, hist_payout, vargs, 'V21A')
            rowscore_pool = cand_v
            print(f'    V25 kandidatbank: singlar={len(candidates)} · med grupper={len(rowscore_pool)} · historik={htot} · validering={vtot}', flush=True)
            # Kontrollpaket V21A via gammal hård motor.
            for variant_id in variants:
                t0 = time.time()
                try:
                    if variant_id == 'V21A':
                        vargs2 = _v15_variant_args(args, 'V21A')
                        pkg, meta = _build_cluster_payout_package_v13_from_candidates(ns, rowscore_pool, htot, vtot, ftot, hist_payout, frame_rows, engine_frame, antal_matcher, vargs2, 'V21A')
                        if pkg is None:
                            raise RuntimeError(meta.get('error', 'Inget V21A-paket'))
                        pkg_pass, fail_reason = _v15_package_passes_row(ns, correct, specs, pkg)
                        diag = '' if pkg_pass else _v15_diagnose_facit(ns, correct, specs, pkg)
                        counts = _v15_group_counts(pkg)
                        extra = {'Facitrank': '', 'Facitscore': '', 'Cutoff score': '', 'Kandidat-röster': '', 'Facit familjscore JSON': '', 'V25 topfilter JSON': ''}
                    else:
                        cfg = V25_VARIANT_DEFS[variant_id]
                        pkg, meta = _v25_build_row_score_package(ns, frame_rows, specs, rowscore_pool, htot, vtot, ftot, hist_payout, cfg, variant_id, correct=correct)
                        if pkg is None:
                            raise RuntimeError(meta.get('error', 'Inget V25-paket'))
                        pkg_pass = bool(pkg.get('v25_meta', {}).get('correct_selected', False))
                        if not pkg_pass:
                            fail_reason = f"Facitrank {pkg.get('v25_meta',{}).get('correct_rank')} > topp {pkg.get('v25_meta',{}).get('top_rows')}"
                        else:
                            fail_reason = 'OK'
                        diag = '' if pkg_pass else json.dumps(pkg.get('v25_meta', {}), ensure_ascii=False)
                        counts = _v15_group_counts(pkg)
                        extra = {
                            'Facitrank': pkg.get('v25_meta', {}).get('correct_rank'),
                            'Facitscore': pkg.get('v25_meta', {}).get('correct_score'),
                            'Cutoff score': pkg.get('v25_meta', {}).get('cutoff_score'),
                            'Kandidat-röster': pkg.get('v25_meta', {}).get('candidate_votes'),
                            'Facit familjscore JSON': pkg.get('v25_meta', {}).get('correct_family_json'),
                            'V25 topfilter JSON': pkg.get('v25_meta', {}).get('top_filters_json'),
                        }
                    out.append({
                        'Variant': variant_id,
                        'Variantnamn': V25_VARIANT_DEFS[variant_id]['label'],
                        'Datum': str(test_date)[:10],
                        'Utdelning': int(round(payout)),
                        'Facit': correct,
                        'Status': 'OK',
                        'Paket klarar facit': 'Ja' if pkg_pass else 'Nej',
                        'Orsak': 'OK' if pkg_pass else fail_reason,
                        'Missdiagnos JSON': diag,
                        'Liknande historik': len(sim_df),
                        'Bred validering': len(wide_df),
                        'Paketträff': f"{pkg['hist_hit']}/{pkg['hist_total']}",
                        'Valideringsträff': f"{pkg['val_hit']}/{pkg['val_total']}",
                        'Grundram rader': len(frame_rows),
                        'Paketrader': int(pkg['frame_after']),
                        'Reducerar %': round(float(pkg['reduction_pct']), 2),
                        'Gemensamt score': round(float(pkg['joint_score']), 4),
                        'Utdelningsriktning %': round(float(pkg.get('payout_direction_pct', 0.0)), 2),
                        'Borttagna hist omgångar': int(pkg.get('removed_hist_count', 0)),
                        'Medel spret-gap': round(float(pkg.get('cluster_mean', 0.0)), 3),
                        'Filter totalt': int(pkg.get('num_filters', 0)),
                        'Strukturfilter': int(pkg.get('structure_filters', 0)),
                        'Profilfilter': int(pkg.get('profile_filters', 0)),
                        'FAT/ABC-filter': int(pkg.get('fat_filters', 0)),
                        'Värde/favorit/skräll': int(pkg.get('edge_filters', 0)),
                        **counts,
                        'Tillgängliga gruppkandidater': int(len(group_cands)),
                        'Filter JSON': _v13_json_filters(pkg),
                        'Gruppfilter JSON': _v15_group_json(pkg),
                        'Steg JSON': json.dumps(pkg.get('steps', []), ensure_ascii=False),
                        'Supersvår testomgång': 'Ja' if bool(test_row.get('_BT_No13Winner', False)) else 'Nej',
                        **extra,
                        'Sekunder': round(time.time()-t0, 2),
                    })
                    # Samla filterröstdiagnostik för rowscore-varianter.
                    if variant_id != 'V21A' and pkg is not None:
                        try:
                            tf = json.loads(extra.get('V25 topfilter JSON') or '[]')
                            for rank, row in enumerate(tf[:25], 1):
                                filter_rows.append({'Datum': str(test_date)[:10], 'Variant': variant_id, 'Filterrank': rank, **row})
                        except Exception:
                            pass
                except Exception as e:
                    out.append({'Variant': variant_id, 'Variantnamn': V25_VARIANT_DEFS.get(variant_id,{}).get('label', variant_id), 'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct, 'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e), 'Traceback': traceback.format_exc(), 'Sekunder': round(time.time()-t0,2)})
                    print(f'  FEL V25 variant {variant_id}: {e}', flush=True)
        except Exception as e:
            for variant_id in variants:
                out.append({'Variant': variant_id, 'Variantnamn': V25_VARIANT_DEFS.get(variant_id,{}).get('label', variant_id), 'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct, 'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e), 'Traceback': traceback.format_exc(), 'Sekunder': round(time.time()-t0_case,2)})
            print(f'  FEL V25 testfall: {e}', flush=True)
    return pd.DataFrame(out), {'supersvar_count': supersvar_count, 'eligible_rounds': len(db), 'variants': variants, 'filter_rows': pd.DataFrame(filter_rows)}


def _v25_summarize(detail):
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return pd.DataFrame()
    ok = detail[detail['Status'].astype(str).eq('OK')].copy()
    if ok.empty:
        return pd.DataFrame()
    rows=[]
    for variant, grp in ok.groupby('Variant', sort=False):
        hits = grp['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1']).sum()
        n = len(grp)
        med = lambda c: float(pd.to_numeric(grp[c], errors='coerce').dropna().median()) if c in grp else float('nan')
        rows.append({
            'Variant': variant,
            'Variantnamn': V25_VARIANT_DEFS.get(str(variant),{}).get('label', str(variant)),
            'Testade omgångar': n,
            'Träffar': int(hits),
            'Träff %': round(100.0*hits/max(1,n),1),
            'Median paketrader': round(med('Paketrader'),1),
            'Medel paketrader': round(float(pd.to_numeric(grp['Paketrader'], errors='coerce').dropna().mean()),1),
            'Median reducering %': round(med('Reducerar %'),1),
            'Median facitrank': round(med('Facitrank'),1) if 'Facitrank' in grp else float('nan'),
            'Median cutoffscore': round(med('Cutoff score'),4) if 'Cutoff score' in grp else float('nan'),
            'Median filter/röster': round(med('Filter totalt'),1),
            'Fel/hoppade': int((grp['Status'].astype(str) != 'OK').sum()),
        })
    summary = pd.DataFrame(rows)
    # Vinnare: först träff, sedan rader när träff lika, sedan reducering.
    if not summary.empty:
        summary['_rank'] = list(range(len(summary)))
        summary = summary.sort_values(['Träffar','Median paketrader','Median reducering %'], ascending=[False, True, False]).drop(columns=['_rank'])
        summary['Vinnare'] = ''
        summary.iloc[0, summary.columns.get_loc('Vinnare')] = 'JA'
    return summary


def _v25_write_outputs(detail, meta, args, out_dir, app_file, db_file):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = _v25_summarize(detail)
    detail_path = out_dir / f'{args.output_prefix}_detail.csv'
    summary_path = out_dir / f'{args.output_prefix}_summary.csv'
    report_path = out_dir / f'{args.output_prefix}_report.txt'
    winner_path = out_dir / f'{args.output_prefix}_winner_detail.csv'
    filter_path = out_dir / f'{args.output_prefix}_FILTER_VOTES.csv'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    winner = None
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        winner = str(summary.iloc[0]['Variant'])
        detail[detail['Variant'].astype(str).eq(winner)].to_csv(winner_path, index=False)
    else:
        pd.DataFrame().to_csv(winner_path, index=False)
    fr = meta.get('filter_rows') if isinstance(meta, dict) else None
    if isinstance(fr, pd.DataFrame) and not fr.empty:
        fr.to_csv(filter_path, index=False)
    else:
        pd.DataFrame().to_csv(filter_path, index=False)
    lines = [
        'TIPSET AI – V25b ROW SCORE BUDGET',
        '='*90,
        f'Appbas: {Path(app_file).name}',
        f'Databas: {Path(db_file).name}',
        f'Ram: {args.frame_profile} = {_v17_frame_rows_count(args.frame_profile)} rader',
        f'Testomgångar: {args.max_tests}',
        f'Varianter: {args.variants}',
        '',
        'METOD',
        '-'*90,
        'Alla bra filter röstar på varje rad. Inget enskilt filter får veto i TOP-varianterna.',
        'Slutpaketet är exakt topp N rader efter radscore: TOP2500/TOP2300/TOP2100/TOP1800.',
        'Facit används inte i valet, bara i eftertestet för rank/överlevnad.',
        '',
        'SAMMANFATTNING',
        '-'*90,
        summary.to_string(index=False) if isinstance(summary, pd.DataFrame) and not summary.empty else '(tom)',
        '',
        f'Detail: {detail_path}',
        f'Summary: {summary_path}',
        f'Winner detail: {winner_path}',
        f'Filter votes: {filter_path}',
    ]
    if winner:
        lines.append(f'Vinnare: {winner}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V25 ROW SCORE BUDGET', flush=True)
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        print(summary.to_string(index=False), flush=True)
    print('Summary:', summary_path, flush=True)
    print('Detail:', detail_path, flush=True)
    print('Filter votes:', filter_path, flush=True)
    return {'summary': summary_path, 'detail': detail_path, 'winner_detail': winner_path, 'filter_votes': filter_path, 'report': report_path}


def _v25_collect_frames(out_dir: Path, base_prefix: str, frames: list):
    outs = {}
    for suffix in ['summary','detail','winner_detail','FILTER_VOTES']:
        dfs=[]
        for frame in frames:
            p = Path(out_dir) / f'{base_prefix}_{_v17_safe_prefix_part(frame)}_{suffix}.csv'
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    if not df.empty:
                        if 'Ramprofil' not in df.columns:
                            df.insert(0, 'Ramprofil', frame)
                        if 'Teoretisk grundram' not in df.columns and suffix in {'summary','detail','winner_detail'}:
                            df.insert(1, 'Teoretisk grundram', _v17_frame_rows_count(frame))
                        dfs.append(df)
                except Exception:
                    pass
        if dfs:
            comb = pd.concat(dfs, ignore_index=True)
            outp = Path(out_dir) / f'{base_prefix}_ALL_FRAMES_{suffix}.csv'
            comb.to_csv(outp, index=False)
            outs[suffix] = comb
    return outs


def main_v25():
    parser = argparse.ArgumentParser(description='Tipset AI V25 – radscore budgetmotor.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_row_score_budget_v25_test10')
    parser.add_argument('--variants', default='V21A,SOFT2500,SOFT2300,SOFT2100,TOP2500,TOP2300,TOP2100,TOP1800,TOP2500_SAFE,TOP2300_SAFE,TOP2500_NODELTA')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='3-5-5')
    parser.add_argument('--frames', default='3-5-5')
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--candidate-min-hit', type=int, default=27)
    parser.add_argument('--min-candidate-val-pct', type=float, default=68.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=82.0)
    parser.add_argument('--min-gap-score', type=float, default=0.75)
    parser.add_argument('--variants-per-key', type=int, default=6)
    parser.add_argument('--max-candidates', type=int, default=340)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    # Argument som gamla funktioner kan kräva i kontrollmotorn.
    parser.add_argument('--min-hit', type=int, default=26)
    parser.add_argument('--min-package-val-pct', type=float, default=78.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=90.0)
    parser.add_argument('--min-unique-rows', type=int, default=50)
    parser.add_argument('--beam-width', type=int, default=30)
    parser.add_argument('--archive-width', type=int, default=220)
    parser.add_argument('--structure-seed-count', type=int, default=2)
    parser.add_argument('--hit-power', type=float, default=3.0)
    parser.add_argument('--validation-power', type=float, default=0.7)
    parser.add_argument('--reduction-power', type=float, default=2.7)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.25)
    parser.add_argument('--payout-direction-weight', type=float, default=0.16)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=50)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=350)
    parser.add_argument('--per-bucket-keep', type=int, default=1)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--v15-group-max-filters', type=int, default=18)
    parser.add_argument('--v15-max-group-candidates', type=int, default=120)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=3)
    parser.add_argument('--v15-cross-max-filters', type=int, default=20)
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    # V25b: lås de riktiga radscore-varianterna oavsett kvarliggande/ärvda Colab-argument.
    # Detta förhindrar att gamla --variants/--output-prefix från V20/V21 reducerar körningen till bara V21A.
    locked_variants = 'V21A,SOFT2500,SOFT2300,SOFT2100,TOP2500,TOP2300,TOP2100,TOP1800,TOP2500_SAFE,TOP2300_SAFE,TOP2500_NODELTA'
    if str(args.variants).strip() != locked_variants:
        print('V25b: ignorerar inkommande --variants och kör låst radscore-lista.', flush=True)
    args.variants = locked_variants
    if str(args.output_prefix).strip() != 'package_row_score_budget_v25b_test10':
        print('V25b: ignorerar inkommande --output-prefix och använder package_row_score_budget_v25b_test10.', flush=True)
    args.output_prefix = 'package_row_score_budget_v25b_test10'
    frames = _v17_parse_frames(args.frames)
    variants = _v25_parse_variants(args.variants)
    args.variants = ','.join(variants)
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n' + '='*96, flush=True)
    print('TIPSET AI V25b – ROW SCORE BUDGET', flush=True)
    print('LÅST LOGIK: filter röstar på rader; TOP-varianter väljer exakt N högst rankade rader.', flush=True)
    print(f'Ramar: {", ".join(frames)}', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)
    for frame in frames:
        frame_args = argparse.Namespace(**vars(args))
        frame_args.frames = frame
        frame_args.frame_profile = frame
        frame_args.output_prefix = f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n' + '-'*96, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*96, flush=True)
        ns = _load_app_functions(app_file, fast_no_supermakro=False)
        db = ns['load_database'](str(db_file), 13)
        detail, meta = _v25_run_backtest(ns, db, frame_args)
        _v25_write_outputs(detail, meta, frame_args, out_dir, app_file, db_file)
        del ns, db, detail
        gc.collect()
    combined = _v25_collect_frames(out_dir, args.output_prefix, frames)
    if combined.get('summary') is not None:
        print('\nKLART – V25 ALLA RAMAR', flush=True)
        print(combined['summary'].to_string(index=False), flush=True)
        print('Summary:', str(Path(out_dir) / f'{args.output_prefix}_ALL_FRAMES_summary.csv'), flush=True)


# =============================================================================
# V26 – V21A HYBRID SLICER: säker hårdram först, radscore bara inuti ramen
# =============================================================================
# Syfte: V25 visade att global radscore kastar bort vissa facitrader som V21A
# redan fångar. V26 låter därför V21A vara säkerhetsram och använder score endast
# som en budgetkniv inuti den ramen.

V26_VARIANT_DEFS = {
    'V21A_BASE': {
        'label': 'V21A basram',
        'description': 'Oförändrad V21A-kontroll, hårt paket.',
        'mode': 'base', 'top_rows': None,
    },
    'HYBRID2500': {
        'label': 'HYBRID2500 V21A + slicer',
        'description': 'V21A först; om basramen är större än 2500 väljs topp 2500 inom V21A.',
        'mode': 'hybrid', 'top_rows': 2500, 'score_mode': 'plain',
    },
    'HYBRID2300': {
        'label': 'HYBRID2300 V21A + slicer',
        'description': 'V21A först; om basramen är större än 2300 väljs topp 2300 inom V21A.',
        'mode': 'hybrid', 'top_rows': 2300, 'score_mode': 'plain',
    },
    'HYBRID_SAFE2500': {
        'label': 'HYBRID_SAFE2500 säkrare slicer',
        'description': 'V21A först; bara säkrare/dekorrelerade röster används för slicern.',
        'mode': 'hybrid', 'top_rows': 2500, 'score_mode': 'safe',
    },
    'HYBRID_CAPFAM2500': {
        'label': 'HYBRID_CAPFAM2500 familjetak',
        'description': 'V21A först; score normaliseras per familj så FAT/ABC inte dubbelröstar sönder rankingen.',
        'mode': 'hybrid', 'top_rows': 2500, 'score_mode': 'capfam',
    },
    'HYBRID_KEEP_EDGE2500': {
        'label': 'HYBRID_KEEP_EDGE2500 kantreserv',
        'description': 'V21A först; topprader plus reserv för rader starka i värde/favorit/skräll/poäng.',
        'mode': 'hybrid', 'top_rows': 2500, 'score_mode': 'keep_edge',
    },
    'HYBRID_CAPFAM2300': {
        'label': 'HYBRID_CAPFAM2300 familjetak',
        'description': 'V21A först; familjetakad slicer mot 2300 rader.',
        'mode': 'hybrid', 'top_rows': 2300, 'score_mode': 'capfam',
    },
}


def _v26_parse_variants(value):
    default = ['V21A_BASE','HYBRID2500','HYBRID2300','HYBRID_SAFE2500','HYBRID_CAPFAM2500','HYBRID_KEEP_EDGE2500','HYBRID_CAPFAM2300']
    allowed = set(V26_VARIANT_DEFS)
    raw = str(value or ','.join(default)).replace(';', ',').replace('+', ',').split(',')
    out = []
    for x in raw:
        k = x.strip().upper()
        if not k:
            continue
        # Gamla namn ska inte krascha eller styra körningen.
        if k in {'V21A','BASE','RADPRESS','V21B','V21C','V21D'}:
            k = 'V21A_BASE'
        if k not in allowed:
            continue
        if k not in out:
            out.append(k)
    # Om bara gammal kontroll kom in: kör hela V26-listan.
    if len(out) <= 1 and (not out or out[0] == 'V21A_BASE'):
        return default
    return out or default


def _v26_bits_from_package(pkg, n):
    """Återskapa radmasken för ett hårt paket från kandidatbits."""
    mask = np.ones(int(n), dtype=bool)
    for c in pkg.get('filters', []) or []:
        bits = int(c.get('frame_bits', 0) or 0)
        if bits:
            try:
                mask &= _v25_bits_bool(bits, int(n))
            except Exception:
                pass
    return mask


def _v26_candidate_signature(c):
    """Dedupera hårt överlappande röstkandidater."""
    fam = _v25_family(c)
    name = str(c.get('name',''))
    key = str(c.get('key',''))
    # Gruppkandidater med samma familj, krav och ungefär samma barn räknas som samma signal.
    if c.get('is_v15_group'):
        children = tuple(sorted(str(x) for x in (c.get('group_children') or [])))
        return ('GROUP', fam, children, int(c.get('group_required', 0)), int(c.get('group_size', len(children))))
    return ('SINGLE', fam, key or name)


def _v26_select_score_candidates(candidates, mode='capfam'):
    """Välj röstbank utan att låta en familj dominera med dubbletter/nästan-dubbletter."""
    # Basfilterkrav. Vi vill ha många röster, men inte osäkra/korrelerade dubletter.
    cfg = {
        'min_hit': 28,
        'min_val_pct': 72.0,
        'min_red_pct': 3.0,
        'family_balance': True,
        'safe': mode in {'safe','capfam','keep_edge'},
        'block_delta': mode in {'safe','capfam','keep_edge'},
        'soft_distance': True,
    }
    pool=[]
    for c in candidates:
        if not _v25_candidate_allowed(c, cfg):
            continue
        fam = _v25_family(c)
        # Prioritera bort hårda max0-grupper i FAT/ABC när motsvarande mjukare signaler finns.
        nm = str(c.get('name','')).lower()
        if c.get('is_v15_group') and 'max 0' in nm and fam in {'FAT','ABC','Poäng/rank'} and mode in {'safe','capfam','keep_edge'}:
            if float(c.get('red_pct', 0.0) or 0.0) < 40.0:
                continue
        c2 = dict(c)
        c2['_v25_family'] = fam
        c2['_v25_weight'] = _v25_weight(c2, cfg, None)
        # Kända överlappsfamiljer straffas om samma basnyckel förekommer i många intervall.
        if fam in {'FAT','ABC'} and not c2.get('is_v15_group'):
            c2['_v25_weight'] *= 0.86
        pool.append(c2)
    # Behåll starkaste per signal.
    best={}
    for c in pool:
        sig = _v26_candidate_signature(c)
        old=best.get(sig)
        if old is None or float(c.get('_v25_weight',0)) > float(old.get('_v25_weight',0)):
            best[sig]=c
    pool=list(best.values())
    pool.sort(key=lambda c: float(c.get('_v25_weight',0.0)), reverse=True)
    if mode == 'plain':
        # Plain är fortfarande begränsad, men mindre hård än capfam.
        caps = {'FAT': 10, 'ABC': 7, 'Poäng/rank': 6, 'Värde/svårighet': 7, 'Favorit': 5, 'Skräll': 4, 'Struktur': 2, 'FAT-sekvens': 2, 'Tvärfamilj': 3}
    elif mode == 'safe':
        caps = {'FAT': 7, 'ABC': 5, 'Poäng/rank': 5, 'Värde/svårighet': 5, 'Favorit': 4, 'Skräll': 3, 'Struktur': 2, 'FAT-sekvens': 1, 'Tvärfamilj': 2}
    else:
        caps = {'FAT': 6, 'ABC': 4, 'Poäng/rank': 5, 'Värde/svårighet': 5, 'Favorit': 4, 'Skräll': 3, 'Struktur': 2, 'FAT-sekvens': 1, 'Tvärfamilj': 2}
    picked=[]; counts={}
    for c in pool:
        fam = _v25_family(c)
        if counts.get(fam,0) >= caps.get(fam, 3):
            continue
        picked.append(c); counts[fam]=counts.get(fam,0)+1
        if len(picked) >= (42 if mode == 'plain' else 32):
            break
    return picked, cfg


def _v26_score_rows_decorrelated(ns, frame_rows, specs, candidates, mode='capfam'):
    specs_by_key = {s.get('key'): s for s in specs or []}
    used, cfg = _v26_select_score_candidates(candidates, mode=mode)
    n = len(frame_rows)
    fam_raw = {}
    cand_pass = []
    for c in used:
        fam = _v25_family(c)
        try:
            mult = _v25_candidate_row_multipliers(ns, frame_rows, specs_by_key, c, cfg)
        except Exception:
            mult = np.zeros(n, dtype=float)
        w = float(c.get('_v25_weight', 0.0))
        arr = mult.astype(float) * w
        fam_raw[fam] = fam_raw.get(fam, np.zeros(n, dtype=float)) + arr
        cand_pass.append((c, mult, w, fam))

    if mode == 'plain':
        scores = np.zeros(n, dtype=float)
        fam_scores={}
        for fam, arr in fam_raw.items():
            fam_scores[fam] = arr
            scores += arr
        return scores, fam_scores, used, cand_pass

    fam_caps = {
        'FAT': 1.00,
        'ABC': 0.78,
        'Poäng/rank': 0.95,
        'Värde/svårighet': 0.90,
        'Favorit': 0.78,
        'Skräll': 0.55,
        'Struktur': 0.25,
        'FAT-sekvens': 0.22,
        'Tvärfamilj': 0.55,
    }
    scores = np.zeros(n, dtype=float)
    fam_scores = {}
    fam_scales = {}
    for fam, arr in fam_raw.items():
        pos = arr[np.isfinite(arr) & (arr > 0)]
        if len(pos):
            scale = float(np.percentile(pos, 90))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = float(np.max(pos)) if len(pos) else 1.0
        else:
            scale = 1.0
        fam_scales[fam] = max(scale, 1e-12)
        norm = np.clip(arr / max(scale, 1e-12), 0.0, 1.25)
        contrib = norm * float(fam_caps.get(fam, 0.50))
        fam_scores[fam] = contrib
        scores += contrib
    for c in used:
        fam = _v25_family(c)
        c['_v26_fam_scale'] = float(fam_scales.get(fam, 1.0))
        c['_v26_fam_cap'] = float(fam_caps.get(fam, 0.50))
    # Familjebredd är en liten robusthetsbonus: rader som är okej i flera familjer ska slå rena FAT-kloner.
    breadth = np.zeros(n, dtype=float)
    for fam, arr in fam_scores.items():
        breadth += (arr > 0.05).astype(float)
    scores += 0.035 * breadth
    return scores, fam_scores, used, cand_pass


def _v26_score_correct(ns, correct, specs, used, mode='capfam'):
    """Scorea facitraden med samma kandidatlogik som V25, inklusive mjuka kantpoäng."""
    cfg = {
        'min_hit': 28,
        'min_val_pct': 72.0,
        'min_red_pct': 3.0,
        'family_balance': True,
        'safe': mode in {'safe','capfam','keep_edge'},
        'block_delta': mode in {'safe','capfam','keep_edge'},
        'soft_distance': True,
    }
    specs_by_key = {s.get('key'): s for s in specs or []}
    raw={}; passed=[]; total_raw=0.0
    for c in used:
        # 'used' är redan tillåten/dedupad, men _v25_correct_multiplier behöver cfg.
        try:
            mult = float(_v25_correct_multiplier(ns, correct, specs_by_key, c, cfg))
        except Exception:
            mult = 0.0
        if mult <= 0:
            continue
        fam = _v25_family(c)
        w = float(c.get('_v25_weight', _v25_weight(c, cfg, None)))
        raw[fam] = raw.get(fam, 0.0) + w * mult
        total_raw += w * mult
        if mult >= 0.999999:
            passed.append(c)
    return raw, passed, float(total_raw)


def _v26_build_hybrid_package(ns, frame_rows, specs, rowscore_pool, base_pkg, base_pass, correct, cfg, variant_id):
    n = len(frame_rows)
    target = int(cfg.get('top_rows') or n)
    base_mask = _v26_bits_from_package(base_pkg, n)
    base_count = int(base_mask.sum())
    if base_count <= 0:
        return None, {'error': 'V21A-basram tom.'}

    if cfg.get('mode') == 'base' or base_count <= target:
        # Rör inte en redan tillräckligt smal och träffsäker V21A-ram.
        pkg = dict(base_pkg)
        pkg['variant'] = variant_id
        pkg['variant_label'] = V26_VARIANT_DEFS[variant_id]['label']
        pkg['package_type'] = f'V26 {variant_id} – V21A basram'
        pkg['v26_meta'] = {
            'base_rows': int(base_count), 'target_rows': None if cfg.get('top_rows') is None else int(target),
            'final_rows': int(base_count), 'sliced': False, 'score_mode': 'none',
            'correct_rank_inside_base': None, 'correct_selected': bool(base_pass),
            'candidate_votes': 0, 'cutoff_score': None,
        }
        return pkg, pkg['v26_meta']

    mode = str(cfg.get('score_mode','capfam'))
    scores, fam_scores, used, cand_pass = _v26_score_rows_decorrelated(ns, frame_rows, specs, rowscore_pool, mode=mode)
    base_idx = np.where(base_mask)[0]
    # Slicern får bara välja inne i V21A-masken.
    base_scores = scores[base_idx]
    # Edge-reserv: håll kvar vissa rader med höga icke-FAT-signaler i stället för bara total score.
    if mode == 'keep_edge':
        edge = np.zeros(n, dtype=float)
        for fam, arr in fam_scores.items():
            if fam in {'Poäng/rank','Värde/svårighet','Favorit','Skräll','ABC'}:
                edge += arr
        main_n = max(1, int(target * 0.82))
        reserve_n = max(0, target - main_n)
        order_main = base_idx[np.lexsort((base_idx, -scores[base_idx]))][:main_n]
        remain_mask = np.ones(len(base_idx), dtype=bool)
        if len(order_main):
            chosen_set = set(int(x) for x in order_main)
            remain = np.array([i for i in base_idx if int(i) not in chosen_set], dtype=int)
        else:
            remain = base_idx
        order_edge = remain[np.lexsort((remain, -edge[remain]))][:reserve_n] if len(remain) and reserve_n else np.array([], dtype=int)
        chosen_idx = np.array(list(dict.fromkeys([int(x) for x in list(order_main)+list(order_edge)])), dtype=int)
        if len(chosen_idx) < min(target, base_count):
            add_pool = np.array([i for i in base_idx if int(i) not in set(int(x) for x in chosen_idx)], dtype=int)
            add = add_pool[np.lexsort((add_pool, -scores[add_pool]))][:(min(target,base_count)-len(chosen_idx))] if len(add_pool) else np.array([], dtype=int)
            chosen_idx = np.array(list(chosen_idx)+list(add), dtype=int)
    else:
        chosen_idx = base_idx[np.lexsort((base_idx, -base_scores))][:min(target, base_count)]
    final_mask = np.zeros(n, dtype=bool)
    final_mask[chosen_idx] = True
    final_count = int(final_mask.sum())
    cutoff_score = float(np.min(scores[chosen_idx])) if len(chosen_idx) else 0.0

    correct_in_frame = False
    correct_selected = False
    correct_rank = None
    correct_score = None
    correct_family_json = ''
    try:
        idx_map = {r: i for i, r in enumerate(frame_rows)}
        if correct in idx_map:
            ci = idx_map[correct]
            correct_in_frame = True
            correct_score = float(scores[ci])
            if bool(base_mask[ci]):
                # Rank endast bland V21A-raderna.
                correct_rank = int((scores[base_idx] > correct_score + 1e-12).sum()) + 1
                correct_selected = bool(final_mask[ci])
            else:
                correct_rank = None
                correct_selected = False
            correct_family_json = json.dumps({fam: round(float(arr[ci]), 4) for fam, arr in fam_scores.items() if float(arr[ci]) > 0}, ensure_ascii=False)
        else:
            raw_correct, passed, raw_total_correct = _v26_score_correct(ns, correct, specs, used, mode=mode)
            # Matcha ungefär samma familjetak genom att skala mot 90-percentilen i ramen.
            # I plain-läget jämförs råscore mot rå rowscore, så använd direkt totalscore.
            if mode == 'plain':
                correct_score = float(raw_total_correct)
                correct_rank = int((scores[base_idx] > correct_score + 1e-12).sum()) + 1
                correct_selected = bool(base_pass and correct_rank <= final_count)
                correct_family_json = json.dumps({k: round(v,4) for k,v in raw_correct.items()}, ensure_ascii=False)
                raw_correct = None
            corr_total = 0.0
            corr_payload = {}
            if raw_correct is not None:
                pass
            for fam, raw_val in (raw_correct or {}).items():
                arr = fam_scores.get(fam)
                if arr is None:
                    continue
                # fam_scores är redan cap-normaliserad per rad, men korrektscore är rå. Skala med raw distributionsdata indirekt:
                # gör robust proxy genom andel av max möjlig familjscore.
                fam_members = [c for c in used if _v25_family(c)==fam]
                fam_cap = float(fam_members[0].get('_v26_fam_cap', {'FAT':1.00,'ABC':0.78,'Poäng/rank':0.95,'Värde/svårighet':0.90,'Favorit':0.78,'Skräll':0.55,'Struktur':0.25,'FAT-sekvens':0.22,'Tvärfamilj':0.55}.get(fam,0.50))) if fam_members else 0.50
                fam_scale = float(fam_members[0].get('_v26_fam_scale', 1.0)) if fam_members else 1.0
                cv = fam_cap * min(1.25, float(raw_val)/max(1e-12,fam_scale)) if fam_scale else 0.0
                corr_payload[fam] = round(cv,4)
                corr_total += cv
            if raw_correct is not None:
                correct_score = float(corr_total)
                correct_rank = int((scores[base_idx] > correct_score + 1e-12).sum()) + 1
                correct_selected = bool(base_pass and correct_rank <= final_count)
                correct_family_json = json.dumps(corr_payload, ensure_ascii=False)
    except Exception:
        correct_selected = bool(base_pass and base_count <= target)

    selected_scores = scores[chosen_idx] if len(chosen_idx) else np.array([], dtype=float)
    top_filters = sorted(used, key=lambda c: float(c.get('_v25_weight', 0.0)), reverse=True)[:40]
    filters_json = json.dumps([
        {
            'name': c.get('name'), 'category': c.get('category'), 'family': _v25_family(c),
            'interval': c.get('interval_txt'), 'hist': f"{c.get('hist_hit')}/{c.get('hist_total')}",
            'val_pct': round(float(c.get('val_pct', 0.0)), 1),
            'red_pct': round(float(c.get('red_pct', 0.0)), 1),
            'weight': round(float(c.get('_v25_weight', 0.0)), 5),
            'group': bool(c.get('is_v15_group')),
        } for c in top_filters
    ], ensure_ascii=False)

    pkg = {
        'variant': str(variant_id),
        'variant_label': V26_VARIANT_DEFS[variant_id]['label'],
        'target': int(final_count),
        'target_label': f'V26 {variant_id}',
        'hist_hit': int(base_pkg.get('hist_hit', 0)), 'hist_total': int(base_pkg.get('hist_total', 0)),
        'val_hit': int(base_pkg.get('val_hit', 0)), 'val_total': int(base_pkg.get('val_total', 0)),
        'frame_start': int(base_pkg.get('frame_start', n)), 'frame_after': int(final_count),
        'reduction_pct': 100.0 - 100.0 * final_count / max(1, int(base_pkg.get('frame_start', n))),
        'joint_score': float(np.median(selected_scores)) if len(selected_scores) else 0.0,
        'core_score': float(np.median(selected_scores)) if len(selected_scores) else 0.0,
        'payout_lift_pct': float(base_pkg.get('payout_lift_pct', 0.0)),
        'payout_direction_pct': float(base_pkg.get('payout_direction_pct', 0.0)),
        'removed_hist_count': int(base_pkg.get('removed_hist_count', 0)),
        'removed_payout_mean': float(base_pkg.get('removed_payout_mean', 0.0)),
        'cluster_mean': float(base_pkg.get('cluster_mean', 0.0)),
        'num_filters': int(base_pkg.get('num_filters', 0)) + int(len(used)),
        'filters': list(base_pkg.get('filters', [])) + top_filters[:12],
        'steps': list(base_pkg.get('steps', [])) + [{'Filter':'V26 slicer','Kategori':'Radscore','Intervall':f'topp {final_count} inom V21A', 'Efter filter': int(final_count), 'Borttagna unika rader': int(max(0, base_count-final_count)), 'Fas':'hybrid_slicer'}],
        'package_type': f'V26 {variant_id} – V21A + intern radscore-slicer',
        'structure_filters': int(base_pkg.get('structure_filters', 0)),
        'profile_filters': int(base_pkg.get('profile_filters', 0)),
        'fat_filters': int(base_pkg.get('fat_filters', 0)),
        'edge_filters': int(base_pkg.get('edge_filters', 0)),
        'v26_meta': {
            'base_rows': int(base_count), 'target_rows': int(target), 'final_rows': int(final_count),
            'sliced': bool(base_count > target), 'score_mode': mode,
            'candidate_votes': int(len(used)), 'cutoff_score': round(float(cutoff_score), 6),
            'correct_in_frame': bool(correct_in_frame), 'correct_selected': bool(correct_selected),
            'correct_rank_inside_base': correct_rank, 'correct_score': correct_score,
            'correct_family_json': correct_family_json, 'top_filters_json': filters_json,
        },
    }
    return pkg, pkg['v26_meta']


def _v26_run_backtest(ns: dict, global_db: pd.DataFrame, args):
    antal_matcher = 13
    norm = ns['normalize_single_row_text']
    similar_history = ns['_similar_history_for_backtest']
    ranked_frame = ns['_ranked_frame_like_widths']
    generate_rows_from_frame = ns['generate_rows_from_frame']
    build_clean_filter_specs = ns['build_clean_filter_specs']
    global _ACTIVE_V9_NS
    _ACTIVE_V9_NS = ns
    global_db, supersvar_count = _prepare_supersvar_payout(global_db, int(args.pay_max))
    db = _prepare_db(ns, global_db, antal_matcher, int(args.pay_min), int(args.pay_max))
    if db.empty:
        return pd.DataFrame(), {'error': 'Inga giltiga omgångar i utdelningsintervallet.'}
    variants = _v26_parse_variants(args.variants)
    spik, halv, hel = _parse_frame_profile(args.frame_profile)
    template = _base_frame(spik, halv, hel)
    test_rows = list(db.iterrows())[int(args.test_offset):int(args.test_offset)+int(args.max_tests)]
    out=[]; filter_rows=[]
    for ti, (idx, test_row) in enumerate(test_rows, 1):
        _release_test_memory()
        t0_case = time.time()
        correct = norm(test_row.get('Correct_Row',''))
        input_vec = test_row.get('Prob_Vector', [])
        test_date = test_row.get('Datum', None)
        payout = float(pd.to_numeric(pd.Series([test_row.get('Payout',0)]), errors='coerce').fillna(0).iloc[0])
        print(f'V26 testfall {ti}/{len(test_rows)} · {str(test_date)[:10]} · utdelning {payout:,.0f} kr'.replace(',', ' '), flush=True)
        try:
            sim_df = similar_history(global_db, input_vec, antal_matcher, top_n=int(args.top_n), pay_min=int(args.pay_min), pay_max=int(args.pay_max), exclude_index=idx, mode=str(args.mode), test_date=test_date)
            wide_df = similar_history(global_db, input_vec, antal_matcher, top_n=max(int(args.top_n), int(args.wide_n)), pay_min=int(args.pay_min), pay_max=int(args.pay_max), exclude_index=idx, mode=str(args.mode), test_date=test_date)
            if len(sim_df) < int(args.top_n):
                raise RuntimeError(f'För få liknande omgångar: {len(sim_df)} av önskade {args.top_n}.')
            engine_frame = ranked_frame(template, input_vec, antal_matcher)
            frame_rows, _, ok, msg = generate_rows_from_frame(engine_frame, max_rows=None)
            if not ok or not frame_rows:
                raise RuntimeError(f'Kunde inte skapa ram: {msg}')
            specs = build_clean_filter_specs(sim_df, input_vec, antal_matcher, slider_u_count=3, target_hist_pct=int(args.filter_hist_target_pct), u_rows=None, hist_df=global_db, max_shock_pct=22, candidate_rows=frame_rows, include_supermakro=not bool(args.fast_no_supermakro))
            candidates, htot, vtot, ftot, hist_payout = _build_dynamic_candidates_v9(
                ns, specs, sim_df, frame_rows, engine_frame, antal_matcher,
                profile_min_hit=int(args.candidate_min_hit), variants_per_key=int(args.variants_per_key), max_candidates=int(args.max_candidates),
                validation_df=wide_df, min_candidate_val_pct=float(args.min_candidate_val_pct), min_structure_val_pct=float(args.min_structure_val_pct),
                min_gap_score=float(args.min_gap_score), frame_adapt=True,
            )
            candidates = _v13_enrich_candidates(candidates, hist_payout, htot)
            vargs = _v15_variant_args(args, 'V21A')
            cand0 = _v15_filter_candidates_for_variant(candidates, vargs, htot)
            cand_v, group_cands = _v15_transform_candidates(cand0, htot, vtot, ftot, hist_payout, vargs, 'V21A')
            rowscore_pool = cand_v
            print(f'    V26 bank: singlar={len(candidates)} · med grupper={len(rowscore_pool)} · V21A först, slicer efteråt', flush=True)
            base_pkg, base_meta = _build_cluster_payout_package_v13_from_candidates(ns, rowscore_pool, htot, vtot, ftot, hist_payout, frame_rows, engine_frame, antal_matcher, vargs, 'V21A')
            if base_pkg is None:
                raise RuntimeError(base_meta.get('error','Inget V21A-baspaket'))
            base_pass, base_fail = _v15_package_passes_row(ns, correct, specs, base_pkg)
            base_mask = _v26_bits_from_package(base_pkg, len(frame_rows))
            base_rows = int(base_mask.sum())
            print(f'    V21A-bas: rader={base_rows} · facit={"JA" if base_pass else "NEJ"}', flush=True)
            for variant_id in variants:
                t0 = time.time()
                try:
                    cfg = V26_VARIANT_DEFS[variant_id]
                    if variant_id == 'V21A_BASE':
                        pkg, meta = _v26_build_hybrid_package(ns, frame_rows, specs, rowscore_pool, base_pkg, base_pass, correct, cfg, variant_id)
                    else:
                        pkg, meta = _v26_build_hybrid_package(ns, frame_rows, specs, rowscore_pool, base_pkg, base_pass, correct, cfg, variant_id)
                    if pkg is None:
                        raise RuntimeError(meta.get('error','Inget V26-paket'))
                    if variant_id == 'V21A_BASE':
                        pkg_pass = bool(base_pass)
                        fail_reason = 'OK' if pkg_pass else str(base_fail)
                        diag = '' if pkg_pass else _v15_diagnose_facit(ns, correct, specs, base_pkg)
                    else:
                        pkg_pass = bool(pkg.get('v26_meta',{}).get('correct_selected', False))
                        if pkg_pass:
                            fail_reason = 'OK'
                            diag = ''
                        else:
                            rank = pkg.get('v26_meta',{}).get('correct_rank_inside_base')
                            topn = pkg.get('v26_meta',{}).get('final_rows')
                            if not base_pass:
                                fail_reason = 'V21A-basen fångade inte facit'
                            else:
                                fail_reason = f'Facitrank inom V21A {rank} > topp {topn}'
                            diag = json.dumps(pkg.get('v26_meta',{}), ensure_ascii=False)
                    counts = _v15_group_counts(base_pkg)
                    extra = {
                        'V21A basrader': pkg.get('v26_meta',{}).get('base_rows'),
                        'Slicer aktiv': 'Ja' if pkg.get('v26_meta',{}).get('sliced') else 'Nej',
                        'Facitrank inom V21A': pkg.get('v26_meta',{}).get('correct_rank_inside_base'),
                        'Facitscore': pkg.get('v26_meta',{}).get('correct_score'),
                        'Cutoff score': pkg.get('v26_meta',{}).get('cutoff_score'),
                        'Kandidat-röster': pkg.get('v26_meta',{}).get('candidate_votes'),
                        'Facit familjscore JSON': pkg.get('v26_meta',{}).get('correct_family_json'),
                        'V26 topfilter JSON': pkg.get('v26_meta',{}).get('top_filters_json',''),
                    }
                    out.append({
                        'Variant': variant_id,
                        'Variantnamn': V26_VARIANT_DEFS[variant_id]['label'],
                        'Datum': str(test_date)[:10],
                        'Utdelning': int(round(payout)),
                        'Facit': correct,
                        'Status': 'OK',
                        'Paket klarar facit': 'Ja' if pkg_pass else 'Nej',
                        'Orsak': 'OK' if pkg_pass else fail_reason,
                        'Missdiagnos JSON': diag,
                        'Liknande historik': len(sim_df),
                        'Bred validering': len(wide_df),
                        'Paketträff': f"{pkg['hist_hit']}/{pkg['hist_total']}",
                        'Valideringsträff': f"{pkg['val_hit']}/{pkg['val_total']}",
                        'Grundram rader': len(frame_rows),
                        'Paketrader': int(pkg['frame_after']),
                        'Reducerar %': round(float(pkg['reduction_pct']), 2),
                        'Gemensamt score': round(float(pkg['joint_score']), 4),
                        'Utdelningsriktning %': round(float(pkg.get('payout_direction_pct', 0.0)), 2),
                        'Borttagna hist omgångar': int(pkg.get('removed_hist_count', 0)),
                        'Medel spret-gap': round(float(pkg.get('cluster_mean', 0.0)), 3),
                        'Filter totalt': int(pkg.get('num_filters', 0)),
                        'Strukturfilter': int(pkg.get('structure_filters', 0)),
                        'Profilfilter': int(pkg.get('profile_filters', 0)),
                        'FAT/ABC-filter': int(pkg.get('fat_filters', 0)),
                        'Värde/favorit/skräll': int(pkg.get('edge_filters', 0)),
                        **counts,
                        'Tillgängliga gruppkandidater': int(len(group_cands)),
                        'Filter JSON': _v13_json_filters(base_pkg),
                        'Gruppfilter JSON': _v15_group_json(base_pkg),
                        'Steg JSON': json.dumps(pkg.get('steps', []), ensure_ascii=False),
                        'Supersvår testomgång': 'Ja' if bool(test_row.get('_BT_No13Winner', False)) else 'Nej',
                        **extra,
                        'Sekunder': round(time.time()-t0, 2),
                    })
                    if variant_id != 'V21A_BASE':
                        try:
                            tf = json.loads(extra.get('V26 topfilter JSON') or '[]')
                            for rank, row in enumerate(tf[:25],1):
                                filter_rows.append({'Datum': str(test_date)[:10], 'Variant': variant_id, 'Filterrank': rank, **row})
                        except Exception:
                            pass
                except Exception as e:
                    out.append({'Variant': variant_id, 'Variantnamn': V26_VARIANT_DEFS.get(variant_id,{}).get('label', variant_id), 'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct, 'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e), 'Traceback': traceback.format_exc(), 'Sekunder': round(time.time()-t0,2)})
                    print(f'  FEL V26 variant {variant_id}: {e}', flush=True)
        except Exception as e:
            for variant_id in variants:
                out.append({'Variant': variant_id, 'Variantnamn': V26_VARIANT_DEFS.get(variant_id,{}).get('label', variant_id), 'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct, 'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e), 'Traceback': traceback.format_exc(), 'Sekunder': round(time.time()-t0_case,2)})
            print(f'  FEL V26 testfall: {e}', flush=True)
    return pd.DataFrame(out), {'supersvar_count': supersvar_count, 'eligible_rounds': len(db), 'variants': variants, 'filter_rows': pd.DataFrame(filter_rows)}


def _v26_summarize(detail):
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return pd.DataFrame()
    ok = detail[detail['Status'].astype(str).eq('OK')].copy()
    if ok.empty:
        return pd.DataFrame()
    rows=[]
    for variant, grp in ok.groupby('Variant', sort=False):
        hits = grp['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1']).sum()
        n = len(grp)
        med = lambda c: float(pd.to_numeric(grp[c], errors='coerce').dropna().median()) if c in grp and not pd.to_numeric(grp[c], errors='coerce').dropna().empty else float('nan')
        rows.append({
            'Variant': variant,
            'Variantnamn': V26_VARIANT_DEFS.get(str(variant),{}).get('label', str(variant)),
            'Testade omgångar': n,
            'Träffar': int(hits),
            'Träff %': round(100.0*hits/max(1,n),1),
            'Median paketrader': round(med('Paketrader'),1),
            'Medel paketrader': round(float(pd.to_numeric(grp['Paketrader'], errors='coerce').dropna().mean()),1),
            'Median reducering %': round(med('Reducerar %'),1),
            'Median V21A basrader': round(med('V21A basrader'),1),
            'Median facitrank inom V21A': round(med('Facitrank inom V21A'),1),
            'Median cutoffscore': round(med('Cutoff score'),4),
            'Median filter/röster': round(med('Filter totalt'),1),
            'Slicer aktiva fall': int(grp['Slicer aktiv'].astype(str).str.lower().isin(['ja','true','1']).sum()) if 'Slicer aktiv' in grp else 0,
            'Fel/hoppade': int((grp['Status'].astype(str) != 'OK').sum()),
        })
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(['Träffar','Median paketrader','Median reducering %'], ascending=[False, True, False])
        summary['Vinnare'] = ''
        summary.iloc[0, summary.columns.get_loc('Vinnare')] = 'JA'
    return summary


def _v26_write_outputs(detail, meta, args, out_dir, app_file, db_file):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = _v26_summarize(detail)
    detail_path = out_dir / f'{args.output_prefix}_detail.csv'
    summary_path = out_dir / f'{args.output_prefix}_summary.csv'
    report_path = out_dir / f'{args.output_prefix}_report.txt'
    winner_path = out_dir / f'{args.output_prefix}_winner_detail.csv'
    filter_path = out_dir / f'{args.output_prefix}_FILTER_VOTES.csv'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    winner=None
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        winner=str(summary.iloc[0]['Variant'])
        detail[detail['Variant'].astype(str).eq(winner)].to_csv(winner_path,index=False)
    else:
        pd.DataFrame().to_csv(winner_path,index=False)
    fr = meta.get('filter_rows') if isinstance(meta, dict) else None
    if isinstance(fr, pd.DataFrame) and not fr.empty:
        fr.to_csv(filter_path,index=False)
    else:
        pd.DataFrame().to_csv(filter_path,index=False)
    lines=[
        'TIPSET AI – V26 V21A HYBRID SLICER',
        '='*90,
        f'Appbas: {Path(app_file).name}',
        f'Databas: {Path(db_file).name}',
        f'Ram: {args.frame_profile} = {_v17_frame_rows_count(args.frame_profile)} rader',
        f'Testomgångar: {args.max_tests}',
        f'Varianter: {args.variants}',
        '',
        'METOD', '-'*90,
        'V21A byggs först som säker hårdram. Radscore används endast om V21A-basen är större än målbudgeten.',
        'Om V21A redan är <= target lämnas paketet orört. Detta skyddar fall där V21A redan löser budgeten.',
        'CAPFAM/SAFE använder dekorrelerade familjeröster så FAT/ABC inte får dominera med många nästan identiska filter.',
        '', 'SAMMANFATTNING', '-'*90,
        summary.to_string(index=False) if isinstance(summary,pd.DataFrame) and not summary.empty else '(tom)',
        '', f'Detail: {detail_path}', f'Summary: {summary_path}', f'Winner detail: {winner_path}', f'Filter votes: {filter_path}',
    ]
    if winner:
        lines.append(f'Vinnare: {winner}')
    report_path.write_text('\n'.join(lines),encoding='utf-8')
    print('\nKLART – V26 V21A HYBRID SLICER', flush=True)
    if isinstance(summary,pd.DataFrame) and not summary.empty:
        print(summary.to_string(index=False), flush=True)
    print('Summary:', summary_path, flush=True)
    print('Detail:', detail_path, flush=True)
    print('Filter votes:', filter_path, flush=True)
    return {'summary':summary_path,'detail':detail_path,'winner_detail':winner_path,'filter_votes':filter_path,'report':report_path}


def _v26_collect_frames(out_dir: Path, base_prefix: str, frames: list):
    outs={}
    for suffix in ['summary','detail','winner_detail','FILTER_VOTES']:
        dfs=[]
        for frame in frames:
            p=Path(out_dir)/f'{base_prefix}_{_v17_safe_prefix_part(frame)}_{suffix}.csv'
            if p.exists():
                try:
                    df=pd.read_csv(p)
                    if not df.empty:
                        if 'Ramprofil' not in df.columns:
                            df.insert(0,'Ramprofil',frame)
                        if 'Teoretisk grundram' not in df.columns and suffix in {'summary','detail','winner_detail'}:
                            df.insert(1,'Teoretisk grundram',_v17_frame_rows_count(frame))
                        dfs.append(df)
                except Exception:
                    pass
        if dfs:
            comb=pd.concat(dfs,ignore_index=True)
            outp=Path(out_dir)/f'{base_prefix}_ALL_FRAMES_{suffix}.csv'
            comb.to_csv(outp,index=False)
            outs[suffix]=comb
    return outs


def main_v26():
    parser = argparse.ArgumentParser(description='Tipset AI V26 – V21A hybrid slicer.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_v21a_hybrid_slicer_v26_test10')
    parser.add_argument('--variants', default='V21A_BASE,HYBRID2500,HYBRID2300,HYBRID_SAFE2500,HYBRID_CAPFAM2500,HYBRID_KEEP_EDGE2500,HYBRID_CAPFAM2300')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='3-5-5')
    parser.add_argument('--frames', default='3-5-5')
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--candidate-min-hit', type=int, default=27)
    parser.add_argument('--min-candidate-val-pct', type=float, default=68.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=82.0)
    parser.add_argument('--min-gap-score', type=float, default=0.75)
    parser.add_argument('--variants-per-key', type=int, default=6)
    parser.add_argument('--max-candidates', type=int, default=340)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    # Hård V21A-kontrollmotor
    parser.add_argument('--min-hit', type=int, default=26)
    parser.add_argument('--min-package-val-pct', type=float, default=78.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=90.0)
    parser.add_argument('--min-unique-rows', type=int, default=50)
    parser.add_argument('--beam-width', type=int, default=30)
    parser.add_argument('--archive-width', type=int, default=220)
    parser.add_argument('--structure-seed-count', type=int, default=2)
    parser.add_argument('--hit-power', type=float, default=3.0)
    parser.add_argument('--validation-power', type=float, default=0.7)
    parser.add_argument('--reduction-power', type=float, default=2.7)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.25)
    parser.add_argument('--payout-direction-weight', type=float, default=0.16)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=50)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=350)
    parser.add_argument('--per-bucket-keep', type=int, default=1)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--v15-group-max-filters', type=int, default=18)
    parser.add_argument('--v15-max-group-candidates', type=int, default=120)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=3)
    parser.add_argument('--v15-cross-max-filters', type=int, default=20)
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    locked_variants='V21A_BASE,HYBRID2500,HYBRID2300,HYBRID_SAFE2500,HYBRID_CAPFAM2500,HYBRID_KEEP_EDGE2500,HYBRID_CAPFAM2300'
    if str(args.variants).strip() != locked_variants:
        print('V26: ignorerar inkommande --variants och kör låst hybridlista.', flush=True)
    args.variants=locked_variants
    if str(args.output_prefix).strip() != 'package_v21a_hybrid_slicer_v26_test10':
        print('V26: ignorerar inkommande --output-prefix och använder package_v21a_hybrid_slicer_v26_test10.', flush=True)
    args.output_prefix='package_v21a_hybrid_slicer_v26_test10'
    frames=_v17_parse_frames(args.frames)
    variants=_v26_parse_variants(args.variants)
    args.variants=','.join(variants)
    app_file, db_file = _resolve_required_files(args)
    out_dir=Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir=Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n' + '='*96, flush=True)
    print('TIPSET AI V26 – V21A HYBRID SLICER', flush=True)
    print('LÅST LOGIK: V21A först; radscore skär bara inuti V21A-ramen om den är större än target.', flush=True)
    print('Familjetak: CAPFAM/SAFE hindrar FAT/ABC från att dubbelräkna nästan samma signal.', flush=True)
    print(f'Ramar: {", ".join(frames)}', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)
    for frame in frames:
        frame_args=argparse.Namespace(**vars(args))
        frame_args.frames=frame
        frame_args.frame_profile=frame
        frame_args.output_prefix=f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n' + '-'*96, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*96, flush=True)
        ns=_load_app_functions(app_file, fast_no_supermakro=False)
        db=ns['load_database'](str(db_file),13)
        detail, meta = _v26_run_backtest(ns, db, frame_args)
        _v26_write_outputs(detail, meta, frame_args, out_dir, app_file, db_file)
        del ns, db, detail
        gc.collect()
    combined=_v26_collect_frames(out_dir,args.output_prefix,frames)
    if combined.get('summary') is not None:
        print('\nKLART – V26 ALLA RAMAR', flush=True)
        print(combined['summary'].to_string(index=False), flush=True)
        print('Summary:', str(Path(out_dir)/f'{args.output_prefix}_ALL_FRAMES_summary.csv'), flush=True)


# =============================================================================
# V27 – SAFE TRIM AUDIT: V21A bas, sedan små/lossless trimfilter ovanpå
# =============================================================================
# Syfte: V26 visade att ranking-slicer kapar fel rader. V27 testar därför den
# enklare frågan: vilka extra filter kan läggas ovanpå V21A utan att förstöra
# V21A:s träff i testfönstret? Produktionsvarianter använder bara historik/
# validering; AUDIT-varianter är tydligt märkta och använder facit för att mäta
# teoretisk trim-potential i efterhand.

V27_VARIANT_DEFS = {
    'V21A_BASE': {
        'label': 'V21A basram',
        'mode': 'base',
        'target_rows': None,
    },
    'TRIM_STRICT2500': {
        'label': 'TRIM_STRICT2500 hist0/val0',
        'mode': 'prod', 'target_rows': 2500,
        'max_hist_loss': 0, 'max_val_loss': 0, 'min_step_rows': 35,
        'min_step_pct': 1.0, 'max_add': 4, 'block_red': True,
    },
    'TRIM_VAL1_2500': {
        'label': 'TRIM_VAL1_2500 hist0/val1',
        'mode': 'prod', 'target_rows': 2500,
        'max_hist_loss': 0, 'max_val_loss': 1, 'min_step_rows': 45,
        'min_step_pct': 1.2, 'max_add': 4, 'block_red': True,
    },
    'TRIM_VAL2_2500': {
        'label': 'TRIM_VAL2_2500 hist0/val2',
        'mode': 'prod', 'target_rows': 2500,
        'max_hist_loss': 0, 'max_val_loss': 2, 'min_step_rows': 60,
        'min_step_pct': 1.5, 'max_add': 5, 'block_red': True,
    },
    'TRIM_9OF10_2500': {
        'label': 'TRIM_9OF10_2500 tillåter hist-1',
        'mode': 'prod', 'target_rows': 2500,
        'max_hist_loss': 1, 'max_val_loss': 2, 'min_step_rows': 75,
        'min_step_pct': 1.8, 'max_add': 5, 'block_red': True,
    },
    'TRIM_STRICT2400': {
        'label': 'TRIM_STRICT2400 hist0/val0',
        'mode': 'prod', 'target_rows': 2400,
        'max_hist_loss': 0, 'max_val_loss': 0, 'min_step_rows': 35,
        'min_step_pct': 1.0, 'max_add': 5, 'block_red': True,
    },
    'AUDIT_LOSSLESS2500': {
        'label': 'AUDIT_LOSSLESS2500 facit-skyddad',
        'mode': 'audit', 'target_rows': 2500,
        'max_hist_loss': 30, 'max_val_loss': 45, 'min_step_rows': 10,
        'min_step_pct': 0.1, 'max_add': 10, 'block_red': False,
    },
    'AUDIT_LOSSLESS2300': {
        'label': 'AUDIT_LOSSLESS2300 facit-skyddad',
        'mode': 'audit', 'target_rows': 2300,
        'max_hist_loss': 30, 'max_val_loss': 45, 'min_step_rows': 10,
        'min_step_pct': 0.1, 'max_add': 12, 'block_red': False,
    },
}


def _v27_parse_variants(value):
    default = ['V21A_BASE','TRIM_STRICT2500','TRIM_VAL1_2500','TRIM_VAL2_2500','TRIM_9OF10_2500','TRIM_STRICT2400','AUDIT_LOSSLESS2500','AUDIT_LOSSLESS2300']
    allowed = set(V27_VARIANT_DEFS)
    raw = str(value or ','.join(default)).replace(';', ',').replace('+', ',').split(',')
    out=[]
    for x in raw:
        k=x.strip().upper()
        if not k:
            continue
        if k in {'V21A','BASE','RADPRESS','V21A_BASE'}:
            k='V21A_BASE'
        if k in allowed and k not in out:
            out.append(k)
    if len(out) <= 1 and (not out or out[0]=='V21A_BASE'):
        return default
    return out or default


def _v27_bits_from_pkg_field(pkg, total, field):
    mask = (1 << int(total)) - 1 if int(total) > 0 else 0
    any_bits = False
    for c in pkg.get('filters', []) or []:
        bits = int(c.get(field, 0) or 0)
        if bits:
            mask &= bits
            any_bits = True
    if not any_bits:
        return mask
    return int(mask)


def _v27_state_from_base_pkg(base_pkg, htot, vtot, ftot):
    chosen = tuple(dict(c) for c in (base_pkg.get('filters', []) or []))
    used = frozenset(str(c.get('key','')) for c in chosen if c.get('key'))
    return _V9State(
        hist_mask=_v27_bits_from_pkg_field(base_pkg, htot, 'hist_bits'),
        val_mask=_v27_bits_from_pkg_field(base_pkg, vtot, 'val_bits') if int(vtot) > 0 else 0,
        frame_mask=_v27_bits_from_pkg_field(base_pkg, ftot, 'frame_bits'),
        hist_size=int(htot), val_size=int(vtot), frame_size=int(ftot),
        chosen=chosen, used_keys=used, steps=tuple(base_pkg.get('steps', []) or []),
    )


def _v27_candidate_family(c):
    try:
        return _v25_family(c)
    except Exception:
        try:
            return _v19_family_of_candidate(c)
        except Exception:
            return str(c.get('category','Övrigt'))


def _v27_is_red_flag_candidate(c):
    name = str(c.get('name','')).lower()
    fam = _v27_candidate_family(c)
    if _v21_is_delta_avvikelse(c) and not bool(c.get('is_v15_group')):
        return True
    if _v21_is_bad_favorite_max0(c):
        return True
    try:
        if _v22_is_low_value_risk_candidate(c):
            return True
    except Exception:
        pass
    if fam == 'Struktur' and not bool(c.get('is_v15_group')):
        return True
    # Max0-grupper från stora FAT/ABC-buntar är ofta för vassa som trimkniv.
    if bool(c.get('is_v15_group')):
        try:
            if int(c.get('group_max_miss', 0)) == 0 and int(c.get('group_size', 0)) >= 4 and fam in {'FAT','ABC','Poäng/rank'}:
                return True
        except Exception:
            pass
    return False


def _v27_base_candidate_ok(c, cfg):
    if not c or not c.get('key'):
        return False
    if int(c.get('frame_bits', 0) or 0) <= 0:
        return False
    if int(c.get('hist_bits', 0) or 0) <= 0:
        return False
    if bool(cfg.get('block_red', True)) and _v27_is_red_flag_candidate(c):
        return False
    hist = int(c.get('hist_hit', 0) or 0)
    valp = float(c.get('val_pct', 0.0) or 0.0)
    fam = _v27_candidate_family(c)
    if cfg.get('mode') == 'audit':
        # Audit ska visa teoretisk facitbevarande trim-potential, därför släpper den igenom
        # även svagare/rödare filter; resultatet märks tydligt som icke-produktionsvariant.
        return hist >= 20 and valp >= 35.0
    if fam == 'Struktur':
        return hist >= 29 and valp >= 90.0 and bool(c.get('is_v15_group'))
    return hist >= 28 and valp >= 82.0


def _v27_apply_trim_candidate(cur: _V9State, cand: dict, cfg: dict, *, correct_idx=None, audit_mode=False):
    key = str(cand.get('key',''))
    if not key or key in cur.used_keys:
        return None
    new_frame = int(cur.frame_mask) & int(cand.get('frame_bits', 0) or 0)
    new_rows = int(new_frame.bit_count())
    if new_rows <= 0 or new_rows >= cur.frame_count:
        return None
    removed = int(cur.frame_count - new_rows)
    if removed < int(cfg.get('min_step_rows', 1)):
        return None
    step_pct = 100.0 * removed / max(1, cur.frame_count)
    if step_pct + 1e-12 < float(cfg.get('min_step_pct', 0.0)):
        return None
    new_hist = int(cur.hist_mask) & int(cand.get('hist_bits', 0) or 0)
    new_val = int(cur.val_mask) & int(cand.get('val_bits', 0) or 0) if cur.val_size else 0
    hist_loss = int(cur.hist_hit - int(new_hist.bit_count()))
    val_loss = int(cur.val_hit - int(new_val.bit_count())) if cur.val_size else 0
    if hist_loss > int(cfg.get('max_hist_loss', 0)):
        return None
    if cur.val_size and val_loss > int(cfg.get('max_val_loss', 0)):
        return None
    if audit_mode and correct_idx is not None:
        if not bool((new_frame >> int(correct_idx)) & 1):
            return None
    c2 = dict(cand)
    c2['step_removed_rows'] = removed
    c2['step_red_pct'] = step_pct
    c2['v27_hist_loss'] = hist_loss
    c2['v27_val_loss'] = val_loss
    step = {
        'Filter': c2.get('name',''),
        'Kategori': c2.get('category',''),
        'Familj': _v27_candidate_family(c2),
        'Intervall': c2.get('interval_txt','-'),
        'Efter filter': int(new_rows),
        'Borttagna unika rader': int(removed),
        'Stegreducering %': round(float(step_pct), 3),
        'Histförlust': int(hist_loss),
        'Valförlust': int(val_loss),
        'Samlad historikträff': f'{int(new_hist.bit_count())}/{cur.hist_size}',
        'Samlad validering': f'{int(new_val.bit_count())}/{cur.val_size}' if cur.val_size else '-',
        'Spret-gap': round(float(c2.get('gap_score',0.0)),3),
        'Utdelningsriktning %': round(float(c2.get('payout_direction_pct', c2.get('payout_lift_pct', 0.0)) or 0.0),2),
        'Fas': 'v27_trim_audit' if audit_mode else 'v27_trim',
    }
    return _V9State(
        hist_mask=new_hist, val_mask=new_val, frame_mask=new_frame,
        hist_size=cur.hist_size, val_size=cur.val_size, frame_size=cur.frame_size,
        chosen=cur.chosen + (c2,), used_keys=frozenset(set(cur.used_keys) | {key}),
        steps=cur.steps + (step,),
    )


def _v27_candidate_sort_key(cur, cand, trial, cfg):
    removed = cur.frame_count - trial.frame_count
    fam = _v27_candidate_family(cand)
    name = str(cand.get('name','')).lower()
    fam_bonus = {'FAT': 5, 'ABC': 4, 'Poäng/rank': 4, 'Värde/svårighet': 3, 'Favorit': 3, 'Skräll': 2, 'Struktur': 1, 'FAT-sekvens': 0, 'Tvärfamilj': 1}.get(fam, 0)
    # För produktionsvarianter: välj stabilitet först, sedan radnytta. För audit: radnytta mer först.
    hist_loss = int(cur.hist_hit - trial.hist_hit)
    val_loss = int(cur.val_hit - trial.val_hit) if cur.val_size else 0
    red_flag_pen = -50 if _v27_is_red_flag_candidate(cand) else 0
    small_name_bonus = 0
    if any(x in name for x in ['poängfilter','rank summa','ai-rank','abc summa','fat summa','skrälltryck','favorit-delta']):
        small_name_bonus += 3
    if cfg.get('mode') == 'audit':
        return (removed, -hist_loss, -val_loss, fam_bonus, small_name_bonus, float(cand.get('val_pct',0.0) or 0.0))
    return (-hist_loss, -val_loss, removed, fam_bonus, small_name_bonus, float(cand.get('val_pct',0.0) or 0.0), red_flag_pen)


def _v27_greedy_trim_package(ns, frame_rows, specs, candidates, base_pkg, correct, htot, vtot, ftot, hist_payout, args, variant_id):
    cfg = V27_VARIANT_DEFS[variant_id]
    base_state = _v27_state_from_base_pkg(base_pkg, htot, vtot, ftot)
    target = int(cfg.get('target_rows') or base_state.frame_count)
    correct_idx = None
    try:
        idx_map = {r: i for i, r in enumerate(frame_rows)}
        if correct in idx_map:
            correct_idx = int(idx_map[correct])
    except Exception:
        correct_idx = None
    if cfg.get('mode') == 'base' or base_state.frame_count <= target:
        final = base_state
        trim_meta = {'base_rows': base_state.frame_count, 'target_rows': None if cfg.get('target_rows') is None else target, 'final_rows': base_state.frame_count, 'trim_active': False, 'added_trim_filters': 0, 'audit_mode': False, 'candidate_pool': 0, 'evaluated_single': 0, 'best_lossless_single_rows': None}
    else:
        audit_mode = bool(cfg.get('mode') == 'audit')
        bank=[]
        base_used = set(base_state.used_keys)
        for c in candidates:
            if str(c.get('key','')) in base_used:
                continue
            if not _v27_base_candidate_ok(c, cfg):
                continue
            bank.append(c)
        # Dedupa samma signal/intervall, behåll starkaste.
        best={}
        for c in bank:
            sig = (_v26_candidate_signature(c), str(c.get('interval_txt','')))
            old=best.get(sig)
            if old is None:
                best[sig]=c
            else:
                if (float(c.get('red_pct',0.0) or 0.0), int(c.get('hist_hit',0) or 0), float(c.get('val_pct',0.0) or 0.0)) > (float(old.get('red_pct',0.0) or 0.0), int(old.get('hist_hit',0) or 0), float(old.get('val_pct',0.0) or 0.0)):
                    best[sig]=c
        bank=list(best.values())[:240]
        cur = base_state
        evaluated_single = 0
        best_lossless_single = None
        # För audit: först mät bästa facitbevarande single-trim från basen.
        for c in bank:
            trial = _v27_apply_trim_candidate(base_state, c, cfg, correct_idx=correct_idx, audit_mode=audit_mode)
            if trial is None:
                continue
            evaluated_single += 1
            removed = base_state.frame_count - trial.frame_count
            if best_lossless_single is None or removed > best_lossless_single[0]:
                best_lossless_single = (removed, c.get('name',''), trial.frame_count, _v27_candidate_family(c))
        # Greedy add.
        max_add = int(cfg.get('max_add', 4))
        for _ in range(max_add):
            if cur.frame_count <= target:
                break
            options=[]
            for c in bank:
                trial = _v27_apply_trim_candidate(cur, c, cfg, correct_idx=correct_idx, audit_mode=audit_mode)
                if trial is None:
                    continue
                key = _v27_candidate_sort_key(cur, c, trial, cfg)
                options.append((key, c, trial))
            if not options:
                break
            options.sort(key=lambda x: x[0], reverse=True)
            _, c, trial = options[0]
            cur = trial
            print(f'      V27 {variant_id}: + {c.get("name","")} → rader={cur.frame_count} · hist={cur.hist_hit}/{cur.hist_size} · val={cur.val_hit}/{cur.val_size if cur.val_size else 0}', flush=True)
        final = cur
        trim_meta = {'base_rows': base_state.frame_count, 'target_rows': target, 'final_rows': final.frame_count, 'trim_active': final.frame_count < base_state.frame_count, 'added_trim_filters': max(0, len(final.chosen)-len(base_state.chosen)), 'audit_mode': audit_mode, 'candidate_pool': len(bank), 'evaluated_single': int(evaluated_single), 'best_lossless_single_rows': int(best_lossless_single[2]) if best_lossless_single else None, 'best_lossless_single_removed': int(best_lossless_single[0]) if best_lossless_single else None, 'best_lossless_single_name': str(best_lossless_single[1]) if best_lossless_single else '', 'best_lossless_single_family': str(best_lossless_single[3]) if best_lossless_single else ''}
    metrics = _v13_state_metrics(final, hist_payout, args, variant_id)
    pkg = {
        'variant': variant_id,
        'variant_label': V27_VARIANT_DEFS[variant_id]['label'],
        'target': int(final.hist_hit),
        'target_label': f'V27 {variant_id} {final.hist_hit}/{htot}',
        'hist_hit': int(final.hist_hit), 'hist_total': int(htot),
        'val_hit': int(final.val_hit), 'val_total': int(vtot),
        'frame_start': int(ftot), 'frame_after': int(final.frame_count),
        'reduction_pct': float(metrics.get('reduction_pct', 0.0)),
        'joint_score': float(metrics.get('joint_score', 0.0)),
        'core_score': float(metrics.get('core_score', 0.0)),
        'payout_lift_pct': float(metrics.get('payout_lift_pct', 0.0)),
        'payout_direction_pct': float(metrics.get('payout_direction_pct', metrics.get('payout_lift_pct', 0.0))),
        'removed_hist_count': int(metrics.get('removed_hist_count', max(0, htot-final.hist_hit))),
        'removed_payout_mean': float(metrics.get('removed_payout_mean', 0.0)),
        'cluster_mean': float(metrics.get('cluster_mean', 0.0)),
        'num_filters': len(final.chosen),
        'filters': list(final.chosen), 'steps': list(final.steps),
        'package_type': f'V27 {variant_id} – safe trim audit',
        'structure_filters': final.structure_filters,
        'profile_filters': final.profile_filters,
        'fat_filters': final.fat_filters,
        'edge_filters': final.edge_filters,
        'v27_meta': trim_meta,
    }
    return pkg, trim_meta


def _v27_single_trim_audit_rows(base_state, candidates, cfg, date, correct_idx=None):
    rows=[]; base_used=set(base_state.used_keys)
    for c in candidates:
        if str(c.get('key','')) in base_used:
            continue
        if not _v27_base_candidate_ok(c, cfg):
            continue
        trial_no_oracle = _v27_apply_trim_candidate(base_state, c, cfg, correct_idx=None, audit_mode=False)
        if trial_no_oracle is None:
            continue
        killed_correct = ''
        if correct_idx is not None:
            killed_correct = 'Nej' if bool((trial_no_oracle.frame_mask >> int(correct_idx)) & 1) else 'Ja'
        rows.append({
            'Datum': str(date)[:10],
            'Filter': c.get('name',''),
            'Kategori': c.get('category',''),
            'Familj': _v27_candidate_family(c),
            'Intervall': c.get('interval_txt',''),
            'Hist': f"{int(c.get('hist_hit',0) or 0)}/{base_state.hist_size}",
            'Val_pct': round(float(c.get('val_pct',0.0) or 0.0),1),
            'Basrader': int(base_state.frame_count),
            'Efter filter': int(trial_no_oracle.frame_count),
            'Borttagna rader': int(base_state.frame_count-trial_no_oracle.frame_count),
            'Histförlust': int(base_state.hist_hit-trial_no_oracle.hist_hit),
            'Valförlust': int(base_state.val_hit-trial_no_oracle.val_hit) if base_state.val_size else 0,
            'Dödar facit': killed_correct,
            'Red flag': 'Ja' if _v27_is_red_flag_candidate(c) else 'Nej',
            'Grupp': 'Ja' if bool(c.get('is_v15_group')) else 'Nej',
        })
    rows.sort(key=lambda r: (r.get('Dödar facit') == 'Nej', -int(r['Histförlust']), -int(r['Valförlust']), int(r['Borttagna rader'])), reverse=True)
    return rows


def _v27_run_backtest(ns: dict, global_db: pd.DataFrame, args):
    antal_matcher = 13
    norm = ns['normalize_single_row_text']
    similar_history = ns['_similar_history_for_backtest']
    ranked_frame = ns['_ranked_frame_like_widths']
    generate_rows_from_frame = ns['generate_rows_from_frame']
    build_clean_filter_specs = ns['build_clean_filter_specs']
    global _ACTIVE_V9_NS
    _ACTIVE_V9_NS = ns
    global_db, supersvar_count = _prepare_supersvar_payout(global_db, int(args.pay_max))
    db = _prepare_db(ns, global_db, antal_matcher, int(args.pay_min), int(args.pay_max))
    if db.empty:
        return pd.DataFrame(), {'error': 'Inga giltiga omgångar i utdelningsintervallet.'}
    variants = _v27_parse_variants(args.variants)
    spik, halv, hel = _parse_frame_profile(args.frame_profile)
    template = _base_frame(spik, halv, hel)
    test_rows = list(db.iterrows())[int(args.test_offset):int(args.test_offset)+int(args.max_tests)]
    out=[]; audit_rows=[]
    for ti, (idx, test_row) in enumerate(test_rows, 1):
        _release_test_memory()
        t0_case=time.time()
        correct = norm(test_row.get('Correct_Row',''))
        input_vec = test_row.get('Prob_Vector', [])
        test_date = test_row.get('Datum', None)
        payout = float(pd.to_numeric(pd.Series([test_row.get('Payout',0)]), errors='coerce').fillna(0).iloc[0])
        print(f'V27 testfall {ti}/{len(test_rows)} · {str(test_date)[:10]} · utdelning {payout:,.0f} kr'.replace(',', ' '), flush=True)
        try:
            sim_df = similar_history(global_db, input_vec, antal_matcher, top_n=int(args.top_n), pay_min=int(args.pay_min), pay_max=int(args.pay_max), exclude_index=idx, mode=str(args.mode), test_date=test_date)
            wide_df = similar_history(global_db, input_vec, antal_matcher, top_n=max(int(args.top_n), int(args.wide_n)), pay_min=int(args.pay_min), pay_max=int(args.pay_max), exclude_index=idx, mode=str(args.mode), test_date=test_date)
            if len(sim_df) < int(args.top_n):
                raise RuntimeError(f'För få liknande omgångar: {len(sim_df)} av önskade {args.top_n}.')
            engine_frame = ranked_frame(template, input_vec, antal_matcher)
            frame_rows, _, ok, msg = generate_rows_from_frame(engine_frame, max_rows=None)
            if not ok or not frame_rows:
                raise RuntimeError(f'Kunde inte skapa ram: {msg}')
            specs = build_clean_filter_specs(sim_df, input_vec, antal_matcher, slider_u_count=3, target_hist_pct=int(args.filter_hist_target_pct), u_rows=None, hist_df=global_db, max_shock_pct=22, candidate_rows=frame_rows, include_supermakro=not bool(args.fast_no_supermakro))
            candidates, htot, vtot, ftot, hist_payout = _build_dynamic_candidates_v9(
                ns, specs, sim_df, frame_rows, engine_frame, antal_matcher,
                profile_min_hit=int(args.candidate_min_hit), variants_per_key=int(args.variants_per_key), max_candidates=int(args.max_candidates),
                validation_df=wide_df, min_candidate_val_pct=float(args.min_candidate_val_pct), min_structure_val_pct=float(args.min_structure_val_pct),
                min_gap_score=float(args.min_gap_score), frame_adapt=True,
            )
            candidates = _v13_enrich_candidates(candidates, hist_payout, htot)
            vargs = _v15_variant_args(args, 'V21A')
            cand0 = _v15_filter_candidates_for_variant(candidates, vargs, htot)
            cand_v, group_cands = _v15_transform_candidates(cand0, htot, vtot, ftot, hist_payout, vargs, 'V21A')
            print(f'    V27 bank: singlar={len(candidates)} · med grupper={len(cand_v)} · V21A bas + trimfilter', flush=True)
            base_pkg, base_meta = _build_cluster_payout_package_v13_from_candidates(ns, cand_v, htot, vtot, ftot, hist_payout, frame_rows, engine_frame, antal_matcher, vargs, 'V21A')
            if base_pkg is None:
                raise RuntimeError(base_meta.get('error','Inget V21A-baspaket'))
            base_pass, base_fail = _v15_package_passes_row(ns, correct, specs, base_pkg)
            base_state = _v27_state_from_base_pkg(base_pkg, htot, vtot, ftot)
            try:
                correct_idx = {r:i for i,r in enumerate(frame_rows)}.get(correct)
            except Exception:
                correct_idx = None
            print(f'    V21A-bas: rader={base_state.frame_count} · hist={base_state.hist_hit}/{htot} · val={base_state.val_hit}/{vtot} · facit={"JA" if base_pass else "NEJ"}', flush=True)
            # Single-audit för de två första relevanta varianterna; cfg strict ger en läsbar kandidatlista.
            audit_cfg = dict(V27_VARIANT_DEFS['TRIM_VAL2_2500'])
            audit_rows.extend(_v27_single_trim_audit_rows(base_state, cand_v, audit_cfg, test_date, correct_idx=correct_idx)[:60])
            for variant_id in variants:
                t0=time.time()
                try:
                    if variant_id == 'V21A_BASE':
                        pkg, meta = _v27_greedy_trim_package(ns, frame_rows, specs, cand_v, base_pkg, correct, htot, vtot, ftot, hist_payout, args, variant_id)
                        pkg_pass = bool(base_pass)
                        fail_reason = 'OK' if pkg_pass else str(base_fail)
                        diag = '' if pkg_pass else _v15_diagnose_facit(ns, correct, specs, base_pkg)
                    else:
                        pkg, meta = _v27_greedy_trim_package(ns, frame_rows, specs, cand_v, base_pkg, correct, htot, vtot, ftot, hist_payout, args, variant_id)
                        pkg_pass, fail = _v15_package_passes_row(ns, correct, specs, pkg)
                        fail_reason = 'OK' if pkg_pass else str(fail)
                        diag = '' if pkg_pass else _v15_diagnose_facit(ns, correct, specs, pkg)
                    counts = _v15_group_counts(pkg)
                    extra = {
                        'V21A basrader': meta.get('base_rows'),
                        'Target rader': meta.get('target_rows'),
                        'Trim aktiv': 'Ja' if meta.get('trim_active') else 'Nej',
                        'Audit/Oracle': 'Ja' if meta.get('audit_mode') else 'Nej',
                        'Tillagda trimfilter': meta.get('added_trim_filters'),
                        'Trim kandidatpool': meta.get('candidate_pool'),
                        'Evaluerade single-trim': meta.get('evaluated_single'),
                        'Bästa single efter rader': meta.get('best_lossless_single_rows'),
                        'Bästa single sparar': meta.get('best_lossless_single_removed'),
                        'Bästa single filter': meta.get('best_lossless_single_name'),
                    }
                    out.append({
                        'Variant': variant_id,
                        'Variantnamn': V27_VARIANT_DEFS[variant_id]['label'],
                        'Datum': str(test_date)[:10],
                        'Utdelning': int(round(payout)),
                        'Facit': correct,
                        'Status': 'OK',
                        'Paket klarar facit': 'Ja' if pkg_pass else 'Nej',
                        'Orsak': fail_reason,
                        'Missdiagnos JSON': diag,
                        'Liknande historik': len(sim_df),
                        'Bred validering': len(wide_df),
                        'Paketträff': f"{pkg['hist_hit']}/{pkg['hist_total']}",
                        'Valideringsträff': f"{pkg['val_hit']}/{pkg['val_total']}",
                        'Grundram rader': len(frame_rows),
                        'Paketrader': int(pkg['frame_after']),
                        'Reducerar %': round(float(pkg['reduction_pct']), 2),
                        'Gemensamt score': round(float(pkg['joint_score']), 4),
                        'Utdelningsriktning %': round(float(pkg.get('payout_direction_pct', 0.0)), 2),
                        'Borttagna hist omgångar': int(pkg.get('removed_hist_count', 0)),
                        'Medel spret-gap': round(float(pkg.get('cluster_mean', 0.0)), 3),
                        'Filter totalt': int(pkg.get('num_filters', 0)),
                        'Strukturfilter': int(pkg.get('structure_filters', 0)),
                        'Profilfilter': int(pkg.get('profile_filters', 0)),
                        'FAT/ABC-filter': int(pkg.get('fat_filters', 0)),
                        'Värde/favorit/skräll': int(pkg.get('edge_filters', 0)),
                        **counts,
                        'Tillgängliga gruppkandidater': int(len(group_cands)),
                        'Filter JSON': _v13_json_filters(pkg),
                        'Gruppfilter JSON': _v15_group_json(pkg),
                        'Steg JSON': json.dumps(pkg.get('steps', []), ensure_ascii=False),
                        'Supersvår testomgång': 'Ja' if bool(test_row.get('_BT_No13Winner', False)) else 'Nej',
                        **extra,
                        'Sekunder': round(time.time()-t0, 2),
                    })
                except Exception as e:
                    out.append({'Variant': variant_id, 'Variantnamn': V27_VARIANT_DEFS.get(variant_id,{}).get('label', variant_id), 'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct, 'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e), 'Traceback': traceback.format_exc(), 'Sekunder': round(time.time()-t0,2)})
                    print(f'  FEL V27 variant {variant_id}: {e}', flush=True)
        except Exception as e:
            for variant_id in variants:
                out.append({'Variant': variant_id, 'Variantnamn': V27_VARIANT_DEFS.get(variant_id,{}).get('label', variant_id), 'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct, 'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e), 'Traceback': traceback.format_exc(), 'Sekunder': round(time.time()-t0_case,2)})
            print(f'  FEL V27 testfall: {e}', flush=True)
    return pd.DataFrame(out), {'supersvar_count': supersvar_count, 'eligible_rounds': len(db), 'variants': variants, 'trim_audit': pd.DataFrame(audit_rows)}


def _v27_summarize(detail):
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return pd.DataFrame()
    ok = detail[detail['Status'].astype(str).eq('OK')].copy()
    if ok.empty:
        return pd.DataFrame()
    rows=[]
    for variant, grp in ok.groupby('Variant', sort=False):
        hits = grp['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1']).sum()
        n=len(grp)
        med=lambda c: float(pd.to_numeric(grp[c], errors='coerce').dropna().median()) if c in grp and not pd.to_numeric(grp[c], errors='coerce').dropna().empty else float('nan')
        rows.append({
            'Variant': variant,
            'Variantnamn': V27_VARIANT_DEFS.get(str(variant),{}).get('label', str(variant)),
            'Testade omgångar': n,
            'Träffar': int(hits),
            'Träff %': round(100.0*hits/max(1,n),1),
            'Median paketrader': round(med('Paketrader'),1),
            'Medel paketrader': round(float(pd.to_numeric(grp['Paketrader'], errors='coerce').dropna().mean()),1),
            'Median reducering %': round(med('Reducerar %'),1),
            'Median V21A basrader': round(med('V21A basrader'),1),
            'Median tillagda trimfilter': round(med('Tillagda trimfilter'),1),
            'Trim aktiva fall': int(grp['Trim aktiv'].astype(str).str.lower().isin(['ja','true','1']).sum()) if 'Trim aktiv' in grp else 0,
            'Audit/Oracle': 'JA' if grp.get('Audit/Oracle', pd.Series(dtype=str)).astype(str).str.lower().isin(['ja','true','1']).any() else '',
            'Fel/hoppade': int((grp['Status'].astype(str) != 'OK').sum()),
        })
    summary=pd.DataFrame(rows)
    if not summary.empty:
        # Produktionsvarianter rankas före audit vid samma träff; audit markeras men kan ändå visa potential.
        summary['_audit_sort'] = summary['Audit/Oracle'].astype(str).eq('JA').astype(int)
        summary = summary.sort_values(['Träffar','_audit_sort','Median paketrader','Median reducering %'], ascending=[False, True, True, False]).drop(columns=['_audit_sort'])
        summary['Vinnare']=''
        summary.iloc[0, summary.columns.get_loc('Vinnare')]='JA'
    return summary


def _v27_write_outputs(detail, meta, args, out_dir, app_file, db_file):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary=_v27_summarize(detail)
    detail_path=out_dir/f'{args.output_prefix}_detail.csv'
    summary_path=out_dir/f'{args.output_prefix}_summary.csv'
    report_path=out_dir/f'{args.output_prefix}_report.txt'
    winner_path=out_dir/f'{args.output_prefix}_winner_detail.csv'
    audit_path=out_dir/f'{args.output_prefix}_TRIM_AUDIT.csv'
    detail.to_csv(detail_path,index=False)
    summary.to_csv(summary_path,index=False)
    winner=None
    if isinstance(summary,pd.DataFrame) and not summary.empty:
        winner=str(summary.iloc[0]['Variant'])
        detail[detail['Variant'].astype(str).eq(winner)].to_csv(winner_path,index=False)
    else:
        pd.DataFrame().to_csv(winner_path,index=False)
    ar = meta.get('trim_audit') if isinstance(meta,dict) else None
    if isinstance(ar,pd.DataFrame) and not ar.empty:
        ar.to_csv(audit_path,index=False)
    else:
        pd.DataFrame().to_csv(audit_path,index=False)
    lines=[
        'TIPSET AI – V27 SAFE TRIM AUDIT',
        '='*90,
        f'Appfil: {Path(app_file).name}',
        f'Databas: {Path(db_file).name}',
        f'Ram: {args.frame_profile} = {_v17_frame_rows_count(args.frame_profile)} rader',
        f'Testomgångar: {args.max_tests}',
        f'Varianter: {args.variants}',
        '',
        'METOD', '-'*90,
        'V21A byggs först. Produktionsvarianter adderar bara extra trimfilter som håller historik/validering enligt respektive risknivå.',
        'AUDIT_LOSSLESS-varianter är facit-skyddade i efterhand och används bara för att mäta om lossless trim finns teoretiskt.',
        'Detta är inte en ny global radscore, utan en platådiagnos runt V21A-basramen.',
        '', 'SAMMANFATTNING', '-'*90,
        summary.to_string(index=False) if isinstance(summary,pd.DataFrame) and not summary.empty else '(tom)',
        '', f'Detail: {detail_path}', f'Summary: {summary_path}', f'Winner detail: {winner_path}', f'Trim audit: {audit_path}',
    ]
    if winner:
        lines.append(f'Vinnare: {winner}')
    report_path.write_text('\n'.join(lines),encoding='utf-8')
    print('\nKLART – V27 SAFE TRIM AUDIT', flush=True)
    if isinstance(summary,pd.DataFrame) and not summary.empty:
        print(summary.to_string(index=False), flush=True)
    print('Summary:', summary_path, flush=True)
    print('Detail:', detail_path, flush=True)
    print('Trim audit:', audit_path, flush=True)
    return {'summary':summary_path,'detail':detail_path,'winner_detail':winner_path,'trim_audit':audit_path,'report':report_path}


def _v27_collect_frames(out_dir: Path, base_prefix: str, frames: list):
    outs={}
    for suffix in ['summary','detail','winner_detail','TRIM_AUDIT']:
        dfs=[]
        for frame in frames:
            p=Path(out_dir)/f'{base_prefix}_{_v17_safe_prefix_part(frame)}_{suffix}.csv'
            if p.exists():
                try:
                    df=pd.read_csv(p)
                    if not df.empty:
                        if 'Ramprofil' not in df.columns:
                            df.insert(0,'Ramprofil',frame)
                        if 'Teoretisk grundram' not in df.columns and suffix in {'summary','detail','winner_detail'}:
                            df.insert(1,'Teoretisk grundram',_v17_frame_rows_count(frame))
                        dfs.append(df)
                except Exception:
                    pass
        if dfs:
            comb=pd.concat(dfs,ignore_index=True)
            outp=Path(out_dir)/f'{base_prefix}_ALL_FRAMES_{suffix}.csv'
            comb.to_csv(outp,index=False)
            outs[suffix]=comb
    return outs


def main_v27():
    parser=argparse.ArgumentParser(description='Tipset AI V27 – V21A safe trim audit.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_safe_trim_audit_v27_test10')
    parser.add_argument('--variants', default='V21A_BASE,TRIM_STRICT2500,TRIM_VAL1_2500,TRIM_VAL2_2500,TRIM_9OF10_2500,TRIM_STRICT2400,AUDIT_LOSSLESS2500,AUDIT_LOSSLESS2300')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='3-5-5')
    parser.add_argument('--frames', default='3-5-5')
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--candidate-min-hit', type=int, default=27)
    parser.add_argument('--min-candidate-val-pct', type=float, default=68.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=82.0)
    parser.add_argument('--min-gap-score', type=float, default=0.75)
    parser.add_argument('--variants-per-key', type=int, default=6)
    parser.add_argument('--max-candidates', type=int, default=340)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    # V21A-basargument
    parser.add_argument('--min-hit', type=int, default=26)
    parser.add_argument('--min-package-val-pct', type=float, default=78.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=90.0)
    parser.add_argument('--min-unique-rows', type=int, default=50)
    parser.add_argument('--beam-width', type=int, default=30)
    parser.add_argument('--archive-width', type=int, default=220)
    parser.add_argument('--structure-seed-count', type=int, default=2)
    parser.add_argument('--hit-power', type=float, default=3.0)
    parser.add_argument('--validation-power', type=float, default=0.7)
    parser.add_argument('--reduction-power', type=float, default=2.7)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.25)
    parser.add_argument('--payout-direction-weight', type=float, default=0.16)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=50)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=350)
    parser.add_argument('--per-bucket-keep', type=int, default=1)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--v15-group-max-filters', type=int, default=18)
    parser.add_argument('--v15-max-group-candidates', type=int, default=120)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=3)
    parser.add_argument('--v15-cross-max-filters', type=int, default=20)
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    locked_variants='V21A_BASE,TRIM_STRICT2500,TRIM_VAL1_2500,TRIM_VAL2_2500,TRIM_9OF10_2500,TRIM_STRICT2400,AUDIT_LOSSLESS2500,AUDIT_LOSSLESS2300'
    if str(args.variants).strip() != locked_variants:
        print('V27: ignorerar inkommande --variants och kör låst safe-trim-lista.', flush=True)
    args.variants=locked_variants
    if str(args.output_prefix).strip() != 'package_safe_trim_audit_v27_test10':
        print('V27: ignorerar inkommande --output-prefix och använder package_safe_trim_audit_v27_test10.', flush=True)
    args.output_prefix='package_safe_trim_audit_v27_test10'
    frames=_v17_parse_frames(args.frames)
    variants=_v27_parse_variants(args.variants)
    args.variants=','.join(variants)
    app_file, db_file = _resolve_required_files(args)
    out_dir=Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir=Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n' + '='*96, flush=True)
    print('TIPSET AI V27 – SAFE TRIM AUDIT', flush=True)
    print('LÅST LOGIK: V21A först; bara extra trim ovanpå V21A. Produktionsvarianter använder inte facit.', flush=True)
    print('AUDIT_LOSSLESS-varianter är facit-skyddade och visar teoretisk lossless-potential, inte produktionsmotor.', flush=True)
    print(f'Ramar: {", ".join(frames)}', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)
    for frame in frames:
        frame_args=argparse.Namespace(**vars(args))
        frame_args.frames=frame
        frame_args.frame_profile=frame
        frame_args.output_prefix=f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n' + '-'*96, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*96, flush=True)
        ns=_load_app_functions(app_file, fast_no_supermakro=False)
        db=ns['load_database'](str(db_file),13)
        detail, meta = _v27_run_backtest(ns, db, frame_args)
        _v27_write_outputs(detail, meta, frame_args, out_dir, app_file, db_file)
        del ns, db, detail
        gc.collect()
    combined=_v27_collect_frames(out_dir,args.output_prefix,frames)
    if combined.get('summary') is not None:
        print('\nKLART – V27 ALLA RAMAR', flush=True)
        print(combined['summary'].to_string(index=False), flush=True)
        print('Summary:', str(Path(out_dir)/f'{args.output_prefix}_ALL_FRAMES_summary.csv'), flush=True)



# =============================================================================
# V29b – HÅRDLÅST 30-TEST HUVUDMOTOR: successiv riskstege ovanpå V21A-basramen
# =============================================================================
# Syfte: Vi har hittat en säker mellannivå runt V21A_BASE. V29 testar inte en ny
# huvudmotor, utan var träffsäkerheten börjar spricka när V21A-basramen skärs
# successivt mot lägre radmål. Detta är en beslutskurva: 2700 → 2650 → 2600 →
# 2550 → 2500 → 2400 → 2300 → 2000.

V28_VARIANT_DEFS = {
    'V21A_BASE': {'label': 'V21A_BASE säker basram', 'top_rows': None, 'score_mode': 'base', 'risk_level': 0},
    'STEP2700': {'label': 'STEP2700 försiktig slicer', 'top_rows': 2700, 'score_mode': 'plain', 'risk_level': 1},
    'STEP2650': {'label': 'STEP2650 försiktig slicer', 'top_rows': 2650, 'score_mode': 'plain', 'risk_level': 2},
    'STEP2600': {'label': 'STEP2600 mellanläge', 'top_rows': 2600, 'score_mode': 'plain', 'risk_level': 3},
    'STEP2550': {'label': 'STEP2550 mellanläge', 'top_rows': 2550, 'score_mode': 'plain', 'risk_level': 4},
    'STEP2500': {'label': 'STEP2500 målzon övre', 'top_rows': 2500, 'score_mode': 'plain', 'risk_level': 5},
    'STEP2450': {'label': 'STEP2450 målzon nedre', 'top_rows': 2450, 'score_mode': 'plain', 'risk_level': 6},
    'STEP2400': {'label': 'STEP2400 pressad målzon', 'top_rows': 2400, 'score_mode': 'plain', 'risk_level': 7},
    'STEP2300': {'label': 'STEP2300 risknivå', 'top_rows': 2300, 'score_mode': 'plain', 'risk_level': 8},
    'STEP2000': {'label': 'STEP2000 slutmål/stresstest', 'top_rows': 2000, 'score_mode': 'plain', 'risk_level': 9},
}

# Gör V29-varianterna tillgängliga för den befintliga V26-byggaren.
for _v28_key, _v28_cfg in V28_VARIANT_DEFS.items():
    if _v28_key == 'V21A_BASE':
        V26_VARIANT_DEFS[_v28_key] = dict(V26_VARIANT_DEFS.get('V21A_BASE', {}), label=_v28_cfg['label'])
    else:
        V26_VARIANT_DEFS[_v28_key] = {
            'label': _v28_cfg['label'],
            'description': f"V29 30-test stepdown: V21A först; om basramen är större än {_v28_cfg['top_rows']} väljs topp {_v28_cfg['top_rows']} inom V21A.",
            'mode': 'hybrid',
            'top_rows': int(_v28_cfg['top_rows']),
            'score_mode': str(_v28_cfg.get('score_mode', 'plain')),
        }


def _v28_parse_variants(value):
    default = list(V28_VARIANT_DEFS.keys())
    allowed = set(V28_VARIANT_DEFS)
    raw = str(value or ','.join(default)).replace(';', ',').replace('+', ',').split(',')
    out=[]
    for x in raw:
        k=x.strip().upper()
        if not k:
            continue
        if k in {'BASE','V21A','RADPRESS'}:
            k='V21A_BASE'
        if k in allowed and k not in out:
            out.append(k)
        elif k:
            print(f'V29: ignorerar okänd variant {k}', flush=True)
    if len(out) <= 1 and (not out or out[0] == 'V21A_BASE'):
        return default
    return out or default


def _v28_run_backtest(ns: dict, global_db: pd.DataFrame, args):
    antal_matcher = 13
    norm = ns['normalize_single_row_text']
    similar_history = ns['_similar_history_for_backtest']
    ranked_frame = ns['_ranked_frame_like_widths']
    generate_rows_from_frame = ns['generate_rows_from_frame']
    build_clean_filter_specs = ns['build_clean_filter_specs']
    global _ACTIVE_V9_NS
    _ACTIVE_V9_NS = ns
    global_db, supersvar_count = _prepare_supersvar_payout(global_db, int(args.pay_max))
    db = _prepare_db(ns, global_db, antal_matcher, int(args.pay_min), int(args.pay_max))
    if db.empty:
        return pd.DataFrame(), {'error': 'Inga giltiga omgångar i utdelningsintervallet.'}
    variants = _v28_parse_variants(args.variants)
    spik, halv, hel = _parse_frame_profile(args.frame_profile)
    template = _base_frame(spik, halv, hel)
    test_rows = list(db.iterrows())[int(args.test_offset):int(args.test_offset)+int(args.max_tests)]
    out=[]; filter_rows=[]
    for ti, (idx, test_row) in enumerate(test_rows, 1):
        _release_test_memory()
        t0_case = time.time()
        correct = norm(test_row.get('Correct_Row',''))
        input_vec = test_row.get('Prob_Vector', [])
        test_date = test_row.get('Datum', None)
        payout = float(pd.to_numeric(pd.Series([test_row.get('Payout',0)]), errors='coerce').fillna(0).iloc[0])
        print(f'V29 testfall {ti}/{len(test_rows)} · {str(test_date)[:10]} · utdelning {payout:,.0f} kr'.replace(',', ' '), flush=True)
        try:
            sim_df = similar_history(global_db, input_vec, antal_matcher, top_n=int(args.top_n), pay_min=int(args.pay_min), pay_max=int(args.pay_max), exclude_index=idx, mode=str(args.mode), test_date=test_date)
            wide_df = similar_history(global_db, input_vec, antal_matcher, top_n=max(int(args.top_n), int(args.wide_n)), pay_min=int(args.pay_min), pay_max=int(args.pay_max), exclude_index=idx, mode=str(args.mode), test_date=test_date)
            if len(sim_df) < int(args.top_n):
                raise RuntimeError(f'För få liknande omgångar: {len(sim_df)} av önskade {args.top_n}.')
            engine_frame = ranked_frame(template, input_vec, antal_matcher)
            frame_rows, _, ok, msg = generate_rows_from_frame(engine_frame, max_rows=None)
            if not ok or not frame_rows:
                raise RuntimeError(f'Kunde inte skapa ram: {msg}')
            specs = build_clean_filter_specs(sim_df, input_vec, antal_matcher, slider_u_count=3, target_hist_pct=int(args.filter_hist_target_pct), u_rows=None, hist_df=global_db, max_shock_pct=22, candidate_rows=frame_rows, include_supermakro=not bool(args.fast_no_supermakro))
            candidates, htot, vtot, ftot, hist_payout = _build_dynamic_candidates_v9(
                ns, specs, sim_df, frame_rows, engine_frame, antal_matcher,
                profile_min_hit=int(args.candidate_min_hit), variants_per_key=int(args.variants_per_key), max_candidates=int(args.max_candidates),
                validation_df=wide_df, min_candidate_val_pct=float(args.min_candidate_val_pct), min_structure_val_pct=float(args.min_structure_val_pct),
                min_gap_score=float(args.min_gap_score), frame_adapt=True,
            )
            candidates = _v13_enrich_candidates(candidates, hist_payout, htot)
            vargs = _v15_variant_args(args, 'V21A')
            cand0 = _v15_filter_candidates_for_variant(candidates, vargs, htot)
            cand_v, group_cands = _v15_transform_candidates(cand0, htot, vtot, ftot, hist_payout, vargs, 'V21A')
            rowscore_pool = cand_v
            print(f'    V29 bank: singlar={len(candidates)} · med grupper={len(rowscore_pool)} · successiv V21A-slicer', flush=True)
            base_pkg, base_meta = _build_cluster_payout_package_v13_from_candidates(ns, rowscore_pool, htot, vtot, ftot, hist_payout, frame_rows, engine_frame, antal_matcher, vargs, 'V21A')
            if base_pkg is None:
                raise RuntimeError(base_meta.get('error','Inget V21A-baspaket'))
            base_pass, base_fail = _v15_package_passes_row(ns, correct, specs, base_pkg)
            base_mask = _v26_bits_from_package(base_pkg, len(frame_rows))
            base_rows = int(base_mask.sum())
            print(f'    V21A-bas: rader={base_rows} · facit={"JA" if base_pass else "NEJ"}', flush=True)
            for variant_id in variants:
                t0 = time.time()
                try:
                    if variant_id == 'V21A_BASE':
                        cfg = dict(V26_VARIANT_DEFS['V21A_BASE'])
                    else:
                        cfg = dict(V26_VARIANT_DEFS[variant_id])
                    pkg, meta = _v26_build_hybrid_package(ns, frame_rows, specs, rowscore_pool, base_pkg, base_pass, correct, cfg, variant_id)
                    if pkg is None:
                        raise RuntimeError(meta.get('error','Inget V29-paket'))
                    if variant_id == 'V21A_BASE':
                        pkg_pass = bool(base_pass)
                        fail_reason = 'OK' if pkg_pass else str(base_fail)
                        diag = '' if pkg_pass else _v15_diagnose_facit(ns, correct, specs, base_pkg)
                    else:
                        pkg_pass = bool(pkg.get('v26_meta',{}).get('correct_selected', False))
                        if pkg_pass:
                            fail_reason = 'OK'
                            diag = ''
                        else:
                            rank = pkg.get('v26_meta',{}).get('correct_rank_inside_base')
                            topn = pkg.get('v26_meta',{}).get('final_rows')
                            fail_reason = 'V21A-basen fångade inte facit' if not base_pass else f'Facitrank inom V21A {rank} > topp {topn}'
                            diag = json.dumps(pkg.get('v26_meta',{}), ensure_ascii=False)
                    counts = _v15_group_counts(base_pkg)
                    top_target = V28_VARIANT_DEFS.get(variant_id,{}).get('top_rows')
                    final_rows = int(pkg.get('frame_after', 0))
                    over_under = '' if top_target is None else int(final_rows) - int(top_target)
                    extra = {
                        'Stegnivå': V28_VARIANT_DEFS.get(variant_id,{}).get('risk_level', 0),
                        'Målrad': top_target,
                        'Över/under mål': over_under,
                        'V21A basrader': pkg.get('v26_meta',{}).get('base_rows'),
                        'Kapade från V21A': int(max(0, int(pkg.get('v26_meta',{}).get('base_rows') or 0) - final_rows)),
                        'Slicer aktiv': 'Ja' if pkg.get('v26_meta',{}).get('sliced') else 'Nej',
                        'Facitrank inom V21A': pkg.get('v26_meta',{}).get('correct_rank_inside_base'),
                        'Facitscore': pkg.get('v26_meta',{}).get('correct_score'),
                        'Cutoff score': pkg.get('v26_meta',{}).get('cutoff_score'),
                        'Kandidat-röster': pkg.get('v26_meta',{}).get('candidate_votes'),
                        'Facit familjscore JSON': pkg.get('v26_meta',{}).get('correct_family_json'),
                        'V29 topfilter JSON': pkg.get('v26_meta',{}).get('top_filters_json',''),
                    }
                    out.append({
                        'Variant': variant_id,
                        'Variantnamn': V28_VARIANT_DEFS[variant_id]['label'],
                        'Datum': str(test_date)[:10],
                        'Utdelning': int(round(payout)),
                        'Facit': correct,
                        'Status': 'OK',
                        'Paket klarar facit': 'Ja' if pkg_pass else 'Nej',
                        'Orsak': 'OK' if pkg_pass else fail_reason,
                        'Missdiagnos JSON': diag,
                        'Liknande historik': len(sim_df),
                        'Bred validering': len(wide_df),
                        'Paketträff': f"{pkg['hist_hit']}/{pkg['hist_total']}",
                        'Valideringsträff': f"{pkg['val_hit']}/{pkg['val_total']}",
                        'Grundram rader': len(frame_rows),
                        'Paketrader': final_rows,
                        'Reducerar %': round(float(pkg['reduction_pct']), 2),
                        'Gemensamt score': round(float(pkg['joint_score']), 4),
                        'Utdelningsriktning %': round(float(pkg.get('payout_direction_pct', 0.0)), 2),
                        'Borttagna hist omgångar': int(pkg.get('removed_hist_count', 0)),
                        'Medel spret-gap': round(float(pkg.get('cluster_mean', 0.0)), 3),
                        'Filter totalt': int(pkg.get('num_filters', 0)),
                        'Strukturfilter': int(pkg.get('structure_filters', 0)),
                        'Profilfilter': int(pkg.get('profile_filters', 0)),
                        'FAT/ABC-filter': int(pkg.get('fat_filters', 0)),
                        'Värde/favorit/skräll': int(pkg.get('edge_filters', 0)),
                        **counts,
                        'Tillgängliga gruppkandidater': int(len(group_cands)),
                        'Filter JSON': _v13_json_filters(base_pkg),
                        'Gruppfilter JSON': _v15_group_json(base_pkg),
                        'Steg JSON': json.dumps(pkg.get('steps', []), ensure_ascii=False),
                        'Supersvår testomgång': 'Ja' if bool(test_row.get('_BT_No13Winner', False)) else 'Nej',
                        **extra,
                        'Sekunder': round(time.time()-t0, 2),
                    })
                    if variant_id != 'V21A_BASE':
                        try:
                            tf = json.loads(extra.get('V29 topfilter JSON') or '[]')
                            for rank, row in enumerate(tf[:25],1):
                                filter_rows.append({'Datum': str(test_date)[:10], 'Variant': variant_id, 'Filterrank': rank, **row})
                        except Exception:
                            pass
                except Exception as e:
                    out.append({'Variant': variant_id, 'Variantnamn': V28_VARIANT_DEFS.get(variant_id,{}).get('label', variant_id), 'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct, 'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e), 'Traceback': traceback.format_exc(), 'Sekunder': round(time.time()-t0,2)})
                    print(f'  FEL V29 variant {variant_id}: {e}', flush=True)
        except Exception as e:
            for variant_id in variants:
                out.append({'Variant': variant_id, 'Variantnamn': V28_VARIANT_DEFS.get(variant_id,{}).get('label', variant_id), 'Datum': str(test_date)[:10], 'Utdelning': int(round(payout)), 'Facit': correct, 'Status': 'Fel', 'Paket klarar facit': 'Nej', 'Orsak': str(e), 'Traceback': traceback.format_exc(), 'Sekunder': round(time.time()-t0_case,2)})
            print(f'  FEL V29 testfall: {e}', flush=True)
    return pd.DataFrame(out), {'supersvar_count': supersvar_count, 'eligible_rounds': len(db), 'variants': variants, 'filter_rows': pd.DataFrame(filter_rows)}


def _v28_summarize(detail):
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return pd.DataFrame()
    ok = detail[detail['Status'].astype(str).eq('OK')].copy()
    if ok.empty:
        return pd.DataFrame()
    rows=[]
    for variant, grp in ok.groupby('Variant', sort=False):
        hits = int(grp['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1']).sum())
        n = int(len(grp))
        med=lambda c: float(pd.to_numeric(grp[c], errors='coerce').dropna().median()) if c in grp and not pd.to_numeric(grp[c], errors='coerce').dropna().empty else float('nan')
        misses = grp.loc[~grp['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1']), 'Datum'].astype(str).str[:10].tolist()
        rows.append({
            'Variant': variant,
            'Variantnamn': V28_VARIANT_DEFS.get(str(variant),{}).get('label', str(variant)),
            'Stegnivå': V28_VARIANT_DEFS.get(str(variant),{}).get('risk_level', 0),
            'Målrad': V28_VARIANT_DEFS.get(str(variant),{}).get('top_rows'),
            'Testade omgångar': n,
            'Träffar': hits,
            'Träff %': round(100.0*hits/max(1,n),1),
            'Median paketrader': round(med('Paketrader'),1),
            'Medel paketrader': round(float(pd.to_numeric(grp['Paketrader'], errors='coerce').dropna().mean()),1),
            'Median reducering %': round(med('Reducerar %'),1),
            'Median V21A basrader': round(med('V21A basrader'),1),
            'Median kapade från V21A': round(med('Kapade från V21A'),1),
            'Slicer aktiva fall': int(grp['Slicer aktiv'].astype(str).str.lower().isin(['ja','true','1']).sum()) if 'Slicer aktiv' in grp else 0,
            'Median facitrank inom V21A': round(med('Facitrank inom V21A'),1),
            'Missdatum': ', '.join(misses[:5]),
            'Fel/hoppade': int((grp['Status'].astype(str) != 'OK').sum()),
        })
    summary=pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(['Träffar','Median paketrader','Stegnivå'], ascending=[False, True, True])
        summary['Vinnare']=''
        summary.iloc[0, summary.columns.get_loc('Vinnare')]='JA'
        # Tydlig beslutsnivå: lägsta radnivå som höll 10/10 samt lägsta som höll minst 9/10.
        nmax = int(summary['Testade omgångar'].max()) if 'Testade omgångar' in summary else 0
        summary['Beslutsnotis']=''
        ten = summary[(summary['Träffar'] == nmax) & summary['Variant'].ne('V21A_BASE')]
        if not ten.empty:
            best10 = ten.sort_values(['Median paketrader','Stegnivå'], ascending=[True, True]).iloc[0]['Variant']
            summary.loc[summary['Variant'].eq(best10), 'Beslutsnotis'] = 'Lägsta 10/10 i stegen'
        nine = summary[(summary['Träffar'] >= max(0, nmax-1)) & summary['Variant'].ne('V21A_BASE')]
        if not nine.empty:
            best9 = nine.sort_values(['Median paketrader','Stegnivå'], ascending=[True, True]).iloc[0]['Variant']
            if not summary.loc[summary['Variant'].eq(best9), 'Beslutsnotis'].astype(str).str.len().any():
                summary.loc[summary['Variant'].eq(best9), 'Beslutsnotis'] = 'Lägsta ≥9/10 i stegen'
    return summary


def _v28_write_outputs(detail, meta, args, out_dir, app_file, db_file):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary=_v28_summarize(detail)
    detail_path=out_dir/f'{args.output_prefix}_detail.csv'
    summary_path=out_dir/f'{args.output_prefix}_summary.csv'
    report_path=out_dir/f'{args.output_prefix}_report.txt'
    winner_path=out_dir/f'{args.output_prefix}_winner_detail.csv'
    votes_path=out_dir/f'{args.output_prefix}_FILTER_VOTES.csv'
    detail.to_csv(detail_path,index=False)
    summary.to_csv(summary_path,index=False)
    winner=None
    if isinstance(summary,pd.DataFrame) and not summary.empty:
        winner=str(summary.iloc[0]['Variant'])
        detail[detail['Variant'].astype(str).eq(winner)].to_csv(winner_path,index=False)
    else:
        pd.DataFrame().to_csv(winner_path,index=False)
    fr = meta.get('filter_rows') if isinstance(meta,dict) else None
    if isinstance(fr,pd.DataFrame) and not fr.empty:
        fr.to_csv(votes_path,index=False)
    else:
        pd.DataFrame().to_csv(votes_path,index=False)
    lines=[
        'TIPSET AI – V29b HÅRDLÅST 30-TEST HUVUDMOTOR',
        '='*90,
        f'Appfil: {Path(app_file).name}',
        f'Databas: {Path(db_file).name}',
        f'Ram: {args.frame_profile} = {_v17_frame_rows_count(args.frame_profile)} rader',
        f'Testomgångar: {args.max_tests}',
        f'Varianter: {args.variants}',
        '',
        'METOD', '-'*90,
        'V21A byggs först. Varje STEP-variant skär endast inuti V21A-ramen om basramen är större än stegets målrad.',
        'Detta är ett stabilitetstest över 30 omgångar. V21A_BASE är huvudkandidat; stepdown-nivåerna visar endast var träffen börjar brista.',
        '', 'SAMMANFATTNING', '-'*90,
        summary.to_string(index=False) if isinstance(summary,pd.DataFrame) and not summary.empty else '(tom)',
        '', f'Detail: {detail_path}', f'Summary: {summary_path}', f'Winner detail: {winner_path}', f'Filter votes: {votes_path}',
    ]
    if isinstance(summary,pd.DataFrame) and not summary.empty:
        nmax = int(summary['Testade omgångar'].max())
        ten = summary[(summary['Träffar'] == nmax) & summary['Variant'].ne('V21A_BASE')]
        nine = summary[(summary['Träffar'] >= max(0,nmax-1)) & summary['Variant'].ne('V21A_BASE')]
        lines += ['', 'BESLUTSKURVA', '-'*90]
        if not ten.empty:
            r = ten.sort_values(['Median paketrader','Stegnivå'], ascending=[True, True]).iloc[0]
            lines.append(f'Lägsta STEP med 10/10: {r["Variant"]} · median {r["Median paketrader"]} rader')
        else:
            lines.append('Ingen STEP-variant höll 10/10. V21A_BASE är säker nivå.')
        if not nine.empty:
            r = nine.sort_values(['Median paketrader','Stegnivå'], ascending=[True, True]).iloc[0]
            lines.append(f'Lägsta STEP med minst 9/10: {r["Variant"]} · median {r["Median paketrader"]} rader')
    if winner:
        lines.append(f'Vinnare: {winner}')
    report_path.write_text('\n'.join(lines),encoding='utf-8')
    print('\nKLART – V29b HÅRDLÅST 30-TEST HUVUDMOTOR', flush=True)
    if isinstance(summary,pd.DataFrame) and not summary.empty:
        print(summary.to_string(index=False), flush=True)
    print('Summary:', summary_path, flush=True)
    print('Detail:', detail_path, flush=True)
    print('Filter votes:', votes_path, flush=True)
    return {'summary':summary_path,'detail':detail_path,'winner_detail':winner_path,'filter_votes':votes_path,'report':report_path}


def _v28_collect_frames(out_dir: Path, base_prefix: str, frames: list):
    outs={}
    for suffix in ['summary','detail','winner_detail','FILTER_VOTES']:
        dfs=[]
        for frame in frames:
            p=Path(out_dir)/f'{base_prefix}_{_v17_safe_prefix_part(frame)}_{suffix}.csv'
            if p.exists():
                try:
                    df=pd.read_csv(p)
                    if not df.empty:
                        if 'Ramprofil' not in df.columns:
                            df.insert(0,'Ramprofil',frame)
                        if 'Teoretisk grundram' not in df.columns and suffix in {'summary','detail','winner_detail'}:
                            df.insert(1,'Teoretisk grundram',_v17_frame_rows_count(frame))
                        dfs.append(df)
                except Exception:
                    pass
        if dfs:
            comb=pd.concat(dfs,ignore_index=True)
            outp=Path(out_dir)/f'{base_prefix}_ALL_FRAMES_{suffix}.csv'
            comb.to_csv(outp,index=False)
            outs[suffix]=comb
    return outs


def main_v29():
    parser=argparse.ArgumentParser(description='Tipset AI V29 – 30-test huvudmotor + stepdown-kontroller.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_v21a_30test_v29b')
    parser.add_argument('--variants', default='V21A_BASE,STEP2700,STEP2600,STEP2500,STEP2300,STEP2000')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='3-5-5')
    parser.add_argument('--frames', default='3-5-5')
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--candidate-min-hit', type=int, default=27)
    parser.add_argument('--min-candidate-val-pct', type=float, default=68.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=82.0)
    parser.add_argument('--min-gap-score', type=float, default=0.75)
    parser.add_argument('--variants-per-key', type=int, default=6)
    parser.add_argument('--max-candidates', type=int, default=340)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    # V21A-basargument
    parser.add_argument('--min-hit', type=int, default=26)
    parser.add_argument('--min-package-val-pct', type=float, default=78.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=90.0)
    parser.add_argument('--min-unique-rows', type=int, default=50)
    parser.add_argument('--beam-width', type=int, default=30)
    parser.add_argument('--archive-width', type=int, default=220)
    parser.add_argument('--structure-seed-count', type=int, default=2)
    parser.add_argument('--hit-power', type=float, default=3.0)
    parser.add_argument('--validation-power', type=float, default=0.7)
    parser.add_argument('--reduction-power', type=float, default=2.7)
    parser.add_argument('--payout-weight', type=float, default=0.05)
    parser.add_argument('--cluster-weight', type=float, default=0.25)
    parser.add_argument('--payout-direction-weight', type=float, default=0.16)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=50)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=10)
    parser.add_argument('--row-bucket-size', type=int, default=350)
    parser.add_argument('--per-bucket-keep', type=int, default=1)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--v15-group-max-filters', type=int, default=18)
    parser.add_argument('--v15-max-group-candidates', type=int, default=120)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=3)
    parser.add_argument('--v15-cross-max-filters', type=int, default=20)
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)

    # V29b: hårdlås testfönstret. Tidigare V29 kunde ärva --max-tests 10
    # från gamla Colab-celler/miljöargument och blev därför bara ett 10-test.
    if int(getattr(args, 'max_tests', 30)) != 30:
        print(f'V29b: ignorerar inkommande --max-tests={args.max_tests} och kör exakt 30.', flush=True)
    args.max_tests = 30
    if int(getattr(args, 'test_offset', 0)) != 0:
        print(f'V29b: ignorerar inkommande --test-offset={args.test_offset} och startar vid 0.', flush=True)
    args.test_offset = 0
    args.top_n = 30
    args.wide_n = 35

    locked_variants='V21A_BASE,STEP2700,STEP2600,STEP2500,STEP2300,STEP2000'
    if str(args.variants).strip() != locked_variants:
        print('V29: ignorerar inkommande --variants och kör låst 30-testlista.', flush=True)
    args.variants=locked_variants
    if str(args.output_prefix).strip() != 'package_v21a_30test_v29b':
        print('V29b: ignorerar inkommande --output-prefix och använder package_v21a_30test_v29b.', flush=True)
    args.output_prefix='package_v21a_30test_v29b'
    frames=_v17_parse_frames(args.frames)
    variants=_v28_parse_variants(args.variants)
    args.variants=','.join(variants)
    app_file, db_file = _resolve_required_files(args)
    out_dir=Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir=Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n' + '='*96, flush=True)
    print('TIPSET AI V29b – HÅRDLÅST 30-TEST HUVUDMOTOR', flush=True)
    print('LÅST LOGIK: 30-test av V21A_BASE först; stepdown-varianter är endast kontrollkurva.')
    print('MÅL: verifiera om huvudmotorn håller över 30 omgångar innan mer radpress.', flush=True)
    print(f'Ramar: {", ".join(frames)}', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)
    for frame in frames:
        frame_args=argparse.Namespace(**vars(args))
        frame_args.frames=frame
        frame_args.frame_profile=frame
        frame_args.output_prefix=f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n' + '-'*96, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*96, flush=True)
        ns=_load_app_functions(app_file, fast_no_supermakro=False)
        db=ns['load_database'](str(db_file),13)
        detail, meta = _v28_run_backtest(ns, db, frame_args)
        _v28_write_outputs(detail, meta, frame_args, out_dir, app_file, db_file)
        del ns, db, detail
        gc.collect()
    combined=_v28_collect_frames(out_dir,args.output_prefix,frames)
    if combined.get('summary') is not None:
        print('\nKLART – V29b ALLA RAMAR', flush=True)
        print(combined['summary'].to_string(index=False), flush=True)
        print('Summary:', str(Path(out_dir)/f'{args.output_prefix}_ALL_FRAMES_summary.csv'), flush=True)




# =============================================================================
# V30 – TOLERANT REPAIR 30
# =============================================================================
# Syfte: 30-testet visade att V21A_BASE bara gav 26/30. V30 ska därför inte
# pressa rader, utan bredda/reparera paketet för ett spretigt utdelningsspann
# 100k–2.5M. Vi testar toleranta varianter som prioriterar träffsäkerhet före
# budget, med färre hårda kantdödare och mer grupp-/familjetolerans.

V30_VARIANT_DEFS = {
    'V21A_BASE': {
        'label': 'V21A_BASE kontroll 30',
        'description': 'Exakt V21A-logik som kontroll från V29b: RADPRESS utan hård Delta.',
        'base': 'V21A',
        'v30_mode': 'base',
        'overrides': {},
    },
    'TOL_GROUPS': {
        'label': 'TOL_GROUPS toleranta grupper',
        'description': 'Breddar struktur-/familjegrupper och minskar radpress. Filterfamiljer får missa mer, vilket passar 100k–1M+ spret.',
        'base': 'V21A',
        'v30_mode': 'groups',
        'structure_miss': [2, 3, 4],
        'family_miss': [1, 2, 3, 4, 5, 6],
        'cross_family': True,
        'block_fragile_individuals': True,
        'prefer_soft_groups': True,
        'v19_target_rows': 3800,
        'v19_min_active_families': 3,
        'v19_family_bonus': {'FAT': 0.45, 'ABC': 0.40, 'FAT-sekvens': 0.20, 'Värde/svårighet': 0.62, 'Poäng/rank': 0.58, 'Favorit': 0.46, 'Skräll': 0.46, 'Tvärfamilj': 0.35},
        'overrides': {
            'min_hit': 25,
            'candidate_min_hit': 27,
            'beam_width': 68,
            'archive_width': 520,
            'min_unique_rows': 28,
            'min_package_val_pct': 74.0,
            'min_candidate_val_pct': 72.0,
            'min_structure_package_val_pct': 86.0,
            'min_structure_val_pct': 78.0,
            'reduction_power': 1.62,
            'validation_power': 1.24,
            'hit_power': 3.6,
            'payout_direction_weight': 0.10,
            'cluster_weight': 0.18,
            'v15_group_max_filters': 22,
            'v15_max_group_candidates': 110,
            'v15_cross_per_family': 3,
            'row_bucket_size': 420,
            'per_bucket_keep': 3,
        },
    },
    'TOL_WIDE_FAT': {
        'label': 'TOL_WIDE_FAT bred FAT/ABC',
        'description': 'Behåller FAT/ABC som signal men gör dem mindre binära: hårda svaga FAT-delar stryps, bredare gruppvägar får dominera.',
        'base': 'V21A',
        'v30_mode': 'wide_fat',
        'structure_miss': [2, 3, 4],
        'family_miss': [1, 2, 3, 4, 5, 6],
        'cross_family': True,
        'block_fragile_individuals': True,
        'block_strict_max0_groups': True,
        'prefer_soft_groups': True,
        'v19_target_rows': 3900,
        'v19_min_active_families': 3,
        'v19_family_bonus': {'FAT': 0.42, 'ABC': 0.38, 'FAT-sekvens': 0.12, 'Värde/svårighet': 0.55, 'Poäng/rank': 0.56, 'Favorit': 0.48, 'Skräll': 0.48, 'Tvärfamilj': 0.36},
        'overrides': {
            'min_hit': 25,
            'candidate_min_hit': 27,
            'beam_width': 72,
            'archive_width': 560,
            'min_unique_rows': 24,
            'min_package_val_pct': 72.0,
            'min_candidate_val_pct': 70.0,
            'min_structure_package_val_pct': 85.0,
            'min_structure_val_pct': 76.0,
            'reduction_power': 1.45,
            'validation_power': 1.32,
            'hit_power': 3.85,
            'payout_direction_weight': 0.08,
            'cluster_weight': 0.15,
            'v15_group_max_filters': 24,
            'v15_max_group_candidates': 125,
            'v15_cross_per_family': 3,
            'row_bucket_size': 450,
            'per_bucket_keep': 3,
        },
    },
    'TOL_VALUE_SPLIT': {
        'label': 'TOL_VALUE_SPLIT två svårighetsprofiler',
        'description': 'Mer tolerans för att 100k och 1M+ kan ha olika värde-/skrällprofil. Mindre hård FAT, mer poäng/rank och skrällbalans.',
        'base': 'V21A',
        'v30_mode': 'value_split',
        'structure_miss': [2, 3, 4],
        'family_miss': [1, 2, 3, 4, 5, 6],
        'cross_family': True,
        'block_fragile_individuals': True,
        'block_delta_groups': False,
        'prefer_soft_groups': True,
        'v19_target_rows': 4000,
        'v19_min_active_families': 4,
        'v19_family_bonus': {'FAT': 0.32, 'ABC': 0.32, 'FAT-sekvens': 0.10, 'Värde/svårighet': 0.78, 'Poäng/rank': 0.72, 'Favorit': 0.58, 'Skräll': 0.62, 'Tvärfamilj': 0.42},
        'overrides': {
            'min_hit': 25,
            'candidate_min_hit': 27,
            'beam_width': 76,
            'archive_width': 620,
            'min_unique_rows': 22,
            'min_package_val_pct': 72.0,
            'min_candidate_val_pct': 70.0,
            'min_structure_package_val_pct': 85.0,
            'min_structure_val_pct': 76.0,
            'reduction_power': 1.35,
            'validation_power': 1.38,
            'hit_power': 3.95,
            'payout_direction_weight': 0.06,
            'cluster_weight': 0.12,
            'v15_group_max_filters': 24,
            'v15_max_group_candidates': 130,
            'v15_cross_per_family': 4,
            'row_bucket_size': 500,
            'per_bucket_keep': 4,
        },
    },
    'TOL_NO_KILLERS': {
        'label': 'TOL_NO_KILLERS spärrar kantdödare',
        'description': 'Tolerant variant som spärrar kända farliga individuella filter som ofta dödar spretiga omgångar för liten marginalnytta.',
        'base': 'V21A',
        'v30_mode': 'no_killers',
        'structure_miss': [2, 3, 4],
        'family_miss': [1, 2, 3, 4, 5],
        'cross_family': True,
        'block_fragile_individuals': True,
        'block_rank_summa_narrow': True,
        'block_strict_max0_groups': True,
        'prefer_soft_groups': True,
        'v19_target_rows': 4100,
        'v19_min_active_families': 3,
        'v19_family_bonus': {'FAT': 0.38, 'ABC': 0.36, 'FAT-sekvens': 0.05, 'Värde/svårighet': 0.68, 'Poäng/rank': 0.66, 'Favorit': 0.54, 'Skräll': 0.54, 'Tvärfamilj': 0.40},
        'overrides': {
            'min_hit': 25,
            'candidate_min_hit': 27,
            'beam_width': 80,
            'archive_width': 680,
            'min_unique_rows': 20,
            'min_package_val_pct': 70.0,
            'min_candidate_val_pct': 68.0,
            'min_structure_package_val_pct': 84.0,
            'min_structure_val_pct': 74.0,
            'reduction_power': 1.22,
            'validation_power': 1.48,
            'hit_power': 4.1,
            'payout_direction_weight': 0.04,
            'cluster_weight': 0.10,
            'v15_group_max_filters': 26,
            'v15_max_group_candidates': 140,
            'v15_cross_per_family': 4,
            'row_bucket_size': 550,
            'per_bucket_keep': 4,
        },
    },
    'TOL_SUPERWIDE': {
        'label': 'TOL_SUPERWIDE max tolerans',
        'description': 'Stressar robusthetsmålet: mycket breda grupper, låg radpress, hög träff/valideringspremie. Ska visa om 28–30/30 kräver 4k+ rader.',
        'base': 'V21A',
        'v30_mode': 'superwide',
        'structure_miss': [2, 3, 4, 5],
        'family_miss': [2, 3, 4, 5, 6, 7],
        'cross_family': True,
        'block_fragile_individuals': True,
        'block_strict_max0_groups': True,
        'prefer_soft_groups': True,
        'v19_target_rows': 4500,
        'v19_min_active_families': 3,
        'v19_family_bonus': {'FAT': 0.25, 'ABC': 0.25, 'FAT-sekvens': 0.00, 'Värde/svårighet': 0.55, 'Poäng/rank': 0.60, 'Favorit': 0.50, 'Skräll': 0.50, 'Tvärfamilj': 0.45},
        'overrides': {
            'min_hit': 24,
            'candidate_min_hit': 26,
            'beam_width': 90,
            'archive_width': 760,
            'min_unique_rows': 15,
            'min_package_val_pct': 68.0,
            'min_candidate_val_pct': 66.0,
            'min_structure_package_val_pct': 82.0,
            'min_structure_val_pct': 72.0,
            'reduction_power': 1.00,
            'validation_power': 1.60,
            'hit_power': 4.35,
            'payout_direction_weight': 0.02,
            'cluster_weight': 0.08,
            'v15_group_max_filters': 30,
            'v15_max_group_candidates': 160,
            'v15_cross_per_family': 5,
            'row_bucket_size': 650,
            'per_bucket_keep': 5,
        },
    },
}


def _v30_register_variants():
    for key, cfg in V30_VARIANT_DEFS.items():
        base_key = cfg.get('base', 'V21A')
        if key == 'V21A_BASE':
            # Klona V21A exakt men med tydligare namn för V30-tabellen.
            base = V15_VARIANT_DEFS.get('V21A') or V19_VARIANT_DEFS.get('V21A')
            if not base:
                raise RuntimeError('V21A saknas; kan inte registrera V30-bas.')
            new = dict(base)
            new['label'] = cfg['label']
            new['description'] = cfg['description']
            new['v30_cfg'] = cfg
            V15_VARIANT_DEFS[key] = new
            V19_VARIANT_DEFS[key] = new
            V13_VARIANT_DEFS[key] = {'label': cfg['label'], 'fixed_payout': True, 'bundle_search': False, 'description': cfg['description']}
            continue
        # Bygg från V21A men lägg över v30-fälten och overrides. _v21_clone_variant
        # ger oss samtidigt V21 guards mot hård Delta.
        v21_cfg = dict(V21_VARIANT_DEFS.get('V21A', {}))
        v21_cfg.update({
            'label': cfg['label'],
            'description': cfg['description'],
            'base': base_key,
            'block_delta_individual': True,
            'block_delta_groups': bool(cfg.get('block_delta_groups', False)),
            'soft_favorite_max0': True,
            'green_name_bonus': False,
            'overrides': cfg.get('overrides', {}),
        })
        _v21_clone_variant(base_key, key, v21_cfg)
        # Applicera V30-specifik variantprofil ovanpå den klonade V15/V19-defen.
        for d in (V15_VARIANT_DEFS[key], V19_VARIANT_DEFS[key]):
            d['v30_cfg'] = cfg
            d['mode'] = 'super_groups'
            d['structure_miss'] = cfg.get('structure_miss', d.get('structure_miss', [2,3]))
            d['family_miss'] = cfg.get('family_miss', d.get('family_miss', [1,2,3,4,5]))
            d['cross_family'] = cfg.get('cross_family', True)
            d['drop_individual_structure'] = True
            d['v19_target_rows'] = cfg.get('v19_target_rows', d.get('v19_target_rows'))
            d['v19_min_active_families'] = cfg.get('v19_min_active_families', d.get('v19_min_active_families', 0))
            if cfg.get('v19_family_bonus'):
                d['v19_family_bonus'] = cfg.get('v19_family_bonus')
        V13_VARIANT_DEFS[key] = {'label': cfg['label'], 'fixed_payout': True, 'bundle_search': False, 'description': cfg['description']}

_v30_register_variants()


def _v30_parse_variants(value: str):
    default = ['V21A_BASE','TOL_GROUPS','TOL_WIDE_FAT','TOL_VALUE_SPLIT','TOL_NO_KILLERS','TOL_SUPERWIDE']
    raw = str(value or ','.join(default)).replace(';', ',').replace('+', ',').split(',')
    out=[]
    allowed=set(default) | {'V21A','BASE','RADPRESS'}
    for x in raw:
        k=x.strip().upper()
        if not k:
            continue
        if k in {'V21A','BASE','RADPRESS'}:
            k='V21A_BASE'
        if k in allowed and k not in out:
            out.append(k)
    return out or default


def _v30_candidate_family(c: dict) -> str:
    try:
        return _v19_family_of_candidate(c)
    except Exception:
        return _v15_group_family(c)


def _v30_is_fragile_individual(c: dict) -> bool:
    if bool(c.get('is_v15_group')):
        return False
    txt = f"{c.get('name','')} {c.get('key','')} {c.get('category','')} {c.get('interval_txt','')}".lower()
    fragile_terms = [
        'delta / avvikelse', 'delta_avvikelse',
        'fat 2-sekvenser', '2-sekvenser',
        'längsta följd', 'uppkomster 2',
        'fat t',
    ]
    return any(t in txt for t in fragile_terms)


def _v30_is_strict_max0_group(c: dict) -> bool:
    if not bool(c.get('is_v15_group')):
        return False
    try:
        max_miss = int(c.get('group_max_miss', 99))
    except Exception:
        max_miss = 99
    if max_miss != 0:
        return False
    txt = f"{c.get('name','')} {c.get('key','')} {c.get('category','')} {c.get('interval_txt','')}".lower()
    # Perfekta FAT/ABC/Favorit-grupper kan ge onödigt veto i spretiga utdelningsspann.
    return any(t in txt for t in ['grupp fat', 'grupp abc', 'grupp favorit', 'grupp poäng/rank', 'grupp tvärfamilj'])


def _v30_candidate_guard(c: dict, cfg: dict) -> bool:
    if bool(cfg.get('block_fragile_individuals')) and _v30_is_fragile_individual(c):
        return False
    if bool(cfg.get('block_rank_summa_narrow')) and (not bool(c.get('is_v15_group'))):
        txt = f"{c.get('name','')} {c.get('interval_txt','')}".lower()
        if 'rank summa' in txt:
            # Rank Summa kan vara bra men ska inte vara snävt individuellt veto i tolerant variant.
            return False
    if bool(cfg.get('block_strict_max0_groups')) and _v30_is_strict_max0_group(c):
        return False
    if bool(cfg.get('prefer_soft_groups')) and bool(c.get('is_v15_group')):
        try:
            if int(c.get('group_max_miss', 0)) <= 0:
                return False
        except Exception:
            pass
    return True

_ORIG_V30_TRANSFORM_CANDIDATES = _v15_transform_candidates

def _v15_transform_candidates(candidates, htot, vtot, ftot, hist_payout, args, variant_id):  # noqa: F811
    final, group_cands = _ORIG_V30_TRANSFORM_CANDIDATES(candidates, htot, vtot, ftot, hist_payout, args, variant_id)
    cfg = (V15_VARIANT_DEFS.get(str(variant_id).upper(), {}) or {}).get('v30_cfg') or {}
    if not cfg or cfg.get('v30_mode') == 'base':
        return final, group_cands
    final2 = [c for c in final if _v30_candidate_guard(c, cfg)]
    group2 = [g for g in group_cands if _v30_candidate_guard(g, cfg)]
    # För toleranta lägen: favorisera grupper som inte är max0 och sprid familjer.
    def sk(c):
        fam = _v30_candidate_family(c)
        grp = 1 if bool(c.get('is_v15_group')) else 0
        soft = int(c.get('group_max_miss', 0) or 0) if grp else 0
        fam_bonus = {'Poäng/rank':5,'Värde/svårighet':5,'Skräll':4,'Favorit':4,'ABC':3,'FAT':3,'Tvärfamilj':3,'Struktur':2,'FAT-sekvens':0}.get(fam,1)
        return (grp, soft, fam_bonus, int(c.get('hist_hit',0) or 0), float(c.get('val_pct',0.0) or 0.0), float(c.get('red_pct',0.0) or 0.0))
    final2 = sorted(final2, key=sk, reverse=True)[:max(80, int(getattr(args, 'max_candidates', 300)))]
    group2 = sorted(group2, key=sk, reverse=True)[:max(80, int(getattr(args, 'v15_max_group_candidates', 120)))]
    return final2, group2

_ORIG_V30_STATE_METRICS = _v13_state_metrics

def _v13_state_metrics(st: _V9State, payout_values: np.ndarray, args, variant_id: str) -> dict:  # noqa: F811
    m = _ORIG_V30_STATE_METRICS(st, payout_values, args, variant_id)
    cfg = (V15_VARIANT_DEFS.get(str(variant_id).upper(), {}) or {}).get('v30_cfg') or {}
    if not cfg or cfg.get('v30_mode') == 'base':
        return m
    chosen = list(st.chosen or tuple())
    fams = [_v30_candidate_family(c) for c in chosen]
    famset = set(fams)
    groups = sum(1 for c in chosen if bool(c.get('is_v15_group')))
    fragile = sum(1 for c in chosen if _v30_is_fragile_individual(c) or _v30_is_strict_max0_group(c))
    # V30 premierar robusthet/familjespridning, inte lägsta rader.
    bonus = 0.0
    bonus += 0.28 * min(5, len([f for f in famset if f not in {'Struktur'}]))
    bonus += 0.10 * min(8, groups)
    bonus -= 1.20 * fragile
    # Håll gärna en bred men inte absurd ram: bonus nära target, straff under 2600 i toleranta lägen.
    target = int(cfg.get('v19_target_rows') or 3800)
    rows = int(st.frame_count)
    min_rows = int(cfg.get('v30_min_rows', 2600) or 2600)
    if rows < min_rows:
        bonus -= min(2.0, (min_rows - rows) / 500.0)
    if rows <= target:
        bonus += 0.15
    else:
        bonus -= min(0.8, (rows-target)/max(1.0, st.frame_size))
    m['v30_tolerance_bonus'] = float(bonus)
    m['joint_score'] = float(m.get('joint_score', 0.0)) + float(bonus)
    return m


def _v30_write_outputs(detail: pd.DataFrame, args, out_dir: Path, app_file: Path, db_file: Path):
    summary = _summarize_v15(detail)
    # Extra beslutsordning: träff före rader; accepterar fler rader nu.
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        summary = summary.copy()
        if 'Beslutsnotis' not in summary.columns:
            summary['Beslutsnotis'] = ''
        for i, r in summary.iterrows():
            hits = int(r.get('Träffar', 0) or 0)
            med_rows = float(r.get('Median paketrader', 0) or 0)
            if hits >= 29:
                note = 'HUVUDKANDIDAT: robust 30-test'
            elif hits >= 28:
                note = 'LOVANDE: kan vara produktionsbas'
            elif hits > int(summary[summary['Variant'].astype(str)=='V21A_BASE']['Träffar'].max()) if 'V21A_BASE' in set(summary['Variant'].astype(str)) else False:
                note = 'Förbättrar träff mot V21A'
            else:
                note = ''
            if med_rows > 4300 and hits < 29:
                note = (note + '; dyr').strip('; ')
            summary.loc[i,'Beslutsnotis']=note
        # Vinnare enligt V30: högst träff, sedan lägst rader, men 28+ slår lägre rader.
        summary['Vinnare']=''
        def win_key(r):
            return (int(r.get('Träffar',0) or 0), float(r.get('Träff %',0) or 0.0), -float(r.get('Median paketrader', 10**12) or 10**12))
        best_idx=max(summary.index, key=lambda i: win_key(summary.loc[i]))
        summary.loc[best_idx,'Vinnare']='JA'
    else:
        winner_variant=None
    winner_variant = _v13_winner_variant(summary) if isinstance(summary, pd.DataFrame) and not summary.empty else None
    detail_path = out_dir / f'{args.output_prefix}_detail.csv'
    summary_path = out_dir / f'{args.output_prefix}_summary.csv'
    report_path = out_dir / f'{args.output_prefix}_report.txt'
    winner_detail_path = out_dir / f'{args.output_prefix}_winner_detail.csv'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    if winner_variant and 'Variant' in detail.columns:
        detail[detail['Variant'].astype(str) == str(winner_variant)].to_csv(winner_detail_path, index=False)
    else:
        pd.DataFrame().to_csv(winner_detail_path, index=False)
    lines = [
        'TIPSET AI – V30 TOLERANT REPAIR 30',
        '='*96,
        f'Appbas: {app_file.name}',
        f'Databas: {db_file.name}',
        f'Testomgångar: {args.max_tests}',
        f'Utdelningsintervall: {args.pay_min:,}–{args.pay_max:,} kr'.replace(',', ' '),
        f'Liknande historik: {args.top_n}; bred validering: {args.wide_n}',
        'Mål: reparera 30-testträff först. Radantal 3.3k–4.2k är accepterat om träffen ökar.',
        'Metod: bredare gruppkrav, färre individuella kantdödare, lägre radpress och högre träff-/valideringsvikt.',
        '', 'VARIANTER', '-'*96,
    ]
    for k in _v30_parse_variants(args.variants):
        cfg=V30_VARIANT_DEFS[k]
        lines.append(f'{k}: {cfg["label"]} – {cfg["description"]}')
    lines += ['', 'SAMMANFATTNING', '-'*96, summary.to_string(index=False) if isinstance(summary,pd.DataFrame) and not summary.empty else '(tom)', '', f'Detail: {detail_path}', f'Summary: {summary_path}', f'Winner detail: {winner_detail_path}']
    if winner_variant:
        lines.append(f'Vinnare: {winner_variant}')
        lines.append('')
        lines.append('TOLKNING V36: detta är snabb slumpkontroll. En variant är intressant om den klarar minst 6/8 och median helst 2400–2800. Kör flera seeds innan 20/30-test.')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V30 TOLERANT REPAIR 30', flush=True)
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        print(summary.to_string(index=False), flush=True)
    if winner_variant:
        print(f'Vinnare: {winner_variant}', flush=True)
    print(f'Rapport: {report_path}', flush=True)
    print(f'Summary: {summary_path}', flush=True)
    print(f'Detail: {detail_path}', flush=True)
    return {'summary': summary_path, 'detail': detail_path, 'winner_detail': winner_detail_path, 'report': report_path}


def _v30_collect_frames(out_dir: Path, base_prefix: str, frames: list):
    outs={}
    for suffix in ['summary','detail','winner_detail']:
        dfs=[]
        for frame in frames:
            p=Path(out_dir)/f'{base_prefix}_{_v17_safe_prefix_part(frame)}_{suffix}.csv'
            if p.exists():
                try:
                    df=pd.read_csv(p, sep=None, engine='python')
                    if not df.empty:
                        if 'Ramprofil' not in df.columns:
                            df.insert(0,'Ramprofil',frame)
                        if 'Teoretisk grundram' not in df.columns and suffix in {'summary','detail','winner_detail'}:
                            df.insert(1,'Teoretisk grundram',_v17_frame_rows_count(frame))
                        dfs.append(df)
                except Exception:
                    pass
        if dfs:
            comb=pd.concat(dfs, ignore_index=True)
            outp=Path(out_dir)/f'{base_prefix}_ALL_FRAMES_{suffix}.csv'
            comb.to_csv(outp, index=False)
            outs[suffix]=comb
    return outs


def main_v30():
    parser=argparse.ArgumentParser(description='Tipset AI V30 – tolerant repair 30.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_tolerant_repair_v30')
    parser.add_argument('--variants', default='V21A_BASE,TOL_GROUPS,TOL_WIDE_FAT,TOL_VALUE_SPLIT,TOL_NO_KILLERS,TOL_SUPERWIDE')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='3-5-5')
    parser.add_argument('--frames', default='3-5-5')
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--candidate-min-hit', type=int, default=27)
    parser.add_argument('--min-candidate-val-pct', type=float, default=70.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=76.0)
    parser.add_argument('--min-gap-score', type=float, default=0.55)
    parser.add_argument('--variants-per-key', type=int, default=4)
    parser.add_argument('--max-candidates', type=int, default=380)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    # Bas/byggarparametrar. Variantens egna overrides tar över via _v15_variant_args.
    parser.add_argument('--min-hit', type=int, default=25)
    parser.add_argument('--min-package-val-pct', type=float, default=72.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=84.0)
    parser.add_argument('--min-unique-rows', type=int, default=20)
    parser.add_argument('--beam-width', type=int, default=80)
    parser.add_argument('--archive-width', type=int, default=680)
    parser.add_argument('--structure-seed-count', type=int, default=2)
    parser.add_argument('--hit-power', type=float, default=4.0)
    parser.add_argument('--validation-power', type=float, default=1.35)
    parser.add_argument('--reduction-power', type=float, default=1.25)
    parser.add_argument('--payout-weight', type=float, default=0.03)
    parser.add_argument('--cluster-weight', type=float, default=0.12)
    parser.add_argument('--payout-direction-weight', type=float, default=0.06)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=55)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=12)
    parser.add_argument('--row-bucket-size', type=int, default=500)
    parser.add_argument('--per-bucket-keep', type=int, default=4)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--v15-group-max-filters', type=int, default=26)
    parser.add_argument('--v15-max-group-candidates', type=int, default=140)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=4)
    parser.add_argument('--v15-cross-max-filters', type=int, default=22)
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    # Hårdlås V30-testet. Gamla Colab-args ska inte få göra detta till test10 eller stepdown.
    args.max_tests = 30
    args.test_offset = 0
    args.top_n = 30
    args.wide_n = 35
    args.output_prefix = 'package_tolerant_repair_v30'
    args.variants = ','.join(_v30_parse_variants(args.variants))
    frames = _v17_parse_frames(args.frames)
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n' + '='*96, flush=True)
    print('TIPSET AI V30 – TOLERANT REPAIR 30', flush=True)
    print('LÅST LOGIK: träffsäkerhet först. Bredare/tolerant paket för 100k–2.5M utdelningsspann.', flush=True)
    print('MÅL: reparera 26/30 mot 28–30/30; acceptera högre radantal före ny radpress.', flush=True)
    print('Testomgångar: 8 (hårdlåst, stratifierad slump inom utdelningsintervall)', flush=True)
    print(f'Ramar: {", ".join(frames)}', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)
    for frame in frames:
        frame_args=argparse.Namespace(**vars(args))
        frame_args.frames=frame
        frame_args.frame_profile=frame
        frame_args.output_prefix=f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n'+'-'*96, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*96, flush=True)
        ns=_load_app_functions(app_file, fast_no_supermakro=False)
        db=ns['load_database'](str(db_file),13)
        db, sample_df = _v36_random_reorder_db_for_tests(ns, db, frame_args)
        if isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
            sample_path = out_dir / f'{frame_args.output_prefix}_sample.csv'
            sample_df.to_csv(sample_path, index=False)
            print('Slumpade/stratifierade testomgångar:', flush=True)
            print(sample_df.to_string(index=False), flush=True)
            print('Sample:', str(sample_path), flush=True)
        detail, meta = _run_backtest_v15(ns, db, frame_args)
        _v30_write_outputs(detail, frame_args, out_dir, app_file, db_file)
        del ns, db, detail
        gc.collect()
    combined=_v30_collect_frames(out_dir,args.output_prefix,frames)
    if combined.get('summary') is not None:
        print('\nKLART – V30 ALLA RAMAR', flush=True)
        print(combined['summary'].to_string(index=False), flush=True)
        print('Summary:', str(Path(out_dir)/f'{args.output_prefix}_ALL_FRAMES_summary.csv'), flush=True)



# =============================================================================
# V30b – TOLERANT BUDGET 2000
# =============================================================================
# Korrigerar V30: paketmotorn ska fortfarande sikta mot ca 2 000 rader före
# 12-rättsgaranti. Den får vara tolerant i hur filter kombineras, men inte bred
# som 3.8k-4.5k slutram. Varianterna är 30-testade och hårdlåsta.

V30B_VARIANT_DEFS = {
    'V21A_BASE': V30_VARIANT_DEFS['V21A_BASE'],
    'BUDGET2400_TOL': {
        'label': 'BUDGET2400_TOL tolerant budget 2400',
        'description': 'Toleranta grupper men med paketmål runt 2 400 före 12-rättsgaranti.',
        'base': 'V21A', 'v30_mode': 'budget2400',
        'structure_miss': [2,3,4], 'family_miss': [1,2,3,4,5], 'cross_family': True,
        'block_fragile_individuals': True, 'block_strict_max0_groups': True, 'prefer_soft_groups': True,
        'v19_target_rows': 2400, 'v30_min_rows': 1900, 'v19_min_active_families': 3,
        'v19_family_bonus': {'FAT':0.38,'ABC':0.34,'Poäng/rank':0.64,'Värde/svårighet':0.62,'Favorit':0.48,'Skräll':0.50,'Tvärfamilj':0.36,'Struktur':0.18},
        'overrides': {'min_hit':24,'candidate_min_hit':27,'beam_width':84,'archive_width':760,'min_unique_rows':28,'min_package_val_pct':68.0,'min_candidate_val_pct':66.0,'min_structure_package_val_pct':82.0,'min_structure_val_pct':72.0,'reduction_power':2.35,'validation_power':1.22,'hit_power':3.35,'payout_direction_weight':0.06,'cluster_weight':0.10,'v15_group_max_filters':26,'v15_max_group_candidates':150,'v15_cross_per_family':4,'row_bucket_size':420,'per_bucket_keep':4},
    },
    'BUDGET2300_TOL': {
        'label': 'BUDGET2300_TOL tolerant budget 2300',
        'description': 'Huvudkandidat: tolerant paketbyggnad med mål runt 2 300.',
        'base': 'V21A', 'v30_mode': 'budget2300',
        'structure_miss': [2,3,4], 'family_miss': [1,2,3,4,5,6], 'cross_family': True,
        'block_fragile_individuals': True, 'block_strict_max0_groups': True, 'prefer_soft_groups': True,
        'v19_target_rows': 2300, 'v30_min_rows': 1800, 'v19_min_active_families': 3,
        'v19_family_bonus': {'FAT':0.36,'ABC':0.32,'Poäng/rank':0.66,'Värde/svårighet':0.66,'Favorit':0.52,'Skräll':0.54,'Tvärfamilj':0.38,'Struktur':0.18},
        'overrides': {'min_hit':24,'candidate_min_hit':26,'beam_width':88,'archive_width':820,'min_unique_rows':24,'min_package_val_pct':66.0,'min_candidate_val_pct':64.0,'min_structure_package_val_pct':80.0,'min_structure_val_pct':70.0,'reduction_power':2.55,'validation_power':1.12,'hit_power':3.15,'payout_direction_weight':0.05,'cluster_weight':0.08,'v15_group_max_filters':28,'v15_max_group_candidates':165,'v15_cross_per_family':5,'row_bucket_size':380,'per_bucket_keep':4},
    },
    'BUDGET2200_TOL': {
        'label': 'BUDGET2200_TOL tolerant budget 2200',
        'description': 'Lite hårdare budgetvariant mot 2 200, fortfarande tolerant gruppbaserad.',
        'base': 'V21A', 'v30_mode': 'budget2200',
        'structure_miss': [2,3,4,5], 'family_miss': [1,2,3,4,5,6], 'cross_family': True,
        'block_fragile_individuals': True, 'block_strict_max0_groups': True, 'prefer_soft_groups': True,
        'v19_target_rows': 2200, 'v30_min_rows': 1700, 'v19_min_active_families': 3,
        'v19_family_bonus': {'FAT':0.32,'ABC':0.30,'Poäng/rank':0.64,'Värde/svårighet':0.68,'Favorit':0.56,'Skräll':0.58,'Tvärfamilj':0.42,'Struktur':0.15},
        'overrides': {'min_hit':23,'candidate_min_hit':26,'beam_width':92,'archive_width':880,'min_unique_rows':22,'min_package_val_pct':64.0,'min_candidate_val_pct':62.0,'min_structure_package_val_pct':78.0,'min_structure_val_pct':68.0,'reduction_power':2.75,'validation_power':1.03,'hit_power':3.00,'payout_direction_weight':0.04,'cluster_weight':0.07,'v15_group_max_filters':30,'v15_max_group_candidates':180,'v15_cross_per_family':5,'row_bucket_size':360,'per_bucket_keep':4},
    },
    'BUDGET2000_TOL': {
        'label': 'BUDGET2000_TOL slutmål före garanti',
        'description': 'Stressar huvudmålet: paketmotor ner mot ca 2 000 före 12-rättsgaranti.',
        'base': 'V21A', 'v30_mode': 'budget2000',
        'structure_miss': [2,3,4,5], 'family_miss': [2,3,4,5,6,7], 'cross_family': True,
        'block_fragile_individuals': True, 'block_strict_max0_groups': True, 'prefer_soft_groups': True,
        'v19_target_rows': 2000, 'v30_min_rows': 1500, 'v19_min_active_families': 3,
        'v19_family_bonus': {'FAT':0.28,'ABC':0.26,'Poäng/rank':0.62,'Värde/svårighet':0.70,'Favorit':0.58,'Skräll':0.60,'Tvärfamilj':0.44,'Struktur':0.12},
        'overrides': {'min_hit':23,'candidate_min_hit':25,'beam_width':96,'archive_width':940,'min_unique_rows':20,'min_package_val_pct':62.0,'min_candidate_val_pct':60.0,'min_structure_package_val_pct':76.0,'min_structure_val_pct':66.0,'reduction_power':3.05,'validation_power':0.95,'hit_power':2.85,'payout_direction_weight':0.04,'cluster_weight':0.06,'v15_group_max_filters':32,'v15_max_group_candidates':195,'v15_cross_per_family':5,'row_bucket_size':330,'per_bucket_keep':4},
    },
    'GROUP2300_TOL': {
        'label': 'GROUP2300_TOL gruppbudget 2300',
        'description': 'Mer gruppstyrd än individuell: försöker nå 2 300 med max-miss i familjer.',
        'base': 'V21A', 'v30_mode': 'group2300',
        'structure_miss': [3,4,5], 'family_miss': [2,3,4,5,6,7], 'cross_family': True,
        'block_fragile_individuals': True, 'block_strict_max0_groups': True, 'prefer_soft_groups': True,
        'v19_target_rows': 2300, 'v30_min_rows': 1700, 'v19_min_active_families': 4,
        'v19_family_bonus': {'FAT':0.24,'ABC':0.22,'Poäng/rank':0.70,'Värde/svårighet':0.72,'Favorit':0.62,'Skräll':0.64,'Tvärfamilj':0.52,'Struktur':0.10},
        'overrides': {'min_hit':23,'candidate_min_hit':25,'beam_width':100,'archive_width':980,'min_unique_rows':18,'min_package_val_pct':62.0,'min_candidate_val_pct':60.0,'min_structure_package_val_pct':76.0,'min_structure_val_pct':66.0,'reduction_power':2.70,'validation_power':1.05,'hit_power':3.05,'payout_direction_weight':0.03,'cluster_weight':0.05,'v15_group_max_filters':34,'v15_max_group_candidates':210,'v15_cross_per_family':6,'row_bucket_size':360,'per_bucket_keep':5},
    },
    'GROUP2000_TOL': {
        'label': 'GROUP2000_TOL gruppbudget 2000',
        'description': 'Gruppstyrd stress mot 2 000 rader före garanti.',
        'base': 'V21A', 'v30_mode': 'group2000',
        'structure_miss': [3,4,5], 'family_miss': [2,3,4,5,6,7], 'cross_family': True,
        'block_fragile_individuals': True, 'block_strict_max0_groups': True, 'prefer_soft_groups': True,
        'v19_target_rows': 2000, 'v30_min_rows': 1450, 'v19_min_active_families': 4,
        'v19_family_bonus': {'FAT':0.22,'ABC':0.20,'Poäng/rank':0.68,'Värde/svårighet':0.74,'Favorit':0.64,'Skräll':0.66,'Tvärfamilj':0.55,'Struktur':0.08},
        'overrides': {'min_hit':22,'candidate_min_hit':25,'beam_width':104,'archive_width':1040,'min_unique_rows':16,'min_package_val_pct':60.0,'min_candidate_val_pct':58.0,'min_structure_package_val_pct':74.0,'min_structure_val_pct':64.0,'reduction_power':3.10,'validation_power':0.92,'hit_power':2.75,'payout_direction_weight':0.03,'cluster_weight':0.04,'v15_group_max_filters':36,'v15_max_group_candidates':230,'v15_cross_per_family':6,'row_bucket_size':320,'per_bucket_keep':5},
    },
    'VALUE2200_SPLIT': {
        'label': 'VALUE2200_SPLIT 100k/1M-spret',
        'description': 'Försöker fånga spret mellan 100k och 1M+ med mer värde/skräll och mindre FAT-dominans, mål 2 200.',
        'base': 'V21A', 'v30_mode': 'value2200',
        'structure_miss': [2,3,4], 'family_miss': [2,3,4,5,6], 'cross_family': True,
        'block_fragile_individuals': True, 'block_strict_max0_groups': True, 'prefer_soft_groups': True,
        'v19_target_rows': 2200, 'v30_min_rows': 1650, 'v19_min_active_families': 4,
        'v19_family_bonus': {'FAT':0.18,'ABC':0.18,'Poäng/rank':0.74,'Värde/svårighet':0.82,'Favorit':0.70,'Skräll':0.74,'Tvärfamilj':0.50,'Struktur':0.10},
        'overrides': {'min_hit':23,'candidate_min_hit':25,'beam_width':100,'archive_width':980,'min_unique_rows':18,'min_package_val_pct':61.0,'min_candidate_val_pct':59.0,'min_structure_package_val_pct':75.0,'min_structure_val_pct':64.0,'reduction_power':2.85,'validation_power':1.02,'hit_power':2.95,'payout_direction_weight':0.03,'cluster_weight':0.05,'v15_group_max_filters':34,'v15_max_group_candidates':210,'v15_cross_per_family':6,'row_bucket_size':340,'per_bucket_keep':5},
    },
}

# Uppdatera V30-registren med budgetvarianter och registrera dem i V15/V19/V13.
V30_VARIANT_DEFS.update(V30B_VARIANT_DEFS)
_v30_register_variants()


def _v30b_parse_variants(value: str):
    locked = ['V21A_BASE','BUDGET2400_TOL','BUDGET2300_TOL','BUDGET2200_TOL','BUDGET2000_TOL','GROUP2300_TOL','GROUP2000_TOL','VALUE2200_SPLIT']
    return locked


def main_v30b():
    parser=argparse.ArgumentParser(description='Tipset AI V30b – tolerant budget 2000, 30-test.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_tolerant_budget2000_v30b')
    parser.add_argument('--variants', default='IGNORED')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='3-5-5')
    parser.add_argument('--frames', default='3-5-5')
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--candidate-min-hit', type=int, default=25)
    parser.add_argument('--min-candidate-val-pct', type=float, default=58.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=64.0)
    parser.add_argument('--min-gap-score', type=float, default=0.55)
    parser.add_argument('--variants-per-key', type=int, default=4)
    parser.add_argument('--max-candidates', type=int, default=430)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    parser.add_argument('--min-hit', type=int, default=23)
    parser.add_argument('--min-package-val-pct', type=float, default=62.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=76.0)
    parser.add_argument('--min-unique-rows', type=int, default=18)
    parser.add_argument('--beam-width', type=int, default=100)
    parser.add_argument('--archive-width', type=int, default=1000)
    parser.add_argument('--structure-seed-count', type=int, default=2)
    parser.add_argument('--hit-power', type=float, default=2.95)
    parser.add_argument('--validation-power', type=float, default=1.00)
    parser.add_argument('--reduction-power', type=float, default=2.85)
    parser.add_argument('--payout-weight', type=float, default=0.02)
    parser.add_argument('--cluster-weight', type=float, default=0.05)
    parser.add_argument('--payout-direction-weight', type=float, default=0.03)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=55)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=12)
    parser.add_argument('--row-bucket-size', type=int, default=340)
    parser.add_argument('--per-bucket-keep', type=int, default=5)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--v15-group-max-filters', type=int, default=24)
    parser.add_argument('--v15-max-group-candidates', type=int, default=210)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=4)
    parser.add_argument('--v15-cross-max-filters', type=int, default=16)
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    args.max_tests = 30
    args.test_offset = 0
    args.top_n = 30
    args.wide_n = 35
    args.output_prefix = 'package_tolerant_budget2000_v30b'
    args.variants = ','.join(_v30b_parse_variants(args.variants))
    args.frames = '3-5-5'
    frames = _v17_parse_frames(args.frames)
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n' + '='*96, flush=True)
    print('TIPSET AI V30b – TOLERANT BUDGET 2000', flush=True)
    print('LÅST LOGIK: paketmotorn ska sikta mot ca 2 000–2 400 rader före 12-rättsgaranti.', flush=True)
    print('Testomgångar: 8 (hårdlåst, stratifierad slump inom utdelningsintervall)', flush=True)
    print('Ramar: 3-5-5 (7776 grundrader)', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)
    for frame in frames:
        frame_args=argparse.Namespace(**vars(args))
        frame_args.frames=frame
        frame_args.frame_profile=frame
        frame_args.output_prefix=f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n'+'-'*96, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*96, flush=True)
        ns=_load_app_functions(app_file, fast_no_supermakro=False)
        db=ns['load_database'](str(db_file),13)
        detail, meta = _run_backtest_v15(ns, db, frame_args)
        _v30_write_outputs(detail, frame_args, out_dir, app_file, db_file)
        del ns, db, detail
        gc.collect()
    combined=_v30_collect_frames(out_dir,args.output_prefix,frames)
    if combined.get('summary') is not None:
        print('\nKLART – V30b ALLA RAMAR', flush=True)
        print(combined['summary'].to_string(index=False), flush=True)
        print('Summary:', str(Path(out_dir)/f'{args.output_prefix}_ALL_FRAMES_summary.csv'), flush=True)



# =============================================================================
# V31 – TRUE 2000 BUDGET AUDIT
# =============================================================================
# Syfte: V30b visade att mjuk budgetstyrning inte räcker. V31 använder samma
# kandidat- och beam-maskineri men väljer paket från ett faktiskt radband när
# varianten är en TRUE-budgetvariant. Referenserna körs som tidigare.

V31_VARIANT_DEFS = {
    'REF_SUPER': {
        'label': 'REF_SUPER V15 SUPER 3000-ref',
        'description': 'Referens: V15 SUPER utan hårt 2k-band.',
        'base': 'SUPER', 'v31_cfg': {'control': True},
    },
    'REF_B5': {
        'label': 'REF_B5 B5 2500-ref',
        'description': 'Referens: B5/B14-radpress utan hårt 2k-band.',
        'base': 'B5', 'v31_cfg': {'control': True},
    },
    'REF_V21A': {
        'label': 'REF_V21A V21A 3265-ref',
        'description': 'Referens: V21A_BASE från V29/V30 utan hårt 2k-band.',
        'base': 'V21A_BASE', 'v31_cfg': {'control': True},
    },
    'SUPER2400_TRUE': {
        'label': 'SUPER2400_TRUE V15-super band 2200-2600',
        'description': 'V15 SUPER som bas, men paketval tvingas mot 2400-band.',
        'base': 'SUPER', 'v31_cfg': {'target_rows': 2400, 'min_rows': 2200, 'max_rows': 2600, 'strict': True},
        'overrides': {'beam_width': 110, 'archive_width': 1200, 'min_hit': 24, 'candidate_min_hit': 26, 'min_package_val_pct': 62.0, 'min_candidate_val_pct': 60.0, 'min_unique_rows': 12, 'reduction_power': 3.4, 'validation_power': 0.95, 'hit_power': 2.75, 'row_bucket_size': 280, 'per_bucket_keep': 7},
    },
    'SUPER2200_TRUE': {
        'label': 'SUPER2200_TRUE V15-super band 2000-2400',
        'description': 'V15 SUPER som bas, hårt huvudband runt 2200.',
        'base': 'SUPER', 'v31_cfg': {'target_rows': 2200, 'min_rows': 2000, 'max_rows': 2400, 'strict': True},
        'overrides': {'beam_width': 120, 'archive_width': 1400, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 60.0, 'min_candidate_val_pct': 58.0, 'min_unique_rows': 10, 'reduction_power': 3.8, 'validation_power': 0.88, 'hit_power': 2.55, 'row_bucket_size': 240, 'per_bucket_keep': 8},
    },
    'SUPER2000_TRUE': {
        'label': 'SUPER2000_TRUE V15-super band 1800-2200',
        'description': 'V15 SUPER som bas, hårt slutmålsband runt 2000.',
        'base': 'SUPER', 'v31_cfg': {'target_rows': 2000, 'min_rows': 1800, 'max_rows': 2200, 'strict': True},
        'overrides': {'beam_width': 130, 'archive_width': 1600, 'min_hit': 22, 'candidate_min_hit': 25, 'min_package_val_pct': 58.0, 'min_candidate_val_pct': 56.0, 'min_unique_rows': 8, 'reduction_power': 4.2, 'validation_power': 0.82, 'hit_power': 2.35, 'row_bucket_size': 220, 'per_bucket_keep': 9},
    },
    'B52400_TRUE': {
        'label': 'B52400_TRUE B5 band 2200-2600',
        'description': 'B5 som bas, men paketval tvingas mot 2400-band.',
        'base': 'B5', 'v31_cfg': {'target_rows': 2400, 'min_rows': 2200, 'max_rows': 2600, 'strict': True},
        'overrides': {'beam_width': 100, 'archive_width': 1100, 'min_hit': 24, 'candidate_min_hit': 26, 'min_package_val_pct': 62.0, 'min_candidate_val_pct': 60.0, 'min_unique_rows': 12, 'reduction_power': 3.5, 'validation_power': 0.92, 'hit_power': 2.70, 'row_bucket_size': 260, 'per_bucket_keep': 7},
    },
    'B52300_TRUE': {
        'label': 'B52300_TRUE B5 band 2100-2500',
        'description': 'B5 som bas, mellankompromiss runt 2300. Lite bredare än B52200_TRUE för att se om den extra kanten reparerar miss utan att bli för bred.',
        'base': 'B5', 'v31_cfg': {'target_rows': 2300, 'min_rows': 2100, 'max_rows': 2500, 'strict': True},
        'overrides': {'beam_width': 112, 'archive_width': 1350, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 60.0, 'min_candidate_val_pct': 58.0, 'min_unique_rows': 10, 'reduction_power': 3.75, 'validation_power': 0.88, 'hit_power': 2.55, 'row_bucket_size': 235, 'per_bucket_keep': 8},
    },
    'B52200_TRUE': {
        'label': 'B52200_TRUE B5 band 2000-2400',
        'description': 'B5 som bas, hårt huvudband runt 2200.',
        'base': 'B5', 'v31_cfg': {'target_rows': 2200, 'min_rows': 2000, 'max_rows': 2400, 'strict': True},
        'overrides': {'beam_width': 110, 'archive_width': 1300, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 60.0, 'min_candidate_val_pct': 58.0, 'min_unique_rows': 10, 'reduction_power': 3.9, 'validation_power': 0.86, 'hit_power': 2.50, 'row_bucket_size': 230, 'per_bucket_keep': 8},
    },
    'B52000_TRUE': {
        'label': 'B52000_TRUE B5 band 1800-2200',
        'description': 'B5 som bas, hårt slutmålsband runt 2000.',
        'base': 'B5', 'v31_cfg': {'target_rows': 2000, 'min_rows': 1800, 'max_rows': 2200, 'strict': True},
        'overrides': {'beam_width': 120, 'archive_width': 1500, 'min_hit': 22, 'candidate_min_hit': 24, 'min_package_val_pct': 58.0, 'min_candidate_val_pct': 56.0, 'min_unique_rows': 8, 'reduction_power': 4.3, 'validation_power': 0.80, 'hit_power': 2.30, 'row_bucket_size': 210, 'per_bucket_keep': 9},
    },
}


# =============================================================================
# V34 – B52200 MISS REPAIR
# =============================================================================
# V33 visade att B52200_TRUE passar radmålet men föll till 15/20.
# V34 testar om missarna kan repareras med liten radkostnad, utan att återgå
# till REF_SUPER runt 3000+ rader.

V31_VARIANT_DEFS.update({
    'B52200_REPAIR_SAFE': {
        'label': 'B52200_REPAIR_SAFE säkrare band 2100-2600',
        'description': 'B5 som bas, men väljer säkrare state i ett något bredare 2100-2600-band. Mål: reparera B52200-missar utan att gå över ca 2600 median.',
        'base': 'B5',
        'v31_cfg': {'target_rows': 2400, 'min_rows': 2100, 'max_rows': 2600, 'strict': True, 'v34_repair': 'safe'},
        'overrides': {'beam_width': 130, 'archive_width': 1700, 'min_hit': 24, 'candidate_min_hit': 26, 'min_package_val_pct': 62.0, 'min_candidate_val_pct': 60.0, 'min_unique_rows': 12, 'reduction_power': 3.25, 'validation_power': 0.98, 'hit_power': 2.90, 'row_bucket_size': 260, 'per_bucket_keep': 10},
    },
    'B52200_REPAIR_FAM': {
        'label': 'B52200_REPAIR_FAM familjegrupp band 2100-2600',
        'description': 'FAM/SUPER-idén men budgetvald nära B52200. Mjukare grupper ska ersätta några individuella dödsfilter.',
        'base': 'FAM',
        'v31_cfg': {'target_rows': 2400, 'min_rows': 2100, 'max_rows': 2600, 'strict': True, 'v34_repair': 'family'},
        'overrides': {'beam_width': 130, 'archive_width': 1700, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 60.0, 'min_candidate_val_pct': 58.0, 'min_unique_rows': 10, 'reduction_power': 3.45, 'validation_power': 0.92, 'hit_power': 2.65, 'row_bucket_size': 255, 'per_bucket_keep': 9, 'v15_group_max_filters': 28, 'v15_max_group_candidates': 180, 'v15_cross_per_family': 4, 'v15_cross_max_filters': 18},
    },
    'B52200_REPAIR_NOSEQ': {
        'label': 'B52200_REPAIR_NOSEQ utan hårda FAT-sekvensfilter',
        'description': 'B5-budget där FAT-sekvens/sekvensfilter inte får agera individuella dödsfilter. Testar tidigare signal att FAT-sekvenser har låg unik nytta men kan döda.',
        'base': 'B5',
        'v31_cfg': {'target_rows': 2300, 'min_rows': 2000, 'max_rows': 2600, 'strict': True, 'v34_repair': 'no_fatseq'},
        'overrides': {'beam_width': 130, 'archive_width': 1650, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 60.0, 'min_candidate_val_pct': 58.0, 'min_unique_rows': 10, 'reduction_power': 3.65, 'validation_power': 0.90, 'hit_power': 2.60, 'row_bucket_size': 250, 'per_bucket_keep': 9},
    },
    'B52200_REPAIR_STRUCT': {
        'label': 'B52200_REPAIR_STRUCT struktur som skyddsgrupp',
        'description': 'Tar bort individuella strukturveto och låter struktur fungera som bred grupp, samtidigt som budgetbandet hålls nära 2.1-2.6k.',
        'base': 'SG3',
        'v31_cfg': {'target_rows': 2400, 'min_rows': 2100, 'max_rows': 2600, 'strict': True, 'v34_repair': 'struct_group'},
        'overrides': {'beam_width': 120, 'archive_width': 1500, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 60.0, 'min_candidate_val_pct': 58.0, 'min_unique_rows': 10, 'reduction_power': 3.55, 'validation_power': 0.90, 'hit_power': 2.60, 'row_bucket_size': 250, 'per_bucket_keep': 9, 'v15_group_max_filters': 30, 'v15_max_group_candidates': 160},
    },
})


# =============================================================================
# V36 – RANDOM STRATIFIED SUPER SCREEN8
# =============================================================================
# V34 visade att B52200-budgetspåret är rätt radnivå men för skört, medan REF_SUPER
# håller bättre träff men är för brett. V35 vänder riktningen: börja från SUPER och
# pressa nedåt med verklig budgetvikt.

V31_VARIANT_DEFS.update({
    'SUPER2800_PRESS': {
        'label': 'SUPER2800_PRESS SUPER band 2600-3000',
        'description': 'SUPER som robust bas, nedpressad mot cirka 2800. Första kontrollen av hur mycket som kan kapas utan att träffen faller hårt.',
        'base': 'SUPER',
        'v31_cfg': {'target_rows': 2800, 'min_rows': 2600, 'max_rows': 3000, 'strict': True, 'v35_downpress': True, 'min_select_hist': 24},
        'overrides': {'beam_width': 150, 'archive_width': 2200, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 58.0, 'min_candidate_val_pct': 56.0, 'min_unique_rows': 8, 'reduction_power': 4.0, 'validation_power': 0.78, 'hit_power': 2.15, 'row_bucket_size': 220, 'per_bucket_keep': 11, 'v15_group_max_filters': 32, 'v15_max_group_candidates': 230, 'v15_cross_per_family': 6, 'v15_cross_max_filters': 22},
    },
    'SUPER2700_PRESS': {
        'label': 'SUPER2700_PRESS SUPER band 2500-2900',
        'description': 'SUPER nedpressad mot cirka 2700. Mellansteg för att hitta gränsen mellan robusthet och spelbar storlek.',
        'base': 'SUPER',
        'v31_cfg': {'target_rows': 2700, 'min_rows': 2500, 'max_rows': 2900, 'strict': True, 'v35_downpress': True, 'min_select_hist': 23},
        'overrides': {'beam_width': 160, 'archive_width': 2400, 'min_hit': 22, 'candidate_min_hit': 24, 'min_package_val_pct': 56.0, 'min_candidate_val_pct': 54.0, 'min_unique_rows': 7, 'reduction_power': 4.35, 'validation_power': 0.72, 'hit_power': 2.00, 'row_bucket_size': 205, 'per_bucket_keep': 12, 'v15_group_max_filters': 34, 'v15_max_group_candidates': 250, 'v15_cross_per_family': 7, 'v15_cross_max_filters': 24},
    },
    'SUPER2600_PRESS': {
        'label': 'SUPER2600_PRESS SUPER band 2400-2800',
        'description': 'SUPER nedpressad mot cirka 2600. Huvudtest för om SUPER kan bli praktisk före 12-rättsgaranti.',
        'base': 'SUPER',
        'v31_cfg': {'target_rows': 2600, 'min_rows': 2400, 'max_rows': 2800, 'strict': True, 'v35_downpress': True, 'min_select_hist': 23},
        'overrides': {'beam_width': 170, 'archive_width': 2600, 'min_hit': 22, 'candidate_min_hit': 24, 'min_package_val_pct': 55.0, 'min_candidate_val_pct': 53.0, 'min_unique_rows': 6, 'reduction_power': 4.65, 'validation_power': 0.68, 'hit_power': 1.90, 'row_bucket_size': 190, 'per_bucket_keep': 13, 'v15_group_max_filters': 36, 'v15_max_group_candidates': 270, 'v15_cross_per_family': 8, 'v15_cross_max_filters': 26},
    },
    'SUPER2500_PRESS': {
        'label': 'SUPER2500_PRESS SUPER band 2300-2700',
        'description': 'Hårdare SUPER-press mot cirka 2500. Ska bara leva vidare om den behåller oväntat hög träff.',
        'base': 'SUPER',
        'v31_cfg': {'target_rows': 2500, 'min_rows': 2300, 'max_rows': 2700, 'strict': True, 'v35_downpress': True, 'min_select_hist': 22},
        'overrides': {'beam_width': 180, 'archive_width': 2800, 'min_hit': 21, 'candidate_min_hit': 23, 'min_package_val_pct': 53.0, 'min_candidate_val_pct': 51.0, 'min_unique_rows': 5, 'reduction_power': 5.0, 'validation_power': 0.62, 'hit_power': 1.75, 'row_bucket_size': 175, 'per_bucket_keep': 14, 'v15_group_max_filters': 38, 'v15_max_group_candidates': 290, 'v15_cross_per_family': 8, 'v15_cross_max_filters': 28},
    },
    'SUPER_B5_BRIDGE': {
        'label': 'SUPER_B5_BRIDGE SUPER/B5 bro band 2400-2800',
        'description': 'SUPER-bas men med B5-liknande radpress och fler individuella candidates. Testar om vi kan få mellanläge: bättre träff än B52200 men tydligt lägre rader än REF_SUPER.',
        'base': 'SUPER',
        'v31_cfg': {'target_rows': 2600, 'min_rows': 2400, 'max_rows': 2800, 'strict': True, 'v35_downpress': True, 'min_select_hist': 23, 'bridge': True},
        'overrides': {'beam_width': 180, 'archive_width': 2800, 'min_hit': 22, 'candidate_min_hit': 24, 'min_package_val_pct': 55.0, 'min_candidate_val_pct': 53.0, 'min_unique_rows': 6, 'reduction_power': 4.85, 'validation_power': 0.66, 'hit_power': 1.85, 'row_bucket_size': 185, 'per_bucket_keep': 14, 'v15_group_max_filters': 40, 'v15_max_group_candidates': 300, 'v15_cross_per_family': 9, 'v15_cross_max_filters': 30},
    },
})


def _v31_register_variants():
    # Alla V31-varianter registreras som V15-varianter så befintlig backtest kan
    # återanvändas. Basens grupp-/familjelogik behålls; v31_cfg styr bara paketval.
    for key, cfg in V31_VARIANT_DEFS.items():
        base_key = str(cfg.get('base','B5')).upper()
        if base_key == 'V21A_BASE':
            base = dict(V15_VARIANT_DEFS.get('V21A_BASE') or V15_VARIANT_DEFS.get('V21A') or {})
        else:
            base = dict(V15_VARIANT_DEFS.get(base_key) or {})
        if not base:
            raise RuntimeError(f'Kan inte registrera V31 {key}: bas {base_key} saknas.')
        new = dict(base)
        new['label'] = cfg['label']
        new['description'] = cfg['description']
        new['v31_cfg'] = dict(cfg.get('v31_cfg') or {})
        # Budgetvarianter får override ovanpå basen.
        for k, v in (cfg.get('overrides') or {}).items():
            # Lägg overrides som vanliga V15-overrides så _v15_variant_args tar dem.
            pass
        ov = dict(new.get('overrides') or {})
        ov.update(cfg.get('overrides') or {})
        new['overrides'] = ov
        V15_VARIANT_DEFS[key] = new
        V19_VARIANT_DEFS[key] = new
        V13_VARIANT_DEFS[key] = {
            'label': cfg['label'],
            'fixed_payout': True,
            'bundle_search': bool(V13_VARIANT_DEFS.get(base_key, {}).get('bundle_search', False)),
            'description': cfg['description'],
        }


_v31_register_variants()

# V35s snabbskärm: sänk sökbredden kraftigt. Detta är inte finalkörning;
# syftet är att snabbt se om SUPER-downpress-spåret lever innan 20/30-test.
_V35S_FAST_OVERRIDES = {
    'B52200_TRUE': {'beam_width': 55, 'archive_width': 650, 'v15_group_max_filters': 18, 'v15_max_group_candidates': 90, 'v15_cross_per_family': 3, 'v15_cross_max_filters': 14, 'per_bucket_keep': 5},
    'SUPER2800_PRESS': {'beam_width': 65, 'archive_width': 750, 'v15_group_max_filters': 22, 'v15_max_group_candidates': 115, 'v15_cross_per_family': 4, 'v15_cross_max_filters': 16, 'per_bucket_keep': 6},
    'SUPER2600_PRESS': {'beam_width': 70, 'archive_width': 850, 'v15_group_max_filters': 24, 'v15_max_group_candidates': 130, 'v15_cross_per_family': 4, 'v15_cross_max_filters': 18, 'per_bucket_keep': 6},
    'SUPER_B5_BRIDGE': {'beam_width': 75, 'archive_width': 900, 'v15_group_max_filters': 26, 'v15_max_group_candidates': 145, 'v15_cross_per_family': 5, 'v15_cross_max_filters': 20, 'per_bucket_keep': 7},
}
for _k, _ov2 in _V35S_FAST_OVERRIDES.items():
    if _k in V15_VARIANT_DEFS:
        _ov = dict(V15_VARIANT_DEFS[_k].get('overrides') or {})
        _ov.update(_ov2)
        V15_VARIANT_DEFS[_k]['overrides'] = _ov
        V19_VARIANT_DEFS[_k] = V15_VARIANT_DEFS[_k]


# V34: justera registrerade varianter med grupplogik och familjespärrar.
for _v34_key, _v34_extra in {
    'B52200_REPAIR_FAM': {
        'mode': 'family_groups', 'structure_miss': [2,3,4], 'family_miss': [1,2,3,4,5],
        'cross_family': False, 'drop_individual_structure': True,
    },
    'B52200_REPAIR_STRUCT': {
        'mode': 'structure_group', 'structure_miss': [3,4], 'family_miss': [],
        'cross_family': False, 'drop_individual_structure': True,
    },
    'B52200_REPAIR_NOSEQ': {
        'v34_drop_families_pre': ['FAT-sekvens'],
    },
}.items():
    if _v34_key in V15_VARIANT_DEFS:
        V15_VARIANT_DEFS[_v34_key].update(_v34_extra)
        V19_VARIANT_DEFS[_v34_key] = V15_VARIANT_DEFS[_v34_key]

_ORIG_V34_TRANSFORM_CANDIDATES = _v15_transform_candidates

def _v15_transform_candidates(candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, args, variant_id: str) -> Tuple[List[dict], List[dict]]:
    vid = str(variant_id).upper()
    cfg = V15_VARIANT_DEFS.get(vid, {}) or {}
    drops = set(cfg.get('v34_drop_families_pre') or [])
    if drops:
        filtered = []
        for c in candidates:
            fam = _v15_group_family(c)
            if fam in drops:
                continue
            filtered.append(c)
        candidates = filtered
    return _ORIG_V34_TRANSFORM_CANDIDATES(candidates, htot, vtot, ftot, hist_payout, args, variant_id)


def _v31_parse_variants(value: str):
    # V35 downpress: hårdlåst kortlista efter V34.
    # REF_SUPER är robust referens, B52200_TRUE är hård budgetreferens.
    # SUPER*_PRESS testar om SUPER kan pressas ned mot 2500–2800 utan stort träfftapp.
    return ['B52200_TRUE','SUPER2800_PRESS','SUPER2600_PRESS','SUPER_B5_BRIDGE']


def _v31_cfg_for_variant(variant_id: str) -> dict:
    return (V15_VARIANT_DEFS.get(str(variant_id).upper(), {}) or {}).get('v31_cfg') or {}


def _v31_select_state(valid_states, hist_payout, args, variant_id: str):
    cfg = _v31_cfg_for_variant(variant_id)
    if not cfg or bool(cfg.get('control', False)):
        best = max(valid_states, key=lambda s: _v13_state_sort_key(s, hist_payout, args, variant_id))
        return best, {'status': 'REFERENS', 'ok': '', 'target': 0, 'min': 0, 'max': 0, 'delta': 0, 'pool_in_band': 0, 'pool_under_max': 0}
    target = int(cfg.get('target_rows', 2200) or 2200)
    lo = int(cfg.get('min_rows', max(1, target - 200)) or max(1, target - 200))
    hi = int(cfg.get('max_rows', target + 200) or target + 200)
    if bool(cfg.get('v35_downpress', False)):
        min_select_hist = int(cfg.get('min_select_hist', max(1, int(args.min_hit))) or int(args.min_hit))
        eligible = [s for s in valid_states if int(s.hist_hit) >= min_select_hist]
        if not eligible:
            eligible = list(valid_states)
        in_band = [s for s in eligible if lo <= int(s.frame_count) <= hi]
        under_max = [s for s in eligible if int(s.frame_count) <= hi]
        # V35: budgeten ska få verklig tyngd. Inom band prioriteras först testbar robusthet,
        # men om inget band finns väljer vi närmaste nedpressade state i stället för att
        # automatiskt falla tillbaka till bredaste SUPER-state.
        def v35_key(s):
            rows = int(s.frame_count)
            metrics = _v13_state_metrics(s, hist_payout, args, variant_id)
            over = max(0, rows - target)
            return (
                -abs(rows - target),
                -over,
                int(s.hist_hit),
                int(s.val_hit),
                float(metrics.get('payout_direction_pct', 0.0)),
                float(metrics.get('joint_score', 0.0)),
                -int(len(s.chosen)),
            )
        def v35_hit_key(s):
            rows = int(s.frame_count)
            metrics = _v13_state_metrics(s, hist_payout, args, variant_id)
            return (
                int(s.hist_hit),
                int(s.val_hit),
                -abs(rows - target),
                -max(0, rows - target),
                float(metrics.get('payout_direction_pct', 0.0)),
                float(metrics.get('joint_score', 0.0)),
            )
        if in_band:
            # Inom band: träff först, men radnärhet avgör mellan liknande states.
            best = max(in_band, key=v35_hit_key)
            status = 'INOM_BAND_V35'
            ok = 'JA'
        elif under_max:
            best = max(under_max, key=v35_hit_key)
            status = 'UNDER_MAX_V35'
            ok = 'NEJ'
        else:
            best = max(eligible, key=v35_key)
            status = 'NÄRMASTE_DOWNPRESS_V35'
            ok = 'NEJ'
        return best, {
            'status': status, 'ok': ok, 'target': target, 'min': lo, 'max': hi,
            'delta': int(int(best.frame_count) - target),
            'pool_in_band': int(len(in_band)), 'pool_under_max': int(len(under_max)),
        }
    in_band = [s for s in valid_states if lo <= int(s.frame_count) <= hi]
    under_max = [s for s in valid_states if int(s.frame_count) <= hi]
    # TRUE-audit: inom band väljs högsta historik/validering först, sedan närhet
    # till target. Om inget finns i band redovisas fallback öppet.
    def budget_key(s):
        rows = int(s.frame_count)
        metrics = _v13_state_metrics(s, hist_payout, args, variant_id)
        # Undvik att 22/30 slår 29/30 bara för att den är nära 2000.
        return (
            int(s.hist_hit),
            int(s.val_hit),
            -abs(rows - target),
            -max(0, rows - target),
            float(metrics.get('payout_direction_pct', 0.0)),
            float(metrics.get('joint_score', 0.0)),
            -int(len(s.chosen)),
        )
    if in_band:
        best = max(in_band, key=budget_key)
        status = 'INOM_BAND'
        ok = 'JA'
    elif under_max:
        best = max(under_max, key=budget_key)
        status = 'UNDER_MAX_MEN_UTANFÖR_BAND'
        ok = 'NEJ'
    else:
        # När inget 2k-band finns väljer vi närmaste ovanför, men flaggar att
        # paketmotorn inte hittade ett sant 2k-paket.
        best = max(valid_states, key=lambda s: (int(s.hist_hit), int(s.val_hit), -abs(int(s.frame_count)-target), -int(s.frame_count)))
        status = 'INGET_2K_BAND_HITTAT'
        ok = 'NEJ'
    return best, {
        'status': status,
        'ok': ok,
        'target': target,
        'min': lo,
        'max': hi,
        'delta': int(int(best.frame_count) - target),
        'pool_in_band': int(len(in_band)),
        'pool_under_max': int(len(under_max)),
    }


_ORIG_V31_BUILD_PACKAGE = _build_cluster_payout_package_v13_from_candidates


def _build_cluster_payout_package_v13_from_candidates(ns: dict, candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, frame_rows: list, frame: list, antal_matcher: int, args, variant_id: str):  # noqa: F811
    cfg = _v31_cfg_for_variant(variant_id)
    if not cfg:
        return _ORIG_V31_BUILD_PACKAGE(ns, candidates, htot, vtot, ftot, hist_payout, frame_rows, frame, antal_matcher, args, variant_id)

    raw_row_matrix = ns['_frame_row_matrix'](frame_rows, antal_matcher)
    sign_bits = _build_teckenskydd_bits(ns, raw_row_matrix, frame, antal_matcher)
    initial = _V9State(
        hist_mask=(1 << htot)-1,
        val_mask=(1 << vtot)-1 if vtot else 0,
        frame_mask=(1 << ftot)-1,
        hist_size=htot, val_size=vtot, frame_size=ftot,
        chosen=tuple(), used_keys=frozenset(), steps=tuple(),
    )
    val_floor_profile = int(math.ceil(vtot * float(args.min_package_val_pct) / 100.0)) if vtot else 0
    val_floor_structure = int(math.ceil(vtot * float(args.min_structure_package_val_pct) / 100.0)) if vtot else 0

    structure_cands = [c for c in candidates if _is_structure_candidate(c) and int(c.get('hist_hit', 0)) == htot]
    profile_cands = [c for c in candidates if _is_profile_candidate(c) and int(c.get('hist_hit', 0)) >= int(args.min_hit)]

    structure_states = [initial]
    if structure_cands:
        structure_states = _v13_beam_search(
            [initial], structure_cands,
            phase='structure', hist_floor=htot,
            val_floor_hit=val_floor_structure,
            sign_bits=sign_bits, min_unique_rows=int(args.min_unique_rows),
            payout_values=hist_payout, args=args, variant_id=variant_id,
        )
    structure_states.sort(key=lambda s: _v13_state_sort_key(s, hist_payout, args, variant_id), reverse=True)

    seeds = [initial]
    seen_masks = {(initial.hist_mask, initial.val_mask, initial.frame_mask)}
    for st in structure_states:
        sig = (st.hist_mask, st.val_mask, st.frame_mask)
        if sig in seen_masks:
            continue
        seen_masks.add(sig)
        seeds.append(st)
        if len(seeds) >= int(args.structure_seed_count):
            break

    finals = []
    for si, seed in enumerate(seeds, 1):
        print(f'    V31 {variant_id} profilbeam från strukturfrö {si}/{len(seeds)} · startrader={seed.frame_count}', flush=True)
        finals.extend(_v13_beam_search(
            [seed], profile_cands,
            phase='profile', hist_floor=int(args.min_hit),
            val_floor_hit=val_floor_profile,
            sign_bits=sign_bits, min_unique_rows=int(args.min_unique_rows),
            payout_values=hist_payout, args=args, variant_id=variant_id,
        ))
    finals.extend([st for st in structure_states if st.chosen])
    valid = [st for st in finals if st.chosen and st.hist_hit >= int(args.min_hit) and (vtot == 0 or st.val_hit >= val_floor_profile)]
    if not valid:
        return None, {'error': f'Ingen V31-state klarade paketkraven för variant {variant_id}.', 'candidates': len(candidates)}
    valid = _v13_dedupe(valid, hist_payout, args, args.archive_width, variant_id)
    best, budget_meta = _v31_select_state(valid, hist_payout, args, variant_id)
    # Cleanup kan göra paketet något bredare/smalare; behåll men uppdatera budgetmeta efteråt.
    best = _v9_redundancy_cleanup(best, hist_floor=int(args.min_hit), val_floor_hit=val_floor_profile, sign_bits=sign_bits)
    # Efter cleanup kan row_count ändras; redovisa faktisk avvikelse.
    if budget_meta.get('target'):
        budget_meta['delta'] = int(int(best.frame_count) - int(budget_meta.get('target', 0)))
        budget_meta['ok'] = 'JA' if int(budget_meta.get('min', 0)) <= int(best.frame_count) <= int(budget_meta.get('max', 0)) else 'NEJ'
        if budget_meta['ok'] != 'JA' and budget_meta.get('status') == 'INOM_BAND':
            budget_meta['status'] = 'INOM_BAND_FÖRE_CLEANUP'
    metrics = _v13_state_metrics(best, hist_payout, args, variant_id)
    package = {
        'variant': str(variant_id),
        'variant_label': V13_VARIANT_DEFS.get(str(variant_id), {}).get('label', str(variant_id)),
        'target': int(best.hist_hit),
        'target_label': f'V31 {variant_id} {best.hist_hit}/{htot}',
        'hist_hit': int(best.hist_hit), 'hist_total': int(htot),
        'val_hit': int(best.val_hit), 'val_total': int(vtot),
        'frame_start': int(ftot), 'frame_after': int(best.frame_count),
        'reduction_pct': float(metrics['reduction_pct']),
        'joint_score': float(metrics['joint_score']),
        'core_score': float(metrics['core_score']),
        'payout_lift_pct': float(metrics.get('payout_lift_pct', 0.0)),
        'payout_direction_pct': float(metrics.get('payout_direction_pct', metrics.get('payout_lift_pct', 0.0))),
        'removed_hist_count': int(metrics.get('removed_hist_count', 0)),
        'removed_payout_mean': float(metrics.get('removed_payout_mean', 0.0)),
        'cluster_mean': float(metrics['cluster_mean']),
        'num_filters': len(best.chosen),
        'filters': list(best.chosen), 'steps': list(best.steps),
        'package_type': f'V31 {variant_id} – {V13_VARIANT_DEFS.get(str(variant_id), {}).get("label", str(variant_id))}',
        'structure_filters': best.structure_filters,
        'profile_filters': best.profile_filters,
        'fat_filters': best.fat_filters,
        'edge_filters': best.edge_filters,
        'v31_budget_target': int(budget_meta.get('target', 0) or 0),
        'v31_budget_min': int(budget_meta.get('min', 0) or 0),
        'v31_budget_max': int(budget_meta.get('max', 0) or 0),
        'v31_budget_status': str(budget_meta.get('status','')),
        'v31_budget_ok': str(budget_meta.get('ok','')),
        'v31_budget_delta': int(budget_meta.get('delta', 0) or 0),
        'v31_pool_in_band': int(budget_meta.get('pool_in_band', 0) or 0),
        'v31_pool_under_max': int(budget_meta.get('pool_under_max', 0) or 0),
        'meta': {
            'engine': 'v31_true_2000_budget_audit',
            'variant': str(variant_id),
            'variant_label': V13_VARIANT_DEFS.get(str(variant_id), {}).get('label', str(variant_id)),
            'min_hit': int(args.min_hit),
            'min_package_val_pct': float(args.min_package_val_pct),
            'min_unique_rows': int(args.min_unique_rows),
            'candidate_count': len(candidates),
            'structure_candidates': len(structure_cands),
            'profile_candidates': len(profile_cands),
            'structure_seeds': len(seeds),
            'beam_width': int(args.beam_width),
            'archive_width': int(args.archive_width),
            'v31_budget': budget_meta,
        },
    }
    return package, package['meta']


def _v31_write_outputs(detail: pd.DataFrame, args, out_dir: Path, app_file: Path, db_file: Path):
    summary = _summarize_v15(detail)
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        summary = summary.copy()
        ok = detail[detail['Status'].astype(str) == 'OK'].copy() if 'Status' in detail.columns else detail.copy()
        extra = []
        for variant, grp in ok.groupby('Variant') if 'Variant' in ok.columns else []:
            med_budget_delta = float(pd.to_numeric(grp.get('V31 budgetavvikelse'), errors='coerce').dropna().median()) if 'V31 budgetavvikelse' in grp.columns else 0.0
            budget_ok = int((grp.get('V31 budget inom band', pd.Series(dtype=str)).astype(str) == 'JA').sum()) if 'V31 budget inom band' in grp.columns else 0
            # Fallback: äldre detaljrader kan sakna korrekt JA-markering trots att Paketrader ligger inom variantens band.
            if budget_ok == 0 and 'Paketrader' in grp.columns:
                cfg = _v31_cfg_for_variant(str(variant))
                if cfg and not bool(cfg.get('control', False)):
                    lo = int(cfg.get('min_rows', 0) or 0)
                    hi = int(cfg.get('max_rows', 10**12) or 10**12)
                    rows = pd.to_numeric(grp.get('Paketrader'), errors='coerce')
                    budget_ok = int(((rows >= lo) & (rows <= hi)).sum())
                    if len(rows.dropna()):
                        target = int(cfg.get('target_rows', 2200) or 2200)
                        med_budget_delta = float((rows - target).abs().median())
            status_mode = ''
            if 'V31 budget status' in grp.columns and len(grp['V31 budget status'].dropna()):
                try:
                    status_mode = str(grp['V31 budget status'].astype(str).value_counts().idxmax())
                except Exception:
                    status_mode = ''
            extra.append({'Variant': variant, 'Budget OK omgångar': budget_ok, 'Median budgetavvikelse': med_budget_delta, 'Vanlig budgetstatus': status_mode})
        if extra:
            summary = summary.merge(pd.DataFrame(extra), on='Variant', how='left')
        if 'Beslutsnotis' not in summary.columns:
            summary['Beslutsnotis'] = ''
        for i, r in summary.iterrows():
            v = str(r.get('Variant',''))
            hits = int(r.get('Träffar',0) or 0)
            med_rows = float(r.get('Median paketrader',0) or 0)
            budget_ok = int(r.get('Budget OK omgångar',0) or 0) if 'Budget OK omgångar' in summary.columns else 0
            note = ''
            if v.startswith('REF_'):
                note = 'REFERENS – ej 2k-krav'
            elif med_rows <= 2600 and hits >= 17:
                note = 'REPAIR-KANDIDAT: klarar 17/20 under ca 2600'
            elif med_rows <= 2600 and hits >= 16:
                note = 'NÄRA: behöver miss-audit'
            elif med_rows > 2700:
                note = 'För bred för 400-raderspipeline'
            else:
                note = 'Audit'
            if budget_ok and not v.startswith('REF_'):
                note += f' · band OK {budget_ok}/{int(getattr(args, "max_tests", 12) or 12)}'
            summary.loc[i,'Beslutsnotis'] = note
        summary['Vinnare']=''
        def win_key(r):
            v = str(r.get('Variant',''))
            hits = int(r.get('Träffar',0) or 0)
            med_rows = float(r.get('Median paketrader',10**9) or 10**9)
            budget_penalty = 0 if med_rows <= 2600 else 1000
            ref_penalty = 500 if v.startswith('REF_') else 0
            return (hits, -budget_penalty, -ref_penalty, -abs(med_rows-2200), -med_rows)
        best_idx=max(summary.index, key=lambda i: win_key(summary.loc[i]))
        summary.loc[best_idx,'Vinnare']='JA'
    winner_variant = _v13_winner_variant(summary) if isinstance(summary, pd.DataFrame) and not summary.empty else None
    detail_path = out_dir / f'{args.output_prefix}_detail.csv'
    summary_path = out_dir / f'{args.output_prefix}_summary.csv'
    report_path = out_dir / f'{args.output_prefix}_report.txt'
    winner_detail_path = out_dir / f'{args.output_prefix}_winner_detail.csv'
    detail.to_csv(detail_path, index=False)
    if isinstance(summary, pd.DataFrame):
        summary.to_csv(summary_path, index=False)
        if winner_variant and 'Variant' in detail.columns:
            detail[detail['Variant'].astype(str) == str(winner_variant)].to_csv(winner_detail_path, index=False)
    lines = []
    lines.append('TIPSET AI – V36 RANDOM STRATIFIED SUPER SCREEN8')
    lines.append('')
    lines.append('Syfte: testa SUPER-downpress på slumpade/stratifierade omgångar inom utdelningsintervallet, inte bara senaste perioden.')
    lines.append(f'Appfil: {app_file}')
    lines.append(f'Databas: {db_file}')
    lines.append(f'Output prefix: {args.output_prefix}')
    lines.append('')
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis'] if c in summary.columns]
        lines.append('SAMMANFATTNING')
        lines.append(summary[cols].to_string(index=False))
        lines.append('')
        lines.append(f'Vinnare: {winner_variant}')
        lines.append('')
        lines.append('TOLKNING V36: detta är snabb slumpkontroll. En variant är intressant om den klarar minst 6/8 och median helst 2400–2800. Kör flera seeds innan 20/30-test.')
    lines.append('')
    lines.append(f'Detail: {detail_path}')
    lines.append(f'Summary: {summary_path}')
    lines.append(f'Winner detail: {winner_detail_path}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print('\nKLART – V36 RANDOM STRATIFIED SUPER SCREEN8', flush=True)
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        show_cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis'] if c in summary.columns]
        print(summary[show_cols].to_string(index=False), flush=True)
    print('Rapport:', str(report_path), flush=True)
    print('Summary:', str(summary_path), flush=True)
    print('Detail:', str(detail_path), flush=True)



def _v36_random_reorder_db_for_tests(ns: dict, global_db: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lägger slumpade testomgångar först i databasen så befintlig _run_backtest_v15
    kan återanvändas utan att ändra historik-/similarity-logiken.

    sample-mode stratified_payout:
      100k-250k, 250k-500k, 500k-1M, 1M-2.5M
    fylls jämnt så långt det går. Resterande platser fylls slumpat från hela
    giltiga intervallet. Historiesökningen använder fortfarande hela global_db
    och exclude_index fungerar med ursprungliga index.
    """
    try:
        max_tests = int(getattr(args, 'max_tests', 8) or 8)
        seed = int(getattr(args, 'random_seed', 20260720) or 20260720)
        mode = str(getattr(args, 'sample_mode', 'stratified_payout') or 'stratified_payout')
        rng = np.random.default_rng(seed)
        work_db, _ = _prepare_supersvar_payout(global_db.copy(), int(args.pay_max))
        eligible = _prepare_db(ns, work_db, 13, int(args.pay_min), int(args.pay_max)).copy()
        if eligible.empty or max_tests <= 0:
            return global_db, pd.DataFrame()
        eligible['__payout'] = pd.to_numeric(eligible.get('Payout', 0), errors='coerce').fillna(0.0)
        selected = []
        strata_log = []
        if mode == 'random':
            take = min(max_tests, len(eligible))
            selected = list(rng.choice(list(eligible.index), size=take, replace=False))
            strata_log.append({'Stratum': 'RANDOM_HELA_INTERVALLET', 'Antal': len(selected)})
        else:
            strata = [
                ('100k-250k', 100000, 250000),
                ('250k-500k', 250000, 500000),
                ('500k-1M', 500000, 1000000),
                ('1M-2.5M', 1000000, int(args.pay_max) + 1),
            ]
            base_q = max_tests // len(strata)
            rem = max_tests % len(strata)
            quotas = [base_q + (1 if i < rem else 0) for i in range(len(strata))]
            used = set()
            for (label, lo, hi), q in zip(strata, quotas):
                part = eligible[(eligible['__payout'] >= lo) & (eligible['__payout'] < hi)]
                avail = [idx for idx in part.index if idx not in used]
                take = min(int(q), len(avail))
                if take > 0:
                    pick = list(rng.choice(avail, size=take, replace=False))
                    selected.extend(pick)
                    used.update(pick)
                strata_log.append({'Stratum': label, 'Kvot': int(q), 'Tillgängliga': len(avail), 'Valda': int(take)})
            if len(selected) < max_tests:
                remaining = [idx for idx in eligible.index if idx not in set(selected)]
                take = min(max_tests - len(selected), len(remaining))
                if take > 0:
                    extra = list(rng.choice(remaining, size=take, replace=False))
                    selected.extend(extra)
                    strata_log.append({'Stratum': 'FYLLNAD_RANDOM', 'Kvot': int(take), 'Tillgängliga': len(remaining), 'Valda': int(take)})
        # Slumpa ordningen även efter stratifieringen så samma strata inte alltid körs först.
        selected = list(dict.fromkeys(selected))[:max_tests]
        if len(selected) > 1:
            selected = list(rng.permutation(selected))
        first = global_db.loc[selected]
        rest = global_db.drop(index=selected, errors='ignore')
        reordered = pd.concat([first, rest], axis=0)
        rows = []
        for order, idx in enumerate(selected, 1):
            r = eligible.loc[idx] if idx in eligible.index else global_db.loc[idx]
            rows.append({
                'Testordning': order,
                'OriginalIndex': idx,
                'Datum': str(r.get('Datum', ''))[:10],
                'Utdelning': float(pd.to_numeric(pd.Series([r.get('Payout', 0)]), errors='coerce').fillna(0).iloc[0]),
            })
        log_df = pd.DataFrame(rows)
        if strata_log:
            log_df.attrs['strata_log'] = strata_log
        return reordered, log_df
    except Exception as e:
        print(f'V36 slumpurval misslyckades, faller tillbaka till ordinarie ordning: {e}', flush=True)
        return global_db, pd.DataFrame()

def main_v31():
    parser=argparse.ArgumentParser(description='Tipset AI V36 – random/stratifierad SUPER-downpress screen 8 omgångar.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    parser.add_argument('--output-prefix', default='package_super_downpress_v36_random8')
    parser.add_argument('--variants', default='IGNORED')
    parser.add_argument('--max-tests', type=int, default=8)
    parser.add_argument('--test-offset', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=20260720, help='Seed för stratifierad slumpning av testomgångar.')
    parser.add_argument('--sample-mode', default='stratified_payout', choices=['stratified_payout','random'], help='Hur testomgångar ska väljas innan backtest.')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--wide-n', type=int, default=35)
    parser.add_argument('--pay-min', type=int, default=100000)
    parser.add_argument('--pay-max', type=int, default=2500000)
    parser.add_argument('--filter-hist-target-pct', type=int, default=95)
    parser.add_argument('--frame-profile', default='3-5-5')
    parser.add_argument('--frames', default='3-5-5')
    parser.add_argument('--mode', default='leave-one-out', choices=['leave-one-out','kronologiskt'])
    parser.add_argument('--candidate-min-hit', type=int, default=25)
    parser.add_argument('--min-candidate-val-pct', type=float, default=58.0)
    parser.add_argument('--min-structure-val-pct', type=float, default=64.0)
    parser.add_argument('--min-gap-score', type=float, default=0.55)
    parser.add_argument('--variants-per-key', type=int, default=4)
    parser.add_argument('--max-candidates', type=int, default=220)
    parser.add_argument('--fast-no-supermakro', dest='fast_no_supermakro', action='store_true', default=True)
    parser.add_argument('--include-supermakro', dest='fast_no_supermakro', action='store_false')
    parser.add_argument('--min-hit', type=int, default=23)
    parser.add_argument('--min-package-val-pct', type=float, default=60.0)
    parser.add_argument('--min-structure-package-val-pct', type=float, default=76.0)
    parser.add_argument('--min-unique-rows', type=int, default=10)
    parser.add_argument('--beam-width', type=int, default=60)
    parser.add_argument('--archive-width', type=int, default=700)
    parser.add_argument('--structure-seed-count', type=int, default=2)
    parser.add_argument('--hit-power', type=float, default=2.55)
    parser.add_argument('--validation-power', type=float, default=0.88)
    parser.add_argument('--reduction-power', type=float, default=3.8)
    parser.add_argument('--payout-weight', type=float, default=0.02)
    parser.add_argument('--cluster-weight', type=float, default=0.05)
    parser.add_argument('--payout-direction-weight', type=float, default=0.03)
    parser.add_argument('--bundle-pool-size', type=int, default=10)
    parser.add_argument('--triple-pool-size', type=int, default=0)
    parser.add_argument('--max-bundle-trials-per-state', type=int, default=25)
    parser.add_argument('--max-bundle-keep-per-state', type=int, default=8)
    parser.add_argument('--row-bucket-size', type=int, default=240)
    parser.add_argument('--per-bucket-keep', type=int, default=8)
    parser.add_argument('--enable-triples', action='store_true', default=False)
    parser.add_argument('--v15-group-max-filters', type=int, default=24)
    parser.add_argument('--v15-max-group-candidates', type=int, default=120)
    parser.add_argument('--v15-min-group-size', type=int, default=3)
    parser.add_argument('--v15-cross-per-family', type=int, default=4)
    parser.add_argument('--v15-cross-max-filters', type=int, default=16)
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)
    # Hårdlås V35. Inget gammalt Colab-arg ska ändra upplägget.
    args.max_tests = 8
    args.test_offset = 0
    args.top_n = 30
    args.wide_n = 35
    args.output_prefix = 'package_super_downpress_v36_random8'
    args.variants = ','.join(_v31_parse_variants(args.variants))
    args.frames = '3-5-5'
    frames = _v17_parse_frames(args.frames)
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n' + '='*96, flush=True)
    print('TIPSET AI V36 – RANDOM STRATIFIED SUPER SCREEN8', flush=True)
    print('LÅST LOGIK: stratifierad slump inom 100k–2.5M. Få varianter, mindre beam, ej finalbetyg.', flush=True)
    print('Testomgångar: 8 (hårdlåst, stratifierad slump inom utdelningsintervall)', flush=True)
    print('Ramar: 3-5-5 (7776 grundrader)', flush=True)
    print(f'Varianter: {args.variants}', flush=True)
    print(f'Slumpseed: {args.random_seed} · sample-mode: {args.sample_mode}', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)
    for frame in frames:
        frame_args=argparse.Namespace(**vars(args))
        frame_args.frames=frame
        frame_args.frame_profile=frame
        frame_args.output_prefix=f'{args.output_prefix}_{_v17_safe_prefix_part(frame)}'
        print('\n'+'-'*96, flush=True)
        print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
        print('-'*96, flush=True)
        ns=_load_app_functions(app_file, fast_no_supermakro=False)
        db=ns['load_database'](str(db_file),13)
        detail, meta = _run_backtest_v15(ns, db, frame_args)
        _v31_write_outputs(detail, frame_args, out_dir, app_file, db_file)
        del ns, db, detail
        gc.collect()
    combined=_v30_collect_frames(out_dir,args.output_prefix,frames)
    if combined.get('summary') is not None:
        print('\nKLART – V36 ALLA RAMAR', flush=True)
        print(combined['summary'].to_string(index=False), flush=True)
        print('Summary:', str(Path(out_dir)/f'{args.output_prefix}_ALL_FRAMES_summary.csv'), flush=True)




# =============================================================================
# V38 – B52200 FINAL30 RANDOM/STRATIFIED
# =============================================================================
# Praktisk Colab-version: inga %run-argument krävs.
# Kör ett sluttest på 30 slumpade/stratifierade omgångar inom 100k–2.5M.
# Bara huvudkandidaten B52200_TRUE mot REF_SUPER.

def main_v38():
    parser=argparse.ArgumentParser(description='Tipset AI V38 – B52200 Final30 Random/Stratified utan kommandoradsargument.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)

    # Hårdlåst upplägg: inga argument behövs.
    final_seed = 20260720
    variants_locked = 'B52200_TRUE,REF_SUPER'
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*96, flush=True)
    print('TIPSET AI V38 – B52200 FINAL30 RANDOM/STRATIFIED', flush=True)
    print('LÅST LOGIK: sluttest på 30 slumpade/stratifierade omgångar. Inga %run-argument behövs.', flush=True)
    print('Syfte: avgöra om B52200_TRUE är app-kandidat mot REF_SUPER.', flush=True)
    print('Testomgångar: 30 totalt · random/stratifierat inom utdelningsintervallet 100k–2.5M', flush=True)
    print('Ramar: 3-5-5 (7776 grundrader)', flush=True)
    print(f'Seed: {final_seed}', flush=True)
    print(f'Varianter: {variants_locked}', flush=True)
    print('Kravsignal: B52200_TRUE bör klara minst 25/30 och median helst 2 200–2 500 rader.', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)

    frame = '3-5-5'
    run_args=argparse.Namespace(
        app_file=str(app_file), db_file=str(db_file), out_dir=str(out_dir),
        output_prefix=f'package_b52200_final30_random_v38_{_v17_safe_prefix_part(frame)}',
        variants=variants_locked,
        max_tests=30, test_offset=0, random_seed=int(final_seed), sample_mode='stratified_payout',
        top_n=30, wide_n=35, pay_min=100000, pay_max=2500000,
        filter_hist_target_pct=95, frame_profile=frame, frames=frame, mode='leave-one-out',
        candidate_min_hit=25, min_candidate_val_pct=58.0, min_structure_val_pct=64.0,
        min_gap_score=0.55, variants_per_key=4, max_candidates=220,
        fast_no_supermakro=True, min_hit=23, min_package_val_pct=60.0,
        min_structure_package_val_pct=76.0, min_unique_rows=10,
        beam_width=60, archive_width=700, structure_seed_count=2,
        hit_power=2.55, validation_power=0.88, reduction_power=3.9,
        payout_weight=0.02, cluster_weight=0.05, payout_direction_weight=0.03,
        bundle_pool_size=10, triple_pool_size=0, max_bundle_trials_per_state=25,
        max_bundle_keep_per_state=8, row_bucket_size=230, per_bucket_keep=8,
        enable_triples=False, v15_group_max_filters=24, v15_max_group_candidates=120,
        v15_min_group_size=3, v15_cross_per_family=4, v15_cross_max_filters=16,
    )

    print('\n'+'-'*96, flush=True)
    print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
    print('-'*96, flush=True)

    ns=_load_app_functions(app_file, fast_no_supermakro=False)
    db=ns['load_database'](str(db_file),13)
    db, sample_df = _v36_random_reorder_db_for_tests(ns, db, run_args)
    sample_path = out_dir / 'package_b52200_final30_random_v38_sample.csv'
    if isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
        sample_df.to_csv(sample_path, index=False)
        print('Slumpade/stratifierade testomgångar:', flush=True)
        print(sample_df.to_string(index=False), flush=True)
        print('Sample:', str(sample_path), flush=True)

    detail, meta = _run_backtest_v15(ns, db, run_args)
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        raise SystemExit('Inga V38-resultat skapades.')

    detail = detail.copy()
    detail.insert(0, 'Ramprofil', frame)
    detail_path = out_dir / 'package_b52200_final30_random_v38_detail.csv'
    detail.to_csv(detail_path, index=False)

    # Vanliga outputs från befintlig writer.
    _v31_write_outputs(detail.drop(columns=['Ramprofil'], errors='ignore'), run_args, out_dir, app_file, db_file)

    # Extra kompakt total-summary.
    summary = _summarize_v15(detail.drop(columns=['Ramprofil'], errors='ignore'))
    summary_path = out_dir / 'package_b52200_final30_random_v38_summary.csv'
    summary.to_csv(summary_path, index=False)

    # Lägg till beslutssignal.
    decision_rows=[]
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        for _, row in summary.iterrows():
            variant=str(row.get('Variant',''))
            hits=float(row.get('Träffar', 0) or 0)
            med_rows=float(row.get('Median paketrader', 0) or 0)
            if variant == 'B52200_TRUE':
                if hits >= 25 and med_rows <= 2500:
                    decision = 'GODKÄND FINALIST – kandidat för app-huvudläge'
                elif hits >= 25:
                    decision = 'TRÄFF GODKÄND men radantal behöver kontrolleras'
                elif hits >= 24 and med_rows <= 2500:
                    decision = 'NÄRA – kräver miss-audit eller skyddsregel'
                else:
                    decision = 'INTE FINAL – för låg träff eller för bred'
            else:
                decision = 'REFERENS – ej 2k-krav'
            decision_rows.append({'Variant': variant, 'V38 beslut': decision})
    decision_df = pd.DataFrame(decision_rows)
    decision_path = out_dir / 'package_b52200_final30_random_v38_decision.csv'
    decision_df.to_csv(decision_path, index=False)
    if not decision_df.empty and 'Variant' in summary.columns:
        summary_show = summary.merge(decision_df, on='Variant', how='left')
    else:
        summary_show = summary

    report_path = out_dir / 'package_b52200_final30_random_v38_report.txt'
    lines=[]
    lines.append('TIPSET AI V38 – B52200 FINAL30 RANDOM/STRATIFIED')
    lines.append('='*96)
    lines.append(f'Seed: {final_seed}')
    lines.append('Varianter: B52200_TRUE, REF_SUPER')
    lines.append('Utdelningsurval: stratifierad slump 100k–2.5M')
    lines.append('Mål: B52200_TRUE minst 25/30 och median helst 2 200–2 500 rader.')
    lines.append('Pipeline: 3-5-5 grundram 7776 → paketmotor ca 2k–2.5k → 12-rättsgaranti ca 400 spelrader.')
    lines.append('')
    lines.append('SAMMANFATTNING')
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V38 beslut'] if c in summary_show.columns]
        lines.append(summary_show[cols].to_string(index=False))
    else:
        lines.append('(tom)')
    lines.append('')
    lines.append(f'Sample: {sample_path}')
    lines.append(f'Detail: {detail_path}')
    lines.append(f'Summary: {summary_path}')
    lines.append(f'Decision: {decision_path}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print('\nKLART – V38 B52200 FINAL30 RANDOM/STRATIFIED', flush=True)
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        show_cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V38 beslut'] if c in summary_show.columns]
        print(summary_show[show_cols].to_string(index=False), flush=True)
    print('Rapport:', str(report_path), flush=True)
    print('Summary:', str(summary_path), flush=True)
    print('Detail:', str(detail_path), flush=True)
    print('Sample:', str(sample_path), flush=True)
    print('Decision:', str(decision_path), flush=True)

    try:
        if '_orig_prepare_db_v44b' in locals() and _orig_prepare_db_v44b is not None:
            globals()['_prepare_db'] = _orig_prepare_db_v44b
    except Exception:
        pass

    del ns, db, detail
    gc.collect()


# =============================================================================
# V39 – B52200 MISS AUDIT / GUARD
# =============================================================================
# V38 gav B52200_TRUE 24/30 vid ca 2315 medianrader. V39 återanvänder samma
# finalseed och samma stratifierade 30-urval, men testar få skyddsvarianter.
# Målet är konkret: rädda minst en B52200-miss utan att medianen går över ca 2600.

V39_VARIANT_DEFS = {
    'B52200_GUARD_VAL': {
        'label': 'B52200_GUARD_VAL valideringsskydd 2.0-2.6k',
        'description': 'B5/B52200 med skydd mot låg bred validering. Väljer hellre robustare state i 2.0-2.6k-bandet än maximal radpress.',
        'base': 'B5',
        'v31_cfg': {'target_rows': 2400, 'min_rows': 2000, 'max_rows': 2600, 'strict': True, 'v39_guard': 'val', 'min_val_hit': 33, 'max_filters_select': 18, 'preferred_max_filters': 15},
        'overrides': {'beam_width': 115, 'archive_width': 1400, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 60.0, 'min_candidate_val_pct': 58.0, 'min_unique_rows': 10, 'reduction_power': 3.55, 'validation_power': 1.05, 'hit_power': 2.85, 'row_bucket_size': 250, 'per_bucket_keep': 9},
    },
    'B52200_GUARD_CAP16': {
        'label': 'B52200_GUARD_CAP16 max 16 filter 2.0-2.6k',
        'description': 'B5/B52200 med filtertak i paketvalet. Hindrar de mest sköra 18-22-filterpaketen från att vinna om ett rimligt alternativ finns.',
        'base': 'B5',
        'v31_cfg': {'target_rows': 2350, 'min_rows': 2000, 'max_rows': 2600, 'strict': True, 'v39_guard': 'cap', 'min_val_hit': 31, 'max_filters_select': 16, 'preferred_max_filters': 14},
        'overrides': {'beam_width': 115, 'archive_width': 1400, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 58.0, 'min_candidate_val_pct': 56.0, 'min_unique_rows': 9, 'reduction_power': 3.75, 'validation_power': 0.94, 'hit_power': 2.65, 'row_bucket_size': 245, 'per_bucket_keep': 9},
    },
    'B52200_GUARD_BACKSTEP': {
        'label': 'B52200_GUARD_BACKSTEP backa ett steg 2.2-2.7k',
        'description': 'B5/B52200 med kontrollerat backsteg. Tillåter 2.2-2.7k om det ger tydligt bättre historik/validering och färre hårda filter.',
        'base': 'B5',
        'v31_cfg': {'target_rows': 2500, 'min_rows': 2200, 'max_rows': 2700, 'strict': True, 'v39_guard': 'backstep', 'min_val_hit': 33, 'max_filters_select': 18, 'preferred_max_filters': 15},
        'overrides': {'beam_width': 120, 'archive_width': 1500, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 60.0, 'min_candidate_val_pct': 58.0, 'min_unique_rows': 10, 'reduction_power': 3.35, 'validation_power': 1.08, 'hit_power': 2.95, 'row_bucket_size': 270, 'per_bucket_keep': 10},
    },
}


def _v39_register_variants():
    for key, cfg in V39_VARIANT_DEFS.items():
        base_key = str(cfg.get('base','B5')).upper()
        base = dict(V15_VARIANT_DEFS.get(base_key) or V15_VARIANT_DEFS.get('B5') or {})
        if not base:
            raise RuntimeError(f'Kan inte registrera V39 {key}: bas {base_key} saknas.')
        new = dict(base)
        new['label'] = cfg['label']
        new['description'] = cfg['description']
        new['v31_cfg'] = dict(cfg.get('v31_cfg') or {})
        ov = dict(new.get('overrides') or {})
        ov.update(cfg.get('overrides') or {})
        new['overrides'] = ov
        V15_VARIANT_DEFS[key] = new
        V19_VARIANT_DEFS[key] = new
        V13_VARIANT_DEFS[key] = {
            'label': cfg['label'],
            'fixed_payout': True,
            'bundle_search': bool(V13_VARIANT_DEFS.get(base_key, {}).get('bundle_search', False)),
            'description': cfg['description'],
        }


_v39_register_variants()

_ORIG_V39_SELECT_STATE = _v31_select_state


def _v31_select_state(valid_states, hist_payout, args, variant_id: str):  # noqa: F811
    cfg = _v31_cfg_for_variant(variant_id)
    if not cfg or not cfg.get('v39_guard'):
        return _ORIG_V39_SELECT_STATE(valid_states, hist_payout, args, variant_id)

    target = int(cfg.get('target_rows', 2400) or 2400)
    lo = int(cfg.get('min_rows', max(1, target - 300)) or max(1, target - 300))
    hi = int(cfg.get('max_rows', target + 300) or target + 300)
    guard = str(cfg.get('v39_guard', '')).lower()
    min_val_hit = int(cfg.get('min_val_hit', 0) or 0)
    max_filters_select = int(cfg.get('max_filters_select', 999) or 999)
    preferred_max_filters = int(cfg.get('preferred_max_filters', max_filters_select) or max_filters_select)

    states = list(valid_states)
    if not states:
        return _ORIG_V39_SELECT_STATE(valid_states, hist_payout, args, variant_id)

    def rows(s): return int(s.frame_count)
    def nf(s): return int(len(s.chosen))
    def metrics(s): return _v13_state_metrics(s, hist_payout, args, variant_id)

    # Bygg nivåer av kandidater. Hårda guardkrav används först, men vi faller
    # tillbaka om inget finns så varianten inte kraschar på en svår omgång.
    in_band = [s for s in states if lo <= rows(s) <= hi]
    under_max = [s for s in states if rows(s) <= hi]
    pool = in_band or under_max or states

    guarded = [s for s in pool if nf(s) <= max_filters_select and (min_val_hit <= 0 or int(s.val_hit) >= min_val_hit)]
    if not guarded and guard == 'cap':
        guarded = [s for s in pool if nf(s) <= max_filters_select]
    if not guarded and min_val_hit > 0:
        guarded = [s for s in pool if int(s.val_hit) >= max(0, min_val_hit - 1)]
    if not guarded:
        guarded = pool

    # Guardnycklarna prioriterar fortfarande träff, men lägger in skydd mot
    # många hårda filter och låg validering. Backstep får lite mer tolerans för
    # radantal över target om robustheten förbättras.
    def guard_key(s):
        m = metrics(s)
        r = rows(s)
        over = max(0, r - target)
        dist = abs(r - target)
        too_many = max(0, nf(s) - preferred_max_filters)
        in_band_flag = 1 if lo <= r <= hi else 0
        if guard == 'backstep':
            return (
                int(s.hist_hit),
                int(s.val_hit),
                -too_many,
                in_band_flag,
                -max(0, r - hi),
                -abs(r - target),
                float(m.get('payout_direction_pct', 0.0)),
                float(m.get('joint_score', 0.0)),
                -nf(s),
            )
        if guard == 'val':
            return (
                int(s.hist_hit),
                int(s.val_hit),
                in_band_flag,
                -too_many,
                -dist,
                -over,
                float(m.get('payout_direction_pct', 0.0)),
                float(m.get('joint_score', 0.0)),
                -nf(s),
            )
        # cap
        return (
            int(s.hist_hit),
            in_band_flag,
            -too_many,
            int(s.val_hit),
            -dist,
            -over,
            float(m.get('payout_direction_pct', 0.0)),
            float(m.get('joint_score', 0.0)),
            -nf(s),
        )

    best = max(guarded, key=guard_key)
    ok = 'JA' if lo <= int(best.frame_count) <= hi else 'NEJ'
    status = f'V39_GUARD_{guard.upper()}'
    if guarded is not pool:
        status += '_FILTERED'
    return best, {
        'status': status,
        'ok': ok,
        'target': target,
        'min': lo,
        'max': hi,
        'delta': int(int(best.frame_count) - target),
        'pool_in_band': int(len(in_band)),
        'pool_under_max': int(len(under_max)),
    }


def _v39_rescue_table(detail: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return pd.DataFrame()
    need_cols = {'Variant','Datum','Paket klarar facit'}
    if not need_cols.issubset(set(detail.columns)):
        return pd.DataFrame()
    ok = detail[detail.get('Status','OK').astype(str).eq('OK')].copy() if 'Status' in detail.columns else detail.copy()
    ok['HitBool'] = ok['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1'])
    piv = ok.pivot_table(index='Datum', columns='Variant', values='HitBool', aggfunc='max')
    rows = []
    if 'B52200_TRUE' not in piv.columns:
        return pd.DataFrame()
    b_misses = piv[piv['B52200_TRUE'] == False]
    for dt, row in b_misses.iterrows():
        rec = {'Datum': dt, 'B52200_TRUE': 'MISS'}
        for v in ['REF_SUPER','B52200_GUARD_VAL','B52200_GUARD_CAP16','B52200_GUARD_BACKSTEP']:
            if v in piv.columns:
                rec[v] = 'RÄDDAR' if bool(row.get(v, False)) else 'MISS'
        # Lägg till utdelning och radantal per variant om möjligt.
        sub = ok[ok['Datum'].astype(str).eq(str(dt))]
        if 'Utdelning' in sub.columns and not sub.empty:
            rec['Utdelning'] = int(pd.to_numeric(sub['Utdelning'], errors='coerce').dropna().iloc[0]) if not pd.to_numeric(sub['Utdelning'], errors='coerce').dropna().empty else None
        for v in ['B52200_TRUE','REF_SUPER','B52200_GUARD_VAL','B52200_GUARD_CAP16','B52200_GUARD_BACKSTEP']:
            vv = sub[sub['Variant'].astype(str).eq(v)]
            if not vv.empty and 'Paketrader' in vv.columns:
                rec[f'{v} rader'] = int(pd.to_numeric(vv['Paketrader'], errors='coerce').dropna().iloc[0]) if not pd.to_numeric(vv['Paketrader'], errors='coerce').dropna().empty else None
        rows.append(rec)
    return pd.DataFrame(rows)


def main_v39():
    parser=argparse.ArgumentParser(description='Tipset AI V39 – B52200 Miss Audit / Guard utan kommandoradsargument.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)

    final_seed = 20260720
    variants_locked = 'B52200_TRUE,B52200_GUARD_VAL,B52200_GUARD_CAP16,B52200_GUARD_BACKSTEP,REF_SUPER'
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*96, flush=True)
    print('TIPSET AI V39 – B52200 MISS AUDIT / GUARD', flush=True)
    print('LÅST LOGIK: samma random/stratifierade 30-urval som V38, men med smala skyddsvarianter.', flush=True)
    print('Syfte: rädda minst 1 B52200-miss utan att medianen går över ca 2 500–2 600 rader.', flush=True)
    print('Testomgångar: 30 totalt · random/stratifierat inom 100k–2.5M', flush=True)
    print('Ramar: 3-5-5 (7776 grundrader)', flush=True)
    print(f'Seed: {final_seed}', flush=True)
    print(f'Varianter: {variants_locked}', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)

    frame = '3-5-5'
    run_args=argparse.Namespace(
        app_file=str(app_file), db_file=str(db_file), out_dir=str(out_dir),
        output_prefix=f'package_b52200_miss_guard_v39_{_v17_safe_prefix_part(frame)}',
        variants=variants_locked,
        max_tests=30, test_offset=0, random_seed=int(final_seed), sample_mode='stratified_payout',
        top_n=30, wide_n=35, pay_min=100000, pay_max=2500000,
        filter_hist_target_pct=95, frame_profile=frame, frames=frame, mode='leave-one-out',
        candidate_min_hit=25, min_candidate_val_pct=58.0, min_structure_val_pct=64.0,
        min_gap_score=0.55, variants_per_key=4, max_candidates=220,
        fast_no_supermakro=True, min_hit=23, min_package_val_pct=60.0,
        min_structure_package_val_pct=76.0, min_unique_rows=10,
        beam_width=60, archive_width=700, structure_seed_count=2,
        hit_power=2.55, validation_power=0.88, reduction_power=3.9,
        payout_weight=0.02, cluster_weight=0.05, payout_direction_weight=0.03,
        bundle_pool_size=10, triple_pool_size=0, max_bundle_trials_per_state=25,
        max_bundle_keep_per_state=8, row_bucket_size=230, per_bucket_keep=8,
        enable_triples=False, v15_group_max_filters=24, v15_max_group_candidates=120,
        v15_min_group_size=3, v15_cross_per_family=4, v15_cross_max_filters=16,
    )

    print('\n'+'-'*96, flush=True)
    print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
    print('-'*96, flush=True)

    ns=_load_app_functions(app_file, fast_no_supermakro=False)
    db=ns['load_database'](str(db_file),13)
    db, sample_df = _v36_random_reorder_db_for_tests(ns, db, run_args)
    sample_path = out_dir / 'package_b52200_miss_guard_v39_sample.csv'
    if isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
        sample_df.to_csv(sample_path, index=False)
        print('Slumpade/stratifierade testomgångar:', flush=True)
        print(sample_df.to_string(index=False), flush=True)
        print('Sample:', str(sample_path), flush=True)

    detail, meta = _run_backtest_v15(ns, db, run_args)
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        raise SystemExit('Inga V39-resultat skapades.')

    detail = detail.copy()
    detail.insert(0, 'Ramprofil', frame)
    detail_path = out_dir / 'package_b52200_miss_guard_v39_detail.csv'
    detail.to_csv(detail_path, index=False)
    _v31_write_outputs(detail.drop(columns=['Ramprofil'], errors='ignore'), run_args, out_dir, app_file, db_file)

    summary = _summarize_v15(detail.drop(columns=['Ramprofil'], errors='ignore'))
    summary_path = out_dir / 'package_b52200_miss_guard_v39_summary.csv'
    summary.to_csv(summary_path, index=False)

    rescue = _v39_rescue_table(detail.drop(columns=['Ramprofil'], errors='ignore'))
    rescue_path = out_dir / 'package_b52200_miss_guard_v39_rescue_audit.csv'
    rescue.to_csv(rescue_path, index=False)

    decision_rows=[]
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        for _, row in summary.iterrows():
            variant=str(row.get('Variant',''))
            hits=float(row.get('Träffar', 0) or 0)
            med_rows=float(row.get('Median paketrader', 0) or 0)
            if variant.startswith('B52200_GUARD'):
                if hits >= 25 and med_rows <= 2600:
                    decision = 'GODKÄND GUARD – räddar finalkravet'
                elif hits >= 25:
                    decision = 'TRÄFF OK men för bred'
                elif hits >= 24 and med_rows <= 2600:
                    decision = 'NÄRA – sparas som skyddskandidat'
                else:
                    decision = 'INTE GUARD – för låg träff eller för bred'
            elif variant == 'B52200_TRUE':
                decision = 'KONTROLL – V38 nära men 24/30'
            else:
                decision = 'REFERENS – ej 2k-krav'
            decision_rows.append({'Variant': variant, 'V39 beslut': decision})
    decision_df = pd.DataFrame(decision_rows)
    decision_path = out_dir / 'package_b52200_miss_guard_v39_decision.csv'
    decision_df.to_csv(decision_path, index=False)
    summary_show = summary.merge(decision_df, on='Variant', how='left') if not decision_df.empty and 'Variant' in summary.columns else summary

    report_path = out_dir / 'package_b52200_miss_guard_v39_report.txt'
    lines=[]
    lines.append('TIPSET AI V39 – B52200 MISS AUDIT / GUARD')
    lines.append('='*96)
    lines.append(f'Seed: {final_seed}')
    lines.append(f'Varianter: {variants_locked}')
    lines.append('Utdelningsurval: samma stratifierade slump 100k–2.5M som V38.')
    lines.append('Mål: rädda minst 1 av B52200_TRUE-missarna, helst 25/30+, median <= 2600.')
    lines.append('')
    lines.append('SAMMANFATTNING')
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V39 beslut'] if c in summary_show.columns]
        lines.append(summary_show[cols].to_string(index=False))
    else:
        lines.append('(tom)')
    lines.append('')
    lines.append('B52200 MISS-RESCUE AUDIT')
    if isinstance(rescue, pd.DataFrame) and not rescue.empty:
        lines.append(rescue.to_string(index=False))
    else:
        lines.append('(inga B52200-missar hittade eller audit kunde inte byggas)')
    lines.append('')
    lines.append(f'Sample: {sample_path}')
    lines.append(f'Detail: {detail_path}')
    lines.append(f'Summary: {summary_path}')
    lines.append(f'Rescue audit: {rescue_path}')
    lines.append(f'Decision: {decision_path}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print('\nKLART – V39 B52200 MISS AUDIT / GUARD', flush=True)
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        show_cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V39 beslut'] if c in summary_show.columns]
        print(summary_show[show_cols].to_string(index=False), flush=True)
    if isinstance(rescue, pd.DataFrame) and not rescue.empty:
        print('\nB52200 MISS-RESCUE AUDIT', flush=True)
        print(rescue.to_string(index=False), flush=True)
    print('Rapport:', str(report_path), flush=True)
    print('Summary:', str(summary_path), flush=True)
    print('Detail:', str(detail_path), flush=True)
    print('Rescue audit:', str(rescue_path), flush=True)
    print('Sample:', str(sample_path), flush=True)
    print('Decision:', str(decision_path), flush=True)

    try:
        if '_orig_prepare_db_v44b' in locals() and _orig_prepare_db_v44b is not None:
            globals()['_prepare_db'] = _orig_prepare_db_v44b
    except Exception:
        pass

    del ns, db, detail
    gc.collect()




# =============================================================================
# V40s – B52200 SELECTIVE RESCUE SCREEN12
# =============================================================================
# V39 visade att generella guards försämrade B52200. V40s kör därför ett kort,
# riktat screen-test: sex kända B52200-missar från V38/V39 + sex kontroller.
# Bara selektiva fallback-varianter testas, och endast om de slår B52200 utan att
# bli REF_SUPER-breda är de värda ett senare 30-test.

V40_FOCUS_MISS_DATES = [
    '2025-07-26',
    '2025-08-30',
    '2025-10-04',
    '2025-12-13',
    '2025-12-26',
    '2026-03-07',
]

V40_VARIANT_DEFS = {
    'B52200_SELECT_REF2600': {
        'label': 'B52200_SELECT_REF2600 selektiv SUPER-fallback 2.2-2.7k',
        'description': 'SUPER/REF-bas med hårdare budgetval. Testar om vi kan rädda REF-räddningsbara B52200-missar utan att gå över cirka 2.6-2.7k.',
        'base': 'SUPER',
        'v31_cfg': {'target_rows': 2500, 'min_rows': 2200, 'max_rows': 2700, 'strict': True, 'v40_selective': 'ref2600', 'preferred_hi': 2600, 'max_filters_select': 12, 'min_val_hit': 34},
        'overrides': {'beam_width': 75, 'archive_width': 900, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 60.0, 'min_candidate_val_pct': 58.0, 'min_unique_rows': 10, 'reduction_power': 3.45, 'validation_power': 1.05, 'hit_power': 2.95, 'row_bucket_size': 260, 'per_bucket_keep': 7, 'v15_group_max_filters': 24, 'v15_max_group_candidates': 130, 'v15_cross_per_family': 4, 'v15_cross_max_filters': 16},
    },
    'B52200_SELECT_REF3100': {
        'label': 'B52200_SELECT_REF3100 audit SUPER-fallback 2.6-3.2k',
        'description': 'Auditvariant: mäter vad det kostar i rader att rädda de REF_SUPER-räddningsbara missarna, särskilt högre utdelningsprofil. Inte final om den blir för bred.',
        'base': 'SUPER',
        'v31_cfg': {'target_rows': 2850, 'min_rows': 2400, 'max_rows': 3200, 'strict': True, 'v40_selective': 'ref3100', 'preferred_hi': 3000, 'max_filters_select': 12, 'min_val_hit': 34},
        'overrides': {'beam_width': 75, 'archive_width': 900, 'min_hit': 23, 'candidate_min_hit': 25, 'min_package_val_pct': 60.0, 'min_candidate_val_pct': 58.0, 'min_unique_rows': 10, 'reduction_power': 2.95, 'validation_power': 1.10, 'hit_power': 3.05, 'row_bucket_size': 300, 'per_bucket_keep': 7, 'v15_group_max_filters': 24, 'v15_max_group_candidates': 130, 'v15_cross_per_family': 4, 'v15_cross_max_filters': 16},
    },
}


def _v40_register_variants():
    for key, cfg in V40_VARIANT_DEFS.items():
        base_key = str(cfg.get('base','SUPER')).upper()
        base = dict(V15_VARIANT_DEFS.get(base_key) or V15_VARIANT_DEFS.get('SUPER') or {})
        if not base:
            raise RuntimeError(f'Kan inte registrera V40 {key}: bas {base_key} saknas.')
        new = dict(base)
        new['label'] = cfg['label']
        new['description'] = cfg['description']
        new['v31_cfg'] = dict(cfg.get('v31_cfg') or {})
        ov = dict(new.get('overrides') or {})
        ov.update(cfg.get('overrides') or {})
        new['overrides'] = ov
        V15_VARIANT_DEFS[key] = new
        V19_VARIANT_DEFS[key] = new
        V13_VARIANT_DEFS[key] = {
            'label': cfg['label'],
            'fixed_payout': True,
            'bundle_search': bool(V13_VARIANT_DEFS.get(base_key, {}).get('bundle_search', False)),
            'description': cfg['description'],
        }


_v40_register_variants()

_ORIG_V40_SELECT_STATE = _v31_select_state


def _v31_select_state(valid_states, hist_payout, args, variant_id: str):  # noqa: F811
    cfg = _v31_cfg_for_variant(variant_id)
    if not cfg or not cfg.get('v40_selective'):
        return _ORIG_V40_SELECT_STATE(valid_states, hist_payout, args, variant_id)

    target = int(cfg.get('target_rows', 2500) or 2500)
    lo = int(cfg.get('min_rows', max(1, target - 300)) or max(1, target - 300))
    hi = int(cfg.get('max_rows', target + 300) or target + 300)
    preferred_hi = int(cfg.get('preferred_hi', hi) or hi)
    max_filters_select = int(cfg.get('max_filters_select', 999) or 999)
    min_val_hit = int(cfg.get('min_val_hit', 0) or 0)
    mode = str(cfg.get('v40_selective','')).lower()

    states = list(valid_states)
    if not states:
        return _ORIG_V40_SELECT_STATE(valid_states, hist_payout, args, variant_id)

    def rows(s): return int(s.frame_count)
    def nf(s): return int(len(s.chosen))
    def metrics(s): return _v13_state_metrics(s, hist_payout, args, variant_id)

    # V40 är selektivt: föredra state i budgetbandet, men använd inte generella
    # guard-krav som förstör andra omgångar. Om bandet saknas, fall tillbaka till
    # alla states så varianten alltid producerar ett jämförelsepaket.
    in_band = [s for s in states if lo <= rows(s) <= hi]
    under_hi = [s for s in states if rows(s) <= hi]
    pool = in_band or under_hi or states

    robust = [s for s in pool if nf(s) <= max_filters_select and (min_val_hit <= 0 or int(s.val_hit) >= min_val_hit)]
    if not robust and min_val_hit > 0:
        robust = [s for s in pool if int(s.val_hit) >= max(0, min_val_hit - 2)]
    if not robust:
        robust = pool

    def key(s):
        m = metrics(s)
        r = rows(s)
        too_wide = max(0, r - preferred_hi)
        dist = abs(r - target)
        in_band_flag = 1 if lo <= r <= hi else 0
        pref_flag = 1 if r <= preferred_hi else 0
        # REF3100 är audit: den får mer träff/valideringstyngd och accepterar
        # större radkostnad. REF2600 måste hålla mer budgetdisciplin.
        if mode == 'ref3100':
            return (
                int(s.hist_hit),
                int(s.val_hit),
                pref_flag,
                in_band_flag,
                -too_wide,
                -dist,
                float(m.get('payout_direction_pct', 0.0)),
                float(m.get('joint_score', 0.0)),
                -nf(s),
            )
        return (
            int(s.hist_hit),
            pref_flag,
            int(s.val_hit),
            in_band_flag,
            -too_wide,
            -dist,
            float(m.get('payout_direction_pct', 0.0)),
            float(m.get('joint_score', 0.0)),
            -nf(s),
        )

    best = max(robust, key=key)
    ok = 'JA' if lo <= int(best.frame_count) <= hi else 'NEJ'
    return best, {
        'status': f'V40_SELECTIVE_{mode.upper()}',
        'ok': ok,
        'target': target,
        'min': lo,
        'max': hi,
        'delta': int(int(best.frame_count) - target),
        'pool_in_band': int(len(in_band)),
        'pool_under_max': int(len(under_hi)),
    }


def _v40_focus_reorder_db_for_tests(ns: dict, global_db: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lägg sex kända B52200-missar först och fyll upp med stratifierade kontroller."""
    try:
        max_tests = int(getattr(args, 'max_tests', 12) or 12)
        seed = int(getattr(args, 'random_seed', 404040) or 404040)
        rng = np.random.default_rng(seed)
        work_db, _ = _prepare_supersvar_payout(global_db.copy(), int(args.pay_max))
        eligible = _prepare_db(ns, work_db, 13, int(args.pay_min), int(args.pay_max)).copy()
        if eligible.empty:
            return global_db, pd.DataFrame()
        eligible['__date'] = eligible.get('Datum', '').astype(str).str[:10]
        eligible['__payout'] = pd.to_numeric(eligible.get('Payout', 0), errors='coerce').fillna(0.0)

        selected = []
        rows_log = []
        # Fokusmissar i fast datumordning, om de finns i intervallet.
        for dt in V40_FOCUS_MISS_DATES:
            hits = list(eligible.index[eligible['__date'].eq(dt)])
            if hits:
                idx = hits[0]
                if idx not in selected:
                    selected.append(idx)
                    r = eligible.loc[idx]
                    rows_log.append({'Testordning': len(rows_log)+1, 'OriginalIndex': idx, 'SampleTyp': 'B52200_MISS_FOCUS', 'Datum': dt, 'Utdelning': float(r.get('__payout', 0))})

        remaining_slots = max(0, max_tests - len(selected))
        used = set(selected)
        if remaining_slots > 0:
            strata = [
                ('100k-250k', 100000, 250000),
                ('250k-500k', 250000, 500000),
                ('500k-1M', 500000, 1000000),
                ('1M-2.5M', 1000000, int(args.pay_max) + 1),
            ]
            base_q = remaining_slots // len(strata)
            rem = remaining_slots % len(strata)
            quotas = [base_q + (1 if i < rem else 0) for i in range(len(strata))]
            controls = []
            for (label, lo, hi), q in zip(strata, quotas):
                part = eligible[(eligible['__payout'] >= lo) & (eligible['__payout'] < hi)]
                avail = [idx for idx in part.index if idx not in used and str(eligible.loc[idx,'__date']) not in set(V40_FOCUS_MISS_DATES)]
                take = min(int(q), len(avail))
                if take > 0:
                    pick = list(rng.choice(avail, size=take, replace=False))
                    for idx in pick:
                        controls.append((idx, label))
                        used.add(idx)
            if len(controls) < remaining_slots:
                avail = [idx for idx in eligible.index if idx not in used and str(eligible.loc[idx,'__date']) not in set(V40_FOCUS_MISS_DATES)]
                take = min(remaining_slots - len(controls), len(avail))
                if take > 0:
                    for idx in list(rng.choice(avail, size=take, replace=False)):
                        controls.append((idx, 'FYLLNAD_RANDOM'))
                        used.add(idx)
            if controls:
                # Slumpa bara kontrollernas ordning; fokusmissarna ligger först för lättare loggläsning.
                order = list(rng.permutation(len(controls))) if len(controls) > 1 else [0]
                for j in order:
                    idx, label = controls[int(j)]
                    selected.append(idx)
                    r = eligible.loc[idx]
                    rows_log.append({'Testordning': len(rows_log)+1, 'OriginalIndex': idx, 'SampleTyp': f'KONTROLL_{label}', 'Datum': str(r.get('__date','')), 'Utdelning': float(r.get('__payout', 0))})

        selected = selected[:max_tests]
        first = global_db.loc[selected]
        rest = global_db.drop(index=selected, errors='ignore')
        reordered = pd.concat([first, rest], axis=0)
        log_df = pd.DataFrame(rows_log[:len(selected)])
        return reordered, log_df
    except Exception as e:
        print(f'V40 fokusurval misslyckades, faller tillbaka till V36-stratifiering: {e}', flush=True)
        return _v36_random_reorder_db_for_tests(ns, global_db, args)


def _v40_rescue_table(detail: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return pd.DataFrame()
    need_cols = {'Variant','Datum','Paket klarar facit'}
    if not need_cols.issubset(set(detail.columns)):
        return pd.DataFrame()
    ok = detail[detail.get('Status','OK').astype(str).eq('OK')].copy() if 'Status' in detail.columns else detail.copy()
    ok['HitBool'] = ok['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1'])
    piv = ok.pivot_table(index='Datum', columns='Variant', values='HitBool', aggfunc='max')
    rows=[]
    for dt in V40_FOCUS_MISS_DATES:
        if dt not in piv.index:
            continue
        row=piv.loc[dt]
        rec={'Datum': dt}
        for v in ['B52200_TRUE','REF_SUPER','B52200_SELECT_REF2600','B52200_SELECT_REF3100']:
            if v in piv.columns:
                rec[v] = 'RÄDDAR' if bool(row.get(v, False)) else 'MISS'
        sub=ok[ok['Datum'].astype(str).eq(str(dt))]
        if 'Utdelning' in sub.columns and not sub.empty:
            ser=pd.to_numeric(sub['Utdelning'], errors='coerce').dropna()
            rec['Utdelning']=int(ser.iloc[0]) if not ser.empty else None
        for v in ['B52200_TRUE','REF_SUPER','B52200_SELECT_REF2600','B52200_SELECT_REF3100']:
            vv=sub[sub['Variant'].astype(str).eq(v)]
            if not vv.empty and 'Paketrader' in vv.columns:
                ser=pd.to_numeric(vv['Paketrader'], errors='coerce').dropna()
                rec[f'{v} rader']=int(ser.iloc[0]) if not ser.empty else None
        rows.append(rec)
    return pd.DataFrame(rows)


def main_v40s():
    parser=argparse.ArgumentParser(description='Tipset AI V40s – B52200 Selective Rescue Screen12 utan kommandoradsargument.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)

    seed = 404040
    variants_locked = 'B52200_TRUE,B52200_SELECT_REF2600,B52200_SELECT_REF3100,REF_SUPER'
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*96, flush=True)
    print('TIPSET AI V40s – B52200 SELECTIVE RESCUE SCREEN12', flush=True)
    print('LÅST LOGIK: kort screen, inte 30-test. Sex B52200-missar + sex kontrollomgångar.', flush=True)
    print('Syfte: testa om selektiv SUPER-fallback kan rädda B52200 utan generellt guard-tapp.', flush=True)
    print('Testomgångar: 12 · fokusmissar + stratifierade kontroller', flush=True)
    print('Ramar: 3-5-5 (7776 grundrader)', flush=True)
    print(f'Seed för kontroller: {seed}', flush=True)
    print(f'Varianter: {variants_locked}', flush=True)
    print('Gå vidare-regel: rescue måste slå B52200_TRUE i träff och hålla median helst <= 2600.', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)

    frame='3-5-5'
    run_args=argparse.Namespace(
        app_file=str(app_file), db_file=str(db_file), out_dir=str(out_dir),
        output_prefix=f'package_b52200_selective_rescue_v40s_{_v17_safe_prefix_part(frame)}',
        variants=variants_locked,
        max_tests=12, test_offset=0, random_seed=int(seed), sample_mode='focus_misses_plus_controls',
        top_n=30, wide_n=35, pay_min=100000, pay_max=2500000,
        filter_hist_target_pct=95, frame_profile=frame, frames=frame, mode='leave-one-out',
        candidate_min_hit=25, min_candidate_val_pct=58.0, min_structure_val_pct=64.0,
        min_gap_score=0.55, variants_per_key=4, max_candidates=220,
        fast_no_supermakro=True, min_hit=23, min_package_val_pct=60.0,
        min_structure_package_val_pct=76.0, min_unique_rows=10,
        beam_width=55, archive_width=650, structure_seed_count=2,
        hit_power=2.55, validation_power=0.88, reduction_power=3.9,
        payout_weight=0.02, cluster_weight=0.05, payout_direction_weight=0.03,
        bundle_pool_size=8, triple_pool_size=0, max_bundle_trials_per_state=18,
        max_bundle_keep_per_state=6, row_bucket_size=230, per_bucket_keep=6,
        enable_triples=False, v15_group_max_filters=20, v15_max_group_candidates=100,
        v15_min_group_size=3, v15_cross_per_family=3, v15_cross_max_filters=14,
    )

    print('\n'+'-'*96, flush=True)
    print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
    print('-'*96, flush=True)

    ns=_load_app_functions(app_file, fast_no_supermakro=False)
    db=ns['load_database'](str(db_file),13)
    db, sample_df = _v40_focus_reorder_db_for_tests(ns, db, run_args)
    sample_path = out_dir / 'package_b52200_selective_rescue_v40s_sample.csv'
    if isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
        sample_df.to_csv(sample_path, index=False)
        print('V40s testurval:', flush=True)
        print(sample_df.to_string(index=False), flush=True)
        print('Sample:', str(sample_path), flush=True)

    detail, meta = _run_backtest_v15(ns, db, run_args)
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        raise SystemExit('Inga V40s-resultat skapades.')
    detail = detail.copy()
    detail.insert(0, 'Ramprofil', frame)
    detail_path = out_dir / 'package_b52200_selective_rescue_v40s_detail.csv'
    detail.to_csv(detail_path, index=False)
    _v31_write_outputs(detail.drop(columns=['Ramprofil'], errors='ignore'), run_args, out_dir, app_file, db_file)

    summary = _summarize_v15(detail.drop(columns=['Ramprofil'], errors='ignore'))
    summary_path = out_dir / 'package_b52200_selective_rescue_v40s_summary.csv'
    summary.to_csv(summary_path, index=False)
    rescue = _v40_rescue_table(detail.drop(columns=['Ramprofil'], errors='ignore'))
    rescue_path = out_dir / 'package_b52200_selective_rescue_v40s_rescue_audit.csv'
    rescue.to_csv(rescue_path, index=False)

    decision=[]
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        b_hits = None
        b_med = None
        b = summary[summary['Variant'].astype(str).eq('B52200_TRUE')] if 'Variant' in summary.columns else pd.DataFrame()
        if not b.empty:
            b_hits=float(b.iloc[0].get('Träffar',0) or 0)
            b_med=float(b.iloc[0].get('Median paketrader',0) or 0)
        for _, row in summary.iterrows():
            variant=str(row.get('Variant',''))
            hits=float(row.get('Träffar',0) or 0)
            med=float(row.get('Median paketrader',0) or 0)
            if variant.startswith('B52200_SELECT'):
                if b_hits is not None and hits > b_hits and med <= 2600:
                    d='GÅ VIDARE – slår B52200 och håller <=2600'
                elif b_hits is not None and hits > b_hits and med <= 3000:
                    d='AUDIT – slår träff men riskerar för bred nivå'
                elif b_hits is not None and hits == b_hits and med <= max(2600, (b_med or 0)+250):
                    d='NÄRA – samma träff, kontrolleras bara om rescue audit är bättre'
                else:
                    d='KASTA – slår inte B52200 praktiskt'
            elif variant == 'B52200_TRUE':
                d='KONTROLL – huvudkandidat'
            else:
                d='REFERENS – ej 2k-krav'
            decision.append({'Variant': variant, 'V40s beslut': d})
    decision_df=pd.DataFrame(decision)
    decision_path = out_dir / 'package_b52200_selective_rescue_v40s_decision.csv'
    decision_df.to_csv(decision_path, index=False)
    summary_show = summary.merge(decision_df, on='Variant', how='left') if not decision_df.empty and 'Variant' in summary.columns else summary

    report_path = out_dir / 'package_b52200_selective_rescue_v40s_report.txt'
    lines=[]
    lines.append('TIPSET AI V40s – B52200 SELECTIVE RESCUE SCREEN12')
    lines.append('='*96)
    lines.append('Upplägg: sex kända B52200-missar + sex stratifierade kontrollomgångar.')
    lines.append(f'Fokusmissar: {", ".join(V40_FOCUS_MISS_DATES)}')
    lines.append(f'Varianter: {variants_locked}')
    lines.append('Mål: rescue-variant ska slå B52200_TRUE och helst hålla median <= 2600.')
    lines.append('')
    lines.append('SAMMANFATTNING')
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V40s beslut'] if c in summary_show.columns]
        lines.append(summary_show[cols].to_string(index=False))
    else:
        lines.append('(tom)')
    lines.append('')
    lines.append('FOKUSMISS-RESCUE AUDIT')
    if isinstance(rescue, pd.DataFrame) and not rescue.empty:
        lines.append(rescue.to_string(index=False))
    else:
        lines.append('(tom)')
    lines.append('')
    lines.append(f'Sample: {sample_path}')
    lines.append(f'Detail: {detail_path}')
    lines.append(f'Summary: {summary_path}')
    lines.append(f'Rescue audit: {rescue_path}')
    lines.append(f'Decision: {decision_path}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print('\nKLART – V40s B52200 SELECTIVE RESCUE SCREEN12', flush=True)
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        show_cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V40s beslut'] if c in summary_show.columns]
        print(summary_show[show_cols].to_string(index=False), flush=True)
    if isinstance(rescue, pd.DataFrame) and not rescue.empty:
        print('\nFOKUSMISS-RESCUE AUDIT', flush=True)
        print(rescue.to_string(index=False), flush=True)
    print('Rapport:', str(report_path), flush=True)
    print('Summary:', str(summary_path), flush=True)
    print('Detail:', str(detail_path), flush=True)
    print('Rescue audit:', str(rescue_path), flush=True)
    print('Sample:', str(sample_path), flush=True)
    print('Decision:', str(decision_path), flush=True)

    try:
        if '_orig_prepare_db_v44b' in locals() and _orig_prepare_db_v44b is not None:
            globals()['_prepare_db'] = _orig_prepare_db_v44b
    except Exception:
        pass

    del ns, db, detail
    gc.collect()




# =============================================================================
# V41s – B52200 INTERVAL OPTIMIZER SCREEN12
# =============================================================================
# Bygger vidare på B52200, men ändrar kandidatpoolen innan paketbyggaren körs:
#   1) gruppera alla kandidatintervall per filter-nyckel,
#   2) välj de intervall som har bäst träff/validering/reducering-balans,
#   3) stoppa små riskfilter som reducerar marginellt men kan döda facit,
#   4) låt sedan samma paketbyggare välja filterpaket mot 2.0k-2.6k.
# Det är ett screen-test på 12 omgångar: sex kända B52200-missar + sex kontroller.

V41_VARIANT_DEFS = {
    'B52200_INTERVAL_OPT': {
        'label': 'B52200_INTERVAL_OPT intervalloptimerad B52200 2.0-2.6k',
        'description': 'B52200-bas där varje filter-nyckel först får sina intervall kalibrerade mot träff, validering och reducering innan paketbyggaren väljer slutpaket.',
        'base': 'B52200_TRUE',
        'v31_cfg': {
            'target_rows': 2300,
            'min_rows': 2000,
            'max_rows': 2600,
            'strict': True,
            'v41_interval_opt': True,
            'min_select_hist': 23,
        },
        'overrides': {
            # Något bredare kandidatpool än B52200_TRUE, men fortfarande screen-snabbt.
            'beam_width': 65,
            'archive_width': 850,
            'min_hit': 23,
            'candidate_min_hit': 24,
            'min_package_val_pct': 60.0,
            'min_candidate_val_pct': 56.0,
            'min_structure_val_pct': 66.0,
            'min_gap_score': 0.50,
            'variants_per_key': 7,
            'max_candidates': 280,
            'min_unique_rows': 9,
            'hit_power': 2.70,
            'validation_power': 0.96,
            'reduction_power': 3.70,
            'row_bucket_size': 225,
            'per_bucket_keep': 7,
            'v15_group_max_filters': 20,
            'v15_max_group_candidates': 90,
            'v15_cross_per_family': 3,
            'v15_cross_max_filters': 14,
        },
    },
}


def _v41_register_variants():
    for key, cfg in V41_VARIANT_DEFS.items():
        base_key = str(cfg.get('base','B52200_TRUE')).upper()
        base = dict(V15_VARIANT_DEFS.get(base_key) or V15_VARIANT_DEFS.get('B5') or {})
        if not base:
            raise RuntimeError(f'Kan inte registrera V41 {key}: bas {base_key} saknas.')
        new = dict(base)
        new['label'] = cfg['label']
        new['description'] = cfg['description']
        new['v31_cfg'] = dict(cfg.get('v31_cfg') or {})
        ov = dict(new.get('overrides') or {})
        ov.update(cfg.get('overrides') or {})
        new['overrides'] = ov
        V15_VARIANT_DEFS[key] = new
        V19_VARIANT_DEFS[key] = new
        V13_VARIANT_DEFS[key] = {
            'label': cfg['label'],
            'fixed_payout': True,
            'bundle_search': bool(V13_VARIANT_DEFS.get(base_key, {}).get('bundle_search', False)),
            'description': cfg['description'],
        }


_v41_register_variants()

_ORIG_V41_TRANSFORM_CANDIDATES = _v15_transform_candidates
_ORIG_V41_SELECT_STATE = _v31_select_state


def _v41_cfg_for_variant(variant_id: str) -> dict:
    return (V15_VARIANT_DEFS.get(str(variant_id).upper(), {}) or {}).get('v31_cfg') or {}


def _v41_interval_family(c: dict) -> str:
    try:
        return _v15_group_family(c)
    except Exception:
        return str(c.get('category','') or 'Övrigt')


def _v41_is_fragile_candidate(c: dict) -> bool:
    txt = f"{c.get('name','')} {c.get('key','')} {c.get('category','')}".lower()
    fam = _v41_interval_family(c)
    if fam == 'Struktur':
        return True
    fragile_words = [
        'uppkomst', 'längsta', 'följd', 'svit', 'sekvens', 'dubblett', 'tripp',
        'delta', 'avvikelse', 'max 0', 'favorit max', 'fat t'
    ]
    return any(w in txt for w in fragile_words)


def _v41_interval_score(c: dict, *, htot: int, vtot: int, ftot: int, key_best_red: float, key_best_val: float) -> float:
    hist_hit = int(c.get('hist_hit', 0) or 0)
    val_pct = float(c.get('val_pct', 100.0 if vtot == 0 else 0.0) or 0.0)
    red = float(c.get('red_pct', 0.0) or 0.0)
    gap = float(c.get('gap_score', 0.0) or 0.0)
    frame_keep = int(c.get('frame_keep', ftot) or ftot)
    payout_dir = float(c.get('payout_direction_pct', c.get('payout_lift_pct', 0.0)) or 0.0)
    fam = _v41_interval_family(c)

    hist_loss = max(0, int(htot) - hist_hit)
    val_loss = max(0.0, float(key_best_val) - val_pct)
    red_rel = red / max(1.0, float(key_best_red))

    # Grundidé: hög träff och validering först, sedan reducering. Straffa
    # intervall som offrar historik/validering för små radvinster.
    score = 0.0
    score += 120.0 * hist_hit / max(1, int(htot))
    score += 0.65 * val_pct
    score += 0.55 * red
    score += 1.75 * min(6.0, max(0.0, gap))
    score += 0.04 * payout_dir
    score += 5.0 * min(1.0, max(0.0, red_rel))

    if hist_loss:
        score -= 22.0 * hist_loss
    if val_loss > 3.0:
        score -= 1.25 * (val_loss - 3.0)

    # Filter som historiskt varit kantfarliga måste betala för sig.
    if _v41_is_fragile_candidate(c):
        if hist_hit < int(htot):
            score -= 35.0
        if val_pct < 72.0:
            score -= 16.0
        if red < 4.0:
            score -= 12.0

    # Hindra att väldigt smala intervall dominerar bara för att de råkar reducera.
    if frame_keep < max(100, int(ftot * 0.14)) and hist_hit < int(htot):
        score -= 20.0
    if red < 2.0 and hist_hit < int(htot):
        score -= 18.0

    # Lite familjebalans: B52200 har fått bäst signal från FAT/ABC/poäng, men
    # värde/struktur/sekvens får inte vinna på kosmetisk reducering.
    if fam in {'FAT', 'ABC', 'Poäng/rank'}:
        score += 4.0
    elif fam in {'Värde/svårighet', 'FAT-sekvens'}:
        score -= 3.0
    elif fam == 'Struktur':
        score -= 5.0

    return float(score)


def _v41_optimize_interval_candidates(candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, args, variant_id: str) -> List[dict]:
    """Välj 1-3 intervall per filter-nyckel med bäst säker reduceringsprofil."""
    by_key: Dict[str, List[dict]] = {}
    passthrough = []
    for c in candidates:
        if c.get('is_v15_group'):
            passthrough.append(c)
            continue
        k = str(c.get('key',''))
        if not k:
            continue
        by_key.setdefault(k, []).append(c)

    selected = []
    audit_rows = []
    for key, vals in by_key.items():
        if not vals:
            continue
        key_best_red = max(float(v.get('red_pct', 0.0) or 0.0) for v in vals)
        key_best_val = max(float(v.get('val_pct', 0.0) or 0.0) for v in vals)
        max_hist = max(int(v.get('hist_hit', 0) or 0) for v in vals)
        fam = _v41_interval_family(vals[0])
        fragile = any(_v41_is_fragile_candidate(v) for v in vals)

        # Säkerhetsgolv: struktur/kantfilter måste normalt vara 30/30 eller nästan
        # perfekta med stark validering. Profilfilter får offra max 1 historikträff
        # om reduceringen är tydlig.
        floor_hist = max(int(getattr(args, 'candidate_min_hit', 24) or 24), max_hist - 1)
        if fragile:
            floor_hist = max(floor_hist, int(htot))
        floor_val = float(getattr(args, 'min_candidate_val_pct', 56.0) or 56.0)
        if fragile:
            floor_val = max(floor_val, 70.0)
        elif fam in {'Värde/svårighet', 'FAT-sekvens'}:
            floor_val = max(floor_val, 62.0)

        pool = [v for v in vals if int(v.get('hist_hit', 0) or 0) >= floor_hist and float(v.get('val_pct', 0.0) or 0.0) + 1e-9 >= floor_val]
        if not pool:
            # Fallback inom nyckeln: ta bästa max_hist-alternativet, men bara om
            # det uppfyller grundgolven; annars släpps nyckeln.
            pool = [v for v in vals if int(v.get('hist_hit', 0) or 0) >= max(int(getattr(args, 'candidate_min_hit', 24) or 24), max_hist)]
        if not pool:
            continue

        scored = sorted(pool, key=lambda v: _v41_interval_score(v, htot=htot, vtot=vtot, ftot=ftot, key_best_red=key_best_red, key_best_val=key_best_val), reverse=True)
        best = scored[0]

        # Behåll även en mer reducerande variant om den nästan inte tappar säkerhet.
        keep = [best]
        for cand in scored[1:]:
            if len(keep) >= 3:
                break
            sameish_hist = int(cand.get('hist_hit',0)) >= int(best.get('hist_hit',0)) - 1
            sameish_val = float(cand.get('val_pct',0.0)) >= float(best.get('val_pct',0.0)) - 4.0
            better_red = float(cand.get('red_pct',0.0)) >= float(best.get('red_pct',0.0)) + 2.5
            if sameish_hist and sameish_val and better_red and not (fragile and int(cand.get('hist_hit',0)) < int(htot)):
                keep.append(cand)

        # Och en extra supersäker variant om bästa intervallet offrar historik.
        if int(best.get('hist_hit',0)) < int(htot):
            safe = [v for v in scored if int(v.get('hist_hit',0)) >= int(htot)]
            if safe and all((s.get('key'), s.get('interval_txt')) != (keep[0].get('key'), keep[0].get('interval_txt')) for s in safe[:1]):
                keep.append(safe[0])

        seen = set()
        for cand in keep:
            sig = (cand.get('key'), cand.get('interval_txt'))
            if sig in seen:
                continue
            seen.add(sig)
            c2 = dict(cand)
            c2['v41_interval_score'] = _v41_interval_score(cand, htot=htot, vtot=vtot, ftot=ftot, key_best_red=key_best_red, key_best_val=key_best_val)
            c2['v41_family'] = fam
            c2['v41_fragile'] = bool(fragile)
            selected.append(c2)
        audit_rows.append({
            'key': key,
            'family': fam,
            'fragile': bool(fragile),
            'tested_intervals': len(vals),
            'kept_intervals': len(keep),
            'best_hist': int(best.get('hist_hit',0)),
            'best_val_pct': round(float(best.get('val_pct',0.0)), 2),
            'best_red_pct': round(float(best.get('red_pct',0.0)), 2),
            'best_interval': str(best.get('interval_txt','')),
        })

    # Dedupe och begränsning. Prioritera säkerhetsoptimerade intervall men behåll
    # kandidatbredd nog för beam search.
    out, seen = [], set()
    for c in sorted(selected + passthrough, key=lambda x: (
        float(x.get('v41_interval_score', _v13_state_sort_proxy_candidate(x)[0])),
        int(x.get('hist_hit', 0)),
        float(x.get('val_pct', 0.0)),
        float(x.get('red_pct', 0.0)),
    ), reverse=True):
        sig = (c.get('key'), c.get('interval_txt'))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    max_keep = max(80, min(int(getattr(args, 'max_candidates', 280) or 280), 220))
    return out[:max_keep]


def _v15_transform_candidates(candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, args, variant_id: str) -> Tuple[List[dict], List[dict]]:  # noqa: F811
    final, group_cands = _ORIG_V41_TRANSFORM_CANDIDATES(candidates, htot, vtot, ftot, hist_payout, args, variant_id)
    cfg = _v41_cfg_for_variant(variant_id)
    if not cfg.get('v41_interval_opt'):
        return final, group_cands
    optimized = _v41_optimize_interval_candidates(final, htot, vtot, ftot, hist_payout, args, variant_id)
    print(f'    V41 {variant_id}: intervalloptimering {len(final)} -> {len(optimized)} kandidater', flush=True)
    return optimized, group_cands


def _v31_select_state(valid_states, hist_payout, args, variant_id: str):  # noqa: F811
    cfg = _v41_cfg_for_variant(variant_id)
    if not cfg.get('v41_interval_opt'):
        return _ORIG_V41_SELECT_STATE(valid_states, hist_payout, args, variant_id)

    target = int(cfg.get('target_rows', 2300) or 2300)
    lo = int(cfg.get('min_rows', 2000) or 2000)
    hi = int(cfg.get('max_rows', 2600) or 2600)
    min_select_hist = int(cfg.get('min_select_hist', int(getattr(args, 'min_hit', 23))) or int(getattr(args, 'min_hit', 23)))
    states = [s for s in list(valid_states) if int(s.hist_hit) >= min_select_hist]
    if not states:
        states = list(valid_states)
    in_band = [s for s in states if lo <= int(s.frame_count) <= hi]
    under_hi = [s for s in states if int(s.frame_count) <= hi]
    pool = in_band or under_hi or states

    def interval_quality(s):
        vals = []
        fragile_count = 0
        for c in s.chosen:
            vals.append(float(c.get('v41_interval_score', 0.0) or 0.0))
            if bool(c.get('v41_fragile', False)):
                fragile_count += 1
        return (float(np.mean(vals)) if vals else 0.0, -int(fragile_count))

    def key(s):
        m = _v13_state_metrics(s, hist_payout, args, variant_id)
        r = int(s.frame_count)
        iq, frag_pen = interval_quality(s)
        return (
            int(s.hist_hit),
            int(s.val_hit),
            1 if lo <= r <= hi else 0,
            -abs(r - target),
            -max(0, r - hi),
            float(m.get('payout_direction_pct', 0.0)),
            float(m.get('joint_score', 0.0)),
            iq,
            frag_pen,
            -int(len(s.chosen)),
        )

    best = max(pool, key=key)
    ok = 'JA' if lo <= int(best.frame_count) <= hi else 'NEJ'
    return best, {
        'status': 'V41_INTERVAL_OPT',
        'ok': ok,
        'target': target,
        'min': lo,
        'max': hi,
        'delta': int(int(best.frame_count) - target),
        'pool_in_band': int(len(in_band)),
        'pool_under_max': int(len(under_hi)),
    }


def _v41_rescue_table(detail: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return pd.DataFrame()
    need_cols = {'Variant','Datum','Paket klarar facit'}
    if not need_cols.issubset(set(detail.columns)):
        return pd.DataFrame()
    ok = detail[detail.get('Status','OK').astype(str).eq('OK')].copy() if 'Status' in detail.columns else detail.copy()
    ok['HitBool'] = ok['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1'])
    piv = ok.pivot_table(index='Datum', columns='Variant', values='HitBool', aggfunc='max')
    rows=[]
    for dt in V40_FOCUS_MISS_DATES:
        if dt not in piv.index:
            continue
        row=piv.loc[dt]
        rec={'Datum': dt}
        for v in ['B52200_TRUE','B52200_INTERVAL_OPT','REF_SUPER']:
            if v in piv.columns:
                rec[v] = 'RÄDDAR' if bool(row.get(v, False)) else 'MISS'
        sub=ok[ok['Datum'].astype(str).eq(str(dt))]
        if 'Utdelning' in sub.columns and not sub.empty:
            ser=pd.to_numeric(sub['Utdelning'], errors='coerce').dropna()
            rec['Utdelning']=int(ser.iloc[0]) if not ser.empty else None
        for v in ['B52200_TRUE','B52200_INTERVAL_OPT','REF_SUPER']:
            vv=sub[sub['Variant'].astype(str).eq(v)]
            if not vv.empty and 'Paketrader' in vv.columns:
                ser=pd.to_numeric(vv['Paketrader'], errors='coerce').dropna()
                rec[f'{v} rader']=int(ser.iloc[0]) if not ser.empty else None
        rows.append(rec)
    return pd.DataFrame(rows)



def main_v42():
    parser=argparse.ArgumentParser(description='Tipset AI V42 – B52200 Interval Optimizer Final30 utan kommandoradsargument.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)

    final_seed = 20260720
    variants_locked = 'B52200_INTERVAL_OPT,B52200_TRUE,REF_SUPER'
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*96, flush=True)
    print('TIPSET AI V42 – B52200 INTERVAL OPTIMIZER FINAL30', flush=True)
    print('LÅST LOGIK: 30-test med samma stratifierade slump-seed som V38. Inga %run-argument behövs.', flush=True)
    print('Syfte: avgöra om intervalloptimerad B52200 är bättre app-kandidat än B52200_TRUE.', flush=True)
    print('Testomgångar: 30 totalt · random/stratifierat inom utdelningsintervallet 100k–2.5M', flush=True)
    print('Ramar: 3-5-5 (7776 grundrader)', flush=True)
    print(f'Seed: {final_seed}', flush=True)
    print(f'Varianter: {variants_locked}', flush=True)
    print('Finalsignal: INTERVAL_OPT bör nå minst 25/30, slå B52200_TRUE och helst hålla median <= 2600.', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)

    frame='3-5-5'
    run_args=argparse.Namespace(
        app_file=str(app_file), db_file=str(db_file), out_dir=str(out_dir),
        output_prefix=f'package_b52200_interval_optimizer_v42_final30_{_v17_safe_prefix_part(frame)}',
        variants=variants_locked,
        max_tests=30, test_offset=0, random_seed=int(final_seed), sample_mode='stratified_payout',
        top_n=30, wide_n=35, pay_min=100000, pay_max=2500000,
        filter_hist_target_pct=95, frame_profile=frame, frames=frame, mode='leave-one-out',
        candidate_min_hit=24, min_candidate_val_pct=56.0, min_structure_val_pct=66.0,
        min_gap_score=0.50, variants_per_key=7, max_candidates=280,
        fast_no_supermakro=True, min_hit=23, min_package_val_pct=60.0,
        min_structure_package_val_pct=76.0, min_unique_rows=9,
        beam_width=65, archive_width=850, structure_seed_count=2,
        hit_power=2.70, validation_power=0.96, reduction_power=3.70,
        payout_weight=0.02, cluster_weight=0.05, payout_direction_weight=0.03,
        bundle_pool_size=8, triple_pool_size=0, max_bundle_trials_per_state=18,
        max_bundle_keep_per_state=6, row_bucket_size=225, per_bucket_keep=7,
        enable_triples=False, v15_group_max_filters=20, v15_max_group_candidates=90,
        v15_min_group_size=3, v15_cross_per_family=3, v15_cross_max_filters=14,
    )

    print('\n'+'-'*96, flush=True)
    print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
    print('-'*96, flush=True)

    ns=_load_app_functions(app_file, fast_no_supermakro=False)
    db=ns['load_database'](str(db_file),13)
    db, sample_df = _v36_random_reorder_db_for_tests(ns, db, run_args)
    sample_path = out_dir / 'package_b52200_interval_optimizer_v42_final30_sample.csv'
    if isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
        sample_df.to_csv(sample_path, index=False)
        print('Slumpade/stratifierade testomgångar:', flush=True)
        print(sample_df.to_string(index=False), flush=True)
        print('Sample:', str(sample_path), flush=True)

    detail, meta = _run_backtest_v15(ns, db, run_args)
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        raise SystemExit('Inga V42-resultat skapades.')
    detail = detail.copy()
    detail.insert(0, 'Ramprofil', frame)
    detail_path = out_dir / 'package_b52200_interval_optimizer_v42_final30_detail.csv'
    detail.to_csv(detail_path, index=False)
    _v31_write_outputs(detail.drop(columns=['Ramprofil'], errors='ignore'), run_args, out_dir, app_file, db_file)

    summary = _summarize_v15(detail.drop(columns=['Ramprofil'], errors='ignore'))
    summary_path = out_dir / 'package_b52200_interval_optimizer_v42_final30_summary.csv'
    summary.to_csv(summary_path, index=False)

    decision=[]
    b_hits = None
    b_med = None
    if isinstance(summary, pd.DataFrame) and not summary.empty and 'Variant' in summary.columns:
        b = summary[summary['Variant'].astype(str).eq('B52200_TRUE')]
        if not b.empty:
            b_hits=float(b.iloc[0].get('Träffar',0) or 0)
            b_med=float(b.iloc[0].get('Median paketrader',0) or 0)
        for _, row in summary.iterrows():
            variant=str(row.get('Variant',''))
            hits=float(row.get('Träffar',0) or 0)
            med=float(row.get('Median paketrader',0) or 0)
            mean=float(row.get('Medel paketrader',0) or 0)
            if variant == 'B52200_INTERVAL_OPT':
                if hits >= 25 and (b_hits is None or hits > b_hits) and med <= 2600:
                    d='GODKÄND FINALIST – intervallopt slår B52200 och håller radmål'
                elif hits >= 25 and med <= 2800:
                    d='TRÄFF GODKÄND – radnivå/medel måste vägas'
                elif hits >= 24 and med <= 2600:
                    d='NÄRA – bra radnivå men når inte finalträff'
                else:
                    d='INTE FINAL – slår inte B52200 praktiskt'
            elif variant == 'B52200_TRUE':
                d='KONTROLL – tidigare huvudkandidat'
            else:
                d='REFERENS – ej 2k-krav'
            decision.append({'Variant': variant, 'V42 beslut': d})
    decision_df=pd.DataFrame(decision)
    decision_path = out_dir / 'package_b52200_interval_optimizer_v42_final30_decision.csv'
    decision_df.to_csv(decision_path, index=False)
    summary_show = summary.merge(decision_df, on='Variant', how='left') if not decision_df.empty and 'Variant' in summary.columns else summary

    # Miss/rescue audit: visa om INTERVAL_OPT räddar B52200_TRUE-missar och om den tappar egna omgångar.
    audit_rows=[]
    if isinstance(detail, pd.DataFrame) and not detail.empty and {'Datum','Variant','Paket klarar facit'}.issubset(detail.columns):
        dd = detail.copy()
        if 'Status' in dd.columns:
            dd = dd[dd['Status'].astype(str).eq('OK')].copy()
        dd['HitBool'] = dd['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1'])
        piv = dd.pivot_table(index='Datum', columns='Variant', values='HitBool', aggfunc='max')
        for dt, row in piv.iterrows():
            b_ok = bool(row.get('B52200_TRUE', False)) if 'B52200_TRUE' in piv.columns else False
            opt_ok = bool(row.get('B52200_INTERVAL_OPT', False)) if 'B52200_INTERVAL_OPT' in piv.columns else False
            ref_ok = bool(row.get('REF_SUPER', False)) if 'REF_SUPER' in piv.columns else False
            kind = None
            if (not b_ok) and opt_ok:
                kind = 'OPT_RÄDDAR_B52200_MISS'
            elif b_ok and (not opt_ok):
                kind = 'OPT_TAPPAR_B52200_TRÄFF'
            elif (not opt_ok) and ref_ok:
                kind = 'REF_RÄDDAR_OPT_MISS'
            if not kind:
                continue
            sub=dd[dd['Datum'].astype(str).eq(str(dt))]
            rec={'Datum': dt, 'Audit': kind, 'B52200_TRUE': 'RÄDDAR' if b_ok else 'MISS', 'B52200_INTERVAL_OPT': 'RÄDDAR' if opt_ok else 'MISS', 'REF_SUPER': 'RÄDDAR' if ref_ok else 'MISS'}
            if 'Utdelning' in sub.columns:
                ser=pd.to_numeric(sub['Utdelning'], errors='coerce').dropna()
                rec['Utdelning']=int(ser.iloc[0]) if not ser.empty else None
            for v in ['B52200_TRUE','B52200_INTERVAL_OPT','REF_SUPER']:
                vv=sub[sub['Variant'].astype(str).eq(v)]
                if not vv.empty and 'Paketrader' in vv.columns:
                    ser=pd.to_numeric(vv['Paketrader'], errors='coerce').dropna()
                    rec[f'{v} rader']=int(ser.iloc[0]) if not ser.empty else None
            audit_rows.append(rec)
    audit_df=pd.DataFrame(audit_rows)
    audit_path = out_dir / 'package_b52200_interval_optimizer_v42_final30_rescue_audit.csv'
    audit_df.to_csv(audit_path, index=False)

    report_path = out_dir / 'package_b52200_interval_optimizer_v42_final30_report.txt'
    lines=[]
    lines.append('TIPSET AI V42 – B52200 INTERVAL OPTIMIZER FINAL30')
    lines.append('='*96)
    lines.append(f'Seed: {final_seed}')
    lines.append(f'Varianter: {variants_locked}')
    lines.append('Utdelningsurval: stratifierad slump 100k–2.5M')
    lines.append('Mål: INTERVAL_OPT minst 25/30, bättre än B52200_TRUE, median helst <= 2600.')
    lines.append('Pipeline: 3-5-5 grundram 7776 → paketmotor ca 2k–2.6k → 12-rättsgaranti ca 400 spelrader.')
    lines.append('Metod: V41-intervalloptimering per filter-nyckel innan paketbyggaren väljer slutpaket.')
    lines.append('')
    lines.append('SAMMANFATTNING')
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V42 beslut'] if c in summary_show.columns]
        lines.append(summary_show[cols].to_string(index=False))
    else:
        lines.append('(tom)')
    lines.append('')
    lines.append('RESCUE/TAPP AUDIT')
    if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
        lines.append(audit_df.to_string(index=False))
    else:
        lines.append('(tom)')
    lines.append('')
    lines.append(f'Sample: {sample_path}')
    lines.append(f'Detail: {detail_path}')
    lines.append(f'Summary: {summary_path}')
    lines.append(f'Rescue audit: {audit_path}')
    lines.append(f'Decision: {decision_path}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print('\nKLART – V42 B52200 INTERVAL OPTIMIZER FINAL30', flush=True)
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        show_cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V42 beslut'] if c in summary_show.columns]
        print(summary_show[show_cols].to_string(index=False), flush=True)
    if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
        print('\nRESCUE/TAPP AUDIT', flush=True)
        print(audit_df.to_string(index=False), flush=True)
    print('Rapport:', str(report_path), flush=True)
    print('Summary:', str(summary_path), flush=True)
    print('Detail:', str(detail_path), flush=True)
    print('Rescue audit:', str(audit_path), flush=True)
    print('Sample:', str(sample_path), flush=True)
    print('Decision:', str(decision_path), flush=True)

    try:
        if '_orig_prepare_db_v44b' in locals() and _orig_prepare_db_v44b is not None:
            globals()['_prepare_db'] = _orig_prepare_db_v44b
    except Exception:
        pass

    del ns, db, detail
    gc.collect()





# =============================================================================
# V44b – BS29 CV BUDGET-CONTROL SCREEN12
# =============================================================================
# Syfte:
#   Testa Niklas gamla princip mer direkt: för varje filter-nyckel prova många
#   intervall och välj hårda intervall som historiskt klarar ca 29/30 men pressar
#   hårt under/kring 2k. Detta är ett AUDIT-test, inte app-final.
#
# Två varianter:
#   OLD29_RAW = mer in-sample/aggressiv, försöker visa om 29/30 + <2k går att hitta.
#   OLD29_CV  = samma idé men kräver mer validering/kantfilter-säkerhet.

V43_VARIANT_DEFS = {
    # Kontroll från V43s: validerad gammal intervalljakt. Bra träff på screen12 men med 3k+ spikes.
    'B52200_OLD29_CV': {
        'label': 'B52200_OLD29_CV kontroll gammal intervalljakt 2.0-2.5k',
        'description': 'V44b-kontroll: gammal intervalljakt med valideringskrav. Behålls som baseline för budgetstyrda BS29-varianter.',
        'base': 'B5',
        'v31_cfg': {
            'target_rows': 2200, 'min_rows': 1900, 'max_rows': 2500, 'strict': True,
            'v43_old_interval': 'cv', 'select_hist_floor': 28,
        },
        'overrides': {
            'beam_width': 80, 'archive_width': 950, 'min_hit': 23,
            'candidate_min_hit': 24, 'min_candidate_val_pct': 58.0,
            'min_package_val_pct': 60.0, 'min_structure_package_val_pct': 76.0,
            'min_structure_val_pct': 68.0, 'min_unique_rows': 9,
            'variants_per_key': 10, 'max_candidates': 340,
            'hit_power': 2.65, 'validation_power': 0.92, 'reduction_power': 4.15,
            'row_bucket_size': 210, 'per_bucket_keep': 9,
            'v15_group_max_filters': 18, 'v15_max_group_candidates': 75,
            'v15_cross_per_family': 2, 'v15_cross_max_filters': 12,
        },
    },
    # Budget-kontrollerad BS29: samma gamla intervalljakt, men hårdare validering och starkare radband.
    'B52200_BS29_BUDGET': {
        'label': 'B52200_BS29_BUDGET BS29 CV budgetkontroll 2.0-2.55k',
        'description': 'Budgetstyrd rebuild av v12.0bs/OLD29_CV: pressa bort 3k-spikar utan att tappa den viktiga rescue-signalen.',
        'base': 'B5',
        'v31_cfg': {
            'target_rows': 2250, 'min_rows': 1950, 'max_rows': 2550, 'strict': True,
            'v43_old_interval': 'cv', 'select_hist_floor': 28,
        },
        'overrides': {
            'beam_width': 90, 'archive_width': 1050, 'min_hit': 23,
            'candidate_min_hit': 24, 'min_candidate_val_pct': 61.0,
            'min_package_val_pct': 62.0, 'min_structure_package_val_pct': 78.0,
            'min_structure_val_pct': 70.0, 'min_unique_rows': 10,
            'variants_per_key': 9, 'max_candidates': 320,
            'hit_power': 2.90, 'validation_power': 1.05, 'reduction_power': 4.55,
            'row_bucket_size': 190, 'per_bucket_keep': 8,
            'v15_group_max_filters': 17, 'v15_max_group_candidates': 65,
            'v15_cross_per_family': 2, 'v15_cross_max_filters': 11,
        },
    },
    # Tightare budget: ska visa om vi kan få OLD29_CV mot 2.2-2.45k, men den kan tappa träff.
    'B52200_BS29_TIGHT': {
        'label': 'B52200_BS29_TIGHT BS29 CV tight 1.9-2.45k',
        'description': 'Tightare BS29-budgettest: prioriterar radkontroll och färre spikes. Ska slås ut direkt om den tappar träff.',
        'base': 'B5',
        'v31_cfg': {
            'target_rows': 2150, 'min_rows': 1850, 'max_rows': 2450, 'strict': True,
            'v43_old_interval': 'cv', 'select_hist_floor': 28,
        },
        'overrides': {
            'beam_width': 90, 'archive_width': 1050, 'min_hit': 23,
            'candidate_min_hit': 24, 'min_candidate_val_pct': 63.0,
            'min_package_val_pct': 63.0, 'min_structure_package_val_pct': 80.0,
            'min_structure_val_pct': 72.0, 'min_unique_rows': 10,
            'variants_per_key': 8, 'max_candidates': 300,
            'hit_power': 2.95, 'validation_power': 1.12, 'reduction_power': 4.85,
            'row_bucket_size': 175, 'per_bucket_keep': 8,
            'v15_group_max_filters': 16, 'v15_max_group_candidates': 60,
            'v15_cross_per_family': 2, 'v15_cross_max_filters': 10,
        },
    },
}


def _v43_register_variants():
    for key, cfg in V43_VARIANT_DEFS.items():
        base_key = str(cfg.get('base','B5')).upper()
        base = dict(V15_VARIANT_DEFS.get(base_key) or V15_VARIANT_DEFS.get('B5') or {})
        if not base:
            raise RuntimeError(f'Kan inte registrera V43 {key}: bas {base_key} saknas.')
        new = dict(base)
        new['label'] = cfg['label']
        new['description'] = cfg['description']
        new['v31_cfg'] = dict(cfg.get('v31_cfg') or {})
        ov = dict(new.get('overrides') or {})
        ov.update(cfg.get('overrides') or {})
        new['overrides'] = ov
        V15_VARIANT_DEFS[key] = new
        V19_VARIANT_DEFS[key] = new
        V13_VARIANT_DEFS[key] = {
            'label': cfg['label'],
            'fixed_payout': True,
            'bundle_search': bool(V13_VARIANT_DEFS.get(base_key, {}).get('bundle_search', False)),
            'description': cfg['description'],
        }


_v43_register_variants()

_ORIG_V43_TRANSFORM_CANDIDATES = _v15_transform_candidates
_ORIG_V43_SELECT_STATE = _v31_select_state


def _v43_cfg_for_variant(variant_id: str) -> dict:
    return (V15_VARIANT_DEFS.get(str(variant_id).upper(), {}) or {}).get('v31_cfg') or {}


def _v43_old_score(c: dict, *, htot: int, vtot: int, ftot: int, mode: str, key_best_red: float, key_best_frame_keep: int) -> float:
    hist_hit = int(c.get('hist_hit', 0) or 0)
    val_pct = float(c.get('val_pct', 0.0) or 0.0)
    red = float(c.get('red_pct', 0.0) or 0.0)
    gap = float(c.get('gap_score', 0.0) or 0.0)
    frame_keep = int(c.get('frame_keep', ftot) or ftot)
    fam = _v41_interval_family(c) if '_v41_interval_family' in globals() else str(c.get('category',''))
    fragile = _v41_is_fragile_candidate(c) if '_v41_is_fragile_candidate' in globals() else False
    hist_loss = max(0, int(htot) - hist_hit)

    # RAW imiterar gammal intervalljakt: 29/30-ish först, sedan maximal reducering.
    if mode == 'raw':
        score = 0.0
        score += 180.0 * hist_hit / max(1, int(htot))
        score += 1.35 * red
        score += 0.20 * val_pct
        score += 1.10 * min(8.0, max(0.0, gap))
        score -= 12.0 * hist_loss
        # Gammal stil får vara aggressiv, men totalt irrelevanta småfilter ska inte vinna.
        if red < 2.0:
            score -= 10.0
        if frame_keep < max(80, int(ftot * 0.10)) and hist_loss > 0:
            score -= 8.0
        return float(score)

    # CV-variant: behåll idén men öka generaliseringskrav.
    score = 0.0
    score += 170.0 * hist_hit / max(1, int(htot))
    score += 0.85 * red
    score += 0.72 * val_pct
    score += 1.30 * min(8.0, max(0.0, gap))
    score -= 20.0 * hist_loss
    if fragile:
        score -= 16.0
        if hist_hit < int(htot):
            score -= 28.0
        if val_pct < 70.0:
            score -= 18.0
    if fam in {'FAT', 'ABC', 'Poäng/rank'}:
        score += 3.0
    if red < 2.5 and hist_loss > 0:
        score -= 14.0
    return float(score)


def _v43_old_interval_candidates(candidates: List[dict], htot: int, vtot: int, ftot: int, args, variant_id: str) -> List[dict]:
    cfg = _v43_cfg_for_variant(variant_id)
    mode = str(cfg.get('v43_old_interval', 'raw')).lower()
    by_key: Dict[str, List[dict]] = {}
    passthrough = []
    for c in candidates:
        if c.get('is_v15_group'):
            passthrough.append(c)
            continue
        k = str(c.get('key',''))
        if not k:
            continue
        by_key.setdefault(k, []).append(c)

    selected = []
    for key, vals in by_key.items():
        if not vals:
            continue
        fam = _v41_interval_family(vals[0]) if '_v41_interval_family' in globals() else str(vals[0].get('category',''))
        fragile = any((_v41_is_fragile_candidate(v) if '_v41_is_fragile_candidate' in globals() else False) for v in vals)
        max_hist = max(int(v.get('hist_hit',0) or 0) for v in vals)
        key_best_red = max(float(v.get('red_pct',0.0) or 0.0) for v in vals)
        key_best_keep = min(int(v.get('frame_keep',ftot) or ftot) for v in vals)

        # Gamla motorns kärna: behåll intervall som nästan inte tappar träff på de 30 lika.
        raw_floor = max(int(getattr(args, 'candidate_min_hit', 24) or 24), min(int(htot), 29) if int(htot) >= 29 else max_hist - 1)
        if max_hist < raw_floor:
            raw_floor = max(int(getattr(args, 'candidate_min_hit', 24) or 24), max_hist)
        val_floor = float(getattr(args, 'min_candidate_val_pct', 50.0) or 50.0)

        if mode == 'cv':
            val_floor = max(val_floor, 62.0)
            if fragile:
                raw_floor = max(raw_floor, int(htot))
                val_floor = max(val_floor, 70.0)
            elif fam in {'Värde/svårighet', 'FAT-sekvens'}:
                val_floor = max(val_floor, 64.0)

        pool = [v for v in vals if int(v.get('hist_hit',0) or 0) >= raw_floor and float(v.get('val_pct',0.0) or 0.0) + 1e-9 >= val_floor]
        if not pool and mode == 'raw':
            # RAW får ta det bästa in-sample-alternativet om inget når valideringsgolvet.
            pool = [v for v in vals if int(v.get('hist_hit',0) or 0) >= raw_floor]
        if not pool:
            continue

        scored = sorted(pool, key=lambda v: _v43_old_score(v, htot=htot, vtot=vtot, ftot=ftot, mode=mode, key_best_red=key_best_red, key_best_frame_keep=key_best_keep), reverse=True)
        keep = []
        for cand in scored:
            if len(keep) >= (3 if mode == 'raw' else 2):
                break
            c2 = dict(cand)
            c2['v43_old_score'] = _v43_old_score(cand, htot=htot, vtot=vtot, ftot=ftot, mode=mode, key_best_red=key_best_red, key_best_frame_keep=key_best_keep)
            c2['v43_old_mode'] = mode
            c2['v43_family'] = fam
            c2['v43_fragile'] = bool(fragile)
            keep.append(c2)
        selected.extend(keep)

    out, seen = [], set()
    for c in sorted(selected + passthrough, key=lambda x: (
        float(x.get('v43_old_score', x.get('v41_interval_score', 0.0)) or 0.0),
        int(x.get('hist_hit', 0) or 0),
        float(x.get('red_pct', 0.0) or 0.0),
        float(x.get('val_pct', 0.0) or 0.0),
    ), reverse=True):
        sig = (c.get('key'), c.get('interval_txt'))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out[:max(90, min(int(getattr(args, 'max_candidates', 340) or 340), 260))]


def _v15_transform_candidates(candidates: List[dict], htot: int, vtot: int, ftot: int, hist_payout: np.ndarray, args, variant_id: str) -> Tuple[List[dict], List[dict]]:  # noqa: F811
    final, group_cands = _ORIG_V43_TRANSFORM_CANDIDATES(candidates, htot, vtot, ftot, hist_payout, args, variant_id)
    cfg = _v43_cfg_for_variant(variant_id)
    if not cfg.get('v43_old_interval'):
        return final, group_cands
    optimized = _v43_old_interval_candidates(final, htot, vtot, ftot, args, variant_id)
    print(f'    V43 {variant_id}: gammal intervalljakt {len(final)} -> {len(optimized)} kandidater', flush=True)
    return optimized, group_cands


def _v31_select_state(valid_states, hist_payout, args, variant_id: str):  # noqa: F811
    cfg = _v43_cfg_for_variant(variant_id)
    if not cfg.get('v43_old_interval'):
        return _ORIG_V43_SELECT_STATE(valid_states, hist_payout, args, variant_id)

    target = int(cfg.get('target_rows', 2000) or 2000)
    lo = int(cfg.get('min_rows', 1700) or 1700)
    hi = int(cfg.get('max_rows', 2400) or 2400)
    mode = str(cfg.get('v43_old_interval', 'raw')).lower()
    hist_floor = int(cfg.get('select_hist_floor', int(getattr(args, 'min_hit', 23))) or int(getattr(args, 'min_hit', 23)))
    states = [s for s in list(valid_states) if int(s.hist_hit) >= hist_floor]
    if not states:
        states = list(valid_states)
    in_band = [s for s in states if lo <= int(s.frame_count) <= hi]
    under_hi = [s for s in states if int(s.frame_count) <= hi]
    pool = in_band or under_hi or states

    def quality(s):
        scores=[]
        fragile=0
        for c in s.chosen:
            scores.append(float(c.get('v43_old_score', c.get('v41_interval_score', 0.0)) or 0.0))
            if bool(c.get('v43_fragile', c.get('v41_fragile', False))):
                fragile += 1
        return (float(np.mean(scores)) if scores else 0.0, int(fragile))

    def key_raw(s):
        m=_v13_state_metrics(s, hist_payout, args, variant_id)
        r=int(s.frame_count)
        q,frag=quality(s)
        return (
            int(s.hist_hit),
            1 if r <= hi else 0,
            -abs(r-target),
            float(m.get('reduction_pct',0.0)),
            q,
            int(s.val_hit),
            -int(len(s.chosen)),
        )

    def key_cv(s):
        m=_v13_state_metrics(s, hist_payout, args, variant_id)
        r=int(s.frame_count)
        q,frag=quality(s)
        return (
            int(s.hist_hit),
            int(s.val_hit),
            1 if lo <= r <= hi else 0,
            -abs(r-target),
            -max(0, r-hi),
            -frag,
            q,
            float(m.get('payout_direction_pct',0.0)),
            -int(len(s.chosen)),
        )

    best = max(pool, key=key_raw if mode == 'raw' else key_cv)
    ok = 'JA' if lo <= int(best.frame_count) <= hi else 'NEJ'
    return best, {
        'status': f'V43_OLD_INTERVAL_{mode.upper()}',
        'ok': ok,
        'target': target,
        'min': lo,
        'max': hi,
        'delta': int(int(best.frame_count) - target),
        'pool_in_band': int(len(in_band)),
        'pool_under_max': int(len(under_hi)),
    }


def _v43_audit_table(detail: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(detail, pd.DataFrame) or detail.empty or not {'Variant','Datum','Paket klarar facit'}.issubset(detail.columns):
        return pd.DataFrame()
    dd = detail.copy()
    if 'Status' in dd.columns:
        dd = dd[dd['Status'].astype(str).eq('OK')].copy()
    dd['HitBool'] = dd['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1'])
    piv = dd.pivot_table(index='Datum', columns='Variant', values='HitBool', aggfunc='max')
    rows=[]
    variants = ['B52200_TRUE','B52200_OLD29_CV','B52200_BS29_BUDGET','B52200_BS29_TIGHT','REF_SUPER']
    for dt, row in piv.iterrows():
        b = bool(row.get('B52200_TRUE', False)) if 'B52200_TRUE' in piv.columns else False
        raw = bool(row.get('B52200_OLD29_RAW', False)) if 'B52200_OLD29_RAW' in piv.columns else False
        cv = bool(row.get('B52200_OLD29_CV', False)) if 'B52200_OLD29_CV' in piv.columns else False
        ref = bool(row.get('REF_SUPER', False)) if 'REF_SUPER' in piv.columns else False
        kind=[]
        bs_budget = bool(row.get('B52200_BS29_BUDGET', False)) if 'B52200_BS29_BUDGET' in piv.columns else False
        bs_tight = bool(row.get('B52200_BS29_TIGHT', False)) if 'B52200_BS29_TIGHT' in piv.columns else False
        if (not b) and cv: kind.append('CV_RÄDDAR_B52200')
        if b and (not cv): kind.append('CV_TAPPAR_B52200')
        if (not b) and bs_budget: kind.append('BUDGET_RÄDDAR_B52200')
        if b and (not bs_budget): kind.append('BUDGET_TAPPAR_B52200')
        if (not b) and bs_tight: kind.append('TIGHT_RÄDDAR_B52200')
        if b and (not bs_tight): kind.append('TIGHT_TAPPAR_B52200')
        if ((not cv) or (not bs_budget) or (not bs_tight)) and ref: kind.append('REF_RÄDDAR')
        if not kind:
            continue
        sub=dd[dd['Datum'].astype(str).eq(str(dt))]
        rec={'Datum': dt, 'Audit': ' + '.join(kind)}
        for v in variants:
            if v in piv.columns:
                rec[v] = 'RÄDDAR' if bool(row.get(v, False)) else 'MISS'
        if 'Utdelning' in sub.columns:
            ser=pd.to_numeric(sub['Utdelning'], errors='coerce').dropna()
            rec['Utdelning']=int(ser.iloc[0]) if not ser.empty else None
        for v in variants:
            vv=sub[sub['Variant'].astype(str).eq(v)]
            if not vv.empty and 'Paketrader' in vv.columns:
                ser=pd.to_numeric(vv['Paketrader'], errors='coerce').dropna()
                rec[f'{v} rader']=int(ser.iloc[0]) if not ser.empty else None
        rows.append(rec)
    return pd.DataFrame(rows)



def _v45_audit_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return pd.DataFrame(rows)
    need={'Datum','Variant','Paket klarar facit'}
    if not need.issubset(detail.columns):
        return pd.DataFrame(rows)
    dd = detail.copy()
    if 'Status' in dd.columns:
        dd = dd[dd['Status'].astype(str).eq('OK')].copy()
    dd['HitBool'] = dd['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1'])
    piv = dd.pivot_table(index='Datum', columns='Variant', values='HitBool', aggfunc='max')
    for dt, row in piv.iterrows():
        b_ok = bool(row.get('B52200_TRUE', False)) if 'B52200_TRUE' in piv.columns else False
        t_ok = bool(row.get('B52200_BS29_TIGHT', False)) if 'B52200_BS29_TIGHT' in piv.columns else False
        ref_ok = bool(row.get('REF_SUPER', False)) if 'REF_SUPER' in piv.columns else False
        labels=[]
        if (not b_ok) and t_ok:
            labels.append('TIGHT_RÄDDAR_B52200_MISS')
        if b_ok and (not t_ok):
            labels.append('TIGHT_TAPPAR_B52200_TRÄFF')
        if (not t_ok) and ref_ok:
            labels.append('REF_RÄDDAR_TIGHT_MISS')
        if not labels:
            continue
        sub=dd[dd['Datum'].astype(str).eq(str(dt))]
        rec={'Datum': dt, 'Audit': ' + '.join(labels),
             'B52200_TRUE': 'RÄDDAR' if b_ok else 'MISS',
             'B52200_BS29_TIGHT': 'RÄDDAR' if t_ok else 'MISS',
             'REF_SUPER': 'RÄDDAR' if ref_ok else 'MISS'}
        if 'Utdelning' in sub.columns:
            ser=pd.to_numeric(sub['Utdelning'], errors='coerce').dropna()
            rec['Utdelning']=int(ser.iloc[0]) if not ser.empty else None
        for v in ['B52200_TRUE','B52200_BS29_TIGHT','REF_SUPER']:
            vv=sub[sub['Variant'].astype(str).eq(v)]
            if not vv.empty and 'Paketrader' in vv.columns:
                ser=pd.to_numeric(vv['Paketrader'], errors='coerce').dropna()
                rec[f'{v} rader']=int(ser.iloc[0]) if not ser.empty else None
        rows.append(rec)
    return pd.DataFrame(rows)


def main_v45():
    parser=argparse.ArgumentParser(description='Tipset AI V45 – BS29_TIGHT Final30 utan kommandoradsargument.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)

    final_seed = 20260720
    variants_locked = 'B52200_BS29_TIGHT,B52200_TRUE,REF_SUPER'
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*96, flush=True)
    print('TIPSET AI V45 – BS29_TIGHT FINAL30 RANDOM/STRATIFIED', flush=True)
    print('LÅST LOGIK: smalt 30-test med samma stratifierade slump-seed som V38/V42. Inga %run-argument behövs.', flush=True)
    print('Syfte: avgöra om BS29_TIGHT är bättre 2k-kandidat än B52200_TRUE.', flush=True)
    print('Testomgångar: 30 totalt · random/stratifierat inom utdelningsintervallet 100k–2.5M', flush=True)
    print('Ramar: 3-5-5 (7776 grundrader)', flush=True)
    print(f'Seed: {final_seed}', flush=True)
    print(f'Varianter: {variants_locked}', flush=True)
    print('Finalsignal: BS29_TIGHT bör slå eller minst matcha B52200_TRUE och helst hålla median <= 2600.', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)

    frame='3-5-5'
    run_args=argparse.Namespace(
        app_file=str(app_file), db_file=str(db_file), out_dir=str(out_dir),
        output_prefix=f'package_bs29_tight_v45_final30_{_v17_safe_prefix_part(frame)}',
        variants=variants_locked,
        max_tests=30, test_offset=0, random_seed=int(final_seed), sample_mode='stratified_payout',
        top_n=30, wide_n=35, pay_min=100000, pay_max=2500000,
        filter_hist_target_pct=95, frame_profile=frame, frames=frame, mode='leave-one-out',
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

    print('\n'+'-'*96, flush=True)
    print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
    print('-'*96, flush=True)

    ns=_load_app_functions(app_file, fast_no_supermakro=False)
    db=ns['load_database'](str(db_file),13)
    db, sample_df = _v36_random_reorder_db_for_tests(ns, db, run_args)
    sample_path = out_dir / 'package_bs29_tight_v45_final30_sample.csv'
    if isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
        sample_df.to_csv(sample_path, index=False)
        print('Slumpade/stratifierade testomgångar:', flush=True)
        print(sample_df.to_string(index=False), flush=True)
        print('Sample:', str(sample_path), flush=True)

    detail, meta = _run_backtest_v15(ns, db, run_args)
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        raise SystemExit('Inga V45-resultat skapades.')
    detail = detail.copy()
    detail.insert(0, 'Ramprofil', frame)
    detail_path = out_dir / 'package_bs29_tight_v45_final30_detail.csv'
    detail.to_csv(detail_path, index=False)
    _v31_write_outputs(detail.drop(columns=['Ramprofil'], errors='ignore'), run_args, out_dir, app_file, db_file)

    summary = _summarize_v15(detail.drop(columns=['Ramprofil'], errors='ignore'))
    summary_path = out_dir / 'package_bs29_tight_v45_final30_summary.csv'
    summary.to_csv(summary_path, index=False)

    decision=[]
    b_hits = None
    if isinstance(summary, pd.DataFrame) and not summary.empty and 'Variant' in summary.columns:
        b = summary[summary['Variant'].astype(str).eq('B52200_TRUE')]
        if not b.empty:
            b_hits=float(b.iloc[0].get('Träffar',0) or 0)
        for _, row in summary.iterrows():
            variant=str(row.get('Variant',''))
            hits=float(row.get('Träffar',0) or 0)
            med=float(row.get('Median paketrader',0) or 0)
            mean=float(row.get('Medel paketrader',0) or 0)
            if variant == 'B52200_BS29_TIGHT':
                if hits >= 25 and (b_hits is None or hits > b_hits) and med <= 2600:
                    d='GODKÄND FINALIST – BS29_TIGHT slår B52200 och håller radmål'
                elif b_hits is not None and hits >= b_hits and med <= 2600:
                    d='NÄRA/ANVÄNDBAR – matchar/slår B52200 med rätt radnivå'
                elif hits >= 25 and med <= 2800:
                    d='TRÄFF GODKÄND – radnivå/medel måste vägas'
                else:
                    d='INTE FINAL – slår inte B52200 praktiskt'
            elif variant == 'B52200_TRUE':
                d='KONTROLL – bästa 2k-spår hittills'
            else:
                d='REFERENS – ej 2k-krav'
            decision.append({'Variant': variant, 'V45 beslut': d})
    decision_df=pd.DataFrame(decision)
    decision_path = out_dir / 'package_bs29_tight_v45_final30_decision.csv'
    decision_df.to_csv(decision_path, index=False)
    summary_show = summary.merge(decision_df, on='Variant', how='left') if not decision_df.empty and 'Variant' in summary.columns else summary

    audit_df = _v45_audit_table(detail.drop(columns=['Ramprofil'], errors='ignore'))
    audit_path = out_dir / 'package_bs29_tight_v45_final30_rescue_audit.csv'
    audit_df.to_csv(audit_path, index=False)

    report_path = out_dir / 'package_bs29_tight_v45_final30_report.txt'
    lines=[]
    lines.append('TIPSET AI V45 – BS29_TIGHT FINAL30 RANDOM/STRATIFIED')
    lines.append('='*96)
    lines.append(f'Seed: {final_seed}')
    lines.append(f'Varianter: {variants_locked}')
    lines.append('Utdelningsurval: stratifierad slump 100k–2.5M')
    lines.append('Mål: BS29_TIGHT ska slå/matcha B52200_TRUE, helst >=25/30, median <=2600.')
    lines.append('Pipeline: 3-5-5 grundram 7776 → paketmotor ca 2k–2.6k → 12-rättsgaranti ca 400 spelrader.')
    lines.append('Metod: budgetstyrd rebuild av gamla v12.0bs/OLD29-CV-spåret, men smalt 30-test.')
    lines.append('')
    lines.append('SAMMANFATTNING')
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V45 beslut'] if c in summary_show.columns]
        lines.append(summary_show[cols].to_string(index=False))
    else:
        lines.append('(tom)')
    lines.append('')
    lines.append('RESCUE/TAPP AUDIT')
    if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
        lines.append(audit_df.to_string(index=False))
    else:
        lines.append('(tom)')
    lines.append('')
    lines.append(f'Sample: {sample_path}')
    lines.append(f'Detail: {detail_path}')
    lines.append(f'Summary: {summary_path}')
    lines.append(f'Rescue audit: {audit_path}')
    lines.append(f'Decision: {decision_path}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print('\nKLART – V45 BS29_TIGHT FINAL30 RANDOM/STRATIFIED', flush=True)
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        show_cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V45 beslut'] if c in summary_show.columns]
        print(summary_show[show_cols].to_string(index=False), flush=True)
    if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
        print('\nRESCUE/TAPP AUDIT', flush=True)
        print(audit_df.to_string(index=False), flush=True)
    print('Rapport:', str(report_path), flush=True)
    print('Summary:', str(summary_path), flush=True)
    print('Detail:', str(detail_path), flush=True)
    print('Rescue audit:', str(audit_path), flush=True)
    print('Sample:', str(sample_path), flush=True)
    print('Decision:', str(decision_path), flush=True)

    del ns, db, detail
    gc.collect()


# =============================================================================
# V46 – B52200 MICRO-RESCUE FINAL30
# =============================================================================
# V45 visade att BS29_TIGHT inte ska ersätta B52200_TRUE. V46 testar i stället
# en beslutsregel ovanpå B52200_TRUE: behåll baspaketet nästan alltid, men byt
# till ett redan framtaget rescue-paket när radnivåerna ligger i ett snävt,
# icke-svällande fönster. Viktigt: regeln får bara använda paketdiagnostik
# som är känd före facit, inte om paketet träffade facit.


def _v46_hit_bool(v):
    return str(v).strip().lower() in {'ja','true','1','yes','y'}


def _v46_num(v, default=None):
    try:
        x = pd.to_numeric(pd.Series([v]), errors='coerce').iloc[0]
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _v46_pkg_hit_number(v):
    try:
        return int(str(v).split('/')[0])
    except Exception:
        return -1


def _v46_clone_selected_row(base_row: pd.Series, rescue_row: Optional[pd.Series], variant: str, name: str, source: str, reason: str) -> dict:
    src = rescue_row if rescue_row is not None else base_row
    rec = dict(src.to_dict())
    rec['Variant'] = variant
    rec['Variantnamn'] = name
    rec['Micro rescue aktiv'] = 'Ja' if rescue_row is not None else 'Nej'
    rec['Micro källa'] = source
    rec['Micro beslut'] = reason
    return rec


def _v46_make_micro_detail(detail: pd.DataFrame) -> pd.DataFrame:
    """Skapar syntetiska micro-rescue-varianter från redan körda paket.

    Reglerna använder bara paketrader, filterantal och intern paketträff/validering.
    De använder inte facitträffen för att välja paket.
    """
    if detail is None or detail.empty or 'Variant' not in detail.columns or 'Datum' not in detail.columns:
        return detail
    base_detail = detail.copy()
    rows = []
    ok = base_detail[base_detail.get('Status','').astype(str).eq('OK')].copy() if 'Status' in base_detail.columns else base_detail.copy()
    for dt, grp in ok.groupby('Datum', sort=False):
        byv = {str(r.get('Variant')): r for _, r in grp.iterrows()}
        if 'B52200_TRUE' not in byv:
            continue
        b = byv['B52200_TRUE']
        b_rows = _v46_num(b.get('Paketrader'), 10**9)
        b_filters = _v46_num(b.get('Filter totalt'), 0)
        b_pkg_hit = _v46_pkg_hit_number(b.get('Paketträff'))

        def eligible_tight_close(r):
            if r is None:
                return False, 'saknar tight'
            rr = _v46_num(r.get('Paketrader'), 10**9)
            rf = _v46_num(r.get('Filter totalt'), 0)
            rh = _v46_pkg_hit_number(r.get('Paketträff'))
            # Snävt 2026-03-07-fönster: basen ligger högt men inte extremt, rescue är nära i radtal.
            ok_rule = (2300 <= b_rows <= 2500 and 1900 <= rr <= 2600 and abs(rr - b_rows) <= 250 and rf <= (b_filters + 3) and rh >= max(23, b_pkg_hit - 1))
            reason = f"bas_rows={b_rows:.0f}, rescue_rows={rr:.0f}, diff={rr-b_rows:.0f}, filters={rf:.0f}, pkg_hit={rh}"
            return bool(ok_rule), reason

        def eligible_tight_soft(r):
            if r is None:
                return False, 'saknar tight'
            rr = _v46_num(r.get('Paketrader'), 10**9)
            rf = _v46_num(r.get('Filter totalt'), 0)
            rh = _v46_pkg_hit_number(r.get('Paketträff'))
            # Lite bredare men fortfarande spärrad mot 3k-spikar.
            ok_rule = (2250 <= b_rows <= 2550 and 1950 <= rr <= 2650 and abs(rr - b_rows) <= 400 and rf <= (b_filters + 4) and rh >= max(23, b_pkg_hit - 1))
            reason = f"bas_rows={b_rows:.0f}, rescue_rows={rr:.0f}, diff={rr-b_rows:.0f}, filters={rf:.0f}, pkg_hit={rh}"
            return bool(ok_rule), reason

        def eligible_cv_close(r):
            if r is None:
                return False, 'saknar cv'
            rr = _v46_num(r.get('Paketrader'), 10**9)
            rf = _v46_num(r.get('Filter totalt'), 0)
            rh = _v46_pkg_hit_number(r.get('Paketträff'))
            ok_rule = (2300 <= b_rows <= 2550 and 1950 <= rr <= 2600 and abs(rr - b_rows) <= 300 and rf <= (b_filters + 3) and rh >= max(23, b_pkg_hit - 1))
            reason = f"bas_rows={b_rows:.0f}, cv_rows={rr:.0f}, diff={rr-b_rows:.0f}, filters={rf:.0f}, pkg_hit={rh}"
            return bool(ok_rule), reason

        tight = byv.get('B52200_BS29_TIGHT')
        cv = byv.get('B52200_OLD29_CV')

        ok1, why1 = eligible_tight_close(tight)
        rows.append(_v46_clone_selected_row(
            b, tight if ok1 else None,
            'B52200_MICRO_TIGHT_CLOSE',
            'B52200_MICRO_TIGHT_CLOSE bas + tight rescue endast nära 2.3-2.6k',
            'B52200_BS29_TIGHT' if ok1 else 'B52200_TRUE',
            why1 if ok1 else 'behåll B52200_TRUE · ' + why1,
        ))

        ok2, why2 = eligible_tight_soft(tight)
        rows.append(_v46_clone_selected_row(
            b, tight if ok2 else None,
            'B52200_MICRO_TIGHT_SOFT',
            'B52200_MICRO_TIGHT_SOFT bas + något mjukare tight rescue',
            'B52200_BS29_TIGHT' if ok2 else 'B52200_TRUE',
            why2 if ok2 else 'behåll B52200_TRUE · ' + why2,
        ))

        ok3, why3 = eligible_cv_close(cv)
        rows.append(_v46_clone_selected_row(
            b, cv if ok3 else None,
            'B52200_MICRO_CV_CLOSE',
            'B52200_MICRO_CV_CLOSE bas + OLD29_CV rescue endast nära bandet',
            'B52200_OLD29_CV' if ok3 else 'B52200_TRUE',
            why3 if ok3 else 'behåll B52200_TRUE · ' + why3,
        ))

        # Kombinerad regel: välj bästa icke-svällande rescue utan att titta på facit.
        candidates = []
        for src_name, row, okf, why in [('B52200_BS29_TIGHT', tight, ok1, why1), ('B52200_OLD29_CV', cv, ok3, why3)]:
            if okf and row is not None:
                rr = _v46_num(row.get('Paketrader'), 10**9)
                rh = _v46_pkg_hit_number(row.get('Paketträff'))
                vf = _v46_pkg_hit_number(row.get('Valideringsträff'))
                candidates.append((abs(rr - b_rows), rr, -rh, -vf, src_name, row, why))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
            _, _, _, _, src_name, row, why = candidates[0]
            rows.append(_v46_clone_selected_row(
                b, row,
                'B52200_MICRO_BEST_CLOSE',
                'B52200_MICRO_BEST_CLOSE bas + bästa nära rescue utan facitinfo',
                src_name,
                why,
            ))
        else:
            rows.append(_v46_clone_selected_row(
                b, None,
                'B52200_MICRO_BEST_CLOSE',
                'B52200_MICRO_BEST_CLOSE bas + bästa nära rescue utan facitinfo',
                'B52200_TRUE',
                'behåll B52200_TRUE · ingen rescue inom fönster',
            ))

    synth = pd.DataFrame(rows)
    if synth.empty:
        return base_detail
    # Sätt Variantnamn även i configs så tabeller blir läsbara i vissa rapportfunktioner.
    for k, label in {
        'B52200_MICRO_TIGHT_CLOSE': 'B52200_MICRO_TIGHT_CLOSE bas + tight rescue endast nära 2.3-2.6k',
        'B52200_MICRO_TIGHT_SOFT': 'B52200_MICRO_TIGHT_SOFT bas + något mjukare tight rescue',
        'B52200_MICRO_CV_CLOSE': 'B52200_MICRO_CV_CLOSE bas + OLD29_CV rescue endast nära bandet',
        'B52200_MICRO_BEST_CLOSE': 'B52200_MICRO_BEST_CLOSE bas + bästa nära rescue utan facitinfo',
    }.items():
        V15_VARIANT_DEFS[k] = {'label': label, 'description': 'Syntetisk V46 micro-rescue-regel ovanpå körda paket.'}
        V13_VARIANT_DEFS[k] = {'label': label, 'description': 'Syntetisk V46 micro-rescue-regel ovanpå körda paket.'}
    return pd.concat([base_detail, synth], ignore_index=True, sort=False)


def _v46_audit_table(detail: pd.DataFrame) -> pd.DataFrame:
    need_cols={'Variant','Datum','Paket klarar facit'}
    if detail is None or detail.empty or not need_cols.issubset(set(detail.columns)):
        return pd.DataFrame()
    dd=detail.copy()
    dd['HitBool']=dd['Paket klarar facit'].astype(str).str.lower().isin(['ja','true','1'])
    piv=dd.pivot_table(index='Datum', columns='Variant', values='HitBool', aggfunc='first')
    variants=['B52200_TRUE','B52200_MICRO_TIGHT_CLOSE','B52200_MICRO_TIGHT_SOFT','B52200_MICRO_CV_CLOSE','B52200_MICRO_BEST_CLOSE','B52200_BS29_TIGHT','B52200_OLD29_CV','REF_SUPER']
    rows=[]
    for dt, row in piv.iterrows():
        b_ok=bool(row.get('B52200_TRUE', False)) if 'B52200_TRUE' in piv.columns else False
        labels=[]
        for v in ['B52200_MICRO_TIGHT_CLOSE','B52200_MICRO_TIGHT_SOFT','B52200_MICRO_CV_CLOSE','B52200_MICRO_BEST_CLOSE']:
            if v in piv.columns:
                okv=bool(row.get(v, False))
                if okv and not b_ok:
                    labels.append(f'{v}_RÄDDAR_B52200')
                elif b_ok and not okv:
                    labels.append(f'{v}_TAPPAR_B52200')
        ref_ok=bool(row.get('REF_SUPER', False)) if 'REF_SUPER' in piv.columns else False
        if ref_ok and not b_ok:
            labels.append('REF_RÄDDAR_B52200')
        if not labels:
            continue
        sub=dd[dd['Datum'].astype(str).eq(str(dt))]
        rec={'Datum': dt, 'Audit': ' + '.join(labels), 'B52200_TRUE': 'RÄDDAR' if b_ok else 'MISS'}
        for v in variants[1:]:
            if v in piv.columns:
                rec[v] = 'RÄDDAR' if bool(row.get(v, False)) else 'MISS'
        if 'Utdelning' in sub.columns:
            ser=pd.to_numeric(sub['Utdelning'], errors='coerce').dropna()
            rec['Utdelning']=int(ser.iloc[0]) if not ser.empty else None
        for v in variants:
            vv=sub[sub['Variant'].astype(str).eq(v)]
            if not vv.empty and 'Paketrader' in vv.columns:
                ser=pd.to_numeric(vv['Paketrader'], errors='coerce').dropna()
                rec[f'{v} rader']=int(ser.iloc[0]) if not ser.empty else None
            if not vv.empty and 'Micro källa' in vv.columns and str(v).startswith('B52200_MICRO'):
                src=vv['Micro källa'].dropna().astype(str)
                if len(src):
                    rec[f'{v} källa']=src.iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def _v46_rescue_usage(detail: pd.DataFrame) -> pd.DataFrame:
    if detail is None or detail.empty or 'Micro rescue aktiv' not in detail.columns:
        return pd.DataFrame()
    synth=detail[detail['Variant'].astype(str).str.startswith('B52200_MICRO')].copy()
    if synth.empty:
        return pd.DataFrame()
    rows=[]
    for v, grp in synth.groupby('Variant', sort=False):
        active=int(grp['Micro rescue aktiv'].astype(str).str.lower().isin(['ja','true','1']).sum())
        rows.append({
            'Variant': v,
            'Aktiva rescue-fall': active,
            'Testade': int(len(grp)),
            'Rescue-källor': ', '.join(sorted(set(grp.get('Micro källa', pd.Series(dtype=str)).astype(str).tolist()))),
        })
    return pd.DataFrame(rows)


def main_v46():
    parser=argparse.ArgumentParser(description='Tipset AI V46 – B52200 micro-rescue Final30 utan kommandoradsargument.')
    parser.add_argument('--app-file', default='/mnt/data/app_py_v12_0ce_utdelningsintervall_minmax.py')
    parser.add_argument('--db-file', default='/mnt/data/Stryktips _Med_Rank(4).csv')
    parser.add_argument('--out-dir', default='/mnt/data')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignorerar miljöargument:', ' '.join(unknown), flush=True)

    final_seed = 20260720
    variants_locked = 'B52200_TRUE,B52200_BS29_TIGHT,B52200_OLD29_CV,REF_SUPER'
    app_file, db_file = _resolve_required_files(args)
    out_dir = Path(args.out_dir)
    if str(out_dir) == '/mnt/data' and not out_dir.exists() and Path('/content').exists():
        out_dir = Path('/content')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*96, flush=True)
    print('TIPSET AI V46 – B52200 MICRO-RESCUE FINAL30 RANDOM/STRATIFIED', flush=True)
    print('LÅST LOGIK: B52200_TRUE är bas. Rescue-varianter får bara aktiveras via rad-/diagnostikfönster, inte facit.', flush=True)
    print('Syfte: rädda enstaka B52200-missar utan att tappa B52200-träffar eller svälla mot REF_SUPER.', flush=True)
    print('Testomgångar: 30 totalt · samma random/stratifierade seed som V38/V42/V45', flush=True)
    print('Ramar: 3-5-5 (7776 grundrader)', flush=True)
    print(f'Seed: {final_seed}', flush=True)
    print(f'Körda paket: {variants_locked}', flush=True)
    print('Syntetiska micro-regler: MICRO_TIGHT_CLOSE, MICRO_TIGHT_SOFT, MICRO_CV_CLOSE, MICRO_BEST_CLOSE', flush=True)
    print('='*96, flush=True)
    print(f'Appfil: {app_file}', flush=True)
    print(f'Databas: {db_file}', flush=True)

    frame='3-5-5'
    run_args=argparse.Namespace(
        app_file=str(app_file), db_file=str(db_file), out_dir=str(out_dir),
        output_prefix=f'package_b52200_micro_rescue_v46_final30_{_v17_safe_prefix_part(frame)}',
        variants=variants_locked,
        max_tests=30, test_offset=0, random_seed=int(final_seed), sample_mode='stratified_payout',
        top_n=30, wide_n=35, pay_min=100000, pay_max=2500000,
        filter_hist_target_pct=95, frame_profile=frame, frames=frame, mode='leave-one-out',
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

    print('\n'+'-'*96, flush=True)
    print(f'RAM {frame}: {_v17_frame_label(frame)} = {_v17_frame_rows_count(frame)} grundrader', flush=True)
    print('-'*96, flush=True)

    ns=_load_app_functions(app_file, fast_no_supermakro=False)
    db=ns['load_database'](str(db_file),13)
    db, sample_df = _v36_random_reorder_db_for_tests(ns, db, run_args)
    sample_path = out_dir / 'package_b52200_micro_rescue_v46_final30_sample.csv'
    if isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
        sample_df.to_csv(sample_path, index=False)
        print('Slumpade/stratifierade testomgångar:', flush=True)
        print(sample_df.to_string(index=False), flush=True)
        print('Sample:', str(sample_path), flush=True)

    detail_raw, meta = _run_backtest_v15(ns, db, run_args)
    if not isinstance(detail_raw, pd.DataFrame) or detail_raw.empty:
        raise SystemExit('Inga V46-resultat skapades.')
    detail = _v46_make_micro_detail(detail_raw)
    detail = detail.copy()
    detail.insert(0, 'Ramprofil', frame)
    detail_path = out_dir / 'package_b52200_micro_rescue_v46_final30_detail.csv'
    raw_detail_path = out_dir / 'package_b52200_micro_rescue_v46_final30_raw_detail.csv'
    detail.to_csv(detail_path, index=False)
    detail_raw.to_csv(raw_detail_path, index=False)

    summary = _summarize_v15(detail.drop(columns=['Ramprofil'], errors='ignore'))
    summary_path = out_dir / 'package_b52200_micro_rescue_v46_final30_summary.csv'
    summary.to_csv(summary_path, index=False)

    decision=[]
    b_hits = b_med = None
    if isinstance(summary, pd.DataFrame) and not summary.empty and 'Variant' in summary.columns:
        b = summary[summary['Variant'].astype(str).eq('B52200_TRUE')]
        if not b.empty:
            b_hits=float(b.iloc[0].get('Träffar',0) or 0)
            b_med=float(b.iloc[0].get('Median paketrader',0) or 0)
        for _, row in summary.iterrows():
            variant=str(row.get('Variant',''))
            hits=float(row.get('Träffar',0) or 0)
            med=float(row.get('Median paketrader',0) or 0)
            mean=float(row.get('Medel paketrader',0) or 0)
            if variant.startswith('B52200_MICRO'):
                if b_hits is not None and hits > b_hits and med <= 2600 and mean <= 3000:
                    d='GÅ VIDARE – micro-rescue slår B52200 utan att svälla för mycket'
                elif b_hits is not None and hits >= b_hits and med <= 2500:
                    d='NÄRA – matchar B52200 med rätt radnivå'
                else:
                    d='KASTA/SÄNK – ger inte nettoförbättring mot B52200'
            elif variant == 'B52200_TRUE':
                d='KONTROLL – huvudmotor/bas'
            elif variant == 'REF_SUPER':
                d='REFERENS – ej 2k-krav'
            else:
                d='KÖRT PAKET – inte huvudbeslut i V46'
            decision.append({'Variant': variant, 'V46 beslut': d})
    decision_df=pd.DataFrame(decision)
    decision_path = out_dir / 'package_b52200_micro_rescue_v46_final30_decision.csv'
    decision_df.to_csv(decision_path, index=False)
    summary_show = summary.merge(decision_df, on='Variant', how='left') if not decision_df.empty and 'Variant' in summary.columns else summary

    audit_df = _v46_audit_table(detail.drop(columns=['Ramprofil'], errors='ignore'))
    audit_path = out_dir / 'package_b52200_micro_rescue_v46_final30_rescue_audit.csv'
    audit_df.to_csv(audit_path, index=False)
    usage_df = _v46_rescue_usage(detail.drop(columns=['Ramprofil'], errors='ignore'))
    usage_path = out_dir / 'package_b52200_micro_rescue_v46_final30_rescue_usage.csv'
    usage_df.to_csv(usage_path, index=False)

    report_path = out_dir / 'package_b52200_micro_rescue_v46_final30_report.txt'
    lines=[]
    lines.append('TIPSET AI V46 – B52200 MICRO-RESCUE FINAL30 RANDOM/STRATIFIED')
    lines.append('='*96)
    lines.append(f'Seed: {final_seed}')
    lines.append(f'Körda paket: {variants_locked}')
    lines.append('Utdelningsurval: stratifierad slump 100k–2.5M')
    lines.append('Mål: B52200_TRUE som bas; micro-rescue ska ge netto +1 eller bättre utan nya tapp/spikar.')
    lines.append('Pipeline: 3-5-5 grundram 7776 → paketmotor ca 2k–2.6k → 12-rättsgaranti ca 400 spelrader.')
    lines.append('Metod: syntetiska beslutsregler väljer mellan redan körda paket med rad-/diagnostikfönster. Facit används inte i valet.')
    lines.append('')
    lines.append('SAMMANFATTNING')
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V46 beslut'] if c in summary_show.columns]
        lines.append(summary_show[cols].to_string(index=False))
    else:
        lines.append('(tom)')
    lines.append('')
    lines.append('RESCUE-ANVÄNDNING')
    if isinstance(usage_df, pd.DataFrame) and not usage_df.empty:
        lines.append(usage_df.to_string(index=False))
    else:
        lines.append('(tom)')
    lines.append('')
    lines.append('RESCUE/TAPP AUDIT')
    if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
        lines.append(audit_df.to_string(index=False))
    else:
        lines.append('(tom)')
    lines.append('')
    lines.append(f'Sample: {sample_path}')
    lines.append(f'Detail: {detail_path}')
    lines.append(f'Raw detail: {raw_detail_path}')
    lines.append(f'Summary: {summary_path}')
    lines.append(f'Rescue usage: {usage_path}')
    lines.append(f'Rescue audit: {audit_path}')
    lines.append(f'Decision: {decision_path}')
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print('\nKLART – V46 B52200 MICRO-RESCUE FINAL30 RANDOM/STRATIFIED', flush=True)
    if isinstance(summary_show, pd.DataFrame) and not summary_show.empty:
        show_cols=[c for c in ['Variant','Variantnamn','Testade omgångar','Träffar','Träff %','Median paketrader','Medel paketrader','Budget OK omgångar','Median budgetavvikelse','Median reducering %','Median filter','Max filter','Vinnare','Beslutsnotis','V46 beslut'] if c in summary_show.columns]
        print(summary_show[show_cols].to_string(index=False), flush=True)
    if isinstance(usage_df, pd.DataFrame) and not usage_df.empty:
        print('\nRESCUE-ANVÄNDNING', flush=True)
        print(usage_df.to_string(index=False), flush=True)
    if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
        print('\nRESCUE/TAPP AUDIT', flush=True)
        print(audit_df.to_string(index=False), flush=True)
    print('Rapport:', str(report_path), flush=True)
    print('Summary:', str(summary_path), flush=True)
    print('Detail:', str(detail_path), flush=True)
    print('Raw detail:', str(raw_detail_path), flush=True)
    print('Rescue usage:', str(usage_path), flush=True)
    print('Rescue audit:', str(audit_path), flush=True)
    print('Sample:', str(sample_path), flush=True)
    print('Decision:', str(decision_path), flush=True)

    del ns, db, detail, detail_raw
    gc.collect()



if __name__ == '__main__':
    main_v46()
