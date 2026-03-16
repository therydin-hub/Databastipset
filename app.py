else:
        # ALLT RITAS UPP HÄR!
        input_compare = get_structural_vector(input_vec) if cb_structure else input_vec  # <-- LÄGG TILL DENNA RAD!
        visnings_kolumner = [c for c in ['Datum', 'ID_Omg'] if c in v_m.columns] + ['Payout', 'Sim']
        st.success(f"✅ Auto-laddade: **{filnamn}**. Exakt {len(v_m)} liknande omgångar hittades.")
