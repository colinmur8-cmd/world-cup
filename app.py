"""
WC 2026 Prediction App — Streamlit UI

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import time

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ World Cup 2026 — Prediction Model")
st.caption("Dixon-Coles model trained on international results 2016–2026 with time decay")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    odds_key = st.text_input(
        "The Odds API key (optional)",
        type="password",
        help="Get a free key at the-odds-api.com — enables live value bet detection",
    )

    n_sims = st.select_slider(
        "Simulations",
        options=[5_000, 10_000, 25_000, 50_000],
        value=25_000,
        help="More sims = more accurate, but slower",
    )

    use_history = st.checkbox(
        "Full international history",
        value=True,
        help="Train on 6,000+ competitive internationals (2016–2026). Uncheck for fast WC-only model.",
    )

    decay = st.slider(
        "Time decay (per year)",
        min_value=0.0,
        max_value=0.8,
        value=0.3,
        step=0.1,
        help="Higher = recent results count more. 0.3 ≈ match 3 years ago worth 40% of today's.",
    )

    st.divider()
    st.markdown("**Quick match lookup**")
    from predictions.wc2026 import WC2026_GROUPS
    all_teams = sorted({t for grp in WC2026_GROUPS.values() for t in grp})
    home_team = st.selectbox("Home team", all_teams, index=all_teams.index("France"))
    away_team = st.selectbox("Away team", all_teams, index=all_teams.index("England"))

# ── Model loader (cached) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model(use_hist: bool, dec: float):
    from predictions.wc2026 import load_trained_model
    return load_trained_model(use_history=use_hist, decay=dec)


# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_winner, tab_groups, tab_match, tab_backtest = st.tabs([
    "🏆 Tournament Winner",
    "📊 Group Stage",
    "🆚 Match Predictor",
    "📈 Backtest",
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Tournament Winner
# ═══════════════════════════════════════════════════════════
with tab_winner:
    st.subheader("2026 World Cup — Win Probabilities")
    st.write(f"Monte Carlo simulation of the full tournament ({n_sims:,} runs)")

    if st.button("▶  Run Tournament Simulation", type="primary", key="btn_winner"):
        with st.spinner("Loading model…"):
            model = get_model(use_history, decay)

        from predictions.wc2026 import WC2026_GROUPS, R32_BRACKET
        from model.simulate_tournament import simulate_tournament

        prog = st.progress(0, text="Simulating…")
        t0 = time.time()

        probs = simulate_tournament(
            WC2026_GROUPS,
            model,
            n_sims=n_sims,
            seed=42,
            format32=False,
            r32_bracket=R32_BRACKET,
        )

        elapsed = time.time() - t0
        prog.empty()
        st.success(f"Done in {elapsed:.1f}s")

        # Build results table
        rows = []
        for rank, (team, prob) in enumerate(probs.items(), 1):
            if prob == 0:
                continue
            rows.append({
                "Rank": rank,
                "Team": team,
                "Win %": f"{prob*100:.1f}%",
                "Fair Odds": f"{1/max(prob,1e-9):.1f}x",
                "_prob": prob,
            })

        df = pd.DataFrame(rows)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(
                df.drop(columns="_prob"),
                use_container_width=True,
                hide_index=True,
            )

        with col2:
            top10 = df.head(10)
            st.bar_chart(
                top10.set_index("Team")["_prob"].rename("Win probability"),
                horizontal=True,
            )

    else:
        st.info("Press **Run Tournament Simulation** to generate predictions.")


# ═══════════════════════════════════════════════════════════
# TAB 2 — Group Stage
# ═══════════════════════════════════════════════════════════
with tab_groups:
    st.subheader("Group Stage Predictions")

    if st.button("▶  Generate Group Predictions", type="primary", key="btn_groups"):
        with st.spinner("Loading model and generating predictions…"):
            model = get_model(use_history, decay)

            live_odds = None
            if odds_key:
                try:
                    from data.odds_fetcher import fetch_wc_odds
                    live_odds = fetch_wc_odds(odds_key)
                    st.success(f"Live odds loaded — {len(live_odds)} matches")
                except Exception as e:
                    st.warning(f"Could not load odds: {e}")

            from predictions.wc2026 import scan_group_stage
            df_all = scan_group_stage(model, bookmaker_odds=live_odds)

        from predictions.wc2026 import WC2026_GROUPS
        groups = sorted(df_all["Group"].unique())

        cols = st.columns(2)
        for i, grp in enumerate(groups):
            with cols[i % 2]:
                st.markdown(f"**Group {grp}** — {', '.join(WC2026_GROUPS[grp])}")
                gdf = df_all[df_all["Group"] == grp].drop(columns="Group").reset_index(drop=True)

                # Highlight value bets
                def highlight_value(row):
                    if row.get("Value Bets", "—") not in ("—", ""):
                        return ["background-color: #d4edda"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    gdf.style.apply(highlight_value, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("Press **Generate Group Predictions** to scan all 66 group fixtures.")


# ═══════════════════════════════════════════════════════════
# TAB 3 — Match Predictor
# ═══════════════════════════════════════════════════════════
with tab_match:
    st.subheader(f"{home_team}  vs  {away_team}")
    st.caption("Select teams in the sidebar")

    if st.button("▶  Predict Match", type="primary", key="btn_match"):
        with st.spinner("Running…"):
            model = get_model(use_history, decay)

            bk_odds = None
            if odds_key:
                try:
                    from data.odds_fetcher import fetch_wc_odds
                    live = fetch_wc_odds(odds_key)
                    bk_odds = live.get((home_team, away_team),
                                       live.get((away_team, home_team), None))
                except Exception:
                    pass

            from predictions.wc2026 import predict_match
            result = predict_match(home_team, away_team, model, bk_odds=bk_odds)
            mkt = result["markets"]

        # Expected goals
        lam = mkt["expected_goals"]["home"]
        mu  = mkt["expected_goals"]["away"]

        col1, col2, col3 = st.columns(3)
        col1.metric(f"{home_team} xG", f"{lam:.2f}")
        col2.metric("—", "vs")
        col3.metric(f"{away_team} xG", f"{mu:.2f}")

        st.divider()

        # 1X2
        mr = mkt["match_result"]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home_team} win", f"{mr['home']*100:.1f}%", f"Odds {1/mr['home']:.2f}")
        c2.metric("Draw",             f"{mr['draw']*100:.1f}%", f"Odds {1/mr['draw']:.2f}")
        c3.metric(f"{away_team} win", f"{mr['away']*100:.1f}%", f"Odds {1/mr['away']:.2f}")

        st.divider()

        col_ou, col_btts, col_cs = st.columns(3)

        with col_ou:
            st.markdown("**Over / Under**")
            ou_rows = []
            for line, vals in sorted(mkt["over_under"].items()):
                ou_rows.append({"Line": f"O/U {line}", "Over": f"{vals['over']*100:.1f}%", "Under": f"{vals['under']*100:.1f}%"})
            st.dataframe(pd.DataFrame(ou_rows), hide_index=True, use_container_width=True)

        with col_btts:
            st.markdown("**Both Teams to Score**")
            bt = mkt["btts"]
            st.metric("Yes", f"{bt['yes']*100:.1f}%", f"Odds {1/bt['yes']:.2f}")
            st.metric("No",  f"{bt['no']*100:.1f}%",  f"Odds {1/bt['no']:.2f}")

        with col_cs:
            st.markdown("**Top Scorelines**")
            cs_rows = [{"Score": s, "Prob": f"{p*100:.2f}%", "Odds": f"{1/p:.0f}x"}
                       for s, p in mkt["correct_score"][:8]]
            st.dataframe(pd.DataFrame(cs_rows), hide_index=True, use_container_width=True)

        # Value bets
        if result["value_bets"]:
            st.divider()
            st.markdown("**🟢 Value Bets Detected**")
            vb_rows = [{"Market": v.market, "Selection": v.selection,
                        "Model": f"{v.model_prob*100:.1f}%",
                        "Implied": f"{v.implied_prob*100:.1f}%",
                        "Edge": f"{v.edge*100:+.1f}pp",
                        "Odds": v.bk_odds}
                       for v in result["value_bets"]]
            st.dataframe(pd.DataFrame(vb_rows), hide_index=True, use_container_width=True)

    else:
        st.info("Select teams in the sidebar then press **Predict Match**.")


# ═══════════════════════════════════════════════════════════
# TAB 4 — Backtest
# ═══════════════════════════════════════════════════════════
with tab_backtest:
    st.subheader("Winner Prediction Backtest")
    st.write("Trains on pre-tournament data, simulates 2018 and 2022 WC, checks if actual winner was predicted.")

    sims_bt = st.select_slider("Simulations (backtest)", options=[5_000, 10_000, 25_000], value=10_000, key="sims_bt")

    if st.button("▶  Run Backtest", type="primary", key="btn_backtest"):
        with st.spinner("Running backtest (this takes ~2 minutes)…"):
            from backtest.winner import run_winner_backtest
            results = run_winner_backtest(n_sims=sims_bt, verbose=False)

        for res in results:
            with st.expander(f"**{res.year} WC** — Actual winner: {res.actual_winner}  |  Predicted rank: {res.winner_rank}/{len(res.win_probs)}", expanded=True):
                cols = st.columns(4)
                cols[0].metric("Predicted rank", f"{res.winner_rank} / {len(res.win_probs)}")
                cols[1].metric("Predicted prob", f"{res.winner_prob*100:.1f}%")
                cols[2].metric("Top-3?", "✓" if res.top_n_accuracy(3) else "✗")
                cols[3].metric("Top-5?", "✓" if res.top_n_accuracy(5) else "✓")

                top_rows = [{"Rank": i+1, "Team": t, "Win %": f"{p*100:.1f}%",
                             "": "← WINNER" if t == res.actual_winner else ""}
                            for i, (t, p) in enumerate(list(res.win_probs.items())[:12])]
                st.dataframe(pd.DataFrame(top_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Press **Run Backtest** to validate the model against 2018 and 2022.")
