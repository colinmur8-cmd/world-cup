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


@st.cache_resource(show_spinner=False)
def get_stat_models(dec: float):
    from predictions.wc2026 import load_stat_models
    return load_stat_models(decay=dec, verbose=False)


# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_winner, tab_groups, tab_match, tab_backtest, tab_matchbt = st.tabs([
    "🏆 Tournament Winner",
    "📊 Group Stage",
    "🆚 Match Predictor",
    "📈 Winner Backtest",
    "🔬 Match Backtest",
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

        # Value bets + Kelly staking
        if result["value_bets"]:
            st.divider()
            bankroll = st.number_input("Bankroll (£)", min_value=10, value=1000, step=50,
                                       key="bankroll_match")
            st.markdown("**🟢 Value Bets Detected**")
            vb_rows = [{"Market": v.market, "Selection": v.selection,
                        "Model": f"{v.model_prob*100:.1f}%",
                        "Implied": f"{v.implied_prob*100:.1f}%",
                        "Edge": f"{v.edge*100:+.1f}pp",
                        "Odds": v.bk_odds,
                        "Half-Kelly stake": f"£{v.half_kelly * bankroll:.2f}",
                        "Full-Kelly stake": f"£{v.kelly_fraction * bankroll:.2f}"}
                       for v in result["value_bets"]]
            st.dataframe(pd.DataFrame(vb_rows), hide_index=True, use_container_width=True)
            st.caption("Half-Kelly is recommended — full Kelly is theoretically optimal but very aggressive.")

        # ── Stat markets: corners, cards, shots ──
        st.divider()
        st.markdown("#### Ancillary Markets")
        with st.spinner("Loading corner / card / shots models…"):
            stat_models = get_stat_models(decay)

        if not stat_models:
            st.info(
                "Stat models not loaded yet. Click **Build Stat Models** to download StatsBomb data "
                "(one-time, ~2 min) and unlock corners, cards, and shots markets."
            )
            if st.button("📥 Build Stat Models", key="btn_build_stats"):
                with st.spinner("Downloading StatsBomb event data for WC 2018/2022, Copa, AFCON…"):
                    from data.statsbomb_loader import load_statsbomb_match_stats
                    load_statsbomb_match_stats(refresh=True, verbose=False)
                st.cache_resource.clear()
                st.rerun()
        else:
            from model.markets import stat_markets
            smkt = stat_markets(stat_models, home_team, away_team)

            col_cn, col_cd, col_sh = st.columns(3)

            with col_cn:
                st.markdown("**Corners**")
                if "corners" in smkt:
                    c = smkt["corners"]
                    st.caption(f"Expected: {home_team} {c['home_expected']} — {away_team} {c['away_expected']} (total {c['total_expected']})")
                    cn_rows = [
                        {"Line": f"O/U {line}", "Over": f"{v['over']*100:.1f}%", "Under": f"{v['under']*100:.1f}%"}
                        for line, v in c["over_under"].items()
                    ]
                    st.dataframe(pd.DataFrame(cn_rows), hide_index=True, use_container_width=True)
                else:
                    st.info("No corner data")

            with col_cd:
                st.markdown("**Yellow Cards**")
                if "yellow_cards" in smkt:
                    cd = smkt["yellow_cards"]
                    st.caption(f"Expected: {home_team} {cd['home_expected']} — {away_team} {cd['away_expected']} (total {cd['total_expected']})")
                    cd_rows = [
                        {"Line": f"O/U {line}", "Over": f"{v['over']*100:.1f}%", "Under": f"{v['under']*100:.1f}%"}
                        for line, v in cd["over_under"].items()
                    ]
                    st.dataframe(pd.DataFrame(cd_rows), hide_index=True, use_container_width=True)
                else:
                    st.info("No card data")

            with col_sh:
                st.markdown("**Shots on Target**")
                if "shots_on_target" in smkt:
                    sh = smkt["shots_on_target"]
                    st.caption(f"Expected: {home_team} {sh['home_expected']} — {away_team} {sh['away_expected']} (total {sh['total_expected']})")
                    sh_rows = [
                        {"Line": f"O/U {line}", "Over": f"{v['over']*100:.1f}%", "Under": f"{v['under']*100:.1f}%"}
                        for line, v in sh["over_under"].items()
                    ]
                    st.dataframe(pd.DataFrame(sh_rows), hide_index=True, use_container_width=True)
                else:
                    st.info("No shots data")

            if stat_models:
                st.caption(
                    "⚠️ Stat models trained on StatsBomb free data (~200 international matches). "
                    "Add API-Football ($19/mo) for WC qualifier data and significantly better accuracy."
                )

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


# ═══════════════════════════════════════════════════════════
# TAB 5 — Match-level Backtest
# ═══════════════════════════════════════════════════════════
with tab_matchbt:
    st.subheader("Match-Level Backtest — 2018 & 2022 WC")
    st.write(
        "Trains on pre-tournament international history, predicts every WC match, "
        "then checks accuracy, calibration, and simulated P&L."
    )
    st.caption(
        "⚠️ Bookmaker odds are **simulated** (6% margin over model fair odds) — real historical "
        "odds aren't available. This shows calibration and edge detection quality, not guaranteed profit."
    )

    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        bt_margin   = st.slider("Simulated bookie margin", 0.02, 0.12, 0.06, 0.01,
                                 key="bt_margin", help="Typical sportsbook overround 4–8%")
        bt_min_edge = st.slider("Min edge to bet", 0.01, 0.10, 0.03, 0.01, key="bt_edge")
    with col_bt2:
        bt_bankroll = st.number_input("Starting bankroll (£)", 100, 100_000, 1000, 100,
                                       key="bt_bankroll")
        bt_kelly    = st.slider("Kelly fraction", 0.1, 1.0, 0.5, 0.1, key="bt_kelly",
                                help="0.5 = half-Kelly (recommended)")

    if st.button("▶  Run Match Backtest", type="primary", key="btn_matchbt"):
        with st.spinner("Loading pre-tournament histories and predicting all 128 matches…"):
            from backtest.match_backtest import (
                run_match_backtest, summary_metrics, calibration_table, pnl_simulation
            )
            preds = run_match_backtest(years=("2018", "2022"), verbose=False)

        metrics = summary_metrics(preds)

        # ── Headline metrics ──
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Matches",         metrics["matches"])
        m2.metric("Result accuracy", f"{metrics['result_acc']*100:.1f}%",
                  help="Picking correct 1X2 outcome. Naive baseline ~45%.")
        m3.metric("Brier score",     f"{metrics['brier_1x2']:.4f}",
                  help="Lower is better. Random guessing = 0.222")
        m4.metric("BTTS accuracy",   f"{metrics['btts_acc']*100:.1f}%")
        m5.metric("O/U 2.5 acc",     f"{metrics['ou25_acc']*100:.1f}%")

        st.divider()

        # ── Match-by-match table ──
        st.markdown("#### All Predictions vs Actuals")
        match_rows = []
        for p in preds:
            match_rows.append({
                "Year":   p.year,
                "Stage":  p.stage,
                "Home":   p.home,
                "Away":   p.away,
                "Score":  f"{p.actual_hg}–{p.actual_ag}",
                "xG":     f"{p.xg_home:.2f}–{p.xg_away:.2f}",
                "P(H)":   f"{p.p_home*100:.0f}%",
                "P(D)":   f"{p.p_draw*100:.0f}%",
                "P(A)":   f"{p.p_away*100:.0f}%",
                "Pred":   p.predicted_result.upper()[0],
                "Actual": p.actual_result.upper()[0],
                "✓":      "✓" if p.correct_result else "✗",
                "BTTS":   "✓" if p.btts_actual else "✗",
                "O2.5":   "✓" if p.over25_actual else "✗",
            })

        df_matches = pd.DataFrame(match_rows)

        def _color_correct(row):
            color = "#d4edda" if row["✓"] == "✓" else "#f8d7da"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            df_matches.style.apply(_color_correct, axis=1),
            use_container_width=True, hide_index=True, height=400,
        )

        st.divider()

        # ── Calibration ──
        st.markdown("#### Calibration — 1X2 Predictions")
        st.write("When the model says 70%, does it happen 70% of the time? Perfect model = diagonal line.")
        cal = calibration_table(preds)

        cal_chart = cal.set_index("Predicted (mid)")[["Actual freq"]].copy()
        cal_chart.index.name = "Predicted probability"

        col_cal1, col_cal2 = st.columns([2, 1])
        with col_cal1:
            st.line_chart(cal_chart, use_container_width=True)
        with col_cal2:
            st.dataframe(cal, hide_index=True, use_container_width=True)

        st.divider()

        # ── P&L simulation ──
        st.markdown("#### Simulated P&L (synthetic bookie odds)")
        pnl = pnl_simulation(
            preds,
            margin=bt_margin,
            min_edge=bt_min_edge,
            kelly_fraction=bt_kelly,
            flat_stake=bt_bankroll / 100,
            bankroll=bt_bankroll,
        )

        if pnl.empty:
            st.warning(f"No bets found with ≥{bt_min_edge*100:.0f}pp edge at {bt_margin*100:.0f}% margin. "
                       "Lower the min edge or margin.")
        else:
            n_bets = len(pnl)
            wins   = (pnl["Won"] == "✓").sum()
            final_flat  = pnl["Flat Balance"].iloc[-1]
            final_kelly = pnl["Kelly Balance"].iloc[-1]

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Bets placed",       n_bets)
            p2.metric("Win rate",          f"{wins/n_bets*100:.1f}%")
            p3.metric("Flat final balance",
                      f"£{final_flat:.0f}",
                      f"{(final_flat - bt_bankroll)/bt_bankroll*100:+.1f}%")
            p4.metric("Kelly final balance",
                      f"£{final_kelly:.0f}",
                      f"{(final_kelly - bt_bankroll)/bt_bankroll*100:+.1f}%")

            # Equity curve
            eq = pnl[["Flat Balance", "Kelly Balance"]].reset_index(drop=True)
            eq.index.name = "Bet #"
            st.line_chart(eq, use_container_width=True)

            st.markdown("**All bets placed**")
            st.dataframe(
                pnl[["Year","Stage","Match","Selection","Model %","Edge","Bk Odds","Won",
                      "Flat P&L","Flat Balance","Kelly Stake","Kelly P&L","Kelly Balance"]],
                use_container_width=True, hide_index=True,
            )

    else:
        st.info("Press **Run Match Backtest** to see prediction accuracy, calibration, and simulated P&L.")
