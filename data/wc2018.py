"""
2018 FIFA World Cup (Russia) — verified match results.
Format: (stage, date, home_team, away_team, home_goals, away_goals)
All matches played on neutral ground; 'home_team' = first-named side.
"""

MATCHES_2018 = [
    # ── GROUP A ──────────────────────────────────────────────────────────────
    ("Group A", "2018-06-14", "Russia",       "Saudi Arabia", 5, 0),
    ("Group A", "2018-06-15", "Egypt",         "Uruguay",      0, 1),
    ("Group A", "2018-06-19", "Russia",        "Egypt",        3, 1),
    ("Group A", "2018-06-20", "Uruguay",       "Saudi Arabia", 1, 0),
    ("Group A", "2018-06-25", "Uruguay",       "Russia",       3, 0),
    ("Group A", "2018-06-25", "Saudi Arabia",  "Egypt",        2, 1),

    # ── GROUP B ──────────────────────────────────────────────────────────────
    ("Group B", "2018-06-15", "Morocco",       "Iran",         0, 1),
    ("Group B", "2018-06-15", "Portugal",      "Spain",        3, 3),
    ("Group B", "2018-06-20", "Portugal",      "Morocco",      1, 0),
    ("Group B", "2018-06-20", "Iran",          "Spain",        0, 1),
    ("Group B", "2018-06-25", "Iran",          "Portugal",     1, 1),
    ("Group B", "2018-06-25", "Spain",         "Morocco",      2, 2),

    # ── GROUP C ──────────────────────────────────────────────────────────────
    ("Group C", "2018-06-16", "France",        "Australia",    2, 1),
    ("Group C", "2018-06-16", "Peru",          "Denmark",      0, 1),
    ("Group C", "2018-06-21", "Denmark",       "Australia",    1, 1),
    ("Group C", "2018-06-21", "France",        "Peru",         1, 0),
    ("Group C", "2018-06-26", "Australia",     "Peru",         0, 2),
    ("Group C", "2018-06-26", "Denmark",       "France",       0, 0),

    # ── GROUP D ──────────────────────────────────────────────────────────────
    ("Group D", "2018-06-16", "Argentina",     "Iceland",      1, 1),
    ("Group D", "2018-06-16", "Croatia",       "Nigeria",      2, 0),
    ("Group D", "2018-06-21", "Argentina",     "Croatia",      0, 3),
    ("Group D", "2018-06-22", "Nigeria",       "Iceland",      2, 0),
    ("Group D", "2018-06-26", "Iceland",       "Croatia",      1, 2),
    ("Group D", "2018-06-26", "Nigeria",       "Argentina",    1, 2),

    # ── GROUP E ──────────────────────────────────────────────────────────────
    ("Group E", "2018-06-17", "Costa Rica",    "Serbia",       0, 1),
    ("Group E", "2018-06-17", "Brazil",        "Switzerland",  1, 1),
    ("Group E", "2018-06-22", "Brazil",        "Costa Rica",   2, 0),
    ("Group E", "2018-06-22", "Serbia",        "Switzerland",  1, 2),
    ("Group E", "2018-06-27", "Serbia",        "Brazil",       0, 2),
    ("Group E", "2018-06-27", "Switzerland",   "Costa Rica",   2, 2),

    # ── GROUP F ──────────────────────────────────────────────────────────────
    ("Group F", "2018-06-17", "Germany",       "Mexico",       0, 1),
    ("Group F", "2018-06-18", "Sweden",        "South Korea",  1, 0),
    ("Group F", "2018-06-23", "South Korea",   "Mexico",       1, 2),
    ("Group F", "2018-06-23", "Germany",       "Sweden",       2, 1),
    ("Group F", "2018-06-27", "Germany",       "South Korea",  0, 2),
    ("Group F", "2018-06-27", "Mexico",        "Sweden",       0, 3),

    # ── GROUP G ──────────────────────────────────────────────────────────────
    ("Group G", "2018-06-18", "Belgium",       "Panama",       3, 0),
    ("Group G", "2018-06-18", "Tunisia",       "England",      1, 2),
    ("Group G", "2018-06-23", "Belgium",       "Tunisia",      5, 2),
    ("Group G", "2018-06-24", "England",       "Panama",       6, 1),
    ("Group G", "2018-06-28", "England",       "Belgium",      0, 1),
    ("Group G", "2018-06-28", "Panama",        "Tunisia",      1, 2),

    # ── GROUP H ──────────────────────────────────────────────────────────────
    ("Group H", "2018-06-19", "Colombia",      "Japan",        1, 2),
    ("Group H", "2018-06-19", "Poland",        "Senegal",      1, 2),
    ("Group H", "2018-06-24", "Japan",         "Senegal",      2, 2),
    ("Group H", "2018-06-24", "Poland",        "Colombia",     0, 3),
    ("Group H", "2018-06-28", "Japan",         "Poland",       0, 1),
    ("Group H", "2018-06-28", "Senegal",       "Colombia",     0, 1),

    # ── ROUND OF 16 ──────────────────────────────────────────────────────────
    ("R16",     "2018-06-30", "France",        "Argentina",    4, 3),
    ("R16",     "2018-06-30", "Uruguay",       "Portugal",     2, 1),
    ("R16",     "2018-07-01", "Spain",         "Russia",       1, 1),   # Russia won pens
    ("R16",     "2018-07-01", "Croatia",       "Denmark",      1, 1),   # Croatia won pens
    ("R16",     "2018-07-02", "Brazil",        "Mexico",       2, 0),
    ("R16",     "2018-07-02", "Belgium",       "Japan",        3, 2),
    ("R16",     "2018-07-03", "Sweden",        "Switzerland",  1, 0),
    ("R16",     "2018-07-03", "Colombia",      "England",      1, 1),   # England won pens

    # ── QUARTER-FINALS ───────────────────────────────────────────────────────
    ("QF",      "2018-07-06", "Uruguay",       "France",       0, 2),
    ("QF",      "2018-07-06", "Brazil",        "Belgium",      1, 2),
    ("QF",      "2018-07-07", "Sweden",        "England",      0, 2),
    ("QF",      "2018-07-07", "Russia",        "Croatia",      2, 2),   # Croatia won pens

    # ── SEMI-FINALS ──────────────────────────────────────────────────────────
    ("SF",      "2018-07-10", "France",        "Belgium",      1, 0),
    ("SF",      "2018-07-11", "Croatia",       "England",      2, 1),

    # ── THIRD PLACE / FINAL ──────────────────────────────────────────────────
    ("3rd",     "2018-07-14", "Belgium",       "England",      2, 0),
    ("Final",   "2018-07-15", "France",        "Croatia",      4, 2),
]
