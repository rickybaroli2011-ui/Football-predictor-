import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="Calcio Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Calcio Predictor")
st.caption("Previsione risultati e gol basata su machine learning e dati storici reali, su 5 campionati europei.")

LEGHE = {
    "Serie A (Italia)": "I1",
    "Premier League (Inghilterra)": "E0",
    "La Liga (Spagna)": "SP1",
    "Bundesliga (Germania)": "D1",
    "Ligue 1 (Francia)": "F1"
}

lega_label = st.selectbox("Campionato", list(LEGHE.keys()))
codice_lega = LEGHE[lega_label]

@st.cache_data(ttl=3600)
def carica_dati(codice_lega):
    stagioni = ['1920', '2021', '2122', '2223', '2324']
    lista_df = []
    for s in stagioni:
        url = f"https://www.football-data.co.uk/mmz4281/{s}/{codice_lega}.csv"
        try:
            temp = pd.read_csv(url)
            lista_df.append(temp)
        except Exception:
            continue
    df = pd.concat(lista_df, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date', 'FTHG', 'FTAG', 'FTR'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

@st.cache_resource(ttl=3600)
def costruisci_modello_e_stato(codice_lega):
    df = carica_dati(codice_lega)

    storico_gol = {}
    storico_forma = {}
    storico_coppie = {}

    home_goals_avg, home_conceded_avg = [], []
    away_goals_avg, away_conceded_avg = [], []
    home_form, away_form = [], []
    h2h_home_adv = []

    for _, riga in df.iterrows():
        home, away, ftr = riga['HomeTeam'], riga['AwayTeam'], riga['FTR']

        for squadra in [home, away]:
            if squadra not in storico_gol:
                storico_gol[squadra] = {'fatti': [], 'subiti': []}
            if squadra not in storico_forma:
                storico_forma[squadra] = []

        prec_h_fatti = storico_gol[home]['fatti'][-5:]
        prec_h_subiti = storico_gol[home]['subiti'][-5:]
        prec_a_fatti = storico_gol[away]['fatti'][-5:]
        prec_a_subiti = storico_gol[away]['subiti'][-5:]

        home_goals_avg.append(sum(prec_h_fatti)/len(prec_h_fatti) if prec_h_fatti else 1.3)
        home_conceded_avg.append(sum(prec_h_subiti)/len(prec_h_subiti) if prec_h_subiti else 1.3)
        away_goals_avg.append(sum(prec_a_fatti)/len(prec_a_fatti) if prec_a_fatti else 1.3)
        away_conceded_avg.append(sum(prec_a_subiti)/len(prec_a_subiti) if prec_a_subiti else 1.3)

        prec_h_forma = storico_forma[home][-5:]
        prec_a_forma = storico_forma[away][-5:]
        home_form.append(sum(prec_h_forma)/len(prec_h_forma) if prec_h_forma else 1.0)
        away_form.append(sum(prec_a_forma)/len(prec_a_forma) if prec_a_forma else 1.0)

        chiave = tuple(sorted([home, away]))
        precedenti_h2h = storico_coppie.get(chiave, [])[-3:]
        if precedenti_h2h:
            punti_h2h = []
            for (h_prec, ftr_prec) in precedenti_h2h:
                if h_prec == home:
                    punti_h2h.append(3 if ftr_prec == 'H' else (1 if ftr_prec == 'D' else 0))
                else:
                    punti_h2h.append(3 if ftr_prec == 'A' else (1 if ftr_prec == 'D' else 0))
            h2h_home_adv.append(sum(punti_h2h)/len(punti_h2h))
        else:
            h2h_home_adv.append(1.0)

        storico_gol[home]['fatti'].append(riga['FTHG'])
        storico_gol[home]['subiti'].append(riga['FTAG'])
        storico_gol[away]['fatti'].append(riga['FTAG'])
        storico_gol[away]['subiti'].append(riga['FTHG'])

        punti_home = 3 if ftr == 'H' else (1 if ftr == 'D' else 0)
        punti_away = 3 if ftr == 'A' else (1 if ftr == 'D' else 0)
        storico_forma[home].append(punti_home)
        storico_forma[away].append(punti_away)

        storico_coppie.setdefault(chiave, []).append((home, ftr))

    df['HomeGoalsAvg'] = home_goals_avg
    df['HomeConcededAvg'] = home_conceded_avg
    df['AwayGoalsAvg'] = away_goals_avg
    df['AwayConcededAvg'] = away_conceded_avg
    df['HomeFormPoints'] = home_form
    df['AwayFormPoints'] = away_form
    df['GoalDiffAdvantage'] = (df['HomeGoalsAvg'] - df['HomeConcededAvg']) - (df['AwayGoalsAvg'] - df['AwayConcededAvg'])
    df['FormDiff'] = df['HomeFormPoints'] - df['AwayFormPoints']
    df['H2H_HomeAdvantage'] = h2h_home_adv

    features = ['HomeGoalsAvg', 'HomeConcededAvg', 'AwayGoalsAvg', 'AwayConcededAvg',
                'HomeFormPoints', 'AwayFormPoints', 'GoalDiffAdvantage', 'FormDiff', 'H2H_HomeAdvantage']

    # Modello 1: chi vince (classificazione)
    modello_risultato = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=6, class_weight='balanced')
    modello_risultato.fit(df[features], df['FTR'])

    # Modello 2 e 3: quanti gol segna ciascuna squadra (regressione, numero continuo)
    modello_gol_casa = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6)
    modello_gol_casa.fit(df[features], df['FTHG'])

    modello_gol_trasferta = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6)
    modello_gol_trasferta.fit(df[features], df['FTAG'])

    stato_squadre = {}
    for squadra in storico_gol:
        fatti = storico_gol[squadra]['fatti'][-5:]
        subiti = storico_gol[squadra]['subiti'][-5:]
        forma = storico_forma[squadra][-5:]
        stato_squadre[squadra] = {
            'GoalsAvg': sum(fatti)/len(fatti) if fatti else 1.3,
            'ConcededAvg': sum(subiti)/len(subiti) if subiti else 1.3,
            'FormPoints': sum(forma)/len(forma) if forma else 1.0
        }

    return modello_risultato, modello_gol_casa, modello_gol_trasferta, stato_squadre, storico_coppie, features, sorted(storico_gol.keys())

with st.spinner(f"Caricamento dati {lega_label}..."):
    modello_risultato, modello_gol_casa, modello_gol_trasferta, stato_squadre, storico_coppie, features, squadre_disponibili = costruisci_modello_e_stato(codice_lega)

st.divider()
col1, col2 = st.columns(2)
with col1:
    squadra_casa = st.selectbox("Squadra Casa", squadre_disponibili, index=0)
with col2:
    idx_away = 1 if len(squadre_disponibili) > 1 else 0
    squadra_trasferta = st.selectbox("Squadra Trasferta", squadre_disponibili, index=idx_away)

if st.button("🔮 Prevedi risultato", use_container_width=True, type="primary"):
    if squadra_casa == squadra_trasferta:
        st.warning("Scegli due squadre diverse.")
    else:
        s_casa = stato_squadre[squadra_casa]
        s_trasferta = stato_squadre[squadra_trasferta]

        chiave = tuple(sorted([squadra_casa, squadra_trasferta]))
        precedenti_h2h = storico_coppie.get(chiave, [])[-3:]
        if precedenti_h2h:
            punti_h2h = []
            for (h_prec, ftr_prec) in precedenti_h2h:
                if h_prec == squadra_casa:
                    punti_h2h.append(3 if ftr_prec == 'H' else (1 if ftr_prec == 'D' else 0))
                else:
                    punti_h2h.append(3 if ftr_prec == 'A' else (1 if ftr_prec == 'D' else 0))
            h2h_val = sum(punti_h2h)/len(punti_h2h)
        else:
            h2h_val = 1.0

        input_dati = pd.DataFrame([{
            'HomeGoalsAvg': s_casa['GoalsAvg'],
            'HomeConcededAvg': s_casa['ConcededAvg'],
            'AwayGoalsAvg': s_trasferta['GoalsAvg'],
            'AwayConcededAvg': s_trasferta['ConcededAvg'],
            'HomeFormPoints': s_casa['FormPoints'],
            'AwayFormPoints': s_trasferta['FormPoints'],
            'GoalDiffAdvantage': (s_casa['GoalsAvg'] - s_casa['ConcededAvg']) - (s_trasferta['GoalsAvg'] - s_trasferta['ConcededAvg']),
            'FormDiff': s_casa['FormPoints'] - s_trasferta['FormPoints'],
            'H2H_HomeAdvantage': h2h_val
        }])[features]

        probabilita = modello_risultato.predict_proba(input_dati)[0]
        classi = modello_risultato.classes_
        prob_dict = dict(zip(classi, probabilita))

        gol_previsti_casa = max(0, modello_gol_casa.predict(input_dati)[0])
        gol_previsti_trasferta = max(0, modello_gol_trasferta.predict(input_dati)[0])

        st.divider()
        st.subheader(f"{squadra_casa} vs {squadra_trasferta}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"🏠 Vittoria {squadra_casa}", f"{prob_dict.get('H', 0):.0%}")
        with c2:
            st.metric("🤝 Pareggio", f"{prob_dict.get('D', 0):.0%}")
        with c3:
            st.metric(f"✈️ Vittoria {squadra_trasferta}", f"{prob_dict.get('A', 0):.0%}")

        st.divider()
        st.subheader("⚽ Previsione gol")

        g1, g2 = st.columns(2)
        with g1:
            st.metric(f"Gol previsti {squadra_casa}", f"{gol_previsti_casa:.1f}")
        with g2:
            st.metric(f"Gol previsti {squadra_trasferta}", f"{gol_previsti_trasferta:.1f}")

        risultato_probabile = f"{round(gol_previsti_casa)}-{round(gol_previsti_trasferta)}"
        st.info(f"Risultato più probabile secondo il modello: **{risultato_probabile}**")

        st.caption("Previsione basata su forma recente, media gol e scontri diretti storici. I numeri di gol sono medie statistiche (es. 1.8 significa 'quasi 2 gol in media'), non predizioni esatte garantite.")

st.divider()
st.caption("Creato con scikit-learn e Streamlit • Dati storici da football-data.co.uk")
