import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Calcio Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Calcio Predictor")
st.caption("Previsione risultati Serie A basata su machine learning e dati storici reali.")

@st.cache_data(ttl=3600)
def carica_dati():
    stagioni = ['1920', '2021', '2122', '2223', '2324']
    lista_df = []
    for s in stagioni:
        url = f"https://www.football-data.co.uk/mmz4281/{s}/I1.csv"
        temp = pd.read_csv(url)
        lista_df.append(temp)
    df = pd.concat(lista_df, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    return df

@st.cache_resource(ttl=3600)
def costruisci_modello_e_stato():
    df = carica_dati()

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

        # Medie gol PRE-partita
        prec_h_fatti = storico_gol[home]['fatti'][-5:]
        prec_h_subiti = storico_gol[home]['subiti'][-5:]
        prec_a_fatti = storico_gol[away]['fatti'][-5:]
        prec_a_subiti = storico_gol[away]['subiti'][-5:]

        home_goals_avg.append(sum(prec_h_fatti)/len(prec_h_fatti) if prec_h_fatti else 1.3)
        home_conceded_avg.append(sum(prec_h_subiti)/len(prec_h_subiti) if prec_h_subiti else 1.3)
        away_goals_avg.append(sum(prec_a_fatti)/len(prec_a_fatti) if prec_a_fatti else 1.3)
        away_conceded_avg.append(sum(prec_a_subiti)/len(prec_a_subiti) if prec_a_subiti else 1.3)

        # Forma PRE-partita
        prec_h_forma = storico_forma[home][-5:]
        prec_a_forma = storico_forma[away][-5:]
        home_form.append(sum(prec_h_forma)/len(prec_h_forma) if prec_h_forma else 1.0)
        away_form.append(sum(prec_a_forma)/len(prec_a_forma) if prec_a_forma else 1.0)

        # H2H PRE-partita
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

        # Aggiorna storici DOPO aver calcolato le feature
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

    modello = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=6, class_weight='balanced')
    modello.fit(df[features], df['FTR'])

    # Stato attuale di ogni squadra (per previsioni future)
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

    return modello, stato_squadre, storico_coppie, features, sorted(storico_gol.keys())

modello, stato_squadre, storico_coppie, features, squadre_disponibili = costruisci_modello_e_stato()

st.divider()
col1, col2 = st.columns(2)
with col1:
    squadra_casa = st.selectbox("Squadra Casa", squadre_disponibili, index=squadre_disponibili.index("Inter") if "Inter" in squadre_disponibili else 0)
with col2:
    squadra_trasferta = st.selectbox("Squadra Trasferta", squadre_disponibili, index=squadre_disponibili.index("Milan") if "Milan" in squadre_disponibili else 1)

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

        probabilita = modello.predict_proba(input_dati)[0]
        classi = modello.classes_

        prob_dict = dict(zip(classi, probabilita))

        st.divider()
        st.subheader(f"{squadra_casa} vs {squadra_trasferta}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"🏠 Vittoria {squadra_casa}", f"{prob_dict.get('H', 0):.0%}")
        with c2:
            st.metric("🤝 Pareggio", f"{prob_dict.get('D', 0):.0%}")
        with c3:
            st.metric(f"✈️ Vittoria {squadra_trasferta}", f"{prob_dict.get('A', 0):.0%}")

        st.caption("Previsione basata su forma recente, media gol e scontri diretti storici. Nessun modello di ML predice il calcio con certezza — trattalo come un'indicazione statistica, non una garanzia.")

st.divider()
st.caption("Creato con scikit-learn e Streamlit • Dati storici da football-data.co.uk")
