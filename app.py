import streamlit as st

import altair as alt
from util import *
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title=" Tableau de bord Budget", layout="wide")


@st.cache_data
def load_data():
    file_path = r'budget_v2.Csv'
    data_set = pd.read_csv(file_path, sep=';', encoding='latin1')
    data = data_cleaning(data_set)
    return data

df = load_data()

st.markdown("<h2 style='text-align: center; color: navy;'> Tableau de bord de suivi budgétaire</h2>", unsafe_allow_html=True)

# ======== Filtres utilisateur ========
with st.sidebar:
    st.markdown("## 🔍 Filtres")

    # Filtre par Année
    # Vérifie que la colonne date est bien au format datetime
    #if not pd.api.types.is_datetime64_any_dtype(df['BUAP_CREE_DATE']):
        #df['BUAP_CREE_DATE'] = pd.to_datetime(df['BUAP_CREE_DATE'], errors='coerce')

    # Extraction de l'année
    if 'BUAP_CREE_DATE' in df.columns:
        df['ANNEE_CREE'] = pd.to_datetime(df['BUAP_CREE_DATE'], errors='coerce').dt.year
    else:
        st.error("⚠️ La colonne 'BUAP_CREE_DATE' est introuvable dans le fichier importé.")
        st.write("Colonnes disponibles :", df.columns.tolist())
        df['ANNEE_CREE'] = df['BUAP_CREE_DATE'].dt.year

    # Liste des années disponibles
    annees_disponibles = sorted(df['ANNEE_CREE'].dropna().unique())
    selected_annees = st.multiselect(
        "📆 Année(s) de création",
        options=annees_disponibles,
        default=annees_disponibles
    )

    # ======================
    # Filtre par Famille
    # ======================
    codes_familles = sorted(df['FAMB_CODE'].unique())
    selected_familles = st.multiselect(
        "Code(s) famille(s) de budget",
        options=codes_familles,
        default=codes_familles
    )

    # Affichage des libellés correspondants
    if selected_familles:
        libelles = (
            df[df['FAMB_CODE'].isin(selected_familles)][['FAMB_CODE', 'FAMB_LIBELLE']]
            .drop_duplicates()
            .sort_values('FAMB_CODE')
        )
       
    else:
        st.markdown("Aucune famille sélectionnée.")

    
try:
    df_filtered = df[
        (df['FAMB_CODE'].isin(selected_familles)) &
        (df['ANNEE_CREE'].isin(selected_annees))
    ]
except Exception as e:
    st.error(f"❌ Erreur dans les filtres : {e}")
    st.stop()



total_budget, total_engage, pourcentage_depassement = calculate_key_metrics(df_filtered)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Budget total alloué", f"{total_budget:,.2f} €")

with col2:
    st.metric("Montant total depensé", f"{total_engage:,.2f} €")

with col3:
    st.metric("% de budgets dépassés", f"{pourcentage_depassement:.1f} %")


# ======== Onglets (dans la page principale) ========
tab1, tab2 = st.tabs([" Vue d'ensemble", "Analyse détaillée"])

with tab1:
    st.markdown("### 💰 Détail budget")

    col1, col2 = st.columns(2)
    
    with col1:
        evolution_budget_alloue_engage(df_filtered)
    
    with col2:
        repartition_budget_par_famille_donut(df_filtered)
    
        
with tab2:
    st.header("🧮 Analyse détaillée")
    st.dataframe(df_filtered)






