import streamlit as st
import altair as alt
from util import *
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title=" Tableau de bord Budget", layout="wide")


@st.cache_data
def load_data():
    file_path = r'test.csv'
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
    if not pd.api.types.is_datetime64_any_dtype(df['BUAP_DATE_DEB']):
        df['BUAP_DATE_DEB'] = pd.to_datetime(df['BUAP_DATE_DEB'], errors='coerce')

    # Extraction de l'année
    df['ANNEE_CREE'] = df['BUAP_DATE_DEB'].dt.year

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
    st.error(f" Erreur dans les filtres : {e}")
    st.stop()



depenses, budget_global, dep_total, budget_restant, taux_conso = prepare_budget_data(df_filtered)

# 3 KPI cards en colonnes
col1, col2, col3 = st.columns(3)

with col1:
        st.markdown(f"""
        <div style="text-align:center; font-size:18px; padding:10px; white-space: nowrap;">
            💰 Budget total<br><span style="font-size:26px; font-weight:bold">{budget_global:,.0f} €</span>
        </div>
        """, unsafe_allow_html=True)

with col2:
        st.markdown(f"""
        <div style="text-align:center; font-size:18px; padding:10px; white-space: nowrap;">
            📊 Dépenses engagées<br><span style="font-size:26px; font-weight:bold">{dep_total:,.0f} €</span>
        </div>
        """, unsafe_allow_html=True)

with col3:
        st.markdown(f"""
        <div style="text-align:center; font-size:18px; padding:10px; white-space: nowrap;">
            💸 Budget restant<br><span style="font-size:26px; font-weight:bold">{budget_restant:,.0f} €</span>
        </div>
        """, unsafe_allow_html=True)



# ======== Onglets (dans la page principale) ========
tab1, tab2 = st.tabs([" Vue d'ensemble", "Analyse détaillée"])


with tab1:
    st.markdown("### 💰 Détail budget")
    col1, col2,col3 = st.columns([3,2,2])
    
    with col1:
        fig , dfb= create_budget_evolution_figure(df_filtered)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig, summary = create_budget_summary_figure1(df_filtered)
        st.plotly_chart(fig, use_container_width=True)
        fig.update_layout(
        height=350,
        bargap=0.25,
        bargroupgap=0.1,
        margin=dict(l=60, r=40, t=60, b=50),
    )

        #st.subheader("Résumé budgétaire par année")
        #st.dataframe(
        #    summary.style.format({
        #        'Budget_alloué': "{:,.0f} €",
        #        'Budget_dépensé': "{:,.0f} €",
        #        '%_consommation': "{:.1f} %"
        #    }),
        #    use_container_width=True
#)
        

    with col3:
        # Taux de consommation sur toute la largeur, avec hauteur adaptée
        fig_taux = go.Figure(go.Indicator(
        mode="gauge+number",
        value=taux_conso,
        title={'text': "<b>Taux de consommation (%)</b>", 'font': {'size': 16}},  # ✅ simplifié
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#c2185b"},
            'steps': [
                {'range': [0, 50], 'color': "#e0f7fa"},
                {'range': [50, 75], 'color': "#fff3e0"},
                {'range': [75, 100], 'color': "#fce4ec"}
            ]
        }
    ))

        fig_taux.update_layout(
            height=300,
            margin=dict(l=20, r=30, t=20, b=10)
        )

        # Mettre une hauteur généreuse, par exemple 40-50% de la hauteur du viewport
        fig_taux.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=15))

        st.plotly_chart(fig_taux, use_container_width=True)
        #with col2:
            #repartition_budget_par_famille_donut(df_filtered)
        
        
    st.markdown("-----------------------------")
    col_a, col_b = st.columns([3,2])
    with col_a:
        df_summary = prepare_budget_summary(df_filtered)
        display_budget_summary(df_summary)
    with col_b:
        repartition_budget_par_famille_donut(df_filtered)
        
    
        
    
with tab2:
    st.header("🧮 Analyse détaillée")
    st.dataframe(df_filtered)


