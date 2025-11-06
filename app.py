import streamlit as st
import altair as alt
from util import *
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title=" Tableau de bord Budget", layout="wide")


@st.cache_data
def load_data():
    file_path = r'budget_v2.csv'
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
    if not pd.api.types.is_datetime64_any_dtype(df['BUAP_CREE_DATE']):
        df['BUAP_CREE_DATE'] = pd.to_datetime(df['BUAP_CREE_DATE'], errors='coerce')

    # Extraction de l'année
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


#total_budget, total_engage,taux_engagement, nb_supplier = calculate_key_metrics(df_filtered)

#col1, col2, col3, col4 = st.columns(4)

#with col1:
    #st.metric("Budget total alloué", f"{total_budget:,.2f} €")

#with col2:
    #st.metric("Montant total depensé", f"{total_engage:,.2f} €")

#with col3:
    #"st.metric("Taux engagement", f"{taux_engagement:.1f} %")

#with col4:
    #st.metric("Nb fournisseurs", f"{nb_supplier:}")


# ======== Onglets (dans la page principale) ========
tab1, tab2 = st.tabs([" Vue d'ensemble", "Analyse détaillée"])

with tab1:
    st.markdown("### 💰 Détail budget")

    col1, col2 = st.columns([2,1])
    
    with col1:
        fig = create_budget_evolution_figure(df_filtered)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
    
        # Taux de consommation sur toute la largeur, avec hauteur adaptée
        fig_taux = go.Figure(go.Indicator(
            mode="gauge+number",
            value=taux_conso,
            title={'text': "🔥 Taux de consommation (%)", 'font': {'size': 20}},
            gauge={'axis': {'range': [0, 100]},
                'bar': {'color': "#c2185b"},
                'steps': [
                    {'range': [0, 50], 'color': "#e0f7fa"},
                    {'range': [50, 75], 'color': "#fff3e0"},
                    {'range': [75, 100], 'color': "#fce4ec"}
                ]}
        ))
        # Mettre une hauteur généreuse, par exemple 40-50% de la hauteur du viewport
        fig_taux.update_layout(height=350, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_taux, use_container_width=True)
        #with col2:
            #repartition_budget_par_famille_donut(df_filtered)
    
    col_a, col_b = st.columns([2,3])
    with col_a:
        repartition_budget_par_famille_donut(df_filtered)
    with col_b:
        # Exemple d'utilisation
        df_final, color_func = prepare_budget_table(df_filtered)

        # Affichage dans Streamlit
        st.dataframe(
            df_final.style.applymap(color_func, subset=['Solde'])
                .format({
                    'Montant Initial': '{:,.0f} €',
                    'Montant Dépensé': '{:,.0f} €',
                    'Solde': '{:,.0f} €',
                    '% Consommé': '{} %'
                }),
            use_container_width=True
        )
    
        
    df_summary = prepare_budget_summary(df_filtered)
    display_budget_summary(df_summary)
with tab2:
    st.header("🧮 Analyse détaillée")
    st.dataframe(df_filtered)


