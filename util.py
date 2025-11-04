import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

columns = ["BUDA_SOC_CODE","BUAP_CODE","BUDA_CODE","FAMB_CODE",
           "FAMB_LIBELLE","BUDA_LIBELLE","BUDA_CREE_PAR","BUDA_CREE_DATE",
           "BUAP_DATE_DEB","BUAP_CREE_DATE","BUAP_DATE_FIN","BUAP_MONTANT","BUAP_FOUR_CODE",
           "FBL_CODE","FBL_FOUR_RAISON","FBL_MONTANT_HT","BUDA_DEPASSEMENT",
           "FBL_FCMD_CODE","FBL_DATE_EMM","FBL_DATE_LIV","FBL_FOUR_VILLE",
           "FBL_TAUX_CONV","FBL_FOUR_CP","FBL_DUREE_CREDIT",
           "FBL_RETOUR","FBL_ETAT","BUAP_AFFAIRE_CODE"
           ]



def data_cleaning(df):
    df = df[columns]
    col_date = ["BUDA_CREE_DATE", "BUAP_DATE_DEB","BUAP_CREE_DATE", "BUAP_DATE_FIN", "FBL_DATE_EMM", "FBL_DATE_LIV"]
    for col in col_date:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors='coerce')

    # Conversion des montants en float
    # 💰 Conversion des montants en float
    colonnes_montants = ['FBL_MONTANT_HT', 'BUAP_MONTANT']
    for col in colonnes_montants:
        if col in df.columns:
            # Nettoyage des caractères parasites (espaces, € …)
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'[^0-9,.-]', '', regex=True)  # garde uniquement chiffres, virgules, points, signes
                .str.replace(',', '.', regex=False)         # convertit virgule → point
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def calculate_key_metrics(df):
    # Création d'une copie du DataFrame pour éviter les modifications inattendues
    df_copy = df.copy()
    
    # Montant total du budget alloué (sans doublons sur BUDA_CODE)
    budget_unique = df_copy.drop_duplicates(subset='BUAP_CODE')
    print(budget_unique[['BUAP_CODE', 'BUAP_MONTANT', 'BUDA_DEPASSEMENT']])
    
    total_budget = budget_unique['BUAP_MONTANT'].sum()
    
    # Pourcentage de budgets dépassés sans doublons sur BUDA_CODE exprimé en pourcentage
    depasses = budget_unique[budget_unique['BUDA_DEPASSEMENT'] == 'O'].shape[0]
    nb_depassed = budget_unique.shape[0]
    pourcentage_depassement = (depasses / nb_depassed * 100) if nb_depassed > 0 else 0
    
    # Montant total livré (engagé) — attention à la syntaxe
    total_engage_df = df.groupby(['BUAP_FOUR_CODE'])['FBL_MONTANT_HT'].sum().reset_index()
    total_engage = total_engage_df['FBL_MONTANT_HT'].sum()
    
    
    return total_budget, total_engage, pourcentage_depassement


def repartition_budget_par_famille_donut(df):
    # Création d'une copie du DataFrame pour éviter les modifications inattendues
    df_copy = df.copy()
    
    # Élimination des doublons par BUAP_CODE
    df_copy = df_copy.drop_duplicates(subset='BUAP_CODE')
    
    # Agrégation des montants par famille
    budget_par_famille = df_copy.groupby('FAMB_LIBELLE')['BUAP_MONTANT'].sum().reset_index()
    
    # Calcul du montant total
    total_budget = budget_par_famille['BUAP_MONTANT'].sum()
    
    # Création du Donut Chart avec Plotly
    fig = px.pie(
        budget_par_famille,
        names='FAMB_LIBELLE',
        values='BUAP_MONTANT',
        title='Répartition du budget par famille',
        hole=0.5
    )

    # Ajout du montant total au centre
    fig.update_layout(
        annotations=[
            dict(
                text=f"<b>Total</b><br>{total_budget:,.0f} €",  # format lisible avec séparateur
                x=0.5, y=0.5,
                font_size=18,
                font_color='black',
                showarrow=False,
                align='center'
            )
        ],
        showlegend=True,
        margin=dict(t=60, b=20, l=20, r=20)
    )

    # Amélioration de la lisibilité des labels
    fig.update_traces(
        textposition='inside',
        #textinfo='label',
        insidetextorientation='radial'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    
def etat_budget_mensuel(df):
    # Préparation des données
    df['ANNEE_MOIS'] = df['BUAP_CREE_DATE'].dt.to_period('M').astype(str)

    # Budget alloué (sans doublons BUAP_CODE)
    budget_alloue = df.drop_duplicates(subset='BUAP_CODE').groupby('ANNEE_MOIS')['BUAP_MONTANT'].sum().reset_index()

    # Budget engagé (sans doublons BUAP_CODE + BUAP_FOUR_CODE + FBL_CODE)
    factures_unique = df.drop_duplicates(subset=['BUAP_CODE', 'BUAP_FOUR_CODE', 'FBL_CODE'])
    budget_engage = factures_unique.groupby('ANNEE_MOIS')['FBL_MONTANT_HT'].sum().reset_index()

    # Fusion
    data_final = pd.merge(budget_alloue, budget_engage, on='ANNEE_MOIS', how='left')
    data_final['TAUX_ENGAGEMENT'] = (data_final['FBL_MONTANT_HT'] / data_final['BUAP_MONTANT'] * 100).round(1)
    data_final['ECART'] = data_final['BUAP_MONTANT'] - data_final['FBL_MONTANT_HT']

    # Création du graphique
    fig = go.Figure()

    # Barres groupées
    fig.add_trace(go.Bar(x=data_final['ANNEE_MOIS'], y=data_final['BUAP_MONTANT'], name='Budget Alloué'))
    fig.add_trace(go.Bar(x=data_final['ANNEE_MOIS'], y=data_final['FBL_MONTANT_HT'], name='Budget Engagé'))

    # Courbe pour Budget Alloué
    fig.add_trace(go.Scatter(x=data_final['ANNEE_MOIS'], y=data_final['BUAP_MONTANT'],
                             mode='lines+markers', name='Courbe Budget Alloué'))

    # Courbe pour Budget Engagé
    fig.add_trace(go.Scatter(x=data_final['ANNEE_MOIS'], y=data_final['FBL_MONTANT_HT'],
                             mode='lines+markers', name='Courbe Budget Engagé'))

    # Mise en forme
    fig.update_layout(
        barmode='group',
        title='État mensuel du budget : Alloué vs Engagé (Barres + Courbes)',
        xaxis_title='Mois',
        yaxis_title='Montant (€)',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    return data_final

def evolution_budget_alloue_engage(df):
    
    df_copy = df.copy()

    # --- Ajout colonne ANNEE_MOIS ---
    df_copy['ANNEE_MOIS'] = df_copy['BUAP_CREE_DATE'].dt.to_period('M').astype(str)

    # --- Budget alloué (sans doublons BUAP_CODE) ---
    budget_alloue = df_copy.drop_duplicates(subset='BUAP_CODE').groupby('ANNEE_MOIS')['BUAP_MONTANT'].sum().reset_index()

    # --- Budget engagé (sans doublons BUAP_CODE + BUAP_FOUR_CODE + FBL_CODE) ---
    factures_unique = df_copy.drop_duplicates(subset=['BUAP_CODE', 'BUAP_FOUR_CODE', 'FBL_CODE'])
    budget_engage = factures_unique.groupby('ANNEE_MOIS')['FBL_MONTANT_HT'].sum().reset_index()

    # --- Fusion des deux ---
    budget_complet = pd.merge(budget_alloue, budget_engage, on='ANNEE_MOIS', how='left')

    # --- Création du graphique ---
    fig = go.Figure()

    # Ligne pour budget alloué
    fig.add_trace(go.Scatter(x=budget_complet['ANNEE_MOIS'], y=budget_complet['BUAP_MONTANT'],
                             mode='lines+markers', name='Budget Alloué'))

    # Barres groupées
    fig.add_trace(go.Bar(x=budget_complet['ANNEE_MOIS'], y=budget_complet['BUAP_MONTANT'], name='Budget Alloué'))
    fig.add_trace(go.Bar(x=budget_complet['ANNEE_MOIS'], y=budget_complet['FBL_MONTANT_HT'], name='Budget Engagé'))

    fig.update_layout(
        barmode='group',
        title='Évolution mensuelle : Budget Alloué vs Budget Engagé',
        xaxis_title='Mois',
        yaxis_title='Montant (€)',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

