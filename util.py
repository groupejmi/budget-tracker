import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import calendar
import numpy as np
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
    #depasses = budget_unique[budget_unique['BUDA_DEPASSEMENT'] == 'O'].shape[0]
    #nb_depassed = budget_unique.shape[0]
    #pourcentage_depassement = (depasses / nb_depassed * 100) if nb_depassed > 0 else 0
    
    # Montant total livré (engagé) — attention à la syntaxe
    total_engage_df = df.groupby(['BUAP_CODE', 'FBL_CODE'])['FBL_MONTANT_HT'].sum().reset_index()
    total_engage = total_engage_df['FBL_MONTANT_HT'].sum()
    
    rate_engagement = (total_engage / total_budget * 100) if total_budget > 0 else 0
    
    #nombre de fournisseurs actifs
    fournisseurs_actifs = df['FBL_FOUR_RAISON'].nunique()
    
    return total_budget, total_engage, rate_engagement, fournisseurs_actifs


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

    
def prepare_budget_data(df):

    df = df.copy()
    df['FBL_DATE_EMM'] = pd.to_datetime(df['FBL_DATE_EMM'], errors='coerce')

    # Budget global sans doublons
    budget_global = df.drop_duplicates(subset='BUAP_CODE')['BUAP_MONTANT'].sum()

    # Extraire mois et limiter Mars → Février
    df['mois'] = df['FBL_DATE_EMM'].dt.month
    df = df[(df['mois'] >= 3) | (df['mois'] <= 2)]

    depenses = (
        df.groupby('mois')['FBL_MONTANT_HT']
          .sum()
          .reset_index()
    )

    mois_ordre = [3,4,5,6,7,8,9,10,11,12,1,2]
    mois_labels = [calendar.month_name[m][0:3].capitalize() for m in mois_ordre]
    depenses = depenses.set_index('mois').reindex(mois_ordre).fillna(0).reset_index()
    depenses['mois_label'] = mois_labels

    # Calcul du budget début et fin de mois
    depenses['budget_debut'] = 0.0
    depenses['budget_fin'] = 0.0
    budget_courant = budget_global
    for i in range(len(depenses)):
        depenses.loc[i, 'budget_debut'] = budget_courant
        depenses.loc[i, 'budget_fin'] = max(budget_courant - depenses.loc[i, 'FBL_MONTANT_HT'], 0)
        budget_courant = depenses.loc[i, 'budget_fin']

    # KPI globaux
    depenses_total = depenses['FBL_MONTANT_HT'].sum()
    budget_restant_final = depenses['budget_fin'].iloc[-1]
    taux_conso = (depenses_total / budget_global * 100).round(1)

    return depenses, budget_global, depenses_total, budget_restant_final, taux_conso






def create_kpi_section(df):
    """
    Crée une figure Plotly contenant les indicateurs clés du budget :
    - Budget total
    - Dépenses engagées
    - Budget restant
    - Jauge du taux de consommation
    """
    
    depenses, budget_global, depenses_total, budget_restant_final, taux_conso = prepare_budget_data(df)

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="number",
        value=budget_global,
        title={"text": "💰 Budget total"},
        number={'suffix': " €", 'valueformat': ",.0f"},
        domain={'x': [0.0, 0.2], 'y': [0, 1]}
    ))

    fig.add_trace(go.Indicator(
        mode="number",
        value=depenses_total,
        title={"text": "📊 Dépenses engagées"},
        number={'suffix': " €", 'valueformat': ",.0f"},
        domain={'x': [0.25, 0.45], 'y': [0, 1]}
    ))

    fig.add_trace(go.Indicator(
        mode="number",
        value=budget_restant_final,
        title={"text": "💸 Budget restant"},
        number={'suffix': " €", 'valueformat': ",.0f"},
        domain={'x': [0.5, 0.7], 'y': [0, 1]}
    ))

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=taux_conso,
        title={"text": "🔥 Taux de consommation"},
        number={'suffix': " %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 70], 'color': 'rgba(0,200,0,0.15)'},
                {'range': [70, 90], 'color': 'rgba(255,165,0,0.25)'},
                {'range': [90, 100], 'color': 'rgba(255,0,0,0.25)'},
            ],
        },
        domain={'x': [0.75, 1.0], 'y': [0, 1]}
    ))

    fig.update_layout(
        template="plotly_white",
        height=250,
        margin=dict(l=40, r=40, t=40, b=10),
    )
    return fig


def create_budget_evolution(df):
    """
    Crée le graphique principal d’évolution du budget :
    - Barres : budget début de mois et dépenses mensuelles
    - Ligne : budget restant
    - Fond vert clair = budget total
    """

    depenses, budget_global, _, _, _ = prepare_budget_data(df)

    couleur_budget_debut = '#2ca02c'  # vert
    couleur_depense = '#1f77b4'       # bleu
    couleur_restant = '#ff7f0e'       # orange
    couleur_fond = 'rgba(44,160,44,0.08)'  # vert clair transparent

    fig = go.Figure()

    # Fond
    fig.add_shape(
        type="rect", xref="paper", yref="y",
        x0=0, x1=1, y0=0, y1=budget_global,
        fillcolor=couleur_fond, line=dict(width=0), layer='below'
    )

    # Budget début
    fig.add_trace(go.Bar(
        x=depenses['mois_label'],
        y=depenses['budget_debut'],
        name='Budget début de mois',
        marker_color=couleur_budget_debut,
        text=[f"{v:,.0f} €" for v in depenses['budget_debut']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Budget début : %{y:,.0f} €<extra></extra>'
    ))

    # Dépenses du mois
    fig.add_trace(go.Bar(
        x=depenses['mois_label'],
        y=depenses['FBL_MONTANT_HT'],
        name='Dépense du mois',
        marker_color=couleur_depense,
        text=[f"{v:,.0f} €" for v in depenses['FBL_MONTANT_HT']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Dépense : %{y:,.0f} €<extra></extra>'
    ))

    # Ligne du budget restant
    fig.add_trace(go.Scatter(
        x=depenses['mois_label'],
        y=depenses['budget_fin'],
        mode='lines+markers',
        line=dict(color=couleur_restant, width=3),
        marker=dict(size=8, color=couleur_restant),
        name='Budget restant',
        hovertemplate='<b>%{x}</b><br>Reste : %{y:,.0f} €<extra></extra>'
    ))

    # Ligne budget initial
    fig.add_shape(
        type="line", xref="paper", yref="y",
        x0=0, x1=1, y0=budget_global, y1=budget_global,
        line=dict(color="green", width=2, dash="dash")
    )

    fig.add_annotation(
        x=1.02, y=budget_global, xref="paper", yref="y",
        text=f"<b>Budget initial : {budget_global:,.0f} €</b>",
        font=dict(color="green", size=13),
        showarrow=False, align="left",
        bgcolor="rgba(240,255,240,0.8)",
        bordercolor="green", borderwidth=1, borderpad=4
    )

    # Layout
    fig.update_layout(
        title=dict(
            text="<b>Évolution du budget mois par mois (Mars N → Février N+1)</b>",
            x=0.5, font=dict(size=18)
        ),
        xaxis_title="Mois", yaxis_title="Montant (€)",
        barmode='group', template='plotly_white',
        height=350,
        legend=dict(orientation='h', y=-0.25, x=0.25, bgcolor='rgba(255,255,255,0)'),
        margin=dict(l=70, r=160, t=80, b=80),
        plot_bgcolor='white'
    )

    fig.update_traces(textfont_size=11)
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey', zeroline=False)

    return fig



# --- Fonction pour préparer la figure d'évolution du budget ---
def create_budget_evolution_figure(data, height=450):
    df = data.copy()
    df['FBL_DATE_EMM'] = pd.to_datetime(df['FBL_DATE_EMM'], errors='coerce')

    # Budget global sans doublons sur BUAP_CODE
    budget_global = df.drop_duplicates(subset='BUAP_CODE')['BUAP_MONTANT'].sum()

    # Extraire mois et limiter à Mars->Février
    df['mois'] = df['FBL_DATE_EMM'].dt.month
    df = df[(df['mois'] >= 3) | (df['mois'] <= 2)]

    # Dépenses par mois = engagé du mois
    depenses_mensuelles = (
        df.groupby('mois')['FBL_MONTANT_HT']
        .sum()
        .reset_index()
        .sort_values('mois')
    )

    # Ordre Mars->Février et labels
    mois_ordre = [3,4,5,6,7,8,9,10,11,12,1,2]
    mois_labels = [calendar.month_name[m][0:3].capitalize() for m in mois_ordre]
    depenses_mensuelles = depenses_mensuelles.set_index('mois').reindex(mois_ordre).fillna(0).reset_index()
    depenses_mensuelles['mois_label'] = mois_labels

    # Calculs complémentaires
    dfB = depenses_mensuelles.copy()
    dfB['cumul'] = dfB['FBL_MONTANT_HT'].cumsum()
    dfB['budget_restant_B'] = (budget_global - dfB['cumul']).clip(lower=0)
    dfB['alerte'] = dfB['budget_restant_B'] == 0
    dfB['%_consommation'] = (dfB['cumul'] / budget_global * 100).round(1)

    # Couleurs harmonisées
    couleur_budget_restant = ['#FFB347' if not a else '#D62728' for a in dfB['alerte']]
    couleur_depense = '#1f77b4'

    figB = go.Figure()

    # Barres : Budget restant
    figB.add_trace(go.Bar(
        x=dfB['mois_label'],
        y=dfB['budget_restant_B'],
        name='Budget restant après le mois',
        marker_color=couleur_budget_restant,
        textposition='none',  # on retire le texte pour ne pas surcharger
        hovertemplate='<b>%{x}</b><br>Solde du budget : %{y:,.0f} €<extra></extra>'
    ))

    # Barres : Dépenses du mois
    figB.add_trace(go.Bar(
        x=dfB['mois_label'],
        y=dfB['FBL_MONTANT_HT'],
        name='Engagé du mois',
        marker_color=couleur_depense,
        textposition='none',  # idem
        hovertemplate='<b>%{x}</b><br>Dépense du mois : %{y:,.0f} €<extra></extra>'
    ))

    # Ligne de consommation cumulée
    figB.add_trace(go.Scatter(
        x=dfB['mois_label'],
        y=dfB['cumul'],
        mode='lines+markers+text',
        line=dict(color='rgba(0,0,0,0.4)', width=2),
        marker=dict(size=7, color='black'),
        text=[f"{p} %" for p in dfB['%_consommation']],
        textposition='top center',
        name='% consommation cumulée',
        hovertemplate='<b>%{x}</b><br>Cumul : %{y:,.0f} €<br>Consommé : %{text}<extra></extra>'
    ))

    # Ligne horizontale du budget total
    figB.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0, x1=1,
        y0=budget_global * 1.02, y1=budget_global * 1.02,
        line=dict(color="green", width=2.5, dash="dash")
    )

    # Annotation budget total
    figB.add_annotation(
        x=1.01, y=budget_global * 1.02,
        xref="paper", yref="y",
        text=f"<b>Budget total : {budget_global:,.0f} €</b>",
        font=dict(color="green", size=13),
        showarrow=False,
        align="left",
        bgcolor="rgba(240,255,240,0.7)",
        bordercolor="green",
        borderwidth=1,
        borderpad=4
    )

    # Mise en forme générale
    figB.update_layout(
        title=dict(
            text="<b>Analyse du budget : consommations et reste réel (Mars N → Février N+1)</b>",
            x=0.5, xanchor='center', font=dict(size=16)
        ),
        xaxis_title="Mois",
        yaxis_title="Montant (€)",
        barmode='group',
        template='plotly_white',
        height=height,
        legend=dict(orientation='h', y=-0.2, x=0.3, title_text=''),
        margin=dict(l=60, r=80, t=80, b=50)
    )

    return figB


def prepare_budget_table(data):
    df = data.copy()
    
    # Somme des dépenses par budget
    df_budget = df.groupby(
        ['BUDA_CODE', 'BUDA_LIBELLE', 'BUAP_CODE', 'BUAP_MONTANT', 
         'FAMB_CODE', 'FAMB_LIBELLE'],
        as_index=False
    )['FBL_MONTANT_HT'].sum()
    
    df_budget.rename(columns={'FBL_MONTANT_HT':'Montant Dépensé', 
                              'BUAP_MONTANT':'Montant Initial'}, inplace=True)
    
    df_budget['Solde'] = df_budget['Montant Initial'] - df_budget['Montant Dépensé']
    df_budget['% Consommé'] = (df_budget['Montant Dépensé'] / df_budget['Montant Initial'] * 100).round(1)
    
    # Ligne total
    total_row = pd.DataFrame({
        'BUDA_CODE': ['Total'],
        'BUDA_LIBELLE': [''],
        'BUAP_CODE': [''],
        'Montant Initial': [df_budget['Montant Initial'].sum()],
        'FAMB_CODE': [''],
        'FAMB_LIBELLE': ['Total Général'],
        'Montant Dépensé': [df_budget['Montant Dépensé'].sum()],
        'Solde': [df_budget['Solde'].sum()],
        '% Consommé': [np.nan]
    })
    
    df_final = pd.concat([df_budget, total_row], ignore_index=True)
    
    # Coloration du solde : fonction corrigée
    def color_solde(val):
        if pd.isna(val):
            return ''  # pas de couleur si NaN
        elif val == 0:
            return 'background-color: red; color: white'
        elif val < 0.2 * df_final['Montant Initial'].max():
            return 'background-color: orange'
        else:
            return 'background-color: green; color: white'
    
    return df_final, color_solde


def prepare_budget_summary(data):
    df = data.copy()
    
    # --- Budget initial sans doublons ---
    df_budget_initial = (
        df.drop_duplicates(subset=['BUDA_CODE','BUAP_CODE'])
          .groupby(['BUDA_CODE','BUDA_LIBELLE','FAMB_CODE','FAMB_LIBELLE','BUAP_DATE_DEB','BUAP_DATE_FIN'], 
                   as_index=False)['BUAP_MONTANT']
          .sum()
          .rename(columns={'BUAP_MONTANT':'Montant Initial'})
    )
    
    # --- Dépenses par BUDA ---
    df_depenses = (
        df.groupby('BUDA_CODE', as_index=False)['FBL_MONTANT_HT']
          .sum()
          .rename(columns={'FBL_MONTANT_HT':'Montant Dépensé'})
    )
    
    # --- Fusionner pour avoir un tableau complet ---
    df_summary = df_budget_initial.merge(df_depenses, on='BUDA_CODE', how='left')
    df_summary['Montant Dépensé'] = df_summary['Montant Dépensé'].fillna(0)
    df_summary['Solde'] = df_summary['Montant Initial'] - df_summary['Montant Dépensé']
    
    # --- Ajouter total général ---
    total_row = pd.DataFrame({
        'BUDA_CODE': ['Total'],
        'BUDA_LIBELLE': [''],
        'FAMB_CODE': [''],
        'FAMB_LIBELLE': [''],
        'BUAP_DATE_DEB': [''],
        'BUAP_DATE_FIN': [''],
        'Montant Initial': [df_summary['Montant Initial'].sum()],
        'Montant Dépensé': [df_summary['Montant Dépensé'].sum()],
        'Solde': [df_summary['Solde'].sum()]
    })
    
    df_summary = pd.concat([df_summary, total_row], ignore_index=True)
    return df_summary

def display_budget_summary(df_summary):
    # --- Fonction de coloration des soldes ---
    def color_solde(val):
        if val == '':  # Pour la ligne total ou texte vide
            return ''
        if val <= 0:
            color = 'background-color: #ff6961; color: white;'  # rouge
        elif val <= 0.1 * df_summary['Montant Initial'].max():
            color = 'background-color: #ffd966; color: black;'  # jaune
        else:
            color = 'background-color: #77dd77; color: black;'  # vert
        return color

    # --- Appliquer style ---
    styled = df_summary.style.applymap(color_solde, subset=['Solde']) \
                             .format({'Montant Initial':'{:,.0f}', 
                                      'Montant Dépensé':'{:,.0f}', 
                                      'Solde':'{:,.0f}'})
    
    # --- Affichage Streamlit ---
    st.dataframe(styled, use_container_width=True)


# --- Exemple d'utilisation ---
# df_summary = prepare_budget_summary(data)
# display_budget_summary(df_summary)

