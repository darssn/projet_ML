import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Projet ML",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)

tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs(["Traitement des données", "Visualisations", "Modelisation", "Evaluation"])

@st.cache_data
def load_data():
    data = pd.read_csv("vin.csv")
    data = data.drop(columns=['Unnamed: 0'])
    return data

# Traitement de données 
with tabs_1:
    st.header("Traitement de données")
    data = load_data()

    st.write("Données du fichier")
    st.dataframe(data)

    select = st.multiselect("Sélectionner une colonne à supprimer", options=data.columns)

    if st.button(label="Supprimer"):
        data = data.drop(columns=select)
        st.write(f'La colonne {select} à bien été supprimée')
        st.write("Données mises à jour")
        st.dataframe(data)


    st.write("Analyse descriptive du dataframe")
    st.write(data.describe())

    
    
with tabs_2:
    st.header("Visualisation")
    data = load_data()

    st.write("Graphique de distribution")
    select_2 = st.selectbox("Choisissez une colonne", options=data.columns)

    st.title(f"Moyenne de {select_2}, par rapport au type de vin")
    chart = alt.Chart(data, height=500).mark_bar(size=30, color="red", fontSize=25).encode(
        x=(f'average({select_2})'),
        y='target',
    )

    st.altair_chart(chart, use_container_width=True)



with tabs_3:
    pass

with tabs_4:
    pass