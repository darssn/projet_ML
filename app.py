import streamlit as st
import pandas as pd
import altair as alt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

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
    st.header("Modélisation")
    data = load_data()

    target = st.selectbox("Choisissez une colonne cible", options=data.columns)
    y = data[target]

    if data[target].dtype == 'object' or data[target].dtype == 'string':
         X = data.drop(columns=[target])
    else:
        X = data.select_dtypes(exclude=['object','string']).drop(columns=[target])

    st.write(X.columns)

    model_choice = st.selectbox("Choisissez un algorithme", ["Random Forest", "Linear regression"])

    test_size = st.slider("Taille de l'ensemble de test (%)", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    st.write("Taille de l'ensemble d'entraînement :", len(X_train))
    st.write("Taille de l'ensemble de test :", len(X_test))

    st.write("Distribution des classes dans y_train :", y_train.value_counts())
    st.write("Distribution des classes dans y_test :", y_test.value_counts())


    if st.button("Entrainer le modèle"):
        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=1, max_depth=60)
            
        
        model = model.fit(X_train, y_train)
        st.success("Modèle entraîné avec succès")

        result = model.predict(X_test)

        X_test["guess_target"] = result
        X_test["target"] = y_test
        st.write(X_test)

        # Calcul des métriques
        precision, recall, fscore, _ = score(y_test, result, average='macro')
        accuracy = accuracy_score(y_test, result)

        # Affichage des résultats
        st.write("**Métriques du modèle :**")
        st.write(f"- Precision : {round(precision, 3)}")
        st.write(f"- Recall : {round(recall, 3)}")
        st.write(f"- F1-score : {round(fscore, 3)}")
        st.write(f"- Accuracy : {round(accuracy, 3)}")
    

    # if st.button("Évaluer le modèle"):
    #     y_pred = model.predict(X_test)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     st.write(f"### Précision : {accuracy:.2f}")
    #     st.text("Rapport de classification")
    #     st.text(classification_report(y_test, y_pred))


with tabs_4:
    pass
    # st.header("Évaluation")

    # if "model" in st.session_state:
    #     model = st.session_state["model"]
    
    # if st.button("Évaluer le modèle"):
    #     y_pred = model.predict(X_test)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     st.write(f"### Précision : {accuracy:.2f}")
    #     st.text("Rapport de classification")
    #     st.text(classification_report(y_test, y_pred))
    

    # else:
    #     st.warning("Aucun modèle n'a été entraîné. Veuillez entraîner un modèle avant d'évaluer.")