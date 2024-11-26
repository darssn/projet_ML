import streamlit as st
import pandas as pd
import altair as alt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import OneHotEncoder

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
    st.subheader("Données sources")
    st.dataframe(data)
     
    st.divider() 

    st.subheader("Modification des données")

    missing_values = data.columns[data.isnull().any()]
    if not missing_values.empty:
        st.info(f"Colonnes avec valeurs manquantes : {list(missing_values)}", icon="ℹ️")

    data = st.data_editor(data)

    st.divider() 

    

    
    # Suppression d'une colonne
    st.subheader("Supprimer une ou plusieurs colonnes")
    select = st.multiselect("Sélectionner une colonne à supprimer", options=data.columns)

    if st.button(label="Supprimer"):
        data = data.drop(columns=select)
        if len(select) > 1:
            st.success(f'Les colonnes {select} ont bien été supprimées')
        else:
            st.success(f'La colonne {select} a bien été supprimée')
        st.write("Données mises à jour")

    # # Imputation
    # missing_values = data.columns[data.isnull().any()]
    # if not missing_values.empty:
    #     st.warning(f"Colonnes avec valeurs manquantes : {list(missing_values)}")
    #     imputation_method = st.selectbox(
    #     "Choisissez une méthode d'imputation",
    #     ["Remplir par une constante", "Moyenne (numérique)", "Médiane (numérique)", "Mode (plus fréquent)", "Supprimer lignes/colonnes"])
    #     selected_columns = st.multiselect( "Colonnes à imputer", options=list(missing_values), default=list(missing_values))


    #     if imputation_method == "Remplir par une constante":
    #         constant_value = st.text_input("Entrez une constante pour remplir les valeurs manquantes", value="0")
            

    #     if st.button("Appliquer l'imputation"):
    #         if imputation_method == "Remplir par une constante":
    #             data[selected_columns] = data[selected_columns].fillna(constant_value)
    #         elif imputation_method == "Moyenne (numérique)":
    #             for col in selected_columns:
    #                 if pd.api.types.is_numeric_dtype(data[col]):
    #                     data[col] = data[col].fillna(data[col].mean())
    #         elif imputation_method == "Médiane (numérique)":
    #             for col in selected_columns:
    #                 if pd.api.types.is_numeric_dtype(data[col]):
    #                     data[col] = data[col].fillna(data[col].median())
    #         elif imputation_method == "Mode (plus fréquent)":
    #             for col in selected_columns:
    #                 data[col] = data[col].fillna(data[col].mode()[0])
    #         elif imputation_method == "Supprimer lignes/colonnes":
    #             if st.radio("Supprimer", ["Lignes", "Colonnes"]) == "Lignes":
    #                 data = data.dropna(subset=selected_columns)
    #             else:
    #                 data = data.drop(columns=selected_columns)
            
            
    #         st.write("### Données après imputation")
    #         st.dataframe(data)

    # else:
    #     st.write("Aucune colonne avec des valeurs manquantes.")



    st.divider() 
    
    st.header("Graphique de Distribution")    
    numeric_columns = data.select_dtypes(include=["number"]).columns
    selected_column = st.selectbox(
        "Choisissez une colonne",
        data.columns,
        format_func=lambda x: f"{x} (Non numérique)" if x not in numeric_columns else x
    )
    if selected_column in numeric_columns:
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(selected_column, bin=True),
            y="count()",
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.error("⚠️ La colonne sélectionnée n'est pas numérique et ne peut pas être visualisée.")
  
    
    st.divider() 
    st.header("Analyse descriptive du dataframe")
    st.write(data.describe())
    
   
with tabs_2:
    st.header("Visualisation")
    with st.container(border=True):
        select_graph = st.selectbox("Choisissez un model de graphe", ["Horizontal Bar Chart", "Area Chart with Gradient"])
        if select_graph != None :
            select_2 = st.selectbox("Choisissez la première colonne", options=data.columns)
            select_3 = st.selectbox("Choisissez la deuxième colonne", options=data.columns)

            match select_graph:
                case "Horizontal Bar Chart":
                    st.title(f"Moyenne de {select_2}, par rapport au {select_3}")
                    chart = alt.Chart(data, height=500).mark_bar().encode(
                        x=(f'average({select_2})'),
                        y=select_3,
                    ).properties(height=alt.Step(10))

                    st.altair_chart(chart, use_container_width=True)
    
                case "Area Chart with Gradient":
                    st.title(f"Comparaison du taux de {select_2}, par rapport au {select_3}")
                    if select_2 != 'target' and select_3 != 'target':
                        data['taux'] = (data[select_2] / data[select_3]) * 100
                        chart = alt.Chart(data).mark_circle(size=60).encode(
                            x=select_2,
                            y=select_3,
                            color='target',
                            tooltip=['target', select_2, select_3, 'taux']
                        )
                        st.altair_chart(chart, use_container_width=True)
                    
                           
with tabs_3:
    st.header("Modélisation")

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

    data = pd.get_dummies(data)
    st.write(data)
    st.write(data.dtypes)


    if st.button("Entrainer le modèle"):
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=1, max_depth=60)
            
        
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