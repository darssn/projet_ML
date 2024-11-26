import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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

    st.write("Graphique de distribution")
    select_graph = st.selectbox("Choisissez un model de graphe", ["Horizontal Bar Chart", "Area Chart with Gradient"])
    if select_graph != None :
        select_2 = st.selectbox("Choisissez la première colonne", options=data.columns)
        select_3 = st.selectbox("Choisissez la deuxième colonne", options=data.columns)

        match select_graph:
            case "Horizontal Bar Chart":
                st.title(f"Moyenne de {select_2}, par rapport au {select_3}")
                chart = alt.Chart(data, height=500).mark_bar(size=10, color="red", fontSize=25).encode(
                    x=(f'average({select_2})'),
                    y=select_3,
                )

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
    
    metrics_bool = False
    model_choice = None
    eval = True
    

    target = st.selectbox("Choisissez une colonne cible", options=data.columns)
  
    
    y = data[target]
    type_data = data[target].dtype == 'object' or data[target].dtype == 'string'


    model_choice = st.selectbox("Choisissez un algorithme", ["Random Forest", "Linear regression"])
    
    
    if model_choice == "Linear regression" and type_data :
        st.error("La colonne selectionné n'est pas numerique et ne peut pas etre entrainé")
        eval = False
    else:
        eval = True
    

    if eval : 
        if type_data:
            X = data.drop(columns=[target])
        else:
            data = pd.get_dummies(data)
            X = data.drop(columns=[target])

    
        test_size = st.slider("Taille de l'ensemble de test (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        st.write("Taille de l'ensemble d'entraînement :", len(X_train))
        st.write("Taille de l'ensemble de test :", len(X_test))

        st.write("Distribution des classes dans y_train :", y_train.value_counts())
        st.write("Distribution des classes dans y_test :", y_test.value_counts())

        

        if st.button("Entrainer le modèle"):
            
            match model_choice:
                
                case "Random Forest":
        
                    if type_data:
                        model = RandomForestClassifier(n_estimators=100, max_depth=60)
                    else:
                        model = RandomForestRegressor(n_estimators=100, max_depth=60,oob_score=True)
                    
                    model = model.fit(X_train, y_train)
                    st.success("Modèle entraîné avec succès")

                    result = model.predict(X_test)

                    X_test["guess_target"] = result
                    X_test["target"] = y_test
                    st.write(X_test)

                    if type_data :
                        # Calcul des métriques
                        precision, recall, fscore, _ = score(y_test, result, average='weighted')
                        accuracy = accuracy_score(y_test, result)
                        
                        metrics_bool = True

                        # Affichage des résultats
                        st.write("**Métriques du modèle :**")
                        st.write(f"- Precision : {round(precision, 3)}")
                        st.write(f"- Recall : {round(recall, 3)}")
                        st.write(f"- F1-score : {round(fscore, 3)}")
                        st.write(f"- Accuracy : {round(accuracy, 3)}")
                    else :          
                    # Access the OOB Score
                        oob_score = model.oob_score_
                        st.write(f'OOB Score: {oob_score:.4f}')
                        
                        oob_error = 1 - oob_score
                        st.write(f'OOB error: {oob_error:.4f}')
                        
                        # Evaluating the model
                        mse = mean_squared_error(y_test, result)
                        st.write(f'Mean Squared Error: {mse:.4f}')

                        r2 = r2_score(y_test, result)
                        st.write(f'R-squared: {r2:.4f}')
                        
                        metrics_bool = True
            
                case "Linear regression":
                                                               
                    st.write("Linear regression")
                    
                    lm = LinearRegression()  
                    
                    lm.fit(X_train,y_train)
                    
                    st.write(lm.coef_)
                    
                    prediction = lm.predict(X_test)
                    
                    mae = metrics.mean_absolute_error(y_test,prediction)            
                    mse = metrics.mean_squared_error(y_test,prediction)
                    rmse = np.sqrt(metrics.mean_squared_error(y_test, prediction))
                   
                    metrics_bool = True
                    
                    st.write(f"- MAE : {mae}")
                    st.write(f"- MSE : {mse}")
                    st.write(f"- RMSE : {rmse}")
                    
                    
with tabs_4:
    st.header("Évaluation")
    
    if metrics_bool :

        match model_choice:
            
            case "Random Forest":
     
                if type_data :
                    # Affichage des résultat
                    st.write("**Métriques du modèle :**")
                    st.write(f"- Precision : {round(precision, 3)}")
                    st.write(f"- Recall : {round(recall, 3)}")
                    st.write(f"- F1-score : {round(fscore, 3)}")
                    st.write(f"- Accuracy : {round(accuracy, 3)}")
                else :          
                    # Access the OOB Score
                    st.write(f'OOB Score: {oob_score:.4f}')
                    st.write(f'OOB error: {oob_error:.4f}')   
                                 
                    # Evaluating the model
                    st.write(f'Mean Squared Error: {mse:.4f}')
                    st.write(f'R-squared: {r2:.4f}')
                    
            case "Linear regression":
                
                    st.write(f"- MAE : {mae}")
                    st.write(f"- MSE : {mse}")
                    st.write(f"- RMSE : {rmse}")

    else:
        st.warning("Aucun modèle n'a été entraîné. Veuillez entraîner un modèle avant d'évaluer.")