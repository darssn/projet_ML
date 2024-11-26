import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import OneHotEncoder
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
    st.subheader("Données sources")

    st.dataframe(data)
     
    st.divider() 

    st.subheader("Modification des données")
    data = st.data_editor(data)

    missing_values = data.columns[data.isnull().any()]
    if not missing_values.empty:
        st.warning(f"Colonnes avec valeurs manquantes : {list(missing_values)}")

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

    
    st.title("Graphique represente la moyenne des proprietes dans chanque type de vin")

    meanProperties = data.groupby('target').mean().reset_index()
    #creation d'un nouveau tableau avec trois colonnes
    newDf = pd.melt(meanProperties,id_vars=['target'],value_vars=['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280/od315_of_diluted_wines','proline'],ignore_index=False, var_name='Properties', value_name='Moyenne des proprietes')             
    chartmean=alt.Chart(newDf, height=500).mark_bar().encode(
    x="Moyenne des proprietes",
    y="target",
    color="Properties"
    )     
    st.altair_chart(chartmean, use_container_width=True)
    
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
       
        
        if not missing_values.empty:
            st.error(f"Pour entraîner le modèle, veuillez remplir le ou les champs manquants, ou supprimer la colonne {list(missing_values)}, dans l'onglet Transformation de données.")
        else:
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
                        X_test["Prévision_"+ target] = result
                        X_test["target"] = y_test
                        st.write(X_test)

                        st.info("Rendez-vous dans l'onglet Evaluation pour monitorer les résultats")

                        if type_data :
                            # Calcul des métriques
                            precision, recall, fscore, _ = score(y_test, result, average='weighted')
                            accuracy = accuracy_score(y_test, result)                        
                            metrics_bool = True
                        
                        else :          
                        # Access the OOB Score
                            oob_score = model.oob_score_                                                 
                            oob_error = 1 - oob_score                           
                            mse = mean_squared_error(y_test, result)                           
                            r2 = r2_score(y_test, result)                           
                            
                            metrics_bool = True
                
                    case "Linear regression":                                                                                 
                        lm = LinearRegression()                      
                        lm.fit(X_train,y_train)                        
                        prediction = lm.predict(X_test)                        
                        mae = metrics.mean_absolute_error(y_test,prediction)            
                        mse = metrics.mean_squared_error(y_test,prediction)
                        r2 = r2_score(y_test, prediction)
                        rmse = np.sqrt(metrics.mean_squared_error(y_test, prediction))                 
                   
                    
with tabs_4:
    st.header("Évaluation")
    
    if metrics_bool :
        match model_choice:          
            case "Random Forest":   
                if type_data :
                    col1, col2, col3, col4 = st.columns(4)
                    # Affichage des résultat
                    st.write("**Métriques du modèle :**")
                    with col1:
                        st.metric(label="- Precision", value=round(precision, 3), delta=+1.5, delta_color="inverse")
                    with col2:
                        st.metric(label="- Recall", value=(round(recall,3)), delta=-2, delta_color="inverse")
                    with col3:
                        st.metric(label="- F1-score", value=(round(fscore,3)), delta=+0.5, delta_color="inverse")
                    with col4:
                        st.metric(label="- Accuracy", value=(round(accuracy,3)), delta=+3.5, delta_color="inverse")

                else :
                    with st.container():          
                    # Access the OOB Score
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(label="- OOB Score", value=(round(oob_score, 4)), delta=+0.5, delta_color="inverse")

                        with col2:
                            st.metric(label="- OOB error", value=(round(oob_error,4)), delta=+0.5, delta_color="inverse")

                        with col3:
                            st.metric(label="- Mean Squared Error", value=(round(mse,4)), delta=+0.5, delta_color="inverse")

                        with col4:
                            st.metric(label="- R-squared", value=(round(r2,4)), delta=+0.5, delta_color="inverse")
                                 
                    # Evaluating the model                    
            case "Linear regression": 
                col1, col2, col3, col4 = st.columns(4)   
                with col1:
                    st.metric(label="- Mean Absolute Error", value=(round(mae, 4)), delta=+1.5, delta_color="inverse")
                with col2:
                    st.metric(label="- Mean Squared Error", value=(round(mse,4)), delta=-0.5, delta_color="inverse")
                with col3:
                    st.metric(label="- R-squared", value=(round(r2,4)), delta=-1.5, delta_color="inverse")    
                with col4:
                    st.metric(label="- Root Mean Squared Error", value=(round(rmse,4)), delta=-1.5, delta_color="inverse")        

    else:
        st.warning("Aucun modèle n'a été entraîné. Veuillez entraîner un modèle avant d'évaluer.")