import streamlit as st


title = "Classification d'images radio des poumons."
sidebar_name = "Introduction"


def run():
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")

    st.title(title)

    st.markdown("---")

    st.markdown("""
                L'objectif est de definir un modele de classification qui identifiera chaque image parmi l'une des 4 categories suivantes:
                - 'Covid'
                - 'Lung Opacity'
                - 'Normal'
                - 'Pneumonia'.
                """)
    st.markdown("""
                Dans un premier temps, nous allons regarder les différentes images dont nous disposons afin de voir si d'éventuels 
                pré-traitements sont nécessaires. Les principaux résultats sont disponibles dans l'onglet **'analyse exploratoire'**.
                
                Puis, nous avons construit un premier modèle sur l'architecture Lenet. Ce premier modèle sert de point de référence 
                et les résultats sont décrits dans l'onglet **'premier modèle'**. On y teste notament l'impact du 
                prétraitement et de l'augmentation des images.
                
                Afin d'améliorer les performances du modèle, on testera plusieurs méthodes de transfert learning dont les 
                résultats sont disponibles dans l'onglet **'transfert learning'**.
                
                Enfin, disposant d'un jeu de masques des poumons, la dernière étape consiste à générer des jeux de masques 
                de poumons et à utiliser ces masques afin d'améliorer encore la performance du modèle. Les résultats sont décrits 
                dans l'onglet **'Masques des poumons'** et l'onglet **'Génération des masques'**.
                
                En dernier lieu, l'onglet **'prediction live'** permet de choisir une image de l'ensemble de test ou provenant d'internet et d'appliquer 
                un modele au choix pour comparer la prédiction du modèle à la réalité.
                """)

    # st.markdown(
    #     """
    #     Here is a bootsrap template for your DataScientest project, built with [Streamlit](https://streamlit.io).

    #     You can browse streamlit documentation and demos to get some inspiration:
    #     - Check out [streamlit.io](https://streamlit.io)
    #     - Jump into streamlit [documentation](https://docs.streamlit.io)
    #     - Use a neural net to [analyze the Udacity Self-driving Car Image
    #       Dataset] (https://github.com/streamlit/demo-self-driving)
    #     - Explore a [New York City rideshare dataset]
    #       (https://github.com/streamlit/demo-uber-nyc-pickups)
    #     """
    # )
