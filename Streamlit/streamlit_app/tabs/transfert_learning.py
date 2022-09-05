import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from tabs import utils

title = "Utilisation du transfert learning"
sidebar_name = "Transfert learning"


available_models = {'VGG16': 'VGG1622560.2',
                    'VGG16 Fine Tuning': 'VGG16FineTune22560.2',
                    'MobileNet': 'MobileNet22560.2', 
                    'MobileNet Fine Tuning': 'MobileNetFineTune22560.2',
                    'MobileNet HE': 'MobileNet22560.2HE',
                    'MobileNet Fine Tuning HE': 'MobileNetFineTune22560.2HE',
                    'Inception V3': 'InceptionV322560.2',
                    'InceptionV3 Fine Tuning': 'InceptionV3FineTune22560.2'}

mod2name = {v:k for (k,v) in available_models.items()}

    
def run():      
  
    st.title(title)

    st.markdown("""
## Description de l'approche
        
Nous avons utilisé 3 modèles déjà entraînés: **VGG16, MobileNet, et InceptionV3**.
Pour chaque modèle principal, nous supprimons les couches supérieures, que nous remplaçons par un ensemble de couches 
denses entrecoupées de couches de dropout. Le nombre de couches denses, leur nombre de neurones ainsi que le taux de 
dropout sont les nouveaux hyper-paramètres de ce modèle. 

La calibration de ce nouveau modèle s’effectue en deux étapes. Dans la première étape, on fige les poids du modèle de 
base et on entraîne les couches supérieures uniquement. Dans un second temps, on libère les couches initiales, 
on réduit le taux de learning_rate afin de ne pas générer des gradients trop importants qui pourraient trop dégrader 
les poids du modèle issu de la première étape et nuire à la convergence et on réentraîne le modèle complet. 

A notre connaissance, **il n’existe pas de modèle déjà entraîné sur des images en niveau de gris uniquement. Pour utiliser les modèles 
pré-entraînés, nous devons convertir nos images en couleur**, ce qui augmente le temps de traitement et de calcul. 
Comme le temps d'entraînement est globablement assez long pour ces modèles, nous n'avons pas cherché à optimiser les 
hyper-paramètres. Par ailleurs, l'égalisation préalable des histogrammes ne semble pas améliorer la performance du 
modèle Lenet, nous avons donc décidé de ne pas l'utiliser pour les modèles avec transfert learning.
""")

    st.image(Image.open(utils.path_images+'transfer-learning-fine-tuning-approach.width-1200.jpg'))

    st.markdown("""
## Résultats

La comparaison de performance des différents modèles est disponible ci-dessous. Globalement, on constate que:
- Pour les 2 modèles MobileNet et VGG16, il y a un avantage certain à libérer les couches du modèle de base et à poursuivre l’entraînement sur le modèle en entier
- Le modèle à base de MobileNet améliore la précision moyenne autour de 91/93% sur l’ensemble de test (91/92% sur l’ensemble de validation/entraînement) une fois qu’on a libéré les couches du modèle de base ;
- Le modèle à base de VGG16 améliore la précision moyenne autour de 95% sur l’ensemble de test (94/95% sur l’ensemble de validation/entraînement) une fois qu’on a libéré les couches du modèle de base ;
- Sur l’ensemble des 2 modèles (avec et sans libération des couches initiales) même si les scores f1 de chaque classe sont bons (et supérieurs à 95% dans la plupart des cas), c’et la classe « Lung Opacity » qui a le score f1 le plus bas avec un recall de 90% sur cette catégorie

""")

  
    if st.checkbox('afficher les graphes de convergence des différents modèles'):
        st.image(Image.open(utils.path_images+"transfert_learning_compare_convergence_TL.png"))
    if st.checkbox('afficher les matrices de confusion des différents modèles'):
        st.image(Image.open(utils.path_images+"transfert_learningcompare_conf_matrix_TL.png"))
    
    idx_choice = st.multiselect('choisissez les modèles pour comparer matrice de confusion et courbes d\'apprentissage', options=available_models.keys()) 
    chosen_models = [available_models[idx_choice[i]] for i in range(len(idx_choice))]
    show_names = [mod2name[chosen_models[i]] for i in range(len(idx_choice))]

    if len(chosen_models)>0:
        res = utils.load_results(chosen_models)
        utils.show_learning_curve(chosen_models, res, show_names)
        utils.show_confusion_matrices(chosen_models, res, show_names)
    