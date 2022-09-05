import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from tabs import utils

title = "Utilisation des masques des poumons"
sidebar_name = "Masques des poumons"

available_models = {
    'image_initiale_contraste_initial':'Lenet_image_initiale_initial',
    'poumon_uniquement_contraste_initial': 'Lenet_poumon_only_initial',
    'poumons_cachés_contraste_inital': 'Lenet_poumon_cache_initial',
    'image_initiale_contraste_égalisé':'Lenet_image_initiale_equalized',
    'poumon_uniquement_contraste_égalisé': 'Lenet_poumon_only_equalized',
    'poumons_cachés_contraste_égalisé': 'Lenet_poumon_cache_equalized',
    }                    
                    
mod2name = {v:k for (k,v) in available_models.items()}

    
def run():      
  
    st.title(title)

    st.markdown("""
## Description de l'approche
        
Dans cette partie, nous avons cherché à améliorer la performance du modèle en utilisant les masques des poumons disponibles avec 
le jeu initial des images. **L’idée sous-jacente est que l’information médicalement utile pour prédire l’appartenance à une 
classe devrait se situer uniquement dans les poumons et pas ailleurs dans l’image**. Aussi, si on est capable d’entraîner un 
modèle uniquement sur les images qui ne montrent que les poumons, ce modèle devrait être plus performant que celui entraîné sur 
l’ensemble de l’image. On peut également comparer à la performance d'un modèle entraîné sur les images dont on a retiré les poumons. 
On s'attend à ce que ce modèle ait du mal à classifier correctement les images puisque l'information utile a été retirée des images. 
Enfin, on tester l'influence de l'égalisation des histogrammes en pré-traitement de chacune des images. Ceci nous fait donc 6 
configurations au total à tester (pré-traitement des images O/N et image initiale / uniquement les poumons / poumons retirés).
""")
    if st.checkbox('voir des images pour chacune des 6 catégories'):
        fig = plt.figure()
        plt.subplot(231)
        plt.imshow(plt.imread(utils.path_images+'COVID-17initial-image_initiale.png'), cmap='gray')
        plt.axis('off')
        plt.title('image initiale')
        plt.subplot(232)
        plt.imshow(plt.imread(utils.path_images+'COVID-17initial-poumon_only.png'), cmap='gray')
        plt.axis('off')
        plt.title('uniquement poumons')
        plt.subplot(233)
        plt.imshow(plt.imread(utils.path_images+'COVID-17initial-poumon_cache.png'), cmap='gray')
        plt.axis('off')
        plt.title('poumons retirés')  
        plt.subplot(234)
        plt.imshow(plt.imread(utils.path_images+'COVID-17equalized-image_initiale.png'), cmap='gray')
        plt.axis('off')
        plt.title('image initiale')
        plt.subplot(235)
        plt.imshow(plt.imread(utils.path_images+'COVID-17equalized-poumon_only.png'), cmap='gray')
        plt.axis('off')
        plt.title('uniquement poumons')
        plt.subplot(236)
        plt.imshow(plt.imread(utils.path_images+'COVID-17equalized-poumon_cache.png'), cmap='gray')
        plt.axis('off')
        plt.title('poumons retirés')
        st.pyplot(fig)         
    st.markdown("""
## Résultats

En partant du premier modèle Lenet qui atteint 87% de précision sur les images initiales, on a d’abord cherché à l’entraîner 
sur des images qui ne contiennent que les poumons. **Les résultats sont décevants et la précision moyenne sur l'ensemble de test 
tombe à 75%**. Le même modèle entraîné sur les images dont on a retiré les poumons atteint une précision de 83% sur l'ensemble de 
test, légèrement inférieure donc à la performance du modèle entraîné sur l'ensemble de l'image (85%). Les résultats sont 
précisés dans la tableau suivant:
""")

    res = pd.DataFrame([['image initiale', 'contraste initial', 'train', 92],
                       ['image initiale', 'contraste initial', 'validation', 85],
                       ['image initiale', 'contraste initial', 'test', 85],
                       ['image initiale', 'contraste égalisé', 'train', 94],
                       ['image initiale', 'contraste égalisé', 'validation', 85],
                       ['image initiale', 'contraste égalisé', 'test', 86],
                       ['poumon uniquement', 'contraste initial', 'train', 84],
                       ['poumon uniquement', 'contraste initial', 'validation', 75],
                       ['poumon uniquement', 'contraste initial', 'test', 75],
                       ['poumon uniquement', 'contraste égalisé', 'train',88],
                       ['poumon uniquement', 'contraste égalisé', 'validation',78],
                       ['poumon uniquement', 'contraste égalisé', 'test', 78],
                       ['poumons cachés', 'contraste initial', 'train',89],
                       ['poumons cachés', 'contraste initial', 'validation',85],
                       ['poumons cachés', 'contraste initial', 'test', 83],
                       ['poumons cachés', 'contraste égalisé', 'train',87],
                       ['poumons cachés', 'contraste égalisé', 'validation',83],
                       ['poumons cachés', 'contraste égalisé', 'test', 82],
                       ], 
                   columns=['image', 'contraste', 'type', 'precision'])
    tab = pd.pivot_table(res, values='precision', index=['image', 'type'], columns='contraste')
    idx = pd.IndexSlice
    slice_ = idx[idx[:,'test'], :]
    tab = tab.style.format(precision=2).highlight_min(subset=slice_, props='color:white;background-color:darkblue').highlight_max(subset=slice_, props='color:white;background-color:darkred')
    st.write(tab)
    
    idx_choice = st.multiselect('choisissez les modèles pour comparer matrice de confusion et courbes d\'apprentissage', options=available_models.keys()) 
    chosen_models = [available_models[idx_choice[i]] for i in range(len(idx_choice))]
    show_names = [mod2name[chosen_models[i]] for i in range(len(idx_choice))]

    if len(chosen_models)>0:
        res = utils.load_results(chosen_models)
        utils.show_learning_curve(chosen_models, res, show_names)
        utils.show_confusion_matrices(chosen_models, res, show_names)
