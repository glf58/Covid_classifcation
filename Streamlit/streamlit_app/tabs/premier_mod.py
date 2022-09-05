import streamlit as st
import pandas as pd
from PIL import Image
from tabs import utils


title = "Un premier modèle"
sidebar_name = "Premier modèle"

def run():

    st.title(title)

    st.markdown("""
        ## Description du modèle
        
        Nous avons commencé par définir un modèle de base qui servira de point de référence en matière de performance. 
        Comme il s’agit d’un problème de classification d’images, nous sommes partis sur un modèle de type LeNet avec un 
        empilement de trois couches de convolution/Maxpooling pour générer les différentes caractéristiques. A la fin, on 
        dispose de 32 filtres de dimension 35x35. Puis, après une couche de dropout, on a ajouté 2 couches denses avant 
        la dernière couche de sortie.
        

         ## Des premiers résultats
         Tout au long de l'étude, la métrique retenue est la **précision** car les classes sont relativement peu déséquilibrées. 
         Pour chaque modèle, on regardera également le classification report et la matrice de confusion.
         
         
         Ce modèle atteint une **précision de 85% sur les ensembles d’entraînement et de validation. 
         Sur l’ensemble de test, les résultats restent comparables voire un peu meilleur à 87%**, avec des f1-scores un peu 
         déséquilibrés en fonction de chaque classe, le moins bon score étant obtenu pour la catégorie « Lung Opacity » comme le 
         montre le classification report ci-dessous:
             

         """)

    res = pd.DataFrame([[0.86, 0.86, 0.86, 542], [0.84, 0.81, 0.82, 901], [0.88, 0.90, 0.89, 1528], [0.93, 0.94, .93, 201]],
                       columns=['precision', 'recall', 'f1-score', 'support'], 
                       index=['Covid', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'])
    res = res.style.format(precision=2).highlight_min(props='color:white;background-color:darkblue', subset=['f1-score']).highlight_max(props='color:white;background-color:darkred', subset=['f1-score'])
    st.write(res)
    
    st.markdown("""
                ## Influence de l'egalisation des histogrammes et de l'enrichissement des images
                
                A partir de ce modèle, nous avons cherché à tester l'influence de l'égalisation des historgrammes en 
                entrée du modèle, et de l'enrichissement des images. **Cela donne donc 4 configurations à tester** (histogramme égalisé O/N et enrichissement des 
                images O/N). On constate que:
                - Le modèle « de base » se comporte relativement bien mais n’atteint une **précision moyenne que de 85% (87% sur l’ensemble de test)** ; en outre le modèle a du mal à discrimer les catégories « Lung Opacity » et « Normal » en particulier avec des scores f1 de 82% pour la catégorie « Lung Opacity »
                - **Le processus d’augmentation d’image dégrade sensiblement la performance du modèle**, à la fois sur l’ensemble entraînement/validation et celui de test (la performance descend autour de 75%) ; peut-être avons-nous trop bruité les images de départ pour que le modèle arrive à bien généraliser et cela a pénalisé l’entraînement ? 
                - **La normalisation des histogrammes ne semble pas sensiblement améliorer la performance du modèle** ; la précision baisse de 87% à 84% sur l’ensemble de test sur les images brutes alors qu’elle s’améliore de 74% à 78% sur les images avec augmentation

                """)
    res = pd.DataFrame([['initial', 'non', 86.86], 
                        ['augmenté', 'non', 83.58], 
                        ['initial', 'oui', 73.79], 
                        ['augmenté', 'oui', 77.88]], columns=['contraste', 'image augmentée', 'précision'])
    res = res.style.format(precision=2).highlight_min(props='color:white;background-color:darkblue', subset='précision').highlight_max(props='color:white;background-color:darkred', subset='précision')    
    st.write(res)
    if st.checkbox('afficher les graphes de convergence des 4 modèles'):
        st.image(Image.open(utils.path_images+"lenet_compare_convergence.png"))
    if st.checkbox('afficher les matrices de confusion des 4 modèles'):
        st.image(Image.open(utils.path_images+"lenetcompare_conf_matrix.png"))