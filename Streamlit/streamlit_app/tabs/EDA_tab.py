import streamlit as st
import numpy as np
from PIL import Image
import os
from skimage import exposure
from skimage.io import imread
import matplotlib.pyplot as plt

from tabs import utils

#path_data = "C:\\Users\\guillaume\\Downloads\\data\\COVID-19_Radiography_Dataset"
cats = {'Covid': [1, 3616], 
        'Lung_Opacity': [1, 6012], 
        'Normal': [1, 10191], 
        'Viral Pneumonia': [1, 1345]}
categories = list(cats.keys())

title = "Analyse Exploratoire des Données"
sidebar_name = "Analyse Exploratoire"

def preprocess(func, img):
    if func=='rescale_intensity':
        p2, p98 = np.percentile(img, (2, 98))
        return exposure.rescale_intensity(img, in_range=(p2, p98))
    elif func=='equalize_hist':
        return exposure.equalize_hist(img)
    else:
        return exposure.equalize_adapthist(img, clip_limit=0.03)

def run():

    st.title(title)
    st.markdown("""
                ## A quoi resemblent nos données?
                
                Nous disposons de 21160 images réparties en 4 classes inhomogènes, avec la répartition suivante:
                
                - Covid (C): 3616
                - Lung Opacity (LO): 6012
                - Normal (N): 10191
                - Pneumonia (P): 1345
                                
                Bien qu’inhomogènes (la classe ‘Normal’ représente 48% des images alors que la classe ‘Pneumonie’ représente 
                                     un peu plus de 6% de nos images), **le déséquilibre entre ces classes sera géré directement 
                dans le modèle, par exemple à l’aide de poids spécifiques inversement proportionnels aux nombres 
                d’éléments dans chaque classe**. Ce choix est largement guidé par la volonté de garder un maximum d’images 
                disponibles pour entraîner notre modèle.
                
                Les images sont en échelle de gris, de dimension 299x299, et la valeur de chaque pixel est comprise entre 
                0 et 255.
                """
                )
    if st.checkbox('voir la distribution des 4 classes'):
        st.image(Image.open(utils.path_images+"distrib_classes.png"))
        
    cat = st.selectbox('Choisissez une image parmi les catégories suivantes:', cats)
    index = st.slider("choisissez l'index de l'image: ", cats[cat][0], cats[cat][1])
    img_path = os.path.join(utils.path_data_with_masks, cat, 'images', cat+'-'+str(index)+'.png')        
    st.image(Image.open(img_path))
    
    st.markdown("""
                ## Un problème de contraste?
                
                En regardant des images au hasard dans chacune des catégories, on observe que certaine images présentent un faible 
                contraste.
                """)
    if st.checkbox('voir quelques images à faible contraste'):
        st.image(Image.open(utils.path_images+"25_contraste.png"))
    
    st.markdown("""
                On définit une fonction qui calcule le contraste de chaque image afin de regarder la distribution du contraste 
                sur nos images. On constate globalement que les images de la classe COVID sont moins contrastées et lumineuses 
                que les autres classes.
                """)
    st.image(Image.open(utils.path_images+"densite_contraste.png"))   
        
    st.markdown("""
                **Comme le problème semble affecter plus particulièrement une classe que les autres, il faudra tester un 
                pré_traitement qui normalise le contraste entre toutes les images.**

                Plusieurs fonctions de la bibliothèque skimage sont disponibles et 3 en particulier ont été testées: 
                    rescale_intensity, equalize_hist et equalize_adapthist. La comparaison de ces 3 méthodes sur les images 
                    à faible contraste est disponible ci-dessous.
                """)
    if st.checkbox('jouer avec les fonctions de contraste sur l\'image sélectionnée'):           
        func = st.selectbox('choisissez la fonction de pré-traîtement', ['rescale_intensity', 'equalize_hist', 
                                                                        'equalize_adapthist'])
        img = imread(img_path)
        img_mod = preprocess(func, img)
        fig = plt.figure()
        
        plt.subplot(2,2,1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title('original')
        
        plt.subplot(2,2,2)
        plt.imshow(img_mod, cmap='gray')
        plt.axis('off')
        plt.title(func)
        
        plt.subplot(2,2,3)
        plt.hist(img.ravel(), bins=255)
        plt.axis('off')
        
        plt.subplot(2,2,4)
        plt.hist(img_mod.ravel(), bins=255)
        plt.axis('off')
        
        st.pyplot(fig)

    if st.checkbox('voir la comparaison des méthodes sur les images les moins contrastées'):
        st.image(Image.open(utils.path_images+"comparaison_normalisation.png"))
        st.image(Image.open(utils.path_images+"comparaison_histo_contraste.png"))
        
    st.markdown("""
                ## Des images à bord noir
                
                Certaine images présentent de larges bandes noires qui entourent les radios. Dans notre dataframe, nous avons 
                calculé la proportion de pixels noirs présents dans l'image. En regardant la distribution des pixels noirs 
                dans les images, on se rend compte qu'en dessous de 20% de pixels noirs, les bandes noires ont globalement 
                disparu. Si, maintenant, on regarde le nombre d'images par catégories qui ont plus que 20% de pixels noirs, 
                on constate que ce phénomène touche à peu près toutes les classes. **Nous avons donc décidé de ne pas 
                effectuer de pré-traitement pour cette partie**.
                """)
    if st.checkbox('voir quelques images avec des bords noirs'):
        st.image(Image.open(utils.path_images+"ex_noirs.png"))
    if st.checkbox('voir les bandes noires en fonction de la proportion de pixels noirs'):
        st.image(Image.open(utils.path_images+"ex_images_noir.png"))
    if st.checkbox('voir le nombre d\'images par catégorie qui ont plus que 20% de pixels noirs'):
        st.image(Image.open(utils.path_images+"dist_noir.png"))
        
