import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tabs import utils
import os

import tensorflow as tf
from tensorflow.keras.models import load_model

title = "Génération de masques de poumons"
sidebar_name = "Génération de masques"

IMG_SIZE = 256

def load_and_preprocess(img_path, mask_path):
  img = tf.io.read_file(img_path)
  img = tf.io.decode_png(img, channels=1)
  img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method='nearest')
      
  mask = tf.io.read_file(mask_path)
  mask = tf.io.decode_png(mask, channels=1)
  mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method='nearest')/255
      
  return (tf.cast(img, tf.float32)/255.0, tf.cast(mask,tf.uint8))

def run():

    st.title(title)

    st.markdown("""
        ## Description de l'approche et du modèle
        
       Dans cette partie, nous allons générer des masques des poumons à partir d’un modèle Unet. Nous disposons pour chaque image 
       d’un masque des poumons qui a été généré au préalable par l’équipe qui a mis au point le jeu de données. Ces masques sont 
       de dimension 256x256. On construit après un modèle Unet avec 4 couches de convolution sur la partie encoder et 4 couches 
       d’upsampling sur la partie décodeur. 
       
       On compile le modèle avec la perte binary_crossentropy car nos masques sont formés uniquement de 0 et de 1 et on regarde 
       les métriques ‘accuracy’ et ‘Dice_coefficient’. Cette dernière mesure le degré de chevauchement entre le masque généré par le modèle 
       et le masque donné en input. Cette métrique ‘Dice_coefficient’ n’est pas disponible dans Keras et nous l’avons donc 
       implémentée sous tensorflow. Cette métrique est souvent utilisée en segmentation d’image et est similaire à un score 
       f1 entre les 2 classes du masque. Dans notre cas, les images des poumons sont assez larges et couvrent bien l’image 
       initiale, de sorte qu’il n’y a pas de déséquilibre majeur entre les 2 classes (fond et poumon) qui forment le masque et, 
       en fin de compte, le Dice_coefficient n’est pas sensiblement plus faible/discriminant que le score de précision.
        

         ## Résultats
         
         Ce modèle s'entraîne assez rapidement et après 16 epochs, le modèle atteint les scores de 99,48% / 99,34% pour la 
         précision sur entraînement/validation et 98,83% / 98,49% pour le Dice_coef sur l’ensemble d’entraînement/validation.
             

         """)

    if st.checkbox('visualiser la courbe d\'apprentissage du modèle'):
        st.image(Image.open(utils.path_images+'Unet_8filters_bilinear.png'))
        st.image(Image.open(utils.path_images+'Unet_8filters_bilinear_test_results.png'))
    if st.checkbox('visualiser le résultat sur une image de l\'ensemble de test'):
        #Unet = load_model(utils.path_model+'Unet_8filters_bilinear', custom_objects={'metrics': dice_coef_tf})
        #Unet = load_model(utils.path_model+'Unet_8filters_bilinear', compile=False)
        Unet = load_model(utils.path_model+'Unet_custom_save', compile=False)
        
        titres = ['inital img', 'intital mask', 'predicted mask']
        
        cat = st.selectbox('dans l\'ensemble de test, choisissez une image parmi les catégories suivantes:', utils.categories)
        available_imgs = os.listdir(os.path.join(utils.path_data, cat))
        if st.button('nouvelle image'):
            index = np.random.randint(len(available_imgs))
        else:
            index=10    
        
        img_path = os.path.join(utils.path_data_with_masks, cat, 'images', available_imgs[index])
        mask_path = os.path.join(utils.path_data_with_masks, cat, 'masks', available_imgs[index])
        #st.write(img_path, mask_path)
        imgs, masks = load_and_preprocess(img_path, mask_path)
        imgs = tf.expand_dims(imgs, axis=0)
        masks = tf.expand_dims(masks, axis=0)
    
        predicted_prob_masks = Unet.predict(imgs)
        predicted_masks = [np.where(predicted_prob_masks[i,:,:,:]<0.5,0,1) for i in range(1)]
        predicted_masks = np.array(predicted_masks)
        a = np.expand_dims(masks[0,:,:,:], axis=0)
        b = np.expand_dims(predicted_masks[0,:,:,:], axis=0)
        dc = utils.dice_coef_np(a, b)
        fig = plt.figure()
        plt.subplot(311)
        plt.imshow(np.squeeze(imgs[0].numpy(),-1), cmap='gray')
        plt.title(titres[0])
        plt.axis('off')
        plt.subplot(312)
        plt.imshow(np.squeeze(masks[0].numpy(),-1), cmap='gray')
        plt.title(titres[1])
        plt.axis('off')
        plt.subplot(313)
        plt.imshow(np.squeeze(predicted_masks[0],-1), cmap = 'gray')
        plt.title(titres[2]+' Dice Coef: '+str(round(100*dc, 2)))
        plt.axis('off')

        st.pyplot(fig)
        