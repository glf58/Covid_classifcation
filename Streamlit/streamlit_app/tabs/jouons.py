import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
from tensorflow.keras.models import Model, load_model

from tabs import utils

title = "Jouons avecs les modèles et les prédictions"
sidebar_name = "Prédictions en live"
liste_imgs = ['institut-pasteur-covid-19-related-picture_sain.png', 
              'institut-pasteur-covid-19-related-picture_covid.png',
              'pneumonie-infectieuse-pneumopathie-franche-lobaire-aigue.jpeg',
              'IPF_amiodarone.jpg',
              'normal1.jpg', 'normal2.jpg', 'normal3.jpg', 'wikipedia_imagerie.jpg']
#              'https://thoracotomie.files.wordpress.com/2015/07/pneumonie-infectieuse-pneumopathie-franche-lobaire-aigue.jpeg']
# IPF_amiodarone: Fibrose pulmonaire induite par l’amiodarone (wikipedia)
#normal1: https://www.docteurclic.com/galerie-photos/image_3037_400.jpg
#normal2 : https://www.ottawaheart.ca/sites/default/files/legacy/files/5.2.10-Chest-X-ray.jpg
#normal3: https://www.imbm-radiologie.com/wp-content/uploads/2020/03/Capture-d%E2%80%99e%CC%81cran-2020-03-19-a%CC%80-19.18.43.png


filters = {1:8, 2:16, 3:32}
list_layers = list(filters.keys())
list_filters = list(filters.values())

IMG_SIZE = 299
AUTO = tf.data.experimental.AUTOTUNE

def preprocess(img_path):
    img = tf.io.read_file(img_path)
    #img = tf.io.decode_png(img, channels=1)
    img = tf.io.decode_image(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method='nearest')
    img = tf.expand_dims(img, axis=0)
    return tf.cast(img, tf.float32)/255.0

def preprocess_VGG(img_path):
    img = tf.io.read_file(img_path)
    #img = tf.io.decode_png(img, channels=3)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, [224, 224], method='nearest')
    return tf.keras.applications.vgg16.preprocess_input(img)

def preprocess_MobileNet(img_path):
    img = tf.io.read_file(img_path)
    #img = tf.io.decode_png(img, channels=3)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, [224, 224], method='nearest')
    img = tf.cast(img, tf.float32)
    return tf.keras.applications.mobilenet.preprocess_input(img)

def preprocess_InceptionV3(img_path):
    img = tf.io.read_file(img_path)
    #img = tf.io.decode_png(img, channels=3)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, [224, 224], method='nearest')
    img = tf.cast(img, tf.float32)
    return tf.keras.applications.inception_v3.preprocess_input(img)

def get_features(img, liste_name_conv_layer, model, idx_layer, idx_filter):
    # on recupere le resultat des couches de convolution appliquees a l'image choisie et on stocke le resultat pour affichage
    conv_layer = liste_name_conv_layer[idx_layer]
    inputs = model.inputs
    layer = model.get_layer(name=conv_layer)
    new_model = Model(inputs = inputs, outputs = layer.output)
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    res = new_model.predict(img)
    return res[0,:,:,idx_filter]

    
def run():

    st.title(title)
    #st.write("version de tensorflow utilisée: ",tf.__version__)
    st.markdown(
        """
        Dans cette section, vous pouvez choisir un modèle déjà entraîné parmi la liste disponible dans le menu déroulant et une image au choix afin de voir la prédiction rétablie par le modèle.
        """
    )
    models = {'LeNet': ['LeNet_image_initiale_initial', preprocess], 
              'VGG16': ['VGG16FineTune', preprocess_VGG], 
              'MobileNet': ['MobileNetFineTune2', preprocess_MobileNet], 
              'InceptionV3': ['InceptionV3FineTune', preprocess_InceptionV3],
              }
    liste_mods = list(models.keys())
    choix_mod = st.selectbox("choisissez le modèle ", liste_mods)
    #st.write('vous avez choisi', mod, 'et le chemin est ', path_model+mod)
    mod = load_model(utils.path_model+models[choix_mod][0])
    #print(mod.summary())
    cat = st.selectbox('dans l\'ensemble de test, choisissez une image parmi les catégories suivantes:', utils.categories)
    available_imgs = os.listdir(os.path.join(utils.path_data, cat))
    if st.button('nouvelle image'):
        index = np.random.randint(len(available_imgs))
    else:
        index=15    

    img_path = os.path.join(utils.path_data, cat, available_imgs[index])
    st.image(Image.open(img_path))
    if st.checkbox('obtenir la prédiction du modèle'):
        print(choix_mod, models[choix_mod])
        test = models[choix_mod][1](img_path)
        test = tf.expand_dims(test, axis=0)
        y_pred = mod.predict(test)
        res = pd.DataFrame(y_pred*100., columns=utils.categories)
        res = res.style.format(precision=2).highlight_max(props='color:white;background-color:darkblue', axis=1)
        st.write(res)
        st.write('le modèle  {} a prédit {} avec une probabilité de {}%.'.format(choix_mod, utils.categories[int(np.argmax(y_pred, axis=-1))], round(100*np.max(y_pred),2)))
    if st.checkbox('visualiser le résultat des filtres de convolution du modèle LeNet sur l\'image sélectionnée'):
        idx_layer = st.selectbox("indice de la couche de convolution", list(filters.keys()))
        idx_filter = st.slider('indice du filtre',1, filters[idx_layer], step=1)
        # on charge le modele et on recupere les couches de convolution
        model = load_model(utils.path_model+'Lenet_image_initiale_initial')
        liste_name_conv_layer = []
        for layer in model.layers:
            if "conv2d" in layer.name:
                liste_name_conv_layer.append(layer.name)
        #on prepare l'iamge pour l'envoyer au modele
        img = preprocess(img_path)
        img = tf.expand_dims(img, axis=0)
        #on calcule la sortie du filtre de convolution
        res = get_features(img, liste_name_conv_layer, model, idx_layer-1, idx_filter-1)
        fig = plt.figure(figsize=(10,10))
        plt.imshow(res, cmap='gray')
        st.pyplot(fig)
        
    if st.checkbox('tester le modèle sur une image téléchargée d\'internet'):
        img_name = st.selectbox('choisir une image', liste_imgs)
        img_path = os.path.join(utils.path_pictures_from_internet, img_name)
        st.image(Image.open(img_path))
        test = models[choix_mod][1](img_path)
        test = tf.expand_dims(test, axis=0)
        y_pred = mod.predict(test)
        #st.write(y_pred)
        res = pd.DataFrame(y_pred*100., columns=utils.categories)
        res = res.style.format(precision=2).highlight_max(props='color:white;background-color:darkblue', axis=1)
        st.write(res)
        st.write('le modèle  {} a prédit {} avec une probabilité de {}%.'.format(choix_mod, utils.categories[int(np.argmax(y_pred, axis=-1))], round(100*np.max(y_pred),2)))
