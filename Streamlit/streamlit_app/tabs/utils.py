# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 07:43:32 2022

@author: guillaume
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

path_images = "C:/Users/guillaume/Documents/Projet_Covid/Streamlit/images/"
path_model = "C:/Users/guillaume/Documents/Projet_Covid/Streamlit/saved_models/"
path_data = "C:/Users/guillaume/Downloads/data/C19/Test"
path_data_with_masks = "C:/Users/guillaume/Downloads/data/C19_with_masks"
path_pictures_from_internet = "C:/Users/guillaume/Downloads/data/C19_internet"

categories = ['Covid', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
categories = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

def load_results(chosen_models):
    path_res = [path_images+name+'.npy' for name in chosen_models]
    liste_res = {}
    for name, mod in zip(chosen_models, path_res):
        liste_res[name] = np.load(mod, allow_pickle='TRUE').item()
    return liste_res

def show_learning_curve(names, liste_res, show_names):
    n_mods = len(names)
    if n_mods > 1:
        fig, axs = plt.subplots(n_mods, 2, figsize=(10,15))
        for row in range(len(names)):
          hist = liste_res[names[row]]['hist']
          epochs = len(hist['accuracy'])
          axs[row, 0].plot(np.arange(1,epochs+1,1), hist['accuracy'], 'blue', label='training')
          axs[row, 0].plot(np.arange(1,epochs+1,1), hist['val_accuracy'], 'r', label='validation')
          #axs[row, 0].set_xlabel('epochs')
          axs[row, 0].set_ylabel('accuracy')
          axs[row, 0].set_ylim([0.7, 1])
          axs[row, 0].legend()
          axs[row, 0].grid()
          axs[row, 0].set_title(show_names[row])
          
          axs[row, 1].plot(np.arange(1,epochs+1,1), hist['loss'], 'blue', label='training')
          axs[row, 1].plot(np.arange(1,epochs+1,1), hist['val_loss'], 'r', label='validation')
          #axs[row, 1].set_xlabel('epochs')
          axs[row, 1].set_ylabel('loss')
          axs[row, 1].legend()
          axs[row, 1].grid()
          axs[row, 1].set_title(show_names[row])

    else:
        fig = plt.figure()
        hist = liste_res[names[0]]['hist']
        epochs = len(hist['accuracy'])
        plt.subplot(121)
        plt.plot(np.arange(1,epochs+1,1), hist['accuracy'], 'blue', label='training')
        plt.plot(np.arange(1,epochs+1,1), hist['val_accuracy'], 'r', label='validation')
        plt.ylabel('accuracy')
        plt.legend()
        plt.grid()
        plt.title(show_names[0])
        plt.subplot(122)
        plt.plot(np.arange(1,epochs+1,1), hist['loss'], 'blue', label='training')
        plt.plot(np.arange(1,epochs+1,1), hist['val_loss'], 'r', label='validation')
        plt.ylabel('loss')
        plt.legend()
        plt.grid()
        plt.title(show_names[0])
    st.pyplot(fig)


def show_confusion_matrices(names, liste_res, show_names):
    n_mods = len(names)
    n_rows = 4
    n_cols = 2
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12,24))
    d = {0: 'COVID', 1: 'Lung Op', 2: 'Normal', 3: 'Vir Pneum'}
    for i in range(n_mods):
      z = liste_res[names[i]]['confusion_matrix']
      z.columns=d.values()
      z = z.reset_index()
      z['realite'] = z['realite'].replace(d)
      z = z.set_index('realite')
      g = sns.heatmap(data=z, cbar = False, cmap="YlGnBu", fmt='d', linewidths=.5, square=True, annot=True, ax=axs[i//n_cols, i -n_cols*(i//n_cols)])
      g.set_title(str(show_names[i]));
    st.pyplot(fig)


def dice_coef_np(im1, im2):

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    im_sum = im1.sum(axis=(1,2,3)) + im2.sum(axis=(1,2,3))
    #print(im1, im2, im_sum)
    intersection = np.logical_and(im1, im2)
    #print("intersection ",intersection)
    dice =  (2. * intersection.sum(axis=(1,2,3)) / im_sum)

    return np.mean(dice, axis=0)

    