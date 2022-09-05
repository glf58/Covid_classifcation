import streamlit as st


title = "Conclusion"
sidebar_name = "Conclusion"


def run():

    st.title(title)

    st.markdown("---")

    st.markdown("""
                Après avoir défini un modèle de référence sur la base de l'architecture LeNet qui atteint une précision de 85%
                sur l'ensemble de test, nous avons amélioré cette précision à 95% en utilisant le modèle VGG16 et les techniques de 
                transfert learning.
                
                Même si cette performance semble raisonable, **des doutes subsistent pour une utilisation pratique de ce modèle dans 
                un cadre médical car il est difficile d'expliquer les cas où le modèle se trompe, voire de se faire une intuition 
                sur les éléments dans l'image qui ont conduit le modèle à faire sa prédiction**. 
                
                En particulier, l'étude menée sur la suppression des poumons de l'image, ou la concentration de l'image sur les 
                poumons jette un doute sur ce que le modèle observe et la façon dont il détermine la classe de l'image proposée. 
                
                A première vue plutôt contre-intuitifs, ces résultats peuvent en partie s'expliquer par la génération des masques 
                de poumons qui, pour certains, semble assez incomplète et aurait tendance à ne se concentrer que sur les parties 
                'saines' des poumons, oubliant ainsi la partie malade. On note également que l'étude des cartes de caractéristiques
                du modèle simple ne mermet pas non plus de se rassurer sur l'interprétabilité du modèle.
                
                Enfin, on notera que l'étude n'a été menée que sur le modèle simple et n'est donc pas généralisable 
                aux modèles plus compliqués construits à partir de transfert learning.
                
                L'algorithme Unet de segmentation d'image est très efficace et permet d'obtenir rapidement un modèle de génération de
                masques de poumons à partir de nos images. Cependant, comme tout modèle supervisé, la qualité des labels - et donc
                des masques initiaux - est essentielle à la bonne performance du modèle. Pour bien faire, il faudrait faire valider
                les données initiales par des radiologues qui confirmeraient les masques qui ont du sens sur le plan médical, et 
                élimineraient ou modifieraient ceux qui sont trop tronqués.
                
                Les difficultés rencontrées:
                
                - temps de calculs important qui rendent chaque test unitaire compliqué
                - manque de diversité des données
                - difficulté à analyser les images radio pour identifier les facteurs importants dans la prise de décision; échanger avec un expert serait bienvenu.
                
                Les axes d'amélioration et de développement incluent:
                    
                - compléter l'étude d'interprétabilité du modèle sur les modèles avancés
                - implémenter GradCam sur les modèles de transfert learning
                - affiner les labels de masques de poumons
                """)

