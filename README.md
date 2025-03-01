# Cuda Pathtracer

Un moteur de rendu 3d utilisant CUDA pour le rendu sur le GPU.

Peut charger des modèles 3d au format .obj avec les matériaux. Encore quelques erreurs avec des modèles contenant des vertices de texture. 

Je recommande d'exporter le modèle à partir de Blender en sélectionnant les options Triangulated Mesh et pbr extensions. 

Pour charger un modèle : écrire assets/models/[nom du modèle].obj

# Controles 

ZQSD : mouvement de caméra 

Shift et espace : descendre et monter. 

M : prendre un screenshot

P : changer la position du soleil dans la skyBox (si elle est chargée) 

* : lancer le rendu (la caméra ne peut plus bouger)

/ : stopper le rendu

crtl + G : charger un nouveau modèle 
