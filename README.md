# Cuda Pathtracer

Un moteur de rendu 3d utilisant CUDA pour le rendu sur le GPU.

Peut charger des modèles 3d au format .obj avec les matériaux (au format mtl). 

Je recommande d'exporter le modèle à partir de Blender en sélectionnant les options Triangulated Mesh et pbr extensions. 

Pour charger un modèle : écrire assets/models/[nom du modèle].obj. 

Pour changer d'envmap : juste écrire le nom de l'envmap sans le chemin. 

# Controles 

- ZQSD : mouvement de caméra 
- Shift et espace : descendre et monter. 
- M : prendre un screenshot
- P : changer la position du soleil dans la skyBox (si elle est chargée) 
- * : lancer le rendu (la caméra ne peut plus bouger)
- / : stopper le rendu
- crtl + G : charger un nouveau modèle
- crtl + S : changer d'envmap
