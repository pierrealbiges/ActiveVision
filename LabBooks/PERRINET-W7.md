# 2018-04-23
---
Traduire un document avec [nbconvert](https://nbconvert.readthedocs.io/en/latest/usage.html) :

    jupyter nbconvert --to markdown path/to/file
    
# 2018-04-24
---
En retirant le drapeau -f de la commande créant mon tunnel ssh, je devrai pouvoir l'arrêter facilement quand il ne semble plus fonctionnant, puisque il ne devrait plus tourner en fond:

    ssh -N -L 127.0.0.1:8895:127.0.0.1:8898 pierre@10.164.7.21
    
En attendant de trouver une façon pertinente de complexifier le nn, j'ai simplement ajouté une couche linéaire.

# 2018-04-25
---
Pour produire un bruit écologique, je peux me baser sur les générateurs de [perlin noise](https://medium.com/@yvanscher/playing-with-perlin-noise-generating-realistic-archipelagos-b59f004d8401).  
Il existe une librairie python permettant de produire du bruit perlin: [noise](https://github.com/caseman/noise)

# 2018-04-27
---
Toute une partie du travail de la veille est perdu à cause de mes problèmes de connectivité.  
Aujourd'hui Laurent a proposé une solution permettant de me connecter directement à babbage sans créer de tunnel ssh. La connection local-babbage n'est plus sécurisée mais si ça fonctionne je continuerai comme ça.  

Nouvelle méthode pour lancer les notebooks depuis local :

    local > ssh pierre@10.164.7.21
    babbage > jupyter-notebook --no-browser --port=8898
    local browser > 10.167.7.21:8898
    
J'ai écrit à nouveau tout ce qui a été perdu hier : implémententation du bruit dans le modèle, production de nouvelles figures comprenant le bruit.  

Une nouvelle méthode de production de bruit écologique remplace la libraire noise: motioncloud, développée par Laurent et moins lourde computationnellement.

---
# To Do
+ ~~Modifier le modèle pour que les valeurs de sortie soient comprises dans l'intervale [0,1]~~
+ ~~Modifier le modèle pour comprendre les dernières modifications de Laurent (notamment le paramètre rho)~~
+ ~~Modifier la génération des coordonnées pour qu'elles soient des integers~~
+ Complexifier le réseau neuronal
+ ~~Introduire du bruit dans l'environnement visuel pour rendre l'apprentissage et l'évaluation plus difficile~~
+ ~~Remplacer la libraire noise par motion cloud, pour la production de bruit~~
+ Recréer la carte d'accuracy en présence de bruit

### Rapport M2b
+ **Ecrire une ébauche d'introduction**

---
# A lire
+ http://bethgelab.org/media/publications/Kuemmerer_High_Low_Level_Fixations_ICCV_2017.pdf
+ https://pdfs.semanticscholar.org/0182/5573781674bcf85d0f5d2ec456842f75ad3c.pdf
+ Schmidhuber, 1991 (voir mail Daucé)
+ Parr and Friston, 2017 (voir mail Perrinet)
+ http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003005#s1
+ http://rpg.ifi.uzh.ch/docs/RAL18_Loquercio.pdf
+ https://www.nature.com/articles/sdata2016126
+ [Liu et al., 2016](http://ieeexplore.ieee.org/document/7762165/?reload=true) : Learning to Predict Eye Fixations via Multiresolution Convolutional Neural Networks
+ [Papier utilisant une méthode similaire à la notre + intégration en robotique](https://www.researchgate.net/publication/220934961_Fast_Object_Detection_with_Foveated_Imaging_and_Virtual_Saccades_on_Resource_Limited_Robots)
### Magnocellular pathway function  
+ [Selective suppression of the magnocellular visual pathway during saccadic eye movements](http://www.nature.com.lama.univ-amu.fr/articles/371511a0), Burr1994
+ [On Identifying Magnocellular and Parvocellular Responses on the Basis of Contrast-Response Functions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3004196/), Skottun2011
+ [Review: Steady and pulsed pedestals, the how and why of post-receptoral pathway separation](http://jov.arvojournals.org/article.aspx?articleid=2191890), Pokorny2011
+ [An evolving view of duplex vision: separate but interacting cortical pathways for perception and action](http://www.sciencedirect.com/science/article/pii/S0959438804000340?via%3Dihub), Goodale2004
+ [Quantitative measurement of saccade amplitude, duration, and velocity](http://n.neurology.org/content/25/11/1065), Baloh1975
### Peripherical vision function
+ [The Role of Peripheral Vision in Configural Spatial Knowledge Acquisition](https://etd.ohiolink.edu/pg_10?0::NO:10:P10_ACCESSION_NUM:wright1496188017928082), Douglas2017

---
# Satellites
+ [Zhu et al., 2018](https://onlinelibrary.wiley.com/doi/abs/10.1002/adma.201707495) : utiliser la vision par ordinateur pour permettre à une imprimante 3D d'imprimer des circuits électroconducteurs directement sur une main. La vision par ordinateur permet ici de compenser les mouvements faibles et involontaires des sujets.