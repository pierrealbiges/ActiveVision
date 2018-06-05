# 2018-06-04
---

# 2018-06-05
---
Le modèle ne présente toujours pas de performance ne serait-ce qu'acceptable. Pour vérifier une hypothèse de l'origine de ce problème, on crée des inputs artificiels correspondant d'abord à un point non-bruité, puis à un blob non-bruité

Note concernant les problèmes de 403 FORBIDDEN (problème de connexion aux notebooks) : le problème apparaît indépendamment du temps, puisque laisser tourner l'ordi pendant la nuit n'entraîne pas une 403 le matin suivant. Ce n'est donc pas un problème de "time out".

Concernant les problèmes d'apprentissage, il faudrait aussi tester de créer la base de données non pas à la volée comme actuellement mais avant train, pour vérifier si ça empêche son fonctionnement normal.

Après quelques tests, l'apprentissage ne se fait toujours pas et ce même si l'on remplace le stimulus par un simple point blanc sur fond noir, et les performances sont toujours très faibles: 

    0/60000] Loss: 0.8450890779495239 Time: 0.03 mn
    [10000/60000] Loss: 0.4025525152683258 Time: 3.15 mn
    [...]
    [40000/60000] Loss: 0.36951038241386414 Time: 28.25 mn
    [50000/60000] Loss: 0.37394532561302185 Time: 37.15 mn

Après quelques vérifications, les coordonnées d'insertion dans input et accuracy semblent les même, le problème d'apprentissage ne semble donc pas correspondre à une dissociation input/label.  
Cependant, l'apprentissage consiste en la comparaison d'un vecteur input (1152,) et d'un vecteur accuracy  (96,).  Est-ce que la grande différence de valeurs peu entrainer un faible apprentissage?  

Remplacer le stimulus par un bloc blanc sur un fond noir ne semble pas améliorer l'apprentissage : 

    [0/60000] Loss: 0.7251732349395752 Time: 0.06 mn
    [10000/60000] Loss: 0.6256107091903687 Time: 5.59 mn
    [...]
    [40000/60000] Loss: 0.3663122355937958 Time: 16.77 mn
    [50000/60000] Loss: 0.37088751792907715 Time: 21.51 mn
    
Pour observer l'influence du calcul actuel du coût (BCEWithLogitsLoss) sur l'apprentissage, je lance un test en le remplacant par le classique MSELoss. Il faudrait aussi tester [SmoothL1Loss](https://pytorch.org/docs/stable/nn.html?highlight=loss#torch.nn.SmoothL1Loss).  
Ce changement semble avoir un effet fortement bénéfique sur l'apprentissage :

    [0/60000] Loss: 1.873697280883789 Time: 0.05 mn
    [10000/60000] Loss: 0.24662844836711884 Time: 5.39 mn
    [20000/60000] Loss: 0.15121513605117798 Time: 10.73 mn
    [30000/60000] Loss: 0.09705615043640137 Time: 16.07 mn
    [40000/60000] Loss: 0.07907275855541229 Time: 21.54 mn

---
# To Do

### Modèle
+ Créer une carte de certitude persistente et mise à jour après chaque saccade
+ Réaliser des benchmarking pour choisir les paramètres optimaux pour le modèle
    + learning rate
+ Ne garder que N_pic ou N_X/N_Y (doublon)
+ Adapter Regard.py à notre modèle
+ ~~Normaliser les données après transformations 128+noise+logpol~~ -> Normalisation retirée, mauvaise intégration endommageant le modèle
    + ~~Réaliser le benchmarking des paramètres mean et std~~ -> cf parent 
+ Corriger le calcul de la nouvelle position du stimulus avec saccade -> (max(a,b) - min(a,b)) ?
+ ~~Modifer la figure model.odg pour s'adapter aux notes manuscrites~~
+ ~~Intégrer la possibilité de créer les figures dans Where.py et réaliser un cleanup de LogPol_figures.ipynb pour ne faire qu'appeler ses fonctions~~
+ ~~Debug: créer input = 1 point~~
    + ~~Debug: créer input = blob~~
+ Debug : créer database avant train
+ Debug : changer le calcul de la perte
    + ~~MSELoss~~ -> Apprentissage en cours
    + ~~SmoothL1Loss~~ -> Apprentissage en cours

### Rapport de stage
+ **Ecrire une ébauche de Résultats**
    + ~~Ecrire une ébauche de Résultats escomptés~~
    + Ecrire une ébauche de Résultats préliminaires
+ Ecrire une ébauche de Discussion
+ Ecrire une ébauche de Résumé

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
+ Focal Loss for Dense Object Detection, Lin et al. 2017 (cf mail Hugo)
+ [Rosten et al., 2008](https://arxiv.org/abs/0810.2434) : Faster and better: a machine learning approach to corner detection
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
Des choses intéressantes qui gravitent autour de l'IA/ML et des neurosciences mais qui ne concernent pas directement notre sujet...