
# 2018-03-12 - Graphiques Mode 'Multiple'
---
Premier jour de stage, initialisation du rapport de stage (copie+nettoyage du rapport M2a pour conserver la même structure) et du premier notebook de notes personnelles.  
Transmission de la liste de tâches à réaliser + d'articles à lire.

J'ai réussi à résoudre le problème de chargement des données dans le cadre 'Multiple' (comparaison des filtres), en remplacant les lignes : 

    weights_detect_wl = tf.Variable(tf.random_normal([76,2], stddev=0.01), name='weights_detect_wl')    # poids (coordonnées)
    weights_classif_wl = tf.Variable(tf.random_normal([76,10], stddev=0.01), name='weights_classif_wl') # poids (classes)

Par :

    weights_detect = [...]
    weights_classif = [...]

Et en réalisant une réinitialisation du graph après chaque fermeture de session :

    sess_wl.close()
    tf.reset_default_graph()

# 2018-03-13 - Graphiques Mode 'Multiple'
---
Aujourd'hui j'ai pu terminer l'intégration de la majorité des graphiques pour le Mode 'Multiple' (comparaison de la performance des filtres).  
L'optimiseur d'apprentissage [Adam](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) est basé sur les travaux de [Kingma et Lei Ba, 2017](https://arxiv.org/pdf/1412.6980.pdf) et semble intéressant à intégrer à la place d'une simple descente de gradient.

# 2018-03-14 - Classifier_Mapping_TensorFlow
---
[SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) = Stochastic gradient descent ; Corresponds à une approximation de la méthode de descente de gradient  
[CUDA](https://developer.nvidia.com/cuda-education-training) ; Outil d'optimisation de l'apprentissage par l'utilisation de GPU pour réaliser les calculs  
[Doc](https://docs.python.org/2/library/functions.html#super) de la fonction super()  

Lorsqu'on utilise l'optimiseur Adam, il est nécessaire de réaliser l'initialisation des variables après sa création, car il défini implicitement de nouvelles variables (beta_1 et beta_2):
    
    optimizer = tf.train.AdamOptimizer(alpha).minimize(cost)
    sess.run(tf.global_variable_initializer())

# 2018-03-15 - TensorFlow bugged
---
Récupération de l'ordinateur fixe INV-OPE-IMO3 sur lequel j'ai bossé l'année dernière. Clonage du repo ActiveVision pour bosser sur les deux ordis à la fois.

    cd Documents/
    git clone https://github.com/pierrealbiges/ActiveVision
    Username : ...
    Password : ...
    
J'ai un problème lorsque je tente de lancer tensorflow sur la machine INV-OPE-IM03, le script crash et output:

    Illegal instruction (core dumped)

# 2018-03-16 - Carte de certitude
---
J'ai passé l'après-midi d'hier à essayer de régler mon problème avec Tensorflow. Malgré tous mes efforts, je n'arrive pas à le faire tourner sur la machine INV-OPE-IM03. D'après mes recherches, le problème proviendrait d'une incompatibilité entre TensorFlow et le CPU de la machine...  

Après discussion avec l'informaticien du labo, la problème pourrait provenir d'une installation incorrecte de l'OS (donc indépendant de ma volonté), qui provoquerait une incompatibilité driver/CPU/tensorflow.

---
# To Do
+ Traduire le rapport en anglais?
### mnist-logPolar-encoding
+ Créer un classifier pour stopper les saccades lorsque la cible est identifiée
    + Améliorer la méthode d'apprentissage du classifieur (performances très faibles)
+ Créer une ou plusieurs méthodes pour introduire du bruit écologique dans les images (cf Najemnik2005 (chercher la méthode utilisée dans les sources); librairie [SLIP](https://nbviewer.jupyter.org/github/bicv/SLIP/blob/master/SLIP.ipynb) de Laurent)
+ Traduire en modèle probabiliste
+ Introduire le whitening de l'image dans l'apprentissage
+ ~~Se renseigner sur l'[optimiseur Adam](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) pour améliorer l'apprentissage du détecteur~~
+ ~~Intégrer le filtre WaveImage pour réaliser des comparaisons quantitatives WaveImage/LogPolar~~
    + ~~Intégrer les blocs spécifiques à la comparaison (création de double-graphiques)~~ 
    + ~~Résoudre le problème de chargement des données dans le cadre 'Multiple'~~
+ Créer graphique : quantité d'informations original de l'image et pour chaque filtre
+ Créer graphique : économie de bande-passante pour chaque filtre
+ Réaliser une transformation coordonnées -> degrés et adapter le modèle pour les utiliser
+ Remplacer la descente de gradient par un optimiseur Adam
+ Se renseigner sur l'intégration d'un biais à l'apprentissage
### Rapport M2b
+ Ecrire une ébauche d'introduction

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
