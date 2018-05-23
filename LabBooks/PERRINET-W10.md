# 2018-05-14
---
Pas de notes les deux dernières semaines : une semaine de rapport + une semaine de maladie

# 2018-05-15
---
J'ai écrit une version simplissime d'exploration saccadique dans le modèle en m'inspirant de celui que j'avais créé pour le stage précédent.

# 2018-05-16
---
Avancée de la reflexion sur l'état que devrait avoir le modèle. Voir notes manuscrites.

# 2018-05-17
---
Description [cross-entropy](https://stackoverflow.com/questions/41990250/what-is-cross-entropy#41990932) + notes manuscrites.  
Doc pytorch de la fonction calculant la [cross-entropy loss](https://pytorch.org/docs/0.3.1/nn.html?highlight=crossentropy#torch.nn.CrossEntropyLoss).  
Plus d'informations sur [Relu](https://pytorch.org/docs/0.3.1/nn.html?highlight=relu#torch.nn.ReLU) et [Leaky_Relu](https://pytorch.org/docs/0.3.1/nn.html?highlight=relu#torch.nn.LeakyReLU).

# 2018-05-18
---
Ce matin, j'ai créé deux nouveaux scripts dans lesquels j'ai remplacé ma deuxième couche linéaire cachée par des [RNN](https://pytorch.org/docs/0.3.1/nn.html?#torch.nn.RNN), respectivement d'une et de trois couches.  
Je n'ai pas encore créé de script me permettant de comparer leurs performances, mais logiquement le multicouche devrait être plus performant. Comme on pouvait s'en douter, son apprentissage semble plus lent :

    RNN_1couche > Epoch 0: [10000/60000] Loss: [ 0.01175939] Time: 3.41 mn
    RNN_3couches > Epoch 0: [10000/60000] Loss: [ 0.01874864] Time: 4.03 mn


---
# To Do

+ simplifier le script pour avoir une convergence du réseau à une entrée synthétique simple qui fait converger le réseau vers la fonction identité (juste pour voir si on maitrise l'apprentissage)

+ ~~Complexifier le réseau neuronal~~
    + ~~Changer les couches intermédiaire en ReLu~~
    + ~~Introduire du RNN~~
+ Recréer la carte d'accuracy en présence de bruit
+ ~~Créer fonction réalisant une saccade vers la position contenant la valeur maximale de la sortie de net~~
+ Créer une carte de certitude persistente et mise à jour après chaque saccade
+ ~~Créer des graphiques logpolaires pour vérifier l'importance des problèmes de reconstruction dans le fonctionnement du modèle~~
+ Changer le calcul de la perte par une cross-entropy -> Vraiment adapté au problème? Dans le cas où on l'implémente vraiment, plutôt utiliser une [BCE loss](https://pytorch.org/docs/0.3.1/nn.html?highlight=normalize#torch.nn.BCELoss)?

### Rapport M2b
+ **Ecrire une ébauche d'introduction**
+ Ecrire une ébauche de matériel et méthodes

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
+ [Spatial representations of the viewer’s surroundings](https://www.nature.com/articles/s41598-018-25433-5); Shioiri et al., 2017
