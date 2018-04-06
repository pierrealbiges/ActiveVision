
# 2018-04-03
---
L'évaluation de la partie What du modèle retourne une performance bien trop faible, en dessous même du hasard (10%):

    Test set: Average loss: 2.3068, Accuracy: 10/10000 (0%), Elapsed time (%mn): 1.480791
    
Mais lorsque j'imprime la variable "correct":

    (...)
    4
    9
    5
    10
    7
    4
    11
    12
    (...)
    
Après avoir changé l'évaluation du nombre de prédictions correctes de:

        correct = pred.eq(LABEL.data.view_as(pred)).cpu().sum()
        
Vers: 

        correct += pred.eq(LABEL.data.view_as(pred)).cpu().sum()
        
J'obtient toujours une performance très faible mais proche des 10% du hasard:

    Test set: Average loss: 2.3068, Accuracy: 892/10000 (9%), Elapsed time (%mn): 1.486337

# 2018-04-04 - Fighting to make the model learn
---
Confusion matrix souvent utilisées pour réaliser un benchmarking de classifieurs (montre quelles classes sont les plus difficiles à discriminer).  
Pour obtenir les poids entrainés:

    model.parameters()
    
Pour les imprimer:

    print(list(model.parameters()))
    
Ce [lien](https://stackoverflow.com/questions/48477198/problems-with-pytorch-mlp-when-training-the-mnist-dataset-retrieved-from-keras) explique la performance nulle (10% même au centre) par l'absence de normalisation des données. Celles-ci le sont lorsqu'elles sont chargées en début de script mais pas après transformation vers du 128x128.

La performance est toujours minime, même si je réalise l'entrainement au centre de l'image :

    Train Epoch: 1 [0/60000 (0%)]	Loss: 2.296373	Elapsed time: 0.00 mn
    Train Epoch: 1 [5000/60000 (8%)]	Loss: 2.305507	Elapsed time: 0.03 mn
    (...)
    Train Epoch: 1 [50000/60000 (83%)]	Loss: 2.298825	Elapsed time: 0.32 mn
    Train Epoch: 1 [55000/60000 (92%)]	Loss: 2.300486	Elapsed time: 0.36 mn

    Test set: Average loss: 2.3024, Accuracy: 982/10000 (10%), Elapsed time: 0.03 mn
    
Le problème étant que j'ai besoin d'obtenir une bonne performance au moins dans le centre de l'image pour construire la carte de certitude qui doit être intégrée dans la base de données que j'essaie de construire.  

A force de bidouillages pour essayer d'améliorer ces performances, j'ai tout de même réussi à fortement réduire sa durée (moins d'une minute par epoch).  
Le coût et la performance n'évoluent pas même lorsque je défini un paramètre alpha très élevé (1).

# 2018-04-05
---
Première journée du séminaire "Probabilities and optimal inference to understand the brain".  
Pour traduire les script de notes personnelles du format ipynb vers markdown, il est possible d'utiliser l'outil [nbconvert](https://nbconvert.readthedocs.io/en/latest/index.html).  

    pip3 install --used nbconvert
    
(A noter que nbconvert est inclus dans l'installer de jupyter, donc la ligne précédente n'est pas nécessaire si celui-ci est déjà installé).

# 2018-04-06
---
Seconde journée du séminaire.  
Traduire les fichiers de notes existant vers du markdown :

    jupyter nbconvert --to markdown PERRINET-W* 
    
D'autres formats sont disponibles (cf [lien](https://ipython.org/ipython-doc/3/notebook/nbconvert.html) et [API](https://nbconvert.readthedocs.io/en/latest/execute_api.html)): html, latex, pdf, slides, script (python), notebook, ...

---
# To Do
+ **Créer la base de données qui servira pour l'apprentissage et l'évaluation**. Cette base devra comprendre pour l'ensemble des situations possibles (coordonnées transformables) les cartes rétinienne (LogPolaire de l'image) et colliculaires (LogPolaire de la carte de certitude) correspondantes
+ Transformer les script ipynb de notes en nb vi l'outil nbconvert
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
+ Bonnes pratiques avec les NN: [lien](https://cs231n.github.io/neural-networks-3/)
