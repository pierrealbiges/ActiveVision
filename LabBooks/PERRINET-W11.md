# 2018-05-24
---
Lorsque je tente de lancer les scripts dans lequels Laurent à intégré la possibilité de réaliser les calculs sur GPU (Where.py), j'ai une erreur que je n'arrive pas à débugger (je ne trouve rien sur la doc pytorch, ni sur internet) :

    net = Net(n_feature=N_theta*N_orient*N_scale*N_phase, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=N_orient*N_scale).to(device)
    >> 'Net' object has no attribute 'to'
    
Alors ce que ces lignes semblent fonctionner quand Laurent lance le script + l'intégration est similaire dans cet [exemple](https://github.com/pytorch/examples/blob/master/mnist/main.py).

# 2018-05-25
---
Ce matin lorsque j'ai voulu lancer jupyter-notebook sur babbage, j'ai reçu une erreur :

    AttributeError: '_NamespacePath' object has no attribute 'sort'

Pour régler ce problème, j'ai dû installer setuptools :

    pip3 install --user --upgrade pip setuptools
    
Concernant les conflits sur les derniers scripts entre mon local et origin, j'ai trouvé une explication du fonctionnement de [mergetool](https://stackoverflow.com/questions/161813/how-to-resolve-merge-conflicts-in-git)

---
# To Do

### Modèle
+ simplifier le script pour avoir une convergence du réseau à une entrée synthétique simple qui fait converger le réseau vers la fonction identité (juste pour voir si on maitrise l'apprentissage) -> Qu'est-ce que tu veux dire pr fonction idendité?
+ Recréer la carte d'accuracy en présence de bruit
+ Créer une carte de certitude persistente et mise à jour après chaque saccade
+ ~~Changer le calcul de la perte par une cross-entropy -> Vraiment adapté au problème? Dans le cas où on l'implémente vraiment, plutôt utiliser une [BCE loss](https://pytorch.org/docs/0.3.1/nn.html?highlight=normalize#torch.nn.BCELoss)?~~
+ ~~Remettre valeur rho par défaut~~
+ ~~Mettre à jour les librairies python utilisées~~
+ Intégrer le calcul GPU aux nouveaux scripts -> Cf notes 2018-05-24
+ ~~Retirer la fonction sigmoid, doublon avec F.Sigmoid~~ -> Retrait aussi de F.Sigmoid, doublon depuis le changement de fonction loss vers BCELossWithLogits
+ Réaliser des benchmarking pour choisir les paramètre optimaux pour le modèle
    + learning rate
+ Ne garder que N_pic ou N_X/N_Y (doublon)

### Rapport de stage
+ **Ecrire une ébauche d'Introduction**
+ Ecrire une ébauche de Matériel et méthodes
+ Ecrire une ébauche de Résultats
+ Ecrire une ébauche de Discussion

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
Des choses intéressantes qui gravitent autour de l'IA/ML et des neurosciences mais qui ne concernent pas directement notre sujet...