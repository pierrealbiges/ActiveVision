# 2018-04-16
---
Pour construire mon autoencodeur, j'adapte cet [exemple](https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/404_autoencoder.py) à mon problème.

# 2018-04-17
---
Quand j'essaie de modifier le calcul du coût de:

    loss_func = torch.nn.MSELoss()
    
Vers: 

    loss_func = torch.nn.CrossEntropyLoss()
 
Pour réaliser une [Cross-entropy error function](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression), j'obtient toute une série d'erreur qui débouchent finalement sur:

    multi-target not supported at /pytorch/torch/lib/THNN/generic/ClassNLLCriterion.c:22
 
# 2018-04-18
---
Toute la journée j'ai eu des problèmes de connexion avec babbage.  
Au bout de quelques minutes les notebooks jupyter passent en erreur 403 (Forbidden). Cette erreur arrivait déjà avant mais je pouvais continuer à travailler sur le script (y compris le lancer).   
Hors depuis hier, lorsque l'erreur 403 apparaît, je ne peux même plus lancer les scripts et actualiser la page pour y entrer le MdP (solution habituelle à l'erreur) ne donne rien (chargement infini). Je ne semble pas le [seul touché](https://github.com/jupyter/notebook/issues/1845). 
La seule solution que j'ai trouvé pour le moment, c'est de tuer le kernel jupyter sur babbage et de le relancer...  

Cette erreur apparait alors: 

    [W 16:10:32.025 NotebookApp] WebSocket ping timeout after 119997 ms.
    
# 2018-04-19
---
A cause des problèmes de connectivité d'hier, j'ai perdu toute une partie de mon travail.  
Pour reproduire le filtre LogPol moyenné qu'on a construit dans le script 2018-04-16_regression_couples.ipynb dans le script 2018-04-18_Produce_LogPol_figures.ipynb, j'ai dû modifier le calcul de la variable env:

    # Dans le filtre LogPol classique
    env = np.sqrt(phi[i_theta, i_orient, i_scale, 0, :]**2 + phi[i_theta, i_orient, i_scale, 1, :]**2).reshape((N_X, N_Y))
    
    # Dans le filtre LogPol moyenné, version originale
    env = energy[i_orient, i_scale, :].reshape((N_X, N_Y))
    
    # Dans le nouveau script, version fonctionnelle
    env = np.sqrt(energy[i_orient, i_scale, :]**2.5).reshape((N_X, N_Y))


---
# To Do
+ ~~Créer un réseau simple (idéalement une couche) réalisant une régression linéaire v -> a~~
+ ~~Modifier le script pour que l'apprentissage se réalise~~
+ ~~Vérifier l'efficacité de l'apprentissage en visualisant les prédictions~~
+ Appliquer le nouveau filtre moyenné à la carte de certitude
+ Produire et enregistrer des figures des images/cartes de certitude
+ Relancer l'apprentissage avec ces nouvelles données
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
+ [Scientific communication correlates with increased citations in Ecology and Conservation](https://peerj.com/articles/4564/)
+ [Theory of the tensile actuation of fiber reinforced coiled muscles](http://iopscience.iop.org/article/10.1088/1361-665X/aab52b)