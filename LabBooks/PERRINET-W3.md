
# 2018-03-26 - Apprentissage pytorch
---
## Journal Club InVibe Notes  
    Functional brain networks for learning predictive statistics, ? et al., 2017  
Only apparition probabilities are kept, signs-letter relation is randomized and changed between subjects and sessions.  
0th order markov paradigm (no memory but fixed non-equiprobable probabilities) then 1st order markov paradigm (apparition probabilities directly depends on the last sign that appared).  
Notable time differencies between train and evaluation steps, not totally explained by f%RI constraints (unloy used during evaluation steps).   
Kullback-Lieber methode used to compute differences between data distributions.  
High performance variability between subjects.  
Performance seems better when using the maximization (vs matching) strategy but results are too weaks to make any conclusion.  
Paprer contrains weird (too far?) interpretations considering methods and results.  
Actived areas are differents for strategies and for markov paradigm used.  
Biais of using students (not explicit, but subjects have a mean age of 21) for a study investigating learning methodes and implicated cerebral areas?  

## connexion
J'ai un problème avec la connexion à distance et l'autorisation d'écrire sur les notebooks jupyter: très régulièrement (au moins toutes les 10mn) l'accès est bloqué (FORBIDDEN) empéchant d'enregistrer le travail en cours et nécessitant d'entrer à nouveau le MdP défini pour continuer à travailler.  

## Problème actuel à régler: 

    (...)
    x1 = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x0)), 2))
    x2 = x1.view(-1, 480)
    x3 = F.relu(self.fc1(x2))
    (...)
    
Les formats sont:

    x1: [2000,177]
    x3: [480,50]
    
Pour pouvoir réaliser la troisième ligne de l'apprentissage, il faut utiliser x2 pour changer le format du tenseur, sauf que les tailles sont imcompatibles:

    x2 = x1.view(-1,480)
    > size '[-1 x 480]' is invalid for input with 234000 elements
    
Changer l'argument valeur 480 vers 500 résoud ce problème.  

Après avoir résolu le problème de calcul du coût (voir notes manuscrites), je me retrouve confronté à une valeur de coût qui augmente exponentiellement :

    Train Epoch: 1 [0/60000 (0%)]	Loss: 165.038376
    Train Epoch: 1 [10/60000 (0%)]	Loss: 224673.953125
    Train Epoch: 1 [20/60000 (0%)]	Loss: 4220524311485612032.000000
    Train Epoch: 1 [30/60000 (0%)]	Loss: nan
    Train Epoch: 1 [40/60000 (0%)]	Loss: nan
    
Mais au moins l'apprentissage se lance!

# 2018-03-27 - Reducing learning time
---
L'apprentissage se lance donc, mais il semble avoir besoin d'être fortement optimisé.  
Pour première étape, j'ai augmenté la taille de l'échantillon d'apprentissage de 10 à 100, mais l'apprentissage semble très lent.  
J'ai ajouté quelques verboses affichant le temps de calcul, et la transformation de chaque image vers le vecteur LogPolar est très longue (donc le calcul des 100 image de l'échantillon d'apprentissage semble interminable):

    > time to compute one image (s): 49.64
    
Plus précisemment, ce sont les transformations réalisées à chaque image qui sont longues :

    > Time to reshape one image and process the vectorization (s): 48.93
    > Time to compute one image (s): 48.94
    
Encore plus précisemment, c'est la transformation de l'image par la foncton mnist_reshape_128() qui explique ce temps de calcul important:

    > Time to reshape one image (s): 49.60
    > Time to reshape one image and process the vectorization (s): 49.61
    
En simplifiant (passant la variable x_reshape de 4D à 2D) la fonction mnist_reshape_128() pour passer de:

    def mnist_reshape_128(x, i_off=0, j_off=0):
        N_pix = 28
        assert x.shape[2:4] == (N_pix,N_pix)
        x_translate = np.zeros((data.shape[0], 1, N_pix*(128/N_pix), N_pix*(128/N_pix)))
        x_translate[:,:,(N_pix+22+i_off):(2*N_pix+22+i_off), (N_pix+22+j_off):(2*N_pix+22+j_off)] = x
        return x_translate
        
Vers:

    def mnist_reshape_128(x, i_off=0, j_off=0):
        N_pix = 28
        assert x.shape[2:4] == (N_pix,N_pix)
        x_translate = np.zeros((N_pix*(128/N_pix), N_pix*(128/N_pix)))
        x_translate[(N_pix+22+i_off):(2*N_pix+22+i_off), (N_pix+22+j_off):(2*N_pix+22+j_off)] = x[2,-1]
        return x_translate
        
Il semblerait que j'ai fortement réduit ce temps de calcul:

    Time to reshape one image (s): 0.52
    Time to reshape one image and process the vectorization (s): 0.53
    Time to compute the whole dataset (s): 53.12
    
Maintenant que l'entraînement se déroule plus rapidement, il faut que je me penche sur deux choses:  
La valeur de perte stagne, l'apprentissage n'est donc pas efficace  
L'apprentissage plante après 600 entrées, avec l'erreur suivante:    

    x_translate[(N_pix+22+i_off):(2*N_pix+22+i_off), (N_pix+22+j_off):(2*N_pix+22+j_off)] = x[2,-1]
    > ValueError: cannot copy sequence with size 28 to array axis with dimension 24    
    
Après avoir relancé plusieurs fois l'apprentissage, ce crash semble apparaître de façon aléatoire, exemple:

    Train Epoch: 1 [1800/60000 (3%)]	Loss: 159.942780, elapsed time: 16.98 mn
    (...)
    x_translate[(N_pix+22+i_off):(2*N_pix+22+i_off), (N_pix+22+j_off):(2*N_pix+22+j_off)] = x[2,-1]
    # x_translate[104:132,52:80] = x
    # 104:132 = 28, 52:80 = 28
    # i_off = 54
    > ValueError: cannot copy sequence with size 28 to array axis with dimension 24 
    
    x_translate[(N_pix+22+i_off):(2*N_pix+22+i_off), (N_pix+22+j_off):(2*N_pix+22+j_off)] = x[2,-1]
    # x_translate[48:76,-5:23] = x
    # 48:76 = 28, -5:23 = 28
    # j_off = -55
    > ValueError: cannot copy sequence with size 28 to array axis with dimension 0
    
    x_translate[(N_pix+22+i_off):(2*N_pix+22+i_off), (N_pix+22+j_off):(2*N_pix+22+j_off)] = x[2,-1]
    # x_translate[108:136,45:73] = x
    # 108:136 = 28, 45:73 = 28
    # i_off = 58
    > ValueError: cannot copy sequence with size 28 to array axis with dimension 20
    
Dans tous les cas l'une des coordonnées semble >50, est-ce que l'erreur provient de là?  
Après avoir introduit la fonction minmax() pour limiter i_off et j_off à [-50,50], l'erreur ne semble plus apparaître.  

Concernant l'évolution du coût au cours de l'apprentissage, après changement de la méthode de calcul, sa valeur semble stagner (après un pic très important en début d'epoch) :

    loss = F.mse_loss(OUTPUT, coord, size_average=True)
    Training model...
    Train Epoch: 1 [0/60000 (0%)]	Loss: 179.025833, elapsed time: 0.94 mn
    Train Epoch: 1 [100/60000 (0%)]	Loss: 5389.903320, elapsed time: 1.88 mn
    Train Epoch: 1 [200/60000 (0%)]	Loss: 2657.710938, elapsed time: 2.81 mn
    Train Epoch: 1 [300/60000 (0%)]	Loss: 206.353592, elapsed time: 3.74 mn
    Train Epoch: 1 [400/60000 (1%)]	Loss: 208.627625, elapsed time: 4.68 mn
    Train Epoch: 1 [500/60000 (1%)]	Loss: 206.825912, elapsed time: 5.61 mn
    Train Epoch: 1 [600/60000 (1%)]	Loss: 183.498032, elapsed time: 6.54 mn
    Train Epoch: 1 [700/60000 (1%)]	Loss: 217.954712, elapsed time: 7.47 mn
    
Après avoir augmenté très légèrement le paramètre alpha (passant d'une valeur de 0.01 à 0.03), on observe un sur-apprentissage très important:

    Training model...
    Train Epoch: 1 [0/60000 (0%)]	Loss: 192.580429, elapsed time: 0.94 mn
    Train Epoch: 1 [100/60000 (0%)]	Loss: 94805.078125, elapsed time: 1.87 mn
    Train Epoch: 1 [200/60000 (0%)]	Loss: 45865764.000000, elapsed time: 2.80 mn
    Train Epoch: 1 [300/60000 (0%)]	Loss: 365114096091136.000000, elapsed time: 3.74 mn
    Train Epoch: 1 [400/60000 (1%)]	Loss: 6157510041585363631207419192803328.000000, elapsed time: 4.68 mn
    Train Epoch: 1 [500/60000 (1%)]	Loss: 2382964004579637025202740658176.000000, elapsed time: 5.62 mn

Avec une valeur d'alpha de 0.02, le sur-apprentissage est moins important mais semble toujours présent:

    Train Epoch: 1 [0/60000 (0%)]	Loss: 173.929565, elapsed time: 0.96 mn
    Train Epoch: 1 [100/60000 (0%)]	Loss: 332601.937500, elapsed time: 1.90 mn
    Train Epoch: 1 [200/60000 (0%)]	Loss: 797965.500000, elapsed time: 2.85 mn
    Train Epoch: 1 [300/60000 (0%)]	Loss: 316.674377, elapsed time: 3.80 mn
    Train Epoch: 1 [400/60000 (1%)]	Loss: 464.111053, elapsed time: 4.75 mn
    Train Epoch: 1 [500/60000 (1%)]	Loss: 1239.878296, elapsed time: 5.71 mn
    Train Epoch: 1 [600/60000 (1%)]	Loss: 2562.166016, elapsed time: 6.66 mn
    Train Epoch: 1 [700/60000 (1%)]	Loss: 4086.354736, elapsed time: 7.61 mn
    Train Epoch: 1 [800/60000 (1%)]	Loss: 6221.288086, elapsed time: 8.57 mn
    Train Epoch: 1 [900/60000 (2%)]	Loss: 8012.681152, elapsed time: 9.52 mn
    Train Epoch: 1 [1000/60000 (2%)]	Loss: 9431.127930, elapsed time: 10.47 mn

La solution semblerait de modifier complètement l'optimiseur qu'on utilise (actuellement [SGD](http://pytorch.org/docs/master/optim.html?#torch.optim.SGD)).

# 2018-03-28
---
Même avec un paramètre alpha d'une valeur de 0.01, on observe après un certain nombre d'échantillons un important sur-apprentissage:

    (...)
    Train Epoch: 1 [700/60000 (1%)]	Loss: 222.513367, elapsed time: 7.07 mn
    Train Epoch: 1 [800/60000 (1%)]	Loss: 327.580444, elapsed time: 7.94 mn
    Train Epoch: 1 [900/60000 (2%)]	Loss: 397.229797, elapsed time: 8.82 mn
    (...)
    Train Epoch: 1 [4100/60000 (7%)]	Loss: 10234600448.000000, elapsed time: 36.76 mn
    Train Epoch: 1 [4200/60000 (7%)]	Loss: 3827566080.000000, elapsed time: 37.64 mn
    Train Epoch: 1 [4300/60000 (7%)]	Loss: 424681184.000000, elapsed time: 38.52 mn
    (...)
    
Après avoir modifié l'optimiseur vers un [Adam](http://pytorch.org/docs/master/optim.html?#torch.optim.Adam), je n'observe plus de sur-apprentissage pour un paramètre alpha à 0.01, mais une stagnation de la perte:

    Train Epoch: 1 [0/60000 (0%)]	Loss: 187.287643, elapsed time: 0.93 mn
    Train Epoch: 1 [100/60000 (0%)]	Loss: 114.464531, elapsed time: 1.82 mn
    Train Epoch: 1 [200/60000 (0%)]	Loss: 77.370071, elapsed time: 2.75 mn
    (...)
    Train Epoch: 1 [2700/60000 (4%)]	Loss: 175.603241, elapsed time: 25.89 mn
    Train Epoch: 1 [2800/60000 (5%)]	Loss: 189.255295, elapsed time: 26.81 mn
    Train Epoch: 1 [2900/60000 (5%)]	Loss: 256.063812, elapsed time: 27.74 mn

Après des essais d'apprentissage avec des valeurs d'alpha de 0.03, 0.05, 0.08 et 0.1, le coût semble stagner autour d'une valeur de 200. Peut-être qu'il faudrait changer le graph du réseau?

# 2018-03-29 - Reducing the learning time
---
L'entrainement CPU est encore plus lent que prévu, et seulement deux epochs ont eu lieu pendant la nuit :

    Train Epoch: 1 [59800/60000 (100%)]	Loss: 204.7736, elapsed time: 545.76 mn
    # 545 mn = 9h
    
Pour chercher ce qui prends tant de temps dans l'apprentissage, je rends le script bavard (temps en % de s) :

    time to init pytorch argument: 0.0043
    time to init the loaders: 0.053
    time to init the nn: 2.02
    time to init the optimizer: 0.00025
    time to init the mnist_reshape128 and the minmax functions: 0.00035
    time to init the vectorization function 0.00018
    time to init the train function: 0000.24
    time to init the eval function: 0000.26
    
Aucune initialisation dans le script LP_detect.py ne semble prendre de temps.  
Entre temps, j'ai réussi à lancer l'apprenssitage en utilisant l'accélération GPU, mais ça ne semble pas avoir d'effet sur le temps de calcul (qui reste donc similaire à celui n'utilisant que le CPU):

    Train Epoch: 1 [0/60000 (0%)]	Loss: 201.2468, elapsed time: 1.03 mn
    Train Epoch: 1 [100/60000 (0%)]	Loss: 1753.0425, elapsed time: 1.99 mn
    Train Epoch: 1 [200/60000 (0%)]	Loss: 166.1081, elapsed time: 2.94 mn
    
Encore du bavardage:

    time to achieve the vectorization function: 2.6996970176696777
    time to achieve the minmax function: 7.3909759521484375e-06
    time to achieve the minmax function: 2.1457672119140625e-06
    time to achieve the mnist_reshape_128 function: 0.5925393104553223    

Pour un échantillon de 100 images:
    
    vectorization: 2.69 # une seule réalisation par epoch
    minmax: 9.53e-4
    mnist_reshape_128: 59
    
Ce serait donc bien le temps de calcul de la fonction mnist_reshape128 qui explique la durée d'apprentissage. Pré-traiter la base de donnée devrait donc fortement réduire ce temps.  

J'ai trouvé d'où venait le problème. Lors de l'appel de la fonction mnist_reshape_128():

    data_reshaped = mnist_reshape_128(data, i_off, j_off)

L'argument "data" est de forme [batch_size,1,28,28]; donc à chaque appel :

    for idx in range(args.batch_size):
        (...)
        data_reshaped = mnist_reshape_128(data, i_off, j_off)

Je calculais batch_size x batch_size images... Ce qui en plus entraînait la production de 100 échantillons identiques...  
Le problème est donc résolu en remplancant le bloc par:

    for idx in range(args.eval_batch_size):
        (...)
        data_reshaped = mnist_reshape_128(data[idx,0,:], i_off, j_off)

    Train Epoch: 1 [0/60000 (0%)]	Loss: 220.5616, elapsed time: 0.08 mn
    Train Epoch: 1 [100/60000 (0%)]	Loss: 431.8232, elapsed time: 0.11 mn
    Train Epoch: 1 [200/60000 (0%)]	Loss: 180.9021, elapsed time: 0.14 mn

# 2018-03-30
---
Lorsque j'essai de lancer l'évaluation de mon classifieur, je tombe sur l'erreur:

    cuda runtime error (2) : out of memory at /pytorch/torch/lib/THC/generic/THCStorage.cu:58
    
Je n'ai pas trouvé de solution pour régler ce problème. Relancer régulièrement le script fini par autoriser le lancement.  

Lorsque l'évaluation se lance :

    invalid argument 2: size '[-1 x 320]' is invalid for input with 1682000 elements at /pytorch/torch/lib/TH/THStorage.c:37
    
Je ne comprends pas d'où provient l'erreur puisque lors de l'évaluation les entrées, les sorties et la taille de l'échantillon sont censé être les mêmes que lors de l'entraînement.  
Le problème tirait son origine d'une erreur de ma part, lorsque j'ai lancé l'apprentissage:

    def train_classifier(epoch):
        (...)
            (...)
            OUTPUT = model(data)
        
Au lieu de:

    def train_classifier(epoch):
        (...)
            (...)
            OUTPUT = model(INPUT)

---
# To Do
+ ~~Traduire le modèle de TensorFlow vers Pytorch~~
+ Créer la base de données qui servira pour l'apprentissage et l'évaluation. Cette base devra comprendre pour l'ensemble des situations possibles (coordonnées transformables) les cartes rétinienne (LogPolaire de l'image) et colliculaires (LogPolaire de la carte de certitude) correspondantes
+ Transformer les script ipynb de notes en nb
### mnist-logPolar-encoding
+ Créer un classifier pour stopper les saccades lorsque la cible est identifiée
    + ~~Améliorer la méthode d'apprentissage du classifieur (performances très faibles)~~
+ Créer une ou plusieurs méthodes pour introduire du bruit écologique dans les images (cf Najemnik2005 (chercher la méthode utilisée dans les sources); librairie [SLIP](https://nbviewer.jupyter.org/github/bicv/SLIP/blob/master/SLIP.ipynb) de Laurent)
+ Traduire en modèle probabiliste
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
+ [Science AMA Series: We’re roboticists at MIT’s Computer Science and Artificial Intelligence Laboratory who developed a soft robot fish that can swim in the ocean. Ask us anything!](https://www.reddit.com/r/science/comments/87hthf/science_ama_series_were_roboticists_at_mits/) ([Archive](https://www.thewinnower.com/papers/8768-science-ama-series-we-re-roboticists-at-mit-s-computer-science-and-artificial-intelligence-laboratory-who-developed-a-soft-robot-fish-that-can-swim-in-the-ocean-ask-us-anything))
+ [Harvard Biodesign Lab](https://biodesign.seas.harvard.edu/soft-exosuits)
