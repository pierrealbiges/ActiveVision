
# 2018-03-19 - ssh avec machine distante (GPU)
---
Laurent m'a créé un compte sur la machine contenant le GPU. Je peux y accéder à distance avec la commande :

    ssh pierre@10.164.7.21
    
Lorsque je veux installer quoi que ce soit sur l'ordinateur, y compris des librairies python, il faut que je pense à ne le faire que sur mon compte utilisateur :

    pip3 install --user lib_example
    
En suivant les instructions sur leur [site](http://pytorch.org/), j'ai installé pytorch sur les deux machines (perso et GPU).

    pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
    pip3 install torchvision
    
L'installation de pytorch nécessite de connaitre la version de CUDA installée :

    nvcc --version
    
J'ai pu accéder aux notebooks présents sur la machine distante et je peux maintenant les modifier directement. J'ai suivi ces [consignes](http://kawahara.ca/how-to-run-an-ipythonjupyter-notebook-on-a-remote-machine/) :

    remote : jupyter-notebook --no-browser --port=8898
    local : ssh -N -f -L 127.0.0.1:8898:127.0.0.1:8898 pierre@10.164.7.21
    local (browser) : 127.0.0.1:8898
    
J'ai du réaliser quelques étapes supplémentaires de configuration (dont définir un mot de passe pour les prochaines connexions)

# 2018-03-20 - Distant connection + Trad modèle
---
Pour vérifier quels notebooks sont ouverts et leurs tokens :
    
    remote :  jupyter-notebook list

# 2018-03-21 - Trad modèle
---
Parfois, certains notebooks ne chargent pas, je n'ai pas encore trouvé de solution à ce problème.  
Autre problème, dans l'ancien modèle:

    x.shape = 28,28
    
Alors qu'avec pytorch:

    x.shape = [100,1,28,28]
    
Donc la ligne:

    assert x.shape == 28,28
    
Doit devenir:

    assert x.shape[2:4] == (28,28)
    
Pour que l'assert soit validé.  
Mais nouveau problème, la ligne d'après:

    image = x.shape(28, 28)
    > 'Variable' object has no attribute 'reshape'
    
Les deux fonctiones suivants ne fonctionnant pas comme désiré:

    image = a.unsqueeze(24,24)
    image = a.resize_(24,24)
    
J'ai fini par récupérer directement les données via:
    
    images = x[2,-1]
    print(image.size())
    > torch.Size([28,28])
    
Mais nouvelle erreur:

    image = np.append(np.zeros((128 + 2, 28)), image, axis = 0)
    > all the input arrays must have same number of dimensions
    
Pour tenter de résoudre cette erreur, j'ai re-écrit entièrement ma fonction mnist_reshape_128, devenant :

    def mnist_reshape_128(x, i_offset=0, j_offset=0, N_pix=24):
        assert x.shape[2:4] == (28,28)
        x_translate = np.multiply(x.min(), np.ones((x.shape[0], 1, N_pix*(128/N_pix), N_pix*(128/N_pix))))
        image = x_translate[:, :, (N_pix+22+i_offset):(2*N_pix+22+i_offset), (N_pix+22+j_ofsset):(2*N_pix+22+j_ofsset)]
        return image

# 2018-03-22
---

---
# To Do
+ **Traduire le modèle de TensorFlow vers Pytorch**
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
