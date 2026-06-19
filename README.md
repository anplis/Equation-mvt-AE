# Equation mvt AE
Peut-on à l’aide seulement des coordonnées d’un point et les paramètres d’un nombre fini d’expériences approcher les équations du mouvement et généraliser celles-ci?

Intro :
Pour des systèmes élémentaires, comme le masse ressort  il est simple de trouver les équations du mouvement. Mais pour des systèmes incluant plus de paramètres et de composants, il peut s’avérer plus complexe de trouver des équations les décrivant. Ainsi pour réussir à approcher ces équations on peut utiliser une autre approche par une ‘approximation évolutive ’ présenté dans ce TIPE.

Idée :
On fait plusieurs expérience d’un même phénomène en faisant varier ses paramètres, et on récolte les coordonnées du système étudié à des mêmes instants afin de recréer leur trajectoire.
Peut on avec ces données seulement retrouver les équations régissant le mouvement du système ?

Problématique :
Peut-on à l’aide seulement des coordonnées d’un point et les paramètres d’un nombre fini d’expériences approcher les équations du mouvement et généraliser celles-ci?

Réalisation :
Utiliser un algorithme évolutionnaire afin d’approcher les équations du mouvement (en fonction des paramètres comme la constante de raideur). On étudiera des mouvements plan pour simplifier l’acquisition des données. 

Protocole :
	- Réaliser n expériences dans les mêmes conditions en faire seulement varier les paramètres associés à l’expérience(ex : raideur du ressort, longueur à vide,position initiale).
	- Pour chaque expérience prendre une vidéo d’une durée T de la trajectoire d’un point M sur un support fixe.
	- Réaliser un pointage des positions du point étudié afin de collecter les coordonnées(x,y) tout les ∆t.
	- Intégrer les données dans l’algorithme et entraîner le modèle jusqu’à avoir un écart inférieur à un certain ε.
	-  Observer l’écart a l’équation théorique du mouvement(si on la) avec des paramétrés différents de ceux des expériences. 

(M, T, ∆t identiques pour chaque expériences)


Algorithme évolutif :
principe :
reproduction du principe de l’évolution selon Darwin 

boucle d’évolution :
- population initiale
	- évaluation de la population
	- sélections 
	- création nouvelle population par mutation des sélectionnés

Appropriation :
Individu = 2 fonctions (pour x et y) de t qui tente d’approcher la trajectoire réelle.
Évaluation =  fonction nommée ‘fitness’ qui attribue un une valeur à chaque individu sur son aptitude à répondre au problème et permet de les classer, celle-ci comptant l’écart à la trajectoire et la complexité de ses fonctions.
Sélection = on garde qu’un certain nombre d’individus parmi les mieux classés (selon la fonction fitness) de notre population population.
Mutation = modification(s) aléatoire(s) des fonctions des individus sélectionnés, (d’amplitude proportionnelle au rang de l’individu).

mise en œuvre :
Utilisation de la bibliothèque « Sympy » pour modéliser et modifier des fonctions mathématiques. 
