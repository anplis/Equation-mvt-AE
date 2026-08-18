import matplotlib.pyplot as plt
import sympy as sy
import random as rd
from func_timeout import func_timeout, FunctionTimedOut 
import time as time

t = sy.symbols('t', real = True, nonnegative = True)

max_memory = 1000
max_gen = 100
max_size = 20
max_compute = 0.5 # durée maximal(s) qu'une fonction peut prendre pour être évaluer(calculer l'écart)
n_voisin_S = 6
n_voisin_N = 6
select = 10
w_node = 0.02   # coef malus du nombre de noeuds de la fonction
w_cst = 5       # coef malus si la fonction est constante
eps = 0.1


#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#
#                       EVALUATION                          #
#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#

def compute(f, K_val_i):
    vals = []
    dict = {var: val for var, val in zip(K_var, K_val_i)}
    for t_val in T:
        expr = f.subs({t: t_val, **dict})
        if expr.has(sy.zoo) or expr.has(sy.oo) or expr.has(-sy.oo) or expr.has(sy.nan):
            return False
        val = expr.evalf()
        if val.is_infinite or not val.is_real:
            return False
        vals.append(val)
    return vals

def eval(f):
    f_eval = []
    for i in range(len(K_val)):
        
        try:
            f_eval_i = func_timeout(max_compute, compute, args = (f, K_val[i]))
        except FunctionTimedOut:
            return False

        if f_eval_i is False:
            return False

        f_eval.append(f_eval_i)

    return f_eval



def ecart(f,i_var):
    # Calcule l'écart au carré entre la fonction f et les données expérimentales
    f_eval = eval(f)
    if f_eval is False:
        return [float('inf')]*2  # Retourne une valeur infinie si l'évaluation a échoué
    ecart_total, L_ec = 0, []
    for i in range(len(C)):
        C_i = C[i]
        ecart_i = sum(((f_eval[i][j] - C_i[j][i_var])**2) for j in range(len(T)))
        L_ec.append(ecart_i)
        ecart_total += ecart_i
    return (ecart_total, ec_type(L_ec))


#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#
#                       MUTATION                            #
#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#

def rd_cst():#rd cst
    c = rd.choice([t,*K_var,sy.Float(rd.uniform(0, 10))])
    if rd.random()>0.5:
        c = -c
    return c

def mutation(f, memory, n_mut, max_try = 20):
    new_f = f
    for _ in range(n_mut):
        i = 0
        while i<max_try and (new_f == f or stucture(new_f) in memory) :
            try:
                # Applique une mutation à la fonction f
                mut = rd.choices([change_node, del_op, extract], weights=[1, 0.8, 0.3])[0]
                new_f = func_timeout(1, lambda: mut(f))  # Timeout de 1 seconde pour la mutation
            except FunctionTimedOut:
                print('mutation a pris trop temps')
                return None  # Retourne None si la mutation prend trop de temps
            i += 1
        if i == max_try:
            return None
        f = new_f
    return new_f

def change_cst(f):  # Modifie aléatoirement les cst(numériques) d'une fonction
    CSTES = list(f.atoms(sy.Float, sy.Integer))
    c = rd.choice(CSTES)
    if c != 0:
        new_c = sy.Float(c*(1 + rd.uniform(-1, 1)))  # Ajuste la constante aléatoirement
    else:
        new_c = sy.Float(rd.uniform(-10, 10))
    return f.xreplace({c : new_c})

def Div(x, y):
    return x/y

OP_1 = [sy.cos, sy.sin, sy.tan, sy.Abs, sy.sqrt, sy.exp, sy.log]
OP_2 = [sy.Add, sy.Mul, Div, sy.Pow]
OP = {1 : OP_1, 2  :OP_2}

def change_node(f):
    # Change un nœud de la fonction f
    nodes = sorted(list(sy.preorder_traversal(f)), key = lambda x: count_nodes(x), reverse=True)
    if not nodes:
        return rd_cst()
    
    node = rd.choices(nodes, weights=[1/(i+1) for i in range(len(nodes))])[0]
    n_arg = rd.choices([0, 1, 2],weights=[1,1,1])[0]  #nb of args of the new node
    if n_arg == 0:
        new_cst = rd_cst()
        return f.xreplace({node : new_cst})
    new_op = rd.choice(OP[n_arg])
    node_arg = len(node.args)
    
    if node_arg == 0:
        return f.xreplace({node : new_op(node,*[rd_cst() for _ in range(n_arg-1)])})
    
    new_arg_needed = max(0, n_arg - node_arg)
    new_args = rd.sample(list(node.args), min(n_arg, len(node.args))) + [rd_cst() for _ in range(new_arg_needed)]
    return f.xreplace({node : new_op(*new_args)})

def del_op(f):
    # Supprime un opérateur de la fonction f
    ops = [expr for expr in sy.preorder_traversal(f) if len(expr.args)>=1]
    if not ops :
        return f
    op = rd.choice(ops)
    op_arg = rd.choice(op.args)
    return f.xreplace({op : op_arg})

def extract(f):
    # Extrait un sous-arbre de la fonction f
    nodes = sorted(list(sy.preorder_traversal(f)), key = lambda x: count_nodes(x), reverse=True)[1:]#pas interet d'extraire la f elle même
    if nodes:
        return rd.choices(nodes, weights=[1/(i+1) for i in range(len(nodes))])[0]
    else:
        return f

def validate_function(f):
    # Vérifie si la fonction f est valide (pas de division par zéro, pas de logarithme négatif, etc.)
    if f is None:
        return False
    if f.has(sy.nan) or f.has(sy.zoo) or f.has(sy.oo) or f.has(-sy.oo):
        return False
    if f.is_real is False:
        return False
    if count_nodes(f)>max_size:
        return False
    return True

#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#
#                       MAIN                                #
#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#

def ini_f():
    # Génère une fonction aléatoire définie pour tout réel
    c1 = rd_cst()
    c2 = rd_cst()
    if rd.random() < 0.3:
        op1 = rd.choice([sy.cos, sy.sin, sy.tan])
        return rd.choice([sy.Add, sy.Mul])(c1, op1(c2))
    else:
        return rd.choice([sy.Add, sy.Mul])(c1, c2)

def check_cst(f):
    if t in f.atoms(sy.Symbol):
        return 0
    return 1

def moy(L):
    return sum(L)/len(L)

def ec_type(L):
    L_2 = [l**2 for l in L]
    return (max(0, moy(L_2) - moy(L)**2))**(1/2)    #   max(0,...) -> évite les nombres complexes

def fitness(f,i_var):
    ec,ec_typ = ecart(f, i_var)  # Calcule l'écart et l'écart type de la nouvelle fonction
    return ec + w_node*count_nodes(f) + w_cst*check_cst(f) + ec_typ

def insert_sorted(population, new_individual):  # Trouve l'emplacement adapté du nouvel élément et l'insert
    k = 0
    while k<len(population) and new_individual[1]>=population[k][1]:  # Garde population triée dans ordre croissant de fitness
        k+=1
    return population[:k] + [new_individual] + population[k:] 

def count_nodes(f): #   Compte le nombre de noeuds
    return 1 + sum(count_nodes(arg) for arg in f.args)

Nb = sy.symbols('Nb', real = True)  #   Un nombre quelquonque

def stucture(f):    #   renvoie une fonction sans nombres par remplacement des nombres par le symbole Nb 
    f_bis = f
    for cst in f.atoms(sy.Float, sy.Integer):
        f_bis = f_bis.xreplace({cst : Nb})
    return f_bis

def insert_ini_f(population, memory, i_var, n):
    while len(population)<n :  # Continue jusqu'à ce que la population atteigne la taille souhaitée
            new_f = ini_f()
            while stucture(new_f) in memory:
                new_f = ini_f()  # Génère une nouvelle fonction aléatoire
                
            if new_f != None and validate_function(new_f):  # Vérifie si la nouvelle fonction est valide
                fit = fitness(new_f, i_var)
                if isinstance(fit, (int, sy.Float)) and abs(fit) < float('inf'):  # Vérifie si l'écart est un nombre réel
                    memory.append(stucture(new_f))  #  Rajout de la nouvelle structure à la mémoire
                    population = insert_sorted(population, (new_f, fit))  # Insère la nouvelle fonction dans la population triée
    return population

def evolution_one_var(i_var):
    
    # Initialisation de la population
    memory,population = [],[]  # Mémoire pour stocker les individus déjà évalués, population pour stocker les individus triés par écart
    population = insert_ini_f(population, memory, i_var, n_voisin_S)  

    print("start evolution")
    # Boucle d'évolution
    gen = 0     # Compteur de générations
    n_mut = 1   # Nombre de mutation appliquées
    best_f_10 = population[0][0]
    a = time.time()
    while gen<max_gen and ecart(population[0][0],i_var)[0]>eps:  # Continue tant que le nombre de générations est inférieur à max_gen et que l'écart du meilleur individu est supérieur à eps
        Parents = rd.choices(population, weights=[1/(i+1) for i in range(len(population))], k=min(select, len(population)))  # Sélectionne des individu avec une probabilité décroissante
        
        for parent in Parents:
            p_f, p_fit = parent
            # Génération par mutation de nouveaux individus :
            
            # - Structurels
            for _ in range(n_voisin_S):
                new_f = mutation(p_f, memory, n_mut)
                if new_f != None and validate_function(new_f):  # Vérifie si la nouvelle fonction est valide
                    new_fit = fitness(new_f, i_var)
                    if isinstance(new_fit, (int, sy.Float)) and abs(new_fit) < float('inf'):  # Vérifie si l'écart est un nombre réel
                        memory.append(stucture(new_f))  #  Rajout de la nouvelle structure à la mémoire
                        population = insert_sorted(population, (new_f, new_fit))  # Insère la nouvelle fonction dans la population triée 
                    
            # - numériques
            New_f = []
            for _ in range(n_voisin_N):
                if list(p_f.atoms(sy.Float, sy.Integer)): # Si la fonction a des cst(s) numérique(s)
                    new_f = change_cst(p_f)
                    if validate_function(new_f):  # Vérifie si la nouvelle fonction est valide
                            new_fit = fitness(new_f, i_var)
                            if isinstance(new_fit, (int, sy.Float)) and abs(new_fit) < float('inf'):  # Vérifie si l'écart est un nombre réel
                                New_f.append((new_f,new_fit))
            if New_f:
                best_f = min(New_f, key = lambda i : i[1])
                if best_f[1] < p_fit:   # Si f est meilleur que son parent 
                    population = insert_sorted(population, best_f)  # Insère la nouvelle fonction dans la population triée
                    if parent in population: #   supprime parent (pour éviter d'avoir des mêmes structures dans la population)
                        population.remove(parent)
        
        gen += 1  # Incrémente le compteur de générations
        if gen % 10 == 0:  # Affiche l'état de la population tous les 10 générations
            population = population[:max_memory]    # Limite la taille de la population et memory
            memory = memory[max_memory:]    # Enlève les plus anciens
            print(f"Generation {gen} : Durée {time.time()-a} : Best function : {population[0][0]}, Fitness : {population[0][1]}")
            if best_f_10 == population[0][0]:   # Si stagnation du meilleur individu
                n_mut += 1
                population = insert_ini_f(population, memory, i_var, 10)    # Ajoute 10 individus aléatoires
            else:
                n_mut = 1
            best_f_10 = population[0][0]
            a = time.time()
            
    return population[0]  # Retourne le meilleur individu après l'évolution

def evolution():
    F = [evolution_one_var(i_var)[0] for i_var in range(len(C[0][0]))]  # Évolue pour chaque variable meusurée
    print(F)
    return trace_anim_F_all_loop(F)

#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#
#                       GRAPH                               #
#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#

def plot(f,i_var):
    # Trace la fonction f et les données expérimentales
    f_eval = eval(f)
    if f_eval is False:
        print("Erreur lors de l'évaluation de la fonction.")
        return
    plt.figure(figsize=(10, 6))
    for i in range(len(C)):
        plt.plot(T, [C[i][j][i_var] for j in range(len(T))], 'o', label=f'Données expérimentales {i+1}')
        plt.plot(T, f_eval[i], '-', label=f'Fonction estimée {i+1}')
    plt.xlabel('Temps')
    plt.ylabel('Valeur')
    plt.title('Comparaison des données expérimentales et de la fonction estimée')
    plt.legend()
    plt.grid()
    plt.show()

def plot2(f, i_var):
    # Évalue la fonction f
    f_eval = eval(f)
    if f_eval is False:
        print("Erreur lors de l'évaluation de la fonction.")
        return
    n = len(C)  # nombre de sets expérimentaux
    # Une seule fenêtre avec n subplots
    fig, axes = plt.subplots(n, 1, figsize=(10, 4*n), sharex=True)
    # Si n == 1, axes n'est pas une liste → on le transforme
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        # Données expérimentales pour ce set
        exp_vals = [C[i][j][i_var] for j in range(len(T))]
        # Approximation f pour ce set
        approx_vals = f_eval[i]
        # Traces
        ax.plot(T, exp_vals, 'o')
        ax.plot(T, approx_vals, '-')
        # Mise en forme
        ax.set_ylabel('Valeur')
        ax.set_title(f'Set {i+1}')
        ax.grid()
        ax.legend()
    axes[-1].set_xlabel('Temps')
    plt.tight_layout()
    plt.show()


def trace(F):
    # Trace la courbe F et les données expérimentales
    F_eval = [eval(f)   for f in F]
    if any(f_eval is False for f_eval in F_eval):
        print("Erreur lors de l'évaluation de la fonction.")
        return
    plt.figure(figsize=(10, 6))
    for i in range(len(C)):
        plt.plot([C[i][j][0] for j in range(len(T))], [C[i][j][1] for j in range(len(T))], 'o', label=f'Données expérimentales {i+1}')
        plt.plot(F_eval[0][i], F_eval[1][i], '-', label=f'Fonction estimée {i+1}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparaison des données expérimentales et de la fonction estimée')
    plt.legend()
    plt.grid()
    plt.show()

from matplotlib.animation import FuncAnimation
import numpy as np
import time

def trace_anim_F_all_loop(F):
    # Évalue F = [Fx(t), Fy(t)]
    F_eval = [eval(f) for f in F]
    if any(f_eval is False for f_eval in F_eval):
        print("Fonction F mal définie")
        return

    n_sets = len(C)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Préparation des données pour chaque set
    X1, Y1, X2, Y2 = [], [], [], []

    for i in range(n_sets):
        Xi = np.array([p[0] for p in C[i]], dtype=float)
        Yi = np.array([p[1] for p in C[i]], dtype=float)

        XFi = np.array(F_eval[0][i], dtype=float)
        YFi = np.array(F_eval[1][i], dtype=float)

        X1.append(Xi)
        Y1.append(Yi)
        X2.append(XFi)
        Y2.append(YFi)

    # Nombre de frames synchronisé
    n_frames = min(len(T), *(len(x) for x in X1), *(len(x) for x in X2))

    # Points animés pour chaque graphe
    points_exp = []
    points_F = []

    for i in range(n_sets):
        ax = axes[i]
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Set {i+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Tracés fixes
        ax.plot(X1[i], Y1[i], color="gray", alpha=0.4, label="Trajectoire réelle")
        ax.plot(X2[i], Y2[i], color="tab:blue", alpha=0.4, label="Approximation F")

        # Points animés
        p_exp, = ax.plot([], [], "ro", ms=6)
        p_F,   = ax.plot([], [], "bo", ms=6)

        points_exp.append(p_exp)
        points_F.append(p_F)

        ax.legend()

    # Animation
    def init():
        for p1, p2 in zip(points_exp, points_F):
            p1.set_data([], [])
            p2.set_data([], [])
        return points_exp + points_F

    def update(frame):
        # Si on arrive à la fin → pause 2 secondes puis recommencer
        if frame == n_frames - 1:
            time.sleep(1)

        for i in range(n_sets):
            points_exp[i].set_data([X1[i][frame]], [Y1[i][frame]])
            points_F[i].set_data([X2[i][frame]], [Y2[i][frame]])

        return points_exp + points_F

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=40,
        repeat=True,   # boucle infinie
        blit=True,
    )

    plt.tight_layout()
    plt.show()

    return ani

#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#
#                       DATA                                #
#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#

teta0 = sy.symbols('teta0',real=True)
l = sy.symbols('l',real=True)


import math as ma
K_val = [(1,10),(2,5)]
K_var = [l,teta0]

def f_data(t,x,y):
    return t + ma.sin(2*t + x) + y

def g_data(t,x,y):
    return t + x + y
T=[]
to = 0
for _ in range(50):
    T.append(to)
    to += 0.5

C = [[(f_data(to,x,y),g_data(to,x,y)) for to in T] for x,y in K_val]

evolution()
