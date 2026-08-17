import matplotlib.pyplot as plt
import sympy as sy
import random as rd
from func_timeout import func_timeout, FunctionTimedOut 
import time as time

t = sy.symbols('t')

max_memory = 1000
max_gen = 100
max_size = 30
max_compute = 0.5 # durée maximal(s) qu'une fonction peut prendre pour être évaluer(calculer l'écart)
n_voisin_S = 10
n_voisin_N = 10
select = 20
w_node = 0.01   # coef malus du nombre de noeuds de la fonction
w_cst = 5       # coef malus si la fonction est constante
eps = 0.1


#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#
#                       EVALUATION                          #
#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#

def eval(f):
    try:
        f_eval = []
        for i in range(len(K_val)):
            K_val_i = K_val[i]*len(T)
            def compute():
                vals = []
                for t_val in T:
                    expr = f.subs({t: t_val, **{var: val for var, val in zip(K_var, K_val_i)}})
                    # Sécurité : si l'expression contient zoo, oo, -oo, nan → on arrête
                    if expr.has(sy.zoo) or expr.has(sy.oo) or expr.has(-sy.oo) or expr.has(sy.nan):
                        return False
                    val = expr.evalf()
                    if val.is_infinite or not val.is_real:
                        return False
                    vals.append(val)
                return vals
            f_eval_i = func_timeout(max_compute, compute)
            if f_eval_i is False:
                return False
            f_eval.append(f_eval_i)
        return f_eval
    except FunctionTimedOut:
        return False

def ecart(f,i_var):
    # Calcule l'écart au carré entre la fonction f et les données expérimentales
    f_eval = eval(f)
    if f_eval is False:
        return [float('inf')]*2  # Retourne une valeur infinie si l'évaluation a échoué
    ecart_total, L_ec = 0, []
    for i in range(len(C)):
        C_i = C[i]
        ecart_i = sum((f_eval[i][j] - C_i[j][i_var])**2 for j in range(len(T)))
        L_ec.append(ecart_i)
        ecart_total += ecart_i
    return (ecart_total, ec_type(L_ec))


#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#
#                       MUTATION                            #
#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#

def rd_cst():#rd cst
    return rd.choice([t,*K_var,sy.Float(rd.uniform(-10, 10))])

def mutation(f, memory, max_try = 20):
    i = 0
    new_f = f
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
    return new_f

def change_cst(f):  # Modifie aléatoirement les cst(numériques) d'une fonction
    CSTES = list(f.atoms(sy.Float, sy.Integer))
    c = rd.choice(CSTES)
    new_c = sy.Float(c*(1 + rd.uniform(-1, 1)))  # Ajuste la constante aléatoirement
    return f.xreplace({c : new_c})

OP_1 = [sy.cos, sy.sin, sy.tan, sy.Abs]
OP_2 = [sy.Add, sy.Mul]
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
    return (moy(L_2)-moy(L)**2)**(1/2)

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

def fit_insert(new_f, population, i_var, memory, old_f_fit = None):
    if validate_function(new_f):  # Vérifie si la nouvelle fonction est valide
        fit = fitness(new_f, i_var)
        
        if memory != None:  # Pour les mutations structurel
            memory.append(stucture(new_f))  #  Rajout de la nouvelle structure à la mémoire
        
        if old_f_fit != None :   # Pour les mutations numérique
            if fit < old_f_fit[1] and old_f_fit in population:  # si f est meilleur que son parent on supprime celui-ci (pour éviter d'avoir des mêmes structures dans la population)
                population.remove(old_f_fit)
            else:# si f est pire que son parent on ne le prend pas en compte
                return population
            
        if isinstance(fit, (int, sy.Float)) and abs(fit) < float('inf'):  # Vérifie si l'écart est un nombre réel
            population = insert_sorted(population, (new_f, fit))  # Insère la nouvelle fonction dans la population triée 
    return population

def evolution_one_var(i_var):
    # Initialisation de la population
    memory,population = [],[]  # Mémoire pour stocker les individus déjà évalués, population pour stocker les individus triés par écart
    while len(population)<n_voisin_S :  # Continue jusqu'à ce que la population atteigne la taille souhaitée
        f = ini_f()
        while stucture(f) in memory:
            f = ini_f()  # Génère une nouvelle fonction aléatoire
        population = fit_insert(f, population, i_var, memory)
    print("start evolution")
    # Boucle d'évolution
    gen = 0  # Compteur de générations
    a = time.time()
    while gen<max_gen and ecart(population[0][1],i_var)[0]>eps:  # Continue tant que le nombre de générations est inférieur à max_gen et que l'écart du meilleur individu est supérieur à eps
        f_fits = rd.choices(population, weights=[1/(i+1) for i in range(len(population))], k=min(select, len(population)))  # Sélectionne des individu avec une probabilité décroissante
        
        for f_fit in f_fits:
            f = f_fit[0]
            # Génération par mutation de nouveaux individus :
            
            # - Structurels
            for _ in range(n_voisin_S):
                new_f = mutation(f, memory)
                if new_f == None:
                    continue
                population = fit_insert(new_f, population, i_var, memory)
                    
            # - numériques
            for _ in range(n_voisin_N):
                if list(f.atoms(sy.Float, sy.Integer)): # Si la fonction a des cst(s) numérique(s)
                    new_f = change_cst(f)
                    population = fit_insert(new_f, population, i_var, None, f_fit)
        
        gen += 1  # Incrémente le compteur de générations
        if gen % 10 == 0:  # Affiche l'état de la population tous les 10 générations
            population = population[:max_memory]    # Limite la taille de la population et memory
            memory = memory[:max_memory]
            print(f"Generation {gen} : Durée {time.time()-a} : Best function : {population[0][0]}, Fitness : {population[0][1]}")
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

K_val = [(1,0.35),(0.7,0.31),(0.6,0.31)]
T = [0.0, 0.1339556, 0.2678111, 0.4017222, 0.5356333, 0.6695444, 0.8034556, 0.9374778, 1.071389, 1.2053, 1.339211, 1.473233, 1.607144, 1.741056, 1.874967, 2.008878, 2.142922, 2.276878]
C1 = [(0.3354059, -0.9193614), (0.2980415, -0.9314253), (0.2059416, -0.958923), (0.08786583, -0.9746665), (-0.04630045, -0.9808834), (-0.1801937, -0.9642491), (-0.2728376, -0.9413672), (-0.3141366, -0.928531), (-0.3035328, -0.9335538), (-0.2354452, -0.9530872), (-0.1205178, -0.9724283), (0.01319871, -0.979944), (0.1468501, -0.9675977), (0.2562368, -0.9435996), (0.3204178, -0.9218338), (0.3343702, -0.9162529), (0.2964197, -0.9324377), (0.2121473, -0.9553196)]
C2 = [(0.2136436, -0.6739894), (0.1707408, -0.6871742), (0.0849351, -0.7032889), (-0.02054311, -0.7083117), (-0.12058, -0.6965919), (-0.1839925, -0.6819421), (-0.2049207, -0.6760822), (-0.1741562, -0.6846628), (-0.1017446, -0.7009868), (-0.000661304, -0.7091488), (0.1012591, -0.7028704), (0.1791121, -0.6859185), (0.2165736, -0.674408), (0.2052724, -0.6779658), (0.1504404, -0.6951269), (0.05730986, -0.7064282), (-0.04774979, -0.7093581), (-0.1385783, -0.6944991)]
C = [C1,C2]

teta0 = sy.symbols('teta0',real=True)
l = sy.symbols('l',real=True)

K_var = [l, teta0]

F_sol = [l*sy.sin(teta0*sy.cos(sy.sqrt(9.81/l)*t)),-l*sy.cos(teta0*sy.cos(sy.sqrt(9.81/l)*t))]

F = [(t*teta0)**5.03777002541506, l*teta0*(sy.sin(t) - 3.55200841525158)]

F = [-0.209479169543627*sy.sin(3.49129263616927*sy.cos(t))*l, teta0*(0.064042431187835*t - 3.39002876658668)*sy.sin(l)]

evolution()
