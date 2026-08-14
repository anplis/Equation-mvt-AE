import matplotlib.pyplot as plt
import sympy as sy
import random as rd
from func_timeout import func_timeout, FunctionTimedOut 
import time as time

t = sy.symbols('t')

max_memory = 1000
max_gen = 100
n_voisin = 30
select = 5
eps = 50


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
                    # Si evalf retourne ComplexInfinity → on arrête
                    if val.is_infinite:
                        return False
                    vals.append(val)
                return vals
            f_eval_i = func_timeout(1, compute)
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
        return float('inf')  # Retourne une valeur infinie si l'évaluation a échoué
    ecart_total = 0
    for i in range(len(C)):
        C_i = C[i]
        ecart_i = sum((f_eval[i][j] - C_i[j][i_var])**2 for j in range(len(T)))
        ecart_total += ecart_i
    return ecart_total


#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#
#                       MUTATION                            #
#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#

def rd_cst():#rd cst
    return rd.choice([t,*K_var,sy.Float(rd.uniform(-10, 10))])


def mutation(f, n):
    for _ in range(n):
        try:
            # Applique une mutation à la fonction f
            mut = rd.choices([opti_cst, change_node, del_op, swap_child, extract], weights=[1, 1, 0.3, 0.5, 0.2])[0]
            f = func_timeout(1, lambda: mut(f))  # Timeout de 1 seconde pour la mutation
        except FunctionTimedOut:
            return f  # Retourne la fonction originale si la mutation prend trop de temps
    return f
    

def opti_cst(f):
    # Optimise les constantes de la fonction f
    CSTES = [c for c in f.atoms(sy.Number)]
    if CSTES:
        c = rd.choice(CSTES)
        new_c = sy.Float(c*(1 + rd.uniform(-1, 1)))  # Ajuste la constante aléatoirement
        return f.xreplace({c : new_c})
    else:#no cste in f
        return f

OP_1 = [sy.exp, sy.log, sy.cos, sy.sin, sy.tan, sy.Abs]
OP_2 = [sy.Add, sy.Mul]#pow est source de beaucoup de temps de calcul
OP = {1 : OP_1, 2  :OP_2}

def change_node(f):
    # Change un nœud de la fonction f
    nodes = list(sy.preorder_traversal(f))
    if not nodes:
        return rd_cst()
    
    node = rd.choice(nodes)
    n_arg = rd.choice([0, 1, 2])#nb of args of the new node
    if n_arg == 0:
        new_node = rd_cst()
        return f.xreplace({node : new_node})
    new_node = rd.choice(OP[n_arg])
    node_arg = len(node.args)
    if node_arg == 0:
        return f.xreplace({node : new_node(*[rd_cst() for _ in range(n_arg)])})
    new_arg_needed = max(0, n_arg-node_arg)
    new_args = rd.sample(list(node.args), min(n_arg, len(node.args))) + [rd_cst() for _ in range(new_arg_needed)]
    return f.xreplace({node : new_node(*new_args)})

def del_op(f):
    # Supprime un opérateur de la fonction f
    ops = list(f.atoms(sy.Add, sy.Mul, sy.Pow))
    if not ops :
        return f
    op = rd.choice(ops)
    if len(op.args) <= 1:
        return f
    new_arg = rd.choice(op.args)
    return f.xreplace({op : new_arg})

def extract(f):
    # Extrait un sous-arbre de la fonction f
    nodes = list(f.atoms(sy.Function, sy.Add, sy.Mul, sy.Pow))
    if nodes:
        node = rd.choice(nodes)
        return node
    else:
        return f

def swap_child(f):
    # Échange deux enfants d'un nœud de la fonction f
    nodes = list(f.atoms(sy.Add, sy.Mul, sy.Pow))
    if nodes:
        node = rd.choice(nodes)
        new = node.func(*node.args[::-1])
        return f.xreplace({node : new})
    return f


def validate_function(f):
    # Vérifie si la fonction f est valide (pas de division par zéro, pas de logarithme négatif, etc.)
    if f is None:
        #print("Fonction invalide : est None.")
        return False
    if f.has(sy.nan) or f.has(sy.zoo) or f.has(sy.oo) or f.has(-sy.oo):
        #print("Fonction invalide : contient NaN ou infini.")
        return False
    if f.is_real is False:
        #print("Fonction invalide : n'est pas réelle.")
        return False
    return True

#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#
#                       MAIN                                #
#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#

def new_valid_f():
    # Génère une fonction aléatoire définie pour tout réel
    c1 = rd_cst()
    c2 = rd_cst()
    if rd.random() < 0.5:
        op1 = rd.choice([sy.cos, sy.sin, sy.tan])
        return rd.choice([sy.Add, sy.Mul])(c1, op1(c2))
    else:
        return rd.choice([sy.Add, sy.Mul])(c1, c2)

def add_memory(memory, f):
    # Ajoute la fonction f à la mémoire si elle n'y est pas déjà
    if f not in memory:
        memory.append(f)
        if len(memory) > max_memory:
            memory.pop(0)  # Supprime le plus ancien élément si la mémoire est pleine

def insert_sorted(population, new_individual):
    k = 0
    while k<len(population) and new_individual[1]>=population[k][1]:  # trouve l'emplacement adapté du nouvel élément et l'insert
        k+=1
    return population[:k] + [new_individual] + population[k:] 

def count_nodes(f):
    return 1 + sum(count_nodes(arg) for arg in f.args)


def evolution_one_var(i_var):
    # Initialisation de la population
    memory,population = [],[]  # Mémoire pour stocker les individus déjà évalués, population pour stocker les individus triés par écart
    while len(population)<n_voisin:  # Continue jusqu'à ce que la population atteigne la taille souhaitée
        f = new_valid_f()  # Génère une nouvelle fonction aléatoire
        add_memory(memory, f)
        if validate_function(f):
            ec = ecart(f, i_var)  # Calcule l'écart de la nouvelle fonction
            if isinstance(ec, (int, sy.Float)) and abs(ec) < float('inf'):  # Vérifie si l'écart est un nombre réel:
                population = insert_sorted(population, (f, ec))  # Insère la nouvelle fonction dans la population triée
    
    # Boucle d'évolution
    g = 0  # Compteur de générations
    while g<max_gen and population[0][1]>eps:  # Continue tant que le nombre de générations est inférieur à max_gen et que l'écart du meilleur individu est supérieur à eps
        f_fits = rd.choices(population, weights=[1/(i+1) for i in range(len(population))], k=min(select, len(population)))  # Sélectionne des individu avec une probabilité décroissante
        for f_fit in f_fits:
            # Génération de nouveaux individus par mutation
            for _ in range(n_voisin):
                new_f = mutation(f_fit[0], 1)  # Applique une mutation au meilleur individu
                m = 0  # Compteur pour les fonctions déjà dans la mémoire
                n_mut = 1 # Nombre de mutations consécutives
                while new_f in memory and m<50:
                    if m>10:
                        n_mut += 1
                    m += 1
                    new_f = mutation(f_fit[0], n_mut)
                if m>15:
                    print("nombre d'accés à mémoire : ", m)
                if new_f not in memory:  # Vérifie si la nouvelle fonction n'est pas déjà dans la mémoire
                    add_memory(memory, new_f)  # Ajoute la nouvelle fonction à la mémoire
                    if validate_function(new_f):  # Vérifie si la nouvelle fonction est valide
                        a = time.time()
                        ec = ecart(new_f, i_var)  # Calcule l'écart de la nouvelle fonction
                        b = time.time()
                        if b-a>0.5:
                            print(b-a,"écart", new_f)
                        if isinstance(ec, (int, sy.Float)) and abs(ec) < float('inf'):  # Vérifie si l'écart est un nombre réel
                            population = insert_sorted(population, (new_f, ec+count_nodes(new_f)))  # Insère la nouvelle fonction dans la population triée

        g += 1  # Incrémente le compteur de générations
        print(g)
        if g % 10 == 0:  # Affiche l'état de la population tous les 10 générations
            population = population[:max_memory]    # Limite la taille de la population à max_memory
            print(f"Generation {g}: Best function : {population[0][0]}, Ecart : {population[0][1]}")
    print(m, "fonctions déjà dans la mémoire")
    return population[0]  # Retourne le meilleur individu après l'évolution

def evolution():
    F = [evolution_one_var(i_var)[0] for i_var in range(len(C[0][0]))]  # Évolue pour chaque variable meusurée
    print(F)
    return trace(F)

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
        repeat=True,   # <-- boucle infinie
        blit=True,
    )

    plt.tight_layout()
    plt.show()

    return ani


#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#
#                       DATA                                #
#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#

K_val = [(10, 0.602),(20, 0.586),(30,0.492),(40,0.511)]
C1 = [(-0.852, -1.24), (-0.612, -1.355), (-0.133, -1.454), (0.403, -1.421), (0.799, -1.256), (0.849, -1.24), (0.585, -1.372), (0.3996, -1.446), (-0.472, -1.397), (-0.785, -1.256), (-0.802, -1.256), (-0.53, -1.372), (-0.009563, -1.479), (0.535, -1.388), (0.849, -1.248), (0.849, -1.248), (0.486, -1.413), (-0.02607, -1.471), (-0.554, -1.397), (-0.802, -1.223), (-0.752, -1.256), (-0.431, -1.372), (0.08949, -1.471), (0.585, -1.38), (0.882, -1.24), (0.816, -1.265), (0.395, -1.413), (-0.142, -1.438), (-0.604, -1.355), (-0.802, -1.24), (-0.703, -1.281), (-0.331, -1.397), (0.189, -1.454), (0.684, -1.331), (0.874, -1.223), (0.75, -1.281), (0.296, -1.454), (-0.241, -1.438), (-0.662, -1.306)]

C2 = [(-1.672, -2.519), (-1.522, -2.602), (-1.023, -2.844), (-0.315, -3.002), (0.559, -2.985), (1.233, -2.769), (1.716, -2.519), (1.783, -2.436), (1.633, -2.536), (1.192, -2.769), (0.459, -2.96), (-0.407, -2.969), (-1.114, -2.769), (-1.547, -2.577), (-1.68, -2.478), (-1.464, -2.602), (-0.981, -2.819), (-0.249, -3.002), (0.692, -2.952), (1.35, -2.727), (1.716, -2.494), (1.766, -2.419), (1.6, -2.552), (1.092, -2.802), (0.284, -2.96), (-0.515, -2.969), (-1.173, -2.794), (-1.564, -2.586), (-1.655, -2.511), (-1.422, -2.669), (-0.865, -2.877), (-0.107, -3.002), (0.75, -2.927), (1.408, -2.669), (1.733, -2.486), (1.766, -2.428), (1.55, -2.586), (0.984, -2.827), (0.101, -2.994)]

C3 = [(-2.099, -3.919), (-1.968, -3.965), (-1.543, -4.148), (-0.921, -4.312), (-0.181, -4.397), (0.643, -4.364), (1.35, -4.168), (1.827, -3.965), (2.05, -3.827), (2.011, -3.84), (1.703, -4.017), (1.206, -4.213), (0.401, -4.384), (-0.443, -4.338), (-1.071, -4.253), (-1.719, -4.076), (-2.007, -3.952), (-2.079, -3.899), (-1.817, -4.004), (-1.346, -4.174), (-0.666, -4.318), (0.185, -4.377), (0.898, -4.253), (1.559, -4.063), (1.939, -3.899), (2.083, -3.827), (1.952, -3.88), (1.592, -4.063), (0.957, -4.285), (0.139, -4.377), (-0.659, -4.318), (-1.333, -4.148), (-1.817, -3.978), (-2.02, -3.886), (-1.981, -3.88), (-1.673, -4.03), (-1.13, -4.226), (-0.365, -4.338), (0.453, -4.377)]

C4 = [(-2.854, -5.095), (-2.724, -5.179), (-2.315, -5.347), (-1.626, -5.588), (-0.752, -5.737), (0.197, -5.765), (1.174, -5.644), (1.946, -5.421), (2.597, -5.151), (2.932, -4.965), (2.951, -4.937), (2.746, -5.086), (2.244, -5.337), (1.481, -5.551), (0.588, -5.719), (-0.435, -5.756), (-1.31, -5.654), (-2.054, -5.402), (-2.566, -5.179), (-2.798, -5.058), (-2.761, -5.095), (-2.454, -5.263), (-1.812, -5.495), (-1.012, -5.7), (-0.04462, -5.802), (0.96, -5.644), (1.76, -5.505), (2.458, -5.216), (2.876, -4.993), (3.007, -4.9), (2.867, -5.012), (2.467, -5.254), (1.769, -5.533), (0.886, -5.654), (-0.1, -5.747), (-1.04, -5.654), (-1.84, -5.449), (-2.398, -5.244), (-2.705, -5.086)]

T = [0, 0.067, 0.133, 0.2, 0.267, 0.333, 0.4, 0.467, 0.533, 0.6, 0.667, 0.733, 0.8, 0.867, 0.933, 1, 1.067, 1.133, 1.2, 1.267, 1.333, 1.4, 1.467, 1.533, 1.6, 1.667, 1.733, 1.8, 1.867, 1.933, 2, 2.067, 2.133, 2.2, 2.267, 2.333, 2.4, 2.467, 2.533]

C = [C1,C2,C3,C4]
l = sy.symbols('l',real=True)
teta0 = sy.symbols('teta0',real=True)
K_var = [l,teta0]
