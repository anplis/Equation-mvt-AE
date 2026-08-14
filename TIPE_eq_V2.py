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
w_node = 1 #    coef du malus de la taille de la fonction 
w_cst = 10 #    coef du malus si la fonction est constante
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

def check_cst(f):
    if t in f.atoms(sy.Symbol):
        return 0
    return 1

def fitness(f,i_var):
    a = time.time()
    ec = ecart(f, i_var)  # Calcule l'écart de la nouvelle fonction
    b = time.time()
    if b-a>0.5:
        print(b-a,"time used for : écart", f)
        
    return ec + w_node*count_nodes(f) + w_cst*check_cst(f)

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
            fit = fitness(f, i_var)
            if isinstance(fit, (int, sy.Float)) and abs(fit) < float('inf'):  # Vérifie si l'écart est un nombre réel:
                population = insert_sorted(population, (f, fit))  # Insère la nouvelle fonction dans la population triée
    
    # Boucle d'évolution
    gen = 0  # Compteur de générations
    while gen<max_gen and ecart(population[0][1],i_var)>eps:  # Continue tant que le nombre de générations est inférieur à max_gen et que l'écart du meilleur individu est supérieur à eps
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
                if m>25:
                    print("nombre d'accés à mémoire : ", m)
                
                if validate_function(new_f):  # Vérifie si la nouvelle fonction est valide
                    if new_f not in memory:  # Vérifie si la nouvelle fonction n'est pas déjà dans la mémoire
                        add_memory(memory, new_f)  # Ajoute la nouvelle fonction à la mémoire
                        fit = fitness(new_f)
                        if isinstance(fit, (int, sy.Float)) and abs(fit) < float('inf'):  # Vérifie si l'écart est un nombre réel
                                population = insert_sorted(population, (new_f, fit))  # Insère la nouvelle fonction dans la population triée        
        gen += 1  # Incrémente le compteur de générations
        print(gen)
        if gen % 10 == 0:  # Affiche l'état de la population tous les 10 générations
            population = population[:max_memory]    # Limite la taille de la population à max_memory
            print(f"Generation {gen}: Best function : {population[0][0]}, Ecart : {population[0][1]}")
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

K_val = [(9.81,1,0.35),(9.81,0.7,0.31),(9.81,0.6,0.31)]
T = [0.0, 0.03348889, 0.06697778, 0.1004667, 0.1339556, 0.1674444, 0.2008333, 0.2342222, 0.2678111, 0.3012889, 0.3347667, 0.3682444, 0.4017222, 0.4352, 0.4686778, 0.5021556, 0.5356333, 0.5691111, 0.6025889, 0.6360667, 0.6695444, 0.7030222, 0.7365, 0.7699778, 0.8034556, 0.8369333, 0.8705222, 0.904, 0.9374778, 0.9709556, 1.004433, 1.037911, 1.071389, 1.104867, 1.138344, 1.171822, 1.2053, 1.238778, 1.272256, 1.305733, 1.339211, 1.372689, 1.406278, 1.439756, 1.473233, 1.506711, 1.540189, 1.573667, 1.607144, 1.640622, 1.6741, 1.707578, 1.741056, 1.774533, 1.808011, 1.841489, 1.874967, 1.908444, 1.941922, 1.9754, 2.008878, 2.042356, 2.075944, 2.109433, 2.142922, 2.176411, 2.2099, 2.243389, 2.276878, 2.310367, 2.343856, 2.377344]
C1 = [(0.3354059, -0.9193614), (0.3298711, -0.9222184), (0.3225056, -0.9245859), (0.3117204, -0.9285317), (0.2980415, -0.9314253), (0.2771837, -0.939746), (0.2591337, -0.9455007), (0.2365146, -0.9512176), (0.2059416, -0.958923), (0.180889, -0.9640353), (0.1481094, -0.968465), (0.1179876, -0.9720087), (0.08786583, -0.9746665), (0.05420029, -0.9790962), (0.01876288, -0.981754), (-0.01247987, -0.9811966), (-0.04630045, -0.9808834), (-0.08134781, -0.9782102), (-0.119443, -0.9728946), (-0.152289, -0.9687139), (-0.1801937, -0.9642491), (-0.2097728, -0.9553196), (-0.2332128, -0.9502967), (-0.2533042, -0.9452738), (-0.2728376, -0.9413672), (-0.2867899, -0.9363443), (-0.2985099, -0.9329958), (-0.3079976, -0.9296472), (-0.3141366, -0.928531), (-0.3169271, -0.9274148), (-0.3152528, -0.9279729), (-0.310788, -0.9296472), (-0.3035328, -0.9335538), (-0.2918128, -0.9374605), (-0.2767442, -0.9419253), (-0.2588852, -0.9469481), (-0.2354452, -0.9530872), (-0.2114471, -0.9586681), (-0.1835423, -0.9659234), (-0.1556375, -0.9687139), (-0.1205178, -0.9724283), (-0.08982879, -0.9783782), (-0.05694767, -0.9793177), (-0.02156133, -0.979944), (0.01319871, -0.979944), (0.04695108, -0.9787596), (0.0809949, -0.9765272), (0.1161549, -0.9731786), (0.1468501, -0.9675977), (0.1786616, -0.9625748), (0.2065663, -0.9569939), (0.2311225, -0.9502967), (0.2562368, -0.9435996), (0.2757702, -0.9385767), (0.2936292, -0.9324377), (0.3092559, -0.9274148), (0.3204178, -0.9218338), (0.327673, -0.9184853), (0.3349283, -0.9162529), (0.3354864, -0.9162529), (0.3343702, -0.9162529), (0.3315797, -0.9201596), (0.3226502, -0.9235081), (0.3126045, -0.9268567), (0.2964197, -0.9324377), (0.280793, -0.9385767), (0.2595854, -0.9435996), (0.2361454, -0.9497386), (0.2121473, -0.9553196), (0.1859168, -0.9625748), (0.1563378, -0.9687139), (0.1217359, -0.9720624)]
C2 = [(0.2136436, -0.6739894), (0.2073652, -0.6760822), (0.1994125, -0.6790122), (0.1847627, -0.6827793), (0.1707408, -0.6871742), (0.1519054, -0.6919877), (0.1295122, -0.6965919), (0.1108861, -0.7001497), (0.0849351, -0.7032889), (0.06065837, -0.7066374), (0.03177743, -0.7091488), (0.00457075, -0.7095674), (-0.02054311, -0.7083117), (-0.04712195, -0.7072653), (-0.07244509, -0.7047539), (-0.09546613, -0.7024518), (-0.12058, -0.6965919), (-0.1402525, -0.6934527), (-0.1565765, -0.6884299), (-0.1720634, -0.6848721), (-0.1839925, -0.6819421), (-0.1929916, -0.6802679), (-0.198433, -0.6777565), (-0.2038743, -0.6765008), (-0.2049207, -0.6760822), (-0.2017815, -0.6771287), (-0.195503, -0.6792215), (-0.1867132, -0.6817329), (-0.1741562, -0.6846628), (-0.1607622, -0.6886392), (-0.1419268, -0.6930341), (-0.1239285, -0.697429), (-0.1017446, -0.7009868), (-0.07684001, -0.7037075), (-0.05319113, -0.707056), (-0.02640301, -0.7087303), (-0.000661304, -0.7091488), (0.0250804, -0.7097767), (0.05354278, -0.7074746), (0.0790752, -0.7045446), (0.1012591, -0.7028704), (0.1265823, -0.6978476), (0.1456269, -0.6942898), (0.1655087, -0.689267), (0.1791121, -0.6859185), (0.1937618, -0.6819421), (0.2048538, -0.6779658), (0.210923, -0.675873), (0.2165736, -0.674408), (0.2178293, -0.6737801), (0.2153179, -0.6748265), (0.2117601, -0.6762915), (0.2052724, -0.6779658), (0.1964825, -0.681105), (0.1841349, -0.6857092), (0.1678108, -0.6901041), (0.1504404, -0.6951269), (0.1301401, -0.6993126), (0.1083747, -0.7018239), (0.08347013, -0.7030796), (0.05730986, -0.7064282), (0.032196, -0.708521), (0.005617161, -0.7087303), (-0.02075239, -0.7093581), (-0.04774979, -0.7093581), (-0.07411935, -0.7066374), (-0.09546613, -0.7016147), (-0.1186964, -0.6978476), (-0.1385783, -0.6944991), (-0.1544837, -0.6901041), (-0.1678778, -0.6869649), (-0.1814811, -0.6827793)]
C = [C1,C2]

teta0 = sy.symbols('teta0',real=True)
l = sy.symbols('l',real=True)
g = sy.symbols('g',real=True)

K_var = [g, l, teta0]

F_sol = [l*sy.sin(teta0*sy.cos(sy.sqrt(9.81/l)*t)),-l*sy.cos(teta0*sy.cos(sy.sqrt(9.81/l)*t))]
