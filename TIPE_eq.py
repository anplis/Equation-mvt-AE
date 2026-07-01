import math as ma
import matplotlib.pyplot as plt
import sympy as sy
import random as rd
import copy as copy

t = sy.symbols('t',real = True)

"si plus de que un paramètre(k) modifier les lignes 33"

#C = [Ci] liste des trajectoires
#F = [f,g] les deux fonctions pour x et y qui approximent C
#P = [Fi] liste des individus
#K_var = [k_var_i] paramétres qui varient type(k_var_i) = sy.Symbol
#K_val = [k_val_i] valeurs des paramétres type(k_val_i) = list
#eps l'écart minimal à la trajéctoire
#g_max nb maximal de générations
#unchg nb d'individus inchangés entre 2 générations
#select nb d'individus séléctionnés pour la prochaines génération
#n_P nb d'individus
#m_prob la proba qu'un individu subisse une mutation
#m_r l'amplitude de la mutation
#range_k l'amplitude des valeur des cstes

## modélisation

#liste des valeurs de fonctions pour x valeurs de temps dans X
def value(Ki,f):
    if f is None:
        return False
    V=[]
    for x in T:
        val = f.evalf(subs={t:x,L:Ki[0],x0:Ki[1],y0:Ki[2]})
        if not valide(val):
            return False
        V.append(val)
    return V

# validité d'une fonction sur un l'intervale de temps
def valide(val):
    return type(val) is not sy.core.numbers.NaN and float(sy.im(val))==0 and val.is_finite

# somme des écarts quadratiques aux points d'une trajéctoire
def ecart(Ci,Ki,f,var):
    X = [x[var] for x in Ci]#coord x ou y
    A = value(Ki,f)
    if A==False:
        return False
    return sum((x-a)**2 for x,a in zip(X,A))

# somme des écarts quadratiques aux points des trajéctoires
def ecart_tot(f,var):
    e = 0
    for Ci,Ki in zip(C,K_val):
        ec = ecart(Ci,Ki,f,var)
        if ec==False:
            return False
        e += ec
    return e

## graphique

# trajéctoire et la courbe l'approximent
def trace(i_exp,F):
    fig, ax = plt.subplots()
    Ci,Ki = C[i_exp],K_val[i_exp]
    A,B = [value(Ki,f) for f in F]
    X,Y = [[x[i] for x in Ci] for i in [0,1]]

    ax.plot(X,Y,label='traj',color='black')
    ax.plot(A,B,label='F',color='blue')
    ax.scatter(X, Y, color='black', s=5)
    ax.scatter(A, B, color='blue', s=5)
    ax.set_aspect('equal',adjustable='datalim')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.margins(0.1)
    plt.title('approximation F de la trajéctoire')
    plt.legend()
    plt.show()

from matplotlib.animation import FuncAnimation

def trace_anim(i_exp):
    X, Y = [[x[i] for x in C[i_exp]] for i in [0,1]]
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='datalim')
    ax.plot(X, Y, color='gray', alpha=0.4)

    point, = ax.plot([], [], 'ro', markersize=6)

    def init():
        point.set_data([], [])
        return point,

    def update(i):
        point.set_data([X[i]], [Y[i]])
        return point,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(X),
        init_func=init,
        interval=40,
        repeat=False)
    plt.show()

def trace_anim2(i_exp,F):
    # Trajectoire 1
    X1, Y1 = [[x[i] for x in C[i_exp]] for i in [0,1]]

    # Trajectoire 2 (par exemple F)
    Ki = K_val[i_exp]
    A, B = [value(Ki, f) for f in F]   # A = X2, B = Y2
    X2, Y2 = A, B

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='datalim')

    # Traces statiques
    ax.plot(X1, Y1, color='gray', alpha=0.4, label="traj 1")
    ax.plot(X2, Y2, color='lightblue', alpha=0.4, label="traj 2")

    # Points animés
    p1, = ax.plot([], [], 'ro', markersize=6)   # point rouge
    p2, = ax.plot([], [], 'bo', markersize=6)   # point bleu

    def init():
        p1.set_data([], [])
        p2.set_data([], [])
        return p1, p2

    def update(i):
        # IMPORTANT : toujours passer des listes
        p1.set_data([X1[i]], [Y1[i]])
        p2.set_data([X2[i]], [Y2[i]])
        return p1, p2

    # Nombre de frames = min des deux longueurs
    n_frames = min(len(X1), len(X2))

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=40,
        repeat=False)

    plt.legend()
    plt.show()


## modification génétique

def gauss(x):
    return ma.exp(-(x/n_P)**2)

def mutation(f):
    if f is None:
        return cst()
    mut = rd.choices([add_op,del_op,swap_op,change_cst,swap_child],weights=[1,1,5,5,5])[0]
    f_mut = mut(f)
    if valide_expr(f_mut):
        return f_mut
    return f

def parcour(f,target_index,i=0):#Si i!∈[0,nb_node_f(f)-1] parcour=None
    if i == target_index:
        return f
    for arg in f.args:
        i += 1
        res = parcour(arg,target_index,i)
        if res is not None:
            return res
    return None

def valide_expr(expr):
    if expr is None:
            return False
    if expr.has(sy.nan) or expr.has(sy.zoo) or expr.has(sy.oo) or expr.has(-sy.oo):
        return False
    if sy.im(expr) != 0:
        return False
    return True

#ajout d'une opération entre une partie de f et une cste k (comme définie dans cst)
def add_op(f):
    expr = parcour(f,rd.randrange(0,nb_node(f)))
    if expr==None:
        return f
    type = rd.randint(1,2)
    op = rd.choice(OP[type])
    if type==2 :
        new = op(expr,cst())
        return f.subs(expr,new)
    else:
        new = op(expr)
        return f.subs(expr,new)

#cste ∈ [t, paramètres, Réel∈[-range_k,range_k]]
def cst(old=0):
    if old != 0:
        new = old*(1/range_k)#arbitraire
    else:
        new = sy.Float(rd.uniform(-range_k,range_k))
    new_cst = rd.choice([t,*K_var,new])
    sign = rd.choice([sy.Integer(-1),sy.Integer(1)])
    return sy.Mul(sign,new_cst)

#modifie un opérateur ex: + -> x
#choisi une fonction parmis f qui soit une opération entre deux fonctions et la substitue par une autre opération
def swap_op(f):
    EXPR = [exp for exp in sy.postorder_traversal(f) if len(exp.args)>=2]
    if not EXPR:
        return f
    expr = rd.choice(EXPR)
    n = len(expr.args)
    l = rd.choice([1,2])
    op = rd.choice(OP[l])
    if n==l:
        return f.subs(expr,op(*expr.args))
    elif n==1 and l>=2:
        new = [cst() for _ in range(l-n)]
        return f.subs(expr,op(*expr.args,*new))
    elif n==2 and l == 1:
        return f.subs(expr,op(rd.choice(expr.args)))

#modifie valeur d'une constante aléatoirement choisie
def change_cst(f):
    EXPR = [ex for ex in sy.postorder_traversal(f) if not ex.args]
    return f.subs(rd.choice(EXPR),cst())

OP_1 = [sy.exp,sy.ln,sy.cos,sy.sin,sy.tan,sy.acos,sy.asin,sy.atan]
OP_2 = [sy.Add,sy.Mul,sy.Pow]
OP = {1:OP_1,2:OP_2}

#nombre d'expression en entré de l'opérateur(ex: +:2, cos:1)
def type_op(expr):
    for OP_n,n in [(OP_1,1),(OP_2,1)]:
        for op in OP_n:
            if type(expr) is op:
                return n
    return 0

#supprime une partie de la fonction
def del_op(f):
    expr = parcour(f,rd.randrange(0,nb_node(f)))
    if expr==None:
        return f
    if len(expr.args)>1:
        return f.subs(expr,rd.choice(expr.args))
    return f

def swap_child(f):
    EXPR = [exp for exp in sy.postorder_traversal(f) if len(exp.args)>=2]
    if len(EXPR)>=1:
        expr = rd.choice(EXPR)
        return expr.func(*expr.args[::-1])
    return f

## évolution

def nb_node(f):
    if not f.args:
        return 1
    else:
        return sum(nb_node(arg) for arg in f.args) + 1

# évaluation d'une approximation
def fitness(f,var):
    ec_tot = ecart_tot(f,var)
    if ec_tot==False:
        return float('inf')
    return ec_tot + c*nb_node(f)

def ini_P(var):#P = [(F,fit(F), ...]
    ini_P = []
    for _ in range(n_P):
        f = cst()
        ini_P.append((f,fitness(f,var)))
    return ini_P

#P[0][0] = F
#P[0] = (f,fit(f))
def evolution():
    F,FIT = [],[]
    W = [gauss(i) for i in range(n_P)]
    for var in [0,1]:
        i=1
        P = ini_P(var)
        while i<=g_max and ecart_tot(P[0][0],var)>eps:
            i += 1
            if i%20==0:
                print(i,"-ième génération")
                print(P[0][0],",",P[0][1],"\n")

            W = [gauss(i) for i in range(len(P))]
            for Pi in rd.choices(P,weights=W,k=baby):
                F_mut = mutation(Pi[0])
                F_mut_fit = fitness(F_mut,var)
                P = insertion(P,(F_mut,F_mut_fit))
            if i%div==0:
                P = P[:n_P]

        F.append(P[0][0])
        FIT.append(P[0][1])
    return F,FIT

#insert un individu en gardant un ordre un ordre croissant (de fitness)
def insertion(P,f_fit):
    k = 0
    while k<len(P) and f_fit[1]>=P[k][1] : # trouve l'emplacement adapté du nouvel élément et on l'insert
        k+=1
    return P[:k] + [f_fit] + P[k:]

## evolution matriciel
import numpy as np

def ini_M(var):
    M = np.zeros((n_M,n_M),dtype=object)
    for i,j in COORD:
        M[i][j] = {'f':cst(),'e':[]}
    for i,j in COORD:
        M[i][j]['adj_co'] = ini_voisin_co(M,i,j)#coord des voisins
        M[i][j]['fit'] = fitness(M[i][j]['f'],var)
    return M

VOISIN_DIR = [(0,1),(0,-1),(1,0),(-1,0)]

#les voisins d'une coord (ortho + diago + lui même)
def ini_voisin_co(M,i,j):
    V = [(i,j)]
    for a,b in VOISIN_DIR:
        if 0<=i+a<n_M and 0<=j+b<n_M :#si dans le matrice
            V.append((i+a,j+b))
    return V

#n_M la taille de la matrice
def evolution_mat():
    F_f = []
    for var in [0,1]:
        M = ini_M(var)
        fit_min,f_best = best_L(np.ravel(M))
        i = 1
        while i<=g_max and ecart_tot(f_best,var)>eps:
            i += 1
            if i%20==0:
                print(i,"-ième génération")
            #Pour chaque individu créé des mutations(enfants) de lui même à tout ses voisins
            for i,j in COORD:
                for i_v,j_v in M[i][j]['adj_co']:
                    e = mutation(M[i_v][j_v]['f'])
                    M[i][j]['e'].append({'f':e,'fit':fitness(e,var)})
            #pour tout individus si son meilleur enfant est meilleur que lui on le remplace
            for i,j in COORD:
                e_fit,e_f = best_L(M[i][j]['e'])
                if e_fit<M[i][j]['fit']:
                    M[i][j]['f'] = e_f
                    M[i][j]['fit'] = e_fit
                M[i][j]['e'] = []

            fit_min,f_best = best_L(np.ravel(M))
        F_f.append((f_best,ecart_tot(f_best,var)))
    return F_f

#l'écart du meilleur individu d'aprés fitness, prend une liste
def best_L(L):
    fit_min, F_best = L[0]['fit'],L[0]['f']
    for F_dict in L:
        if F_dict['fit']<fit_min:
            fit_min = F_dict['fit']
            F_best = F_dict['f']
    return fit_min,F_best

def moy_M(M,pow=1):
    s = 0
    for i,j in COORD:
        s +=M[i][j]['fit']**pow
    return s/(n_M**2)

def var_M(M,COORD,n_M,pow):
    return moy_M(M,2) - moy_M(M)**2

def ecrat_type_M(M):
    return ma.sqrt(var_M(M))


##Paramètres génétiques:

#Paramètres
eps     = 5
g_max   = 100
range_k = 10
c       = 0.1

#evolution
n_P     = 300
div     = 4
baby    = 150

#evolution mat
n_M     = 5
COORD = [(i,j) for i in range(n_M) for j in range(n_M)]
