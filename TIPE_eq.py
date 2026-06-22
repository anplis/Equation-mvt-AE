import math as ma
import matplotlib.pyplot as plt
import sympy as sy
import random as rd
import copy as copy

t = sy.symbols('t',real = True)

"si plus de que un paramètre(k) modifier les lignes 30 et 38 et les jouter en variable sympy"

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
def value(X,Ki,F):
    return [[float(f.evalf(subs={t:x,m:Ki[0],x0:Ki[1],y0:Ki[2]})) for x in X] for f in F]

# validité d'une fonction sur un l'intervale de temps
def valide(X,Ki,F):
    if any(f is None for f in F):
        return False
    for x in X:
        for f in F:
            val = f.subs({t:x,m:Ki[0],x0:Ki[1],y0:Ki[2]})
            if type(val) is sy.core.numbers.NaN or float(sy.im(val))!=0:
                return False
    return True

# somme des écarts aux points d'une trajéctoire
def ecart(Ci,Ki,F):
    X_Y = [[x[i] for x in Ci.values()] for i in [0,1]]
    A_B = value(Ci.keys(),Ki,F)
    e = 0
    for i in [0,1]:
        for a,b in zip(X_Y[i],A_B[i]):
            e += abs(a-b)
    return e

# somme des écarts aux trajéctoires
def ecart_tot(F):
    if not all(valide(Ci.keys(), Ki, F) for Ci, Ki in zip(C, K_val)):
        return float('inf')
    e = 0
    for Ci,Ki in zip(C,K_val):
        e += ecart(Ci,Ki,F)
    return e

## graphique

# trajéctoire et la courbe l'approximent
def trace(i_exp,F):
    Ci,Ki = C[i_exp],K_val[i_exp]
    if not valide(Ci.keys(),Ki,F):
        return "fonctions pas biens définis sur l'intervalle"
    X,Y = [[x[i] for x in Ci.values()] for i in [0,1]]
    A,B = value(Ci.keys(),Ki,F)

    plt.figure()
    plt.plot(X,Y,label='f',color='black')
    plt.plot(A,B,label='g',color='blue')
    plt.margins(0.01)
    plt.title('approximation g de la trajéctoire f')
    plt.legend()
    plt.show()

## modification génétique

def gauss(a,x):
    return ma.exp(-(x/a)**2)

def mutation(F):
    return [mutation_f(f) for f in F]

def mutation_f(f):
    if f is None:
        f = add_op(f)
    f_mut = copy.copy(f)
    if rd.random()<m_prob:
        mut = rd.choices([add_op,del_op,swap_op,change_cst],weights=[1,1,7,10])[0]
        f_mut = mut(f)
    return f_mut

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
    expr = parcour(f,rd.randrange(0,nb_node_f(f)))
    type = rd.randint(1,2)
    op = rd.choice(OP[type])
    if type==2 :
        new = op(expr,cst())
        if valide_expr(new):
            return f.subs(expr,new)
        else:
            return f
    else:#opération à une seule vaiable
        new = op(expr)
        if valide_expr(new):
            return f.subs(expr,new)
        else:
            return f

#cste ∈ [t, paramètres, Réel∈[-range_k,range_k]]
def cst(old=0):
    if old != 0:
        new = old*(1/range_k)#arbitraire
    else:
        new = sy.Float(rd.uniform(-range_k,range_k))
    return rd.choice([t,rd.choice(K_var),new])

#modifie un opérateur ex: + -> x
#choisi une fonction parmis f qui soit une opération entre deux fonctions et la substitue par une autre opération
def swap_op(f):
    EXPR = [exp for exp in sy.postorder_traversal(f) if len(exp.args)>=2]
    if not EXPR:
        return f
    expr = rd.choice(EXPR)
    n = len(expr.args)

    if n in OP:
        op = rd.choice(OP[n])
        return f.subs(expr,op(*expr.args))

#modifie valeur d'une constante aléatoirement choisie
def change_cst(f):
    EXPR = [ex for ex in sy.postorder_traversal(f) if not ex.args]
    return f.subs(rd.choice(EXPR),cst())

OP_1 = [sy.exp,sy.ln,sy.cos,sy.sin,sy.tan,sy.acos,sy.asin,sy.atan,sy.cosh,sy.sinh,sy.tanh]
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
    expr = parcour(f,rd.randrange(0,nb_node_f(f)))
    t = len(expr.args)
    if t==2:#on del un des deux aléatoirement
        return f.subs(expr,rd.choice(expr.args))
    elif t==1:
        return f.subs(expr,expr.args[0])
    else:#Integer
        return f

def creation(P):
    n = len(P)
    W = [gauss(n,i) for i in range(n)]
    return P[:unchg] + [mutation(F) for F in rd.choices(P,weights=W,k=n_P-unchg)]

## évolution

def nb_node(F):
    return sum(nb_node_f(f) for f in F)

def nb_node_f(f):
    if not f.args:
        return 1
    else:
        return sum(nb_node_f(arg) for arg in f.args) + 1

# évaluation d'une approximation
def fitness(F):
    return ecart_tot(F) + nb_node(F)

def selection(P):#tri décroissant en fonction de fitness(x[0])
    FIT_P = sorted([(fitness(F),F) for F in P], key=lambda x: x[0])
    return [FIT_P[i][1] for i in range(n_sel)]

def ini_P():
    F = [t]*2
    return [[add_op(f) for f in F] for _ in range(n_P)]


def evolution(i=0):
    P = ini_P()
    while i<g_max and ecart_tot(P[0])>eps:#P[0]=meilleur
        i += 1
        P = creation(P)
        P = selection(P)
        if i%40==0:
            print(i,"-ième génération")
    return P[0],ecart_tot(P[0])

## evolution matriciel
import numpy as np

def ini_M(COORD,K_var,n_M,m_prob,range_k):
    M = np.zeros((n_M,n_M),dtype=object)
    F = [sy.Integer(0)]*2
    for i,j in COORD:
        M[i][j] = {'F':[add_op(K_var,f,m_prob,range_k) for f in F],'e':[]}
    for i,j in COORD:
        M[i][j]['adj_co'] = ini_voisin(M,n_M,i,j)
        M[i][j]['fit'] = fitness(M[i][j]['F'])
    return M

VOISIN_DIR = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1)]

#les voisins d'une coord (ortho + diago + lui même)
def ini_voisin(M,n_M,i,j):
    V = [M[i][j]]
    for a,b in VOISIN_DIR:
        if 0<=i+a<n_M and 0<=j+b<n_M :#si dans le matrice
            V.append((i+a,j+b))
    return V

#n_M la taille de la matrice
def evolution_mat(C,K_var,K,val,eps,g_max,n_M,select,m_prob,unchg,range_k,i=0):
    COORD = [(i,j) for i in range(n_M) for j in range(n_M)]
    M = ini_M(COORD,K_var,n_M,m_prob,range_k)
    ecart_best,F_best = best_M(M,COORD,C,K_val)
    while i<g_max and ecart_min>eps:
        i += 1
        for i,j in COORD:
            for a_v,b_v in M[i][j]['adj_co']:
                F_v = M[a_v][b_v]
                M[i][j]['e'].append(mutation(K_var,F_v,m_prob,range_k))
        for i,j in COORD:
            e_fit,e_F = best_e(M[i][j]['e'])#meilleur enfant
            if e_fit<M[i][j]['fit']:
                M[i][j]['F'] = e_F
                M[i][j]['fit'] = e_fit
        ecart_best,F_best = best_M(M,COORD,C,K_val)
    print (best_M(M,K_val))
    return M


#l'écart du meilleur individu d'aprés fitness, l'individu
def best_M(M,COORD,C,K_val):
    fit_min, F_best = M[0][0]['fit'],M[0][0]
    for F in (M[lign] for lign in M):
        if F['fit']<fit_min:
            fit_min = F['fit']
            F_best = F
    return ecart_tot(C,K_val,F),F_best

def moy_M(M,COORD,n_M,pow=1):
    s = 0
    for i,j in COORD:
        s +=M[i][j]['fit']**pow
    return s/(n_M**2)

def var_M(M,COORD,n_M,pow):
    return moy_M(M,COORD,n_M,2) - moy_M(M,COORD,n_M)**2

def ecrat_type_M(M,COORD,n_M):
    return ma.sqrt(var_M(M,COORD,n_M,pow))


##Paramètres génétiques:

eps =       10
n_P =       100
g_max =     100
n_sel =     60
m_prob =    0.01
unchg =     5
range_k =   10

## data
#ballon m :
k1 = (19,1.53,2.64)
C1 = {
    0.00: (1.53, 2.65),
    0.04: (1.53, 2.64),
    0.08: (1.53, 2.62),
    0.12: (1.53, 2.59),
    0.16: (1.53, 2.56),
    0.20: (1.52, 2.52),
    0.24: (1.53, 2.48),
    0.28: (1.53, 2.43),
    0.32: (1.53, 2.38),
    0.36: (1.53, 2.32),
    0.40: (1.53, 2.26),
    0.44: (1.53, 2.20),
    0.48: (1.53, 2.12),
    0.52: (1.53, 2.07),
    0.56: (1.53, 1.98),
    0.60: (1.53, 1.91),
    0.64: (1.53, 1.83),
    0.68: (1.52, 1.76),
    0.72: (1.52, 1.67),
    0.76: (1.51, 1.59),
    0.80: (1.51, 1.52),
    0.84: (1.51, 1.44),
    0.88: (1.49, 1.36),
    0.92: (1.49, 1.29),
    0.96: (1.48, 1.21),
    1.00: (1.47, 1.13),
    1.04: (1.45, 1.04),
    1.08: (1.44, 0.97),
    1.12: (1.44, 0.89),
    1.16: (1.42, 0.81)}
k2 = (22,1.48,2.61)
C2 = {
    0.00: (1.48, 2.61),
    0.04: (1.48, 2.60),
    0.08: (1.48, 2.58),
    0.12: (1.48, 2.56),
    0.16: (1.48, 2.53),
    0.20: (1.48, 2.49),
    0.24: (1.49, 2.45),
    0.28: (1.49, 2.42),
    0.32: (1.49, 2.35),
    0.36: (1.49, 2.30),
    0.40: (1.49, 2.22),
    0.44: (1.49, 2.16),
    0.48: (1.48, 2.09),
    0.52: (1.49, 2.03),
    0.56: (1.48, 1.95),
    0.60: (1.48, 1.86),
    0.64: (1.48, 1.77),
    0.68: (1.48, 1.69),
    0.72: (1.47, 1.61),
    0.76: (1.47, 1.51),
    0.80: (1.47, 1.42),
    0.84: (1.47, 1.34),
    0.88: (1.47, 1.25),
    0.92: (1.46, 1.16),
    0.96: (1.46, 1.06),
    1.00: (1.45, 0.98),
    1.04: (1.45, 0.90),
    1.08: (1.45, 0.79),
    1.12: (1.45, 0.69),
    1.16: (1.44, 0.59)}
k3 = (27,1.44,2.47)
C3 = {
    0.00: (1.44, 2.47),
    0.04: (1.44, 2.45),
    0.08: (1.44, 2.42),
    0.12: (1.44, 2.39),
    0.16: (1.44, 2.34),
    0.20: (1.44, 2.31),
    0.24: (1.45, 2.26),
    0.28: (1.44, 2.21),
    0.32: (1.44, 2.13),
    0.36: (1.44, 2.05),
    0.40: (1.44, 1.96),
    0.44: (1.44, 1.88),
    0.48: (1.44, 1.79),
    0.52: (1.44, 1.71),
    0.56: (1.44, 1.61),
    0.60: (1.44, 1.51),
    0.64: (1.43, 1.40),
    0.68: (1.43, 1.30),
    0.72: (1.42, 1.20),
    0.76: (1.42, 1.09),
    0.80: (1.41, 0.99),
    0.84: (1.41, 0.88),
    0.88: (1.39, 0.77),
    0.92: (1.39, 0.66),
    0.96: (1.38, 0.54),
    1.00: (1.37, 0.42),
    1.04: (1.36, 0.32),
    1.08: (1.35, 0.21),
    1.12: (1.34, 0.10),
    1.16: (1.33, -0.02)}

K_val = [k1,k2,k3]
m = sy.symbols('m',real=True)
x0 = sy.symbols('x0',real=True)
y0 = sy.symbols('y0',real=True)
K_var = [m,x0,y0]

C = [C1,C2,C3]
