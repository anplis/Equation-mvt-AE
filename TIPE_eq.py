import math as ma
import matplotlib.pyplot as plt
import sympy as sy
import random as rd
import copy as copy
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut     #timeout
import time

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
def value_no_timeout(Ki, f):
    if f is None:
        return False
    try :
        f_num = sy.lambdify((t, *K_var), f, modules="numpy")
        with np.errstate(all="ignore"):
            A = f_num(T, *Ki)

        if not valide(A):
            return False

        return A

    except Exception:
        return False

def value(Ki,f):#timeout de 0.1s pour évaluer la fonction
    try:
        A = func_timeout(0.01, value_no_timeout, args=(Ki,f))
        return A
    except:
        return False

# validité d'une fonction sur un l'intervale de temps
def valide(A):
    return np.all(np.isfinite(A)) and not np.iscomplexobj(A) and np.max(np.abs(A)) < 1e3

# somme des écarts quadratiques aux points d'une trajéctoire
def ecart(Ci,Ki,f,var):
    X = np.array([x[var] for x in Ci])#coord x ou y
    A = value(Ki,f)
    if A is False:
        return False
    return np.sum((X - A)**2)

# somme des écarts quadratiques aux points des trajéctoires
def ecart_tot(f,var):
    e = 0
    for Ci,Ki in zip(C,K_val):
        ec = ecart(Ci,Ki,f,var)
        if ec is False:
            return False
        e += ec
    return e

## graphique

# trajéctoire et la courbe l'approximent
def trace(i_exp,F):
    fig, ax = plt.subplots()
    Ci,Ki = C[i_exp],K_val[i_exp]
    A,B = [value(Ki,f) for f in F]
    if A is False or B is False:
        return "fonction mal définie"
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

def trace_anim2(i_exp, F):
    # Trajectoire réelle
    X1 = np.array([p[0] for p in C[i_exp]], dtype=float)
    Y1 = np.array([p[1] for p in C[i_exp]], dtype=float)

    # Trajectoire approchée
    Ki = K_val[i_exp]

    X2 = value(Ki, F[0])
    Y2 = value(Ki, F[1])

    if X2 is False or Y2 is False:
        print("Fonction mal définie")
        return

    # Si la fonction est constante
    if np.isscalar(X2):
        X2 = np.full(len(T), float(X2))
    else:
        X2 = np.asarray(X2, dtype=float)

    if np.isscalar(Y2):
        Y2 = np.full(len(T), float(Y2))
    else:
        Y2 = np.asarray(Y2, dtype=float)

    n_frames = min(len(X1), len(X2), len(Y1), len(Y2))

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="datalim")

    # Tracés fixes
    ax.plot(X1, Y1, color="gray", alpha=0.4, label="Trajectoire réelle")
    ax.plot(X2, Y2, color="tab:blue", alpha=0.4, label="Approximation")

    # Points animés
    p1, = ax.plot([], [], "ro", ms=6)
    p2, = ax.plot([], [], "bo", ms=6)

    def init():
        p1.set_data([], [])
        p2.set_data([], [])
        return p1, p2

    def update(i):
        p1.set_data([X1[i]], [Y1[i]])
        p2.set_data([X2[i]], [Y2[i]])
        return p1, p2

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=40,
        repeat=False,
        blit=True,)

    ax.legend()
    plt.show()

    return ani

## modification génétique

def gauss(x):
    return ma.exp(-(x/n_P)**2)

def mutation(f,n_mut):
    if n_mut==0:
        return f
    else:
        if f is None:
            return mutation(cst(),n_mut-1)
        mut = rd.choices([add_op,del_op,swap_op,change_cst,swap_child],weights=[1,1,10,10,3])[0]
        f_mut = mut(f)
        if valide_expr(f_mut) and nb_node(f_mut)<=max_size:
            return mutation(f_mut,n_mut-1)
        return mutation(f,n_mut-1)

def parcour(expr, target):
    nodes = list(sy.preorder_traversal(expr))
    if target < len(nodes):
        return nodes[target]
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
        if op==sy.Pow:
            new = op(abs(expr),abs(cst()))
        new = op(expr,cst())
        return f.subs(expr,new,evaluate=False)
    else:
        new = op(expr,evaluate=False)
        return f.subs(expr,new,evaluate=False)

#cste ∈ [t, paramètres, Réel∈[-range_k,range_k]]
def cst(old=None):
    if old is not None:
        new = old + rd.uniform(-1,1)*(1/range_k)#arbitraire
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
        return f.subs(expr,op(*expr.args),evaluate=False)
    elif n==1 and l>=2:
        new = [cst() for _ in range(l-n)]
        return f.subs(expr,op(*expr.args,*new),evaluate=False)
    elif n==2 and l == 1:
        return f.subs(expr,op(rd.choice(expr.args)),evaluate=False)

#modifie valeur d'une constante aléatoirement choisie
def change_cst(f):
    EXPR = [ex for ex in sy.postorder_traversal(f) if not ex.args]
    if not EXPR:
        return f
    try:
        return f.subs(rd.choice(EXPR),cst(),evaluate=False)
    except Exception:
        return f

OP_1 = [sy.exp,sy.ln,sy.cos,sy.sin,sy.tan]
OP_2 = [sy.Add,sy.Mul,sy.Pow]
OP = {1:OP_1,2:OP_2}

#to much errors with :sy.acos,sy.asin,sy.atan

#nombre d'expression en entré de l'opérateur(ex: +:2, cos:1)
def type_op(expr):
    for OP_n,n in [(OP_1,1),(OP_2,2)]:
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
        return f.subs(expr,rd.choice(expr.args),evaluate=False)
    return f

def swap_child(f):
    EXPR = [exp for exp in sy.postorder_traversal(f) if len(exp.args)>=2]
    if EXPR:
        expr = rd.choice(EXPR)
        new = expr.func(*expr.args[::-1])
        return f.subs(expr, new,evaluate=False)
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
    if ec_tot is False:
        return float('inf')
    return ec_tot + c*nb_node(f)

def ini_P(var):#P = [(F,fit(F), ...]
    ini_P = []
    for _ in range(n_P):
        f = mutation(cst(),5)
        fit = fitness(f,var)
        if fit!='inf':
            ini_P.append((f,fit))
    return ini_P

#nombre mutation en fonction écart type de la pop
#lim(ec_ty->0)=max_mut+1, petit écart type -> beaucoup mut
#lim(ec_ty->inf)=1, grand écart type -> peu mutation
def nb_mut(rk,ec_typ):
    return round(max_mut*inverse_1(ec_typ) / (1+ma.exp(-rk+5)) + 1)

def inverse_1(x):
    return 1/(x+1)

def ecar_type(P):
    fits = [p[1] for p in P if np.isfinite(p[1])]
    if len(fits) == 0:
        return 1
    m = sum(fits)/len(fits)
    var = sum((x - m)**2 for x in fits)/len(fits)
    return ma.sqrt(var)

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
            e_typ = ecar_type(P[:round(n_P/2)])#écart type de la première moitié de la pop

            if i%20==0:
                print(i,"-ième génération")
                print(P[0][0],",",P[0][1],",",e_typ,"\n")

            W = [gauss(j) for j in range(len(P))]
            for rk in rd.choices([j for j in range(len(P))],weights=W,k=baby):
                Pi = P[rk]
                n_mut = nb_mut(rk,e_typ)
                F_mut = mutation(Pi[0],n_mut)
                F_mut_fit = fitness(F_mut,var)
                if F_mut_fit!='inf':
                    P = insertion(P,F_mut,F_mut_fit)
            if i%div==0:
                P = P[:n_P]

        F.append(P[0][0])
        FIT.append(float(P[0][1]))
    return F,FIT

#insert un individu en gardant un ordre un ordre croissant (de fitness)
def insertion(P,f,fit):
    k = 0
    while k<len(P) and fit>=P[k][1] : # trouve l'emplacement adapté du nouvel élément et on l'insert
        k+=1
    return P[:k] + [(f,fit)] + P[k:]

## evolution matriciel

def ini_M(var):
    M = np.zeros((n_M,n_M),dtype=object)
    for i,j in COORD:
        M[i][j] = {'f':mutation(cst(),5),'e':[]}
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
                    e = mutation(M[i_v][j_v]['f'],1)
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
eps         = 5
g_max       = 500
range_k     = 10
c           = 0.3
max_size    = 20
max_mut     = 10

#evolution
n_P     = 100
div     = 5
baby    = 200

#evolution mat
n_M     = 5
COORD = [(i,j) for i in range(n_M) for j in range(n_M)]

## data

K_val = [(10,-0.852, -1.24),(20,-1.672, -2.519),(30,-2.099, -3.919),(40,-2.854, -5.095)]
C1 = [
(-0.852, -1.24), (-0.761, -1.273), (-0.612, -1.355), (-0.398, -1.413),
(-0.133, -1.454), (0.139, -1.454), (0.403, -1.421), (0.659, -1.331),
(0.799, -1.256), (0.865, -1.215), (0.849, -1.24), (0.758, -1.298),
(0.585, -1.372), (0.321, -1.438), (0.3996, -1.446), (-0.241, -1.43),
(-0.472, -1.397), (-0.662, -1.314), (-0.785, -1.256), (-0.827, -1.223),
(-0.802, -1.256), (-0.695, -1.298), (-0.53, -1.372), (-0.29, -1.421),
(-0.009563, -1.479), (0.296, -1.454), (0.535, -1.388), (0.733, -1.314),
(0.849, -1.248), (0.874, -1.215), (0.849, -1.248), (0.692, -1.322),
(0.486, -1.413), (0.213, -1.446), (-0.02607, -1.471), (-0.298, -1.446),
(-0.554, -1.397), (-0.686, -1.306), (-0.802, -1.223), (-0.81, -1.223),
(-0.752, -1.256), (-0.62, -1.322), (-0.431, -1.372), (-0.158, -1.438),
(0.08949, -1.471), (0.378, -1.454), (0.585, -1.38), (0.783, -1.298),
(0.882, -1.24), (0.882, -1.232), (0.816, -1.265), (0.659, -1.314),
(0.395, -1.413), (0.156, -1.454), (-0.142, -1.438), (-0.406, -1.405),
(-0.604, -1.355), (-0.744, -1.273), (-0.802, -1.24), (-0.802, -1.232),
(-0.703, -1.281), (-0.554, -1.339), (-0.331, -1.397), (-0.09211, -1.438),
(0.189, -1.454), (0.461, -1.43), (0.684, -1.331), (0.816, -1.256),
(0.874, -1.223), (0.832, -1.24), (0.75, -1.281), (0.535, -1.388),
(0.296, -1.454), (-0.001309, -1.454), (-0.241, -1.438), (-0.472, -1.372),
(-0.662, -1.306), (-0.777, -1.24)]

C2 = [
(-1.672, -2.519), (-1.639, -2.519), (-1.522, -2.602), (-1.356, -2.719),
(-1.023, -2.844), (-0.715, -2.919), (-0.315, -3.002), (0.0928, -3.035),
(0.559, -2.985), (0.917, -2.877), (1.233, -2.769), (1.5, -2.627),
(1.716, -2.519), (1.749, -2.453), (1.783, -2.436), (1.741, -2.469),
(1.633, -2.536), (1.433, -2.652), (1.192, -2.769), (0.842, -2.885),
(0.459, -2.96), (0.001224, -2.969), (-0.407, -2.969), (-0.806, -2.869),
(-1.114, -2.769), (-1.364, -2.677), (-1.547, -2.577), (-1.647, -2.503),
(-1.68, -2.478), (-1.63, -2.527), (-1.464, -2.602), (-1.256, -2.719),
(-0.981, -2.819), (-0.598, -2.935), (-0.249, -3.002), (0.234, -3.019),
(0.692, -2.952), (1.034, -2.852), (1.35, -2.727), (1.55, -2.594),
(1.716, -2.494), (1.783, -2.436), (1.766, -2.419), (1.708, -2.461),
(1.6, -2.552), (1.4, -2.669), (1.092, -2.802), (0.717, -2.91),
(0.284, -2.96), (-0.149, -3.01), (-0.515, -2.969), (-0.906, -2.902),
(-1.173, -2.794), (-1.414, -2.677), (-1.564, -2.586), (-1.68, -2.511),
(-1.655, -2.511), (-1.581, -2.577), (-1.422, -2.669), (-1.181, -2.769),
(-0.865, -2.877), (-0.515, -2.944), (-0.107, -3.002), (0.334, -2.985),
(0.75, -2.927), (1.125, -2.819), (1.408, -2.669), (1.625, -2.544),
(1.733, -2.486), (1.766, -2.453), (1.766, -2.428), (1.7, -2.486),
(1.55, -2.586), (1.325, -2.711), (0.984, -2.827), (0.634, -2.952),
(0.101, -2.994), (-0.265, -3.002)]

C3 = [
(-2.099, -3.919), (-2.06, -3.939), (-1.968, -3.965), (-1.791, -4.037),
(-1.543, -4.148), (-1.287, -4.233), (-0.921, -4.312), (-0.581, -4.351),
(-0.181, -4.397), (0.211, -4.384), (0.643, -4.364), (1.003, -4.266),
(1.35, -4.168), (1.625, -4.063), (1.827, -3.965), (1.984, -3.88),
(2.05, -3.827), (2.083, -3.808), (2.011, -3.84), (1.906, -3.925),
(1.703, -4.017), (1.481, -4.109), (1.206, -4.213), (0.8, -4.298),
(0.401, -4.384), (0.001745, -4.397), (-0.443, -4.338), (-0.744, -4.312),
(-1.071, -4.253), (-1.418, -4.181), (-1.719, -4.076), (-1.844, -4.004),
(-2.007, -3.952), (-2.105, -3.899), (-2.079, -3.899), (-1.988, -3.939),
(-1.817, -4.004), (-1.634, -4.096), (-1.346, -4.174), (-1.006, -4.246),
(-0.666, -4.318), (-0.273, -4.384), (0.185, -4.377), (0.486, -4.312),
(0.898, -4.253), (1.225, -4.154), (1.559, -4.063), (1.749, -3.958),
(1.939, -3.899), (2.03, -3.847), (2.083, -3.827), (2.043, -3.84),
(1.952, -3.88), (1.821, -3.971), (1.592, -4.063), (1.304, -4.174),
(0.957, -4.285), (0.512, -4.357), (0.139, -4.377), (-0.253, -4.39),
(-0.659, -4.318), (-1.019, -4.246), (-1.333, -4.148), (-1.628, -4.063),
(-1.817, -3.978), (-1.942, -3.939), (-2.02, -3.886), (-2.04, -3.873),
(-1.981, -3.88), (-1.85, -3.939), (-1.673, -4.03), (-1.431, -4.135),
(-1.13, -4.226), (-0.738, -4.305), (-0.365, -4.338), (0.04101, -4.37),
(0.453, -4.377), (0.839, -4.266)]

C4 = [
(-2.854, -5.095),(-2.835, -5.114),(-2.724, -5.179),(-2.547, -5.254),(-2.315, -5.347),
(-1.989, -5.486),(-1.626, -5.588),(-1.235, -5.691),(-0.752, -5.737),(-0.286, -5.775),
(0.197, -5.765),(0.69, -5.737),(1.174, -5.644),(1.63, -5.533),
(1.946, -5.421),(2.328, -5.263),(2.597, -5.151),(2.811, -5.04),
(2.932, -4.965),(2.997, -4.919),(2.951, -4.937),(2.895, -5.012),
(2.746, -5.086),(2.504, -5.198),(2.244, -5.337),(1.89, -5.468),
(1.481, -5.551),(1.053, -5.626),(0.588, -5.719),(0.07631, -5.775),(-0.435, -5.756),
(-0.863, -5.7),(-1.31, -5.654),(-1.738, -5.514),(-2.054, -5.402),(-2.333, -5.291),
(-2.566, -5.179),(-2.715, -5.114),(-2.798, -5.058),(-2.835, -5.03),(-2.761, -5.095),
(-2.649, -5.188),(-2.454, -5.263),(-2.147, -5.384),(-1.812, -5.495),(-1.421, -5.588),(-1.012, -5.7),
(-0.547, -5.784),(-0.04462, -5.802),(0.411, -5.756),(0.96, -5.644),
(1.332, -5.607),(1.76, -5.505),(2.16, -5.347),
(2.458, -5.216),(2.69, -5.105),(2.876, -4.993),(2.969, -4.928),
(3.007, -4.9),(2.951, -4.937),(2.867, -5.012),(2.69, -5.133),
(2.467, -5.254),(2.095, -5.402),(1.769, -5.533),
(1.332, -5.598),(0.886, -5.654),(0.393, -5.737),
(-0.1, -5.747),(-0.566, -5.719),(-1.04, -5.654),(-1.459, -5.588),(-1.84, -5.449),(-2.156, -5.347),
(-2.398, -5.244),(-2.575, -5.161),(-2.705, -5.086),(-2.789, -5.049)]

T = np.array([
0, 0.033, 0.067, 0.1, 0.133, 0.167, 0.2, 0.233, 0.267, 0.3, 0.333, 0.367,
0.4, 0.433, 0.467, 0.5, 0.533, 0.567, 0.6, 0.633, 0.667, 0.7, 0.733, 0.767,
0.8, 0.833, 0.867, 0.9, 0.933, 0.967, 1, 1.033, 1.067, 1.1, 1.133, 1.167,
1.2, 1.233, 1.267, 1.3, 1.333, 1.367, 1.4, 1.433, 1.467, 1.5, 1.533, 1.567,
1.6, 1.633, 1.667, 1.7, 1.733, 1.767, 1.8, 1.833, 1.867, 1.9, 1.933, 1.967,
2, 2.033, 2.067, 2.1, 2.133, 2.167, 2.2, 2.233, 2.267, 2.3, 2.333, 2.367,
2.4, 2.433, 2.467, 2.5, 2.533, 2.567])

C = [C1,C2,C3,C4]
L = sy.symbols('L',real=True)
x0 = sy.symbols('x0',real=True)
y0 = sy.symbols('y0',real=True)
K_var = [L,x0,y0]

#results:
F = [x0*sy.sin(sy.sin(sy.sin(4.963740378792*t + y0 + 0.1847517814011))), 1.05995729841294*y0
