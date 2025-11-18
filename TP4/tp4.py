import numpy as np
import time
from scipy.special import hankel1

# --- Classes Modifiées pour le Stockage ---
class NoeudCluster:
    def __init__(self):
        self.gauche = None
        self.droit = None
        self.est_feuille = False
        self.indices = None
        self.boite = None
        self.nb_points = 0
        # (Autres attributs géométriques si nécessaire)
        self.axe_division = None
        self.coord_division = None
        self.profondeur = 0

class NoeudBloc:
    def __init__(self, tau, sigma):
        self.tau = tau
        self.sigma = sigma
        self.est_feuille = False
        self.admissible = False
        self.gauche = None
        self.droit = None
        self.droite_bas = None
        self.bas_droite = None
        
        # --- NOUVEAU : Stockage des données ---
        self.U = None           # Pour ACA (Admissible)
        self.V = None           # Pour ACA (Admissible)
        self.dense_block = None # Pour Dense (Non-Admissible)

# --- Fonctions Géométriques (Inchangées) ---
def calculer_diametre_boite(boite):
    if boite is None: return 0.0
    min_coords, max_coords = boite
    etendues = max_coords - min_coords
    return np.sqrt(np.sum(etendues**2))

def distance_entre_boites(boite_A, boite_B):
    if boite_A is None or boite_B is None: return np.inf
    min_A, max_A = boite_A
    min_B, max_B = boite_B
    delta = np.maximum(0.0, np.maximum(min_A - max_B, min_B - max_A))
    return np.linalg.norm(delta)

def is_admissible(noeud1, noeud2, eta):
    boite1 = noeud1.boite
    boite2 = noeud2.boite
    if boite1 is None or boite2 is None: return False
    diam1 = calculer_diametre_boite(boite1)
    diam2 = calculer_diametre_boite(boite2)
    minimum = min(diam1, diam2)
    distance = distance_entre_boites(boite1, boite2)
    if distance < 1e-15: return False
    return (minimum < eta * distance)

# --- Construction des Arbres (Inchangée) ---
def construire_arbre_bsp(points, indices_originaux, Nleaf=5, profondeur=0):
    noeud = NoeudCluster()
    noeud.profondeur = profondeur
    N = len(points)
    noeud.nb_points = N
    if N == 0:
        noeud.est_feuille = True
        noeud.points = np.empty((0, 2))
        noeud.indices = np.array([], dtype=int)
        return noeud
    noeud.indices = np.array(indices_originaux, dtype=int)
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    noeud.boite = (min_coords, max_coords)
    if N <= Nleaf:
        noeud.est_feuille = True
        noeud.points = points
        return noeud
    etendues = max_coords - min_coords
    noeud.axe_division = int(np.argmax(etendues))
    coords_sur_axe = points[:, noeud.axe_division]
    idx_median = N // 2
    perm = np.argpartition(coords_sur_axe, idx_median)
    left_idx_local = perm[:idx_median]
    right_idx_local = perm[idx_median:]
    points_gauche = points[left_idx_local]
    points_droit = points[right_idx_local]
    indices_gauche_originaux = noeud.indices[left_idx_local]
    indices_droit_originaux = noeud.indices[right_idx_local]
    if len(points_gauche) == 0 or len(points_droit) == 0:
        noeud.est_feuille = True
        noeud.points = points
        return noeud
    noeud.gauche = construire_arbre_bsp(points_gauche, indices_gauche_originaux, Nleaf, profondeur + 1)
    noeud.droit = construire_arbre_bsp(points_droit, indices_droit_originaux, Nleaf, profondeur + 1)
    return noeud

def construire_arbre_blocs(noeud_tau, noeud_sigma, eta=3):
    bloc = NoeudBloc(noeud_tau, noeud_sigma)
    if is_admissible(noeud_tau, noeud_sigma, eta):
        bloc.est_feuille = True
        bloc.admissible = True
        return bloc
    if noeud_tau.est_feuille or noeud_sigma.est_feuille:
        bloc.est_feuille = True
        bloc.admissible = False
        return bloc
    bloc.gauche = construire_arbre_blocs(noeud_tau.gauche, noeud_sigma.gauche, eta)
    bloc.droit = construire_arbre_blocs(noeud_tau.gauche, noeud_sigma.droit, eta)
    bloc.droite_bas = construire_arbre_blocs(noeud_tau.droit, noeud_sigma.gauche, eta)
    bloc.bas_droite = construire_arbre_blocs(noeud_tau.droit, noeud_sigma.droit, eta)
    return bloc

# --- Physique BEM (Inchangée) ---
def generate_nodes(N, a):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = a * np.cos(angles)
    y = a * np.sin(angles)
    return angles, np.column_stack((x, y))

def generate_segments_indices(N):
    return [(k, (k+1) % N) for k in range(N)]

def G(n1, n2, k):
    x1, y1 = n1
    x2, y2 = n2
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    if np.ndim(dist) == 0:
        if dist < 1e-15: return 0j
        return 1j*hankel1(0, k*dist)/4
    else:
        res = np.zeros_like(dist, dtype=np.complex128)
        mask = dist >= 1e-15
        res[mask] = 1j*hankel1(0, k*dist[mask])/4
        return res

def GaussLegendre(f, a, b, points, poids):
    t = points
    u = (b-a)/2*t + (b+a)/2
    vals = f(u)
    if np.ndim(vals) == 0: return (b-a)/2 * vals * np.sum(poids)
    if vals.shape == points.shape: return (b-a)/2 * np.sum(vals * poids)
    return (b-a)/2 * np.dot(vals, poids)

def integrale2D(f, n1, n2, points, poids):
    x1, y1 = n1
    x2, y2 = n2
    norm = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    g = lambda t: f(x1*(1-t)+x2*t, y1*(1-t)+y2*t) * norm
    return GaussLegendre(g, 0, 1, points, poids)

def calculer_Aij(i, j, N, nodes, G_func, points, poids, k):
    seg_i_n1 = nodes[i]
    seg_i_n2 = nodes[(i+1)%N]
    seg_j_n1 = nodes[j]
    seg_j_n2 = nodes[(j+1)%N]
    terme_ij = 0j
    if i == j:
        coeff = -1/(2*np.pi)
        x1, y1 = seg_j_n1
        x2, y2 = seg_j_n2
        long = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        d_j = lambda x, y: np.sqrt((x1-x)**2 + (y1-y)**2)
        d_j_plus_1 = lambda x, y: np.sqrt((x2-x)**2 + (y2-y)**2)
        f = lambda x, y: coeff * (d_j_plus_1(x,y) * np.log(d_j_plus_1(x,y) + 1e-15) + d_j(x,y) * np.log(d_j(x,y) + 1e-15) - long)
        terme_ij += integrale2D(f, seg_i_n1, seg_i_n2, points, poids)
        const = 1j/4 + coeff*(np.log(k/2)+np.euler_gamma)
        long_j = integrale2D(lambda x,y: 1.0, seg_j_n1, seg_j_n2, points, poids)
        terme_ij += integrale2D(lambda x,y: const * long_j, seg_i_n1, seg_i_n2, points, poids)
    else:
        I_inner = lambda X, Y: integrale2D(
            lambda x,y: G_func((x,y),(X[:,None], Y[:,None]), k), 
            seg_i_n1, seg_i_n2, points, poids
        )
        terme_ij += integrale2D(I_inner, seg_j_n1, seg_j_n2, points, poids)
    return terme_ij

def matriceA(N, nodes, G_func, points, poids, k):
    A = np.zeros((N,N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            A[i][j] += calculer_Aij(i, j, N, nodes, G_func, points, poids, k)
    return A

# --- Getters Corrigés pour ACA (Taille du bloc) ---
def get_bem_row_c(i_global, indices_colonne_cible, N, nodes, G_func, points, poids, k):
    n_cible = len(indices_colonne_cible)
    row_values = np.zeros(n_cible, dtype=np.complex128)
    for idx_local, j_global in enumerate(indices_colonne_cible):
        row_values[idx_local] = calculer_Aij(i_global, j_global, N, nodes, G_func, points, poids, k)
    return row_values

def get_bem_col_c(j_global, indices_ligne_cible, N, nodes, G_func, points, poids, k):
    n_cible = len(indices_ligne_cible)
    col_values = np.zeros(n_cible, dtype=np.complex128)
    for idx_local, i_global in enumerate(indices_ligne_cible):
        col_values[idx_local] = calculer_Aij(i_global, j_global, N, nodes, G_func, points, poids, k)
    return col_values

# --- ACA (Inchangé, mais utilisé pour l'assemblage) ---
def aca_partial_pivot_on_the_fly(indices_ligne, indices_colonne, get_row_func, get_col_func, eps=1e-4, max_rank=None):
    m, n = len(indices_ligne), len(indices_colonne)
    if max_rank is None: max_rank = min(m, n)
    U_cols = []
    V_rows = []
    used_rows = np.zeros(m, dtype=bool)
    used_cols = np.zeros(n, dtype=bool)
    norm_A_k_F_squared = 0.0
    current_row_search_idx = 0
    for k in range(max_rank):
        i_local = -1
        for idx in range(current_row_search_idx, m):
            if not used_rows[idx]:
                i_local = idx
                break
        if i_local == -1:
            for idx in range(m):
                if not used_rows[idx]: i_local = idx; break
        if i_local == -1: break
        current_row_search_idx = i_local + 1
        i_global = indices_ligne[i_local]
        R_row = get_row_func(i_global, indices_colonne).copy()
        for l in range(len(U_cols)):
            R_row -= U_cols[l][i_local] * V_rows[l]
        abs_row = np.abs(R_row)
        abs_row[used_cols] = -1.0
        j_local = np.argmax(abs_row)
        if abs_row[j_local] < 1e-15:
            used_rows[i_local] = True
            continue
        used_rows[i_local] = True
        used_cols[j_local] = True
        j_global = indices_colonne[j_local]
        pivot_val = R_row[j_local]
        v_k = R_row / pivot_val
        C_col = get_col_func(j_global, indices_ligne).copy()
        for l in range(len(U_cols)):
            C_col -= U_cols[l] * V_rows[l][j_local]
        u_k = C_col
        norm_u_sq = np.real(np.vdot(u_k, u_k))
        norm_v_sq = np.real(np.vdot(v_k, v_k))
        norm_update_sq = norm_u_sq * norm_v_sq
        cross_term = 0.0
        for l in range(len(U_cols)):
            dot_u = np.vdot(U_cols[l], u_k)
            dot_v = np.vdot(V_rows[l], v_k)
            cross_term += np.real(dot_u * np.conj(dot_v))
        norm_A_k_F_squared += norm_update_sq + 2 * cross_term
        U_cols.append(u_k)
        V_rows.append(v_k)
        if norm_update_sq < (eps**2) * norm_A_k_F_squared and k>0: break
    if len(U_cols) > 0:
        U = np.column_stack(U_cols)
        V = np.vstack(V_rows)
    else:
        U = np.zeros((m, 0), dtype=np.complex128)
        V = np.zeros((0, n), dtype=np.complex128)
    return U, V

# --- NOUVEAU : Assemblage de la H-Matrice ---
def assembler_h_matrice(bloc, Nsegments, nodes, G, points, poids, k):
    """
    Parcourt l'arbre et calcule/stocke les données (U, V ou Dense) dans chaque feuille.
    """
    if bloc.est_feuille:
        indices_ligne = np.sort(bloc.tau.indices)
        indices_colonne = np.sort(bloc.sigma.indices)
        m, n = len(indices_ligne), len(indices_colonne)
        
        if m == 0 or n == 0: return # Bloc vide

        if not bloc.admissible:
            # --- Calcul Dense et Stockage ---
            bloc.dense_block = np.zeros((m, n), dtype=np.complex128)
            for idx_i, I in enumerate(indices_ligne):
                for idx_j, J in enumerate(indices_colonne):
                    bloc.dense_block[idx_i, idx_j] = calculer_Aij(I, J, Nsegments, nodes, G, points, poids, k)
        else:
            # --- Calcul ACA et Stockage ---
            bloc.U, bloc.V = aca_partial_pivot_on_the_fly(
                indices_ligne, indices_colonne,
                lambda i, cols: get_bem_row_c(i, cols, Nsegments, nodes, G, points, poids, k),
                lambda j, rows: get_bem_col_c(j, rows, Nsegments, nodes, G, points, poids, k),
                eps=1e-4
            )
    else:
        # Appel récursif
        assembler_h_matrice(bloc.gauche, Nsegments, nodes, G, points, poids, k)
        assembler_h_matrice(bloc.droit, Nsegments, nodes, G, points, poids, k)
        assembler_h_matrice(bloc.droite_bas, Nsegments, nodes, G, points, poids, k)
        assembler_h_matrice(bloc.bas_droite, Nsegments, nodes, G, points, poids, k)

# --- NOUVEAU : Hmatvec Optimisé ---
def Hmatvec(bloc, x):
    """
    Produit matrice-vecteur utilisant les données stockées dans 'bloc'.
    """
    N = len(x)
    b = np.zeros(N, dtype=np.complex128)
    Hmatvec_recursif(bloc, x, b)
    return b

def Hmatvec_recursif(bloc, x, b):
    if bloc.est_feuille:
        indices_ligne = np.sort(bloc.tau.indices)
        indices_colonne = np.sort(bloc.sigma.indices)
        m, n = len(indices_ligne), len(indices_colonne)
        if m == 0 or n == 0: return

        x_sigma = x[indices_colonne]
        y_tau_contrib = None

        if not bloc.admissible:
            # Utilisation du bloc dense STOCKÉ
            y_tau_contrib = bloc.dense_block @ x_sigma
        else:
            # Utilisation de ACA STOCKÉ
            if bloc.U.shape[1] == 0: # Rang 0
                 y_tau_contrib = np.zeros(m, dtype=np.complex128)
            else:
                 # Produit rapide : U * (V * x)
                 z = bloc.V @ x_sigma
                 y_tau_contrib = bloc.U @ z
        
        b[indices_ligne] += y_tau_contrib
        return
    else:
        Hmatvec_recursif(bloc.gauche, x, b)
        Hmatvec_recursif(bloc.droit, x, b)
        Hmatvec_recursif(bloc.droite_bas, x, b)
        Hmatvec_recursif(bloc.bas_droite, x, b)

# --- Exécution Principale ---
if __name__ == "__main__":
    # Paramètres
    Nsegments = 1600 # Augmenté pour voir la différence
    a = 1.0
    k_wave = 2*np.pi
    n_gauss = 2
    
    # Géométrie
    angles, nodes = generate_nodes(Nsegments, a)
    points, poids = np.polynomial.legendre.leggauss(n_gauss)
    indices_originaux = np.arange(Nsegments)

    print(f"--- Test pour N={Nsegments} ---")

    # 1. Matrice Dense (pour comparaison)
    print("Construction matrice dense...")
    t0 = time.time()
    BEM = matriceA(Nsegments, nodes, G, points, poids, k_wave)
    t_dense_build = time.time() - t0
    print(f"Temps construction Dense: {t_dense_build:.4f}s")

    # 2. Construction H-Matrice (Assemblage)
    print("Construction H-Matrice (Arbres + Assemblage)...")
    t0 = time.time()
    racine_binaire = construire_arbre_bsp(nodes, indices_originaux, Nleaf=20)
    racine_blocs = construire_arbre_blocs(racine_binaire, racine_binaire, eta=3)
    
    # C'est ici que la magie opère : on pré-calcule tout !
    assembler_h_matrice(racine_blocs, Nsegments, nodes, G, points, poids, k_wave)
    t_hmat_build = time.time() - t0
    print(f"Temps assemblage H-Matrice: {t_hmat_build:.4f}s")

    # 3. Comparaison des Produits Matrice-Vecteur (MVM)
    print("\nComparaison des produits MVM (moyenne sur 5 itérations)...")
    x = 10*np.random.rand(Nsegments)
    
    # Warmup
    _ = BEM @ x
    _ = Hmatvec(racine_blocs, x)

    t_dense_accum = 0
    t_hmat_accum = 0
    n_iter = 5

    for i in range(n_iter):
        x = np.random.rand(Nsegments) + 1j*np.random.rand(Nsegments)
        
        # Dense
        t1 = time.time()
        b_exact = BEM @ x
        t2 = time.time()
        t_dense_accum += (t2 - t1)
        
        # H-Matrice
        t1 = time.time()
        b_Hmat = Hmatvec(racine_blocs, x)
        t2 = time.time()
        t_hmat_accum += (t2 - t1)
        
        # Vérif erreur
        err = np.linalg.norm(b_exact - b_Hmat) / np.linalg.norm(b_exact)
        if i == 0: print(f"Erreur relative (run 0): {err:.2e}")

    avg_dense = t_dense_accum / n_iter
    avg_hmat = t_hmat_accum / n_iter

    print(f"\nTemps moyen MVM Dense     : {avg_dense:.6f} s")
    print(f"Temps moyen MVM H-Matrice : {avg_hmat:.6f} s")
    print(f"Accélération (Speedup)    : {avg_dense / avg_hmat:.2f}x")