using LinearAlgebra
using Statistics
using Random
using SpecialFunctions
using FastGaussQuadrature
using Plots

# --- 1. Structures de Données ---

mutable struct NoeudCluster
    gauche::Union{NoeudCluster, Nothing}
    droit::Union{NoeudCluster, Nothing}
    est_feuille::Bool
    points::Union{Matrix{Float64}, Nothing}
    indices::Union{Vector{Int}, Nothing}
    boite::Union{Tuple{Vector{Float64}, Vector{Float64}}, Nothing}
    axe_division::Union{Int, Nothing}
    coord_division::Union{Float64, Nothing}
    profondeur::Int
    nb_points::Int

    NoeudCluster() = new(nothing, nothing, false, nothing, nothing, nothing, nothing, nothing, 0, 0)
end

mutable struct NoeudBloc
    tau::NoeudCluster
    sigma::NoeudCluster
    est_feuille::Bool
    admissible::Bool
    gauche::Union{NoeudBloc, Nothing}
    droit::Union{NoeudBloc, Nothing}
    droite_bas::Union{NoeudBloc, Nothing}
    bas_droite::Union{NoeudBloc, Nothing}
    
    # Stockage des données
    U::Union{Matrix{ComplexF64}, Nothing}
    V::Union{Matrix{ComplexF64}, Nothing}
    dense_block::Union{Matrix{ComplexF64}, Nothing}

    NoeudBloc(tau, sigma) = new(tau, sigma, false, false, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

# --- 2. Fonctions Géométriques ---

function calculer_diametre_boite(boite)
    if boite === nothing
        return 0.0
    end
    min_coords, max_coords = boite
    etendues = max_coords - min_coords
    return sqrt(sum(etendues.^2))
end

function distance_entre_boites(boite_A, boite_B)
    if boite_A === nothing || boite_B === nothing
        return Inf
    end
    min_A, max_A = boite_A
    min_B, max_B = boite_B
    
    delta = max.(0.0, max.(min_A - max_B, min_B - max_A))
    return norm(delta)
end

function is_admissible(noeud1, noeud2, eta)
    boite1 = noeud1.boite
    boite2 = noeud2.boite
    if boite1 === nothing || boite2 === nothing
        return false
    end
    
    diam1 = calculer_diametre_boite(boite1)
    diam2 = calculer_diametre_boite(boite2)
    minimum = min(diam1, diam2)
    distance = distance_entre_boites(boite1, boite2)
    
    if distance < 1e-15
        return false
    end
    
    return minimum < eta * distance
end

# --- 3. Construction des Arbres ---

function construire_arbre_bsp(points, indices_originaux; Nleaf=5, profondeur=0)
    noeud = NoeudCluster()
    noeud.profondeur = profondeur
    N = size(points, 1)
    noeud.nb_points = N
    
    if N == 0
        noeud.est_feuille = true
        noeud.points = zeros(0, 2)
        noeud.indices = Int[]
        return noeud
    end
    
    noeud.indices = indices_originaux
    
    # Calcul BBox (mins et maxs par colonne)
    min_coords = vec(minimum(points, dims=1))
    max_coords = vec(maximum(points, dims=1))
    noeud.boite = (min_coords, max_coords)
    
    if N <= Nleaf
        noeud.est_feuille = true
        noeud.points = points
        return noeud
    end
    
    etendues = max_coords - min_coords
    # argmax renvoie un index cartésien en Julia, on veut l'entier
    axe = argmax(etendues) 
    noeud.axe_division = axe
    
    coords_sur_axe = points[:, axe]
    idx_median = N ÷ 2
    
    # Tri partiel pour trouver la médiane (équivalent argpartition)
    perm = sortperm(coords_sur_axe)
    
    left_idx_local = perm[1:idx_median]
    right_idx_local = perm[idx_median+1:end]
    
    points_gauche = points[left_idx_local, :]
    points_droit = points[right_idx_local, :]
    
    indices_gauche_originaux = noeud.indices[left_idx_local]
    indices_droit_originaux = noeud.indices[right_idx_local]
    
    if size(points_gauche, 1) == 0 || size(points_droit, 1) == 0
        noeud.est_feuille = true
        noeud.points = points
        return noeud
    end
    
    noeud.gauche = construire_arbre_bsp(points_gauche, indices_gauche_originaux, Nleaf=Nleaf, profondeur=profondeur+1)
    noeud.droit = construire_arbre_bsp(points_droit, indices_droit_originaux, Nleaf=Nleaf, profondeur=profondeur+1)
    
    return noeud
end

function construire_arbre_blocs(noeud_tau, noeud_sigma, eta=3.0)
    bloc = NoeudBloc(noeud_tau, noeud_sigma)
    
    if is_admissible(noeud_tau, noeud_sigma, eta)
        bloc.est_feuille = true
        bloc.admissible = true
        return bloc
    end
    
    if noeud_tau.est_feuille || noeud_sigma.est_feuille
        bloc.est_feuille = true
        bloc.admissible = false
        return bloc
    end
    
    bloc.gauche = construire_arbre_blocs(noeud_tau.gauche, noeud_sigma.gauche, eta)
    bloc.droit = construire_arbre_blocs(noeud_tau.gauche, noeud_sigma.droit, eta)
    bloc.droite_bas = construire_arbre_blocs(noeud_tau.droit, noeud_sigma.gauche, eta)
    bloc.bas_droite = construire_arbre_blocs(noeud_tau.droit, noeud_sigma.droit, eta)
    
    return bloc
end

# --- 4. Physique BEM ---

function generate_nodes(N, a)
    angles = range(0, 2*pi, length=N+1)[1:end-1] # 0 à 2pi exclu
    x = a .* cos.(angles)
    y = a .* sin.(angles)
    return angles, hcat(x, y)
end

function hankel1_safe(order, z)
    if abs(z) < 1e-15
        return 0.0 + 0.0im
    end
    return hankelh1(order, z)
end

function G_func(n1, n2, k)
    x1, y1 = n1
    x2, y2 = n2
    dist = sqrt((x2-x1)^2 + (y2-y1)^2)
    if dist < 1e-15
        return 0.0im
    end
    return 1im * hankel1_safe(0, k*dist) / 4.0
end

# Intégration Gauss-Legendre
function integrale2D(f, n1, n2, points_gauss, poids_gauss)
    x1, y1 = n1
    x2, y2 = n2
    norm_seg = sqrt((x2-x1)^2 + (y2-y1)^2)
    
    # Changement de variable t E [0, 1] -> u E [-1, 1] (Gauss standard est sur [-1, 1])
    # Mais ici votre code python intégrait sur [0, 1].
    # FastGaussQuadrature donne des points sur (-1, 1).
    # Transformation [-1, 1] -> [0, 1] : t = (u + 1) / 2, dt = 1/2 du
    
    S = 0.0im
    for i in 1:length(points_gauss)
        u = points_gauss[i] # dans [-1, 1]
        t = (u + 1) / 2.0   # dans [0, 1]
        
        X = x1*(1-t) + x2*t
        Y = y1*(1-t) + y2*t
        
        val = f(X, Y)
        S += poids_gauss[i] * val
    end
    
    return (1.0/2.0) * S * norm_seg # Jacobien de la transfo t->u est 1/2
end

function calculer_Aij(i, j, N, nodes, points_gauss, poids_gauss, k)
    # Attention indices Julia : i, j dans 1..N
    # Segments i : noeud i à i+1 (modulo N)
    
    idx_i1 = i
    idx_i2 = mod1(i+1, N)
    
    idx_j1 = j
    idx_j2 = mod1(j+1, N)
    
    seg_i_n1 = nodes[idx_i1, :]
    seg_i_n2 = nodes[idx_i2, :]
    
    seg_j_n1 = nodes[idx_j1, :]
    seg_j_n2 = nodes[idx_j2, :]
    
    terme_ij = 0.0im
    
    if i == j
        coeff = -1/(2*pi)
        x1, y1 = seg_j_n1
        x2, y2 = seg_j_n2
        long_seg = sqrt((x2-x1)^2 + (y2-y1)^2)
        
        d_j(x,y) = sqrt((x1-x)^2 + (y1-y)^2)
        d_j_plus_1(x,y) = sqrt((x2-x)^2 + (y2-y)^2)
        
        f_sing(x, y) = coeff * (
            d_j_plus_1(x,y) * log(d_j_plus_1(x,y) + 1e-15) + 
            d_j(x,y) * log(d_j(x,y) + 1e-15) - long_seg
        )
        
        terme_ij += integrale2D(f_sing, seg_i_n1, seg_i_n2, points_gauss, poids_gauss)
        
        const_val = 1im/4 + coeff*(log(k/2) + 0.5772156649) # Euler gamma
        
        # Intégrale de 1 sur segment j = longueur
        long_j = long_seg 
        
        # Intégrale de (const * long_j) sur segment i
        # Comme segment i == segment j ici, c'est const * long_j * long_i
        terme_ij += const_val * long_j * long_seg # approximation directe intégrale constante
        
    else
        # Double intégrale
        # Fonction interne : intégrale sur x (segment i)
        function I_inner(X, Y)
            return integrale2D((x,y) -> G_func([x, y], [X, Y], k), seg_i_n1, seg_i_n2, points_gauss, poids_gauss)
        end
        
        terme_ij += integrale2D(I_inner, seg_j_n1, seg_j_n2, points_gauss, poids_gauss)
    end
    
    return terme_ij
end

function matriceA(N, nodes, points_gauss, poids_gauss, k)
    A = zeros(ComplexF64, N, N)
    # Threads.@threads pour paralléliser si besoin
    for j in 1:N
        for i in 1:N
            A[i, j] += calculer_Aij(i, j, N, nodes, points_gauss, poids_gauss, k)
        end
    end
    return A
end

# --- 5. Helpers pour ACA ---

function get_bem_row_c(i_global, indices_colonne_cible, N, nodes, points_gauss, poids_gauss, k)
    n_cible = length(indices_colonne_cible)
    row_values = zeros(ComplexF64, n_cible)
    for (idx_local, j_global) in enumerate(indices_colonne_cible)
        row_values[idx_local] = calculer_Aij(i_global, j_global, N, nodes, points_gauss, poids_gauss, k)
    end
    return row_values
end

function get_bem_col_c(j_global, indices_ligne_cible, N, nodes, points_gauss, poids_gauss, k)
    n_cible = length(indices_ligne_cible)
    col_values = zeros(ComplexF64, n_cible)
    for (idx_local, i_global) in enumerate(indices_ligne_cible)
        col_values[idx_local] = calculer_Aij(i_global, j_global, N, nodes, points_gauss, poids_gauss, k)
    end
    return col_values
end

# --- 6. ACA Partial Pivot ---

function aca_partial_pivot_on_the_fly(indices_ligne, indices_colonne, get_row_func, get_col_func; eps=1e-4, max_rank=nothing)
    m = length(indices_ligne)
    n = length(indices_colonne)
    
    limit_rank = (max_rank === nothing) ? min(m, n) : max_rank
    
    U_cols = Vector{Vector{ComplexF64}}()
    V_rows = Vector{Vector{ComplexF64}}()
    
    used_rows = falses(m)
    used_cols = falses(n)
    
    norm_A_k_F_squared = 0.0
    current_row_search_idx = 1
    
    for k in 1:limit_rank
        # 1. Trouver pivot ligne
        i_local = -1
        for idx in current_row_search_idx:m
            if !used_rows[idx]
                i_local = idx
                break
            end
        end
        if i_local == -1
            for idx in 1:m
                if !used_rows[idx]
                    i_local = idx
                    break
                end
            end
        end
        if i_local == -1; break; end
        
        current_row_search_idx = i_local + 1
        i_global = indices_ligne[i_local]
        
        # 2. Calcul ligne résiduelle
        R_row = get_row_func(i_global, indices_colonne) # copie implicite
        for l in 1:length(U_cols)
            # R_row -= U[l][i] * V[l]
            ul_val = U_cols[l][i_local]
            axpy!(-ul_val, V_rows[l], R_row) # Optimisation BLAS
        end
        
        # 3. Trouver pivot colonne
        abs_row = abs.(R_row)
        # Mettre à -1 les colonnes utilisées
        abs_row[used_cols] .= -1.0
        
        max_val, j_local = findmax(abs_row)
        
        if max_val < 1e-15
            used_rows[i_local] = true
            continue
        end
        
        used_rows[i_local] = true
        used_cols[j_local] = true
        j_global = indices_colonne[j_local]
        
        pivot_val = R_row[j_local]
        v_k = R_row ./ pivot_val
        
        # 4. Calcul colonne résiduelle
        C_col = get_col_func(j_global, indices_ligne)
        for l in 1:length(U_cols)
            vl_val = V_rows[l][j_local]
            axpy!(-vl_val, U_cols[l], C_col)
        end
        u_k = C_col
        
        # 5. Mise à jour norme
        norm_u_sq = real(dot(u_k, u_k))
        norm_v_sq = real(dot(v_k, v_k))
        norm_update_sq = norm_u_sq * norm_v_sq
        
        cross_term = 0.0
        for l in 1:length(U_cols)
            dot_u = dot(U_cols[l], u_k)
            dot_v = dot(V_rows[l], v_k)
            cross_term += real(dot_u * conj(dot_v))
        end
        
        norm_A_k_F_squared += norm_update_sq + 2 * cross_term
        
        push!(U_cols, u_k)
        push!(V_rows, v_k)
        
        if norm_update_sq < (eps^2) * norm_A_k_F_squared && k > 1
            break
        end
    end
    
    if length(U_cols) > 0
        U = hcat(U_cols...)
        V = vcat(transpose.(V_rows)...) # V_rows sont des vecteurs, on veut une matrice (r, n)
        # Note : dans Python v_k était 1D array. Ici V_rows est Vector{Vector}.
        # On veut V de taille (Rank, n). Transpose convertit vecteur colonne en ligne.
        # vcat empile les lignes.
    else
        U = zeros(ComplexF64, m, 0)
        V = zeros(ComplexF64, 0, n)
    end
    
    return U, V
end

# --- 7. Assemblage H-Matrice ---

function assembler_h_matrice!(bloc, Nsegments, nodes, points_gauss, poids_gauss, k)
    if bloc.est_feuille
        indices_ligne = sort(bloc.tau.indices)
        indices_colonne = sort(bloc.sigma.indices)
        m = length(indices_ligne)
        n = length(indices_colonne)
        
        if m == 0 || n == 0
            return
        end
        
        if !bloc.admissible
            # Dense
            bloc.dense_block = zeros(ComplexF64, m, n)
            for (j_loc, j_glob) in enumerate(indices_colonne)
                for (i_loc, i_glob) in enumerate(indices_ligne)
                    bloc.dense_block[i_loc, j_loc] = calculer_Aij(i_glob, j_glob, Nsegments, nodes, points_gauss, poids_gauss, k)
                end
            end
        else
            # ACA
            get_row = (i, cols) -> get_bem_row_c(i, cols, Nsegments, nodes, points_gauss, poids_gauss, k)
            get_col = (j, rows) -> get_bem_col_c(j, rows, Nsegments, nodes, points_gauss, poids_gauss, k)
            
            U, V = aca_partial_pivot_on_the_fly(indices_ligne, indices_colonne, get_row, get_col, eps=1e-4)
            bloc.U = U
            bloc.V = V
        end
    else
        assembler_h_matrice!(bloc.gauche, Nsegments, nodes, points_gauss, poids_gauss, k)
        assembler_h_matrice!(bloc.droit, Nsegments, nodes, points_gauss, poids_gauss, k)
        assembler_h_matrice!(bloc.droite_bas, Nsegments, nodes, points_gauss, poids_gauss, k)
        assembler_h_matrice!(bloc.bas_droite, Nsegments, nodes, points_gauss, poids_gauss, k)
    end
end

# --- 8. Produit H-Matrice Vecteur ---

function Hmatvec(bloc, x)
    N = length(x)
    b = zeros(ComplexF64, N)
    Hmatvec_recursif!(bloc, x, b)
    return b
end

function Hmatvec_recursif!(bloc, x, b)
    if bloc.est_feuille
        indices_ligne = sort(bloc.tau.indices)
        indices_colonne = sort(bloc.sigma.indices)
        
        if isempty(indices_ligne) || isempty(indices_colonne)
            return
        end
        
        x_sigma = x[indices_colonne]
        
        if !bloc.admissible
            # Produit Dense
            # b[indices] += Block * x_sub
            res = bloc.dense_block * x_sigma
            b[indices_ligne] .+= res
        else
            # Produit ACA : U * (V * x)
            if bloc.U !== nothing && size(bloc.U, 2) > 0
                z = bloc.V * x_sigma
                res = bloc.U * z
                b[indices_ligne] .+= res
            end
        end
    else
        Hmatvec_recursif!(bloc.gauche, x, b)
        Hmatvec_recursif!(bloc.droit, x, b)
        Hmatvec_recursif!(bloc.droite_bas, x, b)
        Hmatvec_recursif!(bloc.bas_droite, x, b)
    end
end

# --- 9. Main Execution ---

function main()
    Nsegments = 12800
    a = 1.0
    k_wave = 2 * pi
    n_gauss = 2
    
    println("--- Test pour N=$Nsegments ---")
    
    # Géométrie
    angles, nodes = generate_nodes(Nsegments, a)
    points_gauss, poids_gauss = gausslegendre(n_gauss)
    indices_originaux = collect(1:Nsegments)
    
    # 1. Matrice Dense
    println("Construction matrice dense...")
    t0 = time()
    BEM = matriceA(Nsegments, nodes, points_gauss, poids_gauss, k_wave)
    t_dense_build = time() - t0
    println("Temps construction Dense: $(round(t_dense_build, digits=4)) s")
    
    # 2. H-Matrice
    println("Construction H-Matrice (Arbres + Assemblage)...")
    t0 = time()
    racine_binaire = construire_arbre_bsp(nodes, indices_originaux, Nleaf=20)
    racine_blocs = construire_arbre_blocs(racine_binaire, racine_binaire, 3.0) # eta = 3.0
    
    assembler_h_matrice!(racine_blocs, Nsegments, nodes, points_gauss, poids_gauss, k_wave)
    t_hmat_build = time() - t0
    println("Temps assemblage H-Matrice: $(round(t_hmat_build, digits=4)) s")
    
    # 3. Comparaison MVM
    println("\nComparaison des produits MVM (moyenne sur 5 itérations)...")
    
    # Warmup (compilation JIT)
    x_warm = rand(ComplexF64, Nsegments)
    _ = BEM * x_warm
    _ = Hmatvec(racine_blocs, x_warm)
    
    t_dense_accum = 0.0
    t_hmat_accum = 0.0
    n_iter = 5
    
    for i in 1:n_iter
        x = rand(Nsegments) + 1im * rand(Nsegments)
        
        t1 = time()
        b_exact = BEM * x
        t2 = time()
        t_dense_accum += (t2 - t1)
        
        t1 = time()
        b_Hmat = Hmatvec(racine_blocs, x)
        t2 = time()
        t_hmat_accum += (t2 - t1)
        
        if i == 1
            err = norm(b_exact - b_Hmat) / norm(b_exact)
            println("Erreur relative (run 1): $(Float64(err))")
        end
    end
    
    avg_dense = t_dense_accum / n_iter
    avg_hmat = t_hmat_accum / n_iter
    
    println("\nTemps moyen MVM Dense     : $(avg_dense) s")
    println("Temps moyen MVM H-Matrice : $(avg_hmat) s")
    println("Accélération (Speedup)    : $(avg_dense / avg_hmat)x")
end

main()