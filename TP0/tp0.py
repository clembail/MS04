import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, jv, hankel1

def generate_nodes(N, a):
    """
    Génère les N noeuds sur le cercle de rayon a centré en 0.
    Retourne un tableau numpy de taille (N, 2).
    """
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)  # angles uniformes
    x = a * np.cos(angles)
    y = a * np.sin(angles)
    return np.column_stack((x, y))

def generate_segments_indices(N):
    """
    Génère la liste des segments comme paires d’indices.
    Chaque noeud est relié au suivant (avec retour au premier).
    """
    return [(k, (k+1) % N) for k in range(N)]

def plot_mesh(nodes, segments, a):
    """
    Trace le cercle de rayon a et le maillage (segments + noeuds).
    """
    fig, ax = plt.subplots(figsize=(6,6))

    # Tracé du cercle "continu" pour comparaison
    circle = plt.Circle((0,0), a, color='lightgray', fill=False, linestyle='--')
    ax.add_artist(circle)

    # Tracé des segments
    for (u,v) in segments:
        x_vals = [nodes[u,0], nodes[v,0]]
        y_vals = [nodes[u,1], nodes[v,1]]
        ax.plot(x_vals, y_vals, 'b-')  # segments en bleu

    # Tracé des noeuds
    ax.scatter(nodes[:,0], nodes[:,1], color='red', zorder=5)

    # Mise en forme
    ax.set_aspect('equal')
    ax.set_title(f"Maillage du bord du disque (N={len(nodes)})")
    plt.show()


## calcul de p sur Gamma

def p(r,theta,k):
    costheta = np.cos(theta)
    j = complex(0,1)
    sum = 0
    for n in range(-100,100):
        Jn = jv(n,k*a)
        Hn = hankel1(n,k*a)
        Hnm = hankel1(n-1,k*r)
        Hnp = hankel1(n+1,k*r)
        const = (-j)**n * k/2
        sum += const * Jn * (Hnm-Hnp)*np.exp(-n*theta*j)/Hn
    sum += j * k * costheta * np.exp(-j*k*r*costheta)
    return sum


# ----------------- Programme principal -----------------
if __name__ == "__main__":
    N = 300  # nombre de noeuds (par ex. 12)
    a = 1.0  # rayon du cercle
    k = 20

    nodes = generate_nodes(N, a)
    segments = generate_segments_indices(N)

    print(f"Nombre de noeuds: {len(nodes)}")
    print(f"Nombre de segments: {len(segments)} (égal à N)")

    # print("\nNoeuds:")
    # for i, (x,y) in enumerate(nodes):
    #     print(f"{i}: ({x:.6f}, {y:.6f})")

    # print("\nSegments:")
    # for i, (u,v) in enumerate(segments):
    #     print(f"{i}: {u} -> {v}")

    pVec = np.vectorize(p)
    theta = np.linspace(0,2*np.pi,N)
    Y = np.abs(p(a,theta,k))

    # Tracé
    plt.plot(theta,Y)
    plt.show()
    #plot_mesh(nodes, segments, a)