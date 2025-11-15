import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import roots_legendre, jv, j0, hankel1

# TP0

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

def uPlus(a,r,theta,k,ite=60):
    S = 0+0j
    for i in range(-ite,ite+1):
        Jn = jv(i,k*a)
        Hna = hankel1(i,k*a)
        Hnr = hankel1(i,k*r)
        S += (-1j)**i * Jn * Hnr * np.exp(-i*theta*1j)/Hna
    return -S
# uplus = np.vectorize(uplus)

def p(a,r,theta,k,ite=60):
    costheta = np.cos(theta)
    S = 0+0j
    for i in range(-ite,ite+1):
        Jn = jv(i,k*a)
        Hn = hankel1(i,k*a)
        Hnm = hankel1(i-1,k*r)
        Hnp = hankel1(i+1,k*r)
        const = (-1j)**i
        S += const * Jn * (Hnm-Hnp)*np.exp(-i*theta*1j)/Hn
    S *= k/2
    S += 1j * k * costheta * np.exp(-1j*k*r*costheta)
    return S
# p = np.vectorize(p)

# TP1

### question 4

def GaussLegendre(f,a,b,points,poids):
    n = len(points)
    x, w = points, poids
    S = 0.0
    for i in range(n):
        t = x[i]
        u = (b-a)/2*t + (b+a)/2
        S += w[i]*f(u)
    return (b-a)/2*S

def integrale2D(f,n1,n2,points,poids): 
    # n1,n2 points dans le plan aux extrémités du segment
    # f fonction à deux variables
    x1,y1 = n1[0],n1[1]
    x2,y2 = n2[0],n2[1]
    norm = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    def g(t):
        X = x1*(1-t)+x2*t
        Y = y1*(1-t)+y2*t
        # print("X,Y =", X,Y)
        return f(X,Y)*norm
    return GaussLegendre(g,0,1,points,poids)

### question 5

def G(n1,n2,k):
    x1,y1 = n1
    x2,y2 = n2
    z1 = x1 + 1j*y1
    z2 = x2 + 1j*y2
    return 1j*hankel1(0,k*np.abs(z2-z1))/4

def milieu(e,nodes):
    j = e[0]
    j2 = e[1]
    (x1,y1) = nodes[j]
    (x2,y2) = nodes [j2]
    return ((x1+x2)/2,(y1+y2)/2)

def representationIntegrale(N,points,poids,q,G,segments,nodes,a,k):
    l = len(q)
    M = np.zeros((l,N),dtype=complex)
    V = np.zeros(N,dtype=complex)
    for e in segments: #calcul des milieux pe
        j = e[0]
        (x,y) = milieu(e,nodes)
        z = x + 1j*y
        r = np.abs(z)
        theta = np.angle(z)
        V[j] = p(a,r,theta,k)
    for i in range(l): #calcul de la matrice contenant les intégrales
        n1 = q[i]
        for e in segments:
            j=e[0]
            j2=e[1]
            M[i][j] = integrale2D((lambda X,Y: G(n1,(X,Y),k)),nodes[j],nodes[j2],points,poids)
    U = np.matmul(M,V)
    return U

### question 6

if __name__ == "__main__":
    N = 200  # nombre de noeuds (par ex. 12)
    Nxi = 100
    a = 1.0  # rayon du cercle
    k = 2*np.pi
    n = 2 # nombre de terme dans la somme pour Gauss-Legendre

    nodes = generate_nodes(N, a)
    segments = generate_segments_indices(N)

    print(f"Nombre de noeuds: {len(nodes)}")
    print(f"Nombre de segments: {len(segments)} (égal à N)")

    points, poids = np.polynomial.legendre.leggauss(n)

    ################ Tracés ##################

    ######## partie reelle/imaginaire de p

    # theta = np.linspace(-np.pi,np.pi,N)
    # Y = p(a,a,theta,k)
    # Yr = np.real(Y)
    # Yi = np.imag(Y)
    # Ym = np.abs(Y)

    # plt.plot(theta,Ym, label="module de p sur $\Gamma$")
    # plt.plot(theta,Yr, '+', label="partie réelle de p sur $\Gamma$")
    # plt.plot(theta,Yi, '*', label="partie imaginaire de p sur $\Gamma$")
    # plt.title("tracé des parties réelles/imaginaires de p")
    # plt.xlabel("theta")
    # plt.legend()
    # plt.grid()
    # plt.show()

    ######## partie reelle/imaginaire de u+

    xi = np.linspace(a,a*5,Nxi)
    x = [(0,i) for i in xi]
    N = np.arange(50,500,25,dtype=int)

    for k in [np.pi/2,np.pi,np.pi*2,np.pi*4]:
        e = []
        for n in N:
            nodes = generate_nodes(n, a)
            segments = generate_segments_indices(n)
            uplus = uPlus(a,xi,0,k)
            uNum = representationIntegrale(n,points,poids,x,G,segments,nodes,a,k)
            E = np.abs((uplus - uNum)/uplus)
            e.append(np.max(E))
        plt.plot(N,e, marker="*", linestyle="None", label="k="+str(k))

    uplus = uPlus(a,xi,np.pi/2,k)
    uNum = representationIntegrale(N,points,poids,x,G,segments,nodes,a,k)
            

    R1 = np.real(uNum)
    R = np.real(uplus)
    I1 = np.imag(uNum)
    I = np.imag(uplus)

    # plt.xlabel("nombre de noeuds dans le maillage")
    # plt.ylabel("erreur relative maximale")
    # plt.title("Convergence représentaton intégrale")
    # plt.legend()
    # plt.grid()
    # plt.show()

    plt.plot(xi,R1, color="blue", label="Re($u^+$) numérique", marker="+", linestyle='None')
    plt.plot(xi,R, color="orange", label="Re($u^+$) analytique")
    plt.plot(xi,I1, color="green", label="Im($u^+$) numérique", marker="*", linestyle="None")
    plt.plot(xi,I, color="red", label="Im($u^+$) analytique", linestyle="--")
    plt.title("tracé des parties réelles et imaginaires de $u^+$")

    plt.xlabel("x")
    plt.legend()
    plt.grid()
    plt.show()