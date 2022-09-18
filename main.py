import argparse
import numpy as np
import math
import random
import sklearn.datasets
import matplotlib.pyplot as plt
import Interface

epsilon=0.0001
MAX_ITER = 300
MAX_K_2=20
MAX_N_2=500
MAX_K_3=20
MAX_N_3=500


def QR(A):
    """returns an orthogonal matrix Q1 whose columns approach the
    eigenvectors of A, and a matrix A1 whose diagonal elements
    approach the eigenvalues of A"""
    N=A.shape[0]
    Z = Interface.QR(A.tolist(),N)
    A1=np.asarray(Z[0])
    Q1=np.asarray(Z[1])
    return A1,Q1

def weighted_adjacency_matrix(X):
    """returns the Weighted Adjacency Matrix of X"""
    N=X.shape[0]
    d=X.shape[1]
    W = np.array(Interface.weighted_adjacency_matrix(X.tolist(),N,d))
    return W

def degree_matrix(W):
    """calculate D - the diagonal degree matrix of W , and returns D^-0.5"""
    S = np.sum(W, axis=1)
    f = lambda t: 1 / (math.sqrt(t)) if math.pow(t, 2) > epsilon else 0
    D = np.array(list(map(f, S)))
    return np.diag(D)

def normalized_graph_laplacian(X):
    """returns the Normalized Graph Laplacian of W """
    N=X.shape[0]
    W=weighted_adjacency_matrix(X)
    D=degree_matrix(W)
    # in order to make sure there is no division by zero in the mgs algorithm,
    # we chose to add an small constant to L's diagonal. this makes sure
    # that L is an inverse matrix, but due to the fact we added such a small number,
    # this does not impair the correctness of the algorithm, increases the jacquard
    # score significantly in comparison to other solutions we tried, and also minimizes
    # the number of times where there is still a division by zero.
    e=np.diag(np.full((N,),epsilon*1.5))
    L= np.identity(N)-(D @ W @ D)+e
    return L

def determine_num_of_clusters(eigenvalues,k,Random):
    """returns the number of clusters determined by The Eigengap Heuristic,
    and the sorting order of the eigenvalues"""
    sortingorder = eigenvalues.argsort()
    if (not Random):
        return k,sortingorder
    N = eigenvalues.shape[0]
    eigenvalues=eigenvalues[sortingorder]
    maxdis =eigenvalues[1]-eigenvalues[0]
    maxindex =0
    for i in range(1, N//2):
        dis = eigenvalues[i+1]-eigenvalues[i]
        if (dis > maxdis):
            maxdis = dis
            maxindex = i
    return maxindex+1,sortingorder

def makeT(U):
    """form a matrix T from U by renormalizing each of U’s rows to have unit length"""
    N=U.shape[0]
    for i in range(N):
        norm=np.linalg.norm(U[i])
        if (norm>=epsilon):
            U[i] = U[i]/norm
    return U

def normalized_spectral_clustering(X,Random,k):
    """given n vectors (X's rows) clusters them to K cluster.
    if rand is set to False, the algorithem uses the given k. otherwise,
    K is computed using the eigengap heuristic. """
    lnorm = normalized_graph_laplacian(X)
    A, Q= QR(lnorm)
    k, ind = determine_num_of_clusters(A,k,Random)
    N=A.shape[0]
    if(k==1):
        Interface.clusters_to_file_k_is_1(N)
        return(1)
    U = Q[:, ind][:,:k]
    T = makeT(U)
    kmeans(T,k)
    return k

def kmeans(X, K):
    """given n vectors (X's rows), select randomly K of them, while the
    probability to choose a specific vector determine by the distance of
     it to the vectors that have already been selected."""
    N=X.shape[0]
    d=X.shape[1]
    np.random.seed(0)
    M =np.zeros((K, d))
    XinS=np.full((1,N),-1, dtype=int)[0]
    indieces = np.zeros(K)
    r=np.random.choice(N, 1)[0]
    M[0] = X[r]
    XinS[r]=0
    indieces[0]=r
    D=np.power((X - M[0]), 2).sum(axis=1)
    P = D/D.sum(keepdims=1)
    r=np.random.choice(N, 1, p=P)
    indieces[1]=r
    M[1]=X[r]
    XinS[r] = 1
    for j in range(2,K):
        D=np.minimum(np.power((X - M[j-1]), 2).sum(axis=1),D)
        P=D / D.sum(keepdims=1)
        r=np.random.choice(N, 1, p=P)
        indieces[j]=r
        M[j]=X[r]
        XinS[r] = j
    Interface.kmeans(K, N, d, MAX_ITER, X.tolist(), M.tolist(), XinS.tolist())

def visualization(X,N,K,Y):
    """create clusters.pdf  file and fill it with the visualization of the clusters"""
    f = open("clusters.txt", "r")
    usedK = int(f.readline())
    fig = plt.figure()
    dim=X.shape[1]
    if (dim==2):
        countkmeans, countnsc, bothkmeans, bothnsc = add_2_dim_plot(f,fig,X,Y,usedK)
    else:
        countkmeans, countnsc, bothkmeans, bothnsc = add_3_dim_plot(f,fig,X,Y,usedK)
    if (usedK!=1):
        jnsc,jkmeans=jaccard(countkmeans, countnsc, bothkmeans, bothnsc,Y)
    else:
        jnsc,jkmeans=1.,1.
    add_text(fig,N, K, jnsc, jkmeans, usedK)
    fig.savefig('clusters.pdf')
    plt.close(fig)
    f.close()

def add_text(fig,N,K,jnsc,jkmeans,usedK):
    """adds text to the visualization file"""
    p3 = fig.add_subplot(235)
    p3.set_axis_off()
    l1 = "Data was genrated from the values:\n"
    l2 = "n = " + str(N) + " , k = " + str(K) + "\n"
    l3 = "The k that was used for both algorithms was " + str(usedK) + "\n"
    l4 = "The jaccard measure for Spectral Clustering: " + str(round(jnsc, 2)) + "\n"
    l5 = "The jaccard measure for K-means: " + str(round(jkmeans, 2))
    txt = l1 + l2 + l3 + l4 + l5
    p3.text(0.5, 0.2, txt, ha="center", size=14)

def jaccard(countkmeans, countnsc, bothkmeans, bothnsc,Y):
    """calculate the jaccard measure of the k-means and the Normalized Spectral Clustering"""
    unique, counts = np.unique(Y, return_counts=True)
    choose2 = lambda t: 0.5 * (t * (t - 1))
    counts = np.array(list(map(choose2, counts)))
    y = np.sum(counts)
    kmeans = bothkmeans / (countkmeans + y - bothkmeans)
    nsc = bothnsc / (countnsc + y - bothnsc)
    return nsc,kmeans



def add_2_dim_plot(f,fig,X,Y,K):
    """creates a 2 dim plot of the clusters calculated by kmeans and
    normalized spectral clustering.
     returns the follows:
    countkmeans: total number of pairs that are in the same cluster in kmeans.
    countnsc: total number of pairs that are in the same cluster in normalized
    spectral clustering.
    bothkmeans: total number of pairs that are in the same cluster in both
    kmeans and the generated dataset.
    bothnsc: total number of pairs that are in the same cluster in both normalized
    spectral clustering and the generated dataset."""
    countkmeans ,countnsc ,bothkmeans ,bothnsc = 0,0,0,0
    p1 = fig.add_subplot(221)
    for i in range(1, K + 1):
        bothnsc, countnsc = add_2_dim_vectors(f, p1, X, Y, bothnsc, countnsc)
    p1.set_title('Normalized Spectral Clustering', size=10)
    p1.grid(lw=0.2)
    p2 = fig.add_subplot(222)
    for i in range(K + 1, 2 * K + 1):
        bothkmeans, countkmeans = add_2_dim_vectors(f, p2, X, Y, bothkmeans, countkmeans)
    p2.set_title('k-means', size=10)
    p2.grid(lw=0.2)
    return countkmeans, countnsc, bothkmeans, bothnsc

def add_3_dim_plot(f,fig,X,Y,K):
    """creates a 3 dim plot of the clusters calculated by kmeans and
    normalized spectral clustering.
     returns the follows:
    countkmeans: total number of pairs that are in the same cluster in kmeans.
    countnsc: total number of pairs that are in the same cluster in normalized
    spectral clustering.
    bothkmeans: total number of pairs that are in the same cluster in both
    kmeans and the generated dataset.
    bothnsc: total number of pairs that are in the same cluster in both normalized
    spectral clustering and the generated dataset."""
    countkmeans ,countnsc ,bothkmeans ,bothnsc = 0,0,0,0
    p1 = fig.add_subplot(221, projection='3d')
    for i in range(1, K + 1):
        bothnsc, countnsc = add_3_dim_vectors(f, p1, X, Y, bothnsc, countnsc)
    p1.set_title('Normalized Spectral Clustering', size=10)
    p1.grid()
    p2 = fig.add_subplot(222, projection='3d')
    for i in range(K + 1, 2 * K + 1):
        bothkmeans, countkmeans = add_3_dim_vectors(f, p2, X, Y, bothkmeans, countkmeans)
    p2.set_title('k-means', size=10)
    p2.grid()
    return countkmeans, countnsc, bothkmeans, bothnsc

def add_2_dim_vectors(f,p,X,Y,both,count):
    """adds to a plot 2 dim vectors, cluster by cluster.
     returns the follows:
    both: total number of pairs that are in the same cluster in both
    the computed solution and the generated dataset.
    count: total number of pairs that are in the same cluster
    in the computed solution."""
    indices = list(map(int, f.readline().split(',')))
    filtered_label = X[indices, :]
    p.scatter(filtered_label[:, 0], filtered_label[:, 1],linewidths=0.4)
    for i in range(len(indices)):
        for j in range(i+1,len(indices)):
            if (Y[indices[i]]==Y[indices[j]]):
                both+=1
            count+=1
    return both,count

def add_3_dim_vectors(f,p,X,Y,both,count):
    """adds to a plot 3 dim vectors, cluster by cluster.
     returns the follows:
    both: total number of pairs that are in the same cluster in both
    the computed solution and the generated dataset.
    count: total number of pairs that are in the same cluster
    in the computed solution."""
    indices = list(map(int, f.readline().split(',')))
    filtered_label = X[indices, :]
    p.scatter(filtered_label[:, 0], filtered_label[:, 1], filtered_label[:,2], linewidths=0.4)
    for i in range(len(indices)):
        for j in range(i+1,len(indices)):
            if (Y[indices[i]]==Y[indices[j]]):
                both+=1
            count+=1
    return both,count

def generation(N,K,Random):
    """returns X- N samples with K centers generated by make_blobs,
     Y- The integer labels for cluster membership of each sample.
    returns the N,K that were used."""
    d = random.choice([2, 3])
    if (Random):
        if (d==2):
            K=random.randint(MAX_K_2//2,MAX_K_2+1)
            N = random.randint(MAX_N_2 // 2, MAX_N_2+1)
        else:
            K=random.randint(MAX_K_3//2,MAX_K_3+1)
            N = random.randint(MAX_N_3//2, MAX_N_3+1)
    X, Y = sklearn.datasets.make_blobs(n_samples=N, n_features=d, centers=K)
    Interface.data_to_file(X.tolist(), Y.tolist(),N, d)
    return X,Y,N,K

def main(N,K,Random):
    X,Y,N,K = generation(N,K,Random)
    clusters = open("clusters.txt", "w")
    usedK = normalized_spectral_clustering(X,Random,K)
    if(usedK!=1):
        kmeans(X, usedK)
    clusters.close()
    visualization(X,N,K,Y)

def arguments():
    """checks that the supplied arguments are legal. if not, raises an error, otherwise returns them"""
    parser = argparse.ArgumentParser()
    parser.add_argument('arguments', type=int ,nargs='+')
    try:
        args = parser.parse_args()
    except:
        raise Exception("k and N must be positive integers.")
    if (len(args.arguments)<3):
        raise Exception("k and n must be provided when Random is set to false.")
    K = args.arguments[0]
    N =  args.arguments[1]
    Random = bool(args.arguments[2])
    if ((not Random) and (K < 1 or N < 1)):
        raise Exception("k and N must be positive integers.")
    if ((not Random) and (K >= N)):
        raise Exception("K must be smaller then N.")
    return N,K,Random

print("The maximum capacity for 2-dimensional data points is K = "+str(MAX_K_2) +" and n = "+str(MAX_N_2)+".")
print("The maximum capacity for 3-dimensional data points is K = "+str(MAX_K_3) +" and n = "+str(MAX_N_3)+".")
N,K,Random = arguments()
main(N,K,Random)




