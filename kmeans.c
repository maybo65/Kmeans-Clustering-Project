/*
 this c file contains functions that are responsible for implementing the kmeans algorithm.
 the kmeans algorithm clusters n vectors into k centers. the n vectors are given as X's rows.
 the centroids of the clusters are M's rows.
 we chose to implement the algorithm using few data structures that are
 meant to improve run-time preferences.
 one of them is "a"- an array of linked-lists. each cluster is represented
 by a single linked list,
 and the nodes of that list are the vectors in X which are currently mapped to this cluster.
 we also use "XinS", which is an array of tuples. each tuple represents a
 specific vector. the first item
 in the tuple is the number of the cluster that the vector is currently mapped to,
 and the other item is a pointer to the node of that vector in a.
 lastly, the algorithm also uses Schenges, an array of len k which
 its ith cell indicates whether the ith centroid is not updated.

 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "kmeans.h"


/// Given two vectors of length d, x1 and x2, returns the squared euclidean distance between them.
static double squared_euclidean_distance(double *x1, double *x2, int d)
{
    double sum=0;
    int i;
    for(i=0; i<d; i++) {
        sum+= pow((x1[i]-x2[i]),2);
    }
    return sum;
}

/// Updates the centroid of the Sth cluster.
/// returns 1 if the Sth cluster is empty and 0 otherwise.
static int update_centroid(double **X, node **a, int S, int d, double* newcentroid)
{
    int counter,i,j;
    node* p;
    counter=0;i=0;
    p=a[S];
    for (j=0;j<d;j++)
        newcentroid[j] = 0;
    while (p){
        i=p->val;
        for (j = 0; j < d; j++) {
            newcentroid[j] += X[i][j];
        }
        p=p->next;
        counter++;
    }
    if (counter==0){
        return 1;
    }
    for (i=0;i<d;i++){
        newcentroid[i]=newcentroid[i]/counter;
    }
    return 0;
}

/// Updates the centroids of all of the clusters.
static void update_all_centroids(int K, int d, double **X, double **M, int *Schanges, double *newcentroid, node **a)
{
    int empty,i,j;
    for ( i=0 ; i<K ; i++ ){
        if (Schanges[i]==1){
            empty=update_centroid(X,a,i,d,newcentroid);
            if (empty==0) {
                for (j = 0; j < d; j++) {
                    M[i][j] = newcentroid[j];
                }
            }
            Schanges[i]=0;
        }
    }
    return;
}

/// Deletes the node represented by t from a.
static void delete_node(tuple* t,node **a)
{
    int index=t->index;
    node *pprev,*pnext;
    node *p=t->p;
    if (a[index]==p && !p->next){
        a[index]=NULL;
    }
    else if(a[index]==p && p->next){
        a[index]=p->next;
        p->next->prev=NULL;
    }
    else if(a[index]!=p && !p->next){
        p->prev->next=NULL;
    }
    else{

        pprev=p->prev;
        pnext=p->next;
        pprev->next=p->next;
        pnext->prev=p->prev;
    }
    return;
}

/// Inserts the node p at the beginning of the linked list in a[dest].
static void insert_node(node *p, int dest, node **a)
{
    p->next=a[dest];
    a[dest]=p;
    p->prev=NULL;
    if(p->next) {
        p->next->prev = p;
    }
    return;
}

/// Given a vector x, finds the closest cluster to it and returns its number.
static int find_closest_cluster(double *x, double **M, int K, int d)
{
    int minIndex,j;
    double minVal, dis;
    minIndex=0;
    minVal=squared_euclidean_distance(x,M[0],d);
    for(j=1; j<K; j++){
        dis = squared_euclidean_distance(x,M[j],d);
        if(dis<minVal){
            minIndex=j;
            minVal=dis;
        }
    }
    return minIndex;
}

/// Given N vectors of length d, determines for each vector which cluster it need to be mapped to.
/// returns flag that indicates if there is a vector that need to be in a different cluster than its current one.
static int rearange_clusters(int N,int K,int d, double **X,double **M, tuple **XinS, int *Schanges ,node **a)
{
    int flag,newcluster,i;
    flag=0;
    for(i=0 ; i<N ; i++ ){
        newcluster=find_closest_cluster(X[i],M,K,d);
        if (XinS[i]->index != newcluster){
            flag=1;
            if (XinS[i]->index!=-1) {
                Schanges[XinS[i]->index] = 1;
                delete_node(XinS[i],a);
                insert_node(XinS[i]->p,newcluster,a);
            }
            else {
                node *b;
                b=(node*)malloc(sizeof(node));
                if (b==NULL) {
                    PyErr_SetString(PyExc_TypeError, "rearange_clusters: Memory allocation failed");
                    return -1;
                }
                b->val=i;
                b->prev=NULL;
                b->next=NULL;
                XinS[i]->p=b;
                insert_node(b,newcluster,a);
            }
            Schanges[newcluster]=1;
            XinS[i]->index=newcluster;
        }
    }
    return flag;
}

/// Errors handling for clusters_to_file function.
static int error_clusters_to_file()
{
    PyErr_SetString(PyExc_TypeError, "clusters_to_file : could not open clusters file");
    return -1;
}

/// Print the clusters to the file "clusters.txt"
static int clusters_to_file(tuple **XinS, int K, int N)
{
    int i,j;
    int first=1;
    FILE *f = fopen("clusters.txt", "a");
    if (f==NULL){
        return (error_clusters_to_file());
    }
    long savedOffset = ftell(f);
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0){
        fprintf(f,"%d",K);
    }
    fseek(f, savedOffset, SEEK_SET);
    for ( i=0 ; i<K ; i++ ){
        fprintf(f,"\n");
        for (j=0 ; j<N ; j++ ){
            if (XinS[j]->index==i) {
                if (first==1){
                    fprintf(f,"%d", j);
                }
                else{
                    fprintf(f,",%d", j);
                }
                first=0;
            }
        }
        first=1;
    }
    fclose(f);
    return 1;
}

/// Prevents memory leak.
/// in case of failure frees the memory allocated in kmeams_c.
static int error_kmeams_c(int *Schanges, double *newcentroid)
{
    if (Schanges!=NULL)
        free (Schanges);
    if (newcentroid!=NULL)
        free (newcentroid);
    PyErr_SetString(PyExc_TypeError, "kmeans_c : memory allocation failed");
    return -1;
}

/// Frees the memory of the given arrays.
void free_memory(double **X, double **M, node **a, tuple **XinS, int N, int K)
{
    int i;
    for(i=0; i<N;i++){
        free(X[i]);
        free(XinS[i]->p);
        free(XinS[i]);
        if (i<K){
            free(M[i]);
        }
    }
    free(X);
    free(a);
    free(M);
    free(XinS);
    return;
}

/// Given n vectors of length d- the rows of X, clusters them into K clusters.
/// The K initial centroids are M's rows.
/// Returns -1 in case of a failure , and 1 if the function succeeded.
int kmeans_c(int K, int N, int d, int MAX_ITER, double **X,double **M, tuple **XinS, node **a)
{
    int flag,i;
    int* Schanges;
    double* newcentroid;
    flag=1;
    Schanges= (int*)malloc(K*sizeof(int));
    newcentroid= (double*)malloc(d*sizeof(double));
    if (Schanges==NULL || newcentroid==NULL){
        return error_kmeams_c(Schanges, newcentroid);
    }
    for (i=0;i<K;i++){
        Schanges[i]=0;
    }
    while(MAX_ITER>0 && flag==1) {
        flag=rearange_clusters(N,K,d,X,M,XinS,Schanges,a);
        if (flag==-1){
            return -1;
        }
        update_all_centroids(K,d,X,M,Schanges,newcentroid,a);
        MAX_ITER--;
    }
    if (clusters_to_file(XinS, K, N)==-1){
        return -1;
    }
    free(Schanges);
    free(newcentroid);
    return 1;
}


