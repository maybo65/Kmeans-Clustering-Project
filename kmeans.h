#ifndef KMEANS_H_
#define KMEANS_H_

typedef struct node{
    int val;
    struct node* next;
    struct node* prev;
}node;

typedef struct tuple{
    int index;
    node* p;
}tuple;

int kmeans_c(int K, int N,int d,int MAX_ITER,double** X,double** M,tuple ** XinS, node ** a);
void free_memory(double**X, double**M, node**a, tuple** XinS,int N, int K);

#endif