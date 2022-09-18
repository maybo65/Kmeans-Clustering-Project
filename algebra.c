/*
 this c file contains functions that are responsible for algebraic calculations
 for the normalized spectral clustering algorithm.
 we chose to implement those functions in C inorder to improve the runtime of the project.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "algebra.h"
#define epsilon 0.0001

/// Returns the norm of the its ith column.
static double norm(double **U, int i, int n)
{
    double sum=0;
    int j;
    for(j=0; j<n; j++) {
        sum+=pow((U[j][i]),2);
    }
    return sqrt(sum);
}

/// Given two vectors of length d, x1 and x2 ,returns the euclidean
/// distance between them.
static double euclidean_distance(double *x1, double *x2, int d)
{
    double sum=0;
    int i;
    for(i=0; i<d; i++) {
        sum+= pow((x1[i]-x2[i]),2);
    }
    return sqrt(sum);
}

/// Given two matrices A and B with dim nxn, returns the dot product
/// of the ith column of A and and jth column of B.
static double col_col_mul(double **A, int i, double **B, int j, int n)
{
    double sum=0;
    int l;
    for(l=0; l<n; l++) {
        sum+=(A[l][i]*B[l][j]);
    }
    return sum;
}

/// Given 3 matrices U,Q,R from dim nxn, decomposes the matrix U
/// into a product U=QR of an orthogonal matrix Q and an upper
/// triangular matrix R, and updates the given matrices R and Q accordingly.
static void modified_gram_schmidt(double **U, double **R, double **Q, int n)
{
    int i,j,l;
    double s;
    for(i=0;i<n;i++){
        s=norm(U,i,n);
        R[i][i]=s;
        for (j=0;j<n;j++){
            if(s<=epsilon){
                Q[j][i]=0;
            }
            else{
                Q[j][i] = U[j][i]/s;
            }
        }
        for (j=i+1;j<n;j++){
            s=col_col_mul(Q,i,U,j,n);
            R[i][j]=s;
            for (l=0;l<n;l++){
                U[l][j]=(U[l][j]-s*Q[l][i]);
            }
        }
    }
    return;

}

/// Given two matrices A and B with dim nxn, returns the multiplication
/// of the ith row of A and and jth column of B.
/// l indicate the first cell in the ith row of A which is not necessarily zero
/// (in  some cases, A is an upper triangular matrix and in order to improve runtime,
/// there is no need to iterate over the cells under A's diagonal).
static double row_col_mul(double **A, int i,double **B, int j, int n, int l)
{
    double sum=0;
    for(;l<n; l++) {
        sum+=(A[i][l]*B[l][j]);
    }
    return sum;
}

/// Given three matrices A,B,C with dim nxn, updates the matrix C to be C=AxB.
static void mat_mul(double **A, double **B, double **C, int n)
{
    int i,j;
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            C[i][j]=row_col_mul(A,i,B,j,n,0);
        }
    }
    return;
}

/// Given three matrices A,B,C with dim nxn, where A ia an upper-triangular matrix,
/// updates the matrix C to be C=AxB
static void mat_mul_triangular(double **A, double **B, double **C, int n)
{
    int i,j;
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            C[i][j]=row_col_mul(A,i,B,j,n,i);
        }
    }
    return;
}

/// Returns a boolean flag of whether the QR iteration has converged
static int converge(double **Q1, double **Q , int n)
{
    int i,j;
    double d;
    for (i=0;i<n;i++){
        for(j=0;j<n;j++){
            d=fabs(fabs(Q1[i][j])-fabs(row_col_mul(Q1,i,Q,j,n,0)));
            if(d>epsilon){
                return 0;
            }
        }
    }
    return 1;
}

/// Prevents memory leak.
/// in case of failure frees the memory allocated in init_QR.
static int error_init_QR(double **Q, double **Q1, double **R, double **Q1copy, int i)
{
    int j;
    for(j=0;j<i;j++){
        free(Q[j]);
        free(Q1[j]);
        free(R[j]);
        free(Q1copy[j]);
    }
    if(Q[i]!=NULL)
        free(Q[i]);
    if(Q1[i]!=NULL)
        free(Q1[i]);
    if(R[i]!=NULL)
        free(R[i]);
    if(Q1copy[i]!=NULL)
        free(Q1copy[i]);
    PyErr_SetString(PyExc_TypeError, "init_QR : Memory allocation failed");
    return -1;
}

/// An auxiliary function to QR. initialize the matrices for the QR algorithm.
/// Returns -1 if there was an memory allocation which failed,
/// and 1 if the initialization has succeeded.
static int init_QR(double **Q, double **Q1, double **R, double **Q1copy, int n)
{
    int i,j;
    for (i = 0; i < n; i++) {
        Q1[i] = (double *)malloc(sizeof(double) * n);
        Q[i] = (double *)malloc(sizeof(double) * n);
        R[i] = (double *)malloc(sizeof(double) * n);
        Q1copy[i] = (double *)malloc(sizeof(double) * n);
        if (Q1[i]==NULL || Q[i]==NULL || R[i]==NULL || Q1copy[i]==NULL){
            return error_init_QR(Q,Q1,R,Q1copy, i);
        }
        Q1[i][i] = 1;
        for (j=0;j<n;j++){
            if(j!=i){
                Q1[i][j]=0;
            }
            if(i<j){
                R[i][j]=0;
            }
        }
    }
    return 1;
}

/// Frees the memory of given 2-dim arrays from size nxn.
static void free_matrices(double **A,double **B, double **C, int n)
{
    int i;
    for (i=0;i<n;i++){
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    return;
}


///Prevents memory leak. in case of failure frees the memory allocated in QR_iteration_c.
static double*** error_QR_iteration_c(double **Q, double **Q1,double **R, double **Q1copy, double ***ret){
    if(Q!=NULL)
        free(Q);
    if(Q1!=NULL)
        free(Q1);
    if(R!=NULL)
        free(R);
    if(Q1copy!=NULL)
        free(Q1copy);
    if(ret!=NULL)
        free(ret);
    PyErr_SetString(PyExc_TypeError, "QR_iteration_c : Memory allocation failed");
    return NULL;
}

/// Given a matrix A from dim nxn, returns two matrices, Q and A1 such:
/// Q is an orthogonal matrix whose columns approach the eigenvectors of A,
/// A1 whose diagonal elements approach the eigenvalues of A.
double*** QR_iteration_c(double **A, int n)
{
    double **Q,**Q1,**R,**Q1copy,**tmp;
    double *** ret;
    int i;
    Q1=(double **)malloc(sizeof (double*) * n);
    Q=(double **)malloc(sizeof (double*)  * n);
    R=(double **)malloc(sizeof (double*)  * n);
    Q1copy=(double **)malloc(sizeof (double*)  * n);
    ret=(double ***)malloc(sizeof (double**)  * 2);
    if (Q1==NULL || Q==NULL || R==NULL || Q1copy==NULL || ret==NULL){
        return error_QR_iteration_c(Q,Q1,R,Q1copy,ret);
    }
    if (init_QR(Q,Q1,R,Q1copy,n)==-1)
        return NULL;
    for (i=0;i<n;i++){
        modified_gram_schmidt(A,R,Q,n);
        mat_mul_triangular(R,Q,A,n);
        if (converge(Q1,Q,n)){
            free_matrices(Q,R,Q1copy,n);
            ret[0]=A;
            ret[1]=Q1;
            return ret;
        }
        mat_mul(Q1,Q,Q1copy,n);
        tmp=Q1;
        Q1=Q1copy;
        Q1copy=tmp;

    }
    free_matrices(Q,R,Q1copy,n);
    ret[0]=A;
    ret[1]=Q1;
    return ret;
}

///Prevents memory leak.
/// in case of failure frees the memory allocated in weighted_adjacency_matrix_c.
static double** error_weighted_adjacency_matrix_c(double **W, int i)
{
    int j;
    for (j=0;j<i;j++){
        free(W[i]);
    }
    if (W!=NULL)
        free(W);
    PyErr_SetString(PyExc_TypeError, "weighted_adjacency_matrix_c: Memory allocation failed");
    return NULL;
}

/// Given n vectors of length d, returns their affinity matrix.
double** weighted_adjacency_matrix_c(double **X, int n, int d)
{
    int i,j;
    i=-1;
    double val;
    double **W;
    W=(double **)malloc(n*sizeof(double *));
    if (W==NULL)
        return error_weighted_adjacency_matrix_c(NULL,i);
    for(i=0; i<n; i++) {
        W[i] = (double *)malloc(n * sizeof(double));
        if (W[i] == NULL)
            return error_weighted_adjacency_matrix_c(W,i);
    }
    for(i=0; i<n; i++) {
        W[i][i] = 0;
        for (j = 0; j < i; j++) {
            val = exp(-0.5 * (euclidean_distance(X[i], X[j], d)));
            W[i][j] = val;
            W[j][i] = val;
        }

    }
    for(i=0; i<n; i++)
        free(X[i]);
    free(X);
    return W;

}
