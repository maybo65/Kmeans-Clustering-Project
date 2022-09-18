/*This file is a C extension interface for python.
 * it contains five wrapper function that are invoked from python:
  1. "clusters_to_file_k_is_1"- handling the case which K is 1.
  2. "data_to_file"- printing the data to the data.txt file.
  3. "QR"- calculates the QR decomposition.
  4. weighted_adjacency_matrix"- calculate the weighted matrix for a given graph.
  5. "kmeans"- clusters n vectors into k clusters via the K-means algorithm.
 those functions uses some other c functions from the imported headers files, which
 get pure c objects. thus the wrappers functions convert the inputs from python to c
 object using some auxiliary functions in this module. likewise, before returning values
 to python, convert from c object to python object with other auxiliary functions in this
 file. the remaining functions in this file are error-handling functions that making sure
 that in case of failure there is no memory-leak in the program.
 */


#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "kmeans.h"
#include "algebra.h"
#include "print.h"

#define epsilon 0.0001

/// Given a PyObject array of length list_size, returns a converted list of doubles.
static double* list_of_doubles_to_c(PyObject *list, Py_ssize_t list_size)
{
    Py_ssize_t i;
    PyObject *item;
    double *lst = (double*)malloc(sizeof (double) * list_size);
    if (lst == NULL){
        return NULL;
    }
    for (i = 0; i < list_size; i++) {
        item = PyList_GetItem(list, i);
        if (!PyFloat_Check(item)) continue;
        lst[i] = PyFloat_AsDouble(item);
    }
    return lst;


}

/// Prevents memory leak. in case of failure frees the memory allocated in list_of_lists_to_c.
static double** error_list_of_lists_to_c(double **list, Py_ssize_t i)
{
    int j;
    for (j=0;j<i;j++){
        free(list[i]);
    }
    if (list!=NULL)
        free(list);
    if (i==-1)
        PyErr_SetString(PyExc_TypeError, "list_of_lists_to_c: Memory allocation failed");
    else
        PyErr_SetString(PyExc_TypeError, "list_of_doubles_to_c: Memory allocation failed");
    return NULL;

}

/// Given a PyObject matrix of size list_size*list_size, returns a converted list of lists
static double** list_of_lists_to_c(PyObject *_list)
{
    PyObject *item;
    Py_ssize_t i, n;
    if (!PyList_Check(_list)){
        return NULL;
    }
    n = PyList_Size(_list);
    double **lst = (double**)malloc(sizeof (double*)* n);
    if (lst == NULL){
        return error_list_of_lists_to_c(lst,-1);
    }
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(_list, i);
        if (!PyList_Check(item)){
            continue;
        }
        lst[i]=list_of_doubles_to_c(item, PyList_Size(item));
        if (lst[i]==NULL){
            return error_list_of_lists_to_c(lst,i);;
        }

    }
    return lst;


}

/// Prevents memory leak. in case of failure frees A.
static PyObject* error_in_wrapper_function(double **A, int n)
{
    int i;
    for (i=0;i<n;i++){
        if (A[i]!=NULL)
            free(A[i]);
    }
    free(A);
    return NULL;
}

/// Prevents memory leak. in case of failure frees Aret.
static void free_PyList(PyObject *Aret, int i)
{
    int j;
    for (j = 0; j < i; j++){
        Py_XDECREF(PyList_GetItem(Aret, j));
    }
    if (i!=-1)
        Py_XDECREF(Aret);
}

/// Prevents memory leak.
/// in case of failure frees the memory allocated in list_of_lists_to_python.
static PyObject* error_list_of_lists_to_python(PyObject *Aret, int i)
{
    free_PyList(Aret,i);
    PyErr_SetString(PyExc_TypeError, "list_of_lists_to_python: Memory allocation failed");
    return NULL;

}

/// Given a two dim C array A from dim nxn, returns a PyObject of A
static PyObject* list_of_lists_to_python(double **A, int n)
{
    PyObject *A_i, *a, *Aret;
    int i,j;
    Aret = PyList_New(n);
    if(Aret==NULL)
        return error_list_of_lists_to_python(Aret, -1);
    for (i = 0; i < n; i++)
    {
        A_i = PyList_New(n);
        if(A_i==NULL)
            return error_list_of_lists_to_python(Aret, i);
        for (j = 0; j < n; j++){
            a = Py_BuildValue("f",A[i][j]);
            PyList_SetItem(A_i,j,a);
        }
        PyList_SetItem(Aret,i,A_i);
        free(A[i]);
    }
    free(A);
    return Aret;

}

/// Prevents memory leak.
/// in case of failure frees the memory allocated in list_of_doubles_to_python.
static PyObject* error_list_of_doubles_to_python()
{
    PyErr_SetString(PyExc_TypeError, "list_of_doubles_to_python: Memory allocation failed");
    return NULL;
}

/// Given a C array A of length n, returns a PyObject of A
static PyObject* list_of_doubles_to_python(double **A, int n)
{
    PyObject *a, *Aret;
    int i;
    Aret = PyList_New(n);
    if (Aret==NULL)
        return error_list_of_doubles_to_python();
    for (i = 0; i < n; i++)
    {
        a = Py_BuildValue("f",A[i][i]);
        PyList_SetItem(Aret,i,a);
        free(A[i]);
    }
    free(A);
    return Aret;

}

/// Prevents memory leak.
/// in case of failure frees the memory allocated in build_and_ret.
static PyObject* error_build_and_ret(PyObject* Aret, PyObject* Qret, double ***A_Q ,int n){
    if (Aret==NULL)
        error_in_wrapper_function(A_Q[0],n);
    else
        Py_XDECREF(Aret);
    if (Qret==NULL)
        error_in_wrapper_function(A_Q[1],n);
    else
        free_PyList(Qret,n);
    free(A_Q);
    return NULL;
}

/// Given two matrices A and Q from dim nxn,
/// returns a PyObject (tuple) of the diagonal of A and the matrix Q as PyObjects
static PyObject* build_and_ret(double ***A_Q, int n)
{
    PyObject* Aret ,*Qret;
    double**A,**Q;
    A=A_Q[0];
    Q=A_Q[1];
    Aret=list_of_doubles_to_python(A,n);
    Qret=list_of_lists_to_python(Q,n);
    if (Qret==NULL || Aret==NULL){
        return error_build_and_ret(Aret,Qret,A_Q,n);
    }
    PyObject *t = PyList_New(2);
    PyList_SetItem(t,0,Aret);
    PyList_SetItem(t,1,Qret);
    free(A_Q);
    return t;
}

/// Wrapper function to QR_iteration_c
static PyObject* QR(PyObject *self, PyObject *args)
{
    PyObject *A1;
    int n;
    double **A;
    double*** A_Q;
    if(!PyArg_ParseTuple(args, "Oi",&A1,&n)) {
        return NULL;
    }
    A=list_of_lists_to_c(A1);
    if (A==NULL)
        return NULL;
    A_Q=QR_iteration_c(A,n);
    if (A_Q==NULL)
        return error_in_wrapper_function(A,n);
    return(build_and_ret(A_Q,n));

}

/// Wrapper function to weighted_adjacency_matrix_c
static PyObject* weighted_adjacency_matrix(PyObject *self, PyObject *args)
{
    PyObject *X1, *ret;
    int n,d;
    double **X, **W;
    if(!PyArg_ParseTuple(args, "Oii",&X1,&n,&d)) {
        return NULL;
    }
    X=list_of_lists_to_c(X1);
    if (X==NULL){
        return NULL;
    }
    W=weighted_adjacency_matrix_c(X,n,d);
    if (W==NULL){
        return error_in_wrapper_function(X,n);
    }
    ret=list_of_lists_to_python(W,n);
    if (ret==NULL)
        return error_in_wrapper_function(W,n);
    return ret;
}

/// Frees XinS
static void free_XinS(tuple **XinS,int n)
{
    int i;
    tuple* x;
    node* y;
    for (i=0;i<n;i++){
        x=XinS[i];
        if (x){
            y=x->p;
            if (y){
                free(y);
            }
            free(x);
        }
    }
    free(XinS);
    return;
}

/// Prevents memory leak.
/// in case of failure frees the memory allocated in list_of_tuples.
static tuple** error_list_of_tuples(tuple **XinS,int n)
{
    free_XinS(XinS,n);
    PyErr_SetString(PyExc_TypeError, "list_of_tuples: Memory allocation failed");
    return NULL;
}

/// Builds a list of tuples where every tuple represents a vector from X.
/// Updates the tuples of the initiate vectors and the list of linked lists accordingly.
static tuple** list_of_tuples(PyObject * list, Py_ssize_t list_size ,node** a)
{
    Py_ssize_t i;
    PyObject *item;
    tuple **lst = (tuple**)malloc(sizeof (tuple*) * list_size);
    if (lst==NULL){
        PyErr_SetString(PyExc_TypeError, "list_of_tuples: Memory allocation failed");
        return NULL;
    }
    for (i = 0; i < list_size; i++) {
        item = PyList_GetItem(list, i);
        if (!PyLong_Check(item)){
            continue;
        }
        tuple* c;
        c=(tuple*)malloc(sizeof(tuple));
        if (c==NULL){
            return error_list_of_tuples(lst,(int)i);
        }
        c->index=PyLong_AsLong(item);
        c->p=NULL;
        lst[i] = c;
        if (lst[i]->index!=-1){
            node* b;
            b=(node*)malloc(sizeof(node));
            if (b==NULL){
                return error_list_of_tuples(lst,(int)list_size);
            }
            b->val=i;
            b->prev=NULL;
            b->next=NULL;
            a[lst[i]->index]=b;
            lst[i]->p=b;
        }
    }
    return lst;
}

/// Prevents memory leak.
/// in case of failure frees the memory allocated in the initialization part of kmeans function.
static PyObject* error_kmeans_init(double **X, double **M, node **a, int n, int k)
{
    if (X!=NULL)
        error_in_wrapper_function(X,n);
    if (M!=NULL)
        error_in_wrapper_function(M,k);
    if(a!=NULL)
        free(a);
    else
        PyErr_SetString(PyExc_TypeError, "kmeans: Memory allocation failed");
    return NULL;
}

/// Prevents memory leak.
/// in case of failure frees the memory allocated in the kmeans function.
static PyObject* error_kmeans(double **X, double **M, node **a,tuple** XinS, int n, int k)
{
    free_XinS(XinS,n);
    error_kmeans_init(X, M,a, n, k);
    return NULL;
}

/// Wrapper function to kmeans_c
static PyObject* kmeans(PyObject *self, PyObject *args)
{
    double **X, **M;
    tuple **XinS;
    node** a;
    PyObject *M1,*X1;
    PyObject *XinS1;
    int K,N,d,MAX_ITER;
    if(!PyArg_ParseTuple(args, "iiiiOOO", &K, &N, &d, &MAX_ITER, &X1, &M1, &XinS1)) {
        PyErr_SetString(PyExc_TypeError, "kmeans: incomptible argument");
        return NULL;
    }
    X = list_of_lists_to_c(X1);
    M = list_of_lists_to_c(M1);
    a = (node**)malloc((sizeof(node*))*K);
    if (a==NULL || M==NULL || X==NULL){
        return error_kmeans_init(X, M, a, N, K);
    }
    XinS = list_of_tuples(XinS1,N,a);
    if (XinS==NULL){
        return error_kmeans_init(X, M, a, N, K);
    }
    if (kmeans_c(K,  N, d, MAX_ITER, X, M, XinS,a)==-1){
        return error_kmeans(X, M, a,XinS, N, K);
    }
    free_memory(X,M, a, XinS, N,K);
    Py_RETURN_NONE;
}

/// Given a PyObject array of length list_size, returns a converted list of ints
static int* list_of_ints_to_c(PyObject * list, Py_ssize_t list_size)
{
    Py_ssize_t i;
    PyObject *item;
    int *lst = (int*)malloc(sizeof (int) * list_size);
    if (lst==NULL){
        PyErr_SetString(PyExc_TypeError, "list_of_ints_to_c: Memory allocation failed");
        return NULL;
    }
    for (i = 0; i < list_size; i++) {
        item = PyList_GetItem(list, i);
        lst[i] = PyLong_AsLong(item);
        lst[i] = PyLong_AsLong(item);
    }
    return lst;

}

/// Wrapper function to clusters_to_file_k_is_1_c
static PyObject* clusters_to_file_k_is_1(PyObject *self, PyObject *args)
{
    int N;
    if(!PyArg_ParseTuple(args, "i",&N)) {
        PyErr_SetString(PyExc_TypeError, "clusters_to_file_k_is_1: incomptible argument");
        return NULL;
    }
    if(clusters_to_file_k_is_1_c(N)==-1){
        printf("aaa\n");
        return NULL;
    }
    Py_RETURN_NONE;

}

/// Prevents memory leak.
/// in case of failure frees the memory allocated in data_to_file.
static PyObject* error_data_to_file(double **X, int *Y, int n)
{
    if (Y!=NULL)
        free(Y);
    if(X!=NULL)
        return error_in_wrapper_function(X,n);
    return NULL;
}

/// Wrapper function to data_to_file
static PyObject* data_to_file(PyObject *self, PyObject *args)
{
    PyObject *X1, *Y1;
    int n,d;
    double **X;
    int *Y;
    if(!PyArg_ParseTuple(args, "OOii",&X1,&Y1,&n,&d)) {
        PyErr_SetString(PyExc_TypeError, "data_to_file : incomptible argument");
        return NULL;
    }
    X=list_of_lists_to_c(X1);
    Y=list_of_ints_to_c(Y1,n);
    if (X==NULL || Y==NULL){
        return error_data_to_file(X,Y,n);
    }
    if (data_to_file_c(X,Y,n,d)==-1){
        return error_data_to_file(X,Y,n);
    }
    Py_RETURN_NONE;
}

static PyMethodDef capiMethods[] = {
        {"clusters_to_file_k_is_1",
                (PyCFunction) clusters_to_file_k_is_1,
                     METH_VARARGS,
                        PyDoc_STR("handling the case which K==1")},
        {"data_to_file",
                (PyCFunction) data_to_file,
                     METH_VARARGS,
                        PyDoc_STR("printing the data to the data.txt file")},
        {"QR",
                (PyCFunction) QR,
                     METH_VARARGS,
                        PyDoc_STR("calculate the QR decomposition")},
        {"weighted_adjacency_matrix",
                (PyCFunction) weighted_adjacency_matrix,
                     METH_VARARGS,
                        PyDoc_STR("calculate the weighted matrix for a given graph")},
        {"kmeans",
                (PyCFunction) kmeans,
                     METH_VARARGS,
                        PyDoc_STR("clusters n vectors into k clusters via the K-means algorithm")},

        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "Interface",
        NULL,
        -1,
        capiMethods
};

///initiate the nodule
PyMODINIT_FUNC
PyInit_Interface(void) {
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        PyErr_SetString(PyExc_TypeError, "module initialization failed");
        return NULL;
    }
    return m;
}
