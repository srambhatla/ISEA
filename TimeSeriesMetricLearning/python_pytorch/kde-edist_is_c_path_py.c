#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <../python_pytorch/kde.h>

//gcc -c -fPIC -I/usr/include kde-edist_is_c_path_py.c -o kde-edist_is_c_path_py.o
//gcc kde-edist_is_c_path_py.o -shared -o kde-edist_is_c_path_py.so -L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas -lm

//#include </usr/include/python3.5m/Python.h>
//#include </usr/include/python3.5m/numpy/arrayobject.h>

//#include <~/anaconda3/include/python3.7m/Python.h> // <path_to_library>
//#include <~/anaconda3/lib/python3.7/site-packages/numpy/core/include/numpy/arrayobject.h> // <path_to_library>

//#include </usr/include/python2.7/Python.h>
//#include </usr/include/python2.7/numpy/arrayobject.h>

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION



///usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h
//#include </usr/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h>
//#include <~/anaconda3/pkgs/numpy-base-1.19.2-py37hfa32c7d_0/lib/python3.7/site-packages/numpy/core/include/numpy/arrayobject.h> // <path_to_library>

/** Estimate bandwidth using Silverman's "rule of thumb" 
 * (Silverman 1986, pg 48 eq 3.31).  This is the default
 * bandwith estimator for the R 'density' function.  */




void init() {
    srand(time(NULL));    
}

// From http://cis-linux1.temple.edu/~giorgio/cis71/code/randompermute.c
int *rpermute(int n) {
    int *a = malloc(n*sizeof(int));
    int k, j, temp;
    for (k = 0; k < n; k++) {
    	a[k] = k;
    }
    for (k = n-1; k > 0; k--) {
        j = rand() % (k+1);
        temp = a[j];
        a[j] = a[k];
        a[k] = temp;
    }
    return a;
}

int compare_func(const void * a, const void * b){
  return ( *(int*)a - *(int*)b );
}

void sample_one_path(int T, int L, int *dpath, int t_shift) {
    // 1. get first T-1 elements from the permutation of (T+L-1).
    int i, val, loc;
    int *elements;
    elements = rpermute(T+L-1);
    // 2. sort the first T-1 elements.
    qsort(elements, T-1, sizeof(int), compare_func);
    // 3. from sorted T-1 elements, generate output of length L. put them start from dpath[t_shift].
    val = 0; loc = 0; elements[T-1] = T+L-1;
    for (i=0; i<L; ++i) {
        while (i+val == elements[loc]) {
            ++val; ++loc;
        }    // i+val < elements[loc]
        dpath[t_shift+i] = val;
    }
    free(elements);
}


double nrd0(double x[], const int N)
{
	gsl_sort(x, 1, N);
	
	double iqr =
		gsl_stats_quantile_from_sorted_data (x,1, N,0.75) - 
        gsl_stats_quantile_from_sorted_data (x,1, N,0.25);
    double hi = gsl_stats_sd(x, 1, N);
	double lo = GSL_MIN(hi, iqr/1.34);
	double bw = 0.9 * lo * pow(N,-0.2);
	return(bw);
}

/* kernels for kernel density estimates */
double gauss_kernel(double x)
{ 
	return exp(-(gsl_pow_2(x)/2))/(M_SQRT2*sqrt(M_PI)); 
}

double kerneldensity(double *samples, double *ws, double obs, size_t n)
{
	size_t i;
	double h = GSL_MAX(nrd0(samples, n), 1e-6);
	double prob = 0;
	for(i=0; i < n; i++)
	{
		prob += ws[i]*gauss_kernel( (samples[i] - obs)/h)/(n*h);
	}
	return prob;
}

//g++ edist_c_path_py.c -O3 -fPIC -shared -o edist_c_path.so
/*
void edist_c(float *X1, int t1, float *X2, int t2, int p, int T_low, int T_high, int n_sample, float *d, int *cntR, int *dpath) {
    // sample n_sample paths
    int i, j, k;
    int ii, jj;
    int *T_this_list=(int *)malloc(n_sample*sizeof(int));
    int T_shift, T_diff;
    float dist_sum, dist_this, discrep, dist_sum_path;
    int nT_high;
	// 1. get length of each sample, and total length.
    T_diff = T_high - T_low;
    nT_high = n_sample*T_high;
    if (T_diff == 0) {
        for (i=0; i<n_sample; ++i) {
            T_this_list[i] = T_low;
        }
    } else {
        for (i=0; i<n_sample; ++i) {
            T_this_list[i] = T_low + (rand() % T_diff);
        }
    }
    // 2. sample each path.
    T_shift = 0;
    for (i=0; i<n_sample; ++i) {
        sample_one_path(t1, T_this_list[i], dpath, T_shift);
        sample_one_path(t2, T_this_list[i], dpath, T_shift+nT_high);
        T_shift += T_this_list[i];
    }
    // 3. compute distance along paths.
    dist_sum = 0;
    T_shift = 0;
    for (i=0; i<n_sample; ++i) {
        dist_this = 0; // remove this (Sirisha 2020)
        //dist_sum_path = 0;
        for (j=0; j<T_this_list[i]; ++j) {
            // dist_this = 0;
            ii = dpath[T_shift + j];
            jj = dpath[nT_high + T_shift + j];
            for (k=0; k<p; ++k) {
                discrep = X1[ii*p + k] - X2[jj*p + k];
                dist_this += (discrep * discrep);
            }
           // Local distance (Sirisha, Dec. 2020)
           // dist_sum_path = sqrt(dist_this);
        }
        T_shift += T_this_list[i];
		dist_sum += dist_this / T_this_list[i];
        //dist_sum += dist_sum_path / T_this_list[i];
    }
    *d = dist_sum / n_sample;
	*cntR = T_shift;
	free(T_this_list);
}
*/

float max(float a, float b){
   if ((a) < (b)){   
      return b;  
   } else {   
      return a; 
} 
}

// void temp_func() {
//     // https://stackoverflow.com/questions/52074167/import-array-error-while-embedding-python-and-numpy-to-c
//     import_array();
    
// }


// Sirisha, Dec 2020
// edist-IS
void edist_IS_c(float *X1, int t1, float *X2, int t2, int p, int T_low, int T_high, int n_sample, float *d, int *cntR, int *dpath) {
    
    //FILE *fp;

    //fp = fopen("../python_pytorch/log/test.txt", "a");
    //fprintf(fp,"Lets give this a shot!!!!!!\n");

    // sample n_sample paths
    int i, j, k;
    int ii, jj;
    int iii;
    int is_iter = 5; //5
    int *T_this_list=(int *)malloc(n_sample*sizeof(int));
    float *path_score_list=malloc(n_sample*sizeof(float));
    float *dist_list=malloc(n_sample*sizeof(float));
    //float *q = malloc(n_sample*sizeof(float));
    
    int T_shift, T_diff, flag, y, T_curr;
    float dist_sum, dist_this, discrep, dist_sum_path, dist_sum_ea, eps, dot, denom_x1, denom_x2, u, dist_sum_avg;
    eps = 0.000001;
    int nT_high;
	// 1. get length of each sample, and total length.
    T_diff = T_high - T_low;
    nT_high = n_sample*T_high;
    
    double f_next[T_diff], g[T_diff], weights[T_diff], w_bar;
    double x_sam[n_sample], sum_f;
    
    for (i = 0; i < T_diff; ++i){
        g[i] = 1.0/((float) T_diff);
        f_next[i] = 1.0/((float) T_diff);
    }
    
    dist_sum_avg = 0.0;
    // IS: For loop for Non-parameteric Adaptive Importance Sampling
    for (iii = 0; iii < is_iter; ++iii ){
        
        if (T_diff == 0) {
            for (i=0; i<n_sample; ++i) {
                T_this_list[i] = T_low;
            }
        } 
        else {
            for (i=0; i<n_sample; ++i) {
            //T_this_list[i] = T_low + (rand() % T_diff);
            
                // Rejection Sampling
                flag = 0;
            
                while (flag == 0){
                    // Generate a sample
                    y =  T_low + (rand() % T_diff);
                    
                    // Generate a random number from Unif(0,1)
                    u = (double)rand()/((double) RAND_MAX); 
                    
                    // Decide to accept or reject, M = 30
                    if (u < f_next[y - T_low]/(30.0*g[y - T_low])){
                        flag = 1;
                    }
                }
                // Store y
                T_this_list[i] = y;
            }
        }
        
        // 2. sample each path.
        T_shift = 0;
        for (i=0; i<n_sample; ++i) {
            sample_one_path(t1, T_this_list[i], dpath, T_shift);
            sample_one_path(t2, T_this_list[i], dpath, T_shift+nT_high);
            T_shift += T_this_list[i];
        }
               
        
        
        // 3. compute distance along paths.
        //fprintf(fp, ",%d \n ", sizeof(dpath));
        dist_sum = 0;
        T_shift = 0;
        w_bar = 0;
        for (i=0; i<n_sample; ++i) {
        
            dist_sum_path = 0;
            dot = 0;
            denom_x1 = 0;
            denom_x2 = 0;
            for (j=0; j<T_this_list[i]; ++j) {
                dist_this = 0;
                ii = dpath[T_shift + j];
                jj = dpath[nT_high + T_shift + j];
            
                //   if (j==0){
                //   fprintf(fp, ",%d,%d \n ", T_shift, nT_high);
                //   }
            
                // compute distance across features 
                for (k=0; k<p; ++k) {
                    discrep = X1[ii*p + k] - X2[jj*p + k];
                    dist_this += (discrep * discrep);
                    
                    // cosine similarity in Frobenius norm sense
                
                    dot += X1[ii*p + k] * X2[jj*p + k] ;
                    denom_x1 += X1[ii*p + k] * X1[ii*p + k] ;
                    denom_x2 += X2[jj*p + k] * X2[jj*p + k] ;
                }
                
            // Local Distance
            dist_sum_path += sqrt(dist_this); // += on 01/26 sirisha
           
            }
        
            // scores for each alignment path, since p = path score / sum(path scores), and path score = dist_sum_path
            // we first evaluate the cosine similarity between matrices and add 1 to keep the scores between 0 and 1
            dist_list[i] = dist_sum_path/((float) T_this_list[i]);
        
            //dist_list[i] = dist_sum_path; // remove T_this_list from the denominator 03/29
            path_score_list[i] = 1.0 + dot/max(eps, sqrt(denom_x1*denom_x2));
        
            // Update the distribution to sample from
            weights[i] = max(weights[i], path_score_list[i]);
            w_bar = w_bar + weights[i];
        
            //q[i] = 1.0/T_this_list[i];
        
        
            //path_score_list[i] = dist_sum_path * dist_sum_path;
            //dist_sum += dist_sum_path;
            dist_sum += path_score_list[i]; 
        
            T_shift += T_this_list[i];
            //dist_sum += dist_this / T_this_list[i];
        }
        
        // since q = 1/n_samples, it gets cancelled with the 1/n_samples at the outer sum
    
        //fputs("The weights are, ", fp);
        dist_sum_ea = 0;
        for (i=0; i<n_sample; ++i) {
     // fprintf(fp, "dist is = %f, \n", dist_list[i]);
     //fprintf(fp, "%f, ", path_score_list[i]);
     //fprintf(fp, "q is = %f, \n", q[i]);
       
            dist_sum_ea += dist_list[i]*(path_score_list[i]/dist_sum); 
        }
        //dist_sum_avg = dist_sum_avg + dist_sum_ea/n_sample;
        *d = dist_sum_ea/n_sample;
        *cntR = T_shift;
       // fprintf(fp, "\n");
       // fprintf(fp, "dist is = %f, \n", dist_sum_ea/n_sample);
       // fputs("The dist is \n", fp);
  
        
       //fprintf(fp, "Entering KDE \n");    
    // KDE    
   

        for (i = 0; i<n_sample; ++i){
            x_sam[i] = (double) T_this_list[i];
            weights[i] = weights[i]/w_bar;
       // fprintf(fp, "%f \n", x_sam[i]);
        }
       
        T_curr = T_low;
        sum_f = 0;
       for (i = 0; i<T_diff; ++i){
           f_next[i] = kerneldensity(x_sam, weights, T_curr, n_sample);
           sum_f = sum_f + f_next[i];
        
           T_curr = T_curr + 1; 
        }
        
       for (i = 0; i<T_diff; ++i){
           f_next[i] = f_next[i]/sum_f;
           //fprintf(fp, "%f\t", f_next[i]);
       } 
        
      //fprintf(fp, "%f\n", dist_sum_ea/n_sample);   
    }
    //*d = dist_sum_avg/is_iter;
    
    //fflush(fp);    
    //fclose(fp);
	free(T_this_list);
    free(path_score_list);
    free(dist_list);
    
}


void path_c(int t1, int t2, int T_low, int T_high, int n_sample, int *cntR, int *dpath) {
    // sample n_sample paths
    int i;
    int *T_this_list=(int *)malloc(n_sample*sizeof(int));
    int T_shift, T_diff;
    int nT_high;
	// 1. get length of each sample, and total length.
    T_diff = T_high - T_low;
    nT_high = n_sample*T_high;
    if (T_diff == 0) {
        for (i=0; i<n_sample; ++i) {
            T_this_list[i] = T_low;
        }
    } else {
        for (i=0; i<n_sample; ++i) {
            T_this_list[i] = T_low + (rand() % T_diff);
        }
    }
    // 2. sample each path.
    T_shift = 0;
    for (i=0; i<n_sample; ++i) {
        sample_one_path(t1, T_this_list[i], dpath, T_shift);
        sample_one_path(t2, T_this_list[i], dpath, T_shift+nT_high);
        T_shift += T_this_list[i];
    }
	*cntR = T_shift;
	free(T_this_list);
}


void edist_cp(float *X1, int t1, float *X2, int t2, int p, int n_sample, int *t_list, int *dpath1, int *dpath2, float *d)
{
    float dist_this, dist_sum, discrep;
	int T_shift, i, j, ii, jj, k;
    dist_sum = 0;
    T_shift = 0;
    for (i=0; i<n_sample; ++i) {
        dist_this = 0;
        for (j=0; j<t_list[i]; ++j) {
            ii = dpath1[T_shift + j];
            jj = dpath2[T_shift + j];
            for (k=0; k<p; ++k) {
                discrep = X1[ii*p + k] - X2[jj*p + k];
                dist_this += (discrep * discrep);
            }
        }
        T_shift += t_list[i];
		dist_sum += dist_this / t_list[i];
    }
    *d = dist_sum / n_sample;
}


// int main(){
//     double x_final = gauss_kernel(0.0);
    
//     return 0;
// }



//https://ubuntuforums.org/archive/index.php/t-1266059.html
// int main() {
//     // test void sample_one_path(int T, int L, int *dpath, int t_shift)
//     srand(time(NULL));    
//     int dpath[100] = {0};
//     double w[1][15];
//     int x[1][20];
    
//     int size = sizeof w / sizeof w[0][0];


    
      
//     int i;
//     sample_one_path(5, 8, dpath, 2);
//     for (i=0; i<15; ++i){
        
//         printf("%d\t", dpath[i]);
//         w[0][i] = 0.1;
//     }
//        printf("\n");
    
//     for (i=0; i<20; ++i){
//         x[0][i] = rand() % 5 + 30;
//         printf("%d\t", x[0][i]);
//     }
        
//     printf("\n");
    
//     printf("Now I will print ws \n");
    
//     w[0][2] = 0.4;
//     for (i=0; i<size; ++i)
//         printf("%g\t", w[0][i]);
//     printf("\n");
        
//    // Set PYTHONPATH TO working directory
//    setenv("PYTHONPATH",".",1);



//    PyObject *pName, *pModule, *pDict, *pFunc, *presult, *args, *g_init, *f, *l_lim, *u_lim, *x_sam;


//    // Initialize the Python Interpreter
//    Py_Initialize();
//    temp_func();
//    // import_array1(-1); 
//     //PyObject *sys_path = PySys_GetObject("path");
//     //PyList_Append(sys_path, PyUnicode_FromString("../python_pytorch/"));
    
//         PyRun_SimpleString("import  sys");
//      PyRun_SimpleString("import  joblib");
//    //  PyRun_SimpleString("sys.path.append('~/anaconda3/lib/python3.7/site-packages')"); // <path_to_library>
//     //PyRun_SimpleString("sys.path.append('/usr/lib/python2.7/dist-packages')");
//     //PyRun_SimpleString("sys.path.remove('/usr/lib/python3/dist-packages')");
    
//     PyRun_SimpleString("print(sys.path)");
//     //PyRun_SimpleString("import statsmodels.api as sm");
   
   
//   //  PyRun_SimpleString("from sklearn.externals.joblib import parallel");
//   //  PyRun_SimpleString("import scipy"); 
//   //  PyRun_SimpleString("import sklearn as sk"); 
//   //  PyRun_SimpleString("print(sk.__version__)");
//     PyRun_SimpleString("import kde"); 
//    //import_array();
    



   
 
//    // Build the name object
//   pName = PyString_FromString("kde");
//   //pName = PyUnicode_FromString("kde");
    
//   //int chk = PyUnicode_Check(pName);
//   //  printf("%d\t", chk);

//    printf("I got this \n");  
    
//    // Load the module object
//    pModule = PyImport_Import(pName);

//     if (!pModule)
//     {
//         printf("pName\n");
//         return 0; 
//     }

//     pDict = PyModule_GetDict(pModule);
    
   
    
//     args = PyTuple_New(5);
    
//     pFunc = PyDict_GetItemString(pDict, (char*)"calc_next_dist");


    
//    if (PyCallable_Check(pFunc))
//    {
      
       
//        npy_intp dims = 15;
//        npy_intp x_dims = 20;


//        g_init = PyArray_SimpleNewFromData( 1, &dims,  PyArray_DOUBLE, w );
//        f = PyArray_SimpleNewFromData( 1, &dims,  PyArray_DOUBLE, w );
//        l_lim =  PyFloat_FromDouble(30.0);
//        u_lim =  PyFloat_FromDouble(35.0); 
//        x_sam = PyArray_SimpleNewFromData( 1, &x_dims,  PyArray_INT, x ); 
       
//        PyTuple_SetItem(args, 0,  g_init);
//        PyTuple_SetItem(args, 1,  f);
//        PyTuple_SetItem(args, 2,  l_lim);
//        PyTuple_SetItem(args, 3,  u_lim);
//        PyTuple_SetItem(args, 4,  x_sam);
         
//        PyErr_Print();
//        printf("Let's give this a shot!\n");
//        presult=PyObject_CallObject(pFunc,args);

//        PyErr_Print();
      
//    } else 
//    {
//        PyErr_Print();
//    }

    
//     double temp[0][size];
//      PyObject *ptemp, *objectsRepresentation ;
//         char* a11;
//      printf("Size if %d\t", size);
//     for (i = 0 ; i < size ; i++ )
//         {
        
//             ptemp = PyList_GetItem(presult,i);
//             objectsRepresentation = PyObject_Repr(ptemp);
//             a11 = PyBytes_AS_STRING(objectsRepresentation);
//             temp[0][i] = (double)strtod(a11,NULL);
//             printf("%f\t", temp[0][i]);
//         } 
//    Py_DECREF(g_init);
//   Py_DECREF(f);
//     Py_DECREF(u_lim);
//     Py_DECREF(l_lim);
//     Py_DECREF(x_sam);
//     Py_DECREF(args);
    

//    // Clean up
//    Py_DECREF(pModule);
//    Py_DECREF(pName);

//    // Finish the Python Interpreter
//    Py_Finalize();

//     return 0;
// }

