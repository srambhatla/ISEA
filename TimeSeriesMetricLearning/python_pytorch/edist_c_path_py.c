#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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


//g++ edist_c_path_py.c -O3 -fPIC -shared -o edist_c_path.so
/* original
void edist_c(float *X1, int t1, float *X2, int t2, int p, int T_low, int T_high, int n_sample, float *d, int *cntR, int *dpath) {
    // sample n_sample paths
    int i, j, k;
    int ii, jj;
    int *T_this_list=(int *)malloc(n_sample*sizeof(int));
    int T_shift, T_diff;
    float dist_sum, dist_this, discrep;
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
        dist_this = 0;
        for (j=0; j<T_this_list[i]; ++j) {
            ii = dpath[T_shift + j];
            jj = dpath[nT_high + T_shift + j];
            for (k=0; k<p; ++k) {
                discrep = X1[ii*p + k] - X2[jj*p + k];
                dist_this += (discrep * discrep);
            }
        }
        T_shift += T_this_list[i];
		dist_sum += dist_this / T_this_list[i];
    }
    *d = dist_sum / n_sample;
	*cntR = T_shift;
	free(T_this_list);
}
*/


void edist_c(float *X1, int t1, float *X2, int t2, int p, int T_low, int T_high, int n_sample, float *d, int *cntR, int *dpath) {
   // FILE *fp;
   // fp = fopen("../python_pytorch/log/test.txt", "a");
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
        //dist_this = 0; // remove this (Sirisha 2020)
        dist_sum_path = 0;
        for (j=0; j<T_this_list[i]; ++j) {
            dist_this = 0;
            ii = dpath[T_shift + j];
            jj = dpath[nT_high + T_shift + j];
            for (k=0; k<p; ++k) {
                discrep = X1[ii*p + k] - X2[jj*p + k];
                dist_this += (discrep * discrep);
            }
           // Local distance (Sirisha, Dec. 2020)
            dist_sum_path += sqrt(dist_this); // 01/26 +=
        }
        T_shift += T_this_list[i];
		//dist_sum += dist_this / T_this_list[i];
        dist_sum += dist_sum_path / ((float) T_this_list[i]);
        //dist_sum += dist_sum_path; // remove T_this_list from the denominator 03/29
    }
    *d = dist_sum / n_sample;
	*cntR = T_shift;
	free(T_this_list);
    
   //fprintf(fp, "dist is = %f, \n", dist_sum / n_sample);
   //fputs("The dist is \n", fp);
   //fclose(fp);
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


//int main() {
//    // test void sample_one_path(int T, int L, int *dpath, int t_shift)
//    srand(time(NULL));    
//    int dpath[100] = {0};
//    int i;
//    sample_one_path(5, 8, dpath, 2);
//    for (i=0; i<15; ++i)
//        printf("%d\t", dpath[i]);
//    printf("\n");
//}

