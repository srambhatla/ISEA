/**
 * Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>,
 * Signal Analysis and Machine Perception Laboratory,
 * Department of Electrical, Computer, and Systems Engineering,
 * Rensselaer Polytechnic Institute, Troy, NY 12180, USA
 */

/** 
 * This is the C/MEX code of dynamic time warping of two multivariate signals
 *
 * compile: 
 *     mex mdtw_c.c
 *
 * usage:
 *     d=mdtw_c(s,t)  or  d=mdtw_c(s,t,w)
 *     where s is signal 1, t is signal 2, w is window parameter 
 */

#include "mex.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double mdtw_c(double *X1, int t1, double *X2, int t2, int p, int w)
{
    double d = 0;
    int sizediff = t1-t2>0 ? t1-t2 : t2-t1;
    double **D;
    int i, j, k;
    int j1, j2;
    double cost, discrep, temp;

    if(w!=-1 && w<sizediff) w=sizediff; /* adapt window size */

    /*
    printf("t1=%3d t2=%3d p=%3d w=%3d\n", t1, t2, p, w);
	printf("X1: %d x %d\n", t1, p);
    printf("X2: %d x %d\n", t2, p);
    int tmax = t1>t2 ? t1 : t2;
    for(i=0;i<tmax;i++)
    {
        for(j=0; j<p; j++)
        {	
        	if(i < t1)
        	{
        		printf("X1[%d,%d]=%f", i+1, j+1, X1[i+j*t1]);
        	}
        	if(i < t2) {
        		printf("\tX2[%d,%d]=%f", i+1, j+1, X2[i+j*t2]);
        	}
        	printf("\n");
        }
        
    }
    printf("---\n\n");
    */

    /* create D */
    D=(double **)malloc((t1+1)*sizeof(double *));
    for(i=0;i<t1+1;i++)
    {
        D[i]=(double *)malloc((t2+1)*sizeof(double));
    }

    /* initialization */
    for(i=0;i<t1+1;i++)
    {
        for(j=0;j<t2+1;j++)
        {
            D[i][j]=-1;
        }
    }
    D[0][0]=0;

    /* dynamic programming */
    for(i=1;i<=t1;i++)
    {
        if(w==-1)
        {
            j1=1;
            j2=t2;
        }
        else
        {
            j1= i-w>1 ? i-w : 1;
            j2= i+w<t2 ? i+w : t2;
        }
        for(j=j1;j<=j2;j++)
        {
        	cost = 0;
        	for(k=0; k<p; k++)
        	{
        		discrep = X1[i-1 + k*t1] - X2[j-1 + k*t2];
        		cost += (discrep * discrep);
        	}
            
            temp=D[i-1][j];
            if(D[i][j-1]!=-1) 
            {
                if(temp==-1 || D[i][j-1]<temp) temp=D[i][j-1];
            }
            if(D[i-1][j-1]!=-1) 
            {
                if(temp==-1 || D[i-1][j-1]<temp) temp=D[i-1][j-1];
            }
            
            D[i][j]=cost+temp;
        }
    }
    
    
    d=D[t1][t2];
    
    /* view matrix D */
    /*
    for(i=0;i<t1+1;i++)
    {
        for(j=0;j<t2+1;j++)
        {
            printf("%f  ",D[i][j]);
        }
        printf("\n");
    }
    */
    
    /* free D */
    for(i=0;i<t1+1;i++)
    {
        free(D[i]);
    }
    free(D);
    
    return d;
}

/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    double *X1, *X2;
    int t1, p1, t2, p2;
    int w, p;
    double *dp;
    
    /*  check for proper number of arguments */
    if(nrhs!=2&&nrhs!=3)
    {
        mexErrMsgIdAndTxt( "MATLAB:mdtw_c:nrhs",
                "Two or three inputs required.");
    }
    if(nlhs>1)
    {
        mexErrMsgIdAndTxt( "MATLAB:dmtw_c:nlhs",
                "mdtw_c: One output required.");
    }
    
    /* check to make sure w is a scalar */
    if(nrhs==2)
    {
        w=-1;
    }
    else if(nrhs==3)
    {
        if( !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) ||
                mxGetN(prhs[2])*mxGetM(prhs[2])!=1 )
        {
            mexErrMsgIdAndTxt( "MATLAB:mdtw_c:w",
                    "mdtw_c: Input w must be a scalar.");
        }
        
        /*  get the scalar input w */
        w = (int) mxGetScalar(prhs[2]);
    }
    
    
    /*  create a pointer to the input matrix s */
    X1 = mxGetPr(prhs[0]);
    
    /*  create a pointer to the input matrix t */
    X2 = mxGetPr(prhs[1]);
    
    /*  get the dimensions of the matrix input s */
    t1 = mxGetM(prhs[0]);
    p1 = mxGetN(prhs[0]);
    
    /*  get the dimensions of the matrix input t */
    t2 = mxGetM(prhs[1]);
    p2 = mxGetN(prhs[1]);

    p = p1<p2 ? p1 : p2;
    if(p1 != p2)
    {
    	mexErrMsgIdAndTxt("MATLAB:mdtw_c:p1",
						"mdtw_c: size(X1,2) != size(X2,2)");
    }
    
    /* printf("p1=%3d p2=%3d p=%3d\n", p1, p2, p); */

    /*  set the output pointer to the output matrix */
    plhs[0] = mxCreateDoubleMatrix( 1, 1, mxREAL);
    
    /*  create a C pointer to a copy of the output matrix */
    dp = mxGetPr(plhs[0]);
    
    /*  call the C subroutine */
    dp[0]=mdtw_c(X1,t1,X2,t2,p,w);
    
    return;
    
}
