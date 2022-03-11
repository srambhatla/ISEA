#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//g++ mdtw_c_path_py.c -O3 -fPIC -shared -o mdtw_c_path.so
void mdtw_c(float *X1, int t1, float *X2, int t2, int p, int w, float *d, int *cntR, int *dpath)
{
    int sizediff = t1-t2>0 ? t1-t2 : t2-t1;
    float **D;
	int **P;
    int i, j, k, cnt;
    int j1, j2;
    float cost, discrep, temp;
	int tempp;
	int *fullP=(int *)malloc(((t1+t2)*2)*sizeof(int));

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
    D=(float **)malloc((t1+1)*sizeof(float *));
	P=(int **)malloc((t1+1)*sizeof(int *));
    for(i=0;i<t1+1;i++)
    {
        D[i]=(float *)malloc((t2+1)*sizeof(float));
		P[i]=(int *)malloc((t2+1)*sizeof(int));
    }

    /* initialization */
    for(i=0;i<t1+1;i++)
    {
        for(j=0;j<t2+1;j++)
        {
            D[i][j]=-1;
			P[i][j]=-2;
        }
    }
    D[0][0]=0;
	P[0][0]=0;

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
        		discrep = X1[(i-1)*p + k] - X2[(j-1)*p + k];
        		cost += (discrep * discrep);
        	}
            
            temp=D[i-1][j];
			tempp=-1;
            if(D[i][j-1]!=-1) 
            {
                if(temp==-1 || D[i][j-1]<temp) 
				{
					temp=D[i][j-1];
					tempp=1;
				}
            }
            if(D[i-1][j-1]!=-1) 
            {
                if(temp==-1 || D[i-1][j-1]<temp) 
				{
					temp=D[i-1][j-1];
					tempp=0;
				}
            }
            D[i][j]=cost+temp;
			P[i][j]=tempp;
        }
    }
    
    *d=D[t1][t2];
    i = t1;
	j=t2;
	cnt = 0;
	while (i>0 && j>0)
	{
		//printf("%d: (%d,%d)-%d\t",cnt, i, j, P[i][j]);
		fullP[cnt]=i-1; fullP[cnt+(t1+t2)]=j-1;
		//printf("%(%d,%d)\n",fullP[cnt], fullP[cnt+(t1+t2)]);
		cnt++;
		if (P[i][j]==0)
		{
			--i; --j;
		}
		else if (P[i][j]==-1)
		{
			--i;
		} 
		else /*if (P[i][j] == 1)*/
		{
			--j;
		}
	}

	/* view matrix D */
    /*
    for(i=0;i<t1+1;i++)
    {
        for(j=0;j<t2+1;j++)
        {
            printf("(%d,%d): %f  %f \t\t",i, j, D[i][j], P[i][j]);
        }
        printf("\n");
    }
    */
    /* free D */
    for(i=0;i<t1+1;i++)
    {
        free(D[i]);
		free(P[i]);
    }
    free(D);
	free(P);
    *cntR=cnt;
    
    for(i=0; i<cnt; ++i)
	{
		dpath[i]=fullP[cnt-i-1];
		dpath[t1+t2+i]=fullP[t1+t2+cnt-i-1];
	}
	free(fullP);
    
}

void mdtw_c_exp(float *X1, int t1, float *X2, int t2, int p, float *d)
{
    float **D;
	float **M;
    int i, j, k;
    float cost, discrep, total;

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

    /* create D and M*/
    D=(float **)malloc((t1)*sizeof(float *));
    for(i=0;i<t1;i++)
    {
        D[i]=(float *)malloc((t2)*sizeof(float));
    }
	for(i=0; i<t1; ++i)
		D[i][0] = 1;
	for(j=0; j<t2; ++j)
		D[0][j] = 1;
	for(i=1; i<t1; ++i)
		for(j=1; j<t2; ++j)
			D[i][j]=D[i-1][j] + D[i][j-1] + D[i-1][j-1];
	
	float sumM = 0;
    M=(float **)malloc((t1)*sizeof(float *));
    for(i=0;i<t1;i++)
    {
        M[i]=(float *)malloc((t2)*sizeof(float));
    }	
	
	for(i=0;i<t1; ++i)
		for(j=0; j<t2; ++j)
		{
			M[i][j] = D[i][j] * D[t1-i-1][t2-j-1];
			sumM += M[i][j];
		}
	
	for(i=0;i<t1; ++i)
		for(j=0; j<t2; ++j)	
			M[i][j] /= sumM;
			
    /* initialization */

    /* dynamic programming */
	total = 0;
    for(i=0;i<t1;i++)
    {
        for(j=0;j<t2;j++)
        {
        	cost = 0;
        	for(k=0; k<p; k++)
        	{
        		discrep = X1[i*p + k] - X2[j*p + k];
        		cost += (discrep * discrep);
        	}
            total += cost*M[i][j];
        }
    }
    
    *d=total;

	/* view matrix D */
    /*
    for(i=0;i<t1+1;i++)
    {
        for(j=0;j<t2+1;j++)
        {
            printf("(%d,%d): %f  %f \t\t",i, j, D[i][j], P[i][j]);
        }
        printf("\n");
    }
    */
    /* free D */
    for(i=0;i<t1;i++)
    {
        free(D[i]);
		free(M[i]);
    }
    free(D);
	free(M);
}


