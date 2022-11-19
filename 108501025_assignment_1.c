
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define INITIAL_ZERO 0
#define INPUT 2
#define HIDDENNODE 2
#define OUTPUT 1
#define TRAINNUM 4
#define LEARNING_RATE 0.1f
#define EPOCHS 10000
#define HALF 0.5f
#define ZERO_POS_ARRAY 0
#define FIRST_POS_ARRAY 1
#define SECOND_POS_ARRAY 2
#define THIRD_POS_ARRAY 3
#define FOURTH_POS_ARRAY 4
#define TRUE 1
#define STOP_CHAR 5
#define NUM_1 1

double sigmoid(double x)
{
    return NUM_1 / (NUM_1 + exp(-x));
} // sigmoid function
double dsigmoid(double x)
{
    return x * (NUM_1 - x);
} // differential sigmoid function

double initial_weight()
{

    return (double)rand() / (double)(RAND_MAX); 
}
void shuffle(int *array, size_t n)
{
    if (n > NUM_1)
    {
        size_t i;
        for (i = INITIAL_ZERO; i < n - NUM_1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + NUM_1);
            int t = *(array + j);
            *(array + j) = *(array + i);
            *(array + i) = t;
        }
    }
}

void doubleptr(double **c, int x, int y)
{
    for (int i = INITIAL_ZERO; i < x; i++)
    {
        *(c + i) = (double *)malloc(y * sizeof(double *));
    }
}
void answer(int ain, int bin, double hiddenweight1, double hiddenweight2, double hiddenweight3, double hiddenweight4, double hiddenbias1,
            double hiddenbias2, double outweight1, double outweight2, double outbias)
{
    double temp1;
    double temp2;
    double result;
    temp1 = sigmoid(ain * hiddenweight1 + bin * hiddenweight3 + hiddenbias1);
    temp2 = sigmoid(ain * hiddenweight2 + bin * hiddenweight4 + hiddenbias2);
    result = sigmoid(temp1 * outweight1 + temp2 * outweight2 + outbias);
    printf("The result is %.0f!\n", result);
}
int main()
{
    int ain, bin;
    FILE *fp=NULL;
    fp = fopen("loss.txt", "w+");
    double loss;
    double *hiddenlayer = (double *)malloc(HIDDENNODE * sizeof(double));
    double *outputlayer = (double *)malloc(OUTPUT * sizeof(double));
    double *hiddenbias = (double *)malloc(HIDDENNODE * sizeof(double));
    double *outputbias = (double *)malloc(OUTPUT * sizeof(double));


    double **hiddenweight = (double **)malloc(INPUT * sizeof(double *));
    doubleptr(hiddenweight, INPUT, HIDDENNODE);
    double **outputweight = (double **)malloc(HIDDENNODE * sizeof(double *)); //HIDDENNODE
    doubleptr(outputweight, HIDDENNODE, OUTPUT);

    

    double **traininput = (double **)malloc(TRAINNUM * sizeof(double *));
    doubleptr(traininput, TRAINNUM , INPUT);
    double **trainoutput = (double **)malloc(TRAINNUM * sizeof(double *));
    doubleptr(trainoutput, TRAINNUM, OUTPUT);
    // initialize

    *(*(traininput + ZERO_POS_ARRAY) + ZERO_POS_ARRAY) = 0.0f;
    *(*(traininput + ZERO_POS_ARRAY) + FIRST_POS_ARRAY) = 0.0f;
    *(*(traininput + FIRST_POS_ARRAY) + ZERO_POS_ARRAY) = 0.0f;
    *(*(traininput + FIRST_POS_ARRAY) + FIRST_POS_ARRAY) = 1.0f;
    *(*(traininput + SECOND_POS_ARRAY) + ZERO_POS_ARRAY) = 1.0f;
    *(*(traininput + SECOND_POS_ARRAY) + FIRST_POS_ARRAY) = 0.0f;
    *(*(traininput + THIRD_POS_ARRAY) + ZERO_POS_ARRAY) = 1.0f;
    *(*(traininput + THIRD_POS_ARRAY) + FIRST_POS_ARRAY) = 1.0f;

    *(*(trainoutput + ZERO_POS_ARRAY) + ZERO_POS_ARRAY) = 0.0f;
    *(*(trainoutput + FIRST_POS_ARRAY) + ZERO_POS_ARRAY) = 1.0f;
    *(*(trainoutput + SECOND_POS_ARRAY) + ZERO_POS_ARRAY) = 1.0f;
    *(*(trainoutput + THIRD_POS_ARRAY) + ZERO_POS_ARRAY) = 0.0f;

    for (int i = INITIAL_ZERO; i < INPUT; i++)
    {
        for (int j = INITIAL_ZERO; j < HIDDENNODE; j++)
        {
            *(*(hiddenweight + i) + j) = initial_weight();
        }
    }
    for (int i = INITIAL_ZERO; i < HIDDENNODE; i++)
    {
        *(hiddenbias + i) = initial_weight();
        for (int j = INITIAL_ZERO; j < OUTPUT; j++)
        {
            *(*(outputweight + i) + j) = initial_weight();
        }
    }
    for (int i = INITIAL_ZERO; i < OUTPUT; i++)
    {
        *(outputbias + i) = initial_weight();
    }
    int *trainorder = (int *)malloc(TRAINNUM * sizeof(int));
    for (int i = INITIAL_ZERO; i < TRAINNUM; i++)
    {
        *(trainorder + i) = i;
    }
    
    for (int n = INITIAL_ZERO; n < EPOCHS; n++)
    {                                  
        shuffle(trainorder, TRAINNUM); // choosepair randomly

        for (int j = INITIAL_ZERO; j < TRAINNUM; j++)
        {
            int result = *(trainorder + j);

            for (int k = INITIAL_ZERO; k < HIDDENNODE; k++)
            {
                double actv = *(hiddenbias + k); // give bias to activation funciton
                for (int m = INITIAL_ZERO; m < INPUT; m++)
                {
                    actv += *(*(traininput + result) + m) * *(*(hiddenweight + m) + k);
                }
                *(hiddenlayer + k) = sigmoid(actv);
            } // end for(ends of forward activation at hidden layer)

            for (int k = INITIAL_ZERO; k < OUTPUT; k++)
            {
                double actv = *(outputbias + k);
                for (int m = INITIAL_ZERO; m < HIDDENNODE; m++)
                {
                    actv += *(hiddenlayer + m) * *(*(outputweight + m) + k);
                }

                *(outputlayer + k) = sigmoid(actv);
            } // end for(ends of forward activation at output layer)

            printf("Input:");
            printf("%.0f ", *(*(traininput + result) + ZERO_POS_ARRAY));
            printf("%.0f ", *(*(traininput + result) + FIRST_POS_ARRAY));
            printf("  Output:");
            printf("%.6f ", *(outputlayer + ZERO_POS_ARRAY));
            printf("  Expected Output:");
            printf("%.0f ", *(*(trainoutput + result) + ZERO_POS_ARRAY));
            printf("  Epochs:");
            printf("%d ",n);
            printf("  Loss:");
            printf("%.6f\n",loss);

            // backward
            double *deltaoutput = (double *)malloc(OUTPUT * sizeof(double));
            
            for (int k = INITIAL_ZERO; k < OUTPUT; k++)
            {
                double derror = (*(*(trainoutput + result) + k) - *(outputlayer + k)); // MSE:error=1/2(theoritical-practical)^2
                *(deltaoutput + k) = dsigmoid(*(outputlayer + k)) * derror;            // dE/dx=(dy/dx)(dE/dy)=(d/dx)f(x)(dE/dy)
                loss=derror*derror* HALF;
                fprintf(fp, "%.6f\n  ",loss);
            }
            //fprintf(fp, "%.6f\n  ",loss);

            double *deltahidden = (double *)malloc(HIDDENNODE * sizeof(double));
            for (int k = INITIAL_ZERO; k < HIDDENNODE; k++)
            {
                double errorhidden = INITIAL_ZERO;
                for (int m = INITIAL_ZERO; m < OUTPUT; m++)
                {
                    errorhidden += *(deltaoutput + m) * *(*(outputweight + k) + m);
                }
                *(deltahidden + k) = errorhidden * dsigmoid(*(hiddenlayer + k));
            }
            
            for (int k = INITIAL_ZERO; k < OUTPUT; k++)
            {
                *(outputbias + k) += *(deltaoutput + k) * LEARNING_RATE; 
                for (int m = INITIAL_ZERO; m <HIDDENNODE; m++)
                {
                    *(*(outputweight + m) + k) += *(hiddenlayer + m) * *(deltaoutput + k) * LEARNING_RATE;
                }
            }

            for (int k = INITIAL_ZERO; k < HIDDENNODE; k++)
            {
                *(hiddenbias + k) += *(deltahidden + k) * LEARNING_RATE;
                for (int m = INITIAL_ZERO; m < INPUT; m++)
                {
                    *(*(hiddenweight + m) + k) += *(*(traininput + result) + m) * *(deltahidden + k) * LEARNING_RATE;
                }
            }
        }
    }
    fclose(fp);
    // printfinal
    printf("Final Hidden Weights\n[ ");
    for (int k = INITIAL_ZERO; k < HIDDENNODE; k++)
    {
        printf("[ ");
        for (int m = INITIAL_ZERO; m < INPUT; m++)
        {
            printf("%.6f ", *(*(hiddenweight + m) + k));
        }
        printf("] ");
    }
    printf("]\n");

    printf("Final Hidden Biases\n[ ");
    for (int k = INITIAL_ZERO; k < HIDDENNODE; k++)
    {
        printf("%.6f ", *(hiddenbias + k));
    }
    printf("]\n");
    printf("Final Output Weights");
    for (int k = INITIAL_ZERO; k < OUTPUT; k++)
    {
        printf("[ ");
        for (int m = INITIAL_ZERO; m < HIDDENNODE; m++)
        {
            printf("%.6f ", *(*(outputweight + m) + k));
        }
        printf("]\n");
    }
    printf("Final Output Biases\n[ ");
    for (int k = INITIAL_ZERO; k < OUTPUT; k++)
    {
        printf("%.6f ", *(outputbias + k));
    }
    printf("]\n");
    printf("Please type 2 inputs (If you want to exit, just press 5) :\n");
    
    while (TRUE)
    {
        scanf("%d", &ain);
        if (ain == STOP_CHAR)
        {
            break;
        }
        scanf("%d", &bin);
        answer(ain, bin, *(*(hiddenweight + ZERO_POS_ARRAY) + ZERO_POS_ARRAY), *(*(hiddenweight + ZERO_POS_ARRAY) + FIRST_POS_ARRAY), *(*(hiddenweight + FIRST_POS_ARRAY) + ZERO_POS_ARRAY), *(*(hiddenweight + FIRST_POS_ARRAY) + FIRST_POS_ARRAY),
               *(hiddenbias + ZERO_POS_ARRAY), *(hiddenbias + FIRST_POS_ARRAY), *(*(outputweight + ZERO_POS_ARRAY) + ZERO_POS_ARRAY), *(*(outputweight + FIRST_POS_ARRAY) + ZERO_POS_ARRAY), *(outputbias + ZERO_POS_ARRAY));
    }

    return 0;
}
