#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "CycleTimer.h"

using namespace std;

typedef struct {
  // Control work assignments
  int start, end;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  double *minDist;
  int *counts;
  double *accum; 
  int *clusterAssignments;
  double *currCost;
  int M, N, K, Mstart, Mend;
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args) {
  
  // Initialize arrays
  for (int m =args->Mstart; m < args->Mend; m++) {
    args->minDist[m] = 1e30;
    args->clusterAssignments[m] = -1;
  }

  // Assign datapoints to closest centroids
  // Should be computed in parallel with m workers
  // for (int k = args->start; k < args->end; k++) {
  //   for (int m = 0; m < args->M; m++) {
  //     double d = dist(&args->data[m * args->N],
  //                     &args->clusterCentroids[k * args->N], args->N);
  //     if (d < minDist[m]) {
  //       minDist[m] = d;
  //       args->clusterAssignments[m] = k;
  //     }
  //   }
  // }
  // Switch order of loop for parallel
  for (int m = args->Mstart; m < args->Mend; m++) {
    for (int k = 0; k < args->K; k++) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      if (d < args->minDist[m]) {
        args->minDist[m] = d;
        args->clusterAssignments[m] = k;
      }
    }
  }
}

/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args) {

  // Zero things out
  for (int k = args->start; k < args->end; k++) {
    args->counts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }


  // Sum up contributions from assigned examples
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    if (k>=args->start && k < args->end)
    {
      for (int n = 0; n < args->N; n++) {
        args->clusterCentroids[k * args->N + n] +=
            args->data[m * args->N + n];
      }
      args->counts[k]++;
    }
    
  }

  // Compute means
  for (int k = args->start; k < args->end; k++) {
    args->counts[k] = max(args->counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] /= args->counts[k];
    }
  }
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {
  // Zero things out
  for (int k = args->start; k < args->end; k++) {
    args->accum[k] = 0.0;
  }

  // Sum cost for all data points assigned to centroid
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    if(k>= args->start && k<args->end)
    {
      args->accum[k] += dist(&args->data[m * args->N],
                    &args->clusterCentroids[k * args->N], args->N);
    }
    
    
  }

  // Update costs
  for (int k = args->start; k < args->end; k++) {
    args->currCost[k] = args->accum[k];
  }
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  // WorkerArgs args;
  // args.data = data;
  // args.clusterCentroids = clusterCentroids;
  // args.clusterAssignments = clusterAssignments;
  // args.currCost = currCost;
  // args.M = M;
  // args.N = N;
  // args.K = K;
  // args.Mstart = 0;
  // args.Mend = M;
  WorkerArgs args[3*K];
  std::thread workers[3*K];
  for (int i=0;i<3*K;i++)
  {
    args[i].data = data;
    args[i].clusterCentroids = clusterCentroids;
    args[i].clusterAssignments = clusterAssignments;
    args[i].currCost = currCost;
    args[i].M = M;
    args[i].N = N;
    args[i].K = K;
    args[i].minDist = new double[M];
    args[i].counts = new int[K];
    args[i].accum = new double[K];
    args[i].start = i/3;
    args[i].end = i/3+1;
    args[i].Mstart = (M / (3*K)) * i;
    if(i != (3*K - 1))
    {
      args[i].Mend = (M / (3*K)) * (i+1);
    }
    else
    {
      args[i].Mend = M;
    }
  }
  

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    // Update cost arrays (for checking convergence criteria)
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }

    // Setup args struct
    //float ts = CycleTimer::currentSeconds();
    for (int i=1;i<3*K;i++)
    {
      workers[i] = std::thread(computeAssignments, &args[i]);
    }
    computeAssignments(&args[0]);

    for (int i = 1; i<3*K; i++)
    {
      workers[i].join();
    }
    //printf("Ass time: %.5f\n", CycleTimer::currentSeconds()-ts);
    //ts = CycleTimer::currentSeconds();
    for (int i=1;i<K;i++)
    {
      workers[3*i] = std::thread(computeCentroids, &args[3*i]);
    }
    computeCentroids(&args[0]);

    for (int i = 1; i<K; i++)
    {
      workers[3*i].join();
    }
    //printf("Cent time: %.5f\n", CycleTimer::currentSeconds()-ts);
    //ts = CycleTimer::currentSeconds();
    for (int i=1;i<K;i++)
    {
      workers[3*i] = std::thread(computeCost, &args[3*i]);
    }
    computeCost(&args[0]);

    for (int i = 1; i<K; i++)
    {
      workers[3*i].join();
    }
    //printf("Cost time: %.5f\n", CycleTimer::currentSeconds()-ts);

    iter++;
  }
  for(int i=0;i<K;i++)
  {
    free(args[i].accum);
    free(args[i].minDist);
    free(args[i].counts);
  }
  free(currCost);
  free(prevCost);
}
