#include <omp.h>
#include <stdio.h>

int main() {
    int total_loops = 100;
    int num_threads = 10;
    int iterations_per_thread = total_loops / num_threads; // 10 loops per thread

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int sum = 0;
        
        
        for (int round = 0; round < num_threads; ++round) {
            #pragma omp for
            for (int i = 0; i < iterations_per_thread; ++i) {
                sum+=i;
            }

            // Synchronize all threads before proceeding to the next round
            
        }
        //#pragma omp barrier
        printf("Id: %d, ThreSum: %d\n", thread_id,sum);

    }

    return 0;
}
