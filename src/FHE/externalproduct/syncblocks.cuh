#pragma once

namespace cufhedb {

// synchronizing all blocks in the kernel
// using this version for syncblocks should make sure blocks < 2*SMs
__device__ inline void __syncblocks(int goalVal, volatile int *Syncin, volatile int *Syncout) {
    int idx = threadIdx.x;
    int blk = blockIdx.x;
    int numblk = gridDim.x;

    // lock-free inter-block sync
    if (idx == 0) {
        Syncin[blk] = goalVal; 
        // ensure the write visible to all blocks
        __threadfence();  
    }

    // block 0 check whether all blocks have written their status
    if (blk == 0 && idx == 0) {
      volatile int complete = 0;
      // busy wait
      while (complete != numblk) {
        complete = 0;
        // check
        for (int i = 0; i < numblk; ++i) {
          if (Syncin[i] == goalVal) {
            ++complete;
          }
        }
      }
    }

    __syncthreads();

    if (blk == 0 && idx == 0) {
      for (int i = 0; i < numblk; ++i) {
        // notify all blocks synchronization is complete
        Syncout[i] = goalVal;
      }
      __threadfence();
    }

    __syncthreads(); 

    if (idx == 0) {
        while (Syncout[blk] != goalVal) {
          // ensure consistency within the block
          //__threadfence_block();  
        }
        // reset Syncin and Syncout to 0 
        Syncin[blk] = 0;
        Syncout[blk] = 0;
        __threadfence();  
    }

    __syncthreads();
}
};