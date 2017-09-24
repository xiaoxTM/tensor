#ifndef __U_TENSOR_MEMORY_MANAGER_GPU_HPP__
#define __U_TENSOR_MEMORY_MANAGER_GPU_HPP__

/***
u-tensor.hpp base functions for libu
Copyright (C) 2013  Renweu Gao

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
***/

#ifdef USE_CUDA

#include <cuda.h>

namespace u {
    namespace gpu {
        unsigned char *malloc(size_t size, unsigned int bytes, int init) {
            unsigned char *mem = nullptr;
            size_t total = bytes * size;
            if (cudaMallocManaged((void **)&mem, total) == cudaErrorMemoryAllocation) throw (GPU_ERROR_ERROR);
            cudaMemset(mem, init, total);
            return mem;
        }

        void mfree(unsigned char ** mem) {
            cudaError_t ret = cudaFree((void *)*mem);
            if (ret == cudaSuccess) {
                *mem = nullptr;
            } else {
                throw (ret);
            }
        }

        std::string get_name(int device) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device);
            return props.name;
        }

        int count_device() {
            int ndevice = -1;
            cudaGetDeviceCount(&ndevice);
            return ndevice;
        }

        int get_device() {
            int device = 0;
            cudaGetDevice(&device);
            return device;
        }

        int set_device(int device) {
            cudaGetDevice(device);
        }
    }
}

#endif

#endif
