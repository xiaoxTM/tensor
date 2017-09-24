#ifndef __U_TENSOR_MEMORY_MANAGER_HPP__
#define __U_TENSOR_MEMORY_MANAGER_HPP__

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
#include "mm/u-mm-gpu.cu"
#else
#include "mm/u-mm-cpu.hpp"
#endif

namespace u {
    namespace tensor {
        namespace mm {
            #ifdef USE_CUDA
            // cuda implementation if variable

            inline unsigned char *malloc(size_t size, unsigned int bytes, int init=0) {
                return gpu::malloc(size, bytes, init);
            }

            inline void mfree(unsigned char *mem) {
                gpu::mfree(mem);
            }

            inline void no_free(unsigned char *mem) {}

            template <typename T>
            inline T *download(const unsigned char * const data, size_t size) {
                return reinterpret_cast<T*>(gpu::download(data, size));
            }

            inline unsigned char *download(const unsigned char * const data, size_t size) {
                return gpu::download(data, size);
            }

            inline void upload(unsigned char * const dst, const unsigned char * const src, size_t size) {
                gpu::upload(dst, src, size);
            }

            #else
            // cpu implementation otherwise
            inline unsigned char *malloc(size_t size, unsigned int bytes, int init=0) {
                return cpu::malloc(size, bytes, init);
            }

            inline void mfree(unsigned char *mem) {
                cpu::mfree(mem);
            }

            inline void no_free(unsigned char *mem) {}

            inline unsigned char *copy(const unsigned char * const data, size_t size/*size in byte*/) {
                return cpu::download(data, size);
            }

            template<typename T>
            inline T *download(const unsigned char * const data, size_t size/*size in byte*/) {
                return reinterpret_cast<T*>(cpu::download(data, size));
            }

            inline unsigned char *download(const unsigned char * const data, size_t size/*size in byte*/) {
                return cpu::download(data, size);
            }

            inline void upload(unsigned char * const dst, const unsigned char * const src, size_t size/*size in byte*/) {
                cpu::upload(dst, src, size);
            }
            #endif
        }
    }
}

#endif
