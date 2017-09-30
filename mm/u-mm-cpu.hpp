#ifndef __U_TENSOR_MEMORY_MANAGER_CPU_HPP__
#define __U_TENSOR_MEMORY_MANAGER_CPU_HPP__

/***
u-mm-cpu.hpp base functions for tensor
Copyright (C) 2017  Renweu Gao

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

#include <algorithm>
#include <libu/u-log>
#include <exception>
#include <cstdlib>

namespace u {
    namespace tensor {
        namespace cpu {
            unsigned char *malloc(size_t size, unsigned int bytes, int init = 0) {
                u_fun_enter(2, 0);
                unsigned char *mem = nullptr;
                try {
                    mem = reinterpret_cast<unsigned char*>(calloc(size, bytes));
                    assert(mem != nullptr);
                    size_t total = size * bytes;
                    std::fill_n(mem, init, total);
                } catch (std::exception &e) {
                    bool MALLOC_MEMORY_ERROR = false;
                    assert(MALLOC_MEMORY_ERROR);
                }
                u_fun_exit(0, -2);
                return mem;
            }

            void mfree(unsigned char * mem) {
                u_fun_enter(2, 0);
                if (mem != nullptr) {
                    try {
                        free(mem);
                        mem = nullptr;
                    } catch (std::exception &e) {
                        bool MFREE_MEMORY_ERROR = false;
                        assert(MFREE_MEMORY_ERROR);
                    }
                }
                u_fun_exit(0, -2);
            }

            unsigned char *download(const unsigned char * const data, size_t size) {
                u_fun_enter(2, 0);
                unsigned char *mem = reinterpret_cast<unsigned char*>(malloc(size, sizeof(unsigned char)));
                std::copy_n(data, size, mem);
                u_fun_exit(0, -2);
                return mem;
            }

            void upload(unsigned char * const dst, const unsigned char * const src, size_t size) {
                // @dst: destinate where data is uploaded
                // @size: size in byte
                u_fun_enter(2, 0);
                std::copy_n(src, size, dst);
                u_fun_exit(0, -2);
            }
        }
    }
}

#endif
