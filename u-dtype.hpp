#ifndef __U_TENSOR_DATA_TYPE_HPP__
#define __U_TENSOR_DATA_TYPE_HPP__

/***
u-dtype.hpp base functions for tensor
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

#include <libu/u-log>
#include <typeinfo>

namespace u {
    // basic numerical data type
    namespace tensor {
        enum class DType : size_t{
            int8,
            uint8,
            int16,
            uint16,
            int32,
            uint32,
            int64,
            uint64,
            float32,
            float64,
            invalid
        };

        // convert c/c++ basic data type to tensor basic data type
        template <typename T>
        DType ctype() {
            DType ret = DType::invalid;
            const size_t code = typeid(T).hash_code();
            if (code == typeid(char).hash_code())
            ret = DType::int8;
            else if (code == typeid(unsigned char).hash_code())
            ret = DType::uint8;
            else if (code == typeid(short).hash_code())
            ret = DType::int16;
            else if (code == typeid(unsigned short).hash_code())
            ret = DType::uint16;
            else if (code == typeid(int).hash_code())
            ret = DType::int32;
            else if (code == typeid(unsigned int).hash_code())
            ret = DType::uint32;
            else if (code == typeid(long).hash_code())
            ret = DType::int64;
            else if (code == typeid(unsigned long).hash_code()) // also size_t
            ret = DType::uint64;
            else if (code == typeid(float).hash_code())
            ret = DType::float32;
            else if (code == typeid(double).hash_code())
            ret = DType::float64;
            else {
                bool NON_SUPPORT_DTYPE = false;
                u_assert(NON_SUPPORT_DTYPE,"not supported data type [supported: {u}int{8, 16, 32, 64}, float{32, 64}]");
            }

            return (ret);
        }

        // get the size of `type' in terms of byte
        unsigned int dtype_size(DType type) {
            unsigned int size = 0;
            switch (type) {
                case DType::int8:   size = sizeof(char);break;
                case DType::uint8:  size = sizeof(unsigned char); break;
                case DType::int16:  size = sizeof(short); break;
                case DType::uint16: size = sizeof(unsigned short); break;
                case DType::int32:  size = sizeof(int); break;
                case DType::uint32: size = sizeof(unsigned int); break;
                case DType::int64:  size = sizeof(long); break;
                case DType::uint64: size = sizeof(unsigned long); break;
                case DType::float32:size = sizeof(float); break;
                case DType::float64:size = sizeof(double); break;
                default: size = 0;
            }
            return (size);
        }

        std::string dtype_str(DType type) {
            std::string name;
            switch (type) {
                case DType::int8:   name = "char";break;
                case DType::uint8:  name = "unsigned char"; break;
                case DType::int16:  name = "short"; break;
                case DType::uint16: name = "unsigned short"; break;
                case DType::int32:  name = "int"; break;
                case DType::uint32: name = "unsigned int"; break;
                case DType::int64:  name = "long"; break;
                case DType::uint64: name = "unsigned long"; break;
                case DType::float32:name = "float"; break;
                case DType::float64:name = "double"; break;
                default: name = "NON_SUPPORT_DTYPE";
            }
            return (name);
        }

        const char * dtype_cstr(DType type) {
            return dtype_str(type).c_str();
        }

        template <typename T>
        bool dtype_same(DType type) {
            DType t = ctype<T>();
            return (t == type);
        }

        bool is_float(const DType type) {
            bool ans = false;
            if (type == DType::float32 || type == DType::float64) {
                ans = true;
            }
            return ans;
        }

        bool is_unsigned(const DType type) {
            bool ans = false;
            if (type == DType::uint8 || type == DType::uint16 || type == DType::uint16 || type == DType::uint32 || type == DType::uint64) {
                ans = true;
            }
            return ans;
        }
    }
}

#endif
