#ifndef __U_TENSOR_SHAPE_HPP__
#define __U_TENSOR_SHAPE_HPP__

/***
u-shape.hpp base functions for tensor
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

#include <vector>
#include <tuple>
#include <algorithm>

#include "u-dtype.hpp"
#include <initializer_list>

namespace u {
    namespace tensor {
        static const int all = std::numeric_limits<int>::max();

        class Shape : public std::vector<size_t> {
        private:
            std::tuple<size_t, size_t, size_t, size_t> prepare_for_dimension_operation_(const int axis) {
                // this function is called when do dimensional operation, such as min / max / mean / sum
                // shape are splitted into threee parts:
                //    [previous, laters, offset]
                //    where previous represents the product of all axes before axis
                //          laters represents the product of all axes after axis (include axis)
                //          offset represents the product of all axes after axis (exclude axis)

                u_fun_enter(0, 0);
                size_t dim_size = volume();
                size_t previous = 1;
                size_t laters = 1;
                size_t offset = 1;
                if (axis != u::tensor::all) {
                    size_t _axis = axis_normalize(axis);
                    dim_size = (*this)[_axis];
                    laters = volume(_axis, -1);
                    if (static_cast<size_t>(_axis+1) < rank()) {
                        offset = volume(_axis+1, -1);
                    }
                    if (_axis > 0) {
                        previous = volume(0, _axis-1);
                    }
                }
                u_fun_exit(0, 0);
                return std::make_tuple(dim_size, previous, laters, offset);
            }

            // std::vector<size_t> index_(const int axis) {
            //     std::tuple<size_t, size_t, size_t, size_t> splits = prepare_for_dimension_operation_(axis);
            //     size_t dim_size = std::get<0>(split);
            //     size_t previous = std::get<1>(split);
            //     size_t later    = std::get<2>(split);
            //     size_t offset   = std::get<3>(split);
            //     std::vector<size_t> ret(dim_size);
            //     for (size_t i=0; i<dim_size; ++i) {
            //         ret[i] = prev * later + i * offset + inner;
            //     }
            // }

            const std::tuple<size_t, size_t, size_t, size_t> prepare_for_dimension_operation_(const int axis) const {
                // see @std::tuple<size_t, size_t, size_t, size_t> prepare_for_dimension_operation_(const int axis)
                u_fun_enter(0, 0);
                size_t dim_size = volume();
                size_t previous = 1;
                size_t laters = 1;
                size_t offset = 1;
                if (axis != u::tensor::all) {
                    size_t _axis = axis_normalize(axis);
                    dim_size = (*this)[_axis];
                    laters = volume(_axis, -1);
                    if (static_cast<size_t>(_axis+1) < rank()) {
                        offset = volume(_axis+1, -1);
                    }
                    if (_axis > 0) {
                        previous = volume(0, _axis-1);
                    }
                }
                u_fun_exit(0, 0);
                return std::make_tuple(dim_size, previous, laters, offset);
            }

            Shape prepare_shape_for_dimension_operation_(const int axis, bool keepdims=false) {
                // this function prepare shapes for dimensional operations
                // for example, we have shape = {3, 4, 8, 9, 10} and axis = 2 or axis = -3
                // return {3, 4, 9, 10} if keepdims is false
                // reurn {3, 4, 1, 9, 10} if keepdims is true
                u_fun_enter(0, 0);
                Shape ret(*this);
                if (axis == u::tensor::all) {
                    if (keepdims) {
                        ret.assign(rank(), 1);
                    } else {
                        // scalar
                        ret.clear();
                    }
                } else {
                    size_t _axis = axis_normalize(axis);
                    if (keepdims) {
                        ret[_axis] = 1;
                    } else {
                        ret.erase(ret.begin()+_axis);
                    }
                }
                u_fun_exit(0, 0);
                return ret;
            }

            const Shape prepare_shape_for_dimension_operation_(const int axis, bool keepdims=false) const {
                // see @Shape prepare_shape_for_dimension_operation_(const int axis, bool keepdims=false)
                u_fun_enter(0, 0);
                Shape ret(*this);
                if (axis == u::tensor::all) {
                    if (keepdims) {
                        ret.assign(rank(), 1);
                    } else {
                        // scalar
                        ret.clear();
                    }
                } else {
                    size_t _axis = axis_normalize(axis);
                    if (keepdims) {
                        ret[_axis] = 1;
                    } else {
                        ret.erase(ret.begin()+_axis);
                    }
                }
                u_fun_exit(0, 0);
                return ret;
            }

        public:
            virtual ~Shape(){}
            Shape(){};
            Shape(size_t rank) : std::vector<size_t>(rank) {}
            Shape(size_t rank, size_t value) : std::vector<size_t>(rank, value) {}
            Shape(const std::vector<size_t> &shape) : std::vector<size_t>(shape) {}
            Shape(const Shape &shape) {
                u_fun_enter(0, 0);
                resize(shape.rank());
                assign(shape.begin(), shape.end());
                u_fun_exit(0, 0);
            }

            Shape operator =(const Shape &shape) {
                u_fun_enter(0, 0);
                resize(shape.rank());
                assign(shape.begin(), shape.end());
                u_fun_exit(0, 0);
                return *this;
            }

            bool operator == (const Shape &shape) {
                u_fun_enter(0, 0);
                u_assert(this->rank() == shape.rank(), u::format("cannot compare shapes with different rank. (%zu vs %zu)", this->rank(), shape.rank()));
                bool same = true;
                for(size_t i=0; i<shape.rank(); ++i) {
                    if ((*this)[i] != shape[i]) {
                        same = false;
                        break;
                    }
                }
                u_fun_exit(0, 0);
                return same;
            }

            bool operator != (const Shape &shape) {
                u_fun_enter(0, 0);
                u_assert(this->rank() == shape.rank(), u::format("cannot compare shapes with different rank. (%zu vs %zu)", this->rank(), shape.rank()));
                bool diff = false;
                for(size_t i=0; i<shape.rank(); ++i) {
                    if ((*this)[i] != shape[i]) {
                        diff = true;
                        break;
                    }
                }
                u_fun_exit(0, 0);
                return diff;
            }

            friend std::ostream & operator <<(std::ostream &os, const Shape &shape) {
                u_fun_enter(0, 0);
                os << "{";
                for (size_t i=0; i<shape.rank(); ++i) {
                    os << shape[i];
                    if (i+1 < shape.rank()) {
                        os << ", ";
                    }
                }
                os << "}" << std::flush;
                u_fun_exit(0, 0);
                return os;
            }

            std::string str() {
                u_fun_enter(0, 0);
                std::ostringstream oss;
                oss << *this;
                u_fun_exit(0, 0);
                return oss.str();
            }

            const std::string str() const {
                u_fun_enter(0, 0);
                std::ostringstream oss;
                oss << *this;
                u_fun_exit(0, 0);
                return oss.str();
            }

            template <typename T=size_t>
            T volume(int beg=0, int end=-1) {
                u_fun_enter(0, 0);
                // product of elements from beg to end.
                // NOTE include axes both `beg` and `end`
                T ret = 1;
                if (size() > 0) {
                    unsigned int end_ = axis_normalize(end);
                    unsigned int beg_ = axis_normalize(beg);
                    u_assert(beg_ <= end_, u::format("begin point must be greater than end point (%zu [%d] vs %zu [%d])", beg_, beg, end_, end));
                    // NOTE for scalar, though size() == 0, 1 should be returned
                    for(size_t i=beg_; i<=end_; ++i) {
                        ret *= static_cast<T>((*this)[i]);
                    }
                }
                u_fun_exit(0, 0);
                return ret;
            }

            template <typename T=size_t>
            const T volume(int beg=0, int end=-1) const {
                u_fun_enter(0, 0);
                T ret = 1;
                if (size() > 0) {
                    unsigned int end_ = axis_normalize(end);
                    unsigned int beg_ = axis_normalize(beg);
                    u_assert(beg_ <= end_, u::format("begin point must be greater than end point (%zu [%d] vs %zu [%d])", beg_, beg, end_, end));
                    // NOTE for scalar, though size() == 0, 1 should be returned
                    for(size_t i=beg_; i<=end_; ++i) {
                        ret *= static_cast<T>((*this)[i]);
                    }
                }
                u_fun_exit(0, 0);
                return ret;
            }

            inline const size_t rank() const {return size();}
            inline size_t rank() {return size();}

            static const Shape broadcast(const Shape &s1, const Shape &s2) {return s1.broadcast(s2);}

            Shape broadcast(const Shape &shape) {
                u_fun_enter(0, 0);
                Shape a(*this);
                Shape b(shape);
                if (a.rank() > b.rank()) {
                    b.insert(b.begin(), (a.rank() - b.rank()), 1);
                } else {
                    a.insert(a.begin(), (b.rank() - a.rank()), 1);
                }
                Shape ret(a.rank(), 0);
                for(int i=static_cast<int>(ret.rank()-1); i>=0; --i){
                    size_t adim = a[i];
                    size_t bdim = b[i];
                    u_assert(adim == 1 || bdim == 1 || adim == bdim, u::format("broadcast: either both dimension euqal or one of them be 1. given {%zu, %zu}", adim, bdim));
                    ret[i] = std::max(adim, bdim);
                }
                u_fun_exit(0, 0);
                return ret;
            }

            const Shape broadcast(const Shape &shape) const {
                u_fun_enter(0, 0);
                Shape ret = shape;
                if (shape != *this) {
                    Shape a(*this);
                    Shape b(shape);
                    if (a.rank() > b.rank()) {
                        b.insert(b.begin(), (a.rank() - b.rank()), 1);
                    } else {
                        a.insert(a.begin(), (b.rank() - a.rank()), 1);
                    }
                    ret.resize(a.rank());
                    for(int i=static_cast<int>(ret.rank())-1; i>=0; --i){
                        size_t adim = a[i];
                        size_t bdim = b[i];
                        u_assert(adim == 1 || bdim == 1 || adim == bdim, u::format("broadcast: either both dimension euqal or one of them be 1. given {%zu, %zu}", adim, bdim));
                        ret[i] = std::max(adim, bdim);
                    }
                }
                u_fun_enter(0, 0);
                return ret;
            }

            const Shape broadcast(const std::vector<size_t> &shape) {
                return broadcast(Shape(shape));
            }

            bool broadcastable(const Shape &shape) {
                u_fun_enter(0, 0);
                bool ans = true;
                if (shape != *this) {
                    Shape a(*this);
                    Shape b(shape);
                    if (a.rank() > b.rank()) {
                        b.insert(b.begin(), (a.rank() - b.rank()), 1);
                    } else {
                        a.insert(a.begin(), (b.rank() - a.rank()), 1);
                    }
                    for(int i=static_cast<int>(a.rank())-1; i>=0; --i){
                        size_t adim = a[i];
                        size_t bdim = b[i];
                        if (adim != bdim && adim != 1 && bdim != 1 ) {
                            ans = false;
                            break;
                        }
                    }
                }
                u_fun_enter(0, 0);
                return ans;
            }

            static std::tuple<Shape, Shape, Shape> adapt_shape(const Shape &shape1, const Shape &shape2) {
                u_fun_enter(0, 0);
                Shape ret = shape1;
                Shape a(shape1);
                Shape b(shape2);
                if (shape1 != shape2) {
                    if (a.rank() > b.rank()) {
                        b.insert(b.begin(), (a.rank() - b.rank()), 1);
                    } else if (a.rank() < b.rank()) {
                        a.insert(a.begin(), (b.rank() - a.rank()), 1);
                    }
                    ret.resize(a.rank());
                    for(int i=static_cast<int>(ret.rank())-1; i>=0; --i){
                        size_t adim = a[i];
                        size_t bdim = b[i];
                        u_assert(adim == 1 || bdim == 1 || adim == bdim, u::format("broadcast: either both dimension euqal or one of them be 1. given {%zu, %zu}", adim, bdim));
                        ret[i] = std::max(adim, bdim);
                    }
                }
                u_fun_enter(0, 0);
                return std::make_tuple(ret, a, b);
            }

            inline size_t axis_normalize(const int axis) {return axis_normalize(*this, axis);}

            inline const size_t axis_normalize(const int axis) const {return axis_normalize(*this, axis);}

            inline std::vector<size_t> axis_normalize(const std::vector<int> &axis) {return axis_normalize(*this, axis);}

            inline const std::vector<size_t> axis_normalize(const std::vector<int> &axis) const {return axis_normalize(*this, axis);}

            static size_t axis_normalize(const Shape &shape, const int axis) {
                // normalize axis
                // if axis is positive, do nothing
                // else convert it to positive by adding rank
                // this is like mod operation
                u_fun_enter(0, 0);
                size_t _axis_ = axis >= 0 ? axis : (shape.rank() + axis);
                u_assert(shape.rank() > _axis_,u::format("absolute value of axis should not be greater than shape dimensions (%zu vs %zu) <axis: %d>", shape.rank(), _axis_, axis));
                u_fun_exit(0, 0);
                return (_axis_);
            }

            // transform list of relative axes to absolute axes
            static std::vector<size_t> axis_normalize(const Shape &shape, const std::vector<int> &axis) {
                u_fun_enter(0, 0);
                std::vector<size_t> ret(axis.size());
                for (size_t idx = 0; idx < axis.size(); ++idx) {
                    ret[idx] = axis_normalize(shape, axis[idx]);
                }
                u_fun_exit(0, 0);
                return (ret);
            }

            inline std::tuple<size_t, size_t, size_t, size_t> split(const int axis) {return prepare_for_dimension_operation_(axis);}

            inline const std::tuple<size_t, size_t, size_t, size_t> split(const int axis) const {return prepare_for_dimension_operation_(axis);}

            inline Shape get_shape_dimension(const int axis, bool keepdims=false) {return prepare_shape_for_dimension_operation_(axis, keepdims);}

            inline const Shape get_shape_dimension(const int axis, bool keepdims=false) const {return prepare_shape_for_dimension_operation_(axis, keepdims);}

            inline Shape transpose(const std::vector<int> &axis) {
                u_fun_enter(0, 0);
                std::vector<size_t> _axis_ = axis_normalize(axis);
                //u_assert([&_axis_]()->bool{return std::unique(_axis_.begin(), _axis_.end()) == _axis_.end();}, u::format("axis duplicant"));
                if (std::unique(_axis_.begin(), _axis_.end()) != _axis_.end()) {
                    bool AXIS_DUPLICANT_ERROR = false;
                    u_assert(AXIS_DUPLICANT_ERROR, "axis contains duplicant ones. pay attention to axis indicated by both pisitive and negative value");
                }
                u_fun_exit(0, 0);
                return Shape();
            }

            inline Shape& reshape(const Shape &shape) {
                u_fun_enter(0, 0);
                u_assert(volume() == shape.volume(), u::format("cannot reshape to different volumes. given (%zu, %zu)", volume(), shape.volume()));
                resize(shape.size());
                assign(shape.begin(), shape.end());
                u_fun_exit(0, 0);
                return *this;
            }

            template <typename T>
            inline Shape& reshape(const std::vector<T> &shape) {
                u_fun_enter(0, 0);
                DType type = ctype<T>();
                if (type == DType::float32 || type == DType::float64){
                    bool NON_SUPPORT_DTYPE_FOR_RESHAPE = false;
                    u_assert(NON_SUPPORT_DTYPE_FOR_RESHAPE, u::format("reshape not support %s for specifying dimension size", dtype_str(type)));
                }
                std::vector<size_t> _shape_(shape.size(), 0);
                if (type == DType::int8 || type == DType::int16 || type == DType::int32 || type == DType::int64) {
                    int nneg = -1;
                    size_t capacity = 1;
                    for(size_t i=0; i<shape.size(); ++i) {
                        if (shape[i] <= 0) {
                            u_assert(nneg == -1, "found multiple non-positive rank in shape. inpossible to inference the right shape");
                            nneg = static_cast<int>(i);
                        } else {
                            capacity *= shape[i];
                            _shape_[i] = shape[i];
                        }
                    }
                    if (nneg != -1) {
                        _shape_[nneg] = static_cast<size_t>(volume() / capacity);
                        capacity *= _shape_[nneg];
                    }
                } else {
                    for (size_t i=0; i<shape.size(); ++i) {
                        _shape_[i] = static_cast<size_t>(shape[i]);
                    }
                }
                Shape s(_shape_);
                reshape(s);
                u_fun_exit(0, 0);
                return *this;
            }
        };
    }
}

#endif
