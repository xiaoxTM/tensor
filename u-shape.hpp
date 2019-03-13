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

                u_fun_enter(2, 0);
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
                u_fun_exit(0, -2);
                return std::make_tuple(dim_size, previous, laters, offset);
            }

            const std::tuple<size_t, size_t, size_t, size_t> prepare_for_dimension_operation_(const int axis) const {
                // see @std::tuple<size_t, size_t, size_t, size_t> prepare_for_dimension_operation_(const int axis)
                u_fun_enter(2, 0);
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
                u_fun_exit(0, -2);
                return std::make_tuple(dim_size, previous, laters, offset);
            }

            Shape prepare_shape_for_dimension_operation_(const int axis, bool keepdims=false) {
                // this function prepare shapes for dimensional operations
                // for example, we have shape = {3, 4, 8, 9, 10} and axis = 2 or axis = -3
                // return {3, 4, 9, 10} if keepdims is false
                // reurn {3, 4, 1, 9, 10} if keepdims is true
                u_fun_enter(2, 0);
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
                u_fun_exit(0, -2);
                return ret;
            }

            const Shape prepare_shape_for_dimension_operation_(const int axis, bool keepdims=false) const {
                // see @Shape prepare_shape_for_dimension_operation_(const int axis, bool keepdims=false)
                u_fun_enter(2, 0);
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
                u_fun_exit(0, -2);
                return ret;
            }

        public:
            virtual ~Shape(){}
            Shape(){};
            // to avoid ambigous when shape is 1D or 2D with Shape(const std::vector<size_t> &shape)
            // by using initializer_list feature, we change the following construct to `create` factory
            // Shape(size_t rank) : std::vector<size_t>(rank) {}
            // Shape(size_t rank, const size_t &value, bool=true) : std::vector<size_t>(rank, value) {}
            Shape(const std::vector<size_t> &shape) : std::vector<size_t>(shape) {}
            Shape(const Shape &shape) {
                u_fun_enter(2, 0);
                resize(shape.rank());
                assign(shape.begin(), shape.end());
                u_fun_exit(0, -2);
            }

            static Shape create(size_t rank, size_t value=0) {
                Shape ans;
                ans.assign(rank, value);
                return ans;
            }

            Shape operator =(const Shape &shape) {
                u_fun_enter(2, 0);
                resize(shape.rank());
                assign(shape.begin(), shape.end());
                u_fun_exit(0, -2);
                return *this;
            }

            bool operator == (const Shape &shape) const {
                u_fun_enter(2, 0);
                bool same = true;
                if (rank() != shape.rank()) {
                    same = false;
                } else {
                    for(size_t i=0; i<shape.rank(); ++i) {
                        if ((*this)[i] != shape[i]) {
                            same = false;
                            break;
                        }
                    }
                }
                u_fun_exit(0, -2);
                return same;
            }

            bool operator != (const Shape &shape) const {
                u_fun_enter(2, 0);
                bool diff = false;
                if (rank() != shape.rank()) {
                    diff = true;
                } else {
                    for(size_t i=0; i<shape.rank(); ++i) {
                        if ((*this)[i] != shape[i]) {
                            diff = true;
                            break;
                        }
                    }
                }
                u_fun_exit(0, -2);
                return diff;
            }

            friend std::ostream & operator <<(std::ostream &os, const Shape &shape) {
                u_fun_enter(2, 0);
                os << "{";
                for (size_t i=0; i<shape.rank(); ++i) {
                    os << shape[i];
                    if (i+1 < shape.rank()) {
                        os << ", ";
                    }
                }
                os << "}" << std::flush;
                u_fun_exit(0, -2);
                return os;
            }

            std::string str() const {
                u_fun_enter(2, 0);
                std::ostringstream oss;
                oss << *this;
                u_fun_exit(0, -2);
                return oss.str();
            }

            std::string c_str() const {
                return str().c_str();
            }

            template <typename T=size_t>
            T volume(int beg=0, int end=-1, bool inc_beg=true, bool inc_end=true) const {
                u_fun_enter(2, 0);
                // product of elements from beg to end.
                // NOTE include axes both `beg` and `end`
                T ret = 1;
                if (size() > 0) {
                    unsigned int end_ = axis_normalize(end);
                    unsigned int beg_ = axis_normalize(beg);
                    if (!inc_beg) ++ beg_;
                    if (!inc_end) -- end_;
                    u_assert(beg_ <= end_, u::format("begin point must be greater than end point (%zu[%d] vs %zu[%d])", beg_, beg, end_, end));
                    // NOTE for scalar, though size() == 0, 1 should be returned
                    for(size_t i=beg_; i<=end_; ++i) {
                        ret *= static_cast<T>((*this)[i]);
                    }
                }
                u_fun_exit(0, -2);
                return ret;
            }

            inline size_t rank() const {return size();}

            int next_axis(int begin, const Shape &shape, bool same=true) const {
                u_fun_enter(2, 0);
                u_assert(this->rank() == shape.rank(), u::format("`this` and `shape` must have rank. given (%zu vs %zu)", this->rank(), shape.rank()));
                int axis = -1;
                if (same) {
                    for (int i = begin; i >= 0; --i) {
                        if ((*this)[i] == shape[i]) {
                            axis = i;
                            break;
                        }
                    }
                } else {
                    for (int i = begin; i >= 0; --i) {
                        if ((*this)[i] != shape[i]) {
                            axis = i;
                            break;
                        }
                    }
                }
                u_fun_exit(0, -2);
                return axis;
            }

            /**
             ** sub shape without axis from `axis` to `axis + num`
            */
            Shape esub(int axis, size_t num=1) const {
                u_fun_enter(2, 0);
                if (num > 0) {
                    -- num;
                }
                size_t len = size();
                u_assert(len > 0, u::format("cannot get sub-shape for shape with size of %zu", len));
                size_t beg_axis = axis_normalize(axis);
                size_t end_axis = axis_normalize(axis + num);
                u_assert(len > beg_axis && len > end_axis, u::format("axis overflow: %zu vs (%zu[%d] + %zu)", len, beg_axis, axis, num));
                Shape shape = create(len-1);
                int idx = -1;
                for (size_t i=0; i<len; ++i) {
                    if (i < beg_axis or i >= end_axis) {
                        shape[++ idx] = (*this)[i];
                    }
                }
                u_fun_exit(0, -2);
                return shape;
            }

            size_t offsetmap(const size_t num, int end=-1) const {
                u_fun_enter(2, 0);
                u_assert(num < volume(), u::format("num2offset num overflow (%zu vs %zu)", num, volume()));
                size_t vol = 0;
                size_t left = num;
                size_t capacity = 0;
                size_t end_axis = axis_normalize(end);
                for (size_t i=0; i<=end_axis; ++i) {
                    size_t v = volume(i, end);
                    capacity = static_cast<size_t>(left / v) * v;
                    left -= capacity;
                    vol +=  capacity;
                }
                u_fun_exit(0, -2);
                return vol;
            }

            size_t offsetmap(const Shape &shape, const size_t num, int end=-1) const {
                u_fun_enter(2, 0);
                u_assert(num < shape.volume(), u::format("num2offset num overflow (%zu vs %zu)", num, shape.volume()));
                u_assert(shape.rank() == size(), u::format("num2offset of `shape` should have rank equal `*this`. given (%zu vs %zu)", shape.rank(), this->rank()));
                size_t vol = 0;
                size_t left = num;
                size_t capacity = 0;
                size_t end_axis = axis_normalize(end);
                for (size_t i=0; i<=end_axis; ++i) {
                    size_t v = volume(i, end);
                    capacity = static_cast<size_t>(left / v);
                    left -= (capacity * v);
                    vol += (shape.volume(i, end) * capacity);
                }
                u_fun_exit(0, -2);
                return vol;
            }

            static Shape concatenate(const std::vector<Shape> &shapes, int axis) {
                u_fun_enter(2, 0);
                Shape shape(shapes[0]);
                { // check shape compabitity
                    Shape subshape = shape.esub(axis);
                    for (size_t i=1; i<shapes.size(); ++i) {
                        u_assert(shape.size() == shapes[i].size(), u::format("%zu-th and %zu-th shape have different rank (%zu vs %zu)", shape.rank(), shapes[i].rank()));
                        u_assert(subshape == shapes[i].esub(axis), u::format("%zu-th shape[%s] and %zu-th shape[%s] have different axis besides %d axis", 0, subshape.c_str(), i, shapes[i].esub(axis), axis));
                    }
                }
                size_t naxis = shape.axis_normalize(axis);
                for (size_t i=1; i<shapes.size(); ++i) {
                    shape[naxis] += shapes[i][naxis];
                }
                u_fun_exit(0, -2);
                return shape;
            }

            static const Shape broadcast(const Shape &s1, const Shape &s2) {return s1.broadcast(s2);}

            Shape broadcast(const Shape &shape) const {
                u_fun_enter(2, 0);
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
                u_fun_enter(2, 0);
                return ret;
            }

            Shape broadcast(const std::vector<size_t> &shape) const {
                return broadcast(Shape(shape));
            }

            /**
             ** whether shape_ and shape is broadcast compabitiable
             ** if particial is true
             **    whether shape can be broadcast to shape_
            */
            bool broadcastable(const Shape &shape, bool particial=false) const {
                u_fun_enter(2, 0);
                bool ans = true;
                if (shape != *this) {
                    Shape a(*this);
                    Shape b(shape);
                    if (a.rank() > b.rank()) {
                        b.insert(b.begin(), (a.rank() - b.rank()), 1);
                    } else if (a.rank() < b.rank() && !particial) {
                        a.insert(a.begin(), (b.rank() - a.rank()), 1);
                    }
                    if (particial && a.rank() < b.rank()) {
                        ans =false;
                    } else {
                        for(int i=static_cast<int>(a.rank())-1; i>=0; --i){
                            size_t adim = a[i];
                            size_t bdim = b[i];
                            if (adim != bdim && adim != 1 && bdim != 1 ) {
                                ans = false;
                                break;
                            }
                        }
                    }
                }
                u_fun_enter(2, 0);
                return ans;
            }

            static std::tuple<Shape, Shape, Shape> adapt_shape(const Shape &shape1, const Shape &shape2) {
                u_fun_enter(2, 0);
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
                u_fun_enter(2, 0);
                return std::make_tuple(ret, a, b);
            }

            template<typename T=size_t>
            inline T axis_normalize(const int axis) const {return axis_normalize<T>(*this, axis);}

            template<typename T=size_t>
            inline std::vector<T> axis_normalize(const std::vector<int> &axis) const {return axis_normalize<T>(*this, axis);}

            template<typename T=size_t>
            static T axis_normalize(const Shape &shape, const int axis) {
                // normalize axis
                // if axis is positive, do nothing
                // else convert it to positive by adding rank
                // this is like mod operation
                u_fun_enter(2, 0);
                T _axis_ = axis >= 0 ? axis : (shape.rank() + axis);
                u_assert(shape.rank() > _axis_,u::format("absolute value of axis should not be greater than shape dimensions (%zu vs %zu) <axis: %d>", shape.rank(), _axis_, axis));
                u_fun_exit(0, -2);
                return (_axis_);
            }

            // transform list of relative axes to absolute axes
            template<typename T=size_t>
            static std::vector<T> axis_normalize(const Shape &shape, const std::vector<int> &axis) {
                u_fun_enter(2, 0);
                std::vector<T> ret(axis.size());
                for (size_t idx = 0; idx < axis.size(); ++idx) {
                    ret[idx] = axis_normalize<T>(shape, axis[idx]);
                }
                u_fun_exit(0, -2);
                return (ret);
            }

            inline std::tuple<size_t, size_t, size_t, size_t> split(const int axis) const {return prepare_for_dimension_operation_(axis);}

            inline Shape get_shape_dimension(const int axis, bool keepdims=false) const {return prepare_shape_for_dimension_operation_(axis, keepdims);}

            inline Shape transpose(const std::vector<int> &axis) const {
                u_fun_enter(2, 0);
                std::vector<size_t> _axis_ = axis_normalize(axis);
                //u_assert([&_axis_]()->bool{return std::unique(_axis_.begin(), _axis_.end()) == _axis_.end();}, u::format("axis duplicant"));
                if (std::unique(_axis_.begin(), _axis_.end()) != _axis_.end()) {
                    bool AXIS_DUPLICANT_ERROR = false;
                    u_assert(AXIS_DUPLICANT_ERROR, "axis contains duplicant ones. pay attention to axis indicated by both pisitive and negative value");
                }
                u_fun_exit(0, -2);
                return Shape();
            }

            inline Shape& reshape(const Shape &shape) {
                u_fun_enter(2, 0);
                u_assert(volume() == shape.volume(), u::format("cannot reshape to different volumes. given (%zu, %zu)", volume(), shape.volume()));
                resize(shape.size());
                assign(shape.begin(), shape.end());
                u_fun_exit(0, -2);
                return *this;
            }

            template <typename T>
            inline Shape& reshape(const std::vector<T> &shape) {
                u_fun_enter(2, 0);
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
                u_fun_exit(0, -2);
                return *this;
            }
        };
    }
}

#endif
