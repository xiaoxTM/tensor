#ifndef __U_TENSOR_TENSOR_HPP__
#define __U_TENSOR_TENSOR_HPP__

/***
u-tensor.hpp base functions for tensor
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

#include "u-dtype.hpp"
#include "u-shape.hpp"
#include "u-op.hpp"
#include "u-mm.hpp"
#include <initializer_list>
#include <vector>
#include <iomanip>

namespace u {

    namespace tensor {
        class Tensor{
        protected:

            template <typename T>
            void _init(const T *data, size_t size=1) {
                u_fun_enter(2, 0);
                malloc();
                mm::upload(data_.get(), reinterpret_cast<const unsigned char * const >(data), sizeof(T)*size);
                u_fun_exit(0, -2);
            }

            void _init(const unsigned char *data, DType type, size_t size=1) {
                u_fun_enter(2, 0);
                malloc();
                mm::upload(data_.get(), data, dtype_size(type)*size);
                u_fun_exit(0, -2);
            }

        private:
            std::shared_ptr<unsigned char> data_;
            Shape shape_;
            DType type_;
            int print_precision_;
            int print_width_;

            size_t print_(size_t rank, size_t begin, std::ostream &os) {
                size_t end = begin;
                if (shape_.rank() == 0) {
                    op::run<Tensor, op::Print>(*this, begin, end, os, print_precision_, print_width_);
                } else {
                    os << "[";
                    if (rank < shape_.rank() - 1) {
                        std::string space(rank + 1, ' ');
                        for (size_t i = 0; i < shape_[rank]; ++i) {
                            end = print_(rank + 1, end, os);
                            if (i < shape_[rank] - 1) {
                                os << "],\n" << space;
                            } else {
                                os << "]";
                            }
                        }
                    } else {
                        end += shape_[rank];
                        op::run<Tensor, op::Print>(*this, begin, end, os, print_precision_, print_width_);
                    }
                    if (rank == 0) {
                        os << "]";
                    }
                }
                return (end);
            }

            template <template<typename, typename > class T>
            Tensor dimension_op_run_(const int axis, const DType type, bool keepdims) const {
                u_fun_enter(2, 0);
                Shape shape = shape_.get_shape_dimension(axis, keepdims);
                Tensor ret(shape, (type == DType::invalid ? type_ : type), false);
                op::run2<Tensor, T>(ret, *this, axis);
                u_fun_exit(0, -2);
                return ret;
            }

        public:

            virtual ~Tensor() {}

            Tensor(): data_(nullptr), shape_(), type_(DType::invalid), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {}

            Tensor(const DType type) : data_(nullptr), shape_(), type_(type), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                u_fun_exit(0, -2);
            }

            Tensor(const Shape& shape, const DType type, bool lazy = false) : shape_(shape), type_(type), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, u::format("size must be greater than 0. given %zu", v));
                data_ = nullptr;
                if (!lazy) {
                    malloc();
                }
                u_fun_exit(0, -2);
            }

            Tensor(const std::initializer_list<size_t> &shape, const DType type, bool lazy = false) : shape_(shape), type_(type), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, u::format("size must be greater than 0. given %zu", v));
                data_ = nullptr;
                if (!lazy) {
                    malloc();
                }
                u_fun_exit(0, -2);
            }

            Tensor(unsigned char * data, const std::vector<size_t> &shape, const DType type, bool copy = false) : shape_(shape), type_(type), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, u::format("size must be greater than 0. given %zu", v));
                u_assert(data != nullptr, "data can not be null pointer");
                if (copy) {
                    //size_t size_in_byte = bytesize();
                    _init<unsigned char>(data, bytesize());
                } else {
                    // use void free to avoid freeing referred memory which should be freed by caller
                    data_.reset(data, mm::no_free);
                }
                u_fun_exit(0, -2);
            }

            Tensor(unsigned char * data, const Shape& shape, const DType type, bool copy = false) : shape_(shape), type_(type), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, u::format("size must be greater than 0. given %zu", v));
                u_assert(data != nullptr, "data can not be null pointer");
                if (copy) {
                    _init<unsigned char>(data, bytesize());
                } else {
                    data_.reset(data, mm::no_free);
                }
                u_fun_exit(0, -2);
            }

            Tensor(const unsigned char * const data, const Shape &shape, const DType type) : shape_(shape), type_(type), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, u::format("size must be greater than 0. given %zu", v));
                u_assert(data != nullptr, "data can not be null pointer");
                _init<unsigned char>(data, bytesize());
                u_fun_exit(0, -2);
            }

            Tensor(const unsigned char * const data, const std::vector<size_t> &shape, const DType type) : shape_(shape), type_(type), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, "no allowing copy tensor with 0 size.");
                u_assert(data != nullptr, "data can not be null pointer");
                _init<unsigned char>(data, bytesize());
                u_fun_exit(0, -2);
            }

            // invoked in the following case
            // Tensor t1(t2);
            Tensor(const Tensor &t) : shape_(t.shape()), type_(t.type()), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                u_assert(type_ < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, u::format("size must be greater than 0. given %zu", v));
                u_assert(t.cref() != nullptr, "data can not be null pointer");
                _init<unsigned char>(t.cref(), bytesize());
                u_fun_exit(0, -2);
            }

            Tensor(const char& data) : shape_(), type_(DType::int8), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                _init<char>(&data);
                u_fun_exit(0, -2);
            }

            Tensor(const unsigned char& data) : shape_(), type_(DType::uint8), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                _init<unsigned char>(&data);
                u_fun_exit(0, -2);
            }

            Tensor(const short& data) : shape_(), type_(DType::int16), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                _init<short>(&data);
                u_fun_exit(0, -2);
            }

            Tensor(const unsigned short& data) : shape_(), type_(DType::uint16), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                _init<unsigned short>(&data);
                u_fun_exit(0, -2);
            }

            Tensor(const int& data) : shape_(), type_(DType::int32), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                _init<int>(&data);
                u_fun_exit(0, -2);
            }

            Tensor(const unsigned int& data) : shape_(), type_(DType::uint32), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                _init<unsigned int>(&data);
                u_fun_exit(0, -2);
            }

            Tensor(const long& data) : shape_(), type_(DType::int64), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                _init<long>(&data);
                u_fun_exit(0, -2);
            }

            Tensor(const size_t& data) : shape_(), type_(DType::uint64), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                _init<size_t>(&data);
                u_fun_exit(0, -2);
            }

            Tensor(const float& data) : shape_(), type_(DType::float32), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                _init<float>(&data);
                u_fun_exit(0, -2);
            }

            Tensor(const double& data) : shape_(), type_(DType::float64), print_precision_(std::numeric_limits<long double>::digits10 + 1), print_width_(10) {
                u_fun_enter(2, 0);
                _init<double>(&data);
                u_fun_exit(0, -2);
            }

            friend std::ostream &operator <<(std::ostream &os, const Tensor &t) {
                const_cast<Tensor&>(t).print_(0, 0, os);
                return (os);
            }

            std::string str() const {
                std::ostringstream oss;
                oss << *this;
                return oss.str();
            }

            int precision() const {return print_precision_;}
            void precision(int prec) {print_precision_ = prec;}

            int width() const {return print_width_;}
            void width(int w){print_width_ = w;}

            void malloc(const Shape &shape, DType type) {
                u_fun_enter(2, 0);
                u_assert(data_ == nullptr, "data not empty. memory may be already allocated. use `realloc' instead");
                if (data_ != nullptr) {
                    data_.reset(mm::malloc(shape.volume(), dtype_size(type)), mm::mfree);
                }
                u_fun_exit(0, -2);
            }

            void realloc(const Shape &shape) {data_.reset(mm::malloc(shape.volume(), dtype_size(type_)), mm::mfree);}

            void malloc(DType type) {
                u_fun_enter(2, 0);
                u_assert(data_ == nullptr, "data not empty. memory may be already allocated.");
                if (data_ != nullptr) {
                    data_.reset(mm::malloc(volume(), dtype_size(type)), mm::mfree);
                }
                u_fun_exit(0, -2);
            }

            void malloc() {
                u_fun_enter(2, 0);
                u_assert(data_.get() == nullptr, "data already allocated.");
                u_assert(type_ != DType::invalid, "cannot allocate memory for invalid data type");
                data_.reset(mm::malloc(volume(), dtype_size(type_)), mm::mfree);
                u_fun_exit(0, -2);
            }

            const unsigned char *cref() const {return data_.get();}

            unsigned char * ref() const {return data_.get();}

            std::shared_ptr<unsigned char> data() const {return data_;}

            DType type() const {return type_;}

            // rank
            size_t rank() const {return (shape_.size());}

            size_t volume() const {return shape_.volume<size_t>();}

            size_t bytesize() const {return (volume() * dtype_size(type_));}

            // shape of dimension
            Shape shape() const {return (shape_);}

            // clip in inplace mode
            void clip_inplace(double min, double max) {
                u_fun_enter(2, 0);
                op::run2<Tensor, op::Clip>(*this, *this, min, max);
                u_fun_exit(0, -2);
            }

            Tensor clip(double min, double max) const {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                ret.clip_inplace(min, max);
                u_fun_exit(0, -2);
                return ret;
            }

            void abs_inplace() {
                u_fun_enter(2, 0);
                op::run2<Tensor, op::Absolute>(*this, *this);
                u_fun_exit(0, -2);
            }

            Tensor abs() const {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                ret.abs();
                u_fun_exit(0, -2);
                return ret;
            }

            void exp_inplace() {
                u_fun_enter(2, 0);
                op::run2<Tensor, op::Experiential >(*this, *this);
                u_fun_exit(0, -2);
            }

            Tensor exp() const {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                ret.exp();
                u_fun_exit(0, -2);
                return ret;
            }

            void log_inplace() {
                u_fun_enter(2, 0);
                op::run2<Tensor, op::Log >(*this, *this);
                u_fun_exit(0, -2);
            }

            Tensor log() const {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                ret.log();
                u_fun_exit(0, -2);
                return ret;
            }

            void log10_inplace() {
                u_fun_enter(2, 0);
                op::run2<Tensor, op::Log10>(*this, *this);
                u_fun_exit(0, -2);
            }

            Tensor log10() const {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                ret.log10();
                u_fun_exit(0, -2);
                return ret;
            }

            // shallow-copy
            Tensor &operator =(const Tensor &d) {
                u_fun_enter(2, 0);
                type_ = d.type();
                shape_ = d.shape();
                data_ = d.data();
                print_precision_ = d.precision();
                print_width_ = d.width();
                u_fun_exit(0, -2);
                return (*this);
            }

            // deep-copy assign assignment
            Tensor &operator ()(const Tensor& d) {
                u_fun_enter(2, 0);
                if (type_ == DType::invalid) {
                    // assign to no initialized tensor
                    type_ = d.type();
                    shape_ = d.shape();
                    _init(d.cref(), type_, d.volume());
                    print_precision_ = d.precision();
                    print_width_ = d.width();
                } else {
                    u_assert(shape_.broadcastable(d.shape(), true), u::format("shape %s can not broadcast to %s", d.shape().c_str(), shape_.c_str()));
                    op::run2<Tensor, op::Assign>(*this, d.broadcast(shape_));
                    print_precision_ = d.precision();
                    type_ = d.type();
                    print_width_ = d.width();
                }
                u_fun_exit(0, -2);
                return (*this);
            }

            Tensor &operator ()(const Shape &shape, const DType type, bool lazy) {
                u_fun_enter(2, 0);
                if (shape_.volume() == shape.volume() && type_ == type) {
                    if (shape_ != shape) {
                        shape_ = shape;
                    }
                } else {
                    // if type == invalid, leave type_ as it is
                    if (type != DType::invalid) {
                        // assign to no initialized tensor
                        type_ = type;
                    }
                    data_ = nullptr;
                    shape_ = shape;
                    if (!lazy) {
                        malloc();
                    }
                }
                u_fun_exit(0, -2);
                return (*this);
            }

            Tensor operator ==(const Tensor &d) const {
                u_fun_enter(2, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" == operation can only be applied to tensor have same shape. given (%s, %s)", shape_.c_str(), d.shape().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("equal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::Equal>(ret, *this, d);
                u_fun_exit(0, -2);
                return (ret);
            }

            friend Tensor operator ==(const char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const unsigned char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const unsigned short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const unsigned int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const long &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const size_t &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const float &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const double &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) == rhs);
            }

            Tensor operator !=(const Tensor &d) const {
                u_fun_enter(2, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" != operation can only be applied to tensor have same shape. given (%s, %s)", shape_.c_str(), d.shape().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("nequal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::NotEqual>(ret, *this, d);
                u_fun_exit(0, -2);
                return (ret);
            }

            friend Tensor operator !=(const char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const unsigned char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const unsigned short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const unsigned int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const long &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const size_t &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const float &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const double &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) != rhs);
            }

            Tensor operator >(const Tensor &d) const {
                u_fun_enter(2, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" != operation can only be applied to tensor have same shape. given (%s, %s)", shape_.c_str(), d.shape().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("nequal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::Greater>(ret, *this, d);
                u_fun_exit(0, -2);
                return (ret);
            }

            friend Tensor operator > (const char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator > (const unsigned char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator > (const short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator > (const unsigned short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator > (const int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator > (const unsigned int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator > (const long &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator > (const size_t &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator > (const float &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator > (const double &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) > rhs);
            }

            Tensor operator >= (const Tensor &d) const {
                u_fun_enter(2, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" != operation can only be applied to tensor have same shape. given (%s, %s)", shape_.c_str(), d.shape().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("nequal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::GreaterEqual>(ret, *this, d);
                u_fun_exit(0, -2);
                return (ret);
            }

            friend Tensor operator >= (const char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >= (const unsigned char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >= (const short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >= (const unsigned short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >= (const int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >= (const unsigned int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >= (const long &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >= (const size_t &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >= (const float &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >= (const double &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) >= rhs);
            }

            Tensor operator <(const Tensor &d) const {
                u_fun_enter(2, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" != operation can only be applied to tensor have same shape. given (%s, %s)", shape_.c_str(), d.shape().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("nequal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::Less>(ret, *this, d);
                u_fun_exit(0, -2);
                return (ret);
            }

            friend Tensor operator <(const char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const unsigned char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const unsigned short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const unsigned int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const long &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const size_t &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const float &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const double &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) < rhs);
            }

            Tensor operator <=(const Tensor &d) const {
                u_fun_enter(2, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" != operation can only be applied to tensor have same shape. given (%s, %s)", shape_.c_str(), d.shape().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("nequal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::LessEqual>(ret, *this, d);
                u_fun_exit(0, -2);
                return (ret);
            }

            friend Tensor operator <=(const char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const unsigned char &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const unsigned short &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const unsigned int &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const long &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const size_t &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const float &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const double &lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) <= rhs);
            }

            Tensor operator ++() {
                u_fun_enter(2, 0);
                Tensor one(1);
                op::run3<Tensor, op::Add>(*this, *this, one);
                u_fun_exit(0, -2);
                return (*this);
            }

            Tensor operator ++(int) {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                ++*this;
                u_fun_exit(0, -2);
                return (ret);
            }

            Tensor operator --() {
                u_fun_enter(2, 0);
                Tensor one(1);
                op::run3<Tensor, op::Subtract>(*this, *this, one);
                u_fun_exit(0, -2);
                return (*this);
            }

            Tensor operator --(int) {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                --*this;
                u_fun_exit(0, -2);
                return (ret);
            }

            void operator +=(const Tensor &d) {
                u_fun_enter(2, 0);
                op::run3<Tensor, op::Add>(*this, *this, d);
                u_fun_exit(0, -2);
            }

            void operator -=(const Tensor &d) {
                u_fun_enter(2, 0);
                op::run3<Tensor, op::Subtract>(*this, *this, d);
                u_fun_exit(0, -2);
            }

            void operator *=(const Tensor &d) {
                u_fun_enter(2, 0);
                op::run3<Tensor, op::Multiply>(*this, *this, d);
                u_fun_exit(0, -2);
            }

            void operator /=(const Tensor &d) {
                u_fun_enter(2, 0);
                op::run3<Tensor, op::Divide>(*this, *this, d);
                u_fun_exit(0, -2);
            }

            void operator ^=(const Tensor &d) {
                u_fun_enter(2, 0);
                op::run3<Tensor, op::Pow>(*this, *this, d);
                u_fun_exit(0, -2);
            }

            friend Tensor operator +(short lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(unsigned short lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(int lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(unsigned int lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(long lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(size_t lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(float lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(double lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) + rhs);
            }

            Tensor operator +(const Tensor &d) const {
                u_fun_enter(2, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" == operation can only be applied to tensor have same shape. given (%s, %s)", shape_.c_str(), d.shape().c_str()));
                Tensor ret(shape_.broadcast(d.shape()), std::max(type_, d.type()), false);
                op::run3<Tensor, op::Add>(ret, *this, d);
                u_fun_exit(0, -2);
                return (ret);
            }

            friend Tensor operator -(short lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(unsigned short lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(int lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(unsigned int lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(long lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(size_t lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(float lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(double lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) - rhs);
            }

            Tensor operator -(const Tensor &d) const {
                u_fun_enter(2, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" == operation can only be applied to tensor have same shape. given (%s, %s)", shape_.c_str(), d.shape().c_str()));
                Tensor ret(shape_.broadcast(d.shape()), std::max(type_, d.type()), false);
                op::run3<Tensor, op::Subtract>(ret, *this, d);
                u_fun_exit(0, -2);
                return (ret);
            }

            // positive -> negative, or
            // negative -> positive
            Tensor operator -() const {
                u_fun_enter(2, 0);
                u_assert(! is_unsigned(type_), u::format("cannot apply minus operation on unsigned type. given `%s`", dtype_str(type_).c_str()));
                Tensor ret(shape_, type_, false);
                op::run2<Tensor, op::Minus>(ret, *this);
                u_fun_exit(0, -2);
                return (ret);
            }

            friend Tensor operator *(short lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(unsigned short lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(int lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(unsigned int lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(long lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(size_t lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(float lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(double lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) * rhs);
            }

            Tensor operator *(const Tensor &d) const {
                u_fun_enter(2, 0);
                Tensor ret(shape_.broadcast(d.shape()), std::max(type_, d.type()), false);
                op::run3<Tensor, op::Multiply>(ret, *this, d);
                u_fun_exit(0, -2);
                return (ret);
            }

            friend Tensor operator /(short lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(unsigned short lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(int lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(unsigned int lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(long lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(size_t lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(float lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(double lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) / rhs);
            }

            Tensor operator /(const Tensor &d) const {
                u_fun_enter(2, 0);
                Tensor ret(shape_.broadcast(d.shape()), std::max(type_, d.type()), false);
                op::run3<Tensor, op::Divide>(ret, *this, d);
                u_fun_exit(0, -2);
                return (ret);
            }

            friend Tensor operator ^(short lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(unsigned short lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(int lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(unsigned int lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(long lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(size_t lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(float lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(double lhs, const Tensor &rhs) {
                u_fun_here(0, 0);
                return (Tensor(lhs) ^ rhs);
            }

            Tensor operator ^(const Tensor &d) const {
                u_fun_enter(2, 0);
                Tensor ret(shape_.broadcast(d.shape()), std::max(type_, d.type()), false);
                op::run3<Tensor, op::Pow>(ret, *this, d);
                u_fun_exit(0, -2);
                return (ret);
            }

            Tensor sqrt() const {
                u_fun_enter(2, 0);
                Tensor t(shape_, DType::float64);
                op::run2<Tensor, op::SquareRoot>(t, *this);
                u_fun_exit(0, -2);
                return t;
            }

            Tensor sin() const {
                u_fun_enter(2, 0);
                Tensor t(shape_, DType::float64);
                op::run2<Tensor, op::Sine>(t, *this);
                u_fun_exit(0, -2);
                return t;
            }

            Tensor cos() const {
                u_fun_enter(2, 0);
                Tensor t(shape_, DType::float64);
                op::run2<Tensor, op::Cosine>(t, *this);
                u_fun_exit(0, -2);
                return t;
            }

            Tensor tan() const {
                u_fun_enter(2, 0);
                Tensor t(shape_, DType::float64);
                op::run2<Tensor, op::Tangent>(t, *this);
                u_fun_exit(0, -2);
                return t;
            }

            Tensor tanh() const {
                u_fun_enter(2, 0);
                Tensor t(shape_, DType::float64);
                op::run2<Tensor, op::HyperbolicTangent>(t, *this);
                u_fun_exit(0, -2);
                return t;
            }

            Tensor round() const {
                u_fun_enter(2, 0);
                Tensor t(shape_, type_);
                op::run2<Tensor, op::Round>(t, *this);
                u_fun_exit(0, -2);
                return t;
            }

            Tensor ceil() const {
                u_fun_enter(2, 0);
                Tensor t(shape_, type_);
                op::run2<Tensor, op::Ceil>(t, *this);
                u_fun_exit(0, -2);
                return t;
            }

            Tensor floor() const {
                u_fun_enter(2, 0);
                Tensor t(shape_, type_);
                op::run2<Tensor, op::Floor>(t, *this);
                u_fun_exit(0, -2);
                return t;
            }

            Tensor isinf() const {
                u_fun_enter(2, 0);
                Tensor t(shape_, DType::uint8);
                op::run2<Tensor, op::IsInfinite>(t, *this);
                u_fun_exit(0, -2);
                return t;
            }

            Tensor isnan() const {
                u_fun_enter(2, 0);
                Tensor t(shape_, DType::uint8);
                op::run2<Tensor, op::IsNaN>(t, *this);
                u_fun_exit(0, -2);
                return t;
            }

            Tensor isfinite() const {
                u_fun_enter(2, 0);
                Tensor t(shape_, DType::uint8);
                op::run2<Tensor, op::IsFinite>(t, *this);
                u_fun_exit(0, -2);
                return t;
            }

            // date type cast
            // if same type, copy no data, return tensor description **ONLY**
            Tensor astype(DType type) const {
                u_fun_enter(2, 0);
                u_assert(data_.get()!= nullptr, "data empty. memory not allocated.");
                Tensor ret(shape_, type, false);
                op::run2<Tensor, op::Assign>(ret, *this);
                u_fun_exit(0, -2);
                return (ret);
            }

            template<typename T>
            Tensor astype() const {
                u_fun_here(0, 0);
                return astype(ctype<T>());
            }

            template<typename T>
            const T* cast() const {
                u_fun_enter(2, 0);
                u_assert(ctype<T>() == type_, u::format("cast to different data type not supported."));
                u_fun_exit(0, -2);
                return reinterpret_cast<T*>(data_.get());
            }

            // flatten change tensor description **ONLY** if copy is false
            void flatten_inplace(int beg_axis=0, int end_axis=-1) {
                u_fun_enter(2, 0);
                size_t beg = shape_.axis_normalize(beg_axis);
                size_t end = shape_.axis_normalize(end_axis);
                std::vector<size_t> shape;
                for (size_t i=0; i<shape_.size(); ++i) {
                    if (i <= beg) {
                        shape.push_back(shape_[i]);
                    } else if (i >= end) {
                        shape.push_back(shape_[i]);
                    } else {
                        shape.back() *= shape_[i];
                    }
                }
                shape_.reshape(shape);
                u_fun_exit(0, -2);
            }

            Tensor flatten(int beg_axis=0, int end_axis=-1) const {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                ret.flatten(beg_axis, end_axis);
                u_fun_exit(0, -2);
                return ret;
            }

            void reshape_inplace(const Shape& shape) {
                u_fun_here(0, 0);
                shape_.reshape(shape);
            }

            void reshape_inplace(const std::vector<int> &shape) {
                u_fun_here(0, 0);
                shape_.reshape(shape);
            }

            Tensor reshape(const Shape& shape) const {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                u_fun_exit(0, -2);
                return ret.reshape(shape);
            }

            Tensor reshape(const std::vector<int> &shape) const {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                u_fun_exit(0, -2);
                return ret.reshape(shape);
            }

            void squeeze_inplace() {
                u_fun_enter(2, 0);
                Shape::iterator it = shape_.begin();
                while (it != shape_.end()) {
                    if (*it == 1) {
                        if (shape_.size() > 1) {
                            it = shape_.erase(it);
                        }
                    } else {
                        ++it;
                    }
                }
                u_fun_exit(0, -2);
            }

            Tensor squeeze() const {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                ret.squeeze();
                u_fun_exit(0, -2);
                return ret;
            }

            void expand_dims_inplace(const int axis) {
                u_fun_enter(2, 0);
                size_t _axis_ = shape_.axis_normalize(axis);
                shape_.insert(shape_.begin() + _axis_, 1);
                u_fun_exit(0, -2);
            }

            Tensor expand_dims(const int axis) const {
                u_fun_enter(2, 0);
                Tensor ret(*this);
                ret.expand_dims(axis);
                u_fun_exit(0, -2);
                return ret;
            }

            Tensor zeros_like(const DType type=DType::invalid) const {
                u_fun_enter(2, 0);
                DType _type_ = (type == DType::invalid ? type_ : type);
                u_fun_exit(0, -2);
                return Tensor(shape_, _type_);
            }

            Tensor ones_like(const DType type=DType::invalid) const {
                u_fun_enter(2, 0);
                DType _type_ = (type == DType::invalid ? type_ : type);
                Tensor t(shape_, _type_);
                t += 1;
                u_fun_exit(0, -2);
                return t;
            }

            inline Tensor sum(const int axis = u::tensor::all, const DType type = DType::float32, bool keepdims=false) const {
                u_fun_here(0, 0);
                return dimension_op_run_<op::Sum>(axis, type, keepdims);
            }

            inline Tensor mean(const int axis = u::tensor::all, const DType type = DType::float32, bool keepdims=false) const {
                u_fun_here(0, 0);
                return dimension_op_run_<op::Mean>(axis, type, keepdims);
            }

            inline Tensor stddev(const int axis = u::tensor::all, const DType type = DType::float32, bool keepdims=false) const {
                u_fun_here(0, 0);
                return dimension_op_run_<op::StdDev>(axis, type, keepdims);
            }

            inline Tensor max(const int axis = u::tensor::all, const DType type = DType::invalid, bool keepdims=false) const {
                u_fun_here(0, 0);
                return dimension_op_run_<op::Max>(axis, type, keepdims);
            }

            inline Tensor max(const Tensor &t) const {
                u_fun_enter(2, 0);
                Tensor ret(shape_.broadcast(t.shape()), std::max(type_, t.type()), false);
                op::run3<Tensor, op::Maximum>(ret, *this, t);
                u_fun_exit(0, -2);
                return ret;
            }

            inline Tensor min(const int axis = u::tensor::all, const DType type = DType::invalid, bool keepdims=false) const {
                u_fun_here(0, 0);
                return dimension_op_run_<op::Min>(axis, type, keepdims);
            }

            inline Tensor min(const Tensor &t) const {
                u_fun_enter(2, 0);
                Tensor ret(shape_.broadcast(t.shape()), std::max(type_, t.type()), false);
                op::run3<Tensor, op::Minimum>(ret, *this, t);
                u_fun_exit(0, -2);
                return ret;
            }

            inline Tensor argmax(const int axis = u::tensor::all, const DType type = DType::uint32, bool keepdims=false) const {
                u_fun_here(0, 0);
                return dimension_op_run_<op::ArgMax>(axis, type, keepdims);
            }

            inline Tensor argmin(const int axis = u::tensor::all, const DType type = DType::uint32, bool keepdims=false) const {
                u_fun_here(0, 0);
                return dimension_op_run_<op::ArgMin>(axis, type, keepdims);
            }

            Tensor transpose(const std::vector<int> &dim_changes) const {
                u_fun_enter(2, 0);
                u_assert(shape_.size() == dim_changes.size(), u::format("dimensions not match (%zu vs %zu)", shape_.size(), dim_changes.size()));
                std::vector<size_t> norm_dims = shape_.axis_normalize(dim_changes);

                std::map<size_t, size_t> dim_map;
                std::vector<size_t>::iterator it = norm_dims.end();
                Shape _shape_=Shape::create(shape_.size());
                for (size_t i = 0; i < shape_.size(); ++i) {
                    u_assert(norm_dims[i] < shape_.size(), u::format("dimension overflow (%zu [%d] vs %zu)", norm_dims[i], dim_changes[i], shape_.size()));
                    _shape_[i] = shape_[norm_dims[i]];
                    it = std::find(norm_dims.begin(), norm_dims.end(), i);
                    u_assert(it != norm_dims.end(), u::format("cannot find corresponding dimension for %zu-th dimension", i));
                    dim_map[i] = static_cast<unsigned int>(std::distance(norm_dims.begin(), it));
                }

                Tensor ret(_shape_, type_, false);
                op::run2<Tensor, op::Transpose>(ret, *this, norm_dims, dim_map);
                u_fun_exit(0, -2);
                return (ret);
            }

            static bool any(const Tensor &t, bool positive=true) {
                u_fun_enter(2, 0);
                bool ans = false;
                Tensor tans(static_cast<unsigned char>(1));
                op::run2<Tensor, op::Any>(tans, t, positive);
                ans = (*tans.cast<unsigned char>() == 1);
                u_fun_exit(0, -2);
                return ans;
            }

            static bool all(const Tensor &t, bool positive=true) {
                u_fun_enter(2, 0);
                bool ans = false;
                Tensor tans(static_cast<unsigned char>(1));
                op::run2<Tensor, op::All>(tans, t, positive);
                ans = (*tans.cast<unsigned char>() == 1);
                u_fun_exit(0, -2);
                return ans;
            }

            std::vector<Tensor> split(const std::vector<size_t> &splits, int axis) const {
                u_fun_enter(2, 0);
                size_t naxis = shape_.axis_normalize(axis);
                size_t sum = 0;
                std::vector<Shape> shapes(splits.size());
                for (size_t i=0; i<splits.size(); ++i) {
                    sum += splits[i];
                    shapes[i] = shape_;
                    shapes[i][naxis] = splits[i];
                }
                u_assert(sum == shape_[naxis], u::format("splits and exist size of axis %zu[%d] not match. given (%zu vs %zu)", naxis, axis, sum, shape_[naxis]));
                std::vector<Tensor> tensors;
                for (size_t t=0; t<splits.size(); ++t) {
                    tensors.push_back(Tensor(shapes[t], type_));
                }
                op::runv<Tensor, op::Split>(tensors, *this, shapes, naxis);
                u_fun_exit(0, -2);
                return tensors;
            }

            Tensor tile(const std::vector<int> axis, const std::vector<size_t> &repeats) {
                u_fun_enter(2, 0);
                u_assert(axis.size() == repeats.size(), u::format("repeats should specify each axis for tile. need [%zu] axes but given [%zu]", axis.size(), repeats.size()));
                std::vector<size_t> naxis = shape_.axis_normalize(axis);
                std::vector<size_t> _repeats_(shape_.rank(), 1);
                for (size_t i=0; i<naxis.size(); ++i) {
                    u_assert(naxis[i] < shape_.rank(), u::format("%zu-th axis: %zu [%d] out of range", i, naxis[i], axis[i]));
                    _repeats_[naxis[i]] = repeats[i];
                }
                u_fun_exit(0, -2);
                return tile(_repeats_);
            }

            Tensor tile(const std::vector<size_t> &repeats) {
                u_fun_enter(2, 0);
                // tile can be considered as broadcast by treat repeats as reshape
                u_assert(shape_.rank() == repeats.size(), u::format("repeats should specify each axis for tile. need [%zu] axes but given [%zu]", shape_.rank(), repeats.size()));
                Shape repeated_shape = Shape::create(shape_.rank());
                Shape broadcasted_shape = Shape::create(shape_.rank());
                Shape temp_shape = shape_;
                for (int i=temp_shape.rank()-1; i>=0; --i) {
                    repeated_shape[i] = temp_shape[i];
                    broadcasted_shape[i] = temp_shape[i];
                    if (repeats[i] > 1) {
                        repeated_shape[i] *= repeats[i];
                        if (broadcasted_shape[i] != 1) {
                            broadcasted_shape.insert(broadcasted_shape.begin()+i, repeats[i]);
                            shape_.insert(shape_.begin()+i, 1);
                        } else {
                            broadcasted_shape[i] *= repeats[i];
                        }
                    }
                }
                Tensor ans;
                if (shape_.size() == temp_shape.size()) {
                    ans = broadcast(broadcasted_shape);
                } else {
                    ans = broadcast(broadcasted_shape);
                    ans.reshape_inplace(repeated_shape);
                    shape_ = temp_shape;
                }

                // // BEGIN Experimental
                // Shape shape(shape_);
                // for (size_t i=0; i<shape_.size(); ++i) {
                //     if (repeats[i] > 1) {
                //         shape[i] *= repeats[i];
                //     }
                // }
                // std::cout << shape_ << " ==> " << shape << std::endl;
                // Tensor ans(shape, type_);
                // op::run2<Tensor, op::Broadcast>(ans, *this);
                // // END Experimental

                u_fun_exit(0, -2);
                return ans;
            }

            Tensor broadcast(const Shape &shape) const {
                u_fun_enter(2, 0);
                Tensor ans;
                if (shape_ == shape) {
                    u::log::warning("It seems you try to broadcast tensor to the same shape. Tensor will ignore this operation and return `*this`");
                    ans = *this;
                } else {
                    u_assert(shape_.rank() == shape.rank(), u::format("cannot broadcast from '%s' to '%s'", shape_.c_str(), shape.c_str()));
                    for (int i=static_cast<int>(shape_.rank()-1); i>=0; --i) {
                        u_assert(shape_[i] == shape[i] || (shape_[i] == 1), u::format("Tensor with shape %s cannot broadcast to Tensor with shape %s", shape_.c_str(), shape.c_str()));
                    }
                    ans(shape, type_, false);
                    op::run2<Tensor, op::Broadcast>(ans, *this);
                }
                return ans;
                u_fun_exit(0, -2);
            }
        };

        Tensor concatenate(const std::vector<Tensor> &tensors, int axis) {
            u_fun_enter(2, 0);
            Shape shape;
            DType type = tensors[0].type();
            { // get newshape
                std::vector<Shape> shapes;
                for (size_t i=0; i<tensors.size(); ++i) {
                    shapes.push_back(tensors[i].shape());
                    u_assert(type == tensors[i].type(), u::format("%zu-th and %zu-th have different data type. given (%s vs %s)", 0, i, dtype_cstr(type), dtype_cstr(tensors[i].type())));
                }
                shape = Shape::concatenate(shapes, axis);
            }
            Tensor ans(shape, type);
            op::runv<Tensor, op::Concatenate>(ans, tensors, axis);
            u_fun_exit(0, -2);
            return ans;
        }

        std::vector<Tensor> split(const Tensor &tensor, const std::vector<size_t> &splits, int axis) {
            u_fun_here(0, 0);
            return tensor.split(splits, axis);
        }

        Tensor zeros(const Shape &shape, const DType type) {
            u_fun_here(0, 0);
            return (Tensor(shape, type, false));
        }

        template <typename T=int>
        Tensor fill(const Shape &shape, const DType type, T value) {
            u_fun_enter(2, 0);
            Tensor t(shape, type, false);
            t += value;
            u_fun_exit(0, -2);
            return (t);
        }

        Tensor ones(const Shape &shape, const DType type) {
            u_fun_enter(2, 0);
            Tensor t(shape, type, false);
            t += 1;
            u_fun_exit(0, -2);
            return (t);
        }
    }
}

#endif
