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

#include <vector>

namespace u {

    namespace tensor {
        class Tensor{
        protected:
            template <typename T>
            void _init(const T *data, size_t size=1){
                malloc();
                mm::upload(data_.get(), reinterpret_cast<const unsigned char * const >(data), sizeof(T)*size);
            }

        private:
            std::shared_ptr<unsigned char> data_;
            DType type_;
            Shape shape_;

            size_t print_(size_t rank, size_t begin, std::ostream &os) {
                size_t end = begin;
                if (shape_.rank() == 0) {
                    op::run<Tensor, op::Print>(*this, begin, end, os);
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
                        op::run<Tensor, op::Print>(*this, begin, end, os);
                    }
                    if (rank == 0) {
                        os << "]";
                    }
                }
                return (end);
            }

            template <template<typename, typename > class T>
            Tensor dimension_op_run_(const int axis, const DType type, bool keepdims) {
                u_fun_enter(0, 0);
                Shape shape = shape_.get_shape_dimension(axis, keepdims);
                Tensor ret(shape, (type == DType::invalid ? type_ : type), false);
                op::run2<Tensor, T>(ret, *this, axis);
                u_fun_exit(0, 0);
                return ret;
            }

        public:

            virtual ~Tensor() {}

            Tensor(): data_(nullptr), type_(DType::invalid), shape_() {}

            Tensor(DType type) : data_(nullptr), type_(type), shape_() {u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");}

            Tensor(const Shape& shape, DType type, bool lazy = false) : shape_(shape), type_(type) {
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, u::format("size must be greater than 0. given %zu", v));
                data_ = nullptr;
                if (!lazy) {
                    malloc();
                }
            }

            Tensor(const std::vector<size_t> &shape, DType type, bool lazy = false) : shape_(shape), type_(type) {
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, u::format("size must be greater than 0. given %zu", v));
                data_ = nullptr;
                if (!lazy) {
                    malloc();
                }
            }

            Tensor(unsigned char * data, const std::vector<size_t> &shape, DType type, bool copy = false) : shape_(shape), type_(type) {
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
            }

            Tensor(unsigned char * data, const Shape& shape, DType type, bool copy = false) : shape_(shape), type_(type) {
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, u::format("size must be greater than 0. given %zu", v));
                u_assert(data != nullptr, "data can not be null pointer");
                if (copy) {
                    _init<unsigned char>(data, bytesize());
                } else {
                    data_.reset(data, mm::no_free);
                }
            }

            Tensor(const unsigned char * const data, const Shape &shape, DType type) : shape_(shape), type_(type) {
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, u::format("size must be greater than 0. given %zu", v));
                u_assert(data != nullptr, "data can not be null pointer");
                _init<unsigned char>(data, bytesize());
            }

            Tensor(const unsigned char * const data, const std::vector<size_t> &shape, DType type) : shape_(shape), type_(type) {
                u_assert(type < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, "no allowing copy tensor with 0 size.");
                u_assert(data != nullptr, "data can not be null pointer");
                _init<unsigned char>(data, bytesize());
            }

            // invoked in the following case
            // Tensor t1(t2);
            Tensor(const Tensor &t) : type_(t.type()), shape_(t.shape()) {
                u_assert(type_ < DType::invalid, "no allowing explicitly constrcuting invalid type of tensor");
                size_t v = volume();
                u_assert(v > 0, u::format("size must be greater than 0. given %zu", v));
                u_assert(t.cref() != nullptr, "data can not be null pointer");
                _init<unsigned char>(t.cref(), bytesize());
            }

            Tensor(const char& data) : type_(DType::int8), shape_() {_init<char>(&data);}

            Tensor(const unsigned char& data) : type_(DType::uint8), shape_() {_init<unsigned char>(&data);}

            Tensor(const short& data) : type_(DType::int16), shape_() {_init<short>(&data);}

            Tensor(const unsigned short& data) : type_(DType::uint16), shape_() {_init<unsigned short>(&data);}

            Tensor(const int& data) : type_(DType::int32), shape_() {_init<int>(&data);}

            Tensor(const unsigned int& data) : type_(DType::uint32), shape_() {_init<unsigned int>(&data);}

            Tensor(const long& data) : type_(DType::int64), shape_() {_init<long>(&data);}

            Tensor(const size_t& data) : type_(DType::uint64), shape_() {_init<size_t>(&data);}

            Tensor(const float& data) : type_(DType::float32), shape_() {_init<float>(&data);}

            Tensor(const double& data) : type_(DType::float64), shape_() {_init<double>(&data);}

            friend std::ostream & operator <<(std::ostream &os, const Tensor &t) {
                const_cast<Tensor&>(t).print_(0, 0, os);
                return (os);
            }

            std::string str() {
                std::ostringstream oss;
                oss << *this;
                return oss.str();
            }

            void malloc(const Shape &shape, DType type) {
                u_assert(data_ == nullptr, "data not empty. memory may be already allocated. use `realloc' instead");
                if (data_ != nullptr) {
                    data_.reset(mm::malloc(shape.volume(), dtype_size(type)), mm::mfree);
                }
            }

            void realloc(const Shape &shape) {data_.reset(mm::malloc(shape.volume(), dtype_size(type_)), mm::mfree);}

            void malloc(DType type) {
                u_assert(data_ == nullptr, "data not empty. memory may be already allocated.");
                if (data_ != nullptr) {
                    data_.reset(mm::malloc(volume(), dtype_size(type)), mm::mfree);
                }
            }

            void malloc() {
                u_assert(data_.get() == nullptr, "data already allocated.");
                data_.reset(mm::malloc(volume(), dtype_size(type_)), mm::mfree);
            }

            const unsigned char *cref() const {return data_.get();}

            unsigned char *ref() {return data_.get();}

            DType type() {return type_;}

            const DType type() const {return type_;}

            // rank
            size_t rank() {return (shape_.size());}

            const size_t rank() const {return (shape_.size());}

            size_t volume() {return shape_.volume<size_t>();}

            const size_t volume() const {return shape_.volume<size_t>();}

            size_t bytesize() {return (volume() * dtype_size(type_));}

            const size_t bytesize() const {return (volume() * dtype_size(type_));}

            // shape of dimension
            Shape shape() {return (shape_);}

            const Shape shape() const {return (shape_);}

            // clip in inplace mode
            void clip_inplace(double min, double max) {
                u_fun_enter(0, 0);
                op::run2<Tensor, op::Clip>(*this, *this, min, max);
                u_fun_exit(0, 0);
            }

            Tensor clip(double min, double max) {
                u_fun_enter(0, 0);
                Tensor ret(*this);
                ret.clip_inplace(min, max);
                u_fun_exit(0, 0);
                return ret;
            }

            void abs_inplace() {
                u_fun_enter(0, 0);
                op::run2<Tensor, op::Absolute>(*this, *this);
                u_fun_exit(0, 0);
            }

            Tensor abs() {
                u_fun_enter(0, 0);
                Tensor ret(*this);
                ret.abs();
                u_fun_exit(0, 0);
                return ret;
            }

            void exp_inplace() {
                u_fun_enter(0, 0);
                op::run2<Tensor, op::Experiential >(*this, *this);
                u_fun_exit(0, 0);
            }

            Tensor exp() {
                u_fun_enter(0, 0);
                Tensor ret(*this);
                ret.exp();
                u_fun_exit(0, 0);
                return ret;
            }

            void log_inplace() {
                u_fun_enter(0, 0);
                op::run2<Tensor, op::Log >(*this, *this);
                u_fun_exit(0, 0);
            }

            Tensor log() {
                u_fun_enter(0, 0);
                Tensor ret(*this);
                ret.log();
                u_fun_exit(0, 0);
                return ret;
            }

            void log10_inplace() {
                u_fun_enter(0, 0);
                op::run2<Tensor, op::Log10>(*this, *this);
                u_fun_exit(0, 0);
            }

            Tensor log10() {
                u_fun_enter(0, 0);
                Tensor ret(*this);
                ret.log10();
                u_fun_exit(0, 0);
                return ret;
            }

            // non-copy assign assignment
            Tensor operator =(Tensor &d) {
                u_fun_enter(0, 0);
                type_ = d.type();
                shape_ = d.shape();
                data_.reset(d.ref(), mm::no_free);
                u_fun_exit(0, 0);
                return (*this);
            }

            // copy assignment object assignment
            Tensor &operator ()(const Tensor &d) {
                u_fun_enter(0, 0);
                if (type_ == DType::invalid) {
                    // assign to no initialized tensor       ^
                    type_ = d.type();
                    shape_ = d.shape();
                    _init(d.cref(), d.volume());
                } else {
                    Shape broadcasted = shape_.broadcast(d.shape());
                    u_assert(broadcasted == shape_, u::format("shape not match even after broadcast (%s vs %s)", broadcasted.str().c_str(), shape_.str().c_str()));
                    op::run2<Tensor, op::Assign>(*this, d);
                }
                u_fun_exit(0, 0);
                return (*this);
            }

            Tensor &operator ()(const Shape &shape, const DType type, bool lazy) {
                u_fun_enter(0, 0);
                if (type_ != DType::invalid) {
                    // assign to no initialized tensor
                    type_ = type;
                }
                data_ = nullptr;
                if (!lazy) {
                    malloc();
                }
                u_fun_exit(0, 0);
                return (*this);
            }

            Tensor operator ==(const Tensor &d) {
                u_fun_enter(0, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" == operation can only be applied to tensor have same shape. given (%s, %s)", shape_.str().c_str(), d.shape().str().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("equal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::Equal>(ret, *this, d);
                u_fun_exit(0, 0);
                return (ret);
            }

            friend Tensor operator ==(const char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const unsigned char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const unsigned short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const unsigned int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const long &lhs, const Tensor &rhs) {
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const size_t &lhs, const Tensor &rhs) {
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const float &lhs, const Tensor &rhs) {
                return (Tensor(lhs) == rhs);
            }

            friend Tensor operator ==(const double &lhs, const Tensor &rhs) {
                return (Tensor(lhs) == rhs);
            }

            Tensor operator !=(const Tensor &d) {
                u_fun_enter(0, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" != operation can only be applied to tensor have same shape. given (%s, %s)", shape_.str().c_str(), d.shape().str().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("nequal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::NotEqual>(ret, *this, d);
                u_fun_exit(0, 0);
                return (ret);
            }

            friend Tensor operator !=(const char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const unsigned char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const unsigned short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const unsigned int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const long &lhs, const Tensor &rhs) {
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const size_t &lhs, const Tensor &rhs) {
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const float &lhs, const Tensor &rhs) {
                return (Tensor(lhs) != rhs);
            }

            friend Tensor operator !=(const double &lhs, const Tensor &rhs) {
                return (Tensor(lhs) != rhs);
            }

            Tensor operator >(const Tensor &d) {
                u_fun_enter(0, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" != operation can only be applied to tensor have same shape. given (%s, %s)", shape_.str().c_str(), d.shape().str().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("nequal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::Greater>(ret, *this, d);
                u_fun_exit(0, 0);
                return (ret);
            }

            friend Tensor operator >(const char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator >(const unsigned char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator >(const short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator >(const unsigned short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator >(const int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator >(const unsigned int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator >(const long &lhs, const Tensor &rhs) {
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator >(const size_t &lhs, const Tensor &rhs) {
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator >(const float &lhs, const Tensor &rhs) {
                return (Tensor(lhs) > rhs);
            }

            friend Tensor operator >(const double &lhs, const Tensor &rhs) {
                return (Tensor(lhs) > rhs);
            }

            Tensor operator >=(const Tensor &d) {
                u_fun_enter(0, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" != operation can only be applied to tensor have same shape. given (%s, %s)", shape_.str().c_str(), d.shape().str().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("nequal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::GreaterEqual>(ret, *this, d);
                u_fun_exit(0, 0);
                return (ret);
            }

            friend Tensor operator >=(const char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >=(const unsigned char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >=(const short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >=(const unsigned short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >=(const int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >=(const unsigned int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >=(const long &lhs, const Tensor &rhs) {
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >=(const size_t &lhs, const Tensor &rhs) {
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >=(const float &lhs, const Tensor &rhs) {
                return (Tensor(lhs) >= rhs);
            }

            friend Tensor operator >=(const double &lhs, const Tensor &rhs) {
                return (Tensor(lhs) >= rhs);
            }

            Tensor operator <(const Tensor &d) {
                u_fun_enter(0, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" != operation can only be applied to tensor have same shape. given (%s, %s)", shape_.str().c_str(), d.shape().str().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("nequal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::Less>(ret, *this, d);
                u_fun_exit(0, 0);
                return (ret);
            }

            friend Tensor operator <(const char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const unsigned char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const unsigned short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const unsigned int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const long &lhs, const Tensor &rhs) {
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const size_t &lhs, const Tensor &rhs) {
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const float &lhs, const Tensor &rhs) {
                return (Tensor(lhs) < rhs);
            }

            friend Tensor operator <(const double &lhs, const Tensor &rhs) {
                return (Tensor(lhs) < rhs);
            }

            Tensor operator <=(const Tensor &d) {
                u_fun_enter(0, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" != operation can only be applied to tensor have same shape. given (%s, %s)", shape_.str().c_str(), d.shape().str().c_str()));
                if (is_float(type_) || is_float(d.type())) {
                    u::log::warning("nequal operation between float number is not recomamnded");
                }
                Tensor ret(shape_.broadcast(d.shape()), DType::uint8, false);
                op::run3<Tensor, op::LessEqual>(ret, *this, d);
                u_fun_exit(0, 0);
                return (ret);
            }

            friend Tensor operator <=(const char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const unsigned char &lhs, const Tensor &rhs) {
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const unsigned short &lhs, const Tensor &rhs) {
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const unsigned int &lhs, const Tensor &rhs) {
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const long &lhs, const Tensor &rhs) {
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const size_t &lhs, const Tensor &rhs) {
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const float &lhs, const Tensor &rhs) {
                return (Tensor(lhs) <= rhs);
            }

            friend Tensor operator <=(const double &lhs, const Tensor &rhs) {
                return (Tensor(lhs) <= rhs);
            }

            Tensor operator ++() {
                u_fun_enter(0, 0);
                Tensor one(1);
                op::run3<Tensor, op::Add>(*this, *this, one);
                u_fun_exit(0, 0);
                return (*this);
            }

            Tensor operator ++(int) {
                u_fun_enter(0, 0);
                Tensor ret(*this);
                ++*this;
                u_fun_exit(0, 0);
                return (ret);
            }

            Tensor operator --() {
                u_fun_enter(0, 0);
                Tensor one(1);
                op::run3<Tensor, op::Subtract>(*this, *this, one);
                u_fun_exit(0, 0);
                return (*this);
            }

            Tensor operator --(int) {
                u_fun_enter(0, 0);
                Tensor ret(*this);
                --*this;
                u_fun_exit(0, 0);
                return (ret);
            }

            void operator +=(const Tensor &d) {
                op::run3<Tensor, op::Add>(*this, *this, d);
            }

            void operator -=(const Tensor &d) {
                op::run3<Tensor, op::Subtract>(*this, *this, d);
            }

            void operator *=(const Tensor &d) {
                op::run3<Tensor, op::Multiply>(*this, *this, d);
            }

            void operator /=(const Tensor &d) {
                op::run3<Tensor, op::Divide>(*this, *this, d);
            }

            void operator ^=(const Tensor &d) {
                op::run3<Tensor, op::Pow>(*this, *this, d);
            }

            friend Tensor operator +(short lhs, const Tensor &rhs) {
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(unsigned short lhs, const Tensor &rhs) {
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(int lhs, const Tensor &rhs) {
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(unsigned int lhs, const Tensor &rhs) {
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(long lhs, const Tensor &rhs) {
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(size_t lhs, const Tensor &rhs) {
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(float lhs, const Tensor &rhs) {
                return (Tensor(lhs) + rhs);
            }

            friend Tensor operator +(double lhs, const Tensor &rhs) {
                return (Tensor(lhs) + rhs);
            }

            Tensor operator +(const Tensor &d) {
                u_fun_enter(0, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" == operation can only be applied to tensor have same shape. given (%s, %s)", shape_.str().c_str(), d.shape().str().c_str()));
                Tensor ret(shape_.broadcast(d.shape()), std::max(type_, d.type()), false);
                op::run3<Tensor, op::Add>(ret, *this, d);
                u_fun_exit(0, 0);
                return (ret);
            }

            friend Tensor operator -(short lhs, const Tensor &rhs) {
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(unsigned short lhs, const Tensor &rhs) {
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(int lhs, const Tensor &rhs) {
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(unsigned int lhs, const Tensor &rhs) {
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(long lhs, const Tensor &rhs) {
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(size_t lhs, const Tensor &rhs) {
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(float lhs, const Tensor &rhs) {
                return (Tensor(lhs) - rhs);
            }

            friend Tensor operator -(double lhs, const Tensor &rhs) {
                return (Tensor(lhs) - rhs);
            }

            Tensor operator -(const Tensor &d) {
                u_fun_enter(0, 0);
                u_assert(shape_.broadcastable(d.shape()), u::format(" == operation can only be applied to tensor have same shape. given (%s, %s)", shape_.str().c_str(), d.shape().str().c_str()));
                Tensor ret(shape_.broadcast(d.shape()), std::max(type_, d.type()), false);
                op::run3<Tensor, op::Subtract>(ret, *this, d);
                u_fun_exit(0, 0);
                return (ret);
            }

            // positive -> negative, or
            // negative -> positive
            Tensor operator -() {
                u_fun_enter(0, 0);
                u_assert(! is_unsigned(type_), u::format("cannot apply minus operation on unsigned type. given `%s`", dtype_str(type_).c_str()));
                Tensor ret(shape_, type_, false);
                op::run2<Tensor, op::Minus>(ret, *this);
                u_fun_exit(0, 0);
                return (ret);
            }

            friend Tensor operator *(short lhs, const Tensor &rhs) {
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(unsigned short lhs, const Tensor &rhs) {
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(int lhs, const Tensor &rhs) {
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(unsigned int lhs, const Tensor &rhs) {
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(long lhs, const Tensor &rhs) {
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(size_t lhs, const Tensor &rhs) {
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(float lhs, const Tensor &rhs) {
                return (Tensor(lhs) * rhs);
            }

            friend Tensor operator *(double lhs, const Tensor &rhs) {
                return (Tensor(lhs) * rhs);
            }

            Tensor operator *(const Tensor &d) {
                u_fun_enter(0, 0);
                Tensor ret(shape_.broadcast(d.shape()), std::max(type_, d.type()), false);
                op::run3<Tensor, op::Multiply>(ret, *this, d);
                u_fun_exit(0, 0);
                return (ret);
            }

            friend Tensor operator /(short lhs, const Tensor &rhs) {
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(unsigned short lhs, const Tensor &rhs) {
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(int lhs, const Tensor &rhs) {
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(unsigned int lhs, const Tensor &rhs) {
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(long lhs, const Tensor &rhs) {
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(size_t lhs, const Tensor &rhs) {
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(float lhs, const Tensor &rhs) {
                return (Tensor(lhs) / rhs);
            }

            friend Tensor operator /(double lhs, const Tensor &rhs) {
                return (Tensor(lhs) / rhs);
            }

            Tensor operator /(const Tensor &d) {
                u_fun_enter(0, 0);
                Tensor ret(shape_.broadcast(d.shape()), std::max(type_, d.type()), false);
                op::run3<Tensor, op::Divide>(ret, *this, d);
                u_fun_exit(0, 0);
                return (ret);
            }

            friend Tensor operator ^(short lhs, const Tensor &rhs) {
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(unsigned short lhs, const Tensor &rhs) {
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(int lhs, const Tensor &rhs) {
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(unsigned int lhs, const Tensor &rhs) {
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(long lhs, const Tensor &rhs) {
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(size_t lhs, const Tensor &rhs) {
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(float lhs, const Tensor &rhs) {
                return (Tensor(lhs) ^ rhs);
            }

            friend Tensor operator ^(double lhs, const Tensor &rhs) {
                return (Tensor(lhs) ^ rhs);
            }

            Tensor operator ^(const Tensor &d) {
                u_fun_enter(0, 0);
                Tensor ret(shape_.broadcast(d.shape()), std::max(type_, d.type()), false);
                op::run3<Tensor, op::Pow>(ret, *this, d);
                u_fun_exit(0, 0);
                return (ret);
            }

            // date type cast
            // if same type, copy no data, return tensor description **ONLY**
            Tensor astype(DType type) {
                u_fun_enter(0, 0);
                u_assert(data_.get()!= nullptr, "data empty. memory not allocated.");
                Tensor ret(shape_, type, false);
                op::run2<Tensor, op::Assign>(ret, *this);
                u_fun_exit(0, 0);
                return (ret);
            }

            template<typename T>
            Tensor astype() {
                return astype(ctype<T>());
            }

            template<typename T>
            const T* cast() {
                u_assert(ctype<T>() == type_, u::format("cast to different data type not supported."));
                return reinterpret_cast<T*>(data_.get());
            }

            // flatten change tensor description **ONLY** if copy is false
            void flatten_inplace(int beg_axis=0, int end_axis=-1) {
                u_fun_enter(0, 0);
                int beg = shape_.axis_normalize(beg_axis);
                int end = shape_.axis_normalize(end_axis);
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
                u_fun_exit(0, 0);
            }

            Tensor flatten(int beg_axis=0, int end_axis=-1) {
                u_fun_enter(0, 0);
                Tensor ret(*this);
                ret.flatten(beg_axis, end_axis);
                u_fun_exit(0, 0);
                return ret;
            }

            void reshape_inplace(const Shape& shape) {shape_.reshape(shape);}

            void reshape_inplace(const std::vector<int> &shape) {shape_.reshape(shape);}

            Tensor reshape(const Shape& shape) {
                Tensor ret(*this);
                return ret.reshape(shape);
            }

            Tensor reshape(const std::vector<int> &shape) {
                Tensor ret(*this);
                return ret.reshape(shape);
            }

            void squeeze_inplace() {
                u_fun_enter(0, 0);
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
                u_fun_exit(0, 0);
            }

            Tensor squeeze() {
                Tensor ret(*this);
                ret.squeeze();
                return ret;
            }

            void expand_dims_inplace(const int axis) {
                size_t _axis_ = shape_.axis_normalize(axis);
                shape_.insert(shape_.begin() + _axis_, 1);
            }

            Tensor expand_dims(const int axis) {
                Tensor ret(*this);
                ret.expand_dims(axis);
                return ret;
            }

            static Tensor zeros(const Shape &shape, const DType type) {
                return (Tensor(shape, type, false));
            }

            static Tensor zeros_like(const Tensor &t, const DType type=DType::invalid) {
                DType _type_ = (type == DType::invalid ? t.type() : type);
                return zeros(t.shape(), _type_);
            }

            static Tensor ones(const Shape &shape, const DType type) {
                Tensor t(shape, type, true);
                t.malloc();
                t += 1;
                return (t);
            }

            static Tensor ones_like(const Tensor &t, const DType type=DType::invalid) {
                DType _type_ = (type == DType::invalid ? t.type() : type);
                return ones(t.shape(), _type_);
            }

            inline Tensor sum(const int axis = u::tensor::all, const DType type = DType::float32, bool keepdims=false) {return dimension_op_run_<op::Sum>(axis, type, keepdims);}

            inline Tensor mean(const int axis = u::tensor::all, const DType type = DType::float32, bool keepdims=false) {return dimension_op_run_<op::Mean>(axis, type, keepdims);}

            inline Tensor stddev(const int axis = u::tensor::all, const DType type = DType::float32, bool keepdims=false) {return dimension_op_run_<op::StdDev>(axis, type, keepdims);}

            inline Tensor max(const int axis = u::tensor::all, const DType type = DType::invalid, bool keepdims=false) {return dimension_op_run_<op::Max>(axis, type, keepdims);}

            inline Tensor min(const int axis = u::tensor::all, const DType type = DType::invalid, bool keepdims=false) {return dimension_op_run_<op::Min>(axis, type, keepdims);}

            inline Tensor argmax(const int axis = u::tensor::all, const DType type = DType::uint32, bool keepdims=false) {return dimension_op_run_<op::ArgMax>(axis, type, keepdims);}

            inline Tensor argmin(const int axis = u::tensor::all, const DType type = DType::uint32, bool keepdims=false) {return dimension_op_run_<op::ArgMin>(axis, type, keepdims);}

            Tensor transpose(const std::vector<int> &dim_changes) {
                u_fun_enter(0, 0);
                u_assert(shape_.size() == dim_changes.size(), u::format("dimensions not match (%zu vs %zu)", shape_.size(), dim_changes.size()));
                std::vector<size_t> norm_dims = shape_.axis_normalize(dim_changes);

                std::map<size_t, size_t> dim_map;
                std::vector<size_t>::iterator it = norm_dims.end();
                Shape _shape_(shape_.size());
                for (size_t i = 0; i < shape_.size(); ++i) {
                    u_assert(norm_dims[i] < shape_.size(), u::format("dimension overflow (%zu [%d] vs %zu)", norm_dims[i], dim_changes[i], shape_.size()));
                    _shape_[i] = shape_[norm_dims[i]];
                    it = std::find(norm_dims.begin(), norm_dims.end(), i);
                    u_assert(it != norm_dims.end(), u::format("cannot find corresponding dimension for %zu-th dimension", i));
                    dim_map[i] = static_cast<unsigned int>(std::distance(norm_dims.begin(), it));
                }

                Tensor ret(_shape_, type_, false);
                op::run2<Tensor, op::Transpose>(ret, *this, norm_dims, dim_map);
                u_fun_exit(0, 0);
                return (ret);
            }

            static bool any(const Tensor &t, bool positive=true) {
                bool ans = false;
                Tensor tans(static_cast<unsigned char>(1));
                op::run2<Tensor, op::Any>(tans, t, positive);
                ans = (*tans.cast<unsigned char>() == 1);
                return ans;
            }

            static bool all(const Tensor &t, bool positive=true) {
                bool ans = false;
                Tensor tans(static_cast<unsigned char>(1));
                op::run2<Tensor, op::All>(tans, t, positive);
                ans = (*tans.cast<unsigned char>() == 1);
                return ans;
            }

            //
            // Tensor tile(const std::vector<size_t> &repeats) {
            //     u_assert(shape_.rank() == repeats.size(), u::format("repeats should specify each axis for tile. need [%zu] axes but given [%zu]", shape_.rank(), repeats.size()));
            //     Shape nshape(shape_.rank());
            //     for (size_t i=shape_.rank()-1; i>=0; --i) {
            //         nshape[i] = shape_[i];
            //         if (repeats[i] > 0) {
            //             nshape[i] *= repeats[i];            //
            // Tensor tile(const std::vector<size_t> &repeats) {
            //     u_assert(shape_.rank() == repeats.size(), u::format("repeats should specify each axis for tile. need [%zu] axes but given [%zu]", shape_.rank(), repeats.size()));
            //     Shape nshape(shape_.rank());
            //     for (size_t i=shape_.rank()-1; i>=0; --i) {
            //         nshape[i] = shape_[i];
            //         if (repeats[i] > 0) {
            //             nshape[i] *= repeats[i];
            //         }
            //     }
            //     Tensor ans(nshape, type_, false);
            //     for (size_t i=shape_.rank()-1; i>=0; --i) {
            //         nshape[i] = shape_[i];
            //         if (repeats[i] > 0) {
            //             nshape[i] *= repeats[i];
            //         }
            //     }
            // }
            //
            // Tensor tile(const std::vector<size_t> &repeats) {
            //     u_assert(shape_.rank() == repeats.size(), u::format("repeats should specify each axis for tile. need [%zu] axes but given [%zu]", shape_.rank(), repeats.size()));
            //     Shape nshape(shape_.rank());
            //     for (size_t i=shape_.rank()-1; i>=0; --i) {
            //         nshape[i] = shape_[i];
            //         if (repeats[i] > 0) {
            //             nshape[i] *= repeats[i];
            //         }
            //     }
            //     Tensor ans(nshape, type_, false);
            //     for (size_t i=shape_.rank()-1; i>=0; --i) {
            //         nshape[i] = shape_[i];
            //         if (repeats[i] > 0) {
            //             nshape[i] *= repeats[i];
            //         }
            //     }
            // }

            Tensor broadcast(const Shape &shape) {
                u_fun_enter(0, 0);
                Tensor ans;
                if (shape_ == shape) {
                    u::log::warning("It seems you try to broadcast tensor to the same shape. Tensor will ignore this operation and return `*this`");
                    ans = *this;
                } else {
                    u_assert(shape_.rank() == shape.rank(), u::format("cannot broadcast from '%s' to '%s'", shape_.str().c_str(), shape.str().c_str()));
                    for (int i=static_cast<int>(shape_.rank()-1); i>=0; --i) {
                        u_assert(shape_[i] == shape[i] || (shape_[i] == 1), u::format("Tensor with shape %s cannot broadcast to Tensor with shape %s", shape_.str().c_str(), shape.str().c_str()));
                    }
                    ans(shape, type_, false);
                    op::run2<Tensor, op::Broadcast>(ans, *this);
                }
                u_fun_exit(0, 0);
            }
        };
    }
}

#endif
