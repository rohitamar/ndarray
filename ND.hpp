#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

#include "Adder.hpp"

template<typename T>
class ND {
    public:
    ND<T> operator+(ND<T> a) {
        if(shape_.size() != a.shape().size()) {
            throw std::invalid_argument("Cannot brodcast shapes.");
        }

        std::vector<size_t> shapeA = a.shape();
    
        std::vector<size_t> shapeS(shapeA.size());
        size_t tot = 1;
        for(size_t i = 0; i < a.shape().size(); i++) {
            if(shape_[i] == shapeA[i] || shape_[i] == 1 || shapeA[i] == 1) {
                shapeS[i] = std::max(shape_[i], shapeA[i]);
                tot *= shapeS[i];
            } else {
                throw std::invalid_argument("Cannot brodcast shapes.");
            }
        }

        ND<T> S(shapeS);
        std::vector<size_t> s_stride = S.strides();

        threaded_normal_add(
            data_,
            strides_,
            shape_,
            a.data(),
            a.strides(),
            a.shape(),
            S.data_,
            s_stride
        );

        return S;
    }

    ND<T> operator[](size_t index) const {
        std::vector<size_t> new_shape(shape_.begin() + 1, shape_.end());
        std::vector<size_t> new_strides(strides_.begin() + 1, strides_.end());
        size_t new_offset = contiguous_ ? offset_ : 0;
        new_offset += index * strides_[0];
        return ND<T>(data_, new_shape, new_strides, new_offset, contiguous_);
    }

    static ND<T> zeros(std::vector<size_t> shape) {
        ND<T> ret(shape);
        std::fill(ret.data_.begin(), ret.data_.end(), static_cast<T>(0));
        return ret;
    } 

    static ND<T> ones(std::vector<size_t> shape) {
        ND<T> ret(shape);
        std::fill(ret.data_.begin(), ret.data_.end(), static_cast<T>(1));
        return ret;
    }

    static ND<T> arange(T start, T stop, T step = static_cast<T>(1)) {
        std::vector<size_t> shape = {static_cast<size_t>((stop - start) / step)};
        ND<T> ret(shape);
        T v = start;
        for(size_t i = 0; i < shape[0]; i++) {
            ret.data_[i] = start + i * step; 
        }
        return ret;
    }

    std::vector<size_t> shape() const noexcept {
        return shape_;
    }

    std::vector<size_t> strides() const noexcept {
        return strides_;
    }

    std::vector<T> data() const noexcept {
        return data_;
    }

    bool contiguous() const noexcept {
        return contiguous_;
    }

    ND<T> transpose() {
        std::vector<size_t> shapeT = shape_, stridesT = strides_;
        reverse(shapeT.begin(), shapeT.end());
        reverse(stridesT.begin(), stridesT.end());
        ND<T> ret(data_, shapeT, stridesT, offset_, false);
        return ret;
    }

    void reshape(std::vector<size_t> shape) {
        if(!contiguous_) {
            throw std::invalid_argument("array is not contiguous");
        }
        shape_ = shape;
        strides_.resize(shape.size());
        strides_[strides_.size() - 1] = 1;
        if(shape.size() == 1) return;
        size_t tmp = 1;
        for(int i = (int) shape.size() - 2; i >= 0; i--) {
            strides_[i] = strides_[i + 1] * shape[i + 1];
        }
    }

    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const ND<U>& obj);

    private:
    std::vector<T> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_ = 0;
    bool contiguous_ = true;

    ND(std::vector<T> data, 
       std::vector<size_t> shape, 
       std::vector<size_t> strides, 
       size_t offset, 
       bool contiguous) : data_(data), shape_(shape), strides_(strides), offset_(offset), contiguous_(contiguous) { }

    ND(std::vector<size_t> shape) : shape_(shape) {
        size_t sz = 1;
        for(size_t s : shape) sz *= s;
        data_.resize(sz);
        strides_.resize(shape.size());
        strides_[strides_.size() - 1] = 1;
        if(shape.size() == 1) return;
        size_t tmp = 1;
        for(int i = (int) shape.size() - 2; i >= 0; i--) {
            strides_[i] = strides_[i + 1] * shape[i + 1];
        }
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const ND<T>& obj) {
    if(obj.shape().size() == 0) {
        os << obj.data_[obj.offset_];
    } else {
        os << '[';
        for(size_t i = 0; i < obj.shape()[0] - 1; i++) {
            os << obj[i] << ", ";
            if(obj.shape().size() > 1) os << "\n";
        }
        os << obj[obj.shape()[0] - 1] << "]";
    }
    return os;
}