#include <thread>
#include <vector>

const size_t num_threads = 16;

std::vector<size_t> decompose(
    size_t index, 
    const std::vector<size_t>& stride 
) {
    std::vector<size_t> ret(stride.size());
    for(size_t i = 0; i < stride.size(); i++) {
        ret[i] = index / stride[i];
        index = index % stride[i];
    }
    return ret;
}

size_t compose(
    const std::vector<size_t>& index,
    const std::vector<size_t>& shape, 
    const std::vector<size_t>& stride
) {
    size_t ret = 0;
    for(size_t i = 0; i < index.size(); i++) {
        if(shape[i] == 1) continue;
        ret += index[i] * stride[i];
    }
    return ret;
}

size_t get_index(
    size_t index, 
    const std::vector<size_t>& shape, 
    const std::vector<size_t>& stride,
    const std::vector<size_t>& s_stride
) {
    return compose(
        decompose(index, s_stride),
        shape, 
        stride
    );
}

template<typename T>
void normal_add(
    const std::vector<T>& a_data,
    const std::vector<size_t>& a_stride, 
    const std::vector<size_t>& a_shape,
    const std::vector<T>& b_data, 
    const std::vector<size_t>& b_stride, 
    const std::vector<size_t>& b_shape,
    std::vector<T>& final_data,
    const std::vector<size_t>& s_stride
) {
    for(size_t i = 0; i < final_data.size(); i++) {
        size_t a_i = get_index(i, a_shape, a_stride, s_stride);
        size_t b_i = get_index(i, b_shape, b_stride, s_stride);
        final_data[i] = a_data[a_i] + b_data[b_i];
    }
}

template<typename T>
void threaded_normal_add(
    const std::vector<T>& a_data,
    const std::vector<size_t>& a_stride, 
    const std::vector<size_t>& a_shape,
    const std::vector<T>& b_data, 
    const std::vector<size_t>& b_stride, 
    const std::vector<size_t>& b_shape,
    std::vector<T>& final_data,
    const std::vector<size_t>& s_stride
) { 
    size_t chunk = (final_data.size() + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    auto worker = [&](size_t start, size_t end) {
        for(size_t i = start; i < end; i++) {
            size_t a_i = get_index(i, a_shape, a_stride, s_stride);
            size_t b_i = get_index(i, b_shape, b_stride, s_stride);
            final_data[i] = a_data[a_i] + b_data[b_i];
        }
    };

    for(size_t i = 0; i < num_threads; i++) {
        size_t start = i * chunk;
        size_t end = std::min(start + chunk, final_data.size());
        threads.emplace_back(
            worker,
            start,
            end
        );
    }

    for(auto &th : threads) {
        th.join();
    }
}

