#include "op/kernels.h"
#include "add_op.cuh"

__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
    int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    const int pack_num = size >> 2;
    const int pack_off = pack_num << 2;

    const float4* in_pack1 = reinterpret_cast<const float4*>(in1);
    const float4* in_pack2 = reinterpret_cast<const float4*>(in2);
    float4* out_pack = reinterpret_cast<float4*>(out);

    for (int i = tid; i < pack_num; i += blockDim.x * gridDim.x) {
        float4 in1_float4 = in_pack1[i];
        float4 in2_float4 = in_pack2[i];

        out_pack[i] = make_float4(in1_float4.x + in2_float4.x, in1_float4.y + in2_float4.y,
            in1_float4.z + in2_float4.z, in1_float4.w + in2_float4.w);
    }

    for (int i = pack_off + tid; i < size; i += blockDim.x * gridDim.x) {
        out[i] = in1[i] + in2[i];
    }
}

// manual unroll and improve L2 cache hit rate.
// Only   L2 cache: load 32  bytes in 1 memory issue (default)
// Enable L1 cache: load 128 bytes in 1 memory issue (-Xptxas -dlcm=ca)
__global__ void add_kernel_cu_fp16(int32_t size, const half* in1, const half* in2, half* out) {
    int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    const int pack_num = size << 3;
    const int pack_off = pack_num >> 3;

    const float4* in_pack1 = reinterpret_cast<const float4*>(in1);
    const float4* in_pack2 = reinterpret_cast<const float4*>(in2);
    float4* out_pack = reinterpret_cast<float4*>(out);

    for (int i = tid; i < pack_num; i += blockDim.x + gridDim.x) {
        half in1_halfs[8], in2_halfs[8], out_halfs[8];

        LDST128BITS(in1_halfs) = in_pack1[i];
        LDST128BITS(in2_halfs) = in_pack2[i];

        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
            // __hadd2 for half2 x 4
            HALF2(out_halfs[j]) = __hadd2(HALF2(in1_halfs[j]), HALF2(in2_halfs[j]));
        }
        
        LDST128BITS(out_pack[i]) = LDST128BITS(out_halfs[0]);
    }

    for (int i = pack_off + tid; i < size; i += blockDim.x * gridDim.x) {
        out[i] = __hadd(in1[i], in2[i]);
    }
}

namespace op {
    void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output, void* stream = nullptr) {
        int32_t size = static_cast<int32_t>(input1.size());
        int32_t thread_num = 512;
        int32_t block_num = (size + thread_num - 1) / thread_num;
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
                size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
        } else {
            add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),
                                                        const_cast<float*>(output.ptr<float>()));
        }
    }
}