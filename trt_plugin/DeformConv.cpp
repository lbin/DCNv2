#include "DeformConv.hpp"
#include <cuda_runtime.h>
#include <experimental/source_location>
#include <stdexcept>
#include <string_view>

namespace
{
size_t dtype_step(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
        return sizeof(float) / 2;
    case nvinfer1::DataType::kINT8:
        return sizeof(int8_t);
    case nvinfer1::DataType::kINT32:
        return sizeof(int32_t);
    case nvinfer1::DataType::kBOOL:
        return sizeof(bool);
    default:
        return sizeof(float);
    }
}

size_t dtype_step(nvinfer1::PluginFieldType t)
{
    switch (t)
    {
    case nvinfer1::PluginFieldType::kFLOAT16:
        return sizeof(float) / 2;
    case nvinfer1::PluginFieldType::kFLOAT32:
        return sizeof(float);
    case nvinfer1::PluginFieldType::kFLOAT64:
        return sizeof(double);
    case nvinfer1::PluginFieldType::kINT8:
        return sizeof(int8_t);
    case nvinfer1::PluginFieldType::kINT16:
        return sizeof(int16_t);
    case nvinfer1::PluginFieldType::kINT32:
        return sizeof(int32_t);
    case nvinfer1::PluginFieldType::kCHAR:
        return sizeof(char);
    case nvinfer1::PluginFieldType::kDIMS:
        return sizeof(nvinfer1::Dims);
    case nvinfer1::PluginFieldType::kUNKNOWN:
    default:
        return sizeof(float);
    }
}

using srcloc = std::experimental::source_location;

void check_cuda(cudaError_t e, srcloc loc = srcloc::current())
{
    if (e != cudaSuccess)
    {
        const char *fstr = "%s:%d|%s| cuda error: (%d)%s";
        int len = snprintf(nullptr, 0, fstr,
                           loc.file_name(), loc.line(), loc.function_name(),
                           static_cast<int>(e), cudaGetErrorString(e));
        if (len > 0)
        {
            std::vector<char> msg(len + 1);
            snprintf(msg.data(), len, fstr,
                     loc.file_name(), loc.line(), loc.function_name(),
                     static_cast<int>(e), cudaGetErrorString(e));
            throw std::runtime_error(std::string(msg.data(), msg.data() + len));
        }
        throw std::runtime_error("cuda error (failed to format)");
    }
}
} // namespace

void DCNPlugin::deserialize(const void *data, size_t length)
{
    deserialize_value(&data, &length, &weight);
    deserialize_value(&data, &length, &bias);

    deserialize_value(&data, &length, &output_c);
    deserialize_value(&data, &length, &input_c);
    deserialize_value(&data, &length, &kernel_h);
    deserialize_value(&data, &length, &kernel_w);
    deserialize_value(&data, &length, &stride_h);
    deserialize_value(&data, &length, &stride_w);
    deserialize_value(&data, &length, &pad_h);
    deserialize_value(&data, &length, &pad_w);
    deserialize_value(&data, &length, &dilation_h);
    deserialize_value(&data, &length, &dilation_w);
    deserialize_value(&data, &length, &deformable_group);

    deserialize_value(&data, &length, &input_h);
    deserialize_value(&data, &length, &input_w);
}

size_t DCNPlugin::getSerializationSize() const
{
    return serialized_size(weight) +
           serialized_size(bias) +
           serialized_size(output_c) +
           serialized_size(input_c) +
           serialized_size(kernel_h) +
           serialized_size(kernel_w) +
           serialized_size(stride_h) +
           serialized_size(stride_w) +
           serialized_size(pad_h) +
           serialized_size(pad_w) +
           serialized_size(dilation_h) +
           serialized_size(dilation_w) +
           serialized_size(deformable_group) +
           serialized_size(input_h) +
           serialized_size(input_w);
}

void DCNPlugin::serialize(void *buffer) const
{
    serialize_value(&buffer, weight);
    serialize_value(&buffer, bias);

    serialize_value(&buffer, output_c);
    serialize_value(&buffer, input_c);
    serialize_value(&buffer, kernel_h);
    serialize_value(&buffer, kernel_w);
    serialize_value(&buffer, stride_h);
    serialize_value(&buffer, stride_w);
    serialize_value(&buffer, pad_h);
    serialize_value(&buffer, pad_w);
    serialize_value(&buffer, dilation_h);
    serialize_value(&buffer, dilation_w);
    serialize_value(&buffer, deformable_group);

    serialize_value(&buffer, input_h);
    serialize_value(&buffer, input_w);
}

DCNPlugin::DCNPlugin(const float *layer_weight,
                     const float *layer_bias,
                     nvinfer1::Dims4 kernel_dim,
                     nvinfer1::DimsHW stride_dim,
                     nvinfer1::DimsHW pad_dim,
                     nvinfer1::DimsHW dilation_dim,
                     int deformable_group)
    : output_c(kernel_dim.d[0]), input_c(kernel_dim.d[1]),
      kernel_h(kernel_dim.d[2]), kernel_w(kernel_dim.d[3]),
      stride_h(stride_dim.h()), stride_w(stride_dim.w()),
      pad_h(pad_dim.h()), pad_w(pad_dim.w()),
      dilation_h(dilation_dim.h()), dilation_w(dilation_dim.w()),
      deformable_group(deformable_group)
{
    weight = {layer_weight, layer_weight + output_c * input_c * kernel_h * kernel_w};
    bias = {layer_bias, layer_bias + output_c};
}

DCNPlugin::~DCNPlugin() = default;

nvinfer1::Dims DCNPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputs, int nbInputDims)
{
    // NOTE here use the provided input dims, because this function is called before configuration
    assert(nbInputDims == 3);
    // in order of:
    // input[ic, ih, iw], offset, mask
    // NOTE they are not batched yet
    const auto &inputDim = inputs[0];
    assert(inputDim.d[0] == input_c);

    // they shadows the vars inside plugin
    const int input_h = inputDim.d[1];
    const int input_w = inputDim.d[2];

    const int out_h = (input_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int out_w = (input_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    return nvinfer1::Dims3(output_c, out_h, out_w);
}

void DCNPlugin::configurePlugin(const nvinfer1::Dims *inputDims, int nbInputs,
                                const nvinfer1::Dims *outputDims, int nbOutputs,
                                const nvinfer1::DataType *inputTypes, const nvinfer1::DataType *outputTypes,
                                const bool *inputIsBroadcast, const bool *outputIsBroadcast,
                                nvinfer1::PluginFormat floatFormat, int maxBatchSize)
{
    assert(nbInputs == 3);
    // in order of:
    // input[ic, ih, iw], offset, mask
    // NOTE we are using explicit batch, but they are non-batched
    // and weight[oc, ic, kh, kw], bias[oc] are moved to attributes
    assert(inputDims[0].d[0] == input_c);
    input_h = inputDims[0].d[1];
    input_w = inputDims[0].d[2];
}

void DCNPlugin::attachToContext(cudnnContext * /* cudnn */,
                                cublasContext *cublas_context,
                                nvinfer1::IGpuAllocator *allocator)
{
    cublas_handle = cublas_context;
    gpu_alloc = allocator;

    static const size_t CUDA_ALIGNMENT = 256;
    weight_dev = static_cast<float *>(gpu_alloc->allocate(sizeof(float) * weight.size(), CUDA_ALIGNMENT, 0));
    bias_dev = static_cast<float *>(gpu_alloc->allocate(sizeof(float) * bias.size(), CUDA_ALIGNMENT, 0));

    // upload weight and bias to device
    check_cuda(cudaMemcpy(weight_dev, weight.data(), sizeof(float) * weight.size(), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(bias_dev, bias.data(), sizeof(float) * bias.size(), cudaMemcpyHostToDevice));
}

void DCNPlugin::detachFromContext()
{
    cublas_handle = nullptr;
    gpu_alloc->free(weight_dev);
    gpu_alloc->free(bias_dev);
}

nvinfer1::IPluginV2Ext *DCNPlugin::clone() const
{
    auto *p = new DCNPlugin(
        weight.data(),
        bias.data(),
        nvinfer1::Dims4(output_c, input_c, kernel_h, kernel_w),
        nvinfer1::DimsHW(stride_h, stride_w),
        nvinfer1::DimsHW(pad_h, pad_w),
        nvinfer1::DimsHW(dilation_h, dilation_w),
        deformable_group);
    p->input_h = input_h;
    p->input_w = input_w;
    return p;
}

DCNPluginCreator::DCNPluginCreator()
{
    _fields = {
        {"weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0}, // (output_c, input_c, kernel_h, kernel_w)
        {"bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0},   // (output_c)
        {"kernel_shape", nullptr, nvinfer1::PluginFieldType::kINT32, 4},
        {"stride", nullptr, nvinfer1::PluginFieldType::kINT32, 2},
        {"padding", nullptr, nvinfer1::PluginFieldType::kINT32, 2},
        {"dilation", nullptr, nvinfer1::PluginFieldType::kINT32, 2},
        {"deformable_group", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
    };

    _attributes.nbFields = _fields.size();
    _attributes.fields = _fields.data();
}

DCNPluginCreator::~DCNPluginCreator() = default;

nvinfer1::IPluginV2 *DCNPluginCreator::createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
{
    const float *weight{nullptr}, *bias{nullptr};
    size_t weight_bytes{0}, bias_bytes{0};
    nvinfer1::Dims4 kernel_shape;
    nvinfer1::DimsHW stride, pad, dilation;
    int deformable_group{0};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        auto *field = fc->fields + i;
        std::string_view name(field->name);
        if (name == "weight")
        {
            assert(field->type == nvinfer1::PluginFieldType::kFLOAT32);
            weight_bytes = field->length;
            weight = static_cast<const float *>(field->data);
        }
        else if (name == "bias")
        {
            assert(field->type == nvinfer1::PluginFieldType::kFLOAT32);
            bias_bytes = field->length;
            bias = static_cast<const float *>(field->data);
        }
        else if (name == "kernel_shape")
        {
            assert(field->type == nvinfer1::PluginFieldType::kINT32 && field->length == 4 * sizeof(int32_t));
            memcpy(kernel_shape.d, field->data, 4 * sizeof(int32_t));
        }
        else if (name == "stride")
        {
            assert(field->type == nvinfer1::PluginFieldType::kINT32 && field->length == 2 * sizeof(int32_t));
            memcpy(stride.d, field->data, 2 * sizeof(int32_t));
        }
        else if (name == "padding")
        {
            assert(field->type == nvinfer1::PluginFieldType::kINT32 && field->length == 2 * sizeof(int32_t));
            memcpy(pad.d, field->data, 2 * sizeof(int32_t));
        }
        else if (name == "dilation")
        {
            assert(field->type == nvinfer1::PluginFieldType::kINT32 && field->length == 2 * sizeof(int32_t));
            memcpy(dilation.d, field->data, 2 * sizeof(int32_t));
        }
        else if (name == "deformable_group")
        {
            assert(field->type == nvinfer1::PluginFieldType::kINT32 && field->length == 1 * sizeof(int32_t));
            deformable_group = *(static_cast<const int *>(field->data));
        }
    }
    if (!weight_bytes || !bias_bytes)
    {
        return nullptr;
    }
    return new DCNPlugin(weight, bias, kernel_shape, stride, pad, dilation, deformable_group);
}

REGISTER_TENSORRT_PLUGIN(DCNPluginCreator);
