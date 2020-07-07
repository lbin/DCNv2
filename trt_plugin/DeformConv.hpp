#pragma once

#include "serialize.hpp"
#include <NvInferPlugin.h>
#include <cublas_v2.h>
#include <memory>
#include <string>
#include <vector>

#define DCNV2_PLUGIN_NAME "DCNv2"
#define DCNV2_PLUGIN_VERSION "1"

// NOTE compiled as a MODULE (intended to loaded with dlopen rather than linked when compiling)
// requires non-inlined virtual destructors implemented in cpp
// simple inlined destructor implementation in header would cause
// `undefined symbol to vtable'
// error when loaded, and I'm not sure why

// IPluginV2Ext / IPluginV2IOExt / IPluginV2DynamicExt is required to use with explicit_batch_dimension
class DCNPlugin final : public nvinfer1::IPluginV2Ext
{
public:
    // the input pointers are cpu array, whose sizes are encoded in kernel_dim
    DCNPlugin(const float *layer_weight,
              const float *layer_bias,
              nvinfer1::Dims4 kernel_dim,
              nvinfer1::DimsHW stride_dim,
              nvinfer1::DimsHW pad_dim,
              nvinfer1::DimsHW dilation_dim,
              int deformable_group);

    DCNPlugin(const void *data, size_t length) { this->deserialize(data, length); }

    virtual ~DCNPlugin();

    const char *getPluginType() const override { return DCNV2_PLUGIN_NAME; }
    const char *getPluginVersion() const override { return DCNV2_PLUGIN_VERSION; }

    int getNbOutputs() const override { return 1; }

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const override
    {
        return nvinfer1::DataType::kFLOAT;
    }

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs, int nbInputDims) override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const override
    {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const override
    {
        return false;
    }

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override
    {
        return type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kLINEAR;
    }

    void configurePlugin(const nvinfer1::Dims *inputDims, int nbInputs,
                         const nvinfer1::Dims *outputDims, int nbOutputs,
                         const nvinfer1::DataType *inputTypes, const nvinfer1::DataType *outputTypes,
                         const bool *inputIsBroadcast, const bool *outputIsBroadcast,
                         nvinfer1::PluginFormat floatFormat, int maxBatchSize) override;

    int initialize() override { return 0; }
    void terminate() override {}

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void *const *inputs, void **outputs,
                void *workspace, cudaStream_t stream) override;

    void destroy() override { delete this; }

    const char *getPluginNamespace() const override { return _namespace.c_str(); }
    void setPluginNamespace(const char *ns) override { _namespace = ns; }

    IPluginV2Ext *clone() const override;

    void attachToContext(cudnnContext * /* cudnn */,
                         cublasContext *cublas_context,
                         nvinfer1::IGpuAllocator *allocator) override;
    void detachFromContext() override;

private:
    // should be serialized
    std::vector<float> weight; // (output_c, input_c, kernel_h, kernel_w)
    std::vector<float> bias;   // (output_c)
    int output_c, input_c;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dilation_h, dilation_w;
    int deformable_group;

    // computed from input, but also be serialized
    int input_h, input_w;

    // runtime resources
    std::string _namespace;
    cublasHandle_t cublas_handle{};
    nvinfer1::IGpuAllocator *gpu_alloc;
    float *weight_dev = nullptr, *bias_dev = nullptr;

private:
    void deserialize(const void *data, size_t length);
    size_t getSerializationSize() const override;
    void serialize(void *buffer) const override;
};

class DCNPluginCreator : public nvinfer1::IPluginCreator
{
public:
    DCNPluginCreator();
    virtual ~DCNPluginCreator();

    const char *getPluginNamespace() const override { return _namespace.c_str(); }
    const char *getPluginName() const override { return DCNV2_PLUGIN_NAME; }
    const char *getPluginVersion() const override { return DCNV2_PLUGIN_VERSION; }

    nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
    {
        return new DCNPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *ns) override { _namespace = ns; }
    const nvinfer1::PluginFieldCollection *getFieldNames() override
    {
        return &_attributes;
    }
    nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) override;

private:
    std::string _namespace;
    std::vector<nvinfer1::PluginField> _fields;
    nvinfer1::PluginFieldCollection _attributes;
};

#undef DCNV2_PLUGIN_NAME
#undef DCNV2_PLUGIN_VERSION
