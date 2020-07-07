## Deformable Convolutional Networks V2 with Pytorch 1.X

### Build

```bash
    python setup.py build develop         # build
```
or
```bash
    pip install -U 'git+ssh://git@192.168.253.179:2222/icd/dcnv2.git'
```

### Known Issues:

- [x] Gradient check w.r.t offset (solved)
- [ ] Backward is not reentrant (minor)

This is an adaption of the official [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets/tree/master/DCNv2_op).

Update: all gradient check passes with **double** precision. 

Another issue is that it raises `RuntimeError: Backward is not reentrant`. However, the error is very small (`<1e-7` for 
float `<1e-15` for double), 
so it may not be a serious problem (?)

Please post an issue or PR if you have any comments.

# About TRT Plugin

插件本身适用于TensorRT 7版本，在TensorRT的`onnxparser`中被解析为序列化的`TRT_PluginV2`层加入网络。

## Build

插件代码位于`trt_plugin`. 

```bash
mkdir build
cd build
cmake ../trt_plugin -DCMAKE_BUILD_TYPE=Release
make -j
```

## Usage & Test

依赖于`onnx_graphsurgeon`

可在[https://github.com/NVIDIA/TensorRT.git]下载到此工具的源码。
不出意外的话它应该自TensorRT 7.1.3起进入官方安装包。
**编译和安装此工具不依赖TensorRT的版本。**

用例和测试见`tests/testplugin.py`。
