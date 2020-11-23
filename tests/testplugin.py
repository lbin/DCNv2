#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import ctypes
import os
import sys
import tempfile

import packaging.version as pver
import torch

import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
from dcn_v2 import DCN

deformable_groups = 1
N, inC, inH, inW = 2, 2, 4, 4
outC = 2
kH, kW = 3, 3

PLUGIN_REG = trt.get_plugin_registry()

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

network_creation_flag = 0
# network_creation_flag |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
network_creation_flag |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def convert_engine(onnx_fn, engine_fn, batch_size, fp16=True, int8_calibrator=None, workspace=4_000_000_000):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(network_creation_flag) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = workspace
        builder.max_batch_size = batch_size
        builder.fp16_mode = fp16
        if int8_calibrator:
            builder.int8_mode = True
            builder.int8_calibrator = int8_calibrator
        with open(onnx_fn, "rb") as f:
            if not parser.parse(f.read()):
                print("got {} errors: ".format(parser.num_errors))
                for i in range(parser.num_errors):
                    e = parser.get_error(i)
                    print(e.code(), e.desc(), e.node())
                return
            else:
                print("parse successful")
        print("inputs: ", network.num_inputs)
        for i in range(network.num_inputs):
            print(i, network.get_input(i).name, network.get_input(i).shape)
        print("outputs: ", network.num_outputs)
        for i in range(network.num_outputs):
            output = network.get_output(i)
            print(i, output.name, output.shape)
        engine = builder.build_cuda_engine(network)
    with open(engine_fn, "wb") as f:
        f.write(engine.serialize())
        print("done")


def run_engine(engine_fn, inputs, input_names, output_names):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_fn, "rb") as f:
        ebuf = f.read()
    engine = runtime.deserialize_cuda_engine(ebuf)
    context = engine.create_execution_context()
    bindings = [None] * (len(input_names) + len(output_names))

    def torch_dtype_from_trt(trt_dtype):
        return {
            trt.float16: torch.float16,
            trt.float32: torch.float32,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
        }[trt_dtype]

    # create output tensors
    outputs = [None] * len(output_names)
    for i, output_name in enumerate(output_names):
        idx = engine.get_binding_index(output_name)
        dtype = torch_dtype_from_trt(engine.get_binding_dtype(idx))
        shape = tuple(engine.get_binding_shape(idx))
        device = "cuda"
        output = torch.empty(size=shape, dtype=dtype, device=device)
        outputs[i] = output
        bindings[idx] = output.data_ptr()

    for i, input_name in enumerate(input_names):
        idx = engine.get_binding_index(input_name)
        bindings[idx] = inputs[i].data_ptr()

    context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    outputs = tuple(outputs)
    if len(outputs) == 1:
        outputs = outputs[0]

    return outputs


def convert_dcn_node_to_plugin(node):
    type_str = "DCNv2"
    version_str = "1"
    namespace_str = ""
    dcn_plugin_creator = PLUGIN_REG.get_plugin_creator(type_str, version_str, namespace_str)

    var_inputs = []
    val_inputs = {}
    for inp in node.inputs:
        iname = inp.name.split(".")[-1]
        if iname in ["weight", "bias"]:
            assert isinstance(inp, gs.Constant)
            val_inputs[iname] = inp.values
        else:
            var_inputs.append(inp)

    def value2bytes(v, size_of_type=4):
        return (v).to_bytes(size_of_type, byteorder=sys.byteorder)

    def iterable2bytes(p, size_of_type=4):
        return b"".join([value2bytes(s, size_of_type) for s in p])

    # print(val_inputs)

    fields = [
        trt.PluginField(name="weight", data=val_inputs["weight"].tobytes(), type=trt.PluginFieldType.FLOAT32),
        trt.PluginField(name="bias", data=val_inputs["bias"].tobytes(), type=trt.PluginFieldType.FLOAT32),
        trt.PluginField(
            name="kernel_shape",
            data=iterable2bytes(val_inputs["weight"].shape),
            type=trt.PluginFieldType.INT32,
        ),
        trt.PluginField(name="stride", data=iterable2bytes(node.attrs["stride"]), type=trt.PluginFieldType.INT32),
        trt.PluginField(
            name="padding",
            data=iterable2bytes(node.attrs["padding"]),
            type=trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            name="dilation",
            data=iterable2bytes(node.attrs["dilation"]),
            type=trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            name="deformable_group",
            data=value2bytes(node.attrs["deformable_groups"]),
            type=trt.PluginFieldType.INT32,
        ),
    ]
    params = trt.PluginFieldCollection(fields)
    dcn_plugin = dcn_plugin_creator.create_plugin(type_str, params)
    plugin_serialization = bytes(dcn_plugin.serialize())

    # print(plugin_serialization)

    plugin_attrs = {
        "name": type_str,
        "version": version_str,
        "namespace": namespace_str,
        "data": plugin_serialization,
    }

    node.op = "TRT_PluginV2"
    node.inputs = var_inputs
    node.attrs = plugin_attrs


NODE_CONVERSION_REGISTRY = {"DCNv2_2": convert_dcn_node_to_plugin}


def fix_nodes(onnx_fn, output_onnx_fn=None):
    g = gs.import_onnx(onnx.load(onnx_fn))

    for n in g.nodes:
        conv = NODE_CONVERSION_REGISTRY.get(n.op)
        if conv:
            conv(n)

    g.toposort()

    onproto = gs.export_onnx(g)
    # NOTE you may see CI error: 'ascii' codec can't handle serialized plugin data
    # print(onnx.helper.printable_graph(onproto.graph))

    onnx.save(onproto, output_onnx_fn or onnx_fn)


def test_export_engine():

    input = torch.rand(N, inC, inH, inW) * 0.01
    input.requires_grad = True
    input = input.cuda()

    # offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW) * 2
    # # offset.data.zero_()
    # # offset.data -= 0.5
    # offset.requires_grad = True

    # mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW)
    # # mask.data.zero_()
    # mask.requires_grad = True
    # mask = torch.sigmoid(mask)

    weight = torch.randn(outC, inC, kH, kW)
    weight = torch.nn.Parameter(weight)

    bias = torch.rand(outC)
    bias = torch.nn.Parameter(bias)

    stride = 1
    padding = 1
    dilation = 1

    model = DCN(
        inC,
        outC,
        (kH, kW),
        stride=stride,
        padding=padding,
        dilation=dilation,
        deformable_groups=deformable_groups,
    )
    model.weight = weight
    model.bias = bias

    model = model.cuda().eval()

    pyout = model(input)

    onnx_fn = os.path.join(tempfile.gettempdir(), "dcn_test.onnx")

    versioned_args = {}
    if pver.parse(torch.__version__) >= pver.parse("1.3"):
        # this default behavior changed in pytorch 1.3
        versioned_args["keep_initializers_as_inputs"] = True
    if pver.parse(torch.__version__) >= pver.parse("1.5"):
        # new in torch 1.5, turn off or it yells at your unregistered custom op
        versioned_args["enable_onnx_checker"] = False
    with torch.no_grad():
        torch.onnx.export(
            model,
            input,
            onnx_fn,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            opset_version=11,  # boost version to play with trt7
            verbose=True,
            **versioned_args,
        )
    print("ONNX export finished")

    fix_nodes(onnx_fn)
    print("ONNX surgeon finished")

    engine_fn = os.path.join(tempfile.gettempdir(), "dcn_test.engine")
    convert_engine(onnx_fn, engine_fn, batch_size=N, fp16=False)
    print("TRT conversion finished")

    trtout = run_engine(engine_fn, [input], ["input"], ["output"])

    if not torch.allclose(pyout, trtout, atol=1e-4, rtol=1e-2):
        print("result mismatch!")
        print("pyout = \n{}".format(pyout.detach().cpu().numpy()))
        print("trtout = \n{}".format(trtout.detach().cpu().numpy()))
        return False

    return True


def convert_centertrack():
    batch_size = 32

    onnx_fn = "/root/centertrack/output_nopost.onnx"
    output_onnx_fn = "/root/dcnv2/centertrack.onnx"

    fix_nodes(onnx_fn, output_onnx_fn)
    print("ONNX surgeon finished")

    engine_fn = "/root/dcnv2/centertrack.engine"
    convert_engine(output_onnx_fn, engine_fn, batch_size=batch_size, fp16=False)
    print("TRT conversion finished")


if __name__ == "__main__":
    ctypes.CDLL(os.path.join(os.path.dirname(__file__), "../build/libdcnv2_trt_plugin.so"))

    print("test export engine: ", test_export_engine())
    # convert_centertrack()
