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
