# Machine environments

Fill this in for each machine the project is built on. When something builds
on one machine and not the other, check here first.

| | Person A | Person B |
|---|---|---|
| OS | | |
| GPU | RTX 3060 Ti | |
| GPU compute capability | 86 | |
| NVIDIA driver version | | |
| CUDA Toolkit version | | |
| cuDNN version | | |
| TensorRT version | | |
| OpenCV version | | |
| CMake version | | |

To find your GPU's compute capability:
```
nvidia-smi --query-gpu=compute_cap --format=csv
```

Once both rows are filled in, update `CMAKE_CUDA_ARCHITECTURES` in
`CMakeLists.txt` to include both compute capabilities.
