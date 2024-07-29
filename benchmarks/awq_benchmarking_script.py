"""Benchmark the latency of awq layers on VLLM."""

from vllm import _custom_ops as ops
import torch
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Compare the effectiveness of different kernels.')
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Path to the .pt file containing the tensors that will be used for benchmarking.')

    loaded_tensors = torch.load(args.file)

    qweight = loaded_tensors["qweight"].cuda()
    scales = loaded_tensors["scales"].cuda()
    qzeros = loaded_tensors["qzeros"].to(dtype=torch.int32).cuda()
    reshaped_x = loaded_tensors["reshaped_x"].to(dtype=torch.float16).cuda()
    pack_factor = 8

    out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)
