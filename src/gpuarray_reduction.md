# `GpuReduction` Internals Documentation

This document is intended as a developers' guide to the internals of the reduction kernel generator found in `gpuarray_reduction.[c|h]`.

## Reduction Overview

### Mathematics of Reductions

A **reduction** is to be understood here as the **iterated application** of a **binary**, **associative** mathematical **operator** equiped with an **identity element** over the **elements** of an array of _arbitrary_ size, _folding it_ down to size **1**.

An **array reduction** over a rank-`S` **source array** will be understood as a reduction where folding is only performed over elements along _some_ axes of the source array. The subset of axes along which the source array is folded is given as an ordered list `s`. The cardinality of `s` is `R`, which must lie in the range `[0, S]`. The result of an array reduction is another array, folded to just `D=S-R` axes, called the **destination array**.

The axes in the list `s` shall be called the **reduced axes**. The axes outside the list `s` (equivalently, the axes of the destination array) shall be called the **free axes**. The free and reduced axes are a _partition_ of the source array's axes.

If the array reduction is sensitive to the order of the axes in `s` or to a reversal of an axis, it shall be called **sensitive**. Otherwise it shall be called **insensitive**.

_Note: Actually, this definition of sensitivity might actually be equivalent to non-commutativity._

### Computing Reductions

Any array reduction over any source array involves `M` parallel reductions over `M` independent arrays of size `N` elements, with `M > 0` the product of the lengths of all _free_ axes, and `N >= 0` the product of the lengths of all _reduced_ axes. The source array has size `M*N` elements and the destination array has size `M` elements.

## Introduction to `GpuReduction`

The objective of `GpuReduction` is to execute arbitrary array reductions as fast as possible on GPU threads. For this to happen, the workload of `M*N` reduction operator applications must be partitioned as evenly as possible between GPU threads and worked on in parallel as much as possible.

At the same time, however, array reductions are by nature of low computational intensity; They involve one binary reduction operation per element loaded. The primary determinant of performance will therefore be memory bandwidth. It is thus essential to maximize the global memory read throughput, with global memory write throughput also becoming a consideration for destination tensors that are large relative to the source tensor.

An additional challenge is that the compilation of a GPU kernel takes a non-trivial amount of time and cache space. Therefore, the generation of a kernel must not require intimate knowledge of the shape and stride of the tensor to be reduced; For if it did,

1. The kernel compilation could only be done at invocation time.
2. The kernel compilation time would have to be added to the kernel execution time for fair performance comparisons.
3. The kernel compilation time would dominate the kernel execution time for the smallest reductions.

`GpuReduction` has thus been designed to generate extremely flexible, ahead-of-time-compilable kernels whose behaviour is as _"runtime-programmable"_ as possible, thus deferring decisions to as late as possible - invocation time. When the kernel is then invoked with properly-chosen arguments, it can simultaneously achieve nearly-optimal per-thread workload partitioning, compute speed and memory throughput.

### Maximizing Memory Throughput

On AMD and NVIDIA GPU devices, threads are spawned in 3D **blocks** of up to 1024 threads, but scheduled in **warps** or **wavefronts**. These are groups of 64 or 32 threads with consecutive linear thread IDs (A linear thread ID is computed as `lineartid = tid.x + bdim.x*(tid.y + bdim.y*(tid.z))`). Global memory throughput is maximized when scheduled global memory accesses from all active threads in a **warp** can be **coalesced** into a few large _memory transactions_ to contiguous addresses in global memory.

The likelihood of this happening is highest when the axes of a tensor are sorted in order of _ascending stride_, and the elements of the first few axes then distributed to threads with consecutive linear thread IDs. This minimizes the stride between consecutive threads, maximizing the chances of their memory accesses coalescing.

### Fair Workload Partitioning
