# Emily Sillars
I am currently developing a tiling cost model in MLIR for the **Snitch Cluster** [1] [2], a cluster of RISC-V AI accelerators. I use a work-in-progress compiler flow called **Quidditch** [3] which is a combination of **IREE, MLIR,** and **xDSL**.
## More Project Info: To be updated!!!
## Compiler (Un)School Landing Page
In general, I am interested in

- automating optimizations for Neural Networks in MLIR
- designing a cost model to decide when and what specific optimization should be performed in MLIR
- How can we integrate a modular/reusable cost model into the MLIR compiler flow (provided we restrict our domain to Neural Networks)?

I know a little bit about:

- polyhedral optimization, affine transformations at the affine and linalg dialect levels
- tiling tensors in the linalg dialect (great tutorial on how to do this in IREE: [4])
- tiling memrefs in the affine dialect

I would love to learn more about:

- polyhedral optimization :) Good resources for learning about it + simple examples? Do we need the polyhedral model to work with simple, perfectly nested loops like matmuls? Why or why not?

- transform dialect [5], and when we should use it instead of writing a transformation pass
- bufferization and transforming tensors into memrefs (how to do this, what is provided by MLIR or IREE to use as tools to get started)
- how to systematically and automatically insert calls that transfer memory from one memory level to another (automatic code generation for computing with scratchpads)

1. https://ieeexplore.ieee.org/document/9216552
2. https://github.com/pulp-platform/snitch_cluster?tab=readme-ov-file
3. https://github.com/opencompl/Quidditch
4. https://www.youtube.com/watch?v=FLEb30WyroA&t=2s
5. https://mlir.llvm.org/docs/Dialects/Transform/#overview
