// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/kernel.h"
#include "core/tensor_accessor.h"

#if !defined(OIDN_COMPILE_METAL_DEVICE)
  #include "core/pool.h"
#endif

OIDN_NAMESPACE_BEGIN

#if !defined(OIDN_COMPILE_METAL_DEVICE)
  template<typename T, TensorLayout layout>
  struct GPUPoolKernel
  {
    TensorAccessor3D<T, layout> src;
    TensorAccessor3D<T, layout> dst;

    oidn_device_inline void operator ()(const WorkItem<3>& it) const
    {
      const int c = it.getGlobalID<0>();
      const int h = it.getGlobalID<1>();
      const int w = it.getGlobalID<2>();

      const T x0 = src(c, h*2,   w*2);
      const T x1 = src(c, h*2,   w*2+1);
      const T x2 = src(c, h*2+1, w*2);
      const T x3 = src(c, h*2+1, w*2+1);

      dst(c, h, w) = math::max(math::max(x0, x1), math::max(x2, x3));
    }
  };

  // Optimized for HWC layout (memory coalescing)
  template<typename T>
  struct GPUPoolKernel<T, TensorLayout::hwc>
  {
    TensorAccessor3D<T, TensorLayout::hwc> src;
    TensorAccessor3D<T, TensorLayout::hwc> dst;

    oidn_device_inline void operator ()(const WorkItem<3>& it) const
    {
      const int h = it.getGlobalID<0>();
      const int w = it.getGlobalID<1>();
      const int c = it.getGlobalID<2>();

      const T x0 = src(c, h*2,   w*2);
      const T x1 = src(c, h*2,   w*2+1);
      const T x2 = src(c, h*2+1, w*2);
      const T x3 = src(c, h*2+1, w*2+1);

      dst(c, h, w) = math::max(math::max(x0, x1), math::max(x2, x3));
    }
  };

#endif // !defined(OIDN_COMPILE_METAL_DEVICE)

  // 2x2 max pooling over (h, w) work items, every channel per item: for
  // Metal, whose kernels have 2D work items only (glint temporal fork).
  template<typename T>
  struct GPUPool2DKernel
  {
    TensorAccessor3D<T, TensorLayout::hwc> src;
    TensorAccessor3D<T, TensorLayout::hwc> dst;

    oidn_device_inline void operator ()(const oidn_private WorkItem<2>& it) const
    {
      const int h = it.getGlobalID<0>();
      const int w = it.getGlobalID<1>();
      for (int c = 0; c < dst.C; ++c)
      {
        const T x0 = src(c, h*2,   w*2);
        const T x1 = src(c, h*2,   w*2+1);
        const T x2 = src(c, h*2+1, w*2);
        const T x3 = src(c, h*2+1, w*2+1);
        dst(c, h, w) = math::max(math::max(x0, x1), math::max(x2, x3));
      }
    }
  };

#if !defined(OIDN_COMPILE_METAL_DEVICE)

  template<typename EngineT, typename SrcDstT, TensorLayout srcDstLayout>
  class GPUPool : public Pool
  {
  public:
    GPUPool(EngineT* engine,
            const PoolDesc& desc)
      : Pool(desc),
        engine(engine) {}

    Engine* getEngine() const override { return engine; }

  #if defined(OIDN_COMPILE_METAL)
    void finalize() override
    {
      pipeline = engine->newPipeline("pool_" + toString(DataTypeOf<SrcDstT>::value) +
                                     "_" + toString(srcDstLayout));
    }
  #endif

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      if (!src || !dst)
        throw std::logic_error("pooling source/destination not set");

    #if defined(OIDN_COMPILE_METAL)
      GPUPool2DKernel<SrcDstT> kernel2D;
      kernel2D.src = *src;
      kernel2D.dst = *dst;
      engine->submitKernel(WorkDim<2>(dst->getH(), dst->getW()), kernel2D,
                           pipeline, {src->getBuffer(), dst->getBuffer()});
    #else

      GPUPoolKernel<SrcDstT, srcDstLayout> kernel;
      kernel.src = *src;
      kernel.dst = *dst;

      if (srcDstLayout == TensorLayout::hwc)
        engine->submitKernel(WorkDim<3>(dst->getH(), dst->getW(), dst->getPaddedC()), kernel);
      else
        engine->submitKernel(WorkDim<3>(dst->getPaddedC(), dst->getH(), dst->getW()), kernel);
    #endif
    }

  private:
    EngineT* engine;

  #if defined(OIDN_COMPILE_METAL)
    Ref<MetalPipeline> pipeline;
  #endif
  };

#endif // !defined(OIDN_COMPILE_METAL_DEVICE)

OIDN_NAMESPACE_END
