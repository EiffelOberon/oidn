// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "input_process.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  InputProcess::InputProcess(Engine* engine, const InputProcessDesc& desc)
    : InputProcessDesc(desc)
  {
    if (srcDims.size() != 3)
      throw std::invalid_argument("invalid input processing source shape");

    TensorDims dstDims = srcDims;

    TensorDims dstPaddedDims {
      round_up(srcDims[0], engine->getDevice()->getTensorBlockC()), // round up C
      dstDims[1],
      dstDims[2]
    };

    dstDesc = {dstDims, dstPaddedDims, engine->getDevice()->getTensorLayout(), engine->getDevice()->getTensorDataType()};

    setTile(0, 0, 0, 0, 0, 0);
  }

  void InputProcess::setSrc(const Ref<Image>& color,
                            const Ref<Image>& albedo,
                            const Ref<Image>& normal,
                            const Ref<Image>& history,
                            const Ref<Image>& historyConfidence,
                            const Ref<Image>& extra0,
                            const Ref<Image>& extra1,
                            const Ref<Image>& extra2)
  {
    int C = 0;
    if (color)  C += 3; // always broadcast to 3 channels
    if (albedo) C += 3;
    if (normal) C += 3;
    if (history) C += 3;
    if (historyConfidence) C += 1;
    const bool extra = extra0 || extra1 || extra2;
    if (extra) C += 9;
    if (C != srcDims[0])
      throw std::invalid_argument("invalid input processing source");
    if (bool(history) != bool(historyConfidence) ||
        (history && !(color && albedo && normal)))
      throw std::invalid_argument("history needs color, albedo, normal and confidence");
    if (extra && !(history && extra0 && extra1 && extra2))
      throw std::invalid_argument("extra inputs need history and all of extra0..2");

    this->color  = color;
    this->albedo = albedo;
    this->normal = normal;
    this->history = history;
    this->historyConfidence = historyConfidence;
    this->extra0 = extra0;
    this->extra1 = extra1;
    this->extra2 = extra2;
    updateSrc();
  }

  void InputProcess::setDst(const Ref<Tensor>& dst)
  {
    if (!dst || dst->getDesc() != dstDesc)
      throw std::invalid_argument("invalid input processing destination");

    this->dst = dst;
  }

  void InputProcess::setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
  {
    tile.hSrcBegin = hSrc;
    tile.wSrcBegin = wSrc;
    tile.hDstBegin = hDst;
    tile.wDstBegin = wDst;
    tile.H = H;
    tile.W = W;
  }

  void InputProcess::check()
  {
    if (!getMainSrc() || !dst)
      throw std::logic_error("input processing source/destination not set");
    if (tile.hSrcBegin + tile.H > getMainSrc()->getH() ||
        tile.wSrcBegin + tile.W > getMainSrc()->getW() ||
        tile.hDstBegin + tile.H > dst->getH() ||
        tile.wDstBegin + tile.W > dst->getW())
      throw std::out_of_range("input processing source/destination out of bounds");
  }

OIDN_NAMESPACE_END