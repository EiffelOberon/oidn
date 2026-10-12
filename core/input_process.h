// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"
#include "tensor.h"
#include "color.h"
#include "tile.h"

OIDN_NAMESPACE_BEGIN

  struct InputProcessDesc
  {
    TensorDims srcDims;
    std::shared_ptr<TransferFunction> transferFunc;
    bool hdr;
    bool snorm;
  };

  class InputProcess : public BaseOp, protected InputProcessDesc
  {
  public:
    InputProcess(Engine* engine, const InputProcessDesc& desc);

    TensorDesc getDstDesc() const { return dstDesc; }
    Ref<Tensor> getDst() const { return dst; }

    // glint temporal: `history` (the previous output, warped; encoded like
    // color) and `historyConfidence` (one channel, [0, 1]) follow the
    // normal as channels 9..12.
    void setSrc(const Ref<Image>& color,
                const Ref<Image>& albedo,
                const Ref<Image>& normal,
                const Ref<Image>& history = nullptr,
                const Ref<Image>& historyConfidence = nullptr);
    void setDst(const Ref<Tensor>& dst);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);

  protected:
    virtual void updateSrc() {}
    void check();

    Image* getMainSrc()
    {
      return color ? color.get() : (albedo ? albedo.get() : normal.get());
    }

    TensorDesc dstDesc;
    Ref<Image> color;
    Ref<Image> albedo;
    Ref<Image> normal;
    Ref<Image> history;
    Ref<Image> historyConfidence;
    Ref<Tensor> dst;
    Tile tile;
  };

OIDN_NAMESPACE_END
