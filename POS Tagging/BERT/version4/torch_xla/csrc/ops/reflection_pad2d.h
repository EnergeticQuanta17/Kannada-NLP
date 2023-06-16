#ifndef XLA_TORCH_XLA_CSRC_OPS_REFLECTION_PAD2D_H_
#define XLA_TORCH_XLA_CSRC_OPS_REFLECTION_PAD2D_H_

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ReflectionPad2d : public XlaNode {
 public:
  ReflectionPad2d(const torch::lazy::Value& input,
                  std::vector<int64_t> padding);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& padding() const { return padding_; }

 private:
  std::vector<int64_t> padding_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_REFLECTION_PAD2D_H_
