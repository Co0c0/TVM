# Auto Differentiation
-----------------------

file path : tvm\src\te\autodiff\jocobian.cc

主要用來存放operator的詳細使用方式


    PrimExpr VisitExpr_(const CallNode* op) {
      PrimExpr expr = GetRef<PrimExpr>(op);
      if (op->op.same_as(op_exp_)) {
        return Mul(Mutate(op->args[0]), expr);
      } else if (op->op.same_as(op_log_)) {
        return Div(Mutate(op->args[0]), op->args[0]);
      } else if (op->op.same_as(op_sigmoid_)) {
        return Mul(Mutate(op->args[0]), Mul(expr, Sub(FloatImm(expr.dtype(), 1.0), expr)));
      } else if (op->op.same_as(op_sqrt_)) {
        return Div(Mutate(op->args[0]), Mul(expr, FloatImm(expr.dtype(), 2.0)));
      } else if (op->op.same_as(op_tanh_)) {
        return Mul(Mutate(op->args[0]), Sub(FloatImm(expr.dtype(), 1.0), Mul(expr, expr)));
      } else if (op->op.same_as(op_pow_)) {
        auto x = op->args[0], y = op->args[1];
        return expr * (Mutate(y) * log(x) + Mutate(x) * y / x);
      } else if (op->op.same_as(op_fabs_)) {
        auto type = op->args[0].dtype();
        return Mul(Mutate(op->args[0]), Select(GE(op->args[0], make_zero(type)), FloatImm(type, 1.0),
                                               FloatImm(type, -1.0)));
      } else if (op->op.same_as(op_if_then_else_)) {
        Array<PrimExpr> new_args = {op->args[0], Mutate(op->args[1]), Mutate(op->args[2])};
        return Call(op->dtype, op->op, new_args);
      } else if (piecewise_const.count(op->op)) {
        return FloatImm(expr.dtype(), 0.0);
      } else {
        LOG(FATAL) << "Derivative of this intrinsic is not implemented: " << op->op;
        return PrimExpr();
      }
    }

    PrimExpr VisitExpr_(const AddNode* op) { return Add(Mutate(op->a), Mutate(op->b)); }

    PrimExpr VisitExpr_(const SubNode* op) { return Sub(Mutate(op->a), Mutate(op->b)); }

    PrimExpr VisitExpr_(const MulNode* op) {
      return Add(Mul(Mutate(op->a), op->b), Mul(op->a, Mutate(op->b)));
    }

    PrimExpr VisitExpr_(const DivNode* op) {
      return Div(Sub(Mul(Mutate(op->a), op->b), Mul(op->a, Mutate(op->b))), Mul(op->b, op->b));
    }

    PrimExpr VisitExpr_(const ModNode* op) NOT_IMPLEMENTED;

    PrimExpr VisitExpr_(const FloorDivNode* op) {
      return FloorDiv(Sub(Mul(Mutate(op->a), op->b), Mul(op->a, Mutate(op->b))), Mul(op->b, op->b));
    }

    PrimExpr VisitExpr_(const FloorModNode* op) NOT_IMPLEMENTED;

    PrimExpr VisitExpr_(const MinNode* op) {
      return Select(LE(op->a, op->b), Mutate(op->a), Mutate(op->b));
    }

    PrimExpr VisitExpr_(const MaxNode* op) {
      return Select(GE(op->a, op->b), Mutate(op->a), Mutate(op->b));
    }

    PrimExpr VisitExpr_(const EQNode* op) NOT_IMPLEMENTED;
    PrimExpr VisitExpr_(const NENode* op) NOT_IMPLEMENTED;
    PrimExpr VisitExpr_(const LTNode* op) NOT_IMPLEMENTED;
    PrimExpr VisitExpr_(const LENode* op) NOT_IMPLEMENTED;
    PrimExpr VisitExpr_(const GTNode* op) NOT_IMPLEMENTED;
    PrimExpr VisitExpr_(const GENode* op) NOT_IMPLEMENTED;
    PrimExpr VisitExpr_(const AndNode* op) NOT_IMPLEMENTED;
    PrimExpr VisitExpr_(const OrNode* op) NOT_IMPLEMENTED;


file path : tvm\src\te\autodiff\adjoint.cc

autodiff主要有三個步驟:
1. 計算輸入到輸出的tensor之間的依賴關係
2. 針對每一個有直接依賴關係的tensor進行微分、乘法...的運算
3. 求和

        Array<Tensor> Gradient(const Tensor& output, const Array<Tensor>& inputs,
                               const Tensor& head_or_null) {
          // Diagonal identity tensor
          Tensor head = head_or_null.get() ? head_or_null : Identity(output);

          // This Map{input -> outputs} maps a tensor to the list of tensors
          // immediately depending on it (using it in their bodies)
          std::unordered_map<Tensor, std::vector<Tensor>> reverse_dependencies;
          std::vector<Tensor> stack({output});
          while (!stack.empty()) {
            Tensor tensor = stack.back();
            stack.pop_back();
            for (const Tensor& input : tensor->op->InputTensors()) {
              if (!reverse_dependencies.count(input)) {
                stack.push_back(input);
              }
              reverse_dependencies[input].push_back(tensor);
            }
          }

          // This map maps tensors to the corresponding adjoints (dLoss/dTensor)
          std::unordered_map<Tensor, Tensor> adjoints;
          // head is the adjoint of output by definition
          adjoints[output] = head;

          // This is a recursive function that does all the work. It computes the adjoint for a given
          // tensor, adds it to the map, and returns it
          std::function<Tensor(const Tensor&)> compute_adjoint;
          compute_adjoint = [&compute_adjoint, &adjoints, &reverse_dependencies, &head,
                             &output](const Tensor& tensor) {
            if (!adjoints.count(tensor)) {
              // Here the adjoint hasn't been computed yet
              Tensor res_adjoint;
              std::vector<Tensor> direct_consumers = reverse_dependencies[tensor];
              if (direct_consumers.empty()) {
                // No reverse dependencies means that the output does not depend on this tensor,
                // return a zero tensor of the appropriate shape
                // (i.e., output shape + tensor shape, aka shape of Jacobian)
                Array<PrimExpr> result_shape(head->shape.begin(), head->shape.end() - output->shape.size());
                for (auto e : tensor->shape) {
                  result_shape.push_back(e);
                }
                res_adjoint = topi::full(result_shape, output->dtype, make_zero(output->dtype));
              } else {
                // The new adjoint is computed as a sum of the reverse dependencies' adjoints multiplied
                // by the corresponding "local" jacobians (dDep/dTensor). The computation of the jacobian
                // and the multiplication is done in the function VectorJacobianProduct
                for (const Tensor& direct_consumer : direct_consumers) {
                  // part = (adjoint of direct_consumer) * Jacobian(direct_consumer, tensor)
                  Tensor part =
                      VectorJacobianProduct(direct_consumer, tensor, compute_adjoint(direct_consumer));
                  res_adjoint = res_adjoint.get() ? topi::add(res_adjoint, part) : part;
                }
              }

              adjoints[tensor] = res_adjoint;
              return res_adjoint;
            } else {
              return adjoints[tensor];
            }
          };

          // Adjoints corresponding to inputs
          Array<Tensor> result;
          // Compute an adjoint for each input
          for (const Tensor& input : inputs) {
            result.push_back(compute_adjoint(input));
          }

          return result;
        }
