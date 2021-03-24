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
