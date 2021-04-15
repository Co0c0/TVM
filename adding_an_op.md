# 加入一個operator進入TVM

為了從Relay IR加入TVM的operator，這些operators都需要在relay註冊是為了確保他們都會符合relay的系統

註冊oprators需要下列三個步驟:

* 利用RELAY_REGISTER_OP(c++)來註冊有關這個operator的屬性跟資料屬性資訊
* 定義一個c++函式來產生一個專門呼叫operator的node跟註冊相對應函式的python的API hook
* 把上面的python API hook包進一個neater的介面


file path : src/relay/op/tensor/binary.cc 說明了上面前兩個步驟
file paht : python/tvm/relay/op/tensor.py 則是給了最後一個步驟


# Registering an Operator

雖然TVM已經有operator註冊，但是Relay不能在沒有其他額外資訊的情況下適當地把它合併。

為了使註冊的operator更有彈性，通常operator都用input跟output之間的關係來表示

舉例來說 src/relay/op/type_relations.h 還有他們的實作 BroadcastRel 吃兩個input跟輸出一個output，檢查是否都是tensor type，最後確保output的型態跟input一樣。

如果現存的並沒有被讀取到的話或許可能需要再加一個type relation到type_relations.h

再來說明 RELAY_REGISTER_OP 允許開發者標明下列關於operator的資訊在relay裡

* Arity(number of arguments)
* Names and descriptions for positional arguments
* Support level (1 indicates an internal intrinsic; higher numbers indicate less integral or externally supported operators)
* A type relation for the operator

下面的例子是從 binary.cc 抓的，是有關用broadcasting的用法

    RELAY_REGISTER_OP("add")

      .set_num_inputs(2)
      .add_argument("lhs", "Tensor", "The left hand side tensor.")
      .add_argument("rhs", "Tensor", "The right hand side tensor.")
      .set_support_level(1)
      .add_type_rel("Broadcast", BroadcastRel);


# Creating a Call Node

簡單寫一下要這個函式做什麼然後將call node 回傳給operator

對現在的call來說attributes跟type arguments都不支援，所以只要用Op::Get 就可以從operator registry抓到operator的跟資訊然後將資訊傳到call node就好

以下範例:

    TVM_REGISTER_GLOBAL("relay.op._make.add")
        .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {
            static const Op& op = Op::Get("add");
          return Call(op, {lhs, rhs}, Attrs(), {});
        });
        
# Including a Python API Hook

函式藉由TVM_REGISTER_GLOBAL輸出，而且被分別包裝python function而不是直接呼叫，這件事式非常常見的，為了讓函式產生呼叫訊號去呼叫operator，把函式綁在一起似乎比較方便，詳情請看python/tvm/relay/op/tensor.py。

例子:

    def add(lhs, rhs):
        """Elementwise addition.

        Parameters
        ----------
        lhs : relay.Expr
            The left hand side input data
        rhs : relay.Expr
            The right hand side input data

        Returns
        -------
        result : relay.Expr
            The computed result.
        """
        return _make.add(lhs, rhs)

上述的Python wrappers可能是一個好機會去提供好的介面給operator使用

例如: concat operaotr 被註冊成只使用一個operator，也就是一個tuple可以將tensor連起來，但是python wrapper 將tensor視為argument然後將他們兩個合起來成為一個tuple在生成call node之前:

    def concat(*args):
        """Concatenate the input tensors along the zero axis.

        Parameters
        ----------
        args: list of Tensor

        Returns
        -------
        tensor: The concatenated tensor.
        """
        tup = Tuple(list(args))
        return _make.concat(tup


# Gradient Operators

，所以需要一個Gradient Operators在relay的環境下寫不同的程式，扮演很重要的角色，然而relay的aotudiff演算法在operator是透明的情況下是可以一階微分的，原因是relay不能接觸到實作的部分，所以需要一個明確的微分方式。

# Adding a Gradient in Python

有一堆Python gradient operators集合可以藉由python/tvm/relay/op/_tensor_grad.py 找到

我們先以sigmoid跟multiply舉例

    @register_gradient("sigmoid")
    def sigmoid_grad(orig, grad):
        """Returns [grad * sigmoid(x) * (1 - sigmoid(x))]."""
        return [grad * orig * (ones_like(orig) - orig)]

input分別為original operator跟gradient，ouput是一個list，第i個list的東西就是operator的第i個東西經過微分後所產生的。


再來考慮multiply

    @register_gradient("multiply")
    def multiply_grad(orig, grad):
        """Returns [grad * y, grad * x]"""
        x, y = orig.args
        return [collapse_sum_like(grad * y, x),
                collapse_sum_like(grad * x, y)]

# Adding a Gradient in C++

首先確保src/relay/pass/pattern_utils.h有包含在裡面，

    tvm::Array<Expr> MultiplyGrad(const Expr& orig_call, const Expr& output_grad) {
        const Call& call = orig_call.Downcast<Call>();
        return { CollapseSumLike(Multiply(output_grad, call.args[1]), call.args[0]),
                 CollapseSumLike(Multiply(output_grad, call.args[0]), call.args[1]) };
    }

最後我們要set_attr給“FPrimalGradient”在最後面加上


    RELAY_REGISTER_OP("multiply")
        // ...
        // Set other attributes
        // ...
        .set_attr<FPrimalGradient>("FPrimalGradient", MultiplyGrad);




reference:

### https://tvm.apache.org/docs/dev/relay_add_op.html



