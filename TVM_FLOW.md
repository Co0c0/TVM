# TVM compile 流程
------------------
# Example Compilation Flow

1. import : 將所有包含可以將model轉換成IRModule的函式import進來

2. Transformation :  編譯器(compiler)會將得到的IRModule轉換成可以拿來使用的IRModule或是大致一致的IRModule(像是量化的時候)，很多的Transformation是獨立的(target independent)，我們甚至允許target去影響configuration of the transformation pipeline.

3. Target Translation : 編譯器會轉換(codegen)(translate)IRModule成一個可執行的形式然後包裝進runtime.Module，讓他可以在target runtime environment進行匯出、匯入、執行 。

4. Runtime Execution : 使用者將把runtime.Module匯入回來，跑裡面的compiled functions 在支援runtime的環境下

![Image of Yaktocat](https://raw.githubusercontent.com/tlc-pack/web-data/main/images/design/tvm_dyn_workflow.svg)

# Key data structures

最好理解一個複雜系統的方式就是去了解它的最關鍵的資料型態(key data structure)還有他的API，API有關操作(transform)這些資料型態。

**IRModule**(intermediate representation module)是在整個stack裡面最主要的資料型態，裡面包含了有許多的函式，主要有兩大函式

* **relay::Function** is a high-level functional progame representation。relay.function通常跟終端機連終端機的模型有關，可以將它視為對於計算圖的一些支援(control-flow,recursion,and complex data structures)。

* **tir::PrimFunc** is a low-level program representation that contains elements including loop-nest chioces,multi-dimensional load/store,threading,and vector/tensor instructions.
-------------------------------------------------------
# Transformation

每個Transformation都具有至少下列其中一個目的

* **optimization** : 將程式轉變(transform)成一個更好的版本

* **lowering** : 將程式轉變成一個更低階的表示方式以方便跟目標函式更吻合

**relay/transform** contains a collection of passes that optimize the model.The optimizations include common program optimizations such as constant folding and dead-code elimination, and tensor-computation specific passes such as layout transformation and scaling factor folding.

在relay optimization快結束時，pass(FuseOps)會執行並且把end-to-end function分解成兩個小函式，我們稱之為segments of functions.這個舉動將原本的問題分解的更小了:

* Compilation and optimization for each sub-function
* Overall execution structure : we need to do a sequence of calls into the generated sub-function and use exeternal code genarators.

**tir/transform** contains transformation passes for TIR level functions.

最後提到在低階的優化可以交給LLVM,CUDA.C,還有其他target compiler.總之就是我們把低階的優化工作交給下層的編譯器去處理，因此就只要專心放在上層的優化工作就好。

--------------------------------------
### Search-space and Learning-based Transformations

TVM stack的最終目標是支援對於不同硬體具有高度的優化表現，為了實現我們必須盡可能的探索有關優化的各種方式，舉例multi-dimensional tensor access, loop tiling behavior, special accelerator memory hierarchy, and threading.

上述這些很難用heuristic去得出一個最佳解，取而代之的是我們利用因為我們很難用heuristic去得出一個最佳解，取而代之的是我們收集一群actions，這些action是可以transform program的，舉例:loop transformations, inlining, vectorization.上述這些我們統稱scheduling primitives

這群scheduling primitives定義了一個空間裡面都是優化器，所以系統以後只要從這個空間裡面找到最好的優化組合就可以了。至於搜尋的方式就是被機器學習演算法所控制。

我們可以記錄對每個運算子來說最好的序列，如此一來編譯器就可以快速找到最好的序列並套用到程式身上。

我們利用search based optimizations去解決initial tir function generation problem。這一部份的模型稱作AutoTVM(auto_schedule)。

# Target Translation

translate : IRModule -> corresponding target executable format


# Runtime Execution

TVM的runtime最主要的目的是提供最小程度API給他loading跟execute其他程式語言。












