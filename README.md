# TVM
------
TVM透過兩種方式生成代碼:

* tvm.build   主要針對單一運算子進行編譯優化
* relay.build 是針對整個網路計算圖進行編譯優化

先介紹relay.build,code如下

### relay.build
對relay.build進行trace code
首先是path: tvm/python/tvm/relay/build_module.py


    def build(mod, target=None, target_host=None, params=None, mod_name='default'):
        // ignore some code.....

        # If current dispatch context is fallback context (the default root context),
        # then load pre-tuned parameters from TopHub
        if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
            tophub_context = autotvm.tophub.context(list(target.values()))
        else:
            tophub_context = autotvm.util.EmptyContext()

        with tophub_context:
            bld_mod = BuildModule()
            graph_json, mod, params = bld_mod.build(mod, target, target_host, params)
            mod = _graph_runtime_factory.GraphRuntimeFactoryModule(graph_json, mod, mod_name, params)
            return mod
首先是尋找AutoTVM是否有預先tune好的參數紀錄，然後建構tophub_context，並且創建一個class紀錄如何進行build動作，BuildModule的class在build_module.py中,code如下:

    class BuildModule(object):
        """Build an IR module to run on TVM graph runtime. This class is used
        to expose the `RelayBuildModule` APIs implemented in C++.
        """
        def __init__(self):
            self.mod = _build_module._BuildModule()
            self._get_graph_json = self.mod["get_graph_json"]
            self._get_module = self.mod["get_module"]
            self._build = self.mod["build"]
            self._optimize = self.mod["optimize"]
            self._set_params_func = self.mod["set_params"]
            self._get_params_func = self.mod["get_params"]

        def build(self, mod, target=None, target_host=None, params=None):
            target = _update_target(target)

            # Setup the params.
            if params:
                self._set_params(params)
            # Build the IR module
            self._build(mod, target, target_host)
            # Get artifacts
            graph_json = self.get_json()
            mod = self.get_module()
            params = self.get_params()

            return graph_json, mod, params


透過FFI連結在_module_build.py與tvm/src/relay/beckend/build.module.cc使得python跟c++可以互相連結

### _module_build.py

    """The interface for building Relay functions exposed from C++."""
    import tvm._ffi

    tvm._ffi._init_api("relay.build_module", __name__)
    
### build.module.cc(Line 551~554)

    runtime::Module RelayBuildCreate() {
      auto exec = make_object<RelayBuildModule>();
      return runtime::Module(exec);
    }

為了實現Python跟C++混和編程，TVM使用了統一的PackedFunc機制。PackedFunc可以將C++中的各類函數打包成統一函數的接口，並自動導出到Python模塊中進行調用，並且也支持Python中註冊一個函數，偽裝成PackedFunc在C++和Python中調用
在build_module.cc中，利用Packed形成一個function->FuncRelayBuildModule

### FuncRelayBuildModule code(Line 123)

    class RelayBuildModule : public runtime::ModuleNode {
     public:
      /*!
       * \brief Get member function to front-end
       * \param name The name of the function.
       * \param sptr_to_self The pointer to the module node.
       * \return The corresponding member function.
       */
      PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
      //ignore...
        } else if (name == "build") {
          return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
            ICHECK_EQ(args.num_args, 3);
            this->Build(args[0], args[1], args[2]);
          });
        //ignore...

TVM對Build函數做了一次封裝返回一個PackedFunc，即RelayBuildModule類成員函數Build:

### this->Build(...) code (Line 237)

    void Build(IRModule mod, const TargetsMap& targets, const tvm::Target& target_host) {
      targets_ = targets;
      target_host_ = target_host;
      BuildRelay(mod, params_);
      // Clear compile engine so that tuning schedules can be changed between runs. See issue #6096.
      CompileEngine::Global()->Clear();

進一步調用成員函數BuildRelay

# BuildRelay code (Line 461)

    void BuildRelay(IRModule relay_module,
                      const std::unordered_map<std::string, tvm::runtime::NDArray>& params) {
        // Relay IRModule -> IRModule optimizations.
        relay_module = Optimize(relay_module, targets_, params);
        // Get the updated function.
        auto func = Downcast<Function>(relay_module->Lookup("main"));

        // Generate code for the updated function.
        graph_codegen_ = std::unique_ptr<GraphCodegen>(new GraphCodegen());
        graph_codegen_->Init(nullptr, targets_);
        graph_codegen_->Codegen(func);

        ret_.graph_json = graph_codegen_->GetJSON();
        ret_.params = graph_codegen_->GetParams();

        auto lowered_funcs = graph_codegen_->GetIRModule();

        Target target_host = GetTargetHost();
        // If no target_host has been set, we choose a default one, which is
        // llvm if "codegen.LLVMModuleCreate" is accessible.
        const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
        if (!target_host.defined()) target_host = (pf != nullptr) ? Target("llvm") : Target("stackvm");

        // Generate a placeholder function that attaches linked params as its arguments.
        if (target_host->GetAttr<Bool>("link-params").value_or(Bool(false))) {
          CHECK(pf != nullptr) << "Unable to link-params with no target_host and no llvm codegen.";
          auto param_ids = graph_codegen_->GetParamIds();
          auto link_params = Map<String, tir::LinkedParam>();
          for (auto param : ret_.params) {
            link_params.Set(param.first, tir::LinkedParam(param_ids[param.first], param.second));
          }

          Map<String, ObjectRef> dict;
          dict.Set(tvm::tir::attr::kLinkedParams, link_params);
          dict.Set(tvm::attr::kGlobalSymbol, String(::tvm::runtime::symbol::tvm_lookup_linked_param));
          DictAttrs attrs{dict};
          auto prim = tir::PrimFunc(Array<tir::Var>(), tir::SeqStmt(Array<tir::Stmt>()), VoidType(),
                                    Map<tir::Var, tir::Buffer>(), attrs);
          if (lowered_funcs.find(target_host->str()) == lowered_funcs.end()) {
            lowered_funcs.Set(target_host->str(), IRModule(Map<GlobalVar, BaseFunc>({})));
          }
          lowered_funcs[target_host->str()]->Add(
              GlobalVar(::tvm::runtime::symbol::tvm_lookup_linked_param), prim);
        }

        // When there is no lowered_funcs due to reasons such as optimization.
        if (lowered_funcs.size() == 0) {
          if (target_host.defined() && target_host->kind->name == "llvm") {
            // If we can decide the target is LLVM, we then create an empty LLVM module.
            ret_.mod = (*pf)(target_host->str(), "empty_module");
          } else {
            // If we cannot decide the target is LLVM, we create an empty CSourceModule.
            // The code content is initialized with ";" to prevent complaining
            // from CSourceModuleNode::SaveToFile.
            ret_.mod = tvm::codegen::CSourceModuleCreate(";", "", Array<String>{});
          }
        } else {
          ret_.mod = tvm::build(lowered_funcs, target_host_);
        }

        auto ext_mods = graph_codegen_->GetExternalModules();
        ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods, GetTargetHost());
      }
      
### GetTargetHost() code（Line 526)

     private:
      Target GetTargetHost() {
        Target target_host = target_host_;
        if (!target_host_.defined()) {
          for (const auto& it : targets_) {
            if (it.second->kind->device_type == kDLCPU) {
              target_host = it.second;
              break;
            }
          }
        }
        return target_host;
GetTargetHost():是改良之前的

    if (!ext_mods.empty()) {
          ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods);
        }

變成可以動態的決定device的type變成比較有可塑性，功能也比之前的好很多。

    

下面看到TVM逐步的動作

1. 計算圖優化: relay_module=Optimize(relay_module,targets_,params)
2. 計算圖生成
3. 後端運行代碼生成


# 計算圖優化

    IRModule RunDeviceAnnotationPass(const IRModule& relay_module, int fallback_device) {
        UpdateHeterogeneousInputs(fallback_device);
        auto rewrite = transform::RewriteAnnotatedOps(fallback_device);
        auto updated_module = rewrite(relay_module);
        ICHECK(updated_module.defined());

        tvm::Map<Expr, Integer> device_map;
        for (const auto& it : updated_module->functions) {
          device_map = relay::CollectDeviceInfo(it.second);
          if (!device_map.empty()) break;
        }

        if (device_map.empty()) {
          tvm::Map<Expr, Integer> annotation_map;
          for (const auto& it : relay_module->functions) {
            annotation_map = relay::CollectDeviceAnnotationOps(it.second);
            if (!annotation_map.empty()) break;
          }
          // None op is annotated but they are fallen back to the default device.
          if (annotation_map.empty()) {
            targets_.Set(0, CreateDefaultTarget(fallback_device));
          } else {
            // All ops are annotated to the same device type.
            int64_t dev_type = -1;
            for (auto kv : annotation_map) {
              dev_type = kv.second->value;
              break;
            }
            for (auto kv : annotation_map) {
              ICHECK_EQ(kv.second->value, dev_type) << "Expressions in the function are "
                                                    << "annotated with various device types,"
                                                    << "but not device copy operators "
                                                    << "found. Please check the "
                                                    << "RewriteAnnotation pass.";
            }
            targets_.Set(0, CreateDefaultTarget(dev_type));
          }
        }
        return updated_module;
      }

      /*!
       * \brief Compile a Relay IR module to runtime module.
       *
       * \param relay_module The Relay IR module.
       * \param params The parameters.
       */
      void BuildRelay(IRModule relay_module,
                      const std::unordered_map<std::string, tvm::runtime::NDArray>& params) {
        // Relay IRModule -> IRModule optimizations.
        relay_module = Optimize(relay_module, targets_, params);
        // Get the updated function.
        auto func = Downcast<Function>(relay_module->Lookup("main"));

        // Generate code for the updated function.
        graph_codegen_ = std::unique_ptr<GraphCodegen>(new GraphCodegen());
        graph_codegen_->Init(nullptr, targets_);
        graph_codegen_->Codegen(func);

        ret_.graph_json = graph_codegen_->GetJSON();
        ret_.params = graph_codegen_->GetParams();

        auto lowered_funcs = graph_codegen_->GetIRModule();

        Target target_host = GetTargetHost();
        // If no target_host has been set, we choose a default one, which is
        // llvm if "codegen.LLVMModuleCreate" is accessible.
        const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
        if (!target_host.defined()) target_host = (pf != nullptr) ? Target("llvm") : Target("stackvm");

        // Generate a placeholder function that attaches linked params as its arguments.
        if (target_host->GetAttr<Bool>("link-params").value_or(Bool(false))) {
          CHECK(pf != nullptr) << "Unable to link-params with no target_host and no llvm codegen.";
          auto param_ids = graph_codegen_->GetParamIds();
          auto link_params = Map<String, tir::LinkedParam>();
          for (auto param : ret_.params) {
            link_params.Set(param.first, tir::LinkedParam(param_ids[param.first], param.second));
          }

          Map<String, ObjectRef> dict;
          dict.Set(tvm::tir::attr::kLinkedParams, link_params);
          dict.Set(tvm::attr::kGlobalSymbol, String(::tvm::runtime::symbol::tvm_lookup_linked_param));
          DictAttrs attrs{dict};
          auto prim = tir::PrimFunc(Array<tir::Var>(), tir::SeqStmt(Array<tir::Stmt>()), VoidType(),
                                    Map<tir::Var, tir::Buffer>(), attrs);
          if (lowered_funcs.find(target_host->str()) == lowered_funcs.end()) {
            lowered_funcs.Set(target_host->str(), IRModule(Map<GlobalVar, BaseFunc>({})));
          }
          lowered_funcs[target_host->str()]->Add(
              GlobalVar(::tvm::runtime::symbol::tvm_lookup_linked_param), prim);
        }

        // When there is no lowered_funcs due to reasons such as optimization.
        if (lowered_funcs.size() == 0) {
          if (target_host.defined() && target_host->kind->name == "llvm") {
            // If we can decide the target is LLVM, we then create an empty LLVM module.
            ret_.mod = (*pf)(target_host->str(), "empty_module");
          } else {
            // If we cannot decide the target is LLVM, we create an empty CSourceModule.
            // The code content is initialized with ";" to prevent complaining
            // from CSourceModuleNode::SaveToFile.
            ret_.mod = tvm::codegen::CSourceModuleCreate(";", "", Array<String>{});
          }
        } else {
          ret_.mod = tvm::build(lowered_funcs, target_host_);
        }

        auto ext_mods = graph_codegen_->GetExternalModules();
        ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods, GetTargetHost());
      }

     private:
      Target GetTargetHost() {
        Target target_host = target_host_;
        if (!target_host_.defined()) {
          for (const auto& it : targets_) {
            if (it.second->kind->device_type == kDLCPU) {
              target_host = it.second;
              break;
            }
          }
        }
        return target_host;
      }

# 計算圖生成 GraphCodegen
path: tvm/src/relay/backend/build_module.cc

### code ( Line 60 ~ 117 )

    struct GraphCodegen {
     public:
      GraphCodegen() {
        auto pf = GetPackedFunc("relay.build_module._GraphRuntimeCodegen");
        mod = (*pf)();
      }
      ~GraphCodegen() {}

      void Init(runtime::Module* m, TargetsMap targets) { CallFunc("init", m, targets); }

      void Codegen(const Function& func) { CallFunc("codegen", func); }

      std::string GetJSON() { return CallFunc<std::string>("get_graph_json", nullptr); }

      Array<tvm::runtime::Module> GetExternalModules() {
        return CallFunc<Array<tvm::runtime::Module>>("get_external_modules", nullptr);
      }

      Map<String, IRModule> GetIRModule() {
        return CallFunc<Map<String, IRModule>>("get_irmodule", nullptr);
      }

      std::unordered_map<std::string, tvm::runtime::NDArray> GetParams() {
        std::unordered_map<std::string, tvm::runtime::NDArray> ret;
        auto names = CallFunc<Array<runtime::String>>("list_params_name", nullptr);
        for (const auto& expr : names) {
          // Implicit cast from runtime::String to std::string
          std::string key = expr;
          ret[key] = CallFunc<runtime::NDArray>("get_param_by_name", key);
        }
        return ret;
      }

      std::unordered_map<std::string, int64_t> GetParamIds() {
        std::unordered_map<std::string, int64_t> ret;
        auto names = CallFunc<Array<runtime::String>>("list_params_name", nullptr);
        for (const auto& expr : names) {
          // Implicit cast from runtime::String to std::string
          std::string key = expr;
          ret[key] = CallFunc<int64_t>("get_param_id", key);
        }
        return ret;
      }

     protected:
      tvm::runtime::Module mod;
      template <typename R, typename... Args>
      R CallFunc(const std::string& name, Args... args) {
        auto pf = mod.GetFunction(name, false);
        return pf(std::forward<Args>(args)...);
      }
      template <typename... Args>
      void CallFunc(const std::string& name, Args... args) {
        auto pf = mod.GetFunction(name, false);
        pf(std::forward<Args>(args)...);
        return;
      }
    };

# 實際使用的地方 ( Line 185 ~ 223 )
path : tvm/src/relay/backend/.graph_runtime_codegen.cc

    class GraphRuntimeCodegen : public backend::MemoizedExprTranslator<std::vector<GraphNodeRef>> {
     public:
      GraphRuntimeCodegen(runtime::Module* mod, const TargetsMap& targets) : mod_(mod) {
        compile_engine_ = CompileEngine::Global();
        targets_ = targets;
      }

      LoweredOutput Codegen(relay::Function func) {
        auto pf = GetPackedFunc("relay.backend.GraphPlanMemory");
        storage_device_map_ = (*pf)(func);
        // First we convert all the parameters into input nodes.
        for (auto param : func->params) {
          auto node_ptr = GraphInputNode::make_node_ptr(param->name_hint(), GraphAttrs());
          var_map_[param.get()] = AddNode(node_ptr, param);
        }
        heads_ = VisitExpr(func->body);
        std::ostringstream os;
        dmlc::JSONWriter writer(&os);
        GetJSON(&writer);
        LoweredOutput ret;
        ret.graph_json = os.str();
        ret.params = std::unordered_map<std::string, std::pair<int, const tvm::runtime::NDArray>>();
        for (auto param : params_) {
          ret.params.emplace(std::make_pair(
              param.first,
              std::make_pair(static_cast<int>(param_storage_ids_[param.first]), param.second)));
        }

        for (auto& kv : lowered_funcs_) {
          if (ret.lowered_funcs.count(kv.first) == 0) {
            ret.lowered_funcs.Set(kv.first, IRModule(Map<GlobalVar, BaseFunc>({})));
          }
          auto& mod = ret.lowered_funcs[kv.first];
          mod->Update(kv.second);
          ret.lowered_funcs.Set(kv.first, mod);
        }
        ret.external_mods = compile_engine_->LowerExternalFunctions();
        return ret;
      }
      ...
     }


# 後端程式碼生成
等relay做完以後就會交給tvm::build做code生成，跳到tvm/src/driver/driver_api.cc中的build函數

# code ( Line 256 ~ 304 )

    // Build for heterogeneous execution.
    runtime::Module build(const Map<Target, IRModule>& inputs, const Target& target_host) {
      auto pass_ctx = transform::PassContext::Current();

      std::vector<runtime::Module> device_modules;
      Target target_host_val = target_host;
      if (!target_host.defined()) {
        for (const auto& it : inputs) {
          if (it.first->kind->device_type == kDLCPU || it.first->kind->device_type == kDLMicroDev) {
            target_host_val = it.first;
            break;
          }
        }
      }

      if (!target_host_val.defined()) {
        target_host_val = DefaultTargetHost(target_host_val);
      }

      IRModule mhost_all = IRModule(Map<GlobalVar, BaseFunc>());

      ICHECK(mhost_all.defined()) << "The host module must be defined";

      for (const auto& it : inputs) {
        if (it.second.defined()) {
          auto pair = SplitDevHostFuncs(it.second, it.first, target_host_val, pass_ctx);
          auto& mhost = pair.first;
          auto& mdevice = pair.second;

          ICHECK(mhost.defined()) << "The split host module must be defined";

          ICHECK(mhost_all.defined()) << "The host module must be defined";

          mhost_all->Update(mhost);

          if (mdevice->functions.size() != 0) {
            device_modules.push_back(codegen::Build(mdevice, it.first));
          }
        }
      }

      runtime::Module mhost = codegen::Build(mhost_all, target_host_val);
      // Import all modules
      for (const auto& it : device_modules) {
        if (it.operator->()) {
          mhost.Import(it);
        }
      }
      return mhost;
    }


















