//===- SystolicArrayTimeLoop.cc -------------------------------------------===//
//
// This file implements passes that TODO
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "phism/mlir/Transforms/PhismTransforms.h"

#include "mlir/Analysis/CallGraph.h"
// #include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h" // 1) not found -> works when replaced by removing /Dialect/Affine/
#include "mlir/Analysis/AffineAnalysis.h"
// #include "mlir/Dialect/Affine/Analysis/Utils.h" // 2) not found -> works when  replaced by removing /Dialect/Affine/
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h" // 3) not found -> doesn't exist in LLVM 14, use mlir:: namespace instead of func::
// #include "mlir/Dialect/Linalg/IR/Linalg.h" // 4) not found -> seems to be not needed
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
// #include "mlir/Dialect/SCF/IR/SCF.h" // 5) not found -> works when replaced by removing /IR/
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
// #include "mlir/Tools/mlir-translate/Translation.h" // 6) not found -> works when replaced by removing /Tools/mlir-translate/
#include "mlir/Translation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;
using namespace phism;

#define DEBUG_TYPE "systolic-array-time-loop"

//----------------------------------------------------------------------------//
// SystolicArrayTimeLoopPass:
// * TODO
//----------------------------------------------------------------------------//

namespace {
struct SystolicArrayTimeLoopPipelineOptions : public mlir::PassPipelineOptions<SystolicArrayTimeLoopPipelineOptions> {
  Option<std::string> fileName{
    *this, "file-name",
    llvm::cl::desc("The output HLS code")
  };
};
} // namespace

static bool isNumberString(std::string s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it))
    ++it;
  return !s.empty() && it == s.end();
}

template <typename OpType>
static bool areVecSame(std::vector<OpType> &v1, std::vector<OpType> &v2) {
  if (v1.size() != v2.size())
    return false;
  return std::equal(v1.begin(), v1.begin() + v1.size(), v2.begin());
}

static int countSubstring(std::string pat, std::string txt) {
  int M = pat.length();
  int N = txt.length();
  int res = 0;

  /* A loop to slide pat[] one by one */
  for (int i = 0; i <= N - M; i++) {
    /* For current index i, check for
       pattern match */
    int j;
    for (j = 0; j < M; j++)
      if (txt[i + j] != pat[j])
        break;

    // if pat[0...M-1] = txt[i, i+1, ...i+M-1]
    if (j == M) {
      res++;
    }
  }
  return res;
}

// TODO llvm::DenseMap?
std::map<llvm::StringRef, mlir::FuncOp> PE_FuncOp_old_to_new_map;


static void handleCallerPE(mlir::CallOp PE_call_op) {
  LLVM_DEBUG({
    dbgs() << " * CallOp to handle:\n";
    PE_call_op.dump();
  });

  FuncOp newCallee = PE_FuncOp_old_to_new_map[PE_call_op.getCallee()];

  MLIRContext *context = PE_call_op.getContext();

  OpBuilder b(context);
  b.setInsertionPointAfter(PE_call_op);

  SmallVector<Value> operands;
  for (auto arg : PE_call_op.getOperands())
    operands.push_back(arg);

  Value some_constant = b.create<arith::ConstantIndexOp>(PE_call_op.getLoc(), 123);
  operands.push_back(some_constant);

  CallOp newCaller = b.create<CallOp>(
    PE_call_op.getLoc(),
    newCallee,
    operands
  );

  LLVM_DEBUG({
    dbgs() << " * New caller created:\n";
    newCaller.dump();
  });

  // Erase original CallOp
  PE_call_op->erase();


}

// TODO this could be done better with stringstream
static std::string convertArgumentTypes(const std::vector<std::string>& argument_types) {
  char delimiter = '.';

  std::string result = "";
  for (const auto &argument_type: argument_types) {
    result += argument_type + delimiter;
  }

  // Remove trailing delimiter
  result.pop_back();

  return result;
}

static void handleCalleePE(mlir::FuncOp PE_func_op) {
  LLVM_DEBUG({
    dbgs() << " * FuncOp to handle:\n";
    PE_func_op.dump();
  });

  MLIRContext *context = PE_func_op.getContext();

  OpBuilder b(context);


  // Add indexes to arguments and instantiate as local variables (e.g. for easier debug and monitoring)
  // New callee argument types.
  SmallVector<Type> newArgTypes;
  for (auto arg : PE_func_op.getArguments()) {
    newArgTypes.push_back(arg.getType());
  }
  // TODO this is last argument and not first to avoid conflicts when changing order of argument usage,
  // i.e. operations in blocks still refer to the old argument order
  Value zeroConstant = b.create<arith::ConstantIndexOp>(PE_func_op.getLoc(), 0);
  // TODO use setAttr to set name for EmitHLS (like idx etc) -> how to do it for a Value type
  // zeroConstant.setAttr("phism.name.idx", b.getUnitAttr());
  newArgTypes.push_back(zeroConstant.getType());
  

  // New callee function type.
  FunctionType newFuncType = b.getFunctionType(newArgTypes, PE_func_op->getResultTypes());
  b.setInsertionPointAfter(PE_func_op);
  FuncOp newCallee = b.create<FuncOp>(
    PE_func_op.getLoc(),
    std::string(PE_func_op.getName()),
    newFuncType
  );

  Block *entry = newCallee.addEntryBlock();
  b.setInsertionPointToEnd(entry);
  b.create<mlir::ReturnOp>(PE_func_op.getLoc());
  LLVM_DEBUG({
    dbgs() << " * New callee created (body empty):\n";
    newCallee.dump();
  });

  // Argument map.
  BlockAndValueMapping vmap;
  vmap.map(PE_func_op.getArguments(), newCallee.getArguments());

  // Iterate every operation in the original callee and clone it to the new one.
  b.setInsertionPointToStart(entry);
  for (Operation &op : PE_func_op.getBlocks().begin()->getOperations()) {
    if (isa<mlir::ReturnOp>(op))
      continue;
    b.clone(op, vmap);
  }

  // Annotate each local variable with pragmas (e.g. resource or array partition if needed)
  // newCallee.walk([&](Operation *op) {
  //   // for (auto result : op->getResults()) {
  //   //   // TODO how to give attributes to OpResult type?
  //   //   result->setAttr("phism.hls_pragma.resource", b.getUnitAttr());
  //   // }
  //   op->setAttr("phism.hls_pragma.resource", b.getUnitAttr());
  // });




  // Split affine for loops into smaller loops



  // Add systolic array specific I/O


  // Copy all original attributes
  newCallee->setAttr("phism.pe", b.getUnitAttr());
  // TODO why reusing all old args doesnt work? (i.e. causes issue with operand types/number)
  // newCallee->setAttrs(PE_func_op->getAttrs());

  // Add function pragmas
  newCallee->setAttr("phism.hls_pragma.inline", StringAttr::get(context, "OFF"));


  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
    "ap_fixed<8, 5>",
    "unsigned",
    "unsigned",
    "ap_fixed<8, 5>",
    "ap_fixed<8, 5>",
    "unsigned"
  };
  newCallee->setAttr(
    "phism.argument_types",
    StringAttr::get(context, convertArgumentTypes(argument_types))
  );

  // Link original PE function with the new one in a map, so that callers can get their arguments updated
  PE_FuncOp_old_to_new_map[PE_func_op.getName()] = newCallee;

  // Erase original PE function
  PE_func_op->erase();
}

// template <typename OpType>
// static bool contains(Block &block) {
//   for (auto &op : block) {
//     if (dyn_cast<OpType>(op)) {
//       return true;
//     }
//   }
//   return false;
// }


namespace {
class SystolicArrayTimeLoopPass : public phism::SystolicArrayTimeLoopPassBase<SystolicArrayTimeLoopPass> {
public:

  std::string fileName = "";

  SystolicArrayTimeLoopPass() = default;
  SystolicArrayTimeLoopPass(const SystolicArrayTimeLoopPipelineOptions & options)
    : fileName(!options.fileName.hasValue() ? "" : options.fileName.getValue()){
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Modify each PE callee
    SmallVector<mlir::FuncOp> PE_func_ops;
    m.walk([&](mlir::FuncOp op) {
      if (op->hasAttr("phism.pe"))
        PE_func_ops.push_back(op);
    });

    for (mlir::FuncOp op : PE_func_ops) {
      handleCalleePE(op);
    }
    
    // Update each PE caller arguments to match the new callee arguments
    SmallVector<mlir::CallOp> PE_call_ops;
    m.walk([&](mlir::CallOp op) {
      if (op->hasAttr("phism.pe"))
        PE_call_ops.push_back(op);
    });

    for (mlir::CallOp op : PE_call_ops) {
      handleCallerPE(op);
    }

  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
phism::createSystolicArrayTimeLoopPass() {
  return std::make_unique<SystolicArrayTimeLoopPass>();
}

void phism::registerSystolicArrayTimeLoopPass() {
  PassPipelineRegistration<SystolicArrayTimeLoopPipelineOptions>(
    "systolic-array-time-loop", "Systolic array time loop TODO.",
    [](OpPassManager &pm, const SystolicArrayTimeLoopPipelineOptions &options) {
      pm.addPass(std::make_unique<SystolicArrayTimeLoopPass>(options));
    }
  );
}