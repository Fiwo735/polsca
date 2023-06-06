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

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/BitVector.h"

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

  newCaller->setAttr("phism.pe", b.getUnitAttr());

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

static void printRegionInfo(Region &region, const std::string& info = "") {
  LLVM_DEBUG({dbgs() << "----------------------" << info << "----------------------\n";});
  LLVM_DEBUG({dbgs() << "Region with " << region.getBlocks().size() << " blocks:\n";});

  for (Block &block : region.getBlocks()) {
    LLVM_DEBUG({dbgs() << "\t" << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors() << " successors, and "
        << block.getOperations().size() << " operations\n";});

    for (Operation &op : block.getOperations()){
      LLVM_DEBUG({dbgs() << "\t\t" << "Visiting op: '" << op.getName() << "' with "<< op.getNumOperands() << " operands and " << op.getNumResults() << " results\n";});
    }
  }

  LLVM_DEBUG({dbgs() << "\n";});
}

static void handleCalleePE(mlir::FuncOp PE_func_op) {
  LLVM_DEBUG({dbgs() << " * FuncOp to handle:\n"; PE_func_op.dump();});

  MLIRContext *context = PE_func_op.getContext();
  // OpBuilder b(context);
  ConversionPatternRewriter b(context);

  // Save pointer to original yield op
  Operation *oldRet = PE_func_op.getBody().back().getTerminator();

  // Add indexes to arguments and instantiate as local variables (e.g. for easier debug and monitoring)
  SmallVector<Type> newArgTypes;
  for (auto arg : PE_func_op.getArguments()) {
    newArgTypes.push_back(arg.getType());
  }
  // TODO this is last argument and not first to avoid conflicts when changing order of argument usage,
  // i.e. operations in blocks still refer to the old argument order
  newArgTypes.push_back(IndexType::get(context));
  
  // New callee function type.
  FunctionType newFuncType = b.getFunctionType(newArgTypes, PE_func_op->getResultTypes());
  b.setInsertionPointAfter(PE_func_op);
  FuncOp newCallee = b.create<FuncOp>(
    PE_func_op.getLoc(),
    std::string(PE_func_op.getName()),
    newFuncType
  );

  // Create a block in the new Func Op and add original arguments to it
  Block *block = b.createBlock(&newCallee.getBody());
  for (auto argType : newArgTypes) {
    block->addArgument(argType, newCallee.getLoc());
  }

  // Add additional loops
  b.setInsertionPointToStart(block);
  AffineForOp loop0 = b.create<AffineForOp>(PE_func_op.getLoc(), 0, 100, 5);

  // Copy PE_func_op body into loop0 by inlining
  b.setInsertionPointToStart(loop0.getBody());
  b.inlineRegionBefore(PE_func_op.getBody(), loop0.region(),  loop0.region().end());
  printRegionInfo(loop0.region(), "After inlineRegionBefore");
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

  // Move affine yield (created by default during create<AffineForOp>) after the inlined region,
  // i.e. to the end of outer loop body
  for (Block &block : loop0.region().getBlocks()) {
    block.getTerminator()->moveAfter(oldRet);
    block.erase();
    break;
    // TODO instead of breaking we could have if statement checking getTerminator type
    // return op -> moveafter + erase block | yield op -> moveafter ===> this would avoid saving and using oldRet
  }
  printRegionInfo(loop0.region(), "After moving yield op and erasing block 0");
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

  // Move old return op to the end of function call
  oldRet->moveAfter(loop0);
  printRegionInfo(loop0.region(), "After moving old return to the end of function");
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

  // Replace original value uses with new values
  auto i = 0;
  for (auto arg : loop0.region().getArguments()) {
    arg.replaceAllUsesWith(newCallee.getArgument(i++));
  }
  printRegionInfo(loop0.region(), "After replaceAllUsesWith");
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

  // Change outer affine for body block argument types to a single index
  for (Block &block : loop0.region().getBlocks()) {
    // Erase all existing block argument types (that came from the original FuncOP) using BitVector of all 1s
    llvm::BitVector eraseIndices(block.getNumArguments(), true);
    block.eraseArguments(eraseIndices);

    // Add one new block argument type of index for the affine for induction variable
    block.addArgument(IndexType::get(context), newCallee.getLoc());
  }
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

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
  // LLVM_DEBUG({
  //   dbgs() << "New callee created:\n";
  //   newCallee.dump();
  // });
  LLVM_DEBUG({dbgs() << "About to erase PE_func_op\n";});
  PE_func_op->erase();
}

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