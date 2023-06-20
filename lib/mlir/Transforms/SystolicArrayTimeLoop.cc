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

static int indent = 0;
struct IdentRAII {
  int &indent;
  IdentRAII(int &indent) : indent(indent) {}
  ~IdentRAII() { --indent; }
};
void resetIndent() { indent = 0; }
IdentRAII pushIndent() { return IdentRAII(++indent); }

// llvm::raw_ostream &printIndent() {
//   for (int i = 0; i < indent; ++i)
//     llvm::outs() << "  ";
//   return llvm::outs();
// }

std::string printIndent() {
  std::string result = "";
  for (int i = 0; i < indent; ++i)
    result += "  ";
  return result;
}

void printOperation(Operation *op, const std::string& info = "");

void printBlock(Block &block, const std::string& info = "") {
  // Print the block intrinsics properties (basically: argument list)
  LLVM_DEBUG({dbgs() << printIndent()
      << info << "\n" << "Block with " << block.getNumArguments() << " arguments, "
      << block.getNumSuccessors()
      << " successors, and "
      << block.getOperations().size() << " operations\n";});

  // A block main role is to hold a list of Operations: let's recurse into
  // printing each operation.
  auto indent = pushIndent();
  for (Operation &op : block.getOperations())
    printOperation(&op);
}

void printRegion(Region &region, const std::string& info = "") {
  // A region does not hold anything by itself other than a list of blocks.
  LLVM_DEBUG({dbgs() << printIndent() << info << "\n" << "Region with " << region.getBlocks().size()
                << " blocks:\n";});
  auto indent = pushIndent();
  for (Block &block : region.getBlocks())
    printBlock(block);
}

void printOperation(Operation *op, const std::string& info/*= ""*/) {
  // Print the operation itself and some of its properties
  LLVM_DEBUG({dbgs() << printIndent() << info << "\n" << "visiting op: '" << op->getName() << "' with "
                << op->getNumOperands() << " operands and "
                << op->getNumResults() << " results\n";});
  // Print the operation attributes
  if (!op->getAttrs().empty()) {
    LLVM_DEBUG({dbgs() << printIndent() << op->getAttrs().size() << " attributes:\n";});
    for (NamedAttribute attr : op->getAttrs())
      LLVM_DEBUG({dbgs() <<  printIndent() << " - '" << attr.first << "' : '" << attr.second << "'\n";});
  }

  // Recurse into each of the regions attached to the operation.
  LLVM_DEBUG({dbgs() << printIndent() << " " << op->getNumRegions() << " nested regions:\n";});
  auto indent = pushIndent();
  for (Region &region : op->getRegions())
    printRegion(region);
}

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

static unsigned dummy_func_op_ID = 0;

static mlir::FuncOp createDummyFuncOp(mlir::FuncOp current_func_op, Type arg_type, Type res_type, std::string suffix, ConversionPatternRewriter& b){
  SmallVector<Type> func_result_types;
  func_result_types.push_back(res_type);
  SmallVector<Type> func_arg_types;
  func_arg_types.push_back(arg_type);

  b.setInsertionPointAfter(current_func_op); // Insertion point right before the current Func Op
  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    std::string(current_func_op.getName()) + "_" + suffix + "_" + std::to_string(dummy_func_op_ID++),
    b.getFunctionType(func_arg_types, func_result_types)
  );
  func_op->setAttr("phism.no_emit", b.getUnitAttr()); // TODO this could be added to constructor create
  // func_op->setAttr("phism.pe", b.getUnitAttr()); // TODO this could be added to constructor create

  LLVM_DEBUG({dbgs() << "Dummy func op:\n"; func_op.dump();});

  Block *entry = func_op.addEntryBlock();
  b.setInsertionPointToStart(entry);
  b.create<mlir::ReturnOp>(func_op.getLoc(), func_op.getArguments());

  LLVM_DEBUG({dbgs() << "Dummy func op after adding block with return op:\n"; func_op.dump();});

  return func_op;
}

static void initializeInputVariable(ConversionPatternRewriter &b, MLIRContext *context, Value v_in, FuncOp &callee, Block *innermost_block) {
  // Add a dummy func op that won't be emitted directly, but allows for custom .read() call
  memref::AllocaOp v_local = b.create<memref::AllocaOp>(callee.getLoc(), v_in.getType().cast<MemRefType>());
  FuncOp hls_stream_read_func_op = createDummyFuncOp(callee, v_in.getType(), v_local.getType(), "hls_stream_read", b);

  b.setInsertionPointToStart(innermost_block);
  
  SmallVector<Value> operands;
  operands.push_back(v_in);

  // Use a custom call op for reading from hls stream
  CallOp hls_stream_read = b.create<CallOp>(
    hls_stream_read_func_op.getLoc(),
    hls_stream_read_func_op,
    operands
  );
  hls_stream_read->setAttr("phism.hls_stream_read", b.getUnitAttr()); // TODO this could be added to constructor create
  LLVM_DEBUG({dbgs() << "hls_stream_read:\n"; hls_stream_read.dump();});

  // Add inner loop
  AffineForOp inner_loop = b.create<AffineForOp>(callee.getLoc(), 0, 2, 1);
  inner_loop->setAttr("phism.hls_pragma", StringAttr::get(context, "UNROLL")); // TODO this could be added to constructor create
  inner_loop->setAttr("phism.include_union_hack", b.getUnitAttr());// TODO this could be added to constructor create
  // Block *inner_block = b.createBlock(&inner_loop.getBody());
  // inner_loop.getBody();
  
  // local_A[0][n] = u.ut; -> memref::store op from u to local_A + attrbitured with ".ut"
  // store op takies vector of indexes, here: 0, n

  // TODO maybe use actual MLIR for union hack by having a custom callop that gets emitted as the two lines of u.ui -> u.ut instead of emit hacks?

  // TODO maybe use actual MLIR shift right instead of emit hacks?
  // auto thirty_two = b.create<arith::ConstantOp>(callee.getLoc(), b.getI32IntegerAttr(32));
  // arith::ShRUIOp shift_right_op = b.create<arith::ShRUIOp>(callee.getLoc(), hls_stream_read.getResults()[0], thirty_two);
}

static void handleCalleePE(mlir::FuncOp PE_func_op) {
  std::string IO_type = "IO_t";
  std::string local_type = "local_t";


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
  // printRegionInfo(loop0.region(), "After inlineRegionBefore");
  printOperation(loop0, "After inlineRegionBefore"); resetIndent();
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
  // printRegionInfo(loop0.region(), "After moving yield op and erasing block 0");
  printOperation(loop0, "After moving yield op and erasing block 0"); resetIndent();
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

  // Move old return op to the end of function call
  oldRet->moveAfter(loop0);
  printOperation(loop0, "After moving old return to the end of function"); resetIndent();
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

  // Replace original value uses with new values
  auto i = 0;
  for (auto arg : loop0.region().getArguments()) {
    arg.replaceAllUsesWith(newCallee.getArgument(i++));
  }
  printOperation(loop0, "After replaceAllUsesWith"); resetIndent();
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

  // Change outer affine for body block argument types to a single index + find innermost affine for op
  for (Block &block : loop0.region().getBlocks()) {
    // Erase all existing block argument types (that came from the original FuncOP) using BitVector of all 1s
    llvm::BitVector eraseIndices(block.getNumArguments(), true);
    block.eraseArguments(eraseIndices);

    // Add one new block argument type of index for the affine for induction variable
    block.addArgument(IndexType::get(context), newCallee.getLoc());
  }
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});
  SmallVector<AffineForOp> affine_for_ops;
  loop0.walk([&](AffineForOp op) {
    affine_for_ops.push_back(op);
  });
  LLVM_DEBUG({dbgs() << "Found " << affine_for_ops.size() << " affineForOps\n";});
  AffineForOp innermost_affine_for_op = affine_for_ops[0];
  printOperation(innermost_affine_for_op, "Found AffineForOp"); resetIndent();

  // Find innermost block
  Block* innermost_block = &(innermost_affine_for_op->getRegions().front().getBlocks().front());
  printBlock(*innermost_block, "Found innermost_block"); resetIndent();
  LLVM_DEBUG({dbgs() << "Found innermost_block: \n";});

  // Add systolic array specific I/O
  // 1. Load A_in to A_local
  Value A_in = newCallee.getArguments()[3];
  initializeInputVariable(b, context, A_in, newCallee, innermost_block);
  
  // 2. Load B_in to B_local
  Value B_in = newCallee.getArguments()[4];
  initializeInputVariable(b, context, B_in, newCallee, innermost_block);

  // 3. C_local := op(A_local, B_local) or 0
  // 4. C_local drain to C_out
  // 5. Drain B_local to B_out
  // 6. Drain A_local to A_out










  // Copy all original attributes
  newCallee->setAttr("phism.pe", b.getUnitAttr());
  // TODO why reusing all old args doesnt work? (i.e. causes issue with operand types/number)
  // newCallee->setAttrs(PE_func_op->getAttrs());

  // Add function pragmas
  newCallee->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE OFF"));


  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
    "hls::stream<" + IO_type + ">",
    "unsigned",
    "unsigned",
    "hls::stream<" + IO_type + ">",
    "hls::stream<" + IO_type + ">",
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