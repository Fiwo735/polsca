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
      << info << "\n" << printIndent() << "Block with " << block.getNumArguments() << " arguments, "
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
  LLVM_DEBUG({dbgs() << printIndent() << info << "\n" << printIndent() << "Region with " << region.getBlocks().size()
                << " blocks:\n";});
  auto indent = pushIndent();
  for (Block &block : region.getBlocks())
    printBlock(block);
}

void printOperation(Operation *op, const std::string& info/*= ""*/) {
  // Print the operation itself and some of its properties
  LLVM_DEBUG({dbgs() << printIndent() << info << "\n" << printIndent() << "visiting op: '" << op->getName() << "' with "
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

static inline std::string makeHLSStream(const std::string &t) {
  return "hls::stream<" + t + ">";
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

  operands.push_back(PE_call_op.getOperands()[3]);
  operands.push_back(PE_call_op.getOperands()[4]);

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
static std::string vec2str(const std::vector<std::string>& argument_types, char delimiter = '.') {
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

static std::pair<StringAttr, Attribute> getStringAttrPair(const std::string &s, ConversionPatternRewriter &b) {
  return std::pair<StringAttr, Attribute>(b.getStringAttr(s), b.getUnitAttr());
}

static unsigned dummy_func_op_ID = 0; // avoids error of function redefinition for dummy functions with the same name

static FuncOp createDummyFuncOp(FuncOp current_func_op, SmallVector<Type> arg_types, SmallVector<Type> res_types,
                                std::string suffix, ConversionPatternRewriter &b) {
  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);
  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    std::string(current_func_op.getName()) + "_" + suffix + "_" + std::to_string(dummy_func_op_ID++),
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.no_emit", b.getUnitAttr()); // TODO this could be added to constructor create

  LLVM_DEBUG({dbgs() << "Dummy func op:\n"; func_op.dump();});

  Block *entry = func_op.addEntryBlock();
  b.setInsertionPointToStart(entry);
  b.create<ReturnOp>(func_op.getLoc(), func_op.getArguments()[0]); // TODO this could be a void function, i.e. w/o ReturnOp?

  LLVM_DEBUG({dbgs() << "Dummy func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static Value initializeInputVariable(ConversionPatternRewriter &b, MLIRContext *context, Value v_in, FuncOp &callee,
                                     const std::string &variable_name, const std::string &type_name) {
  // Add a dummy func op that won't be emitted directly, but allows for custom .read() call
  memref::AllocaOp v_local = b.create<memref::AllocaOp>(callee.getLoc(), v_in.getType().cast<MemRefType>());
  v_local->setAttr("phism.hls_pragma", StringAttr::get(context, "ARRAY_PARTITION variable=$ dim=0 complete"));
  v_local->setAttr("phism.variable_name", StringAttr::get(context, variable_name));
  v_local->setAttr("phism.type_name", StringAttr::get(context, type_name));

  FuncOp hls_stream_read_func_op = createDummyFuncOp(callee, {v_in.getType()}, {v_local.getType()}, "hls_stream_read", b);

  SmallVector<Value> operands{v_in};

  // Use a custom call op for reading from hls stream
  // llvm::ArrayRef<std::pair<mlir::StringAttr, mlir::Attribute>> ppp{getStringAttrPair("phism.hls_stream_read", b)};
  CallOp hls_stream_read = b.create<CallOp>(
    hls_stream_read_func_op.getLoc(),
    // hls_stream_read_func_op->getResultTypes(),
    hls_stream_read_func_op,
    operands
    // ppp
  );
  hls_stream_read->setAttr("phism.hls_stream_read", b.getUnitAttr()); // TODO this could be added to constructor create
  LLVM_DEBUG({dbgs() << "hls_stream_read:\n"; hls_stream_read.dump();});

  hls_stream_read.getResults()[0].getDefiningOp()->setAttr("phism.type_name", StringAttr::get(context, type_name));

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
  // b.setInsertionPointToStart(inner_loop.getBody());
  // arith::ConstantIndexOp thirty_two = b.create<arith::ConstantIndexOp>(callee.getLoc(), 32);
  // arith::ShRUIOp shift_right_op = b.create<arith::ShRUIOp>(callee.getLoc(), hls_stream_read.getResults()[0], thirty_two);

  b.setInsertionPointAfter(inner_loop);
  return v_local;
}

// static Block* findInnermostBlock(AffineForOp for_op) {
//   SmallVector<AffineForOp> affine_for_ops;
//   for_op.walk([&](AffineForOp op) {
//     affine_for_ops.push_back(op);
//   });
//   LLVM_DEBUG({dbgs() << "Found " << affine_for_ops.size() << " affineForOps\n";});
//   AffineForOp innermost_affine_for_op = affine_for_ops[0];
//   printOperation(innermost_affine_for_op, "Found AffineForOp"); resetIndent();

//   // Find innermost block
//   return &(innermost_affine_for_op->getRegions().front().getBlocks().front());
// }

static void reduceBlockArguments(Block &block, Type new_type, Location loc) {
  llvm::BitVector eraseIndices(block.getNumArguments(), true);
  block.eraseArguments(eraseIndices);
  // Add one new block argument type for the affine for induction variable
  block.addArgument(new_type, loc);
}

static void propagateUses(AffineForOp for_loop, FuncOp func_op, Type block_argument_type) {
  unsigned i = 0;
  for (auto arg : for_loop.region().getArguments()) {
    arg.replaceAllUsesWith(func_op.getArgument(i++));
  }

  reduceBlockArguments(for_loop.region().getBlocks().front(), block_argument_type, func_op.getLoc());
}

static CallOp createWriteCallOp(ConversionPatternRewriter &b, Value from, Value to, FuncOp &callee) {
  FuncOp hls_stream_write_func_op = createDummyFuncOp(callee, {to.getType(), from.getType()}, {from.getType()}, "hls_stream_write", b);

  SmallVector<Value, 2> operands{to, from};

  // Use a custom call op for writing to hls_stream
  CallOp hls_stream_write = b.create<CallOp>(
    // callee.getLoc(), // TODO which to use?
    hls_stream_write_func_op.getLoc(),
    hls_stream_write_func_op,
    operands
  );
  hls_stream_write->setAttr("phism.hls_stream_write", b.getUnitAttr()); // TODO this could be added to constructor create
  LLVM_DEBUG({dbgs() << "hls_stream_write:\n"; hls_stream_write.dump();});

  return hls_stream_write;
}

static scf::IfOp createIfWithConstantConditions(ConversionPatternRewriter &b, Location loc, const std::vector<std::pair<Value, int>>& cond_pairs) {
  // Structure: if (... & value == equal_to & ...)
  Value condition = nullptr;
  for (const auto& [value, equal_to] : cond_pairs) {
    Value value_to_equal_to = b.create<arith::ConstantIndexOp>(loc, equal_to);
    Value curr_condition = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, value, value_to_equal_to);
    condition = ((condition == nullptr) ? curr_condition : b.create<arith::AndIOp>(loc, condition, curr_condition));
  }

  scf::IfOp compute_if = b.create<scf::IfOp>(loc, condition, /*hasElse*/ false);

  b.setInsertionPointToStart(&(compute_if.thenRegion().front()));
  return compute_if;
}

static llvm::simple_ilist<mlir::Operation>::iterator getFirstOp(AffineForOp op) {
  return op.region().getBlocks().front().getOperations().begin();
}

static void handleCalleePE(FuncOp PE_func_op) {
  std::string IO_type = "ap_fixed<64,32>";

  LLVM_DEBUG({dbgs() << " * FuncOp to handle:\n"; PE_func_op.dump();});

  MLIRContext *context = PE_func_op.getContext();
  ConversionPatternRewriter b(context);

  // Save pointer to original yield op
  Operation *oldRet = PE_func_op.getBody().back().getTerminator();

  // Add indexes to arguments and instantiate as local variables (e.g. for easier debug and monitoring)
  SmallVector<Type> newArgTypes;
  for (auto arg : PE_func_op.getArguments()) {
    newArgTypes.push_back(arg.getType());
  }
  newArgTypes.push_back(PE_func_op.getArguments()[3].getType());
  newArgTypes.push_back(PE_func_op.getArguments()[4].getType());
  // TODO this is last argument and not first to avoid conflicts when changing order of argument usage,
  // i.e. operations in blocks still refer to the old argument order
  // TODO maybe this could be made into first argument
  newArgTypes.push_back(IndexType::get(context));
  
  // New callee function type.
  FunctionType newFuncType = b.getFunctionType(newArgTypes, PE_func_op->getResultTypes());
  b.setInsertionPointAfter(PE_func_op);
  FuncOp newCallee = b.create<FuncOp>(
    PE_func_op.getLoc(),
    std::string(PE_func_op.getName()),
    newFuncType
  );

  Location loc = newCallee.getLoc();

  // Create a block in the new Func Op and add original arguments to it
  Block *block = b.createBlock(&newCallee.getBody());
  for (auto argType : newArgTypes) {
    block->addArgument(argType, loc);
  }

  // Add additional loops
  b.setInsertionPointToStart(block);
  AffineForOp loop0 = b.create<AffineForOp>(loc, 0, 2, 1);

  b.setInsertionPointToStart(loop0.getBody());
  AffineForOp loop1 = b.create<AffineForOp>(loc, 0, 2, 1);

  b.setInsertionPointToStart(loop1.getBody());
  AffineForOp loop2 = b.create<AffineForOp>(loc, 0, 2, 1);

  b.setInsertionPointToStart(loop2.getBody());
  AffineForOp loop3 = b.create<AffineForOp>(loc, 0, 8, 1);

  b.setInsertionPointToStart(loop3.getBody());
  AffineForOp loop4 = b.create<AffineForOp>(loc, 0, 8, 1);

  b.setInsertionPointToStart(loop4.getBody());
  AffineForOp loop5 = b.create<AffineForOp>(loc, 0, 8, 1);

  loop5->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));

  // 3. C_local := op(A_local, B_local) or 0
  // b.setInsertionPointToStart(loop0.getBody());
  b.setInsertionPointToStart(loop5.getBody());
  AffineForOp compute_loop = b.create<AffineForOp>(loc, 0, 2, 1);
  compute_loop->setAttr("phism.hls_pragma", StringAttr::get(context, "UNROLL")); // TODO this could be added to constructor create

  // Copy PE_func_op body into loop0 by inlining
  b.setInsertionPointToStart(compute_loop.getBody());
  b.inlineRegionBefore(PE_func_op.getBody(), compute_loop.region(),  compute_loop.region().end());
  printOperation(loop0, "After inlineRegionBefore"); resetIndent();
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

  // Link original PE function with the new one in a map, so that callers can get their arguments updated and then erase it
  LLVM_DEBUG({dbgs() << "Erasing PE_func_op\n";});
  PE_FuncOp_old_to_new_map[PE_func_op.getName()] = newCallee;
  PE_func_op->erase();

  // Move affine yield (created by default during create<AffineForOp>) at the end of loop0
  compute_loop.region().getBlocks().front().getTerminator()->moveAfter(&(loop0.region().getBlocks().back().getOperations().back()));
  compute_loop.region().getBlocks().front().erase();
  printOperation(compute_loop, "compute_loop after reordering affine yield"); resetIndent();

  // Move affine yield (created by default during create<AffineForOp>) after the inlined region, i.e. to the end of outer loop body
  loop0.region().getBlocks().front().getTerminator()->moveAfter(oldRet);
  printOperation(loop0, "After moving yield op and erasing block 0"); resetIndent();
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

  // Move old return op to the end of function call
  oldRet->moveAfter(loop0);
  printOperation(loop0, "After moving old return to the end of function"); resetIndent();
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

  // Replace original value uses with new values
  propagateUses(compute_loop, newCallee, IndexType::get(context));
  printOperation(compute_loop, "After replaceAllUsesWith and reduceBlockArguments for compute_loop"); resetIndent();
  LLVM_DEBUG({dbgs() << "Callee:\n"; newCallee.dump();});

  // Add systolic array specific I/O
  b.setInsertionPoint(compute_loop);
  // 1. Load A_in to A_local
  Value A_in = newCallee.getArguments()[3];
  Value A_local = initializeInputVariable(b, context, A_in, newCallee, "local_A", IO_type);
  LLVM_DEBUG({dbgs() << "A_in -> A_local done\n";});
  // memref::LoadOp loaded = b.create<memref::LoadOP>(loc, ); // TODO maybe LoadOp with custom attribute to make it into .read() would make more sense?
  
  // 2. Load B_in to B_local
  Value B_in = newCallee.getArguments()[4];
  Value B_local = initializeInputVariable(b, context, B_in, newCallee, "local_B", IO_type);
  LLVM_DEBUG({dbgs() << "B_in -> B_local done\n";});

  printOperation(newCallee, "newCallee after steps 1. and 2.");

  
  // 3. Zeroing condition for compute operation
  // SmallVector<Value, 4> lbOperands(loop2.getLowerBoundOperands());
  // LLVM_DEBUG({dbgs() << "loop2.getLowerBoundOperands() obtained\n";});
  // for (auto v : lbOperands) {
  //   LLVM_DEBUG({dbgs() << "loop2.getLowerBoundOperands(): " << v << "\n";});
  // }

  // auto lbMap = loop3.getLowerBoundMap();
  // LLVM_DEBUG({dbgs() << "loop3.getLowerBoundMap() obtained: " << lbMap << "\n";});
  // auto ubMap = loop3.getUpperBoundMap();
  // LLVM_DEBUG({dbgs() << "loop3.getUpperBoundMap() obtained: " << ubMap << "\n";});

  // LLVM_DEBUG({dbgs() << "loop3.getLowerBound() obtained: " << loop3.getLowerBound().getOperand(0) << "\n";});
  // LLVM_DEBUG({dbgs() << "loop3.getUpperBound() obtained: " << loop3.getUpperBound().getOperand(0) << "\n";});

  // SmallVector<Value, 4> ubOperands(loop2.getUpperBoundOperands());
  // for (auto v : ubOperands) {
  //   LLVM_DEBUG({dbgs() << "loop2.getUpperBoundOperands(): " << v << "\n";});
  // }

  // TODO how to obtain numerical value of lower and upper bound of affine for op?

  scf::IfOp compute_if = createIfWithConstantConditions(b, loc, {{loop2.getInductionVar(), 0}, {loop3.getInductionVar(), 0}});
  memref::AllocaOp C_local = b.create<memref::AllocaOp>(loc, newCallee.getArguments()[0].getType().cast<MemRefType>());
  C_local->setAttr("phism.variable_name", StringAttr::get(context, "local_C"));
  C_local->setAttr("phism.type_name", StringAttr::get(context, IO_type));

  // arith::ConstantIntOp i_zero = b.create<arith::ConstantIntOp>(loc, 15, 32);
  // arith::ConstantIndexOp i0 = b.create<arith::ConstantIndexOp>(loc, 5432);
  // SmallVector<Value, 2> idxs = {i0, i0};
  // SmallVector<Value, 2> idxs = {loop0.getInductionVar(), loop0.getInductionVar()};
  // memref::StoreOp C_local_store = b.create<memref::StoreOp>(loc, i_zero, C_local, idxs);
  // TODO debug above 'memref.store' op failed to verify that type of 'value' matches element type of 'memref'

  // TODO temporary solution
  compute_if->setAttr("phism.zero_local_C", b.getUnitAttr());

  // Modify compute loop insides
  printBlock(compute_loop.region().getBlocks().front(), "compute loop before:");

  // Delete zero initializing as it's already handled (in reverse order to avoid ever being in an illegal state)
  (++getFirstOp(compute_loop))->erase();
  getFirstOp(compute_loop)->erase();
  printBlock(compute_loop.region().getBlocks().front(), "compute loop after erasing zero initializing:");

  // Replace uses in original computation with new SA variables
  AffineForOp inner_compute_loop = dyn_cast<AffineForOp>(*getFirstOp(compute_loop));
  LLVM_DEBUG({dbgs() << "*getFirstOp(inner_compute_loop):" << *getFirstOp(inner_compute_loop) << "\n";});
  LLVM_DEBUG({dbgs() << "*(++getFirstOp(inner_compute_loop)):" << *(++getFirstOp(inner_compute_loop)) << "\n";});
  
  // getFirstOp(inner_compute_loop)->replaceAllUsesWith(C_local); // This works

  // (++getFirstOp(inner_compute_loop))->replaceAllUsesWith(A_local); // TODO This doesn't work -> why?
  // (++(++getFirstOp(inner_compute_loop)))->replaceAllUsesWith(B_local);



  printBlock(compute_loop.region().getBlocks().front(), "compute loop after replaceAllUsesWith:");

  // static void reduceBlockArguments(Block &block, Type new_type, Location loc) {
  //   llvm::BitVector eraseIndices(block.getNumArguments(), true);
  //   block.eraseArguments(eraseIndices);
  //   // Add one new block argument type for the affine for induction variable
  //   block.addArgument(new_type, loc);
  // }

  // static void propagateUses(AffineForOp for_loop, FuncOp func_op, Type block_argument_type) {
  //   unsigned i = 0;
  //   for (auto arg : for_loop.region().getArguments()) {
  //     arg.replaceAllUsesWith(func_op.getArgument(i++));
  //   }

  //   reduceBlockArguments(for_loop.region().getBlocks().front(), block_argument_type, func_op.getLoc());
  // }
  

  
  b.setInsertionPointAfter(compute_loop);

  // 4. C_local drain to C_out
  scf::IfOp C_drain_if = createIfWithConstantConditions(b, loc, {{loop2.getInductionVar(), 1}, {loop3.getInductionVar(), 7}});
  // CallOp write_C_local_call_op = createWriteCallOp(b, C_local, newCallee.getArguments()[0], newCallee);
  // TODO why the above causes "error: operand #1 does not dominate this use"?

  // TODO temporary solution
  C_drain_if->setAttr("phism.write_C_drain", b.getUnitAttr());

  b.setInsertionPointAfter(C_drain_if);

  // 5. Drain B_local to B_out
  CallOp write_B_local_call_op = createWriteCallOp(b, B_local, newCallee.getArguments()[6], newCallee);
  // memref::AllocaOp u = b.create<memref::AllocaOp>(loc, newCallee.getArguments()[0].getType().cast<MemRefType>());
  // u->setAttr("phism.type_name", StringAttr::get(context, "union {unsigned int ui; float ut;}"));

  // arith::ConstantIndexOp ui = b.create<arith::ConstantIndexOp>(loc, 999);
  // SmallVector<Value, 2> u_idxs = {ui, ui};
  // memref::StoreOp u_store = b.create<memref::StoreOp>(loc, u, u, u_idxs);


  write_B_local_call_op->setAttr("phism.include_reverse_union_hack", b.getUnitAttr());

  // 6. Drain A_local to A_out
  // Block *new_block = newCallee.addBlock();
  // new_block->addArgument(A_local.getType(), loc);
  // new_block->addArgument(newCallee.getArguments()[3].getType(), loc);
  // b.setInsertionPointToStart(new_block);
  // TODO error: use of undeclared SSA value name?

  CallOp write_A_local_call_op = createWriteCallOp(b, A_local, newCallee.getArguments()[5], newCallee);
  write_A_local_call_op->setAttr("phism.include_reverse_union_hack", b.getUnitAttr());

  // b.create<ReturnOp>(loc);

  //////////////
  printOperation(newCallee, "newCallee after steps 5. and 6.");

  // Copy all original attributes
  newCallee->setAttr("phism.pe", b.getUnitAttr());
  // TODO why reusing all old args doesnt work? (i.e. causes issue with operand types/number)
  // newCallee->setAttrs(PE_func_op->getAttrs());

  // Add function pragmas
  newCallee->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE OFF"));


  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
    makeHLSStream(IO_type),
    "unsigned",
    "unsigned",
    makeHLSStream(IO_type),
    makeHLSStream(IO_type),
    makeHLSStream(IO_type),
    makeHLSStream(IO_type),
    "unsigned"
  };
  newCallee->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  std::vector<std::string> argument_names = {
    "fifo_C_drain_out",
    "unwanted_index0",
    "unwanted_index1",
    "fifo_A_in",
    "fifo_B_in",
    "fifo_A_out",
    "fifo_B_out",
    "wanted_index"
  };
  newCallee->setAttr(
    "phism.argument_names",
    StringAttr::get(context, vec2str(argument_names))
  );


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