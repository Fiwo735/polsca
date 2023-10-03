//===- SystolicArraySpaceLoop.cc ------------------------------------------===//
//
// This file implements passes that TODO
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "phism/mlir/Transforms/PhismTransforms.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Translation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Block.h"

#include "mlir/Transforms/Utils.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;
using namespace phism;

#define DEBUG_TYPE "systolic-array-space-loop"

//----------------------------------------------------------------------------//
// SystolicArraySpaceLoopPass:
// * TODO
//----------------------------------------------------------------------------//

namespace {
struct SystolicArraySpaceLoopPipelineOptions : public mlir::PassPipelineOptions<SystolicArraySpaceLoopPipelineOptions> {
  Option<std::string> fileName{
    *this, "file-name",
    llvm::cl::desc("The output HLS code")
  };
};
} // namespace

static inline std::string makeHLSStream(const std::string &t) {
  return "hls::stream<" + t + ">";
}

static inline std::string makePointer(const std::string &t) {
  return t + "*";
}

static std::string vec2str(const std::vector<std::string>& argument_types, char delimiter = '.') {
  std::string result = "";
  for (const auto &argument_type: argument_types) {
    result += argument_type + delimiter;
  }

  // Remove trailing delimiter
  result.pop_back();

  return result;
}

static unsigned dummy_func_op_ID = 0; // avoids error of function redefinition for dummy functions with the same name

static FuncOp createDummyFuncOp(FuncOp current_func_op, SmallVector<Type> arg_types, SmallVector<Type> res_types,
                                std::string suffix, ConversionPatternRewriter &b, bool return_inner_type = false) {
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

  if (return_inner_type) {
    // Value value_to_return = b.create<memref::LoadOp>(func_op.getLoc(), func_op.getArguments()[0], SmallVector<Value, 2>({0, 0}));
    // b.create<ReturnOp>(func_op.getLoc(), value_to_return);
    // TODO why does the above not work? -> runtime crash :( it would be better to avoid removal during canonicalize

    Value f_zero = b.create<arith::ConstantOp>(func_op.getLoc(), FloatAttr::get(b.getF64Type(), 0.0));
    b.create<ReturnOp>(func_op.getLoc(), f_zero);
  }
  else {
    b.create<ReturnOp>(func_op.getLoc(), func_op.getArguments()[0]); // TODO this could be a void function, i.e. w/o ReturnOp?
  }

  LLVM_DEBUG({dbgs() << "Dummy func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static CallOp createWriteCallOp(ConversionPatternRewriter &b, Value from, Value to, FuncOp &callee) {
  FuncOp hls_stream_write_func_op = createDummyFuncOp(callee, {to.getType(), from.getType()}, {to.getType()}, "hls_stream_write", b);

  SmallVector<Value, 2> operands{to, from};

  // Use a custom call op for writing to hls_stream
  CallOp hls_stream_write = b.create<CallOp>(
    hls_stream_write_func_op.getLoc(),
    hls_stream_write_func_op,
    operands
  );
  hls_stream_write->setAttr("phism.hls_stream_write", b.getUnitAttr()); // TODO this could be added to constructor create
  LLVM_DEBUG({dbgs() << "hls_stream_write:\n"; hls_stream_write.dump();});

  return hls_stream_write;
}

static CallOp createReadCallOp(ConversionPatternRewriter &b, Value from, FuncOp &callee) {
  FuncOp hls_stream_read_func_op = createDummyFuncOp(callee, {from.getType()}, {b.getF64Type()}, "hls_stream_read", b, /*return_inner_type*/ true);

  SmallVector<Value, 1> operands{from};

  // Use a custom call op for writing to hls_stream
  CallOp hls_stream_read = b.create<CallOp>(
    hls_stream_read_func_op.getLoc(),
    hls_stream_read_func_op,
    operands
  );
  hls_stream_read->setAttr("phism.hls_stream_read", b.getUnitAttr()); // TODO this could be added to constructor create
  LLVM_DEBUG({dbgs() << "hls_stream_write:\n"; hls_stream_read.dump();});

  return hls_stream_read;
}

static scf::IfOp createIfWithConstantConditions(ConversionPatternRewriter &b, Location loc, const std::vector<std::pair<Value, int>>& cond_pairs, bool has_else = false) {
  // Structure: if (... & value == equal_to & ...)
  Value condition = nullptr;
  for (const auto& [value, equal_to] : cond_pairs) {
    Value inner_value = value;

    // If shaped type (i.e. most likely a 1 element array mimicking a variable) then load the underying value
    bool is_shaped_type = (value.getType().dyn_cast<ShapedType>() != nullptr);
    if (is_shaped_type) {
      arith::ConstantIndexOp index_zero = b.create<arith::ConstantIndexOp>(loc, 0);
      inner_value = b.create<memref::LoadOp>(loc, value, SmallVector<Value, 1>({index_zero}));
    }

    unsigned width = inner_value.getType().dyn_cast<IntegerType>().getWidth();
    Value value_to_equal_to = b.create<arith::ConstantOp>(loc, IntegerAttr::get(b.getIntegerType(width), equal_to));

    Value curr_condition = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, inner_value, value_to_equal_to);

    curr_condition.getDefiningOp()->setAttr("phism.is_condition", b.getUnitAttr());
    condition = ((condition == nullptr) ? curr_condition : b.create<arith::AndIOp>(loc, condition, curr_condition));
    condition.getDefiningOp()->setAttr("phism.is_condition", b.getUnitAttr());
  }

  scf::IfOp compute_if = b.create<scf::IfOp>(loc, condition, /*hasElse*/ has_else);

  b.setInsertionPointToStart(&(compute_if.thenRegion().front()));
  return compute_if;
}

static scf::IfOp createIfWithConditions(ConversionPatternRewriter &b, Location loc, const std::vector<std::pair<Value, Value>>& cond_pairs, bool has_else = false) {
  // Structure: if (... & value == equal_to & ...)
  Value condition = nullptr;
  for (const auto& [value, equal_to] : cond_pairs) {
    Value inner_value = value;

    // If shaped type (i.e. most likely a 1 element array mimicking a variable) then load the underying value
    bool is_shaped_type = (value.getType().dyn_cast<ShapedType>() != nullptr);
    if (is_shaped_type) {
      arith::ConstantIndexOp index_zero = b.create<arith::ConstantIndexOp>(loc, 0);
      inner_value = b.create<memref::LoadOp>(loc, value, SmallVector<Value, 1>({index_zero}));
    }

    Value curr_condition = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, inner_value, equal_to);

    curr_condition.getDefiningOp()->setAttr("phism.is_condition", b.getUnitAttr());
    condition = ((condition == nullptr) ? curr_condition : b.create<arith::AndIOp>(loc, condition, curr_condition));
    condition.getDefiningOp()->setAttr("phism.is_condition", b.getUnitAttr());
  }

  scf::IfOp compute_if = b.create<scf::IfOp>(loc, condition, /*hasElse*/ has_else);

  b.setInsertionPointToStart(&(compute_if.thenRegion().front()));
  return compute_if;
}

static FuncOp create_IO_L3_in_serialize(FuncOp current_func_op, mlir::MemRefType in_type, const std::string& in_type_name, const std::string& name, ConversionPatternRewriter &b) {
  MLIRContext *context = current_func_op.getContext();

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  SmallVector<Type> arg_types = {in_type, in_type};
  SmallVector<Type> res_types = {};

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE OFF"));

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

  b.setInsertionPointToStart(block);
  AffineForOp loop = b.create<AffineForOp>(loc, 0, 128, 1);
  loop->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));

  // Load from input pointer to temporary
  b.setInsertionPointToStart(loop.getBody());
  memref::LoadOp load_op = b.create<memref::LoadOp>(loc, func_op.getArguments()[0], SmallVector<Value, 2>({loop.getInductionVar(), loop.getInductionVar()})); // TODO double index to hack memref as ptr

  // Use a custom call op for writing to hls stream
  CallOp write_C_local_call_op = createWriteCallOp(b, load_op, func_op.getArguments()[1], func_op);

  // Set insertion to after all for loops and add func return
  b.setInsertionPointAfter(loop);
  b.create<ReturnOp>(func_op.getLoc());

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
    makePointer(in_type_name),
    makeHLSStream(in_type_name)
  };
  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "create_IO_L3_in_serialize func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static FuncOp create_IO_L3_in(FuncOp current_func_op, mlir::MemRefType in_type, const std::string& in_type_name, const std::string& name, ConversionPatternRewriter &b) {
  MLIRContext *context = current_func_op.getContext();

  SmallVector<Type> arg_types = {in_type, in_type};
  SmallVector<Type> res_types = {};

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE OFF"));

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

  // Add additional loops
  b.setInsertionPointToStart(block);
  AffineForOp loop0 = b.create<AffineForOp>(loc, 0, 2, 1);

  b.setInsertionPointToStart(loop0.getBody());
  AffineForOp loop1 = b.create<AffineForOp>(loc, 0, 2, 1);

  b.setInsertionPointToStart(loop1.getBody());
  AffineForOp loop2 = b.create<AffineForOp>(loc, 0, 2, 1);

  b.setInsertionPointToStart(loop2.getBody());
  AffineForOp loop3 = b.create<AffineForOp>(loc, 0, 2, 1);

  b.setInsertionPointToStart(loop3.getBody());
  AffineForOp loop4 = b.create<AffineForOp>(loc, 0, 8, 1);
  loop4->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));

  b.setInsertionPointToStart(loop4.getBody());

  // Use a custom call op for reading from hls stream
  CallOp read_call_op = createReadCallOp(b, func_op.getArguments()[0], func_op);

  // Use a custom call op for writing to hls stream
  CallOp write_call_op = createWriteCallOp(b, read_call_op.getResults()[0], func_op.getArguments()[1], func_op);
  
  // Set insertion to after all for loops and add func return
  b.setInsertionPointAfter(loop0);
  b.create<ReturnOp>(func_op.getLoc());

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
    makeHLSStream(in_type_name),
    makeHLSStream(in_type_name)
  };
  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "create_IO_L3_in func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static memref::AllocaOp allocateIntVariable(Location loc, ConversionPatternRewriter &b, int init_value = 0, unsigned width = 32) {
  arith::ConstantOp init = b.create<arith::ConstantOp>(loc, IntegerAttr::get(b.getIntegerType(width), init_value));
  
  auto var_type = MemRefType::get({1}, b.getIntegerType(width));
  memref::AllocaOp var = b.create<memref::AllocaOp>(loc, var_type);
  arith::ConstantIndexOp index_zero = b.create<arith::ConstantIndexOp>(loc, 0);
  memref::StoreOp var_store = b.create<memref::StoreOp>(loc, init, var, SmallVector<Value, 1>({index_zero}));

  return var;
}

static memref::AllocaOp allocateIndexVariable(Location loc, ConversionPatternRewriter &b, int init_value = 0) {
  arith::ConstantIndexOp init = b.create<arith::ConstantIndexOp>(loc, init_value);
  
  auto var_type = MemRefType::get({1}, b.getIndexType());
  memref::AllocaOp var = b.create<memref::AllocaOp>(loc, var_type);
  arith::ConstantIndexOp index_zero = b.create<arith::ConstantIndexOp>(loc, 0);
  memref::StoreOp var_store = b.create<memref::StoreOp>(loc, init, var, SmallVector<Value, 1>({index_zero}));

  return var;
}

static void assignIntVariable(Location loc, ConversionPatternRewriter &b, memref::AllocaOp variable, int new_value, unsigned width) {
  // unsigned width = variable.getElementType().dyn_cast<IntegerType>().getWidth(); //TODO 'class mlir::memref::AllocaOp' has no member named 'getElementType'
  arith::ConstantOp value_to_store = b.create<arith::ConstantOp>(loc, IntegerAttr::get(b.getIntegerType(width), new_value));
  arith::ConstantIndexOp index_zero = b.create<arith::ConstantIndexOp>(loc, 0);
  memref::StoreOp variabe_store = b.create<memref::StoreOp>(loc, value_to_store, variable, SmallVector<Value, 1>({index_zero}));
}

static void assignIntVariable(Location loc, ConversionPatternRewriter &b, memref::AllocaOp variable, Value new_value) {
  arith::ConstantIndexOp index_zero = b.create<arith::ConstantIndexOp>(loc, 0);
  memref::StoreOp variabe_store = b.create<memref::StoreOp>(loc, new_value, variable, SmallVector<Value, 1>({index_zero}));
}

static void negateIntVariable(Location loc, ConversionPatternRewriter &b, memref::AllocaOp var) {
  // Create constant with value "1" to emulate logical NOT as XOR 1 (!A -> A ^ 1)
  arith::ConstantOp one = b.create<arith::ConstantOp>(loc, IntegerAttr::get(b.getIntegerType(1), 1));

  arith::ConstantIndexOp index_zero = b.create<arith::ConstantIndexOp>(loc, 0);
  memref::LoadOp inner_var = b.create<memref::LoadOp>(loc, var, SmallVector<Value, 1>({index_zero}));
  arith::XOrIOp not_var = b.create<arith::XOrIOp>(loc, inner_var, one);
  memref::StoreOp arb_store = b.create<memref::StoreOp>(loc, not_var, var, SmallVector<Value, 1>({index_zero}));
}

static FuncOp create_IO_L2_in_inter_trans(FuncOp current_func_op, mlir::MemRefType in_type, const std::string& in_type_name,
                                          const std::string& name, bool is_boundary, ConversionPatternRewriter &b) {
  // When boundary IO then there's one less argument, so offest for that when accessing. Reverse accessing (i.e. [-1]) is illegal for func arguments
  unsigned arg_offset = is_boundary ? 1 : 0;
  
  MLIRContext *context = current_func_op.getContext();

  auto index_type = IndexType::get(context);
  auto bool_type = b.getIntegerType(1);
  auto local_type = MemRefType::get({8, 1}, in_type.getElementType());

  SmallVector<Type> arg_types = {index_type, index_type, index_type, local_type, in_type};
  if (!is_boundary) arg_types.push_back(in_type);
  SmallVector<Type> remaining_arg_types = {bool_type, index_type};
  arg_types.insert(arg_types.end(), remaining_arg_types.begin(), remaining_arg_types.end());
  SmallVector<Type> res_types = {};

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.create_index", IntegerAttr::get(b.getIntegerType(32), 1));
  func_op->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE OFF"));

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

  // Add early if return condition -> actually reverse the condition to avoid having MLIR issue of return inside scf::if
  b.setInsertionPointToStart(block);
  scf::IfOp early_if = createIfWithConstantConditions(b, loc, {{func_op.getArguments()[6 - arg_offset], 1}}, /*has_else*/ false);

  // Add additional loop
  AffineForOp loop = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop.getBody());

  // Add if iter == idx condition
  scf::IfOp iter_if = createIfWithConditions(b, loc, {{loop.getInductionVar(), func_op.getArguments()[7 - arg_offset]}}, /*has_else*/ !is_boundary);

  // Add then loop
  AffineForOp then_loop = b.create<AffineForOp>(loc, 0, 8, 1);
  then_loop->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));
  b.setInsertionPointToStart(then_loop.getBody());

  // Use a custom call op for reading from hls stream
  CallOp then_read_call_op = createReadCallOp(b, func_op.getArguments()[4], func_op);

  // Write to local
  arith::ConstantIndexOp index_zero = b.create<arith::ConstantIndexOp>(loc, 0);
  memref::StoreOp variabe_store = b.create<memref::StoreOp>(loc, then_read_call_op.getResults()[0], func_op.getArguments()[3],
                                                            SmallVector<Value, 2>({then_loop.getInductionVar(), index_zero}));

  if (!is_boundary) {
    // Add else loop
    b.setInsertionPointToStart(&(iter_if.elseRegion().front()));
    AffineForOp else_loop = b.create<AffineForOp>(loc, 0, 8, 1);
    else_loop->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));
    b.setInsertionPointToStart(else_loop.getBody());

    // Use a custom call op for reading from hls stream
    CallOp else_read_call_op = createReadCallOp(b, func_op.getArguments()[4], func_op);

    // Use a custom call op for writing to hls stream
    CallOp else_write_call_op = createWriteCallOp(b, else_read_call_op.getResults()[0], func_op.getArguments()[5], func_op);
  }

  // Set insertion to after all for loops and add func return
  b.setInsertionPointAfter(early_if);
  b.create<ReturnOp>(func_op.getLoc());

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
    "ap_uint<2>", //TODO ensure this matches the caller's index width
    "ap_uint<2>", //TODO ensure this matches the caller's index width
    "ap_uint<2>", //TODO ensure this matches the caller's index width
    in_type_name,
    makeHLSStream(in_type_name)
  };
  if (!is_boundary) argument_types.push_back(makeHLSStream(in_type_name));
  std::vector<std::string> remaining_argument_types = {"bool", "unsigned"};
  argument_types.insert(argument_types.end(), remaining_argument_types.begin(), remaining_argument_types.end());

  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "create_IO_L2_in_inter_trans func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static FuncOp create_IO_L2_in_intra_trans(FuncOp current_func_op, mlir::MemRefType in_type, const std::string& in_type_name,
                                          const std::string& name, ConversionPatternRewriter &b) {
  MLIRContext *context = current_func_op.getContext();

  auto index_type = IndexType::get(context);
  auto bool_type = b.getIntegerType(1);
  auto local_type = MemRefType::get({8, 1}, in_type.getElementType());

  SmallVector<Type> arg_types = {index_type, index_type, index_type, local_type, in_type, bool_type, index_type};
  SmallVector<Type> res_types = {};

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.create_index", IntegerAttr::get(b.getIntegerType(32), 1));
  func_op->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE OFF"));

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

  b.setInsertionPointToStart(block);

  // Allocate data_split
  auto data_split_type = MemRefType::get({8}, in_type.getElementType());
  memref::AllocaOp data_split = b.create<memref::AllocaOp>(loc, data_split_type);
  data_split->setAttr("phism.variable_name", StringAttr::get(context, "data_split"));
  data_split->setAttr("phism.hls_pragma", StringAttr::get(context, "ARRAY_PARTITION variable=$ complete"));

  // Add early if return condition -> actually reverse the condition to avoid having MLIR issue of return inside scf::if
  scf::IfOp early_if = createIfWithConstantConditions(b, loc, {{func_op.getArguments()[5], 1}}, /*has_else*/ false);

  // Add additional loops
  AffineForOp loop0 = b.create<AffineForOp>(loc, 0, 8, 1);

  b.setInsertionPointToStart(loop0.getBody());
  AffineForOp loop1 = b.create<AffineForOp>(loc, 0, 8, 1);

  b.setInsertionPointToStart(loop1.getBody());
  AffineForOp loop2 = b.create<AffineForOp>(loc, 0, 8, 1);
  loop2->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));

  b.setInsertionPointToStart(loop2.getBody());

  // Load from local
  arith::ConstantIndexOp eight = b.create<arith::ConstantIndexOp>(loc, 8);
  arith::DivUIOp outer_index = b.create<arith::DivUIOp>(loc, loop0.getInductionVar(), eight);
  memref::LoadOp in_data = b.create<memref::LoadOp>(loc, func_op.getArguments()[3], SmallVector<Value, 2>({loop2.getInductionVar(), outer_index}));
  in_data->setAttr("phism.variable_name", StringAttr::get(context, "in_data"));

  // Add inner loop
  AffineForOp inner_loop = b.create<AffineForOp>(loc, 0, 8, 1);
  inner_loop->setAttr("phism.hls_pragma", StringAttr::get(context, "UNROLL"));
  b.setInsertionPointToStart(inner_loop.getBody());

  // Use emit hacks to avoid having to deal with bit accessing and MLIR shift type consistency issues
  memref::StoreOp data_split_store = b.create<memref::StoreOp>(loc, in_data, data_split, SmallVector<Value, 1>({inner_loop.getInductionVar()}));
  data_split_store->setAttr("phism.store_through_bit_access", StringAttr::get(context, "64"));
  inner_loop->setAttr("phism.include_ShRUIOp", StringAttr::get(context, "in_data,64"));

  b.setInsertionPointAfter(inner_loop);

  // Write to local
  arith::RemUIOp split_idx = b.create<arith::RemUIOp>(loc, loop0.getInductionVar(), eight);
  memref::LoadOp data_split_load = b.create<memref::LoadOp>(loc, data_split, SmallVector<Value, 1>({split_idx}));
  createWriteCallOp(b, data_split_load, func_op.getArguments()[4], func_op);

  // Set insertion to after all for loops and add func return
  b.setInsertionPointAfter(early_if);
  b.create<ReturnOp>(func_op.getLoc());

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
    "ap_uint<2>", //TODO ensure this matches the caller's index width
    "ap_uint<2>", //TODO ensure this matches the caller's index width
    "ap_uint<2>", //TODO ensure this matches the caller's index width
    in_type_name,
    makeHLSStream(in_type_name),
    "bool",
    "unsigned"
  };
  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "create_IO_L2_in_inter_trans func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static FuncOp create_IO_L2_in(FuncOp current_func_op, FuncOp IO_L2_in_inter_trans, FuncOp IO_L2_in_inter_trans_boundary, FuncOp IO_L2_in_intra_trans, mlir::MemRefType in_type, const std::string& in_type_name, const std::string& name, bool is_boundary, ConversionPatternRewriter &b) {
  // When boundary IO then there's one less argument, so offest for that when accessing. Reverse accessing (i.e. [-1]) is illegal for func arguments
  unsigned arg_offset = is_boundary ? 1 : 0;
  
  MLIRContext *context = current_func_op.getContext();

  SmallVector<Type> arg_types = {in_type, in_type};
  if (!is_boundary) arg_types.push_back(in_type);
  SmallVector<Type> remaining_arg_types = {IndexType::get(context)};
  arg_types.insert(arg_types.end(), remaining_arg_types.begin(), remaining_arg_types.end());
  SmallVector<Type> res_types = {};

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.create_index", IntegerAttr::get(b.getIntegerType(32), 1));
  func_op->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE OFF"));

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

  b.setInsertionPointToStart(block);

  // Allocate ping
  auto local_ping_type = MemRefType::get({8, 1}, in_type.getElementType());
  memref::AllocaOp local_ping = b.create<memref::AllocaOp>(loc, local_ping_type);
  local_ping->setAttr("phism.variable_name", StringAttr::get(context, "local_ping"));
  local_ping->setAttr("phism.hls_pragma", StringAttr::get(context, "RESOURCE variable=$ core=RAM_2P_BRAM"));

  // Allocate pong
  auto local_pong_type = MemRefType::get({8, 1}, in_type.getElementType());
  memref::AllocaOp local_pong = b.create<memref::AllocaOp>(loc, local_pong_type);
  local_pong->setAttr("phism.variable_name", StringAttr::get(context, "local_pong"));
  local_pong->setAttr("phism.hls_pragma", StringAttr::get(context, "RESOURCE variable=$ core=RAM_2P_BRAM"));

  // Create state values
  memref::AllocaOp arb = allocateIntVariable(loc, b, /*init_value*/ 0, /*width*/ 1);
  arb->setAttr("phism.variable_name", StringAttr::get(context, "arb"));

  memref::AllocaOp inter_trans_en = allocateIntVariable(loc, b, /*init_value*/ 1, /*width*/ 1);
  inter_trans_en->setAttr("phism.variable_name", StringAttr::get(context, "inter_trans_en"));

  memref::AllocaOp intra_trans_en = allocateIntVariable(loc, b, /*init_value*/ 0, /*width*/ 1);
  intra_trans_en->setAttr("phism.variable_name", StringAttr::get(context, "intra_trans_en"));

  memref::AllocaOp iter0_prev = allocateIndexVariable(loc, b, /*init_value*/ 0);
  iter0_prev->setAttr("phism.variable_name", StringAttr::get(context, "iter0_prev"));

  memref::AllocaOp iter1_prev = allocateIndexVariable(loc, b, /*init_value*/ 0);
  iter1_prev->setAttr("phism.variable_name", StringAttr::get(context, "iter1_prev"));

  memref::AllocaOp iter2_prev = allocateIndexVariable(loc, b, /*init_value*/ 0);
  iter2_prev->setAttr("phism.variable_name", StringAttr::get(context, "iter2_prev"));

  // 0 index for loading
  arith::ConstantIndexOp index_zero = b.create<arith::ConstantIndexOp>(loc, 0);

  // Add additional loops
  AffineForOp loop0 = b.create<AffineForOp>(loc, 0, 2, 1);

  b.setInsertionPointToStart(loop0.getBody());
  AffineForOp loop1 = b.create<AffineForOp>(loc, 0, 2, 1);

  b.setInsertionPointToStart(loop1.getBody());
  AffineForOp loop2 = b.create<AffineForOp>(loc, 0, 2, 1);

  b.setInsertionPointToStart(loop2.getBody());

  // Load inter_trans_en, intra_trans_en, and iter_prevs for passing to functions
  memref::LoadOp inter_trans_en_load = b.create<memref::LoadOp>(loc, inter_trans_en, SmallVector<Value, 1>({index_zero}));
  memref::LoadOp intra_trans_en_load = b.create<memref::LoadOp>(loc, intra_trans_en, SmallVector<Value, 1>({index_zero}));
  memref::LoadOp iter0_prev_load = b.create<memref::LoadOp>(loc, iter0_prev, SmallVector<Value, 1>({index_zero}));
  memref::LoadOp iter1_prev_load = b.create<memref::LoadOp>(loc, iter1_prev, SmallVector<Value, 1>({index_zero}));
  memref::LoadOp iter2_prev_load = b.create<memref::LoadOp>(loc, iter2_prev, SmallVector<Value, 1>({index_zero}));

  // Create inner if arb == 0 condition
  scf::IfOp inner_if = createIfWithConstantConditions(b, loc, {{arb, 0}}, /*has_else*/ true);

  // Then branch
  // Inter call
  SmallVector<Value, 8> IO_L2_in_inter_trans_boundary_operands = {
    loop0.getInductionVar(),
    loop1.getInductionVar(),
    loop2.getInductionVar(),
    local_pong,
    func_op.getArguments()[0],
    inter_trans_en_load,
    func_op.getArguments()[3 - arg_offset]
  };
  SmallVector<Value, 8> IO_L2_in_inter_trans_operands = {
    loop0.getInductionVar(),
    loop1.getInductionVar(),
    loop2.getInductionVar(),
    local_pong,
    func_op.getArguments()[0],
    func_op.getArguments()[1],
    inter_trans_en_load,
    func_op.getArguments()[3 - arg_offset]
  };
  if (is_boundary) {
    CallOp then_inter_call = b.create<CallOp>(
      IO_L2_in_inter_trans_boundary.getLoc(),
      IO_L2_in_inter_trans_boundary,
      IO_L2_in_inter_trans_boundary_operands
    );
  }
  else {
    CallOp then_inter_call = b.create<CallOp>(
      IO_L2_in_inter_trans.getLoc(),
      IO_L2_in_inter_trans,
      IO_L2_in_inter_trans_operands
    );
  }

  // Intra call
  SmallVector<Value, 7> IO_L2_in_intra_trans_operands = {
    iter0_prev_load,
    iter1_prev_load,
    iter2_prev_load,
    local_ping,
    func_op.getArguments()[2 - arg_offset],
    intra_trans_en_load,
    func_op.getArguments()[3 - arg_offset]
  };
  CallOp then_intra_call = b.create<CallOp>(
    IO_L2_in_intra_trans.getLoc(),
    IO_L2_in_intra_trans,
    IO_L2_in_intra_trans_operands
  );

  // Else branch
  b.setInsertionPointToStart(&(inner_if.elseRegion().front()));

  // Inter call
  if (is_boundary) {
    IO_L2_in_inter_trans_boundary_operands[3] = local_ping; // swap ping with pong
    CallOp else_inter_call = b.create<CallOp>(
      IO_L2_in_inter_trans_boundary.getLoc(),
      IO_L2_in_inter_trans_boundary,
      IO_L2_in_inter_trans_boundary_operands
    );
  }
  else {
    IO_L2_in_inter_trans_operands[3] = local_ping; // swap ping with pong
    CallOp else_inter_call = b.create<CallOp>(
      IO_L2_in_inter_trans.getLoc(),
      IO_L2_in_inter_trans,
      IO_L2_in_inter_trans_operands
    );
  }

  // Intra call
  IO_L2_in_intra_trans_operands[3] = local_pong; // swap ping with pong
  CallOp else_intra_call = b.create<CallOp>(
    IO_L2_in_intra_trans.getLoc(),
    IO_L2_in_intra_trans,
    IO_L2_in_intra_trans_operands
  );

  // Update state values
  b.setInsertionPointAfter(inner_if);
  assignIntVariable(loc, b, intra_trans_en, /*new_value*/ 1, /*width*/ 1); // intra_trans_en = 1
  negateIntVariable(loc, b, arb);                                          // arb = !arb
  assignIntVariable(loc, b, iter0_prev, loop0.getInductionVar());          // iter0_prev = iter0
  assignIntVariable(loc, b, iter1_prev, loop1.getInductionVar());          // iter1_prev = iter1
  assignIntVariable(loc, b, iter2_prev, loop2.getInductionVar());          // iter2_prev = iter2
  
  // Set insertion to after all for loops
  b.setInsertionPointAfter(loop0);

  // Load intra_trans_en and iter_prevs for passing to functions
  memref::LoadOp outer_intra_trans_en_load = b.create<memref::LoadOp>(loc, intra_trans_en, SmallVector<Value, 1>({index_zero}));
  memref::LoadOp outer_iter0_prev_load = b.create<memref::LoadOp>(loc, iter0_prev, SmallVector<Value, 1>({index_zero}));
  memref::LoadOp outer_iter1_prev_load = b.create<memref::LoadOp>(loc, iter1_prev, SmallVector<Value, 1>({index_zero}));
  memref::LoadOp outer_iter2_prev_load = b.create<memref::LoadOp>(loc, iter2_prev, SmallVector<Value, 1>({index_zero}));

  SmallVector<Value, 7> outer_IO_L2_in_intra_trans_operands = {
    outer_iter0_prev_load,
    outer_iter1_prev_load,
    outer_iter2_prev_load,
    local_ping,
    func_op.getArguments()[2 - arg_offset],
    outer_intra_trans_en_load,
    func_op.getArguments()[3 - arg_offset]
  };

  // Add outer if arb == 0 condition
  scf::IfOp outer_if = createIfWithConstantConditions(b, loc, {{arb, 0}}, /*has_else*/ true);

  // Then branch
  CallOp outer_then_intra_call = b.create<CallOp>(
    IO_L2_in_intra_trans.getLoc(),
    IO_L2_in_intra_trans,
    outer_IO_L2_in_intra_trans_operands
  );

  // Else branch
  b.setInsertionPointToStart(&(outer_if.elseRegion().front()));

  outer_IO_L2_in_intra_trans_operands[3] = local_pong; // swap ping with pong
  CallOp outer_else_intra_call = b.create<CallOp>(
    IO_L2_in_intra_trans.getLoc(),
    IO_L2_in_intra_trans,
    outer_IO_L2_in_intra_trans_operands
  );

  // Add func return
  b.setInsertionPointAfter(outer_if);
  b.create<ReturnOp>(func_op.getLoc());

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
    makeHLSStream(in_type_name),
    makeHLSStream(in_type_name)
  };
  if (!is_boundary) argument_types.push_back(makeHLSStream(in_type_name));
  std::vector<std::string> remaining_argument_types = {"unsigned"};
  argument_types.insert(argument_types.end(), remaining_argument_types.begin(), remaining_argument_types.end());

  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "create_IO_L2_in func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static FuncOp createDummyPEIn(FuncOp current_func_op, Value in_value, const std::string& in_type_name, const std::string& name, ConversionPatternRewriter &b) {
  MLIRContext *context = current_func_op.getContext();

  mlir::MemRefType in_type = in_value.getType().cast<MemRefType>();

  SmallVector<Type> arg_types = {in_type, IndexType::get(context), IndexType::get(context)};
  SmallVector<Type> res_types = {};

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.create_index", IntegerAttr::get(b.getIntegerType(32), 2));
  
  LLVM_DEBUG({dbgs() << "createDummyPEIn func op:\n"; func_op.dump();});

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

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

  b.setInsertionPointToStart(loop5.getBody());

  // Create the dummy read func op
  FuncOp hls_stream_read_func_op = createDummyFuncOp(func_op, {in_type}, {b.getF64Type()}, "hls_stream_read", b, /*return_inner_type*/ true);
  SmallVector<Value> operands{func_op.getArguments()[0]};

  // Use a custom call op for reading from hls stream
  CallOp hls_stream_read = b.create<CallOp>(
    hls_stream_read_func_op.getLoc(),
    hls_stream_read_func_op,
    operands
  );
  hls_stream_read->setAttr("phism.hls_stream_read", b.getUnitAttr());


  // Set insertion to after all for loops and add func return
  b.setInsertionPointAfter(loop0);
  b.create<ReturnOp>(func_op.getLoc());

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
    makeHLSStream(in_type_name),
    "unsigned",
    "unsigned"
  };
  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "createDummyPEIn func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static FuncOp create_drain_IO_L1_out_inter_trans(FuncOp current_func_op, mlir::MemRefType out_type, const std::string& out_type_name,
                                          const std::string& name, bool is_boundary, ConversionPatternRewriter &b) {
  // When boundary IO then there's one less argument, so offest for that when accessing. Reverse accessing (i.e. [-1]) is illegal for func arguments
  unsigned arg_offset = is_boundary ? 1 : 0;
  
  MLIRContext *context = current_func_op.getContext();

  auto index_type = IndexType::get(context);
  auto bool_type = b.getIntegerType(1);
  auto local_type = MemRefType::get({8, 2}, out_type.getElementType());

  SmallVector<Type> arg_types = {index_type, index_type, local_type};
  if (!is_boundary) arg_types.push_back(out_type);
  SmallVector<Type> remaining_arg_types = {out_type, index_type, index_type};
  arg_types.insert(arg_types.end(), remaining_arg_types.begin(), remaining_arg_types.end());
  SmallVector<Type> res_types = {};

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.create_index", IntegerAttr::get(b.getIntegerType(32), 2));
  func_op->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE"));

  LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_inter_trans func op after 1:\n"; func_op.dump();});

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

  b.setInsertionPointToStart(block);

  // Add additional loop
  AffineForOp loop = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop.getBody());

  LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_inter_trans func op after 2:\n"; func_op.dump();});

  // Add if iter == idx condition
  scf::IfOp iter_if = createIfWithConditions(b, loc, {{loop.getInductionVar(), func_op.getArguments()[5 - arg_offset]}}, /*has_else*/ !is_boundary);

  LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_inter_trans func op after 3:\n"; func_op.dump();});

  // Add then loops
  AffineForOp then_loop0 = b.create<AffineForOp>(loc, 0, 8, 1);
  b.setInsertionPointToStart(then_loop0.getBody());
  AffineForOp then_loop1 = b.create<AffineForOp>(loc, 0, 2, 1);
  then_loop1->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));
  b.setInsertionPointToStart(then_loop1.getBody());

  LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_inter_trans func op after 4:\n"; func_op.dump();});

  // Read from local
  memref::LoadOp in_data = b.create<memref::LoadOp>(loc, func_op.getArguments()[2], SmallVector<Value, 2>({then_loop0.getInductionVar(), then_loop1.getInductionVar()}));

  LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_inter_trans func op after 5:\n"; func_op.dump();});

  // Use a custom call op for writing to hls stream
  CallOp out_write_call_op = createWriteCallOp(b, in_data, func_op.getArguments()[4 - arg_offset], func_op);

  LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_inter_trans func op after 6:\n"; func_op.dump();});

  if (!is_boundary) {
    // Add else loops
    b.setInsertionPointToStart(&(iter_if.elseRegion().front()));
    AffineForOp else_loop0 = b.create<AffineForOp>(loc, 0, 8, 1);
    b.setInsertionPointToStart(else_loop0.getBody());
    AffineForOp else_loop1 = b.create<AffineForOp>(loc, 0, 2, 1);
    else_loop1->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));
    b.setInsertionPointToStart(else_loop1.getBody());

    LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_inter_trans func op after 7:\n"; func_op.dump();});

    // Use a custom call op for reading from hls stream
    CallOp else_read_call_op = createReadCallOp(b, func_op.getArguments()[3], func_op);

    LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_inter_trans func op after 8:\n"; func_op.dump();});

    // Use a custom call op for writing to hls stream
    CallOp else_write_call_op = createWriteCallOp(b, else_read_call_op.getResults()[0], func_op.getArguments()[4], func_op);

    LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_inter_trans func op after 9:\n"; func_op.dump();});
  }

  // Set insertion to after all for loops and add func return
  b.setInsertionPointAfter(loop);
  b.create<ReturnOp>(func_op.getLoc());

  LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_inter_trans func op after 10:\n"; func_op.dump();});

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
    "ap_uint<2>", //TODO ensure this matches the caller's index width
    "ap_uint<2>", //TODO ensure this matches the caller's index width
    out_type_name
  };
  if (!is_boundary) argument_types.push_back(makeHLSStream(out_type_name));
  std::vector<std::string> remaining_argument_types = {makeHLSStream(out_type_name), "unsigned", "unsigned"};
  argument_types.insert(argument_types.end(), remaining_argument_types.begin(), remaining_argument_types.end());

  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_inter_trans func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static FuncOp create_drain_IO_L1_out_intra_trans(FuncOp current_func_op, mlir::MemRefType out_type, const std::string& out_type_name,
                                          const std::string& name, ConversionPatternRewriter &b) {
  MLIRContext *context = current_func_op.getContext();

  auto index_type = IndexType::get(context);
  auto local_type = MemRefType::get({8, 2}, out_type.getElementType());

  SmallVector<Type> arg_types = {index_type, index_type, local_type, out_type, index_type, index_type};
  SmallVector<Type> res_types = {};

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.create_index", IntegerAttr::get(b.getIntegerType(32), 2));
  func_op->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE"));

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

  b.setInsertionPointToStart(block);

  // Allocate data_split
  auto data_split_type = MemRefType::get({4}, out_type.getElementType());
  memref::AllocaOp data_split = b.create<memref::AllocaOp>(loc, data_split_type);
  data_split->setAttr("phism.variable_name", StringAttr::get(context, "data_split"));
  data_split->setAttr("phism.hls_pragma", StringAttr::get(context, "ARRAY_PARTITION variable=$ complete"));

  // Add additional loops
  AffineForOp loop0 = b.create<AffineForOp>(loc, 0, 8, 1);

  b.setInsertionPointToStart(loop0.getBody());
  AffineForOp loop1 = b.create<AffineForOp>(loc, 0, 8, 1);
  loop1->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));

  b.setInsertionPointToStart(loop1.getBody());

  // Use a custom call op for reading from hls stream
  CallOp then_read_call_op = createReadCallOp(b, func_op.getArguments()[3], func_op);

  // Load from local
  arith::ConstantIndexOp four = b.create<arith::ConstantIndexOp>(loc, 4);
  arith::DivUIOp outer_index = b.create<arith::DivUIOp>(loc, loop0.getInductionVar(), four);
  arith::RemUIOp split_idx = b.create<arith::RemUIOp>(loc, loop0.getInductionVar(), four);
  memref::LoadOp out_data = b.create<memref::LoadOp>(loc, func_op.getArguments()[2], SmallVector<Value, 2>({loop1.getInductionVar(), outer_index}));
  out_data->setAttr("phism.variable_name", StringAttr::get(context, "out_data"));

  // Add inner loop
  AffineForOp inner_loop = b.create<AffineForOp>(loc, 0, 4, 1);
  inner_loop->setAttr("phism.hls_pragma", StringAttr::get(context, "UNROLL"));
  b.setInsertionPointToStart(inner_loop.getBody());

  // Use emit hacks to avoid having to deal with bit accessing and MLIR shift type consistency issues
  memref::StoreOp data_split_store = b.create<memref::StoreOp>(loc, out_data, data_split, SmallVector<Value, 1>({inner_loop.getInductionVar()}));
  data_split_store->setAttr("phism.store_through_bit_access", StringAttr::get(context, "32"));
  inner_loop->setAttr("phism.include_ShRUIOp", StringAttr::get(context, "out_data,32"));

  b.setInsertionPointAfter(inner_loop);

  // Write to data_split
  memref::StoreOp outer_data_split_store = b.create<memref::StoreOp>(loc, then_read_call_op.getResults()[0], data_split, SmallVector<Value, 1>({split_idx}));
  outer_data_split_store->setAttr("phism.load_through_ap_uint", StringAttr::get(context, "32"));

  // Write to local
  memref::StoreOp local_store = b.create<memref::StoreOp>(loc, out_data, func_op.getArguments()[2], SmallVector<Value, 2>({loop1.getInductionVar(), outer_index}));
  local_store->setAttr("phism.assemble_with", StringAttr::get(context, "data_split,4"));

  // Set insertion to after all for loops and add func return
  b.setInsertionPointAfter(loop0);
  b.create<ReturnOp>(func_op.getLoc());

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
    "ap_uint<2>", //TODO ensure this matches the caller's index width
    "ap_uint<2>", //TODO ensure this matches the caller's index width
    out_type_name,
    makeHLSStream(out_type_name),
    "unsigned",
    "unsigned"
  };
  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out_intra_trans func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static FuncOp create_drain_IO_L1_out(FuncOp current_func_op, FuncOp drain_IO_L1_out_inter_trans, FuncOp drain_IO_L1_out_inter_trans_boundary, FuncOp drain_IO_L1_out_intra_trans,
                                     mlir::MemRefType out_type, const std::string& out_type_name, const std::string& name, bool is_boundary, ConversionPatternRewriter &b) {
  // When boundary IO then there's one less argument, so offest for that when accessing. Reverse accessing (i.e. [-1]) is illegal for func arguments
  unsigned arg_offset = is_boundary ? 1 : 0;
  
  MLIRContext *context = current_func_op.getContext();

  SmallVector<Type> arg_types = {};
  if (!is_boundary) arg_types.push_back(out_type);
  SmallVector<Type> remaining_arg_types = {out_type, out_type, IndexType::get(context), IndexType::get(context)};
  arg_types.insert(arg_types.end(), remaining_arg_types.begin(), remaining_arg_types.end());
  SmallVector<Type> res_types = {};

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.create_index", IntegerAttr::get(b.getIntegerType(32), 2));
  std::string inline_pragma = (is_boundary) ? "INLINE" : "INLINE OFF";
  func_op->setAttr("phism.hls_pragma", StringAttr::get(context, inline_pragma));

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

  b.setInsertionPointToStart(block);

  // Allocate local buffer
  auto local_type = MemRefType::get({8, 2}, out_type.getElementType());
  memref::AllocaOp local = b.create<memref::AllocaOp>(loc, local_type);

  // Add additional loops
  AffineForOp loop0 = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop0.getBody());
  AffineForOp loop1 = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop1.getBody());

// Intra call
  SmallVector<Value, 7> drain_IO_L1_out_intra_trans_operands = {
    loop0.getInductionVar(),
    loop1.getInductionVar(),
    local,
    func_op.getArguments()[2 - arg_offset],
    func_op.getArguments()[3 - arg_offset],
    func_op.getArguments()[4 - arg_offset]
  };
  CallOp intra_call = b.create<CallOp>(
    drain_IO_L1_out_intra_trans.getLoc(),
    drain_IO_L1_out_intra_trans,
    drain_IO_L1_out_intra_trans_operands
  );

  // Inter call
  if (is_boundary) {
    SmallVector<Value, 6> drain_IO_L1_out_inter_trans_boundary_operands= {
      loop0.getInductionVar(),
      loop1.getInductionVar(),
      local,
      func_op.getArguments()[1 - arg_offset],
      func_op.getArguments()[3 - arg_offset],
      func_op.getArguments()[4 - arg_offset]
    };
    CallOp inter_call = b.create<CallOp>(
      drain_IO_L1_out_inter_trans_boundary.getLoc(),
      drain_IO_L1_out_inter_trans_boundary,
      drain_IO_L1_out_inter_trans_boundary_operands
    );
  }
  else {
    SmallVector<Value, 7> drain_IO_L1_out_inter_trans_operands = {
      loop0.getInductionVar(),
      loop1.getInductionVar(),
      local,
      func_op.getArguments()[0],
      func_op.getArguments()[1 - arg_offset],
      func_op.getArguments()[3 - arg_offset],
      func_op.getArguments()[4 - arg_offset]
    };
    CallOp inter_call = b.create<CallOp>(
      drain_IO_L1_out_inter_trans.getLoc(),
      drain_IO_L1_out_inter_trans,
      drain_IO_L1_out_inter_trans_operands
    );
  }

  // Add func return
  b.setInsertionPointAfter(loop0);
  b.create<ReturnOp>(func_op.getLoc());

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
  };
  if (!is_boundary) argument_types.push_back(makeHLSStream(out_type_name));
  std::vector<std::string> remaining_argument_types = {makeHLSStream(out_type_name), makeHLSStream(out_type_name), "unsigned", "unsigned"};
  argument_types.insert(argument_types.end(), remaining_argument_types.begin(), remaining_argument_types.end());

  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "create_drain_IO_L1_out func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static FuncOp create_drain_IO_L2_out(FuncOp current_func_op, mlir::MemRefType out_type, const std::string& out_type_name, const std::string& name, bool is_boundary, ConversionPatternRewriter &b) {
  // When boundary IO then there's one less argument, so offest for that when accessing. Reverse accessing (i.e. [-1]) is illegal for func arguments
  unsigned arg_offset = is_boundary ? 1 : 0;

  MLIRContext *context = current_func_op.getContext();

  SmallVector<Type> arg_types = {};
  if (!is_boundary) arg_types.push_back(out_type);
  SmallVector<Type> remaining_arg_types = {out_type, out_type, IndexType::get(context)};
  arg_types.insert(arg_types.end(), remaining_arg_types.begin(), remaining_arg_types.end());
  SmallVector<Type> res_types = {};

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE OFF"));
  func_op->setAttr("phism.create_index", IntegerAttr::get(b.getIntegerType(32), 1));

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

  // Add additional loops
  b.setInsertionPointToStart(block);
  AffineForOp loop0 = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop0.getBody());
  AffineForOp loop1 = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop1.getBody());
  AffineForOp loop2 = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop2.getBody());

  // Add if iter == idx
  scf::IfOp iter_if = createIfWithConditions(b, loc, {{loop2.getInductionVar(), func_op.getArguments()[3 - arg_offset]}}, /*has_else*/ !is_boundary);

  // Add more additional loops
  AffineForOp loop3 = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop3.getBody());
  AffineForOp loop4 = b.create<AffineForOp>(loc, 0, 8, 1);
  b.setInsertionPointToStart(loop4.getBody());
  AffineForOp loop5 = b.create<AffineForOp>(loc, 0, 2, 1);
  loop5->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));
  b.setInsertionPointToStart(loop5.getBody());

  // Use a custom call op for reading from hls stream
  CallOp read_call_op = createReadCallOp(b, func_op.getArguments()[2 - arg_offset], func_op);

  // Use a custom call op for writing to hls stream
  CallOp write_call_op = createWriteCallOp(b, read_call_op.getResults()[0], func_op.getArguments()[1 - arg_offset], func_op);

  if (!is_boundary) {
    // Handle else branch
    b.setInsertionPointToStart(&(iter_if.elseRegion().front()));

    // Add more additional loops
    AffineForOp else_loop3 = b.create<AffineForOp>(loc, 0, 2, 1);
    b.setInsertionPointToStart(else_loop3.getBody());
    AffineForOp else_loop4 = b.create<AffineForOp>(loc, 0, 8, 1);
    b.setInsertionPointToStart(else_loop4.getBody());
    AffineForOp else_loop5 = b.create<AffineForOp>(loc, 0, 2, 1);
    else_loop5->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));
    b.setInsertionPointToStart(else_loop5.getBody());

    // Use a custom call op for reading from hls stream
    CallOp read_call_op = createReadCallOp(b, func_op.getArguments()[0], func_op);

    // Use a custom call op for writing to hls stream
    CallOp write_call_op = createWriteCallOp(b, read_call_op.getResults()[0], func_op.getArguments()[1], func_op);
  }
  
  // Set insertion to after all for loops and add func return
  b.setInsertionPointAfter(loop0);
  b.create<ReturnOp>(func_op.getLoc());

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {
  };
  if (!is_boundary) argument_types.push_back(makeHLSStream(out_type_name));
  std::vector<std::string> remaining_argument_types = {makeHLSStream(out_type_name), makeHLSStream(out_type_name), "unsigned"};
  argument_types.insert(argument_types.end(), remaining_argument_types.begin(), remaining_argument_types.end());
  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "create_drain_IO_L2_out func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static FuncOp create_drain_IO_L3_out(FuncOp current_func_op, mlir::MemRefType out_type, const std::string& out_type_name, const std::string& name, ConversionPatternRewriter &b) {
  MLIRContext *context = current_func_op.getContext();

  SmallVector<Type> arg_types = {out_type, out_type};
  SmallVector<Type> res_types = {};

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE OFF"));

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

  // Add additional loops
  b.setInsertionPointToStart(block);
  AffineForOp loop0 = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop0.getBody());
  AffineForOp loop1 = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop1.getBody());
  AffineForOp loop2 = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop2.getBody());
  AffineForOp loop3 = b.create<AffineForOp>(loc, 0, 2, 1);
  b.setInsertionPointToStart(loop3.getBody());
  AffineForOp loop4 = b.create<AffineForOp>(loc, 0, 8, 1);
  b.setInsertionPointToStart(loop4.getBody());
  AffineForOp loop5 = b.create<AffineForOp>(loc, 0, 2, 1);
  loop5->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));
  b.setInsertionPointToStart(loop5.getBody());

  // Use a custom call op for reading from hls stream
  CallOp read_call_op = createReadCallOp(b, func_op.getArguments()[1], func_op);

  // Use a custom call op for writing to hls stream
  CallOp write_call_op = createWriteCallOp(b, read_call_op.getResults()[0], func_op.getArguments()[0], func_op);
  
  // Set insertion to after all for loops and add func return
  b.setInsertionPointAfter(loop0);
  b.create<ReturnOp>(func_op.getLoc());

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {makeHLSStream(out_type_name), makeHLSStream(out_type_name)};
  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "create_drain_IO_L3_out func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static FuncOp create_drain_IO_L3_out_serialize(FuncOp current_func_op, mlir::MemRefType out_type, const std::string& out_type_name, const std::string& name, ConversionPatternRewriter &b) {
  MLIRContext *context = current_func_op.getContext();

  SmallVector<Type> arg_types = {out_type, out_type};
  SmallVector<Type> res_types = {};

  auto insertion_point = b.saveInsertionPoint();
  b.setInsertionPoint(current_func_op);

  FuncOp func_op = b.create<FuncOp>(
    current_func_op.getLoc(),
    name,
    b.getFunctionType(arg_types, res_types)
  );
  func_op->setAttr("phism.hls_pragma", StringAttr::get(context, "INLINE OFF"));

  Location loc = func_op.getLoc();
  Block *block = func_op.addEntryBlock();

  // Add additional loop
  b.setInsertionPointToStart(block);
  AffineForOp loop0 = b.create<AffineForOp>(loc, 0, 64, 1);
  loop0->setAttr("phism.hls_pragma", StringAttr::get(context, "PIPELINE II=1"));
  b.setInsertionPointToStart(loop0.getBody());

  // Allocate mem_data
  auto mem_data_type = MemRefType::get({1}, out_type.getElementType());
  memref::AllocaOp mem_data = b.create<memref::AllocaOp>(loc, mem_data_type);
  mem_data->setAttr("phism.variable_name", StringAttr::get(context, "mem_data"));

  // Allocate mem_data_split
  auto mem_data_split_type = MemRefType::get({4}, out_type.getElementType());
  memref::AllocaOp mem_data_split = b.create<memref::AllocaOp>(loc, mem_data_split_type);
  mem_data_split->setAttr("phism.variable_name", StringAttr::get(context, "mem_data_split"));
  mem_data_split->setAttr("phism.hls_pragma", StringAttr::get(context, "ARRAY_PARTITION variable=$ complete"));
  
  // Add additional loop
  AffineForOp loop1 = b.create<AffineForOp>(loc, 0, 4, 1);
  b.setInsertionPointToStart(loop1.getBody());

  // Use a custom call op for reading from hls stream
  CallOp read_call_op = createReadCallOp(b, func_op.getArguments()[1], func_op);

  // Store to mem_data_split
  memref::StoreOp mem_data_split_store = b.create<memref::StoreOp>(loc, read_call_op.getResults()[0], mem_data_split, SmallVector<Value, 1>({loop1.getInductionVar()}));

  // Store to output
  b.setInsertionPointAfter(loop1);
  arith::ConstantIndexOp index_zero = b.create<arith::ConstantIndexOp>(loc, 0);
  memref::LoadOp mem_data_load = b.create<memref::LoadOp>(loc, mem_data, SmallVector<Value, 1>({index_zero}));
  memref::StoreOp output_store = b.create<memref::StoreOp>(loc, mem_data_load, func_op.getArguments()[0], SmallVector<Value, 2>({loop0.getInductionVar(), loop0.getInductionVar()})); // TODO double index to hack memref as ptr
  output_store->setAttr("phism.assemble_with", StringAttr::get(context, "mem_data_split,4"));

  // Set insertion to after all for loops and add func return
  b.setInsertionPointAfter(loop0);
  b.create<ReturnOp>(func_op.getLoc());

  // Expicitly mark argument types
  std::vector<std::string> argument_types = {makePointer(out_type_name), makeHLSStream(out_type_name)};
  func_op->setAttr(
    "phism.argument_types",
    StringAttr::get(context, vec2str(argument_types))
  );

  LLVM_DEBUG({dbgs() << "create_drain_IO_L3_out_serialize func op after adding block with return op:\n"; func_op.dump();});

  b.restoreInsertionPoint(insertion_point);
  return func_op;
}

static inline std::string makeIndexName(const std::string &name, const SmallVector<unsigned> &coords) {
  std::string index_name = name;

  for (const auto coord : coords) {
    index_name += "_" + std::to_string(coord);
  }

  return index_name;
}

void addIndexesAsOperands(SmallVector<Value> &operands, const SmallVector<unsigned> &coords, const std::string &name, MLIRContext *context, Location loc, ConversionPatternRewriter &b) {
  SmallVector<std::string, 3> id_names = {"x", "y", "z"};
  assert(coords.size() <= id_names.size() && "Too many coordinates provided");

  for (unsigned i = 0; i < coords.size(); i++) {
    arith::ConstantIndexOp id_value = b.create<arith::ConstantIndexOp>(loc, coords[i]);
    id_value->setAttr("phism.variable_name", StringAttr::get(context, makeIndexName("id" + id_names[i] + "_" + name, coords)));
    operands.push_back(id_value);
  }
}

static void handleTopFuncOp(FuncOp top_func_op, FuncOp PE_func_op) {
  std::string IO_type = "float";

  MLIRContext *context = top_func_op.getContext();
  Location loc = top_func_op.getLoc();
  ConversionPatternRewriter b(context);

  // Find affine for ops
  SmallVector<AffineForOp> for_ops;
  top_func_op.walk([&](AffineForOp op) {
    // LLVM_DEBUG({dbgs() << "op.getConstantLowerBound(): " << op.getConstantLowerBound() << ", op.getConstantUpperBound(): " << op.getConstantUpperBound() << "\n";});
    for_ops.push_back(op);
  });

  // Find PE call op
  SmallVector<CallOp> PE_call_ops;
  top_func_op.walk([&](CallOp op) {
    if (op->hasAttr("phism.pe"))
      PE_call_ops.push_back(op);
  });

  assert(PE_call_ops.size() == 1 && "There can only be 1 PE call op");
  CallOp original_PE_call_op = PE_call_ops[0];

  // Set insertion point to before the affine for ops
  b.setInsertionPoint(for_ops[1]);

  unsigned PE_dims = 2; // In the future this will come from analysing the space loops
  unsigned PE_IOs_dims = PE_dims + 1;
  SmallVector<CallOp, 4> new_PE_call_ops;

  // 3D vector: [A|B][idx][idy]
  SmallVector<SmallVector<SmallVector<memref::AllocaOp, 3>, 3>, 2> PE_IOs(
    2, SmallVector<SmallVector<memref::AllocaOp, 3>, 3>(
      3, SmallVector<memref::AllocaOp, 3>(
        3
      )
    )
  );

  LLVM_DEBUG({dbgs() << " PE_IOs dims: " << PE_IOs.size() << ", " << PE_IOs[0].size() << ", " << PE_IOs[0][0].size() << "\n";});

  Type C_type = original_PE_call_op.getOperands()[0].getType();
  Type A_type = original_PE_call_op.getOperands()[1].getType();
  Type B_type = original_PE_call_op.getOperands()[2].getType();

  // Declare all PE I/O beforehand
  for (unsigned x = 0; x < PE_IOs_dims; x++) {
    for (unsigned y = 0; y < PE_IOs_dims; y++) {

      memref::AllocaOp fifo_A = b.create<memref::AllocaOp>(loc, A_type.cast<MemRefType>());
      fifo_A->setAttr("phism.variable_name", StringAttr::get(context, makeIndexName("fifo_A_PE", {x, y})));
      fifo_A->setAttr("phism.hls_stream", b.getUnitAttr());
      fifo_A->setAttr("phism.type_name", StringAttr::get(context, IO_type));
      fifo_A->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));
      PE_IOs[0][x][y] = fifo_A;

      memref::AllocaOp fifo_B = b.create<memref::AllocaOp>(loc, B_type.cast<MemRefType>());
      fifo_B->setAttr("phism.variable_name", StringAttr::get(context, makeIndexName("fifo_B_PE", {x, y})));
      fifo_B->setAttr("phism.hls_stream", b.getUnitAttr());
      fifo_B->setAttr("phism.type_name", StringAttr::get(context, IO_type));
      fifo_B->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));
      PE_IOs[1][x][y] = fifo_B;
    }
  }

  // --------------------------------------------------------------------------------------------------------- 
  
  // Add top level serialization for A
  FuncOp A_IO_L3_in_serialize = create_IO_L3_in_serialize(top_func_op, A_type.cast<MemRefType>(), IO_type, "A_IO_L3_in_serialize", b);

  memref::AllocaOp fifo_A_IO_L3_in_serialize = b.create<memref::AllocaOp>(loc, A_type.cast<MemRefType>());
  fifo_A_IO_L3_in_serialize->setAttr("phism.variable_name", StringAttr::get(context, "fifo_A_IO_L3_in_serialize"));
  fifo_A_IO_L3_in_serialize->setAttr("phism.hls_stream", b.getUnitAttr());
  fifo_A_IO_L3_in_serialize->setAttr("phism.type_name", StringAttr::get(context, IO_type));
  fifo_A_IO_L3_in_serialize->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2"));

  // TODO how to not hard code the arg number?
  SmallVector<Value, 2> A_IO_L3_in_serialize_operands = {top_func_op.getArguments()[7], fifo_A_IO_L3_in_serialize};

  CallOp A_IO_L3_in_serialize_call = b.create<CallOp>(
    A_IO_L3_in_serialize.getLoc(),
    A_IO_L3_in_serialize,
    A_IO_L3_in_serialize_operands
  );
  

  // Add L3 IO for A
  FuncOp A_IO_L3_in = create_IO_L3_in(top_func_op, A_type.cast<MemRefType>(), IO_type, "A_IO_L3_in", b);

  memref::AllocaOp fifo_A_IO_L2_in_0 = b.create<memref::AllocaOp>(loc, A_type.cast<MemRefType>());
  fifo_A_IO_L2_in_0->setAttr("phism.variable_name", StringAttr::get(context, "fifo_A_IO_L2_in_0"));
  fifo_A_IO_L2_in_0->setAttr("phism.hls_stream", b.getUnitAttr());
  fifo_A_IO_L2_in_0->setAttr("phism.type_name", StringAttr::get(context, IO_type));
  fifo_A_IO_L2_in_0->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));

  SmallVector<Value, 2> A_IO_L3_in_operands = {fifo_A_IO_L3_in_serialize, fifo_A_IO_L2_in_0};

  CallOp A_IO_L3_in_call = b.create<CallOp>(
    A_IO_L3_in.getLoc(),
    A_IO_L3_in,
    A_IO_L3_in_operands
  );

  // Add L2 IO for A
  FuncOp A_IO_L2_in_inter_trans = create_IO_L2_in_inter_trans(top_func_op, A_type.cast<MemRefType>(), IO_type, "A_IO_L2_in_inter_trans", /*is_boundary*/ false, b);
  FuncOp A_IO_L2_in_inter_trans_boundary = create_IO_L2_in_inter_trans(top_func_op, A_type.cast<MemRefType>(), IO_type, "A_IO_L2_in_inter_trans_boundary", /*is_boundary*/ true, b);
  FuncOp A_IO_L2_in_intra_trans = create_IO_L2_in_intra_trans(top_func_op, A_type.cast<MemRefType>(), IO_type, "A_IO_L2_in_intra_trans", b);
  FuncOp A_IO_L2_in = create_IO_L2_in(top_func_op, A_IO_L2_in_inter_trans, A_IO_L2_in_inter_trans_boundary, A_IO_L2_in_intra_trans, A_type.cast<MemRefType>(), IO_type, "A_IO_L2_in", /*is_boundary*/ false, b);

  memref::AllocaOp fifo_A_IO_L2_in_1 = b.create<memref::AllocaOp>(loc, A_type.cast<MemRefType>());
  fifo_A_IO_L2_in_1->setAttr("phism.variable_name", StringAttr::get(context, "fifo_A_IO_L2_in_1"));
  fifo_A_IO_L2_in_1->setAttr("phism.hls_stream", b.getUnitAttr());
  fifo_A_IO_L2_in_1->setAttr("phism.type_name", StringAttr::get(context, IO_type));
  fifo_A_IO_L2_in_1->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));

  SmallVector<Value> A_IO_L2_in_operands = {fifo_A_IO_L2_in_0, fifo_A_IO_L2_in_1, PE_IOs[0][0][0]};
  addIndexesAsOperands(A_IO_L2_in_operands, {0}, "A_IO_L2_in", context, A_IO_L2_in.getLoc(), b);

  CallOp A_IO_L2_in_call = b.create<CallOp>(
    A_IO_L2_in.getLoc(),
    A_IO_L2_in,
    A_IO_L2_in_operands
  );

  // Add boundary L2 IO for A
  FuncOp A_IO_L2_in_boundary = create_IO_L2_in(top_func_op, A_IO_L2_in_inter_trans, A_IO_L2_in_inter_trans_boundary, A_IO_L2_in_intra_trans, A_type.cast<MemRefType>(), IO_type, "A_IO_L2_in_boundary", /*is_boundary*/ true, b);

  SmallVector<Value> A_IO_L2_in_boundary_operands = {fifo_A_IO_L2_in_1, PE_IOs[0][1][0]};
  addIndexesAsOperands(A_IO_L2_in_boundary_operands, {1}, "A_IO_L2_in_boundary", context, A_IO_L2_in_boundary.getLoc(), b);

  CallOp A_IO_L2_in_boundary_call = b.create<CallOp>(
    A_IO_L2_in_boundary.getLoc(),
    A_IO_L2_in_boundary,
    A_IO_L2_in_boundary_operands
  );

  // --------------------------------------------------------------------------------------------------------- 
  
  // Add top level serialization for B
  FuncOp B_IO_L3_in_serialize = create_IO_L3_in_serialize(top_func_op, B_type.cast<MemRefType>(), IO_type, "B_IO_L3_in_serialize", b);

  memref::AllocaOp fifo_B_IO_L3_in_serialize = b.create<memref::AllocaOp>(loc, B_type.cast<MemRefType>());
  fifo_B_IO_L3_in_serialize->setAttr("phism.variable_name", StringAttr::get(context, "fifo_B_IO_L3_in_serialize"));
  fifo_B_IO_L3_in_serialize->setAttr("phism.hls_stream", b.getUnitAttr());
  fifo_B_IO_L3_in_serialize->setAttr("phism.type_name", StringAttr::get(context, IO_type));
  fifo_B_IO_L3_in_serialize->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2"));

  // TODO how to not hard code the arg number?
  SmallVector<Value, 2> B_IO_L3_in_serialize_operands = {top_func_op.getArguments()[8], fifo_B_IO_L3_in_serialize};

  CallOp B_IO_L3_in_serialize_call = b.create<CallOp>(
    B_IO_L3_in_serialize.getLoc(),
    B_IO_L3_in_serialize,
    B_IO_L3_in_serialize_operands
  );
  
  // Add L3 IO for B
  FuncOp B_IO_L3_in = create_IO_L3_in(top_func_op, B_type.cast<MemRefType>(), IO_type, "B_IO_L3_in", b);

  memref::AllocaOp fifo_B_IO_L2_in_0 = b.create<memref::AllocaOp>(loc, B_type.cast<MemRefType>());
  fifo_B_IO_L2_in_0->setAttr("phism.variable_name", StringAttr::get(context, "fifo_B_IO_L2_in_0"));
  fifo_B_IO_L2_in_0->setAttr("phism.hls_stream", b.getUnitAttr());
  fifo_B_IO_L2_in_0->setAttr("phism.type_name", StringAttr::get(context, IO_type));
  fifo_B_IO_L2_in_0->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));

  SmallVector<Value, 2> B_IO_L3_in_operands = {fifo_B_IO_L3_in_serialize, fifo_B_IO_L2_in_0};

  CallOp B_IO_L3_in_call = b.create<CallOp>(
    B_IO_L3_in.getLoc(),
    B_IO_L3_in,
    B_IO_L3_in_operands
  );

  // Add L2 IO for B
  FuncOp B_IO_L2_in_inter_trans = create_IO_L2_in_inter_trans(top_func_op, B_type.cast<MemRefType>(), IO_type, "B_IO_L2_in_inter_trans", /*is_boundary*/ false, b);
  FuncOp B_IO_L2_in_inter_trans_boundary = create_IO_L2_in_inter_trans(top_func_op, B_type.cast<MemRefType>(), IO_type, "B_IO_L2_in_inter_trans_boundary", /*is_boundary*/ true, b);
  FuncOp B_IO_L2_in_intra_trans = create_IO_L2_in_intra_trans(top_func_op, B_type.cast<MemRefType>(), IO_type, "B_IO_L2_in_intra_trans", b);
  FuncOp B_IO_L2_in = create_IO_L2_in(top_func_op, B_IO_L2_in_inter_trans, B_IO_L2_in_inter_trans_boundary, B_IO_L2_in_intra_trans, B_type.cast<MemRefType>(), IO_type, "B_IO_L2_in", /*is_boundary*/ false, b);

  memref::AllocaOp fifo_B_IO_L2_in_1 = b.create<memref::AllocaOp>(loc, B_type.cast<MemRefType>());
  fifo_B_IO_L2_in_1->setAttr("phism.variable_name", StringAttr::get(context, "fifo_B_IO_L2_in_1"));
  fifo_B_IO_L2_in_1->setAttr("phism.hls_stream", b.getUnitAttr());
  fifo_B_IO_L2_in_1->setAttr("phism.type_name", StringAttr::get(context, IO_type));
  fifo_B_IO_L2_in_1->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));

  SmallVector<Value> B_IO_L2_in_operands = {fifo_B_IO_L2_in_0, fifo_B_IO_L2_in_1, PE_IOs[1][0][0]};
  addIndexesAsOperands(B_IO_L2_in_operands, {0}, "B_IO_L2_in", context, B_IO_L2_in.getLoc(), b);

  CallOp B_IO_L2_in_call = b.create<CallOp>(
    B_IO_L2_in.getLoc(),
    B_IO_L2_in,
    B_IO_L2_in_operands
  );

  // Add boundary L2 IO for B
  FuncOp B_IO_L2_in_boundary = create_IO_L2_in(top_func_op, B_IO_L2_in_inter_trans, B_IO_L2_in_inter_trans_boundary, B_IO_L2_in_intra_trans, B_type.cast<MemRefType>(), IO_type, "B_IO_L2_in_boundary", /*is_boundary*/ true, b);

  SmallVector<Value> B_IO_L2_in_boundary_operands = {fifo_B_IO_L2_in_1, PE_IOs[1][1][0]};
  addIndexesAsOperands(B_IO_L2_in_boundary_operands, {1}, "B_IO_L2_in_boundary", context, B_IO_L2_in_boundary.getLoc(), b);

  CallOp B_IO_L2_in_boundary_call = b.create<CallOp>(
    B_IO_L2_in_boundary.getLoc(),
    B_IO_L2_in_boundary,
    B_IO_L2_in_boundary_operands
  );

  // ---------------------------------------------------------------------------------------------------------

  // 2D vector: [idx][idy]
  SmallVector<SmallVector<memref::AllocaOp, 2>, 2> fifo_C_drains(2, SmallVector<memref::AllocaOp, 2>(2));

  // Instantiate PEs and connect them using existing I/O
  for (unsigned x = 0; x < PE_dims; x++) {
    for (unsigned y = 0; y < PE_dims; y++) {

      SmallVector<Value> operands;

      memref::AllocaOp fifo_C_drain = b.create<memref::AllocaOp>(loc, C_type.cast<MemRefType>());
      fifo_C_drain->setAttr("phism.variable_name", StringAttr::get(context, makeIndexName("fifo_C_drain_PE", {x, y})));
      fifo_C_drain->setAttr("phism.hls_stream", b.getUnitAttr());
      fifo_C_drain->setAttr("phism.type_name", StringAttr::get(context, IO_type));
      fifo_C_drain->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));

      fifo_C_drains[x][y] = fifo_C_drain;

      operands.push_back(fifo_C_drain);
      operands.push_back(PE_IOs[0][x][y]);
      operands.push_back(PE_IOs[1][x][y]);
      operands.push_back(PE_IOs[0][x][y + 1]);
      operands.push_back(PE_IOs[1][x + 1][y]);

      addIndexesAsOperands(operands, {x, y}, "PE", context, loc, b);

      CallOp new_PE_call_op = b.create<CallOp>(
        loc,
        PE_func_op,
        operands
      );
      new_PE_call_ops.push_back(new_PE_call_op);


    }
  }

  // ---------------------------------------------------------------------------------------------------------

  // Add dummies for consuming output A's from PEs on the boundaries
  FuncOp A_PE_dummy_in = createDummyPEIn(top_func_op, PE_IOs[0][0][0], IO_type, "A_PE_dummy_in", b);
  for (unsigned x = 0; x < PE_dims; x++) {
    unsigned y = PE_dims;

    Location inner_loc = A_PE_dummy_in.getLoc();

    SmallVector<Value> operands;

    operands.push_back(PE_IOs[0][x][y]);

    addIndexesAsOperands(operands, {x, y}, "A_PE_dummy_in", context, inner_loc, b);

    CallOp A_PE_dummy_in_call = b.create<CallOp>(
      inner_loc,
      A_PE_dummy_in,
      operands
    );
  }

  // Add dummies for consuming output B's from PEs on the boundaries
  FuncOp B_PE_dummy_in = createDummyPEIn(top_func_op, PE_IOs[1][0][0], IO_type, "B_PE_dummy_in", b);
  for (unsigned y = 0; y < PE_dims; y++) {
    unsigned x = PE_dims;

    Location inner_loc = B_PE_dummy_in.getLoc();

    SmallVector<Value> operands;

    operands.push_back(PE_IOs[1][x][y]);

    addIndexesAsOperands(operands, {x, y}, "B_PE_dummy_in", context, inner_loc, b);

    CallOp B_PE_dummy_in_call = b.create<CallOp>(
      inner_loc,
      B_PE_dummy_in,
      operands
    );
  }

  // ---------------------------------------------------------------------------------------------------------

  // Allocate drain IO for L1
  // 2D vector: [idx][idy]
  SmallVector<SmallVector<memref::AllocaOp, 2>, 2> fifo_C_drain_C_drain_IO_L1_outs(2, SmallVector<memref::AllocaOp, 2>(2));

  for (unsigned x = 0; x < PE_dims; x++) {
    for (unsigned y = 0; y < PE_dims; y++) {
      memref::AllocaOp fifo_C_drain_IO_L1_out = b.create<memref::AllocaOp>(loc, C_type.cast<MemRefType>());
      fifo_C_drain_IO_L1_out->setAttr("phism.variable_name", StringAttr::get(context, makeIndexName("fifo_C_drain_IO_L1_out", {x, y})));
      fifo_C_drain_IO_L1_out->setAttr("phism.hls_stream", b.getUnitAttr());
      fifo_C_drain_IO_L1_out->setAttr("phism.type_name", StringAttr::get(context, IO_type));
      fifo_C_drain_IO_L1_out->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));

      fifo_C_drain_C_drain_IO_L1_outs[x][y] = fifo_C_drain_IO_L1_out;
    }
  }

  // Add C drain L1
  FuncOp C_drain_IO_L1_out_inter_trans = create_drain_IO_L1_out_inter_trans(top_func_op, C_type.cast<MemRefType>(), IO_type, "C_drain_IO_L1_in_inter_trans", /*is_boundary*/ false, b);
  FuncOp C_drain_IO_L1_out_inter_trans_boundary = create_drain_IO_L1_out_inter_trans(top_func_op, C_type.cast<MemRefType>(), IO_type, "C_drain_IO_L1_in_inter_trans_boundary", /*is_boundary*/ true, b);
  FuncOp C_drain_IO_L1_out_intra_trans = create_drain_IO_L1_out_intra_trans(top_func_op, C_type.cast<MemRefType>(), IO_type, "C_drain_IO_L1_in_intra_trans", b);
  FuncOp C_drain_IO_L1_out = create_drain_IO_L1_out(top_func_op, C_drain_IO_L1_out_inter_trans, C_drain_IO_L1_out_inter_trans_boundary, C_drain_IO_L1_out_intra_trans, C_type.cast<MemRefType>(), IO_type, "C_drain_IO_L1_out", /*is_boundary*/ false, b);
  FuncOp C_drain_IO_L1_out_boundary = create_drain_IO_L1_out(top_func_op, C_drain_IO_L1_out_inter_trans, C_drain_IO_L1_out_inter_trans_boundary, C_drain_IO_L1_out_intra_trans, C_type.cast<MemRefType>(), IO_type, "C_drain_IO_L1_out_boundary", /*is_boundary*/ true, b);

  for (unsigned x = 0; x < PE_dims; x++) {
    for (unsigned y = 0; y < PE_dims; y++) {

      bool is_boundary = (y == (PE_dims - 1));

      // C_drain indices need to be swapped (TODO how to scale to 3x3 arrays)
      // 0,0 -> 0,0 | 0,1 -> 1,0 | 1,0 -> 0,1 | 1,1 -> 1,1
      unsigned fifo_C_drain_x = (x == y) ? x : (x ^ 1);
      unsigned fifo_C_drain_y = (x == y) ? y : (y ^ 1);

      if (is_boundary) {
        SmallVector<Value> C_drain_IO_L1_out_boundary_operands = {fifo_C_drain_C_drain_IO_L1_outs[x][y], fifo_C_drains[fifo_C_drain_x][fifo_C_drain_y]}; // TODO how to scale fifo_C_drain_C_drain_IO_L1_outs indices to 3x3
        addIndexesAsOperands(C_drain_IO_L1_out_boundary_operands, {x , y}, "C_drain_IO_L1_out_boundary", context, C_drain_IO_L1_out_boundary.getLoc(), b);

        CallOp C_drain_IO_L1_out_boundary_call = b.create<CallOp>(
          C_drain_IO_L1_out_boundary.getLoc(),
          C_drain_IO_L1_out_boundary,
          C_drain_IO_L1_out_boundary_operands
        );

      }
      else {
        SmallVector<Value> C_drain_IO_L1_out_operands = {fifo_C_drain_C_drain_IO_L1_outs[x][1], fifo_C_drain_C_drain_IO_L1_outs[x][y], fifo_C_drains[fifo_C_drain_x][fifo_C_drain_y]}; // TODO how to scale fifo_C_drain_C_drain_IO_L1_outs indices to 3x3
        addIndexesAsOperands(C_drain_IO_L1_out_operands, {x , y}, "C_drain_IO_L1_out", context, C_drain_IO_L1_out.getLoc(), b);

        CallOp C_drain_IO_L1_out_call = b.create<CallOp>(
          C_drain_IO_L1_out.getLoc(),
          C_drain_IO_L1_out,
          C_drain_IO_L1_out_operands
        );
      }
    }
  }

  // ---------------------------------------------------------------------------------------------------------

  // Allocate C drain L2 IO
  // 1D vector: [id]
  SmallVector<memref::AllocaOp, 2> fifo_C_drain_IO_L2_outs(2);
  for (unsigned id = 0; id < PE_dims; id++) {
    memref::AllocaOp fifo_C_drain_IO_L2_out = b.create<memref::AllocaOp>(loc, C_type.cast<MemRefType>());
    fifo_C_drain_IO_L2_out->setAttr("phism.variable_name", StringAttr::get(context, makeIndexName("fifo_C_drain_IO_L2_out", {id})));
    fifo_C_drain_IO_L2_out->setAttr("phism.hls_stream", b.getUnitAttr());
    fifo_C_drain_IO_L2_out->setAttr("phism.type_name", StringAttr::get(context, IO_type));
    fifo_C_drain_IO_L2_out->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));

    fifo_C_drain_IO_L2_outs[id] = fifo_C_drain_IO_L2_out;
  } 

  // Add C drain L2
  FuncOp C_drain_IO_L2_out = create_drain_IO_L2_out(top_func_op, C_type.cast<MemRefType>(), IO_type, "C_drain_IO_L2_out", /*is_boundary*/ false, b);

  SmallVector<Value> C_drain_IO_L2_out_operands = {fifo_C_drain_IO_L2_outs[1], fifo_C_drain_IO_L2_outs[0], fifo_C_drain_C_drain_IO_L1_outs[0][0]};
  addIndexesAsOperands(C_drain_IO_L2_out_operands, {0}, "C_drain_IO_L2_out", context, C_drain_IO_L2_out.getLoc(), b);

  CallOp C_drain_IO_L2_out_call = b.create<CallOp>(
    C_drain_IO_L2_out.getLoc(),
    C_drain_IO_L2_out,
    C_drain_IO_L2_out_operands
  );

  // Add C drain L2 boundary
  FuncOp C_drain_IO_L2_out_boundary = create_drain_IO_L2_out(top_func_op, C_type.cast<MemRefType>(), IO_type, "C_drain_IO_L2_out_boundary", /*is_boundary*/ true, b);

  SmallVector<Value> C_drain_IO_L2_out_boundary_operands = {fifo_C_drain_IO_L2_outs[1], fifo_C_drain_C_drain_IO_L1_outs[1][0]};
  addIndexesAsOperands(C_drain_IO_L2_out_boundary_operands, {0}, "C_drain_IO_L2_out_boundary", context, C_drain_IO_L2_out_boundary.getLoc(), b);

  CallOp C_drain_IO_L2_out_boundary_call = b.create<CallOp>(
    C_drain_IO_L2_out_boundary.getLoc(),
    C_drain_IO_L2_out_boundary,
    C_drain_IO_L2_out_boundary_operands
  );

  // ---------------------------------------------------------------------------------------------------------

  // Allocate C drain L3 IO
  memref::AllocaOp fifo_C_drain_IO_L3_out_serialize = b.create<memref::AllocaOp>(loc, C_type.cast<MemRefType>());
  fifo_C_drain_IO_L3_out_serialize->setAttr("phism.variable_name", StringAttr::get(context, "fifo_C_drain_IO_L3_out_serialize"));
  fifo_C_drain_IO_L3_out_serialize->setAttr("phism.hls_stream", b.getUnitAttr());
  fifo_C_drain_IO_L3_out_serialize->setAttr("phism.type_name", StringAttr::get(context, IO_type));
  fifo_C_drain_IO_L3_out_serialize->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2"));

  // Add C drain L3
  FuncOp C_drain_IO_L3_out = create_drain_IO_L3_out(top_func_op, C_type.cast<MemRefType>(), IO_type, "C_drain_IO_L3_out", b);

  SmallVector<Value> C_drain_IO_L3_out_operands = {fifo_C_drain_IO_L3_out_serialize, fifo_C_drain_IO_L2_outs[0]};

  CallOp C_drain_IO_L3_out_call = b.create<CallOp>(
    C_drain_IO_L3_out.getLoc(),
    C_drain_IO_L3_out,
    C_drain_IO_L3_out_operands
  );

  // ---------------------------------------------------------------------------------------------------------

  // Add C drain L3 serialize
  FuncOp C_drain_IO_L3_out_serialize = create_drain_IO_L3_out_serialize(top_func_op, C_type.cast<MemRefType>(), IO_type, "C_drain_IO_L3_out_serialize", b);

  SmallVector<Value> C_drain_IO_L3_out_serialize_operands = {top_func_op.getArguments()[10], fifo_C_drain_IO_L3_out_serialize};

  CallOp C_drain_IO_L3_out_serialize_call = b.create<CallOp>(
    C_drain_IO_L3_out_serialize.getLoc(),
    C_drain_IO_L3_out_serialize,
    C_drain_IO_L3_out_serialize_operands
  );

  // ---------------------------------------------------------------------------------------------------------

  original_PE_call_op.erase();
  for_ops[0].erase();
  for_ops[1].erase();
}

namespace {
class SystolicArraySpaceLoopPass : public phism::SystolicArraySpaceLoopPassBase<SystolicArraySpaceLoopPass> {
public:

  std::string fileName = "";

  SystolicArraySpaceLoopPass() = default;
  SystolicArraySpaceLoopPass(const SystolicArraySpaceLoopPipelineOptions & options)
    : fileName(!options.fileName.hasValue() ? "" : options.fileName.getValue()){
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Modify top function
    SmallVector<mlir::FuncOp> top_funcs;
    SmallVector<mlir::FuncOp> PE_funcs;
    m.walk([&](mlir::FuncOp op) {
      if (op->hasAttr("phism.top"))
        top_funcs.push_back(op);
      if (op->hasAttr("phism.pe"))
        PE_funcs.push_back(op);
    });

    assert(top_funcs.size() == 1 && "There can only be 1 top function");
    assert(PE_funcs.size() == 1 && "There can only be 1 PE function");

    handleTopFuncOp(top_funcs[0], PE_funcs[0]);

  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
phism::createSystolicArraySpaceLoopPass() {
  return std::make_unique<SystolicArraySpaceLoopPass>();
}

void phism::registerSystolicArraySpaceLoopPass() {
  PassPipelineRegistration<SystolicArraySpaceLoopPipelineOptions>(
    "systolic-array-space-loop", "Systolic array space loop TODO.",
    [](OpPassManager &pm, const SystolicArraySpaceLoopPipelineOptions &options) {
      pm.addPass(std::make_unique<SystolicArraySpaceLoopPass>(options));
    }
  );
}