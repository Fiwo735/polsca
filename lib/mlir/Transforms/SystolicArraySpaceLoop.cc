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
  memref::LoadOp load_op = b.create<memref::LoadOp>(loc, func_op.getArguments()[0], SmallVector<Value, 2>({loop.getInductionVar(), loop.getInductionVar()}));

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

  LLVM_DEBUG({dbgs() << "create_IO_L3_in_serialize func op after adding block with return op:\n"; func_op.dump();});

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
  func_op->setAttr("phism.create_index", b.getUnitAttr());
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

static inline std::string makeIndexName(const std::string &name, unsigned idx, unsigned idy) {
  return name + "_" + std::to_string(idx) + "_" + std::to_string(idy);
}

void addIndexesAsOperands(SmallVector<Value> &operands, unsigned x, unsigned y, const std::string &name, MLIRContext *context, Location loc, ConversionPatternRewriter &b) {
  arith::ConstantIndexOp idx_value = b.create<arith::ConstantIndexOp>(loc, x);
  idx_value->setAttr("phism.variable_name", StringAttr::get(context, makeIndexName("idx_" + name, x, y)));
  operands.push_back(idx_value);

  arith::ConstantIndexOp idy_value = b.create<arith::ConstantIndexOp>(loc, y);
  idy_value->setAttr("phism.variable_name", StringAttr::get(context, makeIndexName("idy_" + name, x, y)));
  operands.push_back(idy_value);
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
      fifo_A->setAttr("phism.variable_name", StringAttr::get(context, makeIndexName("fifo_A_PE", x, y)));
      fifo_A->setAttr("phism.hls_stream", b.getUnitAttr());
      fifo_A->setAttr("phism.type_name", StringAttr::get(context, IO_type));
      fifo_A->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));
      PE_IOs[0][x][y] = fifo_A;

      memref::AllocaOp fifo_B = b.create<memref::AllocaOp>(loc, B_type.cast<MemRefType>());
      fifo_B->setAttr("phism.variable_name", StringAttr::get(context, makeIndexName("fifo_B_PE", x, y)));
      fifo_B->setAttr("phism.hls_stream", b.getUnitAttr());
      fifo_B->setAttr("phism.type_name", StringAttr::get(context, IO_type));
      fifo_B->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));
      PE_IOs[1][x][y] = fifo_B;
    }
  }

  // Instantiate PEs and connect them using existing I/O
  for (unsigned x = 0; x < PE_dims; x++) {
    for (unsigned y = 0; y < PE_dims; y++) {

      SmallVector<Value> operands;

      memref::AllocaOp fifo_C_drain = b.create<memref::AllocaOp>(loc, C_type.cast<MemRefType>());
      fifo_C_drain->setAttr("phism.variable_name", StringAttr::get(context, makeIndexName("fifo_C_drain_PE", x, y)));
      fifo_C_drain->setAttr("phism.hls_stream", b.getUnitAttr());
      fifo_C_drain->setAttr("phism.type_name", StringAttr::get(context, IO_type));
      fifo_C_drain->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));

      operands.push_back(fifo_C_drain);
      operands.push_back(PE_IOs[0][x][y]);
      operands.push_back(PE_IOs[1][x][y]);
      operands.push_back(PE_IOs[0][x][y + 1]);
      operands.push_back(PE_IOs[1][x + 1][y]);

      addIndexesAsOperands(operands, x, y, "PE", context, loc, b);

      CallOp new_PE_call_op = b.create<CallOp>(
        loc,
        PE_func_op,
        operands
      );
      new_PE_call_ops.push_back(new_PE_call_op);


    }
  }

  
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

  memref::AllocaOp fifo_A_IO_L3_in = b.create<memref::AllocaOp>(loc, A_type.cast<MemRefType>());
  fifo_A_IO_L3_in->setAttr("phism.variable_name", StringAttr::get(context, "fifo_A_IO_L3_in"));
  fifo_A_IO_L3_in->setAttr("phism.hls_stream", b.getUnitAttr());
  fifo_A_IO_L3_in->setAttr("phism.type_name", StringAttr::get(context, IO_type));
  fifo_A_IO_L3_in->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));

  SmallVector<Value, 2> A_IO_L3_in_operands = {fifo_A_IO_L3_in_serialize, fifo_A_IO_L3_in};

  CallOp A_IO_L3_in_call = b.create<CallOp>(
    A_IO_L3_in.getLoc(),
    A_IO_L3_in,
    A_IO_L3_in_operands
  );
  
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

  memref::AllocaOp fifo_B_IO_L3_in = b.create<memref::AllocaOp>(loc, B_type.cast<MemRefType>());
  fifo_B_IO_L3_in->setAttr("phism.variable_name", StringAttr::get(context, "fifo_B_IO_L3_in"));
  fifo_B_IO_L3_in->setAttr("phism.hls_stream", b.getUnitAttr());
  fifo_B_IO_L3_in->setAttr("phism.type_name", StringAttr::get(context, IO_type));
  fifo_B_IO_L3_in->setAttr("phism.hls_pragma", StringAttr::get(context, "STREAM variable=$ depth=2,RESOURCE variable=$ core=FIFO_SRL"));

  SmallVector<Value, 2> B_IO_L3_in_operands = {fifo_B_IO_L3_in_serialize, fifo_B_IO_L3_in};

  CallOp B_IO_L3_in_call = b.create<CallOp>(
    B_IO_L3_in.getLoc(),
    B_IO_L3_in,
    B_IO_L3_in_operands
  );

  // Add dummies for consuming output A's from PEs on the boundaries
  FuncOp A_PE_dummy_in = createDummyPEIn(top_func_op, PE_IOs[0][0][0], IO_type, "A_PE_dummy_in", b);
  for (unsigned x = 0; x < PE_dims; x++) {
    unsigned y = PE_dims;

    Location inner_loc = A_PE_dummy_in.getLoc();

    SmallVector<Value> operands;

    operands.push_back(PE_IOs[0][x][y]);

    addIndexesAsOperands(operands, x, y, "A_PE_dummy_in", context, inner_loc, b);

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

    addIndexesAsOperands(operands, x, y, "B_PE_dummy_in", context, inner_loc, b);

    CallOp B_PE_dummy_in_call = b.create<CallOp>(
      inner_loc,
      B_PE_dummy_in,
      operands
    );
  }

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