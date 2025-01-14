#ifndef PHISM_MLIR_TRANSFORMS_PASSES_H
#define PHISM_MLIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace phism {

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
createLoopBoundHoistingPass();

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
createEliminateAffineLoadStorePass();

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createSplitNonAffinePass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createArrayPartitionPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createSimpleArrayPartitionPass();
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createSimplifyPartitionAccessPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createStripExceptTopPass();
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createAffineLoopUnswitchingPass();
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createAnnotatePointLoopPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createOutlineProcessElementPass();
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createRewritePloopIndvarPass();
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createLoadSwitchPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLiftMemRefSubviewPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createSCoPDecompositionPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInlineSCoPAffinePass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createEmitHLSPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createSystolicArrayTimeLoopPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createSystolicArraySpaceLoopPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "phism/mlir/Transforms/Passes.h.inc"

} // namespace phism

#endif
