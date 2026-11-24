set(LLVM_LINK_COMPONENTS
  Support
)

add_mlir_tool(zen-opt
  zen-opt.cpp
)

target_link_libraries(zen-opt
  PRIVATE
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRAllDialects
  MLIRAllPasses
)
