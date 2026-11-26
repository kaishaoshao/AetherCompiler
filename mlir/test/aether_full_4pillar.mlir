// RUN: aether-opt %s --aether-swiglu-fusion --aether-data-movement-routing | FileCheck %s
module @Aether_4Pillar_Reference {
  func.func @forward(
      %x: tensor<8x16xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>,
      %w_norm: tensor<16xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>,
      %w_gate: tensor<16x32xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>,
      %w_up: tensor<16x32xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>,
      %w_down: tensor<32x16xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>)
      -> (tensor<8x16xf32, #aether_mem.space<"PIM_Near_Memory_Banks_Tier">>) {
    %x_asic = aether_mem.dma_transfer(%x) {src_tier = "3D_DRAM_Stack_Tier", dst_tier = "ASIC_Local_Line_Buffer"} : (tensor<8x16xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>) -> (tensor<8x16xf32, #aether_mem.space<"ASIC_Local_Line_Buffer">>)
    %wn_asic = aether_mem.dma_transfer(%w_norm) {src_tier = "3D_DRAM_Stack_Tier", dst_tier = "ASIC_Local_Line_Buffer"} : (tensor<16xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>) -> (tensor<16xf32, #aether_mem.space<"ASIC_Local_Line_Buffer">>)
    %h_norm = aether_graph.rmsnorm(%x_asic, %wn_asic) {placement = #aether_place.spec<chip = "fixed_asic_die", tp = 1>, eps = 1.0e-5 : f32} : (tensor<8x16xf32, #aether_mem.space<"ASIC_Local_Line_Buffer">>, tensor<16xf32, #aether_mem.space<"ASIC_Local_Line_Buffer">>) -> (tensor<8x16xf32, #aether_mem.space<"ASIC_Local_Line_Buffer">>)
    %h_npu = aether_mem.dma_transfer(%h_norm) {src_tier = "ASIC_Local_Line_Buffer", dst_tier = "NPU_Local_Scratchpad_SRAM"} : (tensor<8x16xf32, #aether_mem.space<"ASIC_Local_Line_Buffer">>) -> (tensor<8x16xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>)
    %wg_npu = aether_mem.dma_transfer(%w_gate) {src_tier = "3D_DRAM_Stack_Tier", dst_tier = "NPU_Local_Scratchpad_SRAM"} : (tensor<16x32xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>) -> (tensor<16x32xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>)
    %wu_npu = aether_mem.dma_transfer(%w_up) {src_tier = "3D_DRAM_Stack_Tier", dst_tier = "NPU_Local_Scratchpad_SRAM"} : (tensor<16x32xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>) -> (tensor<16x32xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>)
    %wd_npu = aether_mem.dma_transfer(%w_down) {src_tier = "3D_DRAM_Stack_Tier", dst_tier = "NPU_Local_Scratchpad_SRAM"} : (tensor<32x16xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>) -> (tensor<32x16xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>)
    %g_out = aether_graph.matmul(%h_npu, %wg_npu) {placement = #aether_place.spec<chip = "systolic_npu_die", tp = 2>} : (tensor<8x16xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>, tensor<16x32xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>) -> (tensor<8x32xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>)
    %u_out = aether_graph.matmul(%h_npu, %wu_npu) {placement = #aether_place.spec<chip = "systolic_npu_die", tp = 2>} : (tensor<8x16xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>, tensor<16x32xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>) -> (tensor<8x32xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>)
    %sw_out = aether_graph.custom_kernel(%g_out, %u_out) {placement = #aether_place.spec<chip = "systolic_npu_die", tp = 2>, framework = "triton", symbol = "@triton_swiglu"} : (tensor<8x32xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>, tensor<8x32xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>) -> (tensor<8x32xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>)
    %d_out = aether_graph.matmul(%sw_out, %wd_npu) {placement = #aether_place.spec<chip = "systolic_npu_die", tp = 2>} : (tensor<8x32xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>, tensor<32x16xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>) -> (tensor<8x16xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>)
    %ar_out = aether_place.all_reduce(%d_out) {placement = #aether_place.spec<chip = "systolic_npu_die", tp = 2>} : (tensor<8x16xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>) -> (tensor<8x16xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>)
    %ar_pim = aether_mem.dma_transfer(%ar_out) {src_tier = "NPU_Local_Scratchpad_SRAM", dst_tier = "PIM_Near_Memory_Banks_Tier"} : (tensor<8x16xf32, #aether_mem.space<"NPU_Local_Scratchpad_SRAM">>) -> (tensor<8x16xf32, #aether_mem.space<"PIM_Near_Memory_Banks_Tier">>)
    %x_pim = aether_mem.dma_transfer(%x) {src_tier = "3D_DRAM_Stack_Tier", dst_tier = "PIM_Near_Memory_Banks_Tier"} : (tensor<8x16xf32, #aether_mem.space<"3D_DRAM_Stack_Tier">>) -> (tensor<8x16xf32, #aether_mem.space<"PIM_Near_Memory_Banks_Tier">>)
    %out = aether_graph.add(%x_pim, %ar_pim) {placement = #aether_place.spec<chip = "pim_near_memory_die", tp = 1>} : (tensor<8x16xf32, #aether_mem.space<"PIM_Near_Memory_Banks_Tier">>, tensor<8x16xf32, #aether_mem.space<"PIM_Near_Memory_Banks_Tier">>) -> (tensor<8x16xf32, #aether_mem.space<"PIM_Near_Memory_Banks_Tier">>)
    return %out : tensor<8x16xf32, #aether_mem.space<"PIM_Near_Memory_Banks_Tier">>
  }
}
