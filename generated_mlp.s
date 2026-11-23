	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.file	"-"
	.globl	run_model                       # -- Begin function run_model
	.p2align	1
	.type	run_model,@function
run_model:                              # @run_model
# %bb.0:
	addi	sp, sp, -176
	sd	ra, 168(sp)                     # 8-byte Folded Spill
	sd	s0, 160(sp)                     # 8-byte Folded Spill
	sd	s1, 152(sp)                     # 8-byte Folded Spill
	sd	s2, 144(sp)                     # 8-byte Folded Spill
	sd	s3, 136(sp)                     # 8-byte Folded Spill
	sd	s4, 128(sp)                     # 8-byte Folded Spill
	sd	s5, 120(sp)                     # 8-byte Folded Spill
	sd	s6, 112(sp)                     # 8-byte Folded Spill
	sd	s7, 104(sp)                     # 8-byte Folded Spill
	sd	s8, 96(sp)                      # 8-byte Folded Spill
	sd	s9, 88(sp)                      # 8-byte Folded Spill
	sd	s10, 80(sp)                     # 8-byte Folded Spill
	sd	s11, 72(sp)                     # 8-byte Folded Spill
	slliw	s2, a1, 3
	li	a2, 24
	mv	s10, a0
	mv	a0, s2
	blt	s2, a2, .LBB0_2
# %bb.1:
	li	a0, 24
.LBB0_2:
	addiw	a4, a0, 8
	bge	s2, a4, .LBB0_10
# %bb.3:
	lui	a0, 4
	lui	a2, 8
	add	a0, a0, s10
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	add	a0, s10, a2
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	addi	a0, s10, 2047
	addi	a0, a0, 1
	sd	a0, 24(sp)                      # 8-byte Folded Spill
	lui	a0, 4
	addiw	a0, a0, 32
	add	a0, a0, s10
	sd	a0, 16(sp)                      # 8-byte Folded Spill
	slliw	s8, a1, 9
.LBB0_13:                               # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c)
	addi	s1, s1, %pcrel_lo(.LBB0_13)
.LBB0_14:                               # Label of block must be emitted
	auipc	a6, %pcrel_hi(run_model.sram_bias)
	addi	a6, a6, %pcrel_lo(.LBB0_14)
.LBB0_15:                               # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_a)
	addi	s9, s9, %pcrel_lo(.LBB0_15)
	li	s11, 1024
.LBB0_16:                               # Label of block must be emitted
	auipc	s3, %pcrel_hi(run_model.sram_b)
	addi	s3, s3, %pcrel_lo(.LBB0_16)
	lui	a0, 1
	addiw	s4, a0, -2048
	li	s5, 1
	li	s6, 48
	sd	s2, 8(sp)                       # 8-byte Folded Spill
	sd	a4, 48(sp)                      # 8-byte Folded Spill
.LBB0_4:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_5 Depth 2
	slli	a0, s8, 2
	ld	a1, 16(sp)                      # 8-byte Folded Reload
	add	s7, a1, a0
	slliw	a0, s2, 6
	slli	a1, a0, 2
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	sd	a1, 64(sp)                      # 8-byte Folded Spill
	add	a0, a0, a1
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	lui	a2, 2
	mv	a0, s1
	li	a1, 0
	mv	s0, a6
	call	memset@plt
	mv	a6, s0
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	li	a1, 256
	#APP
	.insn r 11, 0, 1, zero, s0, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s3, s10

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	li	a2, 0
	li	a0, -8
	ld	a1, 24(sp)                      # 8-byte Folded Reload
.LBB0_5:                                #   Parent Loop BB0_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	slli	a5, a2, 32
	subw	a2, s5, a2
	slli	a4, a2, 32
	srli	a3, a4, 22
	add	a3, a3, s9
	#APP
	.insn r 11, 0, 1, zero, a3, s7

	#NO_APP
	srli	a4, a4, 21
	add	a4, a4, s3
	#APP
	.insn r 11, 0, 1, zero, a4, a1

	#NO_APP
	srli	s0, a5, 22
	add	s0, s0, s9
	srli	a5, a5, 21
	add	a5, a5, s3
	#APP
	.insn r 11, 0, 2, s1, s0, a5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, a0, 8
	add	a1, a1, s4
	addi	s7, s7, 32
	bltu	a0, s6, .LBB0_5
# %bb.6:                                #   in Loop: Header=BB0_4 Depth=1
	#APP
	.insn r 11, 0, 2, s1, a3, a4

	#NO_APP
	#APP
	.insn r 11, 0, 3, s1, a6, s4

	#NO_APP
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	add	a0, a0, s10
	lui	a1, 2
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s2, s2, 32
	addw	s8, s8, s4
	ld	a4, 48(sp)                      # 8-byte Folded Reload
	blt	s2, a4, .LBB0_4
# %bb.7:
	ld	s2, 8(sp)                       # 8-byte Folded Reload
	bge	s2, a4, .LBB0_10
# %bb.8:
	lui	a0, 6
	add	a0, a0, s10
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	lui	a0, 7
	addiw	a1, a0, -2048
	add	s3, s10, a1
	add	s4, s10, a0
	lui	a0, 8
	addiw	a0, a0, -2048
	add	s5, s10, a0
.LBB0_17:                               # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_c.3)
	addi	s9, s9, %pcrel_lo(.LBB0_17)
	lui	a0, 1
.LBB0_18:                               # Label of block must be emitted
	auipc	s8, %pcrel_hi(run_model.sram_a.1)
	addi	s8, s8, %pcrel_lo(.LBB0_18)
	addiw	s1, a0, -2048
.LBB0_19:                               # Label of block must be emitted
	auipc	s0, %pcrel_hi(run_model.sram_b.2)
	addi	s0, s0, %pcrel_lo(.LBB0_19)
	add	s7, s8, s1
	add	s11, s0, s1
.LBB0_9:                                # =>This Inner Loop Header: Depth=1
	slliw	a0, s2, 6
	slli	a0, a0, 2
	add	s6, s10, a0
	lui	a2, 1
	mv	a0, s9
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s8, s6

	#NO_APP
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s0, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s6, 64
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s11, s3

	#NO_APP
	#APP
	.insn r 11, 0, 2, s9, s8, s0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s6, 128
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s0, s4

	#NO_APP
	#APP
	.insn r 11, 0, 2, s9, s7, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s6, 192
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s11, s5

	#NO_APP
	#APP
	.insn r 11, 0, 2, s9, s8, s0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s9, s7, s11

	#NO_APP
	slliw	a0, s2, 5
	slli	a0, a0, 2
	add	a0, a0, s10
	lui	a1, 1
	#APP
	.insn r 11, 0, 1, zero, a0, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s2, s2, 32
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	blt	s2, a0, .LBB0_9
.LBB0_10:
	li	a0, 1024
	lui	a1, 8
.LBB0_11:                               # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(s10)
	add	a2, s10, a1
	flw	ft1, 256(a2)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(s10)
	addi	a0, a0, -1
	addi	s10, s10, 4
	bnez	a0, .LBB0_11
# %bb.12:
	ld	ra, 168(sp)                     # 8-byte Folded Reload
	ld	s0, 160(sp)                     # 8-byte Folded Reload
	ld	s1, 152(sp)                     # 8-byte Folded Reload
	ld	s2, 144(sp)                     # 8-byte Folded Reload
	ld	s3, 136(sp)                     # 8-byte Folded Reload
	ld	s4, 128(sp)                     # 8-byte Folded Reload
	ld	s5, 120(sp)                     # 8-byte Folded Reload
	ld	s6, 112(sp)                     # 8-byte Folded Reload
	ld	s7, 104(sp)                     # 8-byte Folded Reload
	ld	s8, 96(sp)                      # 8-byte Folded Reload
	ld	s9, 88(sp)                      # 8-byte Folded Reload
	ld	s10, 80(sp)                     # 8-byte Folded Reload
	ld	s11, 72(sp)                     # 8-byte Folded Reload
	addi	sp, sp, 176
	ret
.Lfunc_end0:
	.size	run_model, .Lfunc_end0-run_model
                                        # -- End function
	.type	run_model.sram_a,@object        # @run_model.sram_a
	.local	run_model.sram_a
	.comm	run_model.sram_a,2048,4
	.type	run_model.sram_b,@object        # @run_model.sram_b
	.local	run_model.sram_b
	.comm	run_model.sram_b,4096,4
	.type	run_model.sram_c,@object        # @run_model.sram_c
	.local	run_model.sram_c
	.comm	run_model.sram_c,8192,4
	.type	run_model.sram_bias,@object     # @run_model.sram_bias
	.local	run_model.sram_bias
	.comm	run_model.sram_bias,256,4
	.type	run_model.sram_a.1,@object      # @run_model.sram_a.1
	.local	run_model.sram_a.1
	.comm	run_model.sram_a.1,4096,4
	.type	run_model.sram_b.2,@object      # @run_model.sram_b.2
	.local	run_model.sram_b.2
	.comm	run_model.sram_b.2,4096,4
	.type	run_model.sram_c.3,@object      # @run_model.sram_c.3
	.local	run_model.sram_c.3
	.comm	run_model.sram_c.3,4096,4
	.ident	"Ubuntu clang version 15.0.7"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym run_model.sram_a
	.addrsig_sym run_model.sram_b
	.addrsig_sym run_model.sram_c
	.addrsig_sym run_model.sram_bias
	.addrsig_sym run_model.sram_a.1
	.addrsig_sym run_model.sram_b.2
	.addrsig_sym run_model.sram_c.3
