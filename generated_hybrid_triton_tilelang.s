	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.file	"generated_hybrid_triton_tilelang.c"
	.section	.sdata,"aw",@progbits
	.p2align	2                               # -- Begin function run_model
.LCPI0_0:
	.word	0x3f000000                      # float 0.5
.LCPI0_1:
	.word	0x3fb504f3                      # float 1.41421354
.LCPI0_2:
	.word	0x3f800000                      # float 1
	.text
	.globl	run_model
	.p2align	1
	.type	run_model,@function
run_model:                              # @run_model
# %bb.0:
	addi	sp, sp, -160
	sd	ra, 152(sp)                     # 8-byte Folded Spill
	sd	s0, 144(sp)                     # 8-byte Folded Spill
	sd	s1, 136(sp)                     # 8-byte Folded Spill
	sd	s2, 128(sp)                     # 8-byte Folded Spill
	sd	s3, 120(sp)                     # 8-byte Folded Spill
	sd	s4, 112(sp)                     # 8-byte Folded Spill
	sd	s5, 104(sp)                     # 8-byte Folded Spill
	sd	s6, 96(sp)                      # 8-byte Folded Spill
	sd	s7, 88(sp)                      # 8-byte Folded Spill
	sd	s8, 80(sp)                      # 8-byte Folded Spill
	sd	s9, 72(sp)                      # 8-byte Folded Spill
	sd	s10, 64(sp)                     # 8-byte Folded Spill
	sd	s11, 56(sp)                     # 8-byte Folded Spill
	fsd	fs0, 48(sp)                     # 8-byte Folded Spill
	fsd	fs1, 40(sp)                     # 8-byte Folded Spill
	fsd	fs2, 32(sp)                     # 8-byte Folded Spill
	fsd	fs3, 24(sp)                     # 8-byte Folded Spill
	slliw	s5, a1, 3
	li	a2, 24
	mv	s3, a0
	mv	a0, s5
	blt	s5, a2, .LBB0_2
# %bb.1:
	li	a0, 24
.LBB0_2:
	addiw	a0, a0, 8
	sd	a0, 16(sp)                      # 8-byte Folded Spill
	bge	s5, a0, .LBB0_16
# %bb.3:
	sd	a1, 8(sp)                       # 8-byte Folded Spill
	lui	a0, 2
	add	s6, s3, a0
	lui	a0, 3
	addiw	a1, a0, -2048
	add	s7, s3, a1
	add	s8, s3, a0
	lui	a0, 4
	addiw	a0, a0, -2048
	add	s9, s3, a0
.LBB0_17:                               # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_tile_c)
	addi	s1, s1, %pcrel_lo(.LBB0_17)
	lui	a0, 1
	addiw	s0, a0, -2048
.LBB0_18:                               # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_tile_a)
	addi	s10, s10, %pcrel_lo(.LBB0_18)
	li	s11, 1024
.LBB0_19:                               # Label of block must be emitted
	auipc	s2, %pcrel_hi(run_model.sram_tile_b)
	addi	s2, s2, %pcrel_lo(.LBB0_19)
	fmv.w.x	fs0, zero
	sd	s5, 0(sp)                       # 8-byte Folded Spill
	j	.LBB0_5
.LBB0_4:                                #   in Loop: Header=BB0_5 Depth=1
	slliw	a0, s5, 5
	slli	a0, a0, 2
	add	a0, a0, s3
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s5, s5, 16
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	bge	s5, a0, .LBB0_9
.LBB0_5:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_7 Depth 2
	slliw	s4, s5, 6
	mv	a0, s1
	li	a1, 0
	mv	a2, s0
	call	memset@plt
	slli	a0, s4, 2
	add	a0, a0, s3
	#APP
	.insn r 11, 0, 1, zero, s10, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s2, s6

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s2

	#NO_APP
	addi	a1, a0, 64
	#APP
	.insn r 11, 0, 1, zero, s10, a1

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s2, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s2

	#NO_APP
	addi	a1, a0, 128
	#APP
	.insn r 11, 0, 1, zero, s10, a1

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s2, s8

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s2

	#NO_APP
	addi	a0, a0, 192
	#APP
	.insn r 11, 0, 1, zero, s10, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s2, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s2

	#NO_APP
	li	a0, 0
	j	.LBB0_7
.LBB0_6:                                #   in Loop: Header=BB0_7 Depth=2
	addi	a0, a0, 4
	beq	a0, s0, .LBB0_4
.LBB0_7:                                #   Parent Loop BB0_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s1, a0
	flw	ft0, 0(a1)
	flt.s	a2, ft0, fs0
	beqz	a2, .LBB0_6
# %bb.8:                                #   in Loop: Header=BB0_7 Depth=2
	sw	zero, 0(a1)
	j	.LBB0_6
.LBB0_9:
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	ld	a1, 8(sp)                       # 8-byte Folded Reload
	ld	a2, 0(sp)                       # 8-byte Folded Reload
	bge	a2, a0, .LBB0_16
# %bb.10:
	lui	a0, 4
	add	s6, s3, a0
	slli	a0, a2, 32
	srli	s2, a0, 32
	slliw	s9, a1, 8
	slli	a0, s9, 2
.LBB0_20:                               # Label of block must be emitted
	auipc	a1, %pcrel_hi(.LCPI0_0)
	addi	a1, a1, %pcrel_lo(.LBB0_20)
	flw	fs0, 0(a1)
.LBB0_21:                               # Label of block must be emitted
	auipc	a1, %pcrel_hi(.LCPI0_1)
	addi	a1, a1, %pcrel_lo(.LBB0_21)
	flw	fs1, 0(a1)
.LBB0_22:                               # Label of block must be emitted
	auipc	a1, %pcrel_hi(.LCPI0_2)
	addi	a1, a1, %pcrel_lo(.LBB0_22)
	flw	fs2, 0(a1)
	add	s5, s3, a0
	li	s7, 1023
	li	s8, 32
	j	.LBB0_12
.LBB0_11:                               #   in Loop: Header=BB0_12 Depth=1
	addiw	a0, s2, 1
	addi	s2, s2, 1
	addi	s5, s5, 128
	addi	s9, s9, 32
	ld	a1, 16(sp)                      # 8-byte Folded Reload
	beq	a1, a0, .LBB0_16
.LBB0_12:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_14 Depth 2
	li	s0, 0
	mv	s1, s5
	j	.LBB0_14
.LBB0_13:                               #   in Loop: Header=BB0_14 Depth=2
	addi	s0, s0, 1
	addi	s1, s1, 4
	beq	s0, s8, .LBB0_11
.LBB0_14:                               #   Parent Loop BB0_12 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a0, s9, s0
	blt	s7, a0, .LBB0_13
# %bb.15:                               #   in Loop: Header=BB0_14 Depth=2
	flw	ft0, 0(s1)
	sraiw	a1, a0, 31
	srliw	a1, a1, 27
	add	a1, a1, a0
	andi	a1, a1, -32
	subw	a0, a0, a1
	slli	a0, a0, 2
	add	a0, a0, s6
	flw	ft1, 0(a0)
	fadd.s	ft0, ft0, ft1
	fmul.s	fs3, ft0, fs0
	fdiv.s	fa0, ft0, fs1
	call	erff@plt
	fadd.s	ft0, fa0, fs2
	fmul.s	ft0, fs3, ft0
	fsw	ft0, 0(s1)
	j	.LBB0_13
.LBB0_16:
	lui	a0, 4
	addiw	a0, a0, 128
	add	a0, a0, s3
	#APP
	.insn r 11, 0, 4, s3, s3, a0

	#NO_APP
	ld	ra, 152(sp)                     # 8-byte Folded Reload
	ld	s0, 144(sp)                     # 8-byte Folded Reload
	ld	s1, 136(sp)                     # 8-byte Folded Reload
	ld	s2, 128(sp)                     # 8-byte Folded Reload
	ld	s3, 120(sp)                     # 8-byte Folded Reload
	ld	s4, 112(sp)                     # 8-byte Folded Reload
	ld	s5, 104(sp)                     # 8-byte Folded Reload
	ld	s6, 96(sp)                      # 8-byte Folded Reload
	ld	s7, 88(sp)                      # 8-byte Folded Reload
	ld	s8, 80(sp)                      # 8-byte Folded Reload
	ld	s9, 72(sp)                      # 8-byte Folded Reload
	ld	s10, 64(sp)                     # 8-byte Folded Reload
	ld	s11, 56(sp)                     # 8-byte Folded Reload
	fld	fs0, 48(sp)                     # 8-byte Folded Reload
	fld	fs1, 40(sp)                     # 8-byte Folded Reload
	fld	fs2, 32(sp)                     # 8-byte Folded Reload
	fld	fs3, 24(sp)                     # 8-byte Folded Reload
	addi	sp, sp, 160
	ret
.Lfunc_end0:
	.size	run_model, .Lfunc_end0-run_model
                                        # -- End function
	.type	run_model.sram_tile_a,@object   # @run_model.sram_tile_a
	.local	run_model.sram_tile_a
	.comm	run_model.sram_tile_a,1024,4
	.type	run_model.sram_tile_b,@object   # @run_model.sram_tile_b
	.local	run_model.sram_tile_b
	.comm	run_model.sram_tile_b,2048,4
	.type	run_model.sram_tile_c,@object   # @run_model.sram_tile_c
	.local	run_model.sram_tile_c
	.comm	run_model.sram_tile_c,2048,4
	.ident	"Ubuntu clang version 15.0.7"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym run_model.sram_tile_a
	.addrsig_sym run_model.sram_tile_b
	.addrsig_sym run_model.sram_tile_c
