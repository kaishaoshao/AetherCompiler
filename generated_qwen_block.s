	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.file	"generated_qwen_block.c"
	.section	.sdata,"aw",@progbits
	.p2align	2                               # -- Begin function run_model
.LCPI0_0:
	.word	0x3c800000                      # float 0.015625
.LCPI0_1:
	.word	0x358637bd                      # float 9.99999997E-7
.LCPI0_2:
	.word	0x3f800000                      # float 1
	.text
	.globl	run_model
	.p2align	1
	.type	run_model,@function
run_model:                              # @run_model
# %bb.0:
	addi	sp, sp, -208
	sd	ra, 200(sp)                     # 8-byte Folded Spill
	sd	s0, 192(sp)                     # 8-byte Folded Spill
	sd	s1, 184(sp)                     # 8-byte Folded Spill
	sd	s2, 176(sp)                     # 8-byte Folded Spill
	sd	s3, 168(sp)                     # 8-byte Folded Spill
	sd	s4, 160(sp)                     # 8-byte Folded Spill
	sd	s5, 152(sp)                     # 8-byte Folded Spill
	sd	s6, 144(sp)                     # 8-byte Folded Spill
	sd	s7, 136(sp)                     # 8-byte Folded Spill
	sd	s8, 128(sp)                     # 8-byte Folded Spill
	sd	s9, 120(sp)                     # 8-byte Folded Spill
	sd	s10, 112(sp)                    # 8-byte Folded Spill
	sd	s11, 104(sp)                    # 8-byte Folded Spill
	fsd	fs0, 96(sp)                     # 8-byte Folded Spill
	fsd	fs1, 88(sp)                     # 8-byte Folded Spill
	fsd	fs2, 80(sp)                     # 8-byte Folded Spill
	fsd	fs3, 72(sp)                     # 8-byte Folded Spill
	mv	s6, a0
	lui	a0, 26
	add	a0, a0, s6
	lui	a2, 25
	add	s1, s6, a2
	lui	a2, 24
	add	s9, s6, a2
	lui	a2, 40
	add	s8, s6, a2
	lui	a2, 43
	add	a3, s6, a2
	#APP
	.insn r 11, 0, 4, a3, s8, a3

	#NO_APP
	#APP
	.insn r 11, 0, 2, s9, a3, s9

	#NO_APP
	lui	a2, 28
	add	a2, a2, s6
	#APP
	.insn r 11, 0, 2, s1, a3, a2

	#NO_APP
	lui	a2, 32
	add	a2, a2, s6
	#APP
	.insn r 11, 0, 2, a0, a3, a2

	#NO_APP
	lui	a2, 44
	add	a5, s6, a2
	li	a2, 1024
	mv	a3, s9
	mv	a4, a5
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a3)
	flw	ft1, 0(a4)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a3)
	addi	a4, a4, 4
	addi	a2, a2, -1
	addi	a3, a3, 4
	bnez	a2, .LBB0_1
# %bb.2:
	addi	a2, a5, 256
	li	a3, 1024
	mv	a4, s1
.LBB0_3:                                # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a4)
	flw	ft1, 0(a2)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a4)
	addi	a2, a2, 4
	addi	a3, a3, -1
	addi	a4, a4, 4
	bnez	a3, .LBB0_3
# %bb.4:
	addi	a3, a5, 512
	li	a2, 1024
.LBB0_5:                                # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a0)
	flw	ft1, 0(a3)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a0)
	addi	a3, a3, 4
	addi	a2, a2, -1
	addi	a0, a0, 4
	bnez	a2, .LBB0_5
# %bb.6:
	lui	a0, 41
	add	a0, a0, s6
	#APP
	.insn r 11, 0, 6, s9, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 6, s1, s1, a0

	#NO_APP
	sd	s1, 16(sp)                      # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 7, s9, s9, s1

	#NO_APP
	sd	a1, 8(sp)                       # 8-byte Folded Spill
	slliw	a2, a1, 2
	li	a1, 12
	mv	a0, a2
	blt	a2, a1, .LBB0_8
# %bb.7:
	li	a0, 12
.LBB0_8:
	addiw	a3, a0, 4
	sd	a2, 64(sp)                      # 8-byte Folded Spill
	sd	a3, 56(sp)                      # 8-byte Folded Spill
	bge	a2, a3, .LBB0_11
# %bb.9:
	lui	a0, 36
	add	a0, a0, s6
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	lui	a0, 37
	add	a0, a0, s6
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	lui	a0, 38
	add	a0, a0, s6
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	lui	a0, 39
	add	a0, a0, s6
	sd	a0, 24(sp)                      # 8-byte Folded Spill
.LBB0_27:                               # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_c)
	addi	s7, s7, %pcrel_lo(.LBB0_27)
	lui	s0, 1
.LBB0_28:                               # Label of block must be emitted
	auipc	s2, %pcrel_hi(run_model.sram_a)
	addi	s2, s2, %pcrel_lo(.LBB0_28)
	li	s3, 1024
.LBB0_29:                               # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_b)
	addi	s5, s5, %pcrel_lo(.LBB0_29)
	addi	s10, s2, 1024
	add	s11, s5, s0
	ld	s4, 64(sp)                      # 8-byte Folded Reload
.LBB0_10:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s4, 6
	slli	a0, a0, 2
	add	s1, s9, a0
	lui	a2, 1
	mv	a0, s7
	li	a1, 0
	call	memset@plt
	ld	a3, 56(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, s1

	#NO_APP
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s5, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s1, 64
	#APP
	.insn r 11, 0, 1, zero, s10, a0

	#NO_APP
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s7, s2, s5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s1, 128
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s5, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s7, s10, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s1, 192
	#APP
	.insn r 11, 0, 1, zero, s10, a0

	#NO_APP
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s7, s2, s5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s7, s10, s11

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s1, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s4, s4, 16
	blt	s4, a3, .LBB0_10
.LBB0_11:
	li	s7, 0
	lui	a0, 44
	addiw	a0, a0, 768
	add	s10, s6, a0
.LBB0_30:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_0)
	addi	a0, a0, %pcrel_lo(.LBB0_30)
	flw	fs0, 0(a0)
.LBB0_31:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_1)
	addi	a0, a0, %pcrel_lo(.LBB0_31)
	flw	fs1, 0(a0)
.LBB0_32:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_2)
	addi	a0, a0, %pcrel_lo(.LBB0_32)
	flw	fs2, 0(a0)
	fmv.w.x	fs3, zero
	li	s0, 256
	li	s5, 16
	ld	s2, 16(sp)                      # 8-byte Folded Reload
	mv	s1, s2
	ld	s4, 64(sp)                      # 8-byte Folded Reload
.LBB0_12:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_13 Depth 2
                                        #     Child Loop BB0_16 Depth 2
	li	a0, 0
	fmv.s	ft0, fs3
.LBB0_13:                               #   Parent Loop BB0_12 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s8, a0
	flw	ft1, 0(a1)
	add	a1, s9, a0
	flw	ft2, 0(a1)
	fadd.s	ft1, ft1, ft2
	fsw	ft1, 0(a1)
	addi	a0, a0, 4
	fmadd.s	ft0, ft1, ft1, ft0
	bne	a0, s0, .LBB0_13
# %bb.14:                               #   in Loop: Header=BB0_12 Depth=1
	fmul.s	ft0, ft0, fs0
	fadd.s	ft0, ft0, fs1
	fsqrt.s	fa0, ft0
	feq.s	a0, fa0, fa0
	beqz	a0, .LBB0_18
.LBB0_15:                               #   in Loop: Header=BB0_12 Depth=1
	li	a0, 0
	fdiv.s	ft0, fs2, fa0
.LBB0_16:                               #   Parent Loop BB0_12 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s9, a0
	flw	ft1, 0(a1)
	add	a1, s10, a0
	flw	ft2, 0(a1)
	fmul.s	ft1, ft0, ft1
	fmul.s	ft1, ft1, ft2
	add	a1, s1, a0
	addi	a0, a0, 4
	fsw	ft1, 0(a1)
	bne	a0, s0, .LBB0_16
# %bb.17:                               #   in Loop: Header=BB0_12 Depth=1
	addi	s7, s7, 1
	addi	s9, s9, 256
	addi	s8, s8, 256
	addi	s1, s1, 256
	bne	s7, s5, .LBB0_12
	j	.LBB0_19
.LBB0_18:                               #   in Loop: Header=BB0_12 Depth=1
	fmv.s	fa0, ft0
	call	sqrtf@plt
	ld	a3, 56(sp)                      # 8-byte Folded Reload
	j	.LBB0_15
.LBB0_19:
	#APP
	.insn r 11, 0, 2, s6, s2, s6

	#NO_APP
	lui	a0, 8
	add	a0, a0, s6
	lui	a1, 2
	add	a1, a1, s6
	#APP
	.insn r 11, 0, 2, a1, s2, a0

	#NO_APP
	#APP
	.insn r 11, 0, 5, s6, s6, a1

	#NO_APP
	bge	s4, a3, .LBB0_24
# %bb.20:
	lui	a0, 16
	add	a0, a0, s6
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	lui	a0, 17
	add	a0, a0, s6
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	addi	a0, s6, 64
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	slliw	s2, a0, 9
.LBB0_33:                               # Label of block must be emitted
	auipc	s11, %pcrel_hi(run_model.sram_c.3)
	addi	s11, s11, %pcrel_lo(.LBB0_33)
	lui	s9, 1
.LBB0_34:                               # Label of block must be emitted
	auipc	s8, %pcrel_hi(run_model.sram_a.1)
	addi	s8, s8, %pcrel_lo(.LBB0_34)
	li	s10, 1024
.LBB0_35:                               # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_b.2)
	addi	s1, s1, %pcrel_lo(.LBB0_35)
	li	s5, 1
	li	s3, 96
.LBB0_21:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_22 Depth 2
	slli	a0, s2, 2
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	s7, a1, a0
	sd	s4, 64(sp)                      # 8-byte Folded Spill
	slliw	a0, s4, 7
	slli	a0, a0, 2
	add	s4, s6, a0
	lui	a2, 1
	mv	a0, s11
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s8, s4

	#NO_APP
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s1, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	li	a2, 0
	li	a0, -16
	ld	a1, 40(sp)                      # 8-byte Folded Reload
.LBB0_22:                               #   Parent Loop BB0_21 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	slli	a5, a2, 32
	subw	a2, s5, a2
	slli	a4, a2, 32
	srli	a3, a4, 22
	add	a3, a3, s8
	#APP
	.insn r 11, 0, 1, zero, a3, s7

	#NO_APP
	srli	a4, a4, 20
	add	a4, a4, s1
	#APP
	.insn r 11, 0, 1, zero, a4, a1

	#NO_APP
	srli	s0, a5, 22
	add	s0, s0, s8
	srli	a5, a5, 20
	add	a5, a5, s1
	#APP
	.insn r 11, 0, 2, s11, s0, a5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, a0, 16
	add	a1, a1, s9
	addi	s7, s7, 64
	bltu	a0, s3, .LBB0_22
# %bb.23:                               #   in Loop: Header=BB0_21 Depth=1
	#APP
	.insn r 11, 0, 2, s11, a3, a4

	#NO_APP
	ld	s4, 64(sp)                      # 8-byte Folded Reload
	slliw	a0, s4, 6
	slli	a0, a0, 2
	add	a0, a0, s6
	#APP
	.insn r 11, 0, 1, zero, a0, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s4, s4, 16
	addiw	a0, s2, 2047
	addiw	s2, a0, 1
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	blt	s4, a0, .LBB0_21
.LBB0_24:
	li	a0, 1024
	lui	a1, 24
.LBB0_25:                               # =>This Inner Loop Header: Depth=1
	add	a2, s6, a1
	flw	ft0, 0(a2)
	flw	ft1, 0(s6)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(s6)
	addi	a0, a0, -1
	addi	s6, s6, 4
	bnez	a0, .LBB0_25
# %bb.26:
	ld	ra, 200(sp)                     # 8-byte Folded Reload
	ld	s0, 192(sp)                     # 8-byte Folded Reload
	ld	s1, 184(sp)                     # 8-byte Folded Reload
	ld	s2, 176(sp)                     # 8-byte Folded Reload
	ld	s3, 168(sp)                     # 8-byte Folded Reload
	ld	s4, 160(sp)                     # 8-byte Folded Reload
	ld	s5, 152(sp)                     # 8-byte Folded Reload
	ld	s6, 144(sp)                     # 8-byte Folded Reload
	ld	s7, 136(sp)                     # 8-byte Folded Reload
	ld	s8, 128(sp)                     # 8-byte Folded Reload
	ld	s9, 120(sp)                     # 8-byte Folded Reload
	ld	s10, 112(sp)                    # 8-byte Folded Reload
	ld	s11, 104(sp)                    # 8-byte Folded Reload
	fld	fs0, 96(sp)                     # 8-byte Folded Reload
	fld	fs1, 88(sp)                     # 8-byte Folded Reload
	fld	fs2, 80(sp)                     # 8-byte Folded Reload
	fld	fs3, 72(sp)                     # 8-byte Folded Reload
	addi	sp, sp, 208
	ret
.Lfunc_end0:
	.size	run_model, .Lfunc_end0-run_model
                                        # -- End function
	.type	run_model.sram_a,@object        # @run_model.sram_a
	.local	run_model.sram_a
	.comm	run_model.sram_a,2048,4
	.type	run_model.sram_b,@object        # @run_model.sram_b
	.local	run_model.sram_b
	.comm	run_model.sram_b,8192,4
	.type	run_model.sram_c,@object        # @run_model.sram_c
	.local	run_model.sram_c
	.comm	run_model.sram_c,4096,4
	.type	run_model.sram_a.1,@object      # @run_model.sram_a.1
	.local	run_model.sram_a.1
	.comm	run_model.sram_a.1,2048,4
	.type	run_model.sram_b.2,@object      # @run_model.sram_b.2
	.local	run_model.sram_b.2
	.comm	run_model.sram_b.2,8192,4
	.type	run_model.sram_c.3,@object      # @run_model.sram_c.3
	.local	run_model.sram_c.3
	.comm	run_model.sram_c.3,4096,4
	.ident	"Ubuntu clang version 15.0.7"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym run_model.sram_a
	.addrsig_sym run_model.sram_b
	.addrsig_sym run_model.sram_c
	.addrsig_sym run_model.sram_a.1
	.addrsig_sym run_model.sram_b.2
	.addrsig_sym run_model.sram_c.3
