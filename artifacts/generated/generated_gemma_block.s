	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.file	"-"
	.section	.sdata,"aw",@progbits
	.p2align	2                               # -- Begin function run_model
.LCPI0_0:
	.word	0x3c800000                      # float 0.015625
.LCPI0_1:
	.word	0x358637bd                      # float 9.99999997E-7
.LCPI0_2:
	.word	0x3f800000                      # float 1
.LCPI0_3:
	.word	0x3f000000                      # float 0.5
.LCPI0_4:
	.word	0x3fb504f3                      # float 1.41421354
	.text
	.globl	run_model
	.p2align	1
	.type	run_model,@function
run_model:                              # @run_model
# %bb.0:
	addi	sp, sp, -224
	sd	ra, 216(sp)                     # 8-byte Folded Spill
	sd	s0, 208(sp)                     # 8-byte Folded Spill
	sd	s1, 200(sp)                     # 8-byte Folded Spill
	sd	s2, 192(sp)                     # 8-byte Folded Spill
	sd	s3, 184(sp)                     # 8-byte Folded Spill
	sd	s4, 176(sp)                     # 8-byte Folded Spill
	sd	s5, 168(sp)                     # 8-byte Folded Spill
	sd	s6, 160(sp)                     # 8-byte Folded Spill
	sd	s7, 152(sp)                     # 8-byte Folded Spill
	sd	s8, 144(sp)                     # 8-byte Folded Spill
	sd	s9, 136(sp)                     # 8-byte Folded Spill
	sd	s10, 128(sp)                    # 8-byte Folded Spill
	sd	s11, 120(sp)                    # 8-byte Folded Spill
	fsd	fs0, 112(sp)                    # 8-byte Folded Spill
	fsd	fs1, 104(sp)                    # 8-byte Folded Spill
	fsd	fs2, 96(sp)                     # 8-byte Folded Spill
	fsd	fs3, 88(sp)                     # 8-byte Folded Spill
	lui	a2, 40
	add	s6, a0, a2
	lui	a2, 43
	add	a3, a0, a2
	#APP
	.insn r 11, 0, 4, a3, s6, a3

	#NO_APP
	lui	a2, 24
	add	s4, a0, a2
	#APP
	.insn r 11, 0, 2, s4, a3, s4

	#NO_APP
	lui	a2, 28
	add	a4, a0, a2
	lui	a2, 25
	add	a5, a0, a2
	#APP
	.insn r 11, 0, 2, a5, a3, a4

	#NO_APP
	lui	a2, 32
	add	a4, a0, a2
	lui	a2, 26
	add	a2, a2, a0
	#APP
	.insn r 11, 0, 2, a2, a3, a4

	#NO_APP
	lui	a2, 41
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	add	a0, a0, a2
	#APP
	.insn r 11, 0, 6, s4, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 6, a5, a5, a0

	#NO_APP
	sd	a5, 8(sp)                       # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 7, s4, s4, a5

	#NO_APP
	sd	a1, 16(sp)                      # 8-byte Folded Spill
	slliw	a3, a1, 2
	li	a1, 12
	mv	a0, a3
	blt	a3, a1, .LBB0_2
# %bb.1:
	li	a0, 12
.LBB0_2:
	addiw	a2, a0, 4
	sd	a2, 80(sp)                      # 8-byte Folded Spill
	sd	a3, 24(sp)                      # 8-byte Folded Spill
	bge	a3, a2, .LBB0_20
# %bb.3:
	lui	a0, 36
	ld	a1, 56(sp)                      # 8-byte Folded Reload
	add	s8, a1, a0
	lui	a0, 24
	add	a3, a1, a0
	sd	a3, 40(sp)                      # 8-byte Folded Spill
	ld	a3, 16(sp)                      # 8-byte Folded Reload
	slliw	a3, a3, 8
	addiw	a0, a0, 64
	add	a0, a0, a1
	sd	a0, 48(sp)                      # 8-byte Folded Spill
.LBB0_51:                               # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_c)
	addi	s7, s7, %pcrel_lo(.LBB0_51)
.LBB0_52:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(run_model.sram_a)
	addi	a0, a0, %pcrel_lo(.LBB0_52)
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	li	s2, 64
.LBB0_53:                               # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_b)
	addi	s9, s9, %pcrel_lo(.LBB0_53)
	li	s11, 47
	li	s3, 48
	ld	s1, 24(sp)                      # 8-byte Folded Reload
	j	.LBB0_6
.LBB0_4:                                #   in Loop: Header=BB0_6 Depth=1
	slli	a0, a3, 32
	srli	a1, a0, 22
.LBB0_54:                               #   in Loop: Header=BB0_6 Depth=1
                                        # Label of block must be emitted
	auipc	a2, %pcrel_hi(run_model.sram_a)
	addi	a2, a2, %pcrel_lo(.LBB0_54)
	add	a1, a1, a2
	srli	a0, a0, 20
	add	a0, a0, s9
	#APP
	.insn r 11, 0, 2, s7, a1, a0

	#NO_APP
.LBB0_5:                                #   in Loop: Header=BB0_6 Depth=1
	#APP
	.insn r 11, 0, 1, zero, a6, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a3, 72(sp)                      # 8-byte Folded Reload
	addiw	a3, a3, 1024
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	mv	s1, a0
	ld	a2, 80(sp)                      # 8-byte Folded Reload
	bge	a0, a2, .LBB0_20
.LBB0_6:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_11 Depth 2
                                        #     Child Loop BB0_14 Depth 2
                                        #       Child Loop BB0_19 Depth 3
	addi	a0, s1, 16
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	blt	a2, a0, .LBB0_8
# %bb.7:                                #   in Loop: Header=BB0_6 Depth=1
	li	s10, 16
	j	.LBB0_9
.LBB0_8:                                #   in Loop: Header=BB0_6 Depth=1
	subw	s10, a2, s1
.LBB0_9:                                #   in Loop: Header=BB0_6 Depth=1
	sd	a3, 72(sp)                      # 8-byte Folded Spill
	slli	s0, a3, 2
	slli	a0, s10, 32
	srli	s5, a0, 32
	lui	a2, 1
	mv	a0, s7
	li	a1, 0
	call	memset@plt
	blez	s10, .LBB0_12
# %bb.10:                               #   in Loop: Header=BB0_6 Depth=1
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	add	a0, a0, s0
	mv	a1, s5
	ld	a2, 32(sp)                      # 8-byte Folded Reload
.LBB0_11:                               #   Parent Loop BB0_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a2, a0

	#NO_APP
	addi	a0, a0, 256
	addi	a1, a1, -1
	addi	a2, a2, 64
	bnez	a1, .LBB0_11
.LBB0_12:                               #   in Loop: Header=BB0_6 Depth=1
	li	a4, 0
	li	a3, 0
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	add	s0, s0, a0
	slliw	a0, s1, 6
	lui	a1, 1
	#APP
	.insn r 11, 0, 1, zero, s9, s8

	#NO_APP
	slli	a0, a0, 2
	add	a6, s4, a0
	slli	a7, s10, 8
	li	t3, 1
	li	t4, 33
	j	.LBB0_14
.LBB0_13:                               #   in Loop: Header=BB0_14 Depth=2
	addi	a4, s1, 16
	srli	a0, a1, 32
	slli	a0, a0, 12
	add	a0, a0, s9
	slli	a1, a4, 8
	add	a1, a1, s8
	slli	a2, t2, 8
	#APP
	.insn r 11, 0, 1, zero, a0, a1

	#NO_APP
	slli	a0, a3, 32
	srli	a1, a0, 22
	add	a1, a1, t0
	srli	a0, a0, 20
	add	a0, a0, s9
	#APP
	.insn r 11, 0, 2, s7, a1, a0

	#NO_APP
	addi	s0, s0, 64
	mv	a3, t1
	bgeu	s1, s3, .LBB0_5
.LBB0_14:                               #   Parent Loop BB0_6 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_19 Depth 3
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	bltu	s11, a4, .LBB0_4
# %bb.15:                               #   in Loop: Header=BB0_14 Depth=2
	mv	s1, a4
	li	t2, 16
	bltu	a4, t4, .LBB0_17
# %bb.16:                               #   in Loop: Header=BB0_14 Depth=2
	subw	t2, s3, s1
.LBB0_17:                               #   in Loop: Header=BB0_14 Depth=2
	subw	t1, t3, a3
	slli	a1, t1, 32
.LBB0_55:                               #   in Loop: Header=BB0_14 Depth=2
                                        # Label of block must be emitted
	auipc	t0, %pcrel_hi(run_model.sram_a)
	addi	t0, t0, %pcrel_lo(.LBB0_55)
	blez	s10, .LBB0_13
# %bb.18:                               #   in Loop: Header=BB0_14 Depth=2
	srli	a0, a1, 22
	add	a4, t0, a0
	slli	a0, t2, 32
	srli	a0, a0, 30
	mv	a5, s5
	mv	a2, s0
.LBB0_19:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_14 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	#APP
	.insn r 11, 0, 1, zero, a4, a2

	#NO_APP
	addi	a2, a2, 256
	addi	a5, a5, -1
	addi	a4, a4, 64
	bnez	a5, .LBB0_19
	j	.LBB0_13
.LBB0_20:
	li	s2, 0
	lui	a0, 44
	ld	a1, 56(sp)                      # 8-byte Folded Reload
	add	s5, a1, a0
.LBB0_56:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_0)
	addi	a0, a0, %pcrel_lo(.LBB0_56)
	flw	fs1, 0(a0)
.LBB0_57:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_1)
	addi	a0, a0, %pcrel_lo(.LBB0_57)
	flw	fs2, 0(a0)
.LBB0_58:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_2)
	addi	a0, a0, %pcrel_lo(.LBB0_58)
	flw	fs0, 0(a0)
	fmv.w.x	fs3, zero
	li	s1, 256
	li	s3, 16
	ld	s7, 8(sp)                       # 8-byte Folded Reload
	mv	s0, s7
.LBB0_21:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_22 Depth 2
                                        #     Child Loop BB0_25 Depth 2
	li	a0, 0
	fmv.s	ft0, fs3
.LBB0_22:                               #   Parent Loop BB0_21 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s6, a0
	flw	ft1, 0(a1)
	add	a1, s4, a0
	flw	ft2, 0(a1)
	fadd.s	ft1, ft1, ft2
	fsw	ft1, 0(a1)
	addi	a0, a0, 4
	fmadd.s	ft0, ft1, ft1, ft0
	bne	a0, s1, .LBB0_22
# %bb.23:                               #   in Loop: Header=BB0_21 Depth=1
	fmul.s	ft0, ft0, fs1
	fadd.s	ft0, ft0, fs2
	fsqrt.s	fa0, ft0
	feq.s	a0, fa0, fa0
	beqz	a0, .LBB0_27
.LBB0_24:                               #   in Loop: Header=BB0_21 Depth=1
	li	a0, 0
	fdiv.s	ft0, fs0, fa0
.LBB0_25:                               #   Parent Loop BB0_21 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s4, a0
	flw	ft1, 0(a1)
	add	a1, s5, a0
	flw	ft2, 0(a1)
	fmul.s	ft1, ft0, ft1
	fmul.s	ft1, ft1, ft2
	add	a1, s0, a0
	addi	a0, a0, 4
	fsw	ft1, 0(a1)
	bne	a0, s1, .LBB0_25
# %bb.26:                               #   in Loop: Header=BB0_21 Depth=1
	addi	s2, s2, 1
	addi	s4, s4, 256
	addi	s6, s6, 256
	addi	s0, s0, 256
	bne	s2, s3, .LBB0_21
	j	.LBB0_28
.LBB0_27:                               #   in Loop: Header=BB0_21 Depth=1
	fmv.s	fa0, ft0
	call	sqrtf@plt
	j	.LBB0_24
.LBB0_28:
	ld	s1, 56(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 2, s1, s7, s1

	#NO_APP
	lui	a0, 8
	add	a0, a0, s1
	lui	s2, 2
	add	s3, s1, s2
	#APP
	.insn r 11, 0, 2, s3, s7, a0

	#NO_APP
.LBB0_59:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_3)
	addi	a0, a0, %pcrel_lo(.LBB0_59)
	flw	fs1, 0(a0)
.LBB0_60:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_4)
	addi	a0, a0, %pcrel_lo(.LBB0_60)
	flw	fs2, 0(a0)
	li	s0, 0
.LBB0_29:                               # =>This Inner Loop Header: Depth=1
	add	s4, s1, s0
	flw	ft0, 0(s4)
	fmul.s	fs3, ft0, fs1
	fdiv.s	fa0, ft0, fs2
	call	erff@plt
	add	a0, s3, s0
	flw	ft0, 0(a0)
	fadd.s	ft1, fa0, fs0
	fmul.s	ft1, fs3, ft1
	fmul.s	ft0, ft0, ft1
	addi	s0, s0, 4
	fsw	ft0, 0(s4)
	bne	s0, s2, .LBB0_29
# %bb.30:
	ld	a1, 80(sp)                      # 8-byte Folded Reload
	ld	a2, 24(sp)                      # 8-byte Folded Reload
	bge	a2, a1, .LBB0_48
# %bb.31:
	lui	a0, 16
	add	s10, s1, a0
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	slliw	a3, a0, 9
	addi	a0, s1, 64
	sd	a0, 48(sp)                      # 8-byte Folded Spill
.LBB0_61:                               # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_c.3)
	addi	s4, s4, %pcrel_lo(.LBB0_61)
.LBB0_62:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(run_model.sram_a.1)
	addi	a0, a0, %pcrel_lo(.LBB0_62)
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	li	s9, 64
.LBB0_63:                               # Label of block must be emitted
	auipc	s3, %pcrel_hi(run_model.sram_b.2)
	addi	s3, s3, %pcrel_lo(.LBB0_63)
	li	s11, 1
	li	s7, 111
	li	s8, 112
	li	s2, 97
	j	.LBB0_34
.LBB0_32:                               #   in Loop: Header=BB0_34 Depth=1
	slli	a0, a3, 32
	srli	a1, a0, 22
.LBB0_64:                               #   in Loop: Header=BB0_34 Depth=1
                                        # Label of block must be emitted
	auipc	a2, %pcrel_hi(run_model.sram_a.1)
	addi	a2, a2, %pcrel_lo(.LBB0_64)
	add	a1, a1, a2
	srli	a0, a0, 20
	add	a0, a0, s3
	#APP
	.insn r 11, 0, 2, s4, a1, a0

	#NO_APP
.LBB0_33:                               #   in Loop: Header=BB0_34 Depth=1
	slli	a0, a7, 2
	ld	s1, 56(sp)                      # 8-byte Folded Reload
	add	a0, a0, s1
	#APP
	.insn r 11, 0, 1, zero, a0, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	addiw	a0, a0, 2047
	addiw	a3, a0, 1
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	mv	a2, a0
	ld	a1, 80(sp)                      # 8-byte Folded Reload
	bge	a0, a1, .LBB0_48
.LBB0_34:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_39 Depth 2
                                        #     Child Loop BB0_42 Depth 2
                                        #       Child Loop BB0_47 Depth 3
	addi	a0, a2, 16
	mv	s1, a2
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	blt	a1, a0, .LBB0_36
# %bb.35:                               #   in Loop: Header=BB0_34 Depth=1
	li	s6, 16
	j	.LBB0_37
.LBB0_36:                               #   in Loop: Header=BB0_34 Depth=1
	subw	s6, a1, s1
.LBB0_37:                               #   in Loop: Header=BB0_34 Depth=1
	sd	a3, 72(sp)                      # 8-byte Folded Spill
	slli	s0, a3, 2
	slli	a0, s6, 32
	srli	s5, a0, 32
	lui	a2, 1
	mv	a0, s4
	li	a1, 0
	call	memset@plt
	blez	s6, .LBB0_40
# %bb.38:                               #   in Loop: Header=BB0_34 Depth=1
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	add	a0, a0, s0
	mv	a1, s5
	ld	a2, 40(sp)                      # 8-byte Folded Reload
.LBB0_39:                               #   Parent Loop BB0_34 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a2, a0

	#NO_APP
	addi	a0, a0, 512
	addi	a1, a1, -1
	addi	a2, a2, 64
	bnez	a1, .LBB0_39
.LBB0_40:                               #   in Loop: Header=BB0_34 Depth=1
	li	a4, 0
	li	a3, 0
	lui	a0, 1
	#APP
	.insn r 11, 0, 1, zero, s3, s10

	#NO_APP
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	add	a0, a0, s0
	slliw	a7, s1, 6
	slli	a6, s6, 8
	j	.LBB0_42
.LBB0_41:                               #   in Loop: Header=BB0_42 Depth=2
	addi	a4, s0, 16
	srli	a1, s1, 32
	slli	a1, a1, 12
	add	a1, a1, s3
	slli	a2, a4, 8
	add	a2, a2, s10
	slli	a5, t2, 8
	#APP
	.insn r 11, 0, 1, zero, a1, a2

	#NO_APP
	slli	a1, a3, 32
	srli	a2, a1, 22
	add	a2, a2, t0
	srli	a1, a1, 20
	add	a1, a1, s3
	#APP
	.insn r 11, 0, 2, s4, a2, a1

	#NO_APP
	addi	a0, a0, 64
	mv	a3, t1
	bgeu	s0, s8, .LBB0_33
.LBB0_42:                               #   Parent Loop BB0_34 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_47 Depth 3
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	bltu	s7, a4, .LBB0_32
# %bb.43:                               #   in Loop: Header=BB0_42 Depth=2
	mv	s0, a4
	li	t2, 16
	bltu	a4, s2, .LBB0_45
# %bb.44:                               #   in Loop: Header=BB0_42 Depth=2
	subw	t2, s8, s0
.LBB0_45:                               #   in Loop: Header=BB0_42 Depth=2
	subw	t1, s11, a3
	slli	s1, t1, 32
.LBB0_65:                               #   in Loop: Header=BB0_42 Depth=2
                                        # Label of block must be emitted
	auipc	t0, %pcrel_hi(run_model.sram_a.1)
	addi	t0, t0, %pcrel_lo(.LBB0_65)
	blez	s6, .LBB0_41
# %bb.46:                               #   in Loop: Header=BB0_42 Depth=2
	srli	a1, s1, 22
	add	a4, t0, a1
	slli	a1, t2, 32
	srli	a1, a1, 30
	mv	a5, s5
	mv	a2, a0
.LBB0_47:                               #   Parent Loop BB0_34 Depth=1
                                        #     Parent Loop BB0_42 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	#APP
	.insn r 11, 0, 1, zero, a4, a2

	#NO_APP
	addi	a2, a2, 512
	addi	a5, a5, -1
	addi	a4, a4, 64
	bnez	a5, .LBB0_47
	j	.LBB0_41
.LBB0_48:
	li	a0, 1024
	lui	a1, 24
.LBB0_49:                               # =>This Inner Loop Header: Depth=1
	add	a2, s1, a1
	flw	ft0, 0(a2)
	flw	ft1, 0(s1)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(s1)
	addi	a0, a0, -1
	addi	s1, s1, 4
	bnez	a0, .LBB0_49
# %bb.50:
	ld	ra, 216(sp)                     # 8-byte Folded Reload
	ld	s0, 208(sp)                     # 8-byte Folded Reload
	ld	s1, 200(sp)                     # 8-byte Folded Reload
	ld	s2, 192(sp)                     # 8-byte Folded Reload
	ld	s3, 184(sp)                     # 8-byte Folded Reload
	ld	s4, 176(sp)                     # 8-byte Folded Reload
	ld	s5, 168(sp)                     # 8-byte Folded Reload
	ld	s6, 160(sp)                     # 8-byte Folded Reload
	ld	s7, 152(sp)                     # 8-byte Folded Reload
	ld	s8, 144(sp)                     # 8-byte Folded Reload
	ld	s9, 136(sp)                     # 8-byte Folded Reload
	ld	s10, 128(sp)                    # 8-byte Folded Reload
	ld	s11, 120(sp)                    # 8-byte Folded Reload
	fld	fs0, 112(sp)                    # 8-byte Folded Reload
	fld	fs1, 104(sp)                    # 8-byte Folded Reload
	fld	fs2, 96(sp)                     # 8-byte Folded Reload
	fld	fs3, 88(sp)                     # 8-byte Folded Reload
	addi	sp, sp, 224
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
