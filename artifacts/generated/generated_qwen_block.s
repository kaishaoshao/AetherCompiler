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
	mv	s1, a1
	lui	a1, 26
	add	a5, a0, a1
	lui	a1, 25
	add	s0, a0, a1
	lui	a1, 24
	add	s6, a0, a1
	lui	a1, 40
	add	s4, a0, a1
	lui	a1, 43
	add	a1, a1, a0
	#APP
	.insn r 11, 0, 4, a1, s4, a1

	#NO_APP
	#APP
	.insn r 11, 0, 2, s6, a1, s6

	#NO_APP
	lui	a2, 28
	add	a2, a2, a0
	#APP
	.insn r 11, 0, 2, s0, a1, a2

	#NO_APP
	lui	a2, 32
	add	a2, a2, a0
	#APP
	.insn r 11, 0, 2, a5, a1, a2

	#NO_APP
	lui	a1, 44
	add	a1, a1, a0
	li	a2, 1024
	mv	a3, s6
	mv	a4, a1
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
	addi	a2, a1, 256
	li	a3, 1024
	mv	a4, s0
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
	addi	a1, a1, 512
	li	a2, 1024
.LBB0_5:                                # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a5)
	flw	ft1, 0(a1)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a5)
	addi	a1, a1, 4
	addi	a2, a2, -1
	addi	a5, a5, 4
	bnez	a2, .LBB0_5
# %bb.6:
	lui	a1, 41
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	add	a0, a0, a1
	#APP
	.insn r 11, 0, 6, s6, s6, a0

	#NO_APP
	#APP
	.insn r 11, 0, 6, s0, s0, a0

	#NO_APP
	sd	s0, 8(sp)                       # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 7, s6, s6, s0

	#NO_APP
	sd	s1, 16(sp)                      # 8-byte Folded Spill
	slliw	a3, s1, 2
	li	a1, 12
	mv	a0, a3
	blt	a3, a1, .LBB0_8
# %bb.7:
	li	a0, 12
.LBB0_8:
	addiw	a2, a0, 4
	sd	a2, 80(sp)                      # 8-byte Folded Spill
	sd	a3, 24(sp)                      # 8-byte Folded Spill
	bge	a3, a2, .LBB0_26
# %bb.9:
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
.LBB0_55:                               # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_c)
	addi	s7, s7, %pcrel_lo(.LBB0_55)
.LBB0_56:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(run_model.sram_a)
	addi	a0, a0, %pcrel_lo(.LBB0_56)
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	li	s2, 64
.LBB0_57:                               # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_b)
	addi	s9, s9, %pcrel_lo(.LBB0_57)
	li	s11, 47
	li	s3, 48
	ld	s0, 24(sp)                      # 8-byte Folded Reload
	j	.LBB0_12
.LBB0_10:                               #   in Loop: Header=BB0_12 Depth=1
	slli	a0, a3, 32
	srli	a1, a0, 22
.LBB0_58:                               #   in Loop: Header=BB0_12 Depth=1
                                        # Label of block must be emitted
	auipc	a2, %pcrel_hi(run_model.sram_a)
	addi	a2, a2, %pcrel_lo(.LBB0_58)
	add	a1, a1, a2
	srli	a0, a0, 20
	add	a0, a0, s9
	#APP
	.insn r 11, 0, 2, s7, a1, a0

	#NO_APP
.LBB0_11:                               #   in Loop: Header=BB0_12 Depth=1
	#APP
	.insn r 11, 0, 1, zero, a6, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a3, 72(sp)                      # 8-byte Folded Reload
	addiw	a3, a3, 1024
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	mv	s0, a0
	ld	a2, 80(sp)                      # 8-byte Folded Reload
	bge	a0, a2, .LBB0_26
.LBB0_12:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_17 Depth 2
                                        #     Child Loop BB0_20 Depth 2
                                        #       Child Loop BB0_25 Depth 3
	addi	a0, s0, 16
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	blt	a2, a0, .LBB0_14
# %bb.13:                               #   in Loop: Header=BB0_12 Depth=1
	li	s5, 16
	j	.LBB0_15
.LBB0_14:                               #   in Loop: Header=BB0_12 Depth=1
	subw	s5, a2, s0
.LBB0_15:                               #   in Loop: Header=BB0_12 Depth=1
	sd	a3, 72(sp)                      # 8-byte Folded Spill
	slli	s1, a3, 2
	slli	a0, s5, 32
	srli	s10, a0, 32
	lui	a2, 1
	mv	a0, s7
	li	a1, 0
	call	memset@plt
	blez	s5, .LBB0_18
# %bb.16:                               #   in Loop: Header=BB0_12 Depth=1
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	add	a0, a0, s1
	mv	a1, s10
	ld	a2, 32(sp)                      # 8-byte Folded Reload
.LBB0_17:                               #   Parent Loop BB0_12 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a2, a0

	#NO_APP
	addi	a0, a0, 256
	addi	a1, a1, -1
	addi	a2, a2, 64
	bnez	a1, .LBB0_17
.LBB0_18:                               #   in Loop: Header=BB0_12 Depth=1
	li	a4, 0
	li	a3, 0
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	add	s1, s1, a0
	slliw	a0, s0, 6
	lui	a1, 1
	#APP
	.insn r 11, 0, 1, zero, s9, s8

	#NO_APP
	slli	a0, a0, 2
	add	a6, s6, a0
	slli	a7, s5, 8
	li	t3, 1
	li	t4, 33
	j	.LBB0_20
.LBB0_19:                               #   in Loop: Header=BB0_20 Depth=2
	addi	a4, s0, 16
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
	addi	s1, s1, 64
	mv	a3, t1
	bgeu	s0, s3, .LBB0_11
.LBB0_20:                               #   Parent Loop BB0_12 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_25 Depth 3
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	bltu	s11, a4, .LBB0_10
# %bb.21:                               #   in Loop: Header=BB0_20 Depth=2
	mv	s0, a4
	li	t2, 16
	bltu	a4, t4, .LBB0_23
# %bb.22:                               #   in Loop: Header=BB0_20 Depth=2
	subw	t2, s3, s0
.LBB0_23:                               #   in Loop: Header=BB0_20 Depth=2
	subw	t1, t3, a3
	slli	a1, t1, 32
.LBB0_59:                               #   in Loop: Header=BB0_20 Depth=2
                                        # Label of block must be emitted
	auipc	t0, %pcrel_hi(run_model.sram_a)
	addi	t0, t0, %pcrel_lo(.LBB0_59)
	blez	s5, .LBB0_19
# %bb.24:                               #   in Loop: Header=BB0_20 Depth=2
	srli	a0, a1, 22
	add	a4, t0, a0
	slli	a0, t2, 32
	srli	a0, a0, 30
	mv	a5, s10
	mv	a2, s1
.LBB0_25:                               #   Parent Loop BB0_12 Depth=1
                                        #     Parent Loop BB0_20 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	#APP
	.insn r 11, 0, 1, zero, a4, a2

	#NO_APP
	addi	a2, a2, 256
	addi	a5, a5, -1
	addi	a4, a4, 64
	bnez	a5, .LBB0_25
	j	.LBB0_19
.LBB0_26:
	li	s2, 0
	lui	a0, 44
	addiw	a0, a0, 768
	ld	a1, 56(sp)                      # 8-byte Folded Reload
	add	s5, a1, a0
.LBB0_60:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_0)
	addi	a0, a0, %pcrel_lo(.LBB0_60)
	flw	fs0, 0(a0)
.LBB0_61:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_1)
	addi	a0, a0, %pcrel_lo(.LBB0_61)
	flw	fs1, 0(a0)
.LBB0_62:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_2)
	addi	a0, a0, %pcrel_lo(.LBB0_62)
	flw	fs2, 0(a0)
	fmv.w.x	fs3, zero
	li	s0, 256
	li	s3, 16
	ld	s7, 8(sp)                       # 8-byte Folded Reload
	mv	s1, s7
.LBB0_27:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_28 Depth 2
                                        #     Child Loop BB0_31 Depth 2
	li	a0, 0
	fmv.s	ft0, fs3
.LBB0_28:                               #   Parent Loop BB0_27 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s4, a0
	flw	ft1, 0(a1)
	add	a1, s6, a0
	flw	ft2, 0(a1)
	fadd.s	ft1, ft1, ft2
	fsw	ft1, 0(a1)
	addi	a0, a0, 4
	fmadd.s	ft0, ft1, ft1, ft0
	bne	a0, s0, .LBB0_28
# %bb.29:                               #   in Loop: Header=BB0_27 Depth=1
	fmul.s	ft0, ft0, fs0
	fadd.s	ft0, ft0, fs1
	fsqrt.s	fa0, ft0
	feq.s	a0, fa0, fa0
	beqz	a0, .LBB0_33
.LBB0_30:                               #   in Loop: Header=BB0_27 Depth=1
	li	a0, 0
	fdiv.s	ft0, fs2, fa0
.LBB0_31:                               #   Parent Loop BB0_27 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s6, a0
	flw	ft1, 0(a1)
	add	a1, s5, a0
	flw	ft2, 0(a1)
	fmul.s	ft1, ft0, ft1
	fmul.s	ft1, ft1, ft2
	add	a1, s1, a0
	addi	a0, a0, 4
	fsw	ft1, 0(a1)
	bne	a0, s0, .LBB0_31
# %bb.32:                               #   in Loop: Header=BB0_27 Depth=1
	addi	s2, s2, 1
	addi	s6, s6, 256
	addi	s4, s4, 256
	addi	s1, s1, 256
	bne	s2, s3, .LBB0_27
	j	.LBB0_34
.LBB0_33:                               #   in Loop: Header=BB0_27 Depth=1
	fmv.s	fa0, ft0
	call	sqrtf@plt
	ld	a2, 80(sp)                      # 8-byte Folded Reload
	j	.LBB0_30
.LBB0_34:
	ld	t4, 56(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 2, t4, s7, t4

	#NO_APP
	lui	a0, 8
	add	a0, a0, t4
	lui	a1, 2
	add	a1, a1, t4
	#APP
	.insn r 11, 0, 2, a1, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 5, t4, t4, a1

	#NO_APP
	ld	s0, 24(sp)                      # 8-byte Folded Reload
	bge	s0, a2, .LBB0_52
# %bb.35:
	lui	a0, 16
	add	s6, t4, a0
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	slliw	a1, a0, 9
	addi	a0, t4, 64
	sd	a0, 48(sp)                      # 8-byte Folded Spill
.LBB0_63:                               # Label of block must be emitted
	auipc	s3, %pcrel_hi(run_model.sram_c.3)
	addi	s3, s3, %pcrel_lo(.LBB0_63)
	li	s8, 64
	li	s9, 1
	li	s10, 111
	li	s11, 112
	li	s7, 97
	j	.LBB0_38
.LBB0_36:                               #   in Loop: Header=BB0_38 Depth=1
	slli	a0, a5, 32
	srli	a1, a0, 22
	add	a1, a1, t0
	srli	a0, a0, 20
	add	a0, a0, t1
	#APP
	.insn r 11, 0, 2, s3, a1, a0

	#NO_APP
.LBB0_37:                               #   in Loop: Header=BB0_38 Depth=1
	slli	a0, a7, 2
	add	a0, a0, t4
	#APP
	.insn r 11, 0, 1, zero, a0, s3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	addiw	a0, a0, 2047
	addiw	a1, a0, 1
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	mv	s0, a0
	ld	a2, 80(sp)                      # 8-byte Folded Reload
	bge	a0, a2, .LBB0_52
.LBB0_38:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_43 Depth 2
                                        #     Child Loop BB0_46 Depth 2
                                        #       Child Loop BB0_51 Depth 3
	addi	a0, s0, 16
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	blt	a2, a0, .LBB0_40
# %bb.39:                               #   in Loop: Header=BB0_38 Depth=1
	li	s4, 16
	j	.LBB0_41
.LBB0_40:                               #   in Loop: Header=BB0_38 Depth=1
	subw	s4, a2, s0
.LBB0_41:                               #   in Loop: Header=BB0_38 Depth=1
	sd	a1, 72(sp)                      # 8-byte Folded Spill
	slli	s2, a1, 2
	slli	a0, s4, 32
	srli	s5, a0, 32
	lui	a2, 1
	mv	a0, s3
	li	a1, 0
	call	memset@plt
.LBB0_64:                               #   in Loop: Header=BB0_38 Depth=1
                                        # Label of block must be emitted
	auipc	t0, %pcrel_hi(run_model.sram_a.1)
	addi	t0, t0, %pcrel_lo(.LBB0_64)
	blez	s4, .LBB0_44
# %bb.42:                               #   in Loop: Header=BB0_38 Depth=1
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	add	a0, a0, s2
	mv	a1, s5
	mv	a2, t0
.LBB0_43:                               #   Parent Loop BB0_38 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a2, a0

	#NO_APP
	addi	a0, a0, 512
	addi	a1, a1, -1
	addi	a2, a2, 64
	bnez	a1, .LBB0_43
.LBB0_44:                               #   in Loop: Header=BB0_38 Depth=1
	li	a1, 0
	li	a5, 0
.LBB0_65:                               #   in Loop: Header=BB0_38 Depth=1
                                        # Label of block must be emitted
	auipc	t1, %pcrel_hi(run_model.sram_b.2)
	addi	t1, t1, %pcrel_lo(.LBB0_65)
	lui	a0, 1
	#APP
	.insn r 11, 0, 1, zero, t1, s6

	#NO_APP
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	add	s1, a0, s2
	slliw	a7, s0, 6
	slli	a6, s4, 8
	ld	t4, 56(sp)                      # 8-byte Folded Reload
	j	.LBB0_46
.LBB0_45:                               #   in Loop: Header=BB0_46 Depth=2
	addi	a1, a0, 16
	srli	a2, s0, 32
	slli	a2, a2, 12
	add	a2, a2, t1
	slli	a3, a1, 8
	add	a3, a3, s6
	slli	a4, t3, 8
	#APP
	.insn r 11, 0, 1, zero, a2, a3

	#NO_APP
	slli	a2, a5, 32
	srli	a3, a2, 22
	add	a3, a3, t0
	srli	a2, a2, 20
	add	a2, a2, t1
	#APP
	.insn r 11, 0, 2, s3, a3, a2

	#NO_APP
	addi	s1, s1, 64
	mv	a5, t2
	bgeu	a0, s11, .LBB0_37
.LBB0_46:                               #   Parent Loop BB0_38 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_51 Depth 3
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	bltu	s10, a1, .LBB0_36
# %bb.47:                               #   in Loop: Header=BB0_46 Depth=2
	mv	a0, a1
	li	t3, 16
	bltu	a1, s7, .LBB0_49
# %bb.48:                               #   in Loop: Header=BB0_46 Depth=2
	subw	t3, s11, a0
.LBB0_49:                               #   in Loop: Header=BB0_46 Depth=2
	subw	t2, s9, a5
	slli	s0, t2, 32
	blez	s4, .LBB0_45
# %bb.50:                               #   in Loop: Header=BB0_46 Depth=2
	srli	a1, s0, 22
	add	a1, a1, t0
	slli	a2, t3, 32
	srli	a3, a2, 30
	mv	a2, s5
	mv	a4, s1
.LBB0_51:                               #   Parent Loop BB0_38 Depth=1
                                        #     Parent Loop BB0_46 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	#APP
	.insn r 11, 0, 1, zero, a1, a4

	#NO_APP
	addi	a4, a4, 512
	addi	a2, a2, -1
	addi	a1, a1, 64
	bnez	a2, .LBB0_51
	j	.LBB0_45
.LBB0_52:
	li	a0, 1024
	lui	a1, 24
.LBB0_53:                               # =>This Inner Loop Header: Depth=1
	add	a2, t4, a1
	flw	ft0, 0(a2)
	flw	ft1, 0(t4)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(t4)
	addi	a0, a0, -1
	addi	t4, t4, 4
	bnez	a0, .LBB0_53
# %bb.54:
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
