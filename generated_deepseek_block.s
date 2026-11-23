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
	addi	sp, sp, -272
	sd	ra, 264(sp)                     # 8-byte Folded Spill
	sd	s0, 256(sp)                     # 8-byte Folded Spill
	sd	s1, 248(sp)                     # 8-byte Folded Spill
	sd	s2, 240(sp)                     # 8-byte Folded Spill
	sd	s3, 232(sp)                     # 8-byte Folded Spill
	sd	s4, 224(sp)                     # 8-byte Folded Spill
	sd	s5, 216(sp)                     # 8-byte Folded Spill
	sd	s6, 208(sp)                     # 8-byte Folded Spill
	sd	s7, 200(sp)                     # 8-byte Folded Spill
	sd	s8, 192(sp)                     # 8-byte Folded Spill
	sd	s9, 184(sp)                     # 8-byte Folded Spill
	sd	s10, 176(sp)                    # 8-byte Folded Spill
	sd	s11, 168(sp)                    # 8-byte Folded Spill
	fsd	fs0, 160(sp)                    # 8-byte Folded Spill
	fsd	fs1, 152(sp)                    # 8-byte Folded Spill
	fsd	fs2, 144(sp)                    # 8-byte Folded Spill
	fsd	fs3, 136(sp)                    # 8-byte Folded Spill
	lui	a2, 76
	add	s0, a0, a2
	sd	a1, 0(sp)                       # 8-byte Folded Spill
	li	a6, 0
	mv	s3, s0
	lui	a1, 86
	add	s2, a0, a1
	lui	a1, 88
	add	a1, a1, a0
	#APP
	.insn r 11, 0, 4, a1, s2, a1

	#NO_APP
	#APP
	.insn r 11, 0, 2, s0, a1, s0

	#NO_APP
	lui	a2, 87
	add	a2, a2, a0
	lui	a3, 83
	addiw	a3, a3, -2048
	add	a3, a3, a0
	#APP
	.insn r 11, 0, 2, a3, a1, a2

	#NO_APP
	lui	a2, 82
	add	a2, a2, a0
	#APP
	.insn r 11, 0, 2, a2, a1, a2

	#NO_APP
	lui	a1, 90
	addiw	a4, a1, 256
	add	a4, a4, a0
	#APP
	.insn r 11, 0, 4, s0, s0, a4

	#NO_APP
	lui	a4, 78
	add	a4, a4, a0
	#APP
	.insn r 11, 0, 2, s0, s0, a4

	#NO_APP
	lui	a5, 80
	add	a5, a5, a0
	lui	s1, 77
	add	s1, s1, a0
	sd	s0, 120(sp)                     # 8-byte Folded Spill
	sd	s1, 24(sp)                      # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 2, s1, s0, a5

	#NO_APP
	addiw	a5, a1, -2048
	add	a5, a5, a0
	lui	s1, 79
	add	s0, a0, s1
	sd	s0, 112(sp)                     # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 6, s0, a3, a5

	#NO_APP
	lui	a3, 3
	addiw	a1, a1, 384
	add	a1, a1, a0
	#APP
	.insn r 11, 0, 4, a4, a2, a1

	#NO_APP
	lui	a1, 84
	add	a1, a1, a0
	#APP
	.insn r 11, 0, 2, a4, a4, a1

	#NO_APP
	lui	a1, 89
	add	a1, a1, a0
	addiw	a2, s1, 1024
	add	a2, a2, a0
	#APP
	.insn r 11, 0, 2, a2, a4, a1

	#NO_APP
	sd	a2, 104(sp)                     # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 6, a2, a2, a5

	#NO_APP
	add	a1, a0, a3
	sd	a1, 16(sp)                      # 8-byte Folded Spill
	li	a1, 256
	li	a2, 16
	mv	a3, s3
	mv	s1, s3
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	li	a5, 0
	mv	a4, s1
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	s0, a3, a5
	flw	ft0, 0(s0)
	fsw	ft0, 0(a4)
	addi	a5, a5, 4
	addi	a4, a4, 64
	bne	a5, a1, .LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	addi	a6, a6, 1
	addi	s1, s1, 4
	addi	a3, a3, 256
	bne	a6, a2, .LBB0_1
# %bb.4:
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	ld	a0, 0(sp)                       # 8-byte Folded Reload
	slliw	a2, a0, 2
	li	a1, 12
	mv	a0, a2
	blt	a2, a1, .LBB0_6
# %bb.5:
	li	a0, 12
.LBB0_6:
	addiw	a5, a0, 4
	sd	a5, 128(sp)                     # 8-byte Folded Spill
	sd	a2, 8(sp)                       # 8-byte Folded Spill
	bge	a2, a5, .LBB0_18
# %bb.7:
	lui	a0, 77
	addiw	a0, a0, -2048
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	lui	a0, 78
	add	a2, a1, a0
	sd	a2, 88(sp)                      # 8-byte Folded Spill
	ld	a2, 0(sp)                       # 8-byte Folded Reload
	slliw	s0, a2, 8
	addiw	a0, a0, 128
	add	a0, a0, a1
	sd	a0, 80(sp)                      # 8-byte Folded Spill
.LBB0_160:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_c)
	addi	s9, s9, %pcrel_lo(.LBB0_160)
.LBB0_161:                              # Label of block must be emitted
	auipc	a0, %pcrel_hi(run_model.sram_a)
	addi	a0, a0, %pcrel_lo(.LBB0_161)
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	li	s8, 128
.LBB0_162:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_b)
	addi	s10, s10, %pcrel_lo(.LBB0_162)
	lui	a0, 1
	addiw	s11, a0, -2048
	ld	s1, 8(sp)                       # 8-byte Folded Reload
	j	.LBB0_9
.LBB0_8:                                #   in Loop: Header=BB0_9 Depth=1
	slliw	a0, s1, 4
	slli	a1, s6, 6
	add	a2, s10, s11
	ld	a3, 96(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, a2, a3

	#NO_APP
.LBB0_163:                              #   in Loop: Header=BB0_9 Depth=1
                                        # Label of block must be emitted
	auipc	a3, %pcrel_hi(run_model.sram_a)
	addi	a3, a3, %pcrel_lo(.LBB0_163)
	#APP
	.insn r 11, 0, 2, s9, a3, s10

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	add	a3, a3, s11
	#APP
	.insn r 11, 0, 2, s9, a3, a2

	#NO_APP
	slli	a0, a0, 2
	ld	a2, 120(sp)                     # 8-byte Folded Reload
	add	a0, a0, a2
	#APP
	.insn r 11, 0, 1, zero, a0, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addiw	s0, s0, 1024
	mv	s1, s5
	bge	s5, a5, .LBB0_18
.LBB0_9:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_14 Depth 2
                                        #     Child Loop BB0_17 Depth 2
	addi	s5, s1, 16
	blt	a5, s5, .LBB0_11
# %bb.10:                               #   in Loop: Header=BB0_9 Depth=1
	li	s6, 16
	j	.LBB0_12
.LBB0_11:                               #   in Loop: Header=BB0_9 Depth=1
	subw	s6, a5, s1
.LBB0_12:                               #   in Loop: Header=BB0_9 Depth=1
	slli	s7, s0, 2
	slli	a0, s6, 32
	srli	s4, a0, 32
	li	a2, 1024
	mv	a0, s9
	li	a1, 0
	call	memset@plt
	blez	s6, .LBB0_15
# %bb.13:                               #   in Loop: Header=BB0_9 Depth=1
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	add	a0, a0, s7
	mv	a1, s4
	ld	a2, 72(sp)                      # 8-byte Folded Reload
.LBB0_14:                               #   Parent Loop BB0_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a2, a0

	#NO_APP
	addi	a0, a0, 256
	addi	a1, a1, -1
	addi	a2, a2, 128
	bnez	a1, .LBB0_14
.LBB0_15:                               #   in Loop: Header=BB0_9 Depth=1
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a5, 128(sp)                     # 8-byte Folded Reload
	blez	s6, .LBB0_8
# %bb.16:                               #   in Loop: Header=BB0_9 Depth=1
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	add	a0, a0, s7
.LBB0_164:                              #   in Loop: Header=BB0_9 Depth=1
                                        # Label of block must be emitted
	auipc	a1, %pcrel_hi(run_model.sram_a)
	addi	a1, a1, %pcrel_lo(.LBB0_164)
	addi	a1, a1, 2047
	addi	a1, a1, 1
.LBB0_17:                               #   Parent Loop BB0_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a1, a0

	#NO_APP
	addi	a0, a0, 256
	addi	s4, s4, -1
	addi	a1, a1, 128
	bnez	s4, .LBB0_17
	j	.LBB0_8
.LBB0_18:
	lui	a2, 79
	addiw	a0, a2, 512
	ld	s1, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, s1
	li	a1, -64
	addiw	a2, a2, 60
.LBB0_19:                               # =>This Inner Loop Header: Depth=1
	add	a3, s1, a2
	flw	ft0, -60(a3)
	add	a4, a0, a1
	fsw	ft0, -448(a4)
	flw	ft0, -56(a3)
	fsw	ft0, -384(a4)
	flw	ft0, -52(a3)
	fsw	ft0, -320(a4)
	flw	ft0, -48(a3)
	fsw	ft0, -256(a4)
	flw	ft0, -44(a3)
	fsw	ft0, -192(a4)
	flw	ft0, -40(a3)
	fsw	ft0, -128(a4)
	flw	ft0, -36(a3)
	fsw	ft0, -64(a4)
	flw	ft0, -32(a3)
	fsw	ft0, 0(a4)
	flw	ft0, -28(a3)
	fsw	ft0, 64(a4)
	flw	ft0, -24(a3)
	fsw	ft0, 128(a4)
	flw	ft0, -20(a3)
	fsw	ft0, 192(a4)
	flw	ft0, -16(a3)
	fsw	ft0, 256(a4)
	flw	ft0, -12(a3)
	fsw	ft0, 320(a4)
	flw	ft0, -8(a3)
	fsw	ft0, 384(a4)
	flw	ft0, -4(a3)
	fsw	ft0, 448(a4)
	flw	ft0, 0(a3)
	fsw	ft0, 512(a4)
	addi	a1, a1, 4
	addi	a2, a2, 64
	bnez	a1, .LBB0_19
# %bb.20:
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	bge	a0, a5, .LBB0_26
# %bb.21:
	lui	a0, 76
	addiw	a0, a0, 1024
	add	s8, s1, a0
.LBB0_165:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_c.3)
	addi	s9, s9, %pcrel_lo(.LBB0_165)
.LBB0_166:                              # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_a.1)
	addi	s4, s4, %pcrel_lo(.LBB0_166)
.LBB0_167:                              # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_b.2)
	addi	s5, s5, %pcrel_lo(.LBB0_167)
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	j	.LBB0_24
.LBB0_22:                               #   in Loop: Header=BB0_24 Depth=1
	li	s6, 1024
.LBB0_23:                               #   in Loop: Header=BB0_24 Depth=1
	mv	s11, a5
	slliw	a0, a0, 4
	slli	s1, a0, 2
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	add	s10, a0, s1
	li	a2, 1024
	mv	a0, s9
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s4, s10

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s5, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s9, s4, s5

	#NO_APP
	add	a0, s8, s1
	#APP
	.insn r 11, 0, 1, zero, a0, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	mv	a0, s0
	mv	a5, s11
	bge	s0, s11, .LBB0_26
.LBB0_24:                               # =>This Inner Loop Header: Depth=1
	addi	s0, a0, 16
	li	s7, 1024
	bge	a5, s0, .LBB0_22
# %bb.25:                               #   in Loop: Header=BB0_24 Depth=1
	subw	a1, a5, a0
	slli	s6, a1, 6
	j	.LBB0_23
.LBB0_26:
	li	a0, 256
	mv	a1, s3
.LBB0_27:                               # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a1)
	flw	ft1, 1024(a1)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a1)
	addi	a0, a0, -1
	addi	a1, a1, 4
	bnez	a0, .LBB0_27
# %bb.28:
	fmv.w.x	fs0, zero
	li	s11, 64
.LBB0_29:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_60 Depth 2
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	slli	a0, a0, 6
	ld	a1, 120(sp)                     # 8-byte Folded Reload
	add	s0, a1, a0
	flw	ft1, 0(s0)
	flw	ft0, 4(s0)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_31
# %bb.30:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft0, ft1
.LBB0_31:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft1, 8(s0)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_33
# %bb.32:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft1, ft0
.LBB0_33:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft0, 12(s0)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_35
# %bb.34:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft0, ft1
.LBB0_35:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft1, 16(s0)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_37
# %bb.36:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft1, ft0
.LBB0_37:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft0, 20(s0)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_39
# %bb.38:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft0, ft1
.LBB0_39:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft1, 24(s0)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_41
# %bb.40:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft1, ft0
.LBB0_41:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft0, 28(s0)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_43
# %bb.42:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft0, ft1
.LBB0_43:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft1, 32(s0)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_45
# %bb.44:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft1, ft0
.LBB0_45:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft0, 36(s0)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_47
# %bb.46:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft0, ft1
.LBB0_47:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft1, 40(s0)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_49
# %bb.48:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft1, ft0
.LBB0_49:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft0, 44(s0)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_51
# %bb.50:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft0, ft1
.LBB0_51:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft1, 48(s0)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_53
# %bb.52:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft1, ft0
.LBB0_53:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft0, 52(s0)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_55
# %bb.54:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft0, ft1
.LBB0_55:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft1, 56(s0)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_57
# %bb.56:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	ft1, ft0
.LBB0_57:                               #   in Loop: Header=BB0_29 Depth=1
	flw	fs2, 60(s0)
	flt.s	a0, ft1, fs2
	bnez	a0, .LBB0_59
# %bb.58:                               #   in Loop: Header=BB0_29 Depth=1
	fmv.s	fs2, ft1
.LBB0_59:                               #   in Loop: Header=BB0_29 Depth=1
	li	s9, 0
	addi	s7, s0, 4
	addi	s10, s0, 8
	addi	s8, s0, 12
	addi	s6, s0, 16
	addi	s5, s0, 20
	addi	s4, s0, 24
	addi	a0, s0, 28
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	addi	a0, s0, 32
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	addi	a0, s0, 36
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	addi	a0, s0, 40
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	addi	a0, s0, 44
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	addi	a0, s0, 48
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addi	a0, s0, 52
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	addi	a0, s0, 56
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	addi	a0, s0, 60
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	fmv.s	fs1, fs0
.LBB0_60:                               #   Parent Loop BB0_29 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	s1, s3, s9
	flw	ft0, 0(s1)
	fsub.s	ft0, ft0, fs2
	fcvt.d.s	fa0, ft0
	call	exp@plt
	fcvt.s.d	ft0, fa0
	fsw	ft0, 0(s1)
	addi	s9, s9, 4
	fadd.s	fs1, fs1, ft0
	bne	s9, s11, .LBB0_60
# %bb.61:                               #   in Loop: Header=BB0_29 Depth=1
	flw	ft0, 0(s0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s0)
	flw	ft0, 0(s7)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s7)
	flw	ft0, 0(s10)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s10)
	flw	ft0, 0(s8)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s8)
	flw	ft0, 0(s6)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s6)
	flw	ft0, 0(s5)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s5)
	flw	ft0, 0(s4)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s4)
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1
	addi	s3, s3, 64
	ld	a2, 128(sp)                     # 8-byte Folded Reload
	li	a1, 16
	bne	a0, a1, .LBB0_29
# %bb.62:
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	bge	a0, a2, .LBB0_86
# %bb.63:
.LBB0_168:                              # Label of block must be emitted
	auipc	s6, %pcrel_hi(run_model.sram_c.6)
	addi	s6, s6, %pcrel_lo(.LBB0_168)
	lui	s3, 1
.LBB0_169:                              # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_a.4)
	addi	s4, s4, %pcrel_lo(.LBB0_169)
.LBB0_170:                              # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_b.5)
	addi	s5, s5, %pcrel_lo(.LBB0_170)
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	ld	s11, 120(sp)                    # 8-byte Folded Reload
	j	.LBB0_66
.LBB0_64:                               #   in Loop: Header=BB0_66 Depth=1
	li	a1, 16
.LBB0_65:                               #   in Loop: Header=BB0_66 Depth=1
	mv	s10, a2
	slliw	a2, a0, 4
	slli	a2, a2, 2
	add	s7, s11, a2
	slli	s1, a1, 6
	slliw	s8, a0, 6
	slli	s9, a1, 8
	lui	a2, 1
	mv	a0, s6
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s4, s7

	#NO_APP
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s5, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s6, s4, s5

	#NO_APP
	slli	a0, s8, 2
	add	a0, a0, s11
	#APP
	.insn r 11, 0, 1, zero, a0, s6

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	mv	a0, s0
	mv	a2, s10
	bge	s0, s10, .LBB0_68
.LBB0_66:                               # =>This Inner Loop Header: Depth=1
	addi	s0, a0, 16
	bge	a2, s0, .LBB0_64
# %bb.67:                               #   in Loop: Header=BB0_66 Depth=1
	subw	a1, a2, a0
	j	.LBB0_65
.LBB0_68:
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	bge	a0, a2, .LBB0_86
# %bb.69:
	lui	a0, 72
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	s7, a1, a0
	lui	a0, 76
	add	a3, a1, a0
	sd	a3, 96(sp)                      # 8-byte Folded Spill
	ld	a3, 0(sp)                       # 8-byte Folded Reload
	slliw	a3, a3, 8
	addiw	a0, a0, 64
	add	a0, a0, a1
	sd	a0, 104(sp)                     # 8-byte Folded Spill
.LBB0_171:                              # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_c.9)
	addi	s4, s4, %pcrel_lo(.LBB0_171)
.LBB0_172:                              # Label of block must be emitted
	auipc	a0, %pcrel_hi(run_model.sram_a.7)
	addi	a0, a0, %pcrel_lo(.LBB0_172)
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	li	s11, 64
.LBB0_173:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_b.8)
	addi	s9, s9, %pcrel_lo(.LBB0_173)
	li	s10, 47
	li	s5, 48
	ld	s8, 8(sp)                       # 8-byte Folded Reload
	j	.LBB0_72
.LBB0_70:                               #   in Loop: Header=BB0_72 Depth=1
	slli	a0, a3, 32
	srli	a1, a0, 22
.LBB0_174:                              #   in Loop: Header=BB0_72 Depth=1
                                        # Label of block must be emitted
	auipc	a2, %pcrel_hi(run_model.sram_a.7)
	addi	a2, a2, %pcrel_lo(.LBB0_174)
	add	a1, a1, a2
	srli	a0, a0, 20
	add	a0, a0, s9
	#APP
	.insn r 11, 0, 2, s4, a1, a0

	#NO_APP
.LBB0_71:                               #   in Loop: Header=BB0_72 Depth=1
	slli	a0, a7, 2
	add	a0, a0, s7
	#APP
	.insn r 11, 0, 1, zero, a0, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a3, 120(sp)                     # 8-byte Folded Reload
	addiw	a3, a3, 1024
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	mv	s8, a0
	ld	a2, 128(sp)                     # 8-byte Folded Reload
	bge	a0, a2, .LBB0_86
.LBB0_72:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_77 Depth 2
                                        #     Child Loop BB0_80 Depth 2
                                        #       Child Loop BB0_85 Depth 3
	addi	a0, s8, 16
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	blt	a2, a0, .LBB0_74
# %bb.73:                               #   in Loop: Header=BB0_72 Depth=1
	li	s3, 16
	j	.LBB0_75
.LBB0_74:                               #   in Loop: Header=BB0_72 Depth=1
	subw	s3, a2, s8
.LBB0_75:                               #   in Loop: Header=BB0_72 Depth=1
	sd	a3, 120(sp)                     # 8-byte Folded Spill
	slli	s0, a3, 2
	slli	a0, s3, 32
	srli	s6, a0, 32
	lui	a2, 1
	mv	a0, s4
	li	a1, 0
	call	memset@plt
	blez	s3, .LBB0_78
# %bb.76:                               #   in Loop: Header=BB0_72 Depth=1
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	add	a0, a0, s0
	mv	a1, s6
	ld	a2, 88(sp)                      # 8-byte Folded Reload
.LBB0_77:                               #   Parent Loop BB0_72 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a2, a0

	#NO_APP
	addi	a0, a0, 256
	addi	a1, a1, -1
	addi	a2, a2, 64
	bnez	a1, .LBB0_77
.LBB0_78:                               #   in Loop: Header=BB0_72 Depth=1
	li	a4, 0
	li	a3, 0
	lui	a0, 1
	#APP
	.insn r 11, 0, 1, zero, s9, s7

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	add	a0, a0, s0
	slliw	a7, s8, 6
	slli	a6, s3, 8
	li	t3, 1
	li	t4, 33
	j	.LBB0_80
.LBB0_79:                               #   in Loop: Header=BB0_80 Depth=2
	addi	a4, s1, 16
	srli	a1, s0, 32
	slli	a1, a1, 12
	add	a1, a1, s9
	slli	a2, a4, 8
	add	a2, a2, s7
	slli	a5, t2, 8
	#APP
	.insn r 11, 0, 1, zero, a1, a2

	#NO_APP
	slli	a1, a3, 32
	srli	a2, a1, 22
	add	a2, a2, t0
	srli	a1, a1, 20
	add	a1, a1, s9
	#APP
	.insn r 11, 0, 2, s4, a2, a1

	#NO_APP
	addi	a0, a0, 64
	mv	a3, t1
	bgeu	s1, s5, .LBB0_71
.LBB0_80:                               #   Parent Loop BB0_72 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_85 Depth 3
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	bltu	s10, a4, .LBB0_70
# %bb.81:                               #   in Loop: Header=BB0_80 Depth=2
	mv	s1, a4
	li	t2, 16
	bltu	a4, t4, .LBB0_83
# %bb.82:                               #   in Loop: Header=BB0_80 Depth=2
	subw	t2, s5, s1
.LBB0_83:                               #   in Loop: Header=BB0_80 Depth=2
	subw	t1, t3, a3
	slli	s0, t1, 32
.LBB0_175:                              #   in Loop: Header=BB0_80 Depth=2
                                        # Label of block must be emitted
	auipc	t0, %pcrel_hi(run_model.sram_a.7)
	addi	t0, t0, %pcrel_lo(.LBB0_175)
	blez	s3, .LBB0_79
# %bb.84:                               #   in Loop: Header=BB0_80 Depth=2
	srli	a1, s0, 22
	add	a4, t0, a1
	slli	a1, t2, 32
	srli	a1, a1, 30
	mv	a5, s6
	mv	a2, a0
.LBB0_85:                               #   Parent Loop BB0_72 Depth=1
                                        #     Parent Loop BB0_80 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	#APP
	.insn r 11, 0, 1, zero, a4, a2

	#NO_APP
	addi	a2, a2, 256
	addi	a5, a5, -1
	addi	a4, a4, 64
	bnez	a5, .LBB0_85
	j	.LBB0_79
.LBB0_86:
	li	s4, 0
	lui	a0, 90
	ld	s8, 32(sp)                      # 8-byte Folded Reload
	add	s5, s8, a0
	lui	a0, 73
	add	s3, s8, a0
	lui	a0, 72
	add	s1, s8, a0
.LBB0_176:                              # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_0)
	addi	a0, a0, %pcrel_lo(.LBB0_176)
	flw	fs0, 0(a0)
.LBB0_177:                              # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_1)
	addi	a0, a0, %pcrel_lo(.LBB0_177)
	flw	fs1, 0(a0)
.LBB0_178:                              # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_2)
	addi	a0, a0, %pcrel_lo(.LBB0_178)
	flw	fs2, 0(a0)
	fmv.w.x	fs3, zero
	li	s7, 256
	li	s6, 16
	mv	s0, s3
	ld	s10, 8(sp)                      # 8-byte Folded Reload
.LBB0_87:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_88 Depth 2
                                        #     Child Loop BB0_91 Depth 2
	li	a0, 0
	fmv.s	ft0, fs3
.LBB0_88:                               #   Parent Loop BB0_87 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s2, a0
	flw	ft1, 0(a1)
	add	a1, s1, a0
	flw	ft2, 0(a1)
	fadd.s	ft1, ft1, ft2
	fsw	ft1, 0(a1)
	addi	a0, a0, 4
	fmadd.s	ft0, ft1, ft1, ft0
	bne	a0, s7, .LBB0_88
# %bb.89:                               #   in Loop: Header=BB0_87 Depth=1
	fmul.s	ft0, ft0, fs0
	fadd.s	ft0, ft0, fs1
	fsqrt.s	fa0, ft0
	feq.s	a0, fa0, fa0
	beqz	a0, .LBB0_93
.LBB0_90:                               #   in Loop: Header=BB0_87 Depth=1
	li	a0, 0
	fdiv.s	ft0, fs2, fa0
.LBB0_91:                               #   Parent Loop BB0_87 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s1, a0
	flw	ft1, 0(a1)
	add	a1, s5, a0
	flw	ft2, 0(a1)
	fmul.s	ft1, ft0, ft1
	fmul.s	ft1, ft1, ft2
	add	a1, s0, a0
	addi	a0, a0, 4
	fsw	ft1, 0(a1)
	bne	a0, s7, .LBB0_91
# %bb.92:                               #   in Loop: Header=BB0_87 Depth=1
	addi	s4, s4, 1
	addi	s1, s1, 256
	addi	s2, s2, 256
	addi	s0, s0, 256
	bne	s4, s6, .LBB0_87
	j	.LBB0_94
.LBB0_93:                               #   in Loop: Header=BB0_87 Depth=1
	fmv.s	fa0, ft0
	call	sqrtf@plt
	ld	a2, 128(sp)                     # 8-byte Folded Reload
	j	.LBB0_90
.LBB0_94:
	#APP
	.insn r 11, 0, 2, s8, s3, s8

	#NO_APP
	lui	a0, 8
	add	a3, s8, a0
	lui	a0, 2
	add	a5, s8, a0
	#APP
	.insn r 11, 0, 2, a5, s3, a3

	#NO_APP
	lui	a0, 24
	add	a0, a0, s8
	lui	a1, 4
	add	a4, s8, a1
	#APP
	.insn r 11, 0, 2, a4, s3, a0

	#NO_APP
	lui	a0, 32
	add	a0, a0, s8
	lui	a1, 6
	add	s1, s8, a1
	#APP
	.insn r 11, 0, 2, s1, s3, a0

	#NO_APP
	lui	a0, 48
	add	a0, a0, s8
	#APP
	.insn r 11, 0, 2, a3, s3, a0

	#NO_APP
	lui	a0, 56
	add	a0, a0, s8
	lui	a1, 10
	add	a1, a1, s8
	#APP
	.insn r 11, 0, 2, a1, s3, a0

	#NO_APP
	sd	a5, 88(sp)                      # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 5, s8, s8, a5

	#NO_APP
	bge	s10, a2, .LBB0_112
# %bb.95:
	sd	s1, 56(sp)                      # 8-byte Folded Spill
	sd	a4, 64(sp)                      # 8-byte Folded Spill
	sd	a1, 72(sp)                      # 8-byte Folded Spill
	sd	a3, 80(sp)                      # 8-byte Folded Spill
	lui	a0, 16
	add	s9, s8, a0
	lui	a0, 1
	addiw	a0, a0, -2048
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	ld	a0, 0(sp)                       # 8-byte Folded Reload
	slliw	a1, a0, 9
	addi	a0, s8, 64
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	mv	s6, s10
	li	s11, 64
	li	s5, 1
	li	s8, 111
	li	s7, 112
	li	s10, 97
.LBB0_179:                              # Label of block must be emitted
	auipc	s2, %pcrel_hi(run_model.sram_c.12)
	addi	s2, s2, %pcrel_lo(.LBB0_179)
	j	.LBB0_98
.LBB0_96:                               #   in Loop: Header=BB0_98 Depth=1
	slli	a0, a5, 32
	srli	a1, a0, 22
	add	a1, a1, t0
	srli	a0, a0, 20
	add	a0, a0, t1
	#APP
	.insn r 11, 0, 2, s2, a1, a0

	#NO_APP
.LBB0_97:                               #   in Loop: Header=BB0_98 Depth=1
	slli	a0, a7, 2
	ld	a1, 88(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	#APP
	.insn r 11, 0, 1, zero, a0, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	ld	a1, 120(sp)                     # 8-byte Folded Reload
	addw	a1, a1, a0
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	mv	s6, a0
	ld	a2, 128(sp)                     # 8-byte Folded Reload
	bge	a0, a2, .LBB0_113
.LBB0_98:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_103 Depth 2
                                        #     Child Loop BB0_106 Depth 2
                                        #       Child Loop BB0_111 Depth 3
	addi	a0, s6, 16
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	blt	a2, a0, .LBB0_100
# %bb.99:                               #   in Loop: Header=BB0_98 Depth=1
	li	s3, 16
	j	.LBB0_101
.LBB0_100:                              #   in Loop: Header=BB0_98 Depth=1
	subw	s3, a2, s6
.LBB0_101:                              #   in Loop: Header=BB0_98 Depth=1
	sd	a1, 120(sp)                     # 8-byte Folded Spill
	slli	s0, a1, 2
	slli	a0, s3, 32
	srli	s4, a0, 32
	lui	a2, 1
	mv	a0, s2
	li	a1, 0
	call	memset@plt
.LBB0_180:                              #   in Loop: Header=BB0_98 Depth=1
                                        # Label of block must be emitted
	auipc	t0, %pcrel_hi(run_model.sram_a.10)
	addi	t0, t0, %pcrel_lo(.LBB0_180)
	blez	s3, .LBB0_104
# %bb.102:                              #   in Loop: Header=BB0_98 Depth=1
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, s0
	mv	a1, s4
	mv	a2, t0
.LBB0_103:                              #   Parent Loop BB0_98 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a2, a0

	#NO_APP
	addi	a0, a0, 512
	addi	a1, a1, -1
	addi	a2, a2, 64
	bnez	a1, .LBB0_103
.LBB0_104:                              #   in Loop: Header=BB0_98 Depth=1
	li	a1, 0
	li	a5, 0
.LBB0_181:                              #   in Loop: Header=BB0_98 Depth=1
                                        # Label of block must be emitted
	auipc	t1, %pcrel_hi(run_model.sram_b.11)
	addi	t1, t1, %pcrel_lo(.LBB0_181)
	lui	a0, 1
	#APP
	.insn r 11, 0, 1, zero, t1, s9

	#NO_APP
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	add	s0, s0, a0
	slliw	a7, s6, 6
	slli	a6, s3, 8
	j	.LBB0_106
.LBB0_105:                              #   in Loop: Header=BB0_106 Depth=2
	addi	a1, a0, 16
	srli	a2, s1, 32
	slli	a2, a2, 12
	add	a2, a2, t1
	slli	a3, a1, 8
	add	a3, a3, s9
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
	.insn r 11, 0, 2, s2, a3, a2

	#NO_APP
	addi	s0, s0, 64
	mv	a5, t2
	bgeu	a0, s7, .LBB0_97
.LBB0_106:                              #   Parent Loop BB0_98 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_111 Depth 3
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	bltu	s8, a1, .LBB0_96
# %bb.107:                              #   in Loop: Header=BB0_106 Depth=2
	mv	a0, a1
	li	t3, 16
	bltu	a1, s10, .LBB0_109
# %bb.108:                              #   in Loop: Header=BB0_106 Depth=2
	subw	t3, s7, a0
.LBB0_109:                              #   in Loop: Header=BB0_106 Depth=2
	subw	t2, s5, a5
	slli	s1, t2, 32
	blez	s3, .LBB0_105
# %bb.110:                              #   in Loop: Header=BB0_106 Depth=2
	srli	a1, s1, 22
	add	a1, a1, t0
	slli	a2, t3, 32
	srli	a3, a2, 30
	mv	a2, s4
	mv	a4, s0
.LBB0_111:                              #   Parent Loop BB0_98 Depth=1
                                        #     Parent Loop BB0_106 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	#APP
	.insn r 11, 0, 1, zero, a1, a4

	#NO_APP
	addi	a4, a4, 512
	addi	a2, a2, -1
	addi	a1, a1, 64
	bnez	a2, .LBB0_111
	j	.LBB0_105
.LBB0_112:
	#APP
	.insn r 11, 0, 5, s8, a4, s1

	#NO_APP
	lui	a0, 3
	add	a2, s8, a0
	j	.LBB0_132
.LBB0_113:
	ld	s8, 32(sp)                      # 8-byte Folded Reload
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	ld	a1, 56(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 5, s8, a0, a1

	#NO_APP
	ld	s6, 8(sp)                       # 8-byte Folded Reload
	bge	s6, a2, .LBB0_131
# %bb.114:
	lui	a0, 40
	add	s3, s8, a0
	ld	a0, 0(sp)                       # 8-byte Folded Reload
	slliw	a0, a0, 9
.LBB0_182:                              # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_c.15)
	addi	s4, s4, %pcrel_lo(.LBB0_182)
.LBB0_183:                              # Label of block must be emitted
	auipc	a1, %pcrel_hi(run_model.sram_a.13)
	addi	a1, a1, %pcrel_lo(.LBB0_183)
	sd	a1, 88(sp)                      # 8-byte Folded Spill
	li	s11, 64
	li	s5, 1
	li	s8, 111
	li	s7, 112
	li	s10, 97
	j	.LBB0_117
.LBB0_115:                              #   in Loop: Header=BB0_117 Depth=1
	slli	a0, a4, 32
	srli	a1, a0, 22
.LBB0_184:                              #   in Loop: Header=BB0_117 Depth=1
                                        # Label of block must be emitted
	auipc	a2, %pcrel_hi(run_model.sram_a.13)
	addi	a2, a2, %pcrel_lo(.LBB0_184)
	add	a1, a1, a2
	srli	a0, a0, 20
	add	a0, a0, t0
	#APP
	.insn r 11, 0, 2, s4, a1, a0

	#NO_APP
.LBB0_116:                              #   in Loop: Header=BB0_117 Depth=1
	slli	a0, a7, 2
	ld	a1, 16(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	#APP
	.insn r 11, 0, 1, zero, a0, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	ld	a1, 120(sp)                     # 8-byte Folded Reload
	addw	a0, a0, a1
	ld	a1, 112(sp)                     # 8-byte Folded Reload
	mv	s6, a1
	ld	a2, 128(sp)                     # 8-byte Folded Reload
	bge	a1, a2, .LBB0_140
.LBB0_117:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_122 Depth 2
                                        #     Child Loop BB0_125 Depth 2
                                        #       Child Loop BB0_130 Depth 3
	addi	a1, s6, 16
	sd	a1, 112(sp)                     # 8-byte Folded Spill
	blt	a2, a1, .LBB0_119
# %bb.118:                              #   in Loop: Header=BB0_117 Depth=1
	li	s2, 16
	j	.LBB0_120
.LBB0_119:                              #   in Loop: Header=BB0_117 Depth=1
	subw	s2, a2, s6
.LBB0_120:                              #   in Loop: Header=BB0_117 Depth=1
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	slli	s1, a0, 2
	slli	a0, s2, 32
	srli	s9, a0, 32
	lui	a2, 1
	mv	a0, s4
	li	a1, 0
	call	memset@plt
	blez	s2, .LBB0_123
# %bb.121:                              #   in Loop: Header=BB0_117 Depth=1
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, s1
	mv	a1, s9
	ld	a2, 88(sp)                      # 8-byte Folded Reload
.LBB0_122:                              #   Parent Loop BB0_117 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a2, a0

	#NO_APP
	addi	a0, a0, 512
	addi	a1, a1, -1
	addi	a2, a2, 64
	bnez	a1, .LBB0_122
.LBB0_123:                              #   in Loop: Header=BB0_117 Depth=1
	li	a5, 0
	li	a4, 0
.LBB0_185:                              #   in Loop: Header=BB0_117 Depth=1
                                        # Label of block must be emitted
	auipc	t0, %pcrel_hi(run_model.sram_b.14)
	addi	t0, t0, %pcrel_lo(.LBB0_185)
	lui	a0, 1
	#APP
	.insn r 11, 0, 1, zero, t0, s3

	#NO_APP
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	add	a1, a0, s1
	slliw	a7, s6, 6
	slli	a6, s2, 8
	j	.LBB0_125
.LBB0_124:                              #   in Loop: Header=BB0_125 Depth=2
	addi	a5, a2, 16
	srli	a0, s0, 32
	slli	a0, a0, 12
	add	a0, a0, t0
	slli	a3, a5, 8
	add	a3, a3, s3
	slli	s1, t3, 8
	#APP
	.insn r 11, 0, 1, zero, a0, a3

	#NO_APP
	slli	a0, a4, 32
	srli	a3, a0, 22
	add	a3, a3, t1
	srli	a0, a0, 20
	add	a0, a0, t0
	#APP
	.insn r 11, 0, 2, s4, a3, a0

	#NO_APP
	addi	a1, a1, 64
	mv	a4, t2
	bgeu	a2, s7, .LBB0_116
.LBB0_125:                              #   Parent Loop BB0_117 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_130 Depth 3
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	bltu	s8, a5, .LBB0_115
# %bb.126:                              #   in Loop: Header=BB0_125 Depth=2
	mv	a2, a5
	li	t3, 16
	bltu	a5, s10, .LBB0_128
# %bb.127:                              #   in Loop: Header=BB0_125 Depth=2
	subw	t3, s7, a2
.LBB0_128:                              #   in Loop: Header=BB0_125 Depth=2
	subw	t2, s5, a4
	slli	s0, t2, 32
.LBB0_186:                              #   in Loop: Header=BB0_125 Depth=2
                                        # Label of block must be emitted
	auipc	t1, %pcrel_hi(run_model.sram_a.13)
	addi	t1, t1, %pcrel_lo(.LBB0_186)
	blez	s2, .LBB0_124
# %bb.129:                              #   in Loop: Header=BB0_125 Depth=2
	srli	a0, s0, 22
	add	a5, t1, a0
	slli	a0, t3, 32
	srli	s1, a0, 30
	mv	a0, s9
	mv	a3, a1
.LBB0_130:                              #   Parent Loop BB0_117 Depth=1
                                        #     Parent Loop BB0_125 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	#APP
	.insn r 11, 0, 1, zero, a5, a3

	#NO_APP
	addi	a3, a3, 512
	addi	a0, a0, -1
	addi	a5, a5, 64
	bnez	a0, .LBB0_130
	j	.LBB0_124
.LBB0_131:
	ld	a2, 16(sp)                      # 8-byte Folded Reload
	ld	a3, 80(sp)                      # 8-byte Folded Reload
	ld	a1, 72(sp)                      # 8-byte Folded Reload
.LBB0_132:
	#APP
	.insn r 11, 0, 5, s8, a3, a1

	#NO_APP
.LBB0_133:
	li	a0, 1024
	mv	a1, s8
.LBB0_134:                              # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a2)
	flw	ft1, 0(a1)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a1)
	addi	a1, a1, 4
	addi	a0, a0, -1
	addi	a2, a2, 4
	bnez	a0, .LBB0_134
# %bb.135:
	li	a0, 1024
	lui	a1, 2
	mv	a2, s8
.LBB0_136:                              # =>This Inner Loop Header: Depth=1
	add	a3, a2, a1
	flw	ft0, 0(a3)
	flw	ft1, 0(a2)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a2)
	addi	a0, a0, -1
	addi	a2, a2, 4
	bnez	a0, .LBB0_136
# %bb.137:
	li	a0, 1024
	lui	a1, 72
.LBB0_138:                              # =>This Inner Loop Header: Depth=1
	add	a2, s8, a1
	flw	ft0, 0(a2)
	flw	ft1, 0(s8)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(s8)
	addi	a0, a0, -1
	addi	s8, s8, 4
	bnez	a0, .LBB0_138
# %bb.139:
	ld	ra, 264(sp)                     # 8-byte Folded Reload
	ld	s0, 256(sp)                     # 8-byte Folded Reload
	ld	s1, 248(sp)                     # 8-byte Folded Reload
	ld	s2, 240(sp)                     # 8-byte Folded Reload
	ld	s3, 232(sp)                     # 8-byte Folded Reload
	ld	s4, 224(sp)                     # 8-byte Folded Reload
	ld	s5, 216(sp)                     # 8-byte Folded Reload
	ld	s6, 208(sp)                     # 8-byte Folded Reload
	ld	s7, 200(sp)                     # 8-byte Folded Reload
	ld	s8, 192(sp)                     # 8-byte Folded Reload
	ld	s9, 184(sp)                     # 8-byte Folded Reload
	ld	s10, 176(sp)                    # 8-byte Folded Reload
	ld	s11, 168(sp)                    # 8-byte Folded Reload
	fld	fs0, 160(sp)                    # 8-byte Folded Reload
	fld	fs1, 152(sp)                    # 8-byte Folded Reload
	fld	fs2, 144(sp)                    # 8-byte Folded Reload
	fld	fs3, 136(sp)                    # 8-byte Folded Reload
	addi	sp, sp, 272
	ret
.LBB0_140:
	ld	s8, 32(sp)                      # 8-byte Folded Reload
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 5, s8, a0, a1

	#NO_APP
	ld	a1, 8(sp)                       # 8-byte Folded Reload
	bge	a1, a2, .LBB0_158
# %bb.141:
	lui	a0, 64
	add	s3, s8, a0
	ld	a0, 0(sp)                       # 8-byte Folded Reload
	slliw	s2, a0, 9
.LBB0_187:                              # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_c.18)
	addi	s4, s4, %pcrel_lo(.LBB0_187)
.LBB0_188:                              # Label of block must be emitted
	auipc	a0, %pcrel_hi(run_model.sram_a.16)
	addi	a0, a0, %pcrel_lo(.LBB0_188)
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	li	s5, 64
	mv	a0, a1
	li	s11, 1
	li	s7, 111
	li	s10, 112
	li	s8, 97
	j	.LBB0_144
.LBB0_142:                              #   in Loop: Header=BB0_144 Depth=1
	slli	a0, a4, 32
	srli	a1, a0, 22
.LBB0_189:                              #   in Loop: Header=BB0_144 Depth=1
                                        # Label of block must be emitted
	auipc	a2, %pcrel_hi(run_model.sram_a.16)
	addi	a2, a2, %pcrel_lo(.LBB0_189)
	add	a1, a1, a2
	srli	a0, a0, 20
	add	a0, a0, t0
	#APP
	.insn r 11, 0, 2, s4, a1, a0

	#NO_APP
.LBB0_143:                              #   in Loop: Header=BB0_144 Depth=1
	slli	a0, a7, 2
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	#APP
	.insn r 11, 0, 1, zero, a0, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	addw	s2, s2, a0
	ld	a1, 120(sp)                     # 8-byte Folded Reload
	mv	a0, a1
	ld	a2, 128(sp)                     # 8-byte Folded Reload
	bge	a1, a2, .LBB0_159
.LBB0_144:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_149 Depth 2
                                        #     Child Loop BB0_152 Depth 2
                                        #       Child Loop BB0_157 Depth 3
	addi	a1, a0, 16
	sd	a1, 120(sp)                     # 8-byte Folded Spill
	blt	a2, a1, .LBB0_146
# %bb.145:                              #   in Loop: Header=BB0_144 Depth=1
	mv	s1, a0
	li	s6, 16
	j	.LBB0_147
.LBB0_146:                              #   in Loop: Header=BB0_144 Depth=1
	mv	s1, a0
	subw	s6, a2, a0
.LBB0_147:                              #   in Loop: Header=BB0_144 Depth=1
	slli	s0, s2, 2
	slli	a0, s6, 32
	srli	s9, a0, 32
	lui	a2, 1
	mv	a0, s4
	li	a1, 0
	call	memset@plt
	blez	s6, .LBB0_150
# %bb.148:                              #   in Loop: Header=BB0_144 Depth=1
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, s0
	mv	a1, s9
	ld	a2, 112(sp)                     # 8-byte Folded Reload
.LBB0_149:                              #   Parent Loop BB0_144 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a2, a0

	#NO_APP
	addi	a0, a0, 512
	addi	a1, a1, -1
	addi	a2, a2, 64
	bnez	a1, .LBB0_149
.LBB0_150:                              #   in Loop: Header=BB0_144 Depth=1
	li	a5, 0
	li	a4, 0
.LBB0_190:                              #   in Loop: Header=BB0_144 Depth=1
                                        # Label of block must be emitted
	auipc	t0, %pcrel_hi(run_model.sram_b.17)
	addi	t0, t0, %pcrel_lo(.LBB0_190)
	lui	a0, 1
	#APP
	.insn r 11, 0, 1, zero, t0, s3

	#NO_APP
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	add	a1, a0, s0
	slliw	a7, s1, 6
	slli	a6, s6, 8
	j	.LBB0_152
.LBB0_151:                              #   in Loop: Header=BB0_152 Depth=2
	addi	a5, a2, 16
	srli	a0, s0, 32
	slli	a0, a0, 12
	add	a0, a0, t0
	slli	a3, a5, 8
	add	a3, a3, s3
	slli	s1, t3, 8
	#APP
	.insn r 11, 0, 1, zero, a0, a3

	#NO_APP
	slli	a0, a4, 32
	srli	a3, a0, 22
	add	a3, a3, t1
	srli	a0, a0, 20
	add	a0, a0, t0
	#APP
	.insn r 11, 0, 2, s4, a3, a0

	#NO_APP
	addi	a1, a1, 64
	mv	a4, t2
	bgeu	a2, s10, .LBB0_143
.LBB0_152:                              #   Parent Loop BB0_144 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_157 Depth 3
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	bltu	s7, a5, .LBB0_142
# %bb.153:                              #   in Loop: Header=BB0_152 Depth=2
	mv	a2, a5
	li	t3, 16
	bltu	a5, s8, .LBB0_155
# %bb.154:                              #   in Loop: Header=BB0_152 Depth=2
	subw	t3, s10, a2
.LBB0_155:                              #   in Loop: Header=BB0_152 Depth=2
	subw	t2, s11, a4
	slli	s0, t2, 32
.LBB0_191:                              #   in Loop: Header=BB0_152 Depth=2
                                        # Label of block must be emitted
	auipc	t1, %pcrel_hi(run_model.sram_a.16)
	addi	t1, t1, %pcrel_lo(.LBB0_191)
	blez	s6, .LBB0_151
# %bb.156:                              #   in Loop: Header=BB0_152 Depth=2
	srli	a0, s0, 22
	add	a5, t1, a0
	slli	a0, t3, 32
	srli	s1, a0, 30
	mv	a0, s9
	mv	a3, a1
.LBB0_157:                              #   Parent Loop BB0_144 Depth=1
                                        #     Parent Loop BB0_152 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	#APP
	.insn r 11, 0, 1, zero, a5, a3

	#NO_APP
	addi	a3, a3, 512
	addi	a0, a0, -1
	addi	a5, a5, 64
	bnez	a0, .LBB0_157
	j	.LBB0_151
.LBB0_158:
	ld	a2, 16(sp)                      # 8-byte Folded Reload
	j	.LBB0_133
.LBB0_159:
	ld	s8, 32(sp)                      # 8-byte Folded Reload
	ld	a2, 16(sp)                      # 8-byte Folded Reload
	j	.LBB0_133
.Lfunc_end0:
	.size	run_model, .Lfunc_end0-run_model
                                        # -- End function
	.type	run_model.sram_a,@object        # @run_model.sram_a
	.local	run_model.sram_a
	.comm	run_model.sram_a,4096,4
	.type	run_model.sram_b,@object        # @run_model.sram_b
	.local	run_model.sram_b
	.comm	run_model.sram_b,4096,4
	.type	run_model.sram_c,@object        # @run_model.sram_c
	.local	run_model.sram_c
	.comm	run_model.sram_c,1024,4
	.type	run_model.sram_a.1,@object      # @run_model.sram_a.1
	.local	run_model.sram_a.1
	.comm	run_model.sram_a.1,2048,4
	.type	run_model.sram_b.2,@object      # @run_model.sram_b.2
	.local	run_model.sram_b.2
	.comm	run_model.sram_b.2,2048,4
	.type	run_model.sram_c.3,@object      # @run_model.sram_c.3
	.local	run_model.sram_c.3
	.comm	run_model.sram_c.3,1024,4
	.type	run_model.sram_a.4,@object      # @run_model.sram_a.4
	.local	run_model.sram_a.4
	.comm	run_model.sram_a.4,2048,4
	.type	run_model.sram_b.5,@object      # @run_model.sram_b.5
	.local	run_model.sram_b.5
	.comm	run_model.sram_b.5,8192,4
	.type	run_model.sram_c.6,@object      # @run_model.sram_c.6
	.local	run_model.sram_c.6
	.comm	run_model.sram_c.6,4096,4
	.type	run_model.sram_a.7,@object      # @run_model.sram_a.7
	.local	run_model.sram_a.7
	.comm	run_model.sram_a.7,2048,4
	.type	run_model.sram_b.8,@object      # @run_model.sram_b.8
	.local	run_model.sram_b.8
	.comm	run_model.sram_b.8,8192,4
	.type	run_model.sram_c.9,@object      # @run_model.sram_c.9
	.local	run_model.sram_c.9
	.comm	run_model.sram_c.9,4096,4
	.type	run_model.sram_a.10,@object     # @run_model.sram_a.10
	.local	run_model.sram_a.10
	.comm	run_model.sram_a.10,2048,4
	.type	run_model.sram_b.11,@object     # @run_model.sram_b.11
	.local	run_model.sram_b.11
	.comm	run_model.sram_b.11,8192,4
	.type	run_model.sram_c.12,@object     # @run_model.sram_c.12
	.local	run_model.sram_c.12
	.comm	run_model.sram_c.12,4096,4
	.type	run_model.sram_a.13,@object     # @run_model.sram_a.13
	.local	run_model.sram_a.13
	.comm	run_model.sram_a.13,2048,4
	.type	run_model.sram_b.14,@object     # @run_model.sram_b.14
	.local	run_model.sram_b.14
	.comm	run_model.sram_b.14,8192,4
	.type	run_model.sram_c.15,@object     # @run_model.sram_c.15
	.local	run_model.sram_c.15
	.comm	run_model.sram_c.15,4096,4
	.type	run_model.sram_a.16,@object     # @run_model.sram_a.16
	.local	run_model.sram_a.16
	.comm	run_model.sram_a.16,2048,4
	.type	run_model.sram_b.17,@object     # @run_model.sram_b.17
	.local	run_model.sram_b.17
	.comm	run_model.sram_b.17,8192,4
	.type	run_model.sram_c.18,@object     # @run_model.sram_c.18
	.local	run_model.sram_c.18
	.comm	run_model.sram_c.18,4096,4
	.ident	"Ubuntu clang version 15.0.7"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym run_model.sram_a
	.addrsig_sym run_model.sram_b
	.addrsig_sym run_model.sram_c
	.addrsig_sym run_model.sram_a.1
	.addrsig_sym run_model.sram_b.2
	.addrsig_sym run_model.sram_c.3
	.addrsig_sym run_model.sram_a.4
	.addrsig_sym run_model.sram_b.5
	.addrsig_sym run_model.sram_c.6
	.addrsig_sym run_model.sram_a.7
	.addrsig_sym run_model.sram_b.8
	.addrsig_sym run_model.sram_c.9
	.addrsig_sym run_model.sram_a.10
	.addrsig_sym run_model.sram_b.11
	.addrsig_sym run_model.sram_c.12
	.addrsig_sym run_model.sram_a.13
	.addrsig_sym run_model.sram_b.14
	.addrsig_sym run_model.sram_c.15
	.addrsig_sym run_model.sram_a.16
	.addrsig_sym run_model.sram_b.17
	.addrsig_sym run_model.sram_c.18
