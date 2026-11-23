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
	add	t0, a0, a2
	mv	a7, a1
	li	a6, 0
	mv	s4, t0
	lui	a1, 3
	lui	a2, 86
	add	s10, a0, a2
	lui	a2, 88
	add	a2, a2, a0
	#APP
	.insn r 11, 0, 4, a2, s10, a2

	#NO_APP
	#APP
	.insn r 11, 0, 2, t0, a2, t0

	#NO_APP
	lui	a3, 87
	add	a3, a3, a0
	lui	a4, 83
	addiw	a4, a4, -2048
	add	a4, a4, a0
	#APP
	.insn r 11, 0, 2, a4, a2, a3

	#NO_APP
	lui	a3, 82
	add	a3, a3, a0
	#APP
	.insn r 11, 0, 2, a3, a2, a3

	#NO_APP
	lui	a2, 90
	addiw	a5, a2, 256
	add	a5, a5, a0
	#APP
	.insn r 11, 0, 4, t0, t0, a5

	#NO_APP
	lui	a5, 78
	add	s5, a0, a5
	#APP
	.insn r 11, 0, 2, t0, t0, s5

	#NO_APP
	lui	a5, 80
	add	a5, a5, a0
	lui	s1, 77
	add	t1, a0, s1
	#APP
	.insn r 11, 0, 2, t1, t0, a5

	#NO_APP
	addiw	a5, a2, -2048
	add	a5, a5, a0
	lui	s1, 79
	add	t2, a0, s1
	#APP
	.insn r 11, 0, 6, t2, a4, a5

	#NO_APP
	addiw	a2, a2, 384
	add	a2, a2, a0
	#APP
	.insn r 11, 0, 4, s5, a3, a2

	#NO_APP
	lui	a2, 84
	add	a2, a2, a0
	#APP
	.insn r 11, 0, 2, s5, s5, a2

	#NO_APP
	lui	a2, 89
	add	a2, a2, a0
	addiw	a3, s1, 1024
	add	t3, a0, a3
	#APP
	.insn r 11, 0, 2, t3, s5, a2

	#NO_APP
	#APP
	.insn r 11, 0, 6, t3, t3, a5

	#NO_APP
	add	a1, a1, a0
	sd	a1, 24(sp)                      # 8-byte Folded Spill
	li	a1, 256
	li	a2, 16
	mv	a3, t0
	mv	s1, t0
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
	sd	t3, 104(sp)                     # 8-byte Folded Spill
	sd	t2, 112(sp)                     # 8-byte Folded Spill
	sd	t1, 0(sp)                       # 8-byte Folded Spill
	sd	t0, 40(sp)                      # 8-byte Folded Spill
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	sd	a7, 8(sp)                       # 8-byte Folded Spill
	slliw	a2, a7, 2
	li	a1, 12
	mv	a0, a2
	blt	a2, a1, .LBB0_6
# %bb.5:
	li	a0, 12
.LBB0_6:
	addiw	s6, a0, 4
	sd	a2, 16(sp)                      # 8-byte Folded Spill
	ld	s3, 40(sp)                      # 8-byte Folded Reload
	bge	a2, s6, .LBB0_9
# %bb.7:
	lui	a0, 77
	addiw	a0, a0, -2048
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 128(sp)                     # 8-byte Folded Spill
.LBB0_91:                               # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_c)
	addi	s9, s9, %pcrel_lo(.LBB0_91)
.LBB0_92:                               # Label of block must be emitted
	auipc	a1, %pcrel_hi(run_model.sram_a)
	addi	a1, a1, %pcrel_lo(.LBB0_92)
	lui	a0, 1
	addiw	s0, a0, -2048
.LBB0_93:                               # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_b)
	addi	s7, s7, %pcrel_lo(.LBB0_93)
	sd	a1, 120(sp)                     # 8-byte Folded Spill
	add	s8, a1, s0
	add	s2, s7, s0
	ld	s11, 16(sp)                     # 8-byte Folded Reload
.LBB0_8:                                # =>This Inner Loop Header: Depth=1
	slliw	a0, s11, 6
	slli	a0, a0, 2
	add	s1, s5, a0
	li	a2, 1024
	mv	a0, s9
	li	a1, 0
	call	memset@plt
	ld	a1, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, a1, s1

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s7, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s1, 128
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s9, a1, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s9, s8, s2

	#NO_APP
	slliw	a0, s11, 4
	slli	a0, a0, 2
	add	a0, a0, s3
	li	a1, 1024
	#APP
	.insn r 11, 0, 1, zero, a0, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s11, s11, 16
	blt	s11, s6, .LBB0_8
.LBB0_9:
	sd	s6, 48(sp)                      # 8-byte Folded Spill
	lui	a2, 79
	addiw	a0, a2, 512
	ld	a5, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, a5
	li	a1, -64
	addiw	a2, a2, 60
.LBB0_10:                               # =>This Inner Loop Header: Depth=1
	add	a3, a5, a2
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
	bnez	a1, .LBB0_10
# %bb.11:
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	ld	s8, 48(sp)                      # 8-byte Folded Reload
	ld	s11, 112(sp)                    # 8-byte Folded Reload
	ld	s2, 104(sp)                     # 8-byte Folded Reload
	bge	a0, s8, .LBB0_14
# %bb.12:
	lui	a0, 76
	addiw	a0, a0, 1024
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 128(sp)                     # 8-byte Folded Spill
.LBB0_94:                               # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_c.3)
	addi	s9, s9, %pcrel_lo(.LBB0_94)
	li	s3, 1024
.LBB0_95:                               # Label of block must be emitted
	auipc	s6, %pcrel_hi(run_model.sram_a.1)
	addi	s6, s6, %pcrel_lo(.LBB0_95)
.LBB0_96:                               # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_b.2)
	addi	s5, s5, %pcrel_lo(.LBB0_96)
	ld	s0, 16(sp)                      # 8-byte Folded Reload
.LBB0_13:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 4
	slli	s1, a0, 2
	add	s7, s2, s1
	li	a2, 1024
	mv	a0, s9
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s6, s7

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s5, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s9, s6, s5

	#NO_APP
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	add	a0, a0, s1
	#APP
	.insn r 11, 0, 1, zero, a0, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, s8, .LBB0_13
.LBB0_14:
	li	a0, 256
	mv	a1, s4
.LBB0_15:                               # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a1)
	flw	ft1, 1024(a1)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a1)
	addi	a0, a0, -1
	addi	a1, a1, 4
	bnez	a0, .LBB0_15
# %bb.16:
	fmv.w.x	fs0, zero
	li	s0, 64
.LBB0_17:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_48 Depth 2
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	slli	a0, a0, 6
	ld	a1, 40(sp)                      # 8-byte Folded Reload
	add	s2, a1, a0
	flw	ft1, 0(s2)
	flw	ft0, 4(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_19
# %bb.18:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft0, ft1
.LBB0_19:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft1, 8(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_21
# %bb.20:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft1, ft0
.LBB0_21:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft0, 12(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_23
# %bb.22:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft0, ft1
.LBB0_23:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft1, 16(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_25
# %bb.24:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft1, ft0
.LBB0_25:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft0, 20(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_27
# %bb.26:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft0, ft1
.LBB0_27:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft1, 24(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_29
# %bb.28:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft1, ft0
.LBB0_29:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft0, 28(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_31
# %bb.30:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft0, ft1
.LBB0_31:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft1, 32(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_33
# %bb.32:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft1, ft0
.LBB0_33:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft0, 36(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_35
# %bb.34:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft0, ft1
.LBB0_35:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft1, 40(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_37
# %bb.36:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft1, ft0
.LBB0_37:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft0, 44(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_39
# %bb.38:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft0, ft1
.LBB0_39:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft1, 48(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_41
# %bb.40:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft1, ft0
.LBB0_41:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft0, 52(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_43
# %bb.42:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft0, ft1
.LBB0_43:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft1, 56(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_45
# %bb.44:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	ft1, ft0
.LBB0_45:                               #   in Loop: Header=BB0_17 Depth=1
	flw	fs2, 60(s2)
	flt.s	a0, ft1, fs2
	bnez	a0, .LBB0_47
# %bb.46:                               #   in Loop: Header=BB0_17 Depth=1
	fmv.s	fs2, ft1
.LBB0_47:                               #   in Loop: Header=BB0_17 Depth=1
	li	s9, 0
	addi	s6, s2, 4
	addi	s3, s2, 8
	addi	s7, s2, 12
	addi	s11, s2, 16
	addi	s8, s2, 20
	addi	s5, s2, 24
	addi	a0, s2, 28
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	addi	a0, s2, 32
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	addi	a0, s2, 36
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	addi	a0, s2, 40
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addi	a0, s2, 44
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	addi	a0, s2, 48
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	addi	a0, s2, 52
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	addi	a0, s2, 56
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	addi	a0, s2, 60
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	fmv.s	fs1, fs0
.LBB0_48:                               #   Parent Loop BB0_17 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	s1, s4, s9
	flw	ft0, 0(s1)
	fsub.s	ft0, ft0, fs2
	fcvt.d.s	fa0, ft0
	call	exp@plt
	fcvt.s.d	ft0, fa0
	fsw	ft0, 0(s1)
	addi	s9, s9, 4
	fadd.s	fs1, fs1, ft0
	bne	s9, s0, .LBB0_48
# %bb.49:                               #   in Loop: Header=BB0_17 Depth=1
	flw	ft0, 0(s2)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s2)
	flw	ft0, 0(s6)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s6)
	flw	ft0, 0(s3)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s3)
	flw	ft0, 0(s7)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s7)
	flw	ft0, 0(s11)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s11)
	flw	ft0, 0(s8)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s8)
	flw	ft0, 0(s5)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s5)
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
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1
	addi	s4, s4, 64
	ld	s8, 48(sp)                      # 8-byte Folded Reload
	li	a1, 16
	bne	a0, a1, .LBB0_17
# %bb.50:
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	ld	s9, 40(sp)                      # 8-byte Folded Reload
	bge	a0, s8, .LBB0_56
# %bb.51:
.LBB0_97:                               # Label of block must be emitted
	auipc	s6, %pcrel_hi(run_model.sram_c.6)
	addi	s6, s6, %pcrel_lo(.LBB0_97)
	lui	s2, 1
.LBB0_98:                               # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_a.4)
	addi	s4, s4, %pcrel_lo(.LBB0_98)
	li	s3, 1024
.LBB0_99:                               # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_b.5)
	addi	s5, s5, %pcrel_lo(.LBB0_99)
	ld	s0, 16(sp)                      # 8-byte Folded Reload
	ld	s11, 0(sp)                      # 8-byte Folded Reload
.LBB0_52:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 4
	slli	a0, a0, 2
	add	s1, s9, a0
	slliw	s7, s0, 6
	lui	a2, 1
	mv	a0, s6
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s4, s1

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s5, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s6, s4, s5

	#NO_APP
	slli	a0, s7, 2
	add	a0, a0, s9
	#APP
	.insn r 11, 0, 1, zero, a0, s6

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, s8, .LBB0_52
# %bb.53:
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	bge	a0, s8, .LBB0_56
# %bb.54:
	mv	s3, s8
	mv	s6, s9
	lui	a0, 72
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	lui	a0, 73
	add	a0, a0, a1
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	lui	a0, 74
	add	a0, a0, a1
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	lui	a0, 75
	add	a0, a0, a1
	sd	a0, 104(sp)                     # 8-byte Folded Spill
.LBB0_100:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.9)
	addi	s1, s1, %pcrel_lo(.LBB0_100)
	lui	s7, 1
.LBB0_101:                              # Label of block must be emitted
	auipc	s8, %pcrel_hi(run_model.sram_a.7)
	addi	s8, s8, %pcrel_lo(.LBB0_101)
.LBB0_102:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_b.8)
	addi	s9, s9, %pcrel_lo(.LBB0_102)
	addi	s11, s8, 1024
	add	s0, s9, s7
	ld	s2, 16(sp)                      # 8-byte Folded Reload
.LBB0_55:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s2, 6
	slli	s4, a0, 2
	add	s5, s6, s4
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	li	a4, 1024
	#APP
	.insn r 11, 0, 1, zero, s8, s5

	#NO_APP
	ld	a1, 128(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s5, 64
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s0, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s5, 128
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s11, s0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s5, 192
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s0, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s11, s0

	#NO_APP
	add	a0, a1, s4
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s2, s2, 16
	blt	s2, s3, .LBB0_55
.LBB0_56:
	li	s4, 0
	lui	a0, 90
	ld	s6, 32(sp)                      # 8-byte Folded Reload
	add	s5, s6, a0
	lui	a0, 73
	add	s2, s6, a0
	lui	a0, 72
	add	s0, s6, a0
.LBB0_103:                              # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_0)
	addi	a0, a0, %pcrel_lo(.LBB0_103)
	flw	fs0, 0(a0)
.LBB0_104:                              # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_1)
	addi	a0, a0, %pcrel_lo(.LBB0_104)
	flw	fs1, 0(a0)
.LBB0_105:                              # Label of block must be emitted
	auipc	a0, %pcrel_hi(.LCPI0_2)
	addi	a0, a0, %pcrel_lo(.LBB0_105)
	flw	fs2, 0(a0)
	fmv.w.x	fs3, zero
	li	s8, 256
	li	s7, 16
	mv	s1, s2
.LBB0_57:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_58 Depth 2
                                        #     Child Loop BB0_61 Depth 2
	li	a0, 0
	fmv.s	ft0, fs3
.LBB0_58:                               #   Parent Loop BB0_57 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s10, a0
	flw	ft1, 0(a1)
	add	a1, s0, a0
	flw	ft2, 0(a1)
	fadd.s	ft1, ft1, ft2
	fsw	ft1, 0(a1)
	addi	a0, a0, 4
	fmadd.s	ft0, ft1, ft1, ft0
	bne	a0, s8, .LBB0_58
# %bb.59:                               #   in Loop: Header=BB0_57 Depth=1
	fmul.s	ft0, ft0, fs0
	fadd.s	ft0, ft0, fs1
	fsqrt.s	fa0, ft0
	feq.s	a0, fa0, fa0
	beqz	a0, .LBB0_63
.LBB0_60:                               #   in Loop: Header=BB0_57 Depth=1
	li	a0, 0
	fdiv.s	ft0, fs2, fa0
.LBB0_61:                               #   Parent Loop BB0_57 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s0, a0
	flw	ft1, 0(a1)
	add	a1, s5, a0
	flw	ft2, 0(a1)
	fmul.s	ft1, ft0, ft1
	fmul.s	ft1, ft1, ft2
	add	a1, s1, a0
	addi	a0, a0, 4
	fsw	ft1, 0(a1)
	bne	a0, s8, .LBB0_61
# %bb.62:                               #   in Loop: Header=BB0_57 Depth=1
	addi	s4, s4, 1
	addi	s0, s0, 256
	addi	s10, s10, 256
	addi	s1, s1, 256
	bne	s4, s7, .LBB0_57
	j	.LBB0_64
.LBB0_63:                               #   in Loop: Header=BB0_57 Depth=1
	fmv.s	fa0, ft0
	call	sqrtf@plt
	j	.LBB0_60
.LBB0_64:
	#APP
	.insn r 11, 0, 2, s6, s2, s6

	#NO_APP
	lui	a0, 8
	add	a3, s6, a0
	lui	a0, 2
	add	a4, s6, a0
	#APP
	.insn r 11, 0, 2, a4, s2, a3

	#NO_APP
	lui	a0, 24
	add	a0, a0, s6
	lui	a1, 4
	add	a2, s6, a1
	#APP
	.insn r 11, 0, 2, a2, s2, a0

	#NO_APP
	lui	a0, 32
	add	a0, a0, s6
	lui	a1, 6
	add	a5, s6, a1
	#APP
	.insn r 11, 0, 2, a5, s2, a0

	#NO_APP
	lui	a0, 48
	add	a0, a0, s6
	#APP
	.insn r 11, 0, 2, a3, s2, a0

	#NO_APP
	lui	a0, 56
	add	a0, a0, s6
	lui	a1, 10
	add	a1, a1, s6
	#APP
	.insn r 11, 0, 2, a1, s2, a0

	#NO_APP
	sd	a4, 112(sp)                     # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 5, s6, s6, a4

	#NO_APP
	ld	s10, 16(sp)                     # 8-byte Folded Reload
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	bge	s10, a0, .LBB0_80
# %bb.65:
	sd	a5, 80(sp)                      # 8-byte Folded Spill
	sd	a2, 104(sp)                     # 8-byte Folded Spill
	sd	a1, 88(sp)                      # 8-byte Folded Spill
	sd	a3, 96(sp)                      # 8-byte Folded Spill
	lui	a0, 16
	add	s9, s6, a0
	lui	s8, 1
	addiw	a0, s8, -2048
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	lui	a0, 17
	add	s2, s6, a0
	addi	a0, s6, 64
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	slliw	s11, a0, 9
	li	s4, 1024
	li	s7, 1
	li	s5, 96
.LBB0_106:                              # Label of block must be emitted
	auipc	s3, %pcrel_hi(run_model.sram_c.12)
	addi	s3, s3, %pcrel_lo(.LBB0_106)
.LBB0_66:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_67 Depth 2
	slli	a0, s11, 2
	ld	a1, 120(sp)                     # 8-byte Folded Reload
	add	s0, a1, a0
	slliw	a0, s10, 7
	slli	a0, a0, 2
	add	s1, s6, a0
	lui	a2, 1
	mv	a0, s3
	li	a1, 0
	call	memset@plt
.LBB0_107:                              #   in Loop: Header=BB0_66 Depth=1
                                        # Label of block must be emitted
	auipc	a6, %pcrel_hi(run_model.sram_a.10)
	addi	a6, a6, %pcrel_lo(.LBB0_107)
	#APP
	.insn r 11, 0, 1, zero, a6, s1

	#NO_APP
.LBB0_108:                              #   in Loop: Header=BB0_66 Depth=1
                                        # Label of block must be emitted
	auipc	a7, %pcrel_hi(run_model.sram_b.11)
	addi	a7, a7, %pcrel_lo(.LBB0_108)
	#APP
	.insn r 11, 0, 1, zero, a7, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	li	a4, 0
	li	a2, -16
	mv	a3, s2
.LBB0_67:                               #   Parent Loop BB0_66 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	slli	a0, a4, 32
	subw	a4, s7, a4
	slli	s1, a4, 32
	srli	a5, s1, 22
	add	a5, a5, a6
	#APP
	.insn r 11, 0, 1, zero, a5, s0

	#NO_APP
	srli	s1, s1, 20
	add	s1, s1, a7
	#APP
	.insn r 11, 0, 1, zero, s1, a3

	#NO_APP
	srli	a1, a0, 22
	add	a1, a1, a6
	srli	a0, a0, 20
	add	a0, a0, a7
	#APP
	.insn r 11, 0, 2, s3, a1, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a2, a2, 16
	add	a3, a3, s8
	addi	s0, s0, 64
	bltu	a2, s5, .LBB0_67
# %bb.68:                               #   in Loop: Header=BB0_66 Depth=1
	#APP
	.insn r 11, 0, 2, s3, a5, s1

	#NO_APP
	slliw	a0, s10, 6
	slli	a0, a0, 2
	ld	a1, 112(sp)                     # 8-byte Folded Reload
	add	a0, a0, a1
	#APP
	.insn r 11, 0, 1, zero, a0, s3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s10, s10, 16
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	addw	s11, s11, a0
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	blt	s10, a0, .LBB0_66
# %bb.69:
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	ld	a1, 80(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 5, s6, a0, a1

	#NO_APP
	ld	s2, 16(sp)                      # 8-byte Folded Reload
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	bge	s2, a0, .LBB0_81
# %bb.70:
	lui	a0, 40
	add	a0, a0, s6
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	lui	a0, 41
	add	a0, a0, s6
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	slliw	s4, a0, 9
.LBB0_109:                              # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_c.15)
	addi	s5, s5, %pcrel_lo(.LBB0_109)
	lui	s9, 1
.LBB0_110:                              # Label of block must be emitted
	auipc	s6, %pcrel_hi(run_model.sram_a.13)
	addi	s6, s6, %pcrel_lo(.LBB0_110)
	li	s7, 1024
.LBB0_111:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_b.14)
	addi	s1, s1, %pcrel_lo(.LBB0_111)
	li	s10, 1
	li	s11, 96
.LBB0_71:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_72 Depth 2
	slli	a0, s4, 2
	ld	a1, 120(sp)                     # 8-byte Folded Reload
	add	s8, a1, a0
	slliw	a0, s2, 7
	slli	a0, a0, 2
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	s3, a1, a0
	lui	a2, 1
	mv	a0, s5
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s6, s3

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s1, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	li	a2, 0
	li	a0, -16
	ld	a1, 104(sp)                     # 8-byte Folded Reload
.LBB0_72:                               #   Parent Loop BB0_71 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	slli	a5, a2, 32
	subw	a2, s10, a2
	slli	a4, a2, 32
	srli	a3, a4, 22
	add	a3, a3, s6
	#APP
	.insn r 11, 0, 1, zero, a3, s8

	#NO_APP
	srli	a4, a4, 20
	add	a4, a4, s1
	#APP
	.insn r 11, 0, 1, zero, a4, a1

	#NO_APP
	srli	s0, a5, 22
	add	s0, s0, s6
	srli	a5, a5, 20
	add	a5, a5, s1
	#APP
	.insn r 11, 0, 2, s5, s0, a5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, a0, 16
	add	a1, a1, s9
	addi	s8, s8, 64
	bltu	a0, s11, .LBB0_72
# %bb.73:                               #   in Loop: Header=BB0_71 Depth=1
	#APP
	.insn r 11, 0, 2, s5, a3, a4

	#NO_APP
	slliw	a0, s2, 6
	slli	a0, a0, 2
	ld	a1, 24(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	#APP
	.insn r 11, 0, 1, zero, a0, s5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s2, s2, 16
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	addw	s4, s4, a0
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	blt	s2, a0, .LBB0_71
# %bb.74:
	ld	s6, 32(sp)                      # 8-byte Folded Reload
	ld	a1, 96(sp)                      # 8-byte Folded Reload
	ld	a2, 88(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 5, s6, a1, a2

	#NO_APP
	ld	a2, 16(sp)                      # 8-byte Folded Reload
	bge	a2, a0, .LBB0_90
# %bb.75:
	lui	a0, 64
	add	a0, a0, s6
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	lui	a0, 65
	add	a0, a0, s6
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	slliw	s4, a0, 9
.LBB0_112:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_c.18)
	addi	s9, s9, %pcrel_lo(.LBB0_112)
	lui	s7, 1
	mv	a1, s6
.LBB0_113:                              # Label of block must be emitted
	auipc	s6, %pcrel_hi(run_model.sram_a.16)
	addi	s6, s6, %pcrel_lo(.LBB0_113)
	li	s5, 1024
.LBB0_114:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_b.17)
	addi	s1, s1, %pcrel_lo(.LBB0_114)
	li	s10, 1
	li	s11, 96
.LBB0_76:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_77 Depth 2
	slli	a0, s4, 2
	ld	a3, 120(sp)                     # 8-byte Folded Reload
	add	s2, a3, a0
	mv	s8, a2
	slliw	a0, a2, 7
	slli	a0, a0, 2
	add	s3, a1, a0
	lui	a2, 1
	mv	a0, s9
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s6, s3

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s1, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	li	a2, 0
	li	a0, -16
	ld	a1, 104(sp)                     # 8-byte Folded Reload
.LBB0_77:                               #   Parent Loop BB0_76 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	slli	a5, a2, 32
	subw	a2, s10, a2
	slli	a4, a2, 32
	srli	a3, a4, 22
	add	a3, a3, s6
	#APP
	.insn r 11, 0, 1, zero, a3, s2

	#NO_APP
	srli	a4, a4, 20
	add	a4, a4, s1
	#APP
	.insn r 11, 0, 1, zero, a4, a1

	#NO_APP
	srli	s0, a5, 22
	add	s0, s0, s6
	srli	a5, a5, 20
	add	a5, a5, s1
	#APP
	.insn r 11, 0, 2, s9, s0, a5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, a0, 16
	add	a1, a1, s7
	addi	s2, s2, 64
	bltu	a0, s11, .LBB0_77
# %bb.78:                               #   in Loop: Header=BB0_76 Depth=1
	#APP
	.insn r 11, 0, 2, s9, a3, a4

	#NO_APP
	mv	a2, s8
	slliw	a0, s8, 6
	slli	a0, a0, 2
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	#APP
	.insn r 11, 0, 1, zero, a0, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a2, s8, 16
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	addw	s4, s4, a0
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	blt	a2, a0, .LBB0_76
# %bb.79:
	mv	s6, a1
	ld	a2, 24(sp)                      # 8-byte Folded Reload
	j	.LBB0_83
.LBB0_80:
	#APP
	.insn r 11, 0, 5, s6, a2, a5

	#NO_APP
	lui	a0, 3
	add	a2, s6, a0
	j	.LBB0_82
.LBB0_81:
	ld	a2, 24(sp)                      # 8-byte Folded Reload
	ld	a3, 96(sp)                      # 8-byte Folded Reload
	ld	a1, 88(sp)                      # 8-byte Folded Reload
.LBB0_82:
	#APP
	.insn r 11, 0, 5, s6, a3, a1

	#NO_APP
.LBB0_83:
	li	a0, 1024
	mv	a1, s6
.LBB0_84:                               # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a2)
	flw	ft1, 0(a1)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a1)
	addi	a1, a1, 4
	addi	a0, a0, -1
	addi	a2, a2, 4
	bnez	a0, .LBB0_84
# %bb.85:
	li	a0, 1024
	lui	a1, 2
	mv	a2, s6
.LBB0_86:                               # =>This Inner Loop Header: Depth=1
	add	a3, a2, a1
	flw	ft0, 0(a3)
	flw	ft1, 0(a2)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a2)
	addi	a0, a0, -1
	addi	a2, a2, 4
	bnez	a0, .LBB0_86
# %bb.87:
	li	a0, 1024
	lui	a1, 72
.LBB0_88:                               # =>This Inner Loop Header: Depth=1
	add	a2, s6, a1
	flw	ft0, 0(a2)
	flw	ft1, 0(s6)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(s6)
	addi	a0, a0, -1
	addi	s6, s6, 4
	bnez	a0, .LBB0_88
# %bb.89:
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
.LBB0_90:
	ld	a2, 24(sp)                      # 8-byte Folded Reload
	j	.LBB0_83
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
