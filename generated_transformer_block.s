	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.file	"generated_transformer_block.c"
	.globl	run_model                       # -- Begin function run_model
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
	mv	a3, a0
	lui	a0, 40
	add	s1, a3, a0
	lui	a0, 41
	add	a5, a3, a0
	#APP
	.insn r 11, 0, 4, a5, s1, a5

	#NO_APP
	slliw	a4, a1, 2
	li	a2, 12
	lui	a0, 24
	mv	s0, a4
	blt	a4, a2, .LBB0_2
# %bb.1:
	li	s0, 12
.LBB0_2:
	addiw	s3, s0, 4
	add	s4, a3, a0
	sd	a3, 40(sp)                      # 8-byte Folded Spill
	sd	a4, 24(sp)                      # 8-byte Folded Spill
	sd	s4, 64(sp)                      # 8-byte Folded Spill
	sd	a1, 8(sp)                       # 8-byte Folded Spill
	sd	s1, 16(sp)                      # 8-byte Folded Spill
	lui	a0, 25
	bge	a4, s3, .LBB0_12
# %bb.3:
	add	a0, a0, a3
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	lui	a0, 26
	add	a0, a0, a3
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	lui	a0, 27
	add	a0, a0, a3
	sd	a0, 112(sp)                     # 8-byte Folded Spill
.LBB0_80:                               # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_c)
	addi	s10, s10, %pcrel_lo(.LBB0_80)
	lui	s5, 1
.LBB0_81:                               # Label of block must be emitted
	auipc	s11, %pcrel_hi(run_model.sram_a)
	addi	s11, s11, %pcrel_lo(.LBB0_81)
	li	s8, 1024
.LBB0_82:                               # Label of block must be emitted
	auipc	s6, %pcrel_hi(run_model.sram_b)
	addi	s6, s6, %pcrel_lo(.LBB0_82)
	addi	s9, s11, 1024
	add	s2, s6, s5
	mv	s0, a4
	sd	a5, 136(sp)                     # 8-byte Folded Spill
.LBB0_4:                                # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 6
	slli	s7, a0, 2
	add	s1, a5, s7
	lui	a2, 1
	mv	a0, s10
	li	a1, 0
	call	memset@plt
	ld	a5, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, s1

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s6, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s1, 64
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s10, s11, s6

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s1, 128
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s6, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s10, s9, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s1, 192
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s10, s11, s6

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s10, s9, s2

	#NO_APP
	add	a0, s4, s7
	#APP
	.insn r 11, 0, 1, zero, a0, s10

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, s3, .LBB0_4
# %bb.5:
	lui	a0, 25
	ld	a3, 40(sp)                      # 8-byte Folded Reload
	add	s11, a3, a0
	ld	s0, 24(sp)                      # 8-byte Folded Reload
	bge	s0, s3, .LBB0_13
# %bb.6:
	lui	a0, 28
	add	a0, a0, a3
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	lui	a0, 29
	add	a0, a0, a3
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	lui	a0, 30
	add	a0, a0, a3
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	lui	a0, 31
	add	a0, a0, a3
	sd	a0, 104(sp)                     # 8-byte Folded Spill
.LBB0_83:                               # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.3)
	addi	s1, s1, %pcrel_lo(.LBB0_83)
	lui	s8, 1
	mv	s4, s3
.LBB0_84:                               # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_a.1)
	addi	s9, s9, %pcrel_lo(.LBB0_84)
	mv	s5, s11
	li	s11, 1024
.LBB0_85:                               # Label of block must be emitted
	auipc	s2, %pcrel_hi(run_model.sram_b.2)
	addi	s2, s2, %pcrel_lo(.LBB0_85)
	addi	s7, s9, 1024
	add	s6, s2, s8
.LBB0_7:                                # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 6
	slli	s10, a0, 2
	add	s3, a5, s10
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	ld	a5, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, s3

	#NO_APP
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s3, 64
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s6, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s3, 128
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s7, s6

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s3, 192
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s6, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s7, s6

	#NO_APP
	add	a0, s5, s10
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, s4, .LBB0_7
# %bb.8:
	sd	s5, 48(sp)                      # 8-byte Folded Spill
	lui	a0, 26
	ld	a1, 40(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	ld	s0, 24(sp)                      # 8-byte Folded Reload
	mv	s3, s4
	bge	s0, s4, .LBB0_79
# %bb.9:
	lui	a0, 32
	add	a0, a0, a1
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	lui	a0, 33
	add	a0, a0, a1
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	lui	a0, 34
	add	a0, a0, a1
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	lui	a0, 35
	add	a0, a0, a1
	sd	a0, 104(sp)                     # 8-byte Folded Spill
.LBB0_86:                               # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.6)
	addi	s1, s1, %pcrel_lo(.LBB0_86)
	lui	s8, 1
	mv	s4, s3
.LBB0_87:                               # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_a.4)
	addi	s9, s9, %pcrel_lo(.LBB0_87)
	li	s11, 1024
.LBB0_88:                               # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_b.5)
	addi	s7, s7, %pcrel_lo(.LBB0_88)
	addi	s6, s9, 1024
	add	s2, s7, s8
	ld	s5, 32(sp)                      # 8-byte Folded Reload
.LBB0_10:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 6
	slli	s10, a0, 2
	add	s3, a5, s10
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	ld	a5, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, s3

	#NO_APP
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s3, 64
	#APP
	.insn r 11, 0, 1, zero, s6, a0

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s3, 128
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s6, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s3, 192
	#APP
	.insn r 11, 0, 1, zero, s6, a0

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s6, s2

	#NO_APP
	add	a0, s5, s10
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, s4, .LBB0_10
# %bb.11:
	mv	s3, s4
	ld	s11, 48(sp)                     # 8-byte Folded Reload
	j	.LBB0_14
.LBB0_12:
	add	s11, a3, a0
.LBB0_13:
	lui	a0, 26
	add	a0, a0, a3
	sd	a0, 32(sp)                      # 8-byte Folded Spill
.LBB0_14:
	li	a0, 0
	li	a1, 256
	li	a2, 16
	mv	a3, s11
	mv	s1, s11
.LBB0_15:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_16 Depth 2
	li	a5, 0
	mv	a4, s1
.LBB0_16:                               #   Parent Loop BB0_15 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	s0, a3, a5
	flw	ft0, 0(s0)
	fsw	ft0, 0(a4)
	addi	a5, a5, 4
	addi	a4, a4, 64
	bne	a5, a1, .LBB0_16
# %bb.17:                               #   in Loop: Header=BB0_15 Depth=1
	addi	a0, a0, 1
	addi	s1, s1, 4
	addi	a3, a3, 256
	bne	a0, a2, .LBB0_15
# %bb.18:
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	mv	s10, s3
	ld	s2, 64(sp)                      # 8-byte Folded Reload
	bge	a0, s3, .LBB0_21
# %bb.19:
	lui	a0, 26
	addiw	a0, a0, -2048
	ld	a1, 40(sp)                      # 8-byte Folded Reload
	add	s3, a1, a0
.LBB0_89:                               # Label of block must be emitted
	auipc	s8, %pcrel_hi(run_model.sram_c.9)
	addi	s8, s8, %pcrel_lo(.LBB0_89)
.LBB0_90:                               # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_a.7)
	addi	s4, s4, %pcrel_lo(.LBB0_90)
	lui	a0, 1
	addiw	s9, a0, -2048
.LBB0_91:                               # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_b.8)
	addi	s5, s5, %pcrel_lo(.LBB0_91)
	add	s6, s4, s9
	add	s7, s5, s9
	ld	s1, 24(sp)                      # 8-byte Folded Reload
.LBB0_20:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s1, 6
	slli	a0, a0, 2
	add	s0, s2, a0
	li	a2, 1024
	mv	a0, s8
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s4, s0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s5, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s0, 128
	#APP
	.insn r 11, 0, 1, zero, s6, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s7, s3

	#NO_APP
	#APP
	.insn r 11, 0, 2, s8, s4, s5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s8, s6, s7

	#NO_APP
	slliw	a0, s1, 4
	slli	a0, a0, 2
	add	a0, a0, s2
	li	a1, 1024
	#APP
	.insn r 11, 0, 1, zero, a0, s8

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s1, s1, 16
	blt	s1, s10, .LBB0_20
.LBB0_21:
	sd	s11, 48(sp)                     # 8-byte Folded Spill
	sd	s10, 56(sp)                     # 8-byte Folded Spill
	li	a0, 0
	fmv.w.x	fs0, zero
	li	s0, 64
	mv	s1, s2
.LBB0_22:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_53 Depth 2
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	slli	a0, a0, 6
	ld	a1, 64(sp)                      # 8-byte Folded Reload
	add	s2, a1, a0
	flw	ft1, 0(s2)
	flw	ft0, 4(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_24
# %bb.23:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft0, ft1
.LBB0_24:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft1, 8(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_26
# %bb.25:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft1, ft0
.LBB0_26:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft0, 12(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_28
# %bb.27:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft0, ft1
.LBB0_28:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft1, 16(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_30
# %bb.29:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft1, ft0
.LBB0_30:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft0, 20(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_32
# %bb.31:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft0, ft1
.LBB0_32:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft1, 24(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_34
# %bb.33:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft1, ft0
.LBB0_34:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft0, 28(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_36
# %bb.35:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft0, ft1
.LBB0_36:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft1, 32(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_38
# %bb.37:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft1, ft0
.LBB0_38:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft0, 36(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_40
# %bb.39:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft0, ft1
.LBB0_40:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft1, 40(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_42
# %bb.41:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft1, ft0
.LBB0_42:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft0, 44(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_44
# %bb.43:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft0, ft1
.LBB0_44:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft1, 48(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_46
# %bb.45:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft1, ft0
.LBB0_46:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft0, 52(s2)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_48
# %bb.47:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft0, ft1
.LBB0_48:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft1, 56(s2)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_50
# %bb.49:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	ft1, ft0
.LBB0_50:                               #   in Loop: Header=BB0_22 Depth=1
	flw	fs2, 60(s2)
	flt.s	a0, ft1, fs2
	bnez	a0, .LBB0_52
# %bb.51:                               #   in Loop: Header=BB0_22 Depth=1
	fmv.s	fs2, ft1
.LBB0_52:                               #   in Loop: Header=BB0_22 Depth=1
	li	s6, 0
	addi	s7, s2, 4
	addi	s9, s2, 8
	addi	s8, s2, 12
	addi	s5, s2, 16
	addi	s4, s2, 20
	addi	s3, s2, 24
	addi	s11, s2, 28
	addi	a0, s2, 32
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	addi	a0, s2, 36
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addi	a0, s2, 40
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	addi	a0, s2, 44
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	addi	a0, s2, 48
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	addi	a0, s2, 52
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	addi	a0, s2, 56
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	addi	a0, s2, 60
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	fmv.s	fs1, fs0
.LBB0_53:                               #   Parent Loop BB0_22 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	s10, s1, s6
	flw	ft0, 0(s10)
	fsub.s	ft0, ft0, fs2
	fcvt.d.s	fa0, ft0
	call	exp@plt
	fcvt.s.d	ft0, fa0
	fsw	ft0, 0(s10)
	addi	s6, s6, 4
	fadd.s	fs1, fs1, ft0
	bne	s6, s0, .LBB0_53
# %bb.54:                               #   in Loop: Header=BB0_22 Depth=1
	flw	ft0, 0(s2)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s2)
	flw	ft0, 0(s7)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s7)
	flw	ft0, 0(s9)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s9)
	flw	ft0, 0(s8)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s8)
	flw	ft0, 0(s5)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s5)
	flw	ft0, 0(s4)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s4)
	flw	ft0, 0(s3)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s3)
	flw	ft0, 0(s11)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s11)
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
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1
	addi	s1, s1, 64
	li	a1, 16
	bne	a0, a1, .LBB0_22
# %bb.55:
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	ld	s8, 56(sp)                      # 8-byte Folded Reload
	ld	a5, 64(sp)                      # 8-byte Folded Reload
	ld	s10, 32(sp)                     # 8-byte Folded Reload
	bge	a0, s8, .LBB0_61
# %bb.56:
.LBB0_92:                               # Label of block must be emitted
	auipc	s6, %pcrel_hi(run_model.sram_c.12)
	addi	s6, s6, %pcrel_lo(.LBB0_92)
	lui	s2, 1
.LBB0_93:                               # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_a.10)
	addi	s4, s4, %pcrel_lo(.LBB0_93)
	li	s3, 1024
.LBB0_94:                               # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_b.11)
	addi	s5, s5, %pcrel_lo(.LBB0_94)
	ld	s0, 24(sp)                      # 8-byte Folded Reload
	mv	s9, a5
.LBB0_57:                               # =>This Inner Loop Header: Depth=1
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
	.insn r 11, 0, 1, zero, s5, s10

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
	blt	s0, s8, .LBB0_57
# %bb.58:
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	mv	a5, s9
	bge	a0, s8, .LBB0_61
# %bb.59:
	mv	s4, s8
	lui	a0, 36
	ld	a1, 40(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a0, 37
	add	a0, a0, a1
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	lui	a0, 38
	add	a0, a0, a1
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	lui	a0, 39
	add	s5, a1, a0
.LBB0_95:                               # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_c.15)
	addi	s10, s10, %pcrel_lo(.LBB0_95)
	lui	s8, 1
.LBB0_96:                               # Label of block must be emitted
	auipc	s11, %pcrel_hi(run_model.sram_a.13)
	addi	s11, s11, %pcrel_lo(.LBB0_96)
	li	s9, 1024
.LBB0_97:                               # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_b.14)
	addi	s7, s7, %pcrel_lo(.LBB0_97)
	addi	s6, s11, 1024
	add	s2, s7, s8
	ld	s0, 24(sp)                      # 8-byte Folded Reload
.LBB0_60:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 6
	slli	a0, a0, 2
	add	s1, a5, a0
	lui	a2, 1
	mv	a0, s10
	li	a1, 0
	mv	s3, a5
	call	memset@plt
	mv	a5, s3
	#APP
	.insn r 11, 0, 1, zero, s11, s1

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s1, 64
	#APP
	.insn r 11, 0, 1, zero, s6, a0

	#NO_APP
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s10, s11, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s1, 128
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s10, s6, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s1, 192
	#APP
	.insn r 11, 0, 1, zero, s6, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s2, s5

	#NO_APP
	#APP
	.insn r 11, 0, 2, s10, s11, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s10, s6, s2

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s1, s10

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, s4, .LBB0_60
.LBB0_61:
	li	a0, 1024
	mv	a1, a5
	ld	a2, 16(sp)                      # 8-byte Folded Reload
.LBB0_62:                               # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a2)
	flw	ft1, 0(a1)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a1)
	addi	a1, a1, 4
	addi	a0, a0, -1
	addi	a2, a2, 4
	bnez	a0, .LBB0_62
# %bb.63:
	lui	a0, 42
	ld	a4, 40(sp)                      # 8-byte Folded Reload
	add	a0, a0, a4
	ld	a3, 48(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 4, a3, a5, a0

	#NO_APP
	ld	s8, 24(sp)                      # 8-byte Folded Reload
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	bge	s8, a0, .LBB0_74
# %bb.64:
	lui	a0, 2
	add	a1, a4, a0
	sd	a1, 120(sp)                     # 8-byte Folded Spill
	lui	a1, 4
	add	a2, a4, a1
	sd	a2, 112(sp)                     # 8-byte Folded Spill
	lui	a2, 6
	add	a5, a4, a2
	sd	a5, 104(sp)                     # 8-byte Folded Spill
	addi	a5, a4, 256
	sd	a5, 96(sp)                      # 8-byte Folded Spill
	addiw	a0, a0, 256
	add	a0, a0, a4
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	addiw	a0, a1, 256
	add	a0, a0, a4
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addiw	a0, a2, 256
	add	a0, a0, a4
	sd	a0, 72(sp)                      # 8-byte Folded Spill
.LBB0_98:                               # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.18)
	addi	s1, s1, %pcrel_lo(.LBB0_98)
	lui	s11, 1
.LBB0_99:                               # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_a.16)
	addi	s10, s10, %pcrel_lo(.LBB0_99)
.LBB0_100:                              # Label of block must be emitted
	auipc	s2, %pcrel_hi(run_model.sram_b.17)
	addi	s2, s2, %pcrel_lo(.LBB0_100)
	addi	s6, s10, 1024
	add	s7, s2, s11
.LBB0_65:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s8, 6
	slli	a0, a0, 2
	ld	a1, 48(sp)                      # 8-byte Folded Reload
	add	s0, a1, a0
	addi	s4, s0, 192
	sd	s4, 128(sp)                     # 8-byte Folded Spill
	addi	s3, s0, 128
	addi	s5, s0, 64
	slliw	a0, s8, 7
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	li	s9, 1024
	#APP
	.insn r 11, 0, 1, zero, s10, s0

	#NO_APP
	ld	a1, 40(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s6, s5

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s6, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s6, s4

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s6, s7

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 2
	add	s4, a1, a0
	#APP
	.insn r 11, 0, 1, zero, s4, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s10, s0

	#NO_APP
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s6, s5

	#NO_APP
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s6, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s6, a0

	#NO_APP
	ld	a5, 56(sp)                      # 8-byte Folded Reload
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s6, s7

	#NO_APP
	addi	a0, s4, 256
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s8, s8, 16
	blt	s8, a5, .LBB0_65
# %bb.66:
	lui	a0, 2
	ld	a4, 40(sp)                      # 8-byte Folded Reload
	add	a0, a0, a4
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	ld	s11, 24(sp)                     # 8-byte Folded Reload
	bge	s11, a5, .LBB0_75
# %bb.67:
	lui	a0, 8
	add	a1, a4, a0
	sd	a1, 112(sp)                     # 8-byte Folded Spill
	lui	a1, 10
	add	a2, a4, a1
	sd	a2, 104(sp)                     # 8-byte Folded Spill
	lui	a2, 12
	add	a3, a4, a2
	sd	a3, 96(sp)                      # 8-byte Folded Spill
	lui	a3, 14
	add	s1, a4, a3
	sd	s1, 88(sp)                      # 8-byte Folded Spill
	addiw	a0, a0, 256
	add	a0, a0, a4
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addiw	a0, a1, 256
	add	a0, a0, a4
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	addiw	a0, a2, 256
	add	a0, a0, a4
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	addiw	a0, a3, 256
	add	a0, a0, a4
	sd	a0, 32(sp)                      # 8-byte Folded Spill
.LBB0_101:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.21)
	addi	s1, s1, %pcrel_lo(.LBB0_101)
	lui	s2, 1
.LBB0_102:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_a.19)
	addi	s10, s10, %pcrel_lo(.LBB0_102)
.LBB0_103:                              # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_b.20)
	addi	s7, s7, %pcrel_lo(.LBB0_103)
	addi	s8, s10, 1024
	add	s9, s7, s2
.LBB0_68:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s11, 6
	slli	a0, a0, 2
	ld	a1, 48(sp)                      # 8-byte Folded Reload
	add	s0, a1, a0
	addi	s4, s0, 192
	addi	s6, s0, 128
	sd	s6, 128(sp)                     # 8-byte Folded Spill
	addi	s5, s0, 64
	slliw	a0, s11, 7
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	li	s3, 1024
	#APP
	.insn r 11, 0, 1, zero, s10, s0

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s5

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s6

	#NO_APP
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s4

	#NO_APP
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s9

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 2
	ld	a1, 120(sp)                     # 8-byte Folded Reload
	add	s6, a1, a0
	#APP
	.insn r 11, 0, 1, zero, s6, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s10, s0

	#NO_APP
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s5

	#NO_APP
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s10, a0

	#NO_APP
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s4

	#NO_APP
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s9

	#NO_APP
	addi	a0, s6, 256
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s11, s11, 16
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	blt	s11, a0, .LBB0_68
# %bb.69:
	ld	a4, 40(sp)                      # 8-byte Folded Reload
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 5, a4, a4, a0

	#NO_APP
	ld	a1, 24(sp)                      # 8-byte Folded Reload
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	bge	a1, a0, .LBB0_76
# %bb.70:
	lui	a0, 16
	add	a0, a0, a4
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a0, 17
	add	a0, a0, a4
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	addi	a0, a4, 64
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	slliw	s5, a0, 9
.LBB0_104:                              # Label of block must be emitted
	auipc	s8, %pcrel_hi(run_model.sram_c.24)
	addi	s8, s8, %pcrel_lo(.LBB0_104)
	lui	s6, 1
.LBB0_105:                              # Label of block must be emitted
	auipc	s3, %pcrel_hi(run_model.sram_a.22)
	addi	s3, s3, %pcrel_lo(.LBB0_105)
	li	s7, 1024
.LBB0_106:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_b.23)
	addi	s1, s1, %pcrel_lo(.LBB0_106)
	li	s9, 1
	li	s11, 96
.LBB0_71:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_72 Depth 2
	slli	a0, s5, 2
	ld	a2, 120(sp)                     # 8-byte Folded Reload
	add	s2, a2, a0
	mv	s4, a1
	slliw	a0, a1, 7
	slli	a0, a0, 2
	add	s10, a4, a0
	lui	a2, 1
	mv	a0, s8
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s3, s10

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s1, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	li	a2, 0
	li	a0, -16
	ld	a1, 128(sp)                     # 8-byte Folded Reload
.LBB0_72:                               #   Parent Loop BB0_71 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	slli	a5, a2, 32
	subw	a2, s9, a2
	slli	a4, a2, 32
	srli	a3, a4, 22
	add	a3, a3, s3
	#APP
	.insn r 11, 0, 1, zero, a3, s2

	#NO_APP
	srli	a4, a4, 20
	add	a4, a4, s1
	#APP
	.insn r 11, 0, 1, zero, a4, a1

	#NO_APP
	srli	s0, a5, 22
	add	s0, s0, s3
	srli	a5, a5, 20
	add	a5, a5, s1
	#APP
	.insn r 11, 0, 2, s8, s0, a5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, a0, 16
	add	a1, a1, s6
	addi	s2, s2, 64
	bltu	a0, s11, .LBB0_72
# %bb.73:                               #   in Loop: Header=BB0_71 Depth=1
	#APP
	.insn r 11, 0, 2, s8, a3, a4

	#NO_APP
	mv	a1, s4
	slliw	a0, s4, 6
	slli	a0, a0, 2
	ld	a4, 40(sp)                      # 8-byte Folded Reload
	add	a0, a0, a4
	#APP
	.insn r 11, 0, 1, zero, a0, s8

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a1, s4, 16
	addiw	a0, s5, 2047
	addiw	s5, a0, 1
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	blt	a1, a0, .LBB0_71
	j	.LBB0_76
.LBB0_74:
	lui	a0, 2
	add	a0, a0, a4
	sd	a0, 120(sp)                     # 8-byte Folded Spill
.LBB0_75:
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 5, a4, a4, a0

	#NO_APP
.LBB0_76:
	li	a0, 1024
	lui	a1, 24
.LBB0_77:                               # =>This Inner Loop Header: Depth=1
	add	a2, a4, a1
	flw	ft0, 0(a2)
	flw	ft1, 0(a4)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a4)
	addi	a0, a0, -1
	addi	a4, a4, 4
	bnez	a0, .LBB0_77
# %bb.78:
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
	addi	sp, sp, 272
	ret
.LBB0_79:
	ld	s11, 48(sp)                     # 8-byte Folded Reload
	j	.LBB0_14
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
	.comm	run_model.sram_a.7,4096,4
	.type	run_model.sram_b.8,@object      # @run_model.sram_b.8
	.local	run_model.sram_b.8
	.comm	run_model.sram_b.8,4096,4
	.type	run_model.sram_c.9,@object      # @run_model.sram_c.9
	.local	run_model.sram_c.9
	.comm	run_model.sram_c.9,1024,4
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
	.type	run_model.sram_a.19,@object     # @run_model.sram_a.19
	.local	run_model.sram_a.19
	.comm	run_model.sram_a.19,2048,4
	.type	run_model.sram_b.20,@object     # @run_model.sram_b.20
	.local	run_model.sram_b.20
	.comm	run_model.sram_b.20,8192,4
	.type	run_model.sram_c.21,@object     # @run_model.sram_c.21
	.local	run_model.sram_c.21
	.comm	run_model.sram_c.21,4096,4
	.type	run_model.sram_a.22,@object     # @run_model.sram_a.22
	.local	run_model.sram_a.22
	.comm	run_model.sram_a.22,2048,4
	.type	run_model.sram_b.23,@object     # @run_model.sram_b.23
	.local	run_model.sram_b.23
	.comm	run_model.sram_b.23,8192,4
	.type	run_model.sram_c.24,@object     # @run_model.sram_c.24
	.local	run_model.sram_c.24
	.comm	run_model.sram_c.24,4096,4
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
	.addrsig_sym run_model.sram_a.19
	.addrsig_sym run_model.sram_b.20
	.addrsig_sym run_model.sram_c.21
	.addrsig_sym run_model.sram_a.22
	.addrsig_sym run_model.sram_b.23
	.addrsig_sym run_model.sram_c.24
