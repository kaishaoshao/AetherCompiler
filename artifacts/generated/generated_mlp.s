	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.file	"-"
	.globl	run_model                       # -- Begin function run_model
	.p2align	1
	.type	run_model,@function
run_model:                              # @run_model
# %bb.0:
	addi	sp, sp, -192
	sd	ra, 184(sp)                     # 8-byte Folded Spill
	sd	s0, 176(sp)                     # 8-byte Folded Spill
	sd	s1, 168(sp)                     # 8-byte Folded Spill
	sd	s2, 160(sp)                     # 8-byte Folded Spill
	sd	s3, 152(sp)                     # 8-byte Folded Spill
	sd	s4, 144(sp)                     # 8-byte Folded Spill
	sd	s5, 136(sp)                     # 8-byte Folded Spill
	sd	s6, 128(sp)                     # 8-byte Folded Spill
	sd	s7, 120(sp)                     # 8-byte Folded Spill
	sd	s8, 112(sp)                     # 8-byte Folded Spill
	sd	s9, 104(sp)                     # 8-byte Folded Spill
	sd	s10, 96(sp)                     # 8-byte Folded Spill
	sd	s11, 88(sp)                     # 8-byte Folded Spill
	slliw	s5, a1, 3
	li	a3, 24
	mv	s8, a0
	mv	a0, s5
	blt	s5, a3, .LBB0_2
# %bb.1:
	li	a0, 24
.LBB0_2:
	addiw	a3, a0, 8
	bge	s5, a3, .LBB0_38
# %bb.3:
	lui	a0, 8
	add	a0, a0, s8
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	lui	a0, 1
	addiw	a0, a0, -2048
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	lui	a0, 4
	add	a4, s8, a0
	sd	a4, 24(sp)                      # 8-byte Folded Spill
	sd	a1, 0(sp)                       # 8-byte Folded Spill
	slliw	a1, a1, 9
	addiw	a0, a0, 32
	add	a0, a0, s8
	sd	a0, 40(sp)                      # 8-byte Folded Spill
.LBB0_41:                               # Label of block must be emitted
	auipc	s2, %pcrel_hi(run_model.sram_c)
	addi	s2, s2, %pcrel_lo(.LBB0_41)
.LBB0_42:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(run_model.sram_bias)
	addi	a0, a0, %pcrel_lo(.LBB0_42)
	sd	a0, 32(sp)                      # 8-byte Folded Spill
.LBB0_43:                               # Label of block must be emitted
	auipc	a0, %pcrel_hi(run_model.sram_a)
	addi	a0, a0, %pcrel_lo(.LBB0_43)
	sd	a0, 16(sp)                      # 8-byte Folded Spill
	li	s11, 32
.LBB0_44:                               # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_b)
	addi	s7, s7, %pcrel_lo(.LBB0_44)
	li	s6, 55
	li	s9, 56
	li	s4, 49
	sd	s5, 8(sp)                       # 8-byte Folded Spill
	sd	a3, 56(sp)                      # 8-byte Folded Spill
	j	.LBB0_6
.LBB0_4:                                #   in Loop: Header=BB0_6 Depth=1
	slli	a0, a4, 32
	srli	a1, a0, 22
.LBB0_45:                               #   in Loop: Header=BB0_6 Depth=1
                                        # Label of block must be emitted
	auipc	a2, %pcrel_hi(run_model.sram_a)
	addi	a2, a2, %pcrel_lo(.LBB0_45)
	add	a1, a1, a2
	srli	a0, a0, 21
	add	a0, a0, s7
	#APP
	.insn r 11, 0, 2, s2, a1, a0

	#NO_APP
.LBB0_5:                                #   in Loop: Header=BB0_6 Depth=1
	#APP
	.insn r 11, 0, 3, s2, t4, t0

	#NO_APP
	slli	a0, a7, 2
	add	a0, a0, s8
	#APP
	.insn r 11, 0, 1, zero, a0, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	addw	a1, a1, a0
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	mv	s5, a0
	ld	a3, 56(sp)                      # 8-byte Folded Reload
	bge	a0, a3, .LBB0_20
.LBB0_6:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_11 Depth 2
                                        #     Child Loop BB0_14 Depth 2
                                        #       Child Loop BB0_19 Depth 3
	addi	a0, s5, 32
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	blt	a3, a0, .LBB0_8
# %bb.7:                                #   in Loop: Header=BB0_6 Depth=1
	li	s3, 32
	j	.LBB0_9
.LBB0_8:                                #   in Loop: Header=BB0_6 Depth=1
	subw	s3, a3, s5
.LBB0_9:                                #   in Loop: Header=BB0_6 Depth=1
	sd	a1, 72(sp)                      # 8-byte Folded Spill
	slli	s0, a1, 2
	slli	a0, s3, 32
	srli	s10, a0, 32
	lui	a2, 2
	mv	a0, s2
	li	a1, 0
	call	memset@plt
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	ld	t4, 32(sp)                      # 8-byte Folded Reload
	li	a1, 256
	#APP
	.insn r 11, 0, 1, zero, t4, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	blez	s3, .LBB0_12
# %bb.10:                               #   in Loop: Header=BB0_6 Depth=1
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	add	a0, a0, s0
	mv	a1, s10
	ld	a2, 16(sp)                      # 8-byte Folded Reload
.LBB0_11:                               #   Parent Loop BB0_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a2, a0

	#NO_APP
	addi	a0, a0, 256
	addi	a1, a1, -1
	addi	a2, a2, 32
	bnez	a1, .LBB0_11
.LBB0_12:                               #   in Loop: Header=BB0_6 Depth=1
	li	a5, 0
	li	a4, 0
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	add	a1, a0, s0
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, s8

	#NO_APP
	slliw	a7, s5, 6
	slliw	t0, s3, 6
	slli	a6, s3, 8
	li	t5, 1
	j	.LBB0_14
.LBB0_13:                               #   in Loop: Header=BB0_14 Depth=2
	addi	a5, s1, 8
	srli	a0, s0, 32
	slli	a0, a0, 11
	add	a0, a0, s7
	slli	a2, a5, 8
	add	a2, a2, s8
	slli	a3, t3, 8
	#APP
	.insn r 11, 0, 1, zero, a0, a2

	#NO_APP
	slli	a0, a4, 32
	srli	a2, a0, 22
	add	a2, a2, t1
	srli	a0, a0, 21
	add	a0, a0, s7
	#APP
	.insn r 11, 0, 2, s2, a2, a0

	#NO_APP
	addi	a1, a1, 32
	mv	a4, t2
	bgeu	s1, s9, .LBB0_5
.LBB0_14:                               #   Parent Loop BB0_6 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_19 Depth 3
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	bltu	s6, a5, .LBB0_4
# %bb.15:                               #   in Loop: Header=BB0_14 Depth=2
	mv	s1, a5
	li	t3, 8
	bltu	a5, s4, .LBB0_17
# %bb.16:                               #   in Loop: Header=BB0_14 Depth=2
	subw	t3, s9, s1
.LBB0_17:                               #   in Loop: Header=BB0_14 Depth=2
	subw	t2, t5, a4
	slli	s0, t2, 32
.LBB0_46:                               #   in Loop: Header=BB0_14 Depth=2
                                        # Label of block must be emitted
	auipc	t1, %pcrel_hi(run_model.sram_a)
	addi	t1, t1, %pcrel_lo(.LBB0_46)
	blez	s3, .LBB0_13
# %bb.18:                               #   in Loop: Header=BB0_14 Depth=2
	srli	a0, s0, 22
	add	a5, t1, a0
	slli	a0, t3, 32
	srli	a2, a0, 30
	mv	a0, s10
	mv	a3, a1
.LBB0_19:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_14 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	#APP
	.insn r 11, 0, 1, zero, a5, a3

	#NO_APP
	addi	a3, a3, 256
	addi	a0, a0, -1
	addi	a5, a5, 32
	bnez	a0, .LBB0_19
	j	.LBB0_13
.LBB0_20:
	ld	a2, 8(sp)                       # 8-byte Folded Reload
	ld	a1, 0(sp)                       # 8-byte Folded Reload
	bge	a2, a3, .LBB0_38
# %bb.21:
	lui	a0, 6
	add	s6, s8, a0
	slliw	a1, a1, 9
	addi	a0, s8, 64
	sd	a0, 48(sp)                      # 8-byte Folded Spill
.LBB0_47:                               # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_c.3)
	addi	s9, s9, %pcrel_lo(.LBB0_47)
	lui	a0, 1
.LBB0_48:                               # Label of block must be emitted
	auipc	a4, %pcrel_hi(run_model.sram_a.1)
	addi	a4, a4, %pcrel_lo(.LBB0_48)
	sd	a4, 32(sp)                      # 8-byte Folded Spill
	li	s4, 64
.LBB0_49:                               # Label of block must be emitted
	auipc	s11, %pcrel_hi(run_model.sram_b.2)
	addi	s11, s11, %pcrel_lo(.LBB0_49)
	addiw	a0, a0, -2048
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	li	s2, 47
	li	s7, 48
	li	s10, 33
	j	.LBB0_24
.LBB0_22:                               #   in Loop: Header=BB0_24 Depth=1
	slli	a0, a3, 32
	srli	a0, a0, 21
.LBB0_50:                               #   in Loop: Header=BB0_24 Depth=1
                                        # Label of block must be emitted
	auipc	a1, %pcrel_hi(run_model.sram_a.1)
	addi	a1, a1, %pcrel_lo(.LBB0_50)
	add	a1, a1, a0
	add	a0, a0, s11
	#APP
	.insn r 11, 0, 2, s9, a1, a0

	#NO_APP
.LBB0_23:                               #   in Loop: Header=BB0_24 Depth=1
	slli	a0, a7, 2
	add	a0, a0, s8
	#APP
	.insn r 11, 0, 1, zero, a0, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	addw	a1, a1, a0
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	mv	a2, a0
	ld	a3, 56(sp)                      # 8-byte Folded Reload
	bge	a0, a3, .LBB0_38
.LBB0_24:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_29 Depth 2
                                        #     Child Loop BB0_32 Depth 2
                                        #       Child Loop BB0_37 Depth 3
	addi	a0, a2, 32
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	blt	a3, a0, .LBB0_26
# %bb.25:                               #   in Loop: Header=BB0_24 Depth=1
	mv	s0, a2
	li	s3, 32
	j	.LBB0_27
.LBB0_26:                               #   in Loop: Header=BB0_24 Depth=1
	mv	s0, a2
	subw	s3, a3, a2
.LBB0_27:                               #   in Loop: Header=BB0_24 Depth=1
	sd	a1, 72(sp)                      # 8-byte Folded Spill
	slli	s1, a1, 2
	slli	a0, s3, 32
	srli	s5, a0, 32
	lui	a2, 1
	mv	a0, s9
	li	a1, 0
	call	memset@plt
	blez	s3, .LBB0_30
# %bb.28:                               #   in Loop: Header=BB0_24 Depth=1
	add	a0, s8, s1
	mv	a1, s5
	ld	a2, 32(sp)                      # 8-byte Folded Reload
.LBB0_29:                               #   Parent Loop BB0_24 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#APP
	.insn r 11, 0, 1, zero, a2, a0

	#NO_APP
	addi	a0, a0, 256
	addi	a1, a1, -1
	addi	a2, a2, 64
	bnez	a1, .LBB0_29
.LBB0_30:                               #   in Loop: Header=BB0_24 Depth=1
	li	a4, 0
	li	a3, 0
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, s6

	#NO_APP
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	add	a0, a0, s1
	slliw	a7, s0, 5
	slli	a6, s3, 7
	li	t3, 1
	j	.LBB0_32
.LBB0_31:                               #   in Loop: Header=BB0_32 Depth=2
	addi	a4, s1, 16
	srli	a1, s0, 32
	slli	a1, a1, 11
	add	a1, a1, s11
	slli	a2, a4, 7
	add	a2, a2, s6
	slli	a5, t2, 7
	#APP
	.insn r 11, 0, 1, zero, a1, a2

	#NO_APP
	slli	a1, a3, 32
	srli	a1, a1, 21
	add	a2, t0, a1
	add	a1, a1, s11
	#APP
	.insn r 11, 0, 2, s9, a2, a1

	#NO_APP
	addi	a0, a0, 64
	mv	a3, t1
	bgeu	s1, s7, .LBB0_23
.LBB0_32:                               #   Parent Loop BB0_24 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_37 Depth 3
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	bltu	s2, a4, .LBB0_22
# %bb.33:                               #   in Loop: Header=BB0_32 Depth=2
	mv	s1, a4
	li	t2, 16
	bltu	a4, s10, .LBB0_35
# %bb.34:                               #   in Loop: Header=BB0_32 Depth=2
	subw	t2, s7, s1
.LBB0_35:                               #   in Loop: Header=BB0_32 Depth=2
	subw	t1, t3, a3
	slli	s0, t1, 32
.LBB0_51:                               #   in Loop: Header=BB0_32 Depth=2
                                        # Label of block must be emitted
	auipc	t0, %pcrel_hi(run_model.sram_a.1)
	addi	t0, t0, %pcrel_lo(.LBB0_51)
	blez	s3, .LBB0_31
# %bb.36:                               #   in Loop: Header=BB0_32 Depth=2
	srli	a1, s0, 21
	add	a4, t0, a1
	slli	a1, t2, 32
	srli	a1, a1, 30
	mv	a5, s5
	mv	a2, a0
.LBB0_37:                               #   Parent Loop BB0_24 Depth=1
                                        #     Parent Loop BB0_32 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	#APP
	.insn r 11, 0, 1, zero, a4, a2

	#NO_APP
	addi	a2, a2, 256
	addi	a5, a5, -1
	addi	a4, a4, 64
	bnez	a5, .LBB0_37
	j	.LBB0_31
.LBB0_38:
	li	a0, 1024
	lui	a1, 8
.LBB0_39:                               # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(s8)
	add	a2, s8, a1
	flw	ft1, 256(a2)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(s8)
	addi	a0, a0, -1
	addi	s8, s8, 4
	bnez	a0, .LBB0_39
# %bb.40:
	ld	ra, 184(sp)                     # 8-byte Folded Reload
	ld	s0, 176(sp)                     # 8-byte Folded Reload
	ld	s1, 168(sp)                     # 8-byte Folded Reload
	ld	s2, 160(sp)                     # 8-byte Folded Reload
	ld	s3, 152(sp)                     # 8-byte Folded Reload
	ld	s4, 144(sp)                     # 8-byte Folded Reload
	ld	s5, 136(sp)                     # 8-byte Folded Reload
	ld	s6, 128(sp)                     # 8-byte Folded Reload
	ld	s7, 120(sp)                     # 8-byte Folded Reload
	ld	s8, 112(sp)                     # 8-byte Folded Reload
	ld	s9, 104(sp)                     # 8-byte Folded Reload
	ld	s10, 96(sp)                     # 8-byte Folded Reload
	ld	s11, 88(sp)                     # 8-byte Folded Reload
	addi	sp, sp, 192
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
