	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.file	"generated_model.c"
	.globl	run_model                       # -- Begin function run_model
	.p2align	1
	.type	run_model,@function
run_model:                              # @run_model
# %bb.0:
	addi	sp, sp, -64
	sd	ra, 56(sp)                      # 8-byte Folded Spill
	sd	s0, 48(sp)                      # 8-byte Folded Spill
	sd	s1, 40(sp)                      # 8-byte Folded Spill
	sd	s2, 32(sp)                      # 8-byte Folded Spill
	sd	s3, 24(sp)                      # 8-byte Folded Spill
	sd	s4, 16(sp)                      # 8-byte Folded Spill
	sd	s5, 8(sp)                       # 8-byte Folded Spill
	mv	s2, a0
	lui	s4, 4
	add	s1, a0, s4
	lui	a0, 8
	add	s0, s2, a0
.LBB0_3:                                # Label of block must be emitted
	auipc	s3, %pcrel_hi(run_model.sram_c)
	addi	s3, s3, %pcrel_lo(.LBB0_3)
	lui	a2, 2
	mv	a0, s3
	li	a1, 0
	call	memset@plt
	li	a2, 0
.LBB0_4:                                # Label of block must be emitted
	auipc	a6, %pcrel_hi(run_model.sram_bias)
	addi	a6, a6, %pcrel_lo(.LBB0_4)
	li	a0, 256
	#APP
	.insn r 11, 0, 1, zero, a6, s0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
.LBB0_5:                                # Label of block must be emitted
	auipc	t2, %pcrel_hi(run_model.sram_a)
	addi	t2, t2, %pcrel_lo(.LBB0_5)
	li	a7, 1024
	#APP
	.insn r 11, 0, 1, zero, t2, s1

	#NO_APP
.LBB0_6:                                # Label of block must be emitted
	auipc	t3, %pcrel_hi(run_model.sram_b)
	addi	t3, t3, %pcrel_lo(.LBB0_6)
	lui	a0, 1
	addiw	t4, a0, -2048
	#APP
	.insn r 11, 0, 1, zero, t3, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addiw	a0, s4, 32
	add	a5, s2, a0
	add	a0, s2, t4
	li	s0, -8
	li	t0, 1
	li	t1, 48
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	slli	a3, a2, 32
	subw	a2, t0, a2
	slli	a1, a2, 32
	srli	s1, a1, 22
	add	s1, s1, t2
	#APP
	.insn r 11, 0, 1, zero, s1, a5

	#NO_APP
	srli	a1, a1, 21
	add	a1, a1, t3
	#APP
	.insn r 11, 0, 1, zero, a1, a0

	#NO_APP
	srli	a4, a3, 22
	add	a4, a4, t2
	srli	a3, a3, 21
	add	a3, a3, t3
	#APP
	.insn r 11, 0, 2, s3, a4, a3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 8
	addi	a5, a5, 32
	add	a0, a0, t4
	bltu	s0, t1, .LBB0_1
# %bb.2:
	#APP
	.insn r 11, 0, 2, s3, s1, a1

	#NO_APP
	#APP
	.insn r 11, 0, 3, s3, a6, zero

	#NO_APP
	lui	a0, 2
	#APP
	.insn r 11, 0, 1, zero, s2, s3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	lui	a0, 6
	add	s4, s2, a0
	lui	s3, 8
	addiw	a0, s3, 256
	add	s0, s2, a0
.LBB0_7:                                # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.3)
	addi	s1, s1, %pcrel_lo(.LBB0_7)
	lui	a2, 1
	lui	s5, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
.LBB0_8:                                # Label of block must be emitted
	auipc	a0, %pcrel_hi(run_model.sram_bias.4)
	addi	a0, a0, %pcrel_lo(.LBB0_8)
	li	a1, 128
	#APP
	.insn r 11, 0, 1, zero, a0, s0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
.LBB0_9:                                # Label of block must be emitted
	auipc	a1, %pcrel_hi(run_model.sram_a.1)
	addi	a1, a1, %pcrel_lo(.LBB0_9)
	addiw	a2, s5, -2048
	#APP
	.insn r 11, 0, 1, zero, a1, s2

	#NO_APP
.LBB0_10:                               # Label of block must be emitted
	auipc	a3, %pcrel_hi(run_model.sram_b.2)
	addi	a3, a3, %pcrel_lo(.LBB0_10)
	#APP
	.insn r 11, 0, 1, zero, a3, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a4, s2, 64
	add	a5, a1, a2
	#APP
	.insn r 11, 0, 1, zero, a5, a4

	#NO_APP
	lui	a6, 7
	addiw	s0, a6, -2048
	add	s0, s0, s2
	add	a4, a3, a2
	#APP
	.insn r 11, 0, 1, zero, a4, s0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, a1, a3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s2, 128
	#APP
	.insn r 11, 0, 1, zero, a1, s0

	#NO_APP
	add	s0, s2, a6
	#APP
	.insn r 11, 0, 1, zero, a3, s0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, a5, a4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s2, 192
	#APP
	.insn r 11, 0, 1, zero, a5, s0

	#NO_APP
	addiw	s0, s3, -2048
	add	s0, s0, s2
	#APP
	.insn r 11, 0, 1, zero, a4, s0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, a1, a3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, a5, a4

	#NO_APP
	#APP
	.insn r 11, 0, 3, s1, a0, zero

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s2, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	ra, 56(sp)                      # 8-byte Folded Reload
	ld	s0, 48(sp)                      # 8-byte Folded Reload
	ld	s1, 40(sp)                      # 8-byte Folded Reload
	ld	s2, 32(sp)                      # 8-byte Folded Reload
	ld	s3, 24(sp)                      # 8-byte Folded Reload
	ld	s4, 16(sp)                      # 8-byte Folded Reload
	ld	s5, 8(sp)                       # 8-byte Folded Reload
	addi	sp, sp, 64
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
	.type	run_model.sram_bias.4,@object   # @run_model.sram_bias.4
	.local	run_model.sram_bias.4
	.comm	run_model.sram_bias.4,128,4
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
	.addrsig_sym run_model.sram_bias.4
