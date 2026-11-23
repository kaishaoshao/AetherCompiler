	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.file	"generated_deepseek_block.c"
	.globl	run_model                       # -- Begin function run_model
	.p2align	1
	.type	run_model,@function
run_model:                              # @run_model
# %bb.0:
	addi	sp, sp, -288
	sd	ra, 280(sp)                     # 8-byte Folded Spill
	sd	s0, 272(sp)                     # 8-byte Folded Spill
	sd	s1, 264(sp)                     # 8-byte Folded Spill
	sd	s2, 256(sp)                     # 8-byte Folded Spill
	sd	s3, 248(sp)                     # 8-byte Folded Spill
	sd	s4, 240(sp)                     # 8-byte Folded Spill
	sd	s5, 232(sp)                     # 8-byte Folded Spill
	sd	s6, 224(sp)                     # 8-byte Folded Spill
	sd	s7, 216(sp)                     # 8-byte Folded Spill
	sd	s8, 208(sp)                     # 8-byte Folded Spill
	sd	s9, 200(sp)                     # 8-byte Folded Spill
	sd	s10, 192(sp)                    # 8-byte Folded Spill
	sd	s11, 184(sp)                    # 8-byte Folded Spill
	fsd	fs0, 176(sp)                    # 8-byte Folded Spill
	fsd	fs1, 168(sp)                    # 8-byte Folded Spill
	fsd	fs2, 160(sp)                    # 8-byte Folded Spill
	lui	a2, 86
	add	a3, a0, a2
	lui	a2, 88
	add	a2, a2, a0
	sd	a3, 32(sp)                      # 8-byte Folded Spill
	sd	a2, 144(sp)                     # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 4, a2, a3, a2

	#NO_APP
	sd	a1, 8(sp)                       # 8-byte Folded Spill
	slliw	a4, a1, 2
	li	a2, 12
	lui	a3, 76
	mv	a1, a4
	blt	a4, a2, .LBB0_2
# %bb.1:
	li	a1, 12
.LBB0_2:
	addiw	s6, a1, 4
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	add	s11, a0, a3
	sd	a4, 56(sp)                      # 8-byte Folded Spill
	ld	s2, 144(sp)                     # 8-byte Folded Reload
	bge	a4, s6, .LBB0_5
# %bb.3:
	lui	a0, 77
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 136(sp)                     # 8-byte Folded Spill
.LBB0_132:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_c)
	addi	s9, s9, %pcrel_lo(.LBB0_132)
	lui	s3, 1
	addiw	s10, s3, -2048
.LBB0_133:                              # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_a)
	addi	s4, s4, %pcrel_lo(.LBB0_133)
.LBB0_134:                              # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_b)
	addi	s5, s5, %pcrel_lo(.LBB0_134)
	add	s7, s4, s10
	add	s8, s5, s3
	ld	s1, 56(sp)                      # 8-byte Folded Reload
.LBB0_4:                                # =>This Inner Loop Header: Depth=1
	slliw	a0, s1, 6
	slli	a0, a0, 2
	add	s0, s2, a0
	mv	a0, s9
	li	a1, 0
	mv	a2, s10
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
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s9, s4, s5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s9, s7, s8

	#NO_APP
	slliw	a0, s1, 5
	slli	a0, a0, 2
	add	a0, a0, s11
	#APP
	.insn r 11, 0, 1, zero, a0, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s1, s1, 16
	blt	s1, s6, .LBB0_4
.LBB0_5:
	lui	a0, 90
	addiw	a0, a0, 256
	ld	a2, 72(sp)                      # 8-byte Folded Reload
	add	a0, a0, a2
	lui	a1, 77
	add	a1, a1, a2
	sd	s11, 64(sp)                     # 8-byte Folded Spill
	sd	a1, 48(sp)                      # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 4, a1, s11, a0

	#NO_APP
	lui	a0, 78
	add	s2, a2, a0
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	mv	a2, s6
	sd	s6, 152(sp)                     # 8-byte Folded Spill
	bge	a0, s6, .LBB0_14
# %bb.6:
	lui	a0, 79
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 136(sp)                     # 8-byte Folded Spill
.LBB0_135:                              # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_c.3)
	addi	s4, s4, %pcrel_lo(.LBB0_135)
	lui	s5, 1
.LBB0_136:                              # Label of block must be emitted
	auipc	s8, %pcrel_hi(run_model.sram_a.1)
	addi	s8, s8, %pcrel_lo(.LBB0_136)
	li	s7, 1024
.LBB0_137:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_b.2)
	addi	s10, s10, %pcrel_lo(.LBB0_137)
	addi	s9, s8, 1024
	add	s11, s10, s5
	ld	s0, 56(sp)                      # 8-byte Folded Reload
	ld	s6, 64(sp)                      # 8-byte Folded Reload
	mv	s3, s2
	ld	s2, 48(sp)                      # 8-byte Folded Reload
.LBB0_7:                                # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 5
	slli	a0, a0, 2
	add	s1, s2, a0
	lui	a2, 1
	mv	a0, s4
	li	a1, 0
	call	memset@plt
	ld	a2, 152(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, s1

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s1, 64
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s4, s8, s10

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s4, s9, s11

	#NO_APP
	slliw	a0, s0, 6
	slli	a0, a0, 2
	add	a0, a0, s6
	#APP
	.insn r 11, 0, 1, zero, a0, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, a2, .LBB0_7
# %bb.8:
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	mv	s2, s3
	bge	a0, a2, .LBB0_14
# %bb.9:
	mv	s6, s2
	lui	a0, 80
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a0, 81
	add	s5, a1, a0
.LBB0_138:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.6)
	addi	s1, s1, %pcrel_lo(.LBB0_138)
	lui	s7, 1
.LBB0_139:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_a.4)
	addi	s9, s9, %pcrel_lo(.LBB0_139)
	li	s8, 1024
.LBB0_140:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_b.5)
	addi	s10, s10, %pcrel_lo(.LBB0_140)
	addi	s11, s9, 1024
	add	s2, s10, s7
	ld	s4, 56(sp)                      # 8-byte Folded Reload
	ld	s0, 48(sp)                      # 8-byte Folded Reload
.LBB0_10:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s4, 5
	slli	a0, a0, 2
	add	s3, s0, a0
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	ld	a2, 152(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, s3

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s10, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s3, 64
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s2, s5

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s10

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s11, s2

	#NO_APP
	slliw	a0, s4, 6
	slli	a0, a0, 2
	add	a0, a0, s0
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s4, s4, 16
	blt	s4, a2, .LBB0_10
# %bb.11:
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	mv	s2, s6
	ld	s6, 144(sp)                     # 8-byte Folded Reload
	bge	a0, a2, .LBB0_14
# %bb.12:
	lui	a0, 87
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	s3, a1, a0
	lui	a0, 88
	addiw	a0, a0, -2048
	add	s7, a1, a0
.LBB0_141:                              # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_c.9)
	addi	s4, s4, %pcrel_lo(.LBB0_141)
.LBB0_142:                              # Label of block must be emitted
	auipc	s8, %pcrel_hi(run_model.sram_a.7)
	addi	s8, s8, %pcrel_lo(.LBB0_142)
	lui	a0, 1
	addiw	s0, a0, -2048
.LBB0_143:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_b.8)
	addi	s9, s9, %pcrel_lo(.LBB0_143)
	add	s10, s8, s0
	add	s11, s9, s0
	ld	s1, 56(sp)                      # 8-byte Folded Reload
.LBB0_13:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s1, 6
	slli	a0, a0, 2
	add	s5, s6, a0
	li	a2, 1024
	mv	a0, s4
	li	a1, 0
	call	memset@plt
	ld	a2, 152(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, s5

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s9, s3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s5, 128
	#APP
	.insn r 11, 0, 1, zero, s10, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s11, s7

	#NO_APP
	#APP
	.insn r 11, 0, 2, s4, s8, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s4, s10, s11

	#NO_APP
	slliw	a0, s1, 4
	slli	a0, a0, 2
	add	a0, a0, s2
	li	a1, 1024
	#APP
	.insn r 11, 0, 1, zero, a0, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s1, s1, 16
	blt	s1, a2, .LBB0_13
.LBB0_14:
	lui	a0, 90
	addiw	a0, a0, -2048
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	a3, a1, a0
	lui	a0, 80
	addiw	a0, a0, -2048
	add	a0, a0, a1
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	sd	a3, 112(sp)                     # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 6, a0, s2, a3

	#NO_APP
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	bge	a0, a2, .LBB0_17
# %bb.15:
	lui	a0, 82
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	s4, a1, a0
	lui	a0, 83
	add	s3, a1, a0
.LBB0_144:                              # Label of block must be emitted
	auipc	s11, %pcrel_hi(run_model.sram_c.12)
	addi	s11, s11, %pcrel_lo(.LBB0_144)
	lui	s5, 1
	addiw	s0, s5, -2048
.LBB0_145:                              # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_a.10)
	addi	s7, s7, %pcrel_lo(.LBB0_145)
.LBB0_146:                              # Label of block must be emitted
	auipc	s8, %pcrel_hi(run_model.sram_b.11)
	addi	s8, s8, %pcrel_lo(.LBB0_146)
	add	s9, s7, s0
	add	s10, s8, s5
	ld	s1, 56(sp)                      # 8-byte Folded Reload
.LBB0_16:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s1, 6
	slli	a0, a0, 2
	ld	a1, 144(sp)                     # 8-byte Folded Reload
	add	s6, a1, a0
	mv	a0, s11
	li	a1, 0
	mv	a2, s0
	call	memset@plt
	ld	a2, 152(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, s6

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s6, 128
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	#APP
	.insn r 11, 0, 2, s11, s7, s8

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s11, s9, s10

	#NO_APP
	slliw	a0, s1, 5
	slli	a0, a0, 2
	add	a0, a0, s2
	#APP
	.insn r 11, 0, 1, zero, a0, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s1, s1, 16
	blt	s1, a2, .LBB0_16
.LBB0_17:
	lui	a0, 90
	addiw	a0, a0, 384
	ld	s10, 72(sp)                     # 8-byte Folded Reload
	add	a0, a0, s10
	lui	a1, 79
	add	s5, s10, a1
	sd	s2, 128(sp)                     # 8-byte Folded Spill
	#APP
	.insn r 11, 0, 4, s5, s2, a0

	#NO_APP
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	mv	s11, a2
	ld	s4, 64(sp)                      # 8-byte Folded Reload
	bge	a0, a2, .LBB0_23
# %bb.18:
	lui	a0, 84
	add	a0, a0, s10
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	lui	a0, 85
	add	a0, a0, s10
	sd	a0, 136(sp)                     # 8-byte Folded Spill
.LBB0_147:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.15)
	addi	s1, s1, %pcrel_lo(.LBB0_147)
	lui	s8, 1
	mv	s4, s11
.LBB0_148:                              # Label of block must be emitted
	auipc	s11, %pcrel_hi(run_model.sram_a.13)
	addi	s11, s11, %pcrel_lo(.LBB0_148)
	li	s9, 1024
.LBB0_149:                              # Label of block must be emitted
	auipc	s0, %pcrel_hi(run_model.sram_b.14)
	addi	s0, s0, %pcrel_lo(.LBB0_149)
	addi	s6, s11, 1024
	add	s7, s0, s8
	ld	s10, 56(sp)                     # 8-byte Folded Reload
	ld	s3, 128(sp)                     # 8-byte Folded Reload
.LBB0_19:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s10, 5
	slli	a0, a0, 2
	add	s2, s5, a0
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s11, s2

	#NO_APP
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s0, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s2, 64
	#APP
	.insn r 11, 0, 1, zero, s6, a0

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s7, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s11, s0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s6, s7

	#NO_APP
	slliw	a0, s10, 6
	slli	a0, a0, 2
	add	a0, a0, s3
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s10, s10, 16
	blt	s10, s4, .LBB0_19
# %bb.20:
	ld	s10, 72(sp)                     # 8-byte Folded Reload
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	mv	s11, s4
	ld	s4, 64(sp)                      # 8-byte Folded Reload
	bge	a0, s11, .LBB0_23
# %bb.21:
	lui	a0, 89
	add	s3, s10, a0
.LBB0_150:                              # Label of block must be emitted
	auipc	s2, %pcrel_hi(run_model.sram_c.18)
	addi	s2, s2, %pcrel_lo(.LBB0_150)
.LBB0_151:                              # Label of block must be emitted
	auipc	s6, %pcrel_hi(run_model.sram_a.16)
	addi	s6, s6, %pcrel_lo(.LBB0_151)
	lui	a0, 1
	addiw	s7, a0, -2048
.LBB0_152:                              # Label of block must be emitted
	auipc	s8, %pcrel_hi(run_model.sram_b.17)
	addi	s8, s8, %pcrel_lo(.LBB0_152)
	ld	s1, 56(sp)                      # 8-byte Folded Reload
.LBB0_22:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s1, 5
	slli	a0, a0, 2
	add	s0, s5, a0
	slliw	s9, s1, 4
	li	a2, 1024
	mv	a0, s2
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s6, s0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s2, s6, s8

	#NO_APP
	slli	a0, s9, 2
	add	a0, a0, s5
	li	a1, 1024
	#APP
	.insn r 11, 0, 1, zero, a0, s2

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s1, s1, 16
	blt	s1, s11, .LBB0_22
.LBB0_23:
	li	a0, 0
	lui	a1, 72
	add	a1, a1, s10
	sd	a1, 24(sp)                      # 8-byte Folded Spill
	lui	a1, 5
	add	a1, a1, s10
	sd	a1, 16(sp)                      # 8-byte Folded Spill
	sd	s5, 144(sp)                     # 8-byte Folded Spill
	ld	a1, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 6, s5, s5, a1

	#NO_APP
	lui	a1, 4
	add	a1, a1, s10
	sd	a1, 40(sp)                      # 8-byte Folded Spill
	li	a1, 256
	li	a2, 16
	mv	a3, s4
	mv	s1, s4
	ld	a6, 56(sp)                      # 8-byte Folded Reload
.LBB0_24:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_25 Depth 2
	li	a5, 0
	mv	a4, s1
.LBB0_25:                               #   Parent Loop BB0_24 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	s0, a3, a5
	flw	ft0, 0(s0)
	fsw	ft0, 0(a4)
	addi	a5, a5, 4
	addi	a4, a4, 64
	bne	a5, a1, .LBB0_25
# %bb.26:                               #   in Loop: Header=BB0_24 Depth=1
	addi	a0, a0, 1
	addi	s1, s1, 4
	addi	a3, a3, 256
	bne	a0, a2, .LBB0_24
# %bb.27:
	ld	s3, 128(sp)                     # 8-byte Folded Reload
	bge	a6, s11, .LBB0_30
# %bb.28:
	lui	a0, 77
	addiw	a0, a0, -2048
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	s5, a1, a0
.LBB0_153:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_c.21)
	addi	s10, s10, %pcrel_lo(.LBB0_153)
.LBB0_154:                              # Label of block must be emitted
	auipc	s6, %pcrel_hi(run_model.sram_a.19)
	addi	s6, s6, %pcrel_lo(.LBB0_154)
	lui	a0, 1
	addiw	s2, a0, -2048
.LBB0_155:                              # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_b.20)
	addi	s7, s7, %pcrel_lo(.LBB0_155)
	add	s8, s6, s2
	add	s9, s7, s2
	ld	s1, 56(sp)                      # 8-byte Folded Reload
.LBB0_29:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s1, 6
	slli	a0, a0, 2
	add	s0, s3, a0
	li	a2, 1024
	mv	a0, s10
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s6, s0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s7, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s0, 128
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s9, s5

	#NO_APP
	#APP
	.insn r 11, 0, 2, s10, s6, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s10, s8, s9

	#NO_APP
	slliw	a0, s1, 4
	slli	a0, a0, 2
	add	a0, a0, s4
	li	a1, 1024
	#APP
	.insn r 11, 0, 1, zero, a0, s10

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s1, s1, 16
	blt	s1, s11, .LBB0_29
.LBB0_30:
	lui	a2, 80
	addiw	a0, a2, -1536
	ld	a5, 72(sp)                      # 8-byte Folded Reload
	add	a0, a0, a5
	li	a1, -64
	addiw	a2, a2, -1988
.LBB0_31:                               # =>This Inner Loop Header: Depth=1
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
	bnez	a1, .LBB0_31
# %bb.32:
	ld	s1, 56(sp)                      # 8-byte Folded Reload
	ld	s9, 120(sp)                     # 8-byte Folded Reload
	ld	s10, 144(sp)                    # 8-byte Folded Reload
	bge	s1, s11, .LBB0_35
# %bb.33:
	lui	a0, 76
	addiw	a0, a0, 1024
	add	s2, a5, a0
.LBB0_156:                              # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_c.24)
	addi	s7, s7, %pcrel_lo(.LBB0_156)
	li	s3, 1024
.LBB0_157:                              # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_a.22)
	addi	s5, s5, %pcrel_lo(.LBB0_157)
.LBB0_158:                              # Label of block must be emitted
	auipc	s6, %pcrel_hi(run_model.sram_b.23)
	addi	s6, s6, %pcrel_lo(.LBB0_158)
.LBB0_34:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s1, 4
	slli	s0, a0, 2
	add	s8, s10, s0
	li	a2, 1024
	mv	a0, s7
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s5, s8

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s6, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s7, s5, s6

	#NO_APP
	add	a0, s2, s0
	#APP
	.insn r 11, 0, 1, zero, a0, s7

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s1, s1, 16
	blt	s1, s11, .LBB0_34
.LBB0_35:
	li	a0, 256
	mv	a1, s4
.LBB0_36:                               # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a1)
	flw	ft1, 1024(a1)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a1)
	addi	a0, a0, -1
	addi	a1, a1, 4
	bnez	a0, .LBB0_36
# %bb.37:
	fmv.w.x	fs0, zero
	li	s1, 64
	mv	s0, s4
.LBB0_38:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_69 Depth 2
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	slli	a0, a0, 6
	add	s10, s4, a0
	flw	ft1, 0(s10)
	flw	ft0, 4(s10)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_40
# %bb.39:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft0, ft1
.LBB0_40:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft1, 8(s10)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_42
# %bb.41:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft1, ft0
.LBB0_42:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft0, 12(s10)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_44
# %bb.43:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft0, ft1
.LBB0_44:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft1, 16(s10)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_46
# %bb.45:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft1, ft0
.LBB0_46:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft0, 20(s10)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_48
# %bb.47:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft0, ft1
.LBB0_48:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft1, 24(s10)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_50
# %bb.49:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft1, ft0
.LBB0_50:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft0, 28(s10)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_52
# %bb.51:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft0, ft1
.LBB0_52:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft1, 32(s10)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_54
# %bb.53:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft1, ft0
.LBB0_54:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft0, 36(s10)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_56
# %bb.55:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft0, ft1
.LBB0_56:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft1, 40(s10)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_58
# %bb.57:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft1, ft0
.LBB0_58:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft0, 44(s10)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_60
# %bb.59:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft0, ft1
.LBB0_60:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft1, 48(s10)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_62
# %bb.61:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft1, ft0
.LBB0_62:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft0, 52(s10)
	flt.s	a0, ft1, ft0
	bnez	a0, .LBB0_64
# %bb.63:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft0, ft1
.LBB0_64:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft1, 56(s10)
	flt.s	a0, ft0, ft1
	bnez	a0, .LBB0_66
# %bb.65:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	ft1, ft0
.LBB0_66:                               #   in Loop: Header=BB0_38 Depth=1
	flw	fs2, 60(s10)
	flt.s	a0, ft1, fs2
	bnez	a0, .LBB0_68
# %bb.67:                               #   in Loop: Header=BB0_38 Depth=1
	fmv.s	fs2, ft1
.LBB0_68:                               #   in Loop: Header=BB0_38 Depth=1
	li	s8, 0
	addi	s3, s10, 4
	addi	s9, s10, 8
	addi	s7, s10, 12
	addi	s2, s10, 16
	addi	s6, s10, 20
	addi	s5, s10, 24
	addi	s4, s10, 28
	addi	a0, s10, 32
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addi	a0, s10, 36
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	addi	a0, s10, 40
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	addi	a0, s10, 44
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	addi	a0, s10, 48
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	addi	a0, s10, 52
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	addi	a0, s10, 56
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	addi	a0, s10, 60
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	fmv.s	fs1, fs0
.LBB0_69:                               #   Parent Loop BB0_38 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	s11, s0, s8
	flw	ft0, 0(s11)
	fsub.s	ft0, ft0, fs2
	fcvt.d.s	fa0, ft0
	call	exp@plt
	fcvt.s.d	ft0, fa0
	fsw	ft0, 0(s11)
	addi	s8, s8, 4
	fadd.s	fs1, fs1, ft0
	bne	s8, s1, .LBB0_69
# %bb.70:                               #   in Loop: Header=BB0_38 Depth=1
	flw	ft0, 0(s10)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s10)
	flw	ft0, 0(s3)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s3)
	flw	ft0, 0(s9)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s9)
	flw	ft0, 0(s7)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s7)
	flw	ft0, 0(s2)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s2)
	flw	ft0, 0(s6)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s6)
	flw	ft0, 0(s5)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s5)
	flw	ft0, 0(s4)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(s4)
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
	flw	ft0, 0(a0)
	fdiv.s	ft0, ft0, fs1
	fsw	ft0, 0(a0)
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1
	addi	s0, s0, 64
	ld	s11, 152(sp)                    # 8-byte Folded Reload
	ld	s4, 64(sp)                      # 8-byte Folded Reload
	li	a1, 16
	bne	a0, a1, .LBB0_38
# %bb.71:
	ld	s8, 56(sp)                      # 8-byte Folded Reload
	bge	s8, s11, .LBB0_78
# %bb.72:
.LBB0_159:                              # Label of block must be emitted
	auipc	s6, %pcrel_hi(run_model.sram_c.27)
	addi	s6, s6, %pcrel_lo(.LBB0_159)
	lui	s2, 1
	mv	s9, s4
.LBB0_160:                              # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_a.25)
	addi	s4, s4, %pcrel_lo(.LBB0_160)
	li	s3, 1024
.LBB0_161:                              # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_b.26)
	addi	s5, s5, %pcrel_lo(.LBB0_161)
	mv	s1, s8
	ld	s10, 48(sp)                     # 8-byte Folded Reload
.LBB0_73:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s1, 4
	slli	a0, a0, 2
	add	s0, s9, a0
	slliw	s7, s1, 6
	lui	a2, 1
	mv	a0, s6
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s4, s0

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
	addi	s1, s1, 16
	blt	s1, s11, .LBB0_73
# %bb.74:
	lui	a0, 72
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	add	s6, a4, a0
	bge	s8, s11, .LBB0_79
# %bb.75:
	lui	a0, 73
	add	a0, a0, a4
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	lui	a0, 74
	add	a0, a0, a4
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a0, 75
	add	a0, a0, a4
	sd	a0, 128(sp)                     # 8-byte Folded Spill
.LBB0_162:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.30)
	addi	s1, s1, %pcrel_lo(.LBB0_162)
	lui	s7, 1
.LBB0_163:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_a.28)
	addi	s9, s9, %pcrel_lo(.LBB0_163)
	mv	s0, s8
	li	s10, 1024
.LBB0_164:                              # Label of block must be emitted
	auipc	s3, %pcrel_hi(run_model.sram_b.29)
	addi	s3, s3, %pcrel_lo(.LBB0_164)
	addi	s2, s9, 1024
	add	s8, s3, s7
	ld	s5, 64(sp)                      # 8-byte Folded Reload
.LBB0_76:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 6
	slli	s11, a0, 2
	add	s4, s5, s11
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s9, s4

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s3, s6

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s4, 64
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s4, 128
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s3, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s2, s8

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, s4, 192
	#APP
	.insn r 11, 0, 1, zero, s2, a0

	#NO_APP
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s3

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s2, s8

	#NO_APP
	add	a0, s6, s11
	ld	s11, 152(sp)                    # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, s11, .LBB0_76
# %bb.77:
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	ld	s8, 56(sp)                      # 8-byte Folded Reload
	j	.LBB0_79
.LBB0_78:
	lui	a0, 72
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	add	s6, a4, a0
.LBB0_79:
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	ld	a2, 24(sp)                      # 8-byte Folded Reload
	li	a0, 1024
.LBB0_80:                               # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a1)
	flw	ft1, 0(a2)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a2)
	addi	a2, a2, 4
	addi	a0, a0, -1
	addi	a1, a1, 4
	bnez	a0, .LBB0_80
# %bb.81:
	lui	a0, 90
	add	a0, a0, a4
	lui	a1, 73
	add	a3, a4, a1
	#APP
	.insn r 11, 0, 4, a3, s6, a0

	#NO_APP
	lui	a0, 2
	bge	s8, s11, .LBB0_116
# %bb.82:
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
	sd	a0, 64(sp)                      # 8-byte Folded Spill
.LBB0_165:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.33)
	addi	s1, s1, %pcrel_lo(.LBB0_165)
	lui	s5, 1
.LBB0_166:                              # Label of block must be emitted
	auipc	s2, %pcrel_hi(run_model.sram_a.31)
	addi	s2, s2, %pcrel_lo(.LBB0_166)
.LBB0_167:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_b.32)
	addi	s9, s9, %pcrel_lo(.LBB0_167)
	mv	s6, s8
	addi	s10, s2, 1024
	add	s11, s9, s5
	sd	a3, 128(sp)                     # 8-byte Folded Spill
.LBB0_83:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s6, 6
	slli	a0, a0, 2
	add	s0, a3, a0
	addi	s8, s0, 192
	sd	s8, 136(sp)                     # 8-byte Folded Spill
	addi	s7, s0, 128
	addi	s3, s0, 64
	slliw	a0, s6, 7
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	li	s4, 1024
	#APP
	.insn r 11, 0, 1, zero, s2, s0

	#NO_APP
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s2, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s2, s7

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s8

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s2, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s11

	#NO_APP
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 2
	add	s8, a1, a0
	#APP
	.insn r 11, 0, 1, zero, s8, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	ld	a3, 128(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s2, s0

	#NO_APP
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s2, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s2, s7

	#NO_APP
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s10, a0

	#NO_APP
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s2, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s11

	#NO_APP
	addi	a0, s8, 256
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s6, s6, 16
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	blt	s6, a0, .LBB0_83
# %bb.84:
	lui	a0, 2
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	add	s1, a4, a0
	#APP
	.insn r 11, 0, 5, s1, a4, a4

	#NO_APP
	ld	s0, 56(sp)                      # 8-byte Folded Reload
	ld	a5, 152(sp)                     # 8-byte Folded Reload
	bge	s0, a5, .LBB0_117
# %bb.85:
	sd	s1, 32(sp)                      # 8-byte Folded Spill
	lui	a6, 8
	add	a1, a4, a6
	sd	a1, 120(sp)                     # 8-byte Folded Spill
	lui	a1, 10
	add	a2, a4, a1
	sd	a2, 112(sp)                     # 8-byte Folded Spill
	lui	a2, 12
	add	s1, a4, a2
	sd	s1, 104(sp)                     # 8-byte Folded Spill
	lui	s1, 14
	add	a0, a4, s1
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	addiw	a0, a6, 256
	add	a0, a0, a4
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	addiw	a0, a1, 256
	add	a0, a0, a4
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addiw	a0, a2, 256
	add	a0, a0, a4
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	addiw	a0, s1, 256
	add	a0, a0, a4
	sd	a0, 48(sp)                      # 8-byte Folded Spill
.LBB0_168:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.36)
	addi	s1, s1, %pcrel_lo(.LBB0_168)
	lui	s6, 1
.LBB0_169:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_a.34)
	addi	s9, s9, %pcrel_lo(.LBB0_169)
.LBB0_170:                              # Label of block must be emitted
	auipc	s11, %pcrel_hi(run_model.sram_b.35)
	addi	s11, s11, %pcrel_lo(.LBB0_170)
	addi	s8, s9, 1024
	add	s5, s11, s6
.LBB0_86:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 6
	slli	a0, a0, 2
	add	s3, a3, a0
	addi	s10, s3, 192
	sd	s10, 136(sp)                    # 8-byte Folded Spill
	addi	s7, s3, 128
	addi	s2, s3, 64
	slliw	a0, s0, 7
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	li	s4, 1024
	#APP
	.insn r 11, 0, 1, zero, s9, s3

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s2

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s5, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s9, s7

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s10

	#NO_APP
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s5, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s5

	#NO_APP
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 2
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	s10, a1, a0
	#APP
	.insn r 11, 0, 1, zero, s10, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	ld	a3, 128(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, s3

	#NO_APP
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s2

	#NO_APP
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s5, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s9, s7

	#NO_APP
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s11, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s5, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s11

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s5

	#NO_APP
	addi	a0, s10, 256
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	blt	s0, a0, .LBB0_86
# %bb.87:
	ld	s4, 56(sp)                      # 8-byte Folded Reload
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	bge	s4, a0, .LBB0_129
# %bb.88:
	lui	a0, 16
	add	a0, a0, a4
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	s6, 1
	addiw	a0, s6, -2048
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	lui	a0, 17
	add	a0, a0, a4
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	lui	a0, 2
	addiw	a0, a0, 64
	add	a0, a0, a4
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	slliw	s8, a0, 9
.LBB0_171:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_c.39)
	addi	s10, s10, %pcrel_lo(.LBB0_171)
.LBB0_172:                              # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_a.37)
	addi	s7, s7, %pcrel_lo(.LBB0_172)
	li	s9, 1024
.LBB0_173:                              # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_b.38)
	addi	s5, s5, %pcrel_lo(.LBB0_173)
	li	s11, 1
	li	s2, 96
.LBB0_89:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_90 Depth 2
	slli	a0, s8, 2
	ld	a1, 112(sp)                     # 8-byte Folded Reload
	add	s0, a1, a0
	slliw	a0, s4, 7
	slli	a0, a0, 2
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	s3, a1, a0
	lui	a2, 1
	mv	a0, s10
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s7, s3

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s5, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	li	a2, 0
	li	a0, -16
	ld	a1, 120(sp)                     # 8-byte Folded Reload
.LBB0_90:                               #   Parent Loop BB0_89 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	slli	a5, a2, 32
	subw	a2, s11, a2
	slli	a4, a2, 32
	srli	a3, a4, 22
	add	a3, a3, s7
	#APP
	.insn r 11, 0, 1, zero, a3, s0

	#NO_APP
	srli	a4, a4, 20
	add	a4, a4, s5
	#APP
	.insn r 11, 0, 1, zero, a4, a1

	#NO_APP
	srli	s1, a5, 22
	add	s1, s1, s7
	srli	a5, a5, 20
	add	a5, a5, s5
	#APP
	.insn r 11, 0, 2, s10, s1, a5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, a0, 16
	add	a1, a1, s6
	addi	s0, s0, 64
	bltu	a0, s2, .LBB0_90
# %bb.91:                               #   in Loop: Header=BB0_89 Depth=1
	#APP
	.insn r 11, 0, 2, s10, a3, a4

	#NO_APP
	slliw	a0, s4, 6
	slli	a0, a0, 2
	ld	a1, 40(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	#APP
	.insn r 11, 0, 1, zero, a0, s10

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s4, s4, 16
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	addw	s8, s8, a0
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	blt	s4, a0, .LBB0_89
# %bb.92:
	ld	s0, 56(sp)                      # 8-byte Folded Reload
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	bge	s0, a0, .LBB0_128
# %bb.93:
	lui	a0, 24
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	add	a1, a4, a0
	sd	a1, 112(sp)                     # 8-byte Folded Spill
	lui	a1, 26
	add	a2, a4, a1
	sd	a2, 104(sp)                     # 8-byte Folded Spill
	lui	a2, 28
	add	a3, a4, a2
	sd	a3, 96(sp)                      # 8-byte Folded Spill
	lui	a3, 30
	add	a5, a4, a3
	sd	a5, 88(sp)                      # 8-byte Folded Spill
	addiw	a0, a0, 256
	add	a0, a0, a4
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addiw	a0, a1, 256
	add	a0, a0, a4
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	addiw	a0, a2, 256
	add	a0, a0, a4
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	addiw	a0, a3, 256
	add	a0, a0, a4
	sd	a0, 24(sp)                      # 8-byte Folded Spill
.LBB0_174:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.42)
	addi	s1, s1, %pcrel_lo(.LBB0_174)
	lui	s6, 1
.LBB0_175:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_a.40)
	addi	s10, s10, %pcrel_lo(.LBB0_175)
.LBB0_176:                              # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_b.41)
	addi	s4, s4, %pcrel_lo(.LBB0_176)
	addi	s9, s10, 1024
	add	s8, s4, s6
.LBB0_94:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 6
	slli	a0, a0, 2
	ld	a1, 128(sp)                     # 8-byte Folded Reload
	add	s3, a1, a0
	addi	s11, s3, 192
	sd	s11, 120(sp)                    # 8-byte Folded Spill
	addi	s7, s3, 128
	addi	s5, s3, 64
	slliw	a0, s0, 7
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	li	s2, 1024
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s9, s5

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s7

	#NO_APP
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s8

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s9, s11

	#NO_APP
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s8

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 2
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	s11, a1, a0
	#APP
	.insn r 11, 0, 1, zero, s11, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s9, s5

	#NO_APP
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s7

	#NO_APP
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s8

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s9, s8

	#NO_APP
	addi	a0, s11, 256
	ld	a1, 152(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, a1, .LBB0_94
# %bb.95:
	ld	s1, 32(sp)                      # 8-byte Folded Reload
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 5, s1, a4, a4

	#NO_APP
	ld	s0, 56(sp)                      # 8-byte Folded Reload
	bge	s0, a1, .LBB0_130
# %bb.96:
	lui	a0, 32
	add	a1, a4, a0
	sd	a1, 112(sp)                     # 8-byte Folded Spill
	lui	a1, 34
	add	a2, a4, a1
	sd	a2, 104(sp)                     # 8-byte Folded Spill
	lui	a2, 36
	add	a3, a4, a2
	sd	a3, 96(sp)                      # 8-byte Folded Spill
	lui	a3, 38
	add	a5, a4, a3
	sd	a5, 88(sp)                      # 8-byte Folded Spill
	addiw	a0, a0, 256
	add	a0, a0, a4
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addiw	a0, a1, 256
	add	a0, a0, a4
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	addiw	a0, a2, 256
	add	a0, a0, a4
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	addiw	a0, a3, 256
	add	a0, a0, a4
	sd	a0, 24(sp)                      # 8-byte Folded Spill
.LBB0_177:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.45)
	addi	s1, s1, %pcrel_lo(.LBB0_177)
	lui	s6, 1
.LBB0_178:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_a.43)
	addi	s10, s10, %pcrel_lo(.LBB0_178)
.LBB0_179:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_b.44)
	addi	s9, s9, %pcrel_lo(.LBB0_179)
	addi	s8, s10, 1024
	add	s4, s9, s6
.LBB0_97:                               # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 6
	slli	a0, a0, 2
	ld	a1, 128(sp)                     # 8-byte Folded Reload
	add	s3, a1, a0
	addi	s11, s3, 192
	sd	s11, 120(sp)                    # 8-byte Folded Spill
	addi	s7, s3, 128
	addi	s2, s3, 64
	slliw	a0, s0, 7
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	li	s5, 1024
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s2

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s7

	#NO_APP
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s11

	#NO_APP
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 2
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	s11, a1, a0
	#APP
	.insn r 11, 0, 1, zero, s11, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s2

	#NO_APP
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s7

	#NO_APP
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	addi	a0, s11, 256
	ld	a1, 152(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, a1, .LBB0_97
# %bb.98:
	ld	s4, 56(sp)                      # 8-byte Folded Reload
	bge	s4, a1, .LBB0_131
# %bb.99:
	lui	a0, 40
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a0, 41
	add	a0, a0, a1
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	lui	a0, 2
	addiw	a0, a0, 64
	add	a0, a0, a1
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	slliw	s8, a0, 9
.LBB0_180:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_c.48)
	addi	s10, s10, %pcrel_lo(.LBB0_180)
	lui	s6, 1
.LBB0_181:                              # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_a.46)
	addi	s7, s7, %pcrel_lo(.LBB0_181)
	li	s9, 1024
.LBB0_182:                              # Label of block must be emitted
	auipc	s5, %pcrel_hi(run_model.sram_b.47)
	addi	s5, s5, %pcrel_lo(.LBB0_182)
	li	s11, 1
	li	s2, 96
.LBB0_100:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_101 Depth 2
	slli	a0, s8, 2
	ld	a1, 112(sp)                     # 8-byte Folded Reload
	add	s0, a1, a0
	slliw	a0, s4, 7
	slli	a0, a0, 2
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	s3, a1, a0
	lui	a2, 1
	mv	a0, s10
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s7, s3

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s5, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	li	a2, 0
	li	a0, -16
	ld	a1, 120(sp)                     # 8-byte Folded Reload
.LBB0_101:                              #   Parent Loop BB0_100 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	slli	a5, a2, 32
	subw	a2, s11, a2
	slli	a4, a2, 32
	srli	a3, a4, 22
	add	a3, a3, s7
	#APP
	.insn r 11, 0, 1, zero, a3, s0

	#NO_APP
	srli	a4, a4, 20
	add	a4, a4, s5
	#APP
	.insn r 11, 0, 1, zero, a4, a1

	#NO_APP
	srli	s1, a5, 22
	add	s1, s1, s7
	srli	a5, a5, 20
	add	a5, a5, s5
	#APP
	.insn r 11, 0, 2, s10, s1, a5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, a0, 16
	add	a1, a1, s6
	addi	s0, s0, 64
	bltu	a0, s2, .LBB0_101
# %bb.102:                              #   in Loop: Header=BB0_100 Depth=1
	#APP
	.insn r 11, 0, 2, s10, a3, a4

	#NO_APP
	slliw	a0, s4, 6
	slli	a0, a0, 2
	ld	a1, 16(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	#APP
	.insn r 11, 0, 1, zero, a0, s10

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s4, s4, 16
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	addw	s8, s8, a0
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	blt	s4, a0, .LBB0_100
# %bb.103:
	ld	s0, 56(sp)                      # 8-byte Folded Reload
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	bge	s0, a0, .LBB0_131
# %bb.104:
	lui	a0, 48
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	add	a1, a4, a0
	sd	a1, 112(sp)                     # 8-byte Folded Spill
	lui	a1, 50
	add	a2, a4, a1
	sd	a2, 104(sp)                     # 8-byte Folded Spill
	lui	a2, 52
	add	a3, a4, a2
	sd	a3, 96(sp)                      # 8-byte Folded Spill
	lui	a3, 54
	add	a5, a4, a3
	sd	a5, 88(sp)                      # 8-byte Folded Spill
	addiw	a0, a0, 256
	add	a0, a0, a4
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addiw	a0, a1, 256
	add	a0, a0, a4
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	addiw	a0, a2, 256
	add	a0, a0, a4
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	addiw	a0, a3, 256
	add	a0, a0, a4
	sd	a0, 24(sp)                      # 8-byte Folded Spill
.LBB0_183:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.51)
	addi	s1, s1, %pcrel_lo(.LBB0_183)
	lui	s6, 1
.LBB0_184:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_a.49)
	addi	s10, s10, %pcrel_lo(.LBB0_184)
.LBB0_185:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_b.50)
	addi	s9, s9, %pcrel_lo(.LBB0_185)
	addi	s8, s10, 1024
	add	s4, s9, s6
.LBB0_105:                              # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 6
	slli	a0, a0, 2
	ld	a1, 128(sp)                     # 8-byte Folded Reload
	add	s3, a1, a0
	addi	s11, s3, 192
	sd	s11, 120(sp)                    # 8-byte Folded Spill
	addi	s7, s3, 128
	addi	s5, s3, 64
	slliw	a0, s0, 7
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	li	s2, 1024
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s5

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s7

	#NO_APP
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s11

	#NO_APP
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 2
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	s11, a1, a0
	#APP
	.insn r 11, 0, 1, zero, s11, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s5

	#NO_APP
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s7

	#NO_APP
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	addi	a0, s11, 256
	ld	a1, 152(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, a1, .LBB0_105
# %bb.106:
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 5, a0, a4, a4

	#NO_APP
	ld	s0, 56(sp)                      # 8-byte Folded Reload
	bge	s0, a1, .LBB0_115
# %bb.107:
	lui	a0, 56
	add	a1, a4, a0
	sd	a1, 112(sp)                     # 8-byte Folded Spill
	lui	a1, 58
	add	a2, a4, a1
	sd	a2, 104(sp)                     # 8-byte Folded Spill
	lui	a2, 60
	add	a3, a4, a2
	sd	a3, 96(sp)                      # 8-byte Folded Spill
	lui	a3, 62
	add	a5, a4, a3
	sd	a5, 88(sp)                      # 8-byte Folded Spill
	addiw	a0, a0, 256
	add	a0, a0, a4
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addiw	a0, a1, 256
	add	a0, a0, a4
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	addiw	a0, a2, 256
	add	a0, a0, a4
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	addiw	a0, a3, 256
	add	a0, a0, a4
	sd	a0, 24(sp)                      # 8-byte Folded Spill
.LBB0_186:                              # Label of block must be emitted
	auipc	s1, %pcrel_hi(run_model.sram_c.54)
	addi	s1, s1, %pcrel_lo(.LBB0_186)
	lui	s6, 1
.LBB0_187:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_a.52)
	addi	s10, s10, %pcrel_lo(.LBB0_187)
.LBB0_188:                              # Label of block must be emitted
	auipc	s9, %pcrel_hi(run_model.sram_b.53)
	addi	s9, s9, %pcrel_lo(.LBB0_188)
	addi	s8, s10, 1024
	add	s4, s9, s6
.LBB0_108:                              # =>This Inner Loop Header: Depth=1
	slliw	a0, s0, 6
	slli	a0, a0, 2
	ld	a1, 128(sp)                     # 8-byte Folded Reload
	add	s3, a1, a0
	addi	s11, s3, 192
	sd	s11, 120(sp)                    # 8-byte Folded Spill
	addi	s7, s3, 128
	addi	s2, s3, 64
	slliw	a0, s0, 7
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	li	s5, 1024
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s2

	#NO_APP
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s7

	#NO_APP
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s11

	#NO_APP
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 2
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	s11, a1, a0
	#APP
	.insn r 11, 0, 1, zero, s11, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	lui	a2, 1
	mv	a0, s1
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s10, s3

	#NO_APP
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s8, s2

	#NO_APP
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 1, zero, s10, s7

	#NO_APP
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s9, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s8, a0

	#NO_APP
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s10, s9

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	#APP
	.insn r 11, 0, 2, s1, s8, s4

	#NO_APP
	addi	a0, s11, 256
	ld	a1, 152(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, a0, s1

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	s0, s0, 16
	blt	s0, a1, .LBB0_108
# %bb.109:
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	ld	a2, 56(sp)                      # 8-byte Folded Reload
	bge	a2, a1, .LBB0_115
# %bb.110:
	lui	a0, 64
	add	a0, a0, a4
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lui	a0, 65
	add	a0, a0, a4
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	lui	a0, 2
	addiw	a0, a0, 64
	add	a0, a0, a4
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	slliw	s6, a0, 9
.LBB0_189:                              # Label of block must be emitted
	auipc	s10, %pcrel_hi(run_model.sram_c.57)
	addi	s10, s10, %pcrel_lo(.LBB0_189)
	lui	s8, 1
.LBB0_190:                              # Label of block must be emitted
	auipc	s7, %pcrel_hi(run_model.sram_a.55)
	addi	s7, s7, %pcrel_lo(.LBB0_190)
	li	s9, 1024
.LBB0_191:                              # Label of block must be emitted
	auipc	s4, %pcrel_hi(run_model.sram_b.56)
	addi	s4, s4, %pcrel_lo(.LBB0_191)
	li	s11, 1
	li	s2, 96
.LBB0_111:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_112 Depth 2
	slli	a0, s6, 2
	ld	a1, 120(sp)                     # 8-byte Folded Reload
	add	s0, a1, a0
	mv	s5, a2
	slliw	a0, a2, 7
	slli	a0, a0, 2
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	add	s3, a1, a0
	lui	a2, 1
	mv	a0, s10
	li	a1, 0
	call	memset@plt
	#APP
	.insn r 11, 0, 1, zero, s7, s3

	#NO_APP
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	#APP
	.insn r 11, 0, 1, zero, s4, a0

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	li	a2, 0
	li	a0, -16
	ld	a1, 128(sp)                     # 8-byte Folded Reload
.LBB0_112:                              #   Parent Loop BB0_111 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	slli	a5, a2, 32
	subw	a2, s11, a2
	slli	a4, a2, 32
	srli	a3, a4, 22
	add	a3, a3, s7
	#APP
	.insn r 11, 0, 1, zero, a3, s0

	#NO_APP
	srli	a4, a4, 20
	add	a4, a4, s4
	#APP
	.insn r 11, 0, 1, zero, a4, a1

	#NO_APP
	srli	s1, a5, 22
	add	s1, s1, s7
	srli	a5, a5, 20
	add	a5, a5, s4
	#APP
	.insn r 11, 0, 2, s10, s1, a5

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a0, a0, 16
	add	a1, a1, s8
	addi	s0, s0, 64
	bltu	a0, s2, .LBB0_112
# %bb.113:                              #   in Loop: Header=BB0_111 Depth=1
	#APP
	.insn r 11, 0, 2, s10, a3, a4

	#NO_APP
	mv	a2, s5
	slliw	a0, s5, 6
	slli	a0, a0, 2
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	#APP
	.insn r 11, 0, 1, zero, a0, s10

	#NO_APP
	#APP
	.insn i 11, 1, zero, zero, 0

	#NO_APP
	addi	a2, s5, 16
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	addw	s6, s6, a0
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	blt	a2, a0, .LBB0_111
# %bb.114:
	ld	a4, 72(sp)                      # 8-byte Folded Reload
.LBB0_115:
	ld	a2, 40(sp)                      # 8-byte Folded Reload
	ld	a3, 16(sp)                      # 8-byte Folded Reload
	j	.LBB0_121
.LBB0_116:
	add	s1, a4, a0
	#APP
	.insn r 11, 0, 5, s1, a4, a4

	#NO_APP
.LBB0_117:
	lui	a0, 4
	add	a2, a4, a0
.LBB0_118:
	#APP
	.insn r 11, 0, 5, s1, a4, a4

	#NO_APP
.LBB0_119:
	lui	a0, 5
	add	a3, a4, a0
.LBB0_120:
	#APP
	.insn r 11, 0, 5, s1, a4, a4

	#NO_APP
.LBB0_121:
	li	a0, 1024
	mv	a1, a4
.LBB0_122:                              # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a3)
	flw	ft1, 0(a1)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a1)
	addi	a1, a1, 4
	addi	a0, a0, -1
	addi	a3, a3, 4
	bnez	a0, .LBB0_122
# %bb.123:
	li	a0, 1024
	mv	a1, a4
.LBB0_124:                              # =>This Inner Loop Header: Depth=1
	flw	ft0, 0(a2)
	flw	ft1, 0(a1)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a1)
	addi	a1, a1, 4
	addi	a0, a0, -1
	addi	a2, a2, 4
	bnez	a0, .LBB0_124
# %bb.125:
	li	a0, 1024
	lui	a1, 72
.LBB0_126:                              # =>This Inner Loop Header: Depth=1
	add	a2, a4, a1
	flw	ft0, 0(a2)
	flw	ft1, 0(a4)
	fadd.s	ft0, ft0, ft1
	fsw	ft0, 0(a4)
	addi	a0, a0, -1
	addi	a4, a4, 4
	bnez	a0, .LBB0_126
# %bb.127:
	ld	ra, 280(sp)                     # 8-byte Folded Reload
	ld	s0, 272(sp)                     # 8-byte Folded Reload
	ld	s1, 264(sp)                     # 8-byte Folded Reload
	ld	s2, 256(sp)                     # 8-byte Folded Reload
	ld	s3, 248(sp)                     # 8-byte Folded Reload
	ld	s4, 240(sp)                     # 8-byte Folded Reload
	ld	s5, 232(sp)                     # 8-byte Folded Reload
	ld	s6, 224(sp)                     # 8-byte Folded Reload
	ld	s7, 216(sp)                     # 8-byte Folded Reload
	ld	s8, 208(sp)                     # 8-byte Folded Reload
	ld	s9, 200(sp)                     # 8-byte Folded Reload
	ld	s10, 192(sp)                    # 8-byte Folded Reload
	ld	s11, 184(sp)                    # 8-byte Folded Reload
	fld	fs0, 176(sp)                    # 8-byte Folded Reload
	fld	fs1, 168(sp)                    # 8-byte Folded Reload
	fld	fs2, 160(sp)                    # 8-byte Folded Reload
	addi	sp, sp, 288
	ret
.LBB0_128:
	ld	a4, 72(sp)                      # 8-byte Folded Reload
.LBB0_129:
	ld	a2, 40(sp)                      # 8-byte Folded Reload
	ld	s1, 32(sp)                      # 8-byte Folded Reload
	j	.LBB0_118
.LBB0_130:
	ld	a2, 40(sp)                      # 8-byte Folded Reload
	j	.LBB0_119
.LBB0_131:
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	ld	a2, 40(sp)                      # 8-byte Folded Reload
	ld	a3, 16(sp)                      # 8-byte Folded Reload
	ld	s1, 32(sp)                      # 8-byte Folded Reload
	j	.LBB0_120
.Lfunc_end0:
	.size	run_model, .Lfunc_end0-run_model
                                        # -- End function
	.type	run_model.sram_a,@object        # @run_model.sram_a
	.local	run_model.sram_a
	.comm	run_model.sram_a,4096,4
	.type	run_model.sram_b,@object        # @run_model.sram_b
	.local	run_model.sram_b
	.comm	run_model.sram_b,8192,4
	.type	run_model.sram_c,@object        # @run_model.sram_c
	.local	run_model.sram_c
	.comm	run_model.sram_c,2048,4
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
	.comm	run_model.sram_a.10,4096,4
	.type	run_model.sram_b.11,@object     # @run_model.sram_b.11
	.local	run_model.sram_b.11
	.comm	run_model.sram_b.11,8192,4
	.type	run_model.sram_c.12,@object     # @run_model.sram_c.12
	.local	run_model.sram_c.12
	.comm	run_model.sram_c.12,2048,4
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
	.comm	run_model.sram_a.16,4096,4
	.type	run_model.sram_b.17,@object     # @run_model.sram_b.17
	.local	run_model.sram_b.17
	.comm	run_model.sram_b.17,4096,4
	.type	run_model.sram_c.18,@object     # @run_model.sram_c.18
	.local	run_model.sram_c.18
	.comm	run_model.sram_c.18,1024,4
	.type	run_model.sram_a.19,@object     # @run_model.sram_a.19
	.local	run_model.sram_a.19
	.comm	run_model.sram_a.19,4096,4
	.type	run_model.sram_b.20,@object     # @run_model.sram_b.20
	.local	run_model.sram_b.20
	.comm	run_model.sram_b.20,4096,4
	.type	run_model.sram_c.21,@object     # @run_model.sram_c.21
	.local	run_model.sram_c.21
	.comm	run_model.sram_c.21,1024,4
	.type	run_model.sram_a.22,@object     # @run_model.sram_a.22
	.local	run_model.sram_a.22
	.comm	run_model.sram_a.22,2048,4
	.type	run_model.sram_b.23,@object     # @run_model.sram_b.23
	.local	run_model.sram_b.23
	.comm	run_model.sram_b.23,2048,4
	.type	run_model.sram_c.24,@object     # @run_model.sram_c.24
	.local	run_model.sram_c.24
	.comm	run_model.sram_c.24,1024,4
	.type	run_model.sram_a.25,@object     # @run_model.sram_a.25
	.local	run_model.sram_a.25
	.comm	run_model.sram_a.25,2048,4
	.type	run_model.sram_b.26,@object     # @run_model.sram_b.26
	.local	run_model.sram_b.26
	.comm	run_model.sram_b.26,8192,4
	.type	run_model.sram_c.27,@object     # @run_model.sram_c.27
	.local	run_model.sram_c.27
	.comm	run_model.sram_c.27,4096,4
	.type	run_model.sram_a.28,@object     # @run_model.sram_a.28
	.local	run_model.sram_a.28
	.comm	run_model.sram_a.28,2048,4
	.type	run_model.sram_b.29,@object     # @run_model.sram_b.29
	.local	run_model.sram_b.29
	.comm	run_model.sram_b.29,8192,4
	.type	run_model.sram_c.30,@object     # @run_model.sram_c.30
	.local	run_model.sram_c.30
	.comm	run_model.sram_c.30,4096,4
	.type	run_model.sram_a.31,@object     # @run_model.sram_a.31
	.local	run_model.sram_a.31
	.comm	run_model.sram_a.31,2048,4
	.type	run_model.sram_b.32,@object     # @run_model.sram_b.32
	.local	run_model.sram_b.32
	.comm	run_model.sram_b.32,8192,4
	.type	run_model.sram_c.33,@object     # @run_model.sram_c.33
	.local	run_model.sram_c.33
	.comm	run_model.sram_c.33,4096,4
	.type	run_model.sram_a.34,@object     # @run_model.sram_a.34
	.local	run_model.sram_a.34
	.comm	run_model.sram_a.34,2048,4
	.type	run_model.sram_b.35,@object     # @run_model.sram_b.35
	.local	run_model.sram_b.35
	.comm	run_model.sram_b.35,8192,4
	.type	run_model.sram_c.36,@object     # @run_model.sram_c.36
	.local	run_model.sram_c.36
	.comm	run_model.sram_c.36,4096,4
	.type	run_model.sram_a.37,@object     # @run_model.sram_a.37
	.local	run_model.sram_a.37
	.comm	run_model.sram_a.37,2048,4
	.type	run_model.sram_b.38,@object     # @run_model.sram_b.38
	.local	run_model.sram_b.38
	.comm	run_model.sram_b.38,8192,4
	.type	run_model.sram_c.39,@object     # @run_model.sram_c.39
	.local	run_model.sram_c.39
	.comm	run_model.sram_c.39,4096,4
	.type	run_model.sram_a.40,@object     # @run_model.sram_a.40
	.local	run_model.sram_a.40
	.comm	run_model.sram_a.40,2048,4
	.type	run_model.sram_b.41,@object     # @run_model.sram_b.41
	.local	run_model.sram_b.41
	.comm	run_model.sram_b.41,8192,4
	.type	run_model.sram_c.42,@object     # @run_model.sram_c.42
	.local	run_model.sram_c.42
	.comm	run_model.sram_c.42,4096,4
	.type	run_model.sram_a.43,@object     # @run_model.sram_a.43
	.local	run_model.sram_a.43
	.comm	run_model.sram_a.43,2048,4
	.type	run_model.sram_b.44,@object     # @run_model.sram_b.44
	.local	run_model.sram_b.44
	.comm	run_model.sram_b.44,8192,4
	.type	run_model.sram_c.45,@object     # @run_model.sram_c.45
	.local	run_model.sram_c.45
	.comm	run_model.sram_c.45,4096,4
	.type	run_model.sram_a.46,@object     # @run_model.sram_a.46
	.local	run_model.sram_a.46
	.comm	run_model.sram_a.46,2048,4
	.type	run_model.sram_b.47,@object     # @run_model.sram_b.47
	.local	run_model.sram_b.47
	.comm	run_model.sram_b.47,8192,4
	.type	run_model.sram_c.48,@object     # @run_model.sram_c.48
	.local	run_model.sram_c.48
	.comm	run_model.sram_c.48,4096,4
	.type	run_model.sram_a.49,@object     # @run_model.sram_a.49
	.local	run_model.sram_a.49
	.comm	run_model.sram_a.49,2048,4
	.type	run_model.sram_b.50,@object     # @run_model.sram_b.50
	.local	run_model.sram_b.50
	.comm	run_model.sram_b.50,8192,4
	.type	run_model.sram_c.51,@object     # @run_model.sram_c.51
	.local	run_model.sram_c.51
	.comm	run_model.sram_c.51,4096,4
	.type	run_model.sram_a.52,@object     # @run_model.sram_a.52
	.local	run_model.sram_a.52
	.comm	run_model.sram_a.52,2048,4
	.type	run_model.sram_b.53,@object     # @run_model.sram_b.53
	.local	run_model.sram_b.53
	.comm	run_model.sram_b.53,8192,4
	.type	run_model.sram_c.54,@object     # @run_model.sram_c.54
	.local	run_model.sram_c.54
	.comm	run_model.sram_c.54,4096,4
	.type	run_model.sram_a.55,@object     # @run_model.sram_a.55
	.local	run_model.sram_a.55
	.comm	run_model.sram_a.55,2048,4
	.type	run_model.sram_b.56,@object     # @run_model.sram_b.56
	.local	run_model.sram_b.56
	.comm	run_model.sram_b.56,8192,4
	.type	run_model.sram_c.57,@object     # @run_model.sram_c.57
	.local	run_model.sram_c.57
	.comm	run_model.sram_c.57,4096,4
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
	.addrsig_sym run_model.sram_a.25
	.addrsig_sym run_model.sram_b.26
	.addrsig_sym run_model.sram_c.27
	.addrsig_sym run_model.sram_a.28
	.addrsig_sym run_model.sram_b.29
	.addrsig_sym run_model.sram_c.30
	.addrsig_sym run_model.sram_a.31
	.addrsig_sym run_model.sram_b.32
	.addrsig_sym run_model.sram_c.33
	.addrsig_sym run_model.sram_a.34
	.addrsig_sym run_model.sram_b.35
	.addrsig_sym run_model.sram_c.36
	.addrsig_sym run_model.sram_a.37
	.addrsig_sym run_model.sram_b.38
	.addrsig_sym run_model.sram_c.39
	.addrsig_sym run_model.sram_a.40
	.addrsig_sym run_model.sram_b.41
	.addrsig_sym run_model.sram_c.42
	.addrsig_sym run_model.sram_a.43
	.addrsig_sym run_model.sram_b.44
	.addrsig_sym run_model.sram_c.45
	.addrsig_sym run_model.sram_a.46
	.addrsig_sym run_model.sram_b.47
	.addrsig_sym run_model.sram_c.48
	.addrsig_sym run_model.sram_a.49
	.addrsig_sym run_model.sram_b.50
	.addrsig_sym run_model.sram_c.51
	.addrsig_sym run_model.sram_a.52
	.addrsig_sym run_model.sram_b.53
	.addrsig_sym run_model.sram_c.54
	.addrsig_sym run_model.sram_a.55
	.addrsig_sym run_model.sram_b.56
	.addrsig_sym run_model.sram_c.57
