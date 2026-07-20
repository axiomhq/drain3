#include "textflag.h"

DATA spaceScanWeights<>+0(SB)/8, $0x8040201008040201
DATA spaceScanWeights<>+8(SB)/8, $0x8040201008040201
GLOBL spaceScanWeights<>(SB), RODATA|NOPTR, $16

// func spaceBitmapBlocks(p *byte, blocks int, bm *uint64)
// One uint64 bitmap word per 64 input bytes: bit i set iff byte i == ' '.
// Per 64 bytes: four 16-byte compares against 0x20 (0xFF/0x00 per lane),
// AND with the per-lane bit weights {1,2,4,8,16,32,64,128}x2, then three
// pairwise adds fold the 64 weighted lanes into the 8 bytes of V0.D[0]:
// byte k holds bits 8k..8k+7, so V0.D[0] read little-endian is the word.
TEXT ·spaceBitmapBlocks(SB), NOSPLIT, $0-24
	MOVD	p+0(FP), R0
	MOVD	blocks+8(FP), R1
	MOVD	bm+16(FP), R2
	MOVD	$spaceScanWeights<>(SB), R3
	VLD1	(R3), [V5.B16]
	MOVD	$0x20, R5
	VDUP	R5, V4.B16
loop:
	VLD1.P	64(R0), [V0.B16, V1.B16, V2.B16, V3.B16]
	VCMEQ	V4.B16, V0.B16, V0.B16
	VCMEQ	V4.B16, V1.B16, V1.B16
	VCMEQ	V4.B16, V2.B16, V2.B16
	VCMEQ	V4.B16, V3.B16, V3.B16
	VAND	V5.B16, V0.B16, V0.B16
	VAND	V5.B16, V1.B16, V1.B16
	VAND	V5.B16, V2.B16, V2.B16
	VAND	V5.B16, V3.B16, V3.B16
	VADDP	V1.B16, V0.B16, V0.B16
	VADDP	V3.B16, V2.B16, V2.B16
	VADDP	V2.B16, V0.B16, V0.B16
	VADDP	V0.B16, V0.B16, V0.B16
	VMOV	V0.D[0], R4
	MOVD.P	R4, 8(R2)
	SUBS	$1, R1, R1
	BNE	loop
	RET
