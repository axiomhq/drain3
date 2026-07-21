#include "textflag.h"

// func spaceBitmapBlocksAVX2(p *byte, blocks int, bm *uint64)
// One uint64 bitmap word per 64 input bytes: bit i set iff byte i == ' '.
TEXT ·spaceBitmapBlocksAVX2(SB), NOSPLIT, $0-24
	MOVQ	p+0(FP), AX
	MOVQ	blocks+8(FP), CX
	MOVQ	bm+16(FP), DX
	MOVQ	$0x2020202020202020, BX
	MOVQ	BX, X0
	VPBROADCASTQ	X0, Y0
loop:
	VMOVDQU	(AX), Y1
	VMOVDQU	32(AX), Y2
	VPCMPEQB	Y0, Y1, Y1
	VPCMPEQB	Y0, Y2, Y2
	VPMOVMSKB	Y1, BX
	VPMOVMSKB	Y2, SI
	SHLQ	$32, SI
	ORQ	SI, BX
	MOVQ	BX, (DX)
	ADDQ	$64, AX
	ADDQ	$8, DX
	DECQ	CX
	JNZ	loop
	VZEROUPPER
	RET
