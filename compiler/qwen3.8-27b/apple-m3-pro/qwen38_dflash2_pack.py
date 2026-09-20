#!/usr/bin/env python3
"""Pack z-lab/Qwen3.8-27B-DFlash2 into the Apple-M3 runtime image.

The official checkpoint is 81 BF16 tensors. Large draft linears are encoded
with the runtime's Q4G64 affine layout (32 packed bytes + fp16 scale/bias per
64 weights). The small dynamic-conv/selector linears and all vectors remain
FP16. Token embedding and LM head are shared with the target and are not copied.

Usage:
  python tools/qwen38_dflash2_pack.py \
      --draft-dir ../Qwen3.8-27B-DFlash2 \
      --out ../models/qwen38-runtime-q4q8/dflash2.q38df2
"""
from __future__ import annotations
import argparse, hashlib, json, mmap, os, struct, sys
from pathlib import Path
import numpy as np

MAGIC=b"Q38DF2\0\0"; VERSION=1
H=5120; I=17408; V=248320; L=5; GS=64
TARGET_LAYERS=[5,19,33,47,61]

# Must match qwen38_m3_dflash2_image.h. We build the header with ctypes-like
# explicit little-endian packing and patch descriptors after payload write.
Q4_FMT="<QQQQII"       # 40 bytes
F16M_FMT="<QQII"       # 24 bytes
VEC_FMT="<QQII"        # 24 bytes
Q4_SZ=struct.calcsize(Q4_FMT); F16M_SZ=struct.calcsize(F16M_FMT); VEC_SZ=struct.calcsize(VEC_FMT)
LAYER_SZ = VEC_SZ+VEC_SZ+F16M_SZ + Q4_SZ*3 + VEC_SZ*2 + Q4_SZ + VEC_SZ+VEC_SZ+F16M_SZ + Q4_SZ*3
PREFIX_FMT="<8sIIQ" + "I"*12 + "I"*5 + "ff" + "I"*6 + "4x"
PREFIX_SZ=struct.calcsize(PREFIX_FMT)
HEADER_SZ = PREFIX_SZ + Q4_SZ + VEC_SZ + L*LAYER_SZ + VEC_SZ + F16M_SZ*3 + 64

class SafeTensorFile:
    def __init__(self,path:Path):
        self.f=open(path,"rb"); self.mm=mmap.mmap(self.f.fileno(),0,access=mmap.ACCESS_READ)
        n=struct.unpack_from("<Q",self.mm,0)[0]
        self.header=json.loads(self.mm[8:8+n]); self.data0=8+n
    def close(self): self.mm.close(); self.f.close()
    def keys(self): return [k for k in self.header if k!="__metadata__"]
    def raw(self,name):
        h=self.header[name]; a,b=h["data_offsets"]
        return h, memoryview(self.mm)[self.data0+a:self.data0+b]
    def bf16(self,name):
        h,raw=self.raw(name)
        if h["dtype"]!="BF16": raise ValueError(f"{name}: expected BF16, got {h['dtype']}")
        shape=tuple(h["shape"])
        u=np.frombuffer(raw,dtype="<u2").reshape(shape)
        return u

def bf16_to_f32(u):
    # Exact BF16 expansion.
    return (u.astype(np.uint32)<<16).view(np.float32)

def align(f,n=4096):
    p=f.tell(); q=(p+n-1)//n*n
    if q>p: f.write(b"\0"*(q-p))

def pad_section(f, start):
    logical=f.tell()-start
    align(f,4096)
    return f.tell()-start

def fp16_payload(f,u16_bf):
    align(f); off=f.tell()
    flat=u16_bf.reshape(-1)
    for s in range(0,flat.size,1<<20):
        x=bf16_to_f32(flat[s:s+(1<<20)]).astype("<f2")
        f.write(x.tobytes())
    n=pad_section(f,off)
    return (off,n)

def q4_payload(f,u16_bf,name):
    if u16_bf.ndim!=2 or u16_bf.shape[1]%GS: raise ValueError(f"{name}: bad Q4 shape {u16_bf.shape}")
    rows,cols=u16_bf.shape; groups=cols//GS
    align(f); qoff=f.tell(); meta_chunks=[]
    for r0 in range(0,rows,16):
        x=bf16_to_f32(u16_bf[r0:r0+16]).reshape(-1,groups,GS)
        lo=x.min(axis=2); hi=x.max(axis=2)
        scale=(hi-lo)/15.0
        scale=np.where(scale>0,scale,1.0).astype(np.float32)
        q=np.rint((x-lo[...,None])/scale[...,None]).clip(0,15).astype(np.uint8)
        packed=(q[...,0::2] | (q[...,1::2]<<4)).reshape(-1)
        f.write(packed.tobytes())
        m=np.empty(lo.shape+(2,),dtype="<f2")
        m[...,0]=scale.astype(np.float16); m[...,1]=lo.astype(np.float16)
        meta_chunks.append(m.tobytes())
    qbytes=pad_section(f,qoff)
    moff=f.tell()
    for b in meta_chunks: f.write(b)
    mbytes=pad_section(f,moff)
    return (qoff,qbytes,moff,mbytes,rows,cols)

def shape(sf,name,expected):
    h,_=sf.raw(name)
    got=tuple(h["shape"])
    if got!=tuple(expected): raise ValueError(f"{name}: {got}, expected {expected}")
    return sf.bf16(name)

def pack_q4(f,sf,name,expected): return q4_payload(f,shape(sf,name,expected),name)
def pack_vec(f,sf,name,expected):
    a=shape(sf,name,expected); off,n=fp16_payload(f,a); return (off,n,a.size,0)
def pack_f16m(f,sf,name,expected):
    a=shape(sf,name,expected); off,n=fp16_payload(f,a); return (off,n,expected[0],expected[1])

def cfg_check(cfg):
    d=cfg.get("dflash_config",{})
    expected={"hidden_size":H,"intermediate_size":I,"vocab_size":V,"num_hidden_layers":5,
              "num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,
              "sliding_window":2048,"is_causal":False}
    for k,v in expected.items():
        if cfg.get(k)!=v: raise ValueError(f"config {k}={cfg.get(k)!r}, expected {v!r}")
    checks={"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":248070,
            "selector_rank":256,"selector_top_k":16,"target_layer_ids":TARGET_LAYERS}
    for k,v in checks.items():
        if d.get(k)!=v: raise ValueError(f"dflash_config {k}={d.get(k)!r}, expected {v!r}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--draft-dir",required=True); ap.add_argument("--out",required=True)
    args=ap.parse_args(); root=Path(args.draft_dir); out=Path(args.out)
    cfg=json.loads((root/"config.json").read_text()); cfg_check(cfg)
    st=root/"model.safetensors"
    if not st.exists(): raise SystemExit(f"missing {st}")
    digest=hashlib.sha256()
    with open(st,"rb") as sfh:
        for chunk in iter(lambda: sfh.read(8<<20), b""):
            digest.update(chunk)
    sha=digest.hexdigest().encode("ascii")
    sf=SafeTensorFile(st)
    try:
        # Exact source inventory is a useful guard against accidentally packing a different architecture.
        expected_keys={"fc.weight","hidden_norm.weight","norm.weight",
                       "candidate_selector.hidden_projection.weight",
                       "candidate_selector.predecessor_codebook","candidate_selector.successor_codebook"}
        for n in range(5):
            p=f"layers.{n}."
            expected_keys |= {p+q for q in [
                "input_layernorm.weight","attention_conv.base_kernel","attention_conv.kernel_projection.weight",
                "self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight",
                "self_attn.q_norm.weight","self_attn.k_norm.weight","self_attn.o_proj.weight",
                "post_attention_layernorm.weight","mlp_conv.base_kernel","mlp_conv.kernel_projection.weight",
                "mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight"]}
        got=set(sf.keys())
        if got!=expected_keys:
            missing=sorted(expected_keys-got); extra=sorted(got-expected_keys)
            raise ValueError(f"unexpected DFlash2 tensor inventory; missing={missing[:8]} extra={extra[:8]}")
        out.parent.mkdir(parents=True,exist_ok=True)
        with open(out,"wb+") as f:
            f.write(b"\0"*HEADER_SZ)
            fc=pack_q4(f,sf,"fc.weight",(5120,25600))
            context_norm=pack_vec(f,sf,"hidden_norm.weight",(5120,))
            layers=[]
            for n in range(5):
                p=f"layers.{n}."
                desc=[]
                desc.append(pack_vec(f,sf,p+"input_layernorm.weight",(5120,)))
                desc.append(pack_vec(f,sf,p+"attention_conv.base_kernel",(2,2,5120)))
                desc.append(pack_f16m(f,sf,p+"attention_conv.kernel_projection.weight",(1280,5120)))
                desc.append(pack_q4(f,sf,p+"self_attn.q_proj.weight",(4096,5120)))
                desc.append(pack_q4(f,sf,p+"self_attn.k_proj.weight",(1024,5120)))
                desc.append(pack_q4(f,sf,p+"self_attn.v_proj.weight",(1024,5120)))
                desc.append(pack_vec(f,sf,p+"self_attn.q_norm.weight",(128,)))
                desc.append(pack_vec(f,sf,p+"self_attn.k_norm.weight",(128,)))
                desc.append(pack_q4(f,sf,p+"self_attn.o_proj.weight",(5120,4096)))
                desc.append(pack_vec(f,sf,p+"post_attention_layernorm.weight",(5120,)))
                desc.append(pack_vec(f,sf,p+"mlp_conv.base_kernel",(2,2,5120)))
                desc.append(pack_f16m(f,sf,p+"mlp_conv.kernel_projection.weight",(1280,5120)))
                desc.append(pack_q4(f,sf,p+"mlp.gate_proj.weight",(17408,5120)))
                desc.append(pack_q4(f,sf,p+"mlp.up_proj.weight",(17408,5120)))
                desc.append(pack_q4(f,sf,p+"mlp.down_proj.weight",(5120,17408)))
                layers.append(desc)
                print(f"packed DFlash2 layer {n}",flush=True)
            final_norm=pack_vec(f,sf,"norm.weight",(5120,))
            selector_proj=pack_f16m(f,sf,"candidate_selector.hidden_projection.weight",(256,5120))
            pred=pack_f16m(f,sf,"candidate_selector.predecessor_codebook",(V,256))
            succ=pack_f16m(f,sf,"candidate_selector.successor_codebook",(V,256))
            file_bytes=f.tell()
            prefix=struct.pack(PREFIX_FMT,MAGIC,VERSION,HEADER_SZ,file_bytes,
                H,I,V,L,32,8,128,2048,248070,256,16,GS,*TARGET_LAYERS,10000000.0,1e-6,*([0]*6))
            hdr=bytearray(prefix)
            hdr+=struct.pack(Q4_FMT,*fc); hdr+=struct.pack(VEC_FMT,*context_norm)
            for d in layers:
                fmts=[VEC_FMT,VEC_FMT,F16M_FMT,Q4_FMT,Q4_FMT,Q4_FMT,VEC_FMT,VEC_FMT,Q4_FMT,VEC_FMT,VEC_FMT,F16M_FMT,Q4_FMT,Q4_FMT,Q4_FMT]
                for fmt,item in zip(fmts,d): hdr+=struct.pack(fmt,*item)
            hdr+=struct.pack(VEC_FMT,*final_norm)
            hdr+=struct.pack(F16M_FMT,*selector_proj)+struct.pack(F16M_FMT,*pred)+struct.pack(F16M_FMT,*succ)
            hdr+=sha
            if len(hdr)!=HEADER_SZ: raise AssertionError((len(hdr),HEADER_SZ))
            f.seek(0); f.write(hdr)
        print(f"wrote {out} ({file_bytes/2**30:.3f} GiB), source sha256={sha.decode()}")
    finally: sf.close()
if __name__=="__main__": main()
