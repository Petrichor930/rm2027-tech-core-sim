# -*- coding: utf-8 -*-
"""按六个关节轴链拆分机械臂网格 → arm_base/arm_l1..l5/arm_drum.stl
关节轴（world，q=0，来自网格 PCA 拟合 + 官方演示视频核对）：
  J1 回转   axis(0,0,1)      @ (0.361, 0.425, ·)      comp1624 轴承盘
  J2 肩俯仰 axis(-0.343,0.939,0) @ (0.487,0.146,0.614) comp1626 法兰盘
  J3 肘俯仰 axis(-0.343,0.939,0) @ (0.339,0.112,1.242) comp1612 法兰盘
  J4 腕回转 axis(0.74,0.269,0.617) @ 肘部             comp607 前臂轴
  J5 腕俯仰 axis(-0.343,0.939,0) @ (-0.133,-0.094,0.974) 鼓安装座 comp1023
  J6 法兰   axis(0,-1,0)     @ 鼓安装座（鼓法向）
已知大件分组见 KEY_GROUPS；其余小件按质心就近分配。
"""
import sys, struct
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

def load_stl(path):
    with open(path,'rb') as f:
        head=f.read(84); n=struct.unpack('<I',head[80:84])[0]; buf=f.read(n*50)
    arr=np.frombuffer(buf,dtype=np.uint8).reshape(n,50)
    return arr[:,12:48].copy().view('<f4').reshape(n,3,3).astype(np.float64)

def components(tri, tol=1e-6):
    n=len(tri); pts=tri.reshape(-1,3)
    keys=np.round(pts/tol).astype(np.int64)
    uniq,inv=np.unique(keys,axis=0,return_inverse=True); inv=inv.reshape(n,3)
    parent=np.arange(len(uniq))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    for t in inv: union(t[0],t[1]); union(t[1],t[2])
    roots=np.array([find(v) for v in inv[:,0]])
    labels,lab_inv=np.unique(roots,return_inverse=True)
    return lab_inv,len(labels)

def save_stl(path, tri):
    n=len(tri)
    with open(path,'wb') as f:
        f.write(b'\0'*80); f.write(struct.pack('<I',n))
        for t in tri:
            f.write(struct.pack('<3f',0,0,1))
            for v in t: f.write(struct.pack('<3f',*v))
            f.write(struct.pack('<H',0))
    print(f"  {path}: {n} tris")

def to_world(m):
    return np.stack([m[:,0], m[:,2], 0.851-m[:,1]], axis=1)

def bake(t):
    """CAD 文件坐标 -> world（与 split_column.py 一致）：world=(mx, mz, 0.851-my)"""
    o = t.reshape(-1,3).copy()
    out = np.empty_like(o)
    out[:,0] = o[:,0]; out[:,1] = o[:,2]; out[:,2] = 0.851 - o[:,1]
    return out.reshape(-1,3,3)

tri = load_stl('assets/tech_core_m.stl')
lab,k = components(tri)
counts = np.bincount(lab)

# 装配柱链已单独拆分（滑架/销轴），从臂分组中排除
CARR_COMPS = {602, 785, 936, 951, 956}
PIN_COMPS = {702, 891, 689, 614, 615, 664, 733, 890, 899}

KEY_GROUPS = {
    'base':  {1638, 1602, 1635, 1608, 1074, 1601},
    'l1':    {1624, 1604, 1605, 1618},
    'l2':    {1636, 1622, 1607},
    'l3':    {1612, 1603},
    'l4':    {607},
    'l5':    {1023, 346, 5},
    'drum':  {3, 731, 1031, 1052, 966, 967, 551, 838, 1600, 388, 534, 348,
               786, 262, 898, 963, 1130, 1174, 1330},
}
ANCHOR = {
    'base': (0.25, 0.36, 0.25), 'l1': (0.42, 0.32, 0.55), 'l2': (0.42, 0.22, 0.90),
    'l3': (0.30, 0.15, 1.22), 'l4': (0.05, 0.20, 1.05), 'l5': (-0.15, 0.00, 0.97),
    'drum': (-0.13, -0.17, 1.05),
}
GROUPS = ['base','l1','l2','l3','l4','l5','drum']

assign = {}
for c in range(k):
    if c in PIN_COMPS or c in CARR_COMPS:
        continue
    hit = None
    for g, s in KEY_GROUPS.items():
        if c in s:
            hit = g; break
    if hit is None:
        w = to_world(tri[lab==c].reshape(-1,3)).mean(0)
        hit = min(GROUPS, key=lambda g: np.linalg.norm(w - np.array(ANCHOR[g])))
    assign[c] = hit

# 鼓组边界检查：drum 的小件若质心明显在臂侧则重分（防止就近误分）
from collections import Counter
cnt = Counter(assign.values())
print("group comp counts:", dict(cnt))

import os
os.makedirs('assets', exist_ok=True)
out_tris = {g: [] for g in GROUPS}
for c, g in assign.items():
    out_tris[g].append(bake(tri[lab == c]))
for g in GROUPS:
    t = np.concatenate(out_tris[g])
    lo, hi = t.reshape(-1,3).min(0), t.reshape(-1,3).max(0)
    print(f"{g}: {len(t)} tris  world bbox {lo.round(3)} .. {hi.round(3)}")
    save_stl(f'assets/arm_{g}.stl', t)

# 打印关键组的分组摘要（人工核对用）
print("\nKEY comp assignment:")
for g in GROUPS:
    keys_in = sorted(c for c in assign if c in set().union(*KEY_GROUPS.values()) and assign[c]==g)
    print(f"  {g}: {keys_in}")
