# -*- coding: utf-8 -*-
"""地形生成：assets/terrain_parts/manifest.csv -> models/terrain.xml
- 视觉 = 地形网格合并（含位姿变换 TERRAIN_POS）
- 碰撞 = 大地面 + 二级台阶盒体（参数实测，打印核对后可调）
运行：python tools/gen_terrain_xml.py
"""
import sys, os, csv, struct
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

MANIFEST = "assets/terrain_parts/manifest.csv"
OUT = "models/terrain.xml"

# ---- 场地 -> 场景 的对齐（一处可调）----
# TERRAIN_POS z 使 装配区科技核心安装底座(manifest z=-1241) 落在场景 z=0
# （与动态科技核心底座法兰 z≈0 对齐；x/y 使装配区围绕科技核心 E≈(-0.129,-0.333,0.793)）
TERRAIN_POS = (-0.25, -1.62, 1.241)
# ---- 二级台阶参数（规则手册图 4-27 A-A 剖面 + 网格实测）----
# A-A 标高（相对当地地面 G）：200/300/350/360/400；科技核心安装台 = 400 级。
# 场景坐标：G=-0.400（高地地面，车行平面）→ 一级台阶 -0.200 → 二级台阶/科技核心台 0.000
# 台阶踢面 y 位置 = 车行进线(x≈0)上网格实测的立面过渡带
G_Z = -0.400            # 驾驶地面（A-A 的 G，manifest -1641）
STEP1_H, STEP1_Y = 0.20, -1.87   # 一级台阶：200mm，踢面 y
STEP2_H, STEP2_Y = 0.20, -1.27   # 二级台阶：200mm，踢面 y（台顶=科技核心安装台 z=0）
STEP_W = 4.0            # 台阶宽（覆盖车道走廊）

def load_stl(path):
    with open(path, 'rb') as f:
        head = f.read(84)
        n = struct.unpack('<I', head[80:84])[0]
        buf = f.read(n * 50)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, 50)
    return arr[:, 12:48].copy().view('<f4').reshape(n, 3, 3).astype(np.float64)

def save_stl(path, tri):
    n = len(tri)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', n))
        for t in tri:
            f.write(struct.pack('<3f', 0, 0, 1))
            for v in t:
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))

rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
print(f"地形 manifest 行数: {len(rows)}")
# ---- 测量打印：按包围盒高度排序，识别台阶/平台 ----
print("\n=== 尺寸较大的件（按 z 跨度，mm，装配系） ===")
big = sorted(rows, key=lambda r: -(float(r["bz1"]) - float(r["bz0"])))
for r in big[:18]:
    lo = np.array([float(r["bx0"]), float(r["by0"]), float(r["bz0"])])
    hi = np.array([float(r["bx1"]), float(r["by1"]), float(r["bz1"])])
    print(f"  {r['product_name'][:22]:22s}  bbox {np.round(lo,0)} .. {np.round(hi,0)}  跨度 {np.round(hi-lo,0)}")

# ---- 合并视觉网格（mm -> m，平移到场景）----
# 清理悬空件：整体浮在空中的部件（能量机关转臂组件 z>1.4m 等）剔除；
# 两侧无人机停机坪（z0≈1.04 板 + 支柱）是真件保留
FLOAT_Z = 1.00
keep = [r for r in rows if float(r["bz0"]) * 0.001 + TERRAIN_POS[2] <= FLOAT_Z]
print(f"悬空件清理: {len(rows)} -> {len(keep)} 件（剔除 {len(rows)-len(keep)} 件 z0>{FLOAT_Z}）")
tri = np.concatenate([load_stl(os.path.join("assets/terrain_parts", r["stl_file"])) for r in keep])
pts = tri.reshape(-1, 3) * 0.001 + np.array(TERRAIN_POS)
tri = pts.reshape(-1, 3, 3)
# 视觉网格抽稀（10mm 顶点聚类）：地形是背景，原 130 万面会让老显卡每帧数秒
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decimate_stl import decimate_tris
n_before = len(tri)
tri = decimate_tris(tri, 0.010)
print(f"视觉网格抽稀: {n_before} -> {len(tri)} tris (cell 10mm)")
if len(tri) > 180000:
    for ci, chunk in enumerate(np.array_split(tri, int(np.ceil(len(tri)/180000.0)))):
        save_stl(f"assets/terrain_vis_{ci}.stl", chunk)
    vis_files = [f"terrain_vis_{ci}.stl" for ci in range(int(np.ceil(len(tri)/180000.0)))]
else:
    save_stl("assets/terrain_vis.stl", tri)
    vis_files = ["terrain_vis.stl"]
lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
print(f"\n视觉网格: {len(tri)} tris  bbox {lo.round(2)} .. {hi.round(2)} (m)")

# ---- 生成 terrain.xml ----
L = ['<mujoco model="terrain">', '  <!-- 场地高地地形（中央高地+二级台阶）：视觉=实测网格，碰撞=盒体 -->']
L.append('  <asset>')
for i, fn in enumerate(vis_files):
    L.append(f'    <mesh name="tv{i}" file="{fn}"/>')
L.append('    <material name="ground_m" rgba="0.62 0.65 0.62 1"/>')
L.append('    <material name="step_m" rgba="0.72 0.68 0.6 1"/>')
L.append('  </asset>')
L.append('  <worldbody>')
for i in range(len(vis_files)):
    L.append(f'    <geom type="mesh" mesh="tv{i}" material="ground_m" contype="0" conaffinity="0"/>')
# 驾驶大平面（A-A 的 G，车在其上行驶）
L.append(f'    <geom name="field_floor" type="box" size="6 6 0.2" pos="0 0 {G_Z-0.2}" contype="1" conaffinity="1" friction="1 0.005 0.0002" solref="0.005 1" rgba="0 0 0 0"/>')
# 二级台阶盒体（图 4-27 A-A：级高 200mm×2；踢面 y 实测于车行进线）
z1 = G_Z + STEP1_H
L.append(f'    <geom name="step1" type="box" size="{STEP_W/2} {(STEP2_Y-STEP1_Y)/2} {STEP1_H/2}" pos="0 {(STEP1_Y+STEP2_Y)/2} {G_Z+STEP1_H/2}" contype="1" conaffinity="1" friction="1 0.005 0.0002" solref="0.005 1" rgba="0 0 0 0"/>')
L.append(f'    <geom name="step2" type="box" size="{STEP_W/2} {1.385} {STEP2_H/2}" pos="0 {STEP2_Y+1.385} {z1+STEP2_H/2}" contype="1" conaffinity="1" friction="1 0.005 0.0002" solref="0.005 1" rgba="0 0 0 0"/>')
print(f"\n碰撞: G={G_Z}  一级台阶顶={z1:.3f}(y>={STEP1_Y})  二级台阶顶={z1+STEP2_H:.3f}(y>={STEP2_Y})")
L.append('  </worldbody>')
L.append('</mujoco>')
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(L) + "\n")
print("已生成", OUT)
