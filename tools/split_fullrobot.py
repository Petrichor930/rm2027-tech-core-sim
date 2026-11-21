# -*- coding: utf-8 -*-
"""整车分件（复用 split_robot 的机器）：full_parts/manifest.csv -> fb_*.stl + robot_full_meta.npy
坐标约定（chassis 系，米）：轮轴 = ±X，车头 = +Y，轮底 z=0
分组：fb_chassis(车体+装饰) / fb_wheel_m1..4(麦轮) / fb_wheel_f / fb_wheel_r(小轮) /
      fb_lift_f / fb_lift_r(升降滑架)。机械臂不在此拆（复用 robot_*.stl）。
运行：python tools/split_fullrobot.py
"""
import sys, os, csv, math, re
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

MANIFEST = "assets/full_parts/manifest.csv"
OUT_ASSETS = "assets"

# 名字分组（正则，按序首匹配；未匹配 -> fb_chassis 默认组，宁简勿繁）
RULES = [
    ("fb_lift_f",   r"前小轮.*(齿条|齿轮|滑块|限位|电机|压紧|固定|垫高|法兰)|6mm-碳齿条$|6mm-碳齿条-伸出|CNC-齿轮30齿|CNC-前小轮|4mm碳-前小轮|5mm碳板-前小轮|5.3mm碳板-前小轮|2mm碳-前小轮|2mm碳板-前小轮|7mm碳板-后小轮固定|5mm碳板-后小轮连接|PLA-前小轮|PLA-滑块限位2"),
    ("fb_lift_r",   r"后小轮.*(齿条|齿轮|滑块|限位|电机|压紧|固定|垫高|法兰)|6mm-碳齿条2|CNC-后小轮|5mm碳板-后小轮|7mm碳板-后小轮|PLA-后小轮|PLA-滑块限位$"),
    ("fb_wheel_f",  r"CNC-轮毂|前导轮|前小轮$|前防撞|防撞轮"),
    ("fb_wheel_r",  r"后小轮$|后防撞|后轮固定板2"),
    ("fb_wheel_m",  r"XROLL麦轮|麦轮|聚氨酯包胶|轮毂|主动轮.*压紧|轴承.*压紧|从动轮"),
]
# 升降滑架通用词补充进 lift 组
RULES += [
    ("fb_lift_f",   r"滑梯形|滑块限位|MGN9|MGW12|直线轴承|齿条"),
]
GROUPS = ["fb_chassis", "fb_lift_f", "fb_lift_r", "fb_wheel_f", "fb_wheel_r",
          "fb_wheel_m1", "fb_wheel_m2", "fb_wheel_m3", "fb_wheel_m4"]
RULES_C = [(g, re.compile(p)) for g, p in RULES]


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_centroid(i, rows):
    r = rows[i]
    return np.array([(float(r["bx0"]) + float(r["bx1"])) / 2,
                     (float(r["by0"]) + float(r["by1"])) / 2,
                     (float(r["bz0"]) + float(r["bz1"])) / 2])


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


import struct

def mesh_volume(tri):
    """网格有向体积（m^3，输入米制三角）"""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    return float(np.abs(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0))


def parse_cyls(s):
    out = []
    if not s:
        return out
    for item in s.split(";"):
        v = item.split(",")
        if len(v) >= 7:
            out.append((float(v[0]), np.array(list(map(float, v[1:4]))),
                        np.array(list(map(float, v[4:7]))), float(v[7]) if len(v) > 7 else 1.0))
    return out


rows = load_manifest()
print("manifest 行数:", len(rows))
CYLCOL = [k for k in rows[0] if k.startswith("cyls")][0]

# ---------------- 分组（装配系：Y=上、Z=前、X=轮轴；宁简勿繁） ----------------
assign = [None] * len(rows)

# 1) 麦轮：XROLL麦轮 行按 (x, z) 象限拆 4 个（前/后 = z）
midx = [i for i, r in enumerate(rows) if r["product_name"] == "XROLL麦轮"]
if midx:
    C = np.array([row_centroid(i, rows) for i in midx])
    mx, mz = float(np.median(C[:, 0])), float(np.median(C[:, 2]))
    for i in midx:
        c = row_centroid(i, rows)
        q = (0 if c[0] <= mx else 1) + (0 if c[2] <= mz else 2)
        assign[i] = f"fb_wheel_m{q+1}"

# 2) 其余按名字分：轮毂/包胶 -> 小轮；齿条/齿轮/滑块/导轨/避震 -> 升降滑架；前后按 z 符号
zx = np.array([row_centroid(i, rows)[2] for i in range(len(rows))])
zmid = float(np.median(zx))
def side(i):
    return "f" if row_centroid(i, rows)[2] > zmid else "r"
for i, r in enumerate(rows):
    if assign[i] is not None:
        continue
    nm = r["product_name"]
    if re.search(r"轮毂|包胶|导轮", nm):
        assign[i] = f"fb_wheel_{side(i)}"
    elif re.search(r"齿条|齿轮|滑块|限位|滑梯形|MGN|MGW|直线轴承|避震|KJ90", nm):
        assign[i] = f"fb_lift_{side(i)}"

# 3) 其余 -> 车体（云台/发射/矿仓/框架/紧固件，静态简化）
for i in range(len(rows)):
    if assign[i] is None:
        assign[i] = "fb_chassis"

named = {g: [] for g in GROUPS}
for i, g in enumerate(assign):
    if g in named:
        named[g].append(i)

print(chr(10) + "=== 分组汇总（装配系 mm） ===")
for g in GROUPS:
    idxs = named[g]
    if not idxs:
        print(f"  {g:14s} !! 空组")
        continue
    pts = np.array([row_centroid(i, rows) for i in idxs])
    tris = sum(int(rows[i]["tris"]) for i in idxs)
    print(f"  {g:14s} 件数 {len(idxs):4d}  三角 {tris:7d}  质心 {pts.mean(0).round(0)}")

# ---------------- 烘焙：装配系(Y=上,Z=前,X=轮轴) -> world(Z=上,Y=前,X=轮轴) ----------------
# 旋转：world_x=-asm_x, world_y=asm_z, world_z=asm_y（行列式 +1 的换轴）
R0 = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
# 前后定向：前小轮类零件（名字含"前"）质心 asm_z 应为正 -> 否则绕 world Z 翻 180
front = [i for i, r in enumerate(rows) if "前" in r["product_name"] and assign[i] != "fb_chassis"]
if front:
    fz = float(np.mean([R0 @ row_centroid(i, rows) for i in front], axis=0)[1])
    if fz < 0:
        R0 = np.diag([-1.0, -1.0, 1.0]) @ R0
# 平移：轮底 z=0，质心 xy 归零
allpts = []
for i in range(len(rows)):
    r = rows[i]
    for p in ([float(r["bx0"]), float(r["by0"]), float(r["bz0"])],
              [float(r["bx1"]), float(r["by1"]), float(r["bz1"])]):
        allpts.append(R0 @ np.array(p))
allpts = np.array(allpts)
t0 = np.array([-allpts[:, 0].mean(), -allpts[:, 1].mean(), -allpts[:, 2].min()])
print("烘焙: 装配系Y上Z前 -> world，t0 =", t0.round(3).tolist())

# ---------------- 导出（米制 + 质量） ----------------
meta = {"files": {}, "mass": {}}
merged = {g: [] for g in GROUPS}
for i, r in enumerate(rows):
    g = assign[i]
    if g not in merged:
        continue
    tri = load_stl(os.path.join("assets/full_parts", r["stl_file"]))
    pts = (R0 @ tri.reshape(-1, 3).T + t0[:, None]).T * 0.001
    merged[g].append(pts.reshape(-1, 3, 3))

print("\n=== 导出（米制，chassis 系） ===")
DENS = {"fb_chassis": 1200.0}   # kg/m^3 综合估算（铝/碳/钢混合）
for g in GROUPS:
    if not merged[g]:
        continue
    T = np.concatenate(merged[g])
    files = []
    if len(T) > 180000:
        nchunks = int(np.ceil(len(T) / 180000.0))
        for ci, chunk in enumerate(np.array_split(T, nchunks)):
            fn = f"{g}_{ci}.stl"
            save_stl(os.path.join(OUT_ASSETS, fn), chunk)
            files.append(fn)
    else:
        save_stl(os.path.join(OUT_ASSETS, g + ".stl"), T)
        files.append(g + ".stl")
    meta["files"][g] = files
    vol = mesh_volume(T)
    mass = vol * DENS.get(g, 2000.0)
    meta["mass"][g] = round(mass, 2)
    lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
    print(f"  {g:14s} {len(T):7d} tris  vol={vol*1e3:.2f}L  mass≈{mass:.1f}kg  bbox {lo.round(2)} .. {hi.round(2)}")

# 轮子/齿条 meta
def group_extent(g):
    if not named[g]:
        return None
    pts = []
    for i in named[g]:
        r = rows[i]
        pts += [[float(r["bx0"]), float(r["by0"]), float(r["bz0"])],
                [float(r["bx1"]), float(r["by1"]), float(r["bz1"])]]
    pts = (R0 @ np.array(pts).T + t0[:, None]).T * 0.001
    return pts.min(0), pts.max(0)

wheel_r = 0.05
for g in ["fb_wheel_m1", "fb_wheel_m2", "fb_wheel_m3", "fb_wheel_m4", "fb_wheel_f", "fb_wheel_r"]:
    ext = group_extent(g)
    if ext:
        lo, hi = ext
        r_est = float(max(hi[1] - lo[1], hi[2] - lo[2])) / 2
        ctr = ((lo + hi) / 2).tolist()
        meta[g + "_ctr"] = [round(c, 4) for c in ctr]
        meta[g + "_r"] = round(r_est, 4)
        if g.startswith("fb_wheel_m"):
            wheel_r = max(wheel_r, r_est)
        print(f"  {g}: 中心 {np.round(ctr,3).tolist()}  r≈{r_est:.3f}")

for g in ["fb_lift_f", "fb_lift_r"]:
    ext = group_extent(g)
    if ext:
        lo, hi = ext
        stroke = float(np.max(hi - lo))
        meta[g + "_stroke"] = round(min(stroke, 0.2), 3)
        print(f"  {g}: 行程估值 {min(stroke,0.2)*1000:.0f}mm")

meta["wheel_r"] = round(wheel_r, 4)
meta["wheel_axis"] = [1.0, 0.0, 0.0]
np.save(os.path.join(OUT_ASSETS, "robot_full_meta.npy"), meta, allow_pickle=True)
print("\n已写", OUT_ASSETS, "/fb_*.stl 与 robot_full_meta.npy")
