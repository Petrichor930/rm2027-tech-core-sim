# -*- coding: utf-8 -*-
"""机械臂分件（M1）：manifest.csv -> 按名字分组 + 关节轴聚类 + 基座系烘焙 -> assets/robot_*.stl
基座系约定：J1 轴 = +Z 过原点，法兰中心方位朝 +Y，底面 z=0（米制）
输出: assets/robot_base.stl, robot_l1..l6.stl, robot_gripper.stl, robot_fing_a/b.stl,
      assets/robot_meta.npy（关节轴/行程/抓取中心）
运行（任意 Python3 + numpy）：python tools/split_robot.py
"""
import sys, os, csv, math
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

MANIFEST = "assets/robot_parts/manifest.csv"
OUT_ASSETS = "assets"
GROUPS = ["robot_base", "robot_l1", "robot_l2", "robot_l3", "robot_l4",
          "robot_l5", "robot_l6", "robot_gripper", "robot_fing_a", "robot_fing_b"]

# 名字分组规则（正则，按序首个匹配；无反斜杠写法；紧固件不写这里 -> 就近分配）
RULES = [
    ("robot_gripper", r"夹爪|T8丝杆|丝杆螺母|MGN5C|STP_MGN5C|打印件-MGN5C|平面推力球轴承|TB-MT1233|dm_3507"),
    ("robot_l6",      r"cnc-J6|转轴固定件|外侧连杆|镜向外侧连杆|主动小连杆|DM-J4310"),
    ("robot_l5",      r"cnc-J5主动轮|cnc-J5从动轮|cnc-J5连接件|cnc-J5电机连接件|J5同步带|碳板-5mm-末端映射"),
    ("robot_l4",      r"cnc-J4轴承座|cnc-J4输出轴|cnc-J4输出光轴|小臂|Link4"),
    ("robot_l3",      r"L2[+]J3|cnc-J3电机输出轴|cnc-J3轴承压件|cnc-J3轴承外压|碳板-10mm-J3传动连杆|Link3"),
    ("robot_l2",      r"J1-J2|J2电机座|cnc-pitch1轴承内压件|碳板-3mm-pitch1|Link2|DM 6248"),
    ("robot_l1",      r"cnc-大yaw4340连接件|大yaw主动轮|大yaw从动轮|yaw轴同步带|J1电机输出轴|dm_4340|深沟球轴承-6707|涨套|F5-10M"),
    ("robot_base",    r"机械臂主体|碳板-新大yaw固定|碳板-8mm-大yaw轴承外压|碳板-大yaw轴承压紧件|打印件-垫高柱|打印件-铝排垫块"),
]

# 手动指定 关节->圆柱簇 序号（按首次运行打印的簇表填；留空走启发式）
JOINT_CLUSTER_IDS = {}

import re
RULES_C = [(g, re.compile(p)) for g, p in RULES]


def load_manifest():
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows


def load_stl(path):
    import struct
    with open(path, 'rb') as f:
        head = f.read(84)
        n = struct.unpack('<I', head[80:84])[0]
        buf = f.read(n * 50)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, 50)
    return arr[:, 12:48].copy().view('<f4').reshape(n, 3, 3).astype(np.float64)


def save_stl(path, tri):
    import struct
    n = len(tri)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', n))
        for t in tri:
            f.write(struct.pack('<3f', 0, 0, 1))
            for v in t:
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))


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


# ---------------- 1. 读入 ----------------
rows = load_manifest()
print(f"manifest 行数: {len(rows)}")
CYLCOL = [k for k in rows[0] if k.startswith("cyls")][0]

# ---------------- 2. 名字分组 ----------------
def row_centroid(i):
    r = rows[i]
    return np.array([(float(r["bx0"]) + float(r["bx1"])) / 2,
                     (float(r["by0"]) + float(r["by1"])) / 2,
                     (float(r["bz0"]) + float(r["bz1"])) / 2])

assign = [None] * len(rows)
named = {g: [] for g in GROUPS}
for i, r in enumerate(rows):
    nm = r["product_name"]
    for g, rx in RULES_C:
        if rx.search(nm):
            assign[i] = g
            named[g].append(i)
            break

# 手指件（内侧连杆 普通+镜向、tpu 垫）按侧边二均值拆分：一指 = 一根普通杆+一根镜向杆+垫
finger_rx = re.compile(r"内侧连杆|tpu-顶端三角")
fidx = [i for i, r in enumerate(rows) if finger_rx.search(r["product_name"])]
if fidx:
    C = np.array([row_centroid(i) for i in fidx])
    ax = int(np.argmax(C.max(0) - C.min(0)))
    med = float(np.median(C[:, ax]))
    for j, i in enumerate(fidx):
        g = "robot_fing_a" if C[j, ax] <= med else "robot_fing_b"
        assign[i] = g
        named[g].append(i)
    print(f"手指件按侧边拆分: {len(fidx)} 件 (轴{ax}, 分界{med:.0f}mm)")


# 就近分配两轮（紧固件/未知件）
anchors = {}
for g in GROUPS:
    if named[g]:
        anchors[g] = np.mean([row_centroid(i) for i in named[g]], axis=0)
for _ in range(2):
    for i, r in enumerate(rows):
        if assign[i] is None:
            assign[i] = min(anchors, key=lambda g: np.linalg.norm(row_centroid(i) - anchors[g]))
            named[assign[i]].append(i)
    for g in GROUPS:
        if named[g]:
            anchors[g] = np.mean([row_centroid(i) for i in named[g]], axis=0)

print("\n=== 分组汇总（烘焙前 mm） ===")
for g in GROUPS:
    idxs = named[g]
    if not idxs:
        print(f"  {g:14s}  !! 空组")
        continue
    pts = []
    tris = 0
    for i in idxs:
        r = rows[i]
        pts += [[float(r["bx0"]), float(r["by0"]), float(r["bz0"])],
                [float(r["bx1"]), float(r["by1"]), float(r["bz1"])]]
        tris += int(r["tris"])
    pts = np.array(pts)
    print(f"  {g:14s} 件数 {len(idxs):3d}  三角 {tris:6d}  bbox {pts.min(0).round(0)} .. {pts.max(0).round(0)}")

print("\n=== 分组明细（产品名 -> 组，仅显示结构件） ===")
seen = {}
for i, r in enumerate(rows):
    seen.setdefault((r["product_name"], assign[i]), 0)
    seen[(r["product_name"], assign[i])] += 1
for (nm, g), c in sorted(seen.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    fastener = re.search(r"螺丝|螺母|垫片|衬套|铝排|杯头|薄头|打印件-垫|打印件-铝", nm)
    if not fastener:
        print(f"  {g:14s} {nm} x{c}")

# ---------------- 3. 圆柱簇 -> 关节轴 ----------------
cyls = []
for i, r in enumerate(rows):
    ctr = row_centroid(i)
    for (rad, p, d, a) in parse_cyls(r[CYLCOL]):
        if a < 20 or rad > 120:     # 太小的面/过大的外壳圆弧不算轴
            continue
        dn = d / (np.linalg.norm(d) + 1e-12)
        if dn[2] < 0 or (abs(dn[2]) < 1e-9 and dn[0] < 0):
            dn = -dn
        # 轴线代表点 = 质心在轴线上的垂足（轴线的 p 可以是线上任意远点，必须归位）
        p = p + np.dot(ctr - p, dn) * dn
        cyls.append((rad, p, dn, a))
print(f"\n圆柱面（过滤后）: {len(cyls)}")

clusters = []   # [sum_area, sum_wd, sum_wp, count, sum_r]
for rad, p, dn, a in cyls:
    hit = None
    for cl in clusters:
        dd = cl[1] / max(np.linalg.norm(cl[1]), 1e-9)
        pp = cl[2] / max(cl[0], 1e-9)   # 均值点 = 加权点和 / 面积和
        if abs(float(np.dot(dn, dd))) > 0.995:
            v = p - pp
            perp = v - np.dot(v, dd) * dd
            if np.linalg.norm(perp) < 1.5:
                hit = cl
                break
    if hit is None:
        clusters.append([a, dn * a, p * a, 1, rad * a])
    else:
        dd = hit[1] / max(np.linalg.norm(hit[1]), 1e-9)
        if float(np.dot(dn, dd)) < 0:
            dn = -dn
        hit[0] += a
        hit[1] += dn * a
        hit[2] += p * a
        hit[3] += 1
        hit[4] += rad * a

cl_data = []
for (sa, swd, swp, n, sr) in clusters:
    d = swd / max(np.linalg.norm(swd), 1e-9)
    p = swp / max(sa, 1e-9)
    rad = sr / max(sa, 1e-9)
    cl_data.append((sa, d, p, n, rad))
cl_data.sort(key=lambda c: -c[0])
print("\n=== 圆柱簇 TOP20（面积序，装配系 mm） ===")
for k, (sa, d, p, n, rad) in enumerate(cl_data[:20]):
    print(f"  簇{k:2d} area={sa:7.0f} n={n:3d} r~{rad:5.1f}  d=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})  p=({p[0]:7.1f},{p[1]:7.1f},{p[2]:7.1f})")

# ---------------- 4. 基座系烘焙 ----------------
JOINT_SCOPES = {
    "r_j1": r"大yaw|6707|涨套|J1电机",
    "r_j2": r"pitch1|J2电机座|J1-J2",
    "r_j3": r"cnc-J3|L2[+]J3",
    "r_j4": r"cnc-J4",
    "r_j5": r"cnc-J5|J5同步带",
    "r_j6": r"cnc-J6|转轴固定件",
}

def scoped_axis(pat):
    """名字圈定的零件范围内取主圆柱簇轴线（装配系 mm）-> (dir, point)"""
    rx = re.compile(pat)
    sel = []
    for i, r in enumerate(rows):
        if not rx.search(r["product_name"]):
            continue
        ctr = row_centroid(i)
        for (rad, p, d, a) in parse_cyls(r[CYLCOL]):
            if a < 10 or rad > 120:
                continue
            dn = d / (np.linalg.norm(d) + 1e-12)
            if dn[2] < 0 or (abs(dn[2]) < 1e-9 and dn[0] < 0):
                dn = -dn
            p = p + np.dot(ctr - p, dn) * dn
            sel.append((rad, p, dn, a))
    cls = []
    for rad, p, dn, a in sel:
        hit = None
        for cl in cls:
            dd = cl[1] / max(np.linalg.norm(cl[1]), 1e-9)
            pp = cl[2] / max(cl[0], 1e-9)
            if abs(float(np.dot(dn, dd))) > 0.99 and np.linalg.norm((p - pp) - np.dot(p - pp, dd) * dd) < 3.0:
                hit = cl
                break
        if hit is None:
            cls.append([a, dn * a, p * a, 1])
        else:
            hit[0] += a; hit[1] += dn * a; hit[2] += p * a; hit[3] += 1
    if not cls:
        return None, None
    sa, swd, swp, n = max(cls, key=lambda c: c[0])
    return swd / max(np.linalg.norm(swd), 1e-9), swp / max(sa, 1e-9)

# J1 轴：大yaw/6707/涨套 的主簇
a1, p1 = scoped_axis(JOINT_SCOPES["r_j1"])
if a1 is None:
    j1 = max([c for c in cl_data if abs(c[1][2]) > 0.97], key=lambda c: c[0], default=cl_data[0])
    a1, p1 = j1[1], j1[2]
if a1[2] < 0:
    a1 = -a1
z = np.array([0.0, 0.0, 1.0])
v = np.cross(a1, z)
c = float(np.dot(a1, z))
if np.linalg.norm(v) < 1e-9:
    R0 = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
else:
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R0 = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

def to_R0(p):
    return R0 @ p

# 底面 z=0：base 组经 R0 后的最低 z；J1 锚点的垂足平移到原点 (x,y 归零, z 归底面)
base_idx = named["robot_base"] or list(range(len(rows)))
bpts = []
for i in base_idx:
    r = rows[i]
    for p in ([float(r["bx0"]), float(r["by0"]), float(r["bz0"])],
              [float(r["bx1"]), float(r["by1"]), float(r["bz1"])]):
        bpts.append(to_R0(np.array(p)))
bpts = np.array(bpts)
zmin = float(bpts[:, 2].min())
p1r = to_R0(p1)
t0 = np.array([-p1r[0], -p1r[1], -zmin])

# 法兰方位朝 +Y：l6 组质心经 R0+t0 后的方位角旋转到 +Y
l6_idx = named["robot_l6"] or []
if l6_idx:
    fc = np.mean([row_centroid(i) for i in l6_idx], axis=0)
    fcr = to_R0(fc) + t0
    beta = math.atan2(fcr[0], fcr[1])
    cb, sb = math.cos(beta), math.sin(beta)
    Rz = np.array([[cb, -sb, 0], [sb, cb, 0], [0, 0, 1]])
else:
    beta = 0.0
    Rz = np.eye(3)

def bake_p(p_mm):
    return Rz @ (R0 @ np.asarray(p_mm) + t0) * 0.001

def bake_d(d_mm):
    return Rz @ (R0 @ np.asarray(d_mm))

print(f"\n=== 烘焙 ===  J1轴->+Z，底面 z=0，法兰方位 +Y（beta={math.degrees(beta):.1f}°）")

# ---------------- 5. 导出分组网格（米） ----------------
merged = {g: [] for g in GROUPS}
for i, r in enumerate(rows):
    g = assign[i]
    tri = load_stl(os.path.join("assets/robot_parts", r["stl_file"]))
    pts = tri.reshape(-1, 3) @ (Rz @ R0).T
    pts = pts @ np.eye(3)  # no-op keeps clarity
    pts = (Rz @ (R0 @ tri.reshape(-1, 3).T + t0[:, None])).T * 0.001
    merged[g].append(pts.reshape(-1, 3, 3))

meta = {"files": {}}
print("\n=== 导出（米制，基座系） ===")
for g in GROUPS:
    if not merged[g]:
        continue
    T = np.concatenate(merged[g])
    files = []
    if len(T) > 180000:   # MuJoCo 单网格上限 20 万面 -> 分块
        nchunks = int(np.ceil(len(T) / 180000.0))
        for ci, chunk in enumerate(np.array_split(T, nchunks)):
            fn = f"{g}_{ci}.stl"
            save_stl(os.path.join(OUT_ASSETS, fn), chunk)
            files.append(fn)
    else:
        save_stl(os.path.join(OUT_ASSETS, g + ".stl"), T)
        files.append(g + ".stl")
    meta["files"][g] = files
    lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
    print(f"  {g:14s} {len(T):6d} tris  bbox {lo.round(3)} .. {hi.round(3)}  -> {files}")

# ---------------- 6. 关节轴（名字圈定）+ 夹爪 meta ----------------
print("\n=== 关节轴（名字圈定主簇，基座系 m） ===")
for jname, pat in JOINT_SCOPES.items():
    d, p = scoped_axis(pat)
    if d is None:
        print(f"  {jname}: !! 该名字范围无圆柱面")
        continue
    db, pb = bake_d(d), bake_p(p)
    meta[jname] = (pb.tolist(), db.tolist())
    print(f"  {jname}: pos={np.round(pb,3).tolist()}  dir={np.round(db,3).tolist()}")

# 夹爪：行程方向 = 两指质心连线（T8 丝杆轴向交叉验证）
fa = named["robot_fing_a"]
fb = named["robot_fing_b"]
travel = None
best_a = 0
for i, r in enumerate(rows):
    if "T8丝杆" in r["product_name"]:
        for (rad, p, d, a) in parse_cyls(r[CYLCOL]):
            if a > best_a:
                best_a = a
                travel = d
if fa and fb:
    ctra = np.mean([row_centroid(i) for i in fa], axis=0)
    ctrb = np.mean([row_centroid(i) for i in fb], axis=0)
    line = ctrb - ctra
    nline = np.linalg.norm(line)
    if nline > 1:
        t2 = line / nline
        if travel is None or abs(float(np.dot(t2, travel / (np.linalg.norm(travel) + 1e-9)))) < 0.5:
            travel = t2   # 丝杆轴与两指连线不一致时以连线为准
if travel is None:
    travel = np.array([1.0, 0.0, 0.0])
travel_b = bake_d(travel)
travel_b /= np.linalg.norm(travel_b)

if fa and fb:
    def corners(idxs):
        cs = []
        for i in idxs:
            r = rows[i]
            lo = np.array([float(r["bx0"]), float(r["by0"]), float(r["bz0"])])
            hi = np.array([float(r["bx1"]), float(r["by1"]), float(r["bz1"])])
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    for z in (lo[2], hi[2]):
                        cs.append([x, y, z])
        return np.array(cs)
    ca, cb = corners(fa), corners(fb)
    pa, pb_ = ca @ travel, cb @ travel
    gap0 = float(min(pb_) - max(pa))
    if gap0 < 0:
        gap0 = float(min(pa) - max(pb_))
    mid = (ca.mean(0) + cb.mean(0)) / 2
    grasp = bake_p(mid)
    meta["travel"] = travel_b.tolist()
    meta["gap0_mm"] = round(gap0, 1)
    meta["grasp_pos"] = grasp.tolist()
    meta["stroke_mm"] = 40.0
    print(f"\n=== 夹爪 ===  行程方向={np.round(travel_b,3).tolist()}  指距gap0={gap0:.1f}mm  抓取中心={np.round(grasp,3).tolist()}")
else:
    print("!! 未找到手指组")

# 臂展
l6c = np.mean([row_centroid(i) for i in named["robot_l6"]], axis=0) if named["robot_l6"] else None
if l6c is not None:
    span_reach = np.linalg.norm((Rz @ (R0 @ l6c + t0))[:2]) * 0.001
    meta["reach_m"] = round(span_reach, 3)
    print(f"臂展(J1轴到l6质心水平距离) ≈ {span_reach:.3f} m")

np.save(os.path.join(OUT_ASSETS, "robot_meta.npy"), meta, allow_pickle=True)
print(f"\n已写 {OUT_ASSETS}/robot_*.stl 与 robot_meta.npy")
