"""拆分科技核心网格：主体 / 滑架(含面板) / 销轴(装配柱)。
组件 ID 来自 stl_components.py + col_ident.py 的分析（网格坐标系 = MJCF body 坐标系）。
world = (mx, mz, 0.851-my)，销轴沿 world -Y，轴线过 world (x=-0.1295, z=0.793)。
"""
import sys, struct
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

def load_stl(path):
    with open(path, 'rb') as f:
        head = f.read(84)
        n = struct.unpack('<I', head[80:84])[0]
        buf = f.read(n * 50)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, 50)
    return arr[:, 12:48].copy().view('<f4').reshape(n, 3, 3).astype(np.float64)

def components(tri, tol=1e-6):
    n = len(tri)
    pts = tri.reshape(-1, 3)
    keys = np.round(pts / tol).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(n, 3)
    parent = np.arange(len(uniq))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for t in inv:
        union(t[0], t[1]); union(t[1], t[2])
    roots = np.array([find(v) for v in inv[:, 0]])
    labels, lab_inv = np.unique(roots, return_inverse=True)
    return lab_inv, len(labels)

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
    print(f"  {path}: {n} tris")

tri = load_stl("assets/tech_core_m.stl")
lab, k = components(tri)
counts = np.bincount(lab)

# 销轴组件（跟随 Q 自旋）：轴侧的小件
PIN_COMPS = {702, 891, 689, 614, 615, 664, 733, 890, 899}
# 滑架组件（随滑槽 + P 铰）：面板/底板
CARR_COMPS = {602, 785, 936, 951, 956}

pin_tris = np.concatenate([tri[lab == c] for c in PIN_COMPS])
carr_tris = np.concatenate([tri[lab == c] for c in CARR_COMPS])
body_mask = ~np.isin(lab, list(PIN_COMPS | CARR_COMPS))
body_tris = tri[body_mask]
# 丢掉主体里残留的退化单三角碎片（z≈-0.226 的薄片，视觉为零）
# （保留不动也可以；此处保留，避免改变主体网格）

# 烘焙竖直变换：world = (mx, mz, 0.851-my)，使 mesh 坐标 == MJCF world 坐标（q=0）
def bake(t):
    o = t.reshape(-1, 3).copy()
    out = np.empty_like(o)
    out[:, 0] = o[:, 0]
    out[:, 1] = o[:, 2]
    out[:, 2] = 0.851 - o[:, 1]
    return out.reshape(-1, 3, 3)

body_tris, carr_tris, pin_tris = bake(body_tris), bake(carr_tris), bake(pin_tris)

print("split:")
save_stl("assets/tech_core_body.stl", body_tris)
save_stl("assets/col_carriage.stl", carr_tris)
save_stl("assets/col_pin.stl", pin_tris)

for name, t in [("body", body_tris), ("carriage", carr_tris), ("pin", pin_tris)]:
    lo, hi = t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)
    print(f"  {name}: file bbox {lo.round(4)} .. {hi.round(4)}")

# 记录销轴几何常量（已烘焙到 world 坐标，q=0 时 tc_body 系 == world）
p_lo, p_hi = pin_tris.reshape(-1, 3).min(0), pin_tris.reshape(-1, 3).max(0)
ax_x = (p_lo[0] + p_hi[0]) / 2
ax_z = (p_lo[2] + p_hi[2]) / 2
print(f"pin axis (world): x={ax_x:.4f} z={ax_z:.4f}  (runs along -Y)")
print(f"pin tip world y: {p_lo[1]:.4f}  base world y: {p_hi[1]:.4f}")
E_local = np.array([ax_x, p_lo[1], ax_z])
print(f"E_local(=E_world at q=0) = {E_local.round(4).tolist()}")
np.save("assets/col_meta.npy", {"E_local": E_local, "ax_x": ax_x, "ax_z": ax_z}, allow_pickle=True)
