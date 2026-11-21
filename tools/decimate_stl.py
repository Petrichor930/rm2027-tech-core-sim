# -*- coding: utf-8 -*-
"""STL 顶点聚类抽稀（视觉网格降面，坐标/轮廓保形）
用法：
  python tools/decimate_stl.py <cell_mm> <stl 文件...>     # 原地覆盖
  from decimate_stl import decimate_tris                    # gen 脚本内用
细薄件（厚度 < cell）会被并掉——装饰件可接受；结构大件在 cell 尺度上保形。
"""
import sys, os, struct
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


def decimate_tris(tri, cell):
    """顶点聚类抽稀：snap 到 cell 网格、簇内取均值、去退化/重复面"""
    if len(tri) == 0:
        return tri
    v = tri.reshape(-1, 3)
    q = np.round(v / cell).astype(np.int64)
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    sums = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(sums, inv, v)
    np.add.at(cnt, inv, 1.0)
    verts = sums / cnt[:, None]
    f = inv.reshape(-1, 3)
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    if len(f) == 0:
        return tri[:0]
    fs = np.sort(f, axis=1)
    _, idx = np.unique(fs, axis=0, return_index=True)
    f = f[np.sort(idx)]
    return verts[f].astype(np.float64)


def drop_small_tris(tri, min_area, weld=0.004):
    """删除小连通件（碎零件）：顶点焊接后按连通域算面积，去掉总面积 < min_area 的件"""
    from collections import defaultdict
    if len(tri) == 0:
        return tri
    v = tri.reshape(-1, 3)
    key = np.round(v / weld).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    parent = np.arange(len(uniq))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in f:
        ra, rb, rc = find(int(a)), find(int(b)), find(int(c))
        if ra != rb:
            parent[rb] = ra
        if ra != rc:
            parent[rc] = ra
    roots = np.array([find(int(i)) for i in range(len(uniq))])
    troot = roots[f[:, 0]]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    comp = defaultdict(float)
    for r, a_ in zip(troot, area):
        comp[r] += a_
    keep = np.array([comp[r] >= min_area for r in troot])
    return tri[keep]


if __name__ == "__main__":
    if sys.argv[1] == "--min-area-cm2":
        min_area = float(sys.argv[2]) * 1e-4
        for p in sys.argv[3:]:
            tri = load_stl(p)
            out = drop_small_tris(tri, min_area)
            save_stl(p, out)
            print(f"{os.path.basename(p)}: {len(tri)} -> {len(out)} tris (删小于 {sys.argv[2]}cm2 的连通件)")
    else:
        cell_mm = float(sys.argv[1])
        cell = cell_mm * 0.001
        for p in sys.argv[2:]:
            tri = load_stl(p)
            out = decimate_tris(tri, cell)
            save_stl(p, out)
            print(f"{os.path.basename(p)}: {len(tri)} -> {len(out)} tris (cell {cell_mm}mm)")
