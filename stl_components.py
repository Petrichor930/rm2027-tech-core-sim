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
    tri = arr[:, 12:48].copy().view('<f4').reshape(n, 3, 3)
    return tri.astype(np.float64)

def components(tri, tol=1e-6):
    # weld vertices on a grid, then union-find over triangles sharing a vertex
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

path = sys.argv[1] if len(sys.argv) > 1 else "assets/tech_core_m.stl"
tri = load_stl(path)
print(path, "triangles:", len(tri))
lo, hi = tri.reshape(-1,3).min(0), tri.reshape(-1,3).max(0)
print("bbox min:", lo.round(4), "max:", hi.round(4), "size(m):", (hi-lo).round(4))

lab, k = components(tri)
print("connected components:", k)
order = np.argsort(np.bincount(lab))[::-1]
for c in order[:15]:
    m = tri[lab == c].reshape(-1, 3)
    lo, hi = m.min(0), m.max(0)
    print(f"  comp {c}: {int((lab==c).sum()):7d} tris  bbox {lo.round(3)} .. {hi.round(3)}  size {(hi-lo).round(3)}")
