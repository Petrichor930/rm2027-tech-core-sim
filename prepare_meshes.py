import numpy as np, struct, os

def read_stl(path):
    with open(path, 'rb') as f:
        head = f.read(5)
    if head[:5] == b'solid':
        verts = []
        with open(path, 'r', errors='replace') as f:
            for line in f:
                p = line.split()
                if len(p) == 4 and p[0] == 'vertex':
                    verts.append([float(p[1]), float(p[2]), float(p[3])])
        v = np.array(verts, dtype=np.float64)
        n = len(v) // 3
    else:
        with open(path, 'rb') as f:
            f.read(80)
            n = struct.unpack('<I', f.read(4))[0]
            raw = f.read()
        dt = np.dtype([('normal', '<f4', 3), ('v', '<f4', 9), ('attr', '<u2')])
        arr = np.frombuffer(raw, dtype=dt, count=n)
        v = arr['v'].reshape(-1, 3).astype(np.float64)
    return v.reshape(n, 3, 3)

def write_stl_binary(path, tris):
    # tris: (N, 3, 3) float
    N = tris.shape[0]
    with open(path, 'wb') as f:
        f.write(b'PinyCaster mujoco mesh'.ljust(80, b'\0'))
        f.write(struct.pack('<I', N))
        normals = np.zeros((N, 3), dtype=np.float32)
        attr = np.zeros((N, 1), dtype=np.uint16)
        buf = np.concatenate([normals, tris.reshape(N, 9).astype(np.float32)], axis=1)
        # pack: normal(3) + verts(9) + attr(2 bytes) = 50 bytes per tri
        for i in range(N):
            f.write(buf[i].tobytes())
            f.write(attr[i].tobytes())

jobs = [
    ("assets/energy_unit_asm.stl", "assets/energy_unit_m.stl"),
    ("assets/tech_core.stl", "assets/tech_core_m.stl"),
]

for src, dst in jobs:
    tris = read_stl(src)          # in mm
    tris_m = tris / 1000.0        # to meters
    centroid = tris_m.reshape(-1, 3).mean(axis=0)
    tris_c = tris_m - centroid    # center at origin
    write_stl_binary(dst, tris_c)
    mn = tris_c.reshape(-1, 3).min(axis=0)
    mx = tris_c.reshape(-1, 3).max(axis=0)
    sz = mx - mn
    print(f"{dst}: {tris.shape[0]} tris, size (m) x {sz[0]:.4f} y {sz[1]:.4f} z {sz[2]:.4f}, "
          f"centroid moved by ({-centroid[0]:.4f}, {-centroid[1]:.4f}, {-centroid[2]:.4f}) m")
