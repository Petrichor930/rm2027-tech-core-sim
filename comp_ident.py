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

tri = load_stl("assets/tech_core_m.stl")
lab, k = components(tri)
counts = np.bincount(lab)
big = np.argsort(counts)[::-1][:40]

cmap = [
    "1 0 0", "0 0.6 0", "0 0.3 1", "1 0.6 0", "0.8 0 0.8",
    "0 0.9 0.9", "1 0.3 0.3", "0.4 0.2 0", "0.6 0.6 0", "1 0 0.5",
    "0.5 0.5 1", "0.9 0.9 0.3", "0.3 0.6 0.3", "1 0.4 0", "0 0.5 0.5",
    "0.7 0 0", "0 0 0.6", "0 0.7 0.7", "1 0.8 0.8", "0.5 0 0",
    "0.6 0.8 0.2", "0.2 0.4 0.8", "0.8 0.4 0.6", "0.4 0.4 0.4", "0.9 0.5 0.7",
    "0.3 0.3 0", "0 0.4 0.2", "0.6 0.2 0.6", "0.8 0.8 0.6", "0.2 0.2 0.6",
    "0.5 0.3 0.1", "0.7 0.7 1", "1 1 0.5", "0.5 1 0.5", "1 0.5 0.5",
    "0.3 0.5 0.5", "0.5 0.5 0.3", "0.9 0.7 0.3", "0.3 0.7 0.9", "0.7 0.3 0.3",
]
lines = ['<mujoco model="comps"><compiler angle="degree" meshdir="comp_meshes"/>',
         '<asset><material name="m0" rgba="0.85 0.85 0.85 1"/>']
for i in range(40):
    lines.append(f'<material name="m{i+1}" rgba="{cmap[i]} 1"/>')
lines.append('</asset><worldbody><light pos="0 -3 4" dir="0 0 -1"/><light pos="2 2 3"/>')
lines.append('<camera name="cam" pos="0 -2.5 1.8" xyaxes="1 0 0 0 0.55 0.85"/>')
lines.append('<geom type="plane" size="4 4 0.1" rgba="0.75 0.75 0.75 1"/>')
lines.append('<body name="tc" pos="0 0 0.851" quat="0.70710678 -0.70710678 0 0">')
import os
os.makedirs("comp_meshes", exist_ok=True)
allm = np.empty((0,3,3))
for i, c in enumerate(big):
    t = tri[lab == c]
    save_stl(f"comp_meshes/c{i:02d}.stl", t)
    lines.append(f'<geom type="mesh" mesh="c{i:02d}" material="m{i+1}" contype="0" conaffinity="0"/>')
    m = t.reshape(-1,3)
    lo, hi = m.min(0), m.max(0)
    print(f"c{i:02d} comp {c} {int(counts[c]):6d} tris bbox {lo.round(3)} .. {hi.round(3)}")
    allm = np.concatenate([allm, t])
lo, hi = allm.reshape(-1,3).min(0), allm.reshape(-1,3).max(0)
lines.append('</body></worldbody>')
for i in range(40):
    lines.append(f'<asset><mesh name="c{i:02d}" file="c{i:02d}.stl"/></asset>' if False else '')
lines.append('</mujoco>')
# rebuild properly: assets must be inside <asset>
xml = ['<mujoco model="comps"><compiler angle="degree" meshdir="comp_meshes"/>', '<asset>']
for i in range(40):
    xml.append(f'<material name="m{i+1}" rgba="{cmap[i]} 1"/>')
    xml.append(f'<mesh name="c{i:02d}" file="c{i:02d}.stl"/>')
xml.append('</asset><worldbody><light pos="0 -3 4" dir="0 0 -1"/><light pos="2 2 3"/>')
xml.append('<camera name="cam" pos="0 -2.5 1.8" xyaxes="1 0 0 0 0.55 0.85"/>')
xml.append('<geom type="plane" size="4 4 0.1" rgba="0.75 0.75 0.75 1"/>')
xml.append('<body name="tc" pos="0 0 0.851" quat="0.70710678 -0.70710678 0 0">')
for i in range(40):
    xml.append(f'<geom type="mesh" mesh="c{i:02d}" material="m{i+1}" contype="0" conaffinity="0"/>')
xml.append('</body></worldbody></mujoco>')
with open("comps.xml", "w") as f:
    f.write("\n".join(xml))
print("wrote comps.xml and comp_meshes/")

import mujoco
from PIL import Image
model = mujoco.MjModel.from_xml_path("comps.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
r = mujoco.Renderer(model, height=480, width=640)

def cam_look(pos, target):
    c = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, c)
    c.lookat[:] = target
    d = float(np.linalg.norm(np.array(pos) - np.array(target)))
    c.distance = d
    c.azimuth = float(np.degrees(np.arctan2(pos[0]-target[0], pos[1]-target[1])))
    c.elevation = float(np.degrees(np.arcsin((pos[2]-target[2]) / max(d, 1e-6))))
    return c

target = (0, 0, 0.9)
for name, pos in {"front": (0,-2.2,1.1), "right": (2.2,0,1.1), "left": (-2.2,0,1.1), "back": (0,2.2,1.1)}.items():
    r.update_scene(data, camera=cam_look(pos, target))
    Image.fromarray(r.render()).save(f"cv_{name}.png")
    print("saved", f"cv_{name}.png")
