import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("tech_core_energy_unit.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tech_core")
print("body xpos:", data.xpos[bid])
R = data.xmat[bid].reshape(3, 3)
print("body xmat:\n", R.round(3))

# mesh vertex range in model: mesh_vert
mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "tech_core")
adr = model.mesh_vertadr[mid]
n = model.mesh_vertnum[mid]
verts = model.mesh_vert[adr:adr+n]
print("mesh verts:", n, "mesh-frame bbox:", verts.min(0).round(3), "..", verts.max(0).round(3))
wv = (R @ verts.T).T + data.xpos[bid]
print("world bbox:", wv.min(0).round(3), "..", wv.max(0).round(3))

# transform candidate component bboxes (mesh coords) to world
comps = {
    "c00_comp731_drumface":  ((-0.357,-0.437,-0.22),  (0.096,0.131,-0.087)),
    "c05_comp1601":          ((0.101,0.554,0.416),   (0.292,0.620,0.476)),
    "c12_comp838_pin?":      ((-0.14,-0.027,-0.189), (-0.12,0.004,-0.169)),
    "c24_comp1622":          ((0.269,-0.337,0.139),  (0.503,0.162,0.31)),
}
for name, (lo, hi) in comps.items():
    corners = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
    wc = (R @ corners.T).T + data.xpos[bid]
    print(f"{name}: world {wc.min(0).round(3)} .. {wc.max(0).round(3)}")
