import sys, os
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import mujoco
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models", "tech_core_energy_unit.xml"))
data = mujoco.MjData(model)
print("loaded OK")
print("  nbody:", model.nbody, " ngeom:", model.ngeom, " nq:", model.nq, " nv:", model.nv)
print("  body names:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)])
print("  geom sizes:", [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i), model.geom_size[i].round(3).tolist()) for i in range(model.ngeom)])

# simulate a bit
for _ in range(200):
    mujoco.mj_step(model, data)
print("  energy_unit pos after 200 steps:", data.qpos[model.jnt_qposadr[model.body('energy_unit').id if hasattr(model,'body') else 0]:][:3] if False else np.round(data.body('energy_unit').xpos, 3).tolist())

# render offscreen
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="cam")
pixels = renderer.render()
Image.fromarray(pixels).save(os.path.join(OUTPUT_DIR, "preview.png"))
arr = pixels.astype(np.float32)
lum = 0.299*arr[:,:,0]+0.587*arr[:,:,1]+0.114*arr[:,:,2]
print("  render mean brightness:", round(float(lum.mean()), 1), " max:", float(lum.max()))
print("  preview saved output/preview.png")
