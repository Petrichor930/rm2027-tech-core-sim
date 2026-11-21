# -*- coding: utf-8 -*-
"""整车爬台阶无头验证：动力学步进 + 运动学锁定（与查看器 drive 模式同构）
运行：python src/verify_climb.py
"""
import sys, os, math
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import mujoco
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models", "tech_core_joints.xml"))
data = mujoco.MjData(model)
J = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_qposadr[i] for i in range(model.njnt)}
act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fb_chassis")

LOCK_NAMES = [j for j in ['j1','j2','j3','j4','j5','j6','j_slide','j_p','j_q',
                          'j_r1','j_r2','j_r3','j_r4','j_r5','j_r6','j_rfa','j_rfb'] if j in J]
LOCK_ADR = np.array([J[j] for j in LOCK_NAMES])
LOCK_DOF = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in LOCK_NAMES])
print(f"[OK] 模型加载：执行器 {model.nu} 个，锁定关节 {len(LOCK_NAMES)} 个")
assert model.nu >= 8, "执行器缺失（robot_full.xml 未包含？）"

r = mujoco.Renderer(model, height=480, width=640)
def cam_look(pos, target):
    c = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, c)
    c.lookat[:] = target
    d = float(np.linalg.norm(np.array(pos)-np.array(target))); c.distance = d
    dx, dy, dz = np.array(pos)-np.array(target)
    c.azimuth = float(np.degrees(np.arctan2(-dy, dx)))
    c.elevation = float(np.degrees(np.arcsin(dz/max(d, 1e-6))))
    return c
def shot(name):
    mujoco.mj_forward(model, data)
    p = data.xpos[bid]
    r.update_scene(data, camera=cam_look(p + np.array((-1.6, -2.2, 1.2)), p))
    Image.fromarray(r.render()).save(os.path.join(OUTPUT_DIR, name))
    print("saved", name, "chassis z =", round(float(p[2]), 3))

mujoco.mj_forward(model, data)
z0 = float(data.xpos[bid][2])
print(f"[OK] 初始车体高度 z={z0:.3f}")

# ---- 1. 升降副单独测试 ----
data.ctrl[act["a_liftf"]] = 0.12
data.ctrl[act["a_liftr"]] = 0.12
saved = data.qpos[LOCK_ADR].copy()
for _ in range(300):
    mujoco.mj_step(model, data)
    data.qpos[LOCK_ADR] = saved
    data.qvel[LOCK_DOF] = 0.0
lift = data.qpos[J['j_lift_f']]
print(f"[{'OK' if abs(lift) > 0.02 else 'WARN'}] 升降副伸出 {lift*1000:.0f}mm（向下顶地面/台阶，预期被挡）")
data.ctrl[act["a_liftf"]] = 0.0
data.ctrl[act["a_liftr"]] = 0.0
for _ in range(300):
    mujoco.mj_step(model, data)
    data.qpos[LOCK_ADR] = saved
    data.qvel[LOCK_DOF] = 0.0

# ---- 2. 爬台阶：四轮差速直行 ----
shot("climb_00_start.png")
y0 = float(data.xpos[bid][1])
zs = []
for i in range(1200):
    for k in range(4):
        data.ctrl[k] = 20.0          # 轮速 rad/s 前进
    data.ctrl[4] = 12.0
    data.ctrl[5] = 12.0
    saved = data.qpos[LOCK_ADR].copy()
    mujoco.mj_step(model, data)
    data.qpos[LOCK_ADR] = saved
    data.qvel[LOCK_DOF] = 0.0
    if i % 100 == 0:
        zs.append(float(data.xpos[bid][2]))
    if i == 400:
        shot("climb_01_mid.png")
    if i == 900:
        shot("climb_02_high.png")
z_end = float(data.xpos[bid][2])
y_end = float(data.xpos[bid][1])
shot("climb_03_end.png")
print(f"z 轨迹: {[round(z,3) for z in zs]}")
FLAT = "--flat" in sys.argv
if FLAT:
    y_move = abs(y_end - y0)
    print(f"[{'OK' if y_move > 0.5 else 'WARN'}] 平地行驶 {y_move:.2f}m")
    print("flat mode done")
    sys.exit(0)
up = data.xmat[bid].reshape(3, 3)[:, 2]
upright = float(up[2]) > 0.7
print(f"[{'OK' if z_end > z0 + 0.05 else 'WARN'}] 车体爬升 {z_end - z0:.3f}m，前进 y={y_end:.2f}m")
print(f"[{'OK' if upright else 'WARN'}] 车体直立 up_z={up[2]:.2f}")
print("climb checks done")
