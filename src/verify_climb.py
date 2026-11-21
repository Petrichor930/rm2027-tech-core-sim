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
JD = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_dofadr[i] for i in range(model.njnt)}
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
    r.update_scene(data, camera=cam_look(np.array([0, -1.0, 0.4]) + np.array((-1.8, -1.6, 1.1)), np.array([0, -1.0, 0.4])))
    Image.fromarray(r.render()).save(os.path.join(OUTPUT_DIR, name))
    print("saved", name, "chassis z =", round(float(p[2]), 3))

mujoco.mj_forward(model, data)
z0 = float(data.xpos[bid][2])
print(f"[OK] 初始车体高度 z={z0:.3f}")

# ---- 1. 升降副单独测试 ----
data.ctrl[act["a_liftf"]] = 0.25
data.ctrl[act["a_liftr"]] = 0.25
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

# ---- 2. 爬台阶（实车机构）：四抬升杆(4×3508)齐升让轮面与台阶齐平，麦轮×4(M3508)
#      +小轮×2(M2006)推进，快走完后杆收起。CAD 中小轮收起高位（高出轮底 218mm），
#      逼近时前脚已悬于台顶上方，直接落脚台面撑升即可。
# 踢面 y（gen_terrain_xml 实测）：一级 -1.62(台顶-0.2)，二级 -1.27(台顶0=科技核心安装台)
RISERS = [(-1.87, -0.2), (-1.27, 0.0)]
FEET_UP = -0.25              # 小轮收起位
shot("climb_00_start.png")
y0 = float(data.xpos[bid][1])
zs = []
y_hist = []
WHEEL_BOT = 0.089            # 轮底相对车体原点
phase = "approach"
active = None                # (踢面y, 台顶z)
prev_f = FEET_UP
prev_r = FEET_UP
tphase = 0
for i in range(8000):
    y = float(data.xpos[bid][1])
    y_front = y + 0.262      # 前轮接地点
    y_axle_r = y - 0.183
    zc = float(data.xpos[bid][2])
    vy = float(data.qvel[0:6][1])
    vz = float(data.qvel[2])
    up = data.xmat[bid].reshape(3, 3)[:, 2]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(up[1])))))
    tphase += 1
    if active is None:
        for rs0, top0 in RISERS:
            if rs0 - y_front > -0.10:
                active = (rs0, top0)
                break
    rs, top = active if active else (None, None)
    z_tgt = (top - WHEEL_BOT) if top is not None else None
    # ---- 状态机：接近→四杆齐升(前脚台面/后脚地面撑，轮面齐平)→六电机推进→后杆收起 ----
    if phase == "approach":
        if rs is not None and y_front > rs - 0.03 and abs(vy) < 0.15:
            phase, tphase = "lift", 0
    elif phase == "lift":
        if z_tgt is not None and (zc >= z_tgt - 0.01 or tphase > 3500):
            phase, tphase = "roll", 0
    elif phase == "roll":
        if (rs is not None and y_axle_r >= rs + 0.05) or tphase > 9000:
            phase, tphase = "settle", 0        # 快走完（后轮已上台）：后杆收起
    elif phase == "settle":
        if abs(pitch) < 4 or tphase > 2500:
            phase, tphase, active = "approach", 0, None
    # ---- 抬升杆控制（bang-bang 目标高度 + 限速 + 垂直速度阻尼）----
    if phase in ("lift", "roll"):
        pred_z = zc + 0.25 * vz
        aim = 0.25 if (z_tgt is not None and pred_z < z_tgt) else -0.10
        fjt = max(min(aim, prev_f + 0.0005), prev_f - 0.003)
        rjt = max(min(aim, prev_r + 0.0005), prev_r - 0.003)
    else:
        fjt, rjt = FEET_UP, FEET_UP           # 收起（approach/settle）
    prev_f, prev_r = fjt, rjt
    # ---- 驱动：麦轮 + 小轮(M2006) ----
    if phase in ("roll",):
        wspd, rspd = 8.0, 8.0
    elif phase == "lift":
        wspd, rspd = 0.0, 0.0
    elif phase == "settle":
        wspd, rspd = 2.0, 2.0
    elif rs is not None and y_front > rs - 0.80:
        wspd, rspd = 3.0, 3.0
    else:
        wspd, rspd = 12.0, 12.0
    for k in range(4):
        data.ctrl[k] = wspd
    data.ctrl[4] = rspd
    data.ctrl[5] = rspd
    data.ctrl[6] = fjt
    data.ctrl[7] = rjt
    saved = data.qpos[LOCK_ADR].copy()
    mujoco.mj_step(model, data)
    data.qpos[LOCK_ADR] = saved
    data.qvel[LOCK_DOF] = 0.0
    y_hist.append(float(data.xpos[bid][1]))
    if float(data.xpos[bid][1]) > -0.55 and zc > -0.15:
        print(f"  [i={i}] 到达科技核心台：y={data.xpos[bid][1]:.2f} z={zc:.3f}，停止")
        break
    if i % 300 == 0:
        wq = [round(float(data.qvel[JD[f'j_w{k}']]),1) for k in range(1,5)]
        print(f"  i={i} y={data.xpos[bid][1]:.2f} z={data.xpos[bid][2]:.3f} vy={vy:+.2f} pitch={pitch:+.0f} "
              f"轮速={wq} 升降q={data.qpos[J['j_lift_f']]:+.3f}/{data.qpos[J['j_lift_r']]:+.3f} ncon={data.ncon} {phase}")
    if i % 100 == 0:
        zs.append(float(data.xpos[bid][2]))
    if i == 1500:
        shot("climb_01_mid.png")
    if i == 4500:
        shot("climb_02_high.png")
z_end = float(data.xpos[bid][2])
print(f"y 前进: {y_hist[-1]-y_hist[0]:.2f}m")
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
climbed = z_end - z0
print(f"[{'OK' if climbed > 0.35 else 'WARN'}] 车体爬升 {climbed:.3f}m（目标≥0.35 到科技核心台顶），前进 y={y_end:.2f}m")
print(f"[{'OK' if upright else 'WARN'}] 车体直立 up_z={up[2]:.2f}")
print("climb checks done")
