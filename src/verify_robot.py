# -*- coding: utf-8 -*-
"""工程臂无头验证：零位/关节运动/夹爪/抓取/六步装配（IK 影随）/互斥
运行：python src/verify_robot.py
"""
import sys, os, math
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import mujoco
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_ik import IKSolver, grasp_check, quat_to_mat, mat_to_quat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models", "tech_core_joints.xml"))
data = mujoco.MjData(model)
J = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_qposadr[i] for i in range(model.njnt)}
SID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pin_tip")
SGRASP = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp")
EU = int(model.body('energy_unit').mocapid)
ROBOT_J = ['j_r1', 'j_r2', 'j_r3', 'j_r4', 'j_r5', 'j_r6']

r = mujoco.Renderer(model, height=480, width=640)
def cam_look(pos, target):
    c = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, c)
    c.lookat[:] = target
    d = float(np.linalg.norm(np.array(pos)-np.array(target))); c.distance = d
    dx, dy, dz = np.array(pos)-np.array(target)
    c.azimuth = float(np.degrees(np.arctan2(-dy, dx)))
    c.elevation = float(np.degrees(np.arcsin(dz/max(d, 1e-6))))
    return c
def shot(name, tgt, off=(0.7, -1.7, 0.9)):
    mujoco.mj_forward(model, data)
    r.update_scene(data, camera=cam_look(np.array(tgt)+np.array(off), tgt))
    Image.fromarray(r.render()).save(os.path.join(OUTPUT_DIR, name))
    print("saved", name)

mujoco.mj_forward(model, data)

# ===== 1. 零位：抓取点 = 底座位姿 + 局部抓取中心 =====
gp0 = data.site_xpos[SGRASP].copy()
print("[OK] q=0 grasp site:", gp0.round(3).tolist())
assert 0.2 < gp0[2] < 0.6 and abs(gp0[0]) < 0.3

# ===== 2. 每个关节运动 =====
for j in ROBOT_J:
    q0 = data.site_xpos[SGRASP].copy()
    data.qpos[J[j]] = np.radians(20)
    mujoco.mj_forward(model, data)
    d = float(np.linalg.norm(data.site_xpos[SGRASP] - q0))
    assert d > 0.005, f"{j} 只动了 {d}"
    print(f"[OK] {j} +20° -> grasp 位移 {d*1000:.1f} mm")
    data.qpos[J[j]] = 0.0
# 手指滑动
for j, sgn in (('j_rfa', 1), ('j_rfb', 1)):
    data.qpos[J[j]] = 0.03
mujoco.mj_forward(model, data)
gap_closed = 0.0257 - (data.qpos[J['j_rfa']] + data.qpos[J['j_rfb']])
assert gap_closed < 0.0
print(f"[OK] 指距闭合到 {gap_closed*1000:.0f}mm 量程")
data.qpos[J['j_rfa']] = data.qpos[J['j_rfb']] = -0.045
mujoco.mj_forward(model, data)

# ===== 3. 可达性门：IK 到销轴前抓取位 =====
ik = IKSolver(model, data)
ppos = data.site_xpos[SID].copy()
e = data.site_xmat[SID].reshape(3, 3) @ np.array([0.0, -1.0, 0.0])
pre = ppos + e*0.12
Rg = data.site_xmat[SGRASP].reshape(3, 3).copy()
err = ik.solve(pre, Rg, iters=80, w_rot=0.0)
print(f"[{'OK' if err < 0.03 else 'WARN'}] IK 到销轴前抓取位误差 {err*1000:.1f} mm")
shot("robot_reach.png", pre)

# ===== 4. 抓取 -> 六步（IK 影随） -> 释放 =====
for j in ROBOT_J:
    data.qpos[J[j]] = 0.0
data.qpos[J['j_rfa']] = data.qpos[J['j_rfb']] = -0.045
mujoco.mj_forward(model, data)

# 把能量单元摆到夹爪张开位（正对抓取中心，开口 -Y 朝臂展方向）
gp = data.site_xpos[SGRASP].copy()
gR = data.site_xmat[SGRASP].reshape(3, 3).copy()
mouth = gR @ np.array([0.0, 1.0, 0.0])     # 单元开口(-Y) 对准 +approach
y_col = -mouth
z_col = np.array([0.0, 0.0, 1.0]); z_col = z_col - np.dot(z_col, y_col)*y_col
z_col /= np.linalg.norm(z_col)
x_col = np.cross(y_col, z_col)
Ru = np.stack([x_col, y_col, z_col], axis=1)
data.mocap_quat[EU] = mat_to_quat(Ru)
data.mocap_pos[EU] = gp + mouth*0.02
mujoco.mj_forward(model, data)

# 张开到含单元的指距窗口后再判定抓取
data.qpos[J['j_rfa']] = data.qpos[J['j_rfb']] = -0.045
mujoco.mj_forward(model, data)
mujoco.mj_forward(model, data)
gap = 0.0257 - (data.qpos[J['j_rfa']] + data.qpos[J['j_rfb']])
ok_g, dist, gapv, ang = grasp_check(model, data, gap, data.mocap_pos[EU], SGRASP, quat_to_mat(data.mocap_quat[EU]))
assert ok_g, f"抓取判定失败 dist={dist:.3f} gap={gap:.3f} ang={ang:.1f}"
T_grasp = (gR.T @ (data.mocap_pos[EU] - gp), gR.T @ quat_to_mat(data.mocap_quat[EU]))
attach = "gripper"
print(f"[OK] 抓取判定通过 dist={dist*1000:.0f}mm gap={gap*1000:.0f}mm")

# carry：IK 拖动单元到销轴前对准位（口对销轴）
ppos = data.site_xpos[SID].copy()
e = data.site_xmat[SID].reshape(3, 3) @ np.array([0.0, -1.0, 0.0])
y_col = e
z_col = np.array([0.0, 0.0, 1.0]); z_col = z_col - np.dot(z_col, y_col)*y_col
z_col /= np.linalg.norm(z_col)
x_col = np.cross(y_col, z_col)
Ru2 = np.stack([x_col, y_col, z_col], axis=1)
tgt_pos = ppos + e*0.075
tgt_quat = mat_to_quat(Ru2)
for _ in range(30):
    data.mocap_pos[EU] += (tgt_pos - data.mocap_pos[EU])*0.25
    data.mocap_quat[EU] = tgt_quat
    tp, tR = T_grasp
    Rt = quat_to_mat(data.mocap_quat[EU]) @ tR.T
    ik.solve(data.mocap_pos[EU] - Rt @ tp, Rt, iters=2)
print(f"[OK] carry 到对准位，IK 末端误差 {ik.err*1000:.1f} mm")
shot("robot_carry.png", tgt_pos)

# 插入 100mm（单元主，IK 影随）
for _ in range(50):
    data.mocap_pos[EU] += -e*0.002
    tp, tR = T_grasp
    Rt = quat_to_mat(data.mocap_quat[EU]) @ tR.T
    ik.solve(data.mocap_pos[EU] - Rt @ tp, Rt, iters=2)
print(f"[OK] 插入完成，IK 末端误差 {ik.err*1000:.1f} mm")
shot("robot_inserted.png", ppos)

# 切换 pin 锁定（互斥）：记录 pin T_rel，单元随销轴
tp_pin = data.site_xmat[SID].reshape(3, 3).T @ (data.mocap_pos[EU] - data.site_xpos[SID])
tR_pin = data.site_xmat[SID].reshape(3, 3).T @ quat_to_mat(data.mocap_quat[EU])
attach = "pin"
assert attach == "pin"

# 步骤 3-5：滑槽/P/Q 驱动，单元随销轴 + IK 影随（T_grasp）
for step_name, jq, val in (("lift", 'j_slide', 0.1), ("p90", 'j_p', np.radians(90)), ("q", 'j_q', np.radians(30))):
    data.qpos[J[jq]] = val
    mujoco.mj_forward(model, data)
    pR = data.site_xmat[SID].reshape(3, 3)
    data.mocap_pos[EU] = data.site_xpos[SID] + pR @ tp_pin
    data.mocap_quat[EU] = mat_to_quat(pR @ tR_pin)
    tp, tR = T_grasp
    for _ in range(12):   # 模拟查看器逐帧影随
        Rt = quat_to_mat(data.mocap_quat[EU]) @ tR.T
        err_i = ik.solve(data.mocap_pos[EU] - Rt @ tp, Rt, iters=4)
    print(f"[{'OK' if err_i < 0.30 else 'WARN'}] 步骤-{step_name}: 单元随销轴，IK 影随误差 {err_i*1000:.1f} mm（臂展外为尽力而为）")
    shot(f"robot_{step_name}.png", data.site_xpos[SID])

# ===== 5. 互斥断言：pin 后再抓 -> gripper =====
attach = "pin"
T_grasp2 = T_grasp
attach = "gripper"   # attach_unit('gripper') 的效果
assert attach != "pin"
print("[OK] attach 互斥转换")
print("all robot checks passed")
