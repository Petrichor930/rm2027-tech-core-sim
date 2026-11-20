# -*- coding: utf-8 -*-
"""无头验证：六轴臂 + 六步装配运动链 + 能量单元锁定数学"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import math
import mujoco
import numpy as np
from PIL import Image

model = mujoco.MjModel.from_xml_path("tech_core_joints.xml")
data = mujoco.MjData(model)
J = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_qposadr[i] for i in range(model.njnt)}
SID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pin_tip")
EU = int(model.body('energy_unit').mocapid)
ARM_J = ['j1','j2','j3','j4','j5','j6']

def mat_to_quat(R):
    q = np.empty(4); tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0: s = math.sqrt(tr+1.0)*2; q[0]=.25*s; q[1]=(R[2,1]-R[1,2])/s; q[2]=(R[0,2]-R[2,0])/s; q[3]=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]: s = math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; q[0]=(R[2,1]-R[1,2])/s; q[1]=.25*s; q[2]=(R[0,1]+R[1,0])/s; q[3]=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]: s = math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; q[0]=(R[0,2]-R[2,0])/s; q[1]=(R[0,1]+R[1,0])/s; q[2]=.25*s; q[3]=(R[1,2]+R[2,1])/s
    else: s = math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; q[0]=(R[1,0]-R[0,1])/s; q[1]=(R[0,2]+R[2,0])/s; q[2]=(R[1,2]+R[2,1])/s; q[3]=.25*s
    return q/np.linalg.norm(q)

def quat_to_mat(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def pin_frame():
    return data.site_xpos[SID].copy(), data.site_xmat[SID].reshape(3,3).copy()

r = mujoco.Renderer(model, height=480, width=640)
def cam_look(pos, target):
    c = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, c)
    c.lookat[:]=target
    d=float(np.linalg.norm(np.array(pos)-np.array(target))); c.distance=d
    dx,dy,dz=np.array(pos)-np.array(target)
    c.azimuth=float(np.degrees(np.arctan2(-dy,dx))); c.elevation=float(np.degrees(np.arcsin(dz/max(d,1e-6))))
    return c
def shot(name, tgt=(0,0,0.9), off=(0.4,-1.6,0.5)):
    mujoco.mj_forward(model, data)
    r.update_scene(data, camera=cam_look(np.array(tgt)+np.array(off), tgt))
    Image.fromarray(r.render()).save(name)
    print("saved", name)

# ===== 1. 零位：E 精确在 CAD 位置 =====
mujoco.mj_forward(model, data)
ppos, pR = pin_frame()
assert np.allclose(ppos, [-0.1293,-0.333,0.7932], atol=1e-6), ppos
print("[OK] 零位 E =", ppos.round(4).tolist())

# ===== 2. 六关节各自运动 =====
E0 = ppos.copy()
for j in ARM_J:
    data.qpos[J[j]] = np.radians(20)
    mujoco.mj_forward(model, data)
    E1 = data.site_xpos[SID]
    d = float(np.linalg.norm(E1-E0))
    # j6 绕鼓法向过安装座转，E 在轴线附近允许小位移；其余关节应明显移动 E
    assert d > 0.005 or j == 'j6', f"{j} moved E only {d}"
    print(f"[OK] {j} +20° -> E 位移 {d*1000:.1f} mm")
    data.qpos[J[j]] = 0
mujoco.mj_forward(model, data)

# ===== 3. 六步装配链（零位姿态） =====
ppos, pR = pin_frame()
e = pR @ np.array([0.0,-1.0,0.0])
y_col = e
z_col = np.array([0.0,0.0,1.0]); z_col = z_col-np.dot(z_col,y_col)*y_col; z_col/=np.linalg.norm(z_col)
x_col = np.cross(y_col,z_col)
R = np.stack([x_col,y_col,z_col],axis=1)
data.mocap_quat[EU] = mat_to_quat(R)
data.mocap_pos[EU] = ppos + e*0.075
ref_pos = data.mocap_pos[EU].copy()
shot("asm_s2_aligned.png", ppos)

for _ in range(50):
    data.mocap_pos[EU] += -e*0.002
insert = -float(np.dot(data.mocap_pos[EU]-ref_pos, e))
assert abs(insert-0.1) < 1e-6
tp = pR.T @ (data.mocap_pos[EU]-ppos)
tR = pR.T @ quat_to_mat(data.mocap_quat[EU])
print(f"[OK] 插入 {insert*1000:.0f}mm 并锁定")
shot("asm_s2_inserted.png", ppos)

data.qpos[J['j_slide']] = 0.1
mujoco.mj_forward(model, data)
ppos2, pR2 = pin_frame()
data.mocap_pos[EU] = ppos2 + pR2 @ tp
data.mocap_quat[EU] = mat_to_quat(pR2 @ tR)
shot("asm_s3_lifted.png", ppos2)

data.qpos[J['j_p']] = np.radians(90)
mujoco.mj_forward(model, data)
ppos3, pR3 = pin_frame()
data.mocap_pos[EU] = ppos3 + pR3 @ tp
data.mocap_quat[EU] = mat_to_quat(pR3 @ tR)
shot("asm_s4_pivoted.png", ppos3)

data.qpos[J['j_q']] = np.radians(30)
mujoco.mj_forward(model, data)
ppos4, pR4 = pin_frame()
data.mocap_pos[EU] = ppos4 + pR4 @ tp
data.mocap_quat[EU] = mat_to_quat(pR4 @ tR)
uR = quat_to_mat(data.mocap_quat[EU])
um = uR @ np.array([0.0,-1.0,0.0])
e4 = pR4 @ np.array([0.0,-1.0,0.0])
assert np.dot(um, -e4) > 0.999
print("[OK] 锁定跟随 slide/P/Q 全程保持对齐")

# ===== 4. 机械臂动起来后锁定仍保持（关节联动） =====
data.qpos[J['j2']] = np.radians(15)
mujoco.mj_forward(model, data)
ppos5, pR5 = pin_frame()
data.mocap_pos[EU] = ppos5 + pR5 @ tp
data.mocap_quat[EU] = mat_to_quat(pR5 @ tR)
uR = quat_to_mat(data.mocap_quat[EU]); um = uR @ np.array([0.0,-1.0,0.0])
e5 = pR5 @ np.array([0.0,-1.0,0.0])
assert np.dot(um, -e5) > 0.999
shot("asm_arm_moved.png", ppos5)
print("all checks passed")
