# -*- coding: utf-8 -*-
"""从 assets/robot_full_meta.npy 生成 models/robot_full.xml（整车动力学模型）
结构：fb_chassis(free) -> 前后升降滑架(滑动) -> 小轮(铰)；麦轮×4(铰)；机械臂链内联挂车顶
执行器：轮速伺服×6、升降位置伺服×2。碰撞：轮=圆柱、车体=盒近似（简化接触，用户确认）。
运行：python tools/gen_fullrobot_xml.py
"""
import sys, os
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

META = np.load("assets/robot_full_meta.npy", allow_pickle=True).item()
ARM = np.load("assets/robot_meta.npy", allow_pickle=True).item()
OUT = "models/robot_full.xml"

FB_START_POS = (-0.13, -2.2, 0.089)  # 世界初始位（轮底贴地；台阶前/装配区旁，可调）
FB_START_YAW = 0.0

files = META["files"]
masses = META["mass"]

def f(v, nd=4):
    return f"{v:.{nd}f}".rstrip("0").rstrip(".")

L = []
L.append('<mujoco model="robot_full">')
L.append('  <!-- 工程整车（总装1.1 简化）：底盘+麦轮×4+前后升降小轮+机械臂（复用 robot_meta） -->')
L.append('  <default>')
L.append('    <default class="dyn">')
L.append('      <geom contype="1" conaffinity="1" friction="0.8 0.02 0.002"/>')
L.append('    </default>')
L.append('    <default class="vis">')
L.append('      <geom contype="0" conaffinity="0"/>')
L.append('    </default>')
L.append('  </default>')
L.append('  <asset>')
meshes = {}
for g, fl in files.items():
    for fn in fl:
        nm = fn.replace(".stl", "")
        meshes.setdefault(g, []).append(nm)
        L.append(f'    <mesh name="{nm}" file="{fn}"/>')
for g in ["robot_base", "robot_l1", "robot_l2", "robot_l3", "robot_l4", "robot_l5", "robot_gripper", "robot_fing_a", "robot_fing_b"]:
    for suf in (["0", "1"] if g == "robot_l6" else [""]):
        nm = g if suf == "" else f"{g}_{suf}"
        if os.path.exists(f"assets/{nm}.stl"):
            meshes.setdefault("arm", []).append(nm)
            L.append(f'    <mesh name="{nm}" file="{nm}.stl"/>')
L.append('  </asset>')
L.append('  <worldbody>')
cy, sy = np.cos(FB_START_YAW), np.sin(FB_START_YAW)
L.append(f'    <body name="fb_chassis" pos="{FB_START_POS[0]} {FB_START_POS[1]} {FB_START_POS[2]}" quat="{f(cy)} 0 0 {f(sy)}">')
L.append('      <freejoint name="j_free"/>')
m_c = masses.get("fb_chassis", 30.0)
L.append(f'      <inertial pos="0 0 0.3" mass="{m_c}" diaginertia="3 4 3"/>')
for nm in meshes.get("fb_chassis", []):
    L.append(f'      <geom type="mesh" mesh="{nm}" class="vis" rgba="0.45 0.42 0.4 1"/>')
L.append('      <geom name="chassis_col" type="box" size="0.28 0.4 0.02" pos="0 -0.05 0.11" class="dyn" rgba="0.5 0.45 0.4 0.35"/>')

def wheel_body(name, g, joint, axis=(1, 0, 0), radius=None):
    ctr = META.get(g + "_ctr", [0, 0, 0.05])
    r = radius if radius is not None else META.get(g + "_r", 0.05)
    out = []
    out.append(f'      <body name="{name}" pos="{f(ctr[0])} {f(ctr[1])} {f(ctr[2])}">')
    out.append(f'        <joint name="{joint}" type="hinge" axis="{axis[0]} {axis[1]} {axis[2]}" damping="0.05"/>')
    m = masses.get(g, 1.5)
    out.append(f'        <inertial pos="0 0 0" mass="{m}" diaginertia="0.01 0.01 0.01"/>')
    for nm in meshes.get(g, []):
        out.append(f'        <geom type="mesh" mesh="{nm}" class="vis" rgba="0.15 0.15 0.17 1"/>')
    out.append(f'        <geom name="{name}_col" type="cylinder" size="{f(r)} 0.025" euler="0 90 0" class="dyn" rgba="0.2 0.2 0.2 0.6"/>')
    out.append('      </body>')
    return out

# 麦轮×4（悬挂 v1 刚性：直接挂车体）
for k in range(1, 5):
    L += wheel_body(f"fb_wm{k}", f"fb_wheel_m{k}", f"j_w{k}")

# 前后升降滑架 -> 小轮
for side, gl, gw, jl, jw in (("f", "fb_lift_f", "fb_wheel_f", "j_lift_f", "j_wf"),
                              ("r", "fb_lift_r", "fb_wheel_r", "j_lift_r", "j_wr")):
    stroke = META.get(gl + "_stroke", 0.12)
    m_l = masses.get(gl, 4.0)
    L.append(f'      <body name="fb_lift_{side}" pos="0 0 0">')
    L.append(f'        <joint name="{jl}" type="slide" axis="0 0 -1" range="0 {f(stroke)}" limited="true" damping="5"/>')
    L.append(f'        <inertial pos="0 0 0" mass="{m_l}" diaginertia="0.02 0.02 0.02"/>')
    for nm in meshes.get(gl, []):
        L.append(f'        <geom type="mesh" mesh="{nm}" class="vis" rgba="0.6 0.55 0.3 1"/>')
    ctr = META.get(gw + "_ctr", [0, 0, 0.05])
    r = META.get(gw + "_r", 0.05)
    L.append(f'        <body name="fb_wheel_{side}" pos="{f(ctr[0]-0)} {f(ctr[1])} {f(ctr[2])}">')
    L.append(f'          <joint name="{jw}" type="hinge" axis="1 0 0" damping="0.05"/>')
    m_w = masses.get(gw, 1.0)
    L.append(f'          <inertial pos="0 0 0" mass="{m_w}" diaginertia="0.005 0.005 0.005"/>')
    for nm in meshes.get(gw, []):
        L.append(f'          <geom type="mesh" mesh="{nm}" class="vis" rgba="0.15 0.15 0.17 1"/>')
    L.append(f'          <geom name="wb_{side}_col" type="cylinder" size="{f(r)} 0.02" euler="0 90 0" class="vis" rgba="0.2 0.2 0.2 0.6"/>')
    L.append('        </body>')
    L.append('      </body>')

# 机械臂（robot_meta 关节链，内联挂车顶；安装位姿可调）
ARM_MOUNT_POS = (0.0, -0.05, 0.55)
ARM_MOUNT_YAW = 0.0    # 臂朝车头(+Y)（robot_meta 的 flange +Y 约定）
ay = np.radians(ARM_MOUNT_YAW)
acy, asy = np.cos(ay/2), np.sin(ay/2)
L.append(f'      <body name="rb_base" pos="{ARM_MOUNT_POS[0]} {ARM_MOUNT_POS[1]} {ARM_MOUNT_POS[2]}" quat="{f(acy)} 0 0 {f(asy)}">')
L.append(f'        <inertial pos="0 0 0.1" mass="8" diaginertia="0.2 0.2 0.2"/>')
for nm in meshes.get("arm", []):
    if nm.startswith("robot_base"):
        L.append(f'        <geom type="mesh" mesh="{nm}" class="vis" rgba="0.55 0.32 0.14 1"/>')
chain = [("j_r1", "robot_l1"), ("j_r2", "robot_l2"), ("j_r3", "robot_l3"),
         ("j_r4", "robot_l4"), ("j_r5", "robot_l5"), ("j_r6", "robot_l6")]
indent = "        "
for jn, g in chain:
    pos, d = ARM.get(jn.replace('j_r', 'r_j'), ([0, 0, 0], [0, 0, 1]))
    L.append(f'{indent}<body name="rb_{jn[2:]}" pos="0 0 0">')
    L.append(f'{indent}  <joint name="{jn}" type="hinge" axis="{f(d[0])} {f(d[1])} {f(d[2])}" pos="{f(pos[0])} {f(pos[1])} {f(pos[2])}" damping="0.5"/>')
    L.append(f'{indent}  <inertial pos="0 0 0" mass="1.5" diaginertia="0.02 0.02 0.02"/>')
    for nm in meshes.get("arm", []):
        if nm.startswith(g):
            L.append(f'{indent}  <geom type="mesh" mesh="{nm}" class="vis" rgba="0.85 0.45 0.12 1"/>')
    indent += "  "
grasp = ARM.get("grasp_pos", [0, 0.727, 0.342])
L.append(f'{indent}<body name="rb_grip" pos="0 0 0">')
L.append(f'{indent}  <inertial pos="0 0 0" mass="0.6" diaginertia="0.005 0.005 0.005"/>')
for nm in meshes.get("arm", []):
    if nm.startswith("robot_gripper"):
        L.append(f'{indent}  <geom type="mesh" mesh="{nm}" class="vis" rgba="0.28 0.28 0.3 1"/>')
L.append(f'{indent}  <site name="grasp" pos="{f(grasp[0])} {f(grasp[1])} {f(grasp[2])}" size="0.012" rgba="0 1 0 0.5"/>')
for side, jn, ax in (("a", "j_rfa", 1), ("b", "j_rfb", -1)):
    L.append(f'{indent}  <body name="rb_fing_{side}" pos="0 0 0">')
    L.append(f'{indent}    <joint name="{jn}" type="slide" axis="{ax} 0 0" range="-0.045 0.045" limited="true" damping="0.2"/>')
    L.append(f'{indent}    <inertial pos="0 0 0" mass="0.3" diaginertia="0.001 0.001 0.001"/>')
    for nm in meshes.get("arm", []):
        if nm.startswith(f"robot_fing_{side}"):
            L.append(f'{indent}    <geom type="mesh" mesh="{nm}" class="vis" rgba="0.85 0.62 0.2 1"/>')
    L.append(f'{indent}  </body>')
L.append(f'{indent}</body>')
for _ in range(len(chain) + 2):   # 关 grip 外的 r6..r1 + rb_base + fb_chassis
    indent = indent[:-2]
    L.append(f'{indent}</body>')

L.append('  </worldbody>')
L.append('  <actuator>')
for k in range(1, 5):
    L.append(f'    <velocity name="a_w{k}" joint="j_w{k}" kv="8" forcerange="-12 12"/>')
L.append('    <velocity name="a_wf" joint="j_wf" kv="4" forcerange="-3 3"/>')
L.append('    <velocity name="a_wr" joint="j_wr" kv="4" forcerange="-3 3"/>')
L.append('    <position name="a_liftf" joint="j_lift_f" kp="200" forcerange="-400 400"/>')
L.append('    <position name="a_liftr" joint="j_lift_r" kp="200" forcerange="-400 400"/>')
L.append('  </actuator>')
L.append('</mujoco>')

with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(L) + "\n")
print("已生成", OUT)
print("车体质量 ≈", round(m_c, 1), "kg   轮半径 ≈", META.get("wheel_r", 0.05))
