# -*- coding: utf-8 -*-
"""从 assets/robot_meta.npy 生成 models/robot_arm.xml（可复跑）
运行：python tools/gen_robot_xml.py
"""
import sys, os
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

META = np.load("assets/robot_meta.npy", allow_pickle=True).item()
OUT = "models/robot_arm.xml"

files = META["files"]
J = {k: META[k] for k in ["r_j1", "r_j2", "r_j3", "r_j4", "r_j5", "r_j6"]}
travel = np.array(META["travel"], dtype=float)
grasp = np.array(META["grasp_pos"], dtype=float)
gap0 = META["gap0_mm"] / 1000.0

# 底座安装位姿（world）：面朝 +Y（朝科技核心），臂展范围内贴近装配柱
BASE_POS = (-0.13, -0.60, 0.25)   # 落在 25cm 简化平台上（保证 P 翻转后 z~1.1 仍在臂展内）

JOINT_RANGE = {
    "r_j1": (-180, 180), "r_j2": (-120, 120), "r_j3": (-150, 150),
    "r_j4": (-180, 180), "r_j5": (-120, 120), "r_j6": (-180, 180),
}

COLORS = {
    "robot_base": "0.55 0.32 0.14", "robot_l1": "0.85 0.45 0.12",
    "robot_l2": "0.9 0.5 0.15", "robot_l3": "0.35 0.35 0.38",
    "robot_l4": "0.85 0.45 0.12", "robot_l5": "0.3 0.3 0.33",
    "robot_l6": "0.42 0.42 0.46", "robot_gripper": "0.28 0.28 0.3",
    "robot_fing_a": "0.85 0.62 0.2", "robot_fing_b": "0.85 0.62 0.2",
}

def f(v, nd=4):
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if isinstance(v, float) else v

lines = []
lines.append('<mujoco model="robot_arm">')
lines.append('  <!-- 工程六轴机械臂+夹爪（STEP 拆分生成，勿手改；tools/gen_robot_xml.py 生成） -->')
lines.append('  <asset>')
mi = 0
mesh_names = {}
for g, fl in files.items():
    for fn in fl:
        nm = fn.replace(".stl", "")
        mesh_names.setdefault(g, []).append(nm)
        lines.append(f'    <mesh name="{nm}" file="{fn}"/>')
for g, c in COLORS.items():
    lines.append(f'    <material name="m_{g}" rgba="{c} 1"/>')
lines.append('  </asset>')
lines.append('  <worldbody>')
lines.append(f'    <body name="rb_base" pos="{BASE_POS[0]} {BASE_POS[1]} {BASE_POS[2]}">')
lines.append('      <geom name="rb_platform" type="box" size="0.22 0.22 0.125" pos="0 0 -0.125" rgba="0.35 0.32 0.3 1"/>')
for nm in mesh_names["robot_base"]:
    lines.append(f'      <geom type="mesh" mesh="{nm}" material="m_robot_base"/>')

chain = ["r_j1", "r_j2", "r_j3", "r_j4", "r_j5", "r_j6"]
groups = ["robot_l1", "robot_l2", "robot_l3", "robot_l4", "robot_l5", "robot_l6"]
indent = "      "
for jn, g in zip(chain, groups):
    pos, d = J[jn]
    lo, hi = JOINT_RANGE[jn]
    lines.append(f'{indent}<body name="rb_{jn[2:]}" pos="0 0 0">')
    jn2 = jn.replace("r_j", "j_r")
    lines.append(f'{indent}  <joint name="{jn2}" type="hinge" axis="{f(d[0])} {f(d[1])} {f(d[2])}" '
                 f'pos="{f(pos[0])} {f(pos[1])} {f(pos[2])}" range="{lo} {hi}"/>')
    for nm in mesh_names[g]:
        lines.append(f'{indent}  <geom type="mesh" mesh="{nm}" material="m_{g}"/>')
    indent += "  "

lines.append(f'{indent}<body name="rb_grip" pos="0 0 0">')
for nm in mesh_names["robot_gripper"]:
    lines.append(f'{indent}  <geom type="mesh" mesh="{nm}" material="m_robot_gripper"/>')
lines.append(f'{indent}  <site name="grasp" pos="{f(grasp[0])} {f(grasp[1])} {f(grasp[2])}" size="0.012" rgba="0 1 0 0.5"/>')
for side, jn, ax in (("a", "j_rfa", travel), ("b", "j_rfb", -travel)):
    g = f"robot_fing_{side}"
    lines.append(f'{indent}  <body name="rb_fing_{side}" pos="0 0 0">')
    lines.append(f'{indent}    <joint name="{jn}" type="slide" axis="{f(ax[0])} {f(ax[1])} {f(ax[2])}" range="-0.035 0.035"/>')
    for nm in mesh_names[g]:
        lines.append(f'{indent}    <geom type="mesh" mesh="{nm}" material="m_{g}"/>')
    lines.append(f'{indent}  </body>')
lines.append(f'{indent}</body>')

for _ in range(len(chain) + 1):   # 关 l6..l1 + rb_base
    indent = indent[:-2]
    lines.append(f'{indent}</body>')
lines.append('  </worldbody>')
lines.append('</mujoco>')

with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"已生成 {OUT}")
print(f"底座安装: {BASE_POS}   抓取中心: {np.round(grasp,3).tolist()}   指距: {gap0*1000:.1f}mm")
