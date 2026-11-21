# -*- coding: utf-8 -*-
"""科技核心-能量单元 六步装配交互查看器（规则手册 5.4.3）

位姿 (x,y,z,θ,φ,α) 直接映射关节：j_px/j_py/j_pz + j_yaw(θ) + j_pitch(90-φ) + j_roll(α)。
装配柱：j_slide(滑槽100mm) → j_p(P铰 0..90°) → j_q(Q自旋 ±90°, 弹簧回正0.3N·m)。

操作：
  Tab          切换控制对象：能量单元 / 科技核心位姿 / 相机
  1/2/3/4      选择装配难度（L4 按 L3 流程演示，双核计时未实现）
  F            进入/退出装配流程（难度选定后按 F 开始）
  G            自动对准（能量单元开口对准销轴，插入深度 0）
  W/S          装配步骤2/3：插入 / 上移（或自由模式平移）
  A/D          步骤4/5：绕P旋转 / 绕Q旋转（或自由模式平移）
  X            取出能量单元（装配失败）
  Enter        步骤6：确认装配
  [ / ]        调整步骤5 指定角度（目标 Q 角）
  Backspace    全部重置
  P / H / Esc  截图 / HUD / 退出
"""
import sys, os, math, time, ctypes
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import glfw
from OpenGL import GL
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mujoco
from robot_ik import IKSolver, grasp_check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "tech_core_joints.xml")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ---------- 关节/站点 id ----------
J = {}
for i in range(model.njnt):
    J[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)] = model.jnt_qposadr[i]
SID_PIN = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pin_tip")
SID_GRASP = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp")
ik = IKSolver(model, data)
IK_ERR = [0.0]
EU = model.body('energy_unit')
EU_MOCAP = int(EU.mocapid)

# 销轴几何（q=0，world）
E_LOCAL = np.array([-0.1293, -0.333, 0.7932])

# ---------- 规则坐标系映射 ----------
# rule -> world: (x,y,z)_rule -> (y, -x, z)_world
def rule_to_world(v):
    return np.array([v[1], -v[0], v[2]])

def e_dir_world(theta_deg, phi_deg):
    """e⃗(θ,φ) 的 world 方向（指向机器人为正）"""
    th, ph = np.radians([theta_deg, phi_deg])
    return np.array([np.sin(ph)*np.sin(th), -np.sin(ph)*np.cos(th), np.cos(ph)])

ARM_J = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
ARM_NAME = ['J1底座回转', 'J2肩俯仰', 'J3肘俯仰', 'J4腕回转', 'J5腕俯仰', 'J6法兰回转']

def zero_arm():
    for j in ARM_J:
        data.qpos[J[j]] = 0.0

ROBOT_J = ['j_r1', 'j_r2', 'j_r3', 'j_r4', 'j_r5', 'j_r6']
ROBOT_NAME = ['r1底座', 'r2肩', 'r3肘', 'r4腕转', 'r5腕俯仰', 'r6法兰']

def zero_robot():
    for j in ROBOT_J:
        data.qpos[J[j]] = 0.0

def finger_gap():
    """指距 = gap0 - (q_a + q_b)（q+ 向内=闭合）"""
    return 0.0257 - (data.qpos[J['j_rfa']] + data.qpos[J['j_rfb']])

def drive_fingers(closing, opening):
    """C 闭合 / O 张开；限位保证指距 5mm..115mm"""
    qa, qb = data.qpos[J['j_rfa']], data.qpos[J['j_rfb']]
    d = 0.0015
    if closing:
        if finger_gap() > 0.005:
            qa += d; qb += d
    if opening:
        if finger_gap() < 0.115:
            qa -= d; qb -= d
    lo, hi = model.jnt_range[J['j_rfa']]
    data.qpos[J['j_rfa']] = min(hi, max(lo, qa))
    lo, hi = model.jnt_range[J['j_rfb']]
    data.qpos[J['j_rfb']] = min(hi, max(lo, qb))

def arm_pose_readout():
    """从运动学正解读出规则位姿 (x,y,z,θ,φ,α)（mm/°）"""
    ppos, pquat, pR = pin_frame()
    e = pR @ np.array([0.0, -1.0, 0.0])
    Pr = world_to_rule(ppos)
    er = world_to_rule(e)
    phi = math.degrees(math.acos(max(-1.0, min(1.0, er[2]))))
    theta = math.degrees(math.atan2(er[1], er[0]))
    # α：鼓绕 e⃗ 相对最小扭转参考系的滚转角
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_drum")
    Rd = data.xmat[bid].reshape(3, 3)
    a0 = np.array([0.0, -1.0, 0.0])
    v = np.cross(a0, e); c = float(np.dot(a0, e))
    if np.linalg.norm(v) < 1e-8:
        Rref = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rref = np.eye(3) + vx + vx @ vx * (1.0/(1.0+c))
    Rrel = Rref.T @ Rd
    alpha = -math.degrees(math.atan2(Rrel[0, 2], Rrel[0, 0]))
    return Pr[0]*1000, Pr[1]*1000, Pr[2]*1000, theta, phi, alpha

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def axis_quat(ax, deg):
    a = math.radians(deg)/2; s = math.sin(a)
    return np.array([math.cos(a), ax[0]*s, ax[1]*s, ax[2]*s])

def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def mat_to_quat(R):
    q = np.empty(4)
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        s = math.sqrt(tr+1.0)*2
        q[0] = 0.25*s; q[1] = (R[2,1]-R[1,2])/s; q[2] = (R[0,2]-R[2,0])/s; q[3] = (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        q[0] = (R[2,1]-R[1,2])/s; q[1] = 0.25*s; q[2] = (R[0,1]+R[1,0])/s; q[3] = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        q[0] = (R[0,2]-R[2,0])/s; q[1] = (R[0,1]+R[1,0])/s; q[2] = 0.25*s; q[3] = (R[1,2]+R[2,1])/s
    else:
        s = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        q[0] = (R[1,0]-R[0,1])/s; q[1] = (R[0,2]+R[2,0])/s; q[2] = (R[1,2]+R[2,1])/s; q[3] = 0.25*s
    return q/np.linalg.norm(q)

def pin_frame():
    """销轴尖端站点的世界位姿 (pos, quat, R)"""
    pos = data.site_xpos[SID_PIN].copy()
    R = data.site_xmat[SID_PIN].reshape(3, 3).copy()
    return pos, mat_to_quat(R), R

# ---------- 装配状态 ----------
STEP_NAMES = {
    1: "步骤1 选择难度",
    2: "步骤2 插入100mm (沿-x)",
    3: "步骤3 上移100mm (沿+z)",
    4: "步骤4 绕P轴转90°",
    5: "步骤5 绕Q轴转至指定角度",
    6: "步骤6 确认装配 (Enter)",
}
LEVEL_REQ = {
    1: [2, 6],
    2: [2, 3, 6],
    3: [2, 3, 4, 5, 6],
    4: [2, 3, 4, 5, 6],   # 双核同步计时未实现，按 L3 流程演示
}

class State:
    def __init__(self):
        self.reset_all()

    def reset_all(self):
        self.pose = dict(x=-0.2, y=0.2, z=0.8, theta=0.0, phi=90.0, alpha=0.0)
        self.difficulty = 1
        self.assembly = False        # 装配流程进行中
        self.step = 1
        self.insert = 0.0            # 步骤2 插入深度 (m)
        self.lift = 0.0              # 步骤3 上移量 (m)
        self.q_target = 30.0         # 步骤5 指定角度 (deg)
        self.latched = False         # 能量单元已锁定（pin 或 gripper）
        self.attach = ""             # "" / "pin" / "gripper"
        self.T_rel = None            # 锁定时 unit 相对站点的变换 (pos, R)
        self.T_grasp = None          # 最近一次夹爪抓持变换（释放后供手臂影随）
        self.align_ref = None        # 对准起点 (unit pos, R)
        self.hold = False            # 步骤5 保持 (0.5N·m)
        self.dwell = 0.0             # 步骤5 目标角度停留时间
        self.result = ""             # SUCCESS / FAIL
        self.q_hold_angle = 0.0
        zero_arm()
        zero_robot()
        data.qpos[J['j_rfa']] = -0.045
        data.qpos[J['j_rfb']] = -0.045   # 初始张开
        e0 = np.array([0.0, -1.0, 0.0])   # 零位销轴方向
        data.mocap_pos[EU_MOCAP] = E_LOCAL + e0*0.35
        data.mocap_quat[EU_MOCAP] = np.array([0.70710678, 0.70710678, 0, 0])  # 顶盖朝上、开口朝下

    def step_list(self):
        return LEVEL_REQ[self.difficulty]

    def advance(self):
        req = self.step_list()
        if self.step in req:
            i = req.index(self.step)
            self.step = req[i+1] if i+1 < len(req) else 6
        else:
            self.step = req[0]

S = State()

# ---------- GLFW ----------
if not glfw.init():
    raise RuntimeError("glfw init failed")
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
glfw.window_hint(glfw.SAMPLES, 4)
window = glfw.create_window(1200, 800, "科技核心-能量单元 六步装配", None, None)
if not window:
    raise RuntimeError("window creation failed")
glfw.make_context_current(window)
glfw.swap_interval(1)
try:
    ctypes.windll.imm32.ImmAssociateContext(glfw.get_win32_window(window), 0)
except Exception:
    pass

scene = mujoco.MjvScene(model, maxgeom=20000)
opt = mujoco.MjvOption()
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)
cam.distance = 2.4
cam.azimuth = 120
cam.elevation = -15
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
HUD_FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)

last_x, last_y = [0.0], [0.0]
btn_left, btn_right, btn_middle = [False], [False], [False]
MODES = ["unit", "arm", "robot", "drive", "camera"]
MODE_NAME = {"unit": "能量单元", "arm": "科技核心臂", "robot": "工程臂", "drive": "驾驶", "camera": "相机"}
DRIVE_SPD = [0.0]      # 轮速目标 rad/s
DRIVE_STEER = [0.0]    # 差速转向
LIFT_TGT = [0.0]       # 升降目标 m (0=收, 上正)
LOCK_NAMES = ['j1','j2','j3','j4','j5','j6','j_slide','j_p','j_q',
              'j_r1','j_r2','j_r3','j_r4','j_r5','j_r6','j_rfa','j_rfb']
LOCK_ADR = np.array([J[j] for j in LOCK_NAMES])
LOCK_DOF = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in LOCK_NAMES])
mode = [0]
show_hud = [True]
pending = set()
held = set()
t0 = [time.time()]; frames = [0]; fps = [0.0]; shot = [0]

MEASURE = {"on": False, "A": None, "B": None}   # 测量工具（M 开关，N 清空）
pose_buf = [None]                               # 数字输入六关节角缓冲（T 开启）
robot_buf = [None]                              # 工程臂六关节角输入缓冲
ARM_SEL = [0]                                   # 科技核心臂模式当前选中关节 0..5
ROBOT_SEL = [0]                                 # 工程臂模式当前选中关节 0..5

MOVE = 0.004      # 每帧平移
ROT = 1.2         # 每帧旋转°

def key_cb(w, key, sc, action, mods):
    if action == glfw.PRESS:
        if key not in held:          # 屏蔽系统按键重复，防止 F 等被重复触发
            held.add(key)
            pending.add(key)
    elif action == glfw.RELEASE:
        held.discard(key)

def char_cb(w, codepoint):
    ch = chr(codepoint)
    for buf in (pose_buf, robot_buf):
        if buf[0] is not None and (ch.isdigit() or ch in " -.+"):
            if len(buf[0]) < 64:
                buf[0] += ch

def mouse_button_cb(w, button, action, mods):
    if button == glfw.MOUSE_BUTTON_LEFT:
        btn_left[0] = (action == glfw.PRESS)
        if action == glfw.PRESS and MEASURE["on"]:
            p = pick_point(last_x[0], last_y[0])
            if p is not None:
                if MEASURE["A"] is None or MEASURE["B"] is not None:
                    MEASURE["A"] = p
                    MEASURE["B"] = None
                    print(f"[测量] A = {np.round(p*1000,1).tolist()} mm (world)")
                else:
                    MEASURE["B"] = p
                    d = np.linalg.norm(MEASURE["B"] - MEASURE["A"])
                    print(f"[测量] B = {np.round(p*1000,1).tolist()} mm, |AB| = {d*1000:.1f} mm")
    elif button == glfw.MOUSE_BUTTON_RIGHT: btn_right[0] = (action == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_MIDDLE: btn_middle[0] = (action == glfw.PRESS)

def cursor_pos_cb(w, x, y):
    dx = x - last_x[0]; dy = y - last_y[0]
    m = MODES[mode[0]]
    if MEASURE["on"]:
        last_x[0] = x; last_y[0] = y
        return   # 测量模式：左键取点，不拖动
    if m == 'camera':
        if btn_left[0]:
            cam.azimuth -= dx * 0.4
            cam.elevation = max(-89.0, min(89.0, cam.elevation - dy*0.4))
        elif btn_right[0]:
            az = math.radians(cam.azimuth)
            s = cam.distance * 0.0015
            cam.lookat[0] += (math.sin(az)*dx - math.cos(az)*dy) * s
            cam.lookat[1] += (-math.cos(az)*dx - math.sin(az)*dy) * s
        elif btn_middle[0]:
            cam.distance = max(0.05, min(50.0, cam.distance*math.exp(dy*0.01)))
    elif m == 'unit':
        if S.latched and S.attach == "pin":
            pass  # 锁定到销轴后跟随机构，不能拖
        else:
            s = cam.distance * 0.003
            if btn_left[0]:
                data.mocap_pos[EU_MOCAP][0] += dx*s
                data.mocap_pos[EU_MOCAP][1] -= dy*s
                if S.align_ref is not None:
                    S.align_ref = None   # 手动拖动后需重新对准
                    S.insert = 0.0
            elif btn_right[0]:
                data.mocap_quat[EU_MOCAP] = quat_mult(axis_quat((0,0,1), -dx*0.4), data.mocap_quat[EU_MOCAP])
            elif btn_middle[0]:
                data.mocap_pos[EU_MOCAP][2] += dy*s
            q = data.mocap_quat[EU_MOCAP]
            data.mocap_quat[EU_MOCAP] = q/np.linalg.norm(q)
    last_x[0] = x; last_y[0] = y

def scroll_cb(w, xo, yo):
    m = MODES[mode[0]]
    if m == 'camera':
        cam.distance = max(0.05, min(50.0, cam.distance*math.exp(-yo*0.1)))
    elif m == 'unit' and not (S.latched and S.attach == "pin"):
        data.mocap_pos[EU_MOCAP][2] += yo * cam.distance * 0.002

glfw.set_key_callback(window, key_cb)
glfw.set_char_callback(window, char_cb)
glfw.set_mouse_button_callback(window, mouse_button_cb)
glfw.set_cursor_pos_callback(window, cursor_pos_cb)
glfw.set_scroll_callback(window, scroll_cb)

# ---------- 测量工具（SolidWorks 式：取两点，显示斜边+直角边） ----------
def world_to_rule(v):
    """world -> rule OXYZ: (x,y,z)_rule = (-wy, wx, wz)_world"""
    return np.array([-v[1], v[0], v[2]])

def pick_point(mx, my):
    """鼠标位置拾取场景表面 3D 点（mjv_select）"""
    w, h = glfw.get_window_size(window)
    if w <= 0 or h <= 0:
        return None
    selpnt = np.zeros(3)
    geomid = np.array([0], dtype=np.int32)
    flexid = np.array([0], dtype=np.int32)
    skinid = np.array([0], dtype=np.int32)
    try:
        gid = mujoco.mjv_select(model, data, opt, w/max(h, 1),
                                mx/w, 1.0 - my/h, scene, selpnt, geomid, flexid, skinid)
    except Exception:
        return None
    if gid < 0:
        return None
    return selpnt.copy()

def draw_measure():
    """场景里画测点、斜边（橙）和直角边折线（灰，rule 笛卡尔坐标系）"""
    if MEASURE["A"] is None or scene.ngeom + 8 > scene.maxgeom:
        return
    A = MEASURE["A"]
    def sphere(p, rgba):
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.008, 0, 0]), np.array(p),
                            np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
        scene.ngeom += 1
    def link(p1, p2, rgba, width):
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.array(p1, dtype=np.float64), np.array(p2, dtype=np.float64))
        g.rgba[:] = rgba
        scene.ngeom += 1
    sphere(A, (1.0, 0.55, 0.0, 1.0))
    if MEASURE["B"] is not None:
        B = MEASURE["B"]
        sphere(B, (1.0, 0.55, 0.0, 1.0))
        link(A, B, (1.0, 0.55, 0.0, 1.0), 0.004)          # 斜边
        Ar, Br = world_to_rule(A), world_to_rule(B)
        c1 = np.array([Br[0], Ar[1], Ar[2]])               # 直角折线（rule 轴向）
        c2 = np.array([Br[0], Br[1], Ar[2]])
        p1, p2 = rule_to_world(c1), rule_to_world(c2)
        for u, v in ((A, p1), (p1, p2), (p2, B)):
            link(u, v, (0.85, 0.85, 0.85, 0.9), 0.002)      # 直角边

def unit_aligned():
    """能量单元正面开口(-Y 体轴, Ø40 口)是否正对销轴（口朝鼓面，横向偏差小）
    结构（手册图 4-39）：+Y=Ø95 平顶盖(背面, 封闭)；-Y=Ø60 正面, 中央 Ø40 开口, 腔内 100mm 深处铁片"""
    ppos, pquat, pR = pin_frame()
    uR = quat_to_mat(data.mocap_quat[EU_MOCAP])
    e = pR @ np.array([0.0, -1.0, 0.0])          # e⃗ 指向机器人
    um = uR @ np.array([0.0, -1.0, 0.0])         # 正面开口法向 = 体轴 -Y
    ang = np.degrees(np.arccos(np.clip(np.dot(um, -e), -1, 1)))
    d = data.mocap_pos[EU_MOCAP] - ppos
    lateral = np.linalg.norm(d - np.dot(d, e)*e)
    return ang < 25 and lateral < 0.06, ang, lateral

def auto_align():
    """自动对准：正面 Ø40 开口(-Y)正对销轴，口面贴销尖，插入深度 0"""
    ppos, pquat, pR = pin_frame()
    e = pR @ np.array([0.0, -1.0, 0.0])
    y_col = e                                    # 体轴 +Y(背面/顶盖) 朝外；-Y(开口) 朝销轴
    z_col = np.array([0.0, 0.0, 1.0])
    z_col = z_col - np.dot(z_col, y_col)*y_col
    if np.linalg.norm(z_col) < 1e-6:
        z_col = np.array([1.0, 0.0, 0.0])
        z_col = z_col - np.dot(z_col, y_col)*y_col
    z_col /= np.linalg.norm(z_col)
    x_col = np.cross(y_col, z_col)
    R = np.stack([x_col, y_col, z_col], axis=1)
    data.mocap_quat[EU_MOCAP] = mat_to_quat(R)
    data.mocap_pos[EU_MOCAP] = ppos + e * 0.075  # 口面(距中心75mm)贴销尖，插入=0
    S.align_ref = (data.mocap_pos[EU_MOCAP].copy(), R.copy())
    S.insert = 0.0

def src_frame(src):
    """锁定源站点的世界位姿 (pos, R)：pin -> pin_tip；gripper -> grasp"""
    sid = SID_PIN if src == "pin" else SID_GRASP
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()

def attach_unit(src):
    """锁定能量单元到 pin 或 gripper（互斥）；gripper 时记住 T_grasp 供影随"""
    ppos, pR = src_frame(src)
    uR = quat_to_mat(data.mocap_quat[EU_MOCAP])
    S.T_rel = (pR.T @ (data.mocap_pos[EU_MOCAP] - ppos), pR.T @ uR)
    S.latched = True
    S.attach = src
    if src == "gripper":
        S.T_grasp = (S.T_rel[0].copy(), S.T_rel[1].copy())

def apply_latched():
    ppos, pR = src_frame(S.attach)
    tp, tR = S.T_rel
    data.mocap_pos[EU_MOCAP] = ppos + pR @ tp
    data.mocap_quat[EU_MOCAP] = mat_to_quat(pR @ tR)

def hand_target_from_unit():
    """由单元位姿反推 grasp 站点目标位姿 (pos, R)（IK 用）"""
    tp, tR = S.T_rel
    uR = quat_to_mat(data.mocap_quat[EU_MOCAP])
    Rt = uR @ tR.T
    pt = data.mocap_pos[EU_MOCAP] - Rt @ tp
    return pt, Rt

def reset_assembly(keep_pose=True):
    arm_q = [data.qpos[J[j]] for j in ARM_J]
    S.reset_all()
    if keep_pose:
        for j in ARM_J:
            data.qpos[J[j]] = arm_q[ARM_J.index(j)]

HELP = ("Tab=切换(能量单元/科技核心臂/工程臂/相机)  1/2/3/4=难度  F=开始装配  G=自动对准\n"
        "装配: W/S=插入·上移  A/D=绕P·绕Q  [ ]=指定角度  X=取出  Enter=确认\n"
        "科技核心臂: ←→选关节 ↑↓=±1° R=归零 T=输入角   工程臂: 同左（C=夹爪闭合 O=张开）\n"
        "测量: M=开/关 点击两点出|AB|斜边+Δx/Δy/Δz直角边  N=清测点\n"
        "相机: 左键=旋转 右键=平移 滚轮=缩放   Backspace=重置  P=截图  H=HUD  Esc=退出")

def render_hud(info_lines):
    fbw, fbh = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, fbw, fbh)
    rgb = np.empty((fbh, fbw, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, viewport, ctx)
    img = Image.fromarray(rgb[::-1].copy()).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    y = 8
    for ln in (HELP.split('\n') + ['-'*46] + info_lines):
        od.text((14, y), ln, font=HUD_FONT, fill=(255, 255, 255, 255))
        y += 28
    img = Image.alpha_composite(img, overlay).convert('RGB')
    mujoco.mjr_drawPixels(np.array(img)[::-1].reshape(-1), None, viewport, ctx)

# ---------- 主循环 ----------
last_step = [1]
last_result = [""]
while not glfw.window_should_close(window):
    dt = 1.0/60.0

    # ---- 数字输入六关节角 (°)，Enter 确认，Esc 取消（科技核心臂 / 工程臂 双缓冲）----
    for buf, jlist, tag in ((pose_buf, ARM_J, "arm"), (robot_buf, ROBOT_J, "robot")):
        if buf[0] is not None:
            if glfw.KEY_BACKSPACE in pending:
                buf[0] = buf[0][:-1]
            if glfw.KEY_ESCAPE in pending:
                buf[0] = None
                print(f"[{tag}] 已取消输入")
            if glfw.KEY_ENTER in pending:
                parts = buf[0].split()
                if len(parts) == 6:
                    try:
                        v = [float(p) for p in parts]
                        for j, a in zip(jlist, v):
                            lo, hi = model.jnt_range[J[j]]
                            data.qpos[J[j]] = np.radians(max(np.degrees(lo), min(np.degrees(hi), a)))
                        print(f"[{tag}] 关节角设定: {v}")
                    except ValueError:
                        print(f"[{tag}] 数字解析失败，格式: 6 个角度（度）")
                else:
                    print(f"[{tag}] 需要 6 个数字（度）")
                buf[0] = None
            pending.clear()

    if glfw.KEY_ESCAPE in pending: break
    if glfw.KEY_TAB in pending: mode[0] = (mode[0]+1) % 4
    if glfw.KEY_H in pending: show_hud[0] = not show_hud[0]
    if glfw.KEY_T in pending and pose_buf[0] is None and robot_buf[0] is None:
        if MODES[mode[0]] == "robot":
            robot_buf[0] = ""
            print("[robot] 输入工程臂六关节角 (°)，Enter 确认，Esc 取消")
        else:
            pose_buf[0] = ""
            mode[0] = 1
            print("[arm] 输入六关节角 J1 J2 J3 J4 J5 J6 (°)，如 0 30 -20 0 0 0，Enter 确认，Esc 取消")
    if glfw.KEY_M in pending:
        MEASURE["on"] = not MEASURE["on"]
        print(f"[测量] {'开：点击物体表面取 A/B 两点' if MEASURE['on'] else '关'}")
    if glfw.KEY_N in pending and (MEASURE["A"] is not None or MEASURE["B"] is not None):
        MEASURE["A"] = None
        MEASURE["B"] = None
        print("[测量] 测点已清空")
    for k, d in ((glfw.KEY_1, 1), (glfw.KEY_2, 2), (glfw.KEY_3, 3), (glfw.KEY_4, 4)):
        if k in pending:
            if S.difficulty != d:
                S.difficulty = d
                S.result = ""
    if glfw.KEY_F in pending:
        # F = 开始 / 重新开始装配（不再做开/关切换，避免误触重置）
        S.assembly = True
        S.step = 2
        S.result = ""
        S.insert = 0.0; S.lift = 0.0; S.latched = False; S.T_rel = None; S.attach = ""
        S.hold = False; S.dwell = 0.0
        data.qpos[J['j_slide']] = 0.0
        data.qpos[J['j_p']] = 0.0
        data.qpos[J['j_q']] = 0.0
        print(f"[F] 开始装配 L{S.difficulty} -> 步骤2，按 G 对准、按 W 插入")
    if glfw.KEY_G in pending and S.assembly and not S.latched:
        auto_align()
        print("[G] 自动对准：开口已对销轴，插入 0mm，按 W 插入")
    if glfw.KEY_BACKSPACE in pending:
        reset_assembly(keep_pose=False)
    if glfw.KEY_P in pending:
        fbw, fbh = glfw.get_framebuffer_size(window)
        rgb = np.empty((fbh, fbw, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, fbw, fbh), ctx)
        shot[0] += 1
        Image.fromarray(rgb[::-1]).save(os.path.join(OUTPUT_DIR, f"asm_shot_{shot[0]:03d}.png"))
        print(f"saved asm_shot_{shot[0]:03d}.png")
    if glfw.KEY_ENTER in pending and S.assembly and S.step == 6 and not S.result:
        S.result = "SUCCESS"
    if glfw.KEY_X in pending and S.assembly and S.latched and not S.result:
        S.result = "FAIL(extracted)"
        S.latched = False
        S.T_rel = None

    pending.clear()

    keys = lambda k: glfw.get_key(window, k) == glfw.PRESS

    # ---- 指定角度微调 ----
    if keys(glfw.KEY_LEFT_BRACKET): S.q_target = max(-90, S.q_target - 1.5)
    if keys(glfw.KEY_RIGHT_BRACKET): S.q_target = min(90, S.q_target + 1.5)

    m = MODES[mode[0]]
    if m == 'unit' and not (S.latched and S.attach == "pin") and not (S.assembly and S.step in (2, 3, 4, 5)):
        # 自由模式键盘移动/旋转（与 view_model.py 一致）
        dpos = np.zeros(3); dquat = np.array([1.0, 0, 0, 0])
        if keys(glfw.KEY_W): dpos[0] += MOVE
        if keys(glfw.KEY_S): dpos[0] -= MOVE
        if keys(glfw.KEY_D): dpos[1] += MOVE
        if keys(glfw.KEY_A): dpos[1] -= MOVE
        if keys(glfw.KEY_E): dpos[2] += MOVE
        if keys(glfw.KEY_Q): dpos[2] -= MOVE
        if keys(glfw.KEY_RIGHT): dquat = quat_mult(axis_quat((0,0,1),  ROT), dquat)
        if keys(glfw.KEY_LEFT):  dquat = quat_mult(axis_quat((0,0,1), -ROT), dquat)
        if keys(glfw.KEY_UP):    dquat = quat_mult(axis_quat((0,1,0),  ROT), dquat)
        if keys(glfw.KEY_DOWN):  dquat = quat_mult(axis_quat((0,1,0), -ROT), dquat)
        if keys(glfw.KEY_Z):     dquat = quat_mult(axis_quat((1,0,0),  ROT), dquat)
        if keys(glfw.KEY_X):     dquat = quat_mult(axis_quat((1,0,0), -ROT), dquat)
        if np.any(dpos) or dquat[0] != 1.0:
            data.mocap_pos[EU_MOCAP] = data.mocap_pos[EU_MOCAP] + dpos
            data.mocap_quat[EU_MOCAP] = quat_mult(dquat, data.mocap_quat[EU_MOCAP])
            data.mocap_quat[EU_MOCAP] /= np.linalg.norm(data.mocap_quat[EU_MOCAP])
            if S.align_ref is not None:
                S.align_ref = None
                S.insert = 0.0
    if m == 'arm' and not (S.assembly and S.latched):
        # 科技核心臂六轴点动：←→选关节，↑↓±1°，R 全部归零
        if keys(glfw.KEY_LEFT):  ARM_SEL[0] = (ARM_SEL[0] - 1) % 6
        if keys(glfw.KEY_RIGHT): ARM_SEL[0] = (ARM_SEL[0] + 1) % 6
        jn = ARM_J[ARM_SEL[0]]
        lo, hi = model.jnt_range[J[jn]]
        if keys(glfw.KEY_UP):
            data.qpos[J[jn]] = min(hi, data.qpos[J[jn]] + math.radians(1.0))
        if keys(glfw.KEY_DOWN):
            data.qpos[J[jn]] = max(lo, data.qpos[J[jn]] - math.radians(1.0))
        if keys(glfw.KEY_R):
            zero_arm()
    if m == 'robot':
        # 工程臂六轴点动：←→选关节，↑↓±1°，R 全部归零
        if keys(glfw.KEY_LEFT):  ROBOT_SEL[0] = (ROBOT_SEL[0] - 1) % 6
        if keys(glfw.KEY_RIGHT): ROBOT_SEL[0] = (ROBOT_SEL[0] + 1) % 6
        jn = ROBOT_J[ROBOT_SEL[0]]
        lo, hi = model.jnt_range[J[jn]]
        if keys(glfw.KEY_UP):
            data.qpos[J[jn]] = min(hi, data.qpos[J[jn]] + math.radians(1.0))
        if keys(glfw.KEY_DOWN):
            data.qpos[J[jn]] = max(lo, data.qpos[J[jn]] - math.radians(1.0))
        if keys(glfw.KEY_R):
            zero_robot()

    # ---- 夹爪：C 闭合 / O 张开（驾驶模式除外，O=升降降）+ 抓取/释放边沿检测 ----
    in_drive = MODES[mode[0]] == "drive"
    drive_fingers(keys(glfw.KEY_C), keys(glfw.KEY_O) and not in_drive)
    if S.attach == "" and keys(glfw.KEY_C):
        uR = quat_to_mat(data.mocap_quat[EU_MOCAP])
        ok_g, dist, gap, ang = grasp_check(model, data, finger_gap(),
                                           data.mocap_pos[EU_MOCAP], SID_GRASP, uR)
        if ok_g:
            attach_unit("gripper")
            print(f"[抓取] 锁定到夹爪 dist={dist*1000:.0f}mm gap={gap*1000:.0f}mm")
    if S.attach == "gripper" and keys(glfw.KEY_O) and finger_gap() > 0.105:
        S.latched = False; S.attach = ""; S.T_rel = None
        print("[释放] 夹爪张开，能量单元释放")
        if S.assembly and not S.result:
            S.result = "FAIL(extracted)"

    # ---- 装配步骤控制 ----
    if S.assembly and not S.result:
        if S.step == 2:
            if not S.latched:
                # 自由对准 + 插入
                ok, ang, lat = unit_aligned()
                if ok and S.align_ref is not None:
                    e = pin_frame()[2] @ np.array([0.0, -1.0, 0.0])
                    dpos = data.mocap_pos[EU_MOCAP] - S.align_ref[0]
                    S.insert = float(np.clip(-np.dot(dpos, e), 0, 0.1))
                elif S.align_ref is not None and not ok:
                    S.insert = 0.0
                if keys(glfw.KEY_W) or keys(glfw.KEY_S):
                    if S.align_ref is None:
                        auto_align()
                    ok, ang, lat = unit_aligned()
                    if ok:
                        e = pin_frame()[2] @ np.array([0.0, -1.0, 0.0])
                        step_dir = -e * (0.0025 if keys(glfw.KEY_W) else -0.0025)
                        data.mocap_pos[EU_MOCAP] += step_dir
                        dpos = data.mocap_pos[EU_MOCAP] - S.align_ref[0]
                        S.insert = float(np.clip(-np.dot(dpos, e), 0, 0.1))
                if S.insert >= 0.0999:
                    S.insert = 0.1
                    attach_unit("pin")
                    drive_fingers(False, True)   # 张开释放（互斥）
                    S.step = 3 if 3 in S.step_list() else 6
        elif S.step == 3:
            if keys(glfw.KEY_W): S.lift = min(0.1, S.lift + 0.0025)
            if keys(glfw.KEY_S): S.lift = max(0.0, S.lift - 0.0025)
            data.qpos[J['j_slide']] = S.lift
            if S.lift >= 0.0999:
                S.lift = 0.1
                S.step = 4 if 4 in S.step_list() else 6
        elif S.step == 4:
            q = np.degrees(data.qpos[J['j_p']])
            if keys(glfw.KEY_D): q = min(90, q + 0.8)
            if keys(glfw.KEY_A): q = max(0, q - 0.8)
            data.qpos[J['j_p']] = np.radians(q)
            if q >= 89.5:
                data.qpos[J['j_p']] = np.radians(90.0)
                S.step = 5
        elif S.step == 5:
            q = np.degrees(data.qpos[J['j_q']])
            manual = False
            if keys(glfw.KEY_D): q = min(90, q + 0.8); manual = True
            if keys(glfw.KEY_A): q = max(-90, q - 0.8); manual = True
            if not manual and not S.hold:
                # 弹簧回正 0.3N·m：未施力时向 0 衰减
                q += (0.0 - q) * 0.04
            data.qpos[J['j_q']] = np.radians(q)
            if abs(q - S.q_target) <= 4.0:
                S.dwell += dt
                if S.dwell >= 2.0:
                    S.hold = True
                if S.dwell >= 0.3 and not S.result:
                    S.step = 6
            else:
                S.dwell = 0.0
                S.hold = False

    # ---- 主从/IK：抓持期 IK 跟随；释放后手臂影随单元（机器人"完成"装配） ----
    if S.latched and not S.result:
        if S.attach == "gripper" and MODES[mode[0]] == "robot":
            apply_latched()                      # 臂主：单元跟随夹爪
        elif S.attach == "pin":
            apply_latched()                      # 单元随销轴/机构
            if S.T_grasp is not None and MODES[mode[0]] != "robot":
                tp, tR = S.T_grasp
                uR = quat_to_mat(data.mocap_quat[EU_MOCAP])
                Rt = uR @ tR.T
                IK_ERR[0] = ik.solve(data.mocap_pos[EU_MOCAP] - Rt @ tp, Rt, iters=2)
        else:
            tp, tR = S.T_grasp if S.T_grasp is not None else S.T_rel
            uR = quat_to_mat(data.mocap_quat[EU_MOCAP])
            Rt = uR @ tR.T
            pt = data.mocap_pos[EU_MOCAP] - Rt @ tp
            IK_ERR[0] = ik.solve(pt, Rt, iters=3)   # 单元主：IK 跟随
        if S.step == 2:
            S.step = 3 if 3 in S.step_list() else 6

    # 状态转换日志（输出到终端，便于排查）
    if S.step != last_step[0]:
        print(f"[step] -> 步骤{S.step}  {STEP_NAMES.get(S.step, '')}")
        last_step[0] = S.step
    if S.result != last_result[0]:
        if S.result:
            print(f"[result] {S.result}")
        last_result[0] = S.result

    # ---- 驾驶模式：动力学步进（车轮/升降动，臂与科技核心关节运动学锁定）----
    if MODES[mode[0]] == "drive":
        if keys(glfw.KEY_I): DRIVE_SPD[0] = min(25.0, DRIVE_SPD[0] + 0.6)
        if keys(glfw.KEY_K): DRIVE_SPD[0] = max(-25.0, DRIVE_SPD[0] - 0.6)
        if not (keys(glfw.KEY_I) or keys(glfw.KEY_K)): DRIVE_SPD[0] *= 0.985
        if keys(glfw.KEY_J): DRIVE_STEER[0] = min(12.0, DRIVE_STEER[0] + 0.5)
        if keys(glfw.KEY_L): DRIVE_STEER[0] = max(-12.0, DRIVE_STEER[0] - 0.5)
        if not (keys(glfw.KEY_J) or keys(glfw.KEY_L)): DRIVE_STEER[0] *= 0.92
        if keys(glfw.KEY_U): LIFT_TGT[0] = min(0.15, LIFT_TGT[0] + 0.002)
        if keys(glfw.KEY_O): LIFT_TGT[0] = max(0.0, LIFT_TGT[0] - 0.002)
        for k in range(4):
            data.ctrl[k] = DRIVE_SPD[0] + DRIVE_STEER[0] * (1 if k % 2 == 0 else -1)
        data.ctrl[4] = DRIVE_SPD[0] * 0.6   # 前小轮
        data.ctrl[5] = DRIVE_SPD[0] * 0.6   # 后小轮
        data.ctrl[6] = LIFT_TGT[0]
        data.ctrl[7] = LIFT_TGT[0]
        saved = data.qpos[LOCK_ADR].copy()
        for _ in range(2):
            mujoco.mj_step(model, data)
            data.qpos[LOCK_ADR] = saved
            data.qvel[LOCK_DOF] = 0.0
    else:
        mujoco.mj_forward(model, data)

    # ---- 渲染 ----
    fbw, fbh = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, fbw, fbh)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    draw_measure()
    mujoco.mjr_render(viewport, scene, ctx)

    if show_hud[0]:
        frames[0] += 1
        now = time.time()
        if now - t0[0] >= 0.5:
            fps[0] = frames[0]/(now-t0[0]); frames[0] = 0; t0[0] = now
        px, py, pz, pth, pph, pal = arm_pose_readout()
        ok, ang, lat = unit_aligned()
        qq = np.degrees(data.qpos[J['j_q']])
        pp = np.degrees(data.qpos[J['j_p']])
        res = f"  ★{S.result}" if S.result else ""
        if not S.assembly:
            nxt = "▶ 按 F 开始装配流程（按 1/2/3/4 选难度）"
        elif S.result:
            nxt = "▶ 按 F 重新开始，或 Backspace 全部重置"
        elif S.step == 2 and not S.latched:
            nxt = f"▶ 按 G 自由对准，再按住 W 插入 100mm（现在{S.insert*1000:.0f}mm）"
        elif S.step == 3:
            nxt = f"▶ 按住 W 上移 100mm（现在{S.lift*1000:.0f}mm）"
        elif S.step == 4:
            nxt = f"▶ 按住 D 绕 P 轴转到 90°（现在{np.degrees(data.qpos[J['j_p']]):.0f}°）"
        elif S.step == 5:
            nxt = f"▶ 按 A/D 绕 Q 转到 {S.q_target:+.0f}°（现在{np.degrees(data.qpos[J['j_q']]):+.0f}°），停 0.3s"
        elif S.step == 6:
            nxt = "▶ 按 Enter 确认装配"
        else:
            nxt = ""
        info = [
            nxt,
            f"FPS {fps[0]:.0f}  控制: {MODE_NAME[m]}  难度: L{S.difficulty}  "
            f"{'装配中' if S.assembly else '待命'}{res}",
            f"位姿读数: x={px:.0f} y={py:.0f} z={pz:.0f}mm  θ={pth:.1f}° φ={pph:.1f}° α={pal:.1f}°",
            "关节: " + "  ".join(
                (">" if i == ARM_SEL[0] else "") + f"{ARM_NAME[i][0:2]}={np.degrees(data.qpos[J[ARM_J[i]]]):+.0f}°"
                for i in range(6)),
            "工程臂: " + "  ".join(
                (">" if i == ROBOT_SEL[0] else "") + f"{ROBOT_NAME[i][0:2]}={np.degrees(data.qpos[J[ROBOT_J[i]]]):+.0f}°"
                for i in range(6)),
            f"驾驶: 轮速 {DRIVE_SPD[0]:+.1f}rad/s  转向 {DRIVE_STEER[0]:+.1f}  升降 {LIFT_TGT[0]*1000:.0f}mm（I/K J/L U/O）",
            f"夹爪: 指距 {finger_gap()*1000:.0f}mm  锁定: {('无','销轴','夹爪')[['', 'pin', 'gripper'].index(S.attach)] if S.attach in ('', 'pin', 'gripper') else S.attach}"
            f"  IK误差 {IK_ERR[0]*1000:.0f}mm",
            f"当前: {STEP_NAMES[S.step]}   要求步骤: {S.step_list()}",
            f"插入 {S.insert*1000:.0f}/100mm   上移 {S.lift*1000:.0f}/100mm   "
            f"P {pp:.0f}/90°   Q {qq:+.0f}° → 目标 {S.q_target:+.0f}°",
            f"对准: {'是' if ok else '否'}(角差{ang:.0f}° 横偏{lat*1000:.0f}mm)  "
            f"{'已锁定' if S.latched else '未锁定'}  "
            f"Q回正力矩≤0.3N·m {'| 保持0.5N·m' if S.hold else ''}",
        ]
        if pose_buf[0] is not None:
            info.insert(0, f"输入六关节角 J1 J2 J3 J4 J5 J6 (°): {pose_buf[0]}▌  Enter确认 Esc取消")
        if robot_buf[0] is not None:
            info.insert(0, f"输入工程臂六关节角 (°): {robot_buf[0]}▌  Enter确认 Esc取消")
        if MEASURE["on"] or MEASURE["A"] is not None:
            ml = f"测量({'开·点取A/B两点' if MEASURE['on'] else '关'}): "
            if MEASURE["A"] is not None:
                Ar = world_to_rule(MEASURE["A"])*1000
                ml += f"A=({Ar[0]:.1f}, {Ar[1]:.1f}, {Ar[2]:.1f})"
            if MEASURE["B"] is not None:
                Br = world_to_rule(MEASURE["B"])*1000
                dr = Br - world_to_rule(MEASURE["A"])*1000
                dist = float(np.linalg.norm(MEASURE["B"]-MEASURE["A"]))*1000
                ml += (f"  B=({Br[0]:.1f}, {Br[1]:.1f}, {Br[2]:.1f})"
                       f"  |AB|(斜边)={dist:.1f}"
                       f"  直角边 Δx={dr[0]:+.1f} Δy={dr[1]:+.1f} Δz={dr[2]:+.1f} mm")
            else:
                ml += "  点击物体取 B 点"
            info.append(ml)
        render_hud(info)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.destroy_window(window)
glfw.terminate()
print("viewer closed")
