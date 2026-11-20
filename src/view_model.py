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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "tech_core_energy_unit.xml")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

if not glfw.init():
    raise RuntimeError("glfw init failed")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
glfw.window_hint(glfw.SAMPLES, 4)

window = glfw.create_window(1200, 800, "科技核心 + 能量单元  手动装配", None, None)
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
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
HUD_FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)

# ---- 对象 ----
eu = model.body('energy_unit')
eu_mocap = int(eu.mocapid)
tc = model.body('tech_core')
tc_mocap = int(tc.mocapid)

init_eu_pos = data.mocap_pos[eu_mocap].copy()
init_eu_quat = data.mocap_quat[eu_mocap].copy()
init_tc_pos = data.mocap_pos[tc_mocap].copy()
init_tc_quat = data.mocap_quat[tc_mocap].copy()

# 科技核心底座在其局部坐标系的位置（+Y 端）
TC_BASE_LOCAL = np.array([0.0, 0.85, 0.0])

# ---- 状态 ----
last_x, last_y = [0.0], [0.0]
btn_left, btn_right, btn_middle = [False], [False], [False]
MODES = ["energy_unit", "tech_core", "camera"]
MODE_NAME = {"energy_unit": "能量单元", "tech_core": "科技核心", "camera": "相机"}
mode = [0]
difficulty = [1]
show_hud = [True]
pending = set()
t0 = [time.time()]; frames = [0]; fps = [0.0]; shot = [0]

TRANS = 0.003
ROT_DEG = 1.0
DRAG_SCALE = 0.004


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


def rotate_tc_pivot(axis_world, deg):
    R_old = quat_to_mat(data.mocap_quat[tc_mocap])
    base_world = data.mocap_pos[tc_mocap] + R_old @ TC_BASE_LOCAL
    q_new = quat_mult(axis_quat(axis_world, deg), data.mocap_quat[tc_mocap])
    q_new = q_new / math.sqrt(float((q_new*q_new).sum()))
    R_new = quat_to_mat(q_new)
    data.mocap_pos[tc_mocap] = base_world - R_new @ TC_BASE_LOCAL
    data.mocap_quat[tc_mocap] = q_new


def key_cb(w, key, scancode, action, mods):
    if action == glfw.PRESS:
        pending.add(key)


def mouse_button_cb(w, button, action, mods):
    if button == glfw.MOUSE_BUTTON_LEFT:
        btn_left[0] = (action == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        btn_right[0] = (action == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        btn_middle[0] = (action == glfw.PRESS)


def cursor_pos_cb(w, x, y):
    dx = x - last_x[0]; dy = y - last_y[0]
    m = MODES[mode[0]]
    if m == 'camera':
        if btn_left[0]:
            cam.azimuth -= dx * 0.4
            cam.elevation -= dy * 0.4
            cam.elevation = max(-89.0, min(89.0, cam.elevation))
        elif btn_right[0]:
            az = math.radians(cam.azimuth)
            right = (math.sin(az), -math.cos(az)); fwd = (-math.cos(az), -math.sin(az))
            s = cam.distance * 0.0015
            cam.lookat[0] += (right[0]*dx + fwd[0]*dy) * s
            cam.lookat[1] += (right[1]*dx + fwd[1]*dy) * s
        elif btn_middle[0]:
            cam.distance *= math.exp(dy * 0.01)
            cam.distance = max(0.05, min(50.0, cam.distance))
    elif m == 'energy_unit':
        s = cam.distance * DRAG_SCALE
        if btn_left[0]:
            data.mocap_pos[eu_mocap][0] += dx * s
            data.mocap_pos[eu_mocap][1] -= dy * s
        elif btn_right[0]:
            data.mocap_quat[eu_mocap] = quat_mult(axis_quat((0,0,1), -dx*0.4), data.mocap_quat[eu_mocap])
        elif btn_middle[0]:
            data.mocap_pos[eu_mocap][2] += dy * s
        q = data.mocap_quat[eu_mocap]
        data.mocap_quat[eu_mocap] = q / math.sqrt(float((q*q).sum()))
    elif m == 'tech_core':
        if btn_left[0]:
            rotate_tc_pivot((0, 0, 1), -dx * 0.4)
        elif btn_right[0]:
            rotate_tc_pivot((0, 1, 0), -dy * 0.4)
        elif btn_middle[0]:
            rotate_tc_pivot((1, 0, 0), -dx * 0.4)
    last_x[0] = x; last_y[0] = y


def scroll_cb(w, xoffset, yoffset):
    m = MODES[mode[0]]
    if m == 'camera':
        cam.distance *= math.exp(-yoffset * 0.1)
        cam.distance = max(0.05, min(50.0, cam.distance))
    elif m == 'energy_unit':
        data.mocap_pos[eu_mocap][2] += yoffset * cam.distance * 0.002
    elif m == 'tech_core':
        rotate_tc_pivot((0, 1, 0), yoffset * 2.0)


glfw.set_key_callback(window, key_cb)
glfw.set_mouse_button_callback(window, mouse_button_cb)
glfw.set_cursor_pos_callback(window, cursor_pos_cb)
glfw.set_scroll_callback(window, scroll_cb)


def apply_energy_keyboard(keys):
    pos = data.mocap_pos[eu_mocap]
    quat = data.mocap_quat[eu_mocap]
    dpos = np.zeros(3)
    dquat = np.array([1.0, 0.0, 0.0, 0.0])
    if keys(glfw.KEY_W): dpos[0] += TRANS
    if keys(glfw.KEY_S): dpos[0] -= TRANS
    if keys(glfw.KEY_D): dpos[1] += TRANS
    if keys(glfw.KEY_A): dpos[1] -= TRANS
    if keys(glfw.KEY_E): dpos[2] += TRANS
    if keys(glfw.KEY_Q): dpos[2] -= TRANS
    if keys(glfw.KEY_RIGHT): dquat = quat_mult(axis_quat((0,0,1), ROT_DEG), dquat)
    if keys(glfw.KEY_LEFT):  dquat = quat_mult(axis_quat((0,0,1), -ROT_DEG), dquat)
    if keys(glfw.KEY_UP):    dquat = quat_mult(axis_quat((0,1,0), ROT_DEG), dquat)
    if keys(glfw.KEY_DOWN):  dquat = quat_mult(axis_quat((0,1,0), -ROT_DEG), dquat)
    if keys(glfw.KEY_X):     dquat = quat_mult(axis_quat((1,0,0), ROT_DEG), dquat)
    if keys(glfw.KEY_Z):     dquat = quat_mult(axis_quat((1,0,0), -ROT_DEG), dquat)
    data.mocap_pos[eu_mocap] = pos + dpos
    data.mocap_quat[eu_mocap] = quat_mult(dquat, quat)
    q = data.mocap_quat[eu_mocap]
    n = math.sqrt(float((q*q).sum()))
    if n > 0:
        data.mocap_quat[eu_mocap] = q / n


def apply_tc_keyboard(keys):
    if keys(glfw.KEY_RIGHT): rotate_tc_pivot((0,0,1), ROT_DEG)
    if keys(glfw.KEY_LEFT):  rotate_tc_pivot((0,0,1), -ROT_DEG)
    if keys(glfw.KEY_UP):    rotate_tc_pivot((0,1,0), ROT_DEG)
    if keys(glfw.KEY_DOWN):  rotate_tc_pivot((0,1,0), -ROT_DEG)
    if keys(glfw.KEY_X):     rotate_tc_pivot((1,0,0), ROT_DEG)
    if keys(glfw.KEY_Z):     rotate_tc_pivot((1,0,0), -ROT_DEG)


HELP = ("Tab=切换(能量单元/科技核心/相机)  1/2/3=难度  P=截图  H=隐藏  Esc=退出\n"
        "能量单元: 左键拖=平移 右键拖=旋转 滚轮=升降  键盘W/A/S/D/Q/E+方向键\n"
        "科技核心: 左键拖=绕底座旋转 右键拖=俯仰 滚轮=滚转  键盘方向键/Z/X (底座固定)\n"
        "相机: 左键=旋转视角 右键=平移 滚轮=缩放  Backspace=重置")

LEVEL_GUIDE = {
    1: "L1: 插入100mm即可 (步骤1,2,6)",
    2: "L2: 插入 + 上移100mm (步骤1,2,3,6)",
    3: "L3: 插入+上移+绕P转90°+绕Q转指定角 (步骤1-6)",
}

while not glfw.window_should_close(window):
    if glfw.KEY_ESCAPE in pending:
        break
    if glfw.KEY_TAB in pending:
        mode[0] = (mode[0] + 1) % 3
    if glfw.KEY_1 in pending:
        difficulty[0] = 1
    if glfw.KEY_2 in pending:
        difficulty[0] = 2
    if glfw.KEY_3 in pending:
        difficulty[0] = 3
    if glfw.KEY_BACKSPACE in pending:
        data.mocap_pos[eu_mocap] = init_eu_pos.copy()
        data.mocap_quat[eu_mocap] = init_eu_quat.copy()
        data.mocap_pos[tc_mocap] = init_tc_pos.copy()
        data.mocap_quat[tc_mocap] = init_tc_quat.copy()
    if glfw.KEY_H in pending:
        show_hud[0] = not show_hud[0]
    if glfw.KEY_P in pending:
        fbw, fbh = glfw.get_framebuffer_size(window)
        rgb = np.empty((fbh, fbw, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, fbw, fbh), ctx)
        shot[0] += 1
        Image.fromarray(rgb[::-1]).save(os.path.join(OUTPUT_DIR, f"shot_{shot[0]:03d}.png"))
        print(f"已保存 shot_{shot[0]:03d}.png")
    pending.clear()

    m = MODES[mode[0]]
    if m == 'energy_unit':
        apply_energy_keyboard(lambda k: glfw.get_key(window, k) == glfw.PRESS)
    elif m == 'tech_core':
        apply_tc_keyboard(lambda k: glfw.get_key(window, k) == glfw.PRESS)

    mujoco.mj_forward(model, data)

    fbw, fbh = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, fbw, fbh)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, ctx)

    if show_hud[0]:
        frames[0] += 1
        now = time.time()
        if now - t0[0] >= 0.5:
            fps[0] = frames[0] / (now - t0[0]); frames[0] = 0; t0[0] = now
        ep = data.mocap_pos[eu_mocap]
        tp = data.mocap_pos[tc_mocap]
        info = (f"FPS {fps[0]:.0f}   当前: {MODE_NAME[m]}   难度: L{difficulty[0]}\n"
                f"{LEVEL_GUIDE[difficulty[0]]}\n"
                f"能量单元: ({ep[0]:.3f}, {ep[1]:.3f}, {ep[2]:.3f}) m\n"
                f"科技核心: ({tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f}) m")

        rgb = np.empty((fbh, fbw, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, ctx)
        img = Image.fromarray(rgb[::-1].copy()).convert('RGBA')
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        y = 8
        for ln in (HELP.split('\n') + info.split('\n')):
            od.text((14, y), ln, font=HUD_FONT, fill=(255, 255, 255, 255))
            y += 30
        img = Image.alpha_composite(img, overlay).convert('RGB')
        mujoco.mjr_drawPixels(np.array(img)[::-1].reshape(-1), None, viewport, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.destroy_window(window)
glfw.terminate()
print("viewer closed")
