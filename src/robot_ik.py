# -*- coding: utf-8 -*-
"""工程臂 IK 与抓取辅助（可独立于查看器测试）
- IKSolver：阻尼最小二乘（6 关节，位置+姿态误差），驱动 grasp site 到目标位姿
- grasp_check：抓取判定（指距/距离/朝向）
独立运行：python src/robot_ik.py  （加载模型做一次 IK 收敛自测）
"""
import sys, os, math
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "tech_core_joints.xml")
ARM_JOINTS = ["j_r1", "j_r2", "j_r3", "j_r4", "j_r5", "j_r6"]


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def mat_to_quat(R):
    q = np.empty(4)
    tr = R[0, 0]+R[1, 1]+R[2, 2]
    if tr > 0:
        s = math.sqrt(tr+1.0)*2
        q[0] = 0.25*s; q[1] = (R[2, 1]-R[1, 2])/s; q[2] = (R[0, 2]-R[2, 0])/s; q[3] = (R[1, 0]-R[0, 1])/s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0+R[0, 0]-R[1, 1]-R[2, 2])*2
        q[0] = (R[2, 1]-R[1, 2])/s; q[1] = 0.25*s; q[2] = (R[0, 1]+R[1, 0])/s; q[3] = (R[0, 2]+R[2, 0])/s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0+R[1, 1]-R[0, 0]-R[2, 2])*2
        q[0] = (R[0, 2]-R[2, 0])/s; q[1] = (R[0, 1]+R[1, 0])/s; q[2] = 0.25*s; q[3] = (R[1, 2]+R[2, 1])/s
    else:
        s = math.sqrt(1.0+R[2, 2]-R[0, 0]-R[1, 1])*2
        q[0] = (R[1, 0]-R[0, 1])/s; q[1] = (R[0, 2]+R[2, 0])/s; q[2] = (R[1, 2]+R[2, 1])/s; q[3] = 0.25*s
    return q/np.linalg.norm(q)


class IKSolver:
    """阻尼最小二乘 IK：驱动 grasp site 到目标 (pos, R)。"""

    def __init__(self, model, data, site_name="grasp", lam=0.08, step=0.08):
        self.model = model
        self.data = data
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.bid = int(model.site_bodyid[self.sid])
        self.qadr = []
        self.dofs = []
        for jn in ARM_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            self.qadr.append(model.jnt_qposadr[jid])
            self.dofs.append(model.jnt_dofadr[jid])
        self.qadr = np.array(self.qadr)
        self.dofs = np.array(self.dofs)
        self.lam = lam
        self.step = step
        self.err = 0.0

    def solve(self, target_pos, target_R, iters=4, w_rot=0.5):
        m, d = self.model, self.data
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jacp, jacr, d.site_xpos[self.sid], self.bid)
            J6 = np.vstack([jacp[:, self.dofs], jacr[:, self.dofs]])
            e_pos = np.asarray(target_pos) - d.site_xpos[self.sid]
            R_cur = d.site_xmat[self.sid].reshape(3, 3)
            R_err = np.asarray(target_R) @ R_cur.T
            e_rot = 0.5*np.array([R_err[2, 1]-R_err[1, 2],
                                  R_err[0, 2]-R_err[2, 0],
                                  R_err[1, 0]-R_err[0, 1]])
            e = np.concatenate([e_pos, w_rot*e_rot])
            lam = self.lam
            JJT = J6 @ J6.T
            if np.linalg.cond(JJT) > 1e8:
                lam = 0.25
            dq = J6.T @ np.linalg.solve(JJT + lam*lam*np.eye(6), e)
            dq = np.clip(dq, -self.step, self.step)
            for k in range(6):
                d.qpos[self.qadr[k]] = d.qpos[self.qadr[k]] + dq[k]
            # 限位截断
            for k in range(6):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS[k])
                lo, hi = m.jnt_range[jid]
                if lo < hi:
                    d.qpos[self.qadr[k]] = float(np.clip(d.qpos[self.qadr[k]], lo, hi))
        mujoco.mj_forward(m, d)
        self.err = float(np.linalg.norm(np.asarray(target_pos) - d.site_xpos[self.sid]))
        return self.err


def grasp_check(model, data, gap, unit_pos, sid_grasp, unit_R=None,
               dist_tol=0.12, gap_win=(0.075, 0.125), ang_tol_deg=35.0):
    """抓取判定：距离、指距窗口、开口朝向（能量单元开口 = 体轴 -Y）"""
    gp = data.site_xpos[sid_grasp]
    dist = float(np.linalg.norm(unit_pos - gp))
    ok_gap = gap_win[0] < gap < gap_win[1]
    if unit_R is not None:
        gR = data.site_xmat[sid_grasp].reshape(3, 3)
        mouth = unit_R @ np.array([0.0, -1.0, 0.0])
        axis = gR @ np.array([0.0, 1.0, 0.0])   # grasp site 的接近轴（沿臂展方向）
        ang = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.dot(mouth, axis)))))))
        ok_ang = ang < ang_tol_deg
    else:
        ok_ang = True
        ang = 99.0
    return (dist < dist_tol and ok_gap and ok_ang), dist, gap, ang


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ik = IKSolver(model, data)
    p0 = data.site_xpos[ik.sid].copy()
    R0 = data.site_xmat[ik.sid].reshape(3, 3).copy()
    tgt = p0 + np.array([0.05, -0.10, 0.08])
    err = ik.solve(tgt, R0, iters=30)
    print(f"IK 自测：目标 {np.round(tgt,3).tolist()}  收敛误差 {err*1000:.1f} mm")
    assert err < 0.02, "IK 未收敛"
    print("robot_ik OK")
