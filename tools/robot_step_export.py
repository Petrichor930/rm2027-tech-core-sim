# -*- coding: utf-8 -*-
"""机械臂 STEP -> 逐实例 STL 缓存 + manifest.csv（M1）
- 同一原型 solid 只网格化一次，实例只做刚体变换（省时间）
- manifest.csv: stl_file,product_name,occurrence_path,solid_index,tri,
                bbox(6), cyls(r,px,py,pz,dx,dy,dz;...) —— 全部装配坐标系,mm
运行（Python 3.12 + OCP）：
  python tools/robot_step_export.py "<STEP路径>"
"""
import sys, os, csv, math
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp


def sm(obj, name, *args):
    f = getattr(obj, name + "_s", None)
    if f is None:
        f = getattr(obj, name)
    return f(*args)


STEP_PATH = sys.argv[1] if len(sys.argv) > 1 else ""
OUT_DIR = "assets/robot_parts"
if not STEP_PATH or not os.path.exists(STEP_PATH):
    print("[错误] 用法: python tools/robot_step_export.py <STEP路径>")
    sys.exit(1)
os.makedirs(OUT_DIR, exist_ok=True)


def name_of(lab):
    try:
        attr = TDataStd_Name()
        if lab.FindAttribute(sm(TDataStd_Name, 'GetID'), attr):
            v = attr.Get()
            t = v.ToExtString() if hasattr(v, "ToExtString") else str(v)
            if isinstance(t, str) and t.strip():
                return t.strip()
    except Exception:
        pass
    return "?"


def trsf_of(loc):
    t = loc.Transformation()
    M = np.eye(4)
    for i in range(3):
        for j in range(4):
            M[i, j] = t.Value(i + 1, j + 1)
    return M


def mesh_solid_local(sol):
    """网格化一个 solid，返回局部三角形 (n,3,3) mm"""
    BRepMesh_IncrementalMesh(sol, 0.5, False, 0.3, True)
    tris = []
    fexp = TopExp_Explorer(sol, TopAbs_FACE)
    while fexp.More():
        face = TopoDS.Face_s(fexp.Current())
        loc = TopLoc_Location()
        tri = sm(BRep_Tool, 'Triangulation', face, loc)
        if tri is not None:
            Mf = trsf_of(loc)
            n = tri.NbNodes()
            pts = np.empty((n, 3))
            for i in range(1, n + 1):
                p = tri.Node(i)
                pts[i - 1] = (p.X(), p.Y(), p.Z())
            pts = pts @ Mf[:3, :3].T + Mf[:3, 3]
            for i in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(i)
                try:
                    a, b, c = t.Get()
                except Exception:
                    a, b, c = t.Value(1), t.Value(2), t.Value(3)
                tris.append((pts[a - 1], pts[b - 1], pts[c - 1]))
        fexp.Next()
    return np.array(tris, dtype=np.float64).reshape(-1, 3, 3)


def cyls_of_solid_local(sol):
    out = []
    fexp = TopExp_Explorer(sol, TopAbs_FACE)
    while fexp.More():
        face = TopoDS.Face_s(fexp.Current())
        try:
            ad = BRepAdaptor_Surface(face)
            if ad.GetType() == GeomAbs_Cylinder:
                cyl = ad.Cylinder()
                ax = cyl.Axis()
                p, d = ax.Location(), ax.Direction()
                props = GProp_GProps()
                sm(BRepGProp, 'SurfaceProperties', face, props)
                out.append((cyl.Radius(), (p.X(), p.Y(), p.Z()), (d.X(), d.Y(), d.Z()), props.Mass()))
        except Exception:
            pass
        fexp.Next()
    return out


def save_stl(path, tri):
    n = len(tri)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct_pack_u32(n))
        for t in tri:
            f.write(struct.pack('<3f', 0, 0, 1))
            for v in t:
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))


import struct
def struct_pack_u32(n):
    return struct.pack('<I', n)


# ---------- 读 STEP ----------
doc = TDocStd_Document(TCollection_ExtendedString("robot"))
rd = STEPCAFControl_Reader()
try:
    rd.SetNameMode(True)
except Exception:
    pass
rd.ReadFile(STEP_PATH)
rd.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

occ_rows = []   # (path, name, solid_idx, trsf4x4, shape_solid, cache_key)


def walk(lab, loc, path):
    if sm(st, 'IsAssembly', lab):
        comps = TDF_LabelSequence()
        sm(st, 'GetComponents', lab, comps)
        for i in range(1, comps.Length() + 1):
            comp = comps.Value(i)
            ref = TDF_Label()
            sm(st, 'GetReferredShape', comp, ref)
            walk(ref, loc.Multiplied(sm(st, 'GetLocation', comp)), path + "/" + name_of(comp))
    else:
        nm = name_of(lab)
        shape = sm(st, 'GetShape', lab)
        M = trsf_of(loc)
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        si = 0
        while exp.More():
            sol = exp.Current()
            occ_rows.append((path, nm, si, M, sol))
            si += 1
            exp.Next()


roots = TDF_LabelSequence()
sm(st, 'GetFreeShapes', roots)
for i in range(1, roots.Length() + 1):
    walk(roots.Value(i), TopLoc_Location(), f"root{i}")
print(f"实例 solid 数: {len(occ_rows)}")

# ---------- 网格缓存（按 产品名+solid序号+局部bbox 去重） ----------
cache = {}
manifest = []
skipped = 0
fi = 0
for path, nm, si, M, sol in occ_rows:
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        box = Bnd_Box()
        sm(BRepBndLib, 'Add', sol, box, True)
        bb = tuple(round(v, 2) for v in box.Get())
    except Exception:
        bb = (si,)
    key = (nm, si, bb)
    if key not in cache:
        tris = mesh_solid_local(sol)
        cyls = cyls_of_solid_local(sol)
        cache[key] = (tris, cyls)
    tris, cyls = cache[key]
    if len(tris) < 40:
        skipped += 1
        continue
    # 实例变换
    R, t = M[:3, :3], M[:3, 3]
    T = tris.reshape(-1, 3) @ R.T + t
    T = T.reshape(-1, 3, 3)
    fi += 1
    fn = f"p{fi:04d}_s{si:02d}.stl"
    save_stl(os.path.join(OUT_DIR, fn), T)
    lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
    cyl_s = []
    for r, p, d, a in cyls:
        pw = R @ np.array(p) + t
        dw = R @ np.array(d)
        cyl_s.append(f"{r:.2f},{pw[0]:.2f},{pw[1]:.2f},{pw[2]:.2f},{dw[0]:.4f},{dw[1]:.4f},{dw[2]:.4f},{a:.0f}")
    manifest.append([fn, nm, path, si, len(tris),
                     lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], ";".join(cyl_s)])

with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["stl_file", "product_name", "occurrence_path", "solid_index", "tris",
                "bx0", "by0", "bz0", "bx1", "by1", "bz1", "cyls(r,px,py,pz,dx,dy,dz,area;)"])
    w.writerows(manifest)

tot = sum(int(m[4]) for m in manifest)
print(f"导出 {fi} 个 STL（跳过 <40 面退化 {skipped} 个），总三角 {tot}")
print(f"清单: {os.path.join(OUT_DIR, 'manifest.csv')}（{len(cache)} 个唯一原型）")
