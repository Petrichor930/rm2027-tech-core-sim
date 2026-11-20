# -*- coding: utf-8 -*-
"""机械臂 STEP 勘察（M0）：XCAF 装配树遍历 -> 输出/robot_step_report.txt
- 每个实例：路径、产品名、solid 数、局部位姿、包围盒
- 每个 solid 的圆柱面：轴线点/方向/半径/面积（关节轴拟合的原料）
运行（Python 3.12 + OCP）：
  python tools/robot_step_dump.py "<STEP路径>" [输出报告路径]
"""
import sys, os, math
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

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
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS


def sm(obj, name, *args):
    """OCP 静态/实例方法双拼写兼容：优先 name_s，回落 name"""
    f = getattr(obj, name + "_s", None)
    if f is None:
        f = getattr(obj, name)
    return f(*args)

STEP_PATH = sys.argv[1] if len(sys.argv) > 1 else ""
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "output/robot_step_report.txt"
if not STEP_PATH or not os.path.exists(STEP_PATH):
    print("[错误] 用法: python tools/robot_step_dump.py <STEP路径> [报告输出路径]")
    sys.exit(1)

os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
out = open(OUT_PATH, "w", encoding="utf-8")


def emit(s=""):
    print(s)
    out.write(s + "\n")


def name_of(lab):
    """取 XCAF 标签的产品名（多路兜底）"""
    try:
        attr = TDataStd_Name()
        ok = lab.FindAttribute(sm(TDataStd_Name, 'GetID'), attr)
        if ok:
            v = attr.Get()
            for meth in ("ToExtString", "ToCString", "__str__"):
                try:
                    t = getattr(v, meth)() if meth != "__str__" else str(v)
                    if isinstance(t, str) and t.strip():
                        return t.strip()
                except Exception:
                    pass
    except Exception:
        pass
    return "?"


def trsf_str(loc):
    try:
        t = loc.Transformation()
        rows = [[round(t.Value(i, j), 4) for j in range(1, 5)] for i in range(1, 4)]
        return rows
    except Exception:
        return None


def solid_report(shape, indent):
    """遍历 solid，打印包围盒与圆柱面轴线"""
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    si = 0
    while exp.More():
        sol = exp.Current()
        si += 1
        box = Bnd_Box()
        sm(BRepBndLib, 'Add', sol, box, True)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        emit(f"{indent}  solid#{si}: bbox ({xmin:.1f},{ymin:.1f},{zmin:.1f})..({xmax:.1f},{ymax:.1f},{zmax:.1f}) mm")
        # 圆柱面
        fexp = TopExp_Explorer(sol, TopAbs_FACE)
        cyls = []
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
                    cyls.append((cyl.Radius(), (p.X(), p.Y(), p.Z()), (d.X(), d.Y(), d.Z()), props.Mass()))
            except Exception:
                pass
            fexp.Next()
        cyls.sort(key=lambda c: -c[3])
        for r, p, d, a in cyls[:12]:
            emit(f"{indent}    cyl r={r:.2f} p=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}) d=({d[0]:.3f},{d[1]:.3f},{d[2]:.3f}) area={a:.0f}")
        if len(cyls) > 12:
            emit(f"{indent}    ... 共 {len(cyls)} 个圆柱面")
        exp.Next()
    return si


def walk(lab, loc, path, stats):
    if sm(st, 'IsAssembly', lab):
        comps = TDF_LabelSequence()
        sm(st, 'GetComponents', lab, comps)
        for i in range(1, comps.Length() + 1):
            comp = comps.Value(i)
            ref = TDF_Label()
            sm(st, 'GetReferredShape', comp, ref)
            cname = name_of(comp)
            nloc = loc.Multiplied(sm(st, 'GetLocation', comp))
            walk(ref, nloc, path + "/" + cname, stats)
    else:
        nm = name_of(lab)
        stats["leaves"] += 1
        stats["names"].add(nm)
        shape = sm(st, 'GetShape', lab)
        emit(f"--- {path}  [{nm}]")
        rows = trsf_str(loc)
        if rows:
            emit(f"    loc: {rows}")
        ns = solid_report(shape, "    ")
        stats["solids"] += ns
        if ns == 0:
            emit("    (无 solid —— 可能是壳/线框/空件)")


emit(f"=== 机械臂 STEP 勘察 ===")
emit(f"文件: {STEP_PATH}  ({os.path.getsize(STEP_PATH)/1e6:.1f} MB)")
doc = TDocStd_Document(TCollection_ExtendedString("robot"))
rd = STEPCAFControl_Reader()
try:
    rd.SetNameMode(True)
except Exception:
    pass
rd.ReadFile(STEP_PATH)
rd.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

roots = TDF_LabelSequence()
sm(st, 'GetFreeShapes', roots)
emit(f"根标签数: {roots.Length()}")
stats = {"leaves": 0, "solids": 0, "names": set()}
for i in range(1, roots.Length() + 1):
    walk(roots.Value(i), TopLoc_Location(), f"root{i}", stats)

emit()
emit(f"=== 汇总 ===")
emit(f"叶子实例(带位置)数: {stats['leaves']}   solid 总数: {stats['solids']}   唯一产品名: {len(stats['names'])}")
out.close()
print(f"\n报告已写入 {OUT_PATH}")
