import sys, os, traceback
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

print("starting ...", flush=True)

from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_ShapeEnum

print("imports ok", flush=True)

def step_to_stl(src, dst, root_id, deflection=0.5, ang_deflection=0.3):
    reader = STEPControl_Reader()
    print(f"reading {src} ...", flush=True)
    status = reader.ReadFile(src)
    print(f"  ReadFile status: {status} (RetDone={int(IFSelect_RetDone.IFSelect_RetDone)})", flush=True)
    if status != IFSelect_RetDone.IFSelect_RetDone:
        raise RuntimeError(f"read failed: {status}")
    nroots = reader.TransferRoots()
    print(f"  TransferRoots: {nroots} roots", flush=True)
    shape = reader.OneShape()
    print(f"  OneShape type: {shape.ShapeType()}", flush=True)
    for name, enum in [("SOLID", TopAbs_ShapeEnum.TopAbs_SOLID),
                       ("SHELL", TopAbs_ShapeEnum.TopAbs_SHELL),
                       ("FACE", TopAbs_ShapeEnum.TopAbs_FACE)]:
        exp = TopExp_Explorer(shape, enum)
        cnt = 0
        while exp.More():
            cnt += 1
            exp.Next()
        print(f"    {name}: {cnt}", flush=True)
    bb = Bnd_Box()
    try:
        BRepBndLib.Add(shape, bb)
        xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
        print(f"  bbox (mm): x {xmin:.2f}..{xmax:.2f}  y {ymin:.2f}..{ymax:.2f}  z {zmin:.2f}..{zmax:.2f}", flush=True)
    except Exception as e:
        print(f"  bbox skipped ({e})", flush=True)
    print(f"  meshing ...", flush=True)
    BRepMesh_IncrementalMesh(shape, deflection, False, ang_deflection, True)
    print(f"  writing STL ...", flush=True)
    writer = StlAPI_Writer()
    writer.Write(shape, dst)
    print(f"  saved {dst} {os.path.getsize(dst)} bytes", flush=True)

try:
    step_to_stl("assets/tech_core.stp", "assets/tech_core.stl", 1751924, deflection=2.0, ang_deflection=0.5)
    step_to_stl("assets/energy_unit_asm.stp", "assets/energy_unit_asm.stl", 4056214)
    print("ALL DONE", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
