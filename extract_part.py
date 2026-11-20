import sys, re, time
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

path = r"D:/RM/RMUC2027_V1.0.0.stp"

# BFS roots: use SHAPE_DEFINITION_REPRESENTATION (SDR) so both upstream
# (PRODUCT structure) and downstream (geometry) are captured.
ROOTS = {
    'tech_core': 1751930,        # SDR for 科技核心 (product 0002) -> shape rep 1751924
    'energy_unit_asm': 4056220,  # SDR for 能量单元 (product 0017) -> shape rep 4056214
}

ere = re.compile(r'^#(\d+)=')
ref_re = re.compile(r'#(\d+)')

# ---- Pass 1: build id -> refs graph (memory-efficient-ish) ----
refs = {}
buf = []
lineno = 0
t0 = time.time()
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    for raw in f:
        lineno += 1
        line = raw.rstrip('\n').rstrip('\r')
        buf.append(line)
        if line.rstrip().endswith(';'):
            s = ''.join(buf)
            buf = []
            m = ere.match(s)
            if not m:
                continue
            eid = int(m.group(1))
            body = s[m.end():]
            r = [int(x) for x in ref_re.findall(body)]
            refs[eid] = r
            if lineno % 2000000 == 0:
                print(f"pass1 line {lineno}, entities {len(refs)}, {time.time()-t0:.0f}s")

print(f"pass1 done: {len(refs)} entities, {time.time()-t0:.0f}s")

# ---- BFS from each root ----
for name, root in ROOTS.items():
    if root not in refs:
        print(f"{name}: root {root} NOT FOUND in graph")
        continue
    seen = set()
    stack = [root]
    while stack:
        e = stack.pop()
        if e in seen:
            continue
        seen.add(e)
        stack.extend(refs.get(e, ()))
    print(f"{name}: reachable entities = {len(seen)}")
    ROOTS[name] = (root, seen)

# ---- Pass 2: output minimal STEP files ----
HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('extract','2026-11-19T00:00:00',(''),(''),'CREO PARAMETRIC BY PTC INC, 2019164','CREO PARAMETRIC BY PTC INC, 2019164','');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN','SHAPE_APPEARANCE_LAYERS_GROUPS'));
ENDSEC;
DATA;
"""

for name, (root, seen) in ROOTS.items():
    out_path = f"assets/{name}.stp"
    buf = []
    t1 = time.time()
    with open(path, 'r', encoding='utf-8', errors='replace') as f, open(out_path, 'w', encoding='utf-8') as fo:
        fo.write(HEADER)
        for raw in f:
            line = raw.rstrip('\n').rstrip('\r')
            buf.append(line)
            if line.rstrip().endswith(';'):
                s = ''.join(buf)
                buf = []
                m = ere.match(s)
                if m:
                    eid = int(m.group(1))
                    if eid in seen:
                        fo.write(s + '\n')
        fo.write("ENDSEC;\nEND-ISO-10303-21;\n")
    import os
    print(f"{name}: wrote {out_path} {os.path.getsize(out_path)} bytes, {time.time()-t1:.0f}s")
