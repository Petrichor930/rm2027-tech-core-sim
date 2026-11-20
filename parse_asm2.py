import sys, re
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

path = r"D:/RM/RMUC2027_V1.0.0.stp"

wanted = {'PRODUCT', 'PRODUCT_DEFINITION', 'PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE',
          'NEXT_ASSEMBLY_USAGE_OCCURRENCE', 'PRODUCT_RELATED_PRODUCT_CATEGORY'}

def decode_step(s):
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '\\':
            if s[i:i+4] == '\\X2\\':
                j = s.find('\\X0\\', i+4)
                if j == -1:
                    out.append(c); i += 1; continue
                hexstr = s[i+4:j]
                for k in range(0, len(hexstr)-3, 4):
                    try:
                        out.append(chr(int(hexstr[k:k+4], 16)))
                    except ValueError:
                        break
                i = j + 4
            elif s[i:i+4] == '\\X4\\':
                j = s.find('\\X0\\', i+4)
                if j == -1:
                    out.append(c); i += 1; continue
                hexstr = s[i+4:j]
                for k in range(0, len(hexstr)-7, 8):
                    try:
                        out.append(chr(int(hexstr[k:k+8], 16)))
                    except ValueError:
                        break
                i = j + 4
            elif s[i:i+3] == '\\X\\':
                j = s.find('\\X0\\', i+3)
                if j == -1:
                    out.append(c); i += 1; continue
                hexstr = s[i+3:j]
                for k in range(0, len(hexstr)-1, 2):
                    try:
                        out.append(chr(int(hexstr[k:k+2], 16)))
                    except ValueError:
                        break
                i = j + 4
            elif s[i:i+2] == '\\_':
                out.append('_'); i += 2
            else:
                out.append(c); i += 1
        else:
            out.append(c); i += 1
    return ''.join(out)

ents = {}
ere = re.compile(r'^#(\d+)=([A-Z0-9_]+)\(')
buf = []
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    for raw in f:
        line = raw.rstrip('\n').rstrip('\r')
        buf.append(line)
        if line.rstrip().endswith(';'):
            s = ''.join(buf)
            buf = []
            m = ere.match(s)
            if m:
                eid = int(m.group(1)); etype = m.group(2)
                if etype in wanted:
                    ents[eid] = (etype, s)

prod = {}
pd = {}
pdf = {}
nauo = []
cat = {}

for eid, (t, s) in ents.items():
    if t == 'PRODUCT':
        m = re.search(r"PRODUCT\('([^']*)'", s)
        prod[eid] = decode_step(m.group(1)) if m else ''
    elif t == 'PRODUCT_DEFINITION':
        m = re.search(r"PRODUCT_DEFINITION\('[^']*','[^']*',#(\d+)", s)
        pd[eid] = int(m.group(1)) if m else None
    elif t == 'PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE':
        m = re.search(r"FORMATION_WITH_SPECIFIED_SOURCE\('[^']*','[^']*',#(\d+)", s)
        pdf[eid] = int(m.group(1)) if m else None
    elif t == 'NEXT_ASSEMBLY_USAGE_OCCURRENCE':
        m = re.search(r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\('[^']*','[^']*','([^']*)',#(\d+),#(\d+)", s)
        if m:
            nauo.append((int(m.group(2)), int(m.group(3)), decode_step(m.group(1))))
    elif t == 'PRODUCT_RELATED_PRODUCT_CATEGORY':
        m = re.search(r"PRODUCT_RELATED_PRODUCT_CATEGORY\('([^']*)','[^']*',\(([^)]*)\)", s)
        if m:
            ids = [int(x) for x in re.findall(r'#(\d+)', m.group(2))]
            cat[m.group(1)] = ids

print("products:", len(prod), "defs:", len(pd), "formations:", len(pdf), "nauo:", len(nauo))
print("categories:", {k: len(v) for k, v in cat.items()})

def prod_of_def(defid):
    fid = pd.get(defid)
    if fid is None: return None, ''
    pid = pdf.get(fid)
    if pid is None: return None, ''
    return pid, prod.get(pid, '')

children = {}
for parent_def, child_def, ref in nauo:
    ppid, _ = prod_of_def(parent_def)
    cpid, _ = prod_of_def(child_def)
    if ppid is None or cpid is None:
        continue
    children.setdefault(ppid, []).append((cpid, ref))

all_child = {c for lst in children.values() for (c, _) in lst}
roots = [p for p in prod if p not in all_child]

def dump(pid, depth):
    print("  " * depth + f"[{pid}] {prod.get(pid,'')}")
    for (cpid, ref) in children.get(pid, []):
        print("  " * (depth+1) + f"-> [{cpid}] {prod.get(cpid,'')}   REF='{ref}'")

for r in roots:
    dump(r, 0)

# search keywords
print("\n===== 搜索关键词 =====")
for kw in ['科技核心', '能量单元', '能量机关', '装配', '核心']:
    hits = [(pid, prod.get(pid,''), ref) for (_, lst) in children.items() for (pid, ref) in lst if kw in ref or kw in prod.get(pid,'')]
    print(f"'{kw}': {len(hits)}")
    for pid, pname, ref in hits:
        print(f"    [{pid}] product='{pname}' REF='{ref}'")
