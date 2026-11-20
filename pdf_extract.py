import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import fitz

rules = r"D:\26engineer\PinyCaster\.docs\UTF-8__RoboMaster 2027 机甲大师超级对抗赛比赛规则手册V1.0.0（20261015）-1.pdf"
spec = r"D:\26engineer\PinyCaster\.docs\UTF-8__RoboMaster 2027 机甲大师高校系列赛机器人制作规范手册V1.0.0（20261015）.pdf"

out = []
def dump(doc, pages, label):
    out.append(f"\n\n########## {label} ##########")
    for p in pages:
        out.append(f"\n----- page {p} -----")
        out.append(doc[p - 1].get_text())

doc = fitz.open(rules)
dump(doc, [6, 8, 11, 12, 32, 52, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90], "比赛规则手册 - 科技核心相关页")
doc.close()

doc = fitz.open(spec)
dump(doc, [16], "制作规范手册 - 科技核心相关页")
doc.close()

with open("tech_core_extract.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("written tech_core_extract.txt, total chars:", sum(len(s) for s in out))
