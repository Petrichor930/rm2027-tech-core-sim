import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import fitz

pdfs = [
    (r"D:\26engineer\PinyCaster\.docs\UTF-8__RoboMaster 2027 机甲大师超级对抗赛比赛规则手册V1.0.0（20261015）-1.pdf", "rules"),
    (r"D:\26engineer\PinyCaster\.docs\UTF-8__RoboMaster 2027 机甲大师高校系列赛机器人制作规范手册V1.0.0（20261015）.pdf", "spec"),
]

keywords = ["科技核心", "能量机关", "核心"]

for path, tag in pdfs:
    doc = fitz.open(path)
    print(f"===== {tag}: {doc.page_count} pages =====")
    for kw in keywords:
        hits = []
        for i in range(doc.page_count):
            if kw in doc[i].get_text():
                hits.append(i + 1)  # 1-based
        print(f"  '{kw}': {len(hits)} pages -> {hits}")
    doc.close()
