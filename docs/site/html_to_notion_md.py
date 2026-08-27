#!/usr/bin/env python3
"""docs/site/looksgood.html -> 노션 붙여넣기용 마크다운(looksgood.md).

    python3 docs/site/html_to_notion_md.py      # 리포 루트에서


범용 변환기가 아니다 — 이 문서가 쓰는 마크업만 안다(.term/.note/.plate/table/.steps).
그래서 표와 콜아웃이 뭉개지지 않는다.
"""
import html as H, re, sys
from pathlib import Path

src = Path("docs/site/looksgood.html").read_text()
body = src.split("<body>", 1)[1].split("</body>", 1)[0]
body = re.sub(r"<!--.*?-->", "", body, flags=re.S)


def inline(s: str) -> str:
    """인라인 태그 -> 마크다운. 컨트롤 이름과 코드는 둘 다 백틱(노션 inline code)."""
    s = re.sub(r"<span class=\"ui\">(.*?)</span>", lambda m: "`" + strip(m.group(1)) + "`", s, flags=re.S)
    s = re.sub(r"<code>(.*?)</code>", lambda m: "`" + strip(m.group(1)) + "`", s, flags=re.S)
    s = re.sub(r"</?(strong|b)>", "**", s)
    s = re.sub(r"</?(em|i)>", "*", s)
    s = re.sub(r"<a [^>]*>(.*?)</a>", r"\1", s, flags=re.S)
    s = re.sub(r"<br\s*/?>", " — ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = H.unescape(s).replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"``+", "`", s)          # 중첩 백틱 정리


def strip(s: str) -> str:
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", "", s))).strip()


def table(block: str) -> str:
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", block, flags=re.S):
        cells = [inline(c).replace("|", "\\|")
                 for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.S)]
        rows.append(cells)
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "|" + "|".join([" --- "] * w) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


out, pos = [], 0
BLOCK = re.compile(
    r'<h([1-4])>(.*?)</h\1>'
    r'|<p>(.*?)</p>'
    r'|<div class="term">\s*<div class="term__tag">(.*?)</div>\s*<pre>(.*?)</pre>\s*</div>'
    r'|<div class="note([^"]*)">(.*?)</div>'
    r'|<div class="plate" data-kind="(.*?)" data-id="(.*?)">(.*?)</div>'
    r'|<table[^>]*>(.*?)</table>'
    r'|<ol class="steps">(.*?)</ol>'
    r'|<nav class="toc">(.*?)</nav>'
    r'|<section id="[^"]*">',
    flags=re.S)

for m in BLOCK.finditer(body):
    g = m.groups()
    if g[0]:                                            # h1..h4
        out.append("#" * int(g[0]) + " " + inline(g[1]))
    elif g[2] is not None and "class=" not in m.group(0):
        t = inline(g[2])
        if t: out.append(t)
    elif g[3] is not None:                              # 터미널 블록
        tag = strip(g[3])
        # 실제로 치는 명령만 bash 로 — 파일 흐름·화면 출력은 언어 없이 둔다.
        lang = "bash" if any(k in tag for k in ("호스트", "컨테이너", "SIM", "REAL", "셸")) else ""
        out.append(f"**{tag}**")
        out.append(f"```{lang}\n" + H.unescape(re.sub(r"<[^>]+>", "", g[4])).strip("\n") + "\n```")
    elif g[5] is not None:                              # 콜아웃
        variant, inner = g[5], g[6]
        icon = "⚠️" if "warn" in variant else ("🤖" if "real" in variant else "💡")
        lab = re.search(r'<span class="note__label">(.*?)</span>', inner, flags=re.S)
        lines = [f"> {icon} **{strip(lab.group(1)) if lab else ''}**"]
        for p in re.findall(r"<p>(.*?)</p>", inner, flags=re.S):
            lines += [">", "> " + inline(p)]
        out.append("\n".join(lines))
    elif g[7] is not None:                              # 사진/영상 자리
        p = re.search(r"<p>(.*?)</p>", g[9], flags=re.S)
        out.append(f"> 📷 **{g[8]} · {g[7]}**\n>\n> " + (inline(p.group(1)) if p else ""))
    elif g[10] is not None:                             # 표
        out.append(table(g[10]))
    elif g[11] is not None:                             # 순서 목록
        for i, li in enumerate(re.findall(r"<li>(.*?)</li>", g[11], flags=re.S), 1):
            out.append(f"{i}. {inline(li)}")
    elif g[12] is not None:                             # 목차
        # 노션 목차 블록은 마크다운 문법이 없다 — 에디터에서만 만들어진다. 그래서 링크
        # 목록을 옮기지 않고, 그 자리에 무엇을 넣어야 하는지 적은 한 줄을 남긴다.
        # (헤딩에서 자동 생성되므로 항목을 손으로 유지할 필요도 없다.)
        out.append("> 💡 **여기에 노션 목차 블록을 넣는다**\n>\n"
                   "> 이 줄에서 `/목차` (영문 `/toc`) 를 입력하면 아래 헤딩으로 목차가 "
                   "자동 생성된다. 넣은 뒤 이 안내는 지운다.")
    else:                                               # <section> 경계
        out.append("---")

md = "\n\n".join(x for x in out if x.strip())
md = re.sub(r"\n{3,}", "\n\n", md)
md = re.sub(r"(?m)^(\d+)\. (.*)\n\n(?=\d+\. )", r"\1. \2\n", md)   # 번호 목록 붙이기
Path("docs/site/looksgood.md").write_text(md + "\n")
print(f"docs/site/looksgood.md  ({len(md.splitlines())} lines, {len(md)} chars)")
