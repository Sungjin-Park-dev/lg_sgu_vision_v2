#!/usr/bin/env python3
"""인쇄용 ChArUco 보드 — 정확한 A4 PDF(+같은 보드의 SVG).

스캔 보드는 물체를 가운데에 올려 두고 매 장 카메라 위치를 잴 용도라 크게 만든다: 물체가 가운데와
그 뒤쪽 코너를 가려도 앞·옆 코너가 남는다. 행 수를 홀수로 두면 OpenCV 4.6 전후 배치 차이(legacy)가
없다. 가장자리 바깥의 짧은 눈금이 보드 중심선 — 물체를 놓을 자리.

    .venv/bin/python scripts/setup/make_charuco_board.py                     # 9×7, 25 mm (A4 가로)
    .venv/bin/python scripts/setup/make_charuco_board.py --squares 6 5 --square-mm 20 --marker-mm 15 \
        --out docs/assets/piper_charuco_20mm.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from core.realscan.calibration import CharucoBoard  # noqa: E402

PX_PER_MM = 20                      # 508 dpi — 칸·마커 칸이 정수 픽셀
PAGES_MM = {"a4": (297, 210), "a3": (420, 297)}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def page_image(board: CharucoBoard, page_mm):
    from PIL import Image, ImageDraw, ImageFont

    s = board.spec
    w, h = (s[k] * s["square_m"] * 1000 for k in ("squares_x", "squares_y"))
    if (w > h) != (page_mm[0] > page_mm[1]):          # 보드 긴 변에 맞춰 가로·세로
        page_mm = page_mm[::-1]
    margin = ((page_mm[0] - w) / 2, (page_mm[1] - h) / 2)
    if min(margin) < 12:
        raise SystemExit(f"보드 {w:g}×{h:g} mm 가 {page_mm[0]}×{page_mm[1]} mm 쪽에 안 들어갑니다 (여백 12 mm 필요)")
    px = lambda mm: int(round(mm * PX_PER_MM))  # noqa: E731
    page = Image.new("L", (px(page_mm[0]), px(page_mm[1])), 255)
    page.paste(Image.fromarray(board.image(PX_PER_MM)), (px(margin[0]), px(margin[1])))
    draw = ImageDraw.Draw(page)
    line = px(0.3)
    x0, y0, x1, y1 = margin[0], margin[1], margin[0] + w, margin[1] + h
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for a, b in (((cx, y0 - 8), (cx, y0 - 3)), ((cx, y1 + 3), (cx, y1 + 8)),        # 중심선 눈금
                 ((x0 - 8, cy), (x0 - 3, cy)), ((x1 + 3, cy), (x1 + 8, cy))):
        draw.line([px(a[0]), px(a[1]), px(b[0]), px(b[1])], fill=0, width=line)
    ruler = 2 * s["square_m"] * 1000                                                 # 두 칸 길이 자
    ry = y1 + 6
    draw.line([px(x0), px(ry), px(x0 + ruler), px(ry)], fill=0, width=line)
    for k in range(int(ruler // 10) + 1):
        tick = 2 if k % 5 else 3.5
        draw.line([px(x0 + 10 * k), px(ry - tick), px(x0 + 10 * k), px(ry)], fill=0, width=line)
    font = ImageFont.truetype(FONT, px(2.6))
    draw.text((px(x0), px(ry + 1)), f"{ruler:g} mm", font=font, fill=0)
    text = (f"ChArUco {s['squares_x']}x{s['squares_y']}  square {s['square_m'] * 1000:g} mm  "
            f"marker {s['marker_m'] * 1000:g} mm  {s['dictionary']}\n"
            f"Print at 100% (actual size) - check one square = {s['square_m'] * 1000:g} mm")
    draw.multiline_text((px(x1), px(y1 + 3)), text, font=font, fill=0, anchor="ra", align="right")
    return page, page_mm


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--squares", type=int, nargs=2, default=(9, 7), metavar=("X", "Y"))
    p.add_argument("--square-mm", type=float, default=25.0)
    p.add_argument("--marker-mm", type=float, default=18.0)
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--page", choices=sorted(PAGES_MM), default="a4")
    p.add_argument("--out", type=Path, default=ROOT / "docs/assets/charuco_9x7_25mm.pdf",
                   help="PDF 경로 — 같은 이름의 .svg 도 쓴다")
    args = p.parse_args()
    board = CharucoBoard({"squares_x": args.squares[0], "squares_y": args.squares[1],
                          "square_m": args.square_mm / 1000, "marker_m": args.marker_mm / 1000,
                          "dictionary": args.dictionary, "layout": "opencv_4.6_even_row_legacy"})
    page, page_mm = page_image(board, PAGES_MM[args.page])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 흑백 1비트로 — PIL 은 회색조를 JPEG(DCT)로 넣어 모서리가 번진다. 1비트는 CCITT G4(무손실)
    page.convert("1", dither=0).save(args.out, "PDF", resolution=PX_PER_MM * 25.4)
    args.out.with_suffix(".svg").write_text(board.svg())
    print(f"{args.out} ({page_mm[0]}×{page_mm[1]} mm) · {args.out.with_suffix('.svg')}\n"
          f"보드 {board.spec} — 마커 {np.prod(args.squares) // 2}개")


if __name__ == "__main__":
    main()
