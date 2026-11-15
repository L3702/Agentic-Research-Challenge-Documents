"""
pdf_to_text.py — 从 PDF（或 zip）中抽取中文正文
===============================================
适用于本项目的两类 PDF：
  A. 简单字体 + GBK 编码（部分期刊 PDF）；
  B. 子集字体 + 自定义编码 + 内嵌 ToUnicode CMap（知网/Word 导出常见）。

做法：解析 PDF 对象表 → 找到每个字体的 ToUnicode CMap → 按内容流里的
`/Font 字号 Tf` 切换当前字体映射 → 逐字符解码。纯标准库实现。

用法：
    python src/data/pdf_to_text.py <文件.pdf | 文件.zip> <输出目录>

输出：每篇一个 .txt；同时打印中文字数，若过少会提示（可能是扫描件）。
"""

from __future__ import annotations

import os
import re
import sys
import zipfile
import zlib

# ------------------------------------------------------------------ 对象表
OBJ_RE = re.compile(rb"(\d+)\s+(\d+)\s+obj\b")
STREAM_RE = re.compile(rb"stream\r?\n(.*?)endstream", re.S)


def build_objects(raw: bytes) -> dict[int, bytes]:
    """把 PDF 解析成 {对象号: 对象体}，并展开压缩对象流（/ObjStm）。

    PDF 1.5+ 会把字体、页面等对象打包进 /ObjStm 里压缩存放，
    只按 "N 0 obj" 扫描会漏掉它们（表现：明明有文字，却找不到任何字体）。
    """
    objs: dict[int, bytes] = {}
    starts = [(int(m.group(1)), m.end()) for m in OBJ_RE.finditer(raw)]
    for idx, (num, start) in enumerate(starts):
        end = raw.find(b"endobj", start)
        if end < 0:
            end = starts[idx + 1][1] if idx + 1 < len(starts) else len(raw)
        objs[num] = raw[start:end]

    # 展开对象流
    for num, body in list(objs.items()):
        if b"/ObjStm" not in body:
            continue
        data = stream_bytes(body)
        if not data:
            continue
        m = re.match(rb"\s*(\d+)\s+(\d+)\s", data)
        if not m:
            continue
        n, first = int(m.group(1)), int(m.group(2))
        head = data[:first].split()
        pairs = [(int(head[i]), int(head[i + 1]))
                 for i in range(0, min(len(head) - 1, 2 * n), 2)]
        for idx, (onum, off) in enumerate(pairs):
            s = first + off
            e = first + pairs[idx + 1][1] if idx + 1 < len(pairs) else len(data)
            objs.setdefault(onum, data[s:e])
    return objs


def stream_bytes(body: bytes) -> bytes | None:
    """取出对象里的流并解压。"""
    m = STREAM_RE.search(body)
    if not m:
        return None
    data = m.group(1)
    try:
        return zlib.decompress(data)
    except Exception:  # noqa: BLE001
        return data


# --------------------------------------------------------------- ToUnicode
def _hex_to_str(h: bytes) -> str:
    b = bytes.fromhex(h.decode("ascii"))
    try:
        return b.decode("utf-16-be")
    except UnicodeDecodeError:
        return b.decode("latin-1", "replace")


def parse_cmap(text: bytes) -> dict[int, str]:
    """解析 ToUnicode CMap 的 bfchar / bfrange（含数组形式）。"""
    cmap: dict[int, str] = {}
    for block in re.findall(rb"beginbfchar(.*?)endbfchar", text, re.S):
        for src, dst in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            cmap[int(src, 16)] = _hex_to_str(dst)

    # bfrange 有两种写法：
    #   <lo> <hi> <dst>                  —— 目标按顺序递增
    #   <lo> <hi> [<d1> <d2> ...]        —— 逐个给出（常见，之前漏掉了！）
    RANGE_RE = re.compile(
        rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[[^\]]*\]|<[0-9A-Fa-f]+>)", re.S)
    for block in re.findall(rb"beginbfrange(.*?)endbfrange", text, re.S):
        for lo_h, hi_h, dst in RANGE_RE.findall(block):
            lo, hi = int(lo_h, 16), int(hi_h, 16)
            if dst.startswith(b"["):
                items = re.findall(rb"<([0-9A-Fa-f]+)>", dst)
                for k, item in enumerate(items):
                    if lo + k > hi:
                        break
                    cmap[lo + k] = _hex_to_str(item)
            else:
                base = int(re.sub(rb"[<>]", b"", dst), 16)
                for k in range(lo, hi + 1):
                    code = base + (k - lo)
                    if code > 0x10FFFF:
                        break
                    try:
                        cmap[k] = chr(code)
                    except ValueError:
                        break
    return cmap


FONT_DICT_RE = re.compile(rb"/Font\s*<<(.*?)>>", re.S)
FONT_ENTRY_RE = re.compile(rb"/([A-Za-z0-9#\-_.+]+)\s+(\d+)\s+0\s+R")
TOUNICODE_RE = re.compile(rb"/ToUnicode\s+(\d+)\s+0\s+R")
TYPE0_RE = re.compile(rb"/Subtype\s*/Type0")


def build_font_maps(raw: bytes) -> dict[str, tuple[dict[int, str], int]]:
    """返回 {资源名(如 F2): (cmap, 每字符字节数)}。"""
    objs = build_objects(raw)

    name2ref: dict[str, int] = {}
    # 字体字典可能被包在对象流里，所以要在展开后的对象里找
    for body in objs.values():
        for m in FONT_DICT_RE.finditer(body):
            for name, ref in FONT_ENTRY_RE.findall(m.group(1)):
                name2ref.setdefault(name.decode("latin-1"), int(ref))
    for m in FONT_DICT_RE.finditer(raw):
        for name, ref in FONT_ENTRY_RE.findall(m.group(1)):
            name2ref.setdefault(name.decode("latin-1"), int(ref))

    out: dict[str, tuple[dict[int, str], int]] = {}
    for name, ref in name2ref.items():
        body = objs.get(ref, b"")
        width = 2 if TYPE0_RE.search(body) else 1
        tu = TOUNICODE_RE.search(body)
        if not tu:
            continue
        cm_body = objs.get(int(tu.group(1)))
        if cm_body is None:
            continue
        data = stream_bytes(cm_body)
        if not data:
            continue
        cmap = parse_cmap(data)
        if cmap:
            out[name] = (cmap, width)
    return out


# ------------------------------------------------------------------ 解码
def unescape_pdf_string(b: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(b):
        c = b[i]
        if c != 0x5C:
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= len(b):
            break
        n = b[i]
        if n in b"nrtbf":
            out.append({0x6E: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12}[n])
            i += 1
        elif n in b"()\\":
            out.append(n)
            i += 1
        elif 0x30 <= n <= 0x37:
            j = i
            while j < len(b) and j < i + 3 and 0x30 <= b[j] <= 0x37:
                j += 1
            out.append(int(b[i:j], 8) & 0xFF)
            i = j
        elif n in b"\r\n":
            i += 1
        else:
            out.append(n)
            i += 1
    return bytes(out)


def decode_gbk(b: bytes) -> str:
    for enc in ("gb18030", "gbk", "utf-8"):
        try:
            s = b.decode(enc)
        except UnicodeDecodeError:
            continue
        bad = sum(1 for ch in s if ord(ch) < 32 and ch not in "\n\t")
        if bad <= max(1, len(s) // 20):
            return s
    return b.decode("latin-1", "replace")


def decode_bytes(b: bytes, font: tuple[dict[int, str], int] | None) -> str:
    if font:
        cmap, width = font
        out = []
        if width == 2:
            for i in range(0, len(b) - 1, 2):
                code = (b[i] << 8) | b[i + 1]
                out.append(cmap.get(code, ""))
        else:
            for byte in b:
                out.append(cmap.get(byte, chr(byte) if 32 <= byte < 127 else ""))
        s = "".join(out)
        if s.strip():
            return s
    return decode_gbk(b)


# ---------------------------------------------------------------- 内容流
NUM_RE = re.compile(rb"-?\d+(?:\.\d+)?")


def _parse_literal(data: bytes, i: int) -> tuple[bytes, int]:
    depth, i, buf = 1, i + 1, bytearray()
    n = len(data)
    while i < n:
        c = data[i]
        if c == 0x5C:
            buf.append(c)
            if i + 1 < n:
                buf.append(data[i + 1])
            i += 2
            continue
        if c == 0x28:
            depth += 1
        elif c == 0x29:
            depth -= 1
            if depth == 0:
                return bytes(buf), i + 1
        buf.append(c)
        i += 1
    return bytes(buf), i


def _parse_hexstr(data: bytes, i: int) -> tuple[bytes, int]:
    j = data.find(b">", i + 1)
    if j < 0:
        return b"", i + 1
    h = re.sub(rb"\s", b"", data[i + 1:j])
    if len(h) % 2:
        h += b"0"
    try:
        return bytes.fromhex(h.decode("ascii")), j + 1
    except ValueError:
        return b"", j + 1


def stream_to_text(stream: bytes, fonts: dict[str, tuple[dict[int, str], int]]) -> str:
    parts: list[str] = []
    cur: tuple[dict[int, str], int] | None = None
    i, n = 0, len(stream)
    while i < n:
        c = stream[i]
        if stream[i:i + 2] == b"Tf":  # 切换字体：形如 "/F3 8.5 Tf"
            back = stream[max(0, i - 64):i]
            m = re.search(rb"/([A-Za-z0-9#\-_.+]+)\s+[\d.]+\s*$", back)
            if m:
                cur = fonts.get(m.group(1).decode("latin-1"))
            i += 2
        elif c == 0x28:  # '('
            s, i = _parse_literal(stream, i)
            parts.append(decode_bytes(unescape_pdf_string(s), cur))
        elif c == 0x3C and stream[i:i + 2] != b"<<":  # '<' 十六进制串
            s, i = _parse_hexstr(stream, i)
            parts.append(decode_bytes(s, cur))
        elif c == 0x5B:  # '[' TJ 数组
            j = i + 1
            chunk = []
            while j < n and stream[j] != 0x5D:
                if stream[j] == 0x28:
                    s, j = _parse_literal(stream, j)
                    chunk.append(decode_bytes(unescape_pdf_string(s), cur))
                elif stream[j] == 0x3C and stream[j:j + 2] != b"<<":
                    s, j = _parse_hexstr(stream, j)
                    chunk.append(decode_bytes(s, cur))
                else:
                    j += 1
            parts.append("".join(chunk))
            i = j + 1
        elif stream[i:i + 2] == b"TD":
            nums = NUM_RE.findall(stream[max(0, i - 32):i])
            if len(nums) >= 2 and abs(float(nums[-1])) > 0.01:
                parts.append("\n")
            i += 2
        elif stream[i:i + 2] == b"ET":
            parts.append("\n")
            i += 2
        else:
            i += 1
    return "".join(parts)


# ------------------------------------------------------------------ 主流程
def pdf_text(raw: bytes) -> str:
    fonts = build_font_maps(raw)
    chunks = []
    for body in build_objects(raw).values():
        st = stream_bytes(body)
        if not st or (b"Tj" not in st and b"TJ" not in st):
            continue
        t = stream_to_text(st, fonts)
        if t.strip():
            chunks.append(t)
    s = "".join(chunks)
    s = "".join(ch for ch in s
                if (ord(ch) >= 32 or ch in "\n\t") and not (0xE000 <= ord(ch) <= 0xF8FF))
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    jobs: list[tuple[str, bytes]] = []
    if src.lower().endswith(".zip"):
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                if not info.filename.lower().endswith(".pdf"):
                    continue
                name = info.filename
                try:
                    name = name.encode("cp437").decode("gbk")
                except Exception:  # noqa: BLE001
                    pass
                jobs.append((name, z.read(info)))
    else:
        jobs.append((os.path.basename(src), open(src, "rb").read()))

    for name, data in jobs:
        txt = pdf_text(data)
        stem = os.path.splitext(os.path.basename(name))[0]
        safe = re.sub(r'[\\/:*?"<>|]', "_", stem)
        dst = os.path.join(outdir, safe + ".txt")
        with open(dst, "w", encoding="utf-8") as f:
            f.write(txt)
        cjk = sum(1 for ch in txt if "\u4e00" <= ch <= "\u9fff")
        print(f"{len(txt):>8} 字符 | 中文字 {cjk:>6} | {os.path.basename(dst)}")
        if cjk < 500:
            print("         [!] 中文过少：可能仍是扫描件，或该 PDF 未内嵌 ToUnicode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
