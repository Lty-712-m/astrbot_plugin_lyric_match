"""歌词库：文件解析、2-gram 倒排索引、模糊匹配。

这个模块不依赖 AstrBot，可以单独导入调试（方便本地验证匹配效果）。

支持两种歌词文件格式：

1. txt（与 tiyi_kb.txt 相同）::

       《歌名》（歌手）
         第一句歌词
         第二句歌词

2. json::

       {"songs": [{"title": "歌名", "lines": ["第一句", "第二句"]}]}

   兼容网易云风格的 list：``[{"title": "...", "snippet": [...]}]``。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# 《歌名》出现在行首
TITLE_RE = re.compile(r"^《([^《》]+)》")
# 标题尾部括号，如「黑凤梨（洛天依AI）」
BRACKET_TAIL_RE = re.compile(r"^(.*?)[（(]([^（）()]*)[）)]\s*$")
# LRC 时间戳 [00:12.34]
LRC_RE = re.compile(r"^\s*\[\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?\]\s*")
# LRC 元数据 [ti:歌名] [ar:歌手]
LRC_META_RE = re.compile(
    r"^\[(ti|ar|al|au|by|offset|re|ve|length|tool):\s*(.*?)\]\s*$",
    re.IGNORECASE,
)
LRC_EXT = (".txt", ".json", ".lrc")
# 行首的项目符号
LINE_PREFIX_RE = re.compile(r"^[\s\-*•·]+")
# 整行只有一个括号注释，如「（洛天依）」「(Live)」——是歌手/版本信息，不是歌词
ARTIST_LINE_RE = re.compile(r"^[（(][^（）()]{0,40}[）)]$")
# 括号里出现这些词时，认为它是歌手/版本信息而不是歌名的一部分
TITLE_ARTIST_KEYS = (
    "洛天依", "翻唱", "cover", "Cover", "COVER", "伴奏", "原唱", "remix",
    "Remix", "live", "Live", "女声", "男声", "纯音乐", "抖音", "AI", "ai",
    "版",
)
# 制作信息行（不是歌词）
CREDIT_KEYS = (
    "作词", "作曲", "编曲", "调声", "调教", "混音", "母带", "演唱", "原唱",
    "制作", "出品", "lyricist", "composer", "arranger", "producer",
    "mixing", "mastering", "tuning", "vocal",
)


def normalize(text: str) -> str:
    """归一化：只保留字母/数字/汉字/假名并转小写，去掉标点、空格、表情。"""
    if not text:
        return ""
    return "".join(ch.lower() for ch in text if ch.isalnum())


def grams(text: str) -> list[str]:
    """返回文本的去重 2-gram 列表。"""
    n = len(text)
    if n < 2:
        return [text] if text else []
    return list({text[i:i + 2] for i in range(n - 1)})


def clean_title(raw: str) -> str:
    """清理歌名：去掉《》、去掉尾部的（洛天依）/（翻唱）等版本信息。"""
    title = (raw or "").strip().strip("《》").strip()
    match = BRACKET_TAIL_RE.match(title)
    if match:
        inner = match.group(2).strip()
        if any(key in inner for key in TITLE_ARTIST_KEYS):
            title = match.group(1).strip()
    return title or (raw or "").strip()


def is_credit_line(line: str) -> bool:
    """判断是不是「作词：xxx」这类制作信息。"""
    if "：" not in line and ":" not in line:
        return False
    head = line.split("：", 1)[0].split(":", 1)[0].strip().lower()
    if not head or len(head) > 14:
        return False
    return any(key in head for key in CREDIT_KEYS)


@dataclass
class Song:
    title: str
    display_title: str
    artists: str = ""
    lines: list[str] = field(default_factory=list)
    source: str = ""


@dataclass
class LyricLine:
    song_index: int
    line_index: int
    text: str
    norm: str


@dataclass
class Match:
    score: float
    song: Song
    line: str
    line_index: int
    next_line: str = ""
    shared: int = 0


def line_score(line: str, query: str) -> float:
    """计算一句歌词与用户消息的相似度（0~1）。

    以「包含关系」为最强信号（用户完整念出一句 / 念出其中一段），
    再用 difflib 的编辑相似度兜底，适应错字、漏字、语气词。
    """
    if not line or not query:
        return 0.0
    if line == query:
        return 1.0

    # 歌词句完整出现在消息里（常见：消息里带「我觉得……这句好听」）
    # 短句（<5 字）不享受包含加成，避免「我爱你」这种到处都有的短语误命中
    if len(line) >= 5 and line in query:
        ratio = SequenceMatcher(None, line, query, autojunk=False).ratio()
        base = 0.80 + 0.20 * (len(line) / max(1, len(query)))
        return max(ratio, base)

    # 消息是歌词句的一部分（用户只哼了半句）
    if len(query) >= 5 and query in line:
        ratio = SequenceMatcher(None, line, query, autojunk=False).ratio()
        base = 0.82 + 0.18 * (len(query) / max(1, len(line)))
        return max(ratio, base)

    matcher = SequenceMatcher(None, line, query, autojunk=False)
    ratio = matcher.ratio()
    block = matcher.find_longest_match(0, len(line), 0, len(query))
    if block.size < 4:
        return ratio * 0.9
    short = min(len(line), len(query))
    cover = block.size / max(1, short)
    return max(ratio, cover * 0.88)


class LyricsLibrary:
    """歌词库：加载目录下所有 txt/json，建 2-gram 倒排索引，按需模糊匹配。"""

    def __init__(self, dirs: list[str] | None = None):
        self.dirs = [d for d in (dirs or []) if d]
        self.songs: list[Song] = []
        self.lines: list[LyricLine] = []
        self.index: dict[str, list[int]] = {}
        self.loaded_at: float = 0.0
        self.files: list[str] = []
        self.errors: list[str] = []
        self.skipped: int = 0

    # ---------------- 加载 ----------------

    def load(self, dirs: list[str] | None = None) -> dict:
        if dirs is not None:
            self.dirs = [d for d in dirs if d]
        self.songs = []
        self.lines = []
        self.index = {}
        self.files = []
        self.errors = []
        self.skipped = 0

        seen_titles: set[str] = set()
        for directory in self.dirs:
            if not os.path.isdir(directory):
                continue
            for root, sub_dirs, names in os.walk(directory):
                sub_dirs[:] = sorted(d for d in sub_dirs if not d.startswith("."))
                for name in sorted(names):
                    if name.startswith("."):
                        continue
                    lower = name.lower()
                    if not lower.endswith(LRC_EXT):
                        continue
                    # 说明文件不是歌词（lyrics/README.txt 里会有格式示例）
                    stem = os.path.splitext(lower)[0]
                    if stem.startswith(("readme", "license", "licence", "说明", "格式")):
                        continue
                    path = os.path.join(root, name)
                    if not os.path.isfile(path):
                        continue
                    try:
                        songs = self._parse_file(path)
                    except Exception as exc:  # 单个文件坏掉不影响整体
                        self.errors.append(f"{name}: {exc}")
                        continue
                    self.files.append(path)
                    for song in songs:
                        key = normalize(song.display_title)
                        if not key or key in seen_titles:
                            self.skipped += 1
                            continue
                        if self._add_song(song, path):
                            seen_titles.add(key)
                        else:
                            self.skipped += 1

        self._rebuild_index()
        self.loaded_at = time.time()
        return self.stats()

    def _parse_file(self, path: str) -> list[Song]:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
            raw = handle.read()
        lower = path.lower()
        if lower.endswith(".json"):
            try:
                data = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"JSON 解析失败：{exc}") from exc
            return self._parse_json(data)
        if lower.endswith(".lrc"):
            stem = os.path.splitext(os.path.basename(path))[0]
            return self._parse_lrc(raw, stem)
        return self._parse_txt(raw)

    def _parse_lrc(self, raw: str, fallback_title: str) -> list[Song]:
        """LRC：靠 [ti:] 或文件名取歌名，逐行剥掉时间戳后当歌词。"""
        title = ""
        lines: list[str] = []
        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            meta = LRC_META_RE.match(line)
            if meta:
                key = meta.group(1).lower()
                value = meta.group(2).strip()
                if key == "ti" and value:
                    title = clean_title(value)
                continue
            # 一行可能带多个时间戳：[00:12.34][01:20.00]歌词
            while True:
                stamp = LRC_RE.match(line)
                if not stamp:
                    break
                line = line[stamp.end():].strip()
            if not line:
                continue
            if is_credit_line(line):
                continue
            if len(normalize(line)) < 2:
                continue
            if lines and lines[-1] == line:
                continue  # 同一句连续重复时只留一条
            lines.append(line)

        if not title:
            stem = re.sub(r"^\d{1,3}[\s._-]+", "", fallback_title).strip()
            title = clean_title(stem or fallback_title)
        if not title or not lines:
            return []
        return [Song(title=title, display_title=title, lines=lines)]

    def file_summary(self, path: str) -> dict:
        """单文件摘要（页面展示用）：首数、句数、解析错误。"""
        try:
            songs = self._parse_file(path)
        except Exception as exc:
            return {"songs": 0, "lines": 0, "error": str(exc)}
        songs = [song for song in songs if song.lines]
        return {
            "songs": len(songs),
            "lines": sum(len(song.lines) for song in songs),
            "error": "",
        }

    def _parse_txt(self, raw: str) -> list[Song]:
        songs: list[Song] = []
        current: Song | None = None
        for raw_line in raw.splitlines():
            line = LINE_PREFIX_RE.sub("", raw_line).strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("//"):
                continue
            if line.startswith("【"):  # 分节标题，如【洛天依 · 我喜欢的歌】
                continue
            title_match = TITLE_RE.match(line)
            if title_match:
                artists = ""
                rest = line[title_match.end():].strip()
                if rest.startswith(("（", "(")):
                    artists = rest.strip("（）() ")
                title = clean_title(title_match.group(1))
                current = Song(title=title, display_title=title, artists=artists)
                songs.append(current)
                continue
            if current is None:
                continue
            if is_credit_line(line):
                continue
            line = LRC_RE.sub("", line).strip()
            if not line or len(normalize(line)) < 2:
                continue
            # 标题下一行如果是「（洛天依）」这种纯括号注释，当作歌手信息
            if not current.lines and ARTIST_LINE_RE.match(line):
                if not current.artists:
                    current.artists = line.strip("（）() ")
                continue
            current.lines.append(line)
        return [song for song in songs if song.lines]

    def _parse_json(self, data) -> list[Song]:
        if isinstance(data, dict):
            seq = data.get("songs")
            if seq is None and any(k in data for k in ("title", "song", "lines", "snippet")):
                seq = [data]
            if not isinstance(seq, list):
                seq = []
        elif isinstance(data, list):
            seq = data
        else:
            seq = []

        songs: list[Song] = []
        for item in seq:
            if isinstance(item, str):
                title = clean_title(item)
                if title:
                    songs.append(Song(title=title, display_title=title))
                continue
            if not isinstance(item, dict):
                continue
            raw_title = item.get("title") or item.get("song") or item.get("name") or ""
            title = clean_title(str(raw_title))
            if not title:
                continue
            artists = item.get("artists") or item.get("artist") or item.get("singer") or ""
            lines = item.get("lines")
            if lines is None:
                lines = item.get("snippet")
            if lines is None:
                lines = item.get("lyrics")
            if lines is None:
                lines = []
            if isinstance(lines, str):
                lines = lines.splitlines()
            cleaned: list[str] = []
            if isinstance(lines, list):
                for raw_line in lines:
                    if not isinstance(raw_line, str):
                        continue
                    line = LRC_RE.sub("", raw_line).strip()
                    if not line or is_credit_line(line):
                        continue
                    if len(normalize(line)) < 2:
                        continue
                    cleaned.append(line)
            songs.append(Song(title=title, display_title=title,
                              artists=str(artists), lines=cleaned))
        return [song for song in songs if song.lines]

    def _add_song(self, song: Song, path: str) -> bool:
        cleaned: list[str] = []
        for raw_line in song.lines:
            line = LINE_PREFIX_RE.sub("", str(raw_line)).strip()
            if not line:
                continue
            if len(normalize(line)) < 2:
                continue
            cleaned.append(line)
        if not cleaned:
            return False
        song.lines = cleaned
        song.source = path
        song_index = len(self.songs)
        self.songs.append(song)
        for line_index, line in enumerate(cleaned):
            self.lines.append(LyricLine(
                song_index=song_index,
                line_index=line_index,
                text=line,
                norm=normalize(line),
            ))
        return True

    def _rebuild_index(self) -> None:
        index: dict[str, list[int]] = {}
        for line_id, line in enumerate(self.lines):
            for gram in grams(line.norm):
                index.setdefault(gram, []).append(line_id)
        self.index = index

    # ---------------- 匹配 ----------------

    def search(self, query: str, top_k: int = 20) -> list[Match]:
        """用 2-gram 倒排索引取候选句，再对候选做相似度计算。"""
        query_norm = normalize(query)
        if len(query_norm) < 2 or not self.lines:
            return []

        shared: dict[int, int] = {}
        for gram in grams(query_norm):
            for line_id in self.index.get(gram, ()):
                shared[line_id] = shared.get(line_id, 0) + 1
        if not shared:
            return []

        top_k = max(1, min(int(top_k), 200))
        min_shared = 2 if len(query_norm) >= 4 else 1
        ranked = sorted(
            shared.items(),
            key=lambda item: (-item[1], -len(self.lines[item[0]].norm)),
        )
        candidates = [item for item in ranked if item[1] >= min_shared][:top_k]
        if not candidates:
            candidates = ranked[:top_k]

        results: list[Match] = []
        for line_id, shared_count in candidates:
            lyric_line = self.lines[line_id]
            song = self.songs[lyric_line.song_index]
            next_line = ""
            if lyric_line.line_index + 1 < len(song.lines):
                next_line = song.lines[lyric_line.line_index + 1]
            results.append(Match(
                score=line_score(lyric_line.norm, query_norm),
                song=song,
                line=lyric_line.text,
                line_index=lyric_line.line_index,
                next_line=next_line,
                shared=shared_count,
            ))
        results.sort(key=lambda match: (match.score, match.shared), reverse=True)
        return results

    def best(self, query: str, top_k: int = 20) -> Match | None:
        matches = self.search(query, top_k)
        return matches[0] if matches else None

    # ---------------- 统计 ----------------

    def stats(self) -> dict:
        return {
            "songs": len(self.songs),
            "lines": len(self.lines),
            "grams": len(self.index),
            "files": len(self.files),
            "skipped": self.skipped,
            "dirs": list(self.dirs),
            "errors": list(self.errors),
            "loaded_at": self.loaded_at,
        }
