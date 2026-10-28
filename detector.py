"""第一段判断：这条消息「像不像在念歌词」。

不调用大模型，纯规则预筛，成本几乎为零。
思路（与需求一致）：

- 长度太短 → 不像（默认 ≥ 8 个字符）
- 日常词 / 问号 / 链接 / @ / 表情过多 → 明显是聊天，直接否
- 「突兀性 / 句式」：短句、少标点、句尾成韵
- 抒情词覆盖率、第二人称「你」+ 抒情词 → 加分

规则只负责「粗筛」，真正的确认交给歌词库匹配的置信度（默认 0.85）。
"""

from __future__ import annotations

import re
import unicodedata

from .lyrics_lib import normalize

# 抒情/意象词：命中率越高越像歌词
LYRIC_WORDS = (
    # 单字意象
    "光", "风", "夜", "心", "梦", "雨", "星", "海", "月", "云", "雪", "花",
    "泪", "歌", "唱", "爱", "念", "远", "空", "城", "街", "灯", "影", "声",
    "笑", "哭", "飞", "落", "等", "望", "醉", "孤", "白", "黑", "蓝", "红",
    "春", "夏", "秋", "冬", "年", "时", "光", "暖", "凉", "甜", "苦", "痛",
    # 双字/多字意象
    "世界", "永远", "天空", "回忆", "时光", "温柔", "寂寞", "孤独", "孤单",
    "拥抱", "眼泪", "微笑", "心动", "喜欢", "想念", "思念", "青春", "远方",
    "自由", "流浪", "翅膀", "童话", "烟火", "黄昏", "清晨", "深夜", "故事",
    "旋律", "和弦", "歌唱", "未来", "曾经", "守护", "承诺", "约定", "距离",
    "风景", "城市", "街道", "阳光", "月光", "星光", "大海", "海浪", "晴天",
    "雨天", "季节", "秋天", "冬天", "夏天", "春天", "心底", "心脏", "灵魂",
    "安静", "沉默", "执着", "等待", "期待", "希望", "失望", "心疼", "遗憾",
    "放下", "离开", "回来", "记得", "忘记", "名字", "呼吸", "心跳", "梦里",
    "耳边", "身旁", "白衣", "少年", "人间", "山河", "江湖", "红尘", "岁月",
)

# 明显日常聊天词：占比过高直接判否
DAILY_WORDS = (
    "吃饭", "吃", "饭", "饿", "渴", "外卖", "奶茶", "火锅", "烧烤", "kfc",
    "麦当劳", "汉堡", "可乐", "零食", "游戏", "上分", "排位", "王者", "原神",
    "上班", "下班", "工作", "加班", "公司", "老板", "同事", "开会", "请假",
    "工资", "睡觉", "睡", "起床", "洗澡", "洗头", "刷牙", "化妆", "考试",
    "作业", "上课", "下课", "学校", "老师", "同学", "宿舍", "地铁", "公交",
    "打车", "堵车", "快递", "网购", "淘宝", "医院", "看病", "感冒", "发烧",
    "咳嗽", "哈哈", "笑死", "无语", "离谱", "绝了", "卧槽", "牛逼", "在吗",
    "干嘛", "咋", "啥", "怎么", "为什么", "手机", "充电", "电量", "流量",
    "wifi", "电脑", "键盘", "鼠标", "追剧", "综艺", "直播", "抖音", "微信",
    "红包", "转账", "点赞", "关注", "天气", "带伞", "衣服", "鞋子", "裤子",
    "超市", "食堂", "做饭", "洗碗", "拖地", "打扫", "厕所", "沙发", "枕头",
    "累", "困", "上班族", "摸鱼", "摆烂",
)

# 明显在「对话/提问」的短语
CHAT_PHRASES = (
    "帮我", "请问", "能不能", "可不可以", "是不是", "要不要", "有没有",
    "怎么样", "怎么办", "你说呢", "在吗", "睡了吗", "吃了吗", "行不行",
    "好不好", "对不对", "为什么", "我觉得", "我想说", "你猜", "随便",
    "算了", "嗯嗯", "哦哦", "好的", "行吧", "笑死", "救命", "emmm",
)

_CLAUSE_SPLIT_RE = re.compile(
    r"[，,。.！!？?；;、：:\s\n\r\t~～…—\-·/\\|（）()【】\[\]「」『』\"'“”‘’]+"
)
_PUNCT_RE = re.compile(r"[，,。.！!？?；;、]")
_EMOJI_CATEGORIES = ("So", "Sk")


def _clauses(text: str) -> list[str]:
    parts = [part.strip() for part in _CLAUSE_SPLIT_RE.split(text)]
    return [part for part in parts if part and normalize(part)]


def _count_words(low: str, words: tuple[str, ...]) -> tuple[int, int]:
    """返回 (命中次数, 覆盖字符数)。"""
    hits = 0
    covered = 0
    for word in words:
        count = low.count(word)
        if count:
            hits += count
            covered += count * len(word)
    return hits, covered


def _covered_chars(norm: str, words: tuple[str, ...]) -> int:
    """统计被词表覆盖到的字符数（同一字符不重复计）。"""
    covered: set[int] = set()
    for word in words:
        start = norm.find(word)
        while start != -1:
            for index in range(start, start + len(word)):
                covered.add(index)
            start = norm.find(word, start + 1)
    return len(covered)


def _structure_score(raw: str, norm: str, clauses: list[str]) -> float:
    count = len(clauses)
    if count == 0:
        return 0.3
    lengths = [len(normalize(clause)) for clause in clauses]
    average = sum(lengths) / max(1, len(lengths))

    if 2 <= count <= 6 and 2 <= average <= 12:
        score = 1.0
    elif count == 1 and 5 <= len(norm) <= 24:
        score = 0.8
    elif count > 6:
        score = 0.45
    else:
        score = 0.4

    # 整句没有标点，是歌词最典型的形态
    if count == 1 and not _PUNCT_RE.search(raw):
        score = max(score, 0.9)
    return score


def analyze(text: str, min_length: int = 8) -> dict:
    """分析一条消息，返回特征与判定结果。

    ``ok`` 表示通过了规则预筛（可以被第二段继续确认），
    ``reason`` 在 ok=False 时说明被否的原因，便于调试。
    """
    raw = (text or "").strip()
    norm = normalize(raw)
    info: dict = {
        "ok": False,
        "reason": "",
        "score": 0.0,
        "len": len(norm),
        "details": {},
    }
    if not raw or not norm:
        info["reason"] = "empty"
        return info
    if len(norm) < max(1, int(min_length)):
        info["reason"] = "too_short"
        return info
    if len(norm) > 240:
        info["reason"] = "too_long"
        return info

    low = raw.lower()
    if "http://" in low or "https://" in low or "[cq:" in low or "@" in raw:
        info["reason"] = "link_or_at"
        return info

    # ---- 日常词 / 问号占比 ----
    daily_hits, daily_chars = _count_words(low, DAILY_WORDS)
    question_marks = low.count("?") + low.count("？")
    exclaim_marks = low.count("!") + low.count("！")
    daily_chars += 3 * question_marks + exclaim_marks
    daily_ratio = daily_chars / max(1, len(norm))
    if daily_hits >= 3 or daily_ratio >= 0.22:
        info["reason"] = "daily_words"
        info["details"] = {"daily_hits": daily_hits, "daily_ratio": round(daily_ratio, 3)}
        return info

    clauses = _clauses(raw)
    structure = _structure_score(raw, norm, clauses)

    # ---- 抒情词覆盖 ----
    lyric_coverage = _covered_chars(norm, LYRIC_WORDS) / max(1, len(norm))
    lyric_score = min(1.0, lyric_coverage / 0.22)

    # ---- 第二人称「你」+ 抒情词 ----
    if "你" in norm:
        second_person = 1.0 if lyric_coverage >= 0.08 else 0.6
    else:
        second_person = 0.0

    # ---- 日常度（反向） ----
    daily_score = max(0.0, 1.0 - daily_ratio / 0.22)

    # ---- 对话感（反向） ----
    chat_hits, _ = _count_words(low, CHAT_PHRASES)
    chat_score = 1.0 if chat_hits == 0 else max(0.0, 1.0 - 0.3 * chat_hits)
    if question_marks:
        chat_score *= 0.4
    emoji = sum(1 for ch in raw if unicodedata.category(ch) in _EMOJI_CATEGORIES)
    if emoji:
        chat_score *= 0.5

    # ---- 韵脚：句尾字重复（无拼音依赖的近似） ----
    endings = [normalize(clause)[-1] for clause in clauses if normalize(clause)]
    rhyme = 1.0 if len(endings) >= 2 and len(set(endings)) < len(endings) else 0.0

    score = (
        0.25 * structure
        + 0.30 * lyric_score
        + 0.20 * daily_score
        + 0.10 * chat_score
        + 0.10 * second_person
        + 0.05 * rhyme
    )
    score = max(0.0, min(1.0, score))

    info["ok"] = True
    info["score"] = score
    info["details"] = {
        "structure": round(structure, 3),
        "lyric_coverage": round(lyric_coverage, 3),
        "daily_hits": daily_hits,
        "daily_ratio": round(daily_ratio, 3),
        "chat_hits": chat_hits,
        "question_marks": question_marks,
        "emoji": emoji,
        "clauses": len(clauses),
        "rhyme": rhyme,
    }
    return info


def is_lyric_like(text: str, min_length: int = 8, threshold: float = 0.5) -> tuple[bool, dict]:
    """规则预筛：返回 (是否像歌词, 分析详情)。"""
    info = analyze(text, min_length=min_length)
    passed = bool(info["ok"] and info["score"] >= float(threshold))
    return passed, info
