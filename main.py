"""AstrBot 插件：歌词识别。

在消息送进大模型之前介入：

1. 规则预筛（detector.py）判断「这句话像不像在念歌词」；
2. 像的话，到本地歌词库（lyrics_lib.py）做 2-gram 索引 + 模糊匹配；
3. 两段都通过，才把「歌名 + 原句」注入 system prompt，让 Bot 认出来并接话。

歌词库需要用户自己准备，插件本身不附带任何歌词（版权原因）。
"""

from __future__ import annotations

import inspect
import os
import re
import time

try:
    from astrbot.api import logger
except Exception:  # 兼容极端情况：拿不到 AstrBot 的 logger
    import logging

    logger = logging.getLogger("astrbot_plugin_lyric_match")

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

try:  # 可视化歌词管理页依赖 4.29 的 Web API；低版本只禁用页面，不影响聊天
    from astrbot.api.web import error_response, json_response, request

    _WEB_AVAILABLE = True
except Exception:  # pragma: no cover - 旧版本没有 astrbot.api.web
    _WEB_AVAILABLE = False

from .detector import is_lyric_like
from .lyrics_lib import LyricsLibrary

PLUGIN_NAME = "astrbot_plugin_lyric_match"

# 允许上传/管理的歌词文件类型
UPLOAD_EXTS = (".txt", ".json", ".lrc")
UPLOAD_MAX_BYTES = 5 * 1024 * 1024
UPLOAD_MAX_FILES = 20

DEFAULT_TEMPLATE = (
    "【歌词识别】主人刚才念的像是《{title}》的歌词：「{line}」。"
    "这首歌紧接着的下一句是：「{next}」。"
    "你要立刻认出来并惊喜地接话；可以顺势哼下一句，但必须用上面给出的这一句，不要自己编，也不要整段背。"
)

# 连续接唱模式：主人在连着唱时用这个，不要重复惊讶
DEFAULT_CONTINUE_TEMPLATE = (
    "【歌词接唱】主人正在连着唱歌，这一句像是《{title}》的：「{line}」。"
    "紧接着的下一句是：「{next}」。"
    "直接接着往下唱这一句就行，不要惊讶、不要重复「这是《{title}》」这类反应，也不要额外解释。"
)

DEFAULTS: dict = {
    "enabled": True,
    "similarity_threshold": 0.85,
    "min_length": 8,
    "private_only": False,
    "inject_template": DEFAULT_TEMPLATE,
    "continue_template": DEFAULT_CONTINUE_TEMPLATE,
    "sequence_window": 180,
    "extra_lyrics_dir": "",
    "candidate_k": 20,
    "rule_threshold": 0.5,
    "debug_log": False,
    "admin_only": True,
}


# ---------------- 配置值转换 ----------------


def as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "是", "开", "启用")
    return default


def as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


class _SafeDict(dict):
    """格式化模板时，未知占位符原样保留，避免 KeyError。"""

    def __missing__(self, key):
        return "{" + key + "}"


class Main(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.library = LyricsLibrary([])
        self.last_stats: dict = {}
        self.last_reload: float = 0.0
        # 连续接唱状态：session_id -> {"ts": float, "song": str}
        self._song_state: dict = {}
        self.reload_library(quiet=True)
        self._register_web_apis()

    # ---------------- 歌词库 ----------------

    def _lyric_dirs(self) -> list[str]:
        dirs: list[str] = [os.path.join(self.plugin_dir, "lyrics")]

        extra = str(self._conf("extra_lyrics_dir") or "").strip()
        if extra:
            for part in re.split(r"[;\n]+", extra):
                part = os.path.expanduser(part.strip())
                if not part:
                    continue
                if not os.path.isabs(part):
                    part = os.path.join(self.plugin_dir, part)
                dirs.append(os.path.abspath(part))

        # 去重，保持顺序
        seen: set[str] = set()
        unique: list[str] = []
        for directory in dirs:
            key = os.path.normcase(directory)
            if key in seen:
                continue
            seen.add(key)
            unique.append(directory)
        return unique

    def reload_library(self, quiet: bool = False) -> dict:
        dirs = self._lyric_dirs()
        library = LyricsLibrary(dirs)
        try:
            stats = library.load()
        except Exception as exc:
            logger.error(f"[{PLUGIN_NAME}] 歌词库加载失败：{exc}")
            stats = library.stats()
        self.library = library
        self.last_stats = stats
        self.last_reload = time.time()

        if stats.get("songs"):
            logger.info(
                f"[{PLUGIN_NAME}] 歌词库加载完成：{stats['songs']} 首 / "
                f"{stats['lines']} 句 / {stats['grams']} 个 2-gram"
            )
        else:
            logger.warning(
                f"[{PLUGIN_NAME}] 歌词库为空。请把 .txt/.json 歌词文件放进 "
                f"{os.path.join(self.plugin_dir, 'lyrics')} 后执行 /lyric reload"
            )
        for error in stats.get("errors", []):
            logger.warning(f"[{PLUGIN_NAME}] 歌词文件解析失败：{error}")
        if not quiet:
            logger.info(f"[{PLUGIN_NAME}] 歌词库已重载。")
        return stats

    # ---------------- 配置读取 ----------------

    def _conf(self, key: str):
        try:
            value = self.config.get(key, DEFAULTS.get(key))
        except Exception:
            value = DEFAULTS.get(key)
        return DEFAULTS.get(key) if value is None else value

    def _threshold(self) -> float:
        value = as_float(self._conf("similarity_threshold"), 0.85)
        return max(0.0, min(1.0, value))

    # ---------------- 事件判断 ----------------

    @staticmethod
    def _message_text(event: AstrMessageEvent) -> str:
        text = getattr(event, "message_str", None)
        if text is None:
            getter = getattr(event, "get_message_str", None)
            if callable(getter):
                try:
                    text = getter()
                except Exception:
                    text = ""
        return str(text or "")

    @staticmethod
    def _session_key(event: AstrMessageEvent) -> str:
        """会话标识：优先 unified_msg_origin，退化到发送者 id。"""
        key = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if key:
            return key
        for attr in ("get_sender_id", "get_session_id"):
            getter = getattr(event, attr, None)
            if callable(getter):
                try:
                    val = str(getter() or "").strip()
                    if val:
                        return val
                except Exception:
                    pass
        return "default"

    @staticmethod
    def _is_private(event: AstrMessageEvent) -> bool:
        origin = str(getattr(event, "unified_msg_origin", "") or "").lower()
        if "friend" in origin:
            return True
        if "group" in origin:
            return False
        getter = getattr(event, "get_message_type", None)
        if callable(getter):
            try:
                name = str(getattr(getter(), "name", "") or getter()).upper()
                return "FRIEND" in name or "PRIVATE" in name
            except Exception:
                pass
        return False

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        if not as_bool(self._conf("admin_only"), True):
            return True

        checker = getattr(event, "is_admin", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                pass

        try:
            config = self.context.get_config()
            admins = config.get("admins_id") or []
            if not admins:
                return True  # 没配管理员时放行，避免把自己锁在外面
            sender = str(event.get_sender_id())
            return sender in {str(admin) for admin in admins}
        except Exception:
            return True

    # ---------------- 钩子 ----------------

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        # 插件安装后 lyrics/ 里可能才刚放进文件，这里再扫一遍
        self.reload_library(quiet=True)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        try:
            injection = self.build_injection(event)
        except Exception as exc:
            logger.warning(f"[{PLUGIN_NAME}] 歌词识别异常：{exc}")
            return
        if not injection:
            return
        try:
            base = getattr(req, "system_prompt", "") or ""
            if injection in base:
                return
            req.system_prompt = (str(base).rstrip() + "\n\n" + injection).strip()
        except Exception as exc:
            logger.warning(f"[{PLUGIN_NAME}] 注入 system_prompt 失败：{exc}")

    async def terminate(self):
        logger.info(f"[{PLUGIN_NAME}] 插件已停止。")

    # ---------------- 识别主流程 ----------------

    def build_injection(self, event: AstrMessageEvent) -> str | None:
        """返回要注入的提示词；不该注入时返回 None。"""
        if not as_bool(self._conf("enabled"), True):
            return None

        text = self._message_text(event).strip()
        if not text or text.startswith("/"):
            return None

        if as_bool(self._conf("private_only"), False) and not self._is_private(event):
            return None

        if not self.library.songs:
            return None

        debug = as_bool(self._conf("debug_log"), False)
        min_length = max(1, as_int(self._conf("min_length"), 8))
        rule_threshold = as_float(self._conf("rule_threshold"), 0.5)

        passed, info = is_lyric_like(text, min_length=min_length,
                                     threshold=rule_threshold)
        if not passed:
            if debug:
                logger.info(
                    f"[{PLUGIN_NAME}] 规则预筛未通过：{info.get('reason') or 'low_score'} "
                    f"score={info.get('score', 0):.3f} text={text[:40]!r}"
                )
            return None

        match = self.library.best(text, top_k=as_int(self._conf("candidate_k"), 20))
        if match is None:
            if debug:
                logger.info(f"[{PLUGIN_NAME}] 规则通过但歌词库无候选：text={text[:40]!r}")
            return None

        threshold = self._threshold()
        if match.score + 1e-9 < threshold:
            if debug:
                logger.info(
                    f"[{PLUGIN_NAME}] 相似度不足：{match.score:.3f} < {threshold:.2f} "
                    f"（{match.song.display_title} / {match.line}）"
                )
            return None

        session_key = self._session_key(event)
        now_ts = time.time()
        window = max(10, as_int(self._conf("sequence_window"), 180))
        prev = self._song_state.get(session_key) or {}
        # 只要窗口内有过命中就算「还在连着唱」，不限同一首
        continuing = bool(prev) and (now_ts - float(prev.get("ts") or 0)) <= window

        if continuing:
            template = str(self._conf("continue_template") or DEFAULT_CONTINUE_TEMPLATE)
            mode = "continue"
        else:
            template = str(self._conf("inject_template") or DEFAULT_TEMPLATE)
            mode = "first"

        self._song_state[session_key] = {
            "ts": now_ts,
            "song": match.song.display_title,
        }
        values = {
            "title": match.song.display_title,
            "song": match.song.display_title,
            "line": match.line,
            "next": match.next_line,
            "score": f"{match.score:.2f}",
        }
        try:
            body = template.format_map(_SafeDict(values))
        except Exception:
            body = DEFAULT_TEMPLATE.format_map(_SafeDict(values))

        if not body.strip():
            return None

        logger.info(
            f"[{PLUGIN_NAME}] 命中歌词[{mode}]：《{match.song.display_title}》"
            f"「{match.line}」 score={match.score:.3f}"
        )
        return body.strip()

    # ---------------- 可视化歌词管理页面 ----------------

    def _lyrics_dir(self) -> str:
        """插件自带歌词目录（页面只允许操作这里）。"""
        directory = os.path.join(self.plugin_dir, "lyrics")
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            logger.warning(f"[{PLUGIN_NAME}] 无法创建歌词目录 {directory}：{exc}")
        return directory

    @staticmethod
    def _is_protected_file(name: str) -> bool:
        stem = os.path.splitext(name)[0].strip().lower()
        return name.startswith(".") or stem in ("readme", "license", "licence")

    @staticmethod
    def _inside_dir(directory: str, path: str) -> bool:
        try:
            base = os.path.realpath(directory)
            target = os.path.realpath(path)
            return os.path.commonpath([base, target]) == base
        except (OSError, ValueError):
            return False

    @staticmethod
    def _safe_upload_name(raw: str) -> str:
        """清洗上传文件名：只留 basename、只允许 txt/json/lrc。"""
        name = str(raw or "").replace("\\", "/").split("/")[-1].strip()
        name = re.sub(r'[\x00-\x1f<>:"|?*]', "_", name)
        ext = os.path.splitext(name)[1].lower()
        if ext not in UPLOAD_EXTS:
            raise ValueError("只支持 .txt / .json / .lrc 歌词文件")
        stem = os.path.splitext(name)[0].strip(" .")
        if not stem:
            raise ValueError("文件名不能为空")
        if stem.lower() in ("readme", "license", "licence"):
            raise ValueError("不能覆盖歌词库说明文件")
        return stem[:100] + ext

    def _stats_payload(self) -> dict:
        stats = self.library.stats()
        return {
            "songs": stats.get("songs", 0),
            "lines": stats.get("lines", 0),
            "grams": stats.get("grams", 0),
            "files": stats.get("files", 0),
            "dirs": stats.get("dirs", []),
            "lyrics_dir": self._lyrics_dir(),
            "loaded_at": stats.get("loaded_at", 0),
            "errors": stats.get("errors", []),
        }

    def _list_lyric_files(self) -> list[dict]:
        directory = self._lyrics_dir()
        try:
            names = sorted(os.listdir(directory))
        except OSError as exc:
            logger.warning(f"[{PLUGIN_NAME}] 读取歌词目录失败：{exc}")
            return []

        files: list[dict] = []
        for name in names:
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            if not name.lower().endswith(UPLOAD_EXTS):
                continue
            if self._is_protected_file(name):
                continue
            try:
                stat = os.stat(path)
            except OSError:
                continue
            summary = self.library.file_summary(path)
            files.append(
                {
                    "name": name,
                    "size": stat.st_size,
                    "songs": summary["songs"],
                    "lines": summary["lines"],
                    "error": summary["error"],
                    "mtime": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
                    ),
                }
            )
        return files

    def _register_web_apis(self) -> None:
        if not _WEB_AVAILABLE:
            logger.warning(
                f"[{PLUGIN_NAME}] 当前 AstrBot 没有 astrbot.api.web，"
                "歌词管理页面不可用（聊天功能不受影响）。"
            )
            return
        register = getattr(self.context, "register_web_api", None)
        if not callable(register):
            logger.warning(
                f"[{PLUGIN_NAME}] 当前 AstrBot 不支持 register_web_api，"
                "歌词管理页面不可用。"
            )
            return

        routes = (
            ("stats", "GET", self._api_stats, "歌词库统计"),
            ("files", "GET", self._api_files, "歌词文件列表"),
            ("upload", "POST", self._api_upload, "上传歌词文件"),
            ("delete", "POST", self._api_delete, "删除歌词文件"),
            ("reload", "POST", self._api_reload, "重新加载歌词库"),
        )
        try:
            for route, method, handler, desc in routes:
                register(f"/{PLUGIN_NAME}/{route}", handler, [method], desc)
            logger.info(f"[{PLUGIN_NAME}] 歌词管理页面 API 已注册。")
        except Exception as exc:
            logger.warning(f"[{PLUGIN_NAME}] 注册歌词管理 API 失败：{exc}")

    async def _api_stats(self):
        return json_response(self._stats_payload())

    async def _api_files(self):
        payload = self._stats_payload()
        payload["files"] = self._list_lyric_files()
        payload["max_bytes"] = UPLOAD_MAX_BYTES
        payload["extensions"] = list(UPLOAD_EXTS)
        return json_response(payload)

    async def _api_reload(self):
        stats = self.reload_library(quiet=False)
        payload = self._stats_payload()
        payload["reloaded"] = True
        payload["stats"] = stats
        return json_response(payload)

    async def _api_upload(self):
        try:
            files = await request.files()
        except Exception as exc:
            return error_response(f"读取上传内容失败：{exc}", status_code=400)

        uploads = []
        try:
            items = list(files.multi_items())
        except Exception:
            items = []
            for key in getattr(files, "keys", lambda: [])():
                items.append((key, files.get(key)))
        for _, value in items:
            if getattr(value, "filename", ""):
                uploads.append(value)

        if not uploads:
            return error_response("没有收到文件，请重新选择。", status_code=400)
        if len(uploads) > UPLOAD_MAX_FILES:
            return error_response(
                f"一次最多上传 {UPLOAD_MAX_FILES} 个文件。", status_code=400
            )

        directory = self._lyrics_dir()
        uploaded: list[dict] = []
        skipped: list[dict] = []

        for upload in uploads:
            raw_name = getattr(upload, "filename", "") or ""
            try:
                name = self._safe_upload_name(raw_name)
            except ValueError as exc:
                skipped.append({"name": raw_name, "reason": str(exc)})
                continue

            target = os.path.join(directory, name)
            if not self._inside_dir(directory, target):
                skipped.append({"name": name, "reason": "非法的文件路径"})
                continue

            saved = False
            try:
                await upload.save(target)
                saved = True
            except Exception:
                # 个别版本 save 的实现不同，退回到读取字节再写
                try:
                    data = upload.read()
                    if inspect.isawaitable(data):
                        data = await data
                    if not isinstance(data, (bytes, bytearray)):
                        raise TypeError("无法读取文件内容")
                    with open(target, "wb") as handle:
                        handle.write(bytes(data))
                    saved = True
                except Exception as exc:
                    skipped.append({"name": name, "reason": f"保存失败：{exc}"})
                    continue
            if not saved:
                skipped.append({"name": name, "reason": "保存失败"})
                continue

            try:
                size = os.path.getsize(target)
            except OSError:
                size = 0

            if size > UPLOAD_MAX_BYTES:
                try:
                    os.remove(target)
                except OSError:
                    pass
                skipped.append(
                    {"name": name, "reason": "超过 5 MB 限制，已丢弃"}
                )
                continue
            if size == 0:
                try:
                    os.remove(target)
                except OSError:
                    pass
                skipped.append({"name": name, "reason": "文件是空的，已丢弃"})
                continue

            uploaded.append({"name": name, "size": size})

        stats = self.reload_library(quiet=False)
        payload = self._stats_payload()
        payload.update(
            {
                "uploaded": uploaded,
                "skipped": skipped,
                "stats": stats,
            }
        )
        return json_response(payload)

    async def _api_delete(self):
        try:
            data = await request.json(default={})
        except Exception:
            data = {}
        name = str((data or {}).get("name", "")).strip()
        if not name:
            return error_response("缺少文件名。", status_code=400)

        # 只允许删除当前歌词目录里真实存在的文件，天然挡住 ../ 与绝对路径
        allowed = {item["name"] for item in self._list_lyric_files()}
        if name not in allowed:
            return error_response("文件不存在或不允许删除。", status_code=404)

        target = os.path.join(self._lyrics_dir(), name)
        if not self._inside_dir(self._lyrics_dir(), target):
            return error_response("非法的文件路径。", status_code=400)
        try:
            os.remove(target)
        except OSError as exc:
            return error_response(f"删除失败：{exc}", status_code=500)

        self.reload_library(quiet=False)
        payload = self._stats_payload()
        payload["deleted"] = name
        return json_response(payload)

    # ---------------- 指令 ----------------

    @staticmethod
    def _command_args(event: AstrMessageEvent) -> list[str]:
        text = Main._message_text(event).strip()
        parts = text.split()
        if parts and parts[0].startswith("/"):
            parts[0] = parts[0][1:]
        if parts and parts[0].lower() in ("lyric", "歌词"):
            parts = parts[1:]
        return [part for part in parts if part]

    @filter.command("lyric")
    async def lyric_command(self, event: AstrMessageEvent):
        args = self._command_args(event)
        sub = args[0].lower() if args else "help"

        if sub in ("reload", "重载", "重载歌词库", "刷新"):
            if not self._is_admin(event):
                yield event.plain_result("只有管理员才能重载歌词库哦。")
                return
            stats = self.reload_library(quiet=False)
            if not stats.get("songs"):
                yield event.plain_result(
                    "歌词库还是空的。\n"
                    "请把 .txt / .json / .lrc 歌词文件放进插件目录下的 lyrics/ 文件夹，"
                    "再执行 /lyric reload；也可以在插件页面「歌词管理」里直接上传。\n"
                    "格式见 lyrics/README.txt。"
                )
                return
            message = (
                f"歌词库已重载：{stats['songs']} 首 / {stats['lines']} 句 / "
                f"{stats['grams']} 个 2-gram"
            )
            if stats.get("skipped"):
                message += f"\n跳过重复或空歌曲：{stats['skipped']} 首"
            if stats.get("errors"):
                message += "\n解析失败：" + "；".join(stats["errors"][:3])
            yield event.plain_result(message)
            return

        if sub in ("list", "列表", "歌单"):
            if not self.library.songs:
                yield event.plain_result(
                    "歌词库是空的。把歌词文件放进 lyrics/ 后执行 /lyric reload。"
                )
                return
            lines = [f"歌词库共 {len(self.library.songs)} 首："]
            for index, song in enumerate(self.library.songs[:60], start=1):
                suffix = f"（{song.artists}）" if song.artists else ""
                lines.append(f"{index}. 《{song.display_title}》{suffix} · {len(song.lines)} 句")
            if len(self.library.songs) > 60:
                lines.append(f"……还有 {len(self.library.songs) - 60} 首未显示")
            yield event.plain_result("\n".join(lines))
            return

        if sub in ("stat", "stats", "统计"):
            stats = self.last_stats or self.library.stats()
            dirs = "\n".join(f"- {path}" for path in stats.get("dirs", []))
            message = (
                "📚 歌词库统计\n"
                f"歌曲：{stats.get('songs', 0)} 首\n"
                f"歌词句：{stats.get('lines', 0)} 句\n"
                f"2-gram 索引：{stats.get('grams', 0)} 个\n"
                f"文件：{stats.get('files', 0)} 个\n"
                f"目录：\n{dirs or '- （无）'}"
            )
            if stats.get("loaded_at"):
                loaded = time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(stats["loaded_at"]))
                message += f"\n最近加载：{loaded}"
            if stats.get("errors"):
                message += "\n解析失败：" + "；".join(stats["errors"][:3])
            yield event.plain_result(message)
            return

        yield event.plain_result(
            "歌词识别插件\n"
            "/lyric reload  重新加载 lyrics/ 下的歌词库（管理员）\n"
            "/lyric list    查看已经加载的歌\n"
            "/lyric stat    查看歌词库统计\n"
            "用法：直接把一句歌词单独发一条消息即可。\n"
            "批量管理歌词：WebUI「插件 → 歌词识别 → 歌词管理」页面。"
        )
