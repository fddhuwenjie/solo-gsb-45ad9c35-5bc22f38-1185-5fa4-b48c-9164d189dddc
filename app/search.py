"""文档版本范围内的字面检索：命中分页、可复现上下文与相交批注状态。

审阅者指定同一文档的版本范围 ``[from_version, to_version]`` 与一段文字，
服务按 **Unicode 码点**做字面子串匹配（**保留重叠命中**，复用
:func:`app.matching.find_occurrences` 的逐码点推进枚举），每个命中返回：

- 版本号与半开范围 ``[start, end)``（码点偏移）；
- **可复现的前后文片段**：``context.before`` / ``context.after`` 与
  ``before_start`` / ``after_end`` 满足
  ``content[before_start:start] == before``、
  ``content[end:after_end] == after``，调用方可按版本全文独立复算核对；
- 与该范围**相交**（半开区间有公共码点）的批注 ID 及**生效状态**
  （``resolved`` / ``confirmed`` / ``ambiguous`` / ``unresolved``）：
  人工确认过的位置以**确认范围**为准（确认范围优先于迁移候选），
  ``resolved`` 取迁移落点，未确认且非顺延的 ``ambiguous`` 取全部候选范围；
  ``unresolved`` 与歧义顺延占位在该版没有生效范围，不与任何命中相交。

**分页与快照**：结果按 ``(version_number, start, end)`` 全序稳定排序
（同一版本内各命中起点互不相同），每页至多 100 个命中。首轮在调用方持有
的同一个 SQLite 读事务中冻结**版本上界**（``to_version`` 缺省取当前最新
版）与**批注 / 位置快照**（annotations / locations / confirmations 三张
只追加表上属于本文档的最大行 ID），续页游标封装文档 ID、筛选条件
（检索文字、版本范围、上下文长度）、快照水位与最后排序键；分页期间新增
的版本 / 批注 / 迁移 / 确认（行 ID 必然大于冻结水位）**留到下一轮**
（即不带游标的新检索），当前轮各页结果不受并发写入影响。

游标为 Base64URL 紧凑 JSON，自带版本号；版本不支持、结构损坏、范围矛盾、
**跨文档使用**或**筛选条件与游标不一致**一律抛 :class:`CursorError`
（路由层转为 422）。接口只读，不写入任何数据。
"""
from __future__ import annotations

import base64
import binascii
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .matching import find_occurrences

CURSOR_VERSION = 1

# 快照水位覆盖的三张只追加表（同时是游标 wm 字段的固定键）：
# 批注、迁移位置、人工确认。版本上界由筛选条件 to_version 冻结，不在水位内。
SNAPSHOT_TABLES = ("annotations", "locations", "confirmations")

DEFAULT_CONTEXT = 20  # 前后文片段的默认长度（码点数）
MAX_CONTEXT = 200     # 前后文片段的最大长度（码点数）
MAX_LIMIT = 100       # 每页最多返回的命中数

# 排序键 = (版本号, 起点, 终点)，同一版本内起点唯一，构成全序
SortKey = Tuple[int, int, int]
Watermark = Dict[str, int]


class CursorError(Exception):
    """游标版本 / 结构 / 范围 / 筛选条件校验失败（路由层转为 HTTP 422）。

    - ``error``：机器可读错误码；
    - ``message``：中文说明。
    """

    def __init__(self, error: str, message: str):
        super().__init__(message)
        self.error = error
        self.message = message


@dataclass(frozen=True)
class EffectiveAnnotation:
    """一条批注在某版上的生效状态与生效范围（用于与命中求交）。

    ``ranges`` 为 ``(start, end)`` 半开区间元组：confirmed 为确认范围
    （优先于迁移候选），resolved 为迁移落点，ambiguous 为全部候选范围；
    unresolved 与歧义顺延占位为空元组（该版无生效范围，不与命中相交）。
    """

    annotation_id: int
    status: str  # resolved / confirmed / ambiguous / unresolved
    ranges: Tuple[Tuple[int, int], ...]


# ---------------------------------------------------------------------------
# 快照水位（单文档，只追加表的最大行 ID）
# ---------------------------------------------------------------------------

# 与 app.events.document_watermark 同一口径的单文档水位，只取检索快照
# 需要的三张表：locations / confirmations 经批注归属限定文档。
_SNAPSHOT_WATERMARK_SQL = {
    "annotations": "SELECT COALESCE(MAX(id), 0) AS m FROM annotations "
    "WHERE document_id = ?",
    "locations": """
        SELECT COALESCE(MAX(l.id), 0) AS m
        FROM locations l
        JOIN annotations a ON a.id = l.annotation_id
        WHERE a.document_id = ?
    """,
    "confirmations": """
        SELECT COALESCE(MAX(c.id), 0) AS m
        FROM confirmations c
        JOIN locations l ON l.id = c.location_id
        JOIN annotations a ON a.id = l.annotation_id
        WHERE a.document_id = ?
    """,
}


def snapshot_watermark(con: sqlite3.Connection, document_id: int) -> Watermark:
    """三张只追加表中**属于指定文档**的行的当前最大 ID（空取 0）。

    只追加 + 自增 ID 单调保证：冻结后新写入的批注 / 迁移 / 确认行 ID 必然
    大于水位，续页时按 ``id <= 水位`` 过滤即可复现首轮快照。调用方必须持有
    读事务，使三个 MAX(id) 与同事务内的其他读取落在同一快照上。
    """
    return {
        table: con.execute(sql, (document_id,)).fetchone()["m"]
        for table, sql in _SNAPSHOT_WATERMARK_SQL.items()
    }


def max_version_number(con: sqlite3.Connection, document_id: int) -> int:
    """文档当前最大版本号（文档创建即有第 1 版，故必然 >= 1）。"""
    return con.execute(
        "SELECT COALESCE(MAX(number), 0) AS m FROM versions WHERE document_id = ?",
        (document_id,),
    ).fetchone()["m"]


# ---------------------------------------------------------------------------
# 数据装载（调用方持有读事务；行级过滤复现冻结快照）
# ---------------------------------------------------------------------------


def load_version_texts(
    con: sqlite3.Connection,
    document_id: int,
    from_version: int,
    to_version: int,
) -> List[Tuple[int, str]]:
    """版本范围内各版全文，按版本号升序（版本内容只追加、不可变）。"""
    rows = con.execute(
        """
        SELECT number, content FROM versions
        WHERE document_id = ? AND number BETWEEN ? AND ?
        ORDER BY number
        """,
        (document_id, from_version, to_version),
    ).fetchall()
    return [(row["number"], row["content"]) for row in rows]


def load_effective_annotations(
    con: sqlite3.Connection,
    document_id: int,
    from_version: int,
    to_version: int,
    wm: Watermark,
) -> Dict[int, List[EffectiveAnnotation]]:
    """版本范围内每版各批注的生效状态与生效范围（按快照水位过滤）。

    只读取行 ID 不超过 ``wm`` 的批注 / 迁移 / 确认行：首轮冻结水位后，
    分页期间并发新增的行不参与任何一页的计算（留到下一轮检索）。
    返回 ``{version_number: [EffectiveAnnotation, ...]}``，每版列表按
    批注 ID 升序。
    """
    loc_rows = con.execute(
        """
        SELECT l.id AS location_id, l.annotation_id, l.status,
               l.start, l."end", l.candidates, l.carried_from,
               v.number AS version_number
        FROM locations l
        JOIN versions v ON v.id = l.version_id
        JOIN annotations a ON a.id = l.annotation_id
        WHERE a.document_id = ? AND a.id <= ? AND l.id <= ?
          AND v.number BETWEEN ? AND ?
        ORDER BY v.number, l.annotation_id
        """,
        (document_id, wm["annotations"], wm["locations"],
         from_version, to_version),
    ).fetchall()

    conf_by_location: Dict[int, sqlite3.Row] = {}
    if loc_rows:
        conf_rows = con.execute(
            """
            SELECT c.location_id, c.start, c."end"
            FROM confirmations c
            JOIN locations l ON l.id = c.location_id
            JOIN annotations a ON a.id = l.annotation_id
            WHERE a.document_id = ? AND a.id <= ? AND c.id <= ?
            """,
            (document_id, wm["annotations"], wm["confirmations"]),
        ).fetchall()
        conf_by_location = {row["location_id"]: row for row in conf_rows}

    by_version: Dict[int, List[EffectiveAnnotation]] = {}
    for row in loc_rows:
        conf = conf_by_location.get(row["location_id"])
        status, ranges = effective_ranges(
            row["status"],
            row["start"],
            row["end"],
            json.loads(row["candidates"]),
            row["carried_from"],
            conf,
        )
        by_version.setdefault(row["version_number"], []).append(
            EffectiveAnnotation(row["annotation_id"], status, ranges)
        )
    return by_version


def effective_ranges(
    status: str,
    start: Optional[int],
    end: Optional[int],
    candidates: Sequence[Dict[str, int]],
    carried_from: Optional[int],
    confirmation: Optional[sqlite3.Row],
) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
    """计算一条迁移记录的生效状态与生效范围（纯函数）。

    **确认范围优先于迁移候选**：有人工确认时生效状态为 confirmed，
    范围取确认所选；否则 resolved 取迁移落点，未确认且非顺延的
    ambiguous 取全部候选范围；unresolved 与歧义顺延占位没有生效范围。
    """
    if confirmation is not None:
        return "confirmed", ((confirmation["start"], confirmation["end"]),)
    if status == "resolved":
        return "resolved", ((start, end),)
    if status == "ambiguous" and carried_from is None:
        return "ambiguous", tuple((c["start"], c["end"]) for c in candidates)
    # unresolved 没有落点；歧义顺延占位未参与本版迁移，均不提供范围
    return status, ()


# ---------------------------------------------------------------------------
# 命中装配（纯函数）
# ---------------------------------------------------------------------------


def _intersecting(
    annotations: Sequence[EffectiveAnnotation], start: int, end: int
) -> List[Dict[str, Any]]:
    """与命中范围 ``[start, end)`` 相交（半开区间有公共码点）的批注。

    每条相交批注给出 ID、生效状态与实际相交的生效范围（confirmed 为
    确认范围，ambiguous 为相交的候选范围），按批注 ID 升序。
    """
    out: List[Dict[str, Any]] = []
    for ann in annotations:
        ranges = [r for r in ann.ranges if r[0] < end and r[1] > start]
        if ranges:
            out.append(
                {
                    "annotation_id": ann.annotation_id,
                    "status": ann.status,
                    "ranges": [[s, e] for s, e in ranges],
                }
            )
    return out


def build_hits(
    version_texts: Sequence[Tuple[int, str]],
    effective_by_version: Dict[int, List[EffectiveAnnotation]],
    query: str,
    context: int,
) -> List[Dict[str, Any]]:
    """对版本范围内的每版全文做字面子串匹配，装配命中列表（纯函数）。

    重叠命中全部保留（如 ``aaaa`` 中检索 ``aa`` 得到 3 个命中）；结果按
    ``(version_number, start, end)`` 全序升序——版本升序处理、
    :func:`find_occurrences` 起点升序枚举，天然有序。每个命中的前后文
    片段取命中两侧至多 ``context`` 个码点，``before_start`` /
    ``after_end`` 使调用方可按版本全文复现核对。
    """
    hits: List[Dict[str, Any]] = []
    for number, content in version_texts:
        annotations = effective_by_version.get(number, [])
        for start in find_occurrences(content, query):
            end = start + len(query)
            before_start = max(0, start - context)
            after_end = min(len(content), end + context)
            hits.append(
                {
                    "version_number": number,
                    "start": start,
                    "end": end,
                    "context": {
                        "before": content[before_start:start],
                        "after": content[end:after_end],
                        "before_start": before_start,
                        "after_end": after_end,
                    },
                    "annotations": _intersecting(annotations, start, end),
                }
            )
    return hits


def sort_key(hit: Dict[str, Any]) -> SortKey:
    """命中的全序排序键（版本号, 起点, 终点）。"""
    return (hit["version_number"], hit["start"], hit["end"])


# ---------------------------------------------------------------------------
# 游标编解码
# ---------------------------------------------------------------------------


def encode_cursor(payload: Dict[str, Any]) -> str:
    """紧凑 JSON → URL 安全 Base64（去掉填充，query 参数中可直接传递）。"""
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def filters_dict(
    query: str, from_version: int, to_version: int, context: int
) -> Dict[str, Any]:
    """游标封装的筛选条件（检索文字、版本范围、上下文长度）。

    续页时请求携带的筛选条件必须与游标冻结值一致（``to_version`` 缺省
    视为沿用游标冻结的版本上界），否则按 ``filter_mismatch`` 拒绝。
    """
    return {"q": query, "from": from_version, "to": to_version, "ctx": context}


def make_page_cursor(
    document_id: int,
    filters: Dict[str, Any],
    wm: Watermark,
    last_key: SortKey,
) -> str:
    """续页游标：冻结筛选条件、版本上界与批注 / 位置快照，推进最后排序键。"""
    return encode_cursor(
        {
            "v": CURSOR_VERSION,
            "kind": "page",
            "doc": document_id,
            "f": filters,
            "wm": {table: wm[table] for table in SNAPSHOT_TABLES},
            "last": [last_key[0], last_key[1], last_key[2]],
        }
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_filters(value: Any) -> Dict[str, Any]:
    """校验游标中的筛选条件：结构完整、取值域合法、版本范围不矛盾。"""
    if not isinstance(value, dict) or set(value.keys()) != {"q", "from", "to", "ctx"}:
        raise CursorError(
            "bad_cursor_structure",
            "游标 f 筛选条件应是恰好包含 q / from / to / ctx 四个键的对象",
        )
    q, frm, to, ctx = value["q"], value["from"], value["to"], value["ctx"]
    if not isinstance(q, str) or not q:
        raise CursorError(
            "bad_cursor_structure", "游标 f.q 检索文字应为非空字符串"
        )
    for name, v in (("from", frm), ("to", to), ("ctx", ctx)):
        if not _is_int(v):
            raise CursorError(
                "bad_cursor_structure", f"游标 f.{name} 应为整数"
            )
    if frm < 1 or to < 1:
        raise CursorError(
            "bad_cursor_range",
            f"游标版本范围端点应不小于 1（from={frm}, to={to}）",
        )
    if frm > to:
        raise CursorError(
            "bad_cursor_range",
            f"游标版本范围矛盾：from={frm} 大于 to={to}",
        )
    if ctx < 0:
        raise CursorError(
            "bad_cursor_range", f"游标上下文长度不能为负数（实际为 {ctx}）"
        )
    return {"q": q, "from": frm, "to": to, "ctx": ctx}


def _check_watermark(value: Any, ceiling: Watermark) -> Watermark:
    """校验游标中的快照水位：结构完整、非负、不超过当前数据库。"""
    if not isinstance(value, dict) or set(value.keys()) != set(SNAPSHOT_TABLES):
        raise CursorError(
            "bad_cursor_structure",
            f"游标 wm 水位应是恰好包含 {list(SNAPSHOT_TABLES)} 三个键的对象",
        )
    wm: Watermark = {}
    for table in SNAPSHOT_TABLES:
        v = value[table]
        if not _is_int(v):
            raise CursorError(
                "bad_cursor_structure", f"游标 wm.{table} 水位应为非负整数"
            )
        if v < 0:
            raise CursorError(
                "bad_cursor_range",
                f"游标 wm.{table} 水位不能为负数（实际为 {v}）",
            )
        if v > ceiling[table]:
            raise CursorError(
                "bad_cursor_range",
                f"游标 {table} 水位 {v} 超过当前最大行 ID {ceiling[table]}"
                "（游标可能来自另一个数据库）",
            )
        wm[table] = v
    return wm


def _check_last_key(value: Any, filters: Dict[str, Any]) -> SortKey:
    """校验最后排序键 ``[version_number, start, end]`` 且落在版本范围内。"""
    if not (
        isinstance(value, list)
        and len(value) == 3
        and all(_is_int(v) for v in value)
    ):
        raise CursorError(
            "bad_cursor_structure",
            "游标 last 应为 [version_number, start, end] 三个整数",
        )
    version_number, start, end = value
    if version_number < 1 or start < 0 or end < 1 or start >= end:
        raise CursorError(
            "bad_cursor_range",
            f"游标最后排序键范围无效：[{version_number}, {start}, {end})"
            "（要求版本号 >= 1 且 0 <= start < end）",
        )
    if not (filters["from"] <= version_number <= filters["to"]):
        raise CursorError(
            "bad_cursor_range",
            f"游标最后排序键的版本号 {version_number} 不在冻结版本范围 "
            f"[{filters['from']}, {filters['to']}] 内",
        )
    return (version_number, start, end)


def decode_cursor(
    token: str, document_id: int, ceiling: Watermark
) -> Dict[str, Any]:
    """解码并完整校验续页游标。

    任何版本不支持 / 解码失败 / 结构错误 / 范围矛盾 / 跨文档使用都抛
    :class:`CursorError`。返回 ``{"filters", "wm", "last"}``；筛选条件与
    请求参数的一致性由路由层另行比对（``filter_mismatch``）。
    """
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.b64decode(padded, altchars=b"-_", validate=True)
        data = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise CursorError(
            "bad_cursor_structure",
            "游标无法解码：应为 Base64URL 编码的紧凑 JSON",
        )

    if not isinstance(data, dict):
        raise CursorError("bad_cursor_structure", "游标应为 JSON 对象")
    if "v" not in data:
        raise CursorError("bad_cursor_structure", "游标缺少版本字段 v")
    version = data["v"]
    if not _is_int(version):
        raise CursorError("bad_cursor_structure", "游标版本 v 应为整数")
    if version != CURSOR_VERSION:
        raise CursorError(
            "bad_cursor_version",
            f"不支持的游标版本 {version}（当前支持版本 {CURSOR_VERSION}）",
        )

    if data.get("kind") != "page":
        raise CursorError(
            "bad_cursor_structure", "游标 kind 应为 'page'（续页）"
        )
    if set(data.keys()) != {"v", "kind", "doc", "f", "wm", "last"}:
        raise CursorError(
            "bad_cursor_structure",
            "page 游标字段应恰好为 ['doc', 'f', 'kind', 'last', 'v', 'wm']",
        )

    doc = data["doc"]
    if not _is_int(doc) or doc < 1:
        raise CursorError("bad_cursor_structure", "游标 doc 应为正整数")
    if doc != document_id:
        raise CursorError(
            "cross_document_cursor",
            f"游标属于文档 {doc}，不能用于文档 {document_id} 的检索"
            "（游标不可跨文档使用）",
        )

    filters = _check_filters(data["f"])
    wm = _check_watermark(data["wm"], ceiling)
    last_key = _check_last_key(data["last"], filters)
    return {"filters": filters, "wm": wm, "last": last_key}
