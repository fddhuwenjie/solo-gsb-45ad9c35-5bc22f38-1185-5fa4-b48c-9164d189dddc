"""版本范围内字面检索测试：字面匹配 / 重叠命中 / emoji 码点偏移 /
可复现上下文 / 相交批注生效状态 / 稳定排序分页 / 快照冻结 / 游标校验。"""
import base64
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_doc(client, content, title="检索测试文档"):
    r = client.post("/documents", json={"title": title, "content": content})
    assert r.status_code == 201, r.text
    return r.json()["document_id"]


def _make_ann(client, doc, text, selection, left="", right="", body="批注"):
    start = text.index(selection)
    r = client.post(
        f"/documents/{doc}/annotations",
        json={
            "body": body,
            "selection": selection,
            "left_context": left,
            "right_context": right,
            "start": start,
            "end": start + len(selection),
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["annotation_id"]


def _new_version(client, doc, content):
    r = client.post(f"/documents/{doc}/versions", json={"content": content})
    assert r.status_code == 201, r.text
    return r.json()["version"]["version_number"]


def _search(client, doc, **params):
    r = client.get(f"/documents/{doc}/search", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _fetch_all(client, doc, **params):
    """沿 next_cursor 翻完整轮，返回全部命中。"""
    page = _search(client, doc, **params)
    hits = list(page["hits"])
    while page["next_cursor"] is not None:
        assert page["has_more"] is True
        page = _search(client, doc, **params, cursor=page["next_cursor"])
        hits.extend(page["hits"])
    assert page["has_more"] is False
    return hits


def _encode(payload):
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _keys(hits):
    return [(h["version_number"], h["start"], h["end"]) for h in hits]


# ---------------------------------------------------------------------------
# 字面子串匹配与重叠命中
# ---------------------------------------------------------------------------


def test_literal_match_and_overlapping_hits(client):
    doc = _make_doc(client, "aaaa")
    resp = _search(client, doc, query="aa")
    # 重叠命中全部保留：[0,2) [1,3) [2,4)
    assert _keys(resp["hits"]) == [(1, 0, 2), (1, 1, 3), (1, 2, 4)]
    assert resp["hit_count"] == 3
    assert resp["has_more"] is False
    assert resp["next_cursor"] is None
    assert resp["to_version"] == 1  # 缺省冻结为当前最新版

    doc2 = _make_doc(client, "ababa")
    resp2 = _search(client, doc2, query="aba")
    assert _keys(resp2["hits"]) == [(1, 0, 3), (1, 2, 5)]

    # 检索词比文本长 / 不存在 → 无命中
    assert _search(client, doc, query="aaaaa")["hit_count"] == 0
    assert _search(client, doc, query="b")["hit_count"] == 0


def test_emoji_offsets_are_codepoints(client):
    # 😀 是 BMP 外字符，按 1 个码点计
    content = "a😀b😀c"
    doc = _make_doc(client, content)
    ann = _make_ann(client, doc, content, "😀b", body="emoji 批注")  # [1,3)

    resp = _search(client, doc, query="😀")
    assert _keys(resp["hits"]) == [(1, 1, 2), (1, 3, 4)]
    # 第一个命中与批注 [1,3) 相交；第二个不相交（3 > 3 不成立）
    assert resp["hits"][0]["annotations"] == [
        {"annotation_id": ann, "status": "resolved", "ranges": [[1, 3]]}
    ]
    assert resp["hits"][1]["annotations"] == []
    # 前后文按码点截取且可复现
    first = resp["hits"][0]
    assert first["context"] == {
        "before": "a",
        "after": "b😀c",
        "before_start": 0,
        "after_end": 5,
    }
    assert content[first["context"]["before_start"] : first["start"]] == "a"
    assert content[first["end"] : first["context"]["after_end"]] == "b😀c"


def test_context_snippet_reproducible_and_clipped(client):
    content = "开头甲乙丙丁中间戊己庚辛结尾"
    doc = _make_doc(client, content)
    start = content.index("中间")

    resp = _search(client, doc, query="中间", context=4)
    hit = resp["hits"][0]
    assert hit["context"]["before"] == "甲乙丙丁"
    assert hit["context"]["after"] == "戊己庚辛"
    assert hit["context"]["before_start"] == start - 4
    assert hit["context"]["after_end"] == start + 2 + 4
    # 可复现：按版本全文与偏移能逐字节还原前后文
    assert content[hit["context"]["before_start"] : hit["start"]] == "甲乙丙丁"
    assert content[hit["end"] : hit["context"]["after_end"]] == "戊己庚辛"

    # context=0 → 无前后文；文本边界处截断不越界
    resp0 = _search(client, doc, query="中间", context=0)
    assert resp0["hits"][0]["context"] == {
        "before": "",
        "after": "",
        "before_start": start,
        "after_end": start + 2,
    }
    edge = _search(client, doc, query="开头", context=10)["hits"][0]
    assert edge["context"]["before"] == ""
    assert edge["context"]["before_start"] == 0
    tail = _search(client, doc, query="结尾", context=10)["hits"][0]
    assert tail["context"]["after"] == ""
    assert tail["context"]["after_end"] == len(content)


# ---------------------------------------------------------------------------
# 版本范围
# ---------------------------------------------------------------------------


def test_version_range_filter(client):
    doc = _make_doc(client, "目标。")
    _new_version(client, doc, "目标。目标。")
    _new_version(client, doc, "目标。目标。目标。")

    # 缺省：第 1 版到最新版
    resp = _search(client, doc, query="目标")
    assert resp["from_version"] == 1
    assert resp["to_version"] == 3
    assert _keys(resp["hits"]) == [
        (1, 0, 2),
        (2, 0, 2), (2, 3, 5),
        (3, 0, 2), (3, 3, 5), (3, 6, 8),
    ]

    # 显式子范围
    sub = _search(client, doc, query="目标", from_version=2, to_version=2)
    assert _keys(sub["hits"]) == [(2, 0, 2), (2, 3, 5)]
    sub2 = _search(client, doc, query="目标", from_version=2)
    assert _keys(sub2["hits"]) == [
        (2, 0, 2), (2, 3, 5),
        (3, 0, 2), (3, 3, 5), (3, 6, 8),
    ]


def test_version_range_validation(client):
    doc = _make_doc(client, "目标。")
    _new_version(client, doc, "目标。目标。")

    r = client.get(f"/documents/{doc}/search", params={"query": "目标", "from_version": 3})
    assert r.status_code == 404  # 起始版不存在
    r = client.get(f"/documents/{doc}/search", params={"query": "目标", "to_version": 3})
    assert r.status_code == 404  # 结束版不存在
    r = client.get(
        f"/documents/{doc}/search",
        params={"query": "目标", "from_version": 2, "to_version": 1},
    )
    assert r.status_code == 422  # from > to
    r = client.get("/documents/999999/search", params={"query": "目标"})
    assert r.status_code == 404  # 文档不存在
    # 结构校验：空检索词 / 上下文与分页越界
    assert client.get(f"/documents/{doc}/search", params={"query": ""}).status_code == 422
    assert client.get(
        f"/documents/{doc}/search", params={"query": "目标", "context": -1}
    ).status_code == 422
    assert client.get(
        f"/documents/{doc}/search", params={"query": "目标", "context": 201}
    ).status_code == 422
    assert client.get(
        f"/documents/{doc}/search", params={"query": "目标", "limit": 0}
    ).status_code == 422
    assert client.get(
        f"/documents/{doc}/search", params={"query": "目标", "limit": 101}
    ).status_code == 422


# ---------------------------------------------------------------------------
# 相交批注的生效状态（确认范围优先于迁移候选）
# ---------------------------------------------------------------------------


def test_intersecting_annotation_statuses(client):
    v1 = "第一段：天气很好。第二段：心情不错。第三段：天气很好。"
    doc = _make_doc(client, v1)
    ann1 = _make_ann(client, doc, v1, "心情不错", left="第二段：", right="。")
    ann2 = _make_ann(client, doc, v1, "天气很好", left="第一段：", right="。")

    # 第 2 版：ann1 两处并列最高分 → ambiguous；ann2 选区消失 → unresolved
    _new_version(client, doc, "第二段：心情不错。第二段：心情不错。")

    resp = _search(client, doc, query="心情不错", from_version=2, to_version=2)
    assert _keys(resp["hits"]) == [(2, 4, 8), (2, 13, 17)]
    # 未确认的歧义批注：按相交的候选范围登记
    assert resp["hits"][0]["annotations"] == [
        {"annotation_id": ann1, "status": "ambiguous", "ranges": [[4, 8]]}
    ]
    assert resp["hits"][1]["annotations"] == [
        {"annotation_id": ann1, "status": "ambiguous", "ranges": [[13, 17]]}
    ]
    # unresolved 批注没有生效范围，不出现在任何命中上
    assert all(
        ann2 not in [a["annotation_id"] for a in h["annotations"]]
        for h in resp["hits"]
    )

    # 人工确认候选 [13,17) → 生效状态 confirmed，确认范围优先于迁移候选
    r = client.post(
        f"/documents/{doc}/annotations/{ann1}/confirmations",
        json={"version_number": 2, "start": 13, "end": 17},
    )
    assert r.status_code == 201, r.text
    resp = _search(client, doc, query="心情不错", from_version=2, to_version=2)
    # 命中 [4,8)：候选 [4,8) 不再参与（确认范围优先），批注不再相交
    assert resp["hits"][0]["annotations"] == []
    # 命中 [13,17)：以确认范围相交
    assert resp["hits"][1]["annotations"] == [
        {"annotation_id": ann1, "status": "confirmed", "ranges": [[13, 17]]}
    ]

    # 第 3 版：ann1 恢复自动迁移 → resolved；ann2 仍 unresolved
    _new_version(client, doc, "第二段：心情不错。")
    resp = _search(client, doc, query="心情不错", from_version=3, to_version=3)
    assert resp["hits"][0]["annotations"] == [
        {"annotation_id": ann1, "status": "resolved", "ranges": [[4, 8]]}
    ]
    # 首尾相接不算相交：批注范围 [4,8)，命中 [8,9) 的「。」与 [3,4) 的「：」
    # 分别与批注范围首尾相接，均不相交
    resp = _search(client, doc, query="。", from_version=3, to_version=3)
    assert _keys(resp["hits"]) == [(3, 8, 9)]
    assert resp["hits"][0]["annotations"] == []
    resp = _search(client, doc, query="：", from_version=3, to_version=3)
    assert _keys(resp["hits"]) == [(3, 3, 4)]
    assert resp["hits"][0]["annotations"] == []


def test_carried_ambiguous_has_no_range(client):
    v1 = "第一段。第二段。"
    doc = _make_doc(client, v1)
    ann = _make_ann(client, doc, v1, "第一段", right="。")
    _new_version(client, doc, "第一段。第一段。")  # v2：歧义（两处得分相同）
    _new_version(client, doc, "第一段。第一段。第一段。")  # v3：顺延占位

    resp = _search(client, doc, query="第一段")
    assert _keys(resp["hits"]) == [
        (1, 0, 3),
        (2, 0, 3), (2, 4, 7),
        (3, 0, 3), (3, 4, 7), (3, 8, 11),
    ]
    by_key = {(h["version_number"], h["start"]): h for h in resp["hits"]}
    # v1：resolved 落点 [0,3)，与唯一的命中相交
    assert by_key[(1, 0)]["annotations"] == [
        {"annotation_id": ann, "status": "resolved", "ranges": [[0, 3]]}
    ]
    # v2：未确认歧义，两个候选各自与对应命中相交
    assert by_key[(2, 0)]["annotations"] == [
        {"annotation_id": ann, "status": "ambiguous", "ranges": [[0, 3]]}
    ]
    assert by_key[(2, 4)]["annotations"] == [
        {"annotation_id": ann, "status": "ambiguous", "ranges": [[4, 7]]}
    ]
    # v3：歧义顺延占位没有生效范围，三个命中都不带批注
    assert by_key[(3, 0)]["annotations"] == []
    assert by_key[(3, 4)]["annotations"] == []
    assert by_key[(3, 8)]["annotations"] == []


# ---------------------------------------------------------------------------
# 稳定排序与分页
# ---------------------------------------------------------------------------


def test_pagination_stable_order(client):
    doc = _make_doc(client, "a" * 7)  # "aa" → 6 个重叠命中
    _new_version(client, doc, "a" * 5)  # → 4 个

    pages, cursor = [], None
    while True:
        params = {"query": "aa", "limit": 4}
        if cursor is not None:
            params["cursor"] = cursor
        page = _search(client, doc, **params)
        assert page["hit_count"] <= 4
        pages.append(page)
        cursor = page["next_cursor"]
        if cursor is None:
            break
    hits = [h for p in pages for h in p["hits"]]
    assert _keys(hits) == [
        (1, 0, 2), (1, 1, 3), (1, 2, 4), (1, 3, 5),
        (1, 4, 6), (1, 5, 7),
        (2, 0, 2), (2, 1, 3), (2, 2, 4), (2, 3, 5),
    ]
    # 全序稳定：严格升序、不重不漏
    assert _keys(hits) == sorted(set(_keys(hits)))
    assert pages[-1]["has_more"] is False
    # 逐页结果与一次性取全量（limit=100 上限内）一致
    assert hits == _fetch_all(client, doc, query="aa", limit=100)


def test_default_limit_is_50_and_max_100(client):
    doc = _make_doc(client, "a" * 61)  # "a" → 61 个命中
    resp = _search(client, doc, query="a")
    assert resp["limit"] == 50
    assert resp["hit_count"] == 50
    assert resp["has_more"] is True
    resp = _search(client, doc, query="a", limit=100)
    assert resp["hit_count"] == 61  # 单页上限 100，61 个一次取完
    assert resp["has_more"] is False


# ---------------------------------------------------------------------------
# 快照冻结：分页期间的并发写入留到下一轮
# ---------------------------------------------------------------------------


def test_snapshot_freezes_during_pagination(client):
    v1 = "目标。目标。"
    doc = _make_doc(client, v1)
    ann1 = _make_ann(client, doc, v1, "目标", right="。")
    _new_version(client, doc, "目标。目标。")  # v2：ann1 歧义未确认

    # 首轮第一页（冻结版本上界=2 与批注/位置快照）
    page1 = _search(client, doc, query="目标", limit=1)
    assert _keys(page1["hits"]) == [(1, 0, 2)]
    assert page1["to_version"] == 2
    frozen_snapshot = page1["snapshot"]
    cursor = page1["next_cursor"]
    assert cursor is not None

    # 并发写入：确认 v2 歧义、提交 v3、在 v3 上新建批注
    r = client.post(
        f"/documents/{doc}/annotations/{ann1}/confirmations",
        json={"version_number": 2, "start": 3, "end": 5},
    )
    assert r.status_code == 201, r.text
    _new_version(client, doc, "目标。目标。目标。")
    ann2 = _make_ann(client, doc, "目标。目标。目标。", "目标", right="。", body="新批注")

    # 续页：仍按冻结快照计算——不含 v3、不含新批注、确认不生效
    rest = []
    while cursor is not None:
        page = _search(client, doc, query="目标", limit=1, cursor=cursor)
        assert page["to_version"] == 2
        assert page["snapshot"] == frozen_snapshot
        rest.extend(page["hits"])
        cursor = page["next_cursor"]
    assert _keys(rest) == [(1, 3, 5), (2, 0, 2), (2, 3, 5)]
    # v2 命中上的 ann1 仍是 ambiguous（确认在冻结水位之后）
    assert rest[1]["annotations"] == [
        {"annotation_id": ann1, "status": "ambiguous", "ranges": [[0, 2]]}
    ]
    assert rest[2]["annotations"] == [
        {"annotation_id": ann1, "status": "ambiguous", "ranges": [[3, 5]]}
    ]
    assert all(
        ann2 not in [a["annotation_id"] for a in h["annotations"]] for h in rest
    )

    # 下一轮（不带游标的新检索）：看到新版本、新批注与确认后的生效状态
    fresh = _search(client, doc, query="目标", limit=100)
    assert fresh["to_version"] == 3
    assert _keys(fresh["hits"]) == [
        (1, 0, 2), (1, 3, 5),
        (2, 0, 2), (2, 3, 5),
        (3, 0, 2), (3, 3, 5), (3, 6, 8),
    ]
    by_key = {(h["version_number"], h["start"]): h for h in fresh["hits"]}
    assert by_key[(2, 3)]["annotations"] == [
        {"annotation_id": ann1, "status": "confirmed", "ranges": [[3, 5]]}
    ]
    assert by_key[(2, 0)]["annotations"] == []  # 确认范围优先，候选 [0,2) 不再参与
    v3_ann_ids = [
        a["annotation_id"] for h in fresh["hits"] if h["version_number"] == 3
        for a in h["annotations"]
    ]
    assert ann2 in v3_ann_ids


# ---------------------------------------------------------------------------
# 游标校验：篡改 / 跨文档 / 筛选条件不一致 → 422
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def paged_doc(client):
    """两版共 4 个命中的文档，首轮 limit=1 取得续页游标。"""
    doc = _make_doc(client, "目标。目标。")
    _new_version(client, doc, "目标。目标。")
    page = _search(client, doc, query="目标", limit=1)
    assert page["next_cursor"] is not None
    return doc, page["next_cursor"]


def test_cursor_tampered_or_malformed(client, paged_doc):
    doc, cursor = paged_doc
    cases = [
        "not-valid-base64!!!",
        _encode("not a dict"),
        _encode({"kind": "page"}),  # 缺版本字段
        _encode({"v": 2, "kind": "page", "doc": doc}),  # 版本不支持
        _encode({"v": 1, "kind": "resume", "doc": doc}),  # kind 非法
        _encode({"v": 1, "kind": "page", "doc": doc}),  # 字段不全
    ]
    for bad in cases:
        r = client.get(f"/documents/{doc}/search", params={"query": "目标", "cursor": bad})
        assert r.status_code == 422, (bad, r.text)
        assert r.json()["detail"]["error"] in (
            "bad_cursor_structure",
            "bad_cursor_version",
        )
    # 原游标依然可用（校验失败不消耗游标）
    assert _search(client, doc, query="目标", limit=1, cursor=cursor)["hit_count"] == 1


def test_cursor_range_errors(client, paged_doc):
    doc, cursor = paged_doc
    base = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))

    def expect_range_error(mutate):
        bad = json.loads(json.dumps(base))
        mutate(bad)
        r = client.get(
            f"/documents/{doc}/search", params={"query": "目标", "cursor": _encode(bad)}
        )
        assert r.status_code == 422, r.text
        assert r.json()["detail"]["error"] == "bad_cursor_range"

    expect_range_error(lambda c: c["wm"].update(annotations=-1))  # 负水位
    expect_range_error(lambda c: c["wm"].update(locations=10 ** 9))  # 超过当前库
    expect_range_error(lambda c: c["f"].update(**{"from": 2, "to": 1}))  # from > to
    expect_range_error(lambda c: c.update(last=[9, 0, 2]))  # 最后排序键越出版本范围
    expect_range_error(lambda c: c.update(last=[1, 2, 2]))  # start >= end


def test_cursor_cross_document(client, paged_doc):
    doc, cursor = paged_doc
    other = _make_doc(client, "目标。目标。")
    r = client.get(f"/documents/{other}/search", params={"query": "目标", "cursor": cursor})
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "cross_document_cursor"


def test_cursor_filter_mismatch(client, paged_doc):
    doc, cursor = paged_doc
    # 检索文字 / 上下文 / 起始版本 / 显式结束版本 任一不一致 → 422
    for params in (
        {"query": "目标。"},
        {"query": "目标", "context": 5},
        {"query": "目标", "from_version": 2},
        {"query": "目标", "to_version": 1},
    ):
        r = client.get(
            f"/documents/{doc}/search", params={**params, "cursor": cursor}
        )
        assert r.status_code == 422, (params, r.text)
        detail = r.json()["detail"]
        assert detail["error"] == "filter_mismatch"
        assert detail["cursor_filters"] == {"q": "目标", "from": 1, "to": 2, "ctx": 20}

    # to_version 缺省即沿用游标冻结上界；显式给出相同值也一致
    assert _search(client, doc, query="目标", limit=1, cursor=cursor)["hit_count"] == 1
    assert _search(
        client, doc, query="目标", limit=1, to_version=2, cursor=cursor
    )["hit_count"] == 1
    # limit 不是筛选条件，续页可改变页大小
    assert _search(client, doc, query="目标", limit=2, cursor=cursor)["hit_count"] == 2
