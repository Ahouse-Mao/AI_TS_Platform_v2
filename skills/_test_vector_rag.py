"""
向量数据库与 RAG 技能综合测试脚本

测试覆盖：
  1. VectorDBSkill.insert()       — 写入经验
  2. VectorDBSkill.similarity_search() — 语义搜索
  3. VectorDBSkill.query_by_metadata() — 标量过滤查询
  4. VectorDBSkill.delete_by_task_id() — 按任务 ID 删除
  5. RAGSkill.search()            — 高层语义检索

每次测试都会披露内部的运转过程（嵌入向量、Milvus 查询、schema 结构等）。

用法：
  uv run python skills/test_vector_rag.py
"""

import sys
import os
import json
import logging
import time

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 临时关闭日志（测试脚本自己打印详细信息）
logging.disable(logging.CRITICAL)

from skills.vector_db_skill import VectorDBSkill
from skills.rag_skill import RAGSkill
from backend.RAG.rag_struct import clear_index, MILVUS_DB_PATH, PERSIST_DIR


# ============================================================
# 格式化输出辅助
# ============================================================

def section(title: str):
    """打印分区标题"""
    print()
    print("╔" + "═" * 70 + "╗")
    print(f"║  {title:^66s}║")
    print("╚" + "═" * 70 + "╝")
    print()


def step(label: str, detail: str = ""):
    """打印步骤"""
    print(f"  ▶ {label}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"    {line}")
    print()


def show_json(label: str, data, indent: int = 2):
    """格式化 JSON 输出"""
    print(f"  📄 {label}:")
    if isinstance(data, (dict, list)):
        printed = json.dumps(data, ensure_ascii=False, indent=indent)
        for line in printed.split("\n"):
            print(f"    {line}")
    else:
        print(f"    {data}")
    print()


def show_dict_table(label: str, data_list: list[dict]):
    """以表格形式展示字典列表"""
    print(f"  📊 {label}:")
    if not data_list:
        print("    (空)")
        print()
        return
    # 收集所有 key
    keys = list(data_list[0].keys())
    # 打印表头
    header = "  │  ".join(keys)
    print(f"    ┌─ {header}")
    print(f"    │")
    for i, item in enumerate(data_list):
        vals = []
        for k in keys:
            v = item.get(k, "")
            s = str(v)
            if len(s) > 30:
                s = s[:27] + "..."
            vals.append(s)
        line = "  │  ".join(vals)
        print(f"    ├─ [{i}] {line}")
    print()


# ============================================================
# 测试准备
# ============================================================

def setup():
    """测试前准备：创建 VectorDBSkill 实例，准备干净的 Milvus 数据库"""
    section("准备工作")

    # 1. 检查 Milvus 数据库路径
    step("检查数据库路径", f"db_path  : {MILVUS_DB_PATH}\npersist  : {PERSIST_DIR}")

    # 2. 清理旧索引，由 VectorDBSkill 自动创建动态 schema
    step("清理旧索引")
    clear_result = clear_index()
    print(f"      {clear_result['message']}")
    print()

    # 3. 创建 VectorDBSkill（首次操作时会自动创建 collection）
    step("创建 VectorDBSkill 实例")
    vdb = VectorDBSkill()
    print(f"      嵌入模型: bge-small-zh-v1.5（本地离线加载）")
    print(f"      向量维度: {len(vdb._embeddings.embed_query('测试'))}")
    print()

    # 4. 显示嵌入模型信息
    test_vec = vdb._embeddings.embed_query("测试向量维度")
    step("嵌入模型信息",
         f"模型路径  : backend/RAG/bge-small-zh-v1.5\n"
         f"向量维度  : {len(test_vec)}\n"
         f"前 5 维  : {[f'{x:.4f}' for x in test_vec[:5]]}")

    return vdb


# ============================================================
# 测试 1：VectorDBSkill.insert()
# ============================================================

def test_insert(vdb: VectorDBSkill):
    section("测试 1：VectorDBSkill.insert() — 写入经验记录")

    # ---- 1.1 插入第一条训练经验 ----
    step("1.1 插入第一条记录（DLinear / ETTh1）")

    data1 = {
        "experience": "在 ETTh1 数据集上使用 DLinear 模型训练 50 轮，"
                      "seq_len=96, pred_len=96, batch_size=64, lr=0.005，"
                      "最终 MSE=0.152, MAE=0.253。模型收敛良好，未出现过拟合。",
        "model": "DLinear",
        "dataset": "ETTh1",
        "seq_len": "96",
        "pred_len": "96",
    }
    metadata1 = {
        "model": "DLinear",
        "dataset": "ETTh1",
        "task_id": "exp_001",
    }

    print("    要插入的数据:")
    show_json("data", data1, indent=4)
    show_json("metadata", metadata1, indent=4)

    # 展示嵌入过程
    print("    嵌入过程（内部）:")
    embedding = vdb._embeddings.embed_query(data1["experience"])
    print(f"      输入文本长度: {len(data1['experience'])} 字")
    print(f"      输出向量维度: {len(embedding)}")
    print(f"      向量前 3 维  : [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}]")
    print(f"      向量后 3 维  : [{embedding[-3]:.4f}, {embedding[-2]:.4f}, {embedding[-1]:.4f}]")
    print()

    id1 = vdb.insert(collection="rag_struct", data=data1, metadata=metadata1)
    step("1.1 结果", f"插入 ID: {id1}")
    print()

    # ---- 1.2 插入第二条训练经验 ----
    step("1.2 插入第二条记录（PatchTST / ETTh1）")

    data2 = {
        "experience": "在 ETTh1 数据集上使用 PatchTST 模型训练 50 轮，"
                      "seq_len=96, pred_len=96, patch_len=16, stride=8，"
                      "最终 MSE=0.168, MAE=0.271。Transformer 架构表现稳定。",
        "model": "PatchTST",
        "dataset": "ETTh1",
        "seq_len": "96",
        "pred_len": "96",
    }
    metadata2 = {
        "model": "PatchTST",
        "dataset": "ETTh1",
        "task_id": "exp_002",
    }

    id2 = vdb.insert(collection="rag_struct", data=data2, metadata=metadata2)
    step("1.2 结果", f"插入 ID: {id2}")
    print()

    # ---- 1.3 插入第三条训练经验（不同数据集） ----
    step("1.3 插入第三条记录（DLinear / weather，不同数据集）")

    data3 = {
        "experience": "在 weather 数据集上使用 DLinear 模型训练 30 轮，"
                      "seq_len=96, pred_len=96，"
                      "最终 MSE=0.234, MAE=0.345。天气数据波动较大，MSE 偏高。",
        "model": "DLinear",
        "dataset": "weather",
        "seq_len": "96",
        "pred_len": "96",
    }
    metadata3 = {
        "model": "DLinear",
        "dataset": "weather",
        "task_id": "exp_003",
    }

    id3 = vdb.insert(collection="rag_struct", data=data3, metadata=metadata3)
    step("1.3 结果", f"插入 ID: {id3}")

    # ---- 1.4 查看当前 collection 统计 ----
    step("1.4 查看 collection 状态")
    stats = vdb.client.get_collection_stats("rag_struct")
    show_json("rag_struct 统计", stats)

    return [id1, id2, id3]


# ============================================================
# 测试 2：VectorDBSkill.similarity_search()
# ============================================================

def test_similarity_search(vdb: VectorDBSkill):
    section("测试 2：VectorDBSkill.similarity_search() — 语义搜索")

    # ---- 2.1 无过滤条件的语义搜索 ----
    step("2.1 无条件语义搜索（query='DLinear 预测 ETTh1 气温'）")

    query1 = "DLinear 预测 ETTh1 气温"
    print(f"      查询文本: \"{query1}\"")
    print()

    # 披露嵌入过程
    q_vec = vdb._embeddings.embed_query(query1)
    print(f"    嵌入过程（内部）:")
    print(f"      查询向量维度: {len(q_vec)}")
    print(f"      前 3 维: [{q_vec[0]:.4f}, {q_vec[1]:.4f}, {q_vec[2]:.4f}]")
    print()

    # 披露 Milvus 搜索调用
    print(f"    即将调用: MilvusStore.similarity_search(query, k=5, filter=None)")
    print(f"      → 底层: MilvusClient.search(collection='rag_struct', data=[query_emb], limit=5)")
    print()

    results1 = vdb.similarity_search(query=query1, top_k=5)

    print(f"    返回 {len(results1)} 条结果:")
    for i, r in enumerate(results1):
        print(f"      [{i}] score={r['score']:.4f} | "
              f"model={r['metadata'].get('model','?')} | "
              f"dataset={r['metadata'].get('dataset','?')}")
        print(f"          text: {r['text'][:80]}{'...' if len(r['text']) > 80 else ''}")
    print()

    # ---- 2.2 带过滤条件的语义搜索 ----
    step("2.2 带过滤条件搜索（model='DLinear'）")

    query2 = "时序预测模型效果"
    print(f"      查询文本: \"{query2}\"")
    print(f"      过滤条件: model='DLinear'")
    print()

    print(f"    即将调用: MilvusStore.similarity_search(query, k=5, filter={{'model': 'DLinear'}})")
    print(f"      → 过滤表达式: model == \"DLinear\"")
    print()

    results2 = vdb.similarity_search(query=query2, top_k=5, filter={"model": "DLinear"})

    print(f"    返回 {len(results2)} 条结果:")
    for i, r in enumerate(results2):
        print(f"      [{i}] score={r['score']:.4f} | "
              f"model={r['metadata'].get('model','?')} | "
              f"dataset={r['metadata'].get('dataset','?')}")
    print()

    # ---- 2.3 对比：无过滤 vs 有过滤 ----
    step("2.3 对比分析")
    print(f"      无条件搜索: 检索全部 3 条插入记录")
    print(f"      model过滤 : 仅检索 DLinear 相关记录（排除了 PatchTST）")
    print(f"      效果: 过滤后结果数减少，相关性更聚焦")
    print()

    return results1, results2


# ============================================================
# 测试 3：VectorDBSkill.query_by_metadata()
# ============================================================

def test_query_by_metadata(vdb: VectorDBSkill):
    section("测试 3：VectorDBSkill.query_by_metadata() — 标量过滤查询")

    # ---- 3.1 按模型查询 ----
    step("3.1 按模型过滤: model == 'DLinear'")

    print(f"    即将调用: MilvusClient.query(collection='rag_struct', filter='model == \"DLinear\"')")
    print(f"      → 这是一个标量查询，不走向量索引，直接按字段过滤")
    print(f"      → 返回所有 model 字段等于 DLinear 的记录")
    print()

    results = vdb.query_by_metadata(
        collection="rag_struct",
        filter_expr='model == "DLinear"',
        limit=10,
    )

    print(f"    返回 {len(results)} 条结果:")
    for i, r in enumerate(results):
        print(f"      [{i}] task_id={r.get('task_id','?')} | "
              f"model={r.get('model','?')} | "
              f"dataset={r.get('dataset','?')}")
        text_preview = r.get("text", "")
        print(f"          text: {text_preview[:80]}{'...' if len(text_preview) > 80 else ''}")
    print()

    # ---- 3.2 按模型+数据集组合查询 ----
    step("3.2 组合过滤: model == 'DLinear' and dataset == 'ETTh1'")

    print(f"    过滤表达式: model == \"DLinear\" and dataset == \"ETTh1\"")
    print()

    results2 = vdb.query_by_metadata(
        collection="rag_struct",
        filter_expr='model == "DLinear" and dataset == "ETTh1"',
        limit=10,
    )

    print(f"    返回 {len(results2)} 条结果:")
    for i, r in enumerate(results2):
        print(f"      [{i}] task_id={r.get('task_id','?')} | "
              f"model={r.get('model','?')} | "
              f"dataset={r.get('dataset','?')}")
        text_preview = r.get("text", "")
        print(f"          text: {text_preview[:80]}{'...' if len(text_preview) > 80 else ''}")
    print()

    # ---- 3.3 查看返回的完整字段 ----
    if results:
        step("3.3 单条记录的完整字段结构")
        show_json("第一条记录的完整内容", results[0], indent=4)

    return results


# ============================================================
# 测试 4：VectorDBSkill.delete_by_task_id()
# ============================================================

def test_delete(vdb: VectorDBSkill, inserted_ids: list):
    section("测试 4：VectorDBSkill.delete_by_task_id() — 按任务 ID 删除")

    task_id_to_delete = "exp_003"  # 删除 weather/DLinear 那条

    # ---- 4.1 删除前确认数据存在 ----
    step(f"4.1 删除前查询: task_id='{task_id_to_delete}'")

    before = vdb.query_by_metadata(
        collection="rag_struct",
        filter_expr=f'task_id == "{task_id_to_delete}"',
    )
    print(f"    删除前匹配记录数: {len(before)}")
    if before:
        print(f"    即将删除: model={before[0].get('model','?')} | "
              f"dataset={before[0].get('dataset','?')} | "
              f"task_id={before[0].get('task_id','?')}")
    print()

    # ---- 4.2 执行删除 ----
    step(f"4.2 执行删除: collection='rag_struct', task_id='{task_id_to_delete}'")

    print(f"    即将调用: MilvusClient.delete(collection='rag_struct', "
          f"filter='task_id == \"{task_id_to_delete}\"')")
    print(f"      → 底层会先 load_collection，然后执行标量删除")
    print()

    result = vdb.delete_by_task_id(collection="rag_struct", task_id=task_id_to_delete)

    print(f"    删除结果: {'✅ 成功' if result else '❌ 失败'}")
    print()

    # ---- 4.3 删除后验证 ----
    step(f"4.3 删除后验证: task_id='{task_id_to_delete}'")

    after = vdb.query_by_metadata(
        collection="rag_struct",
        filter_expr=f'task_id == "{task_id_to_delete}"',
    )
    print(f"    删除后匹配记录数: {len(after)}")
    if len(after) == 0:
        print(f"    ✅ 记录已成功删除")
    else:
        print(f"    ⚠️  记录仍然存在")
    print()

    # ---- 4.4 验证其他记录不受影响 ----
    step("4.4 验证其他记录不受影响")

    remaining = vdb.query_by_metadata(
        collection="rag_struct",
        filter_expr='model == "DLinear"',
        limit=10,
    )
    print(f"    DLinear 记录剩余: {len(remaining)} 条")
    for r in remaining:
        print(f"      - task_id={r.get('task_id','?')} | dataset={r.get('dataset','?')}")
    print()

    return result


# ============================================================
# 测试 5：RAGSkill.search() — 高层语义封装
# ============================================================

def test_rag_skill(vdb: VectorDBSkill):
    section("测试 5：RAGSkill.search() — 高层语义检索封装")

    # ---- 5.1 创建 RAGSkill ----
    step("5.1 创建 RAGSkill（注入 VectorDBSkill）")

    rag = RAGSkill(vector_db_skill=vdb)
    print(f"    RAGSkill 内部依赖: {type(rag._vector_db).__name__}")
    print(f"      → 通过该依赖间接调用 VectorDBSkill.similarity_search()")
    print()

    # ---- 5.2 无条件检索 ----
    step("5.2 search(query='用 DLinear 做时序预测', top_k=3)")

    print(f"    调用链路:")
    print(f"      RAGSkill.search()")
    print(f"        └─→ VectorDBSkill.similarity_search(query, top_k=3, filter=None)")
    print(f"              └─→ MilvusStore.similarity_search(query, k=3)")
    print(f"                    └─→ MilvusClient.search(data=[query_emb], limit=3)")
    print()

    results1 = rag.search(query="用 DLinear 做时序预测", top_k=3)

    print(f"    返回 {len(results1)} 条结果:")
    for i, r in enumerate(results1):
        print(f"      [{i}] similarity={r['similarity']:.4f} | "
              f"model={r['model_name']} | dataset={r['dataset']}")
        print(f"          experience: {r['experience'][:60]}...")
    print()

    # ---- 5.3 带过滤检索 ----
    step("5.3 search(query='时序预测模型效果', model='DLinear')")

    print(f"    调用链路:")
    print(f"      RAGSkill.search(query, model='DLinear')")
    print(f"        └─→ VectorDBSkill.similarity_search(query, filter={{'model':'DLinear'}})")
    print(f"              └─→ MilvusStore.similarity_search(query, filter={{'model':'DLinear'}})")
    print(f"                    └─→ 过滤表达式: model == \"DLinear\"")
    print()

    results2 = rag.search(query="时序预测模型效果", top_k=5, model="DLinear")

    print(f"    返回 {len(results2)} 条结果:")
    for i, r in enumerate(results2):
        print(f"      [{i}] similarity={r['similarity']:.4f} | "
              f"model={r['model_name']} | dataset={r['dataset']}")
    print()

    # ---- 5.4 对比：RAGSkill 输出 vs VectorDBSkill 原始输出 ----
    step("5.4 RAGSkill 输出格式 vs VectorDBSkill 原始输出格式")

    if results1:
        print(f"    RAGSkill.search() 输出格式:")
        show_json("示例", results1[0], indent=4)
        print()

        raw = vdb.similarity_search(query="用 DLinear 做时序预测", top_k=1)
        if raw:
            print(f"    VectorDBSkill.similarity_search() 原始输出格式:")
            show_json("示例", raw[0], indent=4)
            print()

        print(f"    差异总结:")
        print(f"      RAGSkill 将 metadata 展开为顶层字段:")
        print(f"        model_name ← metadata.model")
        print(f"        dataset    ← metadata.dataset")
        print(f"        params     ← seq_len, pred_len, features")
        print(f"        similarity ← score")
        print(f"        experience ← text")
        print()

    return results1


# ============================================================
# 测试 6：边界情况测试
# ============================================================

def test_edge_cases(vdb: VectorDBSkill):
    section("测试 6：边界情况")

    # ---- 6.1 空查询 ----
    step("6.1 空字符串查询")

    results = vdb.similarity_search(query="", top_k=3)
    print(f"      空查询返回 {len(results)} 条结果")
    print()

    # ---- 6.2 不存在的 collection ----
    step("6.2 查询不存在的 collection")

    results = vdb.query_by_metadata(
        collection="不存在的集合",
        filter_expr='model == "DLinear"',
    )
    print(f"      不存在 collection 返回: {results}")
    print()

    # ---- 6.3 删除不存在 task_id ----
    step("6.3 删除不存在的 task_id")

    result = vdb.delete_by_task_id(collection="rag_struct", task_id="不存在的ID")
    print(f"      删除不存在记录结果: {result}")
    print()

    # ---- 6.4 插入空数据 ----
    step("6.4 插入空文本")

    data_empty = {
        "experience": "",
        "model": "test",
    }
    metadata_empty = {"model": "test", "task_id": "exp_empty"}
    try:
        id_empty = vdb.insert(collection="rag_struct", data=data_empty, metadata=metadata_empty)
        print(f"      空文本插入 ID: {id_empty}")
    except Exception as e:
        print(f"      插入异常: {e}")
    print()


# ============================================================
# 主入口
# ============================================================

def main():
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + "      VectorDBSkill + RAGSkill 综合测试".center(66) + "║")
    print("║" + "      披露内部运转过程".center(66) + "║")
    print("╚" + "═" * 70 + "╝")
    print()

    # 初始化
    vdb = setup()

    # 执行测试
    inserted_ids = test_insert(vdb)
    test_similarity_search(vdb)
    test_query_by_metadata(vdb)
    test_delete(vdb, inserted_ids)
    test_rag_skill(vdb)
    test_edge_cases(vdb)

    # ---- 汇总 ----
    section("测试完成")
    print(f"  测试项:")
    print(f"    ✅ VectorDBSkill.insert()              — 插入 3 条经验记录")
    print(f"    ✅ VectorDBSkill.similarity_search()   — 无条件 + 带过滤 语义搜索")
    print(f"    ✅ VectorDBSkill.query_by_metadata()   — 标量字段过滤查询")
    print(f"    ✅ VectorDBSkill.delete_by_task_id()   — 删除 exp_003 并验证")
    print(f"    ✅ RAGSkill.search()                  — 高层封装检索")
    print(f"    ✅ 边界情况测试                        — 空查询/空集合/空文本")
    print()
    print(f"  数据库路径: {MILVUS_DB_PATH}")
    print()


if __name__ == "__main__":
    main()
