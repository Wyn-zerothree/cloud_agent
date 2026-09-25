# 评测原始记录

这里是根目录 README「五、评测」各条结论背后的**原始输出**，未经编辑。每个文件都对应 `data/eval/` 下的一个脚本。

跑批中途产生的临时 dump（如缓存 dump、断点文件）留在本地，未上传。

## 文件

| 文件 | 产出脚本 | 内容 |
|------|---------|------|
| `routing_runs.jsonl` | `run_routing_eval.py` | 50 条题目的路由判定（`gold` / `decision` / `correct` / `note` / `latency_s`） |
| `hybrid_runs.jsonl` | `run_hybrid_eval.py` | 30 题双路检索的**原始规则判定**（字符串匹配口径，已被下面的 judge 输出取代） |
| `hybrid_judged_v1.jsonl` | `judge_hybrid.py` | 30 题 LLM 判官判定，第一次跑（图谱 16/30） |
| `hybrid_judged.jsonl` | `judge_hybrid.py` | 30 题 LLM 判官判定，第二次跑（图谱 13/30）——**同一脚本、同一输入，两次结果不同** |
| `hybrid_contexts_v1.jsonl` / `hybrid_contexts.jsonl` | `rescore_hybrid.py` | 两路检索到的**原文**落盘，供重新判分而不重跑检索 |
| `cache_latency.txt` | `measure_cache_latency.py` | 缓存四档延迟 + 冷启动对照，各 2–3 次 |
| `verify_ltmem.txt` | `verify_ltmem_write.py` | 长期记忆**写路径**验证（抽取 → 落库 → 回读） |
| `verify_ltmem_isolation.txt` | `verify_ltmem_isolation.py` | 长期记忆的跨用户隔离验证（本人能查到、拿他人 user_id 查不到） |
| `verify_cache_isolation.txt` | `verify_cache_isolation.py` | 语义缓存的跨用户隔离验证（含「公共域条目仍可见」的对照，排除缓存整体失效） |
| `verify_poster.txt` + `poster_sample.png` | `verify_poster.py` | `generate_ai_poster` **工具函数**真出图的证据（一次调用 status=200，图已缩至 600*1050 入库）——覆盖到函数级，不含 MCP stdio 调用链 |
| `probe_filter_injection.txt` | `probe_filter_injection.py` | 过滤表达式注入探针 |
| `probe_ltmem.txt` | `probe_ltmem.py` | Milvus 集合清单与 row count 快照 |
| `scan_dead_wiring.txt` | `scan_dead_wiring.py` | 未接线代码静态扫描 |

`probe_stores.py`（Neo4j + Milvus 只读探查）没有对应的产物文件——它只往 stdout 打，结果已摘进根 README。

## 读数据时的几点注意

1. **`hybrid_judged.jsonl` 与 `hybrid_judged_v1.jsonl` 的图谱路数字不一样，这不是 bug。** LLM 生成的 Cypher 在 `temperature=0` 下仍有 26/30 条跨次不同，图谱侧判分因此抖动（16/30 与 13/30）。**向量路的 28/30 与并集的 30/30 在两次判分中完全一致**，所以 README 只报这两个，图谱单独命中率不报。

2. **`hybrid_runs.jsonl` 是废弃的第一版口径，保留是为了留痕。** 它用字符串匹配判「参考答案里的裸值是否出现在检索原文里」，对散文式回答和结构化裸值无法同时适配（如必须包含 `cn-hangzhou`，散文里写「杭州地域」就判不中）。后改用 LLM 判官，见根 README 5.2。

3. **`probe_ltmem.txt` 里 `long_term_memory: num_entities=0` 是「修复前」的快照。** 该探针跑的时候长期记忆的写路径还是死的（调了不存在的方法，集合一直 0 行）；修复后的证据在 `verify_ltmem.txt`（该 user 6 行）。两份文件时间不同，不要当成互相矛盾。

4. **`probe_filter_injection.txt` 里 `long_term_memory` 那一栏是无效的。** 四行全 0，因为当时该集合整体就是 0 行（见上一条）——探针在这里**证不了注入是否被挡住**，只是恰好都查不到。真正有判别力的是 `qa_semantic_cache` 那一栏：未转义 payload 命中全表 12 行，转义后 0 行。

5. **`scan_dead_wiring.txt` 的 A 类（零调用点）混着两类东西。** 一类是真的死代码；另一类是 **MCP server 里以名字暴露给客户端的工具**（`query_user_orders`、`search_product_catalog` 等）——它们由 MCP 客户端按名字经 stdio 调用，静态分析看不见调用点。归类时别把后者当成可以删的死代码。

## 复跑

各脚本用法见根 README「五、评测」。除了 `measure_cache_latency.py`（要打真实后端 + Redis/Milvus）与三个 probe（要 Neo4j/Milvus 在线），其余都只需要 DashScope key。
