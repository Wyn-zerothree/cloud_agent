# -*- coding: utf-8 -*-
"""零成本探针：只看 Neo4j / Milvus 里有什么，不调任何 LLM。"""
import os
import sys
from collections import Counter
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[2] / "agent"
sys.path.insert(0, str(AGENT_DIR))

from dotenv import load_dotenv
load_dotenv(AGENT_DIR / ".env")


def probe_neo4j():
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI")
    drv = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))
    with drv.session() as s:
        print("=" * 70)
        print("NEO4J", uri)
        print("=" * 70)

        labels = [r["l"] for r in s.run("MATCH (n) UNWIND labels(n) AS l RETURN DISTINCT l")]
        print(f"\n节点标签 ({len(labels)}): {sorted(labels)}")

        print("\n各标签节点数:")
        for r in s.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY c DESC"):
            print(f"  {r['l']:28s} {r['c']}")

        rels = [r["t"] for r in s.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t")]
        print(f"\n关系类型 ({len(rels)}): {sorted(rels)}")
        for r in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"):
            print(f"  {r['t']:28s} {r['c']}")

        print("\n--- InstanceType 样本 (最多 25) ---")
        for r in s.run("MATCH (n:InstanceType) RETURN n.id AS id, properties(n) AS p ORDER BY n.id LIMIT 25"):
            print(f"  {r['id']:26s} {r['p']}")

        print("\n--- 其他标签样本 ---")
        for lbl in sorted(labels):
            if lbl == "InstanceType":
                continue
            rows = [r.data() for r in s.run(
                f"MATCH (n:{lbl}) RETURN n.id AS id, n.name AS name LIMIT 4")]
            print(f"  [{lbl}] ", end="")
            print(", ".join(str(x.get("id") or x.get("name")) for x in rows) or "(无 id/name)")

        print("\n--- 关系样本 (最多 30) ---")
        q = """
        MATCH (a)-[r]->(b)
        RETURN labels(a) AS fl, coalesce(a.id,a.name,'') AS f,
               type(r) AS t, labels(b) AS tl, coalesce(b.id,b.name,'') AS tt
        LIMIT 30
        """
        for r in s.run(q):
            print(f"  [{','.join(r['fl'])}] {r['f']}  --{r['t']}-->  [{','.join(r['tl'])}] {r['tt']}")
    drv.close()


def probe_milvus():
    from pymilvus import connections, Collection, utility
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    print("\n" + "=" * 70)
    print(f"MILVUS {host}:{port}")
    print("=" * 70)
    connections.connect(alias="probe", host=host, port=port)
    cols = utility.list_collections(using="probe")
    print(f"\n集合列表: {cols}")
    for name in cols:
        c = Collection(name, using="probe")
        try:
            c.load()
        except Exception as e:
            print(f"  {name}: load 失败 {e}")
            continue
        print(f"\n--- {name} : {c.num_entities} 条 ---")
        try:
            rows = c.query(expr="", output_fields=["*"], limit=5)
            for row in rows:
                brief = {k: (str(v)[:80] + "..." if len(str(v)) > 80 else v)
                         for k, v in row.items() if k != "vector"}
                print(f"    {brief}")
        except Exception as e:
            print(f"    查询失败: {e}")
        if name == "cloud_product_docs":
            try:
                rows = c.query(expr="", output_fields=["source"], limit=200)
                cnt = Counter(os.path.basename(str(r.get("source", ""))) for r in rows)
                print(f"\n    source 分布 ({len(cnt)} 个):")
                for k, v in cnt.most_common():
                    print(f"      {k:45s} {v}")
            except Exception as e:
                print(f"    source 分布查询失败: {e}")


if __name__ == "__main__":
    probe_neo4j()
    probe_milvus()
