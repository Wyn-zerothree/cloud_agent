"""用于 Neo4j 操作的知识图谱模块。

注意：本包没有被运行中的应用引用——图谱检索走 tools/graph_tool.py 的
langchain_neo4j.Neo4jGraph，而当前 Neo4j 里那 122 个节点是 agent/test/build_kg.py
用裸 neo4j driver 建的。这里是一套未被采用的备选 ingestion 实现，保留作参考。
"""

from .client import Neo4jClient
from .ingestor import KnowledgeGraphIngestor
from .parser import KnowledgeGraphParser
from .models import (
    Product, InstanceType, Region, Image, BillingMode,
    DatabaseEngine, StorageType, Relation,
)

__all__ = [
    "Neo4jClient",
    "KnowledgeGraphIngestor",
    "KnowledgeGraphParser",
    "Product",
    "InstanceType",
    "Region",
    "Image",
    "BillingMode",
    "DatabaseEngine",
    "StorageType",
    "Relation",
]
