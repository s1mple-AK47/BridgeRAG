import sys
from pathlib import Path
from collections import Counter

# 将项目根目录添加到 sys.path 以便导入加载函数
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 从主脚本中导入完全相同的函数
from scripts.run_offline_pipeline_small import load_documents_from_jsonl
from bridgerag.utils.logging_config import setup_logging

# 配置日志，以便看到加载函数的输出
setup_logging()

def main():
    """
    此脚本用于验证文档加载函数 `load_documents_from_jsonl`
    是否正确地对源文件 data.jsonl 进行了去重。
    """
    print("--- 开始验证源文件文档去重逻辑 ---")

    data_file = project_root / "data.jsonl"

    # 调用加载函数获取文档列表
    print(f"正在从 {data_file} 加载文档...")
    all_documents = load_documents_from_jsonl(data_file)
    
    if not all_documents:
        print("未能加载任何文档，检查结束。")
        return

    print(f"\n加载函数报告，共加载了 {len(all_documents)} 篇文档。")
    print("现在开始实际检查列表中是否存在重复ID...")

    # 提取所有文档ID到一个列表中
    doc_ids = [doc['id'] for doc in all_documents]

    # 使用 Counter 查找重复项
    id_counts = Counter(doc_ids)
    duplicates = {id: count for id, count in id_counts.items() if count > 1}

    if not duplicates:
        print("\n[✔] 验证成功：加载的文档列表中没有发现任何重复的文档ID。")
        print("这证明 `load_documents_from_jsonl` 函数的去重逻辑是有效的。")
    else:
        print(f"\n[!] 验证失败：在加载的文档列表中发现了 {len(duplicates)} 个重复的文档ID！")
        print("这意味着 `load_documents_from_jsonl` 函数的去重逻辑存在问题。")
        print("以下是重复的ID及其出现次数:")
        for doc_id, count in duplicates.items():
            print(f"  - ID: '{doc_id}', 出现了 {count} 次")

if __name__ == "__main__":
    main()
