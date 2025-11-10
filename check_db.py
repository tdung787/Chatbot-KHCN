import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()

# Config
QDRANT_PATH = "database/qdrant_storage"
COLLECTION_NAME = "KHTN_QA"

def main():
    print("=" * 70)
    print("KIỂM TRA QDRANT DATABASE")
    print("=" * 70)
    
    # Check if storage exists
    if not Path(QDRANT_PATH).exists():
        print(f"❌ Không tìm thấy Qdrant storage tại: {QDRANT_PATH}")
        return
    
    print(f"✓ Qdrant storage tồn tại: {QDRANT_PATH}")
    
    # Connect to Qdrant
    try:
        client = QdrantClient(path=QDRANT_PATH)
        print("✓ Đã kết nối Qdrant client")
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        return
    
    # List collections
    collections = client.get_collections().collections
    print(f"\n📦 Collections: {len(collections)}")
    for col in collections:
        print(f"   - {col.name}")
    
    # Check if our collection exists
    if not any(c.name == COLLECTION_NAME for c in collections):
        print(f"\n❌ Collection '{COLLECTION_NAME}' không tồn tại!")
        return
    
    # Get collection info
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"\n✓ Collection '{COLLECTION_NAME}' tồn tại")
    print(f"   - Vectors count: {collection_info.points_count}")
    print(f"   - Vector dimensions: {collection_info.config.params.vectors.size}")
    print(f"   - Distance metric: {collection_info.config.params.vectors.distance}")
    
    if collection_info.points_count == 0:
        print("\n⚠️  Collection rỗng - chưa có vector nào được upload!")
        return
    
    # Sample some points
    print(f"\n📋 Lấy mẫu {min(5, collection_info.points_count)} vectors:")
    
    try:
        # Scroll through first few points
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        points = scroll_result[0]
        
        for i, point in enumerate(points, 1):
            print(f"\n{i}. ID (hash): {point.id}")
            print(f"   Question ID: {point.payload.get('id', 'N/A')}")
            print(f"   Question: {point.payload.get('question', 'N/A')[:80]}...")
            print(f"   Correct Answer: {point.payload.get('correct_answer', 'N/A')}")
            print(f"   Primary Page: {point.payload.get('primary_page', 'N/A')}")
            print(f"   Subject: {point.payload.get('subject', 'N/A')}")
            
            if 'spans_pages' in point.payload:
                print(f"   Spans Pages: {point.payload['spans_pages']}")
        
    except Exception as e:
        print(f"❌ Lỗi khi lấy mẫu: {e}")
        return
    
    # Test search functionality
    print("\n" + "=" * 70)
    print("TEST SEARCH")
    print("=" * 70)
    
    test_query = "Đối tượng nghiên cứu của Vật lí?"
    print(f"Query: '{test_query}'")
    
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Embed query
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=test_query
        )
        query_vector = response.data[0].embedding
        
        # Search
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        
        print(f"\n🔍 Top 3 kết quả tìm kiếm:")
        for i, result in enumerate(search_results, 1):
            print(f"\n{i}. Score: {result.score:.4f}")
            print(f"   ID: {result.payload.get('id', 'N/A')}")
            print(f"   Question: {result.payload.get('question', 'N/A')[:80]}...")
            print(f"   Answer: {result.payload.get('correct_answer', 'N/A')} - {result.payload.get('correct_answer_text', 'N/A')[:60]}...")
        
        print("\n✅ Search hoạt động tốt!")
        
    except ImportError:
        print("\n⚠️  Cần cài openai để test search: pip install openai")
    except Exception as e:
        print(f"\n❌ Lỗi khi test search: {e}")
    
    # Statistics by page
    print("\n" + "=" * 70)
    print("THỐNG KÊ THEO PAGE")
    print("=" * 70)
    
    try:
        # Count questions per page (sample approach - scroll all)
        all_points = []
        offset = None
        
        while True:
            scroll_result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, offset = scroll_result
            all_points.extend(points)
            
            if offset is None:
                break
        
        # Count by page
        page_counts = {}
        split_count = 0
        
        for point in all_points:
            page = point.payload.get('primary_page', 'unknown')
            page_counts[page] = page_counts.get(page, 0) + 1
            
            if 'spans_pages' in point.payload:
                split_count += 1
        
        print(f"📊 Tổng số câu hỏi: {len(all_points)}")
        print(f"📄 Câu hỏi bị ngắt trang: {split_count}")
        print(f"📁 Số page khác nhau: {len(page_counts)}")
        
        # Top pages
        sorted_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🔝 Top 10 pages có nhiều câu hỏi nhất:")
        for page, count in sorted_pages[:10]:
            print(f"   {page}: {count} câu hỏi")
        
    except Exception as e:
        print(f"❌ Lỗi khi thống kê: {e}")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT KIỂM TRA!")
    print("=" * 70)

if __name__ == "__main__":
    main()