"""
Script đánh giá chất lượng bộ dữ liệu huấn luyện NER (train.jsonl)
- Thống kê tổng quan
- Phân bố nhãn (label distribution)
- Kiểm tra lỗi dữ liệu (tokens vs ner_tags length mismatch)
- Phân tích sự cân bằng dữ liệu
- Phân tích độ dài trung bình các thực thể
"""

import json
import os
from collections import Counter, defaultdict

FILE_PATH = 'data/annotated/train.jsonl'

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                obj['_line_num'] = i + 1
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"  [LỖI] Dòng {i+1}: Không thể parse JSON - {e}")
    return data

def check_data_integrity(data):
    """Kiểm tra tính toàn vẹn dữ liệu"""
    errors = []
    for item in data:
        line_num = item['_line_num']
        tokens = item.get('tokens', [])
        tags = item.get('ner_tags', [])
        
        if len(tokens) != len(tags):
            errors.append({
                'line': line_num,
                'id': item.get('id', 'N/A'),
                'num_tokens': len(tokens),
                'num_tags': len(tags),
            })
        
        # Kiểm tra BIO format hợp lệ
        for j, tag in enumerate(tags):
            if tag.startswith('I-'):
                entity_type = tag[2:]
                if j == 0 or (not tags[j-1].endswith(entity_type)):
                    pass  # I- tag without preceding B- tag (acceptable in some schemes)
    
    return errors

def compute_label_distribution(data):
    """Phân bố nhãn"""
    tag_counter = Counter()
    entity_counter = Counter()
    
    for item in data:
        tags = item.get('ner_tags', [])
        for tag in tags:
            tag_counter[tag] += 1
            if tag != 'O':
                # Extract entity type
                entity_type = tag[2:] if tag.startswith(('B-', 'I-')) else tag
                entity_counter[entity_type] += 1
    
    return tag_counter, entity_counter

def compute_entity_stats(data):
    """Thống kê thực thể: số entity, độ dài trung bình, entity mẫu"""
    entity_counts = Counter()
    entity_lengths = defaultdict(list)
    entity_examples = defaultdict(list)
    
    for item in data:
        tokens = item.get('tokens', [])
        tags = item.get('ner_tags', [])
        
        current_entity = None
        current_tokens = []
        
        for token, tag in zip(tokens, tags):
            if tag.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entity_counts[current_entity] += 1
                    entity_lengths[current_entity].append(len(current_tokens))
                    if len(entity_examples[current_entity]) < 5:
                        entity_examples[current_entity].append(' '.join(current_tokens))
                
                current_entity = tag[2:]
                current_tokens = [token]
            elif tag.startswith('I-') and current_entity:
                current_tokens.append(token)
            else:
                if current_entity:
                    entity_counts[current_entity] += 1
                    entity_lengths[current_entity].append(len(current_tokens))
                    if len(entity_examples[current_entity]) < 5:
                        entity_examples[current_entity].append(' '.join(current_tokens))
                current_entity = None
                current_tokens = []
        
        # Don't forget last entity
        if current_entity:
            entity_counts[current_entity] += 1
            entity_lengths[current_entity].append(len(current_tokens))
            if len(entity_examples[current_entity]) < 5:
                entity_examples[current_entity].append(' '.join(current_tokens))
    
    return entity_counts, entity_lengths, entity_examples

def main():
    print("=" * 70)
    print("  ĐÁNH GIÁ BỘ DỮ LIỆU HUẤN LUYỆN NER")
    print("=" * 70)
    
    if not os.path.exists(FILE_PATH):
        print(f"[LỖI] Không tìm thấy file: {FILE_PATH}")
        return
    
    data = load_data(FILE_PATH)
    
    # 1. Tổng quan
    print(f"\n{'─'*70}")
    print("1. THỐNG KÊ TỔNG QUAN")
    print(f"{'─'*70}")
    print(f"  Tổng số mẫu (CV)           : {len(data)}")
    
    total_tokens = sum(len(item.get('tokens', [])) for item in data)
    avg_tokens = total_tokens / len(data) if data else 0
    max_tokens = max(len(item.get('tokens', [])) for item in data) if data else 0
    min_tokens = min(len(item.get('tokens', [])) for item in data) if data else 0
    
    print(f"  Tổng số tokens              : {total_tokens:,}")
    print(f"  Số tokens trung bình/mẫu    : {avg_tokens:.1f}")
    print(f"  Số tokens tối thiểu/mẫu     : {min_tokens}")
    print(f"  Số tokens tối đa/mẫu        : {max_tokens}")
    
    # 2. Kiểm tra tính toàn vẹn
    print(f"\n{'─'*70}")
    print("2. KIỂM TRA TÍNH TOÀN VẸN DỮ LIỆU")
    print(f"{'─'*70}")
    errors = check_data_integrity(data)
    if errors:
        print(f"  ⚠ Phát hiện {len(errors)} lỗi tokens/tags không khớp:")
        for err in errors[:10]:
            print(f"    - Dòng {err['line']} (ID: {err['id']}): "
                  f"{err['num_tokens']} tokens vs {err['num_tags']} tags")
        if len(errors) > 10:
            print(f"    ... và {len(errors) - 10} lỗi khác")
    else:
        print("  ✓ Không phát hiện lỗi. Tokens và tags khớp hoàn toàn.")
    
    # 3. Phân bố nhãn
    print(f"\n{'─'*70}")
    print("3. PHÂN BỐ NHÃN (TAG DISTRIBUTION)")
    print(f"{'─'*70}")
    tag_counter, entity_counter = compute_label_distribution(data)
    
    total_tags = sum(tag_counter.values())
    o_count = tag_counter.get('O', 0)
    non_o_count = total_tags - o_count
    
    print(f"  Tổng số tags                : {total_tags:,}")
    print(f"  Tags 'O' (không phải entity): {o_count:,} ({o_count/total_tags*100:.1f}%)")
    print(f"  Tags entity (non-O)         : {non_o_count:,} ({non_o_count/total_tags*100:.1f}%)")
    print(f"  Tỷ lệ O / non-O            : {o_count/non_o_count:.1f}:1" if non_o_count > 0 else "")
    
    # Unique entity types
    entity_types = sorted(set(
        tag[2:] for tag in tag_counter if tag.startswith(('B-', 'I-'))
    ))
    print(f"  Số loại thực thể            : {len(entity_types)}")
    print(f"  Danh sách                   : {', '.join(entity_types)}")
    
    # 4. Thống kê chi tiết từng loại thực thể
    print(f"\n{'─'*70}")
    print("4. THỐNG KÊ CHI TIẾT TỪNG LOẠI THỰC THỂ")
    print(f"{'─'*70}")
    
    entity_counts, entity_lengths, entity_examples = compute_entity_stats(data)
    
    print(f"\n  {'Thực thể':<18} {'Số lượng':>10} {'% Tổng':>10} {'Dài TB':>10} {'Dài max':>10}")
    print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    
    total_entities = sum(entity_counts.values())
    for entity_type in sorted(entity_counts, key=entity_counts.get, reverse=True):
        count = entity_counts[entity_type]
        pct = count / total_entities * 100 if total_entities > 0 else 0
        lengths = entity_lengths[entity_type]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        print(f"  {entity_type:<18} {count:>10,} {pct:>9.1f}% {avg_len:>10.1f} {max_len:>10}")
    
    print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'TỔNG':<18} {total_entities:>10,}")
    
    # 5. Ví dụ mẫu cho mỗi thực thể
    print(f"\n{'─'*70}")
    print("5. VÍ DỤ THỰC THỂ (tối đa 5 mẫu/loại)")
    print(f"{'─'*70}")
    for entity_type in sorted(entity_examples):
        print(f"\n  [{entity_type}]")
        for ex in entity_examples[entity_type]:
            print(f"    • {ex}")
    
    # 6. Đánh giá sự cân bằng dữ liệu
    print(f"\n{'─'*70}")
    print("6. ĐÁNH GIÁ SỰ CÂN BẰNG DỮ LIỆU")
    print(f"{'─'*70}")
    
    if entity_counts:
        counts_list = list(entity_counts.values())
        max_count = max(counts_list)
        min_count = min(counts_list)
        max_entity = max(entity_counts, key=entity_counts.get)
        min_entity = min(entity_counts, key=entity_counts.get)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"  Thực thể nhiều nhất         : {max_entity} ({max_count:,} entities)")
        print(f"  Thực thể ít nhất            : {min_entity} ({min_count:,} entities)")
        print(f"  Tỷ lệ mất cân bằng (max/min): {imbalance_ratio:.1f}x")
        
        if imbalance_ratio > 10:
            print(f"\n  ⚠ CẢNH BÁO: Dữ liệu RẤT MẤT CÂN BẰNG (tỷ lệ > 10x)!")
            print(f"    Các thực thể ít mẫu cần được bổ sung thêm dữ liệu:")
            threshold = max_count * 0.1  # < 10% of max
            for et in sorted(entity_counts, key=entity_counts.get):
                if entity_counts[et] < threshold:
                    print(f"      - {et}: {entity_counts[et]:,} mẫu (< {threshold:.0f})")
        elif imbalance_ratio > 5:
            print(f"\n  ⚠ Dữ liệu tương đối mất cân bằng (tỷ lệ > 5x).")
        else:
            print(f"\n  ✓ Dữ liệu tương đối cân bằng.")
    
    # 7. Phân bố độ dài mẫu
    print(f"\n{'─'*70}")
    print("7. PHÂN BỐ ĐỘ DÀI MẪU (TOKENS PER SAMPLE)")
    print(f"{'─'*70}")
    
    lengths = [len(item.get('tokens', [])) for item in data]
    bins = [0, 50, 100, 200, 300, 500, 1000, float('inf')]
    bin_labels = ['0-50', '51-100', '101-200', '201-300', '301-500', '501-1000', '1000+']
    
    for i in range(len(bins) - 1):
        count = sum(1 for l in lengths if bins[i] < l <= bins[i+1])
        bar = '█' * (count * 40 // len(data)) if data else ''
        print(f"  {bin_labels[i]:>10}: {count:>5} ({count/len(data)*100:>5.1f}%) {bar}")
    
    # 8. Tổng kết
    print(f"\n{'='*70}")
    print("  TỔNG KẾT")
    print(f"{'='*70}")
    print(f"  • Số mẫu: {len(data)} CVs")
    print(f"  • Số loại thực thể: {len(entity_types)}")
    print(f"  • Tổng thực thể: {total_entities:,}")
    
    issues = []
    if len(data) < 200:
        issues.append(f"Số mẫu ít ({len(data)} < 200). Nên bổ sung thêm dữ liệu.")
    if imbalance_ratio > 10:
        issues.append(f"Dữ liệu rất mất cân bằng (tỷ lệ {imbalance_ratio:.1f}x).")
    if o_count / total_tags > 0.9:
        issues.append(f"Tỷ lệ tag 'O' quá cao ({o_count/total_tags*100:.1f}%), nhiều token không mang thông tin entity.")
    if errors:
        issues.append(f"Có {len(errors)} mẫu lỗi tokens/tags không khớp.")
    
    if issues:
        print(f"\n  ⚠ CÁC VẤN ĐỀ CẦN LƯU Ý:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  ✓ Dữ liệu huấn luyện có chất lượng tốt!")

if __name__ == '__main__':
    main()
