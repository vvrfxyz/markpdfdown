# utils.py

from typing import List, Dict, Any
import traceback  # 用于打印更详细的错误信息


def _get_sort_key(block: Dict[str, Any], tolerance: int = 5) -> tuple:
    """
    为文本块生成排序键 (ymin, xmin)。
    增加了对 'poly' 键和格式的检查。
    增加了 Y 轴容差，将 Y 坐标相近的块视为同一行。

    Args:
        block: 包含文本块信息的字典，应有 'poly' 键。
        tolerance: Y 坐标的容差值。在此范围内视为同一行。

    Returns:
        一个元组 (y_group, x_min) 用于排序，如果块无效则返回一个表示无穷大的元组。
        y_group 是基于容差分组后的 Y 坐标。
    """
    if not isinstance(block, dict):
        # print(f"警告：输入块不是字典: {block}")
        return (float('inf'), float('inf'))  # 无效块排在最后

    poly = block.get('poly')
    if not isinstance(poly, list) or len(poly) < 2:
        # print(f"警告：块缺少 'poly' 键或格式无效: {block}")
        return (float('inf'), float('inf'))  # 无效块排在最后

    try:
        # poly format: [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
        y_min = float(poly[1])
        x_min = float(poly[0])

        # 根据容差对 y_min 进行分组，使得 Y 坐标相近的块获得相同的 y_group 值
        y_group = int(y_min // tolerance) * tolerance

        return (y_group, x_min)
    except (TypeError, ValueError, IndexError) as e:
        # print(f"警告：处理 poly 时出错: {e}, block: {block}")
        traceback.print_exc()  # 打印详细错误
        return (float('inf'), float('inf'))  # 出错的块排在最后


def concat_text_blocks_ordered(layout_results: List[List[Dict[str, Any]]], y_tolerance: int = 5) -> str:
    """
    将多页布局分析结果中的文本块按推断的阅读顺序 (从上到下, 从左到右) 拼接起来。
    使用块的左上角坐标 ('poly'[1], 'poly'[0]) 进行排序，并加入了Y轴容差。

    同一页内的文本块默认用单个换行符分隔。
    不同页面的内容用两个换行符分隔。

    Args:
        layout_results: 一个列表，其中每个元素是代表一页结果的列表。
                        每页结果列表包含多个字典，每个字典代表一个文本块，
                        且应包含 'text' 和 'poly' 键。
        y_tolerance: Y 坐标排序时的容差。Y坐标差值小于此值的块被视为在同一行进行左右排序。
                     可以根据字体大小或行高来调整，默认值为 5 像素。

    Returns:
        一个包含所有按顺序拼接文本的字符串。
    """
    all_pages_content = []

    if not layout_results:
        return ""  # 如果输入为空，返回空字符串

    for page_index, page_blocks in enumerate(layout_results):
        current_page_texts = []
        if not isinstance(page_blocks, list):
            print(f"警告：页面索引 {page_index} 的数据不是列表，已跳过。数据: {page_blocks}")
            continue  # 跳过格式不正确的页面数据

        # --- 关键步骤：对当前页面的块进行排序 ---
        try:
            # 使用 _get_sort_key 函数生成排序键 (y_group, x_min)
            # 无效块会被排到最后
            sorted_page_blocks = sorted(
                page_blocks, key=lambda block: _get_sort_key(block, y_tolerance))
        except Exception as e:
            print(f"错误：在对页面 {page_index} 的块进行排序时发生异常: {e}")
            traceback.print_exc()
            # 如果排序失败，可以选择跳过此页或使用原始顺序（这里选择跳过）
            continue

        # --- 遍历排序后的块，提取文本 ---
        for block_index, block in enumerate(sorted_page_blocks):
            # 再次检查 block 是否是有效字典
            if not isinstance(block, dict):
                # print(f"调试：跳过页面 {page_index} 排序后的无效块 {block_index}: {block}")
                continue

            # 检查 'text' 键
            text = block.get('text')
            if text and isinstance(text, str):
                current_page_texts.append(text.strip())  # 去除首尾空格
            # else: # 如果需要记录没有文本的块
            #     poly_info = block.get('poly', 'N/A')
            #     print(f"调试：页面 {page_index}, 排序后块 {block_index} (poly: {poly_info}) 的文本为空或非字符串。")

        # 将当前页面的所有文本块用单个换行符连接
        if current_page_texts:  # 只有当页面有内容时才添加
            all_pages_content.append("\n".join(current_page_texts))

    # 将所有页面的内容用两个换行符连接
    return "\n\n".join(all_pages_content)


# --- 示例用法 ---
if __name__ == '__main__':
    # 使用你提供的真实数据
    sample_data = [
        [{'category_id': 1, 'poly': [108, 896, 1331, 896, 1331, 1394, 108, 1394], 'score': 0.98, 'text': 'Page1 BlockA (y=896)'},
         {'category_id': 1, 'poly': [850, 1620, 1561, 1620, 1561, 1886, 850,
                                     1886], 'score': 0.976, 'text': 'Page1 BlockD (y=1620, x=850)'},
         {'category_id': 1, 'poly': [110, 1470, 820, 1470, 820, 1885, 110, 1885],
             # y=1470, x=110
             'score': 0.976, 'text': 'Page1 BlockC (y=1470, x=110)'},
         {'category_id': 1, 'poly': [851, 1470, 1562, 1470, 1562, 1618, 851, 1618],
             # y=1470, x=851
             'score': 0.973, 'text': 'Page1 BlockE (y=1470, x=851)'},
         {'category_id': 0, 'poly': [108, 310, 1559, 310, 1559, 623,
                                     108, 623], 'score': 0.971, 'text': 'Page1 BlockG (y=310)'},
         {'category_id': 2, 'poly': [110, 1947, 1114, 1947, 1114, 2063,
                                     110, 2063], 'score': 0.968, 'text': 'Page1 BlockF (y=1947)'},
         {'category_id': 1, 'poly': [109, 723, 1287, 723, 1287, 827,
                                     109, 827], 'score': 0.957, 'text': 'Page1 BlockB (y=723)'},
         {'category_id': 2, 'poly': [109, 2117, 466, 2117, 466, 2145,
                                     109, 2145], 'score': 0.9, 'text': 'Page1 BlockH (y=2117)'},
         {'category_id': 2, 'poly': [
             108, 47, 614, 47, 614, 102, 108, 102], 'score': 0.878, 'text': 'Page1 BlockI (y=47)'},
         {'category_id': 2, 'poly': [1114, 265, 1560, 265, 1560, 291, 1114, 291], 'score': 0.847,
             # 这个应该在 BlockG (y=310) 前面，但在 y=122 和 y=47 的块后面
             'text': 'Page1 BlockJ (y=265, x=1114)'},
         {'category_id': 2, 'poly': [106, 122, 803, 122, 803, 147,
                                     106, 147], 'score': 0.847, 'text': 'Page1 BlockK (y=122)'},
         {'category_id': 2, 'poly': [1434, 55, 1561, 55, 1561, 93, 1434, 93], 'score': 0.815,
             # y=55, x=1434. 在 y=47 后面，但在 y=122 前面
             'text': 'Page1 BlockL (y=55, x=1434)'},
         {'category_id': 2, 'poly': [1334, 679, 1562, 679, 1562, 709, 1334, 709], 'score': 0.802,
             # 在 BlockG (y=310) 后面，在 BlockB (y=723) 前面
             'text': 'Page1 BlockM (y=679)'},
         {'category_id': 2, 'poly': [1545, 2119, 1560, 2119, 1560, 2140, 1545, 2140],
             # 在 BlockH (y=2117) 附近
             'score': 0.741, 'text': 'Page1 BlockN (y=2119)'},
         {'category_id': 2, 'poly': [1175, 2001, 1561, 2001, 1561, 2061, 1175, 2061],
             # 在 BlockF (y=1947) 后面
             'score': 0.656, 'text': 'Page1 BlockO (y=2001)'},
         {'category_id': 2, 'poly': [1510, 121, 1559, 121, 1559, 193, 1510, 193], 'score': 0.563,
             # y=121 (接近122), x=1510. 应该在 BlockK (y=122, x=106) 后面
             'text': 'Page1 BlockP (y=121, x=1510)'},
         # y=51 (接近55), x=1428. 应该在 BlockI (y=47) 后面, 在 BlockL (y=55) 前面
         {'category_id': 2, 'poly': [1428, 51, 1561, 51, 1561, 194, 1428,
                                     194], 'score': 0.141, 'text': 'Page1 BlockQ (y=51, x=1428)'},
         # 在 BlockE (y=1470) 后面，BlockF (y=1947) 前面
         {'category_id': 2, 'poly': [78, 1942, 1553, 1942, 1553, 2063, 78, 2063], 'score': 0.103, 'text': 'Page1 BlockR (y=1942)'}],
        [
            {'category_id': 1, 'poly': [850, 1383, 1561, 1383, 1561, 2062, 850,
                                        2062], 'score': 0.983, 'text': 'Page2 BlockS (y=1383, x=850)'},
            {'category_id': 1, 'poly': [850, 132, 1561, 132, 1561, 516, 850, 516],
                # y=132, x=850
                'score': 0.983, 'text': 'Page2 BlockT (y=132, x=850)'},
            {'category_id': 1, 'poly': [850, 637, 1561, 637, 1561, 1380, 850,
                                        1380], 'score': 0.983, 'text': 'Page2 BlockU (y=637, x=850)'},
            {'category_id': 1, 'poly': [111, 758, 821, 758, 821, 1348, 111, 1348],
                # y=758, x=111
                'score': 0.983, 'text': 'Page2 BlockV (y=758, x=111)'},
            {'category_id': 1, 'poly': [110, 1350, 821, 1350, 821, 1973, 110, 1973],
                # y=1350, x=110
                'score': 0.982, 'text': 'Page2 BlockW (y=1350, x=110)'},
            {'category_id': 1, 'poly': [111, 131, 820, 131, 820, 516, 111, 516],
                # y=131, x=111
                'score': 0.982, 'text': 'Page2 BlockX (y=131, x=111)'},
            {'category_id': 1, 'poly': [110, 519, 820, 519, 820, 756, 110, 756],
                # y=519, x=110
                'score': 0.966, 'text': 'Page2 BlockY (y=519, x=110)'},
            {'category_id': 1, 'poly': [108, 1978, 819, 1978, 819, 2064,
                                        108, 2064], 'score': 0.961, 'text': 'Page2 BlockZ (y=1978)'},
            {'category_id': 2, 'poly': [
                108, 69, 550, 69, 550, 93, 108, 93], 'score': 0.904, 'text': 'Page2 BlockAA (y=69)'},
            {'category_id': 2, 'poly': [109, 2118, 464, 2118, 464, 2144,
                                        109, 2144], 'score': 0.9, 'text': 'Page2 BlockBB (y=2118)'},
            {'category_id': 2, 'poly': [
                1474, 65, 1559, 65, 1559, 91, 1474, 91], 'score': 0.846, 'text': 'Page2 BlockCC (y=65)'},
            {'category_id': 2, 'poly': [1545, 2120, 1561, 2120, 1561, 2140,
                                        1545, 2140], 'score': 0.772, 'text': 'Page2 BlockDD (y=2120)'},
            {'category_id': 0, 'poly': [848, 546, 1505, 546, 1505, 635, 848, 635],
                # y=546, x=848
                'score': 0.742, 'text': 'Page2 BlockEE (y=546, x=848)'},
            {'category_id': 0, 'poly': [850, 547, 951, 547, 951, 575, 850, 575], 'score': 0.599,
                # y=547 (接近546), x=850. 应该在 BlockEE 附近
                'text': 'Page2 BlockFF (y=547, x=850)'},
            {'category_id': 0, 'poly': [849, 578, 1508, 578, 1508, 634, 849, 634],
                # 在 BlockEE/FF 之后
                'score': 0.575, 'text': 'Page2 BlockGG (y=578, x=849)'},
            {'category_id': 2, 'poly': [1474, 66, 1560, 66, 1560, 91, 1474, 91],
                # y=66, 在 BlockCC (y=65) 附近
                'score': 0.271, 'text': 'Page2 BlockHH (y=66)'},
            {'category_id': 1, 'poly': [110, 519, 820, 519, 820, 756, 110, 756],
                # 重复 BlockY
                'score': 0.194, 'text': 'Page2 BlockY_dup (y=519, x=110)'}
        ],
        [
            {'category_id': 1, 'poly': [109, 1500, 820, 1500, 820, 1707,
                                        109, 1707], 'score': 0.981, 'text': 'Page3 BlockII (y=1500)'},
            {'category_id': 1, 'poly': [110, 1798, 819, 1798, 819, 2065,
                                        110, 2065], 'score': 0.98, 'text': 'Page3 BlockJJ (y=1798)'},
            {'category_id': 1, 'poly': [851, 1502, 1561, 1502, 1561, 2061, 851, 2061],
                # y=1502, x=851
                'score': 0.979, 'text': 'Page3 BlockKK (y=1502, x=851)'},
            {'category_id': 5, 'poly': [717, 623, 1554, 623, 1554, 870, 717, 870], 'score': 0.973,
                'text': 'Page3 BlockLL (y=623)', 'media': 'photo/7ac6116c-65c1-413a-81b8-1e04c67c3103.png'},
            {'category_id': 3, 'poly': [780, 938, 1466, 938, 1466, 1413, 780, 1413], 'score': 0.963,
                'text': 'Page3 BlockMM (y=938)', 'media': 'photo/90e1054d-6542-40b3-8bfc-5025e374847c.png'},
            {'category_id': 3, 'poly': [699, 141, 1562, 141, 1562, 590, 699, 590], 'score': 0.957,
                'text': 'Page3 BlockNN (y=141)', 'media': 'photo/83498b08-8ed6-4707-80d6-70e575571ed2.png'},
            {'category_id': 0, 'poly': [108, 1739, 799, 1739, 799, 1795,
                                        108, 1795], 'score': 0.95, 'text': 'Page3 BlockOO (y=1739)'},
            {'category_id': 1, 'poly': [
                110, 129, 558, 129, 558, 796, 110, 796], 'score': 0.941, 'text': 'Page3 BlockPP (y=129)'},
            {'category_id': 2, 'poly': [
                108, 69, 551, 69, 551, 93, 108, 93], 'score': 0.909, 'text': 'Page3 BlockQQ (y=69)'},
            {'category_id': 2, 'poly': [109, 2118, 464, 2118, 464, 2144,
                                        109, 2144], 'score': 0.9, 'text': 'Page3 BlockRR (y=2118)'},
            {'category_id': 2, 'poly': [1545, 2120, 1561, 2120, 1561, 2139,
                                        1545, 2139], 'score': 0.786, 'text': 'Page3 BlockSS (y=2120)'},
            {'category_id': 2, 'poly': [
                1474, 66, 1560, 66, 1560, 91, 1474, 91], 'score': 0.658, 'text': 'Page3 BlockTT (y=66)'},
            {'category_id': 2, 'poly': [591, 605, 619, 605, 619, 638, 591, 638], 'score': 0.593,
                # 在 BlockPP(y=129) 之后, BlockLL(y=623) 之前
                'text': 'Page3 BlockUU (y=605)'},
            {'category_id': 2, 'poly': [1474, 65, 1560, 65, 1560, 91, 1474, 91], 'score': 0.533,
                # 在 BlockQQ(y=69) 和 BlockTT(y=66) 之前
                'text': 'Page3 BlockVV (y=65)'},
            {'category_id': 2, 'poly': [590, 894, 619, 894, 619, 925, 590, 925], 'score': 0.481,
                # 在 BlockLL(y=623) 之后, BlockMM(y=938) 之前
                'text': 'Page3 BlockWW (y=894)'},
            {'category_id': 0, 'poly': [591, 129, 619, 129, 619, 160, 591, 160], 'score': 0.451,
                # y=129, x=591. 应该在 BlockPP (y=129, x=110) 之后
                'text': 'Page3 BlockXX (y=129, x=591)'},
            {'category_id': 1, 'poly': [590, 894, 619, 894, 619, 925, 590, 925],
                # 重复 BlockWW
                'score': 0.211, 'text': 'Page3 BlockWW_dup (y=894)'}
        ]
    ]  # 原始数据中的 'text' 都被替换成了包含页码和标识符的文本，以便于验证排序结果

    # 调用函数处理数据
    concatenated_text = concat_text_blocks_ordered(
        sample_data, y_tolerance=10)  # 使用 10 像素的 Y 轴容差

    # 打印结果
    print("--- 拼接后的文本 (按推断顺序) ---")
    print(concatenated_text)
    print("--- 结束 ---")

    # 测试空输入
    print("\n--- 测试空输入 ---")
    empty_result = concat_text_blocks_ordered([])
    print(f"空输入结果: '{empty_result}'")
    print("--- 结束 ---")

    # 测试包含无效数据的输入
    print("\n--- 测试包含无效数据 ---")
    invalid_data = [
        [
            {'text': 'Block 1', 'poly': [10, 10, 50, 10, 50, 30, 10, 30]},
            {'text': 'Block 2 (no poly)'},  # 缺少 poly
            {'text': 'Block 3', 'poly': [
                60, 10, 100, 10, 100, 30, 60, 30]},  # 同一行
            {'text': 'Block 4', 'poly': 'invalid poly'},  # poly 格式错误
            {'text': 'Block 5', 'poly': [10, 40, 50, 40, 50, 60, 10, 60]},
            None  # 整个块无效
        ]
    ]
    invalid_result = concat_text_blocks_ordered(invalid_data)
    print(invalid_result)
    print("--- 结束 ---")
