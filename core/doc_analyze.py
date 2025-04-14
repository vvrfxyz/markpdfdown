from uuid import uuid4
import cv2
import os

from core.DocLayoutYOLO import DocLayoutYOLOModel
from core.dataset import Dataset

model_path = "/home/wenruifeng/.cache/modelscope/hub/models/opendatalab/PDF-Extract-Kit-1___0/models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"  # 例如 "models/yolov10l-doc.pt"


def doc_analyze(
    dataset: Dataset,  # 输入的 Dataset 对象，包含文档信息
    start_page_id=0,  # 要分析的起始页面索引（从0开始），默认为0
    end_page_id=None,  # 要分析的结束页面索引（包含），None表示到最后一页，默认为None
):
    end_page_id = (
        end_page_id
        if end_page_id is not None and end_page_id >= 0
        else len(dataset) - 1
    )

    images = []  # 存储需要处理的页面图像
    page_wh_list = []  # 存储对应页面的宽度和高度

    # 遍历数据集中的所有页面
    for index in range(len(dataset)):
        # 只处理指定范围内的页面
        if start_page_id <= index <= end_page_id:
            page_data = dataset.get_page(index)  # 获取页面数据
            img_dict = page_data.get_image()  # 获取页面图像及其信息
            images.append(img_dict['img'])  # 添加图像到列表
            page_wh_list.append(
                (img_dict['width'], img_dict['height']))  # 添加页面宽高到列表

    layout_images = []
    # 初始化存储布局结果的列表
    images_layout_res = []
    model_instance = DocLayoutYOLOModel(model_path, 'cuda')
    # 准备用于布局分析的图像列表
    for image_index, image in enumerate(images):
        layout_images.append(image)
    images_layout_res += model_instance.batch_predict(
        layout_images, 1
    )
    # --- 开始：图片切片功能新增 ---
    output_dir = "output/photo/"
    try:
        os.makedirs(output_dir, exist_ok=True)
        # print(f"创建或确认输出目录: {output_dir}") # 用于调试
    except OSError as e:
        print(f"错误：无法创建目录 {output_dir}: {e}")
        # 根据需要决定是抛出异常还是继续但不保存切片
        output_dir = None  # 设置为None，后续跳过保存步骤
    # --- 结束：图片切片功能新增 ---
    if output_dir:  # 仅在目录创建成功时执行
        print(f"开始保存切片到 {output_dir}...")  # 用于调试
        slice_count = 0
        for image_idx, layout_res in enumerate(images_layout_res):
            original_image = images[image_idx]  # 获取对应的原始图像 (NumPy Array)
            # 检查原始图像是否有效
            if original_image is None or original_image.size == 0:
                print(f"警告：跳过图像 {image_idx} 的切片，因为原始图像无效。")
                continue

            image_height, image_width = original_image.shape[:2]

            for item_idx, item in enumerate(layout_res):
                # 检查 'poly' 键是否存在且不为 None
                if 'poly' in item and item['poly'] is not None:
                    layout_res[item_idx]['text'] = '12345，上山打老虎'
                    try:
                        category_id = item.get('category_id', 'unknown')
                        if category_id != 3 and category_id != 5:
                            continue
                        poly = item['poly']
                        # 确保 poly 是一个列表并且包含足够的元素来提取坐标
                        # (至少需要索引 0, 1, 4, 5，所以长度至少为6)
                        if isinstance(poly, list) and len(poly) >= 6:
                            # 从 'poly' 列表中提取边界框坐标
                            # poly format: [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                            x1 = int(poly[0])  # xmin
                            y1 = int(poly[1])  # ymin
                            # xmax (from bottom-right or top-right corner)
                            x2 = int(poly[4])
                            # ymax (from bottom-right corner)
                            y2 = int(poly[5])
                        else:
                            print(
                                f"警告：跳过图像 {image_idx} 项目 {item_idx} - 'poly' 格式无效或长度不足: {poly}")
                            continue  # 跳过这个无效的项目

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(image_width, x2)
                        y2 = min(image_height, y2)

                        # 检查修正后的框是否有效 (宽度和高度大于0)
                        if x1 >= x2 or y1 >= y2:
                            # print(f"警告：跳过图像 {image_idx} 项目 {item_idx} 的无效边界框 (修正后): [{x1},{y1},{x2},{y2}], 原始poly: {poly}")
                            continue  # 跳过无效或零尺寸的框

                        # 裁剪图像
                        # NumPy 索引是 [y1:y2, x1:x2]
                        slice_img = original_image[y1:y2, x1:x2]

                        # 检查裁剪结果是否为空
                        if slice_img is None or slice_img.size == 0:
                            # print(f"警告：图像 {image_idx} 项目 {item_idx} 裁剪结果为空, 原始poly: {poly}, 修正后: [{x1},{y1},{x2},{y2}]")
                            continue

                        # 构建文件名,随机生成uuid
                        uuid = str(uuid4())
                        filename = f"{uuid}.png"
                        output_path = os.path.join(output_dir, filename)
                        images_layout_res[image_idx]['media'] = 'photo/' + filename
                        # 保存图像
                        # 假设 original_image 是 RGB 格式，转换为 BGR 进行保存
                        # 如果 original_image 已经是 BGR，则移除 cvtColor
                        save_success = cv2.imwrite(
                            output_path, cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR))
                        # save_success = cv2.imwrite(output_path, slice_img) # 如果原始图像是BGR

                        if not save_success:
                            print(f"错误：无法将切片保存到 {output_path}")
                        else:
                            slice_count += 1
                            # print(f"成功保存切片: {output_path}") # 用于详细调试

                    except (ValueError, TypeError) as ve:
                        print(
                            f"错误：处理图像 {image_idx} 项目 {item_idx} 时坐标转换失败: {ve}")
                        print(f"  项目数据: {item}")
                    except Exception as e:
                        print(
                            f"错误：处理图像 {image_idx} 项目 {item_idx} 时发生意外异常: {e}")
                        print(f"  项目数据: {item}")
                # else: # 可以选择性地记录那些没有 'poly' 键的项目用于调试
                #     print(f"信息：跳过图像 {image_idx} 项目 {item_idx}，无 'poly' 键。 Item: {item}")
        print(f"完成切片保存，共保存 {slice_count} 个切片。")  # 用于调试
    # --- 结束：图片切片保存 ---
    print(f"images_layout_res: {images_layout_res}")
