from core.DocLayoutYOLO import DocLayoutYOLOModel
from core.dataset import Dataset


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
            page_wh_list.append((img_dict['width'], img_dict['height']))  # 添加页面宽高到列表

    layout_images = []
    # 初始化存储布局结果的列表
    images_layout_res = []
    model = DocLayoutYOLOModel('/Users/vvrfxyz/Downloads/doclayout_yolo_docstructbench_imgsz1024.onnx','cpu')
    # 准备用于布局分析的图像列表
    for image_index, image in enumerate(images):
        layout_images.append(image)
    images_layout_res += model.batch_predict(
                layout_images, 1
            )
    print(1)