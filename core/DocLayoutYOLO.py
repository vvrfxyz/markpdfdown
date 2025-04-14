import threading  # 导入线程锁，确保线程安全
from doclayout_yolo import YOLOv10  # 假设这个导入是有效的
from tqdm import tqdm
import numpy as np  # 导入numpy用于创建测试图像

# 定义一个名为 DocLayoutYOLOModel 的类，用于文档布局检测。
# 使用单例模式确保只有一个模型实例。


class DocLayoutYOLOModel(object):
    _instance = None  # 类变量，用于存储唯一的实例
    _lock = threading.Lock()  # 增加线程锁，确保在多线程环境下的单例安全
    _initialized = False  # 实例级别的初始化标志

    # __new__ 控制实例的创建
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # 如果实例还不存在
            with cls._lock:  # 获取锁
                # 双重检查，防止多个线程同时通过第一个检查点
                if cls._instance is None:
                    print("Creating new Singleton instance...")
                    # 调用父类的 __new__ 来实际创建实例
                    cls._instance = super().__new__(cls)
        else:
            print("Returning existing Singleton instance...")
        # 始终返回存储的实例
        return cls._instance

    # __init__ 控制实例的初始化
    def __init__(self, model_path=None, device=None):
        """
        初始化 DocLayoutYOLOModel 实例。
        由于是单例模式，此方法中的初始化逻辑仅在第一次创建实例时执行。
        后续调用将直接返回已初始化的实例，并忽略传入的参数。

        Args:
            model_path (str, optional): 模型权重文件的路径。仅在首次创建时需要。
            device (str, optional): 运行模型的设备 ('cpu', 'cuda:0', etc.)。仅在首次创建时需要。
        """
        if self._initialized:  # 如果实例已经初始化过
            print(
                f"Singleton already initialized. Ignoring parameters: model_path={model_path}, device={device}")
            return  # 直接返回，不再执行初始化

        # 只有在第一次创建实例时，才执行这里的初始化逻辑
        with self._lock:  # 确保初始化过程也是线程安全的
            # 再次检查初始化状态，因为可能有线程在等待锁时，另一个线程已经完成了初始化
            if self._initialized:
                print("Initialization already completed by another thread.")
                return

            print(
                f"Initializing Singleton instance with model: {model_path}, device: {device}")
            if model_path is None or device is None:
                raise ValueError(
                    "model_path and device must be provided for the first initialization.")

            # --- 原有的初始化逻辑 ---
            # 使用给定的权重文件初始化 YOLOv10 模型。
            try:
                self.model = YOLOv10(model_path)
            except NameError:
                # 如果 YOLOv10 库或模型文件不存在，提供一个模拟对象以便测试
                print(
                    "Warning: YOLOv10 class not found or model path invalid. Using a mock model for testing.")

                class MockModel:
                    def predict(self, *args, **kwargs):
                        print("MockModel predict called with args:",
                              args, "kwargs:", kwargs)
                        # 返回一个模拟的 YOLO 结果结构

                        class MockBoxes:
                            def __init__(self):
                                # 模拟检测到一个框
                                self.xyxy = np.array([[10, 10, 100, 100]])
                                self.conf = np.array([0.95])
                                self.cls = np.array([0])  # 类别 0

                            def cpu(self):  # 模拟 .cpu() 调用
                                return self

                        class MockResult:
                            def __init__(self):
                                self.boxes = MockBoxes()

                            def cpu(self):  # 模拟 .cpu() 调用 (用于批量预测)
                                return self

                        # predict 返回列表，模拟单个结果
                        # batch_predict 返回列表，这里也模拟返回列表，但需要调用者处理
                        # 根据调用 predict 还是 batch_predict，返回结构可能需要调整
                        # 这里简单返回一个适合单张图预测的模拟结果列表
                        if isinstance(args[0], list):  # 假设 batch_predict 第一个参数是列表
                            return [MockResult() for _ in args[0]]
                        else:
                            return [MockResult()]

                self.model = MockModel()
            except Exception as e:
                print(f"Error initializing YOLOv10 model: {e}")
                raise  # 重新抛出异常，初始化失败

            # 存储指定的运行设备（如 'cpu' 或 'cuda:0'）。
            self.device = device
            # --- 结束原有初始化逻辑 ---

            # 标记为已初始化
            self._initialized = True
            print("Singleton initialization complete.")

    # 定义预测方法，用于对单个图像进行布局检测。

    def predict(self, image):
        # 检查实例是否已初始化
        if not self._initialized:
            raise RuntimeError(
                "Singleton instance has not been initialized. Call with parameters first.")

        # 初始化一个空列表，用于存储布局检测结果。
        layout_res = []
        # 调用 YOLOv10 模型的 predict 方法进行目标检测。
        # [0]: YOLOv10 predict 返回一个列表，通常只包含一个结果对象（针对单张图），所以取第一个元素。
        print(f"Predicting on device: {self.device}")  # 添加打印信息
        doclayout_yolo_res = self.model.predict(
            image,
            imgsz=1280,  # 推理图像大小设置为1280
            conf=0.10,  # 置信度阈值设为0.10
            iou=0.45,  # NMS的IoU阈值设为0.45
            verbose=False,  # 修改为False，减少冗余输出
            device=self.device  # 指定运行设备
        )[0]  # 获取第一个结果，因为是单张图片预测

        # 确保结果在 CPU 上进行后续处理
        boxes = doclayout_yolo_res.boxes
        xyxy_cpu = boxes.xyxy.cpu() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
        conf_cpu = boxes.conf.cpu() if hasattr(boxes.conf, 'cpu') else boxes.conf
        cls_cpu = boxes.cls.cpu() if hasattr(boxes.cls, 'cpu') else boxes.cls

        # 遍历检测结果中的边界框坐标、置信度和类别。
        for xyxy, conf, cla in zip(xyxy_cpu, conf_cpu, cls_cpu):
            # 将边界框坐标从张量（tensor）转换为整数。
            xmin, ymin, xmax, ymax = [int(p.item()) if hasattr(
                p, 'item') else int(p) for p in xyxy]  # 提取并转换坐标为整数
            # 创建一个新的字典来存储格式化的检测结果。
            new_item = {
                # "category_id": 存储检测到的类别ID（整数）。
                # 存储类别ID
                "category_id": int(cla.item()) if hasattr(cla, 'item') else int(cla),
                # "poly": 存储表示边界框的多边形坐标 [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]。
                # 存储矩形框的四个顶点坐标
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                # "score": 存储检测结果的置信度分数，保留3位小数。
                # 存储置信度分数，四舍五入保留3位小数
                "score": round(float(conf.item()) if hasattr(conf, 'item') else float(conf), 3),
            }
            # 将格式化后的结果添加到 layout_res 列表中。
            layout_res.append(new_item)
        # 返回包含所有检测结果的列表。
        return layout_res

    # 定义批量预测方法，用于同时处理多张图像。
    def batch_predict(self, images: list, batch_size: int) -> list:
        # 检查实例是否已初始化
        if not self._initialized:
            raise RuntimeError(
                "Singleton instance has not been initialized. Call with parameters first.")

        # 初始化一个空列表，用于存储所有图像的布局检测结果。
        images_layout_res = []
        print(f"Batch predicting on device: {self.device}")  # 添加打印信息
        # 使用 tqdm 创建一个进度条，按指定的 batch_size 遍历图像列表。
        for index in tqdm(range(0, len(images), batch_size), desc="Layout Predict"):  # 使用tqdm显示批处理进度
            # 获取当前批次的图像
            batch_images = images[index: index + batch_size]
            # 调用 YOLOv10 模型的 predict 方法对当前批次的图像进行推理。
            doclayout_yolo_batch_res = self.model.predict(
                batch_images,  # 输入当前批次的图像。
                imgsz=1280,  # 推理图像大小
                conf=0.10,  # 置信度阈值
                iou=0.45,  # IoU 阈值
                verbose=False,  # 修改为False
                device=self.device,  # 指定运行设备
            )

            # 处理每个图像的预测结果
            for image_res in doclayout_yolo_batch_res:
                # 为当前图像初始化一个空列表，用于存储其布局检测结果。
                layout_res = []
                # 将结果转移到 CPU (如果需要且支持)
                boxes = image_res.boxes
                xyxy_cpu = boxes.xyxy.cpu() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                conf_cpu = boxes.conf.cpu() if hasattr(boxes.conf, 'cpu') else boxes.conf
                cls_cpu = boxes.cls.cpu() if hasattr(boxes.cls, 'cpu') else boxes.cls

                # 遍历当前图像检测到的所有边界框、置信度和类别。
                for xyxy, conf, cla in zip(xyxy_cpu, conf_cpu, cls_cpu):
                    # 将边界框坐标从张量转换为整数。
                    xmin, ymin, xmax, ymax = [int(p.item()) if hasattr(
                        p, 'item') else int(p) for p in xyxy]  # 提取并转换坐标为整数
                    # 创建格式化的结果字典。
                    new_item = {
                        # 存储类别ID
                        "category_id": int(cla.item()) if hasattr(cla, 'item') else int(cla),
                        # 存储矩形框的四个顶点坐标
                        "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                        # 存储置信度分数，保留3位小数
                        "score": round(float(conf.item()) if hasattr(conf, 'item') else float(conf), 3),
                    }
                    # 将当前检测结果添加到该图像的结果列表 layout_res 中。
                    layout_res.append(new_item)
                # 将当前图像的所有检测结果列表 layout_res 添加到最终的结果列表 images_layout_res 中。
                images_layout_res.append(layout_res)
        # 返回包含所有图像检测结果的列表（列表的列表）。
        return images_layout_res


# --- Main 测试方法 ---
if __name__ == "__main__":
    print("--- Testing Singleton DocLayoutYOLOModel ---")

    # 定义模型路径和设备 (请替换为你的实际路径)
    # !! 重要 !!: 替换为你的模型文件路径
    model_path = "/home/wenruifeng/.cache/modelscope/hub/models/opendatalab/PDF-Extract-Kit-1___0/models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"  # 例如 "models/yolov10l-doc.pt"
    # 如果没有 GPU 或不想使用，设为 "cpu"
    device = "cuda"  # 或者 "cuda:0" 如果你有兼容的 GPU 和 CUDA 环境

    # 1. 第一次获取实例 (会进行初始化)
    print("\nAttempting to get instance 1...")
    try:
        instance1 = DocLayoutYOLOModel(model_path=model_path, device=device)
        print(f"Instance 1 obtained. ID: {id(instance1)}")
        print(f"Instance 1 initialized: {instance1._initialized}")
        print(f"Instance 1 device: {instance1.device}")
    except ValueError as e:
        print(f"Error during first initialization: {e}")
        # 如果初始化失败，后续测试可能无法进行
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during first initialization: {e}")
        exit()

    # 2. 第二次获取实例 (应该返回同一个实例，忽略参数)
    print("\nAttempting to get instance 2 with different parameters...")
    instance2 = DocLayoutYOLOModel(
        model_path="another/path.pt", device="cuda:1")
    print(f"Instance 2 obtained. ID: {id(instance2)}")
    print(f"Instance 2 initialized: {instance2._initialized}")
    print(f"Instance 2 device: {instance2.device}")  # 应该仍然是第一次设置的 device

    # 3. 验证是否为同一个实例
    print(
        f"\nIs instance 1 the same object as instance 2? {instance1 is instance2}")

    # 4. 测试 predict 方法
    print("\n--- Testing predict method ---")
    # 创建一个假的输入图像 (例如，一个 640x640 的黑色图像)
    # 注意：YOLOv10 可能期望特定大小或格式，这里的假图像仅用于测试流程
    dummy_image_single = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        print("Calling predict on instance 1...")
        results_single = instance1.predict(dummy_image_single)
        print(f"Predict results (single image): {results_single}")
    except RuntimeError as e:
        print(f"Error calling predict: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during predict: {e}")

    # 5. 测试 batch_predict 方法
    print("\n--- Testing batch_predict method ---")
    # 创建一批假的输入图像
    dummy_image_batch = [
        np.zeros((720, 720, 3), dtype=np.uint8),
        np.ones((800, 600, 3), dtype=np.uint8) * 255,
        np.random.randint(0, 256, size=(500, 500, 3), dtype=np.uint8)
    ]
    batch_size = 2
    try:
        print(
            f"Calling batch_predict on instance 2 with batch size {batch_size}...")
        results_batch = instance2.batch_predict(
            dummy_image_batch, batch_size=batch_size)
        print(
            f"Batch predict results (number of images processed): {len(results_batch)}")
        for i, res in enumerate(results_batch):
            print(f"  Results for image {i}: {res}")
    except RuntimeError as e:
        print(f"Error calling batch_predict: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during batch_predict: {e}")

    print("\n--- Singleton Test Complete ---")
