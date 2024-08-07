'''Inference module for the server'''
import os
import base64
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as rt
import pandas as pd
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from utils.mardi import load_image, filter_out_background, generate_colors, is_white_background, resize_image
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.augmentations import letterbox
from utils.torch_utils import select_device, smart_inference_mode
from database import SessionLocal
from schemas import InferenceUpdate
from crud import update_inference

#os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:128' 
#torch.cuda.set_per_process_memory_fraction(0.8, 0)

class Inference():
    def __init__(self, yolo_classification_weights: str, yolo_detection_weights: str, sam_checkpoint: str, classification1_model_path: str, classification2_model_path: str):
        self.db = SessionLocal()
        self.device = select_device('0')

        # Age Group classification model
        self.age_group_model = DetectMultiBackend(yolo_classification_weights, device=self.device, dnn=False, data='data/coco128.yaml', fp16=False)
        self.age_group_stride = self.age_group_model.stride
        age_group_imgsz = check_img_size((224, 224), s=self.age_group_stride)  # check image size
        self.age_group_model.warmup(imgsz=(1, 3, *age_group_imgsz))


        # Object Detection model
        self.object_detection_model = DetectMultiBackend(yolo_detection_weights, device=self.device, dnn=False, data='data/coco.yaml', fp16=False)
        self.object_detection_stride, object_detection_pt = self.object_detection_model.stride, self.object_detection_model.pt
        object_detection_imgsz = check_img_size((640, 640), s=self.object_detection_stride)  # check image size
        self.object_detection_model.warmup(imgsz=(1 if object_detection_pt or self.object_detection_model.triton else 1, 3, *object_detection_imgsz))  # warmup

        # SAM model
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=self.device, non_blocking=True)
        self.sam_predictor_model = SamPredictor(sam)

        # Mask Generator
        self.mask_generator = SamAutomaticMaskGenerator(sam)

        # Classification Group 1 model
        self.classification1_model = rt.InferenceSession(classification1_model_path, providers=["CPUExecutionProvider"])
        self.classification1_input_name = self.classification1_model.get_inputs()[0].name
        self.classification1_label_name = self.classification1_model.get_outputs()[0].name

        # Classification Group 2 model
        self.classification2_model = rt.InferenceSession(classification2_model_path, providers=["CPUExecutionProvider"])
        self.classification2_input_name = self.classification2_model.get_inputs()[0].name
        self.classification2_label_name = self.classification2_model.get_outputs()[0].name

    def preprocess_image_yolo_classification(self, image: np.ndarray) -> np.ndarray:
        im = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.age_group_model.device)
        im = im.half() if self.object_detection_model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    def preprocess_image_object_detection(self, image: np.ndarray) -> np.ndarray:
        im = letterbox(image, 640, stride=self.object_detection_stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.object_detection_model.device)
        im = im.half() if self.object_detection_model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    def update_progress(self, inference_id: str, status: str, progress: float, age: str = None, image_result: str = None):
        update_data = InferenceUpdate(
            status=status,
            progress=progress
            )
        if age:
            update_data.age = age
        if image_result:
            update_data.image_result = image_result
        return update_inference(self.db, inference_id, update_data)
    
    def detect_tree_area(self, inference_id: str, image: np.ndarray):
        # Define the desired classes
        desired_classes = [25, 58]

        conf_thres = 0.1
        iou_thres = 0.45
        classes = None
        agnostic_nms = False
        max_det = 1000

        # Find the most central bounding box if there are multiple detections
        def find_central_bbox(bboxes):
            center_x, center_y = image_width / 2, image_height / 2
            min_distance = float('inf')
            central_bbox = None
            central_class_id = None
            central_conf_score = None

            for class_id, bbox, conf_score in zip(class_ids, bboxes, conf_scores):
                if class_id in desired_classes:
                    bbox_center_x = (bbox[0] + bbox[2]) / 2
                    bbox_center_y = (bbox[1] + bbox[3]) / 2
                    distance = np.sqrt((center_x - bbox_center_x) ** 2 + (center_y - bbox_center_y) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        central_bbox = bbox
                        central_class_id = class_id
                        central_conf_score = conf_score

            return central_class_id, central_bbox, central_conf_score

        self.update_progress(inference_id, "detect_tree_area", 25)
        im = self.preprocess_image_object_detection(image)
        pred = self.object_detection_model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        image_height, image_width, _ = image.shape
        gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        aggregate_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        self.update_progress(inference_id, "generating_tree_mask", 30)
        class_ids = []
        bboxes = []
        conf_scores = []

        # Process predictions
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to image size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if not cls in desired_classes:
                        continue

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    cx, cy, w, h = xywh

                    # Convert from normalized [0, 1] to image scale
                    cx *= image_width
                    cy *= image_height
                    w *= image_width
                    h *= image_height

                    # Convert center x, y, width and height to xmin, ymin, xmax, ymax
                    xmin = cx - w / 2
                    ymin = cy - h / 2
                    xmax = cx + w / 2
                    ymax = cy + h / 2

                    class_ids.append(cls)
                    bboxes.append((xmin, ymin, xmax, ymax))
                    conf_scores.append(conf)


        # Get the most central bounding box
        central_class_id, central_bbox, _ = find_central_bbox(bboxes)

        if central_class_id is not None and central_bbox is not None:
            # Generate and accumulate masks for each bounding box
            self.sam_predictor_model.set_image(image)
            input_box = np.array(central_bbox).reshape(1, 4)
            masks, _, _ = self.sam_predictor_model.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
            )
            aggregate_mask = np.where(masks[0] > 0.5, 1, aggregate_mask)

            # Convert the aggregated segmentation mask to a binary mask
            binary_mask = np.where(aggregate_mask == 1, 1, 0)

            # Create a white background with the same size as the image
            white_background = np.ones_like(image) * 255

            # Applying the binary mask to the original image
            # Where the binary mask is 0 (background), use white background; otherwise, use the original image.
            segmented_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]

            return segmented_image.astype(np.uint8)
        
        return image

    @smart_inference_mode()
    def predict(self, image_url: str, inference_id: str):
        self.update_progress(inference_id, "loading_image", 0)

        original_image = load_image(image_url)
        image = cv2.resize(original_image, (224, 224))

        self.update_progress(inference_id, "classify_plant_age_group", 10)
        im = self.preprocess_image_yolo_classification(image)
        results = self.age_group_model(im)
        top_class = results.argmax(dim=1).item()
        age_group = self.age_group_model.names[top_class]

        print(f"Age group: {age_group}")
        if age_group == "Class1":
            image = self.detect_tree_area(inference_id, image)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Free up GPU memory before processing each image
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.update_progress(inference_id, "generate_leaf_masks", 60)

        sam_result = self.mask_generator.generate(image_rgb)

        # Filter out the background
        filtered_masks = filter_out_background(sam_result)

        annotated_image = image_rgb.copy()
        colors = generate_colors(len(filtered_masks))

        self.update_progress(inference_id, "classifying", 90)
        segment_count = 0
        mask_data = []
        for i, mask in enumerate(filtered_masks):
            area = mask['area']
            segmentation = mask['segmentation'].astype('uint8')

            # Skip white background areas
            if is_white_background(segmentation, image_rgb):
                continue

            # Color the segmented area
            color = colors[i]
            r, g, b = color
            annotated_image[segmentation > 0] = cv2.addWeighted(annotated_image, 0.5, np.full_like(annotated_image, color), 0.5, 0)[segmentation > 0]
            segment_count += 1
            mask_data.append([area, r, g, b])

        columns = ['area', 'r', 'g', 'b']
        data = pd.DataFrame(mask_data, columns=columns)

        # Create derived features, handling potential division by zero
        data['r/g'] = (data['r'] / (data['g'] + 1e-8)).round(4)  # Add a small constant to avoid division by zero and round to 4 decimals
        data['r/b'] = (data['r'] / (data['b'] + 1e-8)).round(4)
        data['g/b'] = (data['g'] / (data['b'] + 1e-8)).round(4)

        # Calculate aggregated statistics for 'area'
        agg_area = data['area'].agg(['mean', 'median', 'std']).reset_index()
        agg_area.columns = ['statistic', 'value']

        # Round aggregated statistics to 4 decimal places
        area_mean = agg_area.loc[agg_area['statistic'] == 'mean', 'value'].round(4).values[0]
        area_median = agg_area.loc[agg_area['statistic'] == 'median', 'value'].round(4).values[0]
        area_std = agg_area.loc[agg_area['statistic'] == 'std', 'value'].round(4).values[0]

        classification_input = np.array([[segment_count, area_std, area_mean, area_median]]).astype(np.float32)

        if age_group == "Class1":
            age = self.classification1_model.run([self.classification1_label_name], {self.classification1_input_name: classification_input})[0]
        else:
            age = self.classification2_model.run([self.classification2_label_name], {self.classification2_input_name: classification_input})[0]

        if len(age) > 0:
            age = age[0]
        else:
            age = None
        
        print(f"Segment count: {segment_count}")
        print(f"Age: {age}")
        
        # Encode the image to base64
        annotated_image = resize_image(annotated_image, max_size=(256, 256))
        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return self.update_progress(inference_id, "completed", 100, age, image_base64)
