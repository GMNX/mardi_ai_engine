'''Inference module for the server'''
import os
import base64
import cv2
import torch
import numpy as np
import onnxruntime as rt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from utils.mardi import load_image, add_noise, filter_out_background, generate_colors, is_white_background, resize_image
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
    def __init__(self, yolo_weights: str, sam_checkpoint: str, classification_model_path: str):
        self.db = SessionLocal()

        # YOLOv9 model
        self.device = select_device('0')
        self.yolov9_model = DetectMultiBackend(yolo_weights, device=self.device, dnn=False, data='data/coco.yaml', fp16=False)
        self.stride, self.names, pt = self.yolov9_model.stride, self.yolov9_model.names, self.yolov9_model.pt
        self.imgsz = check_img_size((640, 640), s=self.stride)  # check image size
        self.yolov9_model.warmup(imgsz=(1 if pt or self.yolov9_model.triton else 1, 3, *self.imgsz))  # warmup

        # SAM model
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam_model = SamPredictor(sam)

        # Mask Generator
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=self.device, non_blocking=True)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

        # Classification model
        self.classification_model = rt.InferenceSession(classification_model_path, providers=["CPUExecutionProvider"])
        self.classification_input_name = self.classification_model.get_inputs()[0].name
        self.classification_label_name = self.classification_model.get_outputs()[0].name

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        im = letterbox(image, 640, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.yolov9_model.device)
        im = im.half() if self.yolov9_model.fp16 else im.float()  # uint8 to fp16/32
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

    @smart_inference_mode()
    def predict(self, image_url: str, inference_id: str):
        self.update_progress(inference_id, "loading_image", 0)
        # Define the desired classes
        desired_classes = [25, 58]

        conf_thres = 0.1
        iou_thres = 0.45
        classes = None
        agnostic_nms = False
        max_det = 1000

        image = load_image(image_url)
        image = resize_image(image, max_size=(256, 256))
        im = self.preprocess_image(image)

        self.update_progress(inference_id, "detect_tree_area", 25)
        pred = self.yolov9_model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        self.sam_model.set_image(image)

        image_height, image_width, _ = image.shape
        gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        aggregate_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        self.update_progress(inference_id, "generating_tree_mask", 30)
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

                    # Generate and accumulate masks for each bounding box
                    input_box = np.array((xmin, ymin, xmax, ymax)).reshape(1, 4)
                    masks, _, _ = self.sam_model.predict(
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
        new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]
        grayscale_image = cv2.cvtColor(new_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        noisy_img = add_noise(grayscale_image)

        image_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2RGB)

        # Free up GPU memory before processing each image
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.update_progress(inference_id, "generate_leaf_masks", 60)

        sam_result = self.mask_generator.generate(image_rgb)

        # Filter out the background
        filtered_masks = filter_out_background(sam_result)

        annotated_image = cv2.cvtColor(new_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        colors = generate_colors(len(filtered_masks))

        self.update_progress(inference_id, "classifying", 90)
        petal_count = 0
        for i, mask in enumerate(filtered_masks):
            segmentation = mask['segmentation'].astype('uint8')

            # Skip white background areas
            if is_white_background(segmentation, image_rgb):
                continue

            # Color the segmented area
            color = colors[i]
            annotated_image[segmentation > 0] = cv2.addWeighted(annotated_image, 0.5, np.full_like(annotated_image, color), 0.5, 0)[segmentation > 0]
            petal_count += 1

        classification_input = np.array([[petal_count]]).astype(np.int64)
        age = self.classification_model.run([self.classification_label_name], {self.classification_input_name: classification_input})[0]

        if len(age) > 0:
            age = age[0]
        else:
            age = None
        
        # Encode the image to base64
        annotated_image = resize_image(annotated_image, max_size=(256, 256))
        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return self.update_progress(inference_id, "completed", 100, age, image_base64)
