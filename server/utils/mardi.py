'''Utility functions for the server'''
import random
import base64
import cv2
import numpy as np
import requests

def load_image(image_url: str):
    # Fetch the image from the URL
    response = requests.get(image_url, timeout=10)
    if response.status_code != 200:
        raise ValueError("Unable to fetch image from URL")

    # Convert the image content to a NumPy array
    image_array = np.frombuffer(response.content, np.uint8)

    # Read the image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image")

    return image

def show_mask(mask, color):
    '''Function to display mask in an image'''
    h, w = mask.shape[-2:]
    mask_image = (mask.reshape(h, w, 1) * np.array(color)).astype(np.uint8)
    return mask_image


def show_box(image, box, label, conf_score, color):
    '''Function to display bounding box in an image'''
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = int(box[2]), int(box[3])
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    label_text = f'{label} {conf_score:.2f}'
    cv2.putText(image, label_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def add_noise(img):
    '''Function to add noise to an image'''
    # Getting the dimensions of the image
    row, col = img.shape

    # Randomly pick some pixels in the image for coloring them white
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)
        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)
        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in the image for coloring them black
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)
        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)
        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


def resize_image(image, max_size=(512, 512)):
    '''Function to resize image to a maximum size'''
    h, w = image.shape[:2]
    if h > max_size[0] or w > max_size[1]:
        scale = min(max_size[0]/h, max_size[1]/w)
        new_size = (int(w*scale), int(h*scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def filter_out_background(masks, threshold=0.5):
    '''Function to filter out the largest mask, assuming it is the background'''
    areas = [mask["area"] for mask in masks]
    max_area = max(areas)
    filtered_masks = [mask for mask in masks if mask["area"] < threshold * max_area]
    return filtered_masks


def generate_colors(n):
    '''Function to generate random colors for segmentation'''
    colors = np.random.randint(0, 255, size=(n, 3)).tolist()
    return colors


def is_white_background(segmentation, image_rgb, threshold=240):
    '''Function to check if the mask is predominantly white (background)'''
    mask_area = segmentation > 0
    mean_color = np.mean(image_rgb[mask_area], axis=0)
    return np.all(mean_color > threshold)


def add_text_to_image(image_url: str, text: str) -> str:
    '''Function to add text to an image'''
    image = load_image(image_url)

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Define the text and its properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 3

    # Calculate the position for the text to be centered
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    # Add the text to the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

    # Encode the image to base64
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return image_base64
