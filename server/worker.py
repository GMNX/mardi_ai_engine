from pika import BasicProperties
from inference import Inference
from bridge import get_rabbitmq_connection


def callback(ch, method, properties: BasicProperties, body):
    '''Callback function to process incoming inference requests'''
    image_url = body
    print(f"Worker processing image {image_url} with inference id {properties.headers['inference_id']}")
    try:
        result = model.predict(image_url, properties.headers["inference_id"])
        print(result)
    except Exception as e:
        error_string = str(e)
        model.update_progress(properties.headers["inference_id"], status=f"error: {error_string}", progress=100)

    ch.basic_ack(delivery_tag=method.delivery_tag)


# Initialize the pytorch model
yolo_classification_weights = "/data/models/model_classification.onnx"
yolo_detection_weights = "/data/yolov9/weights/gelan-c.pt"
sam_checkpoint = "/data/models/sam_vit_h_4b8939.pth"
classification1_model_path = "/data/models/rf_model_class1_v04.onnx"
classification2_model_path = "/data/models/rf_model_class2_v04.onnx"
model = Inference(yolo_classification_weights, yolo_detection_weights, sam_checkpoint, classification1_model_path, classification2_model_path)

# Initialize the RabbitMQ client
rabbitmq_client = get_rabbitmq_connection()

rabbitmq_client.basic_qos(prefetch_count=1)
rabbitmq_client.basic_consume(queue='mardi_inference_queue', on_message_callback=callback)

rabbitmq_client.start_consuming()