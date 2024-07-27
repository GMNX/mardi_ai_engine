import pika
from pika import PlainCredentials

RABBITMQ_URL = "rabbitmq"
RABBITMQ_USER = "mardi"
RABBITMQ_PASSWORD = "password"
RABBITMQ_QUEUE = "mardi_inference_queue"

def get_rabbitmq_connection():
    '''Get a connection to the RabbitMQ server'''
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_URL,
            port=5672,
            credentials=PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
        )
    )
    rabbitmq_client = connection.channel()
    rabbitmq_client.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
    return rabbitmq_client
