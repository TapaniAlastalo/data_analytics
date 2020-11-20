import argparse
from confluent_kafka import Producer
from time import sleep, time
import requests


def requestTemperature(topic, wait):
    try:
        location = "helsinki"
        apiKey = "f87283ebd7f8ee743f5abf251d32505b"
        url = "https://api.openweathermap.org/data/2.5/weather?q="+location+"&appid="+apiKey
        response = requests.request("GET", url)
        json = response.text
        print(json)
        if p is not None:
            p.produce(topic, json.encode('utf-8'))
        if wait > 0:
            sleep(wait)
    except KeyboardInterrupt:
        pass
    finally:
        p.flush()
        requestTemperature(topic=topic, wait=wait)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temperature data receiver for spark book")
    parser.add_argument("--kafka-brokers", "-k",
                        nargs="?",
                        help="Comma-separated list of kafka brokers. If specified, data is published to kafka.")
    parser.add_argument("--kafka-topic", "-t",
                        nargs="?",
                        help="Topic name to publish temperature data(default: temperature_topic).")
    parser.add_argument("--wait", "-w",
                        nargs="?",
                        help="Number of minutes after to wait next message (default: 1).")
    args = parser.parse_args()
    w = float(args.wait) if args.wait else float(1)
    p = Producer({'bootstrap.servers': args.kafka_brokers}) if args.kafka_brokers else None
    topic = args.kafka_topic or "temperature_topic"
    wait = 60.0 * w # seconds
    requestTemperature(topic = topic, wait = wait)