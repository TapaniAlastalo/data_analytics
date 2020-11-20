import sys
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from datetime import datetime
import pymongo_spark
import json

pymongo_spark.activate()


if __name__ == "__main__":
    conf = SparkConf().setAppName("temperatureDataApp")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("OFF")
    ssc = StreamingContext(sc, 5)

    brokers, topic = sys.argv[1:]
    kvs = KafkaUtils.createDirectStream(ssc, [topic],{"metadata.broker.list": brokers})
    lines = kvs.map(lambda x: json.loads(x[1]))
    print(lines)
    sensors = lines.map(lambda y : "Aikaleima: " + str(y['dt']) + " | ID: " + str(y['id']) + " | Paikka: " + str(y['name']) + " | Temp: " + str(y['main']['temp']) 
    + " | Feels: " + str(y['main']['feels_like']) + " | MinTemp: " + str(y['main']['temp_min']) + " | MaxTemp: " + str(y['main']['temp_max']) )
    sensors.pprint()
    rdd = lines.map(lambda x : {"id": x['id'], "timestamp": x['dt'], "date": x['dt'].strftime("%Y.%m.%d"), "time": x['dt'].strftime("%H:%M:%S"), "name": x['name'], 
    "temp": x['main']['temp'], "feel": x['main']['feels_like'], "min":x['main']['temp_min'], "max":x['main']['temp_max'] })
    rdd.foreachRDD(lambda z: z.saveToMongoDB('mongodb://192.168.1.24:27017/temperaturedatadb.temperaturedata'))
    ssc.start()
    ssc.awaitTermination()