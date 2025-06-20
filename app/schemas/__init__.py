from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, TimestampType,
    ArrayType, FloatType, BooleanType
)
from datetime import datetime
from pyspark.sql import Row

spark = SparkSession.builder.appName("EmoAPI_Schema").getOrCreate()

schema = StructType([
    StructField("id", StringType(), False),
    StructField("modality", StringType(), False),
    StructField("emotion_label", StringType(), False),
    StructField("emotion_idx", IntegerType(), False),
    StructField("source_dataset", StringType(), False),
    StructField("split", StringType(), False),
    StructField("timestamp", TimestampType(), True),

    # Text
    StructField("raw_text", StringType(), True),
    StructField("tokens", ArrayType(StringType()), True),
    StructField("embedding", ArrayType(FloatType()), True),

    # Audio
    StructField("audio_path", StringType(), True),
    StructField("duration_sec", FloatType(), True),
    StructField("mfcc_features", ArrayType(FloatType()), True),

    # Image
    StructField("image_path", StringType(), True),
    StructField("image_shape", ArrayType(IntegerType()), True),
    StructField("face_detected", BooleanType(), True),
    StructField("face_embedding", ArrayType(FloatType()), True),
])


sample = Row(
    id="text_001",
    modality="text",
    emotion_label="joy",
    emotion_idx=0,
    source_dataset="GoEmotions",
    split="train",
    timestamp=datetime.now(),
    raw_text="I love this app!",
    tokens=["I", "love", "this", "app"],
    embedding=None,
    audio_path=None,
    duration_sec=None,
    mfcc_features=None,
    image_path=None,
    image_shape=None,
    face_detected=None,
    face_embedding=None
)

# Write to Parquet
spark.createDataFrame([sample], schema=schema) \
     .write.mode("overwrite") \
     .parquet("data/silver/emotions/")