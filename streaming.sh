#!/bin/bash
HADOOP_HOME=/usr/lib/hadoop-0.20-mapreduce

#The streaming jar location
HADOOP_STREAMING_JAR=$HADOOP_HOME/contrib/streaming/hadoop-streaming.jar

# CONFIGURATION

# INPUT FOLDER
INPUT_FOLDER='input/MI'
# OUTPUT FOLDER
OUTPUT_FOLDER='output/MI'

# Check if relevant folders exists
hadoop fs -test -e OUTPUT_FOLDER
if [ $? -eq 0 ]
    then echo 'Output folder already exists'
    exit 1
fi

hadoop jar $HADOOP_STREAMING_JAR \
-D mapred.reduce.tasks=2000 \
-input $INPUT_FOLDER \
-output $OUTPUT_FOLDER \
-mapper ./feat_group_mapper.py \
-reducer ./MI_reducer.py \
-file ./feat_group_mapper.py \
-file ./MI_reducer.py \
-file ./info.py \
-file ./preproc.py 
