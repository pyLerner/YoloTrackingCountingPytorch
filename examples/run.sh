#!/bin/bash

DIR='/netdisk/Projects/YoloTrackingCountingPytorch'

source ${HOME}/${DIR}/venv/bin/activate

cd ${HOME}/${DIR}

python3 ./examples/traffic_analysis_bottom_trigger.py --source_weights_path ./examples/yolov8x.pt --source_video_path ./video/2.mp4 --target_video_path ./video/2-result.mp4


# python3 ./examples/traffic_analysis.py --source_weights_path ./examples/yolov8x.pt --source_video_path ./video/2.mp4 # --target_video_path ./video/2-result.mp4

python3 ./examples/traffic_count.py

deactivate
