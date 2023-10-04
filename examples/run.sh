# Анализ трафика на перекрестке ул. Республики
#!/bin/bash

DIR='/netdisk/Projects/YoloTrackingCountingPytorch'

source ${HOME}/${DIR}/venv/bin/activate

cd ${HOME}/${DIR}

# Обработка исходного видео в фоне и сохранение результатов в видео файл
python3 ./examples/traffic_analysis_bottom_trigger.py --source_weights_path ./examples/yolov8x.pt --source_video_path ./video/2.mp4 --target_video_path ./video/2-result.mp4

# Обработка и вывод видео в реальном времени отдельном в окне (видео файл не сохраняется)
# python3 ./examples/traffic_analysis_bottom_trigger.py --source_weights_path ./examples/yolov8x.pt --source_video_path ./video/2.mp4

# Обработка лога и вывод отчета в stdout  и сохранение в файл report.html
python3 ./examples/traffic_count.py

deactivate
