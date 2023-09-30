"""Экстрактор изображений из видео файла"""

import os
import cv2
# import numpy as np
from pathlib import Path


# Function to extract frames
def frame_capture(
		video_in: Path,
		pict_directory: Path,
		frequency: int,
		max_frames: int) -> None:
	"""
	Запись из видеофайла  max кадров через каждые frequency
	param: video_in: Path - путь к видео файлу
	param: pict_directory: Path - путь для сохранения изображений
	param: frequency: int - через сколько секунд сохранять кадр
	param: max_frames: int - максимальное количество сохраняемых кадров
	"""
	pict_directory = pict_directory.joinpath(video_in.stem)
	Path.mkdir(pict_directory, parents=True, exist_ok=True)
	# Path to video file
	cap = cv2.VideoCapture(str(video_in))

	# FPS
	# fps = cap.get(5) 	 # FPS - нужен для указания частоты в секундах
	frames = cap.get(7)  # Если не задано, то равен общему количеству кадров
	digits = len(str(frames))

	if max_frames == 0:
		max_frames = frames

	# Used as counter variable
	count = 0
	while cap.isOpened():
		# cap object calls read
		# function extract frames
		success, image = cap.read()

		if ((count // frequency) < max_frames) & success:
			count += 1
			print(count // frequency, count)

			if count % frequency == 0:
				# Сохраняется кадр через каждые frequency кадров
				save_path = (
					pict_directory
					.joinpath(f'image-{count:0{digits}}.png'))
				try:
					cv2.imwrite(str(save_path), image)
				except Exception as e:
					print(e)
		else:
			break


def capture_first_n_frames(path_in, path_out, n):

	# Path to video file
	cap = cv2.VideoCapture(path_in)

	# Used as counter variable
	count = 0

	# checks whether frames were extracted

	while count < n:
		print(count)
		# cap object calls read
		# function extract frames
		success, image = cap.read()
		save_path = os.path.join(path_out, f'frame-{count:05}.jpg')
		# Saves the frames with frame-count
		cv2.imwrite(save_path, image)

		count += 1


if __name__ == "__main__":

	video = Path('../video/gai_1min.mp4').absolute()
	pic_dir = Path('../video/images')
	frame_capture(
		video,
		pic_dir,
		frequency=50,     # every 2 sec
		max_frames=0	 # Не ограничено
	)

	# pass
	#
	# out = np.zeros(3)
	# for file in os.listdir('video'):
	# 	dev = file.split('-')[0]
	# 	try:
	# 		channel = file.split('-')[1]
	# 	except IndexError:
	# 		channel = ''
	# 	file = os.path.join(os.path.join('video'), file)
	# 	print(file)
	# 	print(file[:-3]+'jpg')
	# 	capture_first_n_frames(file, file[:-3]+'jpg', 1)

#     stat = calc_stats_from_video(file)
#     row = np.array([dev, channel, stat])
#     out = np.vstack([out, row])
#     # print(dev, channel, stat)
# out = np.delete(out, (0), axis=0)
#
# print(out)
#
# print(out[:, 2].min())
