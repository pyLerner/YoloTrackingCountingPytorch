"""Экстрактор изображений из видео файла"""

import os
import numpy as np
import cv2


# Function to extract frames
def frame_capture(video_path: str) -> None:

	# Path to video file
	cap = cv2.VideoCapture(video_path)

	# Used as counter variable
	count = 0

	# checks whether frames were extracted
	success = 1

	while success:

		# cap object calls read
		# function extract frames
		success, image = cap.read()

		# Saves the frames with frame-count
		cv2.imwrite(f'frame-{count:05}.jpg', image)

		count += 1


# Driver Code
# if __name__ == '__main__':
#
# 	os.chdir('Images/tablo/')
# 	# Calling the function
# 	FrameCapture("m7316-4-2023-02-09_16-09-58140.avi")


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
	pass

	out = np.zeros(3)
	for file in os.listdir('video'):
		dev = file.split('-')[0]
		try:
			channel = file.split('-')[1]
		except IndexError:
			channel = ''
		file = os.path.join(os.path.join('video'), file)
		print(file)
		print(file[:-3]+'jpg')
		capture_first_n_frames(file, file[:-3]+'jpg', 1)

#     stat = calc_stats_from_video(file)
#     row = np.array([dev, channel, stat])
#     out = np.vstack([out, row])
#     # print(dev, channel, stat)
# out = np.delete(out, (0), axis=0)
#
# print(out)
#
# print(out[:, 2].min())
