import argparse
import csv
import gc
import os.path
import subprocess
import lpips
import pymediainfo
import numpy
import tqdm
import torch
from torchvision.io import VideoReader

def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('reference', type=str)
	parser.add_argument('distorted', type=str)
	parser.add_argument('-o','--out', type=str)
	parser.add_argument('-v','--version', type=str, default='0.1')
	parser.add_argument('-n', '--net', type=str, choices=['alex', 'squeeze', 'vgg'], default='alex')
	parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
	parser.add_argument('--no-resume', action='store_false', dest="resume", help='overwrite existing output instead of continuing it')

	return parser.parse_args()

def main():
	opt = parse_args()

	## Initializing the model
	loss_fn = lpips.LPIPS(net=opt.net,version=opt.version)

	if opt.use_gpu:
		loss_fn.cuda()

	# Get video metadata
	reference_info = pymediainfo.MediaInfo.parse(opt.reference)
	distorted_info = pymediainfo.MediaInfo.parse(opt.distorted)

	reference_video_tracks = [t for t in reference_info.tracks if t.track_type == 'Video']
	distorted_video_tracks = [t for t in distorted_info.tracks if t.track_type == 'Video']
	if len(reference_video_tracks) != 1 or len(distorted_video_tracks) != 1:
		raise Exception("Expected one video track")

	reference_width = int(reference_video_tracks[0].width)
	reference_height = int(reference_video_tracks[0].height)
	reference_fps = f"{reference_video_tracks[0].framerate_num}/{reference_video_tracks[0].framerate_den}"

	distorted_width = int(distorted_video_tracks[0].width)
	distorted_height = int(distorted_video_tracks[0].height)
	distorted_fps = f"{distorted_video_tracks[0].framerate_num}/{distorted_video_tracks[0].framerate_den}"

	if reference_width != distorted_width or reference_height != distorted_height:
		raise Exception("Video dimensions do not match")
	if reference_fps != distorted_fps:
		raise Exception("Video frame rates do not match")

	WIDTH = reference_width
	HEIGHT = reference_height
	FPS = reference_fps

	pbar = tqdm.tqdm(total=int(reference_info.tracks[0].frame_count))

	try:
		total = 0
		count = 0
		if opt.out:
			if os.path.exists(opt.out):
				with open(opt.out, 'r') as f:
					reader = csv.DictReader(f)
					for row in reader:
						total += float(row[opt.net])
						count += 1
			f = open(opt.out, 'w' if count == 0 else 'a', newline='')
			writer = csv.writer(f)
			if count == 0:
				writer.writerow(["Frame",opt.net])

		reference_proc = subprocess.Popen(["/miniconda/bin/ffmpeg", "-i", opt.reference, "-filter_complex", f"[0:v]setpts=PTS-STARTPTS,fps={FPS},select=gte(n\,{count})[out]", "-vsync", "0", "-map", "[out]", "-pix_fmt", "rgb24", "-c:v", "rawvideo", "-f", "image2pipe", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
		distorted_proc = subprocess.Popen(["/miniconda/bin/ffmpeg", "-i", opt.distorted, "-filter_complex", f"[0:v]setpts=PTS-STARTPTS,fps={FPS},select=gte(n\,{count})[out]", "-vsync", "0", "-map", "[out]", "-pix_fmt", "rgb24", "-c:v", "rawvideo", "-f", "image2pipe", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

		pbar.update(count)

		while True:
			reference_raw = reference_proc.stdout.read(WIDTH*HEIGHT*3)
			if not reference_raw:
				break
			reference = numpy.frombuffer(reference_raw, dtype='uint8')
			reference = reference.reshape((HEIGHT,WIDTH,3))[:,:,:3]
			reference = lpips.im2tensor(reference)

			distorted_raw = distorted_proc.stdout.read(WIDTH*HEIGHT*3)
			if not distorted_raw:
				break
			distorted = numpy.frombuffer(distorted_raw, dtype='uint8')
			distorted = distorted.reshape((HEIGHT,WIDTH,3))[:,:,:3]
			distorted = lpips.im2tensor(distorted)

			if opt.use_gpu:
				reference = reference.cuda()
				distorted = distorted.cuda()

			# Compute distance
			dist = loss_fn.forward(reference, distorted)
			total += float(dist)
			writer.writerow([count, float(dist)])

			del reference
			del distorted
			gc.collect()

			count += 1
			pbar.update()
		print("Average distance:", total/count)
	finally:
		if opt.out:
			f.close()
		pbar.close()

if __name__ == "__main__":
	main()
