import argparse
import csv
import gc
import subprocess
import sys
import lpips
import cv2
import numpy
import torch

WIDTH = 1920
HEIGHT = 1080

def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('reference', type=str)
	parser.add_argument('distorted', type=str)
	parser.add_argument('-o','--out', type=str)
	parser.add_argument('-v','--version', type=str, default='0.1')
	parser.add_argument('-n', '--net', type=str, choices=['alex', 'squeeze', 'vgg'], default='alex')
	parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

	return parser.parse_args()

def main():
	opt = parse_args()

	## Initializing the model
	loss_fn = lpips.LPIPS(net=opt.net,version=opt.version)

	if(opt.use_gpu):
		loss_fn.cuda()

	reference_proc = subprocess.Popen(["/usr/bin/ffmpeg", "-i", opt.reference, "-filter_complex", "[0:v]fps=24001/1001[out]", "-map", "[out]", "-pix_fmt", "rgb24", "-c:v", "rawvideo", "-f", "image2pipe", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
	distorted_proc = subprocess.Popen(["/usr/bin/ffmpeg", "-i", opt.distorted, "-filter_complex", "[0:v]fps=24001/1001[out]", "-map", "[out]", "-pix_fmt", "rgb24", "-c:v", "rawvideo", "-f", "image2pipe", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

	try:
		if opt.out:
			f = open(opt.out,'w', newline='')
			writer = csv.writer(f)
			writer.writerow(["Frame","Distance"])

		count = 0
		total = 0
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
			print('Frame %d: %.3f'%(count, dist))
			writer.writerow([count, float(dist)])

			del reference
			del distorted
			gc.collect()

			count += 1
		print("Average distance:", total/count)
	finally:
		if opt.out:
			f.close()

if __name__ == "__main__":
	main()
