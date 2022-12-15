import argparse
import csv
import gc
import subprocess
import sys
import lpips
import cv2
import numpy
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

	return parser.parse_args()

def main():
	opt = parse_args()

	## Initializing the model
	loss_fn = lpips.LPIPS(net=opt.net,version=opt.version)

	if(opt.use_gpu):
		loss_fn.cuda()

	backend = "cuda" if opt.use_gpu else "cpu"
	reference_decoder = VideoReader(opt.reference, device=backend)
	distorted_decoder = VideoReader(opt.distorted, device=backend)

	try:
		if opt.out:
			f = open(opt.out,'w', newline='')
			writer = csv.writer(f)
			writer.writerow(["Frame",opt.net])

		count = 0
		total = 0
		for reference, distorted in zip(reference_decoder, distorted_decoder):
			reference = reference["data"][numpy.newaxis, :, :, :].transpose(1, 3).transpose(2, 3)
			distorted = distorted["data"][numpy.newaxis, :, :, :].transpose(1, 3).transpose(2, 3)
			#if opt.use_gpu:
				#reference = reference.cuda()
				#distorted = distorted.cuda()

			# Compute distance
			dist = loss_fn.forward(reference, distorted)
			total += float(dist)
			#print('Frame %d: %.3f'%(count, dist))
			writer.writerow([count, float(dist)])
			#f.flush()

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
