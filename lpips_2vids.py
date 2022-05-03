import argparse
import csv
import sys
import lpips
import cv2
import numpy
import torch

WIDTH = 1920
HEIGHT = 1080

def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-v','--version', type=str, default='0.1')
	parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

	return parser.parse_args()

def main():
	opt = parse_args()

	## Initializing the model
	loss_fn1 = lpips.LPIPS(net='alex',version=opt.version)
	#loss_fn2 = lpips.LPIPS(net='vgg',version=opt.version)
	#loss_fn3 = lpips.LPIPS(net='squeeze',version=opt.version)

	if(opt.use_gpu):
		loss_fn1.cuda()
		#loss_fn2.cuda()
		#loss_fn3.cuda()

	try:
		stdout = open(sys.__stdout__.fileno(),
	          mode=sys.__stdout__.mode,
	          buffering=1,
	          encoding=sys.__stdout__.encoding,
	          errors=sys.__stdout__.errors,
	          newline='',
	          closefd=False)
		writer = csv.writer(stdout)
		writer.writerow(["Frame","alex"])

		count = 0
		while True:
			reference_raw = sys.stdin.buffer.read(WIDTH*HEIGHT*3)
			if not reference_raw:
				break
			reference = numpy.frombuffer(reference_raw, dtype='uint8')
			reference = reference.reshape((HEIGHT,WIDTH,3))[:,:,:3]
			reference = lpips.im2tensor(reference)

			distorted_raw = sys.stdin.buffer.read(WIDTH*HEIGHT*3)
			if not distorted_raw:
				break
			distorted = numpy.frombuffer(distorted_raw, dtype='uint8')
			distorted = distorted.reshape((HEIGHT,WIDTH,3))[:,:,:3]
			distorted = lpips.im2tensor(distorted)

			if opt.use_gpu:
				reference = reference.cuda()
				distorted = distorted.cuda()

			# Compute distance
			dist1 = float(loss_fn1.forward(reference, distorted))
			#dist2 = float(loss_fn2.forward(reference, distorted))
			#dist3 = float(loss_fn3.forward(reference, distorted))
			writer.writerow([count, dist1])

			count += 1
	finally:
		stdout.close()

if __name__ == "__main__":
	main()
