import argparse
import csv
import sys
import lpips
import cv2

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v0','--video0', type=str, default='./vids/ex_ref.png')
parser.add_argument('-v1','--video1', type=str, default='./vids/ex_p0.png')
parser.add_argument('-o','--out', type=str, default='./vids/example_dists.csv')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)

if(opt.use_gpu):
	loss_fn.cuda()

cap0 = cv2.VideoCapture(opt.video0)
if not cap0.isOpened():
	print("Could not open video %s"%opt.video0, file=sys.stderr)
	sys.exit(1)

cap1 = cv2.VideoCapture(opt.video1)
if not cap1.isOpened():
	print("Could not open video %s"%opt.video1, file=sys.stderr)
	sys.exit(1)

f = open(opt.out,'w', newline='')
writer = csv.writer(f)
writer.writerow(["Frame","Distance"])

count = 0
while True:
	s0, img0 = cap0.read()
	s1, img1 = cap1.read()

	if not (s0 and s1):
		break

	# Load images
	img0 = lpips.im2tensor(img0) # RGB image from [-1,1]
	img1 = lpips.im2tensor(img1)

	if(opt.use_gpu):
		img0 = img0.cuda()
		img1 = img1.cuda()

	# Compute distance
	dist = loss_fn.forward(img0, img1)
	print('Frame %d: %.3f'%(count, dist))
	writer.writerow([count, float(dist)])

	count += 1

f.close()
