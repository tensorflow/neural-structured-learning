# Reference: https://github.com/microsoft/denoised-smoothing/blob/master/code/certify.py

from . import certification_utils
from time import time
import datetime
import os

SKIP = 20
MAX = -1

def perform_certification_test(base_classifier, dataset, sigma,
							   outfile="certification_output/sigma_0.25"):
	smoothed_classifier = certification_utils.Smooth(base_classifier,
								 10, sigma)

	# prepare output file
	if not os.path.exists(outfile.split('sigma')[0]):
		os.makedirs(outfile.split('sigma')[0])

	f = open(outfile, 'w')
	print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f,
		  flush=True)
	print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)
	f.close()

	images, labels = dataset
	for i in range(len(images)):

		# only certify every args.skip examples, and stop after
		# args.max examples
		if i % SKIP != 0:
			continue
		if i == MAX:
			break

		(x, label) = images[i], labels[i]

		before_time = time()
		# certify the prediction of g around x
		prediction, radius = smoothed_classifier.certify(x, 100,
														 10000,
														 0.001,
														 1000)
		after_time = time()
		correct = int(prediction == label)

		time_elapsed = str(
			datetime.timedelta(seconds=(after_time - before_time)))

		f = open(outfile, 'a')
		print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
			i, label, prediction, radius, correct, time_elapsed),
			file=f, flush=True)
		print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
			i, label, prediction, radius, correct, time_elapsed),
			flush=True)
		f.close()