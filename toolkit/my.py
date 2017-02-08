import time

start_time = time.time()

def stopwatch():
	global start_time
	print("\t%.3f" % (time.time() - start_time) + " seconds")
	start_time = time.time()

