import time

for i in range(60):
	time.sleep(1)
	print("Iteration: ", i+1)
	if i == 25:
		raise ValueError()

else:
	print("The code was succesfully executed")


