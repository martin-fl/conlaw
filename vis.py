import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
	unj = np.loadtxt("output.csv", delimiter=",")

	fig, ax = plt.subplots()
	wave_plot = ax.plot(unj[0,:])[0]
	ax.set(ylim=[0.0, 1.0])

	def update(frame):
		wave_plot.set_ydata(unj[frame,:])


	gif = FuncAnimation(
		fig=fig, 
		func=update, 
		frames=unj.shape[0], 
		interval=1
	)

	gif.save(filename="output.gif", writer="ffmpeg", fps=60)
	return


main()