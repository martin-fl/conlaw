#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import argparse
import os
import humanize

def render(file: str):
	with open(file, "rb") as solution:
		file_size = os.fstat(solution.fileno()).st_size
		header = solution.read(len(b"CSFF1"))
		assert header == b"CSFF1", f"file `{file}` not in CSFF1 format"

		float_size = int.from_bytes(solution.read(1),byteorder=sys.byteorder)
		dt = np.float32 if float_size == 4 else np.float64
		space_steps = int.from_bytes(solution.read(4), byteorder=sys.byteorder)
		system_dim = int.from_bytes(solution.read(4), byteorder=sys.byteorder)
		time_steps = int.from_bytes(solution.read(4), byteorder=sys.byteorder)
		[space_lower, space_upper] = np.frombuffer(solution.read(2*float_size), dt)
		[time_lower, time_upper] = np.frombuffer(solution.read(2*float_size), dt)
		method_name_len = int.from_bytes(solution.read(4), byteorder=sys.byteorder)
		method_name = solution.read(method_name_len).decode()

		print(f"""\
rendering CSFF1 file:
	file name: `{file}` 
	file size: {humanize.naturalsize(file_size, binary=True)} 
	output file name: `{file}.gif`
	spatial grid: [{space_lower}, {space_upper}], Δx = {(space_upper-space_lower)/space_steps} ({space_steps} steps)
	temporal grid: [{time_lower}, {time_upper}], Δt = {(time_upper-time_lower)/time_steps} ({time_steps} steps)
	floating-point precision: {float_size*8} bits
	numerical method: {method_name}\
""")

		assert solution.read(4) == b"\xff\xff\xff\xff"
		
		def read_next():
			return np.frombuffer(solution.read(float_size*(space_steps+1)), dt)

		xs = np.linspace(space_lower, space_upper, num=space_steps+1)
		u = read_next()
	
		fig, ax = plt.subplots()
		wave_plot = ax.plot(xs, u)[0]
		ax.set(ylim=[np.floor(np.min(u)), np.ceil(np.max(u))])

		def update(frame):
			wave_plot.set_ydata(read_next())

		gif = FuncAnimation(
			fig=fig, 
			func=update, 
			frames=time_steps-1, 
		)

		gif.save(filename=f"{file}.gif", writer="ffmpeg", fps=60)

		assert solution.read(4) == b"\xff\xff\xff\xff"

parser = argparse.ArgumentParser(
	description="Render gif of conservation law solutions in Conlaw Solution File Format v1 (CSSF1)"
)
parser.add_argument(
	"files", metavar="file", type=str, nargs='+', help="CSSF1 file to render as gif"
)

args = parser.parse_args()

for file in args.files:
	render(file)
