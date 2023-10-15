#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import argparse
import os
import humanize

class CSFF1:
	def __init__(self, src):
		self.src = src

	def __enter__(self):
		self.solution = open(self.src, "rb")

		self.file_size = os.fstat(self.solution.fileno()).st_size

		header = self.solution.read(len(b"CSFF1"))
		assert header == b"CSFF1", f"file `{file}` not in CSFF1 format"

		self.float_size = int.from_bytes(self.solution.read(1),byteorder=sys.byteorder)
		self.dt = np.float32 if self.float_size == 4 else np.float64
		self.space_steps = int.from_bytes(self.solution.read(4), byteorder=sys.byteorder)
		self.time_sampling = int.from_bytes(self.solution.read(4), byteorder=sys.byteorder)
		self.space_sampling = int.from_bytes(self.solution.read(4), byteorder=sys.byteorder)
		self.system_size = int.from_bytes(self.solution.read(4), byteorder=sys.byteorder)
		self.time_steps = int.from_bytes(self.solution.read(4), byteorder=sys.byteorder)
		[self.space_lower, self.space_upper] = np.frombuffer(self.solution.read(2*self.float_size), self.dt)
		[self.time_lower, self.time_upper] = np.frombuffer(self.solution.read(2*self.float_size), self.dt)
		method_name_len = int.from_bytes(self.solution.read(4), byteorder=sys.byteorder)
		self.method_name = self.solution.read(method_name_len).decode()

		assert self.solution.read(4) == b"\xff\xff\xff\xff"

		self.sample_start = self.solution.tell()
		self.samples_length= self.space_steps//self.space_sampling + 1
		self.num_samples = self.time_steps//self.time_sampling + 1

		return self


	def __str__(self):
		return f"""\
CSFF1 reader:
	file name: `{self.src}` 
	file size: {humanize.naturalsize(self.file_size, binary=True)} 
	output file name: `{file}.gif`
	spatial grid: [{self.space_lower}, {self.space_upper}], Δx = {(self.space_upper-self.space_lower)/self.space_steps} ({self.space_steps} steps)
	temporal grid: [{self.time_lower}, {self.time_upper}], Δt = {(self.time_upper-self.time_lower)/self.time_steps} ({self.time_steps} steps)
	time sampling period: {self.time_sampling}
	space sampling period: {self.space_sampling}
	floating-point precision: {self.float_size*8} bits
	numerical method: {self.method_name}"""


	def samples(self):
		self.solution.seek(self.sample_start)

		for _ in range(self.num_samples):
			yield np.frombuffer(self.solution.read(self.float_size*self.system_size*self.samples_length), self.dt)

		assert self.solution.read(4) == b"\xff\xff\xff\xff"
		
		
	def close(self):
		self.solution.close()
	

	def __exit__(self, exc_type, exc_value, traceback):
		self.close()


def render(input: CSFF1):
	xs = np.linspace(input.space_lower, input.space_upper, num=input.samples_length)
	samples = input.samples()

	fig, ax = plt.subplots()
	u = next(samples)
	handle = ax.plot(xs, u)[0]
	ax.set(ylim=[np.floor(np.min(u)), np.ceil(np.max(u))])

	gif = FuncAnimation(
		fig=fig, frames=input.num_samples-2,
		func=lambda frame: handle.set_ydata(next(samples)), 
	)
	gif.save(filename=f"{input.src}.gif", writer="ffmpeg", fps=60)


parser = argparse.ArgumentParser(
	description="Render gif of conservation law solutions in Conlaw Solution File Format v1 (CSSF1)"
)
parser.add_argument(
	"files", metavar="file", type=str, nargs='+', help="CSSF1 file to render as gif"
)

args = parser.parse_args()

for file in args.files:
	with CSFF1(file) as input:
		print(input)
		render(input)
