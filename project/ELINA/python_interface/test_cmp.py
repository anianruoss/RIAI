from elina_interval import *
import numpy as np
from ctypes.util import find_library
import ctypes

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')
printf = libc.printf

def to_str(str):
	return bytes(str, 'utf-8')

def print_c(str):
	printf(to_str(str))








def make_interval(array):
	arr = elina_interval_array_alloc(len(array))
	for i in range(len(array)):
		elina_interval_set_double(arr[i], array[i], array[i])
	
	return arr

def print_interval(interval):
	for i in range(10):
		elina_interval_fprint(cstdout, interval[i])
		print_c("\n")









rnd = np.random.rand(10)

interval = make_interval(rnd)

print_interval(interval)
print(elina_interval_cmp(interval, interval))

elina_interval_array_free(interval, 10)
