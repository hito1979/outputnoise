#refer files
import goodwin_functions
import math
import numpy as np
import matplotlib.pyplot as plt

def goodwin(dt, stepNO, k, D_osci, D_X, a, b, N, flag_osci, flag_X, e):

	import numpy as np
	import matplotlib.pyplot as plt
	import math
	import random

	###########################################################################################################################
	#パラメーターパート
	#関数の引数の用途を示す

	#dt:for euler's method
	#stepNO:how many step you wanna repeat
	#n:demation
	#k:decompotion rate
	#D:noise coefficent
	#a:a parameter
	#b:a parameter
	#N:randam seed
	#flag:with noise:1 or without noise:0


	###########################################################################################################################
	#definition
	#no need to change
	omega = 2*math.pi
	X = [0] * stepNO 
	#X[0] = 0.001
	osci_x = [0] * stepNO
	osci_y = [0] * stepNO
	osci_z = [0] * stepNO
	time = [0]*stepNO #時間


	np.random.seed(N)
	r = np.random.normal(0,1, size=stepNO*2) #random value

	def goodwin_x(x, z):
		goodwin_x = e*(1/(1+z**10)-0.1*x)
		return goodwin_x

	def goodwin_y(x, y):
		goodwin_y = e*(x-0.1*y)
		return goodwin_y

	def goodwin_z(y, z):
		goodwin_z = e*(y-0.1*z)
		return goodwin_z

	def f_1(a, b, k, x, y):
		f_1 = a + b*x - k*y
		return f_1


	###########################################################################################################################
	#オイラー法パート

	for i in range(stepNO-1):
		osci_x[i+1] = osci_x[i] + goodwin_x(osci_x[i], osci_z[i])*dt + flag_osci*r[i]*pow(D_osci,0.5)*pow(dt,0.5)
		osci_y[i+1] = osci_y[i] + goodwin_y(osci_x[i], osci_y[i])*dt
		osci_z[i+1] = osci_z[i] + goodwin_z(osci_y[i], osci_z[i])*dt 
		X[i+1] = X[i] + f_1(a, b, k, osci_z[i], X[i])*dt + flag_osci*r[i+stepNO]*pow(D_X,0.5)*pow(dt,0.5)#
		#print(X[i+1])
		
		time[i+1] = time[i] + dt #時間(横軸)


	return time, osci_x, osci_y, osci_z, X


def euler(dt, stepNO, n, k, D, a, b, N, flag):

	import numpy as np
	import matplotlib.pyplot as plt
	import math
	import random
	import cmath #a modele for complex number

	###########################################################################################################################
	#パラメーターパート
	#関数の引数の用途を示す

	#dt:for euler's method
	#stepNO:how many step you wanna repeat
	#n:demation
	#k:decompotion rate list
	#D:noise coefficent
	#a:a parameter list
	#b:a parameter list
	#N:randam seed
	#theta:phase
	#flag:with noise:1 or without noise:0


	###########################################################################################################################
	#definition
	#no need to change
	omega = 2*math.pi
	X = [[0] * stepNO for i in range(n)] #row = stepNo, colum = n
	theta = [0] * stepNO
	time = [0]*stepNO #時間

	#defined initial values of x
	for i in range(n): 
		#define A_n
		if i == 0:
			A = a[i]/k[i]
		else :
			A = a[i]/k[i]
			for m in range(0, i):
				mul_b = 1 #multiply values of b
				mul_k = 1 #multiply values of k
				for l in range(m, i):
					mul_b = mul_b*b[l+1]
					mul_k = mul_k*k[l+1]
				A = A + mul_b/mul_k*a[m]/k[m]
				#print(A)
		

		#define b_n...b_1/(k^2_n+womega^2)...(k^2_1+womega^2)
		mul_b = 1 #multiply values of b
		mul_denominator = 1 #multiply values of (k^2_n+womega^2)
		for m in range(0, i+1):
			mul_b = mul_b*b[m]
			mul_denominator = mul_denominator*(k[m]**2+omega**2)
		mul_denominator = mul_b/mul_denominator

		#define C_n
		mul_C1 = -1j/2
		mul_C2 = 1/2
		for m in range(0, i+1):
			mul_C1 = mul_C1*(k[m]-omega*1j)
			mul_C2 = mul_C2*(k[m]+omega*1j)
		C = (mul_C1+mul_C2).real

		#define P_n
		P = A + mul_denominator*C
		print(P)
		X[i][0] = P 
		

	np.random.seed(N)

	def f_1(a, b, k, x,theta):
	    f_1 = a + b*math.sin(theta) - k*x
	    return f_1

	def f_n(a, b, k, x, y):
		f_n = a + b*x - k*y
		return f_n


	###########################################################################################################################
	#オイラー法パート

	for i in range(stepNO-1):
		r = np.random.normal(0,1) #乱数
		theta[i+1] = theta[i] + omega*dt + flag*r*pow(D,0.5)*pow(dt,0.5) #euler for theta
		X[0][i+1] = X[0][i] + f_1(a[0], b[0], k[0], X[0][i], theta[i])*dt #x2のオイラー法
		for j in range(1, n):
			X[j][i+1] = X[j][i] + f_n(a[j], b[j], k[j], X[j-1][i], X[j][i])*dt #x2のオイラー法

		time[i+1] = time[i] + dt #時間(横軸)


	return time, X, theta

#the function to take periods and trough. If you want to start from a time point, you can put any time point to "start" value
def findpeaks(X, maxod, minod):
	import numpy as np
	from scipy import signal
	from scipy.signal import argrelmax
	from scipy.signal import argrelmin
	
	###########################################################################################################################
	#find peaks
	X = np.array(X)
	
	#get index of peak or trough 
	maxid_arr = argrelmax(X, order = maxod) #max 
	minid_arr = argrelmin(X, order = minod) #min

	maxid = list(maxid_arr[0])
	minid = list(minid_arr[0])

	return maxid, minid

def findpeaks_ver2(X, dt):
	import numpy as np
	from scipy import signal
	from scipy.signal import argrelmax
	from scipy.signal import argrelmin
	from scipy.signal import find_peaks
	
	###########################################################################################################################
	#find peaks
	X = np.array(X)
	
	#get index of peak or trough 
	peaks, _ = find_peaks(X, distance=int(0.8/dt))
	trough, _ = find_peaks(X, distance=int(0.8/dt))

	maxid = list(peaks)
	minid = list(trough)

	return maxid, minid

def error_peak(time, X, maxid, minid, start, sample):
	import numpy as np

	#start: the start time to measure periods

	error = 0 #error:1, no error:0

	def find_nearest(array, value)->int:
	    array = np.asarray(array)
	    idx = (np.abs(array - value)).argmin()
	    return idx

	nearest_max_time_id = find_nearest(maxid, start)+1 #the start time will be detected as peak or trough so i put +1
	nearest_min_time_id = find_nearest(minid, start)+1

	#from start point, i want to arrange trough, peak, trough, peak...
	while maxid[nearest_max_time_id]<minid[nearest_min_time_id]:
		nearest_max_time_id = nearest_max_time_id + 1

	#peak an trough is taken correclty?
	for i in range(sample):
		if X[maxid[nearest_max_time_id+i]] <= X[minid[nearest_min_time_id+i]]: #if a trough is bigger than the last peak, it is considered as an error
			error = 1
		if X[maxid[nearest_max_time_id+i]] <= X[minid[nearest_min_time_id+i+1]]: #if a trough is bigger than the next peak, it is considered as an error
			error = 1

	return error

def error_period(period, sample):
	
	error = 0 #error:1, no error:0

	#if period is not in the range of 0.8~1.2, we suppose it isn't taken correctly
	for i in range(sample):
		if period[i]<0.8 or 1.2<period[i]:
			error = 1
			print(period[i])

	return error

def periods(time, id_checkpoint, start, sample):
	import numpy as np

	period = [0]*sample #periods

	#checkpoint: the time to measure periods
	#start: from the "start" point, i start to measure periods

	def find_nearest(array, value)->int:
	    array = np.asarray(array)
	    idx = (np.abs(array - value)).argmin()
	    return idx

	nearest_id = find_nearest(id_checkpoint, start)+1 #the start time will be detected as peak or trough so i put +1

	#conver list to array
	time = np.array(time) #x1は位相なのでsinの値に直した
	id_checkpoint = np.array(id_checkpoint)

	#measure periods from the "start" point
	for i in range(sample):
		period[i] = time[id_checkpoint[i+1+nearest_id]] - time[id_checkpoint[i+nearest_id]]


	return period

def amplitude(X, maxid, minid, start, sample):
	import numpy as np

	period = [0]*sample #periods
	X_peak = 0
	X_trough = 0

	#checkpoint: the time to measure periods
	#start: from the "start" point, i start to measure periods

	def find_nearest(array, value)->int:
	    array = np.asarray(array)
	    idx = (np.abs(array - value)).argmin()
	    return idx

	nearest_max_id = find_nearest(maxid, start)+1 #the start time will be detected as peak or trough so i put +1
	nearest_min_id = find_nearest(minid, start)+1 #the start time will be detected as peak or trough so i put +1

	#conver list to array
	X = np.array(X) #x1は位相なのでsinの値に直した
	maxid = np.array(maxid)
	minid = np.array(minid)

	#measure average of peak and trough
	for i in range(sample):
		X_peak = X_peak + X[maxid[i+nearest_max_id]]
		X_trough = X_trough + X[minid[i+nearest_min_id]]

	amplitude = (X_peak - X_trough)/(2*sample)


	return amplitude


def writedata(filename, data, index):
	import numpy as np
	import pandas as pd

	data = np.array(data)
	index = np.array(index)
	df = pd.DataFrame(data = data, index = index)

	#output file
	df.to_csv(filename)

def readdata(filename):
	import math
	import csv #csv file


	###########################################################################################################################
	#read file 

	#file error
	file = 1 #find file? yes:1 no:0
	error_peak_osci = 0 #error in peak of osci? yes:1 no:0
	error_peak_output = 0 #error in peak of output? yes:1 no:0
	error_period_osci = 0 #error in period of osci? yes:1 no:0
	error_period_output = 0 #error in period of output? yes:1 no:0
	
	
	try: #ファイルがあるとき
		#ファイルの読み込み
		with open(filename, 'rt') as f:
			reader = csv.reader(f)
			access_log = [row for row in reader] #ヘッダー含め全てaccess_logに代入する

		#インデックス部分だけリスト化する。そして、取り出したいデータのヘッダーの列を探索し、列番号をindexに入れる
		index=[r[0] for r in access_log ]

		#戻り値に値を入れている。入るのは"no"か"yes"のみ
		osci_period = access_log[index.index("osci_periods")][1]
		X_period = access_log[index.index("output_periods")][1]
		osci_CV = access_log[index.index("osci_CV")][1]
		X_CV = access_log[index.index("output_CV")][1]
		error_peak_osci = access_log[index.index("osci_error in peak")][1]
		error_peak_output = access_log[index.index("output_error in peak")][1]
		error_period_osci = access_log[index.index("osci_error in period")][1]
		error_period_output = access_log[index.index("output_error in period")][1]
		
		#data written by str, so need to conver str to int or float
		osci_period = float(access_log[index.index("osci_periods")][1])
		X_period = float(access_log[index.index("output_periods")][1])
		osci_CV = float(access_log[index.index("osci_CV")][1])
		X_CV = float(access_log[index.index("output_CV")][1])
		error_peak_osci = int(float(access_log[index.index("osci_error in peak")][1]))
		error_peak_output = int(float(access_log[index.index("output_error in peak")][1]))
		error_period_osci = int(float(access_log[index.index("osci_error in period")][1]))
		error_period_output = int(float(access_log[index.index("output_error in period")][1]))

	except IOError: #if not find file
		#set returns
		file = 0
		osci_period = 0
		X_period = 0
		osci_CV = 0
		X_CV = 0
		

	return file, osci_period, X_period, osci_CV, X_CV, error_peak_osci, error_peak_output, error_period_osci, error_period_output

###########################################################################################################################
#set parameters
e = 39.738012006469100 #chenage sclae of time
dt = 10**(-4)  #for euler's methods
stepNO = 13000000 #how many steps you wanna repeat
#k = [10**(-1.0), 10**(-0.5), 10**(0.0), 10**(0.5), 10**(1.0), 10**(1.5), 10**(2.0), 10**(2.5), 10**(3.0)] #decompotion rate
#k = [10**(0.125), 10**(0.25), 10**(0.375), 10**(0.625), 10**(0.75), 10**(0.875), 10**(1.125), 10**(1.25), 10**(1.375), 10**(1.625), 10**(1.75), 10**(1.875)]
k = [10**(0.5)]
a = 1.0
b = 1.0
flag_osci = 1 #with noise:1 or without noise:0
flag_output = 1 #with noise:1 or without noise:0
N = 1
maxod_osci = 10
minod_osci = 10
#maxod_output = [200, 200, 200, 200, 500, 600, 1000, 1000, 1200]
#inod_output = [200, 200, 200, 200, 500, 600, 1000, 1000, 1200]
#maxod_output = [200, 200, 200, 500, 500, 500, 600, 600, 600, 1000, 1000, 1000]
#minod_output = [200, 200, 200, 500, 500, 500, 600, 600, 600, 1000, 1000, 1000]
maxod_output = [500]
minod_output = [500]
start_time_osci = 200 #from the start time, start to measure periods
start_time_X = 200 #from the start time, start to measure periods
sample = 1000
index = ["osci_periods", "output_periods", "osci_CV", "output_CV", "osci_error in peak", "output_error in peak", "osci_error in period", "output_error in period"]
#print(k)

###########################################################################################################################
#definition
start_time_osci_id = int(start_time_osci/dt)
start_time_X_id = int(start_time_X/dt)
D_osci = 0.000035/4.0
D_output = 0.000001
data = [0]*len(index)
getout = 0

###########################################################################################################################
#write graph
#before make graphs, all data is supposed to be list array or values, none of them is numpy array


for i in range(len(k)):
	while getout < 100:
		time, osci_x, osci_y, osci_z, X = goodwin_functions.goodwin(dt, stepNO, k[i], D_osci, D_output, a, b, N, flag_osci, flag_output, e)

		maxid_osci, minid_osci = goodwin_functions.findpeaks(osci_z, maxod_osci, minod_osci)

		maxid_output, minid_output = goodwin_functions.findpeaks(X, maxod_output[i], minod_output[i])

		period_osci = goodwin_functions.periods(time, maxid_osci, start_time_osci_id, sample)

		period_output = goodwin_functions.periods(time, maxid_output, start_time_X_id, sample)

		peakerror_osci = goodwin_functions.error_peak(time, osci_z, maxid_osci, minid_osci, start_time_osci_id, sample)

		peakerror_output = goodwin_functions.error_peak(time, X, maxid_output, minid_output, start_time_X_id, sample)

		perioderror_osci = goodwin_functions.error_period(period_osci, sample)

		perioderror_output = goodwin_functions.error_period(period_output, sample)

		data[0] = np.average(period_osci)
		data[1] = np.average(period_output)
		data[2] = np.std(period_osci)/np.average(period_osci)
		data[3] = np.std(period_output)/np.average(period_output)
		data[4] = int(peakerror_osci)
		data[5] = int(peakerror_output)
		data[6] = int(perioderror_osci)
		data[7] = int(perioderror_output)

		filename = "" #put your file name in here

		goodwin_functions.writedata(filename, data, index)

		if (peakerror_osci == 0) and (peakerror_output == 0) and (perioderror_osci == 0) and (perioderror_output == 0):
			getout = getout + 1

		N = N + 1

		plt.clf()
		plt.close()

	getout = 0

