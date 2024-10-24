import math

def yuragidatamake(dt, stepNO, k, D, a, b, N, theta, con_t, x1od,x2od, flag, sample, filename, graph_flag):

	import numpy as np
	import matplotlib.pyplot as plt
	import japanize_matplotlib   
	from scipy import signal
	from scipy.signal import argrelmax
	from scipy.signal import argrelmin
	import random
	import pandas as pd
	import matplotlib as mpl
	import matplotlib.pyplot as plt

	###########################################################################################################################
	#definition of parameter

	#dt:for euler's method
	#stepNO:how many steps you want for euler's method
	#k:degradation rate
	#D:a parameter of clock and output model
	#a:a parameter of clock and output model
	#b:a parameter of clock and output model
	#N:randam seed
	#theta: checkpoint
	#con_t:the time when graph of converge
	#x1od: order os argrelmax and argrelmin for x1
	#x2od: order os argrelmax and argrelmin for x2
	#flag:with noise or without noise yes:1 no:0
	#sample:how many samples you want
	#flag_graph:do you want to show graphs?


	###########################################################################################################################
	#definition
	omega = 2*math.pi
	x1 = [0]*stepNO 
	x2 = [0]*stepNO 
	x2[0] = a/k - (b*omega)/(k*k+omega*omega) #the initial value of x2
	time = [0]*stepNO 
	x1_tcp = [] 
	x2_tcp = [] 
	x1_line_tcp = float(math.sin(theta)) 
	x2_line_tcp = float(a/k+b/(pow((k*k+omega*omega),0.5))*math.sin(theta)) 
	#midleline = float(a/k+b/(pow((k*k+omega*omega),0.5))*math.sin(math.pi)) 
	error_x1_flag = 0 
	error_x2_flag = 0 
	error_x1_tcp_flag = 0 
	error_x2_tcp_flag = 0 
	error_x1_theta_flag = 0 
	error_x2_theta_flag = 0 
	error_output_flag = 0 
	error_osci_flag = 0 
	error_message_x1_peak = "no" 
	error_message_x2_peak = "no" 
	error_message_x1_theta = "no" 
	error_message_x2_theta = "no" 
	error_message_x1 = "no" 
	error_message_x2 = "no" 
	CV = [0]*2 
	Ave = [0]*2 
	angle = round(360.0*theta/(2*math.pi)) 


	np.random.seed(N)

	timecourse_x = []

	def find_nearest(array, value)->int:
	    array = np.asarray(array)
	    idx = (np.abs(array - value)).argmin()
	    return idx

	def f() -> float:
	    f = omega
	    return f

	def g(X,theta):
	    g = a + b*math.sin(theta) - k*X
	    #g = math.sin(theta)
	    return g


	###########################################################################################################################
	#numerical calculation by euler's method 

	for i in range(stepNO-1):
	    r = np.random.normal(0,1) 
	    x1[i+1] = x1[i] + f()*dt + flag*r*pow(D,0.5)*pow(dt,0.5) 
	    x2[i+1] = x2[i] + g(x2[i],x1[i])*dt 
	    time[i+1] = time[i]+dt 

	    x1[i] = math.sin(x1[i])

	x1 = np.array(x1) 
	x2 = np.array(x2)
	t = np.array(time)

	###########################################################################################################################
	#caluculation for CV_output 

	x2_maxid = argrelmax(x2, order = x2od) 
	x2_minid = argrelmin(x2, order = x2od) 

	nearest_max_time_id = find_nearest(x2_maxid, con_t)+1
	nearest_min_time_id = find_nearest(x2_minid, con_t)+1

	while x2_maxid[0][nearest_max_time_id]<x2_minid[0][nearest_min_time_id]:
	    nearest_max_time_id = nearest_max_time_id + 1

	max_time_id = nearest_max_time_id
	min_time_id = nearest_min_time_id

	#search checkpoints
	#0 <= theta < pi/2
	print("checkpoints of output is taken correctly?")
	if (0.0<=theta and theta<math.pi/2.0):

	    for i in range(min_time_id,min_time_id+sample+1): 

	        time_id = x2_minid[0][i]
	        error_x2_tcp_flag = 1 
	        while time_id < x2_maxid[0][max_time_id]: 

	            if (x2[time_id]<=x2_line_tcp and x2_line_tcp < x2[time_id+1]):

	                x2_tcp.append(int((time_id+1))) 

	                error_x2_tcp_flag = 0 

	                break 

	            time_id = time_id + 1

	        if error_x2_tcp_flag == 1:
	            x2_tcp.append(0) 
	            error_x2_theta_flag = 1 

	        max_time_id = max_time_id + 1

	#pi/2 < theta < 3pi/2
	if (math.pi/2.0<theta and theta<math.pi*3.0/2.0):

	    min_time_id = min_time_id + 1 
	    for i in range(max_time_id,max_time_id+sample+1): 

	        time_id = x2_maxid[0][i]
	        error_x2_tcp_flag = 1 
	        while time_id < x2_minid[0][min_time_id]: 

	            if (x2[time_id+1]<x2_line_tcp and x2_line_tcp <= x2[time_id]):

	                x2_tcp.append(int((time_id+1))) 

	                error_x2_tcp_flag = 0 

	                break 

	            time_id = time_id + 1

	        
	        if error_x2_tcp_flag == 1:
	            x2_tcp.append(0) 
	            error_x2_theta_flag = 1 
	            
	        min_time_id = min_time_id + 1
	    
	#3pi/2 < theta < 2pi
	if (math.pi*3.0/2.0<theta and theta<2.0*math.pi):

	    for i in range(min_time_id,min_time_id+sample+1): 

	        time_id = x2_minid[0][i]
	        error_x2_tcp_flag = 1 
	        while time_id < x2_maxid[0][max_time_id]: 

	            if (x2[time_id]<=x2_line_tcp and x2_line_tcp < x2[time_id+1]):

	                x2_tcp.append(int((time_id+1))) 

	                error_x2_tcp_flag = 0 

	                break 

	            time_id = time_id + 1

	        if error_x2_tcp_flag == 1:
	            x2_tcp.append(0) 
	            error_x2_theta_flag = 1 

	        max_time_id = max_time_id + 1


	if error_x2_theta_flag == 0:
	    print ("no errors in checkpoints of output") #エラーメッセージ

	else:
	    print ("rrors in checkpoints of output") #エラーメッセージ
	  
	max_time_id = nearest_max_time_id
	min_time_id = nearest_min_time_id

	print ("/////////")
	print ("peak and trough are taken correctly？")
	for i in range(min_time_id,min_time_id+sample):
	    for j in range(max_time_id,max_time_id+sample):
	        if x2[x2_maxid[0][j]] <= x2[x2_minid[0][i]]: 
	            error_x2_flag = 1 
	        if x2[x2_maxid[0][j]] <= x2[x2_minid[0][i+1]]: 
	            error_x2_flag = 1 

	if error_x2_flag == 0:
	    print ("no errors in peak and trough of output")

	else:
	    print ("errors in peak and trough of output")


	    max_time_id = max_time_id + 1
	    min_time_id = min_time_id + 1


	x2_tcp = np.array(x2_tcp)
	dif = [0]*sample


	#出力系の平均周期とCVを求める
	#ピークの時
	print ("/////////")
	print ("periods of output are taken correctly？")
	if theta == math.pi/2.0:
	    for i in range(sample):
	        dif[i] = t[x2_maxid[0][i+1]] - t[x2_maxid[0][i]]
	        if dif[i]<=0.8 or 1.2<=dif[i]: #周期の異常性を確認している.閾値を0.9と1.1にしているのは理由はないが、10%以上ずれているならおかしいとは思う
	           error_x2_flag = 1 #異常な周期をとったときにメッセージを出す
	        v = np.std(dif)
	        ave = np.average(dif)
	        CV[1] = v/ave
	        Ave[1] = ave

	#トラフの時
	elif theta == math.pi*3.0/2.0:
	    for i in range(sample):
	        dif[i] = t[x2_minid[0][i+1]] - t[x2_minid[0][i]]
	        if dif[i]<=0.8 or 1.2<=dif[i]: #周期の異常性を確認している.閾値を0.9と1.1にしているのは理由はないが、10%以上ずれているならおかしいとは思う
	            error_x2_flag = 1 #異常な周期をとったときにメッセージを出す
	        v = np.std(dif)
	        ave = np.average(dif)
	        CV[1] = v/ave
	        Ave[1] = ave

	#それ以外
	else:
	    for i in range(sample):
	        dif[i] =  t[x2_tcp[i+1]] - t[x2_tcp[i]]
	        if dif[i]<=0.8 or 1.2<=dif[i]: #周期の異常性を確認している.閾値を0.9と1.1にしているのは理由はないが、10%以上ずれているならおかしいとは思う
	            error_x2_flag = 1 #異常な周期をとったときにメッセージを出す
	        v = np.std(dif)
	        ave = np.average(dif)
	        CV[1] = v/ave
	        Ave[1] = ave

	if error_output_flag == 0:
	    print ("no errors in periods of output")

	else:
	    print ("errors in periods of output")


	#if you want to show plotted graph, take out quotation marks
	"""
	#the graph of output(red:peak,blue:trough,yellow:ckeckpoint)
	plt.figure()  
	plt.title("x2(the range of measurement)")
	plt.xlabel("time(the time of converge~after sample*cycles")
	plt.ylabel("x2") 
	plt.plot(t,x2)
	plt.plot(t[x2_maxid],x2[x2_maxid],'ro')
	plt.plot(t[x2_minid],x2[x2_minid],'bo')
	print(theta)
	print(math.pi/2.0)
	if (theta != math.pi/2.0) and (theta!=math.pi*3.0/2.0):
		plt.plot(t[x2_tcp],x2[x2_tcp],'yo')
	plt.xlim(t[x2_minid[0][nearest_min_time_id]],t[x2_maxid[0][nearest_max_time_id+sample]])
	plt.show(block=graph_flag)


	"""
	######################################################################################################################################################################################
	#caluculation for CV_output 

	x1 = np.array(x1) 
	x1_maxid = argrelmax(x1, order = x1od) 

	#ピーク値のインデックスを取得
	x1_maxid = argrelmax(x1, order = x2od)   
	x1_minid = argrelmin(x1, order = x2od) 

	nearest_max_time_id = find_nearest(x1_maxid, con_t)+1 
	nearest_min_time_id = find_nearest(x1_minid, con_t)+1

	while x1_maxid[0][nearest_max_time_id]<x1_minid[0][nearest_min_time_id]:
	    nearest_max_time_id = nearest_max_time_id + 1
  
	max_time_id = nearest_max_time_id
	min_time_id = nearest_min_time_id

	#search checkpoints
	#0 <= theta < pi/2
	print ("/////////")
	print("checkpoints of clock is taken correctly?")
	if (0.0<=theta and theta<math.pi/2.0):

	    for i in range(min_time_id,min_time_id+sample+1):

	        time_id = x1_minid[0][i]
	        error_x1_tcp_flag = 1 
	        while time_id < x1_maxid[0][max_time_id]: 

	            if (x1[time_id]<=x1_line_tcp and x1_line_tcp < x1[time_id+1]):

	                x1_tcp.append(int((time_id+1))) 

	                error_x1_tcp_flag = 0 

	                break 

	            time_id = time_id + 1

	        if error_x1_tcp_flag == 1:
	            x1_tcp.append(0) 
	            error_x1_theta_flag = 1 

	        max_time_id = max_time_id + 1

	#pi/2 < theta < 3pi/2
	if (math.pi/2.0<theta and theta<math.pi*3.0/2.0):

	    min_time_id = min_time_id + 1 
	    for i in range(max_time_id,max_time_id+sample+1): 

	        time_id = x1_maxid[0][i]
	        error_x1_tcp_flag = 1 
	        while time_id < x1_minid[0][min_time_id]: 

	            if (x1[time_id+1]<x1_line_tcp and x1_line_tcp <= x1[time_id]):

	                x1_tcp.append(int((time_id+1))) 

	                error_x1_tcp_flag = 0 

	                break 

	            time_id = time_id + 1

	        if error_x1_tcp_flag == 1:
	            x1_tcp.append(0)
	            error_x1_theta_flag = 1 
	            
	        min_time_id = min_time_id + 1
	    
	#3pi/2 < theta < 2pi
	if (math.pi*3.0/2.0<theta and theta<2.0*math.pi):

	    for i in range(min_time_id,min_time_id+sample+1): 

	        time_id = x1_minid[0][i]
	        error_x1_tcp_flag = 1 
	        while time_id < x1_maxid[0][max_time_id]: 

	            if (x1[time_id]<=x1_line_tcp and x1_line_tcp < x1[time_id+1]):

	                x1_tcp.append(int((time_id+1))) 

	                error_x1_tcp_flag = 0 

	                break 

	            time_id = time_id + 1

	        if error_x1_tcp_flag == 1:
	            x1_tcp.append(0) 
	            error_x1_theta_flag = 1 

	        max_time_id = max_time_id + 1


	if error_x1_theta_flag == 0:
	    print ("no errors in checkpoint of clock") 

	else:
	    print ("errors in checkpoint of clock") 
	  
	max_time_id = nearest_max_time_id
	min_time_id = nearest_min_time_id

	print ("/////////")
	print ("peak and trough of clock are taken correctly？")

	for i in range(min_time_id,min_time_id+sample):
	    for j in range(max_time_id,max_time_id+sample):
	        if x1[x1_maxid[0][j]] <= x1[x1_minid[0][i]]: 
	            error_x1_flag = 1 
	        if x1[x1_maxid[0][j]] <= x1[x1_minid[0][i+1]]: 
	            error_x1_flag = 1 

	if error_x1_flag == 0:
	    print ("no errors in peak and trough of clock")

	else:
	    print ("errors in peak and trough of clock")


	    max_time_id = max_time_id + 1
	    min_time_id = min_time_id + 1


	x1_tcp = np.array(x1_tcp)
	dif = [0]*sample


	#出力系の平均周期とCVを求める
	#ピークの時
	print ("/////////")
	print ("periods of clock are taken correctly？")
	if theta == math.pi/2.0:
	    for i in range(sample):
	        dif[i] = t[x1_maxid[0][i+1]] - t[x1_maxid[0][i]]
	        if dif[i]<=0.8 or 1.2<=dif[i]: 
	           error_x1_flag = 1 
	        v = np.std(dif)
	        ave = np.average(dif)
	        CV[0] = v/ave
	        Ave[0] = ave

	#トラフの時
	elif theta == math.pi*3.0/2.0:
	    for i in range(sample):
	        dif[i] = t[x1_minid[0][i+1]] - t[x1_minid[0][i]]
	        if dif[i]<=0.8 or 1.2<=dif[i]: 
	            error_x1_flag = 1 
	        v = np.std(dif)
	        ave = np.average(dif)
	        CV[0] = v/ave
	        Ave[0] = ave

	#それ以外
	else:
	    for i in range(sample):
	        dif[i] =  t[x1_tcp[i+1]] - t[x1_tcp[i]]
	        if dif[i]<=0.8 or 1.2<=dif[i]: 
	            error_x1_flag = 1 
	        v = np.std(dif)
	        ave = np.average(dif)
	        CV[0] = v/ave
	        Ave[0] = ave

	if error_x1_flag == 0:
	    print ("no errors in periods of clock")

	else:
	    print ("errors in periods of clock")

	#if you want to show plotted graph, take out quotation marks
	"""
	#the graph of clock(red:peak,blue:trough,yellow:ckeckpoint)
	plt.figure()  
	plt.title("x1(the range of measurement)")
	plt.xlabel("time(the time of converge~after sample*cycles")
	plt.ylabel("x1") 
	plt.plot(t,x1)
	plt.plot(t[x1_maxid],x1[x1_maxid],'ro')
	plt.plot(t[x1_minid],x1[x1_minid],'bo')
	if (theta != math.pi/2.0) and (theta!=math.pi*3.0/2.0): 
	    plt.plot(t[x1_tcp],x1[x1_tcp],'yo')
	plt.xlim(t[x1_minid[0][nearest_min_time_id]],t[x1_maxid[0][nearest_max_time_id+sample]]) 
	plt.ylim(-1,1) #x軸の範囲。実際に周期を測る範囲を表示。
	plt.show(block=graph_flag) 
	
	"""
	######################################################################################################################################################################################
	#output a CSV file

	#any errors in peak and trough of clock? 
	if error_x1_flag == 1:
	    error_message_x1_peak = "yes"

	#any errors in checkpoints of clock? 
	if error_x2_theta_flag == 1:
	    error_message_x1_theta = "yes"

	#any errors in peak and trough of output? 
	if error_x2_flag == 1:
	    error_message_x2_peak = "yes"

	#any errors in checkpoints of output? 
	if error_x2_theta_flag == 1:
	    error_message_x2_theta = "yes"

	#any errors in periods of clock? 
	if error_x1_flag == 1:
	    error_message_x1 = "yes"

	#any errors in periods of output? 
	if error_x2_flag == 1:
	    error_message_x2 = "yes"


	df = pd.DataFrame({'period_oscillator':Ave[0], 'CV_oscillator':CV[0], 'period_output':Ave[1], 'CV_output':CV[1], 'abnormal period in osci':error_message_x1, 'abnormal period in output':error_message_x2, 'error in peak or trough of osci':error_message_x1_peak,'error in checkpoints of osci':error_message_x1_theta, 'error in peak or trough of output':error_message_x2_peak,'error in checkpoints of output':error_message_x2_theta}, index=[0])

	df.to_csv(filename)

	print("all complted")

	plt.clf()
	plt.close()

	return error_message_x1_peak, error_message_x1_theta, error_message_x1_theta, error_message_x2_theta, error_message_x1, error_message_x2


####################################################################################################################################################################################
#simulation

###########################################################################################################################
#parameter
dt = 0.0001  #euler's method
stepNO = 20000000 #how many steps you want for euler's method
k = [10**(-0.5), 10**(0.0), 10**(0.5), 10**(1.0), 10**(1.5), 10**(2.0), 10**(2.5), 10**(3.0)] #the list of degradation rate
D = 0.03 
a = 1
b = 1
theta = [0.0/6.0*math.pi, 1.0/6.0*math.pi, 3.0/6.0*math.pi]#checkpoint
con_t = int(0/dt) #time when graph of converge
x1od = 500 #order os argrelmax and argrelmin for x1
x2od = 200 #order os argrelmax and argrelmin for x2
flag = 1 #with noise or without noise yes:1 no:0
sample = 1000 #how many samples you want
graph_flag = False #do you want to show graphs?

###########################################################################################################################

N = 1
getout = 0
for i in range(len(k)):
	for j in range(len(theta)):
		print("k")
		print(k[i])
		getout = 0
		print(getout)
		while getout < 10: #how many bunch of data do you want not included errors
			angle = round(360.0*theta[j]/(2*math.pi))
			filename = "" #put your file name in here

			result = yuragidatamake(dt, stepNO, k[i], D, a, b, N, theta[j], con_t, x1od, x2od, flag, sample, filename, graph_flag)

			if (result[0] == "no") and (result[1] == "no") and (result[2] == "no") and (result[3] == "no") and (result[4] == "no") and (result[5] == "no"):
				getout = getout + 1
				print(getout)

			N = N + 1

