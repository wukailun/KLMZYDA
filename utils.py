import numpy as np

from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib.pyplot as plt

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum    
    return s

def numerical_gradient(f,y,z,x):

    h=1e-4#0.0001
    grad=np.zeros_like(x)

    for idx in range(x.size):
    	x = np.reshape(x,[107*12])
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        x = np.reshape(x,[107,12])
        fxh1=f(y,z,x)

        x = np.reshape(x,[107*12])
        x[idx]=tmp_val-h
       	x = np.reshape(x,[107,12])
        fxh2=f(y,z,x)


        grad = np.reshape(grad,[107*12])
        grad[idx]=(fxh1-fxh2)/(2*h)
        grad = np.reshape(grad,[107,12])

        x = np.reshape(x,[107*12])
        x[idx]=tmp_val 
        x = np.reshape(x,[107,12])
    return grad

def norm_kl(my_matrix_x,my_matrix_y):
	my_matrix_x = np.mean(np.abs(my_matrix_x),axis=1)+0.0000000001
	my_matrix_x = my_matrix_x/np.sum(my_matrix_x)
	my_matrix_y = np.mean(np.abs(my_matrix_y),axis=1)+0.0000000001
	my_matrix_y = my_matrix_y/np.sum(my_matrix_y)
	return np.sum(np.abs(my_matrix_x)*np.log(np.abs(my_matrix_x)/(np.abs(my_matrix_y)+0.000000001)+0.000000001))
def my_spec(vec):
	spectrum,freqs,ts,fig = plt.specgram(vec,NFFT=512,Fs=44100)
	return spectrum
def mp3_split(music_list,time_list):
	all_audio = []
	for music,time in zip(music_list,time_list):
		song = AudioSegment.from_mp3(music)
		wav_file = 'temp.wav'
		song.export(wav_file , format="wav")
		sampling_freq,audio =wavfile.read(wav_file)
		assert sampling_freq==44100
		file_readlines = open(time)
		my_list = file_readlines.readlines()
		times_list = []
		audio_list = []
		for my_times in my_list:
			if ('[' not in my_times) or ('[' not in my_times):
				continue
			my_times = (my_times.split(']')[0]).split('[')[1]
			my_times = int(my_times.split(":")[0])*60+float(my_times.split(":")[1])
			times_list.append(my_times)
		for my_index in range(0,len(times_list)-1):
			start_index = int(times_list[my_index]*sampling_freq)
			end_index = int(times_list[my_index+1]*sampling_freq)
			audio_list.append(audio[start_index:end_index,:])
		all_audio.append(audio_list)
		file_readlines.close()
	return all_audio

def kl_edge(left_music_split,right_music_split):
	music_index = 0
	anti_music_index = 0
	number_x = 0
	number_y = 0
	SEC =2.0
	spec_x = []
	spec_y = []
	for music in left_music_split:
		for split_x in music:
			number_x+=1
			print np.shape(split_x)
			spec_x.append(my_spec(split_x[int(-44100*SEC):,0]))

	for music in right_music_split:
		for split_y in music:
			print np.shape(split_y)
			spec_y.append(my_spec(split_y[:int(44100*SEC),0]))
			number_y+=1
	music_split_kl = np.ones(shape=[number_x,number_y])*999999
	number_x = 0
	number_y = 0
	for music_x in left_music_split:
		for split_x in music_x:
			if np.shape(split_x)[0]<=88200:
				continue;
			print number_x
			for music_y in right_music_split:
				for split_y in music_y:
					if np.shape(split_y)[0]<=88200:
						continue;
					spec_x_split = spec_x[number_x]
					spec_y_split = spec_y[number_y]
					music_split_kl[number_x,number_y] = 0.0
					music_split_kl[number_x,number_y] += norm_kl(spec_x_split,spec_y_split)

					if number_x == number_y:
						music_split_kl[number_x,number_y]+=999999;
					if music_index == anti_music_index:
						music_split_kl[number_x,number_y] *= 3;
					number_y+=1
				anti_music_index += 1

			number_y = 0
			number_x += 1
		music_index += 1
	for i in range(0,np.shape(music_split_kl)[0]):
		for j in range(0,np.shape(music_split_kl)[1]):
			if np.isnan(music_split_kl[i,j]):
				music_split_kl[i,j] = 999999
	return music_split_kl

def kl_all(left_music_split,right_music_split):
	music_index = 0
	anti_music_index = 0
	number_x = 0
	number_y = 0
	SEC =2.0
	spec_x = []
	spec_y = []
	for music in left_music_split:
		for split_x in music:
			number_x+=1
			print np.shape(split_x)
			spec_x.append(my_spec(split_x[:,0]))

	for music in right_music_split:
		for split_y in music:
			print np.shape(split_y)
			spec_y.append(my_spec(split_y[:,0]))
			number_y+=1
	music_split_kl = np.ones(shape=[number_x,number_y])*999999
	number_x = 0
	number_y = 0
	for music_x in left_music_split:
		for split_x in music_x:
			if np.shape(split_x)[0]<=256:
				continue;
			print number_x
			for music_y in right_music_split:
				for split_y in music_y:
					if np.shape(split_y)[0]<=88200:
						continue;
					spec_x_split = spec_x[number_x]
					spec_y_split = spec_y[number_y]
					music_split_kl[number_x,number_y] = 0.0
					music_split_kl[number_x,number_y] += norm_kl(spec_x_split,spec_y_split)
					number_y+=1
				anti_music_index += 1

			number_y = 0
			number_x += 1
		music_index += 1
	for i in range(0,np.shape(music_split_kl)[0]):
		for j in range(0,np.shape(music_split_kl)[1]):
			if np.isnan(music_split_kl[i,j]):
				music_split_kl[i,j] = 999999
	return music_split_kl
def loss_with_kl(W_kl_self,W_kl_core,weight):
	weight = softmax(weight)
	loss_1 = np.sum(weight*W_kl_core)
	loss_2 = np.sum(np.dot(np.transpose(weight),W_kl_self)[:-1,:]*np.transpose(weight[:,1:]))
	return (loss_1+loss_2)

def learning(W_kl_self,W_kl_core,init_weight):
	W = init_weight
	for i in range(0,100):
		W-=0.00001*numerical_gradient(loss_with_kl,W_kl_self,W_kl_core,W)
		print loss_with_kl(W_kl_self,W_kl_core,W)
	return W
def recover_mp3(list_y,concat_music_split,mp3_filename="final.wav"):
	cnt = 0
	l_old = 9999
	old = np.zeros(shape=[1,2])
	concat_music_split[0]
	for l in list_y:
		if l_old == l:
			continue
		for music in concat_music_split:
			for split in music:
				if l==cnt:
					print "all"+str(l)+" "+str(cnt)
					old = np.concatenate((old,split),axis = 0)
				cnt+=1
		cnt = 0
		l_old = l
	wavfile.write(mp3_filename,44100,old)




