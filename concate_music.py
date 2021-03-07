
from utils import *
import numpy as np
import pydub
core_music_list = ['0.mp3']
core_time_list = ['0.txt']

concat_music_list = ['1.mp3','2.mp3','3.mp3','4.mp3','5.mp3']
concat_time_list = ['1.txt','2.txt','3.txt','4.txt','5.txt']

core_music_split = mp3_split(core_music_list,core_time_list)

concat_music_split = mp3_split(concat_music_list,concat_time_list)

W_kl_self = kl_edge(concat_music_split,concat_music_split)
W_kl_core = kl_all(concat_music_split,core_music_split)
init_weight = np.ones(shape=np.shape(W_kl_core));
loss = loss_with_kl(W_kl_self,W_kl_core,init_weight)
print 'init loss is'+str(loss)
final_weight = learning(W_kl_self,W_kl_core,init_weight)

print np.argmax(final_weight,axis=0)
recover_mp3(np.argmax(final_weight,axis=0),concat_music_split,mp3_filename="final.wav")
sound = pydub.AudioSegment.from_wav("final.wav")
sound.export("final.mp3", format="mp3")
print "done!"

