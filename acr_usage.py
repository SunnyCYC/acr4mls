# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 20:30:10 2022

@author: CITI
"""


#%%
import os
import numpy as np

from madmom.features.downbeats import RNNDownBeatProcessor as mmRNN
from pathlib import Path
import modules as plp_mod
import glob

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import acr_modules_0525 as acr_module
f_measure_threshold = 0.07


def qual_plots(songinfo, start_frame=1000, dur=10, fps = 100,
               legend_fontsize=16, fontsize=18, figsize = (14, 3), 
               ):
    """ generate qualitative results using songinfo """
    colors = list(mcolors.TABLEAU_COLORS.keys())
    end_frame = start_frame + dur*fps
    plt.figure(figsize = figsize)
    plt.plot(songinfo.acti_max, color = 'red')
    plt.vlines(songinfo.est_beats_dict['HMM']*100, 1, 0.75, label = 'HMM', 
               color = colors[0], linewidth = 5, alpha = 0.7)
    plt.vlines(songinfo.est_beats_dict['HMMT0']*100, 0.75, 0.5, label = 'HMMT0', 
               color = colors[1], linewidth = 5, alpha = 0.7)
    plt.vlines(songinfo.est_beats_dict['PLPDP']*100, 0.25, 0.5, label = 'PLPDP', 
               color = colors[2], linewidth = 5, alpha = 0.7)
    plt.vlines(songinfo.ref_beats[:,0]*100, 0., 1, label = 'Reference Beats', linestyle = 'dashed')
    plt.xlabel('Time Frame (FPS =100)', fontsize = fontsize)
    plt.ylabel('Amplitude', fontsize = fontsize)
    plt.xticks( fontsize = fontsize)
    plt.yticks( fontsize = fontsize)
    plt.legend(bbox_to_anchor=(0.5, 1.2), loc = 'upper center', 
               fontsize = legend_fontsize, frameon = False, ncol=4)
    plt.xlim([start_frame, end_frame])
    return plt

class song_info():
    def __init__(self, songpath, post_types, ref_folder, 
                 min_bpm = 30, max_bpm = 300, fps = 100, dp_factor = 5):
        self.songpath = songpath
        self.ref_folder = ref_folder
        self.annpath = os.path.join(self.ref_folder, 
                                    os.path.basename(self.songpath).replace('.wav', '.beats'))
        self.acti_max = self.get_acti_max()
        self.post_types = post_types
        self.mean_temop = plp_mod.getSongMeanTempo(self.annpath, 
                                           mean_type = 'ori_ibi', 
                                           smooth_winlen = None)
        self.ppt_params = {'min_bpm':min_bpm, 'max_bpm':max_bpm, 
                           'fps': fps, 'dp_factor': dp_factor}
        self.est_beats_dict = self.get_est_beats()
        
        self.ref_beats = self.get_ref_beats()
        

    def get_ref_beats(self):
        print("calculating ref beats...")
        ref_beats = np.loadtxt(self.annpath)
        return ref_beats
    
    def get_acti_max(self):
        print("calculating madmom activation...")
        acti_max = mmRNN()(self.songpath).max(axis=1)
        return acti_max
    def get_est_beats(self):
        print("calculating estimated beats...")
        est_beats_dict = {}
        for post_type in self.post_types:
            beat_est = plp_mod.acti2Est(self.acti_max, post_type,  
                              dp_meantempo = self.mean_temop, **self.ppt_params )
            est_beats_dict[post_type] = beat_est
        return est_beats_dict
#%%
def main():
    ### paths for test songs
    songs = glob.glob(os.path.join('./', 'test_songs', '*.wav'))
    ### types of Post-processing trackers to use
    # post_types = ['SPPK', 'DP', 'PLPDP-sk', 'PLPDP', 'HMM', 'HMMT0']
    post_types = ['DP', 'HMM', 'SPPK']
    ### generate beat estimations and other information for each test song


    song_obj = song_info(songs[0], post_types, './test_songs')

    
    ### save qualitative plots and beat estimations
    fig_out_dir = os.path.join('./', 'test_songs', 'qualitative_plots')
    if not os.path.exists(fig_out_dir):
        Path(fig_out_dir).mkdir(parents = True, exist_ok = True)

    
    songname = os.path.basename(song_obj.songpath)
    print("=========Ploting Beat Estimations...=========")
    print("==={}===".format(songname))
    ### plot ACR results
    L = 2
    start_frame = 1000
    dur = 500
    end_frame = start_frame + dur
    ## prepare estimation dict
    simp_legend = {
    'HMM':'$B_\mathrm{HMM}$', 
    'HMMT0': '$B_\mathrm{HMMT0}$', 
    'PLPDP':'$B_\mathrm{PLPDP}$', 
    'DP':'$B_\mathrm{DP}$', 
    'SPPK':'$B_\mathrm{SPPK}$'}
    est_dict = {}
    for post_type in post_types:
        # break
        est_dict[post_type]=song_obj.est_beats_dict[post_type]

    acr_fig, acr_all = acr_module.plotACR(est_dict, song_obj.ref_beats, song_obj.acti_max,  
                                 start_frame, end_frame, L = L, 
                                 simp_legend = simp_legend)
    print("======Fullrecording ACR results======")
    for PPT in acr_all.keys():
        # break
        PPT_acr = acr_all[PPT]
        print("------{}------".format(PPT))
        for acr_metlev in PPT_acr.keys():
            print("{}:{:.3f}".format(acr_metlev, PPT_acr[acr_metlev]))
    

    fig_dir = os.path.join(fig_out_dir, 
                           'ACR_L{}_s{}_e{}'.format(L, start_frame, end_frame)+songname.replace('.wav', '.png'))
    if not os.path.exists(fig_dir):
        acr_fig .savefig(fig_dir, bbox_inches = 'tight', dpi=600)
    
#%%
if __name__=="__main__":
    main()
