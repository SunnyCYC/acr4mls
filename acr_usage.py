# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 20:30:10 2022
@author: Ching-Yu Chiu (SunnyCYC)
@email:  x2009971@gmail.com
"""


#%%
import os
import numpy as np
from pathlib import Path
import glob
import mir_eval.util as util
import mir_eval
import pandas as pd
import acr_modules as acr_module
f_measure_threshold = 0.07

def RnP(reference_beats, estimated_beats, f_measure_threshold=0.07):
    matching = util.match_events(reference_beats,
                                 estimated_beats,
                                 f_measure_threshold)
    ### prevent zero division:
    if len(estimated_beats)==0:
        precision = 0 ## follow sklearn https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    else:
        precision = float(len(matching))/len(estimated_beats)
    recall = float(len(matching))/len(reference_beats)
    return recall, precision

#%%
def main():
    ### set estimation/ground-truth file folder path
    esti_folder = './estimation_folder'
    gt_folder = './groundtruth_folder'
    ### set information of estimation files
    target_modelname = 'madmom_rnn'
    PostProc = 'hmm'

    
    ### get estimation files and evaluate them
    est_files = glob.glob(os.path.join(esti_folder, "*.beats"))
    songlevel_res = []
    for est_file in est_files:
        gt_file = os.path.join(gt_folder, os.path.basename(est_file))
        # acti_file = os.path.join(acti_folder, os.path.basename(est_file).replace('.beats', '.npy'))
        beat_est = np.loadtxt(est_file)
        beat_gt = np.loadtxt(gt_file)
        # acti_max = np.load(acti_file).max(axis=1)
        
        ### conventional evaluation
        beat_est_fmeasure = mir_eval.beat.f_measure(beat_gt[:, 0], beat_est)
        beat_r, beat_p = RnP(beat_gt[:, 0], beat_est, 
                                         f_measure_threshold=f_measure_threshold)
        CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(beat_gt[:, 0], beat_est)
        
        
        ### save single song eval results
        result_dict = {
                        'songinfo': gt_file, 
                        'beat f-score': beat_est_fmeasure, 
                        'R': beat_r, 
                        'P': beat_p, 
                        'PostProc': PostProc, 
                        'dataset': gt_folder,
                        'model': target_modelname,
                        'CMLc': CMLc, 
                        'CMLt': CMLt, 
                        'AMLc': AMLc, 
                        'AMLt': AMLt,
                        }
    
        ### evaluate one song
        acr_res = acr_module.anyMetLev_eval(beat_est, beat_gt[:, 0], tolerance = 0.07, L =2,
                      half_offbeat = True, double= True, half= True, 
                      third_offbeat = True, triple= True, third=True, 
                      quadruple = True,
                      return_dict = False, quarter = True,
                      return_cframe = True, FPS = 100)
        ### print single song evaluation results
        # for k, v in acr_res['all_ratios'].items():
        #     print("{:16s}:{:.2f}".format(k, v))

        
        ### convert to correct_bool_array and calculate MLS frequency
        c_list = []
        for c_type in acr_module.c_types:
            c_list.append(acr_res['correct_frame_results'][c_type][:, np.newaxis])
        c_array = np.array(c_list).squeeze()
        mls_ratio, sw_count, len_true = acr_module.calMLSfreq(c_array)
        mls_freq = {
            'MLSFQ_r': mls_ratio, 
            'MLS_count': sw_count, 
            'Correct_len': len_true}
        result_dict.update(acr_res['all_ratios'])
        result_dict.update(mls_freq)
        songlevel_res.append(result_dict)

   
    
    ### save qualitative plots and beat estimations
    csv_out_dir = os.path.join('./', 'evaluation', 'quantitative_csvs')
    if not os.path.exists(csv_out_dir):
        Path(csv_out_dir).mkdir(parents = True, exist_ok = True)
    csv_fname = target_modelname +"_songlevel_"+PostProc+".csv"
    csv_spath = os.path.join(csv_out_dir, csv_fname)

    df = pd.DataFrame(songlevel_res)
    df.to_csv(csv_spath)
    
   
    

    
#%%
if __name__=="__main__":
    main()
