# -*- coding: utf-8 -*-
"""
0810: fixed empty slicing warning: len(var_template)>0 --> >1
0824: added MLS Frq cal
0825: redesign raioPlotter
@author: CITI
"""
#%%
import numpy as np
import os

from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


### Grouping all metrical levels into 8 cases
tempo_group = {
    ### original tempo cases
    'onbeat':['original'],
    'offbeat': [ 'half_offbeat', 
                  'onethird_offbeat', 'twothird_offbeat'], 
    ### harmonic tempo cases
    'double': ['double'], 
    'triple': ['triple'],
    'quadruple': ['quadruple'],
    ### subharmonic tempo cases
    ## ACR will check step-by-step for each ref beat, 
    ## therefore cases like "half offbeat", "one-third offbeat" , 
    ## "two-third offbeat", ... will all be considered
    'half': ['half'], 
    'third': ['third'], 
    'quarter':['quarter'],
    }



colors = list(mcolors.TABLEAU_COLORS.keys())

# simp_legend = {
#     'HMM':'$B_\mathrm{HMM}$', 
#     'HMMT0': '$B_\mathrm{HMMT0}$', 
#     'PLPDP':'$B_\mathrm{PLPDP}$', 
#     'DP':'$B_\mathrm{DP}$', 
#     'SPPK':'$B_\mathrm{SPPK}$'}

y_tick_dict ={
    'onbeat':'onbeat', 
    'offbeat':'offbeat', 
    'double':'double', 
    'triple':'triple', 
    'quadruple':'quadruple',  
    'half':'half', 
    'third':'third', 
    'quarter': 'quarter', 
    'any': 'any'} 


fontsize = 16
acti_label = '$\Delta(n)$'
# acti_label = 'Beat Activation $\Delta(n)$'
# beat_label = '$\Delta^{ref}(n)$'
beat_label = 'Reference Beats $B$'
c_types=['onbeat', 'offbeat', 'half', 
         'double', 'third', 'triple',  'quarter', 'quadruple', 
         'any']
#%%


def raioPlotter(cdicts, est_dict, beats_ann, title = None,
                start_frame = None, end_frame = None, 
                constrained_layout= False, acti = None, 
                fontsize = fontsize, FPS = 100, c_types =c_types,):
    ### modify the legend
    simp_legend = {k: '$B_\mathrm{'+k+'}$' for k in cdicts.keys()}

    
    if not start_frame:
        start_frame = 0
    if not end_frame:
        end_frame = int(beats_ann[-1]*FPS)+1
    fig, ax = plt.subplots(len(cdicts)+1, 1, figsize = (8,6), 
                        constrained_layout=constrained_layout)
    
    ### set vertical ticks
    
    c_ticks = {y_tick_dict[y]:ind for ind, y in enumerate(c_types)}
    if not isinstance(acti, type(None)):
        start_pos = -1
    else:
        start_pos = 0
        
    ax2 = ax[0].twinx()
    for ind2, (post_proc, est) in enumerate(est_dict.items(), ):
        # break
        ax2.vlines(est*FPS, ind2+start_pos, ind2+0.7+start_pos, label = simp_legend[post_proc], 
                linewidth = 3, linestyle = 'solid', color = colors[ind2])
    ax2.vlines(beats_ann*FPS, start_pos, ind2+0.7, label = beat_label, 
                   linestyle = 'dashdot', linewidth = 2, color = 'black')
    ax2.set_ylim(ax2.get_ylim()[::-1])
    
    
    if not isinstance(acti, type(None)):
        
        ax[0].plot(acti[:], label = acti_label, color = 'red', 
                 linestyle = 'solid', alpha = 0.8)
        ax[0].set_ylim([0, 1])
        ax[0].set_ylabel('Acti. Strength', fontsize = fontsize)
        ax[0].get_yaxis().set_label_coords(-0.2,0.5)
        ax[0].tick_params(axis = 'y', labelsize = fontsize)

        
    ax[0].set_xlim([start_frame, end_frame])
    
    ax[0].tick_params(axis = 'x', labelsize = fontsize)
    
    ax[0].axes.yaxis.set_visible(True)
    ax2.axes.yaxis.set_visible(False)
    for ind, (post_proc, c_arr) in enumerate(cdicts.items(), 1):
        ax[ind].set_yticks(range(len(c_ticks)) )
        ax[ind].tick_params(axis = 'y', rotation = 30)
        ax[ind].set_yticklabels(c_ticks, fontsize = fontsize)
        im = ax[ind].imshow(c_arr, interpolation='none', 
                            cmap=plt.cm.get_cmap('BuGn', 2)) ## for c_array
        #also plot beat ann at acr vis plots
        ax[ind].vlines(beats_ann*FPS, -0.5, 8.5, 
                   linestyle = 'solid', linewidth = 2, color = 'gray', alpha=0.67)
        ax[ind].set_aspect('auto')
        ax[ind].set_xlim([start_frame, end_frame])
        ax[ind].grid(axis='y', linewidth = 1)
        ax[ind].tick_params(axis = 'x', labelsize = fontsize,)
        ax[ind].set_title(simp_legend[post_proc], fontsize = fontsize)
        ax[ind].set_ylabel('Metrical Level '+'$\Phi$', fontsize = fontsize)
        im.set_clim(-0, 1)
    ax[ind].set_xlabel('Time Frame (FPS = 100)', fontsize = fontsize)
    ### legend
    handles, labels = fig.gca().get_legend_handles_labels()
    handle, label = ax[0].get_legend_handles_labels()
    handles.append(handle[0])
    labels.append(label[0])
    # print("len hanles:", len(handles))
    # print(labels)
    
    fig.legend(loc='upper center', bbox_to_anchor = (0.6, 1.04), 
                      frameon=False, fontsize = fontsize, ncol=3, 
                      handletextpad=0.3, columnspacing= 0.7)
    # fig.legend([handles[idx] for idx in [3, 4]],[labels[idx] for idx in [3, 4]], 
    #         loc='upper center', bbox_to_anchor = (0.5, 1.02), 
    #                    frameon=False, fontsize = fontsize, ncol = 2)
    # fig.legend([handles[idx] for idx in [0, 1, 2]],[labels[idx] for idx in [0, 1, 2]], 
    #         loc='upper center', bbox_to_anchor = (0.5, 1.05), 
    #                   frameon=False, fontsize = fontsize, ncol = 3)
    
    
    fig.tight_layout()
    
    return fig
#%%
def faster_variant(sequence, double=False, triple=False, 
                   quadruple = True, allow_extrap = False):
    """
    original source code: https://github.com/CPJKU/madmom/blob/master/madmom/evaluation/beats.py
    Create more harmonic temo variations of the given beat sequence.
    
    Parameters
    ----------
    sequence : np.ndarray
        Beat sequence.
    double : bool, optional
        Create a double tempo sequence.
    triple : bool, optional
        Create triple tempo sequence.
    quadruple : bool, optional
        Create quadruple tempo sequence.
    Returns
    -------
    dict
        Beat sequence variations.
    """
    # create different variants of the annotations
    sequences = {'original': sequence}
    # quadruple tempo beat_ref variation
    if quadruple:
        if len(sequence) == 0:
            # if we have an empty sequence, there's nothing to interpolate
            quadruple_sequence = []
        else:
            # create a sequence with quadruple tempo
            same = np.arange(0, len(sequence))
            # request one item less, otherwise we would extrapolate
            if allow_extrap:
                shifted = np.arange(0, len(sequence), 0.25)
            else:
                shifted = np.arange(0, len(sequence), 0.25)[:-3]

            f = interpolate.interp1d(same, sequence, fill_value='extrapolate')
            quadruple_sequence = f(shifted)
            sequences.update({'quadruple':quadruple_sequence})
    
    # double
    if double : ### both double and offbeat require interp in between 2beats
        if len(sequence) == 0:
            # if we have an empty sequence, there's nothing to interpolate
            double_sequence = []
        else:
            # create a sequence with double tempo
            same = np.arange(0, len(sequence))
            # request one item less, otherwise we would extrapolate
            if allow_extrap:
                shifted = np.arange(0, len(sequence), 0.5)
            else:
                shifted = np.arange(0, len(sequence), 0.5)[:-1]

            f = interpolate.interp1d(same, sequence, fill_value='extrapolate')
            double_sequence = f(shifted)
        sequences.update({'double':double_sequence})

    # triple tempo variations
    if triple:
        if len(sequence) == 0:
            # if we have an empty sequence, there's nothing to interpolate
            triple_sequence = []
        else:
            # create a annotation sequence with triple tempo
            same = np.arange(0, len(sequence))
            # request two items less, otherwise we would extrapolate
            if allow_extrap:
                shifted = np.arange(0, len(sequence), 1. / 3)
            else:
                shifted = np.arange(0, len(sequence), 1. / 3)[:-2]

            f = interpolate.interp1d(same, sequence, fill_value='extrapolate')
            triple_sequence = f(shifted)
        sequences.update({'triple':triple_sequence})
    
    return sequences

def offbeat_variant(sequence, half_offbeat =False, 
                  third_offbeat = True,):
    """
    original source code: https://github.com/CPJKU/madmom/blob/master/madmom/evaluation/beats.py
    Create original tempo offbeat variations of the given beat sequence.
    
    Parameters
    ----------
    sequence : np.ndarray
        Beat sequence.
    half_offbeat : bool, optional
        Create an half_offbeat sequence.
    third_offbeat : bool, optional
        Create third_offbeat sequences.
   
    Returns
    -------
    dict
        Beat sequence variations.
    """
    # create different variants of the annotations
    sequences = {}

    # half offbeat variation
    if half_offbeat: ### both double and offbeat require interp in between 2beats
        if len(sequence) == 0:
            # if we have an empty sequence, there's nothing to interpolate
            halfoff_sequence = []
        else:
            # create a sequence with double tempo
            same = np.arange(0, len(sequence))
            f = interpolate.interp1d(same, sequence, fill_value='extrapolate')
            ## do not extrapolate for offbeat
            shift_double_seq = f(np.arange(0, len(sequence), 0.5)[:-1])
            halfoff_sequence = shift_double_seq[1::2]
        sequences.update({'half_offbeat':halfoff_sequence})

    # triple/third tempo variations
    if third_offbeat:
        if len(sequence) == 0:
            # if we have an empty sequence, there's nothing to interpolate
            onethirdoff_seq = []
            twothirdoff_seq = []
        else:
            # create a annotation sequence with triple tempo
            same = np.arange(0, len(sequence))         
            # triple_sequence = np.interp(shifted, same, sequence)
            f = interpolate.interp1d(same, sequence, fill_value='extrapolate')
            ## do not extrapolate for offbeat
            shift_triple_seq = f(np.arange(0, len(sequence), 1. / 3)[:-2])
            onethirdoff_seq = shift_triple_seq[1::3]
            twothirdoff_seq = shift_triple_seq[2::3]
        sequences.update({'onethird_offbeat':onethirdoff_seq})
        sequences.update({'twothird_offbeat':twothirdoff_seq})
    return sequences

#### a function modified from Peter's L-correct
def countSegCorrect(ref_var, ref_ann, beat_est, tolerance = 0.07, 
                    empty_ref_return = False):
    """
    check if the estimated beats (beat_est) match the reference beat variant (ref_var).
    
    
    Parameters
    ----------
    ref_var : np.ndarray
        reference beat variants. e.g., half tempo variants of reference beats.
    ref_ann : np.ndarray
        original reference beats. 
        Though we have ref_variants, we still have to check within L beat_ann. 
        So we'll need ref_ann to set upper/lower bounds of set this L beats regions to 
        select beat_est for evaluation.
    beat_est : np.ndarray
        full sequence of estimated beats.
    tolerance : float, optional
        size of tolerance window in seconds. The default is 0.07.
    empty_ref_return : bool, optional
        output value when the input reference beat seuqnence is empty. 
        The default is False.

    Returns True (correct) or False (incorrect)
    -------
    TYPE
        bool value.

    """
    
    ## find time range of the L correct beats for this segment 
    if len(ref_var) == 0:
        """for local case, e.g. 4 consecutive beat annotations, the twothird ref variants may be empty, 
        we do not want to count this as correct. (Since this will increase F-any unreasonably)
        """
        return empty_ref_return
    ### for half/third tempo, ref_var will span longer than ref_ann, 
    ### so need to consider ref_var
    start =min( ref_ann[0], ref_var[0]) - tolerance
    end = max(ref_ann[-1], ref_var[-1]) + tolerance 
    ### extract the estimated beats inside the above (start, end) range
    detectedBeats_ind = (start<=beat_est)& (beat_est<=end)
    detectedBeats = beat_est[detectedBeats_ind]
    
    tmp = 0 ## correct or not
    if len(ref_var)!= len(detectedBeats):
        ### not L-correct, directly pass
        return False
    else:
        tmp = 1 ### assume this segment is L-correct, and conduct further check
        for ref_ind in range(len(ref_var)):
            # break
            toll = ref_var[ref_ind]- tolerance # lower bound
            tolh = ref_var[ref_ind]+ tolerance # higher bound
            ## if number of detectedBeats in between lower/higher bounds !=1, not correct
            est_within_id = (detectedBeats>=toll) &(detectedBeats<=tolh)
            if sum(est_within_id)!=1:
                tmp = 0
    if tmp ==1:
        return True
    else:
        return False

def calRatio(segCorrect, eval_type = 'onbeat'):
    """ 
    calculate the coverage ratio using the input segCorrect
        input:
            segCorrect (np.array, dtype = bool): indicating the correct beats,  
            eval_type (str): could be 'onbeat', 'half', 'third', 'any'
        output:
            result (dict). 
            A dictionary containing eval_type and correpsonding coverage ratio
    """

    ratio = sum(segCorrect)/len(segCorrect)
    result = {
            'Ratio-'+eval_type : ratio,

        }

    return result

def multiBitOr(itemlist):
    """
    calculate the union among the items in itemlist

    Parameters
    ----------
    itemlist : list
        Each element of the list corresponds to a sequence of bool values of
        a specific metrical level condition.

    Returns
    -------
    first : np.ndarray
        a sequence of bool values indicating the union result of the input itemlist.

    """
    first = itemlist[0]
    for next_item in itemlist[1:]:
        first = first | next_item
    return first

def anyMetLev_eval(beat_est, beat_ref, tolerance = 0.07, L =2,
                  half_offbeat = True, double= True, half= True, 
                  third_offbeat = True, triple= True, third=True, 
                  quadruple = True,
                  return_dict = False, quarter = True,
                  return_cframe = False, FPS = 100):
    """
    

    Parameters
    ----------
    beat_est : np.ndarray
        array of estimated beat positions (sec).
    beat_ref : np.ndarray
        array of reference beat positions (sec).
    tolerance : float, optional
        tolerance for evaluation (sec). The default is 0.07.
    L : int, optional
        number of continuous beats required to be correct. The default is 2.
    half_offbeat : bool, optional
        The default is True.
    double :  bool, optional
        The default is True.
    half :  bool, optional
        The default is True.
    third_offbeat :  bool, optional
        The default is True.
    triple :  bool, optional
        The default is True.
    third :  bool, optional
        The default is True.
    quadruple :  bool, optional
        The default is True.
    return_dict : bool, optional
        The default is False.
    quarter :  bool, optional
        The default is False.
    return_cframe :  bool, optional
        Return the correct frames. The default is False.
    FPS : int, optional
        Frame-per-second. The default is 100.

    Returns
    -------
    return_dict (dictionary) that contains the following items:
    all_ratios: (dictionary) that contains the metrical level groups and their 
    corresponding annotation coverage ratio.
        
    correct_type_results: (dictionary) that contains the metrical level groups
    and their corresponding array of bool for the 'L-detected' reference beats
        
    correct_frame_results: (dictionary) that contains the metrical level groups 
    and their corresponding array of bool (with time frame as x-axis) for the 
    'L-detected' frames.

    """
    ### prepare index-start and index-end for each frame's evaluation
    segStarts = np.arange(0, len(beat_ref)-(L-1)).tolist()
    segEnds = np.arange(L, len(beat_ref)+1).tolist()
    ### slower tempos
    segEnds_half = np.arange(L*2-1, len(beat_ref)+1).tolist()
    segEnds_third = np.arange(L*3-2, len(beat_ref)+1).tolist()
    segEnds_quarter = np.arange(L*4-3, len(beat_ref)+1).tolist()
    ## for final l-1 frames, use final beat
    if len(segEnds)< len(segStarts):
        segEnds +=[segStarts[-1]+1]*(len(segStarts)-len(segEnds)) 

    ### creat arrays to save correct beats
    variant_dict = {
        'original': True, 
        'half_offbeat': half_offbeat, 
        'double': double, 
        'half': half, 
        'onethird_offbeat': third_offbeat, 
        'twothird_offbeat': third_offbeat, 
        'triple': triple, 
        'third': third,
        'quadruple':quadruple,
        'quarter': quarter,
        }
    correct_dict = {}
    for var_type, bool_v in variant_dict.items():
        if bool_v:
            correct_dict[var_type] = np.zeros(len(segStarts))
    ### create arrays to save correct frames        
    if return_cframe:
        cf_correct_dict = {}
        end_frame = int(beat_ref[-1]*FPS)
        for var_type, bool_v in variant_dict.items():
            if bool_v:
                cf_correct_dict[var_type] = np.zeros(end_frame+1)
    
    ### for each step/segment(frame or beat), 
    ### extract corresponding reference beats and check if L-correct
    for segment in segStarts:
        # break
        # print("start:{}, end:{}".format(segStarts[segment], segEnds[segment]))
        ### original tempo reference beats
        reference = beat_ref[segStarts[segment]:segEnds[segment]]
        ### for original tempo off-beat cases, one more reference beat is required
        ### for interpolation. (Extrapolation is not used in this work concerning
        ### the larger tempo variation)
        ref_offbeat = beat_ref[segStarts[segment]:segEnds[segment]+1]
        ### need to ensure len(ref)>=2 to allow calculation of interval
        if len(reference)<2:
            # break
            # reference = beat_ref[segStarts[segment]-1:segEnds[segment]]
            print('ref too short at seg:{}'.format(segment))
            break
            

        #### collect all reference beat variants into compound_ref_dict
        compound_ref_dict = faster_variant(reference, double=double, 
                                           triple=triple, quadruple = quadruple, 
                                           allow_extrap = False)

        compound_ref_dict.update(offbeat_variant(ref_offbeat, 
                                                 half_offbeat =half_offbeat, 
                  third_offbeat = third_offbeat,))
        ### use longer reference for slower tempo (i.e., half, third, quarter )
        ## only deal with slower tempo when ref is long enougth
        if half:
            if segment < len(segEnds_half):
                ref_half = beat_ref[segStarts[segment]:segEnds_half[segment]]
                compound_ref_half = {
                    'half': ref_half[0::2], 
                     }
    
            else:
                compound_ref_half = {
                    'half': [], 
                     }
            compound_ref_dict.update(compound_ref_half)
        if third:
            if segment < len(segEnds_third):
                ref_third = beat_ref[segStarts[segment]: segEnds_third[segment]]
                compound_ref_third = {
                    'third': ref_third[0::3], 
                    }
            else: 
                compound_ref_third = {
                    'third': [], 
                    }
            compound_ref_dict.update(compound_ref_third)
        if quarter:
            if segment < len(segEnds_quarter):
                ref_quarter = beat_ref[segStarts[segment]: segEnds_quarter[segment]]
                compound_ref_quarter = {
                    'quarter': ref_quarter[0::4], 
                    }
            else: 
                compound_ref_quarter = {
                    'quarter': [], 
                    }
            compound_ref_dict.update(compound_ref_quarter)
        
        ### for each variant type, check if L-correct and update the correct_array
        for var_type, var_correct_arr in correct_dict.items():
            # break
            ### use variant_template to determine stricter tolerance
            var_template = compound_ref_dict[var_type]
            if len(var_template)>1 : # not empty, or with only one beat (0810 modified)
                mean_ibi = (var_template[1:]-var_template[:-1]).mean() ## second
                tol = min(tolerance, 0.175*mean_ibi)
            else:
                mean_ibi = None ## will not be used
                tol = 0.07 
            
            if countSegCorrect(var_template, reference, beat_est,
                               tolerance = tol, 
                               empty_ref_return = False):
                if var_type == 'half':
                    var_correct_arr[segStarts[segment]:segEnds_half[segment]] = 1
                elif var_type =='third':
                    var_correct_arr[segStarts[segment]:segEnds_third[segment]] = 1
                elif var_type =='quarter':
                    var_correct_arr[segStarts[segment]:segEnds_quarter[segment]] = 1
                else:
                    var_correct_arr[segStarts[segment]:segEnds[segment]] = 1
                correct_dict[var_type]=var_correct_arr
                ### also generate correct array with time frame as y-axis
                if return_cframe:
                    cframe_array = cf_correct_dict[var_type]
                    seg_staframe = int(beat_ref[segStarts[segment]]*FPS)
                    if var_type == 'half':
                        seg_endframe = int(beat_ref[segEnds_half[segment]-1]*FPS)
                        cframe_array [seg_staframe:seg_endframe] = 1
                    elif var_type =='third':
                        seg_endframe = int(beat_ref[segEnds_third[segment]-1]*FPS)
                        cframe_array [seg_staframe:seg_endframe] = 1
                    elif var_type =='quarter':
                        seg_endframe = int(beat_ref[segEnds_quarter[segment]-1]*FPS)
                        cframe_array [seg_staframe:seg_endframe] = 1
                    else:
                        seg_endframe = int(beat_ref[segEnds[segment]-1]*FPS)
                        cframe_array [seg_staframe:seg_endframe] = 1
                    cf_correct_dict[var_type]=cframe_array
                

    for k, d in correct_dict.items():
        # break
        correct_dict[k] = np.array(d, dtype= bool)
        if return_cframe:
            cf_correct_dict[k] = np.array(cf_correct_dict[k], 
                                          dtype = bool)
    
    # organize results based on tempo grouping
    tempo_group['any'] = list(correct_dict.keys())
    correct_type_results = {}
    correct_frame_results = {} ## only use it if return_cframe
    for correct_type, var_list in tempo_group.items():

        cdict_list = [correct_dict[i] for i in var_list]
        correct_result = multiBitOr(cdict_list)
        correct_type_results[correct_type] = correct_result
        ### for frame-based correct_results
        if return_cframe:
            cf_dict_list = [cf_correct_dict[i] for i in var_list]
            cf_results=  multiBitOr(cf_dict_list)
            correct_frame_results[correct_type] = cf_results
            
    all_results = {}
    for tempo_type, segC in correct_type_results.items():
        all_results.update(calRatio(segC, eval_type = tempo_type))
    
    return_dict = {
        'all_ratios': all_results, 
        'correct_beat_results': correct_type_results if return_dict else None, 
        'correct_frame_results': correct_frame_results if return_cframe else None, }
    
    return return_dict



# convert est_dict to correct_frame image
def est_dict2carrays(est_dict, beat_ref, L=2, c_types = c_types):
    c_dicts = {} ## dict to save arrays of correct frames
    acr_all = {}
    for post_type in est_dict.keys():
        # break
        b_est =  est_dict[post_type]
        return_dict = anyMetLev_eval(b_est, beat_ref, tolerance = 0.07, 
                      half_offbeat = True, double= True, half= True, L =L, 
                      third_offbeat = True, triple= True, third= True, quarter = True,
                      return_dict = True, return_cframe = True)
        acr_all[post_type] = return_dict['all_ratios']
        c_list = []
        for c_type in c_types:
            c_list.append(return_dict['correct_frame_results'][c_type][:, np.newaxis])
        c_array = np.array(c_list).squeeze()
        c_dicts[post_type] = c_array
    return c_dicts, acr_all

def plotACR(est_dict, beats_ann, acti, start_frame, 
            end_frame, L = 2, c_types = c_types):
    c_dicts, acr_all = est_dict2carrays(est_dict, beats_ann[:, 0], L, c_types)
    fig = raioPlotter(c_dicts, est_dict, beats_ann[:, 0], title = None,
                start_frame = start_frame, end_frame = end_frame, 
                constrained_layout= False, acti = acti, 
                fontsize = fontsize, FPS = 100, c_types =c_types,
                )
    return fig, acr_all

  
def calMLSfreq(c_arr):
    ### get any-metric-level correct row
    any_lev = c_arr[8, :]
    ind_true = np.where(any_lev!=0)[0]
    ## if no correct cases, return 0
    if len(ind_true)==0:
        return 0, 0, 0
    ### get current correct metric-level, and check the next
    cur_lev = np.where(c_arr[:8, ind_true[0]]==True)[0]
    switch_count = 0
    for ind in range(1, len(ind_true)):
        # break
        if cur_lev[0] != np.where(c_arr[:8, ind_true[ind ]]==True)[0][-1]:
            cur_lev = np.where(c_arr[:8, ind_true[ind ]]==True)[0]
            switch_count+=1
    return switch_count/len(ind_true), switch_count, len(ind_true)
#%%
# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(c_array, aspect = 'auto')