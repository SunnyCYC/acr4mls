# ACR4MLS

This is the repo for the paper titled
"*An Analysis Method for Metrical-Level Switching in Beat Tracking*".

[ **| Paper** ](https://arxiv.org/abs/2210.06817)[ **| Code |** ](https://github.com/SunnyCYC/acr4mls/)


## Abstract
For expressive music, the tempo may change over time, posing challenges to tracking the beats by an automatic model. The model may first tap to the correct tempo, but then may fail to adapt to a tempo change, or switch between several incorrect but perceptually plausible ones (e.g., half- or double-tempo). Existing evaluation metrics for beat tracking do not reflect such behaviors, as they typically assume a fixed relationship between the reference beats and estimated beats. In this paper, we propose a new performance analysis method, called annotation coverage ratio (ACR), that accounts for a variety of possible metric-level switching behaviors of beat trackers. The idea is to derive sequences of modified reference beats of all metrical levels for every two consecutive reference beats, and compare every sequence of modified reference beats to the subsequences of estimated beats. We show via experiments on three datasets of different genres the usefulness of ACR when utilized alongside existing metrics, and discuss the new insights to be gained. 


## Usage
In this repo we include five recordings from the ASAP dataset [1, 2] as an example to demonstrate the usage of the proposed ACR method. 

* evaluate one single recording:
    * [acr_example.ipynb](https://github.com/SunnyCYC/acr4mls/blob/main/acr_example.ipynb) shows that with the specified ***L value***, ***ground-truth annotation file path***, and ***beat estimation file path***, the ACR results can be calculated and printed. These results can also be used for visualization.
    * the activation files for visualization were generated using the ***madmom.features.downbeats.RNNDownBeatProcessor*** [3, 4].
* evaluate one full dataset:
    * [acr_usage.py](https://github.com/SunnyCYC/acr4mls/blob/main/acr_usage.py) shows that with the ***folder paths of ground-truth beat files and estimated beat files*** specified, the full dataset can be evaluated. The song-level results will be saved as a ***.csv file*** in the ***./evaluation/quantitative_csvs*** path. Note that the estimated files should share same basenames with the ground-truth files.



## Extended Experiment Results
To strengthen the claimed benefits of this proposed method, we trained the state-of-the-art TCN model [5] from scratch with the data augmentation following the tutorial [6] using SMC [7], HJDB [8], Hainsworth [9], Ballroom [10, 11], and Beatles [12]. As the purpose of these experiments are to gain insights into the behaviors of SOTA beat trackers, rather than to compare their performance, we demonstrate the results in Table 1 (madmom) and Table 2 (TCN) separately. It can be seen from the distribution of the colored cells that the TCN network exhibits similar metric-level behaviors as the madmom network does.

![](https://i.imgur.com/vgStrbC.png)

On the other hand, we also evaluate using three additional beat tracking datasets, namely ASAP [11], GTZAN [12, 13], and Hainsworth [5]. Result shown in Table 3 (madmom) and Table 4 (TCN) again suggests that both the madmom and TCN model exhibit similar MLS behaviors that are nicely reflected by the proposed ACR metric but not by existing evaluation metrics.

![](https://i.imgur.com/iNnE0tQ.png)

## Reference
*[1] F. Foscarin, A. McLeod, P. Rigaux, F. Jacquemard, and M. Sakai,“ASAP: A dataset of aligned scores and performances for piano transcription,” in Proc. Int. Soc. Music Inf. Retr. Conf., 2020, pp. 53*

*[2] https://github.com/fosfrancesco/asap-dataset*

*[3] S. B¨ock, F. Krebs, and G. Widmer, “Joint beat and downbeat tracking with recurrent neural networks,” Proc. Int. Soc. Music Inf. Retr. Conf., pp. 255–261, 2016.*

*[4] https://madmom.readthedocs.io/en/v0.16/modules/features/downbeats.html*

*[5] S. Böck and M. E. P. Davies, “Deconstruct, analyse, reconstruct: How to improve tempo, beat, and downbeat estimation,” in Proc. Int. Soc. Music Inf. Retr. Conf., 2020, pp. 574–582.*

*[6] M. E. P. Davies, S. Böck, and M. Fuentes, Tempo, beat and downbeat estimation. Proc. Int. Soc. Music Inf. Retr. Conf., 2021. https://tempobeatdownbeat.github.io/tutorial/intro.html*

*[7] A. Holzapfel, M. E. P. Davies, J. R. Zapata, J. L. Oliveira, and F. Gouyon, “Selective sampling for beat tracking evaluation,” IEEE Trans. Audio, Speech, and Language Process., vol. 20, no. 9, pp. 2539–2548, 2012.*

*[8] J. A. Hockman, M. E. P. Davies, and I. Fujinaga, “One in the jungle: Downbeat detection in hardcore, jungle, and drum and bass,” Proc. Int. Soc. Music Inf. Retr. Conf., pp. 169–174, 2012.*

*[9] M. Macleod and S. Hainsworth, “Particle filtering applied to musical tempo tracking,” EURASIP Journal on Advances in Signal Processing, vol. 2004, no. 15, pp. 2385–2395, 2004.*

*[10] F. Gouyon, A. Klapuri, S. Dixon, et al., “An experimental comparison of audio tempo induction algorithms,” IEEE Trans. Audio, Speech, and Language Process., vol. 14, no. 5, pp. 1832–1844, 2006.*

*[11] F. Krebs, S. Böck, and G. Widmer, “Rhythmic pattern modeling for beat and downbeat tracking in musical audio,” in Proc. Int. Soc. Music Inf. Retr. Conf., 2013.*

*[12] M. E. P. Davies, N. D. Quintela, and M. Plumbley, “Evaluation methods for musical audio beat tracking algorithms,” in Queen Mary University of London, Centre for Digital Music, Tech. Rep. C4DM-TR-09-06, 2009.*





