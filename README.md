# ACR4MLS

This is the repo for the paper titled
"*An Analysis Method for Metrical-Level Switching in Beat Tracking*".

[ **| Paper** ](https://)[ **| Code |** ](https://github.com/SunnyCYC/acr4mls/)


## Abstract
For expressive music, the tempo may change over time, posing challenges to tracking the beats by an automatic model. The model may first tap to the correct tempo, but then may fail to adapt to a tempo change, or switch between several incorrect but perceptually plausible ones (e.g., half- or double-tempo). Existing evaluation metrics for beat tracking do not reflect such behaviors, as they typically assume a fixed relationship between the reference beats and estimated beats. In this paper, we propose a new performance analysis method, called annotation coverage ratio (ACR), that accounts for a variety of possible metric-level switching behaviors of beat trackers. The idea is to derive sequences of modified reference beats of all metrical levels for every two consecutive reference beats, and compare every sequence of modified reference beats to the subsequences of estimated beats. We show via experiments on three datasets of different genres the superiority of ACR over existing metrics, and discuss the new insights to be gained. 

## Usage
In this repo we include one recording from the ASAP dataset [1, 2] as an example to demonstrate the usage of the proposed ACR method. Users may run [acr_usage.py](https://github.com/SunnyCYC/acr4mls/blob/main/acr_usage.py) directly to see how to generate both the visualization and quantitative results of ACR for the estimated beats and corresponding ground-truth beats.



## Reference
*[1] F. Foscarin, A. McLeod, P. Rigaux, F. Jacquemard, and M. Sakai,“ASAP: A dataset of aligned scores and performances for piano transcription,” in Proc. Int. Soc. Music Inf. Retr. Conf., 2020, pp. 53*

*[2] https://github.com/fosfrancesco/asap-dataset*
