"""
Some details on parameters for each step.

Preprocessor
---------------------

    * highpass_freq (float): frequency of high pass filter typically 250~500Hz. This remove LFP component in the signal
      Theorically a low value is better (250Hz) but if the signal contain oscillation at high freqyencies 
      (during sleep for insctance) theye must removed so 400Hz should be OK.
    * lowpass_freq (float): low pass frequency (typically 3000~10000Hz) This remove noise in high freuqnecy. This help
       to smooth the spike for peak alignement. This must not exceed niquist frequency (sample_rate/2)
    * smooth_size (int): other possibility to smooth signal. This apply a kernel (more or less triangle) the *smooth_size**
       width in sample. This is like a lowpass filter. If you don't known put 0.
    * common_ref_removal (bool): this substracts sample by sample the median across channels
       When there is a strong noise that appears on all channels (sometimes due to reference) you
       can substract it. This is as if all channels would re referenced numerically to there medians.
    * chunksize (int): the whole processing chain is applied chunk by chunk, this is the chunk size in sample. Typically 1024.
       The smaller size lead to less memory but more CPU comsuption in Peeler. For online, this will be more or less the latency.
    * lostfront_chunksize (int): size in sample of the margin at the front edge for each chunk to avoid border effect in backward filter.
       In you don't known put None then lostfront_chunksize will be int(sample_rate/highpass_freq)*3 which is quite robust (<5% error)
       compared to a true offline filtfilt.
    * signalpreprocessor_engine (str): 'numpy' or 'opencl'. There is a double implementation for signal preprocessor : With numpy/scipy
      flavor (and so CPU) or opencl with home made CL kernel (and so use GPU computing). If you have big fat GPU and are able to install
      "opencl driver" (ICD) for your platform the opencl flavor should speedup the peeler because pre processing signal take a quite
      important amoung of time.
    
Peak detector
----------------------

  * peakdetector_engine (str): 'numpy' or 'opencl'.  See signal_preprocessor_engine. Here the speedup is small.
  * peak_sign (str) : sign of the peak ('+' or '-'). The double detection ('+-') is intentionaly NOT implemented is tridesclous
    because it lead to many mistake for users in multi electrode arrays where the same cluster is seen both on negative peak
    and positive rebounce.
  * relative_threshold (str): the threshold without sign with MAD units (robust standard deviation). See :ref:`important_details`.
  * peak_span (float) : this avoid double detection of the same peak in a short span. The units is second.
  
Waveform extraction
--------------------------------

  * n_left (int); size in sample of the left sweep from the peak index. This number is negative.
  * n_right (int): size in sample of the right sweep from the peak index. This number is positive.
  * mode (str): 'rand' or 'all' With 'all' all detected peaks are extracted. With 'all' only an randomized subset is taken.
     Note that if you use tridesclous with the script/notebook method you can also choose by yourself which peak are 
     choosen for waveform extraction. This can be usefull to avoid electrical/optical stimlation periods or force peak around
     stimulus periods.
  * nb_max (int): for 'rand' mode this is the number of peak extracted. This number must be carrefully choosen.
    This highly depend on : the duration on which the catalogue constructor is done + the number of channel (and so the number
    of cells) + the density (firing rate) fo each cluster. Since this can't be known in advance, the user must explore cluster and
    extract again while changing this number given dense enough clusters. This have a strong imptact of the CPU and RAM.
    So do not choose to big number.
  * align_waveform (bool): experimental make an oversampling before extracting waveform. This slow down a lot the process
     with few enhancement for centroids. Note that this is NOT the intersample true peak estimation done by the Peeler (which is
     very important)

Waveform clean
-------------------------

  * alien_value_threshold (float): units=one mad. above this threshold the waveforms is tag as "Alien" and not use for features and clustering

Noise snippet extraction
--------------------------------------

  * nb_snippet (int):  the number of noise snippet taken in the signal in between peaks.
  

Features extraction
-------------------------------
 
Several methods possible. See :ref:`important_details`.


  * **global_pca**:
  
    * n_components (int): number of components of the pca for all the channel.

  * **peak_max** no parameters
  
  * **pca_by_channel**:
  
    * n_components_by_channel (int): number of component for each channel.

  * **neighborhood_pca**:
  
    * n_components_by_neighborhood (int): number of component by channel and its neighborhood
    * radius_um (float): radius around the channel in mircometers.

Cluster
-----------

Several methods possible. See :ref:`important_details`.

  * **kmeans** : `kmeans <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ implemented in sklearn
    
    * n_clusters (int): number of cluster
    
  * **onecluster** no clustering. All label set to 0.
  
  * **gmm** `gaussian mixture model <http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture>`_ implemented in sklearn 
    
    * n_clusters (int): number of cluster
    * covariance_type (str): 'full', 'tied', 'diag', 'spherical'
    * n_init (int) The number of initializations to perform.
  
  * **agglomerative** `AgglomerativeClustering <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_ implemented in sklearn 
  
    * n_clusters: number of cluster
  
  * **dbscan** `DBSCAN <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_ implemented in sklearn
     
    * eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
  
  * **sawchaincut** Home made automatic clustering, usefull for dense arrays. Autodetect well isolated cluster
    and put to trash ambiguous things.


"""

from collections import OrderedDict
import numpy as np

preprocessor_params = [
    {'name': 'highpass_freq', 'type': 'float', 'value':400., 'step': 10., 'suffix': 'Hz', 'siPrefix': True},
    {'name': 'lowpass_freq', 'type': 'float', 'value':5000., 'step': 10., 'suffix': 'Hz', 'siPrefix': True},
    {'name': 'smooth_size', 'type': 'int', 'value':0},
    {'name': 'common_ref_removal', 'type': 'bool', 'value':False},
    {'name': 'chunksize', 'type': 'int', 'value':1024, 'decimals':10},
    {'name': 'lostfront_chunksize', 'type': 'int', 'value':0, 'decimals':10, 'limits': (0, np.inf),},
    {'name': 'signalpreprocessor_engine', 'type': 'list', 'value' : 'numpy', 'values':['numpy', 'opencl']},
]

peak_detector_params = [
    {'name': 'peakdetector_engine', 'type': 'list', 'value' : 'numpy', 'values':['numpy', 'opencl']},
    {'name': 'peak_sign', 'type': 'list',  'value':'-', 'values':['-', '+']},
    {'name': 'relative_threshold', 'type': 'float', 'value': 5., 'step': .1,},
    {'name': 'peak_span', 'type': 'float', 'value':0.0002, 'step': 0.0001, 'suffix': 's', 'siPrefix': True},
]

waveforms_params = [
    {'name': 'n_left', 'type': 'int', 'value':-20},
    {'name': 'n_right', 'type': 'int', 'value':30},
    {'name': 'mode', 'type': 'list', 'values':['rand', 'all']},
    {'name': 'nb_max', 'type': 'int', 'value':20000},
    {'name': 'align_waveform', 'type': 'bool', 'value':False},
    #~ {'name': 'subsample_ratio', 'type': 'int', 'value':20},
]

clean_waveforms_params =[
    {'name': 'alien_value_threshold', 'type': 'float', 'value':100.},
]



noise_snippet_params = [
    {'name': 'nb_snippet', 'type': 'int', 'value':300},
]



features_params_by_methods = OrderedDict([
    ('global_pca',  [{'name' : 'n_components', 'type' : 'int', 'value' : 5}]),
    ('peak_max',  []),
    ('pca_by_channel',  [{'name' : 'n_components_by_channel', 'type' : 'int', 'value' : 3}]),
    ('neighborhood_pca',  [{'name' : 'n_components_by_neighborhood', 'type' : 'int', 'value' : 3}, 
                                        {'name' : 'radius_um', 'type' : 'float', 'value' : 300., 'step':50.}, 
                                        ]),
])


cluster_params_by_methods = OrderedDict([
    ('agglomerative', [{'name' : 'n_clusters', 'type' : 'int', 'value' : 3}]),
    ('kmeans', [{'name' : 'n_clusters', 'type' : 'int', 'value' : 5}]),
    ('onecluster', []),
    ('gmm', [{'name' : 'n_clusters', 'type' : 'int', 'value' : 5},
                    {'name' : 'covariance_type', 'type' : 'list', 'values' : ['full']},
                    {'name' : 'n_init', 'type' : 'int', 'value' : 10}]),
    ('dbscan', [{'name' : 'eps', 'type' : 'float', 'value' : 0.5}]),
    ('sawchaincut', [{'name' : 'max_loop', 'type' : 'int', 'value' : 1000},
                                {'name' : 'nb_min', 'type' : 'int', 'value' : 20},
                                {'name' : 'break_nb_remain', 'type' : 'int', 'value' : 30},
                                {'name' : 'kde_bandwith', 'type' : 'float', 'value' : 1., 'step':0.1},
                                {'name' : 'auto_merge_threshold', 'type' : 'float', 'value' : 2., 'step':0.1},
                                {'name':'print_debug', 'type': 'bool', 'value':False},
                                
                            
                            
                            ]),
])

#~ split_params_by_methods = OrderedDict([
    #~ ('kmeans', [{'name' : 'n_clusters', 'type' : 'int', 'value' : 5}]),
    #~ ('gmm', [{'name' : 'n_clusters', 'type' : 'int', 'value' : 5},
                    #~ {'name' : 'covariance_type', 'type' : 'list', 'values' : ['full']},
                    #~ {'name' : 'n_init', 'type' : 'int', 'value' : 10}]),
#~ ])


fullchain_params = [
    {'name':'duration', 'type': 'float', 'value':300., 'suffix': 's', 'siPrefix': True},
    {'name':'preprocessor', 'type':'group', 'children': preprocessor_params},
    {'name':'peak_detector', 'type':'group', 'children': peak_detector_params},
    {'name':'noise_snippet', 'type':'group', 'children': noise_snippet_params},
    {'name':'extract_waveforms', 'type':'group', 'children' : waveforms_params},
    {'name':'clean_waveforms', 'type':'group', 'children' : clean_waveforms_params},
]

metrics_params = [
    {'name': 'spike_waveforms_similarity', 'type': 'list', 'values' : [ 'cosine_similarity']},
    {'name': 'cluster_similarity', 'type': 'list', 'values' : [ 'cosine_similarity_with_max']},
    {'name': 'cluster_ratio_similarity', 'type': 'list', 'values' : [ 'cosine_similarity_with_max']},
    {'name': 'size_max', 'type': 'int', 'value':10000000},
]


peeler_params = [
    {'name':'limit_duration', 'type': 'bool', 'value':True},
    {'name': 'chunksize', 'type': 'int', 'value':1024, 'decimals':10},
    {'name':'duration', 'type': 'float', 'value':60., 'suffix': 's', 'siPrefix': True},
    {'name': 'use_sparse_template', 'type': 'bool', 'value':False},
    {'name':'sparse_threshold_mad', 'type': 'float', 'value': 1.5, },
    {'name': 'use_opencl_with_sparse', 'type': 'bool', 'value':False},
    
]



possible_tags = ['', 'so_bad', 'bad', 'not_so_bad','not_so_good','good', 'so_good', 'better_than_dreams']




