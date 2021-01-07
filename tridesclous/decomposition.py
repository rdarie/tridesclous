import numpy as np
import pdb, traceback
import sklearn
import sklearn.decomposition

import sklearn.cluster
import sklearn.manifold

import umap
from . import tools


def project_waveforms(waveforms, method='pca', selection=None,  catalogueconstructor=None, **params):
    """
    
    
    """
    if selection is None:
        waveforms2 = waveforms
    else:
        waveforms2 = waveforms[selection]
    
    if waveforms2.shape[0] == 0:
        return None, None, None
    
    
    if method=='global_pca':
        projector = GlobalPCA(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='global_umap':
        projector = GlobalUMAP(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='global_pumap':
        projector = GlobalPUMAP(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='peak_max':
        projector = PeakMaxOnChannel(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='pca_by_channel':
        projector = PcaByChannel(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='neighborhood_pca':
        projector = NeighborhoodPca(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='pca_by_channel_then_tsne':
        projector = PcaByChannelThenTsne(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='global_pca_then_tsne':
        projector = GlobalPCAThenTsne(waveforms2, catalogueconstructor=catalogueconstructor, **params)   
    #~ elif method=='peakmax_and_pca':
        #~ projector = PeakMax_and_PCA(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    #~ elif method=='tsne':
        #~ projector = TSNE(waveforms2, catalogueconstructor=catalogueconstructor, **params)     
    else:
        Raise(NotImplementedError)
    
    features = projector.transform(waveforms2)
    channel_to_features = projector.channel_to_features
    return features, channel_to_features, projector


class GlobalPCA:
    def __init__(
            self, waveforms, catalogueconstructor=None,
            n_components=5, **params):
        cc = catalogueconstructor
        self.n_components = n_components
        self.waveforms = waveforms
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        self.pca = sklearn.decomposition.IncrementalPCA(n_components=n_components, **params)
        self.pca.fit(flatten_waveforms)
        #  In GlobalPCA all feature represent all channels
        self.channel_to_features = np.ones((cc.nb_channel, self.n_components), dtype='bool')

    def transform(self, waveforms):
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        return self.pca.transform(flatten_waveforms)

    def fit(self, waveforms, labels=None):
        return


class GlobalUMAP:
    def __init__(
            self, waveforms, catalogueconstructor=None,
            n_components=2, n_neighbors=5,
            min_dist=0.1, metric='euclidean',
            set_op_mix_ratio=1., init='spectral',
            n_epochs=500,
            **params):
        cc = catalogueconstructor
        self.n_components = n_components
        self.waveforms = waveforms
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        self.umap = umap.UMAP(
            n_components=n_components, n_neighbors=n_neighbors,
            min_dist=min_dist, metric=metric, init=init,
            set_op_mix_ratio=set_op_mix_ratio, n_epochs=n_epochs,
            **params)
        self.umap.fit(flatten_waveforms)
        # try:
        #     self.umap.fit(flatten_waveforms)
        # except:
        #     print('###########################################################')
        #     print(cc)
        #     traceback.print_exc()
        #     print('###########################################################')
        #In GlobalPCA all feature represent all channels
        self.channel_to_features = np.ones((cc.nb_channel, self.n_components), dtype='bool')
    
    def transform(self, waveforms):
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        return self.umap.transform(flatten_waveforms)
        
    def fit(self, waveforms, labels=None):
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        self.umap.fit(flatten_waveforms, y=labels)


class GlobalPUMAP:
    def __init__(
            self, waveforms, catalogueconstructor=None,
            n_components=2, n_neighbors=5,
            min_dist=0.1, metric='euclidean',
            set_op_mix_ratio=1., init='spectral',
            n_epochs=500,
            **params):
        cc = catalogueconstructor
        self.n_components = n_components
        self.waveforms = waveforms
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        ##### parametric umap
        self.data_min = flatten_waveforms.min()
        self.data_range = flatten_waveforms.max() - self.data_min
        scaled_waveforms = (flatten_waveforms - self.data_min) / self.data_range
        self.umap = umap.parametric_umap.ParametricUMAP(
            n_components=n_components, n_neighbors=n_neighbors,
            min_dist=min_dist, metric=metric, init=init,
            set_op_mix_ratio=set_op_mix_ratio, n_epochs=n_epochs,
            **params)
        #
        self.umap.fit(scaled_waveforms)
        self.channel_to_features = np.ones((cc.nb_channel, self.n_components), dtype='bool')
    
    def transform(self, waveforms):
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        scaled_waveforms = (flatten_waveforms - self.data_min) / self.data_range
        return self.umap.transform(scaled_waveforms)
    
    def fit(self, waveforms, labels=None):
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        self.data_min = flatten_waveforms.min()
        self.data_range = flatten_waveforms.max() - self.data_min
        scaled_waveforms = (flatten_waveforms - self.data_min) / self.data_range
        self.umap.fit(scaled_waveforms, y=labels)


class PeakMaxOnChannel:
    def __init__(self, waveforms, catalogueconstructor=None, **params):
        cc = catalogueconstructor
        
        self.waveforms = waveforms
        self.ind_peak = -catalogueconstructor.info['waveform_extractor_params']['n_left']
        #~ print('PeakMaxOnChannel self.ind_peak', self.ind_peak)
        
        
        #In full PeakMaxOnChannel one feature is one channel
        self.channel_to_features = np.eye(cc.nb_channel, dtype='bool')
        
    def transform(self, waveforms):
        #~ print('ici', waveforms.shape, self.ind_peak)
        features = waveforms[:, self.ind_peak, : ].copy()
        return features

    def fit(self, waveforms, labels=None):
        return

class PcaByChannel:
    def __init__(self, waveforms, catalogueconstructor=None, n_components_by_channel=3, **params):
        self.cc = catalogueconstructor
        self.n_components_by_channel = n_components_by_channel
        self.waveforms = waveforms
        self.n_components_by_channel = n_components_by_channel
        self.pcas = []
        for c in range(self.cc.nb_channel):
            #~ print('c', c)
            pca = sklearn.decomposition.IncrementalPCA(n_components=n_components_by_channel, **params)
            pca.fit(waveforms[:,:,c])
            self.pcas.append(pca)
        #In full PcaByChannel n_components_by_channel feature correspond to one channel
        self.channel_to_features = np.zeros((self.cc.nb_channel, self.cc.nb_channel*n_components_by_channel), dtype='bool')
        for c in range(self.cc.nb_channel):
            self.channel_to_features[c, c*n_components_by_channel:(c+1)*n_components_by_channel] = True

    def transform(self, waveforms):
        n = self.n_components_by_channel
        all = np.zeros((waveforms.shape[0], waveforms.shape[2]*n), dtype=waveforms.dtype)
        for c, pca in enumerate(self.pcas):
            all[:, c*n:(c+1)*n] = pca.transform(waveforms[:, :, c])
        return all

    def fit(self, waveforms, labels=None):
        self.pcas = []
        for c in range(self.cc.nb_channel):
            #~ print('c', c)
            pca = sklearn.decomposition.IncrementalPCA(n_components=self.n_components_by_channel, **params)
            pca.fit(waveforms[:,:,c])
            self.pcas.append(pca)
        return
    


class NeighborhoodPca:
    def __init__(self, waveforms, catalogueconstructor=None, n_components_by_neighborhood=6, radius_um=300., **params):
        
        cc = catalogueconstructor
        
        self.n_components_by_neighborhood = n_components_by_neighborhood
        self.neighborhood = tools.get_neighborhood(cc.geometry, radius_um)
        
        self.pcas = []
        for c in range(cc.nb_channel):
            #~ print('c', c)
            neighbors = self.neighborhood[c, :]
            pca = sklearn.decomposition.IncrementalPCA(n_components=n_components_by_neighborhood, **params)
            wfs = waveforms[:,:,neighbors]
            wfs = wfs.reshape(wfs.shape[0], -1)
            pca.fit(wfs)
            self.pcas.append(pca)

        #In full NeighborhoodPca n_components_by_neighborhood feature correspond to one channel
        self.channel_to_features = np.zeros((cc.nb_channel, cc.nb_channel*n_components_by_neighborhood), dtype='bool')
        for c in range(cc.nb_channel):
            self.channel_to_features[c, c*n_components_by_neighborhood:(c+1)*n_components_by_neighborhood] = True

    def transform(self, waveforms):
        n = self.n_components_by_neighborhood
        all = np.zeros((waveforms.shape[0], waveforms.shape[2]*n), dtype=waveforms.dtype)
        for c, pca in enumerate(self.pcas):
            neighbors = self.neighborhood[c, :]
            wfs = waveforms[:,:,neighbors]
            wfs = wfs.reshape(wfs.shape[0], -1)
            all[:, c*n:(c+1)*n] = pca.transform(wfs)
        return all

    def fit(self, waveforms, labels=None):
        return


#~ class PeakMax_and_PCA:
    #~ def __init__(self, waveforms, catalogueconstructor=None, **params):
        #~ self.waveforms = waveforms
        #~ self.ind_peak = -catalogueconstructor.info['waveform_extractor_params']['n_left']+1

        #~ self.pca =  sklearn.decomposition.IncrementalPCA(**params)
        #~ peaks_val = waveforms[:, self.ind_peak, : ].copy()
        #~ self.pca.fit(peaks_val)
        
        
    #~ def transform(self, waveforms):
        #~ peaks_val = waveforms[:, self.ind_peak, : ].copy()
        #~ features = self.pca.transform(peaks_val)
        
        #~ return features
    


#~ class TSNE:
    #~ def __init__(self, waveforms, catalogueconstructor=None, **params):
        #~ self.waveforms = waveforms
        #~ flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        #~ self.tsne = sklearn.manifold.TSNE(**params)
    
    #~ def transform(self, waveforms):
        #~ flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        #~ return self.tsne.fit_transform(flatten_waveforms)

    
class PcaByChannelThenTsne:
    def __init__(
            self, waveforms, catalogueconstructor=None, n_components_by_channel=3,
            n_components_tsne=2, learning_rate=200, perplexity=30,
            method_tsne='barnes_hut', **params):
        self.waveforms = waveforms
        self.n_components_by_channel = n_components_by_channel
        self.pcas = []
        for c in range(self.waveforms.shape[2]):
            pca = sklearn.decomposition.IncrementalPCA(n_components=n_components_by_channel, **params)
            pca.fit(waveforms[:,:,c])
            self.pcas.append(pca)
        #
        tsne = sklearn.manifold.TSNE(
            n_components=n_components_tsne, learning_rate=learning_rate,
            perplexity=perplexity, method=method_tsne, n_iter=2000, init='pca')
        # steps = [('scaler', sklearn.preprocessing.StandardScaler()), ('tsne', tsne)]
        # from sklearn.pipeline import Pipeline
        # pipeline = Pipeline(steps)
        # self.manifold = pipeline
        self.manifold = tsne
        # all feature represent all channels
        self.channel_to_features = np.ones((cc.nb_channel, n_components_tsne), dtype='bool')
    
    def transform(self, waveforms):
        n = self.n_components_by_channel
        all = np.zeros((waveforms.shape[0], waveforms.shape[2]*n), dtype=waveforms.dtype)
        for c, pca in enumerate(self.pcas):
            all[:, c*n:(c+1)*n] = pca.transform(waveforms[:, :, c])
        return self.manifold.fit_transform(all)

    def fit(self, waveforms, labels=None):
        return


class GlobalPCAThenTsne:
    def __init__(
            self, waveforms, catalogueconstructor=None, n_components_pca=3,
            n_components_tsne=2, learning_rate=200, perplexity=30,
            method_tsne='barnes_hut', **params):
        cc = catalogueconstructor
        #
        self.n_components = n_components_pca
        self.waveforms = waveforms
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        self.pca =  sklearn.decomposition.IncrementalPCA(n_components=n_components_pca, **params)
        self.pca.fit(flatten_waveforms)
        #
        tsne = sklearn.manifold.TSNE(
            n_components=n_components_tsne, learning_rate=learning_rate,
            perplexity=perplexity, method=method_tsne, n_iter=2000, init='pca')
        #  steps = [('scaler', sklearn.preprocessing.StandardScaler()), ('tsne', tsne)]
        #  from sklearn.pipeline import Pipeline
        #  pipeline = Pipeline(steps)
        #  self.manifold = pipeline
        self.manifold = tsne
        # all feature represent all channels
        self.channel_to_features = np.ones((cc.nb_channel, n_components_tsne), dtype='bool')

    def transform(self, waveforms):
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        # all_pca = self.pca.transform(flatten_waveforms)
        # return self.manifold.fit_transform(all_pca)
        return self.manifold.fit_transform(flatten_waveforms)

    def fit(self, waveforms, labels=None):
        return