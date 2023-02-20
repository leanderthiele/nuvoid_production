import numpy as np

from compress import Compress
from cut import Cut

class CutCompress :

    def __init__ (self, dm_dphi, C, is_nuisance, Cprior=None, **cut_kwargs) :
        
        if 'have_Cprior' in cut_kwargs : 
            if not cut_kwargs['have_Cprior'] :
                Cprior = None
            del cut_kwargs['have_Cprior']

        if 'do_compression' in cut_kwargs :
            self.do_compression = cut_kwargs['do_compression']
            del cut_kwargs['do_compression']
        else :
            self.do_compression = True

        self.cut = Cut(**cut_kwargs)

        cut_C = self.cut.cut_mat(C)

        if self.do_compression :
            cut_dm_dphi = self.cut.cut_vec(dm_dphi)
            self.compress = Compress(cut_dm_dphi, cut_C, is_nuisance, Cprior)
            self.cut_compression_matrix = self.cut.expand_vec(self.compress.compression_matrix, fill_value=0.0)
        else :
            # no compression, but we still do the transformation such that the covariance
            # becomes the identity
            L = np.linalg.cholesky(cut_C)
            self.cut_compression_matrix = self.cut.expand_vec(np.linalg.inv(L), fill_value=0.0)

    def __call__ (self, x) :
        return np.einsum('ai,...i->...i', self.cut_compression_matrix, x)
