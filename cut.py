import numpy as np

class Cut :
    """ Data vector cutting. The full data vector has the following layout:

    type: vsf        vgplk                  plk
           --zbin     --Rbin
                       --ell                  --ell
            --R         --k                    --k 

    """

    vsf_zbins = [0, 1]
    vsf_R = np.linspace(30, 80, num=32) # FIXME

    vgplk_Rbins = [30, 40, 50]
    vgplk_ell = [0, 2]
    vgplk_k = np.linspace(0.0, 0.2, num=32) # FIXME

    plk_ell = [0, 2]
    plk_k = np.linspace(0.0, 0.2, num=32) # FIXME


    def __init__ (self, use_vsf=True, use_vgplk=True, use_plk=True,
                        vsf_zbins=[0, 1], vsf_Rmin=30, vsf_Rmax=80,
                        vgplk_Rbins=[30, 40, 50], vgplk_ell=[0,2],
                        plk_ell=[0,2],
                        kmin=0.02, kmax=0.1) :

        # vsf part
        vsf_Rmask = (Cut.vsf_R>=vsf_Rmin) * (Cut.vsf_R<=vsf_Rmax)
        vsf_zmask = [ii in vsf_zbins for ii in Cut.vsf_zbins]
        vsf_mask = use_vsf * np.concatenate([ii*vsf_Rmask for ii in vsf_zmask])

        # vgplk part
        vgplk_kmask = (Cut.vgplk_k>=kmin) *(Cut.vgplk_k<=kmax)
        vgplk_ellmask = [ii in vgplk_ell for ii in Cut.vgplk_ell]
        vgplk_Rbinmask = [ii in vgplk_Rbins for ii in Cut.vgplk_Rbins]
        vgplk_mask = use_vgplk * np.concatenate([ii * np.concatenate([jj * vgplk_kmask for jj in vgplk_ellmask]) for ii in vgplk_Rbinmask])

        # plk part
        plk_kmask = (Cut.plk_k>=kmin) * (Cut.plk_k<=kmax)
        plk_ellmask = [ii in plk_ell for ii in Cut.plk_ell]
        plk_mask = use_plk * np.concatenate([ii * plk_kmask for ii in plk_ellmask])

        # complete mask
        self.mask = np.concatenate([vsf_mask, vgplk_mask, plk_mask])

    def cut_vec (self, x) :
        """ cuts the last dimension of x """
        return x[..., self.mask]

    def cut_mat (self, A) :
        """ cuts the last two dimensions of A """
        return A[..., self.mask, :][..., :, self.mask]

    def expand_vec (self, x, fill_value=0.0) :
        """ expands the last dimension of x """
        out = np.full((*x.shape[:-1], len(self.mask)), fill_value, dtype=x.dtype)
        out[..., self.mask] = x
        return out
