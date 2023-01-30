import numpy as np

class Compress :
    """ Linear compression for Gaussianish likelihood
    Implements the nuisance-hardening from 1903.01473,
    with the additional feature that the covariance of
    compressed data is approximately the identity.

    We adopt the convention to call the complete parameter
    vector phi, the interesting parameters theta, and the
    nuisance parameters eta.

    The compression matrix is defined such that

        d[a] = S[a, b] B[b, nu] d[nu]

    is the compressed data vector,
    where

        B[a, nu] = A[a, mu] Cinv[mu, nu],

        A[a, mu] = dm[mu]/dtheta[a] - Fte[a, b] Feeinv[b, c] dm[mu]/deta[c].

    Here, S is chosen such that

        S M S^T = 1,

        with M[a, b] = A[a, mu] A[b, nu] Cinv[mu, nu].

    With this choice, at the fiducial point (for zero mean)

        < d[a] d[b] > = 1[a, b]
    """

    def __init__ (self, dm_dphi, C, is_nuisance, Cprior=None) :
        """ Constructor
        dm_dphi ... model derivatives with respect to all parameters phi,
                    shape is [parameter, data] (can be list in first dimension)
        C ... covariance matrix, shape [data, data]
        is_nuisance ... boolean mask
        Cprior ... optional Gaussian prior covariance matrix
        """

        self.dim_phi = len(dm_dphi) # total dimension
        self.dim_eta = np.count_nonzero(is_nuisance)
        self.dim_theta = self.dim_phi - self.dim_eta

        self.C = C
        self.Cinv = np.linalg.inv(self.C)

        # reorder the parameters such that the interesting ones come first
        self.dm_dtheta = self.dm_dphi[~is_nuisance]
        self.dm_deta = self.dm_dphi[is_nuisance]

        # compute the fisher matrix
        self.F_phi = np.einsum('ai,ij,bj->ab', self.dm_dphi, self.Cinv, self.dm_dphi)
        if Cprior is not None :
            self.F_phi += np.linalg.inv(Cprior)

        # the various blocks of the Fisher matrix
        self.F_tt = self.F_phi[~is_nuisance, :][:, ~is_nuisance]
        self.F_ee = self.F_phi[is_nuisance, :][:, is_nuisance]
        self.F_te = self.F_phi[~is_nuisance, :][:, is_nuisance]

        # do the linear algebra
        self.A = self.dm_dtheta - np.einsum('ab,bc,ci->ai', self.F_te, np.linalg.inv(self.F_ee), self.dm_deta)
        self.B = np.einsum('ai,ij->aj', self.A, self.Cinv)
        self.M = np.einsum('ai,bj,ij->ab', self.A, self.A, self.Cinv)
        L = np.linalg.cholesky(self.M)
        self.S = np.linalg.inv(L)
        self.compression_matrix = np.einsum('ab,bi->ai', self.S, self.B)

    def __call__(self, x) :
        """ Compresses the data x
        x ... shape [..., data]
        """

        return np.einsum('ai,...i->...a', self.compression_matrix, x)


if __name__ == '__main__' :
    # testing code, assumes fiducial model is zero

    np.set_printoptions(formatter={'all': lambda x: '%+.4f'%x})

    # some arbitrary settings
    dim_data = 128
    dim_theta = 3
    dim_eta = 5

    rng = np.random.default_rng(34)

    is_nuisance = np.array([False, ] * dim_theta + [True, ] * dim_eta)
    rng.shuffle(is_nuisance)

    # a random covariance matrix
    A = rng.random((dim_data, dim_data)) - 0.5
    C = np.einsum('ab,bc->ac', A, A.T)

    # random derivatives
    dm_dphi = rng.random((dim_theta+dim_eta, dim_data)) - 0.5

    compressor = Compress(dm_dphi, C, is_nuisance)

    # some random data drawn from covariance
    x = rng.multivariate_normal(np.zeros(dim_data), C, 10000)

    xcompressed = compressor(x)
    cov = np.cov(xcompressed, rowvar=False)
    print(f'covariance of compressed data:\n{cov}')

    # check variation with parameters
    for ii in range(dim_theta + dim_eta) :
        x1 = x + dm_dphi[ii][None, :]
        x1compressed = compressor(x1)
        print(f'nuisance={is_nuisance[ii]}\tdelta={np.mean(x1compressed,axis=0)-np.mean(xcompressed,axis=0)}')
