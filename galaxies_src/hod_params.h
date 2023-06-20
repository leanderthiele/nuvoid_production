/* Holds the HOD free parameters */

#ifndef HOD_PARAMS_H
#define HOD_PARAMS_H

#include "have_if.h"


template<bool have_abias, bool have_vbias, bool have_zdep>
struct HODParams
{
    // TODO constructor
    float log_Mmin, sigma_logM, log_M0, log_M1, alpha;

    // transfP1 is some transformation of P1 to the real line such that P1=0.5 is at transfP1=0
    // and natural perturbations are O(1)
    HAVE_IF(have_abias, float) transfP1, abias;

    HAVE_IF(have_vbias, float) transf_eta_cen, transf_eta_sat;

    HAVE_IF(have_zdep, float) mu_Mmin, mu_M1;
};

#endif // HOD_PARAMS_H
