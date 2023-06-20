#include <cmath>

#include "libqhullcpp/Qhull.h"

#include "globals.h"
#include "err.h"
#include "zobov_qhull.h"
#include "zobov.h"

#define NGUARD 84
#define BF 1e30
#define MAXVERVER 100000

// FIXME check that scaling or not by BoxSize is consistent...

int vozinit (const Globals &globals, ZobovCfg &zc,
             const float *xgal)
{
    zc.width = globals.BoxSize / zc.numdiv;
    zc.width2 = 0.5 * zc.width;

    zc.bf = (zc.border>0.0) ? zc.border : 0.1;

    zc.totwidth = zc.width + 2.0 * zc.bf;
    zc.totwidth2 = zc.width2 + zc.bf;

    zc.s = zc.width / NGUARD;

    CHECK((zc.bf*zc.bf-2.0*zc.s*zc.s)<0.0, return 1);

    zc.g = (0.5*zc.bf) * (1.0 + std::sqrt(1.0 - 2.0*zc.s*zc.s/(zc.bf*zc.bf)));

    zc.nvpmin=globals.Ngals; zc.nvpmax=0; zc.nvpbufmin=globals.Ngals; zc.nvpbufmax=0;

    int b[3];
    double c[3];

    for (b[0]=0; b[0]<zc.numdiv; ++b[0])
    {
        c[0] = (b[0]+0.5) * zc.width;
        for (b[1]=0; b[1]<zc.numdiv; ++b[1])
        {
            c[1] = (b[1]+0.5) * zc.width;
            for (b[2]=0; b[2]<zc.numdiv; ++b[2])
            {
                int64_t nvp = 0;
                int64_t nvpbuf = 0;
                for (int64_t ii=0; ii<globals.Ngals; ++ii)
                {
                    bool isitinbuf=true, isitinmain=true;
                    float rtemp[3];
                    for (int d=0; d<3; ++d)
                    {
                        rtemp[d] = xgal[ii*3+d] - c[d];
                        if (rtemp[d] > +0.5) --rtemp[d];
                        if (rtemp[d] < -0.5) ++rtemp[d];
                        isitinbuf = isitinbuf && (std::fabs(rtemp[d])<zc.totwidth2);
                        isitinmain = isitinmain && (std::fabs(rtemp[d])<=zc.width2);
                    }
                    if (isitinbuf) ++nvpbuf;
                    if (isitinmain) ++nvp;
                }
                if (nvp>zc.nvpmax)       zc.nvpmax = nvp;
                if (nvpbuf>zc.nvpbufmax) zc.nvpbufmax = nvpbuf;
                if (nvp<zc.nvpmin)       zc.nvpmin = nvp;
                if (nvpbuf<zc.nvpbufmin) zc.nvpbufmin = nvpbuf;
            }
        }
    }


    return 0;
}

int voz1b1 (const Globals &globals, ZobovCfg &zc,
            int b[3],
            const float *xgal)
{
    // FIXME not sure if we want Ngals here, probably rather the number of galaxies
    //       in this sub-division

    int status;

    PartAdj *adjs = (PartAdj *)std::malloc(globals.Ngals * sizeof(PartAdj));
    CHECK(!adjs, return 1);

    realT c[3];
    for (int ii=0; ii<3; ++ii) c[ii] = zc.width * b[ii];

    int64_t nvp = 0, nvpbuf = 0;
    for (int64_t ii=0; ii<globals.Ngals; ++ii)
    {
        bool isitinbuf = true, isitinmain = true;
        coordT rtemp[3];
        for (int jj=0; jj<3; ++jj)
        {
            rtemp[jj] = (coordT)xgal[ii*3+jj] - c[jj];
            if (rtemp[jj] > +0.5) --rtemp[jj];
            if (rtemp[jj] < -0.5) ++rtemp[jj];
            isitinbuf = isitinbuf && (std::fabs(rtemp[jj]) < zc.totwidth2);
            isitinmain = isitinmain && (std::fabs(rtemp[jj]) <= zc.width2);
        }

        if (isitinbuf) ++nvpbuf;
        if (isitinmain) ++nvp;
    }

    nvpbuf += 6*(NGUARD+1)*(NGUARD+1);

    coordT *parts = (coordT *)std::malloc(3 * nvpbuf * sizeof(coordT));
    CHECK(!parts, return 1);
    int64_t *orig = (int64_t *)std::malloc(nvpbuf * sizeof(int64_t));
    CHECK(!orig, return 1);

    nvp = 0;

    for (int64_t ii=0; ii<globals.Ngals; ++ii)
    {
        bool isitinmain = true;
        coordT rtemp[3];
        for (int jj=0; jj<3; ++jj)
        {
            rtemp[jj] = (coordT)xgal[ii*3+jj] - c[jj];
            if (rtemp[jj] > +0.5) --rtemp[jj];
            if (rtemp[jj] < -0.5) ++rtemp[jj];
            isitinmain = isitinmain && (std::fabs(rtemp[jj]) <= zc.width2);
        }
        if (isitinmain)
        {
            for (int jj=0; jj<3; ++jj) parts[3*nvp+jj] = rtemp[jj];
            orig[nvp] = ii;
            ++nvp;
        }
    }

    nvpbuf = nvp;

    for (int64_t ii=0; ii<globals.Ngals; ++ii)
    {
        bool isitinbuf = true, isitinmain = true;
        coordT rtemp[3];
        for (int jj=0; jj<3; ++jj)
        {
            rtemp[jj] = (coordT)xgal[ii*3+jj] - c[jj];
            if (rtemp[jj] > +0.5) --rtemp[jj];
            if (rtemp[jj] < -0.5) ++rtemp[jj];
            isitinbuf = isitinbuf && (std::fabs(rtemp[jj]) < zc.totwidth2);
            isitinmain = isitinmain && (std::fabs(rtemp[jj]) <= zc.width2);
        }

        if (isitinbuf && !isitinmain)
        {
            for (int jj=0; jj<3; ++jj) parts[3*nvpbuf+jj] = rtemp[jj];
            orig[nvpbuf] = ii;
            ++nvpbuf;
        }
    }

    int64_t nvpall = nvpbuf;

    for (int64_t ii=0; ii<=NGUARD; ++ii)
        for (int64_t jj=0; jj<=NGUARD; ++jj)
        {
            parts[3*nvpall+0] = -zc.width2 + ii * zc.s;
            parts[3*nvpall+1] = -zc.width2 + jj * zc.s;
            parts[3*nvpall+2] = -zc.width2 - zc.g;
            ++nvpall;
            parts[3*nvpall+0] = -zc.width2 + ii * zc.s;
            parts[3*nvpall+1] = -zc.width2 + jj * zc.s;
            parts[3*nvpall+2] = +zc.width2 + zc.g;
            ++nvpall;
        }
    for (int64_t ii=0; ii<=NGUARD; ++ii)
        for (int64_t jj=0; jj<=NGUARD; ++jj)
        {
            parts[3*nvpall+0] = -zc.width2 + ii * zc.s;
            parts[3*nvpall+1] = -zc.width2 - zc.g;
            parts[3*nvpall+2] = -zc.width2 + jj * zc.s;
            ++nvpall;
            parts[3*nvpall+0] = -zc.width2 + ii * zc.s;
            parts[3*nvpall+1] = +zc.width2 + zc.g;
            parts[3*nvpall+2] = -zc.width2 + jj * zc.s;
            ++nvpall;
        }
    for (int64_t ii=0; ii<=NGUARD; ++ii)
        for (int64_t jj=0; jj<=NGUARD; ++jj)
        {
            parts[3*nvpall+0] = -zc.width2 - zc.g;
            parts[3*nvpall+1] = -zc.width2 + ii * zc.s;
            parts[3*nvpall+2] = -zc.width2 + jj * zc.s;
            ++nvpall;
            parts[3*nvpall+0] = +zc.width2 + zc.g;
            parts[3*nvpall+1] = -zc.width2 + ii * zc.s;
            parts[3*nvpall+2] = -zc.width2 + jj * zc.s;
            ++nvpall;
        }

    status = delaunadj(parts, nvp, nvpbuf, nvpall, &adjs);
    CHECK(status, return 1);

    coordT *deladjs = (coordT *)std::malloc(3 * MAXVERVER * sizeof(coordT));
    CHECK(!deladjs, return 1);
    coordT *points = (coordT *)std::malloc(3 * MAXVERVER * sizeof(coordT));
    CHECK(!points, return 1);
    pointT *intpoints = (pointT *)std::malloc(3 * MAXVERVER * sizeof(pointT));
    CHECK(!intpoints, return 1);

    float *vols = (float *)std::malloc(nvp * sizeof(float));
    for (int64_t ii=0; ii<nvp; ++ii)
    {
        for (int jj=0; jj<adjs[ii].nadj; ++jj)
            for (int kk=0; kk<3; ++kk)
            {
                deladjs[3*jj+kk] = parts[3*adjs[ii].adj[jj]+kk] - parts[3*ii+kk];
                if (deladjs[3*jj+kk] > +0.5) --deladjs[3*jj+kk];
                if (deladjs[3*jj+kk] < -0.5) ++deladjs[3*jj+kk];
            }

        status = vorvol(deladjs, points, intpoints, adjs[ii].nadj, vols+ii);
        CHECK(status, return 1);
        vols[ii] *= globals.Ngals;
    }

    for (int64_t ii=0; ii<nvp; ++ii)
        for (int jj=0; jj<adjs[ii].nadj; ++jj)
            adjs[ii].adj[jj] = orig[adjs[ii].adj[jj]];

    std::free(deladjs);
    std::free(points);
    std::free(intpoints);


    // TODO
    // output should be:
    //   np, nvp
    //   orig, vols
    //   adjacencies

    return 0;
}
