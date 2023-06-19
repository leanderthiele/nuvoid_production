#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>

#include "libqhullcpp/Qhull.h"

#include "err.h"
#include "zobov.h"

// TODO what is this?
#define MAXVERVER 100000

static int compar (const void *a, const void *b)
{
    int i1 = *(int *)a, i2 = *(int *)b;
    return 2*(i1 > i2) - 1 + (i1 == i2);
}

int delaunadj (coordT *x, int64_t nvp, int64_t nvpbuf, int64_t nvpall, PartAdj **adjs)
{
    int status;

    const int dim = 3; const boolT ismalloc = False;
    std::FILE *outfile = stdout, *errfile = stderr;
    
    PartAdj adjst;
    adjst.adj = (int *)std::malloc(MAXVERVER * sizeof(int));
    CHECK(!adjst.adj, return 1);

    // TODO look at qhull/src/qdelaunay/qdelaun_r.c to see how it is done!
    //      also c.f. user_eg3 for the C++ interface example
    //      In particular, there's a class Qhull with a reasonable looking constructor
    //      (which is probably equivalent to runQhull)
    qhT qhdata;
    char cmd[] = "qhull s d Qt";
    status = qh_new_qhull(&qhdata, dim, nvpall, x, ismalloc, cmd, outfile, errfile);
    CHECK(status, return 1);

    int numfacets, numsimplicial, totneighbors, numridges, numcoplanars, numtricoplanars;
    qh_countfacets(&qhdata, qhdata.facet_list, nullptr, 0,
                   &numfacets, &numsimplicial, &totneighbors, &numridges, &numcoplanars, &numtricoplanars);
    qh_vertexneighbors(&qhdata);

    setT *vertices = qh_facetvertices(&qhdata, qhdata.facet_list, NULL, 0);

    setT *vertex_points = qh_settemp(&qhdata, nvpall);
    setT *coplanar_points = qh_settemp(&qhdata, nvpall);
    qh_setzero(&qhdata, vertex_points, 0, nvpall);
    qh_setzero(&qhdata, coplanar_points, 0, nvpall);

    vertexT *vertex, **vertexp;
    FOREACHvertex_(vertices)
        qh_point_add(&qhdata, vertex_points, vertex->point, vertex);

    facetT *facet;
    FORALLfacet_(qhdata.facet_list)
    {
        pointT *point, **pointp;
        FOREACHpoint_(facet->coplanarset)
            qh_point_add(&qhdata, coplanar_points, point, facet);
    }

    int vertex_i, vertex_n, ver = 0;
    FOREACHvertex_i_(&qhdata, vertex_points)
    {
        (*adjs)[ver].nadj = 0;
        if (vertex)
        {
            adjst.nadj = 0;
            facetT *neighbor, **neighborp;
            FOREACHneighbor_(vertex)
            {
                if ((*adjs)[ver].nadj > -1)
                {
                    if (neighbor->visitid)
                    {
                        setT *vertices2 = neighbor->vertices;
                        vertexT *vertex2, **vertex2p;
                        FOREACHsetelement_(vertexT, vertices2, vertex2)
                        {
                            if (ver != qh_pointid(&qhdata, vertex2->point))
                            {
                                adjst.adj[adjst.nadj] = (int)qh_pointid(&qhdata, vertex2->point);
                                ++adjst.nadj;
                                CHECK(adjst.nadj>MAXVERVER, return 1);
                            }
                        }
                    }
                    else
                        (*adjs)[ver].nadj = -1;
                }
            }
        }
        else
            (*adjs)[ver].nadj = -2;

        CHECK(adjst.nadj < 4, return 1);

        std::qsort(adjst.adj, adjst.nadj, sizeof(int), compar);
        int64_t count = 1;

        for (int ii=1; ii<adjst.nadj; ++ii)
            if (adjst.adj[ii] != adjst.adj[ii-1])
            {
                CHECK(adjst.adj[ii] >= nvpbuf, return 1);
                ++ count;
            }
        
        (*adjs)[ver].adj = (int *)std::malloc(count * sizeof(int));
        CHECK((*adjs)[ver].adj, return 1);
        (*adjs)[ver].adj[0] = adjst.adj[0];
        count = 1;
        for (int ii=1; ii<adjst.nadj; ++ii)
            if (adjst.adj[ii] != adjst.adj[ii-1])
            {
                (*adjs)[ver].adj[count] = adjst.adj[ii];
                ++count;
            }
        (*adjs)[ver].nadj = count;

        ++ver;
        if (ver == nvp) break;
    } // FOREACHvertex_i_

    qh_settempfree (&qhdata, &coplanar_points);
    qh_settempfree (&qhdata, &vertex_points);
    qh_settempfree (&qhdata, &vertices);

    qh_freeqhull (&qhdata, !qh_ALL);

    int curlong, totlong;
    qh_memfreeshort (&qhdata, &curlong, &totlong);
    CHECK(curlong || totlong, return 1);

    std::free(adjst.adj);

    return 0;
}

int vorvol (coordT *deladjs, coordT *points, pointT *intpoints, int64_t numpoints, float *vol)
{
    int status;

    const int dim = 3; const boolT ismalloc = False;
    std::FILE *outfile = stdout, *errfile = stderr;
    int curlong, totlong;

    for (int64_t ii=0; ii<numpoints; ++ii)
    {
        coordT *deladj = deladjs + 3*ii, *point = points + 4*ii;
        coordT runsum = 0.0;
        for (int jj=0; jj<3; ++jj)
        {
            runsum += deladj[jj] * deladj[jj];
            point[jj] = deladj[jj];
        }
        point[3] = -0.5 * runsum;
    }

    qhT qhdata;
    char cmd1[] = "qhull H0";
    status = qh_new_qhull(&qhdata, 4, numpoints, points, ismalloc, cmd1, outfile, errfile);
    CHECK(status, return 1);

    numpoints = 0;
    facetT *facet;
    qhT *qh = &qhdata; // required for FORALLfacets
    FORALLfacets
    {
        CHECK(qh_get_feasible_point(&qhdata), return 1);
        coordT *coordp = intpoints + 3*numpoints;
        coordT *point = coordp; // unused?
        coordT *normp = facet->normal;
        coordT *feasiblep = qh_get_feasible_point(&qhdata);
        if (facet->offset < -qhdata.MINdenom)
            for (int jj=qhdata.hull_dim; jj--; )
                *(coordp++) = ( *(normp++) / - facet->offset ) + *(feasiblep++);
        else
            for (int jj=qhdata.hull_dim; jj--; )
            {
                boolT zerodiv;
                *(coordp++) = qh_divzero(*(normp++), facet->offset, qhdata.MINdenom_1, &zerodiv)
                              + *(feasiblep++);
                CHECK(zerodiv, return 1);
            }

        ++numpoints;
    }

    qh_freeqhull(&qhdata, !qh_ALL);
    qh_memfreeshort(&qhdata, &curlong, &totlong);

    char cmd2[] = "qhull FA";
    status = qh_new_qhull(&qhdata, dim, numpoints, intpoints, ismalloc, cmd2, outfile, errfile);
    CHECK(status, return 1);

    qh_getarea(&qhdata, qhdata.facet_list);
    *vol = qhdata.totvol;

    qh_freeqhull(&qhdata, !qh_ALL);
    qh_memfreeshort(&qhdata, &curlong, &totlong);
    CHECK(curlong || totlong, return 1);

    return 0;
}
