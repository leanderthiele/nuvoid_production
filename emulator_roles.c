/* Figure out which cosmology to work on and who copies
 * the data to /tmp.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <glob.h>

#include <mpi.h>

struct ProcInfo
{
    // inputs
    int rank, rank_on_node, node_id;

    // computed
    int cosmo_idx, do_copying;
};

struct CosmoInfo
{
    char *path;
    int cosmo_idx, Nsamples;
};

int compare_node_id (const void *a_, const void *b_)
{
    const struct ProcInfo *a = (struct ProcInfo *)a_;
    const struct ProcInfo *b = (struct ProcInfo *)b_;

    return a->node_id - b->node_id;
}

int compare_rank (const void *a_, const void *b_)
{
    const struct ProcInfo *a = (struct ProcInfo *)a_;
    const struct ProcInfo *b = (struct ProcInfo *)b_;

    return a->rank - b->rank;
}

int compare_cosmo (const void *a_, const void *b_)
{
    const struct CosmoInfo *a = (struct CosmoInfo *)a_;
    const struct CosmoInfo *b = (struct CosmoInfo *)b_;

    return a->Nsamples - b->Nsamples;
}

int get_cosmo_idx (const char *path)
{
    static const char pattern[] = "cosmo_varied_";
    const char *p = strstr(path, pattern);
    assert(p);
    char buffer[8];
    int ii=0;
    for (const char *c = p+strlen(pattern); *c && *c>='0' && *c<='9'; ++c, ++ii)
        buffer[ii++] = *c;
    buffer[ii] = 0;
    return atoi(buffer);
}

void assign_cosmo_indices (int N, int *out)
{
    static const char root[] = "/scratch/gpfs/lthiele/nuvoid_production";

    char buffer[1024];
    glob_t glob_result, glob_result1;

    // find the available cosmologies
    sprintf(buffer, "%s/cosmo_varied_*[0-9]", root);
    glob(buffer, GLOB_TILDE_CHECK | GLOB_ONLYDIR | GLOB_NOSORT, NULL, &glob_result);

    int Ncosmo = glob_result.gl_pathc;
    assert(Ncosmo >= N);
    struct CosmoInfo *cosmo_infos = (struct CosmoInfo *)malloc(Ncosmo * sizeof(struct CosmoInfo));

    for (int ii=0; ii<Ncosmo; ++ii)
    {
        cosmo_infos[ii].path = glob_result.gl_pathv[ii];
        sprintf(buffer, "%s/emulator/*[a-f,0-9]", cosmo_infos[ii].path);
        glob(buffer, GLOB_TILDE_CHECK | GLOB_ONLYDIR | GLOB_NOSORT, NULL, &glob_result1);
        cosmo_infos[ii].Nsamples = glob_result1.gl_pathc;
        cosmo_infos[ii].cosmo_idx = get_cosmo_idx(cosmo_infos[ii].path);
    }

    qsort(cosmo_infos, Ncosmo, sizeof(struct CosmoInfo), compare_cosmo);

    for (int ii=0; ii<N; ++ii)
        out[ii] = cosmo_infos[ii].cosmo_idx;
}

int main(int argc, char **argv)
{
    int rank, world_size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // the node-local rank (the node-local main will do the copying)
    const int rank_on_node = atoi(getenv("SLURM_LOCALID"));
    
    // figure out which node we are on (this is relative to this job!)
    const int node_id = atoi(getenv("SLURM_NODEID"));

    // gather info at root
    int *rank_arr=0, *rank_on_node_arr=0, *node_id_arr=0;
    if (!rank)
    {
        rank_arr = (int *)malloc(world_size * sizeof(int));
        rank_on_node_arr = (int *)malloc(world_size * sizeof(int));
        node_id_arr = (int *)malloc(world_size * sizeof(int));
    }

    MPI_Gather(&rank, 1, MPI_INT, rank_arr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&rank_on_node, 1, MPI_INT, rank_on_node_arr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&node_id, 1, MPI_INT, node_id_arr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *cosmo_idx_arr=0, *do_copying_arr=0, *rank2_arr=0;

    if (!rank) // now figure stuff out
    {
        struct ProcInfo *proc_infos = (struct ProcInfo *)malloc(world_size * sizeof(struct ProcInfo));
        for (int ii=0; ii<world_size; ++ii)
        {
            assert(rank_arr[ii] == ii); // this should be true I believe
            proc_infos[ii].rank = rank_arr[ii];
            proc_infos[ii].rank_on_node = rank_on_node_arr[ii];
            proc_infos[ii].node_id = node_id_arr[ii];
        }

        // sort by nodes
        qsort(proc_infos, world_size, sizeof(struct ProcInfo), compare_node_id);

        // assign copying
        int current_node_id = -1;
        int num_nodes = 0;
        for (int ii=0; ii<world_size; ++ii)
            if (current_node_id != proc_infos[ii].node_id)
            {
                ++num_nodes;
                current_node_id = proc_infos[ii].node_id;
                proc_infos[ii].do_copying = 1;
            }
            else
                proc_infos[ii].do_copying = 0;

        int num_nodes1 = atoi(getenv("SLURM_JOB_NUM_NODES"));
        assert(num_nodes == num_nodes1);

        // get the cosmology indices
        int cosmo_indices[num_nodes];
        assign_cosmo_indices(num_nodes, cosmo_indices);

        current_node_id = -1;
        int counter = -1;
        for (int ii=0; ii<world_size; ++ii)
        {
            if (current_node_id != proc_infos[ii].node_id)
            {
                ++counter;
                current_node_id = proc_infos[ii].node_id;
            }
            proc_infos[ii].cosmo_idx = cosmo_indices[counter];
        }

        cosmo_idx_arr = (int *)malloc(world_size * sizeof(int));
        do_copying_arr = (int *)malloc(world_size * sizeof(int));
        rank2_arr = (int *)malloc(world_size * sizeof(int));

        qsort(proc_infos, world_size, sizeof(struct ProcInfo), compare_rank);
        for (int ii=0; ii<world_size; ++ii)
        {
            assert(proc_infos[ii].rank == ii);
            cosmo_idx_arr[ii] = proc_infos[ii].cosmo_idx;
            do_copying_arr[ii] = proc_infos[ii].do_copying;
            rank2_arr[ii] = proc_infos[ii].rank;
        }
    }

    int cosmo_idx, do_copying, rank2;
    MPI_Scatter(cosmo_idx_arr, 1, MPI_INT, &cosmo_idx, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    MPI_Scatter(do_copying_arr, 1, MPI_INT, &do_copying, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    MPI_Scatter(rank2_arr, 1, MPI_INT, &rank2, 1, MPI_INT, 0, MPI_COMM_WORLD); 

    assert(rank == rank2);

    MPI_Finalize();

    printf("%d %d\n", cosmo_idx, do_copying);
}