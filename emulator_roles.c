/* Figure out which cosmology to work on and who copies
 * the data to /tmp.
 * Compile with:
 *
 * mpicc -ggdb -O3 -Wall -Wextra -o emulator_roles emulator_roles.c valid_outputs.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <glob.h>

#include <mpi.h>

// environment variables we use
#define ENV_HOSTNAME "SLURM_TOPOLOGY_ADDR"
#define ENV_NUMNODES "SLURM_JOB_NUM_NODES"

// which rank is the root
#define ROOT_RANK 0

// 8 bytes should be enough
typedef int64_t node_id_t;

struct ProcInfo
{
    // inputs
    int rank;
    node_id_t node_id;

    // computed
    int cosmo_idx, do_copying;
};

MPI_Datatype create_mpi_procinfo (void)
{
    const int nitems = 4;
    MPI_Datatype types[] = { MPI_INT, MPI_INT64_T, MPI_INT, MPI_INT };
    int blocklengths[]   = { 1, 1, 1, 1 };
    MPI_Aint offsets[]   = { offsetof(struct ProcInfo, rank),
                             offsetof(struct ProcInfo, node_id),
                             offsetof(struct ProcInfo, cosmo_idx),
                             offsetof(struct ProcInfo, do_copying) };

    MPI_Datatype out;
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &out);

    return out;
}

struct CosmoInfo
{
    const char *path;
    int cosmo_idx, Nsamples;
};

int compare_node_id (const void *a_, const void *b_)
{
    const struct ProcInfo *a = (struct ProcInfo *)a_;
    const struct ProcInfo *b = (struct ProcInfo *)b_;

    // we can't just take the difference as that would be too long
    return (a->node_id > b->node_id) ? 1
           : (a->node_id < b->node_id) ? -1
           : 0;
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
        buffer[ii] = *c;
    buffer[ii] = 0;
    return atoi(buffer);
}

// from valid_outputs.c
void check_cosmos (const char *pattern,
                   int *Nvalid, char **valid_paths, 
                   int *Ninvalid, char **invalid_paths);

void assign_cosmo_indices (int N, int *out)
{
    int Ncosmo;
    char *valid_paths[1024]; // should be large enough
    check_cosmos("cosmo_varied_*[0-9]", &Ncosmo, valid_paths, NULL, NULL);

    char buffer[1024];
    glob_t glob_result1;

    assert(Ncosmo >= N);
    struct CosmoInfo *cosmo_infos = (struct CosmoInfo *)malloc(Ncosmo * sizeof(struct CosmoInfo));

    for (int ii=0; ii<Ncosmo; ++ii)
    {
        cosmo_infos[ii].path = valid_paths[ii];
        sprintf(buffer, "%s/emulator/*[a-f,0-9]/sample_*[0-9]/untrimmed_dencut_centers_central_*[0-9].out", cosmo_infos[ii].path);
        glob(buffer, GLOB_TILDE_CHECK | GLOB_NOSORT, NULL, &glob_result1);
        cosmo_infos[ii].Nsamples = glob_result1.gl_pathc;
        cosmo_infos[ii].cosmo_idx = get_cosmo_idx(cosmo_infos[ii].path);
    }

    qsort(cosmo_infos, Ncosmo, sizeof(struct CosmoInfo), compare_cosmo);

    #ifdef PRINT
    for (int ii=0; ii<Ncosmo; ++ii)
        printf("%s\tcosmo_idx=%d\tsamples=%d\n", cosmo_infos[ii].path, cosmo_infos[ii].cosmo_idx, cosmo_infos[ii].Nsamples);
    #endif

    for (int ii=0; ii<N; ++ii)
        out[ii] = cosmo_infos[ii].cosmo_idx;

    free(cosmo_infos);
    globfree(&glob_result1);
    for (int ii=0; ii<Ncosmo; ++ii)
        free(valid_paths[ii]);
}

node_id_t get_nodeid (void)
{
    // this is something like tiger-i26c2n2
    const char *hostname = getenv(ENV_HOSTNAME);

    // the part after "tiger-" is maximum 8 bytes long (iXXcXnXX)
    // so it fits into a 64bit integer
    // In principle we could compress this better but there's no need
    // (but yeah, this is an implementation detail if there ever was one)
    static const char pattern[] = "tiger-";
    assert(strstr(hostname, pattern) == hostname);

    const char *interesting = hostname + strlen(pattern);
    assert(strlen(interesting) <= sizeof(node_id_t));

    node_id_t out = 0;
    memcpy(&out, interesting, strlen(interesting));

    assert(out); // zero would lead to bug below, but is impossible

    return out;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // figure out which node we are on (this is relative to this job!)
    const node_id_t node_id = get_nodeid();
    #ifdef PRINT
    printf("%s %lu\n", getenv(ENV_HOSTNAME), node_id); // something like tiger-i26c2n2
    #endif

    // gather info at root
    struct ProcInfo proc_info = { .rank = rank, .node_id = node_id };

    struct ProcInfo *proc_infos = NULL;
    if (rank == ROOT_RANK)
        proc_infos = (struct ProcInfo *)malloc(world_size * sizeof(struct ProcInfo));

    // define custom MPI type for ProcInfo struct, need to split node_id in two
    MPI_Datatype MPI_ProcInfo = create_mpi_procinfo();
    MPI_Type_commit(&MPI_ProcInfo);

    MPI_Gather(&proc_info, 1, MPI_ProcInfo, proc_infos, 1, MPI_ProcInfo, ROOT_RANK, MPI_COMM_WORLD);

    if (rank == ROOT_RANK) // now figure stuff out
    {
        // sanity check
        for (int ii=0; ii<world_size; ++ii) assert(proc_infos[ii].rank == ii);

        // sort by nodes
        qsort(proc_infos, world_size, sizeof(struct ProcInfo), compare_node_id);

        // assign copying
        node_id_t current_node_id = 0; // it is clear that this cannot be a valid node_id
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

        // sanity check
        assert(num_nodes == atoi(getenv(ENV_NUMNODES)));

        // get the cosmology indices
        int cosmo_indices[num_nodes];
        assign_cosmo_indices(num_nodes, cosmo_indices);

        current_node_id = 0;
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

        qsort(proc_infos, world_size, sizeof(struct ProcInfo), compare_rank);

        // sanity check
        for (int ii=0; ii<world_size; ++ii) assert(proc_infos[ii].rank == ii);
    }

    // information has been computed, give back to the processes
    MPI_Scatter(proc_infos, 1, MPI_ProcInfo, &proc_info, 1, MPI_ProcInfo, ROOT_RANK, MPI_COMM_WORLD);

    MPI_Type_free(&MPI_ProcInfo);

    if (rank == ROOT_RANK)
        free(proc_infos);

    assert(rank == proc_info.rank);
    assert(node_id == proc_info.node_id);

    MPI_Finalize();

    #ifdef PRINT
    printf("node=%lu\tcosmo_idx=%d\tdo_copying=%d\n",
           proc_info.node_id, proc_info.cosmo_idx, proc_info.do_copying);
    #else
    printf("%d %d\n", proc_info.cosmo_idx, proc_info.do_copying);
    #endif
}
