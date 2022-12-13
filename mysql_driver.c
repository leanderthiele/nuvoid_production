#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include <mysql.h>

// possible commands:
// [set_cosmologies]
//     checks /scratch for available cosmo_varied cosmologies
// [get_cosmology]
//     returns index of a cosmology with smallest number of available lightcones
//     argv[2] = some random string for seeding
// [create_trial]
//     returns cosmo_idx,hod_idx
//     argv[2] = some random string for seeding
// [start_trial]
//     argv[2] = cosmo_idx
//     argv[3] = hod_idx
//     argv[4] = hod_hash
// [end_trial]
//     argv[2] = cosmo_idx
//     argv[3] = hod_idx
//     argv[4] = state (0=success, nonzero=failure)

// contents of the database
//
// [cosmologies]
// cosmo_idx (integer)
// num_lc (integer)
//
// [lightcones]
// hod_idx (integer, auto-increment)
// cosmo_idx (integer)
// hod_hash (char 32)
// state ('created','running','fail','success')

// settings for the database
const char db_hst[] = "tigercpu",
           db_usr[] = "lightconesusr",
           db_pwd[] = "pwd",
           db_nme[] = "lightconesdb",
           db_skt[] = "/home/lthiele/mysql/mysql.sock";
const unsigned int
           db_prt = 3310;

// forward declaration from valid_outputs.c
void check_cosmos (const char *pattern,
                   int *Nvalid, char **valid_paths, 
                   int *Ninvalid, char **invalid_paths);

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

#define SAFE_MYSQL(expr) \
    do { \
        int err = (int)(expr); \
        if (err) \
        { \
            fprintf(stderr, "mysql error %d\n", err); \
            assert(0); \
        } \
    } while(0)

void set_cosmologies (MYSQL *p)
{
    const char pattern[] = "cosmo_varied_*";
    int Nvalid;
    char *valid_paths[1024];
    check_cosmos(pattern, &Nvalid, valid_paths, NULL, NULL);

    char query_buffer[1024];
    MYSQL_RES *query_res;

    for (int ii=0; ii<Nvalid; ++ii)
    {
        int cosmo_idx = get_cosmo_idx(valid_paths[ii]);
        sprintf(query_buffer, "SELECT * FROM cosmologies WHERE cosmo_idx=%d", cosmo_idx);
        SAFE_MYSQL(mysql_query(p, query_buffer));
        query_res = mysql_store_result(p);
        uint64_t num_rows = mysql_num_rows(query_res);
        mysql_free_result(query_res);
        if (!num_rows) // not in the database
        {
            sprintf(query_buffer, "INSERT INTO cosmologies VALUES (%d, 0)", cosmo_idx);
            SAFE_MYSQL(mysql_query(p, query_buffer));
        }
        else
            assert(num_rows==1);
    }
}

int get_cosmology (MYSQL *p, const char *seed)
{
    MYSQL_RES *query_res;

    SAFE_MYSQL(mysql_query(p, "SELECT cosmo_idx FROM cosmologies WHERE num_lc=(SELECT MIN(num_lc) FROM cosmologies)"));
    query_res = mysql_store_result(p);
    uint64_t num_rows = mysql_num_rows(query_res);
    assert(num_rows);
    unsigned int num_fields = mysql_num_fields(query_res);
    assert(num_fields==1);

    int cosmo_indices[num_rows];

    MYSQL_ROW row;
    for (int ii=0; ii<num_rows; ++ii)
    {
        row = mysql_fetch_row(query_res);
        assert(row);
        cosmo_indices[ii] = atoi(row[0]);
    }

    mysql_free_result(query_res);

    // hash the seed string
    unsigned long hash = 5381;
    int c;
    while ((c = *seed++))
        hash = ((hash << 5) + hash) + c;
    int rand_row = hash % num_rows;

    return cosmo_indices[rand_row];
}

int create_trial (MYSQL *p, const char *seed, uint64_t *hod_idx)
{
    int cosmo_idx = get_cosmology(p, seed);

    char query_buffer[1024];
    sprintf(query_buffer, "INSERT INTO lightcones (cosmo_idx, state) VALUES (%d, 'created')", cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    *hod_idx = mysql_insert_id(p);
    assert(*hod_idx);

    return cosmo_idx;
}

void start_trial (MYSQL *p, int cosmo_idx, uint64_t hod_idx, const char *hod_hash)
{
    char query_buffer[1024];
    sprintf(query_buffer, "UPDATE lightcones SET hod_hash=%s, state='running' WHERE hod_idx=%lu AND cosmo_idx=%d",
                          hod_hash, hod_idx, cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

void end_trial (MYSQL *p, int cosmo_idx, uint64_t hod_idx, int state)
{
    uint64_t num_rows;
    char query_buffer[1024];

    sprintf(query_buffer, "UPDATE lightcones SET state='%s' WHERE hod_idx=%lu AND cosmo_idx=%d",
                          (state) ? "fail" : "success", hod_idx, cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    num_rows = mysql_affected_rows(p);
    assert(num_rows==1);

    if (!state) // only successful trials are counted
    {
        sprintf(query_buffer, "UPDATE cosmologies SET num_lc=num_lc+1 WHERE cosmo_idx=%d", cosmo_idx);
        SAFE_MYSQL(mysql_query(p, query_buffer));
        num_rows = mysql_affected_rows(p);
        assert(num_rows==1);
    }
}

int main(int argc, char **argv)
{
    const char *mode = argv[1];

    SAFE_MYSQL(mysql_library_init(0, NULL, NULL));

    MYSQL p;
    mysql_init(&p);
    MYSQL *q = mysql_real_connect(&p, db_hst, db_usr, db_pwd, db_nme, db_prt, db_skt, /*client_flag=*/0);
    if (!q)
    {
        fprintf(stderr, "mysql connection failed!\n");
        assert(0);
    }

    if (!strcmp(mode, "set_cosmologies"))
        set_cosmologies(&p);
    else if (!strcmp(mode, "get_cosmology"))
    {
        int cosmo_idx = get_cosmology(&p, argv[2]);
        fprintf(stdout, "%d\n", cosmo_idx);
    }
    else if (!strcmp(mode, "create_trial"))
    {
        uint64_t hod_idx;
        int cosmo_idx = create_trial(&p, argv[2], &hod_idx);
        fprintf(stdout, "%d,%lu\n", cosmo_idx, hod_idx);
    }
    else if (!strcmp(mode, "start_trial"))
    {
        start_trial(&p, atoi(argv[2]), atoll(argv[3]), argv[4]);
    }
    else if (!strcmp(mode, "end_trial"))
    {
        end_trial(&p, atoi(argv[2]), atoll(argv[3]), atoi(argv[4]));
    }
    else
    {
        fprintf(stderr, "invalid mode\n");
        assert(0);
    }

    // these return void
    mysql_close(&p);
    mysql_library_end();
}
