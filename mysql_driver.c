#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <mysql.h>

// possible commands:
// [set_cosmologies]
//     checks /scratch for available cosmo_varied cosmologies
// [get_cosmology]
//     returns index of a cosmology with smallest number of available lightcones

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
// state (bool 0--in progress/fail, 1--competed)

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

void get_cosmology (MYSQL *p)
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

    srandom(time(NULL));
    int rand_row = random() % num_rows;

    fprintf(stdout, "%d\n", cosmo_indices[rand_row]);
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
        get_cosmology(&p);
    else
    {
        fprintf(stderr, "invalid mode\n");
        assert(0);
    }

    // these return void
    mysql_close(&p);
    mysql_library_end();
}
