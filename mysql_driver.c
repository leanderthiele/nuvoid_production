#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <mysql.h>

// usage information, printed when no arguments
const char *usage = R""""(
possible commands:
[new_cosmologies]
    replaces cosmologies table by newly created one
[new_lightcones]
    replaces lightcones table by newly created one
[set_cosmologies]
    checks /scratch for available cosmo_varied cosmologies
[get_cosmology]
    returns index of a cosmology with smallest number of available lightcones
    argv[2] = some random string for seeding
[create_trial]
    returns cosmo_idx hod_idx
    argv[2] = some random string for seeding
[start_trial]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
    argv[4] = hod_hash
[end_trial]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
    argv[4] = state (0=success, nonzero=failure)
[create_plk]
    returns cosmo_idx hod_idx hod_hash
[start_plk]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
[end_plk]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
    argv[4] = state
[reset_lightcones]
    CAUTION: this deletes all data!
[timeout_old_lightcones]
    argv[2] = minutes
[timeout_old_plk]
    argv[2] = minutes
)"""";

// contents of the database
const char *cosmologies_columns =
    "cosmo_idx INT UNSIGNED NOT NULL, "
    "num_lc INT UNSIGNED NOT NULL, "
    "PRIMARY KEY (cosmo_idx)";

const char *lightcones_columns =
    "hod_idx BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, "
    "cosmo_idx INT UNSIGNED NOT NULL, "
    "hod_hash CHAR(40), "
    "state ENUM('created', 'running', 'fail', 'success', 'timeout') NOT NULL, "
    "create_time BIGINT NOT NULL, "
    "plk_state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "plk_create_time BIGINT, "
    "PRIMARY KEY (hod_idx)";

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

// this assumes buffer is a stack array!
#define MYSPRINTF(buffer, ...) \
    do { \
        size_t bufsz = sizeof(buffer); \
        size_t would_write = snprintf(buffer, bufsz, __VA_ARGS__); \
        if (would_write > bufsz-1) \
        { \
            fprintf(stderr, "Exceeded buffer size\n"); \
            assert(0); \
        } \
    } while (0)


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
        MYSPRINTF(query_buffer, "SELECT * FROM cosmologies WHERE cosmo_idx=%d", cosmo_idx);
        SAFE_MYSQL(mysql_query(p, query_buffer));
        query_res = mysql_store_result(p);
        uint64_t num_rows = mysql_num_rows(query_res);
        mysql_free_result(query_res);
        if (!num_rows) // not in the database
        {
            MYSPRINTF(query_buffer, "INSERT INTO cosmologies VALUES (%d, 0)", cosmo_idx);
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

    time_t now = time(NULL); // 8-byte signed integer

    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "INSERT INTO lightcones (cosmo_idx, state, create_time)"
              "VALUES (%d, 'created', %ld)", cosmo_idx, now);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    *hod_idx = mysql_insert_id(p);
    assert(*hod_idx);

    return cosmo_idx;
}

void start_trial (MYSQL *p, int cosmo_idx, uint64_t hod_idx, const char *hod_hash)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE lightcones SET hod_hash='%s', state='running' WHERE hod_idx=%lu AND cosmo_idx=%d",
              hod_hash, hod_idx, cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

void end_trial (MYSQL *p, int cosmo_idx, uint64_t hod_idx, int state)
{
    uint64_t num_rows;
    char query_buffer[1024];

    MYSPRINTF(query_buffer,
              "UPDATE lightcones SET state='%s' WHERE hod_idx=%lu AND cosmo_idx=%d",
              (state) ? "fail" : "success", hod_idx, cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    num_rows = mysql_affected_rows(p);
    assert(num_rows==1);

    if (!state) // only successful trials are counted
    {
        MYSPRINTF(query_buffer,
                  "UPDATE cosmologies SET num_lc=num_lc+1 WHERE cosmo_idx=%d",
                  cosmo_idx);
        SAFE_MYSQL(mysql_query(p, query_buffer));
        num_rows = mysql_affected_rows(p);
        assert(num_rows==1);
    }
}

int create_plk (MYSQL *p, uint64_t *hod_idx, char *hod_hash)
{
    time_t now = time(NULL);

    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "SET @updated_hod_idx := 0; "
              "UPDATE lightcones SET plk_state='created', "
              "plk_create_time=%ld, "
              "hod_idx=(SELECT @updated_hod_idx := hod_idx), "
              "WHERE state='success' AND plk_state IS NULL LIMIT 1; "
              "SELECT cosmo_idx, hod_idx, hod_hash FROM lightcones WHERE hod_idx=@updated_hod_idx;",
              now);
    SAFE_MYSQL(mysql_query(p, query_buffer));

    MYSQL_RES *query_res;

    int cosmo_idx = -1;

    // iterate through multi-statement results
    while (1)
    {
        query_res = mysql_store_result(p);
        if (query_res)
        {
            uint64_t num_rows = mysql_num_rows(query_res);
            if (num_rows == 1) // this is our result
            {
                unsigned int num_fields = mysql_num_fields(query_res);
                assert(num_fields == 3);
                MYSQL_ROW row = mysql_fetch_row(query_res);
                assert(row);
                cosmo_idx = atoi(row[0]);
                *hod_idx = atoll(row[1]);
                sprintf(hod_hash, "%s", row[2]);
            }
            mysql_free_result(query_res);
        }
        int status = mysql_next_result(p);
        assert(status<=0);
        if (status) break;
    }

    assert(cosmo_idx>=0);
    return cosmo_idx;
}

void start_plk (MYSQL *p, int cosmo_idx, uint64_t hod_idx)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE lightcones SET plk_state='running' WHERE hod_idx=%lu AND cosmo_idx=%d",
              hod_idx, cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

void end_plk (MYSQL *p, int cosmo_idx, uint64_t hod_idx, int state)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE lightcones SET plk_state='%s' WHERE hod_idx=%lu AND cosmo_idx=%d",
              (state) ? "fail" : "success", hod_idx, cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

void reset_lightcones (MYSQL *p)
{
    SAFE_MYSQL(mysql_query(p,
                           "DELETE FROM lightcones;"
                           "ALTER TABLE lightcones AUTO_INCREMENT=1;"
                           "UPDATE cosmologies SET num_lc=0;"
                          ));
}

void timeout_old_lightcones (MYSQL *p, float timeout_minutes)
{
    assert(timeout_minutes>60);
    time_t now = time(NULL);
    time_t cutoff = now - (time_t)(60 * timeout_minutes);
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE lightcones SET state='timeout' "
              "WHERE state='running' AND create_time<%ld",
              cutoff);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    fprintf(stdout, "Failed %lu trials\n", num_rows);
}

void timeout_old_plk (MYSQL *p, float timeout_minutes)
{
    assert(timeout_minutes>60);
    time_t now = time(NULL);
    time_t cutoff = now - (time_t)(60 * timeout_minutes);
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE lightcones SET plk_state='timeout' "
              "WHERE plk_state='running' AND plk_create_time<%ld",
              cutoff);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    fprintf(stdout, "Nulled %lu plks\n", num_rows);
}

void new_table (MYSQL *p, const char *name, const char *columns)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "DROP TABLE %s; "
              "CREATE TABLE %s (%s);",
              name, name, columns);
    SAFE_MYSQL(mysql_query(p, query_buffer));

    char spec_buffer[1024];
    char *dst = spec_buffer;
    const char *src = columns;
    for (; *src; ++src, ++dst)
        if (*src==',') *dst = '\n';
        else *dst = *src;
    *dst = '\0';

    fprintf(stdout, "Created table %s with the columns:\n%s\n", name, spec_buffer);
}

void new_cosmologies (MYSQL *p)
{
    new_table(p, "cosmologies", cosmologies_columns);
}

void new_lightcones (MYSQL *p)
{
    new_table(p, "lightcones", lightcones_columns);
}

int main(int argc, char **argv)
{
    if (argc==1)
    {
        fprintf(stderr, "%s", usage);
        return -1;
    }

    const char *mode = argv[1];

    SAFE_MYSQL(mysql_library_init(0, NULL, NULL));

    MYSQL p;
    mysql_init(&p);
    MYSQL *q = mysql_real_connect(&p, db_hst, db_usr, db_pwd, db_nme, db_prt, db_skt,
                                  /*client_flag=*/CLIENT_MULTI_STATEMENTS);
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
        fprintf(stdout, "%d %lu\n", cosmo_idx, hod_idx);
    }
    else if (!strcmp(mode, "start_trial"))
    {
        start_trial(&p, atoi(argv[2]), atoll(argv[3]), argv[4]);
    }
    else if (!strcmp(mode, "end_trial"))
    {
        end_trial(&p, atoi(argv[2]), atoll(argv[3]), atoi(argv[4]));
    }
    else if (!strcmp(mode, "create_plk"))
    {
        uint64_t hod_idx;
        char hod_hash[40];
        int cosmo_idx = create_plk(&p, &hod_idx, hod_hash);
        fprintf(stdout, "%d %lu %s\n", cosmo_idx, hod_idx, hod_hash);
    }
    else if (!strcmp(mode, "start_plk"))
    {
        start_plk(&p, atoi(argv[2]), atoll(argv[3]));
    }
    else if (!strcmp(mode, "end_plk"))
    {
        end_plk(&p, atoi(argv[2]), atoll(argv[3]), atoi(argv[4]));
    }
    else if (!strcmp(mode, "reset_lightcones"))
    {
        reset_lightcones(&p);
    }
    else if (!strcmp(mode, "timeout_old_lightcones"))
    {
        timeout_old_lightcones(&p, atof(argv[2]));
    }
    else if (!strcmp(mode, "timeout_old_plk"))
    {
        timeout_old_plk(&p, atof(argv[2]));
    }
    else if (!strcmp(mode, "new_cosmologies"))
    {
        new_cosmologies(&p);
    }
    else if (!strcmp(mode, "new_lightcones"))
    {
        new_lightcones(&p);
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
