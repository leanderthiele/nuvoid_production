#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <mysql.h>

// usage information, printed when no arguments
const char *usage = R""""(
possible commands:
[new_cosmologies]
    replaces cosmologies table by newly created one
[new_lightcones]
    replaces lightcones table by newly created one
[new_fiducials]
    replaces fiducials table by newly created one
[new_fiducials_lightcones]
    replaces fiducials_lightcones table by newly created one
[set_cosmologies]
    checks /scratch for available cosmo_varied cosmologies
[set_fiducials]
    checks /scratch for available cosmo_fiducial runs
[get_cosmology]
    returns index of a cosmology with smallest number of available lightcones
    argv[2] = some random string for seeding
[create_trial]
    returns cosmo_idx hod_idx
    argv[2] = some random string for seeding
[create_fiducial]
    returns seed_idx
    argv[2] = hod_hash
[start_trial]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
    argv[4] = hod_hash
[start_fiducial]
    argv[2] = seed_idx
    argv[3] = hod_hash
[end_trial]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
    argv[4] = state (0=success, nonzero=failure)
[end_fiducial]
    argv[2] = seed_idx
    argv[3] = hod_hash
    argv[4] = state
[create_plk]
    returns cosmo_idx hod_idx hod_hash
    (cosmo_idx will be negative if no work is remaining)
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

const char *fiducials_columns =
    "seed_idx INT UNSIGNED NOT NULL, "
    "PRIMARY KEY (cosmo_idx)";

const char *lightcones_columns =
    "hod_idx BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, "
    "cosmo_idx INT UNSIGNED NOT NULL, "
    "hod_hash CHAR(40), "
    "state ENUM('created', 'running', 'fail', 'success', 'timeout') NOT NULL, "
    "create_time BIGINT NOT NULL, "
    "plk_state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "plk_create_time BIGINT, "
    "voids_state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "voids_create_time BIGINT, "
    "PRIMARY KEY (hod_idx)";

const char *fiducial_lightcones_columns =
    "idx BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, "
    "seed_idx INT UNSIGNED NOT NULL, "
    "hod_hash CHAR(40), "
    "state ENUM('created', 'running', 'fail', 'success', 'timeout') NOT NULL, "
    "create_time BIGINT NOT NULL, "
    "plk_state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "plk_create_time BIGINT, "
    "voids_state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "voids_create_time BIGINT, "
    "PRIMARY KEY (idx)";


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

int get_run_idx (const char *path, const char *pattern)
{
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
        assert(bufsz > 32); \
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
        int cosmo_idx = get_run_idx(valid_paths[ii], "cosmo_varied_");
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

void set_fiducials (MYSQL *p)
{
    const char pattern[] = "cosmo_fiducial_*";
    int Nvalid;
    char *valid_paths[1024];
    check_cosmos(pattern, &Nvalid, valid_paths, NULL, NULL);

    char query_buffer[1024];
    MYSQL_RES *query_res;

    for (int ii=0; ii<Nvalid; ++ii)
    {
        int seed_idx = get_run_idx(valid_paths[ii], "cosmo_fiducial_");
        MYSPRINTF(query_buffer, "SELECT * FROM fiducials WHERE seed_idx=%d", seed_idx);
        SAFE_MYSQL(mysql_query(p, query_buffer));
        query_res = mysql_store_result(p);
        uint64_t num_rows = mysql_num_rows(query_res);
        mysql_free_result(query_res);
        if (!num_rows) // not in the database
        {
            MYSPRINTF(query_buffer, "INSERT INTO fiducials VALUES (%d)", seed_idx);
            SAFE_MYSQL(mysql_query(p, query_buffer));
        }
        else
            assert(num_rows==1);
    }
}

int get_cosmology (MYSQL *p, const char *seed)
{
    MYSQL_RES *query_res;

    SAFE_MYSQL(mysql_query(p,
                           "SELECT cosmo_idx FROM cosmologies "
                           "WHERE num_lc=(SELECT MIN(num_lc) FROM cosmologies)"
                          ));
    query_res = mysql_store_result(p);
    uint64_t num_rows = mysql_num_rows(query_res);
    assert(num_rows);
    unsigned int num_fields = mysql_num_fields(query_res);
    assert(num_fields==1);

    // hash the seed string
    unsigned long hash = 5381;
    int c;
    while ((c = *seed++))
        hash = ((hash << 5) + hash) + c;
    uint64_t rand_row = hash % num_rows;

    mysql_data_seek(query_res, rand_row);
    MYSQL_ROW row = mysql_fetch_row(query_res);
    assert(row);
    int cosmo_idx = atoi(row[0]);

    mysql_free_result(query_res);

    return cosmo_idx;
}

int create_trial (MYSQL *p, const char *seed, uint64_t *hod_idx)
{
    int cosmo_idx = get_cosmology(p, seed);

    time_t now = time(NULL); // 8-byte signed integer

    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "INSERT INTO lightcones (cosmo_idx, state, create_time) "
              "VALUES (%d, 'created', %ld)", cosmo_idx, now);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    *hod_idx = mysql_insert_id(p);
    assert(*hod_idx);

    // to avoid everyone going to the same cosmology, we make this a bit fuzzy
    // (so num_lc also has some non-completed trials)
    MYSPRINTF(query_buffer,
              "UPDATE cosmologies SET num_lc=num_lc+1 WHERE cosmo_idx=%d",
              cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);

    return cosmo_idx;
}

int create_fiducial (MYSQL *p, const char *hod_hash)
{
    // TODO deal with timeouts here
    time_t now = time(NULL);

    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "INSERT INTO fiducials_lightcones (seed_idx, hod_hash, state, create_time) "
              "VALUES ((SELECT seed_idx FROM fiducials "
                       "WHERE seed_idx NOT IN "
                       "(SELECT seed_idx FROM fiducials_lightcones WHERE hod_hash != '%s') LIMIT 1), "
                      "'%s', 'created', %ld)",
              hod_hash, hod_hash, now);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t idx = mysql_insert_id(p); // this is the new row we created
    assert(idx);

    MYSPRINTF(query_buffer,
              "SELECT cosmo_idx FROM fiducials_lightcones WHERE idx=%lu",
              idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    MYSQL_RES *query_res = mysql_store_result(p);
    uint64_t num_rows = mysql_num_rows(query_res);
    assert(num_rows==1);
    MYSQL_ROW row = mysql_fetch_row(query_res);
    assert(row);
    int seed_idx = atoi(row[0]);

    mysql_free_result(query_res);

    return seed_idx;
}

void start_trial (MYSQL *p, int cosmo_idx, uint64_t hod_idx, const char *hod_hash)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE lightcones SET hod_hash='%s', state='running' "
              "WHERE hod_idx=%lu AND cosmo_idx=%d",
              hod_hash, hod_idx, cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

void start_fiducial (MYSQL *p, int seed_idx, const char *hod_hash)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE fiducials_lightcones SET state='running' "
              "WHERE seed_idx=%d AND hod_hash='%s'",
              seed_idx, hod_hash);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

void end_trial (MYSQL *p, int cosmo_idx, uint64_t hod_idx, int state)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE lightcones SET state='%s' "
              "WHERE hod_idx=%lu AND cosmo_idx=%d",
              (state) ? "fail" : "success", hod_idx, cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

void end_fiducial (MYSQL *p, int seed_idx, const char *hod_hash, int state)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE fiducials_lightcones SET state='%s' "
              "WHERE seed_idx=%d AND hod_hash='%s'",
              (state) ? "fail" : "success", seed_idx, hod_hash);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

int create_plk (MYSQL *p, uint64_t *hod_idx, char *hod_hash)
{
    time_t now = time(NULL);

    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "SET @updated_hod_idx := 0; "
              "UPDATE lightcones SET plk_state='created', "
              "plk_create_time=%ld, "
              "hod_idx=(SELECT @updated_hod_idx := hod_idx) "
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

    if (cosmo_idx<0)
    // nothing left to work on, fill results to avoid UB
    {
        *hod_idx = 0;
        sprintf(hod_hash, "NONE");
    }

    // NOTE that cosmo_idx can be negative, in which case no work is remaining!
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
    fprintf(stdout, "Timed out %lu trials\n", num_rows);
}

void timeout_old_plk (MYSQL *p, float timeout_minutes)
{
    assert(timeout_minutes>30);
    time_t now = time(NULL);
    time_t cutoff = now - (time_t)(60 * timeout_minutes);
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE lightcones SET plk_state='timeout' "
              "WHERE plk_state='running' AND plk_create_time<%ld",
              cutoff);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    fprintf(stdout, "Timed out %lu plks\n", num_rows);
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
    int in_pars = 0; // in parentheses, here we don't break
    for (; *src; ++src, ++dst)
    {
        if (*src=='(') in_pars = 1;
        else if (*src==')') in_pars = 0;

        if (*src==',' && !in_pars) *dst = '\n';
        else *dst = *src;
    }
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

void new_fiducials (MYSQL *p)
{
    new_table(p, "fiducials", fiducials_columns);
}

void new_fiducials_lightcones (MYSQL *p)
{
    new_table(p, "fiducials_lightcones", fiducial_lightcones_columns);
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

    MYSQL *q;
    for (int ii=1; ii</*connection attempts=*/8; ++ii)
    {
        q = mysql_real_connect(&p, db_hst, db_usr, db_pwd, db_nme, db_prt, db_skt,
                               /*client_flag=*/CLIENT_MULTI_STATEMENTS);
        if (q) break; // successful connection

        // give a bit of time
        sleep(1);
    }

    if (!q)
    {
        fprintf(stderr, "mysql connection failed!\n");
        assert(0);
    }

    if (!strcmp(mode, "set_cosmologies"))
    {
        set_cosmologies(&p);
    }
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
    else if (!strcmp(mode, "new_fiducials"))
    {
        new_fiducials(&p);
    }
    else if (!strcmp(mode, "new_fiducials_lightcones"))
    {
        new_fiducials_lightcones(&p);
    }
    else if (!strcmp(mode, "set_fiducials"))
    {
        set_fiducials(&p);
    }
    else if (!strcmp(mode, "create_fiducial"))
    {
        int seed_idx = create_fiducial(&p, argv[2]);
        fprintf(stdout, "%d\n", seed_idx);
    }
    else if (!strcmp(mode, "start_fiducial"))
    {
        start_fiducial(&p, atoi(argv[2]), argv[3]);
    }
    else if (!strcmp(mode, "end_fiducial"))
    {
        end_fiducial(&p, atoi(argv[2]), argv[3], atoi(argv[4]));
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
