#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
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
    argv[2] = version index
[new_fiducials_lightcones]
    replaces fiducials_lightcones table by newly created one
    argv[2] = version index
[set_cosmologies]
    checks /scratch for available cosmo_varied cosmologies
    and extract the cosmological parameters
[set_fiducials]
    checks /scratch for available cosmo_fiducial runs
    argv[2] = version index
[get_cosmology]
    returns index of a cosmology with smallest number of available lightcones
    argv[2] = some random string for seeding
[create_trial]
    returns cosmo_idx hod_idx
    argv[2] = some random string for seeding
[create_fiducial]
    returns seed_idx
    argv[2] = version index
    argv[3] = hod_hash
[start_trial]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
    argv[4] = hod_hash
[start_fiducial]
    argv[2] = version index
    argv[3] = seed_idx
[end_trial]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
    argv[4] = state (0=success, nonzero=failure)
[end_fiducial]
    argv[2] = version index
    argv[3] = seed_idx
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
[create_fiducials_plk]
    returns running_idx seed_idx lightcone_idx hod_hash
    argv[2] = version index
[start_fiducials_plk]
    argv[2] = version index
    argv[3] = running_idx
    argv[4] = seed_idx
    argv[5] = lightcone_idx
[end_fiducials_plk]
    argv[2] = version index
    argv[3] = running_idx
    argv[4] = seed_idx
    argv[5] = lightcone_idx
    argv[6] = state
[create_voids]
    returns cosmo_idx hod_idx hod_hash
    (cosmo_idx will be negative if no work is remaining)
[start_voids]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
[end_voids]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
    argv[4] = state
[create_fiducials_voids]
    returns running_idx seed_idx lightcone_idx hod_hash
    argv[2] = version index
[start_fiducials_voids]
    argv[2] = version index
    argv[3] = running_idx
    argv[4] = seed_idx
    argv[5] = lightcone_idx
[end_fiducials_voids]
    argv[2] = version index
    argv[3] = running_idx
    argv[4] = seed_idx
    argv[5] = lightcone_idx
    argv[6] = state
[create_vgplk]
    returns cosmo_idx hod_idx hod_hash
    (cosmo_idx will be negative if no work is remaining)
[start_vgplk]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
[end_vgplk]
    argv[2] = cosmo_idx
    argv[3] = hod_idx
    argv[4] = state
[create_fiducials_vgplk]
    returns running_idx seed_idx lightcone_idx hod_hash
    argv[2] = version index
[start_fiducials_vgplk]
    argv[2] = version index
    argv[3] = running_idx
    argv[4] = seed_idx
    argv[5] = lightcone_idx
[end_fiducials_vgplk]
    argv[2] = version index
    argv[3] = running_idx
    argv[4] = seed_idx
    argv[5] = lightcone_idx
    argv[6] = state
[reset_lightcones]
    CAUTION: this deletes all data!
[timeout_old_lightcones]
    argv[2] = minutes
[timeout_old_plk]
    argv[2] = minutes
[get_run]
    argv[2] = hod_idx
    returns cosmo_idx hod_hash state plk_state voids_state vgplk_state
    cosmo_idx == -1 if nothing left
)"""";

// contents of the database
const char *cosmologies_columns =
    "cosmo_idx INT UNSIGNED NOT NULL, "
    "Om DOUBLE NOT NULL, "
    "Ob DOUBLE NOT NULL, "
    "h DOUBLE NOT NULL, "
    "ns DOUBLE NOT NULL, "
    "sigma8 DOUBLE NOT NULL, "
    "S8 DOUBLE NOT NULL, "
    "Mnu DOUBLE NOT NULL, "
    "As DOUBLE NOT NULL, "
    "On DOUBLE NOT NULL, "
    "Oc DOUBLE NOT NULL, "
    "Obh2 DOUBLE NOT NULL, "
    "Och2 DOUBLE NOT NULL, "
    "theta DOUBLE NOT NULL, "
    "logA DOUBLE NOT NULL, "
    "num_lc INT UNSIGNED NOT NULL, "
    "PRIMARY KEY (cosmo_idx)";

const char *fiducials_columns =
    "seed_idx INT UNSIGNED NOT NULL, "
    "hod_hash CHAR(40), "
    "state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "create_time BIGINT, "
    "plk_state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "plk_create_time BIGINT, "
    "voids_state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "voids_create_time BIGINT, "
    "PRIMARY KEY (seed_idx)";

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
    "vgplk_state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "vgplk_create_time BIGINT, "
    "PRIMARY KEY (hod_idx)";

const char *fiducials_lightcones_columns =
    "running_idx BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, "
    "seed_idx INT UNSIGNED NOT NULL, "
    "lightcone_idx INT UNSIGNED NOT NULL, "
    "hod_hash CHAR(40) NOT NULL, "
    "plk_state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "plk_create_time BIGINT, "
    "voids_state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "voids_create_time BIGINT, "
    "vgplk_state ENUM('created', 'running', 'fail', 'success', 'timeout'), "
    "vgplk_create_time BIGINT, "
    "PRIMARY KEY (running_idx)";

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
            unsigned int this_errno = mysql_errno(p); \
            fprintf(stderr, "mysql error %u\n", this_errno); \
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

void read_info (const char *fname, const char *ident, double *out)
{
    FILE *fp = fopen(fname, "r");
    assert(fp);

    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    int found = 0;
    while ((read = getline(&line, &len, fp)) != -1)
        if (strstr(line, ident) == line && *(line+strlen(ident)) == '=')
        {
            const char *eq = strchr(line, '=');
            assert(eq);
            *out = atof(eq+1);
            found = 1;
            break;
        }

    assert(found);
    fclose(fp);
}

void read_cosmology (int cosmo_idx,
                     double *Om, double *Ob, double *h, double *ns, double *sigma8, double *S8,
                     double *Mnu, double *As, double *On, double *Oc, double *Obh2, double *Och2,
                     double *theta, double *logA)
{
    // the CMB parameterization
    #define CODEBASE "/home/lthiele/nuvoid_production"
    const char prior_exe[] = CODEBASE "/sample_prior";
    const int cmb_dim = 5;
    const char cmb_prior[] = CODEBASE "/mu_cov_plikHM_TTTEEE_lowl_lowE.dat";
    const int mnu_dim = 1;
    const char mnu_prior[] = CODEBASE "/mnu_prior.dat";
    #undef CODEBASE
    char cmd_buf[1024];
    MYSPRINTF(cmd_buf, "%s %d %d %s %d %s", prior_exe, cosmo_idx, cmb_dim, cmb_prior, mnu_dim, mnu_prior);
    FILE *fp = popen(cmd_buf, "r");
    assert(fp);
    int res = fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf", Obh2, Och2, theta, logA, ns, Mnu);
    assert(res==6);
    pclose(fp);

    // other parameterizations
    char info_buf[1024];
    MYSPRINTF(info_buf, "/scratch/gpfs/lthiele/nuvoid_production/cosmo_varied_%d/cosmo.info", cosmo_idx);
    read_info(info_buf, "Omega_m", Om);
    read_info(info_buf, "Omega_b", Ob);
    read_info(info_buf, "h", h);
    read_info(info_buf, "sigma_8", sigma8);
    read_info(info_buf, "A_s", As);
    read_info(info_buf, "Omega_nu", On);
    read_info(info_buf, "Omega_cdm", Oc);
    *S8 = *sigma8 * sqrt(*Om / 0.3);
}


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
            double Om, Ob, h, ns, sigma8, S8, Mnu, As, On, Oc, Obh2, Och2, theta, logA;
            read_cosmology(cosmo_idx,
                           &Om, &Ob, &h, &ns, &sigma8, &S8,
                           &Mnu, &As, &On, &Oc, &Obh2, &Och2,
                           &theta, &logA);
            MYSPRINTF(query_buffer,
                      "INSERT INTO cosmologies VALUES "
                      "(%d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, 0)",
                      cosmo_idx, Om, Ob, h, ns, sigma8, S8, Mnu, As, On, Oc, Obh2, Och2, theta, logA);
            SAFE_MYSQL(mysql_query(p, query_buffer));
        }
        else
            assert(num_rows==1);
    }
}

void set_fiducials (MYSQL *p, int version)
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
        MYSPRINTF(query_buffer, "SELECT * FROM fiducials_v%d WHERE seed_idx=%d", version, seed_idx);
        SAFE_MYSQL(mysql_query(p, query_buffer));
        query_res = mysql_store_result(p);
        uint64_t num_rows = mysql_num_rows(query_res);
        mysql_free_result(query_res);
        if (!num_rows) // not in the database
        {
            MYSPRINTF(query_buffer, "INSERT INTO fiducials_v%d (seed_idx) VALUES (%d)", version, seed_idx);
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

int create_fiducial (MYSQL *p, int version, const char *hod_hash)
{
    // TODO deal with timeouts here
    time_t now = time(NULL);

    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "SET @updated_seed_idx := 0; "
              "UPDATE fiducials_v%d SET state='created', "
              "create_time=%ld, "
              "hod_hash='%s', "
              "seed_idx=(SELECT @updated_seed_idx := seed_idx) "
              "WHERE state IS NULL LIMIT 1; "
              "SELECT seed_idx FROM fiducials_v%d WHERE seed_idx=@updated_seed_idx;",
              version, now, hod_hash, version);
    SAFE_MYSQL(mysql_query(p, query_buffer));

    MYSQL_RES *query_res;

    int seed_idx = -1;

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
                assert(num_fields == 1);
                MYSQL_ROW row = mysql_fetch_row(query_res);
                assert(row);
                seed_idx = atoi(row[0]);
            }
            mysql_free_result(query_res);
        }
        int status = mysql_next_result(p);
        assert(status<=0);
        if (status) break;
    }

    // may return -1
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

void start_fiducial (MYSQL *p, int version, int seed_idx)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE fiducials_v%d SET state='running' "
              "WHERE seed_idx=%d",
              version, seed_idx);
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

void end_fiducial (MYSQL *p, int version, int seed_idx, int state)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE fiducials_v%d SET state='%s' "
              "WHERE seed_idx=%d",
              version, (state) ? "fail" : "success", seed_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

int create_summary (MYSQL *p, const char *name, uint64_t *hod_idx, char *hod_hash, const char *depends)
{
    time_t now = time(NULL);

    char depends_buffer[256];
    if (depends)
        MYSPRINTF(depends_buffer, "AND %s_state='success'", depends);

    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "SET @updated_hod_idx := 0; "
              "UPDATE lightcones SET %s_state='created', "
              "%s_create_time=%ld, "
              "hod_idx=(SELECT @updated_hod_idx := hod_idx) "
              "WHERE state='success' AND %s_state IS NULL %s LIMIT 1; "
              "SELECT cosmo_idx, hod_idx, hod_hash FROM lightcones WHERE hod_idx=@updated_hod_idx;",
              name, name, now, name, (depends) ? depends_buffer : "");
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

int create_fiducials_summary (MYSQL *p, int version, const char *name,
                              uint64_t *running_idx, int *lightcone_idx, char *hod_hash, const char *depends)
{
    time_t now = time(NULL);

    char depends_buffer[256];
    if (depends)
        MYSPRINTF(depends_buffer, "AND %s_state='success'", depends);

    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "SET @updated_running_idx := 0; "
              "UPDATE fiducials_lightcones_v%d SET %s_state='created', "
              "%s_create_time=%ld, "
              "running_idx=(SELECT @updated_running_idx := running_idx) "
              "WHERE %s_state IS NULL %s LIMIT 1; "
              "SELECT running_idx, seed_idx, lightcone_idx, hod_hash FROM fiducials_lightcones_v%d "
              "WHERE running_idx=@updated_running_idx;",
              version, name, name, now, name, (depends) ? depends_buffer : "", version);
    SAFE_MYSQL(mysql_query(p, query_buffer));

    MYSQL_RES *query_res;

    int seed_idx = -1;

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
                assert(num_fields == 4);
                MYSQL_ROW row = mysql_fetch_row(query_res);
                assert(row);
                *running_idx = atoll(row[0]);
                seed_idx = atoi(row[1]);
                *lightcone_idx = atoi(row[2]);
                sprintf(hod_hash, "%s", row[3]);
            }
            mysql_free_result(query_res);
        }
        int status = mysql_next_result(p);
        assert(status<=0);
        if (status) break;
    }

    if (seed_idx<0)
    // nothing left to work on, fill results to avoid UB
    {
        *running_idx = 0;
        *lightcone_idx = 0;
        sprintf(hod_hash, "NONE");
    }

    // NOTE that cosmo_idx can be negative, in which case no work is remaining!
    return seed_idx;

}

void start_summary (MYSQL *p, const char *name, int cosmo_idx, uint64_t hod_idx)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE lightcones SET %s_state='running' WHERE hod_idx=%lu AND cosmo_idx=%d",
              name, hod_idx, cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

void start_fiducials_summary (MYSQL *p, int version, const char *name, uint64_t running_idx,
                              int seed_idx, int lightcone_idx)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE fiducials_lightcones_v%d SET %s_state='running' "
              "WHERE running_idx=%lu AND seed_idx=%d AND lightcone_idx=%d",
              version, name, running_idx, seed_idx, lightcone_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

void end_summary (MYSQL *p, const char *name, int cosmo_idx, uint64_t hod_idx, int state)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "UPDATE lightcones SET %s_state='%s' WHERE hod_idx=%lu AND cosmo_idx=%d",
              name, (state) ? "fail" : "success", hod_idx, cosmo_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

void end_fiducials_summary (MYSQL *p, int version, const char *name, uint64_t running_idx, 
                            int seed_idx, int lightcone_idx, int state)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer, 
              "UPDATE fiducials_lightcones_v%d SET %s_state='%s' "
              "WHERE running_idx=%lu AND seed_idx=%d AND lightcone_idx=%d",
              version, name, (state) ? "fail" : "success", running_idx, seed_idx, lightcone_idx);
    SAFE_MYSQL(mysql_query(p, query_buffer));
    uint64_t num_rows = mysql_affected_rows(p);
    assert(num_rows==1);
}

int create_plk (MYSQL *p, uint64_t *hod_idx, char *hod_hash)
{
    return create_summary(p, "plk", hod_idx, hod_hash, NULL);
}

int create_voids (MYSQL *p, uint64_t *hod_idx, char *hod_hash)
{
    return create_summary(p, "voids", hod_idx, hod_hash, NULL);
}

int create_vgplk (MYSQL *p, uint64_t *hod_idx, char *hod_hash)
{
    return create_summary(p, "vgplk", hod_idx, hod_hash, "voids");
}

int create_fiducials_voids (MYSQL *p, int version,
                            uint64_t *running_idx, int *lightcone_idx, char *hod_hash)
{
    return create_fiducials_summary(p, version, "voids", running_idx, lightcone_idx, hod_hash, NULL);
}

int create_fiducials_plk (MYSQL *p, int version,
                          uint64_t *running_idx, int *lightcone_idx, char *hod_hash)
{
    return create_fiducials_summary(p, version, "plk", running_idx, lightcone_idx, hod_hash, NULL);
}

int create_fiducials_vgplk (MYSQL *p, int version,
                            uint64_t *running_idx, int *lightcone_idx, char *hod_hash)
{
    return create_fiducials_summary(p, version, "vgplk", running_idx, lightcone_idx, hod_hash, "voids");
}

void start_plk (MYSQL *p, int cosmo_idx, uint64_t hod_idx)
{
    start_summary(p, "plk", cosmo_idx, hod_idx);
}

void start_voids (MYSQL *p, int cosmo_idx, uint64_t hod_idx)
{
    start_summary(p, "voids", cosmo_idx, hod_idx);
}

void start_vgplk (MYSQL *p, int cosmo_idx, uint64_t hod_idx)
{
    start_summary(p, "vgplk", cosmo_idx, hod_idx);
}

void start_fiducials_voids (MYSQL *p, int version, uint64_t running_idx, int seed_idx, int lightcone_idx)
{
    start_fiducials_summary(p, version, "voids", running_idx, seed_idx, lightcone_idx);
}

void start_fiducials_plk (MYSQL *p, int version, uint64_t running_idx, int seed_idx, int lightcone_idx)
{
    start_fiducials_summary(p, version, "plk", running_idx, seed_idx, lightcone_idx);
}

void start_fiducials_vgplk (MYSQL *p, int version, uint64_t running_idx, int seed_idx, int lightcone_idx)
{
    start_fiducials_summary(p, version, "vgplk", running_idx, seed_idx, lightcone_idx);
}

void end_plk (MYSQL *p, int cosmo_idx, uint64_t hod_idx, int state)
{
    end_summary(p, "plk", cosmo_idx, hod_idx, state);
}

void end_voids (MYSQL *p, int cosmo_idx, uint64_t hod_idx, int state)
{
    end_summary(p, "voids", cosmo_idx, hod_idx, state);
}

void end_vgplk (MYSQL *p, int cosmo_idx, uint64_t hod_idx, int state)
{
    end_summary(p, "vgplk", cosmo_idx, hod_idx, state);
}

void end_fiducials_voids (MYSQL *p, int version, uint64_t running_idx,
                          int seed_idx, int lightcone_idx, int state)
{
    end_fiducials_summary(p, version, "voids", running_idx, seed_idx, lightcone_idx, state);
}

void end_fiducials_plk (MYSQL *p, int version, uint64_t running_idx,
                        int seed_idx, int lightcone_idx, int state)
{
    end_fiducials_summary(p, version, "plk", running_idx, seed_idx, lightcone_idx, state);
}

void end_fiducials_vgplk (MYSQL *p, int version, uint64_t running_idx,
                          int seed_idx, int lightcone_idx, int state)
{
    end_fiducials_summary(p, version, "vgplk", running_idx, seed_idx, lightcone_idx, state);
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
              "DROP TABLE IF EXISTS %s; "
              "CREATE TABLE %s (%s);",
              name, name, columns);
    SAFE_MYSQL(mysql_query(p, query_buffer));

    // iterate through multi-statement results, we don't actually use them
    // but otherwise subsequent calls will fail
    MYSQL_RES *query_res;
    while (1)
    {
        query_res = mysql_store_result(p);
        if (query_res)
            mysql_free_result(query_res);
        int status = mysql_next_result(p);
        assert(status<=0);
        if (status) break; // used up all results
    }

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

void new_fiducials (MYSQL *p, int version)
{
    char buffer[128];
    MYSPRINTF(buffer, "fiducials_v%d", version);
    new_table(p, buffer, fiducials_columns);
}

void new_fiducials_lightcones (MYSQL *p, int version)
{
    static const int max_seed_idx = 100;
    static const int num_lightcones = 96; // number of augmentations

    char buffer[128];
    MYSPRINTF(buffer, "fiducials_lightcones_v%d", version);
    new_table(p, buffer, fiducials_lightcones_columns);

    char query_buffer[1024];
    char hod_hash[40];
    MYSQL_RES *query_res;

    for (int seed_idx=0; seed_idx<max_seed_idx; ++seed_idx)
    {
        MYSPRINTF(query_buffer,
                  "SELECT hod_hash FROM fiducials_v%d WHERE seed_idx=%d AND state='success'",
                  version, seed_idx);
        SAFE_MYSQL(mysql_query(p, query_buffer));
        query_res = mysql_store_result(p);
        uint64_t num_rows = mysql_num_rows(query_res);
        if (!num_rows)
        {
            mysql_free_result(query_res);
            continue;
        }
        assert(num_rows==1);
        unsigned int num_fields = mysql_num_fields(query_res);
        assert(num_fields==1);
        MYSQL_ROW row = mysql_fetch_row(query_res);
        assert(row);
        sprintf(hod_hash, "%s", row[0]);
        mysql_free_result(query_res);

        for (int lightcone_idx=0; lightcone_idx<num_lightcones; ++lightcone_idx)
        {
            MYSPRINTF(query_buffer,
                      "INSERT INTO fiducials_lightcones_v%d (seed_idx, lightcone_idx, hod_hash) "
                      "VALUES (%d, %d, '%s')",
                      version, seed_idx, lightcone_idx, hod_hash);
            SAFE_MYSQL(mysql_query(p, query_buffer));
            uint64_t running_idx = mysql_insert_id(p);
            assert(running_idx);
        }
    }
}

int get_run (MYSQL *p, uint64_t hod_idx, char *hod_hash, char *state,
             char *plk_state, char *voids_state, char *vgplk_state)
{
    char query_buffer[1024];
    MYSPRINTF(query_buffer,
              "SELECT cosmo_idx, hod_hash, state, plk_state, voids_state, vgplk_state "
              "FROM lightcones "
              "WHERE hod_idx=%lu",
              hod_idx);

    SAFE_MYSQL(mysql_query(p, query_buffer));

    MYSQL_RES *query_res = mysql_store_result(p);
    uint64_t num_rows = mysql_num_rows(query_res);
    if (num_rows==0)
    {
        sprintf(hod_hash, "NONE"); sprintf(state, "NONE"); sprintf(plk_state, "NONE");
        sprintf(voids_state, "NONE"); sprintf(vgplk_state, "NONE");
        return -1;
    }
    assert(num_rows==1);

    unsigned int num_fields = mysql_num_fields(query_res);
    assert(num_fields==6);
    MYSQL_ROW row = mysql_fetch_row(query_res);
    int cosmo_idx = atoi(row[0]);

    // need to account for possible NULL here!
    sprintf(hod_hash, "%s", (row[1]) ? row[1] : "NULL");
    sprintf(state, "%s", (row[2]) ? row[2] : "NULL");
    sprintf(plk_state, "%s", (row[3]) ? row[3] : "NULL");
    sprintf(voids_state, "%s", (row[4]) ? row[4] : "NULL");
    sprintf(vgplk_state, "%s", (row[5]) ? row[5] : "NULL");

    mysql_free_result(query_res);

    return cosmo_idx;
}

int main(int argc, char **argv)
{
    if (argc==1)
    {
        fprintf(stderr, "%s", usage);
        return -1;
    }

    const char *mode = argv[1];

    int init_err = mysql_library_init(0, NULL, NULL);
    assert(!init_err);

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
    else if (!strcmp(mode, "create_voids"))
    {
        uint64_t hod_idx;
        char hod_hash[40];
        int cosmo_idx = create_voids(&p, &hod_idx, hod_hash);
        fprintf(stdout, "%d %lu %s\n", cosmo_idx, hod_idx, hod_hash);
    }
    else if (!strcmp(mode, "start_voids"))
    {
        start_voids(&p, atoi(argv[2]), atoll(argv[3]));
    }
    else if (!strcmp(mode, "end_voids"))
    {
        end_voids(&p, atoi(argv[2]), atoll(argv[3]), atoi(argv[4]));
    }
    else if (!strcmp(mode, "create_vgplk"))
    {
        uint64_t hod_idx;
        char hod_hash[40];
        int cosmo_idx = create_vgplk(&p, &hod_idx, hod_hash);
        fprintf(stdout, "%d %lu %s\n", cosmo_idx, hod_idx, hod_hash);
    }
    else if (!strcmp(mode, "start_vgplk"))
    {
        start_vgplk(&p, atoi(argv[2]), atoll(argv[3]));
    }
    else if (!strcmp(mode, "end_vgplk"))
    {
        end_vgplk(&p, atoi(argv[2]), atoll(argv[3]), atoi(argv[4]));
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
        new_fiducials(&p, atoi(argv[2]));
    }
    else if (!strcmp(mode, "set_fiducials"))
    {
        set_fiducials(&p, atoi(argv[2]));
    }
    else if (!strcmp(mode, "create_fiducial"))
    {
        int seed_idx = create_fiducial(&p, atoi(argv[2]), argv[3]);
        fprintf(stdout, "%d\n", seed_idx);
    }
    else if (!strcmp(mode, "start_fiducial"))
    {
        start_fiducial(&p, atoi(argv[2]), atoi(argv[3]));
    }
    else if (!strcmp(mode, "end_fiducial"))
    {
        end_fiducial(&p, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    }
    else if (!strcmp(mode, "get_run"))
    {
        char hod_hash[40], state[40], plk_state[40], voids_state[40], vgplk_state[40];
        int cosmo_idx = get_run(&p, atoll(argv[2]), hod_hash, state, plk_state, voids_state, vgplk_state);
        fprintf(stdout, "%d %s %s %s %s %s\n", cosmo_idx, hod_hash, state, plk_state, voids_state, vgplk_state);
    }
    else if (!strcmp(mode, "new_fiducials_lightcones"))
    {
        new_fiducials_lightcones(&p, atoi(argv[2]));
    }
    else if (!strcmp(mode, "create_fiducials_voids"))
    {
        uint64_t running_idx;
        int lightcone_idx;
        char hod_hash[40];
        int seed_idx = create_fiducials_voids(&p, atoi(argv[2]), &running_idx, &lightcone_idx, hod_hash);
        fprintf(stdout, "%lu %d %d %s\n", running_idx, seed_idx, lightcone_idx, hod_hash);
    }
    else if (!strcmp(mode, "start_fiducials_voids"))
    {
        start_fiducials_voids(&p, atoi(argv[2]), atoll(argv[3]), atoi(argv[4]), atoi(argv[5]));
    }
    else if (!strcmp(mode, "end_fiducials_voids"))
    {
        end_fiducials_voids(&p, atoi(argv[2]), atoll(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
    }
    else if (!strcmp(mode, "create_fiducials_plk"))
    {
        uint64_t running_idx;
        int lightcone_idx;
        char hod_hash[40];
        int seed_idx = create_fiducials_plk(&p, atoi(argv[2]), &running_idx, &lightcone_idx, hod_hash);
        fprintf(stdout, "%lu %d %d %s\n", running_idx, seed_idx, lightcone_idx, hod_hash);
    }
    else if (!strcmp(mode, "start_fiducials_plk"))
    {
        start_fiducials_plk(&p, atoi(argv[2]), atoll(argv[3]), atoi(argv[4]), atoi(argv[5]));
    }
    else if (!strcmp(mode, "end_fiducials_plk"))
    {
        end_fiducials_plk(&p, atoi(argv[2]), atoll(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
    }
    else if (!strcmp(mode, "create_fiducials_vgplk"))
    {
        uint64_t running_idx;
        int lightcone_idx;
        char hod_hash[40];
        int seed_idx = create_fiducials_vgplk(&p, atoi(argv[2]), &running_idx, &lightcone_idx, hod_hash);
        fprintf(stdout, "%lu %d %d %s\n", running_idx, seed_idx, lightcone_idx, hod_hash);
    }
    else if (!strcmp(mode, "start_fiducials_vgplk"))
    {
        start_fiducials_vgplk(&p, atoi(argv[2]), atoll(argv[3]), atoi(argv[4]), atoi(argv[5]));
    }
    else if (!strcmp(mode, "end_fiducials_vgplk"))
    {
        end_fiducials_vgplk(&p, atoi(argv[2]), atoll(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
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
