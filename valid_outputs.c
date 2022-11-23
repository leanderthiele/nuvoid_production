/* Can be compiled to an executable, by defining EXE, or to an object file to be used by
 * other code
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <glob.h>
#include <unistd.h>
#include <sys/stat.h>

void check_cosmos (const char *pattern,
                   int *Nvalid, char **valid_paths, 
                   int *Ninvalid, char **invalid_paths)
{
    if (Nvalid) *Nvalid = 0;
    if (Ninvalid) *Ninvalid = 0;

    static const char root[] = "/scratch/gpfs/lthiele/nuvoid_production";

    char buffer[1024];
    glob_t glob_result, glob_result2, glob_result3;
    struct stat st;

    // find the available cosmologies
    sprintf(buffer, "%s/%s", root, pattern);
    glob(buffer, GLOB_TILDE_CHECK | GLOB_ONLYDIR | GLOB_NOSORT, NULL, &glob_result);

    // figure out which ones are valid / done
    for (int ii=0; ii<glob_result.gl_pathc; ++ii)
    {
        sprintf(buffer, "%s/rockstar_0.*[0-9]", glob_result.gl_pathv[ii]);
        glob(buffer, GLOB_TILDE_CHECK | GLOB_ONLYDIR | GLOB_NOSORT, NULL, &glob_result2);

        // hardcoded expected number of snapshots here!
        if (glob_result2.gl_pathc != 20) goto invalid;

        for (int jj=0; jj<glob_result2.gl_pathc; ++jj)
        {
            sprintf(buffer, "%s/out_*[0-9]_hosts.bf", glob_result2.gl_pathv[jj]);
            glob(buffer, GLOB_TILDE_CHECK | GLOB_ONLYDIR | GLOB_NOSORT, NULL, &glob_result3);
            if (glob_result3.gl_pathc != 1) goto invalid;

            // this is the last file being written
            sprintf(buffer, "%s/Header/attr-v2", glob_result3.gl_pathv[0]);
            if (access(buffer, F_OK)) goto invalid;

            // this can also be an issue for some reason
            sprintf(buffer, "%s/Pos/000000", glob_result3.gl_pathv[0]);
            if (access(buffer, F_OK)) goto invalid;
            stat(buffer, &st);
            if (st.st_size < 1000*6*4) goto invalid;
        }

        // all checks passed, valid run
        if (Nvalid && valid_paths)
        {
            valid_paths[*Nvalid] = (char *)malloc(1+strlen(glob_result.gl_pathv[ii]));
            sprintf(valid_paths[(*Nvalid)++], "%s", glob_result.gl_pathv[ii]);
        }
        continue;

        invalid:
        if (Ninvalid && invalid_paths)
        {
            invalid_paths[*Ninvalid] = (char *)malloc(1+strlen(glob_result.gl_pathv[ii]));
            sprintf(invalid_paths[(*Ninvalid)++], "%s", glob_result.gl_pathv[ii]);
        }
        continue;
    }
    
    globfree(&glob_result); globfree(&glob_result2); globfree(&glob_result3);
}

#ifdef EXE
int main (int argc, char **argv)
{
    const char *pattern = argv[1];
    int Nvalid, Ninvalid;
    char *valid_paths[1024], *invalid_paths[1024];
    check_cosmos(pattern, &Nvalid, valid_paths, &Ninvalid, invalid_paths);
    printf("VALID:\n");
    for (int ii=0; ii<Nvalid; ++ii)
        printf("%s\n", valid_paths[ii]);
    printf("INVALID:\n");
    for (int ii=0; ii<Ninvalid; ++ii)
        printf("%s\n", invalid_paths[ii]);

    for (int ii=0; ii<Nvalid; ++ii)
        free(valid_paths[ii]);
    for (int ii=0; ii<Ninvalid; ++ii)
        free(invalid_paths[ii]);
}
#endif
