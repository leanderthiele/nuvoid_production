/* Super ugly and unmaintainable script to normalize the SLURM nodelist format
 * into a comma-separated list of individual nodes.
 * Need this for buggy conda mpi version that ships with nbodykit.
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    const char *slurm_nodelist = argv[1];

    char *result = (char *)malloc(10000);
    char *result_dest = result;

    char *primary_group = (char *)malloc(4096);
    char *primary_group_dest = primary_group;

    int in_parentheses = 0;
    for (const char *c = slurm_nodelist; ; ++c)
    {
        if ((*c == ',' && !in_parentheses) || !*c)
        // we finished a primary group, process into result
        {
            // finish and reset
            *primary_group_dest = '\0';
            primary_group_dest = primary_group;

            char root[64]; // this is only tiger-... so a small buffer is safe
            char *root_dest = root;
            const char *parentheses_start = NULL;
            for (const char *d = primary_group; *d; ++d)
            {
                if (*d == '[')
                {
                    parentheses_start = d;
                    break;
                }
                *(root_dest++) = *d;
            }
            *root_dest = '\0';

            if (!parentheses_start)
            // easy case
            {
                for (const char *d = root; *d; ++d)
                    *(result_dest++) = *d;
                *(result_dest++) = ',';
            }
            else
            // difficult case
            {
                char lo[4]; char *lo_dest = lo;
                char hi[4]; char *hi_dest = hi;
                int after_dash = 0;
                for (const char *d = parentheses_start+1; ; ++d)
                {
                    if (*d >= '0' && *d <= '9') // a digit
                    {
                        if (after_dash) *(hi_dest++) = *d;
                        else *(lo_dest++) = *d;
                    }
                    else if (*d == '-') // found dash
                        after_dash = 1;
                    else if (*d == ',' || *d == ']') // found separator, write into output
                    {
                        assert(lo_dest != lo); // something has been written
                        *lo_dest = '\0'; *hi_dest = '\0';
                        if (!after_dash) // only a single number
                        {
                            assert(hi_dest == hi);
                            int written = sprintf(result_dest, "%s%s,", root, lo);
                            assert(written>0);
                            result_dest += written;
                        }
                        else // difficult case
                        {
                            int lo_int = atoi(lo); int hi_int = atoi(hi);
                            assert(lo_int<hi_int);
                            for (int ii=lo_int; ii<=hi_int; ++ii)
                            {
                                int written = sprintf(result_dest, "%s%d,", root, ii);
                                assert(written>0);
                                result_dest += written;
                            }
                        }

                        // reset our temporaries
                        lo_dest = lo;
                        hi_dest = hi;
                        after_dash = 0;

                        // check if we are done
                        if (*d == ']') break;
                    }
                }
            }

            if (!*c) break;
        }
        else
        {
            *(primary_group_dest++) = *c;
            if (*c == '[')
            {
                assert(!in_parentheses);
                in_parentheses = 1;
            }
            else if (*c == ']')
            {
                assert(in_parentheses);
                in_parentheses = 0;
            }
        }
    }

    // finish output
    --result_dest;
    assert(*result_dest == ',');
    *result_dest = '\0';

    fprintf(stdout, "%s\n", result);

    free(result); free(primary_group_dest);
}
