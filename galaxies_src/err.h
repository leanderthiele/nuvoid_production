#ifndef ERR_H
#define ERR_H

#include <cstdio>

#define CHECK(status, cmd) \
    do \
    { \
        if (status) \
        { \
            std::fprintf(stderr, "[Error] at %s, %s, %d\n", __FILE__, __PRETTY_FUNCTION__, __LINE__); \
            std::fflush(stderr); \
            cmd; \
        } \
    } while(0)

#endif // ERR_H
