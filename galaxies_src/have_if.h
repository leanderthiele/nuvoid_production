#ifndef HAVE_IF_H
#define HAVE_IF_H

struct UNUSED
{
    // define constructor for anything
    template<typename T>
    UNUSED (T) { };

    // default constructor required
    UNUSED () {};
};

// small convenience utility
// if enable is false, the type is empty struct so compiler will complain if we use it
// for anything
template<bool enable, typename T>
struct have_if { typedef UNUSED type; };

// and this is the specialization
template<typename T>
struct have_if<true, T> { typedef T type; }; 

// for convenience
#define HAVE_IF(condition, T) \
    [[maybe_unused]] typename have_if<condition, T>::type

#endif // HAVE_IF_H
