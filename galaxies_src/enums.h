#ifndef ENUMS_H
#define ENUMS_H

// NOTE that the end_ element is required everywhere

// order is fixed here (used to index)!
enum class HaloDef { c, v, m,
                     end_ };

enum class Cat { Rockstar, FOF, RFOF, OLDFOF, 
                 end_ };

// secondary for assembly bias
enum class Sec { None, Conc/*concentration*/, TU/*kinetic to potential*/,
                 end_ };

enum class MAS { NGP, CIC, TSC, PCS,
                 end_ };

enum class VelMode { None, Unbiased, Biased,
                     end_ };

// order is fixed here (used to index)!
enum class RSD { x, y, z, None,
                 end_ };

// file format
enum class ftype { txt, bin, gad2,
                   end_ };


#endif // ENUMS_H
