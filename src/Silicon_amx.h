#ifndef SILICON_AMX_H
#define SILICON_AMX_H

#include <aarch64.h>

#define PTR_ROW_FLAGS(ptr, row, flags) (((uint64_t)&*(ptr)) + (((uint64_t)((row) + (flags) * 64)) << 56))



namespace ASC_HPC
{

  class SiliconAMX
  {
  public:
    SiliconAMX()
      {
        AMX_SET();
      }

    ~SiliconAMX()
      {
        AMX_CLR();
      }

    // close to AMX - functions:
    void LoadX (int nr, SIMD<double,8> val)
    {
      AMX_LDX(PTR_ROW_FLAGS(&val, nr, 0));
    }
  
    void StoreX (int nr, SIMD<double,8> & val)
    {
      AMX_STX(PTR_ROW_FLAGS(&val, nr, 0));
    }

    void LoadY (int nr, SIMD<double,8> val)
    {
      AMX_LDY(PTR_ROW_FLAGS(&val, nr, 0));
    }
  
    void StoreY (int nr, SIMD<double,8> & val)
    {
      AMX_STY(PTR_ROW_FLAGS(&val, nr, 0));
    }
    void LoadZ (int nr, SIMD<double,8> val)
    {
      AMX_LDX(PTR_ROW_FLAGS(&val, nr, 0));
    }
  
    void StoreZ (int nr, SIMD<double,8> & val)
    {
      AMX_STZ(PTR_ROW_FLAGS(&val, nr, 0));
    }

    // outer product of x[ix] and y[iy], added to (z[iz], z[iz+8], ...)
    void OuterProduct (int ix, int iy, int iz)
    {
      AMX_FMA64( (0ul<<63) + ( (iz)<<20) + ( (8*ix)<<13) + ( (8*iy)<<3)+0);
    }


    // convenience:
    SIMD<double,8> X(int nr)
      {
        SIMD<double,8> hv;
        StoreX(nr, hv);
        return hv;
      }

    SIMD<double,8> Y(int nr)
      {
        SIMD<double,8> hv;
        StoreY(nr, hv);
        return hv;
      }

    SIMD<double,8> Z(int nr)
      {
        SIMD<double,8> hv;
        StoreZ(nr, hv);
        return hv;
      }

  };

}


#endif
