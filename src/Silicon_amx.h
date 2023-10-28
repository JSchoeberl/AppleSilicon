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

    void LoadX (int nr, const double * ptr)
    {
      AMX_LDX(PTR_ROW_FLAGS(ptr, nr, 0));
    }

    void StoreX (int nr, double * ptr)
    {
      AMX_STX(PTR_ROW_FLAGS(ptr, nr, 0));
    }

    void LoadY (int nr, double * ptr)
    {
      AMX_LDY(PTR_ROW_FLAGS(ptr, nr, 0));
    }

    void StoreY (int nr, double * ptr)
    {
      AMX_STY(PTR_ROW_FLAGS(ptr, nr, 0));
    }

    void LoadZ (int nr, double * ptr)
    {
      AMX_LDZ(PTR_ROW_FLAGS(ptr, nr, 0));
    }

    void StoreZ (int nr, double * ptr)
    {
      AMX_STZ(PTR_ROW_FLAGS(ptr, nr, 0));
    }

    
    // outer product of x[ix] and y[iy], added to (z[iz], z[iz+8], ...)
    void OuterProduct (int ix, int iy, int iz)
    {
      AMX_FMA64( (0ul<<63) + ( (iz)<<20) + ( (8*ix)<<13) + ( (8*iy)<<3)+0);
    }

    
    void ExtrY (uint64_t iz, uint64_t iy)
    {
      AMX_EXTRY( ((iz)<<20) + ((iy)<<6)+8 );
    }

    
    // convenience:
    SIMD<double,8> X(int nr)
      {
        SIMD<double,8> hv;
        StoreX(nr, hv.Ptr());
        return hv;
      }

    SIMD<double,8> Y(int nr)
      {
        SIMD<double,8> hv;
        StoreY(nr, hv.Ptr());
        return hv;
      }

    SIMD<double,8> Z(int nr)
      {
        SIMD<double,8> hv;
        StoreZ(nr, hv.Ptr());
        return hv;
      }
  };

}


#endif
