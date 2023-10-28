#include <iostream>
#include <sstream>


#include <simd.h>
#include <Silicon_amx.h>

using namespace ASC_HPC;
using std::cout, std::endl;


int main()
{
  SiliconAMX amx;

  SIMD<double,8> x0(1.,2.,3.,4.,5.,6.,7.,8.);
  SIMD<double,8> a;
  
  amx.LoadX(0, x0);
  amx.StoreX(0, a);

  cout << "a = " << a << endl << endl;


  SIMD<double,8> matX[8];
  SIMD<double,8> matY[8];  
  for (int i = 0; i < 8; i++)
    {
      std::array<double,8> sa;
      for (int j = 0; j < 8; j++)
        sa[j] = 10*i+j;
      matX[i] = SIMD<double,8> (sa);
      matY[i] = 2*SIMD<double,8> (sa);      
    }

  for (int i = 0; i < 8; i++)
    amx.LoadX(i, matX[i]);
  for (int i = 0; i < 8; i++)
    amx.LoadY(i, matY[i]);
  
  for (int i = 0; i < 8; i++)
    cout << "x" << i << " = " << amx.X(i) << endl;
  cout << endl;  
  for (int i = 0; i < 8; i++)
    cout << "y" << i << " = " << amx.Y(i) << endl;
  cout << endl;
  
  cout << "outer(" << matX[1] << "; " << matY[2] << ") = " << endl;
  amx.OuterProduct(1,2,0);
  for (int i = 0; i < 64; i+=8)
    cout << "z" << i << " = " << amx.Z(i) << endl;
}
