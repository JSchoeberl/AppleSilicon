#include <iostream>
#include <sstream>
#include <chrono>



#include <simd.h>
#include <Silicon_amx.h>

using namespace ASC_HPC;
using std::cout, std::endl;


/*
  compute Z = X^T*Y
  X ... n*8 ... row major
  Y ... n*8 ... row major
  Z ... 8x8 ... row major
*/
 
void MultMatMat8x8 (size_t n, double * px, double * py, double * pz)
{
  SiliconAMX amx;
  /* 
  for (int i = 0; i < 8; i++)
    amx.LoadZ(8*i, SIMD<double,8>{0.0});
  */

  for (size_t i = 0; i < n; i++)
    {
      size_t i8 = 0; //i%8;
      amx.LoadX(i8, px+8*i);
      amx.LoadY(i8, py+8*i);
      amx.OuterProduct(i8, i8, 0);
    }

  for (int i = 0; i < 8; i++)
    amx.StoreZ(8*i, pz+8*i);
}



/*
  compute Z = X^T*Y
  X ... n*16 ... row major
  Y ... n*16 ... row major
  Z ... 16x16 ... row major
*/

void MultMatMat16x16 (size_t n, double * px, double * py, double * pz)
{
  SiliconAMX amx;

  for (size_t i = 0; i < n; i++)
    {
      amx.LoadX(0, px+16*i);
      amx.LoadX(1, px+16*i+8);
      amx.LoadY(0, py+16*i);
      amx.LoadY(1, py+16*i+8);
      amx.OuterProduct(0, 0, 0);
      amx.OuterProduct(1, 0, 1);
      amx.OuterProduct(0, 1, 2);
      amx.OuterProduct(1, 1, 3);
    }

  for (int i = 0; i < 8; i++)
    {
      amx.StoreZ(8*i, pz+16*i);
      amx.StoreZ(8*i+1, pz+16*i+8);
      amx.StoreZ(8*i+2, pz+16*(i+8));      
      amx.StoreZ(8*i+3, pz+16*(i+8)+8);      
    }
}




int main()
{
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


  // Matrix-matrix multiply:
  {
    // SiliconAMX amx;
    
    for (size_t n = 10; n < 40000; n *= 2)
      {
        cout << "n = " << n << endl;
        double * px = new double[8*n];
        double * py = new double[8*n];
        double * pz = new double[8*8];

        for (size_t i = 0; i < 8*n; i++)
          {
            px[i] = double (rand()) / RAND_MAX;
            py[i] = double (rand()) / RAND_MAX;
          }


        
        MultMatMat8x8 (n, px, py, pz);

        auto start = std::chrono::high_resolution_clock::now();
        size_t flops = 64*n;
        size_t runs = size_t (1e8 / flops) + 1;
        for (size_t i = 0; i < runs; i++)
          MultMatMat8x8 (n, px, py, pz);
      
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::milli>(end-start).count();
        
        cout << "n = " << n << "time = " << time 
             << " ms, GFlops = " << (flops*runs)/time/1e6
             << endl;
        
        for (int i = 0; i < 8; i++)
          cout << "Z" << i << " = " << SIMD<double,8>(pz+8*i) << endl;;
        cout << endl;
        
        for (int i = 0; i < 8; i++)
          for (int j = 0; j < 8; j++)
            {
              double sum = 0;
              for (int k = 0; k < n; k++)
                sum += px[8*k+j]*py[8*k+i];
              pz[8*i+j] = sum;
            }
        for (int i = 0; i < 8; i++)
          cout << "Z" << i << " = " << SIMD<double,8>(pz+8*i) << endl;
        
        delete [] py;
        delete [] px;
      }
  }




  // Matrix-matrix multiply 16x16:
  {
    for (size_t n = 4; n < 40000; n *= 2)
      {
        cout << "n = " << n << endl;
        double * px = new double[16*n];
        double * py = new double[16*n];
        double * pz = new double[16*16];

        for (size_t i = 0; i < 16*n; i++)
          {
            px[i] = double (rand()) / RAND_MAX;
            py[i] = double (rand()) / RAND_MAX;
          }
        for (size_t i = 0; i < 16*16; i++)
          pz[i] = 0;

        
        MultMatMat16x16 (n, px, py, pz);

        auto start = std::chrono::high_resolution_clock::now();
        size_t flops = 16*16*n;
        size_t runs = size_t (1e8 / flops) + 1;
        for (size_t i = 0; i < runs; i++)
          MultMatMat16x16 (n, px, py, pz);
      
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::milli>(end-start).count();
        
        cout << "n = " << n << "time = " << time 
             << " ms, GFlops = " << (flops*runs)/time/1e6
             << endl;
        
        for (int i = 0; i < 16; i++)
          cout << "Z" << i << " = " << SIMD<double,16>(pz+16*i) << endl;;
        cout << endl;
        
        for (int i = 0; i < 16; i++)
          for (int j = 0; j < 16; j++)
            {
              double sum = 0;
              for (int k = 0; k < n; k++)
                sum += px[16*k+j]*py[16*k+i];
              pz[16*i+j] = sum;
            }
        for (int i = 0; i < 16; i++)
          cout << "Z" << i << " = " << SIMD<double,16>(pz+16*i) << endl;
        
        delete [] py;
        delete [] px;
      }
  }


  
}
