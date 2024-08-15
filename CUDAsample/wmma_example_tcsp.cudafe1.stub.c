#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "wmma_example_tcsp.fatbin.c"
extern void __device_stub__Z12wmma_exampleP6__halfS0_Pfiiiff(half *, half *, float *, int, int, int, float, float);
extern void __device_stub__Z17convertFp32ToFp16P6__halfPfi(half *, float *, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z12wmma_exampleP6__halfS0_Pfiiiff(half *__par0, half *__par1, float *__par2, int __par3, int __par4, int __par5, float __par6, float __par7){__cudaLaunchPrologue(8);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 28UL);__cudaSetupArgSimple(__par5, 32UL);__cudaSetupArgSimple(__par6, 36UL);__cudaSetupArgSimple(__par7, 40UL);__cudaLaunch(((char *)((void ( *)(half *, half *, float *, int, int, int, float, float))wmma_example)));}
# 93 "wmma_example_tcsp.cu"
void wmma_example( half *__cuda_0,half *__cuda_1,float *__cuda_2,int __cuda_3,int __cuda_4,int __cuda_5,float __cuda_6,float __cuda_7)
# 93 "wmma_example_tcsp.cu"
{__device_stub__Z12wmma_exampleP6__halfS0_Pfiiiff( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 157 "wmma_example_tcsp.cu"
}
# 1 "wmma_example_tcsp.cudafe1.stub.c"
void __device_stub__Z17convertFp32ToFp16P6__halfPfi( half *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(half *, float *, int))convertFp32ToFp16))); }
# 166 "wmma_example_tcsp.cu"
void convertFp32ToFp16( half *__cuda_0,float *__cuda_1,int __cuda_2)
# 166 "wmma_example_tcsp.cu"
{__device_stub__Z17convertFp32ToFp16P6__halfPfi( __cuda_0,__cuda_1,__cuda_2);




}
# 1 "wmma_example_tcsp.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T44) {  __nv_dummy_param_ref(__T44); __nv_save_fatbinhandle_for_managed_rt(__T44); __cudaRegisterEntry(__T44, ((void ( *)(half *, float *, int))convertFp32ToFp16), _Z17convertFp32ToFp16P6__halfPfi, (-1)); __cudaRegisterEntry(__T44, ((void ( *)(half *, half *, float *, int, int, int, float, float))wmma_example), _Z12wmma_exampleP6__halfS0_Pfiiiff, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
