// Integer-only exact F64 triplet counting. No point values/order are modified.
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,
              "Requires native IEEE754 binary64");
namespace {
constexpr std::uint64_t kMaxN=2147483647ULL;
constexpr std::uint64_t kAbs=0x7fffffffffffffffULL;
constexpr std::uint64_t kExp=0x7ff0000000000000ULL;
struct Slot { std::uint64_t key[3]; std::uint8_t state; };
std::uint64_t bits(const unsigned char* p) {
 std::uint64_t value; std::memcpy(&value,p,8); return value;
}
std::uint64_t canonical(std::uint64_t value) {
 return (value&kAbs)==0 ? 0 : value;
}
std::uint64_t mix(std::uint64_t x) {
 x ^= x>>30; x *= 0xbf58476d1ce4e5b9ULL;
 x ^= x>>27; x *= 0x94d049bb133111ebULL;
 return x^(x>>31);
}
std::uint64_t hash(const std::uint64_t k[3],unsigned bits_used) {
 std::uint64_t h=mix(k[0]^0x9e3779b97f4a7c15ULL);
 h=mix(h^k[1]^0xd6e8feb86659fd93ULL);
 h=mix(h^k[2]^0xa0761d6478bd642fULL);
 return bits_used==64?h:bits_used==0?0:h&((1ULL<<bits_used)-1);
}
// 0 success; 1 unsupported shape/count/alignment; 2 allocation failure;
// 3 other exception; 4 subnormal; 5 nonfinite. Every nonzero invokes stock sort.
int count_impl(const void* input,std::uint64_t n,std::uint64_t bytes,
               std::uint64_t* result,unsigned hash_bits,unsigned fault) {
 if(!result)return 1;
 result[0]=result[1]=result[2]=0;
 if(n<4 || n>kMaxN || bytes!=n*24 || !input ||
    reinterpret_cast<std::uintptr_t>(input)%alignof(double) || hash_bits>64)return 1;
 const auto* p=static_cast<const unsigned char*>(input);
 // Classify by bits before allocation. No F64 compare/FP environment operation.
 for(std::uint64_t i=0;i<n*3;++i) {
  const auto a=bits(p+i*8)&kAbs;
  if((a&kExp)==kExp)return 5;
  if((a&kExp)==0 && a!=0)return 4;
 }
 try {
  if(fault==1)throw std::bad_alloc();
  if(fault==2)throw std::runtime_error("CPU qualification fault");
  std::size_t capacity=8;
  while(capacity<n*2) {
   if(capacity>std::numeric_limits<std::size_t>::max()/2)return 2;
   capacity*=2;
  }
  if(capacity>std::numeric_limits<std::size_t>::max()/sizeof(Slot))return 2;
  result[2]=capacity;
  // Vector only allocates/zero-initializes integer slots. No floating load factor.
  std::vector<Slot> table(capacity);
  const auto mask=capacity-1;
  for(std::uint64_t row=0;row<n;++row) {
   const std::uint64_t k[3]={canonical(bits(p+row*24)),
     canonical(bits(p+row*24+8)),canonical(bits(p+row*24+16))};
   auto bucket=static_cast<std::size_t>(hash(k,hash_bits))&mask;
   bool done=false;
   for(std::size_t probe=0;probe<capacity;++probe,bucket=(bucket+1)&mask) {
    auto& slot=table[bucket];
    if(slot.state==0) {
     slot.key[0]=k[0];slot.key[1]=k[1];slot.key[2]=k[2];slot.state=1;
     ++result[0];done=true;break;
    }
    if(slot.key[0]==k[0] && slot.key[1]==k[1] && slot.key[2]==k[2]) {
     if(slot.state==1){slot.state=2;++result[1];}
     done=true;break;
    }
   }
   if(!done)return 3;
  }
  return 0;
 } catch(const std::bad_alloc&) { return 2; }
 catch(...) { return 3; }
}
}
extern "C" const char* exact_triplet_counts_abi() {
 return "E164_EXACT_NATIVE_F64_TRIPLET_COUNTS_V1";
}
extern "C" std::uint64_t exact_triplet_slot_bytes() {return sizeof(Slot);}
extern "C" int exact_triplet_counts(const void* p,std::uint64_t n,
                                   std::uint64_t bytes,std::uint64_t* out) {
 return count_impl(p,n,bytes,out,64,0);
}
// Bounded fixture-only entry, never called by the candidate validator.
extern "C" int exact_triplet_counts_fixture(const void* p,std::uint64_t n,
 std::uint64_t bytes,std::uint64_t* out,unsigned hash_bits,unsigned fault) {
 if(n>1024 || fault>2)return 1;
 return count_impl(p,n,bytes,out,hash_bits,fault);
}
// Read-only architectural status/control snapshot, no setters and no FP math.
extern "C" int exact_triplet_fp_snapshot(std::uint64_t* out) {
 if(!out)return 1;
#if defined(__aarch64__)
 asm volatile("mrs %0, fpcr":"=r"(out[0]));
 asm volatile("mrs %0, fpsr":"=r"(out[1]));out[2]=0;return 0;
#elif defined(__x86_64__)
 std::uint32_t csr;std::uint16_t cw,sw;
 asm volatile("stmxcsr %0":"=m"(csr));asm volatile("fnstcw %0":"=m"(cw));
 asm volatile("fnstsw %0":"=m"(sw));out[0]=csr;out[1]=cw;out[2]=sw;return 0;
#else
 return 1;
#endif
}
