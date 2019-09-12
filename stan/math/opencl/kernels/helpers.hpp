#ifndef STAN_MATH_OPENCL_KERNELS_HELPERS_HPP
#define STAN_MATH_OPENCL_KERNELS_HELPERS_HPP
#ifdef STAN_OPENCL

#include <string>

namespace stan {
namespace math {
namespace opencl_kernels {

/*
 * Defines helper macros for common matrix indexing operations
 */
static const char* indexing_helpers =
    R"(
    // Matrix access helpers
  #ifndef A_batch
  #define A_batch(i,j,k) A[(k) * cols * rows + (j) * rows + (i)]
  #endif
  #ifndef A
  #define A(i,j) A[(j) * rows + (i)]
  #endif
  #ifndef B
  #define B(i,j) B[(j) * rows + (i)]
  #endif
  #ifndef C
  #define C(i,j) C[(j) * rows + (i)]
  #endif
    // Transpose
  #ifndef BT
  #define BT(i,j) B[(j) * cols + (i)]
  #endif
  #ifndef AT
  #define AT(i,j) A[(j) * cols + (i)]
  #endif
    // Moving between two buffers
  #ifndef src
  #define src(i,j) src[(j) * src_rows + (i)]
  #endif
  #ifndef dst
  #define dst(i,j) dst[(j) * dst_rows + (i)]
  #endif
    // math constants for the generalized logistic model and its lgamma calculations
  #define MATH_PI	3.14159265358979323846 
  #define MATH_LN_SQRT_PI	0.918938533204672741780329736406 
  #define MATH_LN_SQRT_PId2	0.225791352644727432363097614947	
  #define nalgm 5 
  #define xbig  94906265.62425156 
  #define ngam 22 
  #define xmin -170.5674972726612 
  #define xmax  171.61447887182298 
  #define xsml 2.2474362225598545e-308 
  #define dxrel 1.490116119384765696e-8 
  #define S_0 0.083333333333333333333
  #define S1 0.00277777777777777777778
  #define S2 0.00079365079365079365079365
  #define S3 0.000595238095238095238095238
  #define S4 0.0008417508417508417508417508
  #define NUM 256
  #define NUM2 16
  )";

/*
 * Defines a helper macro for kernels with 2D local size
 */
static const char* thread_block_helpers =
    R"(
  // The local memory column for each thread block
  #define THREAD_BLOCK_SIZE_COL THREAD_BLOCK_SIZE/WORK_PER_THREAD
        )";


static const char* lbeta_helpers = 
  R"(
    double stirlerr(double n);  
    double chebyshev_eval(double x, const double *a, const int n) 
    { 
      double b0, b1, b2, twox; 
      int i; 
      twox = x * 2; 
      b2 = b1 = 0; 
      b0 = 0; 
      for (i = 1; i <= n; i++) { 
        b2 = b1; 
        b1 = b0; 
        b0 = twox * b1 - b2 + a[n - i];
      } 
      return (b0 - b2) * 0.5; 
    } 
 
    double lgammacor(double x) 
    { 
    
      const double algmcs[15] = { 
        +.1666389480451863247205729650822e+0, 
        -.1384948176067563840732986059135e-4, 
        +.9810825646924729426157171547487e-8, 
        -.1809129475572494194263306266719e-10, 
        +.6221098041892605227126015543416e-13, 
        -.3399615005417721944303330599666e-15, 
        +.2683181998482698748957538846666e-17, 
        -.2868042435334643284144622399999e-19, 
        +.3962837061046434803679306666666e-21, 
        -.6831888753985766870111999999999e-23, 
        +.1429227355942498147573333333333e-24, 
        -.3547598158101070547199999999999e-26, 
        +.1025680058010470912000000000000e-27, 
        -.3401102254316748799999999999999e-29, 
        +.1276642195630062933333333333333e-30 
      }; 
 
      double tmp; 
    
      /* For IEEE double precision DBL_EPSILON = 2^-52 = 2.220446049250313e-16 : 
      *   xbig = 2 ^ 26.5 */ 
    
      if (x < xbig) { 
        tmp = 10 / x; 
        return chebyshev_eval(tmp * tmp * 2 - 1, algmcs, nalgm) / x; 
      } 
    
      return 1 / (x * 12); 
    } 
 
  double gammafn(double x) 
  { 
  
    const double gamcs[42] = { 
      +.8571195590989331421920062399942e-2, 
      +.4415381324841006757191315771652e-2, 
      +.5685043681599363378632664588789e-1, 
      -.4219835396418560501012500186624e-2, 
      +.1326808181212460220584006796352e-2, 
      -.1893024529798880432523947023886e-3, 
      +.3606925327441245256578082217225e-4, 
      -.6056761904460864218485548290365e-5, 
      +.1055829546302283344731823509093e-5, 
      -.1811967365542384048291855891166e-6, 
      +.3117724964715322277790254593169e-7, 
      -.5354219639019687140874081024347e-8, 
      +.9193275519859588946887786825940e-9, 
      -.1577941280288339761767423273953e-9, 
      +.2707980622934954543266540433089e-10, 
      -.4646818653825730144081661058933e-11, 
      +.7973350192007419656460767175359e-12, 
      -.1368078209830916025799499172309e-12, 
      +.2347319486563800657233471771688e-13, 
      -.4027432614949066932766570534699e-14, 
      +.6910051747372100912138336975257e-15, 
      -.1185584500221992907052387126192e-15, 
      +.2034148542496373955201026051932e-16, 
      -.3490054341717405849274012949108e-17, 
      +.5987993856485305567135051066026e-18, 
      -.1027378057872228074490069778431e-18, 
      +.1762702816060529824942759660748e-19, 
      -.3024320653735306260958772112042e-20, 
      +.5188914660218397839717833550506e-21, 
      -.8902770842456576692449251601066e-22, 
      +.1527474068493342602274596891306e-22, 
      -.2620731256187362900257328332799e-23, 
      +.4496464047830538670331046570666e-24, 
      -.7714712731336877911703901525333e-25, 
      +.1323635453126044036486572714666e-25, 
      -.2270999412942928816702313813333e-26, 
      +.3896418998003991449320816639999e-27, 
      -.6685198115125953327792127999999e-28, 
      +.1146998663140024384347613866666e-28, 
      -.1967938586345134677295103999999e-29, 
      +.3376448816585338090334890666666e-30, 
      -.5793070335782135784625493333333e-31 
    }; 
  
    int i, n; 
    double y; 
    double sinpiy, value; 
  
      
    y = fabs(x); 
      
    if (y <= 10) { 
      
	  n = (int)x; 
	  if (x < 0) --n; 
	  y = x - n; 
	  --n; 
	  value = chebyshev_eval(y * 2 - 1, gamcs, ngam) + .9375; 
	  if (n == 0) 
	  return value; 
	  
	  if (n < 0) { 
	  n = -n; 
	  for (i = 0; i < n; i++) { 
		  value /= (x + i); 
	  } 
	  return value; 
	  } 
	  else { 
	  
	  for (i = 1; i <= n; i++) { 
		  value *= (y + i); 
	  } 
	  return value; 
	  } 
  
	} else { 
          if (y <= 50 && y == (int)y) {  
          value = 1.; 
          for (i = 2; i < y; i++) value *= i; 
          } 
          else {  
          double tmp1; 
          if (2 * y == (int)(2 * y)) { 
              tmp1 = stirlerr(y); 
          } 
          else { 
              tmp1 = lgammacor(y); 
          } 
          value = exp((y - 0.5) * log(y) - y + MATH_LN_SQRT_PI + tmp1); 
          } 
          if (x > 0) 
          return value; 
      
          sinpiy = sin(y*MATH_PI); 
      
          return -MATH_PI / (y * sinpiy * value); 
    } 
      
  } 
      
  double lgammafn(double x) 
  { 
	  double ans, y, sinpiy; 
	  
	  y = fabs(x); 
	  
	  if (y < 1e-306) return -log(y); // denormalized range, R change 
	  if (y <= 10) return log(fabs(gammafn(x))); 
	  
	  if (x > 0) { /* i.e. y = x > 10 */ 
		  if (x > 1e17) 
		  return(x*(log(x) - 1.)); 
		  else if (x > 4934720.) 
		  return(MATH_LN_SQRT_PI + (x - 0.5) * log(x) - x); 
		  else 
		  return MATH_LN_SQRT_PId2 + (x - 0.5) * log(x) - x + lgammacor(x); 
	  } 
	  /* else: x < -10; y = -x */ 
	  sinpiy = fabs(sin(y*MATH_PI)); 
	  
	  ans = MATH_LN_SQRT_PId2 + (x - 0.5) * log(y) - x - log(sinpiy) - lgammacor(y); 
	  
	  return ans; 
  } 
      
  double stirlerr(double n) 
  { 
  
	  const double sferr_halves[31] = { 
		  0.0, /* n=0 - wrong, place holder only */ 
		  0.1534264097200273452913848,  /* 0.5 */ 
		  0.0810614667953272582196702,  /* 1.0 */ 
		  0.0548141210519176538961390,  /* 1.5 */ 
		  0.0413406959554092940938221,  /* 2.0 */ 
		  0.03316287351993628748511048, /* 2.5 */ 
		  0.02767792568499833914878929, /* 3.0 */ 
		  0.02374616365629749597132920, /* 3.5 */ 
		  0.02079067210376509311152277, /* 4.0 */ 
		  0.01848845053267318523077934, /* 4.5 */ 
		  0.01664469118982119216319487, /* 5.0 */ 
		  0.01513497322191737887351255, /* 5.5 */ 
		  0.01387612882307074799874573, /* 6.0 */ 
		  0.01281046524292022692424986, /* 6.5 */ 
		  0.01189670994589177009505572, /* 7.0 */ 
		  0.01110455975820691732662991, /* 7.5 */ 
		  0.010411265261972096497478567, /* 8.0 */ 
		  0.009799416126158803298389475, /* 8.5 */ 
		  0.009255462182712732917728637, /* 9.0 */ 
		  0.008768700134139385462952823, /* 9.5 */ 
		  0.008330563433362871256469318, /* 10.0 */ 
		  0.007934114564314020547248100, /* 10.5 */ 
		  0.007573675487951840794972024, /* 11.0 */ 
		  0.007244554301320383179543912, /* 11.5 */ 
		  0.006942840107209529865664152, /* 12.0 */ 
		  0.006665247032707682442354394, /* 12.5 */ 
		  0.006408994188004207068439631, /* 13.0 */ 
		  0.006171712263039457647532867, /* 13.5 */ 
		  0.005951370112758847735624416, /* 14.0 */ 
		  0.005746216513010115682023589, /* 14.5 */ 
		  0.005554733551962801371038690  /* 15.0 */ 
	  }; 
	  double nn; 
	  
	  if (n <= 15.0) { 
		  nn = n + n; 
		  if (nn == (int)nn) return(sferr_halves[(int)nn]); 
		  return(lgammafn(n + 1.) - (n + 0.5)*log(n) + n - MATH_LN_SQRT_PI); 
	  } 
	  
	  nn = n*n; 
	  if (n>500) return((S_0 - S1 / nn) / n); 
	  if (n> 80) return((S_0 - (S1 - S2 / nn) / nn) / n); 
	  if (n> 35) return((S_0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n); 
	  /* 15 < n <= 35 : */ 
	  return((S_0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n); 
  }

  double lbeta(double a, double b) 
  { 
    double corr, p, q; 
    
    p = a; 
    q = a; 
    
    if (b < p) p = b; 
    if (b > q) q = b; 
    
    if (p >= 10) { 
        corr = lgammacor(p) + lgammacor(q) - lgammacor(p + q); 
        return log(q) * -0.5 + MATH_LN_SQRT_PI + corr + (p - 0.5) * log(p / (p + q)) + q * log(1 - (p / (p + q)));//log1p(-p / (p + q)); 
    } else if (q >= 10) { 
        corr = lgammacor(q) - lgammacor(p + q); 
        return lgammafn(p) + corr + p - p * log(p + q) 
        + (q - 0.5) * log1p(-p / (p + q)); 
    } else {
        if (p < 1e-306) {
          return lgamma(p) + (lgamma(q) - lgamma(p + q)); 
        } else{
          double aa = gammafn(p) * (gammafn(q) / gammafn(p + q)); 
          return log(aa);
        }
    }
    return 0.0; 
  }

  double digamma(double x)
  {
    double c = 8.5;
    double euler_mascheroni = 0.57721566490153286060;
    double r;
    double value;
    double x2;
    if (x <= 0.0)
    {
        value = 0.0;
        return value;
    }
    if (x <= 0.000001)
    {
        value = -euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x;
        return value;
    }
    value = 0.0;
    x2 = x;
    while (x2 < c)
    {
        value = value - 1.0 / x2;
        x2 = x2 + 1.0;
    }
    r = 1.0 / x2;
    value = value + log(x2) - 0.5 * r;

    r = r * r;
    
    value = value
        - r * (1.0 / 12.0
        - r * (1.0 / 120.0
            - r * (1.0 / 252.0
            - r * (1.0 / 240.0
                - r * (1.0 / 132.0)))));

    return value;
  }
  
  double dbeta(const double x, const double a, const double b) {
      return log(x) * (a - 1) + log(1 - x) * (b - 1) - lbeta(a, b);
  }
  )";
}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan
#endif
#endif
