#include <stdio.h>
#include <windows.h>

/* print the compiler version number formatted for AMPL,
 * e.g. for version 15.00.21022.08 for 80x86 we have
 * major = 15
 * minor = 00
 * rev = 21022
 * bld = 8
 */
int main (int argc, char **argv)
{

#if defined(_MSC_BUILD)
  int major, minor, rev, bld;
  int full, tmp;

  full = _MSC_FULL_VER;
  bld = _MSC_BUILD;
  tmp = full;
  rev = tmp % 100000;
  tmp = tmp / 100000;
  minor = tmp % 100;
  tmp = tmp / 100;
  major = tmp;
# if defined(_WIN64)
  fprintf (stdout, "char sysdetails_ASL[] = \""
           "mystery 64-bit C %02d.%02d.%05d.%02d\";\n",
           major, minor, rev, bld);
# else
  fprintf (stdout, "char sysdetails_ASL[] = \""
           "MS 32-bit C %02d.%02d.%05d.%02d\";\n",
           major, minor, rev, bld);
# endif
#elif defined(__INTEL_COMPILER)
  int iclVer, tmp;
  int major, minor, bldDate;

  iclVer = __INTEL_COMPILER;
  tmp = iclVer;
  tmp /= 10;                    /* throw away 1's digit */
  minor = tmp % 10;             /* 10's digit is minor version number  */
  tmp /= 10;
  major = tmp;
  tmp /= 100;                   /* next two are major version number */
  bldDate = __INTEL_COMPILER_BUILD_DATE;
  fprintf (stdout, "char sysdetails_ASL[] = \""
           "Intel 64-bit C %d.%d Build %d\";\n", major, minor, bldDate);
#else
  fprintf (stdout, "char sysdetails_ASL[] = \""
           "Mystery C\";\n");
#endif
} /* main */
