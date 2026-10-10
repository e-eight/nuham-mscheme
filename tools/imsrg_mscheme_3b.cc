// Oracle driver: imsrg++'s own m-scheme three-body matrix element for a me3j file.
//
// Reads a Darmstadt me3j file the way imsrg++ does (ReadWrite::Read_Darmstadt_3body, "full" type),
// then evaluates UnitTest::GetMschemeMatrixElement_3b on each query read from stdin.
//
// usage: imsrg_mscheme_3b FILE.me3j[.gz] FILE_EMAX FILE_E2MAX FILE_E3MAX < queries > values
// The model space is the file's: emax = FILE_EMAX, e2max = FILE_E2MAX, e3max = FILE_E3MAX.
// stdin, one query per line: six states  n l j2 tz2 m2  (bra p q r, ket s t u), tz2 = -1 proton,
// +1 neutron. Output: lines "V value" (%.12e), mixed with imsrg++ log lines. The element is
// <pqr|W|stu> in imsrg++'s convention, with the states in the order given.
//
// Build (in the imsrg-dev image, see tools/run_oracle3.sh).
#include <cstdio>
#include <iostream>
#include <string>

#include "ModelSpace.hh"
#include "Operator.hh"
#include "ReadWrite.hh"
#include "UnitTest.hh"

int main(int argc, char** argv)
{
  if (argc != 5)
  {
    std::cerr << "usage: " << argv[0] << " FILE FILE_EMAX FILE_E2MAX FILE_E3MAX < queries\n";
    return 2;
  }
  std::string file = argv[1];
  int emax = std::stoi(argv[2]);
  int e2max = std::stoi(argv[3]);
  int e3max = std::stoi(argv[4]);

  ModelSpace ms(emax, "He4", "He4");
  ms.SetE2max(e2max);
  ms.SetE3max(e3max);
  ms.SetHbarOmega(16.0);
  Operator H(ms, 0, 0, 0, 3);
  H.SetHermitian();
  ReadWrite rw;
  rw.Read_Darmstadt_3body(file, H, emax, e2max, e3max);
  UnitTest ut(ms);

  int s[6][5];
  while (true)
  {
    bool ok = true;
    for (int i = 0; i < 6 && ok; ++i)
      for (int k = 0; k < 5; ++k)
        if (!(std::cin >> s[i][k]))
        {
          ok = false;
          break;
        }
    if (!ok)
      break;
    int idx[6];
    for (int i = 0; i < 6; ++i)
      idx[i] = ms.GetOrbitIndex(s[i][0], s[i][1], s[i][2], s[i][3]);
    double v = ut.GetMschemeMatrixElement_3b(H, idx[0], s[0][4], idx[1], s[1][4], idx[2], s[2][4],
                                             idx[3], s[3][4], idx[4], s[4][4], idx[5], s[5][4]);
    std::printf("V %.12e\n", v);
  }
  return 0;
}
