// Oracle driver: imsrg++'s own m-scheme two-body matrix element for a me2j file.
//
// Reads a Darmstadt me2j file the way imsrg++ does, then evaluates
// UnitTest::GetMschemeMatrixElement_2b on each tuple read from stdin.
//
// usage: imsrg_mscheme_2b FILE.me2j[.gz] FILE_EMAX FILE_E2MAX [MODE HW A] < tuples > values
// MODE = file (default): the file only. file+trel: the file plus imsrg_util::Trel_Op, as imsrg++
// does when it builds Hbare (needs HW and A). trel: Trel_Op only (the file is still read).
// stdin, one query per line: a body count (1 or 2), then that many x 2 single-particle states
//   n l j2 tz2 m2   (bra p, [q;] ket r, [s])
// tz2 = -1 proton, +1 neutron. Output: lines "V value" (%.12e), mixed with imsrg++ log lines; elements of unique pair states
// are returned as <pq|V|rs> in imsrg++'s convention (a_p a_q order of the bra).
//
// Build (in the imsrg-dev image, see tools/run_oracle.sh).
#include <cstdio>
#include <iostream>
#include <string>

#include "ModelSpace.hh"
#include "Operator.hh"
#include "ReadWrite.hh"
#include "imsrg_util.hh"
#include "UnitTest.hh"

int main(int argc, char** argv)
{
  if (argc != 4 && argc != 7)
  {
    std::cerr << "usage: " << argv[0] << " FILE FILE_EMAX FILE_E2MAX < tuples\n";
    return 2;
  }
  std::string file = argv[1];
  int emax = std::stoi(argv[2]);
  int e2max = std::stoi(argv[3]);
  std::string mode = argc == 7 ? argv[4] : "file";
  double hw = argc == 7 ? std::stod(argv[5]) : 16.0;
  int A = argc == 7 ? std::stoi(argv[6]) : 4;

  ModelSpace ms(emax, "He4", "He4");
  ms.SetE2max(e2max);
  ms.SetHbarOmega(hw);
  ms.SetTargetMass(A);
  Operator H(ms, 0, 0, 0, 2);
  H.SetHermitian();
  ReadWrite rw;
  rw.ReadBareTBME_Darmstadt(file, H, emax, e2max, emax);
  if (mode == "trel")
    H = imsrg_util::Trel_Op(ms);
  else if (mode == "file+trel")
    H += imsrg_util::Trel_Op(ms);
  UnitTest ut(ms);

  int nb;
  while (std::cin >> nb)
  {
    int s[4][5];
    bool ok = true;
    for (int i = 0; i < 2 * nb && ok; ++i)
      for (int k = 0; k < 5; ++k)
        if (!(std::cin >> s[i][k]))
        {
          ok = false;
          break;
        }
    if (!ok)
      break;
    int idx[4];
    for (int i = 0; i < 2 * nb; ++i)
      idx[i] = ms.GetOrbitIndex(s[i][0], s[i][1], s[i][2], s[i][3]);
    double v;
    if (nb == 1)
      v = ut.GetMschemeMatrixElement_1b(H, idx[0], s[0][4], idx[1], s[1][4]);
    else
      v = ut.GetMschemeMatrixElement_2b(H, idx[0], s[0][4], idx[1], s[1][4], idx[2], s[2][4],
                                        idx[3], s[3][4]);
    std::printf("V %.12e\n", v);
  }
  return 0;
}
