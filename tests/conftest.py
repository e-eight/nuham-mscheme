from pathlib import Path

import pytest

REF = Path(__file__).resolve().parents[2] / "nuhamil-alt" / "refs" / "work"
E4 = REF / "TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw16_emax4_e2max8.me2j.gz"
E6 = REF / "TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw16_emax6_e2max12.me2j.gz"


@pytest.fixture(scope="session")
def e4_file():
    if not E4.exists():
        pytest.skip(f"reference file missing: {E4}")
    return E4


@pytest.fixture(scope="session")
def e6_file():
    if not E6.exists():
        pytest.skip(f"reference file missing: {E6}")
    return E6
