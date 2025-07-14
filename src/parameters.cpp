#include "parameters.h"
#include "spark/constants/constants.h"

#include <cmath>

namespace spark {

void Parameters::fixed_parameters() {
    tg = 300.0; // neutral temperature (K)
    te = 30'000.0; // electron temperature (K)
    ti = 300.0; // ion temperature (K)
    m_he = 6.67e-27; // ion mass (kg)
    m_e = 9.109e-31; // electron mass (kg)
    lz = 6.7e-2;
    f = 13.56e6; // frequency (Hz)
}

void Parameters::computed_parameters() {
    dz = lz / static_cast<double>(nz - 1);
    dr = dz;
    lr = dr * static_cast<double>(nr - 1);
    double vth = std::sqrt(spark::constants::kb * te / spark::constants::m_e);
    double A = spark::constants::pi * lr * lr;
    particle_weight = (n0 * lr * lz) / static_cast<double>(ppc * (nz - 1) * (nr - 1));
    n_initial = (nz - 1) * (nr - 1) * ppc;
}

Parameters Parameters::case_1() {
    Parameters p{};
    p.fixed_parameters();

    p.nz = 129; // number of horizontal cells
    p.nr = 4; // number of vertical cells
    p.dt = 1.0 / (400.0 * p.f); // time step (s)
    p.ng = 9.64e20; // neutral density (m^-3)
    p.n0 = 2.56e14; // plasma density (m^-3)
    p.volt = 450.0; // voltage (V)
    p.ppc = 512; // particles per cell (dimensionless)
    p.n_steps = 512'000; // steps to execute (dimensionless)
    p.n_steps_avg = 12'800; // steps to average (dimensionless)

    p.computed_parameters();
    return p;
}

Parameters Parameters::case_2() {
    Parameters p{};
    p.fixed_parameters();

    p.nr = 257;
    p.nz = 4;
    p.dt = 1.0 / (800.0 * p.f);
    p.ng = 32.1e20;
    p.n0 = 5.12e14;
    p.volt = 200.0;
    p.ppc = 256;
    p.n_steps = 4'096'000;
    p.n_steps_avg = 25'600;

    p.computed_parameters();
    return p;
}

Parameters Parameters::case_3() {
    Parameters p{};
    p.fixed_parameters();

    p.nr = 513;
    p.nz = 4;
    p.dt = 1.0 / (1600.0 * p.f);
    p.ng = 96.4e20;
    p.n0 = 5.12e14;
    p.volt = 150.0;
    p.ppc = 128;
    p.n_steps = 8'192'000;
    p.n_steps_avg = 51'200;

    p.computed_parameters();
    return p;
}

Parameters Parameters::case_4() {
    Parameters p{};
    p.fixed_parameters();

    p.nr = 513;
    p.nz = 4;
    p.dt = 1.0 / (3200.0 * p.f);
    p.ng = 321.0e20;
    p.n0 = 3.84e14;
    p.volt = 120.0;
    p.ppc = 64;
    p.n_steps = 49'152'000;
    p.n_steps_avg = 102'400;

    p.computed_parameters();
    return p;
}
}  // namespace spark
