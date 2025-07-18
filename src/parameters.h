#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cstddef>

namespace spark {

struct Parameters {
    size_t nz;
    size_t nr;
    double f;
    double dt;
    double lz;
    double lr;
    double dz;
    double dr;
    double ng;
    double tg;
    double te;
    double ti;
    double n0;
    double m_he;
    double m_e;
    double volt;
    size_t ppc;
    size_t n_steps;
    size_t n_steps_avg;
    double particle_weight;
    size_t n_initial;
    double r_min_factor = 4.0;
    static Parameters case_1();
    static Parameters case_2();
    static Parameters case_3();
    static Parameters case_4();

private:
    void fixed_parameters();
    void computed_parameters();
};

}  // namespace spark
#endif  // PARAMETERS_H
