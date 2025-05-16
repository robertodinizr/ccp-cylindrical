#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cstddef>

namespace spark {

struct Parameters {
    size_t nx;
    size_t ny;
    double f;
    double dt;
    double lx;
    double ly;
    double dx;
    double dy;
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