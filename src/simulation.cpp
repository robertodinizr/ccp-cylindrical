#include "simulation.h"

#include <spark/collisions/mcc.h>
#include <spark/constants/constants.h>
#include <spark/core/matrix.h>
#include <spark/em/electric_field.h>
#include <spark/em/poisson.h>
#include <spark/interpolate/field.h>
#include <spark/interpolate/weight.h>
#include <spark/particle/boundary.h>
#include <spark/particle/pusher.h>
#include <spark/random/random.h>
#include <spark/spatial/grid.h>

#include "reactions.h"

#include <fstream>
#include <filesystem>


namespace spark {

Simulation::Simulation(const Parameters& parameters, const std::string& data_path)
    : parameters_(parameters), data_path_(data_path), state_(StateInterface(*this)) {}

void Simulation::run() {
    set_initial_conditions();

    auto electron_collisions = load_electron_collisions();
    auto ion_collisions = load_ion_collisions();

    em::CylindricalPoissonSolver2D::DomainProp domain_prop;
    domain_prop.extents = {static_cast<int>(parameters_.nz), static_cast<int>(parameters_.nr)};
    domain_prop.dx = {parameters_.dz, parameters_.dr};

    events().notify(Event::Start, state_);

    std::vector<em::CylindricalPoissonSolver2D::Region> regions;

    regions.push_back(em::CylindricalPoissonSolver2D::Region{
        em::CellType::BoundaryNeumann,
        {0, 1},
        {0, static_cast<int>(parameters_.nr - 2)},
        []() { return 0.0; }
    });
    regions.push_back(em::CylindricalPoissonSolver2D::Region{
        em::CellType::BoundaryNeumann,
        {static_cast<int>(parameters_.nz - 1), 1},
        {static_cast<int>(parameters_.nz - 1), static_cast<int>(parameters_.nr - 2)},
        []() { return 0.0; }
    });

    regions.push_back(em::CylindricalPoissonSolver2D::Region{
        em::CellType::BoundaryDirichlet,
        {0, 0},
        {static_cast<int>(parameters_.nz - 1), 0},
        []() { return 0.0; }
    });

    double boundary_voltage = 0.0;
    regions.push_back(em::CylindricalPoissonSolver2D::Region{
        em::CellType::BoundaryDirichlet,
        {0, static_cast<int>(parameters_.nr - 1)},
        {static_cast<int>(parameters_.nz - 1), static_cast<int>(parameters_.nr - 1)},
        [&boundary_voltage]() { return boundary_voltage; }
    });

    auto poisson_solver = em::CylindricalPoissonSolver2D(domain_prop, regions);

    core::TMatrix<core::Vec<3>, 1> zero_magnetic_field_e;
    core::TMatrix<core::Vec<3>, 1> zero_magnetic_field_i;

    for (step = 0; step < parameters_.n_steps; ++step) {
        boundary_voltage = parameters_.volt * std::sin(2.0 * spark::constants::pi * parameters_.f * parameters_.dt * static_cast<double>(step));

        spark::interpolate::weight_to_grid_cylindrical(electrons_, electron_density_);
        spark::interpolate::weight_to_grid_cylindrical(ions_, ion_density_);

        reduce_rho();
        poisson_solver.solve(phi_field_.data(), rho_field_.data());

        spark::em::electric_field_cylindrical(phi_field_, electric_field_.data());

        spark::interpolate::field_at_particles_cylindrical(electric_field_, electrons_, electron_field);
        spark::interpolate::field_at_particles_cylindrical(electric_field_, ions_, ion_field);


        core::TMatrix<core::Vec<3>, 1> zero_magnetic_field_e;
        core::TMatrix<core::Vec<3>, 1> zero_magnetic_field_i;

        zero_magnetic_field_e.resize({electrons_.n()});
        zero_magnetic_field_e.fill({0.0, 0.0, 0.0});

        zero_magnetic_field_i.resize({ions_.n()});
        zero_magnetic_field_i.fill({0.0, 0.0, 0.0});

        core::TMatrix<core::Vec<3>, 1> electron_field_3d;
        electron_field_3d.resize({electrons_.n()});
        for (size_t i = 0; i < electrons_.n(); i++) {
            electron_field_3d[i] = {electron_field[i].x, electron_field[i].y, 0.0};
        }

        core::TMatrix<core::Vec<3>, 1> ion_field_3d;
        ion_field_3d.resize({ions_.n()});
        for (size_t i = 0; i < ions_.n(); i++) {
            ion_field_3d[i] = {ion_field[i].x, ion_field[i].y, 0.0};
        }

        spark::particle::boris_mover_cylindrical(electrons_, electron_field_3d, zero_magnetic_field_e, parameters_.dt);
        spark::particle::boris_mover_cylindrical(ions_, ion_field_3d, zero_magnetic_field_i, parameters_.dt);

        tiled_boundary_.apply(electrons_);
        tiled_boundary_.apply(ions_);

        electron_collisions.react_all();
        ion_collisions.react_all();

        events().notify(Event::Step, state_);
    }
    events().notify(Event::End, state_);
}

void Simulation::reduce_rho() {
    auto& rho = rho_field_.data();
    const auto& ne = electron_density_.data();
    const auto& ni = ion_density_.data();

    const auto& n = rho_field_.n();
    const double dz = parameters_.dz;
    const double dr = parameters_.dr;
    const double k = spark::constants::e * parameters_.particle_weight;

    for (size_t i = 0; i < n.x; ++i) {
        for (size_t j = 0; j < n.y; ++j) {
            const size_t idx = rho_field_.data().index(i, j);
            double cell_volume;

            if (j == 0) {
                cell_volume = spark::constants::pi * (dr * dr / 4.0) * dz;
            } else {
                const double r_mid = (static_cast<double>(j) + 0.5) * dr;
                cell_volume = 2.0 * spark::constants::pi * r_mid * dr * dz;
            }

            if (cell_volume > 0.0) {
                rho(i, j) = k * (ni[idx] - ne[idx]) / cell_volume;
            } else {
                rho(i, j) = 0.0;
            }
        }
    }
}

Events<Simulation::Event, Simulation::EventAction>& Simulation::events() {
    return events_;
}

void Simulation::set_initial_conditions() {
        auto emitter = [this](double t, double m) {
        return [t, m, this](spark::core::Vec<3>& v, spark::core::Vec<2>& x) {

            double z_min = 0.0;
            double z_max = parameters_.lz;

            double r_min = parameters_.dr;
            double r_max = parameters_.lr - parameters_.dr;

            x.x = z_min + (z_max - z_min) * spark::random::uniform();

            double r_rand_sq = spark::random::uniform();
            x.y = std::sqrt(r_min * r_min * (1.0 - r_rand_sq) + r_max * r_max * r_rand_sq);

            double vth = std::sqrt(spark::constants::kb * t / m);
            v = {spark::random::normal(0.0, vth), spark::random::normal(0.0, vth),
                 spark::random::normal(0.0, vth)};
        };
    };

    electrons_ = spark::particle::ChargedSpecies<2, 3>(-spark::constants::e, spark::constants::m_e);
    electrons_.add(parameters_.n_initial, emitter(parameters_.te, spark::constants::m_e));

    ions_ = spark::particle::ChargedSpecies<2, 3>(spark::constants::e, parameters_.m_he);
    ions_.add(parameters_.n_initial, emitter(parameters_.ti, parameters_.m_he));

    electron_density_ = spark::spatial::UniformGrid<2>({parameters_.lz, parameters_.lr},
                                                      {parameters_.nz, parameters_.nr});
    ion_density_ = spark::spatial::UniformGrid<2>({parameters_.lz, parameters_.lr},
                                                 {parameters_.nz, parameters_.nr});
    rho_field_ = spark::spatial::UniformGrid<2>({parameters_.lz, parameters_.lr},
                                               {parameters_.nz, parameters_.nr});
    phi_field_ = spark::spatial::UniformGrid<2>({parameters_.lz, parameters_.lr},
                                               {parameters_.nz, parameters_.nr});

    electric_field_ = spark::spatial::TUniformGrid<core::TVec<double, 2>, 2>(
        {parameters_.lz, parameters_.lr}, {parameters_.nz, parameters_.nr});

    std::vector<spark::particle::TiledBoundary> boundaries = {
    // Fronteiras em Z (axial) -> REFLETORAS
    {{0, 0}, {0, static_cast<int>(parameters_.nr - 1)}, spark::particle::BoundaryType::Specular},
    {{static_cast<int>(parameters_.nz - 1), 0}, {static_cast<int>(parameters_.nz - 1), static_cast<int>(parameters_.nr - 1)}, spark::particle::BoundaryType::Specular},

    // Fronteiras em R (eletrodos) -> ABSORVENTES
    {{0, static_cast<int>(parameters_.nr - 1)}, {static_cast<int>(parameters_.nz - 1), static_cast<int>(parameters_.nr - 1)}, spark::particle::BoundaryType::Absorbing},
    {{0, 0}, {static_cast<int>(parameters_.nz - 1), 0}, spark::particle::BoundaryType::Absorbing}
    };
    tiled_boundary_ = spark::particle::TiledBoundary2D(electric_field_.prop(), boundaries, parameters_.dt);
}

spark::collisions::MCCReactionSet<2, 3> Simulation::load_electron_collisions() {
    auto electron_reactions = reactions::load_electron_reactions(data_path_, parameters_, ions_);

    spark::collisions::ReactionConfig<2, 3> electron_reaction_config{
        .dt = parameters_.dt,
        .target = std::make_shared<spark::collisions::StaticUniformTarget<2, 3>>(parameters_.ng, parameters_.tg),
        .reactions = std::move(electron_reactions),
        .dyn = spark::collisions::RelativeDynamics::FastProjectile
    };

    return spark::collisions::MCCReactionSet(&electrons_, std::move(electron_reaction_config));
}

spark::collisions::MCCReactionSet<2, 3> Simulation::load_ion_collisions() {
    auto ion_reactions = reactions::load_ion_reactions(data_path_, parameters_);

    spark::collisions::ReactionConfig<2, 3> ion_reaction_config{
        .dt = parameters_.dt,
        .target = std::make_shared<spark::collisions::StaticUniformTarget<2, 3>>(parameters_.ng, parameters_.tg),
        .reactions = std::move(ion_reactions),
        .dyn = spark::collisions::RelativeDynamics::SlowProjectile
    };

    return spark::collisions::MCCReactionSet(&ions_, std::move(ion_reaction_config));
}
}
