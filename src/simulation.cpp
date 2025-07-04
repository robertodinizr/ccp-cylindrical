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
#include <iostream>

namespace spark {

Simulation::Simulation(const Parameters& parameters, const std::string& data_path)
    : parameters_(parameters), data_path_(data_path), state_(StateInterface(*this)) {}

void Simulation::run() {
    set_initial_conditions();

    std::cout << "Grid sizes:\n"
          << "  phi: " << phi_field_.n().x << "x" << phi_field_.n().y << "\n"
          << "  rho: " << rho_field_.n().x << "x" << rho_field_.n().y << "\n"
          << "  electric: " << electric_field_.n().x << "x" << electric_field_.n().y << "\n"
          << "  electrons: " << electrons_.n() << "\n"
          << "  ions: " << ions_.n() << "\n";

    auto electron_collisions = load_electron_collisions();
    auto ion_collisions = load_ion_collisions();

    em::CylindricalPoissonSolver2D::DomainProp domain_prop;
    domain_prop.extents = {static_cast<int>(parameters_.nz), static_cast<int>(parameters_.nr)};
    domain_prop.dx = {parameters_.dz, parameters_.dr};

    events().notify(Event::Start, state_);

    std::vector<em::CylindricalPoissonSolver2D::Region> regions;

    regions.push_back(em::CylindricalPoissonSolver2D::Region{
        em::CellType::BoundaryDirichlet,
        {0, 0},
        {0, static_cast<int>(parameters_.nr - 1)},
        []() { return 0.0; }
    });
    double boundary_voltage = 0.0;
    regions.push_back(em::CylindricalPoissonSolver2D::Region{
        em::CellType::BoundaryDirichlet,
        {static_cast<int>(parameters_.nz - 1), 0},
        {static_cast<int>(parameters_.nz - 1), static_cast<int>(parameters_.nr - 1)},
        [&boundary_voltage]() { return boundary_voltage; }
    });

    regions.push_back(em::CylindricalPoissonSolver2D::Region{
        em::CellType::BoundaryNeumann,
        {1, static_cast<int>(parameters_.nr - 1)},
        {static_cast<int>(parameters_.nz - 2), static_cast<int>(parameters_.nr - 1)},
        []() { return 0.0; }
    });

    regions.push_back(em::CylindricalPoissonSolver2D::Region{
        em::CellType::BoundaryNeumann,
        {1, 0},
        {static_cast<int>(parameters_.nz - 2), 0},
        []() { return 0.0; }
    });

    auto poisson_solver = em::CylindricalPoissonSolver2D(domain_prop, regions);

    for (step = 0; step < parameters_.n_steps; ++step) {
        boundary_voltage = parameters_.volt * std::sin(2.0 * spark::constants::pi * parameters_.f * parameters_.dt * static_cast<double>(step));

        spark::interpolate::weight_to_grid_cylindrical(electrons_, electron_density_);
        spark::interpolate::weight_to_grid_cylindrical(ions_, ion_density_);

        reduce_rho();

        poisson_solver.solve(phi_field_.data(), rho_field_.data());

        spark::em::electric_field_cylindrical(phi_field_, electric_field_.data());
        

        electron_field.resize({electrons_.n()});
        ion_field.resize({ions_.n()});

        spark::interpolate::field_at_particles_cylindrical(electric_field_, electrons_, electron_field);
        spark::interpolate::field_at_particles_cylindrical(electric_field_, ions_, ion_field);

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

        spark::particle::boris_mover_cylindrical(electrons_, electron_field_3d, parameters_.dt);
        spark::particle::boris_mover_cylindrical(ions_, ion_field_3d, parameters_.dt);

        tiled_boundary_.apply(electrons_);
        tiled_boundary_.apply(ions_);

        electron_collisions.react_all();
        ion_collisions.react_all();

        if (step % 1000 == 0) {
            debug_field_stats();
            debug_particle_counts();
        }

        events().notify(Event::Step, state_);
    }
    events().notify(Event::End, state_);
}

void Simulation::reduce_rho() {
    auto& rho = rho_field_.data();
    const auto& ne = electron_density_.data();
    const auto& ni = ion_density_.data();

    const auto n = rho_field_.n();
    const double dz = parameters_.dz;
    const double dr = parameters_.dr;
    const double k = spark::constants::e * parameters_.particle_weight;

    for (size_t i = 0; i < n.x; ++i) {
        for (size_t j = 0; j < n.y; ++j) {
            
            double r = (j + 0.5) * dr;
	    double cell_volume = 2.0 * spark::constants::pi * r * dr * dz;

	    if (j == 0) {
                cell_volume = spark::constants::pi * (dr/2) * (dr/2) * dz;
            }

	    rho(i, j) = k * (ni(i, j) - ne(i, j)) / cell_volume;
        }
    }
}

void Simulation::debug_field_stats() {
    double max_ez = 0.0, max_er = 0.0;
    double min_ez = 0.0, min_er = 0.0;

    const auto& field_data = electric_field_.data();
    const auto [nz, nr] = field_data.size().to<int>();

    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nr; j++) {
            const auto& field = field_data(i, j);
            max_ez = std::max(max_ez, field.x);
            min_ez = std::min(min_ez, field.x);
            max_er = std::max(max_er, field.y);
            min_er = std::min(min_er, field.y);
        }
    }

    std::cout << "Field Stats: Ez[" << min_ez << ", " << max_ez
              << "] | Er[" << min_er << ", " << max_er << "]\n";
}

void Simulation::debug_particle_counts() {
    int near_axis = 0;
    const double dr = parameters_.dr;
    const auto* x = electrons_.x();
    const size_t n = electrons_.n();

    for (size_t i = 0; i < n; i++) {
        if (x[i].y < dr) near_axis++;
    }

    std::cout << "Particles: e⁻=" << n
              << " ions=" << ions_.n()
              << " | e⁻ near axis: " << near_axis << "\n";
}

Events<Simulation::Event, Simulation::EventAction>& Simulation::events() {
    return events_;
}

void Simulation::set_initial_conditions() {
        auto emitter = [this](double t, double m) {
        return [t, m, this](spark::core::Vec<3>& v, spark::core::Vec<2>& x) {

            x.x = parameters_.lz * spark::random::uniform();
	    x.y = parameters_.lr * sqrt(spark::random::uniform());


            double vth = std::sqrt(spark::constants::kb * t / m);
            // Velocidades em coordenadas cartesianas
            v = {
                spark::random::normal(0.0, vth),
                spark::random::normal(0.0, vth),
                spark::random::normal(0.0, vth)
            };
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
        {{0, 0}, {static_cast<int>(parameters_.nz - 1), 0}, spark::particle::BoundaryType::Specular},
        {{0, static_cast<int>(parameters_.nr - 1)}, {static_cast<int>(parameters_.nz - 1), static_cast<int>(parameters_.nr - 1)}, spark::particle::BoundaryType::Specular},
        {{0, 0}, {0, static_cast<int>(parameters_.nr - 1)}, spark::particle::BoundaryType::Absorbing},
        {{static_cast<int>(parameters_.nz - 1), 0}, {static_cast<int>(parameters_.nz - 1), static_cast<int>(parameters_.nr - 1)}, spark::particle::BoundaryType::Absorbing}
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
