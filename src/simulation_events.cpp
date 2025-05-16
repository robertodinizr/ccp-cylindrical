#include "simulation_events.h"

#include <chrono>
#include <fstream>
#include <span>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace {
    void save_vec(const char* filename, const std::vector<double>& vec, size_t nx, size_t ny) {
        std::ofstream out_file(filename);
        out_file << std::scientific << std::setprecision(6);
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                out_file << vec[i * ny + j];
                if (j < ny - 1) {
                    out_file << " ";
                }
            }
            out_file << "\n";
        }
    }

    std::vector<double> count_to_density(double particle_weight, double dx, double dy,
        const spark::core::TMatrix<double, 2>& count) {
        auto d = std::vector<double>(count.size().mul());
        std::ranges::transform(count.data().begin(), count.data().end(), d.begin(),
            [particle_weight, dx, dy](const double val) {
                return val * particle_weight / (dx * dy);
            });
        return d;
    }

    template <typename SpeciesType>
    void save_particle_velocities(const char* filename, const SpeciesType& species) {
        std::ofstream out_file(filename);
        out_file << std::scientific << std::setprecision(6);
        size_t n_particles = species.n();
        auto v_ptr = species.v();
        for (size_t i = 0; i < n_particles; ++i) {
            out_file << v_ptr[i].x << " " << v_ptr[i].y << " " << v_ptr[i].z << "\n";
        }
    }
} // namespace

namespace spark {

void setup_events(Simulation& simulation) {
    constexpr size_t print_step_interval = 1000;

    struct PrintStartAction : public Simulation::EventAction {
        void notify(const Simulation::StateInterface&) override { printf("Starting simulation\n"); }
    };
    simulation.events().add_action<PrintStartAction>(Simulation::Event::Start);

    struct PrintEvolutionAction : public Simulation::EventAction {
        typedef std::chrono::steady_clock clk;
        typedef std::chrono::duration<double, std::milli> ms;
        std::chrono::time_point<std::chrono::steady_clock> t_last;
        size_t initial_step = 0;
        void notify(const Simulation::StateInterface& s) override {
            auto step = s.step();
            if (step == 0)
                t_last = clk::now();
            if ((step % (print_step_interval / 10) == 0) && (step > 0)) {
                printf("-");
            }
            if ((step % print_step_interval == 0) && (step > 0)) {
                printf("\n");
                const auto now = clk::now();
                const double dur = std::chrono::duration_cast<ms>(now - t_last).count() /
                                   static_cast<double>(s.step() - initial_step);
                t_last = now;
                initial_step = step;
                const float progress = static_cast<float>(step) /
                    static_cast<float>(std::max(1, (int) s.parameters().n_steps - 1));
                const double dur_per_particle = dur / (static_cast<double>(s.electrons().n() + s.ions().n()));
                printf("Info (Step: %zu/%zu, %.2f%%):\n", step, s.parameters().n_steps, progress * 100.0);
                printf("    Avg step duration: %.2fms (%.2eus/p)\n", dur, dur_per_particle * 1e3);
                printf("    Sim electrons: %zu\n", s.electrons().n());
                printf("    Sim ions: %zu\n", s.ions().n());
                printf("\n");
            }
        }
    };
    simulation.events().add_action<PrintEvolutionAction>(Simulation::Event::Step);

    struct AverageFieldAction : public Simulation::EventAction {
        spark::spatial::AverageGrid<2> av_electron_density;
        spark::spatial::AverageGrid<2> av_ion_density;
        Parameters parameters_;
        explicit AverageFieldAction(const Parameters& parameters) : parameters_(parameters) {
            av_electron_density = spark::spatial::AverageGrid<2>({{parameters_.lx, parameters_.ly}, {parameters_.nx, parameters_.ny}});
            av_ion_density = spark::spatial::AverageGrid<2>({{parameters_.lx, parameters_.ly}, {parameters_.nx, parameters_.ny}});
        }
        void notify(const Simulation::StateInterface& s) override {
            if (s.step() > parameters_.n_steps - parameters_.n_steps_avg) {
                av_electron_density.add(s.electron_density());
                av_ion_density.add(s.ion_density());
            }
        }
    };

    auto avg_field_action = simulation.events().add_action(
        Simulation::Event::Step, AverageFieldAction(simulation.state().parameters()));

    struct SaveDataAction : public Simulation::EventAction {
        std::weak_ptr<AverageFieldAction> avg_field_action_;
        Parameters parameters_;
        explicit SaveDataAction(const std::weak_ptr<AverageFieldAction>& avg_field_action,
                                const Parameters& parameters)
            : avg_field_action_(avg_field_action), parameters_(parameters) {}
        void notify(const Simulation::StateInterface& s) override {
            if (!avg_field_action_.expired()) {
                const auto avg_field_action_ptr = avg_field_action_.lock();
                const auto& avg_e = avg_field_action_ptr->av_electron_density.get();
                const auto& avg_i = avg_field_action_ptr->av_ion_density.get();
                auto density_e = count_to_density(parameters_.particle_weight, parameters_.dx, parameters_.dy, avg_e);
                auto density_i = count_to_density(parameters_.particle_weight, parameters_.dx, parameters_.dy, avg_i);
                save_vec("density_e.txt", density_e, parameters_.nx, parameters_.ny);
                save_vec("density_i.txt", density_i, parameters_.nx, parameters_.ny);
            }
        }
    };
    simulation.events().add_action(Simulation::Event::End, SaveDataAction(avg_field_action, simulation.state().parameters()));

    struct SaveGridInfoAction : public Simulation::EventAction {
        Parameters parameters_;
        explicit SaveGridInfoAction(const Parameters& parameters) : parameters_(parameters) {}
        void notify(const Simulation::StateInterface&) override {
            std::ofstream out_file("grid_info.txt");
            out_file << parameters_.lx << " " << parameters_.ly << "\n";
            out_file << parameters_.nx << " " << parameters_.ny << "\n";
        }
    };
    simulation.events().add_action(Simulation::Event::End, SaveGridInfoAction(simulation.state().parameters()));

    struct SaveFieldDataAction : public Simulation::EventAction {
        Parameters parameters_;
        explicit SaveFieldDataAction(const Parameters& parameters) : parameters_(parameters) {}
        void notify(const Simulation::StateInterface& s) override {
            save_vec("phi_field.txt", s.phi_field().data().data(), parameters_.nx, parameters_.ny);
            const auto& E_field = s.electric_field().data();
            std::vector<double> E_x(parameters_.nx * parameters_.ny);
            std::vector<double> E_y(parameters_.nx * parameters_.ny);
            for (size_t i = 0; i < parameters_.nx * parameters_.ny; i++) {
                E_x[i] = E_field.data()[i].x;
                E_y[i] = E_field.data()[i].y;
            }
            save_vec("electric_field_x.txt", E_x, parameters_.nx, parameters_.ny);
            save_vec("electric_field_y.txt", E_y, parameters_.nx, parameters_.ny);
        }
    };
    simulation.events().add_action(Simulation::Event::End, SaveFieldDataAction(simulation.state().parameters()));

    struct SaveParticleDataAction : public Simulation::EventAction {
        Parameters parameters_;
        explicit SaveParticleDataAction(const Parameters& parameters) : parameters_(parameters) {}
        void notify(const Simulation::StateInterface& s) override {
            save_particle_velocities("velocity_e.txt", s.electrons());
            save_particle_velocities("velocity_i.txt", s.ions());
        }
    };
    simulation.events().add_action(Simulation::Event::End, SaveParticleDataAction(simulation.state().parameters()));
}
} // namespace spark
