#ifndef SIMULATION_H
#define SIMULATION_H

#include <spark/collisions/mcc.h>
#include <spark/particle/species.h>
#include <spark/spatial/grid.h>
#include <spark/em/poisson.h>
#include <spark/core/matrix.h>
#include <spark/particle/boundary.h>

#include <string>
#include <vector>

#include "events.h"
#include "parameters.h"
#include "spark/core/vec.h"

namespace spark {

    class Simulation {
    public:
        spark::spatial::UniformGrid<2> phi_field_;
        spark::spatial::TUniformGrid<spark::core::TVec<double, 2>, 2> electric_field_;
        class StateInterface {
        public:
            StateInterface(Simulation& sim) : sim_(sim) {}
            const spark::spatial::UniformGrid<2>& electron_density() const { return sim_.electron_density_; }
            const spark::spatial::UniformGrid<2>& ion_density() const { return sim_.ion_density_; }
            const spark::particle::ChargedSpecies<2, 3>& ions() const { return sim_.ions_; }
            const spark::particle::ChargedSpecies<2, 3>& electrons() const { return sim_.electrons_; }
            const Parameters& parameters() const { return sim_.parameters_; }
            size_t step() const { return sim_.step; }
            const spark::spatial::UniformGrid<2>& phi_field() const { return sim_.phi_field_; }
            const spark::spatial::TUniformGrid<spark::core::TVec<double, 2>, 2>& electric_field() const { return sim_.electric_field_; }
        private:
            Simulation& sim_;
        };

        friend StateInterface;

        explicit Simulation(const Parameters& parameters, const std::string& data_path);

        void run();

        enum class Event { Start, Step, End };

        struct EventAction {
            virtual void notify(const StateInterface&) = 0;
            virtual ~EventAction() {}
        };

        Events<Event, EventAction>& events();
        StateInterface& state() { return state_; };

        const spark::spatial::UniformGrid<2>& get_phi_field() const { return phi_field_; }
        const spark::spatial::TUniformGrid<spark::core::TVec<double, 2>, 2>& get_electric_field() const { return electric_field_; }

    private:
        Parameters parameters_;
        std::string data_path_;
        StateInterface state_;
        
        size_t step = 0;
        spark::particle::ChargedSpecies<2, 3> ions_;
        spark::particle::ChargedSpecies<2, 3> electrons_;

        spark::spatial::UniformGrid<2> electron_density_;
        spark::spatial::UniformGrid<2> ion_density_;

        spark::spatial::UniformGrid<2> rho_field_;
        //spark::spatial::UniformGrid<2> phi_field_;

        Events<Event, EventAction> events_;

        void reduce_rho();    
        std::vector<spark::em::StructPoissonSolver2D::Region> region() const;
        spark::spatial::TUniformGrid<spark::core::Vec<2>, 2> convert_electric_field() const;
        //spark::spatial::TUniformGrid<spark::core::TVec<double, 2>, 2> electric_field_;
        spark::core::TMatrix<core::Vec<2>, 1> electron_field;
        spark::core::TMatrix<core::Vec<2>, 1> ion_field;
        spark::particle::TiledBoundary2D tiled_boundary_;

        void set_initial_conditions();
        spark::collisions::MCCReactionSet<2, 3> load_electron_collisions();
        spark::collisions::MCCReactionSet<2, 3> load_ion_collisions();
    };
} // namespace spark

#endif // SIMULATION_H