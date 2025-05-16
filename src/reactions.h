#ifndef REACTIONS_H
#define REACTIONS_H

#include <filesystem>

#include "spark/collisions/reaction.h"
#include "parameters.h"

namespace spark::reactions {
spark::collisions::Reactions<2, 3> load_electron_reactions(const std::filesystem::path& dir,
                                                        const Parameters& par,
                                                        spark::particle::ChargedSpecies<2, 3>& ions);

spark::collisions::Reactions<2, 3> load_ion_reactions(const std::filesystem::path& dir,
                                                   const Parameters& par);
}  // namespace spark::reactions

#endif  // REACTIONS_H