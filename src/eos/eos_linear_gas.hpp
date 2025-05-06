#include "eos_crtp.hpp"

/**
 * @file eos_linear_gas.hpp
 * @brief Defines the EOSLinearGas class for the linear gas equation of state.
 *
 * This file provides the implementation of the `EOSLinearGas` class, which
 * models a linear gas equation of state. It computes pressure and speed of
 * sound based on density, internal energy and EOS-specific parameters.
 */

/**
 * @class EOSLinearGas
 * @brief Implements the linear gas equation of state.
 *
 * The `EOSLinearGas` class computes pressure and speed of sound for a linear
 * gas model. It inherits from the `EOSCRTP` base class and provides the
 * implementation for the `PressureSoSImpl` method.
 */
class EOSLinearGas : public EOSCRTP<EOSLinearGas> {
   public:
    /**
     * @brief Constructor for EOSLinearGas.
     *
     * Initializes the EOS id, specific heat ratio (`gamma`) and the minimum
     * cutoff pressure (`pcutoff`).
     *
     * @param id EOS id.
     * @param gamma Specific heat ratio (e.g., 1.4 for air).
     * @param pcutoff Minimum cutoff pressure to avoid negative or unphysical
     * values (e.g. cavitation).
     */
    EOSLinearGas(int id, double gamma, double pcutoff)
        : m_id(id), m_gamma(gamma), m_pcutoff(pcutoff) {}

    /**
     * @brief Computes pressure and speed of sound.
     *
     * This method computes the pressure and speed of sound for a linear gas
     * model based on the given density, internal energy and EOS-specific
     * parameters.
     *
     * @param rho Density Kokkos (sub)view.
     * @param eint Internal energy Kokkos (sub)view.
     * @param p Pressure Kokkos (sub)view (output).
     * @param sos Speed of sound Kokkos (sub)view (output).
     */
    void PressureSoSImpl(Kokkos::View<double *> rho,
                         Kokkos::View<double *> eint, Kokkos::View<double *> p,
                         Kokkos::View<double *> sos) const {
        const auto N = rho.extent(0);
        Kokkos::parallel_for(
            "Initialize", N, KOKKOS_CLASS_LAMBDA(const int i) {
                p(i) =
                    Kokkos::max((m_gamma - 1.0) * rho(i) * eint(i), m_pcutoff);
                sos(i) = m_gamma * p(i) / rho(i);
            });
    }

   private:
    /**
     * @brief Unique identifier for the EOS.
     *
     * This member variable stores a unique integer identifier for the equation
     * of state (EOS). It is used to distinguish between different EOS
     * implementations.
     */
    int m_id;
    /**
     * @brief Specific heat ratio (γ).
     *
     * This member variable represents the specific heat ratio, also known as
     * the adiabatic index. It is a dimensionless quantity that characterizes
     * the thermodynamic properties of the gas. For example, for air, γ is
     * typically 1.4.
     */
    double m_gamma;
    /**
     * @brief Minimum cutoff pressure.
     *
     * This member variable defines the minimum allowable pressure value to
     * avoid negative or unphysical results, such as cavitation. It ensures
     * numerical stability in the computations.
     */
    double m_pcutoff;
};
