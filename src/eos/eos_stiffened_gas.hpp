#include "eos_crtp.hpp"

/**
 * @file eos_stiffened_gas.hpp
 * @brief Defines the EOSStiffenedGas class for the stiffened gas equation of
 * state.
 *
 * This file provides the implementation of the `EOSStiffenedGas` class, which
 * models a stiffened gas equation of state. It computes pressure and speed of
 * sound based on density, internal energy and EOS-specific parameters.
 */

/**
 * @class EOSStiffenedGas
 * @brief Implements the stiffened gas equation of state.
 *
 * The `EOSStiffenedGas` class computes pressure and speed of sound for a
 * stiffened gas model. It inherits from the `EOSCRTP` base class and provides
 * the implementation for the `PressureSoSImpl` method.
 */
class EOSStiffenedGas : public EOSCRTP<EOSStiffenedGas> {
   public:
    /**
     * @brief Constructor for EOSStiffenedGas.
     *
     * Initializes the EOS id, specific heat ratio (`gamma`), the minimum cutoff
     * pressure (`pcutoff`) and the infinite pressure.
     *
     * @param id EOS id.
     * @param gamma Specific heat ratio (e.g., 1.4 for air).
     * @param pcutoff Minimum cutoff pressure to avoid negative or unphysical
     * values (e.g. cavitation).
     * @param pinf Infinite pressure.
     */
    EOSStiffenedGas(int id, double gamma, double pcutoff, double pinf = 0.0)
        : m_id(id), m_gamma(gamma), m_pcutoff(pcutoff), m_pinf(pinf) {}

    /**
     * @brief Computes pressure and speed of sound.
     *
     * This method computes the pressure and speed of sound for a stiffened gas
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
                p(i) = Kokkos::max(
                    (m_gamma - 1.0) * rho(i) * eint(i) - m_gamma * m_pinf,
                    m_pcutoff);
                sos(i) = m_gamma * (p(i) + m_pinf) / rho(i);
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
    /**
     * @brief Infinite pressure.
     */
    double m_pinf;  // Infinitesimal pressure
};