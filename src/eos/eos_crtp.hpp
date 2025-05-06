#pragma once

// Kokkos includes
#include <Kokkos_Core.hpp>

#include <iostream>
#include <vector>

/**
 * @file eos_crtp.hpp
 * @brief Defines the CRTP (Curiously Recurring Template Pattern) base class for
 * equation of state (EOS) implementations.
 *
 * This file provides the `EOSCRTP` class template, which serves as a base class
 * for equation of state (EOS) implementations. It uses CRTP to delegate
 * functionality to derived classes without the overhead of virtual functions in
 * order to be GPU-compliant.
 */

/**
 * @class EOSCRTP
 * @brief Base class for equation of state (EOS) implementations using CRTP.
 *
 * The `EOSCRTP` class template provides a common interface for EOS
 * implementations. Derived classes must implement the `PressureSoSImpl` method,
 * which computes pressure and speed of sound based on density, internal energy
 * and EOS-specific parameters.
 *
 */
template <typename Derived>
class EOSCRTP {
   public:
    /**
     * @brief Returns a reference to the derived class.
     *
     * This function is used in the Curiously Recurring Template Pattern (CRTP)
     * to safely cast the base class pointer or reference to the derived type.
     *
     * @return Reference to the derived class.
     */
    Derived& self() { return static_cast<Derived&>(*this); }
    /**
     * @brief Returns a constant reference to the derived class.
     *
     * Const version of self(), used in const-qualified member functions.
     *
     * @return Const reference to the derived class.
     */
    const Derived& self() const { return static_cast<const Derived&>(*this); }
    /**
     * @brief Calls the derived class's implementation of PressureSoS.
     *
     * This function casts the current object to the derived type and calls
     * the derived class's `PressureSoSImpl` function with default-initialized
     * Kokkos views. It is intended to be used in the CRTP pattern for static
     * polymorphism.
     */
    void PressureSoS() const {
        // Call the derived class's implementation
        static_cast<const Derived*>(this)->PressureSoSImpl(
            Kokkos::View<double*>(), Kokkos::View<double*>(),
            Kokkos::View<double*>(), Kokkos::View<double*>());
    }
};