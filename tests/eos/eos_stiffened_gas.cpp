#include <gtest/gtest.h>
#include "eos_stiffened_gas.hpp"
#include <Kokkos_Core.hpp>

// Test EOSStiffenedGas
TEST(EOSStiffenedGasTest, PressureSoSImpl) {
    Kokkos::initialize();
    {
        EOSStiffenedGas eos(0, 1.4, 1e-6, 0.1);

        Kokkos::View<double*> rho("rho", 1);
        Kokkos::View<double*> eint("eint", 1);
        Kokkos::View<double*> p("p", 1);
        Kokkos::View<double*> sos("sos", 1);

        Kokkos::deep_copy(rho, 1.0);
        Kokkos::deep_copy(eint, 2.0);

        eos.PressureSoSImpl(rho, eint, p, sos);

        auto host_p   = Kokkos::create_mirror_view(p);
        auto host_sos = Kokkos::create_mirror_view(sos);
        Kokkos::deep_copy(host_p, p);
        Kokkos::deep_copy(host_sos, sos);

        EXPECT_NEAR(host_p(0), 0.66, 1e-8);
        EXPECT_NEAR(host_sos(0), 1.064, 1e-8);
    }
    Kokkos::finalize();
}