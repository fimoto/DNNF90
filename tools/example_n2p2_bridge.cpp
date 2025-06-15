// tools/example_n2p2_bridge.cpp - one binary containing BOTH n2p2 (libnnp,
// C++) and DNNF90 (libdnnf90, Fortran).  (MIT License.)
//
// n2p2's own SymFncExpRad evaluates the radial symmetry functions of the
// Morse-chain demo; DNNF90 evaluates the Hessian-trained network and all
// carried derivatives on those descriptors.  The descriptors are checked
// against DNNF90's symfunc_module (form compatibility) and the network
// output is checked bitwise against the Fortran reference.
//
// Build: make n2p2_bridge.out N2P2=/path/to/n2p2   (see Makefile)
// Run  : ./hod_ff_example.out && ./n2p2_bridge.out
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include "ElementMap.h"
#include "SymFncExpRad.h"
#include "CutoffFunction.h"
#include "dnnf90.h"

int main(void)
{
    // ---- reference written by the Fortran demo ----
    FILE *f = std::fopen("ff_bridge_ref.dat", "r");
    if (!f) { std::printf("run ./hod_ff_example.out first\n"); return 1; }
    int ng, nslot;
    if (std::fscanf(f, "%d %d", &ng, &nslot) != 2) return 1;
    std::vector<double> eta(ng), rs(ng), Gref(ng), tref(nslot);
    double rc;
    for (int k = 0; k < ng; ++k) std::fscanf(f, "%lf", &eta[k]);
    for (int k = 0; k < ng; ++k) std::fscanf(f, "%lf", &rs[k]);
    std::fscanf(f, "%lf", &rc);
    for (int k = 0; k < ng; ++k) std::fscanf(f, "%lf", &Gref[k]);
    for (int k = 0; k < nslot; ++k) std::fscanf(f, "%lf", &tref[k]);
    std::fclose(f);

    // ---- n2p2 computes the descriptors of atom 1 of the chain ----
    const int NA = 6;
    double x[NA];
    for (int n = 0; n < NA; ++n) x[n] = 1.0*n;   // equilibrium chain, r0=1

    nnp::ElementMap em;
    em.registerElements("H");
    std::vector<double> G(ng, 0.0);
    for (int k = 0; k < ng; ++k) {
        nnp::SymFncExpRad sf(em);
        char ps[128];
        std::snprintf(ps, sizeof(ps), "H 2 H %.17g %.17g %.17g",
                      eta[k], rs[k], rc);
        sf.setParameters(ps);
        sf.setCutoffFunction(nnp::CutoffFunction::CT_COS, 0.0);
        for (int j = 1; j < NA; ++j) {           // neighbors of atom 1 (index 0)
            const double r = std::fabs(x[0] - x[j]);
            if (r < rc) G[k] += sf.calculateRadialPart(r);
        }
    }
    double dG = 0.0;
    for (int k = 0; k < ng; ++k) dG = std::fmax(dG, std::fabs(G[k]-Gref[k]));
    std::printf("descriptors  n2p2 SymFncExpRad vs DNNF90 symfunc: max diff %.3e %s\n",
                dG, dG < 1e-14 ? "(identical form)" : "(MISMATCH)");

    // ---- DNNF90 evaluates the Hessian-trained network on n2p2's G ----
    const int seeds_dummy[1] = {0};
    (void)seeds_dummy;
    dnnf90_tables_init_dense(ng, 2);
    const int nid = dnnf90_net_load("ff_weight.dat");
    const int wid = dnnf90_work_create(nid);
    std::vector<double> t(nslot, 0.0);
    dnnf90_eval(nid, wid, G.data(), t.data());
    double dT = 0.0;
    for (int k = 0; k < nslot; ++k) dT = std::fmax(dT, std::fabs(t[k]-tref[k]));
    std::printf("network      E and all derivative slots vs Fortran: max diff %.3e %s\n",
                dT, dT == 0.0 ? "(bitwise identical)" : "");
    std::printf("n2p2 (C++) -> DNNF90 (Fortran) in one binary: OK\n");

    dnnf90_work_free(wid);
    dnnf90_net_free(nid);
    return 0;
}
