from openparticle.hamiltonians.yukawa_hamiltonians import *
from openparticle.utils import _get_sign, _get_mass, _get_pminus


def renormalized_Yukawa_hamiltonian(res, t, treg, g, mf, mb):
    # Returns the renormalized Yukawa Hamiltonian up to O(g^2)
    Ham = (
        free_boson_Hamiltonian(res=res, mb=mb)
        + free_fermion_Hamiltonian(res=res, mf=mf)
        + renormalized_Yukawa_first_order(res=res, t=t, treg=treg, g=g, mf=mf, mb=mb)
        + renormalized_Yukawa_second_order_form_factor(
            res=res, t=t, treg=treg, g=g, mf=mf, mb=mb
        )
        + renormalized_Yukawa_second_order_contractions(
            res=res, t=t, treg=treg, g=g, mf=mf, mb=mb
        )
    )
    return Ham


def renormalized_Yukawa_first_order(res, t, treg, g, mf, mb):
    H1 = three_point_yukawa(res=res, g=g, mf=mf, mb=mb)

    ren_H1 = ParticleOperator({})

    for term in H1.to_list():
        exp_factor = 0
        for op in term.split():
            exp_factor += (
                _get_sign(op)
                * _get_mass(op, mf, mb) ** 2
                * _get_pminus(op, mf, mb, res)
            )
        ren_H1 += np.exp(-(exp_factor**2) * (t + treg)) * term

    return ren_H1


def renormalized_Yukawa_second_order_form_factor(res, t, treg, g, mf, mb):
    H1inst = instantaneous_yukawa(res=res, g=g, mf=mf, mb=mb)

    ren_H1inst = ParticleOperator({})

    for term in H1inst.to_list():
        exp_factor = 0
        for op in term.split():
            exp_factor += (
                _get_sign(op)
                * _get_mass(op, mf, mb) ** 2
                * _get_pminus(op, mf, mb, res)
            )
        ren_H1inst += np.exp(-(exp_factor**2) * (t + treg)) * term

    return ren_H1inst


def boson_exchange(t, g, res, mf, mb):

    fermionic_range = np.arange(-res + 1 / 2, res + 1 / 2, 1)
    bosonic_range = [i for i in range(-res, res + 1) if i != 0]

    L = 2 * np.pi * res

    h_tree = ParticleOperator()
    for q1 in fermionic_range:
        for q2 in fermionic_range:
            for q4 in fermionic_range:
                for q5 in fermionic_range:
                    for q3 in bosonic_range:
                        if q1 + q2 + q4 + q5 == 0 and q4 + q5 - q3 == 0 and q3 > 0:
                            q1_, q2_, q3_, q4_, q5_ = pminuses(
                                [q1, q2, q3, q4, q5], [mf, mf, mb, mf, mf]
                            )

                            f123 = np.exp(-t * (q1_ + q2_ + q3_) ** 2)
                            f453_ = np.exp(-t * (q4_ + q5_ - q3_) ** 2)
                            f124_5_ = np.exp(-t * (q1_ + q2_ - q4_ - q5_) ** 2)
                            f1245 = np.exp(-t * (q1_ + q2_ + q4_ + q5_) ** 2)

                            B = (
                                0.5
                                * (1 / (q1_ + q2_ + q3_) - 1 / (q4_ + q5_ - q3_))
                                * (f123 * f453_ / f124_5_ - 1)
                                * f1245
                            )

                            h_tree += (
                                B
                                / np.abs(q3)
                                * (
                                    FermionField(-q1, L, mf).psi_dagger.dot(
                                        FermionField(q2, L, mf).psi.dot(
                                            FermionField(-q4, L, mf).psi_dagger.dot(
                                                FermionField(q5, L, mf).psi
                                            )
                                        )
                                    )
                                )[0][0].normal_order()
                            )

    h_tree = remove_symmetry_terms(h_tree, 4)
    return g**2 / (2 * L) ** 5 * h_tree


def fermion_exchange(t, g, res, mf, mb):

    fermionic_range = np.arange(-res + 1 / 2, res + 1 / 2, 1)
    bosonic_range = [i for i in range(-res, res + 1) if i != 0]

    L = 2 * np.pi * res

    h_tree = ParticleOperator()

    for q1 in fermionic_range:
        for q2 in fermionic_range:
            for q3 in bosonic_range:
                for q5 in fermionic_range:
                    for q6 in bosonic_range:
                        if q1 + q3 + q5 + q6 == 0 and q5 + q6 - q2 == 0:
                            q1_, q2_, q3_, q5_, q6_ = pminuses(
                                [q1, q2, q3, q5, q6], [mf, mf, mb, mf, mb]
                            )

                            f123 = np.exp(-t * (q1_ + q2_ + q3_) ** 2)
                            f562_ = np.exp(-t * (q5_ + q6_ - q2_) ** 2)
                            f135_6_ = np.exp(-t * (q1_ + q3_ - q5_ - q6_) ** 2)
                            f1356 = np.exp(-t * (q1_ + q3_ + q5_ + q6_) ** 2)

                            B = (
                                0.5
                                * (1 / (q1_ + q2_ + q3_) - 1 / (q5_ + q6_ - q1_))
                                * (f123 * f562_ / f135_6_ - 1)
                                * f1356
                            )

                            fermion_field_contractions = FermionField(
                                -q1, L, mf
                            ).psi_dagger.dot(
                                gamma0.dot(
                                    gamma_slash_minus_m(q2, mf).dot(
                                        FermionField(q5, L, mf).psi
                                    )
                                )
                            )[
                                0
                            ][
                                0
                            ]
                            boson_field_contractions = (
                                ScalarField(q3, L, mb).phi * ScalarField(q6, L, mb).phi
                            )

                            if fermion_field_contractions.op_dict != {}:
                                field_contractions = (
                                    fermion_field_contractions
                                    * boson_field_contractions
                                )

                            h_tree += B / (q2) * (field_contractions.normal_order())

    h_tree = remove_symmetry_terms(h_tree, 4)
    return g**2 / (2 * L) ** 5 * h_tree


def fermion_self_energy(t, g, res, mf, mb):

    fermion_loop = ParticleOperator({})

    fermionic_range = np.arange(-res + 1 / 2, res + 1 / 2, 1)
    bosonic_range = np.array([i for i in range(-res, res + 1) if i != 0])

    L = 2 * np.pi * res

    for q1 in fermionic_range:
        for q2 in fermionic_range[fermionic_range > 0]:
            for q3 in bosonic_range[bosonic_range > 0]:
                if q2 + q3 == q1:
                    q1_, q2_, q3_ = pminuses([q1, q2, q3], [mf, mf, mb])

                    f231_ = np.exp(-t * (q2_ + q3_ - q1_) ** 2)

                    B = (f231_**2 - 1) / (q2_ + q3_ - q1_)

                    field_contractions = FermionField(q1, L, mf).psi_dagger.dot(
                        gamma0.dot(
                            gamma_slash_minus_m(q2, mf).dot(FermionField(q1, L, mf).psi)
                        )
                    )[0][0]
                    fermion_loop += B / (q2 * q3) * (field_contractions)

    fermion_loop = remove_symmetry_terms(fermion_loop, 2)
    return g**2 / (2 * L) ** 3 * fermion_loop


def antifermion_self_energy(t, g, res, mf, mb):

    fermion_loop = ParticleOperator({})

    fermionic_range = np.arange(-res + 1 / 2, res + 1 / 2, 1)
    bosonic_range = np.array([i for i in range(-res, res + 1) if i != 0])

    L = 2 * np.pi * res

    for q1 in fermionic_range:
        for q2 in fermionic_range[fermionic_range > 0]:
            for q3 in bosonic_range[bosonic_range > 0]:
                if q2 + q3 == q1:
                    q1_, q2_, q3_ = pminuses([q1, q2, q3], [mf, mf, mb])

                    f231_ = np.exp(-t * (q2_ + q3_ - q1_) ** 2)

                    B = (f231_**2 - 1) / (q2_ + q3_ - q1_)

                    field_contractions = FermionField(-q1, L, mf).psi_dagger.dot(
                        gamma0.dot(
                            gamma_slash_minus_m(-q2, mf).dot(
                                FermionField(-q1, L, mf).psi
                            )
                        )
                    )[0][0]
                    fermion_loop += B / (q2 * q3) * (field_contractions).normal_order()
    fermion_loop = remove_symmetry_terms(fermion_loop, 2)
    return g**2 / (2 * L) ** 3 * fermion_loop


def boson_self_energy(t, g, res, mf, mb):

    boson_loop = ParticleOperator({})

    fermionic_range = np.arange(-res + 1 / 2, res + 1 / 2, 1)
    bosonic_range = np.array([i for i in range(-res, res + 1) if i != 0])

    L = 2 * np.pi * res

    for q1 in fermionic_range[fermionic_range > 0]:
        for q2 in fermionic_range[fermionic_range > 0]:
            for q3 in bosonic_range:
                if q1 + q2 == q3:
                    q1_, q2_, q3_ = pminuses([q1, q2, q3], [mf, mf, mb])

                    f123_ = np.exp(-t * (q1_ + q2_ - q3_) ** 2)

                    B = (f123_**2 - 1) / (q1_ + q2_ - q3_)

                    prop_1 = gamma_slash_minus_m(q1, mf)
                    prop_2 = gamma_slash_minus_m(q2, mf)

                    field_contractions = (
                        ScalarField(-q3, L, mb).phi * ScalarField(q3, L, mb).phi
                    ).normal_order()

                    boson_loop += (
                        B
                        / (q1 * q2)
                        * (np.trace(prop_1.dot(prop_2)))
                        * field_contractions
                    )

    boson_loop = remove_symmetry_terms(boson_loop, 2)
    return g**2 / (2 * L) ** 3 * boson_loop


def fermion_mass_counterterm(res, treg, g, mf):
    L = 2 * np.pi * res

    H_free_fermion = ParticleOperator({})
    for k in np.arange(-res + 1 / 2, res + 1 / 2, 1):
        H_free_fermion += 0.5 * (
            (-np.euler_gamma + np.log((2 * np.pi * k / L) ** 2 / (2 * mf**4 * treg)))
            / p(k, L)
            * (
                FermionField(k, L, mf).psi_dagger.dot(
                    Lambdap.dot(FermionField(k, L, mf).psi)
                )
            )[0][0].normal_order()
        )
    H_free_fermion.remove_identity()
    return 1 / (2 * L) ** 2 * g**2 * H_free_fermion


def boson_mass_counterterm(res, treg, g, mf, mb):
    L = 2 * np.pi * res

    H_free_scalar = ParticleOperator({})
    for k in [i for i in range(-res, res + 1) if i != 0]:
        H_free_scalar += 0.5 * (
            (-np.euler_gamma + np.log((2 * np.pi * k / L) ** 2 / (2 * mf**4 * treg)))
            / p(k, L)
            * (ScalarField(-k, L, mb).phi * ScalarField(k, L, mb).phi).normal_order()
        )
    H_free_scalar.remove_identity()
    return 1 / (2 * L) ** 2 * g**2 * H_free_scalar


def renormalized_Yukawa_second_order_contractions(res, t, treg, g, mf, mb):
    second_order = (
        boson_exchange(t=t + treg, g=g, res=res, mf=mf, mb=mb)
        + fermion_exchange(t=t + treg, g=g, res=res, mf=mf, mb=mb)
        + fermion_self_energy(t=t + treg, g=g, res=res, mf=mf, mb=mb)
        + antifermion_self_energy(t=t + treg, g=g, res=res, mf=mf, mb=mb)
        + boson_self_energy(t=t + treg, g=g, res=res, mf=mf, mb=mb)
        + fermion_mass_counterterm(res=res, treg=treg, g=g, mf=mf)
        + boson_mass_counterterm(res=res, treg=treg, g=g, mf=mf, mb=mb)
    )

    return second_order
