from openparticle.hamiltonians.yukawa_hamiltonians import *
from openparticle.utils import _get_sign, _get_mass, _get_pminus
from scipy.integrate import quad
import numba as nb
import time


def renormalized_yukawa_hamiltonian(res, t, treg=0, g=1, mf=1, mb=1, verbose=False):
    # Returns the renormalized Yukawa Hamiltonian up to O(g^2)
    Ham = (
        free_boson_Hamiltonian(res=res, mb=mb)
        + free_fermion_Hamiltonian(res=res, mf=mf)
        + renormalized_yukawa_first_order(
            res=res, t=t, treg=treg, g=g, mf=mf, mb=mb, verbose=verbose
        )
        + renormalized_yukawa_second_order_form_factor(
            res=res, t=t, treg=treg, g=g, mf=mf, mb=mb, verbose=verbose
        )
        + renormalized_yukawa_second_order_contractions(
            res=res, t=t, treg=treg, g=g, mf=mf, mb=mb, verbose=verbose
        )
    )
    return Ham


def renormalized_yukawa_first_order(res, t, treg, g, mf, mb, verbose=False):
    H1 = three_point_yukawa(res=res, g=g, mf=mf, mb=mb)

    container_dict = dict()

    for term in H1.to_list():
        exp_factor = 0
        for op in term.split():
            exp_factor += (
                _get_sign(op)
                * _get_mass(op, mf, mb) ** 2
                * _get_pminus(op, mf, mb, res)
            )
        helper_variable = np.exp(-(exp_factor**2) * (t + treg)) * term
        for op_str, coeff in helper_variable.op_dict.items():
            container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    ren_H1 = ParticleOperator(container_dict)

    return ren_H1


def renormalized_yukawa_second_order_form_factor(
    res, t, treg, g, mf, mb, verbose=False
):
    start = time.time()
    H1inst = instantaneous_yukawa(res=res, g=g, mf=mf, mb=mb)

    container_dict = dict()

    for term in H1inst.to_list():
        exp_factor = 0
        for op in term.split():
            exp_factor += (
                _get_sign(op)
                * _get_mass(op, mf, mb) ** 2
                * _get_pminus(op, mf, mb, res)
            )

        helper_variable = np.exp(-(exp_factor**2) * (t + treg)) * term
        for op_str, coeff in helper_variable.op_dict.items():
            container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    ren_H1inst = ParticleOperator(container_dict)
    finish = time.time()
    if verbose:
        print("Time of renormalized_yukawa_second_order_form_factor:", finish - start)
    return ren_H1inst


def boson_exchange(t, g, res, mf, mb, verbose=False):
    start = time.time()
    fermionic_range = np.arange(-res + 1 / 2, res + 1 / 2, 1)

    L = 2 * np.pi * res
    container_dict = dict()
    for q1, q4, q5 in product(fermionic_range, fermionic_range, fermionic_range):
        q3 = q4 + q5
        q2 = -q1 - q3
        if q3 > 0 and np.abs(q3) <= res and np.abs(q2) <= res:
            q1_, q2_, q3_, q4_, q5_ = pminuses(
                [q1, q2, q3, q4, q5], [mf, mf, mb, mf, mf]
            )

            f123 = np.exp(-t * (q1_ + q2_ + q3_) ** 2)
            f453_ = np.exp(-t * (q4_ + q5_ - q3_) ** 2)
            f1245 = np.exp(-t * (q1_ + q2_ + q4_ + q5_) ** 2)

            B = (
                0.5
                * (1 / (q1_ + q2_ + q3_) - 1 / (q4_ + q5_ - q3_))
                * (f123 * f453_ / f1245 - 1)
                * f1245
            )

            helper_variable = (
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
            for op_str, coeff in helper_variable.op_dict.items():
                container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    h_tree = ParticleOperator(container_dict)
    h_tree = remove_symmetry_terms(h_tree, 4)
    finish = time.time()
    if verbose:
        print("Time of boson_exchange:", finish - start)
    return g**2 / (2 * L) ** 5 * h_tree


def fermion_exchange(t, g, res, mf, mb, verbose=False):
    start = time.time()
    fermionic_range = np.arange(-res + 1 / 2, res + 1 / 2, 1)
    bosonic_range = [i for i in range(-res, res + 1) if i != 0]

    range_product = product(bosonic_range, fermionic_range, bosonic_range)

    L = 2 * np.pi * res

    container_dict = dict()

    for q3, q5, q6 in product(
        bosonic_range,
        fermionic_range,
        bosonic_range,
    ):
        q1 = -q3 - q5 - q6
        q2 = q5 + q6
        if np.abs(q1) <= res and np.abs(q2) <= res:
            q1_, q2_, q3_, q5_, q6_ = pminuses(
                [q1, q2, q3, q5, q6], [mf, mf, mb, mf, mb]
            )

            f123 = np.exp(-t * (q1_ + q2_ + q3_) ** 2)
            f562_ = np.exp(-t * (q5_ + q6_ - q2_) ** 2)
            f1356 = np.exp(-t * (q1_ + q3_ + q5_ + q6_) ** 2)

            B = (
                0.5
                * (1 / (q1_ + q2_ + q3_) - 1 / (q5_ + q6_ - q2_))
                * (f123 * f562_ / f1356 - 1)
                * f1356
            )

            fermion_field_contractions = FermionField(-q1, L, mf).psi_dagger.dot(
                gamma0.dot(gamma_slash_minus_m(q2, mf).dot(FermionField(q5, L, mf).psi))
            )[0][0]
            boson_field_contractions = (
                ScalarField(q3, L, mb).phi * ScalarField(q6, L, mb).phi
            )

            if fermion_field_contractions.op_dict != {}:
                field_contractions = (
                    fermion_field_contractions * boson_field_contractions
                )

            helper_variable = B / q2 * field_contractions.normal_order()
            for op_str, coeff in helper_variable.op_dict.items():
                container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    h_tree = ParticleOperator(container_dict)

    h_tree = remove_symmetry_terms(h_tree, 4)
    finish = time.time()
    if verbose:
        print("Time of fermion_exchange:", finish - start)
    return g**2 / (2 * L) ** 5 * h_tree


def fermion_loop(t, p, mf, mb, verbose=False):
    start = time.time()

    def integrand(x, t, p, mf, mb):
        return (
            1
            / (x**2 * (1 - x) ** 2)
            * (mf * (2 - x)) ** 2
            / (mf**2 / x + mb**2 / (1 - x) - mf**2)
            * np.exp(-2 * t / p**2 * (mf**2 / x + mb**2 / (1 - x) - mf**2))
        )

    finish = time.time()
    if verbose:
        print("Time of fermion_loop:", finish - start)
    return quad(integrand, 0, 1, args=(t, p, mf, mb))[0]


def fermion_self_energy(t, g, res, mf, mb, verbose: bool = False):

    container_dict = dict()
    start = time.time()

    fermionic_range = np.arange(1 / 2, res, 1)

    L = 2 * np.pi * res

    for k in fermionic_range:
        p1 = p(k, L)
        # TODO: Replace ParticleOperator with a dictionary
        helper_variable = (
            (1 / p1)
            * fermion_loop(t=t, p=p1, mf=mf, mb=mb, verbose=False)
            * ParticleOperator("b" + str(int(k - 1 / 2)) + "^ b" + str(int(k - 1 / 2)))
        )
        for op_str, coeff in helper_variable.op_dict.items():
            container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    _fermion_loop = ParticleOperator(container_dict)

    finish = time.time()
    if verbose:
        print("Time of fermion_self_energy (with loop):", finish - start)
    return g**2 / (2 * L) * _fermion_loop


def antifermion_self_energy(t, g, res, mf, mb, verbose: bool = False):

    container_dict = dict()
    start = time.time()

    fermionic_range = np.arange(1 / 2, res, 1)

    L = 2 * np.pi * res

    for k in fermionic_range:
        p1 = p(k, L)
        # TODO: Replace ParticleOperator with a dictionary
        helper_variable = (
            (1 / p1)
            * fermion_loop(t=t, p=p1, mf=mf, mb=mb, verbose=False)
            * ParticleOperator("d" + str(int(k - 1 / 2)) + "^ d" + str(int(k - 1 / 2)))
        )
        for op_str, coeff in helper_variable.op_dict.items():
            container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    _antifermion_loop = ParticleOperator(container_dict)

    finish = time.time()
    if verbose:
        print("Time of antifermion_self_energy (with loop):", finish - start)
    return g**2 / (2 * L) * _antifermion_loop


def boson_loop(t, p, mf, mb, verbose=False):
    start = time.time()

    def integrand(x, t, p, mf, mb):
        return (
            1
            / (x**2 * (1 - x) ** 2)
            * (mf * (2 * x - 1)) ** 2
            / (mf**2 / x + mf**2 / (1 - x) - mb**2)
            * np.exp(-2 * t / p**2 * (mf**2 / x + mf**2 / (1 - x) - mb**2))
        )

    finish = time.time()
    if verbose:
        print("Time of boson_loop:", finish - start)
    return quad(integrand, 0, 1, args=(t, p, mf, mb))[0]


def boson_self_energy(t, g, res, mf, mb, verbose: bool = False):

    container_dict = dict()
    start = time.time()

    bosonic_range = np.arange(1, res + 1, 1)

    L = 2 * np.pi * res

    for k in bosonic_range:
        p3 = p(k, L)
        # TODO: Replace ParticleOperator with a dictionary
        helper_variable = (
            (1 / p3)
            * boson_loop(t=t, p=p3, mf=mf, mb=mb, verbose=False)
            * ParticleOperator("a" + str(int(k - 1)) + "^ a" + str(int(k - 1)))
        )
        for op_str, coeff in helper_variable.op_dict.items():
            container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    _boson_loop = ParticleOperator(container_dict)

    finish = time.time()
    if verbose:
        print("Time of boson_self_energy (with loop):", finish - start)
    return g**2 / (2 * L) * _boson_loop


def fermion_mass_counterterm(res, treg, g, mf):
    L = 2 * np.pi * res

    container_dict = dict()
    for k in np.arange(-res + 1 / 2, res + 1 / 2, 1):
        # TODO: Use simple formula instead of FermionField
        helper_variable = 0.5 * (
            (-np.euler_gamma + np.log((2 * np.pi * k / L) ** 2 / (2 * mf**4 * treg)))
            / p(k, L)
            * (
                FermionField(k, L, mf).psi_dagger.dot(
                    Lambdap.dot(FermionField(k, L, mf).psi)
                )
            )[0][0].normal_order()
        )
        for op_str, coeff in helper_variable.op_dict.items():
            container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    H_free_fermion = ParticleOperator(container_dict)
    H_free_fermion.remove_identity()
    return 1 / (2 * L) ** 2 * g**2 * H_free_fermion


def boson_mass_counterterm(res, treg, g, mf, mb):
    L = 2 * np.pi * res

    container_dict = dict()
    for k in [i for i in range(-res, res + 1) if i != 0]:
        # TODO: Use simple formula instead of ScalarField
        helper_variable = 0.5 * (
            (-np.euler_gamma + np.log((2 * np.pi * k / L) ** 2 / (2 * mf**4 * treg)))
            / p(k, L)
            * (ScalarField(-k, L, mb).phi * ScalarField(k, L, mb).phi).normal_order()
        )
        for op_str, coeff in helper_variable.op_dict.items():
            container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    H_free_scalar = ParticleOperator(container_dict)
    H_free_scalar.remove_identity()
    return 1 / (2 * L) ** 2 * g**2 * H_free_scalar


def renormalized_yukawa_second_order_contractions(
    res, t, treg, g, mf, mb, verbose: bool = False
):
    second_order = (
        boson_exchange(t=t + treg, g=g, res=res, mf=mf, mb=mb, verbose=verbose)
        + fermion_exchange(t=t + treg, g=g, res=res, mf=mf, mb=mb, verbose=verbose)
        + fermion_self_energy(t=t + treg, g=g, res=res, mf=mf, mb=mb, verbose=verbose)
        + antifermion_self_energy(
            t=t + treg, g=g, res=res, mf=mf, mb=mb, verbose=verbose
        )
        + boson_self_energy(t=t + treg, g=g, res=res, mf=mf, mb=mb, verbose=verbose)
    )

    return second_order
