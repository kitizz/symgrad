# Copyright 2018, Christopher Ham.

from .expression import Expression
from .grad import grad, register_grad, register_grad_rule
from .symbol import *
from .sets import *
from .unary_operator import *
from .binary_operator import *


@register_grad
def grad_float(v: CFloat) -> Zero:
    return Zero()


# TODO: Delete
# @register_grad
# def grad_int(v: CInt) -> Zero:
#     return Zero()


@register_grad
def grad_so3(R: SO3):
    raise NotImplementedError("Direct derivative of SO3 not yet implemented.")


@register_grad
def grad_r33(M: Mat33):
    raise NotImplementedError("Direct derivative of Mat33 is not supported; try verctorizing.")


@register_grad
def grad_r3(v: Vec3) -> Identity3:
    return Identity3()


@register_grad
def grad_r1(v: Reals) -> One:
    return One()


@register_grad_rule(Neg)
def grad_neg(a: Symbol, wrt):
    return -grad(a, wrt)


@register_grad_rule(Sqrt)
def grad_sqrt(a: Symbol, result: Sqrt, wrt):
    return grad(a, wrt) / (CInt(2) * result)


@register_grad_rule(Cos)
def grad_cos(a: Symbol, wrt):
    return -grad(a, wrt) * Sin(a)


@register_grad_rule(Sin)
def grad_sin(a: Symbol, wrt):
    return grad(a, wrt) * Cos(a)


@register_grad_rule(Add)
def grad_add_generic(a: Expression, b: Expression, wrt):
    return grad(a, wrt) + grad(b, wrt)


@register_grad_rule(Multiply)
def grad_mult_generic(a: Symbol, b: Symbol, wrt):
    return grad(a, wrt) * b + a * grad(b, wrt)


@register_grad_rule(ElementWiseInverse)
def grad_inv_generic(a: Symbol, wrt):
    return -grad(a, wrt) * (1 / a**2)


@register_grad_rule(ElementWisePower)
def grad_pow_generic(a: Symbol, b: CInt, wrt):
    # if b == 0:
    #     return 0.0  # (a**0 = 1) is constant.
    return b * (grad(a, wrt) * a ** (b - 1))


@register_grad_rule(Subscript)
def grad_sub_generic(a: Symbol, b: Symbol, wrt):
    return grad(a, wrt)[b]


# @register_grad_rule(Project)
# def grad_project(a: Vec3, result: Vec2, wrt):
#     iz = 1.0 / a[2]
#     da = grad(a * iz, wrt)
#     return (da[:2] - result * da[2])


# -----------------------------------------------
#  Shortcuts on specific arg types for operators
# -----------------------------------------------
@register_grad_rule(Multiply)
def grad_mult_so3_r3(a: SO3, b: Vec3, result: Vec3, wrt) -> SkewSym:
    wrt_a = wrt in a.symbols
    wrt_b = wrt in b.symbols
    if wrt_a and wrt_b:
        raise NotImplementedError("Can only cope with one arg at a time.")

    if wrt_a:
        return SkewSym(result)
    else:
        return a


@register_grad_rule(Dot)
def grad_dot_so3_r3(a: SO3, b: Vec3, result: Vec3, wrt) -> SkewSym:
    wrt_a = wrt in a.symbols
    wrt_b = wrt in b.symbols
    if wrt_a and wrt_b:
        raise NotImplementedError("Can only cope with one arg at a time.")

    if wrt_a:
        return SkewSym(result)
    else:
        return a


@register_grad_rule(Dot)
def grad_dot_r3_r3(a: Vec3, b: Vec3, result: Reals, wrt):
    return dot(grad(a, wrt), b) + dot(a, grad(b, wrt))
