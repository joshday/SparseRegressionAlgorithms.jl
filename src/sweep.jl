#-----------------------------------------------------------------------# sweepmatrix
"""
`sweepmatrix(x, y, addint = true)`

`sweepmatrix(x, y, wts, addint = true)`

Create the matrix [x y]' * [x y] for sweeping, optionally with weights
"""
function sweepmatrix{T <: LinAlg.BlasFloat}(x::AMat{T}, y::AVec{T}, addint::Bool = true)
    n, p = size(x)
    d = p + 1 + addint
    A = zeros(d, d)

    rng = (1:p) + addint
    # operations: x'x, x'y, y'y
    BLAS.syrk!('U', 'T', 1 / n, x, 0.0, view(A, rng, rng))
    BLAS.gemv!('T', 1 / n, x, y, 0.0, view(A, rng, d))
    A[end, end] = dot(y, y) / n

    if addint
        #operations: 1'1, 1'x, 1'y,
        A[1, 1] = 1.0
        A[1, 2:end - 1] = mean(x, 1)
        A[1, end] = mean(y)
    end
    A
end
function sweepmatrix{T <: LinAlg.BlasFloat}(x::AMat{T}, y::AVec{T}, wts::AVec{T}, addint::Bool = true)
    n, p = size(x)
    d = p + 1 + addint
    wts = wts / sum(wts)
    A = zeros(d, d)
    W = Diagonal(sqrt(wts))
    Wx = W * x
    Wy = W * y

    rng = (1:p) + addint
    # operations: x'Wx, x'Wy, y'Wy
    BLAS.syrk!('U', 'T', 1 / n, Wx, 0.0, view(A, rng, rng))
    BLAS.gemv!('T', 1 / n, Wx, Wy, 0.0, view(A, rng, d))
    A[end, end] = dot(y, Wy) / n

    if addint
        #operations: 1'Wx, 1'Wy,
        A[1, 2:end - 1] = mean(Wx, 1)
        A[1, end] = mean(y, StatsBase.WeightVec(wts))
    end
    A
end

"Add a ridge penalty to the sweepmatrix"
function addridge!{T <: LinAlg.BlasFloat}(A::AMat{T}, λ::T, intercept::Bool)
    rng = (1 + intercept):(size(A, 1) - 1)
    for i in rng
        A[i, i] *= λ
    end
    A
end


# Unpenalized or Ridge-regularized regression
function sweepreg{T <: LinAlg.BlasFloat}(x::AMat{T}, y::AVec{T}, addint::Bool = true; λ::T = zero(T))
    A = sweepmatrix(x, y, addint)
    if λ > zero(T)
        addridge!(A, λ, addint)
    end
    SweepOperator.sweep!(A, 1:size(A, 1) - 1)
    β = A[1:end-1, end]
    SSE = A[end, end]
    minusXTXinv = A[1:end-1, 1:end-1]
    β, SSE, minusXTXinv
end
