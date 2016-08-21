

function sweepmatrix{T <: Number}(x::AMat{T}, y::AVec{T}, addint::Bool = true)
    n, p = size(x)
    d = p + 1 + addint
    A = zeros(d, d)

    rng = (1:p) + addint
    # operations: x'x, x'y, y'y
    BLAS.syrk!('U', 'T', 1 / n, x, 0.0, view(A, rng, rng))
    BLAS.gemv!('T', 1 / n, x, y, 0.0, view(A, rng, d))
    A[end, end] = dot(y, y) / n

    if addint
        #operations: 1'x, 1'y,
        A[1, 2:end - 1] = mean(x, 1)
        A[1, end] = mean(y)
    end
    A
end
