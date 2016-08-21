module Tests
using SparseRegressionAlgorithms
S = SparseRegressionAlgorithms
using Base.Test

macro display(ex)
    :(display($ex))
end

@testset "sweep" begin
    n, p = 1000, 5
    x = randn(n, p)
    y = x * collect(1:p) + randn(n)
    w = rand(n)
    @display S.sweepreg(x, y)
end
end
