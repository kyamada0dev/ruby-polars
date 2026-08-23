require_relative "test_helper"

# In-Rust linear algebra via faer (DataFrame#lstsq/#solve/#svd/#pca/#eig_sym/
# #cholesky). Small, known inputs so results are exact up to float tolerance.
class FaerTest < Minitest::Test
  DELTA = 1e-9

  # y = 1 + 2x exactly, so intercept -> 1, x -> 2.
  def test_lstsq
    df = Polars::DataFrame.new(
      {"intercept" => [1.0, 1.0, 1.0], "x" => [1.0, 2.0, 3.0], "y" => [3.0, 5.0, 7.0]}
    )
    out = df.lstsq("y")
    assert_equal %w[feature coefficient], out.columns
    coef = out.to_a.to_h { |r| [r["feature"], r["coefficient"]] }
    assert_in_delta 1.0, coef["intercept"], DELTA
    assert_in_delta 2.0, coef["x"], DELTA
  end

  # diag(2, 3) x = [2, 9] -> x = [1, 3].
  def test_solve
    a = Polars::DataFrame.new({"c0" => [2.0, 0.0], "c1" => [0.0, 3.0]})
    b = Polars::DataFrame.new({"b0" => [2.0, 9.0]})
    out = a.solve(b)
    assert_equal ["b0"], out.columns
    assert_equal [1.0, 3.0], out["b0"].to_a
  end

  # Singular values of diag(2, 3), non-increasing.
  def test_svd
    a = Polars::DataFrame.new({"c0" => [2.0, 0.0], "c1" => [0.0, 3.0]})
    out = a.svd
    assert_equal ["singular_value"], out.columns
    got = out["singular_value"].to_a
    assert_in_delta 3.0, got[0], DELTA
    assert_in_delta 2.0, got[1], DELTA
  end

  # Eigenvalues of a symmetric matrix, non-decreasing (like numpy eigvalsh).
  def test_eig_sym
    a = Polars::DataFrame.new({"c0" => [2.0, 0.0], "c1" => [0.0, 3.0]})
    got = a.eig_sym["eigenvalue"].to_a
    assert_in_delta 2.0, got[0], DELTA
    assert_in_delta 3.0, got[1], DELTA
  end

  # A = [[4,2],[2,3]] -> L = [[2,0],[1,sqrt(2)]], lower-triangular, A = L Lᵀ.
  def test_cholesky
    a = Polars::DataFrame.new({"c0" => [4.0, 2.0], "c1" => [2.0, 3.0]})
    out = a.cholesky
    assert_equal %w[c0 c1], out.columns
    rows = out.to_a
    assert_in_delta 2.0, rows[0]["c0"], DELTA
    assert_in_delta 0.0, rows[0]["c1"], DELTA
    assert_in_delta 1.0, rows[1]["c0"], DELTA
    assert_in_delta Math.sqrt(2), rows[1]["c1"], DELTA
  end

  # Perfectly collinear x, y -> one component explains everything; scores span
  # one column of the row count.
  def test_pca
    df = Polars::DataFrame.new({"x" => [1.0, 2.0, 3.0, 4.0], "y" => [2.0, 4.0, 6.0, 8.0]})
    out = df.pca(1)
    assert_equal ["pc1"], out.columns
    assert_equal 4, out.height
    # Centered + projected: scores are monotone across the ordered rows.
    scores = out["pc1"].to_a
    assert scores.each_cons(2).all? { |a, b| a < b } || scores.each_cons(2).all? { |a, b| a > b }
  end

  # lstsq needs float input and no nulls; integer/nullable input raises.
  def test_lstsq_rejects_nulls
    df = Polars::DataFrame.new({"x" => [1.0, 2.0, nil], "y" => [1.0, 2.0, 3.0]})
    assert_raises(RuntimeError, Polars::Error) { df.lstsq("y") }
  end
end
