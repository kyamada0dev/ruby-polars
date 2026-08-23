require_relative "test_helper"

class NumoTest < Minitest::Test
  def test_roundtrip
    assert_roundtrip Numo::Int8
    assert_roundtrip Numo::Int16
    assert_roundtrip Numo::Int32
    assert_roundtrip Numo::Int64
    assert_roundtrip Numo::UInt8
    assert_roundtrip Numo::UInt16
    assert_roundtrip Numo::UInt32
    assert_roundtrip Numo::UInt64
    assert_roundtrip Numo::SFloat
    assert_roundtrip Numo::DFloat
  end

  def test_series_int
    s = Polars::Series.new([1, 2, 3])
    assert_kind_of Numo::Int64, s.to_numo
    assert_equal s.to_a, s.to_numo.to_a
  end

  # Integers have no NaN, so nulls raise by default rather than silently
  # promoting to float (loss of nulls is never silent).
  def test_series_int_nil_raises
    s = Polars::Series.new([1, nil, 3])
    error = assert_raises(ArgumentError) { s.to_numo }
    assert_match(/null/, error.message)
  end

  # null_value: fills nulls first (in Polars), keeping the integer dtype.
  def test_series_int_null_value
    s = Polars::Series.new([1, nil, 3])
    out = s.to_numo(null_value: 0)
    assert_kind_of Numo::Int64, out
    assert_equal [1, 0, 3], out.to_a
  end

  def test_series_float
    s = Polars::Series.new([1.5, 2.5, 3.5])
    assert_kind_of Numo::DFloat, s.to_numo
    assert_equal s.to_a, s.to_numo.to_a
  end

  # A float column fills nulls with NaN (numpy convention).
  def test_series_float_nil
    s = Polars::Series.new([1.0, nil, 3.0])
    out = s.to_numo
    assert_kind_of Numo::DFloat, out
    assert out[1].nan?
    assert_equal [1.0, 3.0], [out[0], out[2]]
  end

  # null_value: overrides the default NaN fill for floats too.
  def test_series_float_null_value
    s = Polars::Series.new([1.0, nil, 3.0])
    assert_equal [1.0, -1.0, 3.0], s.to_numo(null_value: -1.0).to_a
  end

  def test_series_bool
    s = Polars::Series.new([true, false, true])
    assert_kind_of Numo::Bit, s.to_numo
    assert_equal [1, 0, 1], s.to_numo.to_a
  end

  def test_series_bool_nil
    s = Polars::Series.new([true, false, nil])
    assert_kind_of Numo::RObject, s.to_numo
    assert_equal [true, false, nil], s.to_numo.to_a
  end

  def test_series_str
    s = Polars::Series.new(["one", nil, "three"])
    assert_kind_of Numo::RObject, s.to_numo
    assert_equal s.to_a, s.to_numo.to_a
  end

  # Temporal dtypes convert to their physical representation (numpy's
  # datetime64-as-int64 semantics): Date -> Int32 days since the Unix epoch.
  def test_series_date
    epoch = Date.new(1970, 1, 1)
    today = Date.today
    s = Polars::Series.new([today - 2, today - 1, today])
    out = s.to_numo
    assert_kind_of Numo::Int32, out
    assert_equal [(today - 2 - epoch).to_i, (today - 1 - epoch).to_i, (today - epoch).to_i], out.to_a
  end

  # Date + null: physical Int32 with a null promotes to DFloat (NaN for null).
  def test_series_date_nil
    today = Date.today
    s = Polars::Series.new([today - 2, nil, today])
    out = s.to_numo
    assert_kind_of Numo::DFloat, out
    assert out[1].nan?
  end

  def test_series_2d
    s = Polars::Series.new(Numo::Int64.cast([[1, 2], [3, 4]]))
    assert_series [[1, 2], [3, 4]], s, dtype: Polars::Array.new(Polars::Int64, 2)
  end

  def test_data_frame
    df = Polars::DataFrame.new({"a" => [1, 2, 3], "b" => ["one", "two", "three"]})
    assert_kind_of Numo::RObject, df.to_numo
    assert_equal [[1, "one"], [2, "two"], [3, "three"]], df.to_numo.to_a
  end

  # All-numeric frame: nulls in an integer column raise unless null_value given.
  def test_data_frame_int_null_raises
    df = Polars::DataFrame.new({"a" => [1, nil, 3], "b" => [4, 5, 6]})
    assert_raises(ArgumentError) { df.to_numo }
  end

  # null_value fills every column's nulls first, then a single native copy.
  def test_data_frame_null_value
    df = Polars::DataFrame.new({"a" => [1, nil, 3], "b" => [4, 5, 6]})
    out = df.to_numo(null_value: 0)
    assert_equal [[1, 4], [0, 5], [3, 6]], out.to_a
  end

  # A float frame fills nulls with NaN by default.
  def test_data_frame_float_null
    df = Polars::DataFrame.new({"a" => [1.0, nil, 3.0], "b" => [4.0, 5.0, 6.0]})
    out = df.to_numo
    assert_kind_of Numo::DFloat, out
    assert out[1, 0].nan?
    assert_equal [4.0, 5.0, 6.0], out[true, 1].to_a
  end

  def assert_roundtrip(cls)
    v = cls.cast([1, 2, 3])
    assert_equal v, Polars::Series.new(v).to_numo
  end
end
