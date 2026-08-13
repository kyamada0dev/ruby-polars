use magnus::{Ruby, Value};
use num_traits::{Float, NumCast};
use polars_core::prelude::*;

use crate::RbResult;
use crate::error::RbPolarsErr;
use crate::raise_err;
use crate::ruby::numo::{Element, RbArray1};
use crate::series::RbSeries;

impl RbSeries {
    /// Convert this Series to a Numo array.
    pub fn to_numo(rb: &Ruby, self_: &Self) -> RbResult<Value> {
        series_to_numo(rb, &self_.series.read())
    }
}

/// Convert a Series to a Numo array.
fn series_to_numo(rb: &Ruby, s: &Series) -> RbResult<Value> {
    if let Some(v) = series_to_numo_fast(rb, s) {
        return Ok(v);
    }
    series_to_numo_with_copy(rb, s)
}

/// Fast path: allocate a Numo array and copy the contiguous buffer in a single memcpy
/// (no per-element Ruby Object boxing). Only for fixed-width numeric dtypes with no nulls
/// and a single contiguous chunk; returns None (caller falls back) otherwise.
fn series_to_numo_fast(rb: &Ruby, s: &Series) -> Option<Value> {
    if s.null_count() != 0 {
        return None;
    }
    use DataType::*;

    // $ca: contiguous ChunkedArray, $cls: Numo class name
    macro_rules! copy_into_numo {
        ($ca:expr, $cls:literal) => {{
            let sl = $ca.cont_slice().ok()?;
            let (arr, dst) = super::new_numo_1d(rb, $cls, sl.len())?;
            // Safety: `dst` is a freshly allocated Numo buffer of `sl.len()` elements of the
            // same width; source and destination are non-overlapping and native-endian.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    sl.as_ptr() as *const u8,
                    dst,
                    std::mem::size_of_val(sl),
                );
            }
            Some(arr)
        }};
    }

    match s.dtype() {
        Float64 => copy_into_numo!(s.f64().ok()?, "DFloat"),
        Float32 => copy_into_numo!(s.f32().ok()?, "SFloat"),
        Int64 => copy_into_numo!(s.i64().ok()?, "Int64"),
        Int32 => copy_into_numo!(s.i32().ok()?, "Int32"),
        Int16 => copy_into_numo!(s.i16().ok()?, "Int16"),
        Int8 => copy_into_numo!(s.i8().ok()?, "Int8"),
        UInt64 => copy_into_numo!(s.u64().ok()?, "UInt64"),
        UInt32 => copy_into_numo!(s.u32().ok()?, "UInt32"),
        UInt16 => copy_into_numo!(s.u16().ok()?, "UInt16"),
        UInt8 => copy_into_numo!(s.u8().ok()?, "UInt8"),
        _ => None,
    }
}

/// Convert a Series to a Numo array, copying data in the process.
fn series_to_numo_with_copy(rb: &Ruby, s: &Series) -> RbResult<Value> {
    use DataType::*;
    match s.dtype() {
        Int8 => numeric_series_to_numo::<Int8Type, f32>(rb, s),
        Int16 => numeric_series_to_numo::<Int16Type, f32>(rb, s),
        Int32 => numeric_series_to_numo::<Int32Type, f64>(rb, s),
        Int64 => numeric_series_to_numo::<Int64Type, f64>(rb, s),
        UInt8 => numeric_series_to_numo::<UInt8Type, f32>(rb, s),
        UInt16 => numeric_series_to_numo::<UInt16Type, f32>(rb, s),
        UInt32 => numeric_series_to_numo::<UInt32Type, f64>(rb, s),
        UInt64 => numeric_series_to_numo::<UInt64Type, f64>(rb, s),
        Float32 => numeric_series_to_numo::<Float32Type, f32>(rb, s),
        Float64 => numeric_series_to_numo::<Float64Type, f64>(rb, s),
        Boolean => boolean_series_to_numo(rb, s),
        String => {
            let ca = s.str().unwrap();
            let values = ca.iter();
            RbArray1::from_iter(rb, values)
        }
        dt => {
            raise_err!(
                format!("'to_numo' not supported for dtype: {dt:?}"),
                ComputeError
            );
        }
    }
}

/// Convert numeric types to f32 or f64 with NaN representing a null value.
fn numeric_series_to_numo<T, U>(rb: &Ruby, s: &Series) -> RbResult<Value>
where
    T: PolarsNumericType,
    T::Native: Element,
    U: Float + Element,
{
    let ca: &ChunkedArray<T> = s.as_ref().as_ref();
    if s.null_count() == 0 {
        let values = ca.into_no_null_iter();
        RbArray1::<T::Native>::from_iter(rb, values)
    } else {
        let mapper = |opt_v: Option<T::Native>| match opt_v {
            Some(v) => NumCast::from(v).unwrap(),
            None => U::nan(),
        };
        let values = ca.iter().map(mapper);
        RbArray1::from_iter(rb, values)
    }
}

/// Convert booleans to bit if no nulls are present, otherwise convert to objects.
fn boolean_series_to_numo(rb: &Ruby, s: &Series) -> RbResult<Value> {
    let ca = s.bool().unwrap();
    if s.null_count() == 0 {
        let values = ca.no_null_iter();
        RbArray1::<bool>::from_iter(rb, values)
    } else {
        let values = ca.iter();
        RbArray1::from_iter(rb, values)
    }
}
