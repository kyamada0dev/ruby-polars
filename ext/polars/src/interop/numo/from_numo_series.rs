use magnus::{RString, Value, prelude::*};
use polars::prelude::*;

use crate::{RbResult, RbSeries, RbValueError};

/// Copy `len` contiguous native-endian elements at `buf` into a polars Series.
/// Returns None for unsupported dtypes (caller falls back).
///
/// Safety: `buf` must point to `len` valid, contiguous, native-endian elements of the
/// type named by `class_name`, alive for the duration of the call.
unsafe fn numo_data_to_series(class_name: &str, name: &str, buf: *const u8, len: usize) -> Option<Series> {
    let name: PlSmallStr = name.into();
    // Reinterpret the native buffer as `$t` and copy once into an arrow buffer.
    macro_rules! from_ptr {
        ($t:ty, $chunked:ident) => {{
            // Safety: caller guarantees `len` contiguous native-endian `$t` at `buf`.
            let sl = unsafe { std::slice::from_raw_parts(buf as *const $t, len) };
            $chunked::from_slice(name, sl).into_series()
        }};
    }
    Some(match class_name {
        "Numo::DFloat" => from_ptr!(f64, Float64Chunked),
        "Numo::SFloat" => from_ptr!(f32, Float32Chunked),
        "Numo::Int64" => from_ptr!(i64, Int64Chunked),
        "Numo::Int32" => from_ptr!(i32, Int32Chunked),
        "Numo::Int16" => from_ptr!(i16, Int16Chunked),
        "Numo::Int8" => from_ptr!(i8, Int8Chunked),
        "Numo::UInt64" => from_ptr!(u64, UInt64Chunked),
        "Numo::UInt32" => from_ptr!(u32, UInt32Chunked),
        "Numo::UInt16" => from_ptr!(u16, UInt16Chunked),
        "Numo::UInt8" => from_ptr!(u8, UInt8Chunked),
        _ => return None,
    })
}

impl RbSeries {
    /// Build a Series from a 1-D Numo NArray (public-API path).
    ///
    /// Single bulk copy, no per-element boxing: `numo.to_binary` (one memcpy into a Ruby
    /// String of native-endian bytes) -> reinterpret -> polars ChunkedArray (arrow buffer).
    /// Supports every fixed-width numeric Numo dtype (DFloat/SFloat, Int8..64, UInt8..64),
    /// 1-D, no nulls. Used as the robust fallback for `from_numo_ptr`.
    pub fn from_numo(name: String, numo: Value) -> RbResult<Self> {
        let ndim: usize = numo.funcall("ndim", ())?;
        if ndim != 1 {
            return Err(RbValueError::new_err(format!(
                "from_numo: only 1-D NArray is supported for now (got ndim={ndim})"
            )));
        }

        let class: Value = numo.funcall("class", ())?;
        let class_name: String = class.funcall("name", ())?;

        let bytes: RString = numo.funcall("to_binary", ())?;
        // Safety: the slice is only read within this function, and we copy the data out
        // into an owned Vec before returning (no GC-triggering calls in between).
        let slice: &[u8] = unsafe { bytes.as_slice() };

        let name: PlSmallStr = name.into();
        // Reinterpret `slice` (native-endian bytes) as `$t` elements and build a Series.
        macro_rules! from_bytes {
            ($t:ty, $chunked:ident, $width:literal) => {{
                let v: Vec<$t> = slice
                    .chunks_exact($width)
                    .map(|b| <$t>::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                $chunked::from_vec(name, v).into_series()
            }};
        }
        let s = match class_name.as_str() {
            "Numo::DFloat" => from_bytes!(f64, Float64Chunked, 8),
            "Numo::SFloat" => from_bytes!(f32, Float32Chunked, 4),
            "Numo::Int64" => from_bytes!(i64, Int64Chunked, 8),
            "Numo::Int32" => from_bytes!(i32, Int32Chunked, 4),
            "Numo::Int16" => from_bytes!(i16, Int16Chunked, 2),
            "Numo::Int8" => from_bytes!(i8, Int8Chunked, 1),
            "Numo::UInt64" => from_bytes!(u64, UInt64Chunked, 8),
            "Numo::UInt32" => from_bytes!(u32, UInt32Chunked, 4),
            "Numo::UInt16" => from_bytes!(u16, UInt16Chunked, 2),
            "Numo::UInt8" => from_bytes!(u8, UInt8Chunked, 1),
            other => {
                return Err(RbValueError::new_err(format!(
                    "from_numo: unsupported Numo dtype {other} \
                     (supports DFloat/SFloat and Int8..64 / UInt8..64)"
                )));
            }
        };
        Ok(RbSeries::new(s))
    }

    /// Build a Series from a 1-D Numo NArray by reading its data pointer directly.
    ///
    /// Fast path (pointer-direct single copy): reads `narray_data_t.ptr` via rb-sys' stable
    /// API (`rtypeddata_get_data`) and copies once into a polars arrow buffer
    /// (`from_slice`). No intermediate Ruby String (unlike `from_numo`), so one fewer copy.
    ///
    /// This is a SAFE SUPERSET of `from_numo`: any case the fast path can't handle
    /// (non-1-D, NArray view, unsupported dtype, or an unexpected struct layout) is
    /// delegated to `from_numo` (the robust public-API `to_binary` path). It therefore
    /// never produces garbage — worst case it is as correct/slow as `from_numo`.
    ///
    /// Caller (Ruby `numo_to_rbseries`) must ensure native byte order (`!byte_swapped?`);
    /// byte-swapped arrays are routed to the fully-generic `to_a` path instead.
    ///
    /// ABI-drift guard: the struct's `ndim`/`size` are cross-checked against the values
    /// reported by Ruby. If numo-narray-alt ever changes `narray_data_t`'s layout, the
    /// check fails and we fall back to `from_numo` rather than misread memory.
    pub fn from_numo_ptr(name: String, numo: Value) -> RbResult<Self> {
        let ndim: usize = numo.funcall("ndim", ())?;
        let size: usize = numo.funcall("size", ())?;

        // read_data_ptr が ABI 自己検証 + 連続DATA_T判定を行う(適用不可なら None)。
        if ndim == 1 {
            if let Some(buf) = super::read_data_ptr(numo, ndim, size) {
                let class: Value = numo.funcall("class", ())?;
                let class_name: String = class.funcall("name", ())?;
                // Safety: read_data_ptr guarantees `size` contiguous native-endian elements
                // at `buf`, owned by the live `numo` arg; from_slice copies out.
                if let Some(s) = unsafe { numo_data_to_series(&class_name, &name, buf, size) } {
                    return Ok(RbSeries::new(s));
                }
            }
        }

        // フォールバック: 公開APIのみの堅牢経路(view / 非対応dtype / レイアウト不一致 等)
        Self::from_numo(name, numo)
    }
}
