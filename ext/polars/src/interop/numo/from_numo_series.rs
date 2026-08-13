use magnus::rb_sys::AsRawValue;
use magnus::{RString, Value, prelude::*};
use polars::prelude::*;
use rb_sys::StableApiDefinition;

use crate::{RbResult, RbSeries, RbValueError};

// --- Numo NArray の C 構造体(numo-narray-alt の narray.h と一致させる) ---
// typedef struct RNArray { u8 ndim; u8 type; u8 flag[2]; u16 elmsz;
//                          size_t size; size_t* shape; VALUE reduce; } narray_t;  (32B)
// typedef struct RNArrayData { narray_t base; char* ptr; bool owned; } narray_data_t;
#[repr(C)]
struct NarrayT {
    ndim: std::os::raw::c_uchar,
    ntype: std::os::raw::c_uchar,
    flag: [std::os::raw::c_uchar; 2],
    elmsz: std::os::raw::c_ushort,
    size: usize,
    shape: *const usize,
    reduce: rb_sys::VALUE,
}
#[repr(C)]
struct NarrayDataT {
    base: NarrayT,
    ptr: *const u8,
    owned: bool,
}
const NARRAY_DATA_T: u8 = 0x1;

/// Copy `len` contiguous native-endian elements at `buf` into a polars Series.
/// Returns None for unsupported dtypes (caller falls back).
///
/// Safety: `buf` must point to `len` valid, contiguous, native-endian elements of the
/// type named by `class_name`, alive for the duration of the call.
unsafe fn numo_data_to_series(class_name: &str, name: &str, buf: *const u8, len: usize) -> Option<Series> {
    let name: PlSmallStr = name.into();
    // Safety: caller guarantees `buf` holds `len` valid contiguous native-endian elements
    // of the named dtype for the duration of the call.
    Some(match class_name {
        "Numo::DFloat" => {
            let sl = unsafe { std::slice::from_raw_parts(buf as *const f64, len) };
            Float64Chunked::from_slice(name, sl).into_series()
        }
        "Numo::SFloat" => {
            let sl = unsafe { std::slice::from_raw_parts(buf as *const f32, len) };
            Float32Chunked::from_slice(name, sl).into_series()
        }
        "Numo::Int64" => {
            let sl = unsafe { std::slice::from_raw_parts(buf as *const i64, len) };
            Int64Chunked::from_slice(name, sl).into_series()
        }
        "Numo::Int32" => {
            let sl = unsafe { std::slice::from_raw_parts(buf as *const i32, len) };
            Int32Chunked::from_slice(name, sl).into_series()
        }
        _ => return None,
    })
}

impl RbSeries {
    /// Build a Series from a 1-D Numo NArray.
    ///
    /// PoC #1 (minimal): single-copy, no per-element boxing.
    /// - supported dtypes: Numo::DFloat / SFloat / Int64 / Int32
    /// - 1-D only, no nulls
    ///
    /// Data path: `numo.to_binary` (one bulk memcpy into a Ruby String of native-endian
    /// bytes) -> reinterpret -> polars ChunkedArray (arrow buffer). No element-wise
    /// Ruby Object conversion (unlike the current `values.to_a` path).
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
        let s = match class_name.as_str() {
            "Numo::DFloat" => {
                let v: Vec<f64> = slice
                    .chunks_exact(8)
                    .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                Float64Chunked::from_vec(name, v).into_series()
            }
            "Numo::SFloat" => {
                let v: Vec<f32> = slice
                    .chunks_exact(4)
                    .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                Float32Chunked::from_vec(name, v).into_series()
            }
            "Numo::Int64" => {
                let v: Vec<i64> = slice
                    .chunks_exact(8)
                    .map(|b| i64::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                Int64Chunked::from_vec(name, v).into_series()
            }
            "Numo::Int32" => {
                let v: Vec<i32> = slice
                    .chunks_exact(4)
                    .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                Int32Chunked::from_vec(name, v).into_series()
            }
            other => {
                return Err(RbValueError::new_err(format!(
                    "from_numo: unsupported Numo dtype {other} \
                     (PoC supports DFloat/SFloat/Int64/Int32)"
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

        if ndim == 1 {
            let raw = numo.as_raw();
            let data = unsafe { rb_sys::stable_api::get_default().rtypeddata_get_data(raw) }
                as *const NarrayDataT;
            if !data.is_null() {
                let nd = unsafe { &*data };
                // ABI 自己検証 + 連続所有データ(DATA_T)判定
                let layout_ok = nd.base.ndim as usize == ndim && nd.base.size == size;
                if layout_ok && nd.base.ntype == NARRAY_DATA_T && !nd.ptr.is_null() {
                    let class: Value = numo.funcall("class", ())?;
                    let class_name: String = class.funcall("name", ())?;
                    // Safety: DATA_T guarantees `size` contiguous native-endian elements at
                    // `nd.ptr`, owned by the live `numo` arg; from_slice copies out.
                    if let Some(s) =
                        unsafe { numo_data_to_series(&class_name, &name, nd.ptr, size) }
                    {
                        return Ok(RbSeries::new(s));
                    }
                }
            }
        }

        // フォールバック: 公開APIのみの堅牢経路(view / 非対応dtype / レイアウト不一致 等)
        Self::from_numo(name, numo)
    }
}
