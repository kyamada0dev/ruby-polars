pub mod from_numo_series;
pub mod to_numo_df;
pub mod to_numo_series;

use magnus::rb_sys::AsRawValue;
use magnus::{RClass, RModule, Ruby, Value, prelude::*};
use polars_core::prelude::DataType;
use rb_sys::StableApiDefinition;

/// polars の固定幅数値 dtype に対応する Numo クラス名。対応外は None。
pub(crate) fn numo_class_name(dt: &DataType) -> Option<&'static str> {
    use DataType::*;
    Some(match dt {
        Float64 => "DFloat",
        Float32 => "SFloat",
        Int64 => "Int64",
        Int32 => "Int32",
        Int16 => "Int16",
        Int8 => "Int8",
        UInt64 => "UInt64",
        UInt32 => "UInt32",
        UInt16 => "UInt16",
        UInt8 => "UInt8",
        _ => return None,
    })
}

// numo-narray-alt の narray.h と一致させた C 構造体レイアウト。
//   narray_t     : ndim(u8) type(u8) flag[2] elmsz(u16) size shape* reduce  = 32B
//   narray_data_t: narray_t base; char* ptr; bool owned;   (ptr @32, owned @40)
#[repr(C)]
#[allow(dead_code)] // 一部フィールドはレイアウト目的で保持(直接は読まない)
pub(crate) struct NarrayT {
    pub ndim: std::os::raw::c_uchar,
    pub ntype: std::os::raw::c_uchar,
    pub flag: [std::os::raw::c_uchar; 2],
    pub elmsz: std::os::raw::c_ushort,
    pub size: usize,
    pub shape: *const usize,
    pub reduce: rb_sys::VALUE,
}
#[repr(C)]
#[allow(dead_code)]
pub(crate) struct NarrayDataT {
    pub base: NarrayT,
    pub ptr: *mut u8,
    pub owned: bool,
}
pub(crate) const NARRAY_DATA_T: u8 = 0x1;

/// Numo NArray VALUE の narray_data_t ポインタを取得(rb-sys stable API 経由)。
/// null なら None。
pub(crate) fn narray_data_t(numo: Value) -> Option<*mut NarrayDataT> {
    let raw = numo.as_raw();
    let p =
        unsafe { rb_sys::stable_api::get_default().rtypeddata_get_data(raw) } as *mut NarrayDataT;
    if p.is_null() { None } else { Some(p) }
}

/// 読み出し用: 1-D 連続 DATA_T の検証済みデータポインタを返す。
/// 構造体の ndim/size を Ruby 由来の期待値と照合(ABI 自己検証)。適用不可なら None。
pub(crate) fn read_data_ptr(numo: Value, ndim: usize, len: usize) -> Option<*const u8> {
    let nd = unsafe { &*narray_data_t(numo)? };
    if nd.base.ndim as usize == ndim
        && nd.base.size == len
        && nd.base.ntype == NARRAY_DATA_T
        && !nd.ptr.is_null()
    {
        Some(nd.ptr as *const u8)
    } else {
        None
    }
}

/// 書き込み用: `Numo::<class>.new(len)` を生成し (値, 書き込み可能ポインタ) を返す。
/// 生成物のレイアウトが想定外(view/未割り当て/サイズ不一致)なら None。
pub(crate) fn new_numo_1d(rb: &Ruby, class: &str, len: usize) -> Option<(Value, *mut u8)> {
    let numo_mod: RModule = rb.class_object().const_get("Numo").ok()?;
    let cls: RClass = numo_mod.const_get(class).ok()?;
    let arr: Value = cls.funcall("new", (len,)).ok()?;
    // Numo は `.new` 時点ではデータ未割り当て(遅延)。`allocate` で malloc を強制する
    // (fill はしない=無駄な走査なし)。以後このバッファへ直接 memcpy する。
    let _: Value = arr.funcall("allocate", ()).ok()?;
    let nd = unsafe { &*narray_data_t(arr)? };
    if nd.base.ntype == NARRAY_DATA_T && nd.base.size == len && !nd.ptr.is_null() {
        Some((arr, nd.ptr))
    } else {
        None
    }
}
