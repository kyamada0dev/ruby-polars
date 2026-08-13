use magnus::{Ruby, Value, prelude::*};
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;

use super::{new_numo_1d, numo_class_name};
use crate::dataframe::RbDataFrame;

impl RbDataFrame {
    /// Convert the DataFrame to a 2-D Numo array of shape `[height, width]`.
    ///
    /// Fast path: when all columns share a fixed-width numeric supertype and have no
    /// nulls, cast each column to that supertype and fill a column-major buffer with a
    /// single memcpy per column, then `reshape([width, height]).transpose` to get
    /// `[height, width]`. Returns None (Ruby falls back to `vstack` of per-column
    /// `to_numo`) for empty frames, columns with nulls, or a non-numeric supertype.
    pub fn to_numo(ruby: &Ruby, self_: &Self) -> Option<Value> {
        let df = self_.df.read();
        let ncols = df.width();
        let nrows = df.height();
        if ncols == 0 || nrows == 0 {
            return None;
        }

        // 全列の supertype を求める
        let mut st: Option<DataType> = None;
        for c in df.columns() {
            let dt = c.dtype();
            st = Some(match st {
                None => dt.clone(),
                Some(cur) => try_get_supertype(&cur, dt).ok()?,
            });
        }
        let st = st?;
        let numo_cls = numo_class_name(&st)?; // 固定幅数値以外は None -> フォールバック

        // 列-major の連続バッファを確保(論理 [nrows, ncols] を [ncols, nrows] として詰める)
        let (arr, dst) = new_numo_1d(ruby, numo_cls, nrows * ncols)?;

        for (j, c) in df.columns().iter().enumerate() {
            let casted = c.cast(&st).ok()?;
            let s = casted.as_materialized_series();
            if s.null_count() != 0 {
                return None; // null は Numo で表現できない -> Ruby フォールバック(NaN 化)へ
            }
            if !copy_series_into(s, &st, dst, j * nrows) {
                return None;
            }
        }

        // [ncols, nrows] へ reshape してから転置 -> [nrows, ncols]
        let reshaped: Value = arr.funcall("reshape", (ncols, nrows)).ok()?;
        let transposed: Value = reshaped.funcall("transpose", ()).ok()?;
        Some(transposed)
    }
}

/// no-null で単一連続チャンクの数値 Series を `dst` の要素オフセット `off` から memcpy。
/// 非対応 dtype / 非連続なら false。
fn copy_series_into(s: &Series, st: &DataType, dst: *mut u8, off: usize) -> bool {
    use DataType::*;
    macro_rules! cp {
        ($ca:expr, $native:ty) => {{
            match $ca.cont_slice() {
                Ok(sl) => {
                    // Safety: `dst` has room for the full matrix; this column writes the
                    // contiguous element range [off, off+sl.len()); source is native-endian.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            sl.as_ptr() as *const u8,
                            dst.add(off * std::mem::size_of::<$native>()),
                            std::mem::size_of_val(sl),
                        );
                    }
                    true
                }
                Err(_) => false,
            }
        }};
    }
    match st {
        Float64 => cp!(s.f64().unwrap(), f64),
        Float32 => cp!(s.f32().unwrap(), f32),
        Int64 => cp!(s.i64().unwrap(), i64),
        Int32 => cp!(s.i32().unwrap(), i32),
        Int16 => cp!(s.i16().unwrap(), i16),
        Int8 => cp!(s.i8().unwrap(), i8),
        UInt64 => cp!(s.u64().unwrap(), u64),
        UInt32 => cp!(s.u32().unwrap(), u32),
        UInt16 => cp!(s.u16().unwrap(), u16),
        UInt8 => cp!(s.u8().unwrap(), u8),
        _ => false,
    }
}
