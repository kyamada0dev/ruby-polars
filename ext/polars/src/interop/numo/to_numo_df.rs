use magnus::{Ruby, Value, prelude::*};
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;

use super::{new_numo_1d, numo_class_name};
use crate::dataframe::RbDataFrame;

impl RbDataFrame {
    /// Convert the DataFrame to a **C-contiguous** 2-D Numo array of shape `[height, width]`.
    ///
    /// Matches numpy/pandas `.to_numpy()`: the result is row-major contiguous, so it can be
    /// handed straight to LAPACK-backed ops (`Numo::Linalg.lstsq/solve/svd`) without a
    /// `.dup`. Fast path: when all columns share a fixed-width numeric supertype and have no
    /// nulls, cast each column to the supertype and scatter it into the row-major buffer
    /// (element `[i, j]` at `i*width + j`), then a single `reshape([height, width])`.
    /// Returns None (Ruby falls back to a per-column `vstack`) for empty frames, columns
    /// with nulls, or a non-numeric supertype.
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

        // 行-major(C-contiguous)バッファ。列 j を stride=ncols で散布して詰める。
        let (arr, dst) = new_numo_1d(ruby, numo_cls, nrows * ncols)?;

        for (j, c) in df.columns().iter().enumerate() {
            let casted = c.cast(&st).ok()?;
            let s = casted.as_materialized_series();
            if s.null_count() != 0 {
                return None; // null は Numo で表現できない -> Ruby フォールバック(NaN 化)へ
            }
            if !scatter_series_into(s, &st, dst, j, ncols) {
                return None;
            }
        }

        // 既に C-order で詰めたので reshape のみ(転置不要)-> C-contiguous [nrows, ncols]
        let reshaped: Value = arr.funcall("reshape", (nrows, ncols)).ok()?;
        Some(reshaped)
    }
}

/// no-null で単一連続チャンクの数値 Series(= 行 i の列 j)を、行-major の `dst` へ
/// `dst[i*ncols + col] = s[i]` と stride 付きで書き込む。非対応 dtype / 非連続なら false。
fn scatter_series_into(s: &Series, st: &DataType, dst: *mut u8, col: usize, ncols: usize) -> bool {
    use DataType::*;
    macro_rules! scat {
        ($ca:expr, $native:ty) => {{
            match $ca.cont_slice() {
                Ok(sl) => {
                    // Safety: base points to nrows*ncols elements; for i in 0..sl.len()
                    // (== nrows) the index i*ncols+col stays within the matrix.
                    let base = dst as *mut $native;
                    for (i, &v) in sl.iter().enumerate() {
                        unsafe {
                            *base.add(i * ncols + col) = v;
                        }
                    }
                    true
                }
                Err(_) => false,
            }
        }};
    }
    match st {
        Float64 => scat!(s.f64().unwrap(), f64),
        Float32 => scat!(s.f32().unwrap(), f32),
        Int64 => scat!(s.i64().unwrap(), i64),
        Int32 => scat!(s.i32().unwrap(), i32),
        Int16 => scat!(s.i16().unwrap(), i16),
        Int8 => scat!(s.i8().unwrap(), i8),
        UInt64 => scat!(s.u64().unwrap(), u64),
        UInt32 => scat!(s.u32().unwrap(), u32),
        UInt16 => scat!(s.u16().unwrap(), u16),
        UInt8 => scat!(s.u8().unwrap(), u8),
        _ => false,
    }
}
