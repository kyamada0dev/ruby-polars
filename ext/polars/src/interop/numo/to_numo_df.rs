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

        // 全列を supertype にキャスト + rechunk(単一連続チャンク化)して保持。
        // read_csv 等は multi-chunk を生み、その場合 cont_slice が失敗するため rechunk する
        // (単一チャンクなら rechunk は実質 no-op)。
        let casted: Vec<Series> = match df
            .columns()
            .iter()
            .map(|c| c.cast(&st).map(|col| col.as_materialized_series().rechunk()))
            .collect::<PolarsResult<Vec<_>>>()
        {
            Ok(v) => v,
            Err(_) => return None,
        };
        for s in &casted {
            if s.null_count() != 0 {
                return None; // null は Numo で表現できない -> Ruby フォールバック(NaN 化)へ
            }
        }

        // 行-major(C-contiguous)バッファへ、出力順に書き込み(全アクセスがシーケンシャル)
        let (arr, dst) = new_numo_1d(ruby, numo_cls, nrows * ncols)?;
        if !fill_row_major(&casted, &st, dst, nrows, ncols) {
            return None;
        }

        // 既に C-order で詰めたので reshape のみ(転置不要)-> C-contiguous [nrows, ncols]
        let reshaped: Value = arr.funcall("reshape", (nrows, ncols)).ok()?;
        Some(reshaped)
    }
}

/// 各列(no-null・単一連続チャンク)を、行-major の `dst` へ `dst[i*ncols + j] = col_j[i]`
/// と出力順(row-major)で書き込む。書き込みは完全シーケンシャル、各列読み出しも
/// シーケンシャル(ncols 本のストリーム)なのでキャッシュ効率が良い。
/// 非対応 dtype / 非連続なら false。
fn fill_row_major(casted: &[Series], st: &DataType, dst: *mut u8, nrows: usize, ncols: usize) -> bool {
    use DataType::*;
    macro_rules! fill {
        ($method:ident, $native:ty) => {{
            let mut slices: Vec<&[$native]> = Vec::with_capacity(ncols);
            for s in casted {
                match s.$method() {
                    Ok(ca) => match ca.cont_slice() {
                        Ok(sl) => slices.push(sl),
                        Err(_) => return false,
                    },
                    Err(_) => return false,
                }
            }
            let base = dst as *mut $native;
            for i in 0..nrows {
                let row = i * ncols;
                for (j, sl) in slices.iter().enumerate() {
                    // Safety: base has nrows*ncols elems (row+j < nrows*ncols); each `sl`
                    // has exactly nrows elems (i < nrows).
                    unsafe {
                        *base.add(row + j) = *sl.get_unchecked(i);
                    }
                }
            }
            true
        }};
    }
    match st {
        Float64 => fill!(f64, f64),
        Float32 => fill!(f32, f32),
        Int64 => fill!(i64, i64),
        Int32 => fill!(i32, i32),
        Int16 => fill!(i16, i16),
        Int8 => fill!(i8, i8),
        UInt64 => fill!(u64, u64),
        UInt32 => fill!(u32, u32),
        UInt16 => fill!(u16, u16),
        UInt8 => fill!(u8, u8),
        _ => false,
    }
}
