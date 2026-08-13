// Polars <-> faer bridge(Rust 内で完結する線形代数)。
// Ruby にも Numo にも行列を出さず、DataFrame in -> 線形代数 -> DataFrame out。
// faer は pure Rust(BLAS/LAPACK 不要)。
use faer::Mat;
use faer::prelude::*; // SolveLstsq
use polars::prelude::*;

use crate::dataframe::RbDataFrame;
use crate::{RbPolarsErr, RbResult};

/// DataFrame -> faer::Mat<f64>(列-major, 列ごとコピー)。
/// 全列 Float64 へ cast + rechunk。null は明示エラー(型/欠損は Polars 側の責務)。
fn df_to_faer_f64(df: &DataFrame) -> PolarsResult<Mat<f64>> {
    let nrows = df.height();
    let ncols = df.width();

    let casted: Vec<Series> = df
        .columns()
        .iter()
        .map(|c| {
            c.cast(&DataType::Float64)
                .map(|col| col.as_materialized_series().rechunk())
        })
        .collect::<PolarsResult<_>>()?;

    for s in &casted {
        if s.null_count() != 0 {
            polars_bail!(ComputeError: "column '{}' has nulls; handle them in Polars first (e.g. drop_nulls / fill_null)", s.name());
        }
    }

    let slices: Vec<&[f64]> = casted
        .iter()
        .map(|s| s.f64().unwrap().cont_slice().unwrap())
        .collect();

    Ok(Mat::from_fn(nrows, ncols, |i, j| slices[j][i]))
}

/// 単一の数値 Series -> faer の列ベクトル(shape [n, 1])。
fn series_to_faer_col(s: &Series) -> PolarsResult<Mat<f64>> {
    let s = s.cast(&DataType::Float64)?.rechunk();
    if s.null_count() != 0 {
        polars_bail!(ComputeError: "target column '{}' has nulls", s.name());
    }
    let sl = s.f64().unwrap().cont_slice().unwrap();
    Ok(Mat::from_fn(sl.len(), 1, |i, _| sl[i]))
}

impl RbDataFrame {
    /// 最小二乗回帰(faer, pure Rust)。`target` 列を目的変数 y、その他の列を設計行列 X として
    /// `min ||X b - y||` を解き、`feature` / `coefficient` の 2 列 DataFrame を返す。
    /// 数値計算は Rust 内で完結し、Ruby 側には DataFrame だけが渡る(Numo/BLAS 不使用)。
    pub fn lstsq(&self, target: String) -> RbResult<Self> {
        let df = self.df.read();

        let feature_names: Vec<PlSmallStr> = df
            .get_column_names_owned()
            .into_iter()
            .filter(|n| n.as_str() != target)
            .collect();

        let x = df.select(feature_names.clone()).map_err(RbPolarsErr::from)?;
        let y = df.column(target.as_str()).map_err(RbPolarsErr::from)?;

        let a = df_to_faer_f64(&x).map_err(RbPolarsErr::from)?;
        let b = series_to_faer_col(y.as_materialized_series()).map_err(RbPolarsErr::from)?;

        // 列ピボット QR で最小二乗(過決定 m>=n を想定)
        let beta = a.col_piv_qr().solve_lstsq(&b); // [k, 1]

        let coef: Vec<f64> = (0..beta.nrows()).map(|i| beta[(i, 0)]).collect();
        let feat: Vec<String> = feature_names.iter().map(|s| s.to_string()).collect();

        let result = DataFrame::new(
            coef.len(),
            vec![
                Series::new("feature".into(), feat).into_column(),
                Series::new("coefficient".into(), coef).into_column(),
            ],
        )
        .map_err(RbPolarsErr::from)?;

        Ok(RbDataFrame::new(result))
    }
}
