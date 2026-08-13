// Polars <-> faer bridge(Rust 内で完結する線形代数)。
// Ruby にも Numo にも行列を出さず、DataFrame in -> 線形代数 -> DataFrame out。
// faer は pure Rust(BLAS/LAPACK 不要)。
use faer::Mat;
use faer::prelude::*; // SolveLstsq
use polars::prelude::*;

use crate::dataframe::RbDataFrame;
use crate::{RbPolarsErr, RbResult, RbValueError};

/// faer 側のエラー(SvdError 等)を Ruby 例外へ。
fn faer_err<E: std::fmt::Debug>(e: E) -> magnus::Error {
    RbValueError::new_err(format!("faer error: {e:?}"))
}

/// faer::Mat<f64> -> Vec<Column>(列名付き)。
fn faer_mat_to_columns(m: &Mat<f64>, names: &[PlSmallStr]) -> Vec<Column> {
    (0..m.ncols())
        .map(|j| {
            let v: Vec<f64> = (0..m.nrows()).map(|i| m[(i, j)]).collect();
            Series::new(names[j].clone(), v).into_column()
        })
        .collect()
}

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

    /// 正方線形系 `A x = B` を解く(faer, 部分ピボット LU)。`self` = A(n×n)、
    /// `rhs` = B(n×k)。解 X(n×k)を B の列名で DataFrame として返す。
    pub fn solve(&self, rhs: &RbDataFrame) -> RbResult<Self> {
        let a_df = self.df.read();
        let b_df = rhs.df.read();

        let a = df_to_faer_f64(&a_df).map_err(RbPolarsErr::from)?;
        if a.nrows() != a.ncols() {
            return Err(RbValueError::new_err(format!(
                "solve requires a square matrix, got {}x{}",
                a.nrows(),
                a.ncols()
            )));
        }
        let b = df_to_faer_f64(&b_df).map_err(RbPolarsErr::from)?;
        if b.nrows() != a.nrows() {
            return Err(RbValueError::new_err(format!(
                "rhs rows ({}) must match A rows ({})",
                b.nrows(),
                a.nrows()
            )));
        }

        let x = a.partial_piv_lu().solve(&b); // [n, k]
        let names = b_df.get_column_names_owned();
        let result =
            DataFrame::new(x.nrows(), faer_mat_to_columns(&x, &names)).map_err(RbPolarsErr::from)?;
        Ok(RbDataFrame::new(result))
    }

    /// 特異値(SVD の S)を非増加順で返す(faer thin SVD)。1 列 `singular_value`。
    pub fn svd(&self) -> RbResult<Self> {
        let df = self.df.read();
        let a = df_to_faer_f64(&df).map_err(RbPolarsErr::from)?;
        let svd = a.thin_svd().map_err(faer_err)?;
        let s = svd.S().column_vector();
        let vals: Vec<f64> = (0..s.nrows()).map(|i| s[i]).collect();
        let result = DataFrame::new(
            vals.len(),
            vec![Series::new("singular_value".into(), vals).into_column()],
        )
        .map_err(RbPolarsErr::from)?;
        Ok(RbDataFrame::new(result))
    }

    /// 主成分分析(PCA)。列を中心化して thin SVD し、上位 `n_components` 主成分への
    /// 射影(scores = U[:, :k] * S[:k])を `pc1..pck` の DataFrame として返す。
    /// (寄与率は返り値の各列の分散から Polars 側で算出できる。)
    pub fn pca(&self, n_components: usize) -> RbResult<Self> {
        let df = self.df.read();
        let a = df_to_faer_f64(&df).map_err(RbPolarsErr::from)?;
        let (nrows, ncols) = (a.nrows(), a.ncols());
        if nrows == 0 || ncols == 0 {
            return Err(RbValueError::new_err("pca: empty matrix"));
        }

        // 列を中心化(平均を引く)
        let means: Vec<f64> = (0..ncols)
            .map(|j| (0..nrows).map(|i| a[(i, j)]).sum::<f64>() / nrows as f64)
            .collect();
        let centered = Mat::from_fn(nrows, ncols, |i, j| a[(i, j)] - means[j]);

        let svd = centered.thin_svd().map_err(faer_err)?;
        let u = svd.U();
        let s = svd.S().column_vector();
        let k = n_components.min(ncols).min(nrows);

        let cols: Vec<Column> = (0..k)
            .map(|j| {
                let sv = s[j];
                let v: Vec<f64> = (0..nrows).map(|i| u[(i, j)] * sv).collect();
                Series::new(format!("pc{}", j + 1).into(), v).into_column()
            })
            .collect();
        let result = DataFrame::new(nrows, cols).map_err(RbPolarsErr::from)?;
        Ok(RbDataFrame::new(result))
    }
}
