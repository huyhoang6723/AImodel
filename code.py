import os
import time
import argparse
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

os.environ["PYTENSOR_FLAGS"] = "cxx="
warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)
logging.getLogger("arviz").setLevel(logging.ERROR)

DEFAULT_INPUT_PATH = r"D:\paper\AImodel\data.csv"
DEFAULT_OUTPUT_DIR = r"D:\paper\AImodel"
DEFAULT_WORKBOOK_NAME = "ai_policy_final_results.xlsx"
DEFAULT_FIGURE_DIR_NAME = "figures"

REQUIRED_COLUMNS = [
    "country_code",
    "year",
    "model_count",
    "ai_policy_lag1",
    "mean_ai_policy_lag1",
    "log_gdp",
    "internet_users_pct",
    "regulatory_quality_filled",
    "regulatory_quality_was_filled"
]

FINAL_PREDICTORS = [
    "z_policy_within",
    "z_policy_maturity",
    "z_national_capability",
    "z_prior_ai_output_base",
    "regulatory_quality_was_filled"
]

PREDICTOR_LABELS = {
    "z_policy_within": "Within-country AI policy shock",
    "z_policy_maturity": "Between-country AI policy maturity",
    "z_national_capability": "National capability",
    "z_prior_ai_output_base": "Prior AI output base",
    "regulatory_quality_was_filled": "Regulatory quality imputation indicator"
}


def step(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def smart_read_table(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_excel(path)


def clean_sheet_name(name):
    return str(name)[:31]


def ensure_dirs(output_dir, figure_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)


def coerce_numeric(data, columns):
    out = data.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def zscore(series):
    mean_value = float(series.mean())
    std_value = float(series.std(ddof=0))
    if std_value == 0 or np.isnan(std_value):
        raise ValueError(f"Invalid standard deviation for {series.name}")
    return (series - mean_value) / std_value, mean_value, std_value


def read_and_clean_data(input_path):
    raw = smart_read_table(input_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = raw[REQUIRED_COLUMNS].copy()
    data["country_code"] = data["country_code"].astype(str).str.strip().str.upper()

    numeric_cols = [col for col in REQUIRED_COLUMNS if col != "country_code"]
    data = coerce_numeric(data, numeric_cols)

    missing_before_drop = data[REQUIRED_COLUMNS].isna().sum().reset_index()
    missing_before_drop.columns = ["variable", "missing_count_before_drop"]
    missing_before_drop["missing_ratio_before_drop"] = missing_before_drop["missing_count_before_drop"] / len(data)

    data = data.dropna(subset=REQUIRED_COLUMNS).copy()
    data["year"] = data["year"].round().astype(int)
    data["model_count"] = data["model_count"].round().astype(int)
    data["regulatory_quality_was_filled"] = data["regulatory_quality_was_filled"].round().astype(int)

    if (data["model_count"] < 0).any():
        raise ValueError("model_count contains negative values.")

    duplicated_rows = int(data.duplicated(["country_code", "year"]).sum())
    if duplicated_rows > 0:
        raise ValueError(f"Duplicated country_code-year rows: {duplicated_rows}")

    data = data.sort_values(["country_code", "year"]).reset_index(drop=True)

    return data, missing_before_drop


def create_model_features(data):
    out = data.copy()

    out["policy_within"] = out["ai_policy_lag1"] - out["mean_ai_policy_lag1"]
    out["policy_maturity"] = out["mean_ai_policy_lag1"]

    out = out.sort_values(["country_code", "year"]).reset_index(drop=True)
    out["cum_model_count_lag1"] = (
        out.groupby("country_code")["model_count"]
        .cumsum()
        .groupby(out["country_code"])
        .shift(1)
        .fillna(0)
    )
    out["prior_ai_output_base"] = np.log1p(out["cum_model_count_lag1"])

    scaling_rows = []

    for col in ["log_gdp", "internet_users_pct", "regulatory_quality_filled"]:
        z_col, mean_value, std_value = zscore(out[col])
        out[f"_capability_component_{col}"] = z_col
        scaling_rows.append({
            "variable": col,
            "created_variable": f"_capability_component_{col}",
            "mean": mean_value,
            "std": std_value,
            "role": "component for national_capability"
        })

    out["national_capability"] = out[
        [
            "_capability_component_log_gdp",
            "_capability_component_internet_users_pct",
            "_capability_component_regulatory_quality_filled"
        ]
    ].mean(axis=1)

    feature_scaling = {
        "policy_within": "z_policy_within",
        "policy_maturity": "z_policy_maturity",
        "national_capability": "z_national_capability",
        "prior_ai_output_base": "z_prior_ai_output_base"
    }

    for original, created in feature_scaling.items():
        z_col, mean_value, std_value = zscore(out[original])
        out[created] = z_col
        scaling_rows.append({
            "variable": original,
            "created_variable": created,
            "mean": mean_value,
            "std": std_value,
            "role": "final model predictor"
        })

    out = out.drop(columns=[col for col in out.columns if col.startswith("_capability_component_")])

    return out, pd.DataFrame(scaling_rows)


def gini(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1) / n)


def distribution_diagnostics(data):
    y = data["model_count"].to_numpy()
    mean_y = float(y.mean())
    var_y = float(y.var(ddof=1))
    zero_ratio = float(np.mean(y == 0))
    variance_to_mean = var_y / mean_y if mean_y > 0 else np.nan
    poisson_zero = float(np.exp(-mean_y)) if mean_y > 0 else np.nan

    if var_y > mean_y and mean_y > 0:
        nb_alpha_mom = (var_y - mean_y) / (mean_y ** 2)
        nb_size = 1 / nb_alpha_mom
        nb_prob = nb_size / (nb_size + mean_y)
        nb_zero = nb_prob ** nb_size
    else:
        nb_alpha_mom = np.nan
        nb_size = np.nan
        nb_prob = np.nan
        nb_zero = np.nan

    return pd.DataFrame([
        {"metric": "observations", "value": len(data)},
        {"metric": "countries", "value": data["country_code"].nunique()},
        {"metric": "min_year", "value": data["year"].min()},
        {"metric": "max_year", "value": data["year"].max()},
        {"metric": "total_model_count", "value": int(data["model_count"].sum())},
        {"metric": "positive_model_count_rows", "value": int((data["model_count"] > 0).sum())},
        {"metric": "zero_model_count_rows", "value": int((data["model_count"] == 0).sum())},
        {"metric": "zero_ratio", "value": zero_ratio},
        {"metric": "mean_model_count", "value": mean_y},
        {"metric": "variance_model_count", "value": var_y},
        {"metric": "variance_to_mean_ratio", "value": variance_to_mean},
        {"metric": "poisson_expected_zero_ratio", "value": poisson_zero},
        {"metric": "zero_excess_over_poisson", "value": zero_ratio - poisson_zero},
        {"metric": "nb_alpha_method_of_moments", "value": nb_alpha_mom},
        {"metric": "nb_expected_zero_ratio_method_of_moments", "value": nb_zero}
    ])


def concentration_summary(data):
    country = (
        data.groupby("country_code", as_index=False)
        .agg(total_model_count=("model_count", "sum"))
        .sort_values("total_model_count", ascending=False)
        .reset_index(drop=True)
    )

    total = float(country["total_model_count"].sum())
    rows = []

    for k in [1, 2, 5, 10, 20]:
        captured = float(country.head(k)["total_model_count"].sum())
        rows.append({
            "top_k_countries": k,
            "captured_model_count": captured,
            "captured_share": captured / total if total > 0 else np.nan
        })

    rows.append({
        "top_k_countries": "gini_country_total_output",
        "captured_model_count": np.nan,
        "captured_share": gini(country["total_model_count"])
    })

    return pd.DataFrame(rows), country


def year_summary(data):
    return (
        data.groupby("year", as_index=False)
        .agg(
            observations=("model_count", "size"),
            countries=("country_code", "nunique"),
            total_model_count=("model_count", "sum"),
            mean_model_count=("model_count", "mean"),
            zero_share=("model_count", lambda x: float((x == 0).mean())),
            total_ai_policy_lag1=("ai_policy_lag1", "sum"),
            mean_ai_policy_lag1=("ai_policy_lag1", "mean"),
            mean_policy_maturity=("policy_maturity", "mean"),
            mean_national_capability=("national_capability", "mean")
        )
    )


def variable_summary(data):
    cols = [
        "model_count",
        "ai_policy_lag1",
        "mean_ai_policy_lag1",
        "policy_within",
        "policy_maturity",
        "log_gdp",
        "internet_users_pct",
        "regulatory_quality_filled",
        "regulatory_quality_was_filled",
        "cum_model_count_lag1",
        "prior_ai_output_base",
        "national_capability"
    ]

    return data[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.reset_index().rename(columns={"index": "variable"})


def vif_table(data):
    x = data[FINAL_PREDICTORS].copy()
    x = sm.add_constant(x, has_constant="add")
    rows = []

    for i, col in enumerate(x.columns):
        if col == "const":
            continue
        try:
            value = variance_inflation_factor(x.values, i)
        except Exception:
            value = np.nan
        rows.append({"variable": col, "vif": value})

    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def correlation_tables(data):
    cols = ["model_count"] + FINAL_PREDICTORS
    pearson = data[cols].corr(method="pearson")
    spearman = data[cols].corr(method="spearman")
    return pearson, spearman


def model_selection_table(data, dist_diag, vif):
    vm = float(dist_diag.loc[dist_diag["metric"] == "variance_to_mean_ratio", "value"].iloc[0])
    zero_ratio = float(dist_diag.loc[dist_diag["metric"] == "zero_ratio", "value"].iloc[0])
    max_vif = float(vif["vif"].max())

    return pd.DataFrame([
        {
            "item": "Outcome type",
            "result": "Non-negative count variable",
            "implication": "Use count model rather than OLS"
        },
        {
            "item": "Zero ratio",
            "result": zero_ratio,
            "implication": "Outcome is sparse and zero-heavy"
        },
        {
            "item": "Variance-to-mean ratio",
            "result": vm,
            "implication": "Strong overdispersion; Poisson is not appropriate"
        },
        {
            "item": "Chosen likelihood",
            "result": "Negative Binomial",
            "implication": "Allows variance to exceed mean"
        },
        {
            "item": "Panel structure",
            "result": "Country-year observations",
            "implication": "Use country random effects and year effects"
        },
        {
            "item": "Policy decomposition",
            "result": "Within-country policy shock and between-country policy maturity",
            "implication": "Correlated random-effects interpretation"
        },
        {
            "item": "Maximum VIF",
            "result": max_vif,
            "implication": "No severe multicollinearity if VIF remains below common thresholds"
        }
    ])


def feature_dictionary():
    return pd.DataFrame([
        {"variable": "model_count", "meaning": "Number of recorded AI models produced by a country in a given year", "role": "dependent variable"},
        {"variable": "policy_within", "meaning": "ai_policy_lag1 minus country-level mean_ai_policy_lag1", "role": "within-country policy shock"},
        {"variable": "policy_maturity", "meaning": "country-level average lagged AI policy intensity", "role": "between-country policy maturity"},
        {"variable": "national_capability", "meaning": "average of standardized log_gdp, internet_users_pct, and regulatory_quality_filled", "role": "constructed capability index"},
        {"variable": "prior_ai_output_base", "meaning": "log1p cumulative model_count before the current year", "role": "past AI output base"},
        {"variable": "regulatory_quality_was_filled", "meaning": "indicator for imputed regulatory quality", "role": "data-quality control"},
        {"variable": "country_code", "meaning": "country grouping identifier", "role": "random effect grouping variable"},
        {"variable": "year", "meaning": "observation year", "role": "year effect grouping variable"}
    ])


def summary_to_numeric(table):
    out = table.copy()
    for col in out.columns:
        if col != "term":
            out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


def find_hdi_columns(table):
    hdi_cols = [col for col in table.columns if str(col).startswith("hdi_")]
    if len(hdi_cols) >= 2:
        return hdi_cols[0], hdi_cols[-1]
    return None, None


def build_irr_table(summary_table):
    if summary_table.empty:
        return pd.DataFrame()

    out = summary_table.copy()
    out["coefficient"] = out["term"].astype(str).str.extract(r"\[(.*)\]")

    for col in ["mean", "sd", "ess_bulk", "ess_tail", "r_hat"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    low_col, high_col = find_hdi_columns(out)

    out["irr_mean"] = np.exp(out["mean"])

    if low_col is not None and high_col is not None:
        out["irr_low"] = np.exp(pd.to_numeric(out[low_col], errors="coerce"))
        out["irr_high"] = np.exp(pd.to_numeric(out[high_col], errors="coerce"))
    else:
        out["irr_low"] = np.exp(out["mean"] - 1.96 * out["sd"])
        out["irr_high"] = np.exp(out["mean"] + 1.96 * out["sd"])

    out["interpretation"] = out["coefficient"].map(PREDICTOR_LABELS).fillna(out["coefficient"])

    keep = [
        "coefficient",
        "interpretation",
        "mean",
        "sd",
        "irr_mean",
        "irr_low",
        "irr_high",
        "ess_bulk",
        "ess_tail",
        "r_hat"
    ]

    return out[[col for col in keep if col in out.columns]]


def run_bayesian_model(data, draws, tune, chains, cores, target_accept, random_seed):
    import pymc as pm
    import arviz as az

    country_categories = pd.Categorical(data["country_code"])
    year_categories = pd.Categorical(data["year"])

    country_index = country_categories.codes
    year_index = year_categories.codes

    country_labels = list(country_categories.categories)
    year_labels = [str(x) for x in list(year_categories.categories)]

    x_matrix = data[FINAL_PREDICTORS].to_numpy(dtype=float)
    y = data["model_count"].to_numpy(dtype=int)

    coords = {
        "observation": np.arange(len(data)),
        "country": country_labels,
        "year": year_labels,
        "coefficient": FINAL_PREDICTORS
    }

    with pm.Model(coords=coords) as model:
        x_data = pm.Data("x_data", x_matrix, dims=("observation", "coefficient"))
        country_data = pm.Data("country_index", country_index, dims="observation")
        year_data = pm.Data("year_index", year_index, dims="observation")

        intercept = pm.Normal("intercept", mu=0, sigma=3)
        beta = pm.Normal("beta", mu=0, sigma=1.5, dims="coefficient")

        sigma_country = pm.HalfNormal("sigma_country", sigma=1.5)
        country_raw = pm.Normal("country_raw", mu=0, sigma=1, dims="country")
        country_effect = pm.Deterministic("country_effect", country_raw * sigma_country, dims="country")

        sigma_year = pm.HalfNormal("sigma_year", sigma=1)
        year_raw = pm.Normal("year_raw", mu=0, sigma=1, dims="year")
        year_uncentered = year_raw * sigma_year
        year_effect = pm.Deterministic("year_effect", year_uncentered - pm.math.mean(year_uncentered), dims="year")

        alpha = pm.Exponential("alpha", lam=1)

        eta = intercept + pm.math.dot(x_data, beta) + country_effect[country_data] + year_effect[year_data]
        mu = pm.Deterministic("mu", pm.math.exp(eta), dims="observation")

        pm.NegativeBinomial("model_count", mu=mu, alpha=alpha, observed=y, dims="observation")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True
        )

        try:
            idata = pm.compute_log_likelihood(idata)
        except Exception:
            pass

        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["model_count"],
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True
        )

    main_summary = az.summary(
        idata,
        var_names=["intercept", "beta", "sigma_country", "sigma_year", "alpha"],
        round_to=None
    ).reset_index().rename(columns={"index": "term"})

    beta_summary = az.summary(
        idata,
        var_names=["beta"],
        round_to=None
    ).reset_index().rename(columns={"index": "term"})

    main_summary = summary_to_numeric(main_summary)
    beta_summary = summary_to_numeric(beta_summary)
    irr = build_irr_table(beta_summary)

    beta_posterior = idata.posterior["beta"]
    beta_mean = beta_posterior.mean(dim=("chain", "draw")).values
    beta_positive = (beta_posterior > 0).mean(dim=("chain", "draw")).values
    irr_above_1 = (np.exp(beta_posterior) > 1).mean(dim=("chain", "draw")).values

    posterior_effects = pd.DataFrame({
        "coefficient": FINAL_PREDICTORS,
        "interpretation": [PREDICTOR_LABELS.get(x, x) for x in FINAL_PREDICTORS],
        "posterior_mean_beta": beta_mean,
        "posterior_probability_beta_positive": beta_positive,
        "posterior_probability_irr_above_1": irr_above_1,
        "posterior_mean_irr": np.exp(beta_mean)
    })

    diagnostics_rows = []

    divergences = int(idata.sample_stats["diverging"].sum().values) if "diverging" in idata.sample_stats else np.nan
    max_rhat = safe_float(main_summary["r_hat"].dropna().max()) if "r_hat" in main_summary.columns else np.nan
    min_ess_bulk = safe_float(main_summary["ess_bulk"].dropna().min()) if "ess_bulk" in main_summary.columns else np.nan
    mean_acceptance = safe_float(idata.sample_stats["acceptance_rate"].mean().values) if "acceptance_rate" in idata.sample_stats else np.nan

    diagnostics_rows.extend([
        {"metric": "status", "value": "success"},
        {"metric": "draws", "value": draws},
        {"metric": "tune", "value": tune},
        {"metric": "chains", "value": chains},
        {"metric": "target_accept", "value": target_accept},
        {"metric": "divergences", "value": divergences},
        {"metric": "max_rhat_main_parameters", "value": max_rhat},
        {"metric": "min_ess_bulk_main_parameters", "value": min_ess_bulk},
        {"metric": "mean_acceptance_rate", "value": mean_acceptance}
    ])

    try:
        loo = az.loo(idata, pointwise=True)
        diagnostics_rows.extend([
            {"metric": "loo_elpd", "value": safe_float(loo.elpd_loo)},
            {"metric": "loo_se", "value": safe_float(loo.se)},
            {"metric": "loo_p", "value": safe_float(loo.p_loo)},
            {"metric": "loo_pareto_k_above_0_7", "value": int((loo.pareto_k > 0.7).sum())}
        ])
    except Exception as e:
        diagnostics_rows.append({"metric": "loo_status", "value": str(e)})

    try:
        waic = az.waic(idata)
        diagnostics_rows.extend([
            {"metric": "waic_elpd", "value": safe_float(waic.elpd_waic)},
            {"metric": "waic_se", "value": safe_float(waic.se)},
            {"metric": "waic_p", "value": safe_float(waic.p_waic)}
        ])
    except Exception as e:
        diagnostics_rows.append({"metric": "waic_status", "value": str(e)})

    posterior_y = (
        ppc.posterior_predictive["model_count"]
        .stack(sample=("chain", "draw"))
        .transpose("observation", "sample")
        .values
    )

    predicted_mean = posterior_y.mean(axis=1)
    predicted_low = np.quantile(posterior_y, 0.025, axis=1)
    predicted_high = np.quantile(posterior_y, 0.975, axis=1)

    predictions = data[["country_code", "year", "model_count"] + FINAL_PREDICTORS].copy()
    predictions["predicted_mean"] = predicted_mean
    predictions["predicted_low_95"] = predicted_low
    predictions["predicted_high_95"] = predicted_high
    predictions["residual"] = predictions["model_count"] - predictions["predicted_mean"]
    predictions["abs_residual"] = predictions["residual"].abs()

    observed_y = predictions["model_count"].to_numpy()

    ppc_summary = pd.DataFrame([
        {"metric": "observed_mean", "value": observed_y.mean()},
        {"metric": "predicted_mean_mean", "value": predicted_mean.mean()},
        {"metric": "observed_variance", "value": observed_y.var(ddof=1)},
        {"metric": "predicted_variance_mean", "value": posterior_y.var(axis=0, ddof=1).mean()},
        {"metric": "observed_zero_ratio", "value": np.mean(observed_y == 0)},
        {"metric": "predicted_zero_ratio_mean", "value": np.mean(posterior_y == 0, axis=0).mean()},
        {"metric": "observed_p95", "value": np.quantile(observed_y, 0.95)},
        {"metric": "predicted_p95_mean", "value": np.quantile(posterior_y, 0.95, axis=0).mean()},
        {"metric": "observed_p99", "value": np.quantile(observed_y, 0.99)},
        {"metric": "predicted_p99_mean", "value": np.quantile(posterior_y, 0.99, axis=0).mean()},
        {"metric": "observed_max", "value": observed_y.max()},
        {"metric": "predicted_max_mean", "value": posterior_y.max(axis=0).mean()},
        {"metric": "interval_95_coverage", "value": np.mean((observed_y >= predicted_low) & (observed_y <= predicted_high))}
    ])

    calibration = predictions.copy()
    calibration["prediction_decile"] = pd.qcut(
        calibration["predicted_mean"].rank(method="first"),
        q=10,
        labels=False
    ) + 1

    calibration_summary = calibration.groupby("prediction_decile", as_index=False).agg(
        n=("model_count", "size"),
        observed_mean=("model_count", "mean"),
        predicted_mean=("predicted_mean", "mean"),
        observed_total=("model_count", "sum"),
        predicted_total=("predicted_mean", "sum")
    )

    total_output = predictions["model_count"].sum()
    ranked = predictions.sort_values("predicted_mean", ascending=False).reset_index(drop=True)
    topk_rows = []

    for k in [0.05, 0.10, 0.20, 0.30]:
        n_top = max(1, int(np.ceil(len(ranked) * k)))
        subset = ranked.head(n_top)
        captured = subset["model_count"].sum()
        topk_rows.append({
            "top_k_share": k,
            "n_observations": n_top,
            "captured_model_count": captured,
            "captured_output_share": captured / total_output if total_output > 0 else np.nan,
            "lift_over_random": (captured / total_output) / k if total_output > 0 else np.nan,
            "mean_observed_model_count": subset["model_count"].mean(),
            "mean_predicted_model_count": subset["predicted_mean"].mean()
        })

    topk_capture = pd.DataFrame(topk_rows)

    largest_residuals = predictions.sort_values("abs_residual", ascending=False).head(30)

    country_effects = az.summary(
        idata,
        var_names=["country_effect"],
        round_to=None
    ).reset_index().rename(columns={"index": "term"})
    country_effects = summary_to_numeric(country_effects)
    country_effects["country_code"] = country_effects["term"].astype(str).str.extract(r"\[(.*)\]")
    country_effects = country_effects.sort_values("mean", ascending=False)

    year_effects = az.summary(
        idata,
        var_names=["year_effect"],
        round_to=None
    ).reset_index().rename(columns={"index": "term"})
    year_effects = summary_to_numeric(year_effects)
    year_effects["year"] = year_effects["term"].astype(str).str.extract(r"\[(.*)\]").astype(str)
    year_effects = year_effects.sort_values("year")

    return {
        "model_summary": main_summary,
        "irr": irr,
        "posterior_effects": posterior_effects,
        "model_diagnostics": pd.DataFrame(diagnostics_rows),
        "predictions": predictions,
        "ppc_summary": ppc_summary,
        "calibration_decile": calibration_summary,
        "topk_capture": topk_capture,
        "largest_residuals": largest_residuals,
        "country_effects": country_effects,
        "year_effects": year_effects
    }


def fallback_model(data):
    x = sm.add_constant(data[FINAL_PREDICTORS], has_constant="add")
    y = data["model_count"]
    model = sm.GLM(y, x, family=sm.families.NegativeBinomial())
    result = model.fit()

    summary = pd.DataFrame({
        "term": result.params.index,
        "coef": result.params.values,
        "std_error": result.bse.values,
        "z": result.tvalues.values,
        "p_value": result.pvalues.values,
        "irr": np.exp(result.params.values)
    })

    predicted = result.predict(x)
    predictions = data[["country_code", "year", "model_count"] + FINAL_PREDICTORS].copy()
    predictions["predicted_mean"] = predicted
    predictions["residual"] = predictions["model_count"] - predictions["predicted_mean"]
    predictions["abs_residual"] = predictions["residual"].abs()

    ppc_summary = pd.DataFrame([
        {"metric": "observed_mean", "value": y.mean()},
        {"metric": "predicted_mean", "value": predicted.mean()},
        {"metric": "observed_variance", "value": y.var(ddof=1)},
        {"metric": "observed_zero_ratio", "value": float((y == 0).mean())},
        {"metric": "observed_max", "value": y.max()},
        {"metric": "predicted_max", "value": predicted.max()}
    ])

    calibration = predictions.copy()
    calibration["prediction_decile"] = pd.qcut(
        calibration["predicted_mean"].rank(method="first"),
        q=10,
        labels=False
    ) + 1

    calibration_summary = calibration.groupby("prediction_decile", as_index=False).agg(
        n=("model_count", "size"),
        observed_mean=("model_count", "mean"),
        predicted_mean=("predicted_mean", "mean"),
        observed_total=("model_count", "sum"),
        predicted_total=("predicted_mean", "sum")
    )

    total_output = predictions["model_count"].sum()
    ranked = predictions.sort_values("predicted_mean", ascending=False).reset_index(drop=True)
    topk_rows = []

    for k in [0.05, 0.10, 0.20, 0.30]:
        n_top = max(1, int(np.ceil(len(ranked) * k)))
        subset = ranked.head(n_top)
        captured = subset["model_count"].sum()
        topk_rows.append({
            "top_k_share": k,
            "n_observations": n_top,
            "captured_model_count": captured,
            "captured_output_share": captured / total_output if total_output > 0 else np.nan,
            "lift_over_random": (captured / total_output) / k if total_output > 0 else np.nan,
            "mean_observed_model_count": subset["model_count"].mean(),
            "mean_predicted_model_count": subset["predicted_mean"].mean()
        })

    return {
        "model_summary": summary,
        "irr": summary,
        "posterior_effects": pd.DataFrame(),
        "model_diagnostics": pd.DataFrame([
            {"metric": "status", "value": "fallback_negative_binomial_glm"},
            {"metric": "aic", "value": result.aic},
            {"metric": "deviance", "value": result.deviance}
        ]),
        "predictions": predictions,
        "ppc_summary": ppc_summary,
        "calibration_decile": calibration_summary,
        "topk_capture": pd.DataFrame(topk_rows),
        "largest_residuals": predictions.sort_values("abs_residual", ascending=False).head(30),
        "country_effects": pd.DataFrame(),
        "year_effects": pd.DataFrame()
    }


def save_figures(data, reports, model_outputs, figure_dir):
    figure_paths = {}

    def save_current(name):
        path = os.path.join(figure_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        figure_paths[name] = path

    plt.figure(figsize=(9, 6))
    plt.hist(data["model_count"], bins=40)
    plt.yscale("log")
    plt.xlabel("Model count")
    plt.ylabel("Frequency, log scale")
    plt.title("Distribution of AI model count")
    save_current("model_count_distribution.png")

    ys = reports["year_summary"]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(ys["year"], ys["total_model_count"], marker="o")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total AI model count")
    ax2 = ax1.twinx()
    ax2.plot(ys["year"], ys["total_ai_policy_lag1"], marker="s")
    ax2.set_ylabel("Total lagged AI policy")
    plt.title("Yearly AI output and lagged AI policy")
    path = os.path.join(figure_dir, "yearly_output_and_policy.png")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["yearly_output_and_policy.png"] = path

    plt.figure(figsize=(9, 6))
    plt.plot(ys["year"], ys["zero_share"], marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Year")
    plt.ylabel("Share of country-years with zero AI models")
    plt.title("Zero-output share by year")
    save_current("zero_output_share_by_year.png")

    top = reports["country_totals"].head(15).sort_values("total_model_count", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top["country_code"], top["total_model_count"])
    plt.xlabel("Total AI model count")
    plt.ylabel("Country")
    plt.title("Top countries by AI model output")
    save_current("top_countries_output.png")

    corr = reports["spearman_corr"]
    plt.figure(figsize=(8, 7))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar(label="Spearman correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Spearman correlation among final variables")
    save_current("spearman_correlation_heatmap.png")

    predictions = model_outputs.get("predictions", pd.DataFrame())
    if not predictions.empty and "predicted_mean" in predictions.columns:
        plt.figure(figsize=(8, 8))
        plt.scatter(predictions["predicted_mean"], predictions["model_count"], alpha=0.45)
        max_value = max(predictions["predicted_mean"].max(), predictions["model_count"].max())
        plt.plot([0, max_value], [0, max_value])
        plt.xlabel("Predicted mean model count")
        plt.ylabel("Observed model count")
        plt.title("Observed vs predicted AI model count")
        plt.xscale("symlog")
        plt.yscale("symlog")
        save_current("observed_vs_predicted.png")

    calibration = model_outputs.get("calibration_decile", pd.DataFrame())
    if not calibration.empty:
        plt.figure(figsize=(9, 6))
        plt.plot(calibration["prediction_decile"], calibration["observed_mean"], marker="o", label="Observed mean")
        plt.plot(calibration["prediction_decile"], calibration["predicted_mean"], marker="s", label="Predicted mean")
        plt.xlabel("Prediction decile")
        plt.ylabel("Mean model count")
        plt.title("Calibration by prediction decile")
        plt.legend()
        save_current("calibration_by_decile.png")

    topk = model_outputs.get("topk_capture", pd.DataFrame())
    if not topk.empty:
        plt.figure(figsize=(9, 6))
        plt.plot(topk["top_k_share"], topk["captured_output_share"], marker="o")
        plt.xlabel("Top-k share by predicted output")
        plt.ylabel("Captured observed output share")
        plt.title("Top-k AI output capture")
        save_current("topk_output_capture.png")

    year_effects = model_outputs.get("year_effects", pd.DataFrame())
    if not year_effects.empty and "mean" in year_effects.columns:
        plot_data = year_effects.copy()
        plot_data["year_numeric"] = pd.to_numeric(plot_data["year"], errors="coerce")
        plot_data = plot_data.dropna(subset=["year_numeric"]).sort_values("year_numeric")
        plt.figure(figsize=(9, 6))
        plt.plot(plot_data["year_numeric"], plot_data["mean"], marker="o")
        low_col, high_col = find_hdi_columns(plot_data)
        if low_col is not None and high_col is not None:
            plt.fill_between(
                plot_data["year_numeric"],
                pd.to_numeric(plot_data[low_col], errors="coerce"),
                pd.to_numeric(plot_data[high_col], errors="coerce"),
                alpha=0.2
            )
        plt.axhline(0, linewidth=1)
        plt.xlabel("Year")
        plt.ylabel("Year effect")
        plt.title("Estimated year effects")
        save_current("year_effects.png")

    manifest = pd.DataFrame([
        {"figure_file": key, "path": value}
        for key, value in figure_paths.items()
    ])

    return manifest


def write_workbook(path, sheets):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for sheet_name, table in sheets.items():
            if table is None:
                continue
            if isinstance(table, pd.DataFrame):
                table.to_excel(writer, sheet_name=clean_sheet_name(sheet_name), index=False)
            else:
                pd.DataFrame(table).to_excel(writer, sheet_name=clean_sheet_name(sheet_name), index=False)

        workbook = writer.book
        header_format = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        text_format = workbook.add_format({"text_wrap": True, "valign": "top"})
        number_format = workbook.add_format({"num_format": "0.0000"})

        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, None, header_format)
            worksheet.set_column(0, 0, 34, text_format)
            worksheet.set_column(1, 20, 18, number_format)


def build_readme(input_path, workbook_path, figure_dir, no_bayesian):
    return pd.DataFrame([
        {"item": "input_data", "value": input_path},
        {"item": "workbook_output", "value": workbook_path},
        {"item": "figure_output_directory", "value": figure_dir},
        {"item": "dependent_variable", "value": "model_count"},
        {"item": "main_model", "value": "Bayesian correlated random-effects Negative Binomial panel model"},
        {"item": "country_role", "value": "grouping variable for country random effects"},
        {"item": "year_role", "value": "grouping variable for year effects"},
        {"item": "bayesian_mode", "value": "disabled" if no_bayesian else "enabled"},
        {"item": "note", "value": "CSV remains raw-clean; transformations and standardization are created inside code."}
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workbook-name", default=DEFAULT_WORKBOOK_NAME)
    parser.add_argument("--figure-dir-name", default=DEFAULT_FIGURE_DIR_NAME)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--no-bayesian", action="store_true")
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir
    figure_dir = os.path.join(output_dir, args.figure_dir_name)
    workbook_path = os.path.join(output_dir, args.workbook_name)

    ensure_dirs(output_dir, figure_dir)

    step("Reading raw-clean data")
    raw_clean, missing_before_drop = read_and_clean_data(input_path)

    step("Creating modeling features inside code")
    data, scaling_parameters = create_model_features(raw_clean)

    step("Building overview and diagnostics")
    dist_diag = distribution_diagnostics(data)
    conc_summary, country_totals = concentration_summary(data)
    ys = year_summary(data)
    var_summary = variable_summary(data)
    vif = vif_table(data)
    pearson_corr, spearman_corr = correlation_tables(data)
    model_selection = model_selection_table(data, dist_diag, vif)

    reports = {
        "year_summary": ys,
        "country_totals": country_totals,
        "spearman_corr": spearman_corr
    }

    if args.no_bayesian:
        step("Running fallback Negative Binomial GLM")
        model_outputs = fallback_model(data)
    else:
        step("Running Bayesian correlated random-effects Negative Binomial panel model")
        try:
            model_outputs = run_bayesian_model(
                data=data,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                target_accept=args.target_accept,
                random_seed=args.seed
            )
            step("Bayesian model finished")
        except Exception as error:
            step(f"Bayesian model failed: {error}")
            step("Running fallback Negative Binomial GLM")
            model_outputs = fallback_model(data)

    step("Saving PNG figures")
    figure_manifest = save_figures(data, reports, model_outputs, figure_dir)

    readme = build_readme(input_path, workbook_path, figure_dir, args.no_bayesian)

    country_effects = model_outputs.get("country_effects", pd.DataFrame())
    if not country_effects.empty:
        country_effects_export = pd.concat([country_effects.head(15), country_effects.tail(15)], axis=0)
    else:
        country_effects_export = pd.DataFrame()

    workbook_sheets = {
        "README": readme,
        "data_overview": pd.concat([
            distribution_diagnostics(data),
            pd.DataFrame([{"metric": "duplicated_country_year", "value": int(data.duplicated(["country_code", "year"]).sum())}])
        ], axis=0),
        "model_selection": model_selection,
        "feature_dictionary": feature_dictionary(),
        "scaling_parameters": scaling_parameters,
        "variable_summary": var_summary,
        "year_summary": ys,
        "country_concentration": conc_summary,
        "top_countries": country_totals.head(20),
        "vif": vif,
        "spearman_corr": spearman_corr.reset_index().rename(columns={"index": "variable"}),
        "model_diagnostics": model_outputs.get("model_diagnostics", pd.DataFrame()),
        "posterior_effects": model_outputs.get("posterior_effects", pd.DataFrame()),
        "incidence_rate_ratio": model_outputs.get("irr", pd.DataFrame()),
        "posterior_predictive": model_outputs.get("ppc_summary", pd.DataFrame()),
        "calibration_decile": model_outputs.get("calibration_decile", pd.DataFrame()),
        "topk_capture": model_outputs.get("topk_capture", pd.DataFrame()),
        "largest_residuals": model_outputs.get("largest_residuals", pd.DataFrame()),
        "year_effects": model_outputs.get("year_effects", pd.DataFrame()),
        "country_effects_top_bottom": country_effects_export,
        "figure_manifest": figure_manifest
    }

    step("Writing compact Excel workbook")
    write_workbook(workbook_path, workbook_sheets)

    step(f"Done workbook: {workbook_path}")
    step(f"Done figures: {figure_dir}")


if __name__ == "__main__":
    main()