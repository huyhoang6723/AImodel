import os
import time
import logging
import warnings
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

input_path = r"D:\paper\AImodel\data.csv"
output_dir = r"D:\paper\AImodel\results"
figure_dir = os.path.join(output_dir, "figures")
workbook_path = os.path.join(output_dir, "ai_governance_final_results.xlsx")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

random_seed = 20260707
draws = 1000
tune = 1000
chains = 4
cores = 1
target_accept = 0.92

required_columns = [
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


def step(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def to_numeric_table(table):
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

    numeric_cols = [
        "mean",
        "sd",
        "hdi_3%",
        "hdi_97%",
        "hdi_2.5%",
        "hdi_97.5%",
        "ess_bulk",
        "ess_tail",
        "r_hat"
    ]

    for col in numeric_cols:
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

    keep = [
        "term",
        "coefficient",
        "mean",
        "sd",
        "irr_mean",
        "irr_low",
        "irr_high",
        "ess_bulk",
        "ess_tail",
        "r_hat"
    ]

    keep = [col for col in keep if col in out.columns]
    return out[keep]


def clean_data(path):
    raw = pd.read_csv(path)

    missing_cols = [col for col in required_columns if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = raw[required_columns].copy()
    data["country_code"] = data["country_code"].astype(str).str.strip().str.upper()

    for col in required_columns:
        if col != "country_code":
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=required_columns).copy()
    data["year"] = data["year"].astype(int)
    data["model_count"] = data["model_count"].round().astype(int)
    data["regulatory_quality_was_filled"] = data["regulatory_quality_was_filled"].round().astype(int)

    if (data["model_count"] < 0).any():
        raise ValueError("model_count contains negative values.")

    duplicates = data.duplicated(["country_code", "year"]).sum()
    if duplicates > 0:
        raise ValueError(f"Duplicated country_code-year rows: {duplicates}")

    data = data.sort_values(["country_code", "year"]).reset_index(drop=True)
    data["policy_within"] = data["ai_policy_lag1"] - data["mean_ai_policy_lag1"]

    return data


def standardize_data(data):
    standardize_cols = [
        "policy_within",
        "mean_ai_policy_lag1",
        "log_gdp",
        "internet_users_pct",
        "regulatory_quality_filled"
    ]

    rows = []

    for col in standardize_cols:
        mean_value = data[col].mean()
        std_value = data[col].std(ddof=0)

        if std_value == 0 or np.isnan(std_value):
            raise ValueError(f"Invalid standard deviation for {col}")

        data[f"z_{col}"] = (data[col] - mean_value) / std_value

        rows.append({
            "variable": col,
            "mean": mean_value,
            "std": std_value
        })

    return data, pd.DataFrame(rows)


def distribution_diagnostics(data):
    y = data["model_count"].to_numpy()
    mean_y = y.mean()
    var_y = y.var(ddof=1)
    zero_ratio = np.mean(y == 0)

    poisson_zero = np.exp(-mean_y) if mean_y > 0 else np.nan
    variance_to_mean = var_y / mean_y if mean_y > 0 else np.nan

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
        {"metric": "mean_model_count", "value": mean_y},
        {"metric": "variance_model_count", "value": var_y},
        {"metric": "variance_to_mean_ratio", "value": variance_to_mean},
        {"metric": "observed_zero_ratio", "value": zero_ratio},
        {"metric": "poisson_expected_zero_ratio", "value": poisson_zero},
        {"metric": "zero_excess_over_poisson", "value": zero_ratio - poisson_zero},
        {"metric": "nb_alpha_method_of_moments", "value": nb_alpha_mom},
        {"metric": "nb_size_method_of_moments", "value": nb_size},
        {"metric": "nb_zero_ratio_method_of_moments", "value": nb_zero},
        {"metric": "zero_excess_over_nb", "value": zero_ratio - nb_zero}
    ])


def make_reports(data, scaling_table):
    data_audit = pd.DataFrame([
        {"item": "input_file", "value": input_path},
        {"item": "rows", "value": len(data)},
        {"item": "columns", "value": data.shape[1]},
        {"item": "countries", "value": data["country_code"].nunique()},
        {"item": "min_year", "value": data["year"].min()},
        {"item": "max_year", "value": data["year"].max()},
        {"item": "duplicated_country_year", "value": data.duplicated(["country_code", "year"]).sum()},
        {"item": "total_model_count", "value": int(data["model_count"].sum())},
        {"item": "zero_model_count_rows", "value": int((data["model_count"] == 0).sum())},
        {"item": "positive_model_count_rows", "value": int((data["model_count"] > 0).sum())},
        {"item": "zero_ratio", "value": float((data["model_count"] == 0).mean())}
    ])

    missing_report = data.isna().sum().reset_index()
    missing_report.columns = ["variable", "missing_count"]
    missing_report["missing_ratio"] = missing_report["missing_count"] / len(data)

    descriptive_cols = [
        "year",
        "model_count",
        "ai_policy_lag1",
        "policy_within",
        "mean_ai_policy_lag1",
        "log_gdp",
        "internet_users_pct",
        "regulatory_quality_filled",
        "regulatory_quality_was_filled"
    ]

    descriptive_stats = data[descriptive_cols].describe().T.reset_index().rename(columns={"index": "variable"})

    year_summary = data.groupby("year", as_index=False).agg(
        observations=("model_count", "size"),
        countries=("country_code", "nunique"),
        total_model_count=("model_count", "sum"),
        mean_model_count=("model_count", "mean"),
        zero_model_count_rows=("model_count", lambda x: int((x == 0).sum())),
        total_ai_policy_lag1=("ai_policy_lag1", "sum"),
        mean_ai_policy_lag1=("ai_policy_lag1", "mean"),
        mean_policy_within=("policy_within", "mean"),
        mean_log_gdp=("log_gdp", "mean"),
        mean_internet_users_pct=("internet_users_pct", "mean"),
        mean_regulatory_quality=("regulatory_quality_filled", "mean")
    )

    country_summary = data.groupby("country_code", as_index=False).agg(
        years=("year", "nunique"),
        total_model_count=("model_count", "sum"),
        mean_model_count=("model_count", "mean"),
        max_model_count=("model_count", "max"),
        total_ai_policy_lag1=("ai_policy_lag1", "sum"),
        mean_ai_policy_lag1=("ai_policy_lag1", "mean"),
        mean_log_gdp=("log_gdp", "mean"),
        mean_internet_users_pct=("internet_users_pct", "mean"),
        mean_regulatory_quality=("regulatory_quality_filled", "mean")
    ).sort_values(["total_model_count", "mean_ai_policy_lag1"], ascending=False)

    overdispersion = distribution_diagnostics(data)

    vif_cols = [
        "policy_within",
        "mean_ai_policy_lag1",
        "log_gdp",
        "internet_users_pct",
        "regulatory_quality_filled",
        "regulatory_quality_was_filled"
    ]

    vif_input = sm.add_constant(data[vif_cols], has_constant="add")

    vif_rows = []

    for i, col in enumerate(vif_input.columns):
        if col != "const":
            try:
                vif_value = variance_inflation_factor(vif_input.values, i)
            except Exception:
                vif_value = np.nan

            vif_rows.append({
                "variable": col,
                "vif": vif_value
            })

    vif = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
    correlation_matrix = data[["model_count"] + vif_cols].corr()

    return {
        "data_audit": data_audit,
        "missing_report": missing_report,
        "descriptive_stats": descriptive_stats,
        "year_summary": year_summary,
        "country_summary": country_summary,
        "overdispersion": overdispersion,
        "vif": vif,
        "correlation_matrix": correlation_matrix,
        "scaling_table": scaling_table
    }


def make_figures(data, reports):
    year_summary = reports["year_summary"]
    country_summary = reports["country_summary"]

    paths = {
        "yearly": os.path.join(figure_dir, "yearly_model_count_and_policy.png"),
        "distribution": os.path.join(figure_dir, "model_count_distribution.png"),
        "scatter": os.path.join(figure_dir, "policy_vs_model_count.png"),
        "top_countries": os.path.join(figure_dir, "top_countries_by_model_count.png"),
        "predicted": os.path.join(figure_dir, "observed_vs_predicted.png"),
        "calibration": os.path.join(figure_dir, "calibration_by_decile.png")
    }

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(year_summary["year"], year_summary["total_model_count"], marker="o")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total model count")

    ax2 = ax1.twinx()
    ax2.plot(year_summary["year"], year_summary["total_ai_policy_lag1"], marker="s")
    ax2.set_ylabel("Total AI policy lag1")

    fig.tight_layout()
    fig.savefig(paths["yearly"], dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data["model_count"], bins=40)
    ax.set_xlabel("Model count")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(paths["distribution"], dpi=300)
    plt.close(fig)

    rng = np.random.default_rng(random_seed)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        data["ai_policy_lag1"] + rng.normal(0, 0.08, len(data)),
        data["model_count"] + rng.normal(0, 0.08, len(data)),
        alpha=0.45
    )
    ax.set_xlabel("AI policy lag1")
    ax.set_ylabel("Model count")
    ax.set_yscale("symlog")
    fig.tight_layout()
    fig.savefig(paths["scatter"], dpi=300)
    plt.close(fig)

    top = country_summary.head(15).sort_values("total_model_count", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["country_code"], top["total_model_count"])
    ax.set_xlabel("Total model count")
    ax.set_ylabel("Country code")
    fig.tight_layout()
    fig.savefig(paths["top_countries"], dpi=300)
    plt.close(fig)

    return paths


def run_model(data):
    import pymc as pm
    import arviz as az

    predictors = [
        "z_policy_within",
        "z_mean_ai_policy_lag1",
        "z_log_gdp",
        "z_internet_users_pct",
        "z_regulatory_quality_filled",
        "regulatory_quality_was_filled"
    ]

    country_categories = pd.Categorical(data["country_code"])
    year_categories = pd.Categorical(data["year"])

    country_index = country_categories.codes
    year_index = year_categories.codes

    country_labels = list(country_categories.categories)
    year_labels = [str(x) for x in list(year_categories.categories)]

    x_matrix = data[predictors].to_numpy()
    y = data["model_count"].to_numpy()

    coords = {
        "observation": np.arange(len(data)),
        "country": country_labels,
        "year": year_labels,
        "coefficient": predictors
    }

    with pm.Model(coords=coords) as model:
        x_data = pm.Data("x_data", x_matrix, dims=("observation", "coefficient"))
        country_data = pm.Data("country_index", country_index, dims="observation")
        year_data = pm.Data("year_index", year_index, dims="observation")

        intercept = pm.Normal("intercept", mu=0, sigma=3)
        beta = pm.Normal("beta", mu=0, sigma=1.5, dims="coefficient")

        sigma_country = pm.HalfNormal("sigma_country", sigma=1)
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
            idata_kwargs={"log_likelihood": True},
            progressbar=False
        )

        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["model_count"],
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False
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

    main_summary = to_numeric_table(main_summary)
    beta_summary = to_numeric_table(beta_summary)

    irr = build_irr_table(beta_summary)

    beta_posterior = idata.posterior["beta"]
    beta_means = beta_posterior.mean(dim=("chain", "draw")).values
    beta_positive = (beta_posterior > 0).mean(dim=("chain", "draw")).values
    beta_negative = (beta_posterior < 0).mean(dim=("chain", "draw")).values
    beta_irr_above_1 = (np.exp(beta_posterior) > 1).mean(dim=("chain", "draw")).values

    posterior_effects = pd.DataFrame({
        "coefficient": predictors,
        "posterior_mean_beta": beta_means,
        "posterior_probability_beta_positive": beta_positive,
        "posterior_probability_beta_negative": beta_negative,
        "posterior_probability_irr_above_1": beta_irr_above_1,
        "posterior_mean_irr": np.exp(beta_means)
    })

    try:
        loo = az.loo(idata)
        loo_elpd = safe_float(loo.elpd_loo)
        loo_se = safe_float(loo.se)
        loo_p = safe_float(loo.p_loo)
    except Exception:
        loo_elpd = np.nan
        loo_se = np.nan
        loo_p = np.nan

    try:
        waic = az.waic(idata)
        waic_elpd = safe_float(waic.elpd_waic)
        waic_se = safe_float(waic.se)
        waic_p = safe_float(waic.p_waic)
    except Exception:
        waic_elpd = np.nan
        waic_se = np.nan
        waic_p = np.nan

    max_rhat = main_summary["r_hat"].dropna().max() if "r_hat" in main_summary.columns else np.nan
    min_ess_bulk = main_summary["ess_bulk"].dropna().min() if "ess_bulk" in main_summary.columns else np.nan

    diagnostics = pd.DataFrame([
        {"metric": "status", "value": "success"},
        {"metric": "draws", "value": draws},
        {"metric": "tune", "value": tune},
        {"metric": "chains", "value": chains},
        {"metric": "cores", "value": cores},
        {"metric": "target_accept", "value": target_accept},
        {"metric": "max_rhat_main_parameters", "value": max_rhat},
        {"metric": "min_ess_bulk_main_parameters", "value": min_ess_bulk},
        {"metric": "loo_elpd", "value": loo_elpd},
        {"metric": "loo_se", "value": loo_se},
        {"metric": "loo_p", "value": loo_p},
        {"metric": "waic_elpd", "value": waic_elpd},
        {"metric": "waic_se", "value": waic_se},
        {"metric": "waic_p", "value": waic_p}
    ])

    posterior_y = (
        ppc.posterior_predictive["model_count"]
        .stack(sample=("chain", "draw"))
        .transpose("observation", "sample")
        .values
    )

    predicted_mean = posterior_y.mean(axis=1)
    predicted_low = np.quantile(posterior_y, 0.025, axis=1)
    predicted_high = np.quantile(posterior_y, 0.975, axis=1)

    predictions = data[["country_code", "year", "model_count", "ai_policy_lag1", "policy_within"]].copy()
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
        {"metric": "observed_max", "value": observed_y.max()},
        {"metric": "predicted_max_mean", "value": posterior_y.max(axis=0).mean()}
    ])

    calibration = predictions.copy()
    calibration["prediction_bin"] = pd.qcut(
        calibration["predicted_mean"].rank(method="first"),
        q=10,
        labels=False
    ) + 1

    calibration_summary = calibration.groupby("prediction_bin", as_index=False).agg(
        n=("model_count", "size"),
        observed_mean=("model_count", "mean"),
        predicted_mean=("predicted_mean", "mean"),
        observed_total=("model_count", "sum"),
        predicted_total=("predicted_mean", "sum")
    )

    largest_residuals = predictions.sort_values("abs_residual", ascending=False).head(30)

    country_effects = az.summary(
        idata,
        var_names=["country_effect"],
        round_to=None
    ).reset_index().rename(columns={"index": "term"})

    country_effects = to_numeric_table(country_effects)
    country_effects["country_code"] = country_effects["term"].astype(str).str.extract(r"\[(.*)\]")

    year_effects = az.summary(
        idata,
        var_names=["year_effect"],
        round_to=None
    ).reset_index().rename(columns={"index": "term"})

    year_effects = to_numeric_table(year_effects)
    year_effects["year"] = year_effects["term"].astype(str).str.extract(r"\[(.*)\]")

    return {
        "main_summary": main_summary,
        "irr": irr,
        "posterior_effects": posterior_effects,
        "diagnostics": diagnostics,
        "predictions": predictions,
        "posterior_predictive_check": ppc_summary,
        "calibration_summary": calibration_summary,
        "largest_residuals": largest_residuals,
        "country_effects": country_effects,
        "year_effects": year_effects
    }


def add_model_figures(model_outputs, figure_paths):
    predictions = model_outputs.get("predictions", pd.DataFrame())
    calibration = model_outputs.get("calibration_summary", pd.DataFrame())

    if not predictions.empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(predictions["predicted_mean"], predictions["model_count"], alpha=0.5)
        max_value = max(predictions["predicted_mean"].max(), predictions["model_count"].max())
        ax.plot([0, max_value], [0, max_value])
        ax.set_xlabel("Predicted mean")
        ax.set_ylabel("Observed model count")
        fig.tight_layout()
        fig.savefig(figure_paths["predicted"], dpi=300)
        plt.close(fig)

    if not calibration.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(calibration["prediction_bin"], calibration["observed_mean"], marker="o")
        ax.plot(calibration["prediction_bin"], calibration["predicted_mean"], marker="s")
        ax.set_xlabel("Prediction decile")
        ax.set_ylabel("Mean model count")
        fig.tight_layout()
        fig.savefig(figure_paths["calibration"], dpi=300)
        plt.close(fig)


def export_workbook(data, reports, model_outputs, figure_paths):
    model_notes = pd.DataFrame([
        {"section": "Dependent variable", "content": "model_count"},
        {"section": "Within-country policy variable", "content": "policy_within = ai_policy_lag1 - mean_ai_policy_lag1"},
        {"section": "Between-country policy variable", "content": "mean_ai_policy_lag1"},
        {"section": "Model", "content": "Bayesian Correlated Random-Effects Negative Binomial Mixed Model"},
        {"section": "Country effect", "content": "Random intercept by country_code"},
        {"section": "Year effect", "content": "Regularized year effect"},
        {"section": "Controls", "content": "log_gdp, internet_users_pct, regulatory_quality_filled, regulatory_quality_was_filled"},
        {"section": "Distribution diagnostics", "content": "Poisson and method-of-moments Negative Binomial zero-rate diagnostics are reported before model fitting."},
        {"section": "Posterior inference", "content": "Use posterior_effects and incidence_rate_ratio for effect direction and practical magnitude."},
        {"section": "Model evaluation", "content": "Use posterior_predictive_check, calibration_summary, largest_residuals, WAIC/LOO when available."}
    ])

    manifest = pd.DataFrame([
        {"file_type": "workbook", "path": workbook_path},
        {"file_type": "figure", "path": figure_paths["yearly"]},
        {"file_type": "figure", "path": figure_paths["distribution"]},
        {"file_type": "figure", "path": figure_paths["scatter"]},
        {"file_type": "figure", "path": figure_paths["top_countries"]},
        {"file_type": "figure", "path": figure_paths["predicted"] if os.path.exists(figure_paths["predicted"]) else ""},
        {"file_type": "figure", "path": figure_paths["calibration"] if os.path.exists(figure_paths["calibration"]) else ""}
    ])

    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        data.to_excel(writer, sheet_name="model_input", index=False)
        reports["data_audit"].to_excel(writer, sheet_name="data_audit", index=False)
        reports["missing_report"].to_excel(writer, sheet_name="missing_report", index=False)
        reports["descriptive_stats"].to_excel(writer, sheet_name="descriptive_stats", index=False)
        reports["year_summary"].to_excel(writer, sheet_name="year_summary", index=False)
        reports["country_summary"].to_excel(writer, sheet_name="country_summary", index=False)
        reports["overdispersion"].to_excel(writer, sheet_name="distribution_diagnostics", index=False)
        reports["vif"].to_excel(writer, sheet_name="vif", index=False)
        reports["correlation_matrix"].to_excel(writer, sheet_name="correlation_matrix")
        reports["scaling_table"].to_excel(writer, sheet_name="scaling_table", index=False)
        model_outputs.get("main_summary", pd.DataFrame()).to_excel(writer, sheet_name="bayesian_summary", index=False)
        model_outputs.get("irr", pd.DataFrame()).to_excel(writer, sheet_name="incidence_rate_ratio", index=False)
        model_outputs.get("posterior_effects", pd.DataFrame()).to_excel(writer, sheet_name="posterior_effects", index=False)
        model_outputs.get("diagnostics", pd.DataFrame()).to_excel(writer, sheet_name="model_diagnostics", index=False)
        model_outputs.get("posterior_predictive_check", pd.DataFrame()).to_excel(writer, sheet_name="posterior_predictive_check", index=False)
        model_outputs.get("calibration_summary", pd.DataFrame()).to_excel(writer, sheet_name="calibration_summary", index=False)
        model_outputs.get("largest_residuals", pd.DataFrame()).to_excel(writer, sheet_name="largest_residuals", index=False)
        model_outputs.get("predictions", pd.DataFrame()).to_excel(writer, sheet_name="posterior_predictions", index=False)
        model_outputs.get("country_effects", pd.DataFrame()).to_excel(writer, sheet_name="country_effects", index=False)
        model_outputs.get("year_effects", pd.DataFrame()).to_excel(writer, sheet_name="year_effects", index=False)
        model_notes.to_excel(writer, sheet_name="model_notes", index=False)
        manifest.to_excel(writer, sheet_name="file_manifest", index=False)

        workbook = writer.book
        header_format = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        text_format = workbook.add_format({"text_wrap": True})
        num_format = workbook.add_format({"num_format": "0.0000"})

        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes(1, 0)
            worksheet.set_column(0, 0, 34, text_format)
            worksheet.set_column(1, 50, 18, num_format)

        figures_sheet = workbook.add_worksheet("figures")
        figures_sheet.write(0, 0, "Generated figures", header_format)
        figures_sheet.insert_image(2, 0, figure_paths["yearly"], {"x_scale": 0.72, "y_scale": 0.72})
        figures_sheet.insert_image(25, 0, figure_paths["distribution"], {"x_scale": 0.72, "y_scale": 0.72})
        figures_sheet.insert_image(48, 0, figure_paths["scatter"], {"x_scale": 0.72, "y_scale": 0.72})
        figures_sheet.insert_image(71, 0, figure_paths["top_countries"], {"x_scale": 0.72, "y_scale": 0.72})

        if os.path.exists(figure_paths["predicted"]):
            figures_sheet.insert_image(94, 0, figure_paths["predicted"], {"x_scale": 0.72, "y_scale": 0.72})

        if os.path.exists(figure_paths["calibration"]):
            figures_sheet.insert_image(117, 0, figure_paths["calibration"], {"x_scale": 0.72, "y_scale": 0.72})


def main():
    step("Reading and cleaning data")
    data = clean_data(input_path)

    step("Creating within-country policy variable and standardized predictors")
    data, scaling_table = standardize_data(data)

    step("Building diagnostics: distribution, missing, VIF, correlation and summaries")
    reports = make_reports(data, scaling_table)

    step("Creating descriptive figures")
    figure_paths = make_figures(data, reports)

    step("Running Bayesian CRE Negative Binomial Mixed Model")
    try:
        model_outputs = run_model(data)
        step("Model finished successfully")
    except Exception as error:
        model_outputs = {
            "diagnostics": pd.DataFrame([
                {"metric": "status", "value": "failed"},
                {"metric": "error", "value": str(error)}
            ])
        }
        step(f"Model failed: {error}")

    step("Creating model evaluation figures")
    add_model_figures(model_outputs, figure_paths)

    step("Exporting final workbook")
    export_workbook(data, reports, model_outputs, figure_paths)

    step(f"Done. Output saved to: {workbook_path}")


if __name__ == "__main__":
    main()