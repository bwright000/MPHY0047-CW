from dataloader import NUM_VIEWS, MISSING, gen_impr, crit_perc
from question2 import compute_similarity_metrics
from plot_style import apply_style, scatter_points, reference_line, finish_figure, BLUE, ORANGE
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os

apply_style()
VIEW_NAMES = [f"View {i+1}" for i in range(NUM_VIEWS)]

ssi_vals, mi_vals, cs_vals = compute_similarity_metrics()

def compute_metric_correlations(ssi_vals, mi_vals, cs_vals):
    """
    Calculate Pearson correlation between each pair of similarity metrics
    (SSI-MI, SSI-CS, MI-CS) for each view.
    
    Returns:
        dict mapping pair name -> list of (r, p_value) tuples, one per view
    """
    pairs = [("SSI", "MI",
              ssi_vals, mi_vals),
             ("SSI", "CS",
              ssi_vals, cs_vals),
             ("MI", "CS",
              mi_vals, cs_vals)]
    correlations = {}
    for name1, name2, vals1, vals2 in pairs:
        pair_name = f"{name1}-{name2}"
        correlations[pair_name] = []
        for v in range(NUM_VIEWS):
            # Extract valid (non-NaN) values for this view
            x = vals1[:, v]
            y = vals2[:, v]
            valid_idx = ~np.isnan(x) & ~np.isnan(y)
            if np.sum(valid_idx) < 2:
                # Not enough data to compute correlation
                correlations[pair_name].append((np.nan, np.nan))
                continue
            r, p_value = stats.pearsonr(x[valid_idx], y[valid_idx])
            correlations[pair_name].append((r, p_value))
    return correlations

def polynomial_regression_lasso(x_vals, y_vals, metric_name, score_name, max_degree=7):
    """
    Polynomial regression with LASSO regularization for each view.
    
    Args:
        x_vals: 20×10 array of independent variable (SSI/MI/CS)
        y_vals: 20×10 array of dependent variable (crit_perc/gen_impr)
        metric_name: str, e.g. "SSI"
        score_name: str, e.g. "crit_perc"
        max_degree: maximum polynomial degree (default 7)
    
    Returns:
        list of dicts, one per view, with keys: view, rmse, r2, model, optimal_degree, predicted, y_true
    """
    results = []
    for v in range(NUM_VIEWS):
        x = x_vals[:, v]
        y = y_vals[:, v]
        valid_idx = ~np.isnan(x) & ~np.isnan(y) & (y != -1)
        if np.sum(valid_idx) < 2:
            results.append({
                "view": VIEW_NAMES[v],
                "rmse": np.nan,
                "r2": np.nan,
                "model": None,
                "optimal_degree": 0,
                "predicted": np.array([np.nan]*len(y)),
                "y_true": y
            })
            continue
        x_valid = x[valid_idx].reshape(-1, 1)
        y_valid = y[valid_idx]
        best_cv_rmse = np.inf
        best_model = None
        best_degree = 0
        best_poly = None

        for degree in range(1, max_degree + 1):
            poly = PolynomialFeatures(degree, include_bias=False)
            x_poly = poly.fit_transform(x_valid)
            model = LassoCV(cv=5, max_iter=10000, random_state=0)
            model.fit(x_poly, y_valid)

            # Use LassoCV's cross-validated MSE at optimal alpha for degree selection
            alpha_idx = np.argmin(np.abs(model.alphas_ - model.alpha_))
            cv_rmse = np.sqrt(model.mse_path_[alpha_idx].mean())

            if cv_rmse < best_cv_rmse:
                best_cv_rmse = cv_rmse
                best_model = model
                best_degree = degree
                best_poly = poly

        # Refit best degree on full data and apply coefficient threshold
        x_poly = best_poly.fit_transform(x_valid)
        best_model.fit(x_poly, y_valid)
        best_model.coef_[np.abs(best_model.coef_) < 0.01] = 0
        best_predicted = best_model.predict(x_poly)
        best_rmse = np.sqrt(mean_squared_error(y_valid, best_predicted))
        best_r2 = r2_score(y_valid, best_predicted)

        results.append({
            "view": VIEW_NAMES[v],
            "rmse": best_rmse,
            "r2": best_r2,
            "model": best_model,
            "optimal_degree": best_degree,
            "predicted": best_predicted,
            "y_true": y_valid                                          
        })

    return results

def gaussian_basis_matrix(x, n_bases, sigma=None):
    """
    Construct a Gaussian basis function matrix for input x.
    
    Args:
        x: 1D array of input values
        n_bases: number of Gaussian basis functions
        sigma: width of the Gaussians (if None, set to spacing between centers)
    Returns:
        2D array of shape (len(x), n_bases) with Gaussian basis function values
    """ 
    mu = np.linspace(x.min(), x.max(), n_bases)
    if sigma is None:
        sigma = (x.max() - x.min()) / n_bases
    return np.column_stack([np.exp(-((x - m) ** 2) / (2 * sigma ** 2)) for m in mu])


def gaussian_basis_regression(ssi_vals, gi_vals, max_order=10):
    """
    Gaussian basis linear regression: SSI (independent) -> gen_impr (dependent).
    Tries orders 2 through max_order, selects best via LASSO regularization.

    Args:
        ssi_vals: 20×10 array of SSI values
        gi_vals: 20×10 array of general impression scores
        max_order: maximum number of Gaussian basis functions to try

    Returns:
        list of dicts, one per view, with keys: view, rmse, r2, optimal_order, predicted, y_true
    """
    results = []

    for v in range(NUM_VIEWS):
        x = ssi_vals[:, v]
        y = gi_vals[:, v]
        valid = ~(np.isnan(x) | np.isnan(y)) & (y != -1)
        x_valid = x[valid]
        y_valid = y[valid]

        best_cv_rmse = np.inf
        best_order = 2
        best_model = None

        for order in range(2, max_order + 1):
            # Build Gaussian basis feature matrix
            X_basis = gaussian_basis_matrix(x_valid, order)

            # Fit LASSO with cross-validation
            model = LassoCV(cv=5, max_iter=10000, random_state=0)
            model.fit(X_basis, y_valid)

            # Use LassoCV's cross-validated MSE at optimal alpha for order selection
            alpha_idx = list(model.alphas_).index(model.alpha_)
            cv_rmse = np.sqrt(model.mse_path_[alpha_idx].mean())

            if cv_rmse < best_cv_rmse:
                best_cv_rmse = cv_rmse
                best_order = order
                best_model = model

        # Refit best order on full data
        X_basis = gaussian_basis_matrix(x_valid, best_order)
        best_model.fit(X_basis, y_valid)
        best_predicted = best_model.predict(X_basis)
        best_rmse = np.sqrt(mean_squared_error(y_valid, best_predicted))
        best_r2 = r2_score(y_valid, best_predicted)

        results.append({
            "view": VIEW_NAMES[v],
            "rmse": best_rmse,
            "r2": best_r2,
            "optimal_order": best_order,
            "predicted": best_predicted,
            "y_true": y_valid
        })

    return results

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    # Part i: Correlation between similarity metric pairs.
    print("PART i: Correlation Between Similarity Metrics")

    correlations = compute_metric_correlations(ssi_vals, mi_vals, cs_vals)

    print(f"\n{'View':<10} {'SSI-MI r':<12} {'SSI-CS r':<12} {'MI-CS r':<12}")
    for v in range(NUM_VIEWS):
        r_ssi_mi = correlations["SSI-MI"][v][0]
        r_ssi_cs = correlations["SSI-CS"][v][0]
        r_mi_cs = correlations["MI-CS"][v][0]
        print(f"{VIEW_NAMES[v]:<10} {r_ssi_mi:<12.4f} {r_ssi_cs:<12.4f} {r_mi_cs:<12.4f}")

    # Identify best view (highest |r|) for each pair
    for pair_name, results in correlations.items():
        best_v = max(range(NUM_VIEWS), key=lambda v: abs(results[v][0]))
        print(f"\nHighest agreement for {pair_name}: {VIEW_NAMES[best_v]} "
              f"(r = {results[best_v][0]:.4f})")

    # Part ii: Polynomial Regression with LASSO.
    print("\nPART ii: Polynomial Regression with LASSO Regularization")

    metrics = [("SSI", ssi_vals), ("MI", mi_vals), ("CS", cs_vals)]
    scores = [("crit_perc", crit_perc), ("gen_impr", gen_impr)]

    all_poly_results = {}

    for metric_name, metric_vals in metrics:
        for score_name, score_vals in scores:
            combo = f"{metric_name} -> {score_name}"
            print(f"\n{combo}")
            print(f"{'View':<10} {'RMSE':<10} {'R²':<10} {'Degree':<10}")

            results = polynomial_regression_lasso(metric_vals, score_vals,
                                                   metric_name, score_name)
            all_poly_results[combo] = results

            for r in results:
                print(f"{r['view']:<10} {r['rmse']:<10.4f} {r['r2']:<10.4f} "
                      f"{r['optimal_degree']:<10}")

            # Plot top 3 views by R²
            ranked = sorted(results, key=lambda r: r['r2'], reverse=True)
            for r in ranked[:3]:
                fig, ax = plt.subplots()
                scatter_points(ax, r['y_true'], r['predicted'], color=BLUE)
                min_val = min(r['y_true'].min(), r['predicted'].min())
                max_val = max(r['y_true'].max(), r['predicted'].max())
                reference_line(ax, min_val, max_val)
                ax.set_title(f"{r['view']} \u2014 {combo}\n"
                             f"RMSE = {r['rmse']:.2f}, R\u00b2 = {r['r2']:.4f}, "
                             f"Degree = {r['optimal_degree']}")
                ax.set_xlabel(f'True {score_name}')
                ax.set_ylabel(f'Estimated {score_name}')
                ax.legend()
                fname = f"figures/q3ii_{metric_name}_{score_name}_{r['view'].lower().replace(' ', '_')}.png"
                finish_figure(fig, fname)

    # Part iii: Gaussian Basis Regression (SSI -> gen_impr).
    print("PART iii: Gaussian Basis Regression (SSI -> gen_impr)")

    gb_results = gaussian_basis_regression(ssi_vals, gen_impr)

    print(f"\n{'View':<10} {'RMSE':<10} {'R²':<10} {'Order':<10}")
    for r in gb_results:
        print(f"{r['view']:<10} {r['rmse']:<10.4f} {r['r2']:<10.4f} "
              f"{r['optimal_order']:<10}")

    # Plot top 3 views by R²
    ranked = sorted(gb_results, key=lambda r: r['r2'], reverse=True)
    for r in ranked[:3]:
        fig, ax = plt.subplots()
        scatter_points(ax, r['y_true'], r['predicted'], color=ORANGE)
        min_val = min(r['y_true'].min(), r['predicted'].min())
        max_val = max(r['y_true'].max(), r['predicted'].max())
        reference_line(ax, min_val, max_val)
        ax.set_title(f"{r['view']} \u2014 Gaussian Basis (SSI \u2192 gen_impr)\n"
                     f"RMSE = {r['rmse']:.2f}, R\u00b2 = {r['r2']:.4f}, "
                     f"Order = {r['optimal_order']}")
        ax.set_xlabel('True gen_impr')
        ax.set_ylabel('Estimated gen_impr')
        ax.legend()
        fname = f"figures/q3iii_gaussian_{r['view'].lower().replace(' ', '_')}.png"
        finish_figure(fig, fname)