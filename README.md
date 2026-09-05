# Car Price Prediction

Predicting used car prices on the Azerbaijani market from mileage and production year, using linear regression implemented **from scratch** and compared against scikit-learn.

The point of this project is not the accuracy number — it's the implementation. Gradient descent, the cost function, feature normalization, and the Normal Equation are all written out in NumPy so the math behind `LinearRegression().fit()` is visible rather than hidden.

---

## Dataset

`turboaz.csv` — 1,328 Mercedes-Benz listings scraped from [turbo.az](https://turbo.az), Azerbaijan's largest car marketplace.

| | |
|---|---|
| Rows | 1,328 |
| Columns | 16 (raw) |
| Brand | Mercedes-Benz only |
| Production years | 1989 – 2018 |
| Cities | 51 |

Column names are in Azerbaijani:

| Column | Meaning |
|---|---|
| `Sheher` | City |
| `Marka` / `Model` | Make / model |
| `Buraxilish ili` | Production year |
| `Ban novu` | Body type |
| `Reng` | Colour |
| `Muherrikin hecmi` / `Muherrikin gucu` | Engine volume / horsepower |
| `Yanacaq novu` | Fuel type |
| `Yurush` | Mileage |
| `Suretler qutusu` / `Oturucu` | Transmission / drivetrain |
| `Yeni` | New or used |
| `Qiymet` | **Price (target)** |
| `Extra Info`, `Seller comment` | Free text |

Three columns are used for modelling: `Yurush`, `Buraxilish ili` → `Qiymet`.

### Cleaning

Both numeric columns arrive as strings and need parsing:

- **`Yurush`** — `"366 000 km"` → `366000`
- **`Qiymet`** — listed in either AZN or USD. Dollar prices are converted to manat at a fixed rate of **1.70**, so the target is a single currency: `"31500 $"` → `53550`

---

## Method

### 1. Feature normalization

Mileage is in the hundreds of thousands, year is around 2000. Left unscaled, gradient descent crawls. Each feature is standardized:

```
x_norm = (x - mu) / sigma
```

with `sigma` computed at `ddof=1` (sample standard deviation).

```
mu    = [279649.92, 1999.87]
sigma = [120619.61,    5.33]
```

A column of ones is then prepended to the feature matrix so the bias term `theta[0]` falls out of the same dot product.

### 2. Cost function

Mean squared error, halved so the derivative comes out clean:

```
J(theta) = (1 / 2m) * sum( (h(x) - y)^2 )
```

Implemented as `errors.T @ errors` rather than `np.square().sum()` — same result, one operation.

### 3. Batch gradient descent

Full-batch updates over all 1,328 examples per step, with the cost recorded at each iteration so convergence can be plotted.

| Setting | Value |
|---|---|
| Learning rate `alpha` | 0.001 |
| Iterations | 10,000 |
| Initial `theta` | `[0, 0, 0]` |

Converged parameters:

```
theta = [15115.77, -1343.02, 11272.84]
```

Read on normalized features: each standard deviation of mileage costs about **1,343 AZN**, each standard deviation of production year adds about **11,273 AZN**. Year dominates mileage by roughly 8×.

Cost drops from `2.07e8` to `1.98e7` over the run.

### 4. Learning rate study

The same descent is re-run at `alpha` = 0.005, 0.01, 0.02, 0.03, and 0.15 over 400 iterations and plotted on shared axes to show the convergence/speed trade-off. A final run at `alpha = 1.32` demonstrates divergence — the cost climbs instead of falling.

### 5. Normal Equation

The closed-form solution, for comparison:

```
theta = (X^T X)^-1 X^T y
```

No feature scaling, no iteration, no learning rate — one matrix inversion and the exact optimum. Applied to the raw (unnormalized) features.

### 6. scikit-learn baseline

`LinearRegression` on an 80/20 split (`random_state=101`), to check the from-scratch implementation against a reference.

### 7. Polynomial regression

Degree-2 `PolynomialFeatures` fed into the same linear model, to test whether the price surface is genuinely curved.

---

## Results

**scikit-learn linear regression** (held-out test set, 266 samples):

| Metric | Value |
|---|---|
| R² | 0.747 |
| Adjusted R² | 0.745 |
| MAE | 4,222 AZN |
| MSE | 40,242,455 |
| RMSE | 6,344 AZN |

Learned coefficients: intercept `-4,274,982`, slope `[-0.0102, 2146.68]` — roughly **2,147 AZN of value per model year**, and about **1 AZN lost per 100 km driven**.

**Polynomial regression (degree 2):** R² = **0.922**, measured on the full dataset rather than a held-out split, so it is not directly comparable to the 0.747 above and will be optimistic. Two spot checks:

| Mileage | Year | Actual | Predicted | Error |
|---|---|---|---|---|
| 240,000 | 2000 | 11,500 | 11,481 | 19 |
| 415,558 | 1996 | 8,800 | 9,611 | 811 |

The jump from linear to polynomial is real — depreciation isn't a straight line, it's steep early and flattens out — but the honest linear number is the 0.747.

---

## Visualizations

The notebook produces:

- Scatter plots of mileage vs price and year vs price
- A 3D scatter of all three variables
- Cost vs iteration, showing gradient descent converging
- Cost curves for five learning rates overlaid, plus the diverging case
- Fitted regression lines drawn back over the raw scatter plots
- 3D actual-vs-predicted overlay
- Bar charts comparing actual and predicted prices for the first 50 records, per model

---

## Running it

```bash
git clone https://github.com/RovshanBayramRB/Car-Price-Prediction.git
cd Car-Price-Prediction
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook "Car Price Prediction.ipynb"
```

Run the cells top to bottom — later cells depend on `mu`, `sigma`, and `MainThetas` from earlier ones.

> **Note:** the 3D plots use `Axes3D(figure)`, which was removed in Matplotlib 3.5+. On a newer version, change those two lines to `figure.add_subplot(projection='3d')`.

---

## Repository structure

```
.
├── Car Price Prediction.ipynb   # Full analysis: cleaning → GD → Normal Equation → sklearn → polynomial
├── turboaz.csv                  # 1,328 scraped Mercedes listings
└── README.md
```

---

## Limitations

- **Two features only.** Engine volume, transmission, body type, and city are all in the CSV and all unused. Price would be predicted much better with them.
- **One brand.** Every row is a Mercedes, so nothing here generalizes to the wider market.
- **Fixed exchange rate.** USD prices are converted at a hard-coded 1.70, which ignores when the listing was posted.
- **No outlier handling.** Listings are taken as-is, including obvious data-entry errors in the mileage field.
- **Polynomial R² is in-sample.** It is fitted and scored on the same 1,328 rows.

## License

No license specified.
