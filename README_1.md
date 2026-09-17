# UK Yield Curve Shape Forecaster

A free, Python/Streamlit app that predicts whether the UK gilt yield curve
will be **Normal**, **Flat**, or **Inverted** N business days from now.

- **Data**: Bank of England IADB - daily Bank Rate + 5y/10y/20y nominal par
  gilt yields. Free, no signup, no API key.
- **Model**: scikit-learn `RandomForestClassifier`, trained fresh inside the
  app itself (cached) on curve levels, spreads, and spread momentum. Compared
  against a "today's shape = tomorrow's shape" persistence baseline in an
  out-of-sample backtest, so you can see whether it's actually adding value.
- **Stack**: Streamlit + Plotly, same as your other two dashboards.
- **Trading signal backtest**: turns the shape call into an illustrative
  curve-steepener position and marks it to market day by day over the
  out-of-sample period, next to a persistence-driven version and a passive
  always-long-steepener benchmark. Shows whether the model's signal actually
  has edge, not just whether it classifies well.

## Files

| File | Purpose |
|---|---|
| `appyield.py` | The Streamlit dashboard - run this |
| `data_fetch.py` | Pulls yield data from the Bank of England |
| `model.py` | Feature engineering, labelling, training, backtesting |
| `requirements.txt` | Python dependencies |

## Deploy (GitHub -> Streamlit Community Cloud, no local install needed)

1. **Create a GitHub repo.**
   - Go to [github.com/new](https://github.com/new), name it e.g.
     `yield-curve-shape-forecaster`, keep it public (Streamlit Community
     Cloud's free tier needs a public repo, or a private one it can access
     if you connect your GitHub account with access).
2. **Upload the four files.**
   - On the repo page, click **Add file -> Upload files**, drag in `appyield.py`,
     `data_fetch.py`, `model.py`, and `requirements.txt`, then commit.
   - Make sure each file keeps its `.py` / `.txt` extension and is created as
     a file, not a folder (the same gotcha you hit on the macro dashboard
     repo).
3. **Deploy on Streamlit Community Cloud.**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
     GitHub.
   - Click **New app**, pick this repo, branch `main`, main file `appyield.py`.
   - Click **Deploy**. No secrets to configure - this app needs no API keys.
4. **Wait ~1-2 minutes** for the first build, then open the app URL.

## Using it

- The **sidebar** lets you change the forecast horizon (5-60 business days
  ahead) and how wide the "Flat" band around zero slope is.
- The **headline row** shows today's actual curve shape, the model's
  prediction for N days out, and how the model's backtest accuracy compares
  to just guessing "no change."
- If the model's backtest accuracy is barely better than the persistence
  baseline, that's a genuine and useful finding, not a bug - it means the
  curve mostly drifts and short-horizon shape changes are hard to call, which
  is exactly what you'd expect from an efficient bond market. Try a longer
  horizon (40-60 days) if you want to see the model earn more of its keep.
- The **trading signal backtest** section further down turns the same calls
  into a position and shows cumulative P&L (in spread percentage points,
  notional unit size - not real money) for the model vs a persistence-driven
  version vs always being long the steepener. Read this as "does the signal
  have edge," not as a projected return - there's no transaction costs or
  duration weighting in it. A Sharpe ratio meaningfully above the other two
  is the interesting result; a Sharpe near zero or negative means the model
  isn't earning its keep as a trading signal even if its classification
  accuracy looks fine.

## If the Bank of England fetch ever breaks

The app calls this endpoint under the hood:

```
https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp
```

**403 Forbidden**: the BoE blocks requests that don't look like a browser.
`data_fetch.py` already sends browser-like headers (User-Agent, Referer, etc.)
to get around this - if it comes back after a redeploy, the site has likely
tightened its bot-blocking again and those headers need refreshing. Send me
the error and I'll update them.

**"Could not parse the response as CSV" / a tokenizing error**: the BoE's
export puts a few metadata lines (title, source, notes) before the actual
data table, and sometimes trailing footnotes after it. `data_fetch.py`
handles this by scanning for the first run of consistently-5-column rows and
using only those - it doesn't assume a fixed number of lines to skip. If this
still errors, it means it couldn't find any run of 5-column rows at all,
which means the shape of the whole export changed, not just the preamble.

**Any other data error**: open [the interactive database](https://www.bankofengland.co.uk/boeapps/database/)
yourself, search for series `IUDBEDR`, `IUDSNPY`, `IUDMNPY`, `IUDLNPY`, and
check what the CSV export looks like now. Send me the new format (or just
paste the raw error) and I'll patch `data_fetch.py`.

## Possible next steps

- Add more maturities (1y/2y/30y) if you find better free BoE series codes
  for them, for a richer curve shape.
- Swap the RandomForest for a simple logistic regression to compare
  interpretability vs accuracy.
- Add a "why" panel that shows the top 2-3 feature values behind each
  individual prediction (not just global feature importance).
- Cross-check this against US Treasury shape (FRED, also free) for a
  cross-market comparison.
- Make the trading backtest more realistic: weight positions by duration/DV01
  instead of unit notional, add a simple transaction cost per trade, and
  only re-enter a position when the signal actually changes (right now it
  re-marks daily regardless of whether yesterday's call was the same).
- Log each day's live prediction somewhere persistent (a committed CSV via a
  scheduled GitHub Action, or a free key-value store like the JSONBin setup
  in your trading app) to build a real forward-testing track record over
  time, instead of only a historical backtest.
