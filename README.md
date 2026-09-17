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

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit dashboard - run this |
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
   - On the repo page, click **Add file -> Upload files**, drag in `app.py`,
     `data_fetch.py`, `model.py`, and `requirements.txt`, then commit.
   - Make sure each file keeps its `.py` / `.txt` extension and is created as
     a file, not a folder (the same gotcha you hit on the macro dashboard
     repo).
3. **Deploy on Streamlit Community Cloud.**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
     GitHub.
   - Click **New app**, pick this repo, branch `main`, main file `app.py`.
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

## If the Bank of England fetch ever breaks

The app calls this endpoint under the hood:

```
https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp
```

If the app shows a data error (the BoE occasionally changes this export's
format), open [the interactive database](https://www.bankofengland.co.uk/boeapps/database/)
yourself, search for series `IUDBEDR`, `IUDSNPY`, `IUDMNPY`, `IUDLNPY`, and
check whether the CSV export still looks like `date, value, value, value, value`
in that order. Send me the new format and I'll patch `data_fetch.py`.

## Possible next steps

- Add more maturities (1y/2y/30y) if you find better free BoE series codes
  for them, for a richer curve shape.
- Swap the RandomForest for a simple logistic regression to compare
  interpretability vs accuracy.
- Add a "why" panel that shows the top 2-3 feature values behind each
  individual prediction (not just global feature importance).
- Cross-check this against US Treasury shape (FRED, also free) for a
  cross-market comparison.
