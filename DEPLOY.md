# Deploying to Streamlit Community Cloud

The app reads only committed files (`data/processed/`, `models/`, `reports/`), so no secrets, database or build step is needed.

1. **Push the repo to GitHub** (public, or private with Streamlit access granted):
   ```bash
   git remote add origin https://github.com/<your-user>/epl-match-predictor.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository**: `<your-user>/epl-match-predictor`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
5. Open **Advanced settings** and set **Python version** to **3.12** or newer (the pinned packages in `requirements.txt` need it). No secrets are required.
6. Click **Deploy**. Streamlit installs `requirements.txt` from the repo root and starts the app. The first build takes a few minutes.

## Updating the deployed app

Streamlit Cloud redeploys automatically on every push to the deployed branch. To refresh data and models:

```bash
python -m src.data && python -m src.train && python -m src.evaluate && python -m src.predict
git add data/processed models reports && git commit -m "Refresh data and models" && git push
```
