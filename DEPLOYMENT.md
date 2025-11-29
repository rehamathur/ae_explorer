# Streamlit Cloud Deployment Guide

## Prerequisites
- Repository pushed to GitHub: `https://github.com/rehamathur/ae_explorer.git`
- Streamlit Cloud account (free tier available)

## Deployment Steps

### 1. Go to Streamlit Cloud
Visit https://share.streamlit.io/ and sign in with your GitHub account.

### 2. Create New App
- Click "New app" button
- Select repository: `rehamathur/ae_explorer`
- Set branch: `main`
- Set main file path: `app.py`

### 3. Configure Secrets (Environment Variables)
Click "Advanced settings" → "Secrets" and add:

```
OPENAI_API_KEY=your_openai_api_key_here
REDUCTO_API_KEY=your_reducto_api_key_here
```

**Important:** Never commit these keys to your repository!

### 4. Deploy
Click "Deploy" and wait for the app to build and launch.

### 5. Access Your App
Once deployed, Streamlit Cloud will provide you with a public URL like:
`https://your-app-name.streamlit.app`

## Files Included in Deployment
- `app.py` - Main Streamlit application
- `process_trials.py` - Data processing module
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration
- CSV files (ae.csv, arms.csv, baseline.csv, trials.csv, etc.) - Initial data

## Troubleshooting

### App fails to start
- Check that all environment variables are set correctly
- Verify `requirements.txt` includes all dependencies
- Check Streamlit Cloud logs for error messages

### Missing data
- CSV files should be in the repository root
- App will regenerate CSV files when processing new PDFs
- Existing CSV files will be appended to (not overwritten)

### API errors
- Verify API keys are set correctly in Streamlit Cloud secrets
- Check API key permissions and quotas

## Updating the App
After making changes:
1. Commit and push to GitHub: `git push origin main`
2. Streamlit Cloud will automatically redeploy the app

