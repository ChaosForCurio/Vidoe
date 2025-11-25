# Northflank Deployment Guide

This guide details how to deploy the CPU-friendly video generation API on Northflank's free tier.

## Prerequisites
- A Northflank account (Free tier is sufficient).
- A GitHub account with this repository pushed.

## Step 1: Create a New Project
1. Log in to the Northflank dashboard.
2. Click **"Create Project"**.
3. Name it `cpu-video-gen` (or similar).
4. Select a region (e.g., `us-central` or `europe-west`).
5. Click **"Create Project"**.

## Step 2: Create a New Service
1. Inside your project, click **"Create New"** -> **"Service"**.
2. Select **"Build & Deploy"** (Combined service).
3. **Service Name**: `video-api`.
4. **Repository**: Connect your GitHub and select this repository.
   - **Important**: If you see a warning about Northflank not being installed, click the link to **"Install Northflank"** on your GitHub account/organization and select this repository. This is required for automatic builds.
5. **Branch**: `main` (or your working branch).
6. **Build Type**: `Dockerfile`.
   - **Context**: `/` (Root directory).
   - **Dockerfile Path**: `/Dockerfile`.

## Step 3: Configure Resources (Free Tier)
1. Under **"Resources"**, select **"Sandbox"** (Free).
   - This typically provides shared CPU (up to 2 vCPUs burstable) and limited RAM.
   - Ensure "Always On" is checked if available (Sandbox might sleep, but it's free).

## Step 4: Environment Variables
1. Go to **"Environment"** or **"Runtime Variables"**.
2. Add the following variables:
   - `MODEL_NAME`: `cerspense/zeroscope_v2_576w` (or your preferred model)
   - `API_KEY`: `mysecretapikey123` (Required for access)
   - `RATE_LIMIT`: `1000`
   - `MAX_FRAMES`: `32`
   - `OUTPUT_RESOLUTION`: `256`
   - `ALLOWED_ORIGIN`: `*` (or your frontend URL)
   - `LOGGING`: `true`
   - `BASE_URL`: `https://your-service.onrender.com`

## Step 5: Networking & Ports
1. Under **"Networking"**, ensure Port `8000` is exposed.
2. Protocol: `HTTP`.
3. Publicly accessible: **Yes**.

## Step 6: Deploy
1. Click **"Create Service"**.
2. Northflank will start building the Docker image. This might take 5-10 minutes due to installing PyTorch and dependencies.
3. Once built, the container will start.
4. Check the **"Logs"** tab to see `Uvicorn running on http://0.0.0.0:8000`.

## Step 7: Verify
1. Copy the public URL provided by Northflank (e.g., `https://video-api.northflank.app`).
2. Visit `https://<your-url>/health` in your browser.
3. You should see `{"status": "healthy", ...}`.

## Rate Limiting Note
The application includes a built-in rate limiter (1000 requests/month per key).
- Clients must send `X-API-Key` header.
- Usage is tracked in-memory (resets on restart). For persistence, mount a volume and use a SQLite DB.
