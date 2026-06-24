# Public Demo Deployment

This project deploys as two pieces:

- Backend API: existing FastAPI service on Cloud Run.
- Frontend website: Vite/React static app on Vercel.

The first public demo uses generated URLs only. Add a custom domain later after
the generated deployment is healthy.

## Backend: Cloud Run

The backend is already deployed by `.github/workflows/docker.yml` on pushes to
`main`. That workflow builds the Docker image, pushes it to Artifact Registry,
runs Alembic against Cloud SQL, and rolls `city-concierge-api` forward.

Before sharing the website, confirm the service is public:

```bash
gcloud run services get-iam-policy city-concierge-api \
  --region us-central1 \
  --project mlops-491820 \
  --flatten="bindings[].members" \
  --filter="bindings.role:roles/run.invoker AND bindings.members:allUsers"
```

If no binding is returned, make the API public:

```bash
gcloud run services add-iam-policy-binding city-concierge-api \
  --region us-central1 \
  --project mlops-491820 \
  --member=allUsers \
  --role=roles/run.invoker
```

Health check:

```bash
curl -fsS https://city-concierge-api-6amzjx52nq-uc.a.run.app/health
```

## Frontend: Vercel

Create a Vercel project from the GitHub repository:

- Framework preset: Vite
- Root directory: `frontend`
- Install command: `npm ci`
- Build command: `npm run build`
- Output directory: `dist`

These defaults are committed in `frontend/vercel.json`; setting the root
directory to `frontend` is the important project setting.

Set these Vercel environment variables for Production and Preview:

```bash
VITE_API_URL=https://city-concierge-api-6amzjx52nq-uc.a.run.app
VITE_GOOGLE_MAPS_API_KEY=<browser-restricted Maps key>
VITE_GOOGLE_MAPS_MAP_ID=<map id>
```

Leave the Maps values blank if you want the first deploy to use the graceful
"Map unavailable" fallback. Never add backend secrets to Vercel; every `VITE_*`
value is visible in the browser bundle.

## Google Maps Key

Use a browser key, separate from backend ingestion and Directions keys. Restrict
it to HTTP referrers:

- `http://localhost:5173/*`
- `https://<your-vercel-project>.vercel.app/*`
- `https://*.vercel.app/*` only if you intentionally want all Vercel previews
  for this project to work before tightening the key.

Restrict APIs to Maps JavaScript API and Directions API. See
`docs/google_maps_setup.md` for the full key and Map ID setup.

## CORS

The backend defaults allow:

- `http://localhost:5173`
- `http://localhost:3000`
- any generated `https://*.vercel.app` origin

For a future custom domain, set Cloud Run env vars without code changes:

```bash
gcloud run services update city-concierge-api \
  --region us-central1 \
  --project mlops-491820 \
  --set-env-vars CORS_ALLOW_ORIGINS=https://your-domain.example \
  --set-env-vars CORS_ALLOW_ORIGIN_REGEX=
```

## Smoke Test

Run this before sharing the URL:

1. Open the Vercel production URL.
2. Send a normal itinerary request.
3. Confirm the browser Network tab shows `POST /chat` returning 200 from
   Cloud Run with no CORS errors.
4. Confirm cards render with place names and addresses.
5. If Maps env vars are set, confirm pins and route render.
6. If Maps env vars are blank, confirm the app shows the map fallback instead
   of crashing.

Local verification commands:

```bash
conda run -n city-concierge-py311 python -m pytest -q
conda run -n city-concierge-py311 python -m ruff check .
cd frontend && npm test
cd frontend && npm run build
```

## Cost Guardrails

For a public demo, keep Cloud Run `max-instances` capped, keep provider spend
limits active, and watch Cloud Run, LLM provider, and Google Maps usage after
sharing the link. The current Cloud Run deploy workflow caps the service at 10
instances.
