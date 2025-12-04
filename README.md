# exclaim-scorer

# Docker: Build, Run, and Test the Scorer API

## Prerequisites
- Docker installed
- Project layout:
  - `dockerfile` (at repo root)
  - `requirements.txt`
  - `src/app.py` (FastAPI app, `app`)
  - `src/scorer/backend.py` and related modules

## Build the image
- From the project root:
```bash
docker build -t scorer:latest -f dockerfile .
```


Run the container
- Map port 8000 and start the API:

```bash
docker run --rm -p 8000:8000 --name scorer scorer:latest
```

Expected logs:
- Uvicorn starts at http://0.0.0.0:8000
- Pydantic warnings are cosmetic; the server still runs

## Test the API
- In another terminal, send a POST to /score:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "decoder",
    "model_name": "gpt2",
    "task": null,
    "document": {
      "Claim": { "description": "Test claim", "SubClaims": [] }
    }
  }'
```

Succesful response example
```bash
{"model_type":"decoder","model_name":"gpt2","scored_json":{"Claim":{"description":"Test claim","SubClaims":[],"edge_scores":[{"parent":"","comprehensiveness_score":0.7656,"sufficiency_score":0.2344}]}}}% 
```

## Useful checks
- Verify module path inside the container:

```bash
docker run --rm scorer:latest python -c "import src.app; print('OK')"
```

- Tail logs while testing:
```bash
docker logs -f scorer
```

## Notes
- The Dockerfile uses CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"].
- If you change the app module path, update the Dockerfile CMD accordingly and rebuild.
