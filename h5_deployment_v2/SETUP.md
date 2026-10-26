# Set up and steps
cd ml_zoomcamp/h5-deployment
`uv python pin`

**To update a lock file**
`uv lock`
`uv lock --refresh`
`uv sync --locked`


`curl -L -o pipeline.bin https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/homework/05-deployment/pipeline.bin`

**ensure file pipeline.bin has the following content** 
`sha256sum pipeline.bin`
**it must be**
1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1 *pipeline.bin

`uv run python smoke_test.py`


## Fast api service
1. ** verify the API endpoints** 
`uv run uvicorn predict:app --host 0.0.0.0 --port 9696`

2. **separate terminal window, run your prediction test** 
`uv run python predict-test.py`

3. **Build and Run Your Docker Container** 
`docker build -t zoomcamp-model:2026-hw5 .`

**Command for running containerized application in a live isolated environment.** 
`docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5`

## How to get answers to question
1. check for version
`uv --version`

2. Scikit-Learn version
 `grep -A 5 'name = "scikit-learn"' uv.lock`
  `grep -A 5 'name = "scikit-learn"' pyproject.toml`

3. Conversion probability (loaded model)
**Find and kill the process on port 9696 (if needed)** 
`npx kill-port 9696`

**Or manually check:**
`lsof -i :9696`

**Start your FastAPI app fresh**
`uv run python predict-test.py and value get for Conversion Probability`

**start predict.py or restart**
`uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload`

### check health
` curl http://localhost:9696/health`

`
curl -s http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
  }'
`

4. Conversion probability (API)
**Value produced when executing predict-test.py against your running** 
FastAPI service (http://localhost:9696/predict)

5. Python base image
Found in Dockerfile From python:[base_image]

6. Conversion probability (container)
**Exact probability output by predict-test.py when running against the active Docker container**
`uv run python predict-test.py`
