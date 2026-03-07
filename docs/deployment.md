# Deployment Instructions

1. **Training artifacts**
   - `python training/data_pipeline.py`
   - `python training/train_tinyml_model.py`
2. **Fog + cloud dependencies**
   - `docker compose -f cloud/docker-compose.yml up -d`
3. **Run cloud API**
   - `uvicorn cloud.main:app --host 0.0.0.0 --port 8000`
4. **Run edge/fog simulation**
   - Import `edge.edge_device` and `fog.fog_gateway` from orchestration scripts.
5. **Dashboard**
   - `python -m http.server 9000 -d dashboard`
   - Open `http://localhost:9000/index.html`.
