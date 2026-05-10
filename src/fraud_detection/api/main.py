from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from .schemas import (
    HealthResponse,
    PredictionResponse,
    SwitchModelRequest,
    TransactionRequest,
)
from .services import inference_service
from ..config.settings import settings
from ..db.database import Base, engine, get_db

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", active_model=inference_service.active_model_name)


@app.get("/models/list")
def list_models():
    return inference_service.list_available_models()


@app.post("/models/switch")
def switch_model(request: SwitchModelRequest):
    try:
        inference_service.load_model(request.model_name, request.checkpoint_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "switched", "active_model": request.model_name}


@app.post("/verify-transaction", response_model=PredictionResponse)
def verify_transaction(request: TransactionRequest, db: Session = Depends(get_db)):
    try:
        return inference_service.predict(
            model_name=request.model_name,
            feature_vector=request.feature_vector,
            transaction_id=request.transaction_id,
            user_identifier=request.user_identifier,
            merchant_identifier=request.merchant_identifier,
            db=db,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/verify-ensemble")
def verify_ensemble(request: TransactionRequest, db: Session = Depends(get_db)):
    """Run inference sequentially on all 4 models and return a combined payload."""
    results = {}
    models_to_test = ["cnn", "lstm", "transformer", "hybrid"]
    
    for model_name in models_to_test:
        try:
            res = inference_service.predict(
                model_name=model_name,
                feature_vector=request.feature_vector,
                transaction_id=request.transaction_id,
                user_identifier=request.user_identifier,
                merchant_identifier=request.merchant_identifier,
                db=db,
            )
            results[model_name] = res
        except Exception as exc:
            results[model_name] = {"error": str(exc)}
            
    return results

# Mount static files at the end to prevent route conflicts
from fastapi.staticfiles import StaticFiles
import os

plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "artifacts", "plots")
if os.path.exists(plots_dir):
    app.mount("/plots", StaticFiles(directory=plots_dir), name="plots")
