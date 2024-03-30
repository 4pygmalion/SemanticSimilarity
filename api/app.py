import os
import gc
import sys
from typing import Dict
from contextlib import asynccontextmanager

from omegaconf import OmegaConf
from fastapi import FastAPI, Request, status
from fastapi.responses import RedirectResponse, JSONResponse


SEMANTIC_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SEMANTIC_SIM_DIR)
sys.path.append(ROOT_DIR)
from utils.log_ops import get_logger
from SemanticSimilarity.data_model import (
    PatientSymptoms,
    APIResponse,
    HPOIdFormatError,
)
from SemanticSimilarity.build_datasets import OntologyHandler
from SemanticSimilarity.calculator import ResnikSimilarityCalculator, NodeLevelSimilarityCalculator


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = get_logger(
        "semantic_similarity",
        os.path.join(SEMANTIC_SIM_DIR, "logs", "semantic_similiarty_api.log"),
    )
    app.state.logger = logger

    logger.info("API Startup: load hpo.obo")
    config = OmegaConf.load(os.path.join(SEMANTIC_SIM_DIR, "config.yaml"))
    ontology_handelr = OntologyHandler(config, logger)
    calculator = NodeLevelSimilarityCalculator(
        config=config, hpo_ontology=ontology_handelr.ontology, logger=logger
    )
    calculator.set_disease_pheno_map(ontology_handelr.disease_pheno_map)
    calculator.set_level()
    calculator.set_mica_mat()

    app.state.calculator = calculator

    yield

    del app.state.calculator
    del app.state.logger


app = FastAPI(lifespan=lifespan)


@app.exception_handler(HPOIdFormatError)
def exception_handler(request: Request, exception: HPOIdFormatError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"messsage": str(exception)},
    )


@app.get("/")
def root():
    return RedirectResponse(url="/docs", status_code=302)


@app.on_event("shutdown")
def app_shutdown():
    del app.state.calculator
    gc.collect()

    return


@app.post("/calculate/", response_model=APIResponse)
def calculate(patient_symptoms: PatientSymptoms, request: Request) -> Dict:
    disease_similarity: Dict[str, float] = request.app.state.calculator.get_disease_similarty(
        patient_symptoms.symptoms
    )
    return APIResponse(disease_similarity)
