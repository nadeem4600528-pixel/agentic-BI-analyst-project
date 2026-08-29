"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dashboard import router as dashboard_router
from api.report import router as report_router
from api.upload import router as upload_router
from api.analysis import router as analysis_router
from api.workflow import router as workflow_router
from api.transformation import router as transformation_router



app = FastAPI(title="Agentic BI Analyst")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(dashboard_router)
app.include_router(analysis_router)
app.include_router(report_router)
app.include_router(workflow_router)
app.include_router(upload_router)
app.include_router(transformation_router)