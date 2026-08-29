"""End-to-end workflow orchestration for the BI analyst pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

from analysis.analysis_engine import analyze_dataset
from cleaning.cleaning_agent import CleaningAgent
from dashboard.dashboard_engine import DashboardBuilder
from profiling.profiler import DataProfiler
from reports.report_generator import build_comprehensive_report


logger = logging.getLogger("agentic_bi.workflow")


class WorkflowStatusStore:
    """In-memory job tracker for pipeline executions."""

    _jobs: Dict[str, Dict[str, Any]] = {}
    _lock = Lock()

    @classmethod
    def create_job(cls, payload: Optional[Dict[str, Any]] = None) -> str:
        job_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with cls._lock:
            cls._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "current_step": "queued",
                "stages": {},
                "result": None,
                "error": None,
                "payload": payload or {},
            }
        return job_id

    @classmethod
    def get_job(cls, job_id: str) -> Optional[Dict[str, Any]]:
        with cls._lock:
            return cls._jobs.get(job_id)

    @classmethod
    def list_jobs(cls) -> List[Dict[str, Any]]:
        with cls._lock:
            return list(cls._jobs.values())

    @classmethod
    def update_job(cls, job_id: str, **updates: Any) -> Dict[str, Any]:
        with cls._lock:
            job = cls._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Job {job_id} not found")
            job.update(updates)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            return job


class WorkflowService:
    """Execute the full upload -> profile -> clean -> analyze -> visualize workflow."""

    @staticmethod
    def _stages(job_id: str) -> Dict[str, Any]:
        """Return the current stages after checking that the job exists."""
        job = WorkflowStatusStore.get_job(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        stages = job.get("stages")
        return dict(stages) if isinstance(stages, dict) else {}

    @staticmethod
    def run_pipeline(
        records: List[Dict[str, Any]],
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        category_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not records:
            raise ValueError("Data payload is empty.")

        job_id = WorkflowStatusStore.create_job({
            "date_column": date_column,
            "value_column": value_column,
            "category_column": category_column,
        })

        try:
            WorkflowStatusStore.update_job(
                job_id,
                status="running",
                current_step="load_data",
                stages={"load_data": {"status": "completed"}},
            )
            df = pd.DataFrame(records)

            WorkflowStatusStore.update_job(
                job_id,
                current_step="profile_data",
                stages={**WorkflowService._stages(job_id), "profile_data": {"status": "running"}},
            )
            profiling_report = DataProfiler(df).profile()
            WorkflowStatusStore.update_job(
                job_id,
                current_step="profile_data",
                stages={**WorkflowService._stages(job_id), "profile_data": {"status": "completed", "summary": {"rows": len(df), "columns": len(df.columns), "quality_score": profiling_report.get("quality_score", {}).get("overall_quality_score")}}},
            )

            WorkflowStatusStore.update_job(
                job_id,
                current_step="clean_data",
                stages={**WorkflowService._stages(job_id), "clean_data": {"status": "running"}},
            )
            cleaning_agent = CleaningAgent(df)
            cleaning_result = cleaning_agent.apply(profiling_report, approve_risky=True)
            cleaned_df = cleaning_result.get("cleaned_data", df)
            WorkflowStatusStore.update_job(
                job_id,
                current_step="clean_data",
                stages={**WorkflowService._stages(job_id), "clean_data": {"status": "completed", "rows_before": int(len(df)), "rows_after": int(len(cleaned_df))}},
            )

            # Analysis stage
            WorkflowStatusStore.update_job(
                job_id,
                current_step="analyze_data",
                stages={**WorkflowService._stages(job_id), "analyze_data": {"status": "running"}},
            )
            analysis_result = analyze_dataset(cleaned_df, date_column=date_column, value_column=value_column)
            WorkflowStatusStore.update_job(
                job_id,
                current_step="analyze_data",
                stages={**WorkflowService._stages(job_id), "analyze_data": {"status": "completed", "summary": {"kpi_count": len(analysis_result.get("kpis", [])), "insight_count": len(analysis_result.get("business_insights", []))}}},
            )

            # Prediction stage: placeholder, since no ML model is configured yet
            WorkflowStatusStore.update_job(
                job_id,
                current_step="predict_data",
                stages={**WorkflowService._stages(job_id), "predict_data": {"status": "completed", "summary": {"model_status": "not_configured", "message": "No ML model pipeline is configured yet."}}},
            )

            # Visualization stage
            WorkflowStatusStore.update_job(
                job_id,
                current_step="visualize_data",
                stages={**WorkflowService._stages(job_id), "visualize_data": {"status": "running"}},
            )
            dashboard = DashboardBuilder.build_dashboard(
                cleaned_df,
                date_column=date_column,
                value_column=value_column,
                category_column=category_column,
            )
            report = build_comprehensive_report(
                cleaned_df,
                date_column=date_column,
                value_column=value_column,
                category_column=category_column,
            )
            WorkflowStatusStore.update_job(
                job_id,
                status="completed",
                current_step="visualize_data",
                stages={**WorkflowService._stages(job_id), "visualize_data": {"status": "completed", "summary": {"chart_count": len(dashboard.get("charts", [])), "kpi_count": len(dashboard.get("kpis", []))}}},
                result={
                    "dataset": cleaned_df.to_dict(orient="records"),
                    "profiling_report": profiling_report,
                    "cleaning_report": cleaning_result,
                    "analysis_report": analysis_result,
                    "dashboard": dashboard,
                    "report": report,
                    "predict": {
                        "status": "not_configured",
                        "message": "Prediction model not configured yet.",
                    },
                },
            )

            logger.info("Workflow completed for job %s", job_id)
            job = WorkflowStatusStore.get_job(job_id)
            if job is None:
                raise RuntimeError(f"Workflow job {job_id} was not saved correctly.")
            return job

        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Workflow failed for job %s", job_id)
            WorkflowStatusStore.update_job(
                job_id,
                status="failed",
                current_step="error",
                error={"message": str(exc)},
            )
            raise

    @staticmethod
    def get_status(job_id: str) -> Dict[str, Any]:
        job = WorkflowStatusStore.get_job(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        return job

    @staticmethod
    def list_statuses() -> List[Dict[str, Any]]:
        return WorkflowStatusStore.list_jobs()