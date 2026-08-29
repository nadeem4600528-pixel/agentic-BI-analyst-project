"""Agentic cleaning orchestration built on profiling evidence."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from profiling.profiler import DataProfiler

from .datatype import convert_types
from .duplicates import remove_duplicates
from .missing_values import impute_missing
from .outliers import treat_outliers
from .standardization import mask_pii, normalize_column_names, standardize_columns
from .techniques import execute_technique
from .validation import quarantine, validate_common_formats, validate_dataframe


@dataclass
class CleaningDecision:
	"""A proposed change and the evidence that justified it."""

	action: str
	reason: str
	risk: str = "medium"
	parameters: Dict[str, Any] = field(default_factory=dict)
	approved: bool = False

	def as_dict(self) -> Dict[str, Any]:
		return {
			"action": self.action,
			"reason": self.reason,
			"risk": self.risk,
			"parameters": self.parameters,
			"approved": self.approved,
		}


class CleaningDecisionEngine:
	"""Convert profiler findings into explainable, non-destructive decisions."""

	def propose(self, report: Dict[str, Any]) -> List[CleaningDecision]:
		decisions: List[CleaningDecision] = []
		# 1. Missing values
		missing = report.get("statistics", {}).get("missing_values", {})
		for column, finding in missing.items():
			if finding.get("missing_count", 0):
				metadata_columns = report.get("metadata", {}).get("columns", {})
				metadata_column = metadata_columns.get(column, {}) if isinstance(metadata_columns, dict) else {}
				numeric = bool(metadata_column.get("is_numeric", False))
				if not numeric:
					numeric = column in report.get("statistics", {}).get("numeric_statistics", {})
				strategy = "median" if numeric else "mode"
				decisions.append(CleaningDecision(
					action="impute_missing",
					reason=f"Profiler found {finding['missing_count']} missing values in '{column}'.",
					parameters={"columns": [column], "strategy": strategy},
				))
				decisions.append(CleaningDecision(
					action="impute_missing",
					reason=f"Create missing-value indicator for '{column}'.",
					parameters={"columns": [column], "strategy": "indicator"},
					risk="low",
				))

		# 2. Empty strings normalization
		for column, finding in missing.items():
			if finding.get("empty_string_count", 0):
				decisions.append(CleaningDecision(
					action="empty_string_normalization",
					reason=f"Profiler found {finding['empty_string_count']} empty strings in '{column}'.",
					parameters={},
				))

		# 3. Exact Duplicates
		duplicates = report.get("structure", {}).get("duplicate_rows", {})
		if duplicates.get("duplicate_row_count", 0):
			decisions.append(CleaningDecision(
				action="remove_duplicates",
				reason=f"Profiler found {duplicates['duplicate_row_count']} exact duplicate rows.",
				risk="high",
			))

		# 4. Outliers
		outliers = report.get("statistics", {}).get("outliers", {})
		outlier_columns = [column for column, finding in outliers.items() if finding.get("outlier_count", 0)]
		if outlier_columns:
			decisions.append(CleaningDecision(
				action="treat_outliers",
				reason=f"Profiler identified outliers in {', '.join(outlier_columns)}.",
				risk="high",
				parameters={"columns": outlier_columns, "method": "cap"},
			))

		# 5. PII Detection & Masking
		pii = report.get("semantics", {}).get("pii_detection", {})
		pii_columns = [column for column, finding in pii.items() if finding.get("pii_detected")]
		if pii_columns:
			decisions.append(CleaningDecision(
				action="mask_pii",
				reason=f"Profiler detected sensitive data in {', '.join(pii_columns)}.",
				risk="high",
				parameters={"columns": pii_columns},
			))

		# 6. Column name normalization
		column_names_report = report.get("structure", {}).get("column_names", {})
		if column_names_report or report.get("metadata", {}):
			decisions.append(CleaningDecision(
				action="normalize_column_names",
				reason="Standardize column names to clean snake_case format.",
				risk="low",
				parameters={},
			))

		# 7. Common format validation / Contact fields
		formats = report.get("validation", {}).get("formats", {})
		email_cols = [col for col, data in formats.items() if "invalid_count" in data and data["invalid_count"] > 0]
		if email_cols:
			decisions.append(CleaningDecision(
				action="email_phone_normalization",
				reason=f"Normalize contact fields for email columns: {', '.join(email_cols)}.",
				parameters={"email_columns": email_cols},
			))
		return decisions


class CleaningValidator:
	"""Validate cleaned data and re-run the existing profiler."""

	def validate(
		self,
		before: pd.DataFrame,
		after: pd.DataFrame,
		before_report: Dict[str, Any],
		tables: Optional[Dict[str, pd.DataFrame]] = None,
	) -> Dict[str, Any]:
		try:
			after_report = DataProfiler(after, tables=tables).profile()
		except (TypeError, ValueError, RuntimeError) as error:
			return {
				"passed": False,
				"validation_error": str(error),
				"rows_before": int(len(before)),
				"rows_after": int(len(after)),
			}
		before_score = before_report.get("quality_score", {}).get("overall_quality_score")
		after_score = after_report.get("quality_score", {}).get("overall_quality_score")
		return {
			"passed": bool(after_score is None or before_score is None or after_score >= before_score),
			"rows_before": int(len(before)),
			"rows_after": int(len(after)),
			"quality_score_before": before_score,
			"quality_score_after": after_score,
			"report": after_report,
		}


class CleaningAgent:
	"""Plan, approve, apply, validate, and audit profiling-driven cleaning."""

	def __init__(self, df: pd.DataFrame, tables: Optional[Dict[str, pd.DataFrame]] = None) -> None:
		if not isinstance(df, pd.DataFrame):
			raise TypeError("CleaningAgent requires a pandas DataFrame.")
		self.original = df.copy(deep=True)
		self.tables = tables
		self._rollback = self.original.copy(deep=True)

	def plan(self, profiling_report: Dict[str, Any]) -> Dict[str, Any]:
		decisions = CleaningDecisionEngine().propose(profiling_report)
		return {
			"created_at": datetime.now().isoformat(),
			"source": "profiling_report",
			"decisions": [decision.as_dict() for decision in decisions],
			"requires_approval": any(decision.risk == "high" for decision in decisions),
		}

	def apply(
		self,
		profiling_report: Dict[str, Any],
		decisions: Optional[List[Dict[str, Any]]] = None,
		approve_risky: bool = False,
	) -> Dict[str, Any]:
		plan = self.plan(profiling_report)
		chosen = decisions if decisions is not None else plan["decisions"]
		cleaned = self.original.copy(deep=True)
		audit: List[Dict[str, Any]] = []
		for item in chosen:
			if not item.get("approved", False) and item.get("risk") == "high" and not approve_risky:
				audit.append({**item, "status": "skipped_pending_approval"})
				continue
			cleaned = self._execute(cleaned, item["action"], item.get("parameters", {}))
			audit.append({**item, "status": "applied"})
		validation = CleaningValidator().validate(self.original, cleaned, profiling_report, self.tables)
		return {
			"cleaned_data": cleaned,
			"decision_plan": plan,
			"audit_log": audit,
			"validation": validation,
			"rollback_available": True,
		}

	def rollback(self) -> pd.DataFrame:
		return self._rollback.copy(deep=True)

	def _execute(self, df: pd.DataFrame, action: str, parameters: Dict[str, Any]) -> pd.DataFrame:
		if action == "impute_missing":
			return impute_missing(df, **parameters)
		if action == "remove_duplicates":
			return remove_duplicates(df, **parameters)
		if action == "treat_outliers":
			options = dict(parameters)
			columns = options.pop("columns", None)
			result = treat_outliers(df, **options)
			if columns:
				untouched = [column for column in result.columns if column not in columns and not str(column).endswith("__outlier")]
				result = result[untouched + [column for column in result.columns if column not in untouched]]
			return result
		if action == "standardize_columns":
			return standardize_columns(df, **parameters)
		if action == "normalize_column_names":
			return normalize_column_names(df)
		if action == "convert_types":
			return convert_types(df, parameters)
		if action == "mask_pii":
			return mask_pii(df, **parameters)
		if action == "quarantine":
			valid, _ = quarantine(df, parameters["invalid_mask"])
			return valid
		return execute_technique(action, df, **parameters)


def clean_dataframe(
	df: pd.DataFrame,
	profiling_report: Dict[str, Any],
	decisions: Optional[List[Dict[str, Any]]] = None,
	approve_risky: bool = False,
) -> Dict[str, Any]:
	"""Convenience entry point for profiling-driven cleaning."""
	return CleaningAgent(df).apply(profiling_report, decisions, approve_risky)


