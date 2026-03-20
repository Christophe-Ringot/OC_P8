from fastapi import APIRouter, HTTPException, status
from typing import Optional
from datetime import datetime

from ..monitoring.drift_detection import drift_detector

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@router.get("/drift/summary")
async def get_drift_summary():
    try:
        summary = drift_detector.get_drift_summary()
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting drift summary: {str(e)}"
        )


@router.post("/drift/report")
async def generate_drift_report(
    output_path: Optional[str] = "reports/drift_report.html"
):
    try:
        results = drift_detector.generate_drift_report_from_logs(
            output_path=output_path
        )

        return {
            "status": "success",
            "report_path": output_path,
            "drift_results": results,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating drift report: {str(e)}"
        )


@router.post("/drift/set-reference")
async def set_reference_data(data_path: str):
    try:
        drift_detector.load_reference_data(data_path)
        return {
            "status": "success",
            "message": f"Reference data loaded from {data_path}",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading reference data: {str(e)}"
        )
