from enum import Enum

from fastapi import FastAPI, Path, Query
from redis import Redis
from rq import Queue
from rq.job import Job
from tasks import generate_report


class ReportType(str, Enum):
    sales = "sales"
    inventory = "inventory"
    analytics = "analytics"

app = FastAPI()
redis_conn = Redis(host="redis", port=6379)
queue = Queue(connection=redis_conn)


@app.get("/add")
def add_numbers(
    num1: int = Query(..., description="First number"),
    num2: int = Query(..., description="Second number"),
):
    result = num1 + num2
    return {
        "num1": num1,
        "num2": num2,
        "sum": result,
    }


@app.post("/reports/")
def create_report(report_type: ReportType = Query(ReportType.sales, description="Type of report to generate")):
    """Request a report. The work happens in the background."""
    job = queue.enqueue(generate_report, report_type)
    return {"job_id": job.id, "status": "queued", "report_type": report_type}


@app.get("/reports/{job_id}")
def get_report(job_id: str = Path(..., description="Job ID returned by POST /reports/")):
    """Check if a report is ready."""
    job = Job.fetch(job_id, connection=redis_conn)
    response = {"job_id": job_id, "status": job.get_status()}
    if job.result is not None:
        response["result"] = job.result
    return response


@app.get("/healthz")
def health():
    return {"status": "ok"}
