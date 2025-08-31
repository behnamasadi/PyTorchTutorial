from fastapi import FastAPI, Query

app = FastAPI()


@app.get("/add")
def add_numbers(
    num1: int = Query(..., description="First number"),
    num2: int = Query(..., description="Second number")
):
    result = num1 + num2
    return {
        "num1": num1,
        "num2": num2,
        "sum": result
    }
