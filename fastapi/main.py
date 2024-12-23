from fastapi import FastAPI

app=FastAPI()

@app.get("/") # Decorator for GET requests to the root ("/") route

async def root():
    return {"message":"Hello World for REST API"}

