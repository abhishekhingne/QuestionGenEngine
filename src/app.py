from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from question_gen import QuestionGenEngine


# Create an instance of the FastAPI class
app = FastAPI()

# Define a Pydantic model for request body (if needed)
class QuestionGenAPI(BaseModel):
    goal: str
    manager_role: str
    llm_url: str
    use_grader: bool

# Define the POST endpoint
@app.post("/get_survey_question/")
def simple_rag(payload: QuestionGenAPI):
    engine = QuestionGenEngine(payload.llm_url)
    if payload.use_grader:
        result = engine.execute_graph(payload.goal, payload.manager_role)
    else:
        result = engine.survey_question_generation(payload.goal, payload.manager_role)
    return result