#Importing the python dependencies
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph

class QuestionGenEngine:
    """
    A class representing a survey question generation engine.

    This engine utilizes a language model to generate survey questions based on given goals
    and manager roles, and assesses the relevance of these questions.

    Attributes:
        llm_url (str): The base URL for the language model.

    Methods:
        __init__(self, llm_url):
            Initializes the QuestionGenEngine instance with a language model URL and sets up
            the necessary components for question generation and evaluation.

        generate(self, state):
            Generates survey questions based on the provided state containing goal and manager role.
            Uses a predefined prompt template to interact with the language model for question generation.

        grade_question(self, state):
            Assesses the relevance of generated survey questions to the given goal and manager role.
            Utilizes a grader prompt template to interact with the language model for relevance scoring.

        add_nodes(self):
            Adds nodes to the workflow graph, defining the generation and grading steps.

        build_graph(self):
            Builds a state graph representing the workflow of generating and grading survey questions.

        execute_graph(self, goal, manager_role):
            Executes the built workflow graph using provided goal and manager role inputs,
            generating and grading survey questions accordingly.

    Types:
        GraphState (typing.TypedDict):
            Represents the state of the survey question generation workflow.

            Attributes:
                goal (str): The goal for which questions are being generated.
                manager_role (str): The role of the manager requesting the survey questions.
                question (List[str]): List of generated survey questions.

    Example Usage:
        engine = QuestionGenEngine(llm_url="https://example.com/llm")
        app = engine.build_graph()
        result = engine.execute_graph(goal="Improve customer satisfaction", manager_role="Sales Manager")
        print(result)
    """
    def __init__(self, llm_url):
        """
        Initializes the QuestionGenEngine instance.

        Args:
            llm_url (str): The base URL for the language model.
        """
        self.llm = Ollama(base_url=llm_url, model="llama3", temperature=0.3)
        self.workflow = None
        self.question_gen_prompt_template = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert survey designer.
                Your manager wants to achieve a certain goal in order to do that you've to come up with a list of atleast 10 survey questions that will
                help him achieve the goal. 
                List of questions should be in json format. Format for each json of a question should be fixed as below
                question: Survey Question
                type: Whether it is multiple_choice or open_text
                options: List of options for multiple_choice questions
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Goal: {goal} 
                Manager Role: {manager_role} 
                Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=["goal", "manager_role"],)
        self.question_gen_prompt_chain = self.question_gen_prompt_template | self.llm | JsonOutputParser()
        
        self.grader_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
            your colleague has generated a relevant survey question related to a goal. 
            Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a key 'score' and explaination with a key 'explaination'.
            <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the goal and manger role:
            \n ------- \n
            Goal: {goal}
            Manager Role: {manager_role}
            \n ------- \n
            Here is the generated survey question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["goal", "manger_role", "question"],
        )
        self.grader_chain = self.grader_prompt | self.llm | JsonOutputParser()

    def survey_question_generation(self, manager_role, goal):
        """
        Generates survey questions based on the provided manager role, goal, and language model.

        This method invokes the question generation prompt chain to interact with the language model
        and generate survey questions tailored to the specified manager role and goal.

        Args:
            manager_role (str): The role of the manager requesting the survey questions.
            goal (str): The goal for which the survey questions are intended to achieve.
            llm: The language model instance used for question generation (e.g., LLM model).

        Returns:
            dict: A dictionary containing the result of the question generation process.
                The result typically includes a list of generated survey questions in a structured format,
                such as a JSON-like structure with question details (e.g., question text, type, options).
        """
        result = self.question_gen_prompt_chain.invoke({"goal": goal, "manager_role": manager_role})
        return result

    # Define State
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents 
        """
        goal : str
        manager_role: str
        question : List[str]

    def generate(self, state):
        """
        Generates survey questions based on the provided state.

        Args:
            state (GraphState): The current state containing goal and manager role.

        Returns:
            GraphState: The updated state containing generated survey questions.
        """
        print("---GENERATE---")
        goal = state["goal"]
        manager_role = state["manager_role"]
        
        question = self.question_gen_prompt_chain.invoke({"goal": goal, "manager_role": manager_role})
        print(question)
        return {"goal": goal, "manager_role": manager_role, "question": question}

    def grade_question(self, state):
        """
        Grades the relevance of generated survey questions to the goal and manager role.

        Args:
            state (GraphState): The current state containing goal, manager role, and generated questions.

        Returns:
            GraphState: The updated state containing relevant survey questions.
        """
        print("---CHECK QUESTION RELEVANCE TO GOAL---")
        goal = state["goal"]
        manager_role = state["manager_role"]
        questions = state["question"]
        
        # Score each question
        filtered_question = []
        for q in questions:
            score = self.grader_chain.invoke({"goal": goal, "manager_role": manager_role, "question": q})
            grade = score['score']
            # Question relevant
            if grade.lower() == "yes":
                print("---GRADE: QUESTION RELEVANT---")
                q["explaination"] = score["explaination"]
                filtered_question.append(q)
            # Question not relevant
            else:
                print("---GRADE: QUESTION NOT RELEVANT---")
                continue
        question = filtered_question
        return {"goal": goal, "manager_role": manager_role, "question": question}

    def add_nodes(self):
        """
        Adds nodes to the workflow graph representing generation and grading steps.
        """
        self.workflow = StateGraph(self.GraphState)

        # Define the nodes
        self.workflow.add_node("generate", self.generate) # generatae
        self.workflow.add_node("grade_question", self.grade_question) # grade question

    def build_graph(self):
        """
        Builds a state graph representing the workflow of generating and grading survey questions.

        Returns:
            StateGraph: The compiled state graph representing the workflow.
        """
        self.add_nodes()
        self.workflow.set_entry_point("generate")
        self.workflow.add_edge("generate", "grade_question")
        self.workflow.add_edge("grade_question", END)
  
        app = self.workflow.compile()
        return app

    def execute_graph(self, goal, manager_role):
        """
        Executes the workflow graph to generate and grade survey questions.

        Args:
            goal (str): The goal for which questions are being generated.
            manager_role (str): The role of the manager requesting the survey questions.

        Returns:
            GraphState: The final state containing relevant survey questions.
        """
        inputs = {"goal": goal, "manager_role": manager_role}
        app = self.build_graph()
        for output in app.stream(inputs):
            for key, value in output.items():
                # Node
                print(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            print("\n---\n")
        return value