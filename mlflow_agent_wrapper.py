import mlflow
from uuid import uuid4
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentResponse
from langchain_core.messages import AIMessage

class copilot_agent(ResponsesAgent):
    def __init__(self):
        self.agent = None
        self.objects_documentation = None

    def predict(self, request) -> ResponsesAgentResponse:
        ''' Convert ResponsesAgent format to LangGraph format and back '''
        if self.agent is None:
            from agent import agent
            from utilities import objects_documentation

            self.agent = agent
            self.objects_documentation = objects_documentation

        if isinstance(request.input[0].content, str):
            user_message = request.input[0].content
        else:
            user_message = request.input[0].content[0].text

        # Build state
        state = {
            'objects_documentation': self.objects_documentation,
            'messages_log': [],
            'intermediate_steps': [],
            'current_question': user_message,
            'current_sql_queries': [],
            'generate_answer_details': {},
            'llm_answer': AIMessage(content=''),
            'scenario': ''
        }

        # Run the agent and extract response
        config = {'configurable': {'thread_id': str(uuid4())}}
        result = self.agent.invoke(state, config=config)
        response_text = result['llm_answer'].content

        # Convert back to ResponsesAgent format
        return ResponsesAgentResponse(
            output=[{
                'type': 'message',
                'id': str(uuid4()),
                'content': [{'type': 'output_text', 'text': response_text}],
                'role': 'assistant'
            }]
        )

# Set the model
mlflow.models.set_model(copilot_agent())