from typing import Sequence, Literal, Annotated
from typing_extensions import TypedDict

# LangChain & LangGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.agents import AgentAction
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from databricks_langchain import ChatDatabricks
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import utilities
from utilities import query_genie as query_genie_api

# Genie space ID
GENIE_SPACE_ID = "your-genie-space-id"  

# Initialize LLM (Databricks Foundation Model)
llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4", temperature=0)

# LangGraph state

class State(TypedDict):
    """Agent state"""
    messages_log: Annotated[Sequence[BaseMessage], add_messages]  # Chat history
    current_question: str  # Latest user question
    current_sql_queries: list[dict]  # SQL queries with results: [{'query': str, 'explanation': str, 'result': str}]
    llm_answer: BaseMessage  # AI response
    scenario: str  # A (data query), B (pleasantries), C (data unavailable), D (ambiguous)
    generate_answer_details: dict  # Contains: agent_questions, key_assumptions, ambiguity_explanation
    objects_documentation: str  # Database schema documentation
    intermediate_steps: list[AgentAction]  # Execution tracking

# classes for agent's structured output

class AgentQuestions(TypedDict):
    """Next step suggestions for the user"""
    agent_questions: Annotated[list[str], "max 2 smart next steps for the user to explore further"]

class QueryExplanation(TypedDict):
    """Explanation highlights for SQL query assumptions"""
    explanation: Annotated[list[str], "2-5 concise assumptions/highlights"]

class ScenarioDecision(TypedDict):
    """Decision for routing"""
    next_step: Annotated[Literal["B", "C", "Continue"], "indication of the next step"]

class ClearOrAmbiguous(TypedDict):
    """Conclusion about question clarity"""
    analytical_intent_clearness: Annotated[Literal["CLEAR", "AMBIGUOUS"], "conclusion about the analytical intent"]

class AmbiguityAnalysis(TypedDict):
    """Analysis of ambiguous question with explanation and alternatives"""
    ambiguity_explanation: Annotated[str, "brief explanation of what makes the question ambiguous"]
    agent_questions: Annotated[list[str], "2-3 alternative analytical intents as questions"]

# Helper functions 

def extract_msg_content_from_history(messages_log: list) -> str:
    """Extract content from message history"""
    content = []
    for msg in messages_log:
        content.append(msg.content)
    return "\n".join(content)

def format_sql_query_results_for_prompt(sql_queries: list[dict]) -> str:
    """Format SQL query results for LLM prompt"""
    formatted_queries = []
    for query_index, q in enumerate(sql_queries):
        block = f"Query {query_index+1}:\n{q['query']}\n\nResult of query {query_index+1}:\n{q['result']}"
        formatted_queries.append(block)
    return "\n\n".join(formatted_queries)

def create_query_explanation(sql_query: str) -> dict:
    """Generate explanation highlights for query assumptions"""
    system_prompt = """You are provided with the following SQL query:
{sql_query}.
Your task is to highlight parts of this query to a non-technical user, including only the highlight types below if they exist.

Guidelines:
- Use only bullet points, max 0-3 bullet points, keep just the most important info.
- Keep every bullet very concise, very few words.
- Don't include filters applied to current records.
- Don't include highlights that are not part of the list below.

List of highlight types:
  - filters applied.
    Ex: "excluded inactive affiliates".
  - Show time range of the source table (min/max dates) if the source table for the query has data over time.
    Ex: "account snapshot dates between 2021 and 2022"
  - TOP X rows limits the result.
    Ex: "Results limited to top 10 affiliates by assets"
    """
    prompt = ChatPromptTemplate.from_messages([('user', system_prompt)])
    chain = prompt | llm.with_structured_output(QueryExplanation)
    llm_explanation = chain.invoke({'sql_query': sql_query})
    return {'explanation': llm_explanation['explanation']}

def generate_agent_questions(state: State) -> list[str]:
    """Generate 2 next steps to guide the user. Only runs for scenarios A, B, or C."""
    sys_prompt = """You are a decision support consultant helping users become more data-driven.

Here is the conversation history with the user:
{messages_log}.

Latest user message:
{question}.

Your task is to guide the users to answer their analytical goal that you derive from the conversation history and from the last user message.

Suggest max 2 smart next steps for the user to explore further, chosen from the examples below and tailored to what's available in the database schema:
  {objects_documentation}

  Example of next steps:
  - Trends over time:
    Example: "Want to see how this changed over time?".
    Suggest trends over time only for tables containing multiple dates available.

  - Drill-down suggestions:
    Example: "Would you like to explore this by brand or price tier?"

  - Top contributors to a trend:
    Example: "Want to see the top 5 products that drove this increase in satisfaction?"

  - Explore a possible cause:
    Example: "Curious if pricing could explain the drop? I can help with that."

  - Explore the data at higher granularity levels if the user analyzes on low granularity columns. Use database schema to identify such columns.
    Example: Instead of analyzing at product level, suggest at company level.

  - Explore the data on filtered time ranges. Check the database schema for date range information under "Important considerations about dates available".
    Example: Instead of analyzing for all feedback dates, suggest filtering for a year or for a few months.

  - Filter the data on the value of a specific attribute. Use values from the database schema.
    Example: Instead of analyzing for all companies, suggest filtering for a single company and give a few suggestions.
    """

    prompt = ChatPromptTemplate.from_messages([('system', sys_prompt)])
    chain = prompt | llm.with_structured_output(AgentQuestions)
    result = chain.invoke({
        'messages_log': extract_msg_content_from_history(state['messages_log']),
        'question': state['current_question'],
        'objects_documentation': state['objects_documentation']
    })
    return result['agent_questions']

# Agent nodes

def orchestrator(state: State):
    """ Orchestrator deciding if the user question requires querying the database or is asking for info not available """
    
    sys_prompt = """You are a decision support consultant helping users make data-driven decisions.

    Your task is to decide the next action for this question: {question}.

    Conversation history: {messages_log}.
    Current insights: {insights}.
    Database schema: {objects_documentation}

    Decision process:

    Step 1. Check if question is non-analytical or already answered:
       - If question is just pleasantries ("thank you", "hello", "how are you") → "B"
       - If the same question was already answered in conversation history → "B"

    Step 2. Check if requested data exists in schema:
      - If the user asks for data/metrics not available AND no synonyms or related terms exist in the database schema → "C"

    Step 3. Otherwise → "Continue".
    """

    prompt = ChatPromptTemplate.from_messages([('user', sys_prompt)])
    chain = prompt | llm.with_structured_output(ScenarioDecision)

    result = chain.invoke({
        'messages_log': extract_msg_content_from_history(state['messages_log']),
        'question': state['current_question'],
        'insights': format_sql_query_results_for_prompt(state['current_sql_queries']),
        'objects_documentation': state['objects_documentation']
    })

    if result['next_step'] == 'Continue':
        scenario = ''
        next_tool_name = 'clarification_check'
    elif result['next_step'] == 'B':
        scenario = 'B'
        agent_questions = generate_agent_questions(state)
        state['generate_answer_details'] = {'agent_questions': agent_questions}
        next_tool_name = 'generate_answer'
    else:  # C
        scenario = 'C'
        agent_questions = generate_agent_questions(state)
        state['generate_answer_details'] = {'agent_questions': agent_questions}
        next_tool_name = 'generate_answer'

    # Update state
    state['scenario'] = scenario

    # Log orchestrator run
    action = AgentAction(tool='orchestrator', tool_input='', log='tool ran successfully')
    state['intermediate_steps'].append(action)

    # Control flow
    action = AgentAction(tool=next_tool_name, tool_input='', log='')
    state['intermediate_steps'].append(action)

    return state


@tool
def clarification_check(state: State):
    """Determines if question is clear (scenario A) or ambiguous (scenario D)"""

    sys_prompt = """Decide whether the user question is clear or ambigous based on this specific database schema:
  {objects_documentation}.

  Conversation history:
  "{messages_log}".

  User question:
  "{question}".

  *** The question is CLEAR if ***
  - It has a single, obvious analytical approach in terms of underlying source columns, relationships or past conversations.
    Example: "what is the revenue?" is clear in a database schema that contains just 1 single metric that can answer the question (ex: net_revenue).

  - The column and metric naming in the schema clearly points to one dominant method of interpretation.
    Example: "what is the top client?" is clear in a database schema that contains just 1 single metric that can answer the question (ex: sales_amount).

  - You can apply reasonable assumptions. Examples:
    No specific time periods indicated -> assume a recent period -> CLEAR.
    No level of details specified -> use highest aggregation level -> CLEAR.

  - You can deduct the analytical intent from the conversation history.

  *** The question is AMBIGUOUS if ***
  - Different source columns would give different insights.

  - Different metrics could answer the same question:
    Example: "What is the top client?" is ambigous in a database schema that contains multiple metrics that can answer the question (highest value of sales / highest number of sales).

  Response format:
  If CLEAR -> "CLEAR".
  If AMBIGUOUS -> "AMBIGUOUS".
  """

    prompt = ChatPromptTemplate.from_messages([('user', sys_prompt)])
    chain = prompt | llm.with_structured_output(ClearOrAmbiguous)

    result = chain.invoke({
        'messages_log': extract_msg_content_from_history(state['messages_log']),
        'question': state['current_question'],
        'objects_documentation': state['objects_documentation']
    })

    # Determine next step based on result
    if result['analytical_intent_clearness'] == 'CLEAR':
        state['scenario'] = 'A'
        next_tool_name = 'query_genie'
    else:  # AMBIGUOUS
        state['scenario'] = 'D'
        next_tool_name = 'clarification'

    # Log clarification_check run
    action = AgentAction(tool='clarification_check', tool_input='', log='tool ran successfully')
    state['intermediate_steps'].append(action)

    # Control flow
    action = AgentAction(tool=next_tool_name, tool_input='', log='')
    state['intermediate_steps'].append(action)

    return state


@tool
def clarification(state: State):
    """Generates ambiguity analysis when question is ambiguous (scenario D)"""

    sys_prompt = """The latest user question is ambiguous based on the following database schema:
  {objects_documentation}.

  Here is the conversation history with the user:
  "{messages_log}".

  Latest user message:
  "{question}".

  Step 1: Identify what makes the question ambiguous. The question is ambiguous if:

  - Different source columns would give substantially different insights:
    Example: pre-aggregated vs computed metrics with different business logic.

  - Multiple fundamentally different metrics could answer the same question:
    Example: "What is the top client?" is ambiguous in a database schema that contains multiple metrics that can answer the question (highest value of sales / highest number of sales).

  - Different columns with the same underlying source data (check database schema) do NOT create ambiguity.

  Step 2: Create maximum 3 alternatives of analytical intents to choose from.
      - Do not include redundant intents, be focused.
      - Each analytical intent is for creating one single sql query.
      - Write each analytical intent using 1 sentence.
      - Mention specific column names, tables names, aggregation functions and filters from the database schema.
      - Mention only the useful info for creating sql queries.

  Step 3: Create a brief explanation in this format:
    1. One sentence explaining the ambiguity
    2. Present the 2-3 alternatives as clear options for the user to choose from

  Use simple, non-technical language. Be concise.
  """

    prompt = ChatPromptTemplate.from_messages([('user', sys_prompt)])
    chain = prompt | llm.with_structured_output(AmbiguityAnalysis)

    result = chain.invoke({
        'messages_log': extract_msg_content_from_history(state['messages_log']),
        'question': state['current_question'],
        'objects_documentation': state['objects_documentation']
    })

    # Update state (scenario D already set by clarification_check)
    state['generate_answer_details'] = {
        'ambiguity_explanation': result['ambiguity_explanation'],
        'agent_questions': result['agent_questions']
    }

    # Log clarification run
    action = AgentAction(tool='clarification', tool_input='', log='tool ran successfully')
    state['intermediate_steps'].append(action)

    # Control flow - go to generate_answer
    action = AgentAction(tool='generate_answer', tool_input='', log='')
    state['intermediate_steps'].append(action)

    return state


@tool
def query_genie(state: State):
    """
    Query Genie with the user's question
    Replaces extract_analytical_intent, create_sql, and execute_sql nodes
    """
    user_question = state['current_question']

    # Query Genie
    genie_result = query_genie_api(GENIE_SPACE_ID, user_question)

    # Populate current_sql_queries with Genie result
    # Structure: {'query': str, 'explanation': str, 'result': str}
    sql_query_entry = {
        'query': genie_result.get('generated_sql', ''),
        'explanation': genie_result.get('query_description', ''),  # Genie's interpretation
        'result': genie_result.get('sql_result').to_string() if genie_result.get('sql_result') is not None else ''
    }
    state['current_sql_queries'].append(sql_query_entry)

    # Log query_genie run
    action = AgentAction(tool='query_genie', tool_input=user_question, log='tool ran successfully')
    state['intermediate_steps'].append(action)

    # Control flow - go to generate_answer
    action = AgentAction(tool='generate_answer', tool_input='', log='')
    state['intermediate_steps'].append(action)

    return state


@tool
def add_assumptions(state: State):
    """Add key assumptions to the answer for transparency (scenario A only)"""

    key_assumptions = []

    # Generate assumptions from SQL queries using create_query_explanation
    for query_data in state['current_sql_queries']:
        if query_data.get('query'):
            explanation = create_query_explanation(query_data['query'])
            if explanation.get('explanation') and isinstance(explanation['explanation'], list):
                key_assumptions.extend(explanation['explanation'])

    # Store key_assumptions in generate_answer_details
    if 'generate_answer_details' not in state:
        state['generate_answer_details'] = {}
    state['generate_answer_details']['key_assumptions'] = key_assumptions

    # Format assumptions section
    def format_key_assumptions_for_prompt(key_assumptions: list[str]) -> str:
        """Format key assumptions into single section"""
        if not key_assumptions:
            return ""
        unique_assumptions = list(dict.fromkeys(key_assumptions))
        return "\n\n**Key Assumptions:**\n" + "\n".join([f"• {a}" for a in unique_assumptions])

    key_assumptions_section = format_key_assumptions_for_prompt(key_assumptions)

    if key_assumptions_section:
        current_content = state['llm_answer'].content
        content_with_assumptions = current_content + key_assumptions_section

        # Replace llm_answer with updated answer, as well as the AI answer in the messages log
        ai_msg = AIMessage(content=content_with_assumptions)
        state['llm_answer'] = ai_msg
        state['messages_log'][-1] = ai_msg

    # Log add_assumptions run
    action = AgentAction(tool='add_assumptions', tool_input='', log='tool ran successfully')
    state['intermediate_steps'].append(action)

    # Control flow - go to END
    action = AgentAction(tool=END, tool_input='', log='')
    state['intermediate_steps'].append(action)

    return state


@tool
def generate_answer(state: State):
    """generates the AI answer taking into consideration the explanation and the result of the sql query that was executed"""
    scenario = state['scenario']

    # Scenario-specific prompts (from original agent.py)
    if scenario == 'A':
        # Data query scenario - use SQL query results
        sys_prompt = """You are a decision support consultant helping users become more data-driven.
Your task is to continue the conversation from the last user message by guiding the users to answer their analytical goal.

Here is the conversation history with the user:
{messages_log}.
Latest user message:
{question}.
- Use the raw SQL results below to form your answer: {query_results}.
- Don't assume facts that are not backed up by the data.
- Include all details from these results.
- Suggest these next steps for the user: {agent_questions}.

Response guidelines:
  - Respond in clear, non-technical language.
  - Be concise.
  - Keep it simple and conversational.
  - If the question is smart, reinforce the user's question to build confidence.
    Example: "Great instinct to ask that - it's how data-savvy pros think!"
  - Ask the user which option they prefer from your suggested next steps.
  - Use warm, supportive closing that makes the user feel good.
    Example: "Keep up the great work!", "Have a great day ahead!"
"""

        # Generate agent questions for next steps
        agent_questions = generate_agent_questions(state)

        # Store in generate_answer_details
        state['generate_answer_details'] = {'agent_questions': agent_questions}

        invoke_params = {
            'messages_log': state['messages_log'],
            'question': state['current_question'],
            'query_results': format_sql_query_results_for_prompt(state['current_sql_queries']),
            'agent_questions': agent_questions
        }

    elif scenario == 'B':
        # Pleasantries scenario
        sys_prompt = """You are a decision support consultant helping users become more data-driven.
Your task is to continue the conversation from the last user message by guiding the users to answer their analytical goal.

Here is the conversation history with the user:
{messages_log}.

Latest user message:
{question}.

- Suggest these next steps for the user: {agent_questions}

Response guidelines:
  - Respond in clear, non-technical language.
  - Be concise.
  - Keep it simple and conversational.
  - Ask the user which option they prefer from your suggested next steps."""

        invoke_params = {
            'messages_log': state['messages_log'],
            'question': state['current_question'],
            'agent_questions': state.get('generate_answer_details', {}).get('agent_questions', [])
        }

    elif scenario == 'C':
        # Data unavailable scenario
        sys_prompt = """You are a decision support consultant helping users become more data-driven.
Your task is to continue the conversation from the last user message by guiding the users to answer their analytical goal.

Here is the conversation history with the user:
{messages_log}.

Latest user message:
{question}.

Unfortunately, the requested information from last prompt is not available in our database.

- Suggest these next steps for the user: {agent_questions}.

Response guidelines:
  - Respond in clear, non-technical language.
  - Be concise.
  - Keep it simple and conversational.
  - Ask the user which option they prefer from your suggested next steps."""

        invoke_params = {
            'messages_log': state['messages_log'],
            'question': state['current_question'],
            'agent_questions': state.get('generate_answer_details', {}).get('agent_questions', [])
        }

    else:  # scenario == 'D'
        # Ambiguous question scenario
        sys_prompt = """You are a decision support consultant helping users become more data-driven.
Your task is to continue the conversation from the last user message by guiding the users to answer their analytical goal.

Here is the conversation history with the user:
{messages_log}.

Latest user message:
{question}.

The last user prompt could be interpreted in multiple ways.
Explain the user this ambiguity reason: {ambiguity_explanation}.
And ask user to specify which of these analysis it wants: {agent_questions}.
Respond in clear, non-technical language.
Be concise.
Keep it simple and conversational."""

        invoke_params = {
            'messages_log': state['messages_log'],
            'question': state['current_question'],
            'ambiguity_explanation': state.get('generate_answer_details', {}).get('ambiguity_explanation', ''),
            'agent_questions': state.get('generate_answer_details', {}).get('agent_questions', [])
        }

    # Generate response
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("messages_log"),
        ('user', sys_prompt)
    ])
    llm_answer_chain = prompt | llm
    ai_msg = llm_answer_chain.invoke(invoke_params)

    # Update state
    state['llm_answer'] = ai_msg
    state['messages_log'].append(HumanMessage(state['current_question']))
    state['messages_log'].append(ai_msg)

    # Log generate_answer run
    action = AgentAction(tool='generate_answer', tool_input='', log='tool ran successfully')
    state['intermediate_steps'].append(action)

    # Control flow - add_assumptions if scenario A, otherwise END
    if state['scenario'] == 'A':
        next_tool_name = 'add_assumptions'
    else:
        next_tool_name = END

    action = AgentAction(tool=next_tool_name, tool_input='', log='')
    state['intermediate_steps'].append(action)

    return state

# control flow 

def router(state: State):
    """Router based on last intermediate step"""
    return state['intermediate_steps'][-1].tool


def run_control_flow(state: State):
    """Execute the node indicated by the last intermediate step"""
    tool_name = state['intermediate_steps'][-1].tool

    if tool_name == 'clarification_check':
        state = clarification_check.invoke({'state': state})
    elif tool_name == 'clarification':
        state = clarification.invoke({'state': state})
    elif tool_name == 'query_genie':
        state = query_genie.invoke({'state': state})
    elif tool_name == 'add_assumptions':
        state = add_assumptions.invoke({'state': state})
    elif tool_name == 'generate_answer':
        state = generate_answer.invoke({'state': state})

    return state


# Build the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("orchestrator", orchestrator)
graph.add_node("clarification_check", run_control_flow)
graph.add_node("clarification", run_control_flow)
graph.add_node("query_genie", run_control_flow)
graph.add_node("add_assumptions", run_control_flow)
graph.add_node("generate_answer", run_control_flow)

# Add edges
graph.add_edge(START, "orchestrator")

# Orchestrator routes to clarification_check or generate_answer
graph.add_conditional_edges(source='orchestrator', path=router)

# clarification_check routes to query_genie or clarification
graph.add_conditional_edges(source='clarification_check', path=router)

# clarification routes to generate_answer
graph.add_conditional_edges(source='clarification', path=router)

# query_genie routes to generate_answer
graph.add_conditional_edges(source='query_genie', path=router)

# generate_answer routes to add_assumptions (scenario A) or END (other scenarios)
graph.add_conditional_edges(source='generate_answer', path=router)

# add_assumptions routes to END
graph.add_conditional_edges(source='add_assumptions', path=router)

# Compile graph with checkpointer for memory
checkpointer = MemorySaver()
agent = graph.compile(checkpointer=checkpointer)
