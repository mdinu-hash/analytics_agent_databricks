# Decision-Safe Analytics Agents for Enterprise AI
## An architectural framework built on Databricks Genie

This project wraps a Genie-powered analytics agent with explicit decision-safety policies required in enterprise and regulated environments.

# The problem

GenAI PoCs focused on analytics can be created fast with the most convenient native AI services available.

But they often break in regulated environments where detail and words matter.

This makes decision makers lose trust.

## Common production failures

### Hidden Assumptions
User: "What is the revenue?"
Agent answer 1: "$6M"
Agent answer 2: "$5M"

2 different answers to the same question. 
Both answers are technically correct, but $5M is YTD, $6M is rolling 12 months.
Agents did not disclose these assumptions: This creates confusion.

### Fabricated conclusions
User asks: "Which client segments have the most room to grow?"
Agent retrieves correct data, but concludes without justification that "all segments have most room to grow".

The SQL was correct, the conclusion was not.

### Lack of clarification
Many times, users don't know what is analytically possible, so they start with vague questions like:
"What's the top clients?"
In this cases, the agent shouldn't query anything.
Instead, it should follow up with clarification questions, for example:
Ex: "By top, do you mean by headcount or by revenue?"

### No data acknowledgment
When users ask about data that is not available, agent should explicitly state the requested information is not available.

### Undefined business terminology
Certain business terms are commonly used but not formally defined across the organization.
Example: User asks about "assets under management". 
This term may have no official, approved definition, refer to multiple asset types or be interpreted differently by different teams.

In these cases, the agent often queries the database in the most convenient way.

Instead, it should call out that the term may have multiple meanings & follow up with the user to clarify intent before querying.

### Incorrect Use of Key Terms
User: "What is the compensation for firm X?"
Agent replies: "The compensation for firm X is $Y. This amount of revenue means ..."
The problem is that the agent uses "revenue" not according to the internal revenue statement.
Agent should have enforced the usage of key terms.

### Missing default filters
User: "How many accounts we have?"
In natural language, this is expected to count only open accounts. 

### Acknowledgement of time frame
Date columns refresh frequently. 
If the agent becomes unaware of the data refresh, then when the user asks about
"EOM asset value", then doesn't know that "EOM" means for example, "December 2024".

### Aggregate metrics correctly
Certain metrics can't be aggregated across time, but can be aggregated across other dimensions.
For example: assets.
Agent should be enforced to not aggregate assets over time.

# Architecture framework for safe AI analytics.

These failures break users trust and makes genAI unsafe for decision making in regulated insustries where incorrect answers have consequences.

To tackle these challenges, I made an architecture framework which includes decision safety policies for AI analytics.

This architecture can be implemented on any tools or cloud vendors, as long as the tools provides the necessary capabilities for implementing the policies. 

For example, in this repo I implemented the framework using native Databricks services:
- Genie for NL2SQL engine.
- MLFlow for observability.
- Databricks foundational models.
- LangGraph for orchestration.

## Architecture Diagram

```
[START] ──► ┌─────────────────┐
            │ orchestrator    │
            │ (Decision Logic)│
            └────────┬────────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌────────────────────┐
│ Scenario B   │ │ Scenario C   │ │ clarification_check│
│ Pleasantries │ │ No Data      │ │ (Intent Analysis)  │
└──────┬───────┘ └──────┬───────┘ └─────────┬──────────┘
       │                │                    │
       │                │         ┌──────────┼──────────┐
       │                │         │ CLEAR    │ AMBIGUOUS│
       │                │         ▼          ▼
       │                │   ┌─────────────┐ ┌──────────────┐
       │                │   │ query_genie │ │ clarification│
       │                │   │ (Scenario A)│ │ (Scenario D) │
       │                │   │             │ │              │
       │                │   │ - Gen SQL   │ │ - Generate   │
       │                │   │ - Execute   │ │   Altern.    │
       │                │   │ - Get Data  │ │              │
       │                │   └──────┬──────┘ └──────┬───────┘
       │                │          │                │
       └────────────────┼──────────┘                │
                        │                           │
                        ▼                           ▼
              ┌──────────────────────────────────────────┐
              │ generate_answer                          │
              │ (Create Conversational Response)         │
              └─────────────┬────────────────────────────┘
                            │
                     ┌──────┴──────┐
                     │ Scenario A? │
                     └──────┬──────┘
                            │
                  ┌─────────┼─────────┐
                  │ Yes              │ No
                  ▼                  ▼
          ┌────────────────┐      [END]
          │ add_assumptions│
          │ (Query         │
          │  Explanations) │
          └────────┬───────┘
                   │
                   ▼
                [END]
```

## Decision-Safety Policies

These policies define when the agent is allowed to answer, how it must reason, and what it must disclose.

### Clarification before execution
The agent must not execute queries unless there is a clear analytical intent.
Clear intent means a single dominant metric & a single interpretable analytical method based on the available schema.
When a question is ambiguous, the agent blocks execution, explains why the question cannot be answered and proposes 2–3 alternative interpretations.
This prevents arbitrary unsafe assumptions.

### Explicit Acknowledgment of Missing Data
If the requested data is not available, the agent must explicitly state that it cannot answer the question and suggest viable alternatives.

### Assumptions disclosure
Every analytical answer must disclose its assumptions.
The agent is required to explain assumptions of the SQL they created (timeframe, filters, definitions) in non-technical language.
This ensures the anwers are understood by non-technical decision makers.

### Full Observability for Governance and Audit

Each user–agent interaction should store:
- user question
- agent’s answer
- generated SQL
- query results
- assumptions made by the agent

This enables to explain to governance and compliance why the agent gave a certain answer.

# Conclusion: Why it matters
This framework makes analytics agents decision-grade systems.

It is designed for environments where incorrect answers have financial and regulatory consequences.

# How to use this solution accelerator in your environment
1. Clone this repository into your local environment.

2. Replace the genieID with your genie space.

3. Update the database_schema file with your table and column documentation.

Note: The database_schema file includes available tables and columns from the genie space.
In the near term, these will be available to be fetched automatically from the genie space,
but since currently the Genie API's `include_serialized_space` parameter is in beta
(https://docs.databricks.com/api/workspace/genie/getspace)
I couldn't use it and made the database_schema.py file as a work-around for demo reasons.

4. Before deployment: test the agent using the notebook "test_agent".

5. Run each cell from notebook "deployment" to deploy the agent as a new foundational model.