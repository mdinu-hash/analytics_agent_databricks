# Databricks Genie Agent: Solution Accelerator

Short description with main benefits

Queries an already created genie space.

Uses Genie as NL2SQL engine, add guardrails.

## What makes it fit for use at scale

Checks for edge cases and decide if the question needs to query the database or not.

Questions asks for data not available in database
Clarification

Uses observability (MLflow)

A note about the importance of having custom orchestration

## Diagram

The agent graph has memory for multi-turn conversations.

Uses databricks-claude Databricks foundational model

## Setup

Update `GENIE_SPACE_ID` in agent.py and `objects_documentation` in utilities.py with your schema.

**Note:** Genie API's `include_serialized_space` parameter (for fetching schema) is in beta
(https://docs.databricks.com/api/workspace/genie/getspace)

Python SDK may not support it yet. Manually populate `objects_documentation` until SDK is updated.

## Evaluation

50 benchmark questions from real world questions

Used this evaluation framework to test it against Snowflake Cortex Analyst, Microsoft Power BI Copilot, Microsoft Data Fabric.

All solutions were connected to the same dataset, and had the same prompts as well as documentation for database.

<Show comparison matrix with ranks: 1,2,3 not %s>

Conclusion: This agent scored the highest score amongst x,y,z proved it's the most fit for production use at scale.

## how to use it
1. Clone this repository into your local environment
2. Replace the genieID with your genie space
3. Use the file XX to deploy the agent as a new foundational model