---
name: work-log-reporter
description: Write a structured work log report after completing a task loop.Trigger after: full EDA, data pipeline, testing, review, refactor, or documentation tasks are completed end-to-end including tests passing. Posts report to Notion Work Log database.
---

# Work Log Reporter

## When to use this skill
Use this skill automatically after completing a full task loop:
- EDA script written + tests passing + code review done
- Data pipeline built + tests passing + reviewed
- Refactor completed + regression tests passing
- Any significant task that went through write → test → review → commit

## Report format

Write a concise work log entry covering:

### 1. Task name (one line)
Clear, specific name. Examples:
- "SAP Velocity EDA — initial data profiling"
- "BAPI connection handler — refactor + tests"
- "Data cleaning pipeline — null handling"

### 2. Summary (3-5 sentences)
- What was built or changed
- Key findings or decisions made
- Any blockers encountered and how resolved
- Performance or quality metrics if relevant

### 3. Files changed
List every file created or modified:
- src/eda/eda_sap_velocity.py (created)
- tests/test_eda.py (created)
- config/sap_config.yml (modified)

### 4. Test status
Report: All Passing / Some Failing / Not Run
Include coverage % if available.

### 5. Next steps
2-3 concrete next actions.

## Output
After writing the report in chat, also post it as a new entry to:
Notion Work Log: https://www.notion.so/ec1a5066f7644e3b805be43778dbad1d?v=2843abcbe2ba4fd5bfd93a96fced2b56

Use the Notion MCP tool to create a page in the database with all fields filled.