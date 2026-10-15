# Generic contact-center query patch (Tool Calling 2)

Replace the three files in the matching folders:
- tools/mysql_tools.py
- tools/registry.py
- backend/services/langgraph_service.py

This introduces one allowlisted, parameterized query tool for contact-center data. Existing specialized functions remain as compatibility wrappers for Tool Calling 1 and older code, but LangGraph routes KPI questions through `query_contact_center_data`.

Supported dimensions: work_date, interval_start, channel, market, week_start, month_start.
Supported metrics: offered, answered, abandoned, SLA %, weighted AHT, weighted ASA.
Supported date filters: explicit date list, date range, today, yesterday, same weekday last week, last/this week, and last 7/14/30 complete days. Values and filters are parameterized; SQL identifiers come only from allowlists.

After copying, restart FastAPI and Streamlit. Run your existing tests and then test in Tool Calling 2.
