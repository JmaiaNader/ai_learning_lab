"""Read-only SQL tools for the contact-center demo dataset."""

from config.database import mysql_connection

TABLE_NAME = "contact_center_demo"


def _where_clause(start_date=None, end_date=None, channel=None, market=None):
    """Build a parameterized WHERE clause from supported filters only."""
    conditions = []
    params = []
    if start_date:
        conditions.append("work_date >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("work_date <= %s")
        params.append(end_date)
    if channel:
        conditions.append("channel = %s")
        params.append(channel)
    if market:
        conditions.append("market = %s")
        params.append(market)
    return (" WHERE " + " AND ".join(conditions) if conditions else "", params)


def total_calls_offered(start_date=None, end_date=None, channel=None, market=None):
    """Return offered/answered/abandoned totals and weighted KPIs from contact_center_demo.

    Optional filters: start_date and end_date (YYYY-MM-DD), channel (Voice/Chat/Email),
    and market (Qatar/UAE/Saudi Arabia). Read-only.
    """
    where_sql, params = _where_clause(start_date, end_date, channel, market)
    sql = f"""
        SELECT
            COUNT(*) AS intervals_count,
            MIN(work_date) AS first_date,
            MAX(work_date) AS last_date,
            COALESCE(SUM(offered), 0) AS total_calls_offered,
            COALESCE(SUM(answered), 0) AS total_answered,
            COALESCE(SUM(abandoned), 0) AS total_abandoned,
            ROUND(100.0 * SUM(answered_in_threshold) / NULLIF(SUM(answered), 0), 2) AS sla_percent,
            ROUND(SUM(aht * answered) / NULLIF(SUM(answered), 0), 2) AS weighted_avg_aht_seconds,
            ROUND(SUM(asa * answered) / NULLIF(SUM(answered), 0), 2) AS weighted_avg_asa_seconds
        FROM {TABLE_NAME}{where_sql}
    """
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        result = cur.fetchone()
        cur.close()
    return result or {"total_calls_offered": 0, "total_answered": 0, "total_abandoned": 0}


def market_summary(start_date=None, end_date=None, channel=None, market=None):
    """Summarize contact-center KPIs by market and channel, using weighted AHT/ASA and SLA."""
    where_sql, params = _where_clause(start_date, end_date, channel, market)
    sql = f"""
        SELECT
            market,
            channel,
            COUNT(*) AS intervals_count,
            MIN(work_date) AS first_date,
            MAX(work_date) AS last_date,
            SUM(offered) AS offered_total,
            SUM(answered) AS answered_total,
            SUM(abandoned) AS abandoned_total,
            ROUND(100.0 * SUM(answered_in_threshold) / NULLIF(SUM(answered), 0), 2) AS sla_percent,
            ROUND(SUM(aht * answered) / NULLIF(SUM(answered), 0), 2) AS weighted_avg_aht_seconds,
            ROUND(SUM(asa * answered) / NULLIF(SUM(answered), 0), 2) AS weighted_avg_asa_seconds
        FROM {TABLE_NAME}{where_sql}
        GROUP BY market, channel
        ORDER BY market, channel
        LIMIT 100
    """
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
    return rows


def daily_summary(start_date=None, end_date=None, channel=None, market=None):
    """Return daily offered volume and weighted KPIs, optionally filtered by date/channel/market."""
    where_sql, params = _where_clause(start_date, end_date, channel, market)
    sql = f"""
        SELECT
            work_date,
            channel,
            market,
            SUM(offered) AS offered_total,
            SUM(answered) AS answered_total,
            SUM(abandoned) AS abandoned_total,
            ROUND(100.0 * SUM(answered_in_threshold) / NULLIF(SUM(answered), 0), 2) AS sla_percent,
            ROUND(SUM(aht * answered) / NULLIF(SUM(answered), 0), 2) AS weighted_avg_aht_seconds,
            ROUND(SUM(asa * answered) / NULLIF(SUM(answered), 0), 2) AS weighted_avg_asa_seconds
        FROM {TABLE_NAME}{where_sql}
        GROUP BY work_date, channel, market
        ORDER BY work_date, market, channel
        LIMIT 1000
    """
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
    return rows



def compare_daily_aht(channel=None, market=None):
    """Compare weighted daily AHT for yesterday vs the same weekday last week.

    Uses MySQL CURDATE() so both dates are resolved against the database server's
    calendar: yesterday is CURDATE()-1 day, and the matching weekday last week is
    CURDATE()-8 days. Returns each available period independently so one missing
    date never hides the other period's data. Read-only.
    """
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT
                CURDATE() AS server_today,
                DATE_SUB(CURDATE(), INTERVAL 1 DAY) AS yesterday_date,
                DATE_SUB(CURDATE(), INTERVAL 8 DAY) AS same_weekday_last_week_date
        """)
        date_info = cur.fetchone() or {}
        yesterday_date = str(date_info.get("yesterday_date", ""))[:10]
        last_week_date = str(date_info.get("same_weekday_last_week_date", ""))[:10]

        conditions = ["work_date IN (%s, %s)"]
        params = [yesterday_date, last_week_date]
        if channel:
            conditions.append("channel = %s")
            params.append(channel)
        if market:
            conditions.append("market = %s")
            params.append(market)

        sql = f"""
            SELECT
                work_date,
                channel,
                market,
                COUNT(*) AS interval_count,
                COALESCE(SUM(offered), 0) AS offered_total,
                COALESCE(SUM(answered), 0) AS answered_total,
                ROUND(SUM(aht * answered) / NULLIF(SUM(answered), 0), 2)
                    AS weighted_avg_aht_seconds
            FROM {TABLE_NAME}
            WHERE {' AND '.join(conditions)}
            GROUP BY work_date, channel, market
            ORDER BY work_date DESC, market, channel
        """
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()

    expected_dates = {
        "yesterday": yesterday_date,
        "same_weekday_last_week": last_week_date,
    }
    groups = {}
    for row in rows:
        row_date = str(row.get("work_date", ""))[:10]
        period = "yesterday" if row_date == yesterday_date else (
            "same_weekday_last_week" if row_date == last_week_date else None
        )
        if not period:
            continue
        key = (row.get("channel"), row.get("market"))
        group = groups.setdefault(key, {
            "channel": row.get("channel"),
            "market": row.get("market"),
            "yesterday": None,
            "same_weekday_last_week": None,
        })
        aht = row.get("weighted_avg_aht_seconds")
        group[period] = {
            "date": row_date,
            "interval_count": int(row.get("interval_count") or 0),
            "offered_total": int(row.get("offered_total") or 0),
            "answered_total": int(row.get("answered_total") or 0),
            "weighted_avg_aht_seconds": round(float(aht), 2) if aht is not None else None,
        }

    # Keep an explicit result even when both requested dates have no matching rows,
    # provided the user specified one channel and market.
    if channel and market and (channel, market) not in groups:
        groups[(channel, market)] = {
            "channel": channel,
            "market": market,
            "yesterday": None,
            "same_weekday_last_week": None,
        }

    comparisons = []
    for _, group in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        yesterday = group["yesterday"]
        previous = group["same_weekday_last_week"]
        current_aht = yesterday.get("weighted_avg_aht_seconds") if yesterday else None
        previous_aht = previous.get("weighted_avg_aht_seconds") if previous else None
        change_seconds = round(current_aht - previous_aht, 2) if current_aht is not None and previous_aht is not None else None
        change_percent = round((change_seconds / previous_aht) * 100, 2) if change_seconds is not None and previous_aht not in (None, 0) else None
        missing = []
        if yesterday is None:
            missing.append("yesterday")
        if previous is None:
            missing.append("same_weekday_last_week")
        comparisons.append({
            **group,
            "aht_change_seconds": change_seconds,
            "aht_change_percent": change_percent,
            "missing_periods": missing,
        })

    return {
        "metric": "weighted_avg_aht_seconds",
        "server_today": str(date_info.get("server_today", ""))[:10],
        "expected_dates": expected_dates,
        "filters": {"channel": channel, "market": market},
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "message": None if any(c["yesterday"] or c["same_weekday_last_week"] for c in comparisons)
            else "No rows matched either comparison date and the selected filters.",
    }


def lowest_sla_interval(channel=None, market=None, date_scope="all_data"):
    """Return the single lowest-SLA interval, including market and timestamp.

    date_scope may be 'yesterday', 'today', 'all_data', or an ISO date (YYYY-MM-DD).
    Optional filters: channel and market. Rows with zero answered contacts are excluded
    because their SLA ratio is undefined. Read-only.
    """
    import re

    conditions = ["answered > 0"]
    params = []

    if date_scope == "yesterday":
        conditions.append("work_date = DATE_SUB(CURDATE(), INTERVAL 1 DAY)")
    elif date_scope == "today":
        conditions.append("work_date = CURDATE()")
    elif date_scope == "all_data" or date_scope is None:
        pass
    elif isinstance(date_scope, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_scope):
        conditions.append("work_date = %s")
        params.append(date_scope)
    else:
        raise ValueError("date_scope must be 'yesterday', 'today', 'all_data', or YYYY-MM-DD")

    if channel:
        conditions.append("channel = %s")
        params.append(channel)
    if market:
        conditions.append("market = %s")
        params.append(market)

    sql = f"""
        SELECT
            work_date,
            interval_start,
            channel,
            market,
            offered,
            answered,
            answered_in_threshold,
            abandoned,
            ROUND(100.0 * answered_in_threshold / NULLIF(answered, 0), 2) AS sla_percent,
            aht AS aht_seconds,
            asa AS asa_seconds
        FROM {TABLE_NAME}
        WHERE {" AND ".join(conditions)}
        ORDER BY answered_in_threshold / NULLIF(answered, 0) ASC,
                 work_date DESC, market ASC, interval_start ASC
        LIMIT 1
    """
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        result = cur.fetchone()
        cur.close()
    return result or {"message": "No intervals with answered contacts matched the selected filters."}

def forecast_history(channel=None, market=None, days=14):
    """Return read-only interval history for forecasting; does not generate predictions.

    If channel and/or market is provided, filter to that segment. With no segment filters,
    return one aggregated row per half-hour interval across all channels and markets.
    The days parameter is limited to 1..90 to keep the query bounded.
    """
    try:
        days = int(days)
    except (TypeError, ValueError) as exc:
        raise ValueError("days must be an integer between 1 and 90") from exc
    if not 1 <= days <= 90:
        raise ValueError("days must be between 1 and 90")

    conditions = ["work_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)", "work_date < CURDATE()"]
    params = [days]
    if channel:
        conditions.append("channel = %s")
        params.append(channel)
    if market:
        conditions.append("market = %s")
        params.append(market)
    where_sql = " WHERE " + " AND ".join(conditions)

    # Aggregate by timestamp so the default history is a single all-contact-center series.
    dimension_columns = []
    if channel:
        dimension_columns.append("channel")
    if market:
        dimension_columns.append("market")
    select_dimensions = (", " + ", ".join(dimension_columns)) if dimension_columns else ""
    group_dimensions = (", " + ", ".join(dimension_columns)) if dimension_columns else ""
    sql = f"""
        SELECT
            work_date,
            interval_start{select_dimensions},
            SUM(offered) AS offered,
            SUM(answered) AS answered,
            SUM(abandoned) AS abandoned,
            ROUND(SUM(aht * answered) / NULLIF(SUM(answered), 0), 2) AS aht,
            ROUND(SUM(asa * answered) / NULLIF(SUM(answered), 0), 2) AS asa,
            ROUND(SUM(answered_in_threshold) / NULLIF(SUM(answered), 0), 4) AS sla
        FROM {TABLE_NAME}{where_sql}
        GROUP BY work_date, interval_start{group_dimensions}
        ORDER BY work_date, interval_start
        LIMIT 5000
    """
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
    return rows



def sla_trend(channel=None, market=None, date_scope="all_data"):
    """Return interval-level SLA observations and trend summaries by channel/market.

    date_scope may be 'yesterday', 'today', 'all_data', or an ISO date (YYYY-MM-DD).
    Optional channel/market filters are supported. SLA percentage is
    100 * answered_in_threshold / answered. Read-only; zero-answer rows are excluded.
    """
    import re
    from collections import defaultdict

    conditions = ["answered > 0"]
    params = []

    if date_scope == "yesterday":
        # Use the MySQL server's calendar date, matching the proven lowest-SLA tool.
        conditions.append("work_date = DATE_SUB(CURDATE(), INTERVAL 1 DAY)")
    elif date_scope == "today":
        conditions.append("work_date = CURDATE()")
    elif date_scope == "all_data" or date_scope is None:
        pass
    elif isinstance(date_scope, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_scope):
        conditions.append("work_date = %s")
        params.append(date_scope)
    else:
        raise ValueError("date_scope must be 'yesterday', 'today', 'all_data', or YYYY-MM-DD")

    if channel:
        conditions.append("channel = %s")
        params.append(channel)
    if market:
        conditions.append("market = %s")
        params.append(market)

    sql = f"""
        SELECT
            work_date,
            interval_start,
            channel,
            market,
            offered,
            answered,
            answered_in_threshold,
            abandoned,
            ROUND(100.0 * answered_in_threshold / NULLIF(answered, 0), 2) AS sla_percent,
            aht AS aht_seconds,
            asa AS asa_seconds
        FROM {TABLE_NAME}
        WHERE {" AND ".join(conditions)}
        ORDER BY work_date, channel, market, interval_start
        LIMIT 10000
    """
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()

    # Generate deterministic summaries so the LLM can explain the trend without
    # needing to calculate the basic statistics from a long row list.
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("channel"), row.get("market"))].append(row)

    summaries = []
    for (row_channel, row_market), items in sorted(grouped.items(), key=lambda pair: (str(pair[0][0]), str(pair[0][1]))):
        items = sorted(items, key=lambda item: (str(item.get("work_date")), str(item.get("interval_start"))))
        first, last = items[0], items[-1]
        slas = [float(item["sla_percent"]) for item in items if item.get("sla_percent") is not None]
        min_row = min(items, key=lambda item: float(item["sla_percent"])) if slas else None
        max_row = max(items, key=lambda item: float(item["sla_percent"])) if slas else None
        summaries.append({
            "channel": row_channel,
            "market": row_market,
            "interval_count": len(items),
            "average_sla_percent": round(sum(slas) / len(slas), 2) if slas else None,
            "first_interval": str(first.get("interval_start")),
            "first_sla_percent": float(first["sla_percent"]) if first.get("sla_percent") is not None else None,
            "last_interval": str(last.get("interval_start")),
            "last_sla_percent": float(last["sla_percent"]) if last.get("sla_percent") is not None else None,
            "first_to_last_change_percentage_points": round(float(last["sla_percent"]) - float(first["sla_percent"]), 2) if first.get("sla_percent") is not None and last.get("sla_percent") is not None else None,
            "lowest_sla": ({"work_date": str(min_row.get("work_date")), "interval_start": str(min_row.get("interval_start")), "sla_percent": float(min_row["sla_percent"])} if min_row else None),
            "highest_sla": ({"work_date": str(max_row.get("work_date")), "interval_start": str(max_row.get("interval_start")), "sla_percent": float(max_row["sla_percent"])} if max_row else None),
        })

    return {
        "date_scope": date_scope,
        "channel_filter": channel,
        "market_filter": market,
        "row_count": len(rows),
        "trend_by_channel_market": summaries,
        "intervals": rows,
        "message": None if rows else "No SLA intervals matched the selected date scope and filters.",
    }

def agent_count():
    """Count active agents from the existing agentdetails table (if used by this database)."""
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT COUNT(*) AS active_agents FROM agentdetails WHERE active_status = %s",
            ("Active",),
        )
        result = cur.fetchone()
        cur.close()
    return result or {"active_agents": 0}

# One generic, safe query surface for all contact-center analytics requests.
# The older named functions above remain only for backward compatibility with
# Tool Calling 1 and any saved examples; LangGraph routes through this function.

_GENERIC_DIMENSIONS = {
    "work_date": "work_date",
    "interval_start": "interval_start",
    "channel": "channel",
    "market": "market",
    "week_start": "DATE_SUB(work_date, INTERVAL WEEKDAY(work_date) DAY) AS week_start",
    "month_start": "DATE_FORMAT(work_date, '%Y-%m-01') AS month_start",
}
_GENERIC_GROUP_BY = {
    "work_date": "work_date",
    "interval_start": "interval_start",
    "channel": "channel",
    "market": "market",
    "week_start": "DATE_SUB(work_date, INTERVAL WEEKDAY(work_date) DAY)",
    "month_start": "DATE_FORMAT(work_date, '%Y-%m-01')",
}

_GENERIC_METRICS = {
    "offered": ("COALESCE(SUM(offered), 0)", "offered_total"),
    "answered": ("COALESCE(SUM(answered), 0)", "answered_total"),
    "abandoned": ("COALESCE(SUM(abandoned), 0)", "abandoned_total"),
    "sla": ("ROUND(100.0 * SUM(answered_in_threshold) / NULLIF(SUM(answered), 0), 2)", "sla_percent"),
    "aht": ("ROUND(SUM(aht * answered) / NULLIF(SUM(answered), 0), 2)", "weighted_avg_aht_seconds"),
    "asa": ("ROUND(SUM(asa * answered) / NULLIF(SUM(answered), 0), 2)", "weighted_avg_asa_seconds"),
}

_DATE_PRESETS = {"today", "yesterday", "same_weekday_last_week", "yesterday_and_last_week_same_day", "this_week", "last_week", "last_7_days", "last_14_days", "last_30_days"}
_VALID_CHANNELS = {"Voice", "Chat", "Email"}
_VALID_MARKETS = {"Qatar", "UAE", "Saudi Arabia"}


def query_contact_center_data(
    metrics=None,
    dimensions=None,
    channel=None,
    market=None,
    start_date=None,
    end_date=None,
    dates=None,
    date_preset=None,
    order_by=None,
    sort_direction="ASC",
    limit=500,
    view_mode="auto",
):
    """Generic read-only analytics query against contact_center_demo.

    Supply any combination of allowlisted metrics, grouping dimensions, filters,
    explicit dates/date ranges or a supported relative date preset. It returns
    structured rows; the caller chooses summary, table or chart presentation.
    Metrics: offered, answered, abandoned, sla, aht, asa.
    Dimensions: work_date, interval_start, channel, market, week_start, month_start.
    Date presets: today, yesterday, last_7_days, last_14_days, last_30_days.
    """
    from datetime import date
    import re

    if metrics is None or metrics == "all":
        metrics = list(_GENERIC_METRICS)
    elif isinstance(metrics, str):
        metrics = [part.strip().lower() for part in metrics.split(",") if part.strip()]
    else:
        metrics = [str(item).strip().lower() for item in metrics]
    if not metrics:
        raise ValueError("Select at least one metric: offered, answered, abandoned, sla, aht, asa.")
    unknown_metrics = sorted(set(metrics) - set(_GENERIC_METRICS))
    if unknown_metrics:
        raise ValueError(f"Unsupported metrics: {unknown_metrics}. Allowed: {sorted(_GENERIC_METRICS)}")
    # Stable order and no duplicate select aliases.
    metrics = list(dict.fromkeys(metrics))

    if dimensions is None:
        dimensions = ["work_date", "channel", "market"]
    elif isinstance(dimensions, str):
        dimensions = [part.strip().lower() for part in dimensions.split(",") if part.strip()]
    else:
        dimensions = [str(item).strip().lower() for item in dimensions]
    unknown_dimensions = sorted(set(dimensions) - set(_GENERIC_DIMENSIONS))
    if unknown_dimensions:
        raise ValueError(f"Unsupported dimensions: {unknown_dimensions}. Allowed: {sorted(_GENERIC_DIMENSIONS)}")
    dimensions = list(dict.fromkeys(dimensions))

    if channel is not None and channel not in _VALID_CHANNELS:
        raise ValueError(f"Unsupported channel '{channel}'. Allowed: {sorted(_VALID_CHANNELS)}")
    if market is not None and market not in _VALID_MARKETS:
        raise ValueError(f"Unsupported market '{market}'. Allowed: {sorted(_VALID_MARKETS)}")

    def validate_iso_date(value, label):
        if value is None:
            return None
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(f"{label} must be YYYY-MM-DD")
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{label} is not a real date: {value}") from exc
        return value

    start_date = validate_iso_date(start_date, "start_date")
    end_date = validate_iso_date(end_date, "end_date")
    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    clean_dates = None
    if dates:
        if isinstance(dates, str):
            dates = [dates]
        if not isinstance(dates, (list, tuple, set)):
            raise ValueError("dates must be a list of YYYY-MM-DD strings")
        if len(dates) > 90:
            raise ValueError("At most 90 explicit dates may be queried at once")
        clean_dates = list(dict.fromkeys(validate_iso_date(item, "dates item") for item in dates))
        if any(item is None for item in clean_dates):
            raise ValueError("dates cannot contain empty values")
        if start_date or end_date or date_preset:
            raise ValueError("Use dates OR start_date/end_date OR date_preset, not a combination")

    if date_preset is not None:
        date_preset = str(date_preset).lower().strip()
        if date_preset not in _DATE_PRESETS:
            raise ValueError(f"Unsupported date_preset '{date_preset}'. Allowed: {sorted(_DATE_PRESETS)}")
        if start_date or end_date:
            raise ValueError("Use date_preset OR start_date/end_date, not both")

    try:
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer between 1 and 5000") from exc
    if not 1 <= limit <= 5000:
        raise ValueError("limit must be between 1 and 5000")

    view_mode = str(view_mode or "auto").lower()
    if view_mode not in {"auto", "summary", "table", "chart"}:
        raise ValueError("view_mode must be auto, summary, table, or chart")
    sort_direction = str(sort_direction or "ASC").upper()
    if sort_direction not in {"ASC", "DESC"}:
        raise ValueError("sort_direction must be ASC or DESC")

    conditions = []
    params = []
    if clean_dates:
        conditions.append("work_date IN (" + ", ".join(["%s"] * len(clean_dates)) + ")")
        params.extend(clean_dates)
    elif date_preset == "today":
        conditions.append("work_date = CURDATE()")
    elif date_preset == "yesterday":
        conditions.append("work_date = DATE_SUB(CURDATE(), INTERVAL 1 DAY)")
    elif date_preset == "yesterday_and_last_week_same_day":
        conditions.append("work_date IN (DATE_SUB(CURDATE(), INTERVAL 1 DAY), DATE_SUB(CURDATE(), INTERVAL 8 DAY))")
    elif date_preset == "same_weekday_last_week":
        conditions.append("work_date = DATE_SUB(CURDATE(), INTERVAL 7 DAY)")
    elif date_preset == "this_week":
        conditions.append("work_date >= DATE_SUB(CURDATE(), INTERVAL WEEKDAY(CURDATE()) DAY) AND work_date < CURDATE()")
    elif date_preset == "last_week":
        conditions.append("work_date >= DATE_SUB(DATE_SUB(CURDATE(), INTERVAL WEEKDAY(CURDATE()) DAY), INTERVAL 7 DAY) AND work_date < DATE_SUB(CURDATE(), INTERVAL WEEKDAY(CURDATE()) DAY)")
    elif date_preset == "last_7_days":
        conditions.append("work_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) AND work_date < CURDATE()")
    elif date_preset == "last_14_days":
        conditions.append("work_date >= DATE_SUB(CURDATE(), INTERVAL 14 DAY) AND work_date < CURDATE()")
    elif date_preset == "last_30_days":
        conditions.append("work_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) AND work_date < CURDATE()")
    else:
        if start_date:
            conditions.append("work_date >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("work_date <= %s")
            params.append(end_date)
    if channel:
        conditions.append("channel = %s")
        params.append(channel)
    if market:
        conditions.append("market = %s")
        params.append(market)

    select_parts = [_GENERIC_DIMENSIONS[name] for name in dimensions]
    metric_aliases = {}
    for metric in metrics:
        expression, alias = _GENERIC_METRICS[metric]
        select_parts.append(f"{expression} AS {alias}")
        metric_aliases[metric] = alias
    select_parts.append("COUNT(*) AS source_interval_count")

    where_sql = " WHERE " + " AND ".join(conditions) if conditions else ""
    group_sql = " GROUP BY " + ", ".join(_GENERIC_GROUP_BY[name] for name in dimensions) if dimensions else ""
    allowed_order_fields = set(dimensions) | set(metric_aliases.values())
    if order_by:
        requested_order = str(order_by).strip()
        # Accept both friendly metric names and the SQL output aliases.
        if requested_order in metric_aliases:
            requested_order = metric_aliases[requested_order]
        if requested_order not in allowed_order_fields:
            raise ValueError(f"order_by must be one of {sorted(allowed_order_fields)}")
        order_sql = f" ORDER BY {requested_order} {sort_direction}"
    elif dimensions:
        order_sql = " ORDER BY " + ", ".join(dimensions)
    else:
        order_sql = ""

    sql = f"SELECT {', '.join(select_parts)} FROM {TABLE_NAME}{where_sql}{group_sql}{order_sql} LIMIT {limit}"
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall() if dimensions else [cur.fetchone()]
        cur.close()

    rows = [dict(row) for row in (rows or []) if row is not None]
    # A global aggregate returns one MySQL row even when no rows matched.
    if not dimensions and rows and int(rows[0].get("source_interval_count") or 0) == 0:
        rows = []
    aliases = [metric_aliases[item] for item in metrics]
    columns = list(dimensions) + aliases + ["source_interval_count"]
    if view_mode == "auto":
        view_mode = "summary" if not dimensions else "table"
    resolved_spec = {
        "metrics": metrics,
        "dimensions": dimensions,
        "filters": {"channel": channel, "market": market},
        "start_date": start_date,
        "end_date": end_date,
        "dates": clean_dates,
        "date_preset": date_preset,
        "order_by": order_by,
        "sort_direction": sort_direction,
        "limit": limit,
        "view_mode": view_mode,
    }
    return {
        "tool": "query_contact_center_data",
        "query_spec": resolved_spec,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "has_data": bool(rows),
        "message": None if rows else "No rows matched the selected dates and filters.",
        "requested_view": view_mode,
    }
