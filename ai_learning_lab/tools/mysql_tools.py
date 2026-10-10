from config.database import mysql_connection

def total_calls_offered():
    """Return the read-only total number of calls offered across market intervals."""
    sql = "SELECT COALESCE(SUM(offered), 0) AS total_calls_offered FROM marketleveldataset"
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql)
        result = cur.fetchone(); cur.close()
    return result or {"total_calls_offered": 0}

def market_summary():
    """Read-only aggregate metrics from marketleveldataset."""
    sql = """SELECT market, channel, COUNT(*) AS intervals_count,
        SUM(offered) AS offered_total, SUM(handled) AS handled_total,
        ROUND(AVG(aht),2) AS avg_aht, ROUND(AVG(sla),2) AS avg_sla
        FROM marketleveldataset GROUP BY market, channel ORDER BY market, channel LIMIT 100"""
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True); cur.execute(sql)
        rows = cur.fetchall(); cur.close()
    return rows

def agent_count():
    """Count active agents from agentdetails."""
    with mysql_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) AS active_agents FROM agentdetails WHERE active_status = %s", ("Active",))
        result = cur.fetchone(); cur.close()
    return result or {"active_agents": 0}
