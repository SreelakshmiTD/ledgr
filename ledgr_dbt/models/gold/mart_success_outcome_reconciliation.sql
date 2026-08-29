{{ config(materialized='table') }}

WITH real_calls_only AS (
    -- Exclude synthetic retry rows: they are FAILED by construction 
    -- (added by this project's own injection logic), not genuine 
    -- evidence about the real success/outcome_state relationship
    SELECT *
    FROM {{ source('silver', 'calls_enriched') }}
    WHERE is_synthetic_retry = false
),

session_level AS (
    SELECT
        task_id,
        MAX(success) AS session_success,
        COUNT(*) AS total_calls,
        SUM(CASE WHEN outcome_state = 'SUCCESS' THEN 1 ELSE 0 END) AS successful_calls
    FROM real_calls_only
    GROUP BY task_id
),

reconciled AS (
    SELECT
        *,
        CASE 
            WHEN session_success = true AND successful_calls = total_calls THEN true
            WHEN session_success = false AND successful_calls = 0 THEN true
            ELSE false
        END AS fully_aligned
    FROM session_level
)

SELECT
    COUNT(*) AS total_sessions,
    SUM(CASE WHEN fully_aligned THEN 0 ELSE 1 END) AS mismatched_sessions,
    ROUND(SUM(CASE WHEN fully_aligned THEN 0 ELSE 1 END) / COUNT(*) * 100, 2) AS mismatch_rate_pct
FROM reconciled
