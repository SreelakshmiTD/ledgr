{{ config(materialized='table') }}

WITH daily_metrics AS (
    SELECT
        model_request,
        DATE(start_time) AS call_date,
        SUM(execution_cost_usd) AS daily_total_cost,
        COUNT(CASE WHEN outcome_state = 'SUCCESS' THEN 1 END) AS daily_successful_calls
    FROM {{ source('silver', 'calls_enriched') }}
    GROUP BY model_request, DATE(start_time)
),

daily_cost_per_success AS (
    SELECT
        model_request,
        call_date,
        daily_total_cost,
        daily_successful_calls,
        ROUND(daily_total_cost / NULLIF(daily_successful_calls, 0), 6) AS cost_per_success
    FROM daily_metrics
),

with_rolling_baseline AS (
    SELECT
        *,
        AVG(cost_per_success) OVER (
            PARTITION BY model_request
            ORDER BY call_date
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS rolling_7day_mean,
        STDDEV(cost_per_success) OVER (
            PARTITION BY model_request
            ORDER BY call_date
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS rolling_7day_stddev,
        COUNT(*) OVER (
            PARTITION BY model_request
            ORDER BY call_date
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS days_in_window
    FROM daily_cost_per_success
)

SELECT
    model_request,
    call_date,
    daily_total_cost,
    daily_successful_calls,
    cost_per_success,
    rolling_7day_mean,
    rolling_7day_stddev,
    days_in_window,
    CASE 
        WHEN days_in_window < 3 THEN FALSE  -- too few prior days for a reliable baseline
        WHEN rolling_7day_mean IS NULL OR rolling_7day_stddev IS NULL OR rolling_7day_stddev = 0 
            THEN FALSE
        WHEN ABS(cost_per_success - rolling_7day_mean) > (2 * rolling_7day_stddev) 
            THEN TRUE
        ELSE FALSE
    END AS is_anomaly,
    CASE WHEN days_in_window < 3 THEN TRUE ELSE FALSE END AS baseline_unreliable
FROM with_rolling_baseline
ORDER BY model_request, call_date
