{{ config(materialized='table') }}

SELECT
    model_request,
    harness,
    SUM(execution_cost_usd) AS total_cost_usd,
    SUM(CASE WHEN outcome_state = 'SUCCESS' THEN execution_cost_usd ELSE 0 END) AS successful_cost_usd,
    COUNT(CASE WHEN outcome_state = 'SUCCESS' THEN 1 END) AS successful_calls,
    COUNT(*) AS total_calls,
    SUM(CASE WHEN is_synthetic_retry = true THEN execution_cost_usd ELSE 0 END) AS wasted_retry_cost_usd,
    ROUND(
        SUM(execution_cost_usd) / NULLIF(COUNT(CASE WHEN outcome_state = 'SUCCESS' THEN 1 END), 0), 6
    ) AS cost_per_successful_outcome_including_waste,
    ROUND(
        SUM(CASE WHEN outcome_state = 'SUCCESS' THEN execution_cost_usd ELSE 0 END) 
        / NULLIF(COUNT(CASE WHEN outcome_state = 'SUCCESS' THEN 1 END), 0), 6
    ) AS cost_per_successful_outcome_success_only
FROM {{ source('silver', 'calls_enriched') }}
GROUP BY model_request, harness
