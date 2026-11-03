WITH user_cohorts AS (
    SELECT
        user_id,
        MIN(date_trunc('week', event_date)::DATE) AS cohort_week
    FROM events
    GROUP BY user_id
),
cohorts AS (
    SELECT cohort_week, COUNT(*) AS cohort_size
    FROM user_cohorts
    GROUP BY cohort_week
),
week_grid AS (
    SELECT DISTINCT date_trunc('week', event_date)::DATE AS week
    FROM events
),
active_users AS (
    SELECT DISTINCT
        user_id,
        date_trunc('week', event_date)::DATE AS week
    FROM events
),
retained AS (
    SELECT
        uc.cohort_week,
        au.week,
        COUNT(DISTINCT uc.user_id) AS retained_users
    FROM user_cohorts uc
    JOIN active_users au ON au.user_id = uc.user_id
    GROUP BY uc.cohort_week, au.week
),
retention_rows AS (
    SELECT
        c.cohort_week,
        (date_diff('day', c.cohort_week, g.week) // 7)::INTEGER AS week_offset,
        c.cohort_size,
        COALESCE(r.retained_users, 0) AS retained_users,
        ROUND(COALESCE(r.retained_users, 0) * 1.0 / c.cohort_size, 4) AS retention
    FROM cohorts c
    JOIN week_grid g ON g.week >= c.cohort_week
    LEFT JOIN retained r
        ON r.cohort_week = c.cohort_week
       AND r.week = g.week
),
with_previous AS (
    SELECT
        *,
        LAG(retention) OVER (
            PARTITION BY cohort_week
            ORDER BY week_offset
        ) AS previous_retention
    FROM retention_rows
)
SELECT
    strftime(cohort_week, '%Y-%m-%d') AS cohort_week,
    week_offset,
    cohort_size,
    retained_users,
    retention,
    CASE
        WHEN previous_retention IS NULL THEN 0
        WHEN retention < 0.9 * previous_retention THEN 1
        ELSE 0
    END AS drop_flag
FROM with_previous
ORDER BY cohort_week, week_offset;
