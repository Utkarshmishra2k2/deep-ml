WITH first_month AS (
    SELECT
        user_id,
        date_trunc('month', MIN(activity_date))::DATE AS first_activity_month
    FROM activity
    GROUP BY user_id
),
monthly_users AS (
    SELECT DISTINCT
        user_id,
        date_trunc('month', activity_date)::DATE AS month
    FROM activity
)
SELECT
    strftime(mu.month, '%Y-%m') AS month,
    COUNT(*) FILTER (
        WHERE mu.month = fm.first_activity_month
    ) AS new_users,
    COUNT(*) FILTER (
        WHERE mu.month > fm.first_activity_month
    ) AS returning_users
FROM monthly_users mu
JOIN first_month fm USING (user_id)
GROUP BY mu.month
ORDER BY mu.month ASC;