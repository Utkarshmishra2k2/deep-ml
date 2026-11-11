WITH monthly AS (
    SELECT
        DATE_TRUNC('month', sale_date) AS month,
        SUM(amount)                    AS total
    FROM sales
    GROUP BY DATE_TRUNC('month', sale_date)
)
SELECT
    month,
    total,
    ROUND(
        (total - LAG(total) OVER (ORDER BY month)) * 100.0
        / NULLIF(LAG(total) OVER (ORDER BY month), 0),
        2
    ) AS pct_change
FROM monthly
ORDER BY month;
