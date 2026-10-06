SELECT user_id
FROM purchases
WHERE purchase_date >= '2024-01-01'
  AND purchase_date < '2025-01-01'
GROUP BY user_id
HAVING COUNT(DISTINCT EXTRACT(MONTH FROM purchase_date))
       FILTER (
           WHERE EXTRACT(MONTH FROM purchase_date) IN (
               SELECT EXTRACT(MONTH FROM purchase_date)
               FROM purchases p2
               WHERE p2.user_id = purchases.user_id
                 AND p2.purchase_date >= '2024-01-01'
                 AND p2.purchase_date < '2025-01-01'
               GROUP BY EXTRACT(MONTH FROM purchase_date)
               HAVING COUNT(*) >= 2
           )
       ) = 12
ORDER BY user_id;