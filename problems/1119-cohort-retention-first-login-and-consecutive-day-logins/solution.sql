-- your query
SELECT l.user_id, MIN(l.login_date) AS first_login
FROM logins AS l
GROUP BY l.user_id
HAVING EXISTS (
    SELECT 1
    FROM logins AS d
    WHERE d.user_id = l.user_id
      AND EXISTS (
          SELECT 1
          FROM logins AS next_day
          WHERE next_day.user_id = d.user_id
            AND next_day.login_date = d.login_date + INTERVAL '1 day'
      )
)
ORDER BY l.user_id ASC;
