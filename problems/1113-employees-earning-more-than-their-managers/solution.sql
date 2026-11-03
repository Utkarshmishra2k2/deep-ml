SELECT emp.name AS employee
FROM employees emp
INNER JOIN employees mgr 
    ON emp.manager_id = mgr.id
WHERE emp.salary > mgr.salary;
