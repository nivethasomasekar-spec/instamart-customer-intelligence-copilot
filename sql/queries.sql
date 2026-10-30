-- ====================================================================
-- sql/queries.sql -- Part A Task 6: Analytical SQL Queries
-- ====================================================================
-- Database: data/instamart.db
-- Rule: Unless stated otherwise, only completed orders count as revenue or spend.

-- --------------------------------------------------------------------
-- Q1: WHERE then HAVING
-- Keep only completed orders with order_amount > 500 (WHERE),
-- group by customer, and keep customers whose total of those orders
-- exceeds 8000 (HAVING).
-- Reference: 30 customers.
-- --------------------------------------------------------------------
-- [Q1]
SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;


-- --------------------------------------------------------------------
-- Q2: Three-Table JOIN
-- Revenue by city and product category (join orders, customers, products).
-- Reference: 30 rows; revenues sum to INR 2,998,668.80;
-- largest cell is Bengaluru / Electronics at INR 233,909.45.
-- --------------------------------------------------------------------
-- [Q2]
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- --------------------------------------------------------------------
-- Q3: Scalar Subquery
-- Products priced above the average catalogue price.
-- Reference: 11 products.
-- --------------------------------------------------------------------
-- [Q3]
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;


-- --------------------------------------------------------------------
-- Q4: LEFT JOIN
-- Customers who have no completed order.
-- Reference: 25 customers.
-- --------------------------------------------------------------------
-- [Q4]
SELECT
    c.customer_id,
    c.city,
    c.signup_date
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- --------------------------------------------------------------------
-- Q5: Common Table Expressions (CTE)
-- One CTE computes each customer's total completed spend and a second
-- computes the average of those totals; the main query returns customers
-- whose total is above the average.
-- Reference: 550 customers; average INR 2,552.06 (rounded to 2 decimals).
-- --------------------------------------------------------------------
-- [Q5]
WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT
        AVG(total_spend) AS avg_total_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend, 2) AS total_spend
FROM customer_spend cs
CROSS JOIN avg_spend
WHERE cs.total_spend > avg_spend.avg_total_spend
ORDER BY cs.total_spend DESC;


-- --------------------------------------------------------------------
-- Q6: Ranking Window Functions
-- For each customer with completed spend, compute total spend and then
-- RANK(), DENSE_RANK() and ROW_NUMBER() using
-- OVER (PARTITION BY city ORDER BY total_spend DESC).
-- Reference: top customer per city is:
--   Bengaluru C0082 (12,856.20)
--   Chennai   C0930 (10,151.40)
--   Delhi     C0291 (13,742.60)
--   Hyderabad C0743 (12,730.05)
--   Mumbai    C0693 (14,766.55)
--   Pune      C0426 (10,377.10)
-- Exactly 8 rows have RANK() different from ROW_NUMBER().
-- --------------------------------------------------------------------
-- [Q6]
WITH customer_city_spend AS (
    SELECT
        c.customer_id,
        c.city,
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_dense_rank,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_row_number
FROM customer_city_spend
ORDER BY city, spend_rank, customer_id;


-- --------------------------------------------------------------------
-- Q7: Running Total
-- Daily completed revenue (day is first 10 chars of order_ts) with a
-- cumulative revenue column computed as SUM(revenue) OVER (ORDER BY day).
-- Reference: 183 daily rows; last cumulative value is INR 2,998,668.80.
-- --------------------------------------------------------------------
-- [Q7]
WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS order_day,
        ROUND(SUM(order_amount), 2) AS daily_rev
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    order_day,
    daily_rev,
    ROUND(SUM(daily_rev) OVER (ORDER BY order_day), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY order_day;

