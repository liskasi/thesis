# 1) CUSTOMER'S TOTAL NUMBER OF PURCHASED PRODUCTS
SELECT
    filtered_orders.customer_email,
    SUM(sales_order_item.qty_ordered) AS TotalQty
FROM
    filtered_orders
        JOIN sales_order_item on sales_order_item.order_id = filtered_orders.entity_id
        JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id
WHERE
        filtered_orders.customer_email IN (
        SELECT
            customer_email
        FROM
            (
                SELECT
                    customer_email,
                    COUNT(DISTINCT product_id) as num_products
                FROM
                    filtered_orders
                        JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                GROUP BY
                    customer_email
            ) customer_products
        WHERE
                num_products > 3
    )
GROUP BY
    filtered_orders.customer_email
HAVING
        COUNT(
                DISTINCT sales_order_item.product_id
            ) > 3
ORDER BY
    TotalQty ASC;

# 2) AVERAGE TOTAL QUANTITY - 10.44537718
SELECT
    AVG(TotalProductQty) AS AvgTotalProductQty
FROM
    (
        SELECT
            filtered_orders.customer_email,
            SUM(sales_order_item.qty_ordered) AS TotalProductQty
        FROM
            filtered_orders
                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id
        WHERE
                filtered_orders.customer_email IN (
                SELECT
                    customer_email
                FROM
                    (
                        SELECT
                            customer_email,
                            COUNT(DISTINCT product_id) as num_products
                        FROM
                            filtered_orders
                                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                        GROUP BY
                            customer_email
                    ) customer_products
                WHERE
                        num_products > 3
            )
        GROUP BY
            filtered_orders.customer_email
        HAVING
                COUNT(
                        DISTINCT sales_order_item.product_id
                    ) > 3
    ) AS CustomerProducts;

# 3) STANDARD DEVIATION OF TOTAL QUANTITY - 33.58417053
SELECT
    STDDEV(TotalProductQty) AS AvgTotalProductQty
FROM
    (
        SELECT
            filtered_orders.customer_email,
            SUM(sales_order_item.qty_ordered) AS TotalProductQty
        FROM
            filtered_orders
                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id
        WHERE
                filtered_orders.customer_email IN (
                SELECT
                    customer_email
                FROM
                    (
                        SELECT
                            customer_email,
                            COUNT(DISTINCT product_id) as num_products
                        FROM
                            filtered_orders
                                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                        GROUP BY
                            customer_email
                    ) customer_products
                WHERE
                        num_products > 3
            )
        GROUP BY
            filtered_orders.customer_email
        HAVING
                COUNT(
                        DISTINCT sales_order_item.product_id
                    ) > 3
    ) AS CustomerProducts;

# 4) CUSTOMER'S TOTAL NUMBER OF UNIQUE PURCHASED PRODUCTS
SELECT
    filtered_orders.customer_email,
    COUNT(
            DISTINCT sales_order_item.product_id
        ) AS DistinctProducts
FROM
    filtered_orders
        JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
        JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id # WHERE sales_order_item.product_id IN (
    WHERE
            filtered_orders.customer_email IN (
            SELECT
                customer_email
            FROM
                (
                    SELECT
                        customer_email,
                        COUNT(DISTINCT product_id) as num_products
                    FROM
                        filtered_orders
                            JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                    GROUP BY
                        customer_email
                ) customer_products
            WHERE
                    num_products > 3
        )
      AND sales_order_item.product_id IN (
        SELECT
            product_id
        FROM
            (
                SELECT
                    product_id,
                    COUNT(*) AS num_orders
                FROM
                    sales_order_item
                GROUP BY
                    product_id
            ) AS product_orders
        WHERE
                num_orders > 3
    )
GROUP BY
    filtered_orders.customer_email
HAVING
        DistinctProducts > 3;

# 5) AVERAGE TOTAL NUMBER OF UNIQUE PURCHASED PRODUCTS - 6.9562/6.9461
SELECT
    AVG(DistinctProducts) AS AvgDistinctProducts
FROM
    (
        SELECT
            filtered_orders.customer_email,
            COUNT(
                    DISTINCT sales_order_item.product_id
                ) AS DistinctProducts
        FROM
            filtered_orders
                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id
        WHERE
                filtered_orders.customer_email IN (
                SELECT
                    customer_email
                FROM
                    (
                        SELECT
                            customer_email,
                            COUNT(DISTINCT product_id) as num_products
                        FROM
                            filtered_orders
                                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                        GROUP BY
                            customer_email
                    ) customer_products
                WHERE
                        num_products > 3
            )
          AND sales_order_item.product_id IN (
            SELECT
                product_id
            FROM
                (
                    SELECT
                        product_id,
                        COUNT(*) AS num_orders
                    FROM
                        sales_order_item
                    GROUP BY
                        product_id
                ) AS product_orders
            WHERE
                    num_orders > 3
        )
        GROUP BY
            filtered_orders.customer_email
        HAVING
                COUNT(
                        DISTINCT sales_order_item.product_id
                    ) > 3
    ) AS CustomerProducts;

# 6) STANDARD DEVIATION OF UNIQUE PURCHASED PRODUCTS - 5.7265/5.7099
SELECT
    STDDEV(DistinctProducts) AS AvgDistinctProducts
FROM
    (
        SELECT
            filtered_orders.customer_email,
            COUNT(
                    DISTINCT sales_order_item.product_id
                ) AS DistinctProducts
        FROM
            filtered_orders
                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id
        WHERE
                filtered_orders.customer_email IN (
                SELECT
                    customer_email
                FROM
                    (
                        SELECT
                            customer_email,
                            COUNT(DISTINCT product_id) as num_products
                        FROM
                            filtered_orders
                                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                        GROUP BY
                            customer_email
                    ) customer_products
                WHERE
                        num_products > 3
            )
          AND sales_order_item.product_id IN (
            SELECT
                product_id
            FROM
                (
                    SELECT
                        product_id,
                        COUNT(*) AS num_orders
                    FROM
                        sales_order_item
                    GROUP BY
                        product_id
                ) AS product_orders
            WHERE
                    num_orders > 3
        )
        GROUP BY
            filtered_orders.customer_email
        HAVING
                DistinctProducts > 3
    ) AS CustomerProducts;

#  7) PRODUCT'S TOTAL PURCHASE NUMBER
SELECT
    sales_order_item.product_id,
    SUM(sales_order_item.qty_ordered) AS TotalQty
FROM
    sales_order_item
        JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id
        JOIN filtered_orders on sales_order_item.order_id = filtered_orders.entity_id
WHERE
        filtered_orders.customer_email IN (
        SELECT
            customer_email
        FROM
            (
                SELECT
                    customer_email,
                    COUNT(DISTINCT product_id) as num_products
                FROM
                    filtered_orders
                        JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                GROUP BY
                    customer_email
            ) customer_products
        WHERE
                num_products > 3
    )
  AND sales_order_item.product_id IN (
    SELECT
        product_id
    FROM
        (
            SELECT
                product_id,
                COUNT(*) AS num_orders
            FROM
                sales_order_item
            GROUP BY
                product_id
        ) AS product_orders
    WHERE
            num_orders > 3
)
GROUP BY
    sales_order_item.product_id
HAVING
        TotalQty > 3
ORDER BY
    TotalQty DESC;

# 8) AVERAGE OF PRODUCT'S TOTAL PURCHASE NUMBER - 74.95774648/75.07551867
SELECT
    AVG(TotalQty) AS AvgTotalQty
FROM
    (
        SELECT
            sales_order_item.product_id,
            SUM(sales_order_item.qty_ordered) AS TotalQty
        FROM
            sales_order_item
                JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id
                JOIN filtered_orders on sales_order_item.order_id = filtered_orders.entity_id
        WHERE
                filtered_orders.customer_email IN (
                SELECT
                    customer_email
                FROM
                    (
                        SELECT
                            customer_email,
                            COUNT(DISTINCT product_id) as num_products
                        FROM
                            filtered_orders
                                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                        GROUP BY
                            customer_email
                    ) customer_products
                WHERE
                        num_products > 3
            )
          AND sales_order_item.product_id IN (
            SELECT
                product_id
            FROM
                (
                    SELECT
                        product_id,
                        COUNT(*) AS num_orders
                    FROM
                        sales_order_item
                    GROUP BY
                        product_id
                ) AS product_orders
            WHERE
                    num_orders > 3
        )
        GROUP BY
            sales_order_item.product_id
        HAVING
                TotalQty > 3
    ) AS CustomerProducts;

# 9) STANDARD DEVIATION OF PRODUCT'S TOTAL PURCHASE NUMBER - 252.7444994/252.93761245
SELECT
    STDDEV(TotalQty) AS AvgTotalQty
FROM
    (
        SELECT
            sales_order_item.product_id,
            SUM(sales_order_item.qty_ordered) AS TotalQty
        FROM
            sales_order_item
                JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id
                JOIN filtered_orders on sales_order_item.order_id = filtered_orders.entity_id
        WHERE
                filtered_orders.customer_email IN (
                SELECT
                    customer_email
                FROM
                    (
                        SELECT
                            customer_email,
                            COUNT(DISTINCT product_id) as num_products
                        FROM
                            filtered_orders
                                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                        GROUP BY
                            customer_email
                    ) customer_products
                WHERE
                        num_products > 3
            )
          AND sales_order_item.product_id IN (
            SELECT
                product_id
            FROM
                (
                    SELECT
                        product_id,
                        COUNT(*) AS num_orders
                    FROM
                        sales_order_item
                    GROUP BY
                        product_id
                ) AS product_orders
            WHERE
                    num_orders > 3
        )
        GROUP BY
            sales_order_item.product_id
        HAVING
                TotalQty > 3
    ) AS CustomerProducts;

# 10) NUMBER OF UNIQUE USER PURCHASES PER PRODUCT
SELECT
    sales_order_item.product_id,
    COUNT(
            DISTINCT filtered_orders.customer_email
        ) AS UniqueTotalQty
FROM
    sales_order_item
        JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id
        JOIN filtered_orders on sales_order_item.order_id = filtered_orders.entity_id
WHERE
        filtered_orders.customer_email IN (
        SELECT
            customer_email
        FROM
            (
                SELECT
                    customer_email,
                    COUNT(DISTINCT product_id) as num_products
                FROM
                    filtered_orders
                        JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                GROUP BY
                    customer_email
            ) customer_products
        WHERE
                num_products > 3
    )
  AND sales_order_item.product_id IN (
    SELECT
        product_id
    FROM
        (
            SELECT
                product_id,
                COUNT(*) AS num_orders
            FROM
                sales_order_item
            GROUP BY
                product_id
        ) AS product_orders
    WHERE
            num_orders > 3
)
GROUP BY
    sales_order_item.product_id
ORDER BY
    UniqueTotalQty DESC;

# 11) AVERAGE OF UNIQUE USER PURCHASES PER PRODUCT - 37.7567
SELECT
    AVG(UniqueTotalQty) AS AvgTotalQty
FROM
    (
        SELECT
            sales_order_item.product_id,
            COUNT(
                    DISTINCT filtered_orders.customer_email
                ) AS UniqueTotalQty
        FROM
            sales_order_item
                JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id
                JOIN filtered_orders on sales_order_item.order_id = filtered_orders.entity_id
        WHERE
                filtered_orders.customer_email IN (
                SELECT
                    customer_email
                FROM
                    (
                        SELECT
                            customer_email,
                            COUNT(DISTINCT product_id) as num_products
                        FROM
                            filtered_orders
                                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                        GROUP BY
                            customer_email
                    ) customer_products
                WHERE
                        num_products > 3
            )
          AND sales_order_item.product_id IN (
            SELECT
                product_id
            FROM
                (
                    SELECT
                        product_id,
                        COUNT(*) AS num_orders
                    FROM
                        sales_order_item
                    GROUP BY
                        product_id
                ) AS product_orders
            WHERE
                    num_orders > 3
        )
        GROUP BY
            sales_order_item.product_id
    ) AS CustomerProducts;

# 12) STANDARD DEVIATION OF UNIQUE USER PURCHASES PER PRODUCT - 192.165
SELECT
    STDDEV(UniqueTotalQty) AS AvgTotalQty
FROM
    (
        SELECT
            sales_order_item.product_id,
            COUNT(
                    DISTINCT filtered_orders.customer_email
                ) AS UniqueTotalQty
        FROM
            sales_order_item
                JOIN catalog_product_entity ON sales_order_item.product_id = catalog_product_entity.entity_id
                JOIN filtered_orders on sales_order_item.order_id = filtered_orders.entity_id
        WHERE
                filtered_orders.customer_email IN (
                SELECT
                    customer_email
                FROM
                    (
                        SELECT
                            customer_email,
                            COUNT(DISTINCT product_id) as num_products
                        FROM
                            filtered_orders
                                JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                        GROUP BY
                            customer_email
                    ) customer_products
                WHERE
                        num_products > 3
            )
          AND sales_order_item.product_id IN (
            SELECT
                product_id
            FROM
                (
                    SELECT
                        product_id,
                        COUNT(*) AS num_orders
                    FROM
                        sales_order_item
                    GROUP BY
                        product_id
                ) AS product_orders
            WHERE
                    num_orders > 3
        )
        GROUP BY
            sales_order_item.product_id
        HAVING
                UniqueTotalQty > 3
    ) AS CustomerProducts;

# 13)
SELECT
    fo.customer_email,
    soi.qty_ordered,
    SUM(soi.qty_ordered) AS total_purchases ####
FROM
    filtered_orders fo
        JOIN sales_order_item soi on fo.entity_id = soi.order_id
        JOIN catalog_product_entity cpe ON soi.product_id = cpe.entity_id
WHERE
        fo.customer_email IN (
        SELECT
            customer_email
        FROM
            (
                SELECT
                    customer_email,
                    COUNT(DISTINCT product_id) as num_products
                FROM
                    filtered_orders
                        JOIN sales_order_item ON filtered_orders.entity_id = sales_order_item.order_id
                GROUP BY
                    customer_email
            ) customer_products
        WHERE
                num_products > 3
    )
  AND soi.product_id IN (
    SELECT
        product_id
    FROM
        (
            SELECT
                product_id,
                COUNT(qty_ordered) as num_orders
            FROM
                sales_order_item
            GROUP BY
                product_id
        ) product_orders
    WHERE
            num_orders > 3
)
GROUP BY
    soi.product_id,
    fo.customer_email;
