#SELECT * FROM sales;
#CREATE TABLE sales_copy SELECT * FROM sales;

#Identify duplicates
SELECT Customer_Name, Email, Phone, Order_Date, COUNT(*) as Dupes 
	FROM sales 
    GROUP BY Customer_Name, Email, Phone, Order_Date 
    HAVING Dupes > 1;

#Delete duplicates
DELETE s1 FROM sales as s1
	JOIN sales as s2 
		ON s1.Order_Date = s2.Order_Date
        AND s1.Customer_Name = s2.Customer_Name
        AND s1.Email = s2.Email
        AND s1.Phone = s2.Phone
		AND s1.Order_ID < s2.Order_ID;
        
#Outliers
SELECT * FROM sales WHERE Revenue > (SELECT AVG(Revenue) + 3 * STD(Revenue) FROM sales);
SELECT * FROM sales WHERE `Discount (%)` > (SELECT AVG(`Discount (%)`) + 3 * STD(`Discount (%)`) FROM sales);

#Discount blanks
UPDATE sales
	SET `Discount (%)` = 0
    WHERE `Discount (%)` = "" OR `Discount (%)` IS NULL;

#Email blanks
UPDATE sales 
	SET Email = 'no_email_provided@email.com' 
    WHERE Email IS NULL OR Email = "";

#Phone blanks
UPDATE sales 
	SET Phone = LPAD(Phone, 11, '0')
    WHERE Phone IS NULL OR Phone = "";
    
#Format date
UPDATE sales SET Order_Date = REPLACE(Order_Date, '/', '-');
UPDATE sales SET Order_Date = STR_TO_DATE(Order_Date, '%m/%d/%Y');

#Append 0 at front to 10-digit phone numbers
UPDATE sales 
	SET Phone = LPAD(Phone, 11, '0')
    WHERE LENGTH(Phone) < 11;
    
#Trimming
UPDATE sales SET Product_Category = TRIM(Product_Category);
UPDATE sales SET Email = TRIM(Email);
UPDATE sales SET Phone = TRIM(Phone);



#SUMMARIZING DATA

#By product category
SELECT Product_Category, COUNT(Revenue) as Total_Orders, 
		SUM(Revenue) as Total_Revenue, 
        AVG(Revenue) as Average_Revenue, 
        STD(Revenue) as STD_Revenue,
        AVG(`Discount (%)`) as Average_Discount
	FROM sales
    GROUP BY Product_Category;
    
#By month
SELECT MONTH(Order_Date), 
		COUNT(Revenue) as Total_Orders, 
        SUM(Revenue) as Total_Revenue, 
        AVG(Revenue) as Average_Revenue, 
        AVG(`Discount (%)`) as Average_Discount
	FROM sales
    GROUP BY MONTH(Order_Date);