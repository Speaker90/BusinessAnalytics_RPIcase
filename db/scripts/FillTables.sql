.mode column
.header on

INSERT INTO BugsTemp1 (BugID,Outcome,Opening,Closing,ReporterID,AssigneeID)
	
SELECT
	  reports.id 			AS BugID	
	, reports.current_resolution	AS Outcome
	, reports.opening		AS Opening 
	, temp.Closing			AS Closing
	, reports.reporter		AS ReporterID	
	, temp.AssigneeID		AS AssigneeID	

FROM reports

INNER JOIN

	(
	SELECT 
	  resolution.id		
	, MAX(resolution."when")	AS Closing
	, resolution.who		AS AssigneeID

	FROM resolution

	GROUP BY	
	 resolution.id
	) temp 

ON reports.id = temp.id


WHERE (
	(  reports.current_status = "RESOLVED"
	OR reports.current_status = "CLOSED"
	OR reports.current_status = "VERIFIED"
	) AND reports.current_resolution <> "DUPLICATE"
)	
;

INSERT INTO BugsTemp2(BugID,Priority,CC)

SELECT 
	  cc.id			AS BugsID		
	, temp.Priority 	AS Priority
	, COUNT(cc.what)	AS CC

FROM cc

INNER JOIN

	(
	SELECT 
	  priority.id
	, priority.what 		AS Priority	
	, MAX(priority."when")

	FROM priority

	GROUP BY
	 priority.id
	) temp

ON cc.id = temp.id

INNER JOIN BugsTemp1
	ON cc.id = BugsTemp1.BugID

GROUP BY
	cc.id
;
.output stdout

INSERT INTO BugsTemp3 (BugID,Product,OS)

SELECT 
	  temp.id		AS BugsID
	, temp.what 		AS Product
	, op_sys.what 		AS OS

FROM op_sys

INNER JOIN 
	(
	SELECT 
		  product.id
		, product.what
		, product."when"
		, product.who

	FROM product

	INNER JOIN BugsTemp1
		ON 
		(
			product.id = BugsTemp1.BugID
		AND     product.who =BugsTemp1.ReporterID
		AND 	product."when" = BugsTemp1.Opening
	)	
	) temp	

ON 
	(
		temp.id = op_sys.id
	AND     temp.who =op_sys.who
	AND 	temp."when" = op_sys."when"
)	

;

INSERT INTO Bugs

SELECT 
	  BugsTemp1.BugID
	, BugsTemp1.Outcome
	, BugsTemp1.Opening
	, BugsTemp1.Closing
	, BugsTemp1.ReporterID
	, BugsTemp1.AssigneeID
	, temp.Priority
	, temp.CC
	, temp.Product
	, temp.OS

FROM BugsTemp1

INNER JOIN
	(
	SELECT 
		  BugsTemp2.BugID
		, BugsTemp2.Priority
		, BugsTemp2.CC
		, BugsTemp3.Product
		, BugsTemp3.OS

	FROM BugsTemp2
		
	INNER JOIN BugsTemp3
	ON
		BugsTemp2.BugID = BugsTemp3.BugID
	) temp

ON
	temp.BugID = BugsTemp1.BugID
;



INSERT INTO Assignees

SELECT 
	  temp.AssigneeID
	, ifnull(temp.Success*1.0/temp.Total,0)
FROM 
(
	SELECT
		Bugs.AssigneeID 
		, COUNT(*)  AS Total
		, SUM(CASE WHEN Bugs.Outcome="FIXED" THEN 1 ELSE 0 END) AS Success  
	FROM Bugs
	GROUP BY
		AssigneeID
	) temp
;

INSERT INTO Reporters

SELECT 
	  temp.ReporterID
	, ifnull(temp.Success*1.0/temp.Total,0)
FROM 
(
	SELECT
		Bugs.ReporterID 
		, COUNT(*)  AS Total
		, SUM(CASE WHEN Bugs.Outcome="FIXED" THEN 1 ELSE 0 END) AS Success  
	FROM Bugs
	GROUP BY
		ReporterID
	) temp
;

DROP TABLE BugsTemp1;
DROP TABLE BugsTemp2;
DROP TABLE BugsTemp3;
DROP TABLE assigned_to;
DROP TABLE bug_status;
DROP TABLE cc;
DROP TABLE component;
DROP TABLE op_sys;
DROP TABLE priority;
DROP TABLE product;
DROP TABLE reports;
DROP TABLE resolution;
DROP TABLE severity;
DROP TABLE version;

