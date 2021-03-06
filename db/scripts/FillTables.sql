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

INSERT INTO BugsTemp2(BugID,Assignments,CC)

SELECT 
	  cc.id			AS BugsID		
	, temp.Assignments 	AS Assignments 
	, (COUNT(cc.what)-1)	AS CC

FROM cc

INNER JOIN

	(
	SELECT 
	  assigned_to.id
	, COUNT(*)		AS Assignments

	FROM assigned_to

	GROUP BY
	 assigned_to.id
	) temp

ON cc.id = temp.id

INNER JOIN BugsTemp1
	ON cc.id = BugsTemp1.BugID

GROUP BY
	cc.id
;

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

INSERT INTO BugsTemp4(BugID,Component,Social)

SELECT 
	  temp1.BugID			AS BugID
	, temp1.Component 		AS Component
	, temp1.Social			AS Social

FROM
	(	
	SELECT 
		  BugsTemp1.BugID	AS BugID		
		, temp.Component 	AS Component
		, BugsTemp1.AssigneeID
		, BugsTemp1.ReporterID
		, COUNT(*)		AS Social		

	FROM BugsTemp1

	INNER JOIN

		(
		SELECT 
		  component.id
		, component.what	AS Component	
		, MAX(component."when")

		FROM component

		GROUP BY
		 component.id
		) temp

	ON BugsTemp1.BugID = temp.id

	GROUP BY
		  BugsTemp1.AssigneeID, BugsTemp1.ReporterID
	) temp1	
;

INSERT INTO Bugs

SELECT 
	  BugsTemp1.BugID
	, BugsTemp1.Outcome
	, BugsTemp1.Opening
	, BugsTemp1.Closing
	, BugsTemp1.ReporterID
	, BugsTemp1.AssigneeID
	, temp.Assignments
	, temp.CC
	, temp.Product
	, temp.OS
	, temp.Component
	, temp.Social

FROM BugsTemp1

INNER JOIN
	(
	SELECT 
		  BugsTemp2.BugID
		, BugsTemp2.Assignments
		, BugsTemp2.CC
		, BugsTemp3.Product
		, BugsTemp3.OS
		, BugsTemp4.Component
		, BugsTemp4.Social

	FROM BugsTemp2
		
	INNER JOIN BugsTemp3
	ON
		BugsTemp2.BugID = BugsTemp3.BugID

	INNER JOIN BugsTemp4
	ON
		BugsTemp2.BugID = BugsTemp4.BugID
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
DROP TABLE BugsTemp4;
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

