CREATE TABLE IF NOT EXISTS BugsTemp1
(
	  BugID		INTEGER
	, Time		INTEGER
	, Outcome 	TEXT NOT NULL	
	, ReporterID	INTEGER NOT NULL
	, AssigneeID	INTEGER NOT NULL
	, PRIMARY KEY	(BugID)
)
;

CREATE TABLE IF NOT EXISTS BugsTemp2
(
	  BugID		INTEGER
	, CC		INTEGER
	, PRIMARY KEY	(BugID)
)
;

CREATE TABLE IF NOT EXISTS BugsTemp3
(
	  BugID		INTEGER
	, Product	TEXT 
	, OS		TEXT
	, PRIMARY KEY	(BugID)
)
;

CREATE TABLE IF NOT EXISTS Bugs
(
	  BugID		INTEGER
	, Outcome 	TEXT NOT NULL	
	, ReporterID	INTEGER NOT NULL
	, AssigneeID	INTEGER NOT NULL
	, CC		INTEGER
	, Product	TEXT 
	, OS		TEXT
	, FOREIGN KEY	(ReporterID) REFERENCES Reporters (ReporterID)
	, FOREIGN KEY	(AssigneeID) REFERENCES Assignees (AssigneeID)
	, PRIMARY KEY	(BugID)
)
;

CREATE TABLE IF NOT EXISTS Reporters
(
	  ReporterID		INTEGER
	, SuccessReporter	REAL	
	, PRIMARY KEY		(ReporterID)
)
;

CREATE TABLE IF NOT EXISTS Assignees
(
	  AssigneeID		INTEGER
	, SuccessAssignee	REAL	
	, PRIMARY KEY		(AssigneeID)
)
;


