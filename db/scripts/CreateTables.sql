/*
	Create all the temporary and final tables for the Bugs.db

	Author:	Florian Spychiger
	Date:	02 December 2018

	Stored in scripts/CreateTables.sql
*/;

CREATE TABLE IF NOT EXISTS BugsTemp1
(
	  BugID		INTEGER
	, Outcome 	TEXT NOT NULL	
	, Opening	INTEGER NOT NULL
	, Closing	INTEGER NOT NULL
	, ReporterID	INTEGER NOT NULL
	, AssigneeID	INTEGER NOT NULL
	, PRIMARY KEY	(BugID)
)
;

CREATE TABLE IF NOT EXISTS BugsTemp2
(
	  BugID		INTEGER
	, Priority 	TEXT NOT NULL
	, CC		INTEGER NOT NULL
	, PRIMARY KEY	(BugID)
)
;

CREATE TABLE IF NOT EXISTS BugsTemp3
(
	  BugID		INTEGER
	, Product	TEXT NOT NULL
	, OS		TEXT NOT NULL
	, PRIMARY KEY	(BugID)
)
;

CREATE TABLE IF NOT EXISTS BugsTemp4
(
	  BugID		INTEGER
	, Component	TEXT NOT NULL
	, Social	INTEGER NOT NULL
	, PRIMARY KEY	(BugID)
)
;
CREATE TABLE IF NOT EXISTS Bugs
(
	  BugID		INTEGER
	, Outcome 	TEXT NOT NULL	
	, Opening	INTEGER NOT NULL
	, Closing	INTEGER NOT NULL
	, ReporterID	INTEGER NOT NULL
	, AssigneeID	INTEGER NOT NULL
	, Priority	TEXT NOT NULL
	, CC		INTEGER NOT NULL
	, Product	TEXT NOT NULL 
	, OS		TEXT NOT NULL
	, Component 	TEXT NOT NULL
	, Social 	INTEGER NOT NULL
	, FOREIGN KEY	(ReporterID) REFERENCES Reporters (ReporterID)
	, FOREIGN KEY	(AssigneeID) REFERENCES Assignees (AssigneeID)
	, PRIMARY KEY	(BugID)
)
;

CREATE TABLE IF NOT EXISTS Reporters
(
	  ReporterID		INTEGER
	, SuccessReporter	REAL NOT NULL	
	, PRIMARY KEY		(ReporterID)
)
;

CREATE TABLE IF NOT EXISTS Assignees
(
	  AssigneeID		INTEGER
	, SuccessAssignee	REAL NOT NULL	
	, PRIMARY KEY		(AssigneeID)
)
;


