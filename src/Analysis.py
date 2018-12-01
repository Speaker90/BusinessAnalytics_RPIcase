import pandas as pd
import sqlite3

conn = None


conn = sqlite3.connect("../db/Bugs.db")
df = pd.read_sql_query("SELECT Bugs.*, Assignees.SuccessAssignee, Reporters.SuccessReporter FROM Bugs INNER JOIN Assignees ON Bugs.AssigneeID = Assignees.AssigneeID INNER JOIN Reporters ON Bugs.ReporterID = Reporters.ReporterID;", conn)

print(df)
