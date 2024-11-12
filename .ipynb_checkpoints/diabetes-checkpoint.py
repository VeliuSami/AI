import psycopg2

connection = psycopg2.connect(user = "postgres", password = "Agave!", host = "127.0.0.1", port = "5432", database = "Diabeties")


cursor = connection.cursor()

# Execute a query to select the age column
#cur.execute("SELECT age FROM diabetesdata")

# Fetch all the rows and store the age values in a list
#age_list = [row[0] for row in cur.fetchall()]

cursor.execute("Select blood_pressure from diabetesdata")

blood_pressure = [row[0] for row in cursor.fetchall()]

# Close the cursor and the connection


# Now you can use the age_list variable as needed
#print(age_list)

sum = 0

for values in blood_pressure:
    sum += values
    print("Current value is: ", values)

print("Sum of all values is: ", sum)

average = sum / len(blood_pressure)
print("Average of all values is: ", int(average))

cursor.execute("select outcome from diabetesdata")

counter = 0

for rows in cursor.fetchall():
    outcome = [rows[0]]


    for results in outcome:
        print(results)
        counter+=1
print("Counter total:" , counter)






'''
#cur.execute("SELECT age from diabetesdata where blood_pressure > 70")
cur.execute("SELECT * from diabetesdata where blood_pressure > 70")
rows = cur.fetchall()
print("Blood Pressure > 70")
print(rows)

print()
print("Blood Pressure < 60")
cur.execute("SELECT * from diabetesdata where blood_pressure < 60")
rows = cur.fetchall()
print(rows)

cur.execute("SELECT * FROM diabetesdata ORDER BY pregnancies DESC LIMIT 1")
rows = cur.fetchall()
print(rows)

cur.execute("select * from diabetesdata where BMI = 30.1 and glucose = 126")
rows = cur.fetchall()
print(rows)
'''

cursor.close()
connection.close()

