import pandas as pd
import json
import random
from datetime import datetime
import time


file = pd.read_excel('Machine Info.xlsx')
machine_info = {}
for _, row in file.iterrows():
    machine_info[row[0]] = {
        'Machine K5-4 Output(kg/h)': row[1], 'Machine K5-2 Output(kg/h)': row[2]}
with open("machine_info.json", 'w') as file:
    json.dump(machine_info, file, indent=4)



def log(message):

    print(message, "\n")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open('log.txt', 'a') as the_file:
        log_str = '['+dt_string+'] '+message+'\n'
        the_file.write(log_str)
        the_file.close()


with open('machine_info.json', 'r') as file:
    machine_info = json.loads(file.read())


def suggestion(material_type, quantity, deadline_days):
    with open('response.txt','w') as file:
        file.write(f"You have chosen {material_type} ,\n")

        log(f"You have chosen {material_type}")

        required_hrs = round(
            quantity/machine_info[material_type]['Machine K5-4 Output(kg/h)'], 2)
        file.write(f"It will take {round(required_hrs/24,2)} days that is {required_hrs} hours,\n")
        log(f"It will take {round(required_hrs/24,2)} days that is {required_hrs} hours")

        deadline_hrs = deadline_days*24

        capacity_k5_4 = round(
            deadline_hrs * machine_info[material_type]['Machine K5-4 Output(kg/h)'], 2)

        if required_hrs <= deadline_hrs:
            file.write("This will utilize machine K5000-4 only, \n")
            log("This will utilize machine K5000-4 only")

            file.write(
                f"Total production in {deadline_days} day(s) on k5-4 is: {capacity_k5_4},\n")
            log(
                f"Total production in {deadline_days} day(s) on k5-4 is: {capacity_k5_4}")

            if quantity <= capacity_k5_4:
                file.write(f"K5-4 is enough to produce : {quantity} kg \n")
                log(f"K5-4 is enough to produce : {quantity} kg \n")

        else:
            capacity_k5_2 = round(
                deadline_hrs * machine_info[material_type]['Machine K5-2 Output(kg/h)'], 2)

            k5_2_qty = round(quantity-capacity_k5_4, 2)

            if capacity_k5_2 < k5_2_qty:
                
                file.write("Quantity exceeded! Need to use other machines as well \n")
                log("Quantity exceeded! Need to use other machines as well \n")

            else:
                file.write(f"K5-4: {capacity_k5_4} kgs \nK5-2: {k5_2_qty} kgs \n")
                log(f"K5-4: {capacity_k5_4} kgs \nK5-2: {k5_2_qty} kgs \n")
    file.close()
    time.sleep(1)
    with open('response.txt','r') as file:
        output=file.readlines()
        return output

