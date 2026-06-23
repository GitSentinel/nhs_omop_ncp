from src.data_access.omop_queries import (
    get_person, get_conditions, get_medications,
    get_visits, get_measurements, get_observations,
    get_notes, get_procedures
)

PID = 17247

p  = get_person(PID)
c  = get_conditions(PID)
m  = get_medications(PID)
v  = get_visits(PID)
me = get_measurements(PID)
o  = get_observations(PID)
n  = get_notes(PID)
pr = get_procedures(PID)

print('person     :', p['person_id'], 'yob =', p['year_of_birth'])
print('conditions :', len(c))
print('medications:', len(m))
print('visits     :', len(v))
print('measurements:', len(me))
print('observations:', len(o))
print('notes      :', len(n))
print('procedures :', len(pr))
print('All domain queries OK')
