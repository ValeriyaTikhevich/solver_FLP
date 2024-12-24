import numpy as np
import pulp

# 1. Создаем переменные для объектов (Y_j) и их вместимости (C_k)
def add_facility_variables(range_facility, var_name_y, var_name_c):
    y_vars = [pulp.LpVariable(var_name_y.format(i=i), lowBound=0, upBound=1, cat=pulp.LpInteger) for i in range_facility]
    c_vars = [pulp.LpVariable(var_name_c.format(i=i), lowBound=0, cat=pulp.LpInteger) for i in range_facility]
    return y_vars, c_vars

# 2. Создаем матрицу распределения спроса (Z_ij)
def add_assignment_variables(range_client, range_facility, var_name):
    return np.array([
        [pulp.LpVariable(var_name.format(i=i, j=j), lowBound=0, upBound=1, cat=pulp.LpContinuous) for j in range_facility]
        for i in range_client
    ])

# 3. Ограничение на вместимость объектов
def add_capacity_constraints(problem, y_vars, c_vars, z_vars, demand, range_client, range_facility):
    for j in range_facility:
        problem += pulp.lpSum([demand[i] * z_vars[i, j] for i in range_client]) <= c_vars[j], f"capacity_constraint_{j}"
        problem += c_vars[j] <= y_vars[j] * 10000, f"open_capacity_constraint_{j}"
        problem += c_vars[j] >= y_vars[j] * 50, f"min_capacity_constraint_{j}"

# 4. Ограничение на удовлетворение спроса
def add_demand_constraints(problem, z_vars, accessibility_matrix, range_client, range_facility):
    for i in range_client:
        problem += pulp.lpSum([accessibility_matrix[i, j] * z_vars[i, j] for j in range_facility]) == 1, f"demand_constraint_{i}"

# Основная функция для решения объединенной задачи
def solve_combined_problem(cost_matrix, service_radius, demand_quantity, name="combined_problem"):
    num_clients, num_facilities = cost_matrix.shape
    range_clients = range(num_clients)
    range_facilities = range(num_facilities)

    # Инициализация задачи минимизации
    problem = pulp.LpProblem(name, pulp.LpMinimize)

    # Матрица доступности (a_ij)
    accessibility_matrix = (cost_matrix <= service_radius).astype(int)

    # Переменные
    y_vars, c_vars = add_facility_variables(range_facilities, "y[{i}]", "c[{i}]")
    z_vars = add_assignment_variables(range_clients, range_facilities, "z[{i}_{j}]")

    # Целевая функция: минимизация количества объектов и общей вместимости
    w1, w2 = 1000, 1
    problem += pulp.lpSum([w1 * y_vars[j] + w2 * c_vars[j] for j in range_facilities]), "objective_function"

    # Ограничения
    add_capacity_constraints(problem, y_vars, c_vars, z_vars, demand_quantity, range_clients, range_facilities)
    add_demand_constraints(problem, z_vars, accessibility_matrix, range_clients, range_facilities)

    # Решение задачи
    solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)

    if problem.status != 1:
        raise RuntimeError(f"Problem not solved: {pulp.LpStatus[problem.status]}.")

    fac2cli = []
    for j in range(len(y_vars)):
        if y_vars[j].value() > 0:
            fac_clients = [i for i in range(num_clients) if accessibility_matrix[i, j] > 0]
            fac2cli.append(fac_clients)
        else:
            fac2cli.append([])

    # Формируем результаты
    facilities_open = [j for j in range_facilities if y_vars[j].value() > 0.5]
    assignment = np.array([[z_vars[i, j].value() for j in range_facilities] for i in range_clients])
    capacities = [c_vars[j].value() for j in range_facilities]

    return facilities_open, capacities, fac2cli

def block_coverage(matrix, SERVICE_RADIUS, df, id):
    facilities, capacities, fac2cli = solve_combined_problem(np.array(matrix),
                                                                SERVICE_RADIUS,
                                                                df['demand_without'])
    dict_info_hotels2 = dict([(k,l) for k,l in enumerate(fac2cli) if len(l)>0])
    res_id = {id[key]: [id[val] for val in value] for key, value in dict_info_hotels2.items()}

    return capacities, res_id