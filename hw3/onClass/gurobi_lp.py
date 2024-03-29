#!/usr/bin/env python3

# Copyright 2021, Gurobi Optimization, LLC

# This example reads an LP model from a file and solves it.
# If the model is infeasible or unbounded, the example turns off
# presolve and solves the model again. If the model is infeasible,
# the example computes an Irreducible Inconsistent Subsystem (IIS),
# and writes it to a file

# modifiled based on https://www.gurobi.com/documentation/9.5/examples/lp_py.html

# %%
import sys
import gurobipy as gp
from gurobipy import GRB

if len(sys.argv) < 2:
    print('Usage: lp.py filename')
    sys.exit(0)

# Read and solve model
model = gp.read(sys.argv[1])
model.optimize()

# %%
if model.status == GRB.INF_OR_UNBD:
    # Turn presolve off to determine whether model is infeasible
    # or unbounded
    model.setParam(GRB.Param.Presolve, 0)
    model.optimize()

if model.status == GRB.OPTIMAL:
    print('Optimal objective: ', model.objVal)
    # model.write('model.sol')
    if len(model.x)>3:
        model.write('model.sol')
    else:
        print('Optimal variable: ', model.x)
    sys.exit(0)
elif model.status != GRB.INFEASIBLE:
    print('Optimization was stopped with status %d' % model.status)
    sys.exit(0)


# Model is infeasible - compute an Irreducible Inconsistent Subsystem (IIS)
print('')
print('Model is infeasible')
model.computeIIS()
model.write("model.ilp")
print("IIS written to file 'model.ilp'")
