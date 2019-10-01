# Databricks notebook source
def get_experiment(client, exp_id_or_name):
    if exp_id_or_name.isdigit():
        exp = client.get_experiment(exp_id_or_name)
        which = "ID"
    else:
        exp = client.get_experiment_by_name(exp_id_or_name)
        which = "name"
    if exp is None:
         raise Exception("Cannot find experiment {} '{}'".format(which,exp_id_or_name))
    return exp