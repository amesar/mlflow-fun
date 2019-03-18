from __future__ import print_function

import sys
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("mlflow_analytics").enableHiveSupport().getOrCreate()

def run(query):
    print("==============")
    print(query+"\n")
    spark.sql(query).show(100,False)

def run_queries(database):
    spark.sql("use "+database)

    run("select * from mlflow_status")
    run("select count(*) as num_experiments from experiments")
    run("select count(*) as num_runs from runs")
    run("select * from experiments limit 10")
    #run("select * from runs")

    # Run count per experiment
    run(""" 
select e.experiment_id,e.name experiment_name,count(r.experiment_id) as num_runs from runs r
  right outer join experiments e on e.experiment_id=r.experiment_id
  group by e.experiment_id, e.name order by num_runs desc""")

    # Run count per user
    run("select user_id, count(user_id) as num_runs from runs group by user_id order by num_runs desc")

    run("""
select user_id, round(sum(end_time-start_time)/1000/60,2) as run_time_in_minutes, count(run_uuid) as num_runs
  from runs where start_time != 0
  group by user_id
  order by run_time_in_minutes desc""")

    run("select * from mlflow_status")

if __name__ == "__main__":
    database = sys.argv[1]
    run_queries(database)
