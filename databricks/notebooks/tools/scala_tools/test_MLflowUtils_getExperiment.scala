// Databricks notebook source
// MAGIC %run ./MLflowUtils.scala

// COMMAND ----------

val client = new MlflowClient()

// COMMAND ----------

val ex = "/Shared/experiments/demo/sk_WineQuality"
val exp = MLflowUtils.getExperiment(client, ex)

// COMMAND ----------

val exp = MLflowUtils.getExperiment(client, "2580010")

// COMMAND ----------

val exp = MLflowUtils.getExperiment(client, "foo")
println("exp.type: "+exp.getClass)

// COMMAND ----------

val exp = MLflowUtils.getExperiment(client, "123")

// COMMAND ----------

