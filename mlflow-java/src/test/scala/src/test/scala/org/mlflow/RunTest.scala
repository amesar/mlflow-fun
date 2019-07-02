package org.mlflow

import scala.collection.JavaConversions._
import org.testng.annotations._
import org.testng.Assert._
import org.mlflow.api.proto.Service._

class RunTest extends BaseTest {

  @Test
  def simpleRun() {
    val expId = client.createExperiment(createExperimentName())

    val runInfo = client.createRun(expId)
    val runId = runInfo.getRunUuid()
    client.logParam(runId, "p1","hi")
    client.logMetric(runId, "m1",0.123F)
    client.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
    val runInfo2 = client.getRun(runId).getInfo()
    assertEquals(runInfo.getRunUuid(),runInfo2.getRunUuid())
    assertTrue(runInfo2.getStartTime() > 0)

    val runs = client.listRunInfos(expId)
    assertEquals(runs.size(), 1)
  }

  @Test
  def verboseRun() {
    val expId = client.createExperiment(createExperimentName())

    val request = CreateRun.newBuilder()
        .setExperimentId(expId)
        .build()
    val runInfo = client.createRun(request)
    val runId = runInfo.getRunUuid()

    val runInfo2 = client.getRun(runId).getInfo()
    assertEquals(runInfo.getRunUuid(),runInfo2.getRunUuid())
    assertTrue(runInfo2.getStartTime() > 0)

    val runs = client.listRunInfos(expId)
    assertEquals(runs.size(), 1)
  }
}
