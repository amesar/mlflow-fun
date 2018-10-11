package org.mlflow.tracking;

import org.apache.log4j.Logger;
import org.mlflow.api.proto.Service.RunInfo;
import org.mlflow.api.proto.Service.RunStatus;

public class RunContext implements AutoCloseable {
    private static final Logger logger = Logger.getLogger(RunContext.class);
    private MlflowClient mlflowClient;
    private String runId;

    public RunContext(MlflowClient mlflowClient, long experimentId, String sourceName) 
        throws Exception {
        this.mlflowClient = mlflowClient;
        RunInfo runInfo = mlflowClient.createRun(experimentId, sourceName);
        this.runId = runInfo.getRunUuid();
        logger.debug("runId="+runId);
    }

    public void logParam(String key, String value) throws Exception {
        mlflowClient.logParam(runId, key, value);
    }

    public void logMetric(String key, float value) throws Exception {
        mlflowClient.logMetric(runId, key, value);
    }

    public void logArtifact(String localFile, String artifactPath) throws Exception {
        throw new UnsupportedOperationException();
    }

    public void logModel(String key) throws Exception {
        throw new UnsupportedOperationException("logModel() coming soon");
    }

    public String getRunId() {
        return runId;
    }

    @Override
    public void close() throws Exception {
        mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis());
    }
}
