package org.mlflow.tracking;

import java.io.File;
import org.mlflow.api.proto.Service.RunInfo;
import org.mlflow.api.proto.Service.RunStatus;
import org.mlflow.api.proto.Service.CreateRun;

public class RunContext implements AutoCloseable {
    private MlflowClient mlflowClient;
    private String runId;

    public RunContext(MlflowClient mlflowClient) {
        this.mlflowClient = mlflowClient;
        this.runId = mlflowClient.createRun().getRunUuid();
    }

    public RunContext(MlflowClient mlflowClient, long experimentId) {
        this.mlflowClient = mlflowClient;
        this.runId = mlflowClient.createRun(experimentId).getRunUuid();
    }

    public RunContext(MlflowClient mlflowClient, long experimentId, String sourceName) {
        this.mlflowClient = mlflowClient;
        this.runId = mlflowClient.createRun(experimentId, sourceName).getRunUuid();
    }

    public RunContext(MlflowClient mlflowClient, CreateRun request) {
        this.mlflowClient = mlflowClient;
        this.runId = mlflowClient.createRun(request).getRunUuid();
    }

    public void logParam(String key, String value) {
        mlflowClient.logParam(runId, key, value);
    }

    public void logMetric(String key, float value) {
        mlflowClient.logMetric(runId, key, value);
    }

    public void setTag(String key, String value) {
        mlflowClient.setTag(runId, key, value);
    }

    public void logArtifact(File localFile) {
        mlflowClient.logArtifact(runId, localFile);
    }

    public void logArtifact(File localFile, String artifactPath) {
        mlflowClient.logArtifact(runId, localFile, artifactPath);
    }

    public void logArtifacts(File localDir) {
        mlflowClient.logArtifacts(runId, localDir);
    }

    public void logArtifacts(File localDir, String artifactPath) {
        mlflowClient.logArtifacts(runId, localDir, artifactPath);
    }

    public String getRunId() {
        return runId;
    }

    @Override
    public void close() {
        mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis());
    }
}
