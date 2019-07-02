package org.mlflow.tracking.examples.java2;
  
import org.mlflow.tracking.MlflowClient;
import org.mlflow.tracking.RunContext;

public class HelloWorldFluent {
    public static void main(String [] args) throws Exception {
        String trackingUri = args[0];
        MlflowClient client = new MlflowClient(trackingUri);
        String expId = args[1];
        System.out.println("trackingUri: "+trackingUri);
        System.out.println("expId: "+expId);

        try (RunContext run = new RunContext(client, expId)) {
           run.logParam("alpha", "0.5");
           run.logMetric("rmse", 0.786);
           run.setTag("origin","HelloWorldFluent Java Example");
        } 
    }
}
