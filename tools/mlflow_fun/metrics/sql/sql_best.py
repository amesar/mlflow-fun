
from mlflow.store.dbmodels.models import SqlExperiment, SqlRun, SqlMetric
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

def get_best_run(connection, experiment_id, metric, ascending=True):
    engine = create_engine(connection)
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    best = (session.query(SqlRun.run_uuid,SqlMetric.value)
        .filter(SqlExperiment.experiment_id == SqlRun.experiment_id)
        .filter(SqlRun.run_uuid  == SqlMetric.run_uuid )
        .filter(SqlExperiment.experiment_id == experiment_id)
        .filter(SqlMetric.key == metric))
    best = best.order_by(SqlMetric.value.asc()) if ascending else best.order_by(SqlMetric.value.desc())
    return best.limit(1).one()
