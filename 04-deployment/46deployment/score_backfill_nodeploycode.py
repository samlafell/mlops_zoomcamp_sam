from datetime import datetime
from dateutil.relativedelta import relativedelta

from prefect import flow
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from score import ride_duration_prediction

@flow
def ride_duration_prediction_backfill():
    start_date = datetime(year=2022, month = 3, day = 1)
    end_date = datetime(year=2022, month = 4, day = 1)
    
    d = start_date
    
    while d <= end_date:
        ride_duration_prediction(
            taxi_type='green',
            run_id = '602e2fa2a0df4f5a87eef98f93b79090',
            run_date = d
        )
        
        d += relativedelta(months = 1)

if __name__ == '__main__':
    ride_duration_prediction_backfill()