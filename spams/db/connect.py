from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from spams.config import CONNECTION_STRING


def create_postgres_engine():
    return create_engine(CONNECTION_STRING)


def create_and_return_session(engine=None):
    if not engine:
        engine = create_postgres_engine()
    Session = sessionmaker(bind=engine)
    return Session()
