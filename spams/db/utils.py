from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import Table, MetaData

from spams.config import CONNECTION_STRING


def create_postgres_engine():
    return create_engine(CONNECTION_STRING)


def get_table(tablename, metadata):
    return Table(tablename, metadata, autoload=True)


def get_metadata(engine):
    return MetaData(engine)


def get_connection(engine):
    return engine.connect()


def create_and_return_session(engine=None):
    if not engine:
        engine = create_postgres_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def setup_database():
    engine = create_postgres_engine()
    metadata = get_metadata(engine)
    connection = get_connection(engine)
    return metadata, connection
