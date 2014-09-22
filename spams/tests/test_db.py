from sqlalchemy import MetaData

from spams.db.utils import create_postgres_engine

if __name__ == "__main__":
    engine = create_postgres_engine()
    m = MetaData()
    m.reflect(engine)
    print len(m.tables.keys())
