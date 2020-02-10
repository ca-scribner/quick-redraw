import sqlalchemy as sa
import sqlalchemy.orm

from quick_redraw.data.modelbase import SqlAlchemyBase

# Shared factory
__factory = None


def global_init(db_path: str, echo: bool = True):
    """
    Initializes a single shared factory for all db access in this app

    Args:
        db_path (str): Path to the db file
        echo (bool): If True, engine will echo all db calls

    Returns:
        None
    """
    global __factory

    if __factory:
        return

    db_path = db_path.strip()

    if not db_path:
        print(f"WARNING: Found empty db_path - db will be created in memory")

    # TODO: Handle GCP
    conn_str = "sqlite:///" + db_path
    print(f"Connecting to DB at {conn_str}")

    engine = sa.create_engine(conn_str, echo=echo)

    __factory = sa.orm.sessionmaker(bind=engine)

    # Inform sa about our models
    import quick_redraw.data.__all_models

    SqlAlchemyBase.metadata.create_all(engine)
    print("DB global_init complete")


def create_session() -> sa.orm.Session:
    """
    Create and return a session
    """
    global __factory
    return __factory()


def add_commit_close(x, expire_on_commit: bool=True) -> None:
    """
    Convenience function that a session, adds an object to it, commits it, and closes it
    """
    s = create_session()
    s.expire_on_commit = expire_on_commit
    s.add(x)
    s.commit()
    s.close()
