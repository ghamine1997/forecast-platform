import io


def write_table(df, table, engine):
    """Replace `table` with the contents of `df` using fast COPY."""
    df.head(0).to_sql(table, engine, if_exists="replace", index=False)

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv)", buffer)
        raw.commit()
    finally:
        raw.close()