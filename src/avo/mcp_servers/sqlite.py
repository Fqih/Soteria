"""Read-only SQLite MCP server.

Three tools:

* ``list_tables``     — enumerate tables + their row counts
* ``describe_table``  — column metadata for one table
* ``query``           — execute a SELECT and return rows as JSON

Writes are rejected at two layers:

1. the SQL parser refuses any statement whose first non-whitespace
   token is not ``SELECT`` / ``WITH`` / ``PRAGMA table_info`` / ``PRAGMA table_list``;
2. the connection is opened with ``mode=ro`` via SQLite URI so even a
   parse-slip cannot mutate the file.

Run directly::

    python -m avo.mcp_servers.sqlite --db /path/to/avo.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from avo.mcp_servers._stdio import AvoMCPServer, user_error


class _Server(AvoMCPServer):
    def __init__(self, db_path: Path) -> None:
        super().__init__(server_name="avo-sqlite", server_version="0.1.0")
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self._db_path}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True)
        except sqlite3.Error as exc:
            user_error(f"could not open database read-only: {exc}")
        connection.row_factory = sqlite3.Row
        return connection


class _ListTablesArgs(BaseModel):
    pass


class _DescribeTableArgs(BaseModel):
    table: str = Field(description="Table name to describe.")


class _QueryArgs(BaseModel):
    sql: str = Field(description="A read-only SQL statement (SELECT / WITH / PRAGMA).")
    max_rows: int = Field(default=500, description="Maximum rows to return.")


_ALLOWED_PREFIXES = ("select", "with", "pragma")


def build_server(db_path: Path) -> _Server:
    server = _Server(db_path)

    def _enforce_readonly(sql: str) -> None:
        stripped = sql.lstrip().lower()
        if not stripped.startswith(_ALLOWED_PREFIXES):
            user_error(
                "only SELECT / WITH / PRAGMA queries are allowed; "
                f"got: {sql.split(None, 1)[0] if sql else '(empty)'!r}"
            )

    @server.tool(
        name="list_tables",
        description="List user tables in the SQLite database with row counts.",
        arguments_model=_ListTablesArgs,
    )
    async def _list(args: _ListTablesArgs) -> dict[str, object]:
        del args
        connection = server._connect()
        try:
            rows = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
            tables = []
            for (name,) in rows:
                try:
                    count = connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                except sqlite3.Error:
                    count = None
                tables.append({"name": name, "row_count": count})
            return {"tables": tables}
        finally:
            connection.close()

    @server.tool(
        name="describe_table",
        description="Return column metadata for one table.",
        arguments_model=_DescribeTableArgs,
    )
    async def _describe(args: _DescribeTableArgs) -> dict[str, object]:
        connection = server._connect()
        try:
            rows = connection.execute(
                'SELECT name, type, "notnull", dflt_value, pk '
                "FROM pragma_table_info(?) ORDER BY cid",
                (args.table,),
            ).fetchall()
            if not rows:
                user_error(f"unknown table: {args.table!r}")
            columns = [
                {
                    "name": row["name"],
                    "type": row["type"],
                    "not_null": bool(row["notnull"]),
                    "default": row["dflt_value"],
                    "primary_key": bool(row["pk"]),
                }
                for row in rows
            ]
            return {"table": args.table, "columns": columns}
        finally:
            connection.close()

    @server.tool(
        name="query",
        description="Execute one read-only SQL statement and return rows.",
        arguments_model=_QueryArgs,
    )
    async def _query(args: _QueryArgs) -> dict[str, object]:
        _enforce_readonly(args.sql)
        connection = server._connect()
        try:
            cursor = connection.execute(args.sql)
            rows = cursor.fetchall()
            truncated = len(rows) > args.max_rows
            rows = rows[: args.max_rows]
            description = cursor.description or []
            columns = [col[0] for col in description]
            data = [{column: row[column] for column in columns} for row in rows]
            return {
                "sql": args.sql,
                "columns": columns,
                "rows": data,
                "row_count": len(data),
                "truncated": truncated,
            }
        finally:
            connection.close()

    return server


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="avo.mcp_servers.sqlite")
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to the SQLite database file (opened read-only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import asyncio
    import sys as _sys

    args = _parse_args(_sys.argv[1:] if argv is None else argv)
    if not args.db.exists():
        raise SystemExit(f"database not found: {args.db}")
    server = build_server(args.db)
    asyncio.run(server.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
